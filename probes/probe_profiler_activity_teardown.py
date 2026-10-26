"""Probe which torch.profiler activity combinations survive teardown on XPU.

The DLRM-v3 Profiler wrapper hardcodes ProfilerActivity.CUDA; on XPU that
produced 'PTI_ERROR_NOT_IMPLEMENTED from clearActivities' and a SIGABRT after
the train loop finished. Each case runs in a subprocess so an abort in one does
not take the whole probe down.
"""

import os
import subprocess
import sys

import torch

CASE = os.environ.get("CASE")

if CASE is None:
    print(f"torch {torch.__version__}")
    print("available ProfilerActivity members:")
    for name in dir(torch.profiler.ProfilerActivity):
        if not name.startswith("_"):
            print(f"  {name}")

    cases = ["cpu_only", "cpu_cuda", "cpu_xpu", "xpu_only"]
    results = {}
    for case in cases:
        env = dict(os.environ, CASE=case)
        proc = subprocess.run(
            [sys.executable, "-u", __file__],
            env=env,
            capture_output=True,
            text=True,
        )
        tail = [ln for ln in proc.stdout.strip().splitlines() if ln]
        results[case] = (proc.returncode, tail[-1] if tail else "", proc.stderr.strip())

    print("\n=== results ===")
    for case, (rc, last, err) in results.items():
        verdict = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"{case:10s} {verdict:14s} {last}")
        if rc != 0:
            for ln in err.splitlines()[-3:]:
                print(f"           | {ln}")
    sys.exit(0)

activity_map = {
    "cpu_only": [torch.profiler.ProfilerActivity.CPU],
    "cpu_cuda": [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    "cpu_xpu": [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.XPU,
    ],
    "xpu_only": [torch.profiler.ProfilerActivity.XPU],
}

activities = activity_map[CASE]
dev = torch.device("xpu:0")

# active=2 with a short wait so the profiler actually enters its active window,
# unlike the 10-batch smoke run where wait=10 kept it dormant.
prof = torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    activities=activities,
    record_shapes=True,
)

prof.start()
for _ in range(6):
    a = torch.randn(512, 512, device=dev)
    b = torch.randn(512, 512, device=dev)
    (a @ b).sum().item()
    prof.step()
prof.stop()

torch.xpu.synchronize()
print(f"{CASE}: survived start/step/stop/teardown")

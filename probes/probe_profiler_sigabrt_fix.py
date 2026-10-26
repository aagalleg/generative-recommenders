"""Reproduce the DLRM-v3 profiler SIGABRT on XPU and test candidate fixes.

The generative-recommenders Profiler wrapper constructs torch.profiler.profile
and only ever calls step() -- never start() or stop(). With the debug schedule
(wait=10) a 10-batch run transitions into WARMUP on the final step, which arms
Kineto, and the process then exits with it still armed.

Each case runs in a subprocess so an abort does not take the probe down.
"""

import os
import subprocess
import sys

import torch

CASE = os.environ.get("CASE")

# The exact schedule from generative_recommenders/dlrm_v3/utils.py Profiler,
# with debug.gin's active=10.
SCHEDULE = dict(wait=10, warmup=20, active=10, repeat=1)
STEPS = 10


def build(activities):
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(**SCHEDULE),
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
    )


if CASE is None:
    cases = ["repro_cuda_nostop", "cuda_with_stop", "xpu_with_stop", "xpu_nostop"]
    results = {}
    for case in cases:
        proc = subprocess.run(
            [sys.executable, "-u", __file__],
            env=dict(os.environ, CASE=case),
            capture_output=True,
            text=True,
        )
        out = [ln for ln in proc.stdout.strip().splitlines() if ln]
        results[case] = (proc.returncode, out[-1] if out else "", proc.stderr.strip())

    print("=== results ===")
    for case, (rc, last, err) in results.items():
        sig = ""
        if rc < 0:
            sig = f" (signal {-rc})"
        elif rc == 134:
            sig = " (SIGABRT)"
        verdict = "clean exit" if rc == 0 else f"rc={rc}{sig}"
        print(f"{case:20s} {verdict:22s} {last}")
        for ln in err.splitlines():
            if "PTI" in ln or "terminate" in ln or "Kineto" in ln:
                print(f"                     | {ln.strip()}")
    sys.exit(0)


CPU = torch.profiler.ProfilerActivity.CPU
CUDA = torch.profiler.ProfilerActivity.CUDA
XPU = torch.profiler.ProfilerActivity.XPU

acts = [CPU, XPU] if "xpu" in CASE else [CPU, CUDA]
prof = build(acts)

dev = torch.device("xpu:0")
for _ in range(STEPS):
    a = torch.randn(256, 256, device=dev)
    (a @ a).sum().item()
    prof.step()

print(f"{CASE}: {STEPS} steps done, current_action={prof.current_action}")

if "with_stop" in CASE:
    prof.stop()
    print(f"{CASE}: stop() returned cleanly")

torch.xpu.synchronize()
print(f"{CASE}: reached end of script")

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""End-to-end check: dlrm_v3 training with the XPU workarounds trains correctly
at multi-rank scale (see XPU_DISTRIBUTED_ISSUES.md).

The original DDP code path segfaults at >= 4 ranks, so there is no reference
to compare against at this scale (xpu_e2e_ddp_consistency.py does that at 2
ranks). Instead this runs the real `train_ranker.main()` (MovieLens-1M,
`--mode train-eval`) with the current code on WORLD_SIZE ranks and checks
properties that must hold for correct data-parallel training:

1. completion: the run exits 0 and every rank saves its results;
2. rank sync: final dense weights are bit-identical on every rank (exact; the
   workaround broadcasts initial weights and averages gradients, as DDP does);
3. learning: the last logged train NE is lower than the first by at least
   --min-ne-drop (relative), and every recorded metric is finite.

Optionally (--baseline-world-size N) the same training also runs on N ranks
and final NE values are shown side by side, for information only: the
effective batch size differs, so the numbers are not expected to match.

The training processes reuse xpu_e2e_ddp_consistency.py's `--child` mode
(DLRMV3_CONSISTENCY_MODE=new), which records metrics on rank 0 and saves every
rank's final dense weights and embedding checksums.

Usage (from the repo root, oneAPI environment set up):
    python -m generative_recommenders.dlrm_v3.train.tests.xpu_e2e_multirank_check \\
        --world-size 16 --out-dir /tmp/dlrmv3_multirank

Exit code: 0 all checks pass, 1 a check failed, 2 a run failed.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from generative_recommenders.dlrm_v3.train.tests import xpu_e2e_ddp_consistency as consistency

_TRAIN_NE = "metric/window_ne/rating"
_EVAL_NE = "metric/lifetime_ne/rating"


def _run(name: str, world_size: int, args: argparse.Namespace, repo: str) -> Tuple[bool, str]:
    """Train the current code on `world_size` ranks via the consistency script's child mode."""
    out = os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    gin_overrides = "\n".join([
        f"train_eval_loop.num_epochs = {args.epochs}",
        "train_eval_loop.num_train_batches = None",
        f"train_eval_loop.eval_frequency = {args.eval_every}",
        f"train_eval_loop.num_eval_batches = {args.eval_batches}",
        f"train_eval_loop.metric_log_frequency = {args.log_every}",
        "train_eval_loop.output_trace = False",
    ])
    env = {
        **os.environ,
        "PYTHONPATH": repo + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "WORLD_SIZE": str(world_size),
        "DLRMV3_CONSISTENCY_MODE": "new",
        "DLRMV3_CONSISTENCY_OUT": out,
        "DLRMV3_CONSISTENCY_GIN": gin_overrides,
    }
    cmd = [sys.executable, "-X", "faulthandler", "-m", consistency.__spec__.name, "--child"]
    log_path = os.path.join(args.out_dir, f"{name}.log")
    print(f"[{name}] world_size={world_size} epochs={args.epochs} ... ", end="", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        # Own session, so a hang can be killed with all ranks and DataLoader workers.
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc: Any = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            rc = "timeout"
        # Always kill the session: after a crash the launcher exits, but its rank
        # processes and their DataLoader workers can survive.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    files_ok = os.path.exists(os.path.join(out, "metrics.jsonl")) and all(
        os.path.exists(os.path.join(out, f"rank{r}.pt")) for r in range(world_size))
    ok = rc == 0 and files_ok
    print(f"{'ok' if ok else 'FAILED'} (rc={rc}, {time.time() - t0:.0f}s, log: {log_path})")
    return ok, out


def _series(metrics: Dict[Tuple[str, int, str], float], mode: str, name: str) -> List[Tuple[int, float]]:
    return sorted((step, v) for (m, step, k), v in metrics.items() if m == mode and k == name)


def _rank_sync(ranks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Largest difference of every rank's dense tensors from rank 0's, and where."""
    worst: Dict[str, Any] = {"max_abs_diff": 0.0, "rank": None, "parameter": None,
                             "differing_tensors": 0, "tensors": 0}
    base = ranks[0]["dense"]
    for r, rank in enumerate(ranks[1:], start=1):
        if set(rank["dense"]) != set(base):
            raise ValueError(f"rank {r} has different dense parameter names than rank 0")
        for k, t in rank["dense"].items():
            worst["tensors"] += 1
            if torch.equal(t, base[k]):
                continue
            worst["differing_tensors"] += 1
            d = (t.double() - base[k].double()).abs().max().item() if t.numel() else 0.0
            if d >= worst["max_abs_diff"]:
                worst.update(max_abs_diff=d, rank=r, parameter=k)
    return worst


def _ne_summary(metrics: Dict[Tuple[str, int, str], float]) -> Dict[str, Any]:
    train, evals = _series(metrics, "train", _TRAIN_NE), _series(metrics, "eval", _EVAL_NE)
    return {"train_first": train[0] if train else None, "train_last": train[-1] if train else None,
            "eval_rounds": evals}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world-size", type=int, default=0, help="ranks (default: all visible XPUs)")
    p.add_argument("--epochs", type=int, default=4, help="training epochs (one epoch is short at many ranks)")
    p.add_argument("--eval-every", type=int, default=20, help="train batches between eval rounds")
    p.add_argument("--eval-batches", type=int, default=10, help="batches per eval round")
    p.add_argument("--log-every", type=int, default=10, help="metric_log_frequency")
    p.add_argument("--min-ne-drop", type=float, default=0.05,
                   help="required relative drop of train window NE, first to last log")
    p.add_argument("--sync-tol", type=float, default=0.0,
                   help="allowed max abs dense-weight difference between ranks (0 = bit-identical)")
    p.add_argument("--baseline-world-size", type=int, default=0,
                   help="also train on this many ranks and show NE side by side (0 = skip)")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per run")
    p.add_argument("--out-dir", default="/tmp/dlrmv3_multirank")
    args = p.parse_args()
    if args.world_size <= 0:
        args.world_size = torch.xpu.device_count()
    if args.world_size < 2 or args.epochs < 1 or args.sync_tol < 0 or not 0 <= args.min_ne_drop < 1:
        p.error("need --world-size >= 2, --epochs >= 1, --sync-tol >= 0 and 0 <= --min-ne-drop < 1")

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"world_size={args.world_size} epochs={args.epochs} eval_every={args.eval_every} "
          f"ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK', '<unset>')}")

    ok, out = _run(f"workaround_ws{args.world_size}", args.world_size, args, repo)
    if not ok:
        print("Run failed; see its log. No verdict.")
        return 2
    data = consistency._load(out, args.world_size)
    baseline: Optional[Dict[str, Any]] = None
    if args.baseline_world_size:
        b_ok, b_out = _run(f"baseline_ws{args.baseline_world_size}", args.baseline_world_size, args, repo)
        if b_ok:
            baseline = consistency._load(b_out, args.baseline_world_size)
        else:
            print("Baseline run failed; continuing without it.")

    checks: Dict[str, Dict[str, Any]] = {}
    checks["completion"] = {"ok": True, "detail": f"{args.world_size} ranks saved results"}

    sync = _rank_sync(data["ranks"])
    checks["rank_sync"] = {"ok": sync["max_abs_diff"] <= args.sync_tol, **sync}

    nonfinite = [k for k, v in data["metrics"].items() if not math.isfinite(v)]
    ne = _ne_summary(data["metrics"])
    first, last = ne["train_first"], ne["train_last"]
    learning_ok = (not nonfinite and first is not None and last is not None and last[0] > first[0]
                   and last[1] <= first[1] * (1 - args.min_ne_drop))
    checks["learning"] = {"ok": learning_ok, "train_ne_first": first, "train_ne_last": last,
                          "required_drop": args.min_ne_drop, "nonfinite_metrics": [list(k) for k in nonfinite]}

    print(f"\n{'check':12s} result  detail")
    print(f"{'completion':12s} {'PASS':6s}  {checks['completion']['detail']}")
    s = checks["rank_sync"]
    print(f"{'rank_sync':12s} {'PASS' if s['ok'] else 'FAIL':6s}  {s['tensors']} tensor comparisons vs rank 0, "
          f"{s['differing_tensors']} differ, max abs diff {s['max_abs_diff']:.3e}"
          + (f" (rank {s['rank']}, {s['parameter']})" if s["parameter"] else "")
          + f"; tolerance {args.sync_tol:g}")
    l = checks["learning"]
    if first and last:
        print(f"{'learning':12s} {'PASS' if l['ok'] else 'FAIL':6s}  train window NE {first[1]:.4f} (step {first[0]}) -> "
              f"{last[1]:.4f} (step {last[0]}), drop {1 - last[1] / first[1]:.1%} (need >= {args.min_ne_drop:.0%})"
              + (f"; {len(nonfinite)} non-finite metrics" if nonfinite else ""))
    else:
        print(f"{'learning':12s} FAIL    fewer than two train NE records")
    print("eval lifetime NE by eval step: "
          + ", ".join(f"{st}: {v:.4f}" for st, v in ne["eval_rounds"]) if ne["eval_rounds"] else "no eval rounds")

    summary: Dict[str, Any] = {"args": vars(args), "checks": checks, "eval_ne": ne["eval_rounds"]}
    if baseline is not None:
        b = _ne_summary(baseline["metrics"])
        summary["baseline"] = {"world_size": args.baseline_world_size, **b}
        def fmt(x: Any) -> str:
            return f"{x[1]:.4f} (step {x[0]})" if x else "n/a"
        print(f"\ninfo: final NE, {args.world_size} ranks vs {args.baseline_world_size} ranks "
              f"(different effective batch size, not expected to match):")
        print(f"  train window NE: {fmt(last)} vs {fmt(b['train_last'])}")
        print(f"  eval lifetime NE (last round): {fmt(ne['eval_rounds'][-1] if ne['eval_rounds'] else None)} "
              f"vs {fmt(b['eval_rounds'][-1] if b['eval_rounds'] else None)}")

    passed = all(c["ok"] for c in checks.values())
    summary["passed"] = passed
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n{'ALL CHECKS PASSED' if passed else 'CHECK FAILED'} (details: {args.out_dir}/summary.json)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

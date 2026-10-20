# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""End-to-end check: dlrm_v3 training with the XPU workarounds matches the
original DDP code path (see XPU_DISTRIBUTED_ISSUES.md).

Runs the real `train_ranker.main()` (MovieLens-1M, `--mode train-eval`) in
separate processes:

- reference: the original code path: torchrec's default DDP wrapper (the XPU
  `_XpuGradAveragingDataParallelWrapper` and `_average_grads_pre_step` hook
  patched out) and MetricsLogger on the XPU, synced over the default group.
- new: the current code, unmodified.

Every rank of every run is seeded identically, and a short run is configured
with gin bindings. Rank 0 records every train/eval metric computation; every
rank saves its final dense weights and per-shard embedding checksums.

XPU training is not bit-reproducible, so by default the reference runs five times:
all reference pairs are compared, and their largest difference is the noise floor.
Reference sample counts and metric keys must match before calibration is accepted.
The new run passes
if, against the first reference, (a) NE metrics, final dense weights and
embedding checksums differ by at most FACTOR x that floor (or an absolute
minimum), (b) metric sample counts are identical (same data), and (c) both
runs recorded the same (mode, step, metric) entries. Accuracy and GAUC are
reported but not gated: they change in steps of 1/N whenever a single
prediction crosses the threshold or swaps rank. Use a world size where DDP does
not crash (2 ranks on one card; the original path segfaults at >= 4 ranks).

Maximum-error gates are preserved. Median/p95 NE errors, dense RMS errors,
worst offenders, per-shard checksum errors, and reference-pair spreads are
diagnostics in summary.json. Pairwise comparisons share runs: the noise floor
is an empirical envelope, not a confidence bound. A maximum more than FACTOR
times the median pair difference is flagged as outlier-sensitive calibration;
it is a warning, not a separate PASS/FAIL criterion. One reference run disables
calibration and uses only the absolute tolerance floors.

Usage (from the repo root, oneAPI environment set up):
    ZE_AFFINITY_MASK=0,1 python -m \\
        generative_recommenders.dlrm_v3.train.tests.xpu_e2e_ddp_consistency \\
        --world-size 2 --steps 100 --out-dir /tmp/dlrmv3_consistency

Exit code: 0 consistent, 1 inconsistent, 2 a run failed.
"""

import argparse
import itertools
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_MODE = os.environ.get("DLRMV3_CONSISTENCY_MODE")  # set only in the training processes
_OUT = os.environ.get("DLRMV3_CONSISTENCY_OUT", "")
_GIN = os.environ.get("DLRMV3_CONSISTENCY_GIN", "")

# ---------------------------------------------------------------------------
# Training-process side. This runs at import, so it also runs in every
# mp.spawn child (which re-imports this module as __mp_main__).
# ---------------------------------------------------------------------------
if _MODE is not None:
    import gin
    from generative_recommenders.dlrm_v3 import utils as dlrm_utils
    from generative_recommenders.dlrm_v3.train import train_ranker
    from generative_recommenders.dlrm_v3.train import utils as train_utils
    from torch import distributed as dist
    from torchrec.distributed.model_parallel import DefaultDataParallelWrapper
    from torchrec.distributed.types import ShardedTensor

    # Same initial weights and dropout/data RNG on every rank and in every run.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Short run: the extra gin bindings are applied after the config file.
    _parse_config_file = gin.parse_config_file

    def _parse_with_overrides(*args: Any, **kwargs: Any) -> Any:
        result = _parse_config_file(*args, **kwargs)
        gin.parse_config(_GIN)
        return result

    gin.parse_config_file = _parse_with_overrides

    # Record every metric computation on rank 0 (eval rounds compute twice per
    # step; the comparison keeps the last value per (mode, step)).
    _compute = dlrm_utils.MetricsLogger.compute

    def _recording_compute(self: Any, mode: str = "train") -> Dict[str, Any]:
        result = _compute(self, mode)
        if result and dist.is_initialized() and dist.get_rank() == 0:
            with open(os.path.join(_OUT, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({
                    "mode": mode,
                    "step": self.global_step[mode],
                    "metrics": {k: float(v) for k, v in result.items()},
                }) + "\n")
        return result

    dlrm_utils.MetricsLogger.compute = _recording_compute

    # Save final weights after training.
    _train_eval_loop = train_ranker.train_eval_loop

    def _train_eval_loop_and_dump(*args: Any, **kwargs: Any) -> None:
        _train_eval_loop(*args, **kwargs)
        torch.xpu.synchronize()
        rank, state = kwargs["rank"], kwargs["model"].state_dict()
        dense = {k: v.detach().cpu() for k, v in state.items()
                 if isinstance(v, torch.Tensor) and not isinstance(v, ShardedTensor)}
        emb: Dict[str, List[float]] = {}
        for k, v in state.items():
            if isinstance(v, ShardedTensor):
                for i, shard in enumerate(v.local_shards()):
                    # Shards are ~20 GB: accumulate in float64 over row chunks. Only
                    # non-negative terms (abs, square), so the checksums cannot cancel
                    # to ~0 and blow up the relative comparison.
                    sums = torch.zeros(2, dtype=torch.float64, device=shard.tensor.device)
                    for c in shard.tensor.detach().split(1 << 16):
                        c = c.double()
                        sums += torch.stack([c.abs().sum(), c.square().sum()])
                    emb[f"{k}[{i}]"] = sums.tolist()
        torch.save({"dense": dense, "emb": emb}, os.path.join(_OUT, f"rank{rank}.pt"))

    train_ranker.train_eval_loop = _train_eval_loop_and_dump

    if _MODE == "reference":
        # Original code path: DDP instead of the XPU wrapper, no averaging hook ...
        train_utils._XpuGradAveragingDataParallelWrapper = DefaultDataParallelWrapper
        train_utils._average_grads_pre_step = lambda optimizer, args, kwargs: None

        # ... and metrics on the XPU, synced over the default (XCCL) group.
        class _OriginalMetricsLogger(dlrm_utils.MetricsLogger):
            def __init__(self, *args: Any, device: torch.device, process_group: Any = None,
                         **kwargs: Any) -> None:
                xpu = torch.device("xpu", torch.xpu.current_device())
                super().__init__(*args, device=xpu, process_group=None, **kwargs)

        train_ranker.MetricsLogger = _OriginalMetricsLogger
    else:
        assert _MODE == "new", _MODE


# ---------------------------------------------------------------------------
# Driver side.
# ---------------------------------------------------------------------------
def _run(name: str, mode: str, args: argparse.Namespace, repo: str) -> Tuple[bool, str]:
    out = os.path.join(args.out_dir, name)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    gin_overrides = "\n".join([
        "train_eval_loop.num_epochs = 1",
        f"train_eval_loop.num_train_batches = {args.steps}",
        f"train_eval_loop.eval_frequency = {args.eval_every}",
        f"train_eval_loop.num_eval_batches = {args.eval_batches}",
        f"train_eval_loop.metric_log_frequency = {args.log_every}",
        "train_eval_loop.output_trace = False",
    ])
    env = {
        **os.environ,
        "PYTHONPATH": repo + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "WORLD_SIZE": str(args.world_size),
        "DLRMV3_CONSISTENCY_MODE": mode,
        "DLRMV3_CONSISTENCY_OUT": out,
        "DLRMV3_CONSISTENCY_GIN": gin_overrides,
    }
    cmd = [sys.executable, "-X", "faulthandler", "-m", __spec__.name, "--child"]
    log_path = os.path.join(args.out_dir, f"{name}.log")
    print(f"[{name}] mode={mode} ... ", end="", flush=True)
    t0 = time.time()
    with open(log_path, "w") as log:
        # Own session, so a hang can be killed with all ranks and DataLoader workers.
        proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            rc = "timeout"
    files_ok = os.path.exists(os.path.join(out, "metrics.jsonl")) and all(
        os.path.exists(os.path.join(out, f"rank{r}.pt")) for r in range(args.world_size))
    ok = rc == 0 and files_ok
    print(f"{'ok' if ok else 'FAILED'} (rc={rc}, {time.time() - t0:.0f}s, log: {log_path})")
    return ok, out


def _load(out: str, world_size: int) -> Dict[str, Any]:
    metrics: Dict[Tuple[str, int, str], float] = {}
    with open(os.path.join(out, "metrics.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            for k, v in rec["metrics"].items():
                metrics[(rec["mode"], rec["step"], k)] = v  # last value per step wins
    ranks = [torch.load(os.path.join(out, f"rank{r}.pt")) for r in range(world_size)]
    return {"metrics": metrics, "ranks": ranks}


def _metric_kind(key: Tuple[str, int, str]) -> str:
    """ne: continuous (log loss). discrete: accuracy/GAUC, which jump by 1/N when a single
    prediction crosses the threshold or swaps rank. count: sample counts (data identity)."""
    name = key[2]
    if "num_samples" in name:
        return "count"
    return "ne" if "_ne/" in name else "discrete"


def _diff(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve maximum-error gates and collect distribution/location diagnostics."""
    keys = sorted(set(a["metrics"]) & set(b["metrics"]))
    assert keys, "no common metric records"
    unmatched = len(set(a["metrics"]) ^ set(b["metrics"]))
    by_kind = {"ne": 0.0, "discrete": 0.0, "count": 0.0}
    ne_errors = []
    worst_ne = None
    for k in keys:
        kind = _metric_kind(k)
        if not all(math.isfinite(run["metrics"][k]) for run in (a, b)):
            raise ValueError(f"non-finite metric: {k}")
        error = abs(a["metrics"][k] - b["metrics"][k])
        if kind == "ne":
            ne_errors.append(error)
            if worst_ne is None or error > by_kind[kind]:
                worst_ne = k
        by_kind[kind] = max(by_kind[kind], error)
    dense = emb = 0.0
    dense_squared = 0.0
    dense_count = 0
    dense_parameters = []
    embedding_checksums = []
    worst_dense = worst_emb = None
    assert len(a["ranks"]) == len(b["ranks"]), "rank counts differ"
    for rank, (ra, rb) in enumerate(zip(a["ranks"], b["ranks"])):
        assert set(ra["dense"]) == set(rb["dense"]), "dense parameter names differ"
        for k in sorted(ra["dense"]):
            x, y = ra["dense"][k].double(), rb["dense"][k].double()
            assert x.shape == y.shape, f"dense parameter shapes differ: {k}"
            if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                raise ValueError(f"non-finite dense parameter: rank {rank}, {k}")
            if x.numel():
                errors = (x - y).abs()
                maximum = errors.max().item()
                squared = errors.square().sum().item()
                detail = {"rank": rank, "parameter": k, "max": maximum,
                          "rms": math.sqrt(squared / x.numel()),
                          "flat_index": errors.argmax().item()}
                dense_parameters.append(detail)
                if worst_dense is None or maximum > dense:
                    worst_dense = detail
                dense = max(dense, maximum)
                dense_squared += squared
                dense_count += x.numel()
        assert set(ra["emb"]) == set(rb["emb"]), "embedding shards differ"
        for k in sorted(ra["emb"]):
            assert len(ra["emb"][k]) == len(rb["emb"][k]) == 2, "expected abs/square checksums"
            for name, x, y in zip(("abs_sum", "square_sum"), ra["emb"][k], rb["emb"][k]):
                if not (math.isfinite(x) and math.isfinite(y)):
                    raise ValueError(f"non-finite embedding checksum: rank {rank}, {k}, {name}")
                error = abs(x - y) / max(abs(x), abs(y), 1e-30)
                detail = {"rank": rank, "shard": k, "checksum": name, "relative_error": error}
                embedding_checksums.append(detail)
                if worst_emb is None or error > emb:
                    worst_emb = detail
                emb = max(emb, error)
    return {"metrics_ne": by_kind["ne"], "metrics_accuracy_gauc": by_kind["discrete"],
            "sample_counts": by_kind["count"], "unmatched_metric_records": float(unmatched),
            "dense_weights": dense, "embedding_checksums": emb, "num_metric_records": len(keys),
            "diagnostics": {
                "metrics_ne": {"median": float(np.median(ne_errors)) if ne_errors else None,
                               "p95": float(np.percentile(ne_errors, 95)) if ne_errors else None,
                               "max": by_kind["ne"], "worst_key": worst_ne},
                "dense_weights": {"rms": math.sqrt(dense_squared / dense_count) if dense_count else None,
                                  "max": dense, "worst": worst_dense,
                                  "parameters": dense_parameters},
                "embedding_checksums": {"max": emb, "worst": worst_emb,
                                        "entries": embedding_checksums},
            }}


def _reference_comparisons(data: Dict[str, Any], names: List[str]) -> List[Dict[str, Any]]:
    comparisons = []
    for left, right in itertools.combinations(names, 2):
        diff = _diff(data[left], data[right])
        if diff["sample_counts"] != 0 or diff["unmatched_metric_records"] != 0:
            raise ValueError(f"invalid calibration: {left} vs {right}: "
                             f"sample_counts={diff['sample_counts']}, "
                             f"unmatched_metric_records={diff['unmatched_metric_records']}")
        comparisons.append({"runs": [left, right], "diff": diff})
    return comparisons


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=100, help="train batches per run")
    p.add_argument("--eval-every", type=int, default=50, help="train batches between eval rounds")
    p.add_argument("--eval-batches", type=int, default=10, help="batches per eval round")
    p.add_argument("--log-every", type=int, default=10, help="metric_log_frequency")
    p.add_argument("--reference-runs", type=int, default=5,
                   help="reference runs (default: 5); all pairs calibrate noise; 1 uses absolute floors")
    p.add_argument("--factor", type=float, default=2.0, help="allowed multiple of the noise floor")
    p.add_argument("--min-tol-metrics", type=float, default=1e-3)
    p.add_argument("--min-tol-dense", type=float, default=1e-5)
    p.add_argument("--min-tol-emb", type=float, default=1e-6)
    p.add_argument("--timeout", type=int, default=900, help="seconds per run")
    p.add_argument("--out-dir", default="/tmp/dlrmv3_consistency")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.reference_runs < 1:
        p.error("--reference-runs must be at least 1")
    if not math.isfinite(args.factor) or args.factor < 1:
        p.error("--factor must be finite and at least 1")
    for name in ("min_tol_metrics", "min_tol_dense", "min_tol_emb"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be finite and non-negative")

    if args.child:  # a training process launched by _run()
        sys.argv = ["train_ranker", "--dataset", "movielens-1m-xpu", "--mode", "train-eval"]
        train_ranker.main()
        return 0

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"world_size={args.world_size} steps={args.steps} eval_every={args.eval_every} "
          f"eval_batches={args.eval_batches} ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK', '<unset>')}")

    runs = [(f"reference_{i + 1}", "reference") for i in range(args.reference_runs)] + [("new", "new")]
    outs: Dict[str, str] = {}
    for name, mode in runs:
        ok, out = _run(name, mode, args, repo)
        if not ok:
            print(f"Run {name} failed; see its log. No verdict.")
            return 2
        outs[name] = out
    data = {name: _load(out, args.world_size) for name, out in outs.items()}

    try:
        references = _reference_comparisons(
            data, [name for name, mode in runs if mode == "reference"])
        new_vs_ref = _diff(data["new"], data["reference_1"])
    except (ValueError, AssertionError) as exc:
        print(f"Comparison failed: {exc}. No verdict.")
        return 2
    # Gated quantities and their absolute tolerance floors. Sample counts must match
    # exactly (same data), and both runs must have recorded the same (mode, step,
    # metric) entries. Accuracy/GAUC are reported only: they move in steps of 1/N
    # when one prediction crosses the threshold or swaps rank, so two reference runs
    # cannot estimate their noise.
    exact = ("sample_counts", "unmatched_metric_records")
    mins = {"metrics_ne": args.min_tol_metrics, "dense_weights": args.min_tol_dense,
            "embedding_checksums": args.min_tol_emb, "sample_counts": 0.0,
            "unmatched_metric_records": 0.0}
    print(f"\nCompared {new_vs_ref['num_metric_records']} metric values "
          f"and final weights on {args.world_size} ranks.")
    passed = True
    summary: Dict[str, Any] = {}
    calibration: Dict[str, Any] = {}
    warnings = []
    print(f"Calibration: {args.reference_runs} reference runs, {len(references)} pairs "
          "(shared runs; not independent observations).")
    if not references:
        print("WARNING: no noise calibration; using absolute tolerance floors only.")
    print(f"{'quantity':25s} {'ref min':>11s} {'ref median':>11s} {'ref max':>11s} "
          f"{'new vs ref':>12s} {'tolerance':>11s}  result")
    for q in list(mins) + ["metrics_accuracy_gauc"]:
        values = [pair["diff"][q] for pair in references]
        n = max(values) if values else None
        if values:
            median = float(np.median(values))
            calibration[q] = {"min": min(values), "median": median, "max": n,
                              "outlier_sensitive": q not in exact and n > args.factor * median}
            if calibration[q]["outlier_sensitive"]:
                warnings.append(f"{q}: largest reference difference exceeds "
                                f"{args.factor:g} x median; calibration is outlier-sensitive")
        if q in exact:
            tol = 0.0
        elif q in mins:
            tol = max(args.factor * n, mins[q]) if n is not None else mins[q]
        else:
            tol = None
        ok = None if tol is None else new_vs_ref[q] <= tol
        if ok is not None:
            passed &= ok
        summary[q] = {"noise": n, "new_vs_ref": new_vs_ref[q], "tolerance": tol, "ok": ok}
        spread = " ".join(f"{value:11.3e}" for value in (min(values), median, n)) if values else (
            f"{'n/a':>11s} {'n/a':>11s} {'n/a':>11s}")
        print(f"{q:25s} {spread} {new_vs_ref[q]:12.3e} "
              f"{('%.3e' % tol) if tol is not None else 'n/a':>11s}  "
              f"{'info' if ok is None else 'PASS' if ok else 'FAIL'}")
    print("(metrics_accuracy_gauc is informational: these metrics change in steps of 1/N when a"
          " single prediction crosses the threshold or swaps rank.)")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for quantity, details in new_vs_ref["diagnostics"].items():
        compact = {key: value for key, value in details.items() if key not in ("parameters", "entries")}
        print(f"{quantity} diagnostics: {json.dumps(compact)}")
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"args": vars(args), "results": summary, "consistent": passed,
                   "calibration": {"statistics": calibration, "pairs": references,
                                   "warnings": warnings},
                   "diagnostics": new_vs_ref["diagnostics"]}, f, indent=2, allow_nan=False)
    print(f"\n{'CONSISTENT' if passed else 'INCONSISTENT'} (details: {args.out_dir}/summary.json)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

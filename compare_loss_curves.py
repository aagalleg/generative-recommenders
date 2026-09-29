#!/usr/bin/env python3
"""Diff the per-step training loss of a run against a reference run.

    python compare_loss_curves.py XPU_RUN_DIR CPU_RUN_DIR [--repeat XPU_RUN_DIR_2]

Run folders come from run_dlrm_v3_xpu.sh (losses.csv, manifest.json,
operative_config.gin). Writes loss_diff.json and loss_diff.txt into the
first run folder. Exit status 1 if a loss is not finite or the two runs
were built from different revisions or configs; the loss differences
themselves are reported, not judged: dropout, device-side embedding init
and stochastic rounding differ between devices, so curves agree in trend,
not per step.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional


def read_losses(run: Path) -> Dict[int, float]:
    with open(run / "losses.csv") as f:
        return {int(r["step"]): float(r["total"]) for r in csv.DictReader(f)}


def revisions(run: Path) -> Dict[str, str]:
    comps = json.loads((run / "manifest.json").read_text())["components"]
    out = {}
    for name, c in comps.items():
        git = c.get("git") or c.get("built_from") or {}
        rev = git.get("sha") or c["version"]
        if c["dirty"]:
            rev += f" (dirty {(git.get('diff_sha256') or 'unknown')[:12]})"
        out[name] = rev
    return out


def config_lines(run: Path) -> List[str]:
    text = (run / "operative_config.gin").read_text()
    if text.startswith("# PARTIAL"):
        raise SystemExit(f"{run}: operative_config.gin is PARTIAL (run did not finish)")
    # The effective HSTU_EMBEDDING_DIM / HASH_SIZE are written as comments.
    return [
        line for line in text.splitlines()
        if line.strip() and (not line.startswith("#") or " = " in line)
    ]


def diff_stats(a: Dict[int, float], b: Dict[int, float]) -> Dict[str, object]:
    common = sorted(a.keys() & b.keys())
    # Non-finite losses are reported as problems; statistics use the rest.
    steps = [s for s in common if math.isfinite(a[s]) and math.isfinite(b[s])]
    absd = [abs(a[s] - b[s]) for s in steps]
    reld = [abs(a[s] - b[s]) / abs(b[s]) if b[s] else math.inf for s in steps]
    corr: Optional[float] = None
    if len(steps) > 2:
        xs, ys = [a[s] for s in steps], [b[s] for s in steps]
        if statistics.pstdev(xs) and statistics.pstdev(ys):
            corr = statistics.correlation(xs, ys)
    return {
        "steps_compared": len(steps),
        "steps_non_finite": [s for s in common if s not in steps],
        "steps_only_in_first": sorted(a.keys() - b.keys()),
        "steps_only_in_second": sorted(b.keys() - a.keys()),
        "max_abs_diff": max(absd, default=0.0),
        "mean_abs_diff": statistics.fmean(absd) if absd else 0.0,
        "max_rel_diff": max(reld, default=0.0),
        "mean_rel_diff": statistics.fmean(reld) if reld else 0.0,
        "pearson": corr,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run to check (XPU)")
    ap.add_argument("reference", type=Path, help="reference run (CPU)")
    ap.add_argument("--repeat", type=Path, help="second run like RUN, for the run-to-run noise floor")
    args = ap.parse_args()

    run, ref = read_losses(args.run), read_losses(args.reference)
    problems: List[str] = []

    for name, losses in (("run", run), ("reference", ref)):
        bad = [s for s, v in losses.items() if not math.isfinite(v)]
        if bad:
            problems.append(f"{name}: non-finite loss at steps {bad}")

    rev_run, rev_ref = revisions(args.run), revisions(args.reference)
    rev_diff = {k: (rev_run.get(k), rev_ref.get(k)) for k in rev_run if rev_run.get(k) != rev_ref.get(k)}
    if rev_diff:
        problems.append(f"different revisions: {rev_diff}")

    cfg_run, cfg_ref = config_lines(args.run), config_lines(args.reference)
    cfg_diff = sorted(set(cfg_run) ^ set(cfg_ref))
    if cfg_diff:
        problems.append(f"different operative configs: {cfg_diff}")

    report = {
        "run": str(args.run),
        "reference": str(args.reference),
        "revisions": rev_run,
        "vs_reference": diff_stats(run, ref),
        "noise_floor": diff_stats(run, read_losses(args.repeat)) if args.repeat else None,
        "per_step": [
            {"step": s, "run": run[s], "reference": ref[s], "abs_diff": abs(run[s] - ref[s])}
            for s in sorted(run.keys() & ref.keys())
        ],
        "problems": problems,
    }

    lines = [
        f"run:       {args.run}",
        f"reference: {args.reference}",
        "",
        f"{'step':>5} {'run':>12} {'reference':>12} {'abs diff':>10} {'rel diff':>9}",
    ]
    for r in report["per_step"]:
        rel = r["abs_diff"] / abs(r["reference"]) if r["reference"] else math.inf
        lines.append(f"{r['step']:>5} {r['run']:>12.6f} {r['reference']:>12.6f} {r['abs_diff']:>10.2e} {rel:>9.2%}")
    for label, key in (("vs reference", "vs_reference"), ("noise floor (run vs repeat)", "noise_floor")):
        st = report[key]
        if st:
            corr = "n/a" if st["pearson"] is None else f"{st['pearson']:.3f}"
            lines.append(
                f"\n{label}: {st['steps_compared']} steps, max abs {st['max_abs_diff']:.2e}, "
                f"mean abs {st['mean_abs_diff']:.2e}, mean rel {st['mean_rel_diff']:.2%}, pearson {corr}"
            )
    lines.append("\nproblems: " + ("none" if not problems else "\n  " + "\n  ".join(problems)))
    text = "\n".join(lines) + "\n"

    (args.run / "loss_diff.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.run / "loss_diff.txt").write_text(text)
    print(text, end="")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

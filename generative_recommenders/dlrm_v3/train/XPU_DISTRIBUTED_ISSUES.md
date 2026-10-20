# DLRM-v3 multi-rank training on Intel XPU (PVC): known problems and workarounds

Multi-rank (multi-tile) DLRM-v3 training on Intel Data Center GPU Max (PVC) segfaults inside oneCCL. This document describes the failures, what causes them as far as we can tell, the workarounds in this branch, and how to check them.

## Summary

| # | Problem | Trigger | Status |
|---|---|---|---|
| 1 | SIGSEGV right after the first training step, at 4 or more ranks | DDP's XCCL gradient allreduce running inside `backward()` | Worked around: no DDP on XPU (verified up to 15 ranks) |
| 2 | SIGSEGV at the first metric sync, at 14 ranks | torchrec metric syncs over XCCL | Worked around: metrics on CPU, synced over gloo |
| 3 | PTI warnings and failures once profiling starts | `train_eval_loop.output_trace = True` in the XPU gin file | Avoided: set it to `False` |
| 4 | SIGSEGV in the first optimizer step at 16 ranks, with the workarounds in place | the workaround's own post-backward dense-gradient allreduce over XCCL | **Open:** use at most 15 ranks |

Problems 1, 2 and 4 have the same crash signature, in a oneCCL worker thread; the faulty layer (oneCCL, PyTorch's XCCL backend, or UR/Level Zero) is not yet known. Neither is a regression in this branch: the `feat/dlrm-v3-xpu-v1.8.0` branch crashes the same way on the current software stack.

## Environment

- 8× Intel Data Center GPU Max 1550 (16 tiles)
- torch 2.14.1+xpu, with its bundled `libccl.so.2` and `libccl_legacy.so`
- System oneCCL 2022.1
- oneAPI compiler 2026.1 (UR loader and Level Zero adapter)
- `libze_intel_gpu` 1.6.33578
- torchrec with fbgemm_xpu (TBE kernels); sparse optimizer RowWiseAdagrad (fused); dense optimizer Adam

## The crash signature

Every crash we captured with gdb has the same stack in a oneCCL worker thread:

```
#0  0x0000000000000002 in ?? ()                 <- call through an invalid function pointer
#1  ur::level_zero::urEventGetInfo(...)                     libur_adapter_level_zero
#4  sycl::detail::event_impl::get_info<command_execution_status>()
#5  ccl::native_event_impl::test()                          libccl.so.1 (2022.1)
#6  deps_entry::update()
#8  ccl_sched::do_progress()
#12 ccl_worker_func
```

- **Observed:** while running a collective, oneCCL polls the SYCL events it was given as dependencies, and the status query (`urEventGetInfo`) faults inside `libze_intel_gpu.so`. Frame #0 is either an invalid address such as `0x0` or `0x2`, a call through an invalid function pointer, or an invalid read at offset `0x14a558`.
- **Not established:** the stack doesn't show the event's lifetime or ownership. It fits a stale event handle (released or recycled before the query), a corrupted event object, or a concurrency bug in the UR/Level Zero query path.
- **Hypothesis:** the queried event belongs to an earlier asynchronous `all_to_all_single` with uneven splits, and is no longer valid when the next collective checks its dependencies. The plain-PyTorch reproduction in `upstream_repro/` supports this trigger, but doesn't prove the mechanism.
- Several ranks crash within the same second. In `dmesg` they show up as `segfault at 2 ip 0000000000000002`, or as faults inside `libze_intel_gpu.so`, all with the same stack-pointer pattern.
- How often this happens depends on how many ranks there are and which collectives run in the step. It is not specific to the model.

## Problem 1: DDP gradient allreduce in backward

**Symptom:** all ranks log step 1, then several ranks segfault during the step-2 backward pass. This happens every time at 4 or more ranks, for example 4 ranks on 2 cards, 8 ranks on 8 cards, or 16 tiles.

**What is running when it crashes:** the autograd thread is in
`Reducer::mark_variable_ready → all_reduce_bucket → ProcessGroupXCCL::allreduce → ccl::allreduce(..., deps)`,
meaning DDP is allreducing a gradient bucket while backward continues. That backward includes torchrec's embedding backward: the fused TBE update and the `all_to_all` that returns embedding gradients.

**Isolation:** with a synthetic torchrec model (`EmbeddingCollection` sharded by DMP, plus a dense MLP; no generative-recommenders code):

| Configuration | 4 ranks / 2 cards | 8 ranks / 8 cards |
|---|---|---|
| DMP + DDP (default) | SIGSEGV after step 0 | SIGSEGV after step 0 |
| No dense part, so no DDP | OK | — |
| DDP replaced by one allreduce after `backward()` returns | OK | OK |
| Plain DDP, no torchrec | OK | OK |

In these runs, crashes occurred only with DDP's in-backward allreduce together with the sharded embeddings. Replacing DDP with an allreduce after `backward()` avoided them. That control also drops DDP's reducer and bucketing, so it doesn't show whether the timing, the thread, or DDP itself is what matters. The plain-PyTorch reproduction in `upstream_repro/` has more controls.

**Tested and ruled out (each still crashes):**
- the profiler;
- tile placement;
- `ZE_FLAT_DEVICE_HIERARCHY` (FLAT or COMPOSITE);
- `CCL_ATL_TRANSPORT=ofi FI_PROVIDER=tcp CCL_ZE_IPC_EXCHANGE=pidfd`;
- the v1.8.0 XPU guards: skipping `apply_optimizer_in_backward`, `fused_params` sharders, skipping `init_state`, `xpu.synchronize()` after sharding;
- loading a consistent set of oneAPI libraries from the venv instead of the mixed system/venv set;
- the sharding type: both row-wise and table-wise crash.

**Workaround** (`train/utils.py`, XPU only):
- **`_XpuGradAveragingDataParallelWrapper`** is passed to DMP as `data_parallel_wrapper` instead of torchrec's DDP wrapper. Like DDP at construction, it moves the unsharded modules to the device and broadcasts their parameters and buffers from rank 0.
- **`_average_grads_pre_step`** is a step pre-hook on the dense optimizer. It averages all dense gradients (sum / world size, as DDP does) in one flattened allreduce. It runs inside `optimizer.step()`, after `backward()` has returned. A missing gradient becomes zeros, so every rank reduces the same buffer layout.
- **A guard:** plans with `DATA_PARALLEL` embedding tables are rejected, because torchrec relies on DDP to keep those in sync.
- **Checkpoint keys are unchanged:** DMP strips DDP's `module.` prefix in `state_dict()`.

**Cost:** dense gradient communication no longer overlaps backward. The throughput impact has not been measured.

**Limit:** verified up to 15 ranks. At 16 ranks the workaround's own allreduce crashes (Problem 4).

## Problem 2: torchrec metric sync over XCCL

**Symptom:** with Problem 1 worked around, 14-rank runs (7 cards) crash at the first metric sync: at train step 1, or at the first eval round with `train_eval_loop.eval_frequency = 20`. This happened in 4 out of 4 runs. A single 14-rank run without eval passed, but only by luck.

**Where:**
- `MetricsLogger.compute → torchmetrics sync → RecMetricComputation._sync_dist`. The first sync of each metric runs `_resolve_fixed_shape_sync → invoke_on_rank_and_broadcast_result → dist.broadcast_object_list` over XCCL, with the same oneCCL stack as above.
- When that one-time check was skipped, the crash moved to `barrier` and `all_gather` inside `gather_all_tensors`.
- `torch.xpu.synchronize()` before the sync didn't help.

**Workaround** (XPU only):
- `train_ranker.py`: both `MetricsLogger`s use `device=cpu` and one `dist.new_group(backend="gloo")`.
- `dlrm_v3/utils.py`: `MetricsLogger` takes an optional `process_group`, passes it to every torchrec metric, and moves its inputs to its device in `update()`.
- `checkpoint.py`: restored metric states are moved back to the metrics' device.

**Cost:** one small device-to-CPU copy per step (a synchronization point), and slower metric syncs every `metric_log_frequency` steps. Not measured.

## Problem 3: profiler on XPU

The branch's `train/gin/movielens_1m_xpu.gin` set `train_eval_loop.output_trace = True`; v1.8.0 had `False`. The profiler starts tracing XPU kernels (PTI) after 10 steps, and multi-rank runs then print `zeEventQueryKernelTimestamp returned: 1` warnings and misbehave. Keep `output_trace = False` on XPU.

## Problem 4: the workaround's allreduce at 16 ranks

**Symptom:** with both workarounds in place, 16-rank runs (all 8 cards, both tiles each) segfault before any training metrics are logged. This happened in 2 out of 2 runs, about 36 s after start.

**Where:**
- In Python, rank 0 is in `_average_grads_pre_step → dist.all_reduce` (`train/utils.py`), the workaround's post-backward dense-gradient allreduce, during the first `optimizer.step()`.
- Under gdb, the crashing rank has the same oneCCL worker stack as above (`deps_entry::update → native_event_impl::test → urEventGetInfo`), with frame #0 at an invalid address.
- The kernel logged no segfault lines during these runs, so the signature rests on the gdb stack alone.

**By rank count**, with the real model, 4 epochs, using `train/tests/xpu_e2e_multirank_check.py`:

| Ranks | Cards | Result |
|---|---|---|
| 14 | all except card 6 | pass |
| 14 | all except card 7 (card 6 in) | pass |
| 15 | tiles 0–14 | pass |
| 16 | all | SIGSEGV, 2 out of 2 |
| 16, with `CCL_ALLREDUCE=ring` | all | SIGSEGV, same place |

Every card, including card 6 and both tiles of card 7, is part of a passing configuration, so this isn't one bad card. In the tested configurations the crash starts at 16 ranks. That fits a scale-dependent trigger, as in uxlfoundation/oneCCL#224 (more than about 9 communicators per process), but we haven't counted communicators, so it's a hypothesis.

**Tried:**
- `CCL_ALLREDUCE=ring`, the workaround in intel/torch-xpu-ops#5164. oneCCL confirmed the setting (`value of CCL_ALLREDUCE changed to be ring`), but the run still crashed in the same place.

**Not tried:** averaging the dense gradients over a gloo group, as the metrics workaround does. It would avoid XCCL for that collective at the cost of a CPU copy of the dense gradients every step. The embedding `all_to_all_single` calls would still use XCCL.

**Current guidance:** use at most 15 ranks.



## Tests

The current XPU test directory contains two real-model end-to-end drivers and a
CPU-only regression test for the comparison logic. The earlier
`xpu_workaround_consistency_test.py`, `xpu_ddp_crash_repro_test.py`, and
`xpu_dist_test_utils.py` helpers referenced in older notes are not present in
the current directory.

**Consistency against the original path** (`train/tests/xpu_e2e_ddp_consistency.py`).
This MovieLens-1M `train-eval` check is intended for 2 ranks, where the original
DDP path works. It runs a seeded reference using torchrec's default DDP wrapper
and XPU metrics, then the current workaround. By default it runs five
references and one workaround, compares all ten reference pairs to calibrate
the run-to-run noise, and checks matching sample counts and metric keys. NE,
dense weights, and embedding checksums are gated against the calibrated
tolerance (or absolute floors); accuracy and GAUC are informational. Set
`--reference-runs 2` for a single-pair calibration, or `--reference-runs 1` to
use only absolute tolerance floors. The summary includes maximum-error verdicts
and diagnostics such as median/p95 errors, RMS errors, and worst locations.
Reference pairs share runs, so the calibrated envelope is not a statistical
confidence bound; unusually large reference differences produce a warning,
not a separate verdict. RMS and percentile diagnostics do not replace the
maximum-error gates.

```bash
ZE_AFFINITY_MASK=0,1 python -m \
  generative_recommenders.dlrm_v3.train.tests.xpu_e2e_ddp_consistency \
  --world-size 2 --steps 100 --out-dir /tmp/dlrmv3_consistency
```

**CPU-only comparison regression tests**:

```bash
python -m pytest generative_recommenders/dlrm_v3/train/tests/xpu_e2e_ddp_consistency_test.py
```

These test the comparison and calibration logic only; they do not run XPU
training.

**Multi-rank check with the real model** (`train/tests/xpu_e2e_multirank_check.py`).
The original DDP path can't run at 4 or more ranks, so there is no reference to
compare against at scale. This runs the current MovieLens-1M training code on
`--world-size` ranks (all visible XPUs and 4 epochs by default) using
`xpu_e2e_ddp_consistency.py`'s child mode. It checks:
1. completion: the child exits successfully, writes `metrics.jsonl`, and saves
   `rankN.pt` results for every rank;
2. rank sync: final dense weights match rank 0 exactly by default
   (`--sync-tol 0`);
3. learning: the last recorded training window NE is at least
   `--min-ne-drop` (default 5%) below the first, and all recorded metrics are
   finite.

`--baseline-world-size N` optionally runs another rank count and displays the
final train/eval NE side by side; this is informational because effective
batch sizes differ. A failed baseline is reported but does not change the
primary run's verdict. The script writes `summary.json`, kills the run's
process group on completion or timeout, and exits 0 when all checks pass, 1
when a check fails, or 2 when the primary run fails.

```bash
python -m generative_recommenders.dlrm_v3.train.tests.xpu_e2e_multirank_check \
  --world-size 15 --out-dir /tmp/dlrmv3_multirank
```

Results on 2026-10-20: 14 ranks (twice, with and without card 6) and 15 ranks pass. All dense tensors were bit-identical across ranks, and training NE fell about 27% (1.022 to 0.74–0.75). 16 ranks fail with Problem 4.

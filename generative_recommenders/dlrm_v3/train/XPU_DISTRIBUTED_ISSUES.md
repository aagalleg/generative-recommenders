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

## Problem 3: profiler on XPU

The branch's `train/gin/movielens_1m_xpu.gin` set `train_eval_loop.output_trace = True`; v1.8.0 had `False`. The profiler starts tracing XPU kernels (PTI) after 10 steps, and multi-rank runs then print `zeEventQueryKernelTimestamp returned: 1` warnings and misbehave. Keep `output_trace = False` on XPU. `FINDINGS.md` A-04 on the v1.8.0 branch describes a related profiler problem at shutdown.

## Operational notes

- **A card failed once.** During the 8-card runs on 2026-10-14, the card at PCI `0000:99:00.0` (card 6, flat tiles 12–13) logged stalled compute engines, then `GUC: Engine reset request failed` and `Failed to reset chip`. `xpu-smi discovery` then hung. After a later reboot it was rechecked on 2026-10-20: it initialized cleanly, its firmware status is normal, both tiles compute correctly, and it was part of the passing 14- and 15-rank runs. To leave a card out, use `ZE_AFFINITY_MASK` (flat tile N is card N÷2), e.g. `0,1,2,3,4,5,6,7,8,9,10,11,14,15` without card 6. Check `dmesg` after multi-rank runs.
- **The machine rebooted three times on 2026-10-16.** At least one of those reboots happened while no GPU work was running, so the cause is unknown.
- **Orphaned ranks:** after a crash or kill, `mp.spawn` ranks can survive with parent PID 1, holding about 20 GB per tile and spinning. Check for `spawn_main` processes before trusting a new run, and remember that `timeout` only kills the launcher.
- **Device numbering:** in the default FLAT mode PyTorch sees 16 devices (one per tile). `xpu-smi` numbers cards: use `xpu-smi dump -d <cards> -t 0,1 -m 0,18` to see per-tile utilization and memory. `N/A` means the tile is idle or power-gated.
- **Imports:** running `train_ranker.py` directly needs `PYTHONPATH=<repo root>`.

## Tests

Both tests use dlrm_v3's real `setup()`, `make_optimizer_and_shard()` and `MetricsLogger` with a small synthetic torchrec model. The shared code is in `train/tests/xpu_dist_test_utils.py`. The "reference" mode is the original code path: the workaround patched back to torchrec's default DDP wrapper.

**1. Consistency** (`train/tests/xpu_workaround_consistency_test.py`). On 2 ranks, where DDP works, it trains the reference and the workaround from the same seed and data and checks:
- per-step losses and final dense and embedding weights match;
- dense weights are bit-identical across ranks;
- CPU/gloo metrics equal XPU/XCCL metrics.

It takes about 20 s. Planting a bug (summed instead of averaged gradients) makes the weight check fail: 47% of a tensor's elements differ, by up to 157× relative.

```bash
PYTHONPATH=$PWD ZE_AFFINITY_MASK=0,1 \
  python -m pytest generative_recommenders/dlrm_v3/train/tests/xpu_workaround_consistency_test.py
```

**2. Crash reproduction** (`train/tests/xpu_ddp_crash_repro_test.py`, opt-in). On 4 or more ranks, each mode runs in its own subprocess. The reference must crash with SIGSEGV or hang, and the workaround must finish. If the reference starts passing, the crash is probably fixed in the stack and the workarounds can be reconsidered. It's opt-in because these crashes have already led to a card that could not reset.

```bash
DLRMV3_XPU_CRASH_REPRO=1 PYTHONPATH=$PWD ZE_AFFINITY_MASK=0,1,2,3 \
  python -m pytest generative_recommenders/dlrm_v3/train/tests/xpu_ddp_crash_repro_test.py
```
Set `DLRMV3_XPU_REPRO_WORLD_SIZE` to use more ranks (default 4).

The tests were run with `CCL_ATL_TRANSPORT=ofi FI_PROVIDER=tcp CCL_ZE_IPC_EXCHANGE=pidfd`.

**End-to-end checks with the real model** (MovieLens-1M):
- 2 ranks, seeded, against the original code: train and eval metrics match to within the run-to-run spread. The largest window-NE difference was 6e-4 for the port, while two runs of the original differed by up to 4.4e-3.
- 14 ranks on 7 cards with `eval_frequency = 20`: 2 out of 2 runs completed all 4 epochs, with 4 eval rounds each.

The repeatable driver, `train/tests/xpu_e2e_ddp_consistency.py`, defaults to five
reference runs and one workaround run. It compares all ten reference pairs and
uses their largest difference per quantity as the noise floor. Reference sample
counts and metric keys must match; a mismatch invalidates calibration. Maximum
errors remain gated against `max(factor * noise, absolute_floor)`, while accuracy
and GAUC remain informational. `--reference-runs 2` retains single-pair calibration;
`--reference-runs 1` uses only absolute tolerance floors.

Console output reports reference min/median/max spreads and compact new-run
diagnostics. `summary.json` also stores every pair's diagnostics, NE median/p95
and worst metric/step, dense RMS and per-parameter maxima, and each embedding
shard's abs-sum/square-sum relative errors. A reference maximum greater than
`factor` times its median pair difference triggers an outlier-sensitivity warning,
not a separate verdict. These shared-run pairs are not independent observations,
and the envelope is not a statistical confidence bound. RMS and percentiles are
diagnostics only; they cannot hide an isolated maximum-error failure.

```bash
ZE_AFFINITY_MASK=0,1 python -m \
  generative_recommenders.dlrm_v3.train.tests.xpu_e2e_ddp_consistency \
  --world-size 2 --steps 100 --out-dir /tmp/dlrmv3_consistency
```

CPU-only comparison/calibration regression tests:

```bash
python -m pytest generative_recommenders/dlrm_v3/train/tests/xpu_e2e_ddp_consistency_test.py
```

**Multi-rank check with the real model** (`train/tests/xpu_e2e_multirank_check.py`). The original DDP path can't run at 4 or more ranks, so there's no reference to compare against at scale. This script trains the current code on N ranks (default: all visible XPUs, 4 epochs) and checks properties that correct data-parallel training must have:
1. completion: the run exits 0 and every rank saves its results;
2. rank sync: final dense weights are bit-identical on every rank;
3. learning: training NE drops by at least `--min-ne-drop` (default 5%), and all metrics are finite.

`--baseline-world-size N` adds a run at N ranks and shows final NE side by side, for information only. It reuses `xpu_e2e_ddp_consistency.py`'s `--child` mode, and kills each run's process group when the run ends. Exit codes: 0 pass, 1 a check failed, 2 the run failed.

```bash
python -m generative_recommenders.dlrm_v3.train.tests.xpu_e2e_multirank_check \
  --world-size 15 --out-dir /tmp/dlrmv3_multirank
```

Results on 2026-10-20: 14 ranks (twice, with and without card 6) and 15 ranks pass. All dense tensors were bit-identical across ranks, and training NE fell about 27% (1.022 to 0.74–0.75). 16 ranks fail with Problem 4.

## Open issues

- **Report upstream** to the PyTorch XPU / oneCCL maintainers. The reproduction in `train/tests/xpu_dist_test_utils.py` (`--mode reference_ddp --world-size 4`) and the stack above are enough for a bug report.
- **16 ranks (Problem 4):** the workaround's dense allreduce crashes at 16 ranks. Averaging the dense gradients over gloo is untested; `CCL_ALLREDUCE=ring` didn't help.
- **Other collectives:** the embedding `all_to_all_single` calls and the startup broadcasts still use XCCL. None of them crashed in these runs, but that doesn't prove they are safe, especially at 16 ranks.
- **Not tested:** throughput impact, checkpoint save and load, and the streaming training loops (they need the internal `streaming-*` datasets).
- **A separate hang:** 2 ranks with row-wise sharding forced hang after step 0. This is not covered by the workarounds.
- **When to remove the workarounds:** once the bug is fixed upstream, revert them to get DDP's overlap and on-device metrics back. The crash reproduction test signals when.

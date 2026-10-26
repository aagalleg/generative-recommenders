# DLRM-v3 on XPU — Findings

What the XPU run harness (`run_dlrm_v3_xpu.sh`) and the investigations behind
it have found missing or wrong. One entry per finding: symptom, root cause,
fix or workaround, status on the current pins, evidence, owner.

**Status checked against** (2026-10-26, branch
`probes/dlrm-v3-xpu-fbgemmv1.9.0`): torch `2.14.1+xpu`, fbgemm-gpu-cpu
`1.9.0`, fbgemm-xpu `0.9.0` built from torchlib-xpu `264f0c3` (the
`install_dlrmv3.sh` pin), torchrec `1.10.0a0+c0fc555` (the pin), and
generative-recommenders at the tip of this branch, on one tile of a PVC
(Max 1550), oneAPI 2026.1. The `.so` files are stamped 2026-10-20 18:49. The
torchlib-xpu reflog shows its checkout was at `264f0c3` from 18:47 that day and
only moved to `main` (`2655c44`) on 2026-10-23, with no rebuild. The torchrec
checkout likewise moved from `c0fc555` to `56c531e` (a docstring-only commit)
after the install. The run manifests on this host therefore name the wrong
torchlib-xpu SHA ([H-16](#h-16-the-manifest-records-the-torchlib-xpu-checkout-not-the-revision-the-so-was-built-from)).
This replaces the previous check (2026-09-29: torchlib-xpu `b59acf9`,
fbgemm-gpu-cpu `1.8.0`, torch 2.13, torchrec `e213eb9`, generative-recommenders
`ed103ac`; plus B60 harness runs of 2026-09-30, which H-10 still cites).

**What this branch carries.** It is based on the v1.9.0 minimal XPU port, not
on the harness branch of
[PR #1](https://github.com/aagalleg/generative-recommenders/pull/1). The run
harness of PR #1 has been added to it, with its inference half:
`run_dlrm_v3_xpu.sh` (phases 1–3, `--device cpu`, `--mode infer`; phase 3
guarded), `write_run_manifest.py` (`manifest.json`),
`compare_loss_curves.py`, `dlrm_v3/harness.py` (`losses.csv`,
`operative_config.gin`, `seed_everything`, `apply_size_overrides`), the
`torch.accelerator` port of `inference/model_family.py`, and
`inference/gin/movielens_1m_xpu.gin`. `train_ranker.py` follows PR #1
(`--device_type`, `--world_size`, `--gin_config_file`, `torchrun` path), keeping
this branch's HSTU-Ultra flags and gloo metrics group;
`inference/main.py` gained `--gin_config_file`. Not carried: the
gin-configurable `get_hstu_configs` (and so the model-size bindings of the
`movielens_1m_xpu.gin` files), and PR #1's `common.py`, root `main.py` and
`datasets/dataset.py` changes (the runs below did not need them).
`apply_size_overrides` differs from PR #1: it acts only when
`HSTU_EMBEDDING_DIM` or `HASH_SIZE` is set (the launcher always sets them), so
a direct `train_ranker.py` run keeps the stock 512 / 10M sizes. Setup and the
one command are in the [README](README.md#dlrm-v3-on-intel-xpu-research-harness).
So "status" below is one of: *re-measured* (probe or run on 2026-10-26),
*code read* (not run), or *not re-checked*.

**End-to-end checks on this branch.** (1) `train_ranker.py --dataset debug
--mode train` (`debug.gin`: 10 batches, `output_trace = True`), one tile: exit
0, ten `train - Step` lines with finite NE (1.06 at step 2, 1.38 at step 10),
no NaN in the log, all three tables `TW` / `fused` (`Using fused
exact_row_wise_adagrad`), no SIGABRT at exit. (2) The launcher, phase 1 `--mode
train --device xpu`, 10 steps of MovieLens-1M (`train_loop.num_batches = 10`,
dims 64 / 1M rows): exit 0, `manifest.json`, `operative_config.gin`, `run.gin`
and `losses.csv` written. The same with `--device cpu`, and a phase 2 repeat
(`torchrun`, one XPU), also exit 0. `compare_loss_curves.py`: XPU vs CPU
mean relative difference 10.6%, Pearson 0.932 (10 steps); XPU vs its phase 2
repeat 0.01% mean relative, max abs 3.8e-5; "problems: none". Phase 3 was not
run; its guard was checked (exits 1 without `DLRMV3_ALLOW_MULTI_TILE=1`).
(3) **Unquantized inference runs end to end on XPU.** `run_dlrm_v3_xpu.sh
--mode infer`, random weights, Server scenario: exit 0, 256 queries
processed, 0 `thread: failed` lines (see H-05), LoadGen's only error entry is
the expected `Max query count reached`; result INVALID because the run is
shorter than `min_duration` (expected). The dense HSTU model runs on `xpu:0`
in bf16 with the default Triton kernels. Training forces the PYTORCH kernel on
XPU (Triton backward crashes), but inference only needs the forward, so it
uses Triton. Inference also keeps the stock `max_seq_len=16384`; training on
XPU caps it at 256. The sparse lookup runs on CPU (H-04). A second run with
`run.compute_eval = True` (LoadGen AccuracyOnly, 1510 MovieLens-1M eval
samples): exit 0, all predictions finite (0 NaN / Inf, range 0.19–0.68), and
`inference/accuracy.py` reports gAUC 0.50 and NE 1.11, as expected for random
weights. Not tested: inference from a checkpoint trained by the harness, and
quantized inference (it arrives with the quantized lookup work).

**Two series.**
- **A- entries** come from the first end-to-end DLRM-v3 trial on PVC. That
  trial ran on earlier XPU branches of generative-recommenders, TorchRec and
  torchlib-xpu, not on the current pins, so each A- entry says whether it
  still holds on the pins. Their evidence is a probe in [`probes/`](probes/)
  (index: [PROBES.md](PROBES.md)) or a reproduction on the pins.
- **H- entries** were found while building and running this harness. Each
  gives a command that reproduces it; every harness run folder carries a
  `manifest.json` with the exact revisions.

**Owners** are component-level until names are assigned.

## Index

| ID | Finding | Status on current pins | Owner |
| --- | --- | --- | --- |
| [A-01](#a-01-fp16-embedding-tables-train-to-nan--optimizer-epsilon-not-the-device) | FP16 tables train to NaN — optimizer epsilon, not the device | Not reached; root cause unfixed (dtype guard still missing) | generative-recommenders fork |
| [A-02](#a-02-hstu_-overrides-silently-inert) | `HSTU_*` overrides silently inert | Fixed for the two supported variables; the other dims are not configurable here | harness |
| [A-03](#a-03-two-rank-cross-tile-xccl-took-the-card-off-the-pcie-bus) | Two-rank cross-tile XCCL took the card off the PCIe bus | **Open — hazard present**; launcher phase 3 guarded, other multi-tile paths not; not re-run | harness; platform/driver |
| [A-04](#a-04-profiler-aborts-the-process-on-teardown) | Profiler aborts the process on teardown | **Fixed on this branch** (`Profiler.__del__` calls `stop()`); torch behaviour unchanged | generative-recommenders fork |
| [A-05](#a-05-cpu-on-the-xpu-sharding-plan-fails-in-construct_jagged_tensors) | CPU on the XPU sharding plan fails in `construct_jagged_tensors` | Not reached | torchrec |
| [A-06](#a-06-xpu-was-dense-kernel-only-data_parallel-sharding) | XPU was DENSE-kernel only (DATA_PARALLEL sharding) | Superseded for training; DENSE TBE module still fails on XPU (FBGEMM 1.9.0 Python) | FBGEMM / TorchRec (DENSE path, low) |
| [A-07](#a-07-host-user-mode-driver-too-old-for-torch-213) | Host user-mode driver too old for torch 2.13 | Handled by container; torch 2.14.1 runs | platform |
| [A-08](#a-08-oneccl-needs-devdriby-path-inside-containers) | oneCCL needs `/dev/dri/by-path` inside containers | Handled | platform |
| [A-09](#a-09-movielens-download-failed-expired-tls-certificate) | MovieLens download failed: expired TLS certificate | Resolved externally | — |
| [H-01](#h-01-xpu-kernels-are-only-registered-by-an-explicit-import-fbgemm_xpu) | XPU kernels only registered by an explicit `import fbgemm_xpu` | Worked around (still needed) | torchlib-xpu |
| [H-02](#h-02-fbgemm_xpu-extensions-have-no-rpath) | `fbgemm_xpu` extensions have no RPATH | Not reached (RPATH still missing) | torchlib-xpu |
| [H-03](#h-03-the-fbgemm-wheel-is-patched-in-place) | The FBGEMM wheel is patched in place | Open (by design; 1.9.0 still lacks `XPU` upstream) | FBGEMM / TorchRec upstreaming |
| [H-04](#h-04-unquantized-inference-does-its-embedding-lookup-on-cpu) | Unquantized inference does its embedding lookup on CPU | Open (re-measured: inference runs; sparse on CPU) | quantized embedding lookup work |
| [H-05](#h-05-the-inference-runner-swallows-exceptions) | The inference runner swallows exceptions | Open (not hit in the 2026-10-26 runs) | generative-recommenders upstream |
| [H-06](#h-06-multithreaddataproducer-is-cuda-only) | `MultiThreadDataProducer` is CUDA-only | Worked around | generative-recommenders upstream |
| [H-07](#h-07-offline-scenario-ignores-runnum_queries) | Offline scenario ignores `run.num_queries` | Worked around | generative-recommenders upstream |
| [H-08](#h-08-inverted-metric-log-check-in-train_loop-and-eval_loop) | Inverted metric-log check in `train_loop` and `eval_loop` | Worked around (training loss, via `losses.csv`); bug itself open | generative-recommenders upstream |
| [H-09](#h-09-tensorboard-log-directory-is-fixed-and-shared) | TensorBoard log directory is fixed and shared | Open (code read) | generative-recommenders fork |
| [H-10](#h-10-seeded-xpu-training-is-not-bit-reproducible) | Seeded XPU training is not bit-reproducible | Open (low); re-measured over 10 steps | torchlib-xpu |
| [H-11](#h-11-launcher-sourced-a-version-pinned-oneapi-path-silently) | Launcher sourced a version-pinned oneAPI path silently | Fixed in the launcher (sourcing path not exercised) | harness |
| [H-12](#h-12-mlperf-loadgen-build-writes-into-its-source-tree) | MLPerf LoadGen build writes into its source tree | Not re-checked | harness |
| [H-13](#h-13-two-oneapi-versions-in-one-process) | Two oneAPI versions in one process | Informational; not re-checked | platform |
| [H-14](#h-14-dlrm_v3-is-not-packaged-by-setuppy) | `dlrm_v3` is not packaged by `setup.py` | Open (re-checked) | generative-recommenders upstream |
| [H-15](#h-15-torchautocast-is-hard-coded-to-cuda-in-the-model-modules) | `torch.autocast` is hard-coded to `"cuda"` in the model modules | **Open on this branch** (new, 2026-10-26) | generative-recommenders fork |
| [H-16](#h-16-the-manifest-records-the-torchlib-xpu-checkout-not-the-revision-the-so-was-built-from) | The manifest records the torchlib-xpu checkout, not the revision the `.so` was built from | **Open** (new, 2026-10-26) | harness |

---

## A- findings: first end-to-end trial

### A-01 FP16 embedding tables train to NaN — optimizer epsilon, not the device

**Symptom.** Sharded training on XPU produced `loss = nan` from batch 0; every
embedding table was 100% NaN immediately after `DistributedModelParallel`.

**Root cause.** `KeyedOptimizer.init_state` sets `param.grad = zeros_like(param)`
and calls `step()`. Adam's `eps=1e-8` is below the smallest fp16 subnormal
(5.96e-8) and rounds to zero, so a zero-gradient update is `0 / (sqrt(0) + 0)` =
NaN. It reproduces identically on CPU; it is a dtype bug. XPU met it because the
TorchRec branch used in the trial gave XPU only the DENSE kernel, which put the fp16 tables in the
dense Adam group. In normal training every row a batch does not touch has a zero
gradient, so the table is poisoned on the first step regardless of `init_state`.

**Ruled out.** TBE buffer initialisation, `uniform_`/`normal_` for fp16
on XPU, uninitialised memory (~1.5% NaN, not 100%).

**Workaround used in the trial.** FP32 tables on XPU (the trial branch's
`HSTU_TABLE_DTYPE=FP32`), with both optimizer workarounds kept.

**Status on current pins.** *Not reached, root cause unfixed.* Re-measured
2026-10-26 on fbgemm-gpu-cpu 1.9.0 / torchrec `c0fc555`. The TorchRec gives XPU a
table-wise plan with fused RowWiseAdagrad (`TW` / `fused`, `Using fused
exact_row_wise_adagrad`), so the tables never enter the dense Adam group and
FP16 tables train with finite loss: the 10-step `debug` run has no NaN, and the
sharding-path probes measure 0 NaN after init, after sharding and in the first
forward, on XPU and CPU (XPU aux loss 0.138161, CPU 0.139193, 71 predictions,
0 NaN; see [PROBES.md](PROBES.md)). Any fp16 parameter in a dense Adam group
would still turn NaN; measured with
[`probe_nan_fp16_adam_eps.py`](probes/probe_nan_fp16_adam_eps.py): fp16 with a
partly-zero gradient gives 1016/1024 NaN on XPU, fp32 gives 0; the zero-gradient
fp16 step with `eps=1e-8` gives 1024/1024 NaN on CPU as well, so it is a dtype
problem. The earlier note that the pinned `train/utils.py`
skips `init_state` and `apply_optimizer_in_backward` on XPU **no longer holds
on this branch**: `make_optimizer_and_shard` calls `apply_optimizer_in_backward`
for the sparse parameters and `init_state` on every device. XPU instead skips
DDP (a grad-averaging wrapper replaces it) and rejects `DATA_PARALLEL` plans.
The two workaround comments the finding called wrong are therefore gone.

**Evidence.** Probes, in the order they narrowed it down: `probe_nan_tbe_init_alias.py`,
`probe_nan_rng_initialisers.py`, `probe_nan_shard_weight_alias.py`,
`probe_nan_init_bounds.py`, `probe_nan_corruption_stage.py`,
[`probe_nan_fp16_adam_eps.py`](probes/probe_nan_fp16_adam_eps.py) (the
reproduction), all in [`probes/`](probes/); see [PROBES.md](PROBES.md).
Current plan: `run.log` of any training run (planner table).

**Owner.** generative-recommenders fork; a guard for fp16 + dense Adam belongs
upstream.

### A-02 `HSTU_*` overrides silently inert

**Symptom.** Runs believed to be configured through `HSTU_*` environment
variables ran on stock config; the first visible sign was a 31.8 GiB allocation
failure from the stock `max_seq_len=16384`.

**Root cause (trial branch).** `train_ranker.py` monkeypatched
`configs.get_hstu_configs`, but `train/utils.py` had imported the function
earlier and kept its own reference, so every override was ignored.

**Status on current pins.** *Fixed for the two supported variables; the
rest is ignored (re-measured 2026-10-26).* `dlrm_v3/harness.py:apply_size_overrides`
reads `HSTU_EMBEDDING_DIM` and `HASH_SIZE` (when either is set; the launcher
always sets both, 64 and 1,000,000), and their effective values are written to
`operative_config.gin` (verified in the launcher run). Without them the stock
512 / 10,000,000 are kept: they are module constants in `dlrm_v3/configs.py`.
`get_hstu_configs` is still not `@gin.configurable` on this branch, so the
other HSTU dims (heads, layers, attention and transducer widths) cannot be
set from gin or the environment; the launcher run builds the stock 5-layer,
512-wide transducer with a 64-wide table (`DlrmHSTUConfig` in `run.log`). The
only other size control is `make_model` capping `max_seq_len` at 256 when an
XPU is present (the PYTORCH attention kernel is O(N²) in memory). The
model-building probes set the config fields in-process for that reason (see
[PROBES.md](PROBES.md)). The remaining trial-branch variables
(`HSTU_MAX_SEQ_LEN`, `HSTU_TABLE_DTYPE`, `HSTU_NUM_HEADS`, `DLRM_V3_SEED`, …)
do not exist: setting them has no effect. They are not rejected;
`manifest.json` records every `HSTU_*` / `CCL_*` / `FI_*` / `ZE_*` variable
under `env`, so a mismatch is visible by comparing it with
`operative_config.gin`.

**Evidence.** Reproduced on 2026-09-28 on the harness branch before its fix: a
CPU run with `HSTU_EMBEDDING_DIM=64 HASH_SIZE=1000000` built 512-dim / 10M-row
tables and failed in the planner. On this branch the 10-step `debug` run
without the variables builds the 512-dim / 10M-row tables
(`SplitTableBatchedEmbeddingBagsCodegen` with `(10000000, 512, ...)` in the
run log); the launcher run builds `hstu_embedding_table_dim=64`.

**Owner.** harness (make `get_hstu_configs` gin-configurable as in PR #1;
consider rejecting unknown `HSTU_*` variables).

### A-03 Two-rank cross-tile XCCL took the card off the PCIe bus

**Symptom.** A 2-rank `torchrun` job on the two tiles of one Max 1550 reached
its first heavy collective and took the GPU down:
`iaf i915.iaf.31: PCIE bus error detected. Device has been disabled`, then
`AER: device recovery failed`. After a warm reboot the card no longer
enumerated; a cold power cycle at minimum is needed. No replacement was
available.

**Root cause.** Sustained cross-tile traffic over the Xe Link fabric (`iaf`)
driven by XCCL through Level Zero IPC, on a 2023 kernel driver and Linux 5.15
without pidfd support. Small collectives passed; real training traffic did not.
(A preceding MPI failure, `Duplicate ranks in rank array`, was fixed on the
trial branch.)

**Status on current pins.** **Open — not re-tested; treat the hazard as
present, now guarded in the launcher only.** `run_dlrm_v3_xpu.sh --phase 3`
runs XCCL across all detected devices (`NUM_XPUS`) with
`CCL_ZE_IPC_EXCHANGE=pidfd`, the same cross-tile Level Zero IPC traffic. In
this repo's copy of the launcher, phase 3 now refuses to start unless
`DLRMV3_ALLOW_MULTI_TILE=1` is set (checked 2026-10-26: it exits 1 with an
explanation); PR #1's launcher has no such guard. Direct `torchrun` calls,
`/opt/dlrmv3/run_dlrm_v3_xpu.sh --world-size N`, and `train_ranker.py
--world_size N` are not guarded. The host used on 2026-10-26 differs from the
trial's: Linux 6.8 (so `pidfd_open` works here, checked), not 5.15, and 16
tiles visible. Whether the fabric failure is specific to the trial's
2023 kernel driver and kernel 5.15 is not known, and nobody has run sustained cross-tile
XCCL on this host. The trial's escape, `FBGEMM_XPU_FORCE_GLOO=1` (collectives
staged through the host), does not exist in the pinned trees. Do not run a
multi-tile job on a PVC without a recoverable card. The
`ccl` probes were **not run** on 2026-10-26 for that reason.

**Evidence.** The kernel log lines above. Probes (do not run them across tiles):
[`probes/probe_ccl_2rank_transport.sh`](probes/probe_ccl_2rank_transport.sh),
[`probes/probe_ccl_2rank_config_matrix.sh`](probes/probe_ccl_2rank_config_matrix.sh).

**Owner.** harness (guard phase 3; two-rank via a host-staged gloo backend);
platform/driver for the fabric failure.

### A-04 Profiler aborts the process on teardown

**Symptom.** `train_loop.output_trace = True` causes `SIGABRT` at interpreter
exit on XPU.

**Root cause.** The `Profiler` wrapper only calls `step()`, never `start()` /
`stop()`; Kineto arms itself at the WARMUP step and is never disarmed, and on
XPU the destructor raises `PTI_ERROR_NOT_IMPLEMENTED` through a `noexcept`
path. The activity list is hard-coded to `CPU, CUDA`, so no device data is
collected on XPU.

**Status on current pins.** *Fixed on this branch.* `dlrm_v3/utils.py:Profiler`
now selects `[CPU, XPU]` on an XPU host (`_profiler_activities`) and calls
`stop()` from `__del__` (the comment cites the PTI `clearActivities` abort).
Re-measured 2026-10-26 on torch 2.14.1: the 10-step `debug` run has
`output_trace = True` and exits 0 with no abort. The underlying torch/Kineto
behaviour is unchanged: `probe_profiler_sigabrt_fix.py` still ends in SIGABRT
(`XpuptiActivityApi::clearActivities`) for the `xpu_nostop` case and exits
cleanly with `stop()`. Any new code that builds a `torch.profiler.profile` on
XPU and never calls `stop()` hits it.

**Evidence.** [`probes/probe_profiler_activity_teardown.py`](probes/probe_profiler_activity_teardown.py),
[`probes/probe_profiler_sigabrt_fix.py`](probes/probe_profiler_sigabrt_fix.py).

**Owner.** generative-recommenders fork.

### A-05 CPU on the XPU sharding plan fails in `construct_jagged_tensors`

**Symptom.** Forcing the CPU run onto the trial's XPU plan (DP / dense) failed
with `split_with_sizes expects split_sizes to sum exactly to 96` in
`torchrec/modules/utils.py:construct_jagged_tensors`. Root cause not
investigated.

**Status on current pins.** Not reached (re-measured 2026-10-26: the sharded
`cpu` runs of `probe_nan_after_sharding.py` and `probe_nan_init_bounds.py`
build and forward without error): CPU and XPU both get `TW` / `fused`,
which also makes a CPU-vs-XPU loss comparison like-for-like in
sharding and optimizer (it was not in the trial).

**Owner.** torchrec (low priority).

### A-06 XPU was DENSE-kernel only (DATA_PARALLEL sharding)

**Symptom (trial branch).** XPU exposed only the DENSE compute kernel, so the planner rejected
`table_wise` (`DENSE kernel requires DATA_PARALLEL`), and
`DenseTableBatchedEmbeddingBagsCodegen.__init__` called
`torch.cuda.current_device()` (fixed on the trial branch of torchlib-xpu with `_tbe_dense_compat.py`).

**Status on current pins.** Superseded for training: the pinned TorchRec shards
XPU table-wise with the fused `SplitTableBatchedEmbeddingBagsCodegen`; the dense
constructor is not used. Re-measured 2026-10-26. The DENSE *op* is now there:
torchlib-xpu registers `fbgemm::dense_embedding_codegen_lookup_function` under
`AutogradXPU` (`fbgemm_dense_lookups_ops.cpp`), plus the no-bag forward/backward
kernels. The DENSE *module* is still broken on XPU:
[`probe_kernel_tbe_dense_nobag.py`](probes/probe_kernel_tbe_dense_nobag.py)
and `probe_nan_tbe_init_alias.py` fail with `Torch not compiled with CUDA
enabled`. The cause is in the installed FBGEMM 1.9.0 Python:
`DenseTableBatchedEmbeddingBagsCodegen.__init__` calls
`torch.cuda.current_device()` for any non-CPU, non-MTIA device
(`split_table_batched_embeddings_ops_training.py:5421`). The trial's
`_tbe_dense_compat.py` fix never reached torchlib-xpu, and it would not help
here: the failing line is in `fbgemm_gpu`, not in fbgemm-xpu.

**Owner.** FBGEMM / TorchRec (the constructor); low priority while training
uses the fused path.

### A-07 Host user-mode driver too old for torch 2.13

**Symptom.** Bare metal with torch 2.13.0+xpu: SIGSEGV in `get_device_name(0)`
(torch 2.6 works). The old user-mode driver also hides PVC's matrix engines.

**Status.** Handled: all runs use a container with a current user-mode driver
(`.devcontainer/xpu/Dockerfile.xpu` in the XPU dev-tools repository). On
2026-10-26, torch `2.14.1+xpu` sees the Max 1550 (driver `1.6.33578+77`, matrix
MMA and bf16 convert reported) and a bf16 matmul runs
(`probe_env_xpu_devices.py`).

**Owner.** platform.

### A-08 oneCCL needs `/dev/dri/by-path` inside containers

**Symptom.** Process-group init dies with
`opendir failed: could not open device directory` when the container gets
`--device /dev/dri` but no `by-path` symlinks.

**Status.** Handled: mount `-v /dev/dri:/dev/dri`, or recreate the links with
`fix-dri-by-path.sh` (XPU dev-tools repository). Present on the current host.

**Owner.** platform.

### A-09 MovieLens download failed: expired TLS certificate

**Symptom.** `files.grouplens.org` served a certificate that expired 2026-08-28;
the trial used the synthetic `debug` dataset instead.

**Status.** Resolved externally: download returns HTTP 200 with a verified
certificate (2026-09-29). Harness runs use MovieLens-1M.

**Owner.** —

---

## Harness findings

### H-01 XPU kernels are only registered by an explicit `import fbgemm_xpu`

**Symptom.** First inference run: `NotImplementedError: The operator
'fbgemm::asynchronous_complete_cumsum' is not currently implemented for the XPU
device`, although torchlib-xpu has that kernel.

**Root cause.** `fbgemm_xpu` registers its XPU kernels on import, and nothing
imports it automatically. Only the training path imported it. Import order
does not matter on the pins: `fbgemm_xpu/__init__.py` imports `fbgemm_gpu`
itself before loading its extension. (An abort when importing it first was
seen on an earlier torchlib-xpu; it does not reproduce at `b59acf9` or `264f0c3`.
Re-checked 2026-10-26: `import fbgemm_xpu` as the first import works, and
`asynchronous_complete_cumsum` runs on XPU.)

**Workaround.** An unconditional `import fbgemm_xpu` at the top of
`train/train_ranker.py`, and a guarded one in `inference/main.py` (the harness
branch of PR #1 has guarded imports in `train/utils.py` and
`inference/main.py`). Every new entry point, including probes, needs the same
import. `write_run_manifest.py` locates `fbgemm_xpu` with
`importlib.util.find_spec` instead of importing it.

**Reproduce.** Remove the import from `inference/main.py`;
`bash run_dlrm_v3_xpu.sh --mode infer` (not re-run). Or, without the harness:
`python -c "import torch, fbgemm_gpu; torch.ops.fbgemm.asynchronous_complete_cumsum(torch.tensor([1, 2], device='xpu'))"`
fails without `import fbgemm_xpu`.

**Owner.** torchlib-xpu (registration hook).

### H-02 `fbgemm_xpu` extensions have no RPATH

**Symptom.** `_C.so` / `_C_training.so` have no RPATH/RUNPATH, so
`libtorch*.so` resolves only if torch is already loaded. Loading the `.so`
directly without torch (for example with `ctypes`) fails with
`libtorch.so: cannot open shared object file`.

**Status on current pins.** Not reached: `fbgemm_xpu/__init__.py` imports
torch before loading `_C`, so `import fbgemm_xpu` works without the workaround
(checked 2026-09-30 with torch's `lib/` removed from `LD_LIBRARY_PATH`). The
RPATH is still missing. Re-checked 2026-10-26 on the build from `264f0c3`: `readelf -d` on the
installed `_C` and `_C_training` shows no RPATH/RUNPATH, and `import fbgemm_xpu`
works in a clean environment (`env -i`, no `LD_LIBRARY_PATH`, no oneAPI
`setvars.sh`).

**Workaround.** The launcher prepends torch's `lib/` to `LD_LIBRARY_PATH`;
not needed for `import fbgemm_xpu` on the pins, kept as a safeguard (this
repo's `run_dlrm_v3_xpu.sh` carries it).

**Owner.** torchlib-xpu (bake in an RPATH; low priority).

### H-03 The FBGEMM wheel is patched in place

**Symptom.** Every `manifest.json` reports `fbgemm` as dirty, with
`fbgemm_gpu/tbe/config/embedding_config.py` in `modified_files` (checked
2026-10-26 in the manifest of a launcher run on this branch).

**Root cause.** TorchRec references `ComputeDevice.XPU`, which
fbgemm-gpu-cpu 1.8.0 and 1.9.0 do not define. `install_dlrmv3.sh` step 8 runs
torchrec's `patch/patch_fbgemm_gpu_compute_device_xpu.py`, which adds
`XPU = 3` to the installed wheel. The patch is traceable through the torchrec
SHA.

**Status on current pins.** Open (by design). Re-checked 2026-10-26 against
fbgemm-gpu-cpu 1.9.0: of all files in the wheel's `RECORD`, exactly one hash
differs, `fbgemm_gpu/tbe/config/embedding_config.py`, and it contains the
`XPU = 3` line in `ComputeDevice` (the diff against a pristine wheel was not
taken). A reinstall of the wheel silently drops the
patch; re-run `install_dlrmv3.sh` step 8.

**Owner.** FBGEMM / TorchRec upstreaming.

### H-04 Unquantized inference does its embedding lookup on CPU

**Symptom.** `--mode infer` runs end to end, but only the dense HSTU model runs
on XPU; the sparse lookup runs on CPU.

**Root cause.** Upstream `HSTUSparseInferenceModule` builds a plain TorchRec
`EmbeddingCollection` on `table_device="cpu"`. FBGEMM lookups only enter
through the quantized path (`QuantEmbeddingCollection`).

**Status.** Open. Re-measured 2026-10-26 on this branch: `--mode infer` runs
end to end. The sparse module is built with `table_device="cpu"`
(`inference/model_family.py`); its `EmbeddingCollection` holds plain
`Embedding(1000000, 64)` tables (`run.log`); the dense module runs on `xpu:0`.
Unquantized inference therefore does not exercise any XPU embedding lookup or
any fbgemm-xpu TBE kernel.

**Owner.** quantized embedding lookup work (separate from this harness).

### H-05 The inference runner swallows exceptions

**Symptom.** `Runner.run_one_item` catches every exception, logs
`thread: failed, …`, and still sends predictions to LoadGen from its `finally`
block. A failure after the prediction is unpacked is therefore swallowed: the
run continues and exits 0. A failure inside `model.predict()` leaves
`mt_target_preds` unset, so the `finally` block raises `UnboundLocalError`,
which hides the original error. That error escapes LoadGen's C++ callback and
the process dumps core (exit 139).

**Status.** Open (code read 2026-10-26: `Runner.run_one_item` in
`inference/main.py` still has the broad `except` / `finally`). Not hit in the
two inference runs of 2026-10-26: 0 `thread: failed` lines in either `run.log`.
Check `run.log` for `thread: failed` as well as the exit code.
On a core dump, the real error is the `thread: failed` line just before the
`UnboundLocalError`.

**Owner.** generative-recommenders upstream.

### H-06 `MultiThreadDataProducer` is CUDA-only

**Root cause.** `inference/data_producer.py` uses `torch.cuda.Stream` per
worker thread.

**Status.** Worked around; the code is unchanged. Re-measured 2026-10-26: the
inference gin sets `run.data_producer_threads = 1`, and both inference runs
completed.

**Workaround.** `run.data_producer_threads = 1` in
`inference/gin/movielens_1m_xpu.gin`.

**Owner.** generative-recommenders upstream.

### H-07 Offline scenario ignores `run.num_queries`

**Root cause.** `inference/main.py:get_num_queries` sizes Offline runs as
1.1 × `min_duration` (600 s) × target QPS.

**Status.** Worked around; the code in `inference/main.py:get_num_queries` is
unchanged.

**Workaround.** The inference gin uses the Server scenario, which honours
`num_queries` (re-measured 2026-10-26: 256 queries processed). LoadGen
reports such short runs INVALID, which is expected.

**Owner.** generative-recommenders upstream (low).

### H-08 Inverted metric-log check in `train_loop` and `eval_loop`

**Root cause.** `train_loop` and `eval_loop` log when
`batch_idx % metric_log_frequency != 0` (`train_eval_loop` and
`streaming_train_eval_loop` use `== 0`). With frequency 1 nothing is logged;
with 50, 49 of 50 batches are. Upstream since `418e588`.

**Status.** Open (code read 2026-10-26): `train_loop` and `eval_loop` in
`train/utils.py` still use `!=`, the other two loops `==`.

**Workaround.** `losses.csv` in the run folder records every training step's
loss independently of the metric log (`train_loop` and `train_eval_loop`; not
the streaming loop). Eval metrics in `--mode eval` are still
affected.

**Owner.** generative-recommenders upstream.

### H-09 TensorBoard log directory is fixed and shared

**Root cause.** `MetricsLogger.tensorboard_log_path =
"/tmp/tensorboard_log_path.log"` in `movielens_1m_xpu.gin` and `debug.gin`
(`..._xpu.log` on the harness branch); all runs write to the same directory.

**Status.** Open (code read 2026-10-26).

**Owner.** generative-recommenders fork (bind it per run).

### H-10 Seeded XPU training is not bit-reproducible

**Symptom.** Two XPU runs with the same seed, code and config are identical
at first, then drift apart.
- PVC, 10 steps: steps 0–1 identical, then ~1e-5 absolute drift. CPU-vs-XPU
  differences (mean 6.5% relative) stay below a seed change on CPU (10.1%).
- B60, full epoch (849 steps): identical up to step 21; first difference at
  step 22 (3.6e-7); at most 3.0e-3 absolute, 0.20% mean relative, over the
  epoch. CPU vs XPU: 3.32% mean relative over the epoch (5.2% over the first
  10 steps), about 16 times the XPU repeat difference.

**Root cause.** Not investigated; likely non-deterministic accumulation in
backward kernels or an unseeded stochastic-rounding source (FP16 tables use
`stochastic_rounding=True`). Consistent with the small in-process CPU/XPU
divergence measured by `probe_numerics_train_step_cpu_vs_xpu.py` (worst
relative loss difference 0.002% over 10 steps; re-measured 2026-10-26 on
torch 2.14.1 / fbgemm 1.9.0: 0.003%, worst weight difference 1.2e-2, verdict
"tracks"). Run-to-run XPU reproducibility was re-measured over 10 steps only
(see Reproduce); the full-epoch B60 numbers above were not repeated.

**Reproduce.** Two runs of `bash run_dlrm_v3_xpu.sh --phase 1 --mode train`;
`python compare_loss_curves.py RUN_A CPU_RUN --repeat RUN_B`. Re-measured
2026-10-26 on this branch (PVC, 10 steps; the repeat was a phase 2 run):
XPU vs XPU repeat, max abs 3.8e-5, mean relative 0.01%; XPU vs CPU, mean
relative 10.6%, Pearson 0.932.

**Owner.** torchlib-xpu / FBGEMM XPU kernels (low).

### H-11 Launcher sourced a version-pinned oneAPI path silently

**Root cause.** `source /opt/intel/oneapi/2025.3/oneapi-vars.sh … || true`; on a
2026.1 install the line did nothing and the run relied on the caller's shell.

**Status.** Fixed in the launcher (`ee644e8`): `ONEAPI_ROOT` (default
`/opt/intel/oneapi`), version-independent `setvars.sh`, hard error if it is
missing or fails. An already activated shell (`SETVARS_COMPLETED=1`) is used as
is; the manifest's `runtime.oneapi` records the versions either way. Code read
2026-10-26 for this repo's `run_dlrm_v3_xpu.sh`: it sources
`${ONEAPI_ROOT}/setvars.sh` when `SETVARS_COMPLETED` is not 1. The sourcing
path was **not exercised**: the shell used for the runs already had oneAPI
active (`SETVARS_COMPLETED=1`). `/opt/dlrmv3/run_dlrm_v3_xpu.sh` (the older
launcher outside this repo) was not checked for this.

**Owner.** harness.

### H-12 MLPerf LoadGen build writes into its source tree

**Root cause.** LoadGen's `setup.py` regenerates
`generated/version_generated.cc` and `mlperf_conf.h`, making the fork dirty in
every manifest.

**Workaround.** `install_dlrmv3.sh` builds LoadGen from a temporary copy.
Not re-checked on 2026-10-26.

**Owner.** harness.

### H-13 Two oneAPI versions in one process

**Observation.** The system compiler (2026.1.1) builds fbgemm-xpu; torch loads
the pip runtime wheels (`intel-sycl-rt` 2026.0.0). Both are recorded in
`manifest.json` under `runtime.oneapi`. No failure attributed
to it so far. Not re-checked with torch 2.14.1 (its runtime wheel versions were
not compared).

**Owner.** platform (informational).

### H-14 `dlrm_v3` is not packaged by `setup.py`

**Root cause.** `find_packages()` only finds directories with `__init__.py`;
`dlrm_v3/`, `ops/` and `research/` have none, so an installed
generative-recommenders lacks them.

**Status.** Open (re-checked 2026-10-26: `dlrm_v3/`, `ops/` and `research/`
under `generative_recommenders/` still have no `__init__.py`, and `setup.py`
still uses `find_packages(exclude=["configs"])`).

**Workaround.** The launcher puts the checkout on `PYTHONPATH`
(this repo's `run_dlrm_v3_xpu.sh` does; the probes need `PYTHONPATH=$PWD` too).

**Owner.** generative-recommenders upstream.

### H-15 `torch.autocast` is hard-coded to `"cuda"` in the model modules

**Symptom.** On XPU, the training run logs `UserWarning: CUDA is not available
or torch_xla is imported. Disabling autocast.` from
`modules/dlrm_hstu.py:423`. The bf16 autocast region around the HSTU
transducer is therefore not applied, although `DlrmHSTU` defaults to
`bf16_training=True`.

**Root cause.** `torch.autocast("cuda", dtype=torch.bfloat16, ...)` in
`modules/dlrm_hstu.py`, `modules/multitask_module.py` and
`modules/contextual_interleave_preprocessor.py`. With a `cuda` device type and no
CUDA, autocast disables itself, so on XPU the model runs in the dtype of its
inputs and parameters.

**Status on current pins.** Open on this branch (found 2026-10-26 in the 10-step
`debug` run: the warning is printed once; code read for the other two sites).
The harness branch of PR #1 fixes it by using the input tensor's device type.
Consequence for results: XPU numbers from this branch are not bf16-autocast
numbers, so they are not comparable with CUDA bf16 training, and
bf16 numerics on XPU are not exercised. The CPU-vs-XPU comparisons above are
unaffected (both run without autocast).

**Reproduce.** `python .../dlrm_v3/train/train_ranker.py --dataset debug --mode
train` on an XPU host with `WORLD_SIZE=1`; look for `Disabling autocast` in the
output.

**Owner.** generative-recommenders fork.

### H-16 The manifest records the torchlib-xpu checkout, not the revision the `.so` was built from

**Symptom.** On the 2026-10-26 host, every `manifest.json` gives torchlib-xpu
as `2655c44`, clean, with `binaries_older_than_head_commit: false`. The
fbgemm-xpu that actually ran was built from `264f0c3`.

**Root cause.** fbgemm-xpu is an editable install whose compiled `_C*.so` are
copied into site-packages at build time. `write_run_manifest.py` reports the
SHA the source checkout is at when the run starts. It flags stale binaries
only by comparing the `.so` mtime with the HEAD commit's committer date, not
with the time the checkout moved. Here the checkout moved from
`264f0c3` to `main` (`2655c44`) on 2026-10-23, three days after the build
(reflog), and `2655c44` was committed on 2026-10-19, before the build, so
nothing was flagged. The torchrec component is not affected: it is a
non-editable wheel whose version carries the build SHA (`+c0fc555`), and the
manifest reports `build_sha_matches_source_head: false` for the moved
checkout.

**Status.** Open. The `.so` sha256 in the manifest still identifies the build
exactly, but not which tree it came from.

**Workaround.** Do not move the torchlib-xpu checkout without rebuilding
fbgemm-xpu (`install_dlrmv3.sh` step 4). To check a host:
`git -C src/torchlib-xpu reflog --date=iso | head`.

**Fix.** In `write_run_manifest.py`, compare the `.so` mtime with the time of
the last HEAD move (the reflog entry), not with the commit time, or record the
build SHA at build time.

**Owner.** harness.

---

## Adding a finding

New findings get the next H- ID (`H-17`); the A- series is closed. Add a row to the index and a
section with: symptom, root cause (or "not investigated"), fix or workaround,
status on the current pins, evidence (run folder with its `manifest.json` when
the harness is available, a probe script, or a reproduction command), owner.
Update the "Status checked against" revisions when re-verifying.

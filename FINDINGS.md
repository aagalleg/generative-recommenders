# DLRM-v3 on XPU — Findings

What the XPU run harness (`run_dlrm_v3_xpu.sh`) and the investigations behind
it have found missing or wrong. One entry per finding: symptom, root cause,
fix or workaround, status on the current pins, evidence, owner.

**Status checked against** (2026-10-26, branch
`probes/dlrm-v3-xpu-fbgemmv1.9.0`): torch `2.14.1+xpu`, fbgemm-gpu-cpu
`1.9.0`, fbgemm-xpu `0.9.0` built from torchlib-xpu `264f0c3` (the pin in
`install_dlrmv3.sh`), torchrec `c0fc555` (`1.10.0a0`, branch
`feature/xpu-device-support-v1.9.0_minimal`), generative-recommenders
`120f0f8` plus the probes, on one tile of a PVC (Max 1550), oneAPI 2026.1.
This replaces the previous check (2026-09-29: torchlib-xpu `b59acf9`,
fbgemm-gpu-cpu `1.8.0`, torch 2.13, torchrec `e213eb9`, generative-recommenders
`ed103ac`; plus B60 harness runs of 2026-09-30, which H-10 still cites).

**What this branch carries.** It is based on the v1.9.0 minimal XPU port, not
on the harness branch of
[PR #1](https://github.com/aagalleg/generative-recommenders/pull/1). It has no
`dlrm_v3/harness.py`, run manifest, `operative_config.gin`, `losses.csv` or
`compare_loss_curves.py`, and `get_hstu_configs` is not gin-configurable. The
launcher is `/opt/dlrmv3/run_dlrm_v3_xpu.sh`. So "status" below is one of:
*re-measured* (probe or end-to-end run on 2026-10-26), *code read* (the code
that causes it is unchanged or changed; not run), or *not re-checked*
(depends on the harness or on a host that was not used).

**End-to-end check on this branch.** `train_ranker.py --dataset debug --mode
train` (`debug.gin`: 10 batches, `output_trace = True`), one tile: exit 0, ten
`train - Step` lines with finite NE (1.06 at step 2, 1.38 at step 10), no NaN
in the log, all three tables `TW` / `fused`
(`Using fused exact_row_wise_adagrad`), no SIGABRT at exit.

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
| [A-02](#a-02-hstu_-overrides-silently-inert) | `HSTU_*` overrides silently inert | **Open on this branch**: no override mechanism; probes override in-process | generative-recommenders fork |
| [A-03](#a-03-two-rank-cross-tile-xccl-took-the-card-off-the-pcie-bus) | Two-rank cross-tile XCCL took the card off the PCIe bus | **Open — hazard present** for any multi-tile run; not re-run | harness; platform/driver |
| [A-04](#a-04-profiler-aborts-the-process-on-teardown) | Profiler aborts the process on teardown | **Fixed on this branch** (`Profiler.__del__` calls `stop()`); torch behaviour unchanged | generative-recommenders fork |
| [A-05](#a-05-cpu-on-the-xpu-sharding-plan-fails-in-construct_jagged_tensors) | CPU on the XPU sharding plan fails in `construct_jagged_tensors` | Not reached | torchrec |
| [A-06](#a-06-xpu-was-dense-kernel-only-data_parallel-sharding) | XPU was DENSE-kernel only (DATA_PARALLEL sharding) | Superseded for training; DENSE TBE module still fails on XPU (FBGEMM 1.9.0 Python) | FBGEMM / TorchRec (DENSE path, low) |
| [A-07](#a-07-host-user-mode-driver-too-old-for-torch-213) | Host user-mode driver too old for torch 2.13 | Handled by container; torch 2.14.1 runs | platform |
| [A-08](#a-08-oneccl-needs-devdriby-path-inside-containers) | oneCCL needs `/dev/dri/by-path` inside containers | Handled | platform |
| [A-09](#a-09-movielens-download-failed-expired-tls-certificate) | MovieLens download failed: expired TLS certificate | Resolved externally | — |
| [H-01](#h-01-xpu-kernels-are-only-registered-by-an-explicit-import-fbgemm_xpu) | XPU kernels only registered by an explicit `import fbgemm_xpu` | Worked around (still needed) | torchlib-xpu |
| [H-02](#h-02-fbgemm_xpu-extensions-have-no-rpath) | `fbgemm_xpu` extensions have no RPATH | Not reached (RPATH still missing) | torchlib-xpu |
| [H-03](#h-03-the-fbgemm-wheel-is-patched-in-place) | The FBGEMM wheel is patched in place | Open (by design; 1.9.0 still lacks `XPU` upstream) | FBGEMM / TorchRec upstreaming |
| [H-04](#h-04-unquantized-inference-does-its-embedding-lookup-on-cpu) | Unquantized inference does its embedding lookup on CPU | Not reached: inference is not ported on this branch | quantized embedding lookup work |
| [H-05](#h-05-the-inference-runner-swallows-exceptions) | The inference runner swallows exceptions | Open (code unchanged; not run) | generative-recommenders upstream |
| [H-06](#h-06-multithreaddataproducer-is-cuda-only) | `MultiThreadDataProducer` is CUDA-only | Open (code unchanged; not run) | generative-recommenders upstream |
| [H-07](#h-07-offline-scenario-ignores-runnum_queries) | Offline scenario ignores `run.num_queries` | Not reached (inference not ported) | generative-recommenders upstream |
| [H-08](#h-08-inverted-metric-log-check-in-train_loop-and-eval_loop) | Inverted metric-log check in `train_loop` and `eval_loop` | **Open** (code read) | generative-recommenders upstream |
| [H-09](#h-09-tensorboard-log-directory-is-fixed-and-shared) | TensorBoard log directory is fixed and shared | Open (code read) | generative-recommenders fork |
| [H-10](#h-10-seeded-xpu-training-is-not-bit-reproducible) | Seeded XPU training is not bit-reproducible | Open (low); CPU-vs-XPU tracking re-measured | torchlib-xpu |
| [H-11](#h-11-launcher-sourced-a-version-pinned-oneapi-path-silently) | Launcher sourced a version-pinned oneAPI path silently | Harness launcher; not re-checked | harness |
| [H-12](#h-12-mlperf-loadgen-build-writes-into-its-source-tree) | MLPerf LoadGen build writes into its source tree | Not re-checked | harness |
| [H-13](#h-13-two-oneapi-versions-in-one-process) | Two oneAPI versions in one process | Informational; not re-checked | platform |
| [H-14](#h-14-dlrm_v3-is-not-packaged-by-setuppy) | `dlrm_v3` is not packaged by `setup.py` | Open (re-checked) | generative-recommenders upstream |
| [H-15](#h-15-torchautocast-is-hard-coded-to-cuda-in-the-model-modules) | `torch.autocast` is hard-coded to `"cuda"` in the model modules | **Open on this branch** (new, 2026-10-26) | generative-recommenders fork |

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

**Status on current pins.** *Open on this branch (code read, and exercised by
the probes).* This branch has no `harness.py`, so nothing reads
`HSTU_EMBEDDING_DIM` or `HASH_SIZE` (or any other `HSTU_*` / `DLRM_V3_*`
variable): they are plain module constants in `dlrm_v3/configs.py`
(512 and 10,000,000), and `get_hstu_configs` is not `@gin.configurable`. The
only size control is `make_model` capping `max_seq_len` at 256 when an XPU is
present (the PYTORCH attention kernel is O(N²) in memory). The model-building
probes therefore set the constants and the config fields in-process (see
[PROBES.md](PROBES.md)). On the harness branch of PR #1 the two size
variables are read by `harness.apply_size_overrides` and the other dims are
gin bindings; the remaining trial-branch variables (`HSTU_MAX_SEQ_LEN`,
`HSTU_TABLE_DTYPE`, `HSTU_NUM_HEADS`, `DLRM_V3_SEED`, …) do not exist in either.

**Evidence.** Reproduced on 2026-09-28 on the harness branch before its fix: a
CPU run with `HSTU_EMBEDDING_DIM=64 HASH_SIZE=1000000` built 512-dim / 10M-row
tables and failed in the planner. On this branch the 10-step `debug` run builds
the 512-dim / 10M-row tables (`SplitTableBatchedEmbeddingBagsCodegen` with
`(10000000, 512, ...)` in the run log).

**Owner.** generative-recommenders fork (carry the harness's size overrides, or
make `get_hstu_configs` gin-configurable; consider rejecting unknown `HSTU_*`
variables).

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
present.** This branch has no launcher phase 3, but
`/opt/dlrmv3/run_dlrm_v3_xpu.sh --world-size N` (N > 1) exposes tiles `0..N-1`
and runs XCCL across them with `CCL_ZE_IPC_EXCHANGE=pidfd`, which is the same
cross-tile Level Zero IPC traffic. The host used on 2026-10-26 differs from the
trial's: Linux 6.8 (so `pidfd_open` works here, checked), not 5.15, and 16
tiles visible. Whether the fabric failure is specific to the trial's
2023 kernel driver and kernel 5.15 is not known, and nobody has run sustained cross-tile
XCCL on this host. The trial's escape, `FBGEMM_XPU_FORCE_GLOO=1` (collectives
staged through the host), does not exist in the pinned trees. Do not run a
multi-tile job on a PVC without a recoverable card until it is guarded. The
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

**Workaround.** On this branch an unconditional `import fbgemm_xpu` at the top
of `train/train_ranker.py` (the harness branch of PR #1 has a guarded import in
`train/utils.py` and `inference/main.py`). `inference/main.py` does not import
it here, but inference is not ported on this branch (see H-04). Every new
entry point, including probes, needs the same import.

**Reproduce.** On the harness branch of PR #1: remove the import from
`inference/main.py`; `bash run_dlrm_v3_xpu.sh --mode infer`. On any branch:
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
RPATH is still missing. Re-checked 2026-10-26 on `264f0c3`: `readelf -d` on the
installed `_C` and `_C_training` shows no RPATH/RUNPATH, and `import fbgemm_xpu`
works in a clean environment (`env -i`, no `LD_LIBRARY_PATH`, no oneAPI
`setvars.sh`).

**Workaround.** The launcher prepends torch's `lib/` to `LD_LIBRARY_PATH`;
not needed for `import fbgemm_xpu` on the pins, kept as a safeguard.

**Owner.** torchlib-xpu (bake in an RPATH; low priority).

### H-03 The FBGEMM wheel is patched in place

**Symptom.** Every `manifest.json` reports `fbgemm` as dirty, with
`fbgemm_gpu/tbe/config/embedding_config.py` in `modified_files` (harness
branch). Without the manifest: the installed wheel no longer matches its
`RECORD`.

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
on XPU (~37 ms per batch of 16); the sparse lookup runs on CPU (~2.4 ms).

**Root cause.** Upstream `HSTUSparseInferenceModule` builds a plain TorchRec
`EmbeddingCollection` on `table_device="cpu"`. FBGEMM lookups only enter
through the quantized path (`QuantEmbeddingCollection`).

**Status.** Not reached on this branch: inference is not ported
(`inference/model_family.py` still asserts `torch.cuda.is_available()` and uses
`torch.cuda.*`), so `--mode infer` does not run on XPU here. The numbers above
come from the harness branch of PR #1 (2026-09-29/30) and were not re-measured.
Unquantized inference would not exercise the XPU embedding lookup there either.

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
`inference/main.py` still has the broad `except` / `finally`; not run on this
branch). Check `run.log` for `thread: failed` as well as the exit code.
On a core dump, the real error is the `thread: failed` line just before the
`UnboundLocalError`.

**Owner.** generative-recommenders upstream.

### H-06 `MultiThreadDataProducer` is CUDA-only

**Root cause.** `inference/data_producer.py` uses `torch.cuda.Stream` per
worker thread.

**Status.** Open (code read 2026-10-26: unchanged on this branch; inference is
not ported here).

**Workaround.** `run.data_producer_threads = 1` in
`inference/gin/movielens_1m_xpu.gin` (harness branch of PR #1).

**Owner.** generative-recommenders upstream.

### H-07 Offline scenario ignores `run.num_queries`

**Root cause.** `inference/main.py:get_num_queries` sizes Offline runs as
1.1 × `min_duration` (600 s) × target QPS.

**Status.** Not reached on this branch (no XPU inference path); the code in
`inference/main.py:get_num_queries` is unchanged.

**Workaround.** The harness branch of PR #1 uses the Server scenario, which honours
`num_queries`. LoadGen reports such short runs INVALID, which is expected.

**Owner.** generative-recommenders upstream (low).

### H-08 Inverted metric-log check in `train_loop` and `eval_loop`

**Root cause.** `train_loop` and `eval_loop` log when
`batch_idx % metric_log_frequency != 0` (`train_eval_loop` and
`streaming_train_eval_loop` use `== 0`). With frequency 1 nothing is logged;
with 50, 49 of 50 batches are. Upstream since `418e588`.

**Status.** Open (code read 2026-10-26): `train_loop` and `eval_loop` in
`train/utils.py` still use `!=`, the other two loops `==`.

**Workaround.** The harness branch of PR #1 records every training step's loss in
`RUN_DIR/losses.csv` independently. This branch has no such record. Eval metrics in `--mode eval` are still
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
"tracks"). Run-to-run XPU reproducibility was **not** re-measured.

**Reproduce.** Two runs of `bash run_dlrm_v3_xpu.sh --phase 1 --mode train`;
`python compare_loss_curves.py RUN_A CPU_RUN --repeat RUN_B`.

**Owner.** torchlib-xpu / FBGEMM XPU kernels (low).

### H-11 Launcher sourced a version-pinned oneAPI path silently

**Root cause.** `source /opt/intel/oneapi/2025.3/oneapi-vars.sh … || true`; on a
2026.1 install the line did nothing and the run relied on the caller's shell.

**Status.** Fixed in the launcher (`ee644e8`): `ONEAPI_ROOT` (default
`/opt/intel/oneapi`), version-independent `setvars.sh`, hard error if it is
missing or fails. An already activated shell (`SETVARS_COMPLETED=1`) is used as
is; the manifest's `runtime.oneapi` records the versions either way. This is
the harness branch's launcher; this branch is launched by
`/opt/dlrmv3/run_dlrm_v3_xpu.sh`, which was not re-checked for this.

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
`manifest.json` under `runtime.oneapi` (harness branch). No failure attributed
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
(`run_dlrm_v3_xpu.sh`; the probes need `PYTHONPATH=$PWD` too).

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

New findings get the next H- ID (`H-16`); the A- series is closed. Add a row to the index and a
section with: symptom, root cause (or "not investigated"), fix or workaround,
status on the current pins, evidence (run folder with its `manifest.json` when
the harness is available, a probe script, or a reproduction command), owner.
Update the "Status checked against" revisions when re-verifying.

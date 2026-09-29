# DLRM-v3 on XPU — Findings

What the XPU run harness (`run_dlrm_v3_xpu.sh`) and the investigations behind
it have found missing or wrong. One entry per finding: symptom, root cause,
fix or workaround, status on the current pins, evidence, owner.

**Status checked against** (2026-09-29): torchlib-xpu `b59acf9`,
fbgemm-gpu-cpu `1.8.0`, torchrec `e213eb9`, generative-recommenders `8edcb44`,
on PVC (Max 1550) in the `Dockerfile.xpu` container.

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
| [A-01](#a-01-fp16-embedding-tables-train-to-nan--optimizer-epsilon-not-the-device) | FP16 tables train to NaN — optimizer epsilon, not the device | Not reached; root cause unfixed; code comment wrong | generative-recommenders fork |
| [A-02](#a-02-hstu_-overrides-silently-inert) | `HSTU_*` overrides silently inert | Fixed for the two supported variables; the other trial-branch variables are ignored | harness |
| [A-03](#a-03-two-rank-cross-tile-xccl-took-the-card-off-the-pcie-bus) | Two-rank cross-tile XCCL took the card off the PCIe bus | **Open — hazard present in launcher phase 3** | harness; platform/driver |
| [A-04](#a-04-profiler-aborts-the-process-on-teardown) | Profiler aborts the process on teardown | Open (only with `output_trace = True`) | generative-recommenders fork |
| [A-05](#a-05-cpu-on-the-xpu-sharding-plan-fails-in-construct_jagged_tensors) | CPU on the XPU sharding plan fails in `construct_jagged_tensors` | Not reached | torchrec |
| [A-06](#a-06-xpu-was-dense-kernel-only-data_parallel-sharding) | XPU was DENSE-kernel only (DATA_PARALLEL sharding) | Superseded for training; DENSE TBE still fails on XPU | torchlib-xpu (DENSE path, low) |
| [A-07](#a-07-host-user-mode-driver-too-old-for-torch-213) | Host user-mode driver too old for torch 2.13 | Handled by container | platform |
| [A-08](#a-08-oneccl-needs-devdriby-path-inside-containers) | oneCCL needs `/dev/dri/by-path` inside containers | Handled | platform |
| [A-09](#a-09-movielens-download-failed-expired-tls-certificate) | MovieLens download failed: expired TLS certificate | Resolved externally | — |
| [H-01](#h-01-xpu-kernels-are-only-registered-by-an-explicit-import-fbgemm_xpu) | XPU kernels only registered by an explicit `import fbgemm_xpu` | Worked around | torchlib-xpu |
| [H-02](#h-02-fbgemm_xpu-extensions-have-no-rpath) | `fbgemm_xpu` extensions have no RPATH | Worked around | torchlib-xpu |
| [H-03](#h-03-the-fbgemm-wheel-is-patched-in-place) | The FBGEMM wheel is patched in place | Open (by design) | FBGEMM / TorchRec upstreaming |
| [H-04](#h-04-unquantized-inference-does-its-embedding-lookup-on-cpu) | Unquantized inference does its embedding lookup on CPU | Open | quantized embedding lookup work |
| [H-05](#h-05-the-inference-runner-swallows-exceptions) | The inference runner swallows exceptions | Open | generative-recommenders upstream |
| [H-06](#h-06-multithreaddataproducer-is-cuda-only) | `MultiThreadDataProducer` is CUDA-only | Worked around | generative-recommenders fork |
| [H-07](#h-07-offline-scenario-ignores-runnum_queries) | Offline scenario ignores `run.num_queries` | Worked around | generative-recommenders upstream |
| [H-08](#h-08-inverted-metric-log-check-in-train_loop-and-eval_loop) | Inverted metric-log check in `train_loop` and `eval_loop` | Worked around (training loss) | generative-recommenders upstream |
| [H-09](#h-09-tensorboard-log-directory-is-fixed-and-shared) | TensorBoard log directory is fixed and shared | Open | generative-recommenders fork |
| [H-10](#h-10-seeded-xpu-training-is-not-bit-reproducible) | Seeded XPU training is not bit-reproducible | Open (low) | torchlib-xpu |
| [H-11](#h-11-launcher-sourced-a-version-pinned-oneapi-path-silently) | Launcher sourced a version-pinned oneAPI path silently | Fixed | harness |
| [H-12](#h-12-mlperf-loadgen-build-writes-into-its-source-tree) | MLPerf LoadGen build writes into its source tree | Worked around | harness |
| [H-13](#h-13-two-oneapi-versions-in-one-process) | Two oneAPI versions in one process | Informational | platform |
| [H-14](#h-14-dlrm_v3-is-not-packaged-by-setuppy) | `dlrm_v3` is not packaged by `setup.py` | Worked around | generative-recommenders upstream |

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

**Status on current pins.** *Not reached, root cause unfixed.* The current
TorchRec gives XPU a table-wise plan with fused RowWiseAdagrad (`TW` / `fused`,
`Using fused exact_row_wise_adagrad`), so the tables never enter the dense Adam
group and FP16 tables train with finite loss. The sharding-path probes, adapted
to the pins, measure the same: 0 NaN after init, around DDP, after sharding and
in the first forward, on XPU and CPU (see [PROBES.md](PROBES.md)). Any fp16 parameter in a dense
Adam group would still turn NaN; measured on the pins with
[`probe_nan_fp16_adam_eps.py`](probes/probe_nan_fp16_adam_eps.py): fp16 with a
partly-zero gradient gives 1016/1024 NaN on XPU, fp32 gives 0. The pinned `train/utils.py`
(`make_optimizer_and_shard`) still skips `init_state` and
`apply_optimizer_in_backward` on XPU, with comments attributing the corruption
to the XPU caching allocator. The reproduction above shows it is a dtype
problem, so that explanation is wrong.

**Evidence.** Probes, in the order they narrowed it down: `probe_nan_tbe_init_alias.py`,
`probe_nan_rng_initialisers.py`, `probe_nan_shard_weight_alias.py`,
`probe_nan_init_bounds.py`, `probe_nan_corruption_stage.py`,
[`probe_nan_fp16_adam_eps.py`](probes/probe_nan_fp16_adam_eps.py) (the
reproduction), all in [`probes/`](probes/); see [PROBES.md](PROBES.md).
Current plan: `run.log` of any training run (planner table).

**Owner.** generative-recommenders fork (correct the comments, revisit both XPU
skips); a guard for fp16 + dense Adam belongs upstream.

### A-02 `HSTU_*` overrides silently inert

**Symptom.** Runs believed to be configured through `HSTU_*` environment
variables ran on stock config; the first visible sign was a 31.8 GiB allocation
failure from the stock `max_seq_len=16384`.

**Root cause (trial branch).** `train_ranker.py` monkeypatched
`configs.get_hstu_configs`, but `train/utils.py` had imported the function
earlier and kept its own reference, so every override was ignored.

**Status on current pins.** *Fixed for what is supported; the rest is
ignored.* Only `HSTU_EMBEDDING_DIM` and `HASH_SIZE` are read
(`dlrm_v3/harness.py:apply_size_overrides`), on XPU always and on other devices
when set, and their effective values are written to `operative_config.gin`. The
other HSTU dims are gin bindings. The remaining trial-branch variables
(`HSTU_MAX_SEQ_LEN`, `HSTU_TABLE_DTYPE`, `HSTU_NUM_HEADS`, `DLRM_V3_SEED`, …)
do not exist in the pinned code: setting them has no effect (current runs use
`max_seq_len=16384`). They are recorded in `manifest.json` under `env`, so a
mismatch is visible by comparing it with `operative_config.gin`, but nothing
rejects them.

**Evidence.** Reproduced on 2026-09-28 before the fix: a CPU run with
`HSTU_EMBEDDING_DIM=64 HASH_SIZE=1000000` built 512-dim / 10M-row tables and
failed in the planner.

**Owner.** harness (consider rejecting unknown `HSTU_*` variables).

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

**Status on current pins.** **Open — the hazard is present.** Launcher phase 3
runs XCCL across all visible devices (`NUM_XPUS` defaults to the detected count,
16 on the current host) with `CCL_ZE_IPC_EXCHANGE=pidfd`; this container logs
`pidfd is not supported, fallbacks to drmfd exchange mode`. The trial's escape,
`FBGEMM_XPU_FORCE_GLOO=1` (collectives staged through the host), does not exist
in the pinned trees. Do not run phase 3 on PVC until it is guarded.

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

**Status on current pins.** Open. `dlrm_v3/utils.py:Profiler` is unchanged
(no `stop()`, `ProfilerActivity.CUDA`). Reproduced on the pins with
`probe_profiler_sigabrt_fix.py`: without `stop()` the process ends with
SIGABRT (Kineto `clearActivities`), with `stop()` it exits cleanly. Not hit by
the XPU gin configs, which set `output_trace = False`.

**Evidence.** [`probes/probe_profiler_activity_teardown.py`](probes/probe_profiler_activity_teardown.py),
[`probes/probe_profiler_sigabrt_fix.py`](probes/probe_profiler_sigabrt_fix.py).

**Owner.** generative-recommenders fork.

### A-05 CPU on the XPU sharding plan fails in `construct_jagged_tensors`

**Symptom.** Forcing the CPU run onto the trial's XPU plan (DP / dense) failed
with `split_with_sizes expects split_sizes to sum exactly to 96` in
`torchrec/modules/utils.py:construct_jagged_tensors`. Root cause not
investigated.

**Status on current pins.** Not reached: CPU and XPU both get `TW` / `fused`,
which also makes the harness CPU-vs-XPU loss comparison like-for-like in
sharding and optimizer (it was not in the trial).

**Owner.** torchrec (low priority).

### A-06 XPU was DENSE-kernel only (DATA_PARALLEL sharding)

**Symptom (trial branch).** XPU exposed only the DENSE compute kernel, so the planner rejected
`table_wise` (`DENSE kernel requires DATA_PARALLEL`), and
`DenseTableBatchedEmbeddingBagsCodegen.__init__` called
`torch.cuda.current_device()` (fixed on the trial branch of torchlib-xpu with `_tbe_dense_compat.py`).

**Status on current pins.** Superseded for training: the pinned TorchRec shards
XPU table-wise with the fused `SplitTableBatchedEmbeddingBagsCodegen`; the dense
constructor is not used. The DENSE path itself is still broken on XPU:
[`probe_kernel_tbe_dense_nobag.py`](probes/probe_kernel_tbe_dense_nobag.py)
and `probe_nan_tbe_init_alias.py` fail with `Torch not compiled with CUDA
enabled` in the constructor (the `_tbe_dense_compat.py` fix never reached the pinned torchlib-xpu).

**Owner.** torchlib-xpu (DENSE path; low priority while training uses the fused path).

### A-07 Host user-mode driver too old for torch 2.13

**Symptom.** Bare metal with torch 2.13.0+xpu: SIGSEGV in `get_device_name(0)`
(torch 2.6 works). The old user-mode driver also hides PVC's matrix engines.

**Status.** Handled: all runs use a container with a current user-mode driver
(`.devcontainer/xpu/Dockerfile.xpu` in the XPU dev-tools repository).

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

**Root cause.** `fbgemm_xpu` registers its XPU kernels on import. Only the
training path imported it. It must be imported after `fbgemm_gpu` / `torchrec`:
`fbgemm_gpu`'s registration is not guarded against duplicates, so an earlier
import aborts the process.

**Workaround.** Guarded import in `train/utils.py` and `inference/main.py`.
Every new entry point, including probes, needs the same.

**Reproduce.** Remove the import from `inference/main.py`;
`bash run_dlrm_v3_xpu.sh --mode infer`.

**Owner.** torchlib-xpu (registration hook / import-order robustness).

### H-02 `fbgemm_xpu` extensions have no RPATH

**Symptom.** `_C.so` / `_C_training.so` cannot resolve `libtorch*.so` at import.

**Workaround.** The launcher prepends torch's `lib/` to `LD_LIBRARY_PATH`.

**Owner.** torchlib-xpu.

### H-03 The FBGEMM wheel is patched in place

**Symptom.** Every `manifest.json` reports `fbgemm` as dirty, with
`fbgemm_gpu/tbe/config/embedding_config.py` in `modified_files`.

**Root cause.** TorchRec references `ComputeDevice.XPU`, which
fbgemm-gpu-cpu 1.8.0 does not define. `install_dlrmv3.sh` step 8 runs
torchrec's `patch/patch_fbgemm_gpu_compute_device_xpu.py`, which adds
`XPU = 3` to the installed wheel. The patch is traceable through the torchrec
SHA.

**Owner.** FBGEMM / TorchRec upstreaming.

### H-04 Unquantized inference does its embedding lookup on CPU

**Symptom.** `--mode infer` runs end to end, but only the dense HSTU model runs
on XPU (~37 ms per batch of 16); the sparse lookup runs on CPU (~2.4 ms).

**Root cause.** Upstream `HSTUSparseInferenceModule` builds a plain TorchRec
`EmbeddingCollection` on `table_device="cpu"`. FBGEMM lookups only enter
through the quantized path (`QuantEmbeddingCollection`).

**Status.** Open: unquantized inference does not exercise the XPU embedding
lookup.

**Owner.** quantized embedding lookup work (separate from this harness).

### H-05 The inference runner swallows exceptions

**Symptom.** A batch that fails inside `Runner.run_one_item` logs
`thread: failed, …` and the run continues and exits 0.

**Status.** Open. Check `run.log` for `thread: failed` and the accuracy output,
not the exit code.

**Owner.** generative-recommenders upstream.

### H-06 `MultiThreadDataProducer` is CUDA-only

**Root cause.** `inference/data_producer.py` uses `torch.cuda.Stream` per
worker thread.

**Workaround.** `run.data_producer_threads = 1` in
`inference/gin/movielens_1m_xpu.gin`.

**Owner.** generative-recommenders fork.

### H-07 Offline scenario ignores `run.num_queries`

**Root cause.** `inference/main.py:get_num_queries` sizes Offline runs as
1.1 × `min_duration` (600 s) × target QPS.

**Workaround.** The harness uses the Server scenario, which honours
`num_queries`. LoadGen reports such short runs INVALID, which is expected.

**Owner.** generative-recommenders upstream (low).

### H-08 Inverted metric-log check in `train_loop` and `eval_loop`

**Root cause.** `train_loop` and `eval_loop` log when
`batch_idx % metric_log_frequency != 0` (`train_eval_loop` and
`streaming_train_eval_loop` use `== 0`). With frequency 1 nothing is logged;
with 50, 49 of 50 batches are. Upstream since `418e588`.

**Workaround.** The harness records every training step's loss in
`RUN_DIR/losses.csv` independently. Eval metrics in `--mode eval` are still
affected.

**Owner.** generative-recommenders upstream.

### H-09 TensorBoard log directory is fixed and shared

**Root cause.** `MetricsLogger.tensorboard_log_path =
"/tmp/tensorboard_log_path_xpu.log"` in the XPU gin; all runs write to the same
directory.

**Owner.** generative-recommenders fork (bind it to `RUN_DIR`).

### H-10 Seeded XPU training is not bit-reproducible

**Symptom.** Two XPU runs with the same seed, code and config: steps 0–1
identical, then ~1e-5 absolute drift. CPU-vs-XPU differences (mean 6.5%
relative over 10 steps) stay below a seed change on CPU (10.1%).

**Root cause.** Not investigated; likely non-deterministic accumulation in
backward kernels or an unseeded stochastic-rounding source (FP16 tables use
`stochastic_rounding=True`). Consistent with the small in-process CPU/XPU
divergence measured by `probe_numerics_train_step_cpu_vs_xpu.py` (worst
relative loss difference 0.002% over 10 steps).

**Reproduce.** Two runs of `bash run_dlrm_v3_xpu.sh --phase 1 --mode train`;
`python compare_loss_curves.py RUN_A CPU_RUN --repeat RUN_B`.

**Owner.** torchlib-xpu / FBGEMM XPU kernels (low).

### H-11 Launcher sourced a version-pinned oneAPI path silently

**Root cause.** `source /opt/intel/oneapi/2025.3/oneapi-vars.sh … || true`; on a
2026.1 install the line did nothing and the run relied on the caller's shell.

**Status.** Fixed in the launcher (`ONEAPI_ROOT`, hard error).

**Owner.** harness.

### H-12 MLPerf LoadGen build writes into its source tree

**Root cause.** LoadGen's `setup.py` regenerates
`generated/version_generated.cc` and `mlperf_conf.h`, making the fork dirty in
every manifest.

**Workaround.** `install_dlrmv3.sh` builds LoadGen from a temporary copy.

**Owner.** harness.

### H-13 Two oneAPI versions in one process

**Observation.** The system compiler (2026.1.1) builds fbgemm-xpu; torch loads
the pip runtime wheels (`intel-sycl-rt` 2026.0.0). Both are recorded in
`manifest.json` under `runtime.oneapi`. No failure attributed to it so far.

**Owner.** platform (informational).

### H-14 `dlrm_v3` is not packaged by `setup.py`

**Root cause.** `find_packages()` only finds directories with `__init__.py`;
`dlrm_v3/`, `ops/` and `research/` have none, so an installed
generative-recommenders lacks them.

**Workaround.** The launcher puts the checkout on `PYTHONPATH`.

**Owner.** generative-recommenders upstream.

---

## Adding a finding

New findings get the next H- ID (`H-15`); the A- series is closed. Add a row to the index and a
section with: symptom, root cause (or "not investigated"), fix or workaround,
status on the current pins, evidence (run folder with its `manifest.json`, a
probe script, or a reproduction command), owner. Update the "Status checked
against" revisions when re-verifying.

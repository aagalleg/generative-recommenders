# DLRM-v3 on XPU — Probe index

One-off scripts, each written to answer a single question during the XPU
investigation. They are kept as they were written (no shared framework); this
file says what each one asks, which finding it served, what it needs to run,
and whether it still runs. Findings are in [FINDINGS.md](FINDINGS.md).

## Convention

- **Location:** [`probes/`](probes/) at the repository root.
- **Name:** `probe_<area>_<question>.<py|sh>`; helpers that answer no
  question are `tool_<purpose>.<ext>`. Areas in use: `env`, `kernel`, `nan`,
  `optim`, `ccl`, `profiler`, `numerics`; add new ones as needed.
- **Docstring / header comment:** first line is the question; then what was
  already established and the finding ID it serves.
- **One question per script.** Copy and adapt rather than parameterise.
- **Register every new probe** in the table below: question, finding, what it
  needs, status on the current pins, date checked.
- `import fbgemm_xpu` in every probe that runs fbgemm ops on XPU; without it
  the XPU kernels are not registered ([H-01](FINDINGS.md#h-01-xpu-kernels-are-only-registered-by-an-explicit-import-fbgemm_xpu)).

## Running a probe

Same environment as the launcher, from the repository root, one tile (the
`HSTU_EMBEDDING_DIM` / `HASH_SIZE` defaults, 64 and 1,000,000, are set by the
model-building probes themselves; `HSTU_*` variables other than these two do
nothing):

```bash
source /opt/intel/oneapi/setvars.sh
export PYTHONPATH=$PWD
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH"
ZE_AFFINITY_MASK=0 python probes/<probe>.py [args]
```

On a host with a CPU quota (for example a pod started with `--cpu=4`), also set
`OMP_NUM_THREADS` to the quota before running a `cpu` variant. Otherwise torch
sizes its thread pool from all the node's cores, and the run slows to a crawl.

**Needs** in the table:
- **pins**: runs on the current pinned revisions (see the status line in
  FINDINGS.md).
- **pins (adapted)**: written for the earlier XPU branch used in the first
  end-to-end trial (the A- findings), adapted to the pins on 2026-09-29 for the
  harness branch of PR #1 and again on 2026-10-26 for this branch
  (`probes/dlrm-v3-xpu-fbgemmv1.9.0`), without changing the question. Each
  such probe carries, inline: the repo-relative `train/gin/debug.gin`;
  `torch.device` instead of `xpu_compat.detect_device`; and the model dims it
  was written for. `dlrm_v3/harness.py` is now on this branch, but
  `get_hstu_configs` is still not gin-configurable, so the probe sets
  `configs.HSTU_EMBEDDING_DIM` / `configs.HASH_SIZE` from the environment
  (defaults 64 and 1,000,000) and wraps `train_utils.get_hstu_configs` to set
  the other dims on the returned config
  ([A-02](FINDINGS.md#a-02-hstu_-overrides-silently-inert)). The probes do
  not call `harness.apply_size_overrides()`, so they stay standalone and do
  not depend on the harness files. The
  `max_seq_len = 1024` cap of the harness-branch version is gone: this
  branch's `make_model` already caps it at 256 on an XPU host, so the HSTU
  attention is smaller than when the probes were written. (On the harness
  branch of PR #1 the probes use `harness.apply_size_overrides()` and
  `get_hstu_configs.*` gin bindings instead.)
- **earlier branch**: needs the earlier XPU branch of generative-recommenders,
  which is not in the pins, because it toggles something the pins do not
  expose (`DLRM_V3_*` optimizer switches) or drives `train_ranker.py` with
  variables and a directory layout (`/workspace/src-gr`) that only that branch
  had.

**Status** was measured on 2026-10-26 on the current pins (see FINDINGS.md:
torch 2.14.1, fbgemm-gpu-cpu 1.9.0, fbgemm-xpu built from torchlib-xpu `264f0c3`,
torchrec `c0fc555`), PVC one tile (`ZE_AFFINITY_MASK=0`), unless marked *not
run*. The previous status (2026-09-29, fbgemm 1.8.0, torch 2.13) is quoted
where it differs. The probes were not rerun on a B60. The `cpu` / `xpu` probes were run once
with each argument.

> **Do not run the `ccl` probes on a PVC across tiles** without reading
> [A-03](FINDINGS.md#a-03-two-rank-cross-tile-xccl-took-the-card-off-the-pcie-bus):
> sustained cross-tile XCCL traffic took a card off the PCIe bus permanently.

## Probes

| Probe | Question | Finding | Needs | Status on current pins |
| --- | --- | --- | --- | --- |
| [`probe_env_xpu_devices.py`](probes/probe_env_xpu_devices.py) | What does torch see on the XPU devices in this container? (+ bf16 matmul) | [A-07](FINDINGS.md#a-07-host-user-mode-driver-too-old-for-torch-213) | pins | Runs: torch 2.14.1+xpu sees one Max 1550 (64 GiB, 512 EUs, driver 1.6.33578+77, matrix MMA and bf16 convert reported); bf16 matmul OK. |
| [`probe_kernel_dispatch_registration.py`](probes/probe_kernel_dispatch_registration.py) | Did fbgemm-xpu register an XPU dispatch key for the ops DLRM-v3 needs? | — (bring-up check) | pins | Runs: embedding/TBE ops ("Stage A") 11/12 explicit `XPU`, jagged ops ("Stage B") 1/3. Both counts are misleading, not failures. Stage A's `dense_embedding_codegen_lookup_function` is registered under **`AutogradXPU`** (`fbgemm_dense_lookups_ops.cpp`), which the probe prints as "composite" and does not count; it counts only an exact `XPU` key. The two Stage B "MISSING" (`jagged_to_padded_dense`, `dense_to_jagged`) are `CompositeImplicitAutograd` over `*_forward`, which do have `XPU` kernels (checked). "Stage A/B", "PR #96" in the script are stale labels from the bring-up. Do not use it as a gate as written. |
| [`probe_kernel_jagged_vs_cpu.py`](probes/probe_kernel_jagged_vs_cpu.py) | Do the jagged ops behind the PYTORCH HSTU path match CPU, forward and backward? | — (bring-up check) | pins | Runs: 4/4 pass, max diff 0 (`jagged_to_padded_dense`, `dense_to_jagged`, `jagged_dense_elementwise_add_jagged_output`, and the `jagged_to_padded_dense` backward). |
| [`probe_kernel_tbe_dense_nobag.py`](probes/probe_kernel_tbe_dense_nobag.py) | Does the DENSE no-bag TBE lookup work on XPU, forward and backward? | [A-06](FINDINGS.md#a-06-xpu-was-dense-kernel-only-data_parallel-sharding) | pins + an FBGEMM whose DENSE TBE constructor does not call `torch.cuda` (not in the pins) | **Fails:** `Torch not compiled with CUDA enabled` in the `DenseTableBatchedEmbeddingBagsCodegen` constructor. On the 1.9.0 pins the line is `fbgemm_gpu/split_table_batched_embeddings_ops_training.py:5421` (`torch.cuda.current_device()`), so the cause is in `fbgemm_gpu`, not fbgemm-xpu (A-06). The DENSE *op* is registered for XPU; only the Python module is unusable. Current training does not use the DENSE path. |
| [`probe_nan_first_divergent_module.py`](probes/probe_nan_first_divergent_module.py) | Where does the unsharded XPU forward first diverge from CPU? | [A-01](FINDINGS.md#a-01-fp16-embedding-tables-train-to-nan--optimizer-epsilon-not-the-device) | pins (adapted) | Runs: identical input batches; aux loss CPU 0.138605 = XPU 0.138605; max\|diff\| 0 in the embedding modules shown; no XPU-only NaN in any module (2026-09-29: 0.138660 = 0.138660). |
| [`probe_nan_after_sharding.py`](probes/probe_nan_after_sharding.py) `cpu\|xpu` | Are the weights already NaN right after `DistributedModelParallel`, before any step? | A-01 | pins (adapted) | Runs (`xpu` and `cpu`): 0 NaN/Inf parameters before and after sharding; first forward finite (XPU aux loss 0.138161, CPU 0.139193; 71 predictions, 0 NaN; 2026-09-29: 0.138142 / 0.139062, 82 predictions). The two values come from separate processes and are not a like-for-like comparison; the in-process one is `probe_nan_first_divergent_module.py`. |
| [`probe_nan_tbe_init_alias.py`](probes/probe_nan_tbe_init_alias.py) | Does DENSE TBE init write NaN, or do the fp16 views not alias the real buffer? | A-01 | pins + an FBGEMM whose DENSE TBE constructor does not call `torch.cuda` (not in the pins) | **Fails** at section 3 with the same CUDA error as `probe_kernel_tbe_dense_nobag.py`. Sections 1 and 2 run: fp16 `uniform_` on XPU gives nan=0; an uninitialised fp16 `empty` on XPU gives 15361/1,000,000 NaN (1.5%, so uninitialised memory alone does not explain A-01's 100%). |
| [`probe_nan_rng_initialisers.py`](probes/probe_nan_rng_initialisers.py) | Are `uniform_` / `normal_` correct for fp16 on XPU across sizes? | A-01 | pins | Runs: nan=0, inf=0 for `uniform_` / `normal_` in fp32, fp16 and bf16 at every size up to 64M elements, and for `nn.Embedding.reset_parameters()` on CPU and XPU. |
| [`probe_nan_shard_weight_alias.py`](probes/probe_nan_shard_weight_alias.py) | Which plan is chosen, and do the named parameters still alias the TBE buffer after sharding? | A-01 | pins (adapted) | Runs: plan is `table_wise` + `fused` for all three tables (no `BatchedDenseEmbedding`); the named parameters are `TableBatchedEmbeddingSlice`s, fp16, 0/64M NaN each. |
| [`probe_nan_init_bounds.py`](probes/probe_nan_init_bounds.py) `cpu\|xpu` | Are TorchRec's `uniform_` init bounds NaN? | A-01 | pins (adapted; also reads the FUSED kernel's `weights_dev`) | Runs (`xpu` and `cpu`): one `BatchedFusedEmbedding`, fp16, bounds ±0.001, 0/192M NaN before and after init; parameters still alias the flat buffer (`still_aliases=True`). |
| [`probe_nan_corruption_stage.py`](probes/probe_nan_corruption_stage.py) | At which sharding stage does the embedding buffer turn NaN? | A-01 | pins (adapted; also watches the FUSED kernel's `weights_dev`) | Runs: buffer clean (0/192M NaN) after init and after `make_optimizer_and_shard`. There is no "around DDP" stage on this branch: XPU skips DDP. Without the `weights_dev` fallback it watched nothing and reported "clean" vacuously. |
| [`probe_nan_fp16_adam_eps.py`](probes/probe_nan_fp16_adam_eps.py) | Does a zero-gradient Adam step poison fp16 parameters? | A-01 (reproduction) | pins | Runs: **reproduces** — fp16, `eps=1e-8` (rounds to 0), zero gradient: 1024/1024 NaN on XPU and on CPU; fp32: 0; fp16 with `eps=1e-4`: 0. fp16 with a partly-zero gradient (8/1024 non-zero): 1016/1024 NaN on XPU, fp32: 0. Non-zero gradient: 0. |
| [`probe_optim_workaround_matrix.sh`](probes/probe_optim_workaround_matrix.sh) | Are the two XPU optimizer workarounds still needed? | A-01 | earlier branch (`DLRM_V3_FUSE_SPARSE_OPTIMIZER`, `DLRM_V3_OPTIMIZER_INIT_STATE`) | *Not run; obsolete.* The two switches do not exist on any pin. On this branch `make_optimizer_and_shard` calls `apply_optimizer_in_backward` for the sparse parameters and `init_state` on every device, so the workarounds the probe re-tested are no longer carried (A-01). It also uses `/workspace/src-gr` and `HSTU_*` variables that do nothing. Kept for the record; logs to `/tmp/wa_*.log`. |
| [`probe_ccl_2rank_transport.sh`](probes/probe_ccl_2rank_transport.sh) | Which oneCCL configuration lets a 2-rank XPU job start under `torchrun`? | [A-03](FINDINGS.md#a-03-two-rank-cross-tile-xccl-took-the-card-off-the-pcie-bus) | pins (standalone) | *Not run* (2 ranks across tiles; see A-03). Logs to `/tmp/xccl_*.log`. |
| [`probe_ccl_2rank_config_matrix.sh`](probes/probe_ccl_2rank_config_matrix.sh) | Which oneCCL configuration lets 2-rank DLRM-v3 training survive? | A-03 | earlier branch | *Not run; not adapted on purpose.* **Runs the cross-tile training workload that disabled the card** (A-03). |
| [`probe_profiler_activity_teardown.py`](probes/probe_profiler_activity_teardown.py) | Which `torch.profiler` activity combinations survive teardown on XPU? | [A-04](FINDINGS.md#a-04-profiler-aborts-the-process-on-teardown) | pins | Runs: all four cases (`cpu_only`, `cpu_cuda`, `cpu_xpu`, `xpu_only`) survive start/step/stop/teardown on torch 2.14.1. |
| [`probe_profiler_sigabrt_fix.py`](probes/probe_profiler_sigabrt_fix.py) | Does the DLRM-v3 profiler wrapper abort on XPU, and does calling `stop()` fix it? | A-04 | pins | Runs: still **reproduces** the Kineto `clearActivities` SIGABRT (`rc=-6`) for `xpu_nostop`; `repro_cuda_nostop`, `cuda_with_stop` and `xpu_with_stop` exit cleanly. The branch's `Profiler.__del__` calls `stop()`, so the DLRM-v3 wrapper is no longer affected (A-04); the 10-step `debug` run with `output_trace = True` exits 0. |
| [`probe_numerics_train_step_cpu_vs_xpu.py`](probes/probe_numerics_train_step_cpu_vs_xpu.py) | Do N training steps on XPU track CPU from bitwise-identical weights, in one process? | [H-10](FINDINGS.md#h-10-seeded-xpu-training-is-not-bit-reproducible) | pins (adapted; FP32 tables through a wrapper around `get_embedding_table_config`, otherwise A-01 masks the comparison) | Runs: 10 steps, worst relative loss difference 0.003% (2026-09-29: 0.002%), worst weight difference 1.2e-2 (Adam sign flips on near-zero gradients); verdict "tracks". The sharded, trend-level version is `run_dlrm_v3_xpu.sh --device cpu` + `compare_loss_curves.py` (10 steps on this branch: mean relative 10.6%, Pearson 0.932; see FINDINGS.md). |

## Tools

| Tool | Purpose | Status |
| --- | --- | --- |
| [`tool_sync_to_dut.sh`](probes/tool_sync_to_dut.sh) | Move a local branch to a test host as a **git bundle**, for hosts that cannot reach the GitHub remotes (SHAs are preserved). | Host from `DUT=user@host` (required); expects a clone at `~/dlrmv3-xpu/src-<name>`. *Not run.* |

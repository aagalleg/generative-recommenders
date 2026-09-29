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
- `import fbgemm_xpu` after `fbgemm_gpu` / `torchrec`, never before
  ([H-01](FINDINGS.md#h-01-xpu-kernels-are-only-registered-by-an-explicit-import-fbgemm_xpu)).

## Running a probe

Same environment as the launcher, from the repository root, one tile:

```bash
source /opt/intel/oneapi/setvars.sh
export PYTHONPATH=$PWD
export LD_LIBRARY_PATH="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))'):$LD_LIBRARY_PATH"
ZE_AFFINITY_MASK=0 python probes/<probe>.py [args]
```

**Needs** in the table:
- **pins**: runs on the current pinned revisions (see the status line in
  FINDINGS.md).
- **pins (adapted)**: written for the earlier XPU branch used in the first
  end-to-end trial (the A- findings), and adapted to the pins on 2026-09-29
  without changing the question. Each such probe carries, inline: the
  repo-relative `train/gin/debug.gin`; `harness.apply_size_overrides()`
  instead of `train_ranker._apply_size_overrides()`; `torch.device` instead of
  `xpu_compat.detect_device`; the model dims it was written for as
  `get_hstu_configs.*` gin bindings (as `HSTU_*` variables they are silently
  ignored on the pins,
  [A-02](FINDINGS.md#a-02-hstu_-overrides-silently-inert)); and a
  `max_seq_len = 1024` cap as a wrapper around `get_hstu_configs` (without it
  the PYTORCH HSTU attention asks for 31 GiB and runs out of memory).
- **earlier branch**: needs the earlier XPU branch of generative-recommenders,
  which is not in the pins, because it toggles something the pins do not
  expose (`DLRM_V3_*` optimizer switches) or drives `train_ranker.py` with
  variables and a directory layout (`/workspace/src-gr`) that only that branch
  had.

**Status** was measured on 2026-09-29 on the current pins, PVC one tile,
unless marked *not run*.

> **Do not run the `ccl` probes on a PVC across tiles** without reading
> [A-03](FINDINGS.md#a-03-two-rank-cross-tile-xccl-took-the-card-off-the-pcie-bus):
> sustained cross-tile XCCL traffic took a card off the PCIe bus permanently.

## Probes

| Probe | Question | Finding | Needs | Status on current pins |
| --- | --- | --- | --- | --- |
| [`probe_env_xpu_devices.py`](probes/probe_env_xpu_devices.py) | What does torch see on the XPU devices in this container? (+ bf16 matmul) | [A-07](FINDINGS.md#a-07-host-user-mode-driver-too-old-for-torch-213) | pins | Runs. |
| [`probe_kernel_dispatch_registration.py`](probes/probe_kernel_dispatch_registration.py) | Did fbgemm-xpu register an XPU dispatch key for the ops DLRM-v3 needs? | — (bring-up check) | pins | Runs: embedding/TBE ops ("Stage A") 11/12, jagged ops ("Stage B") 1/3. The two Stage B "MISSING" (`jagged_to_padded_dense`, `dense_to_jagged`) are false negatives: on the pins they are `CompositeImplicitAutograd` over `*_forward`, which have XPU kernels. |
| [`probe_kernel_jagged_vs_cpu.py`](probes/probe_kernel_jagged_vs_cpu.py) | Do the jagged ops behind the PYTORCH HSTU path match CPU, forward and backward? | — (bring-up check) | pins | Runs: 4/4 pass, max diff 0. |
| [`probe_kernel_tbe_dense_nobag.py`](probes/probe_kernel_tbe_dense_nobag.py) | Does the DENSE no-bag TBE lookup work on XPU, forward and backward? | [A-06](FINDINGS.md#a-06-xpu-was-dense-kernel-only-data_parallel-sharding) | pins + a torchlib-xpu with the DENSE TBE fix (not in the pins) | **Fails:** `Torch not compiled with CUDA enabled` in the `DenseTableBatchedEmbeddingBagsCodegen` constructor (A-06; the `_tbe_dense_compat.py` fix never reached the pinned torchlib-xpu). Current training does not use the DENSE path. |
| [`probe_nan_first_divergent_module.py`](probes/probe_nan_first_divergent_module.py) | Where does the unsharded XPU forward first diverge from CPU? | [A-01](FINDINGS.md#a-01-fp16-embedding-tables-train-to-nan--optimizer-epsilon-not-the-device) | pins (adapted) | Runs: identical input batches; aux loss CPU 0.138660 = XPU 0.138660; no XPU-only NaN in any module. |
| [`probe_nan_after_sharding.py`](probes/probe_nan_after_sharding.py) `cpu\|xpu` | Are the weights already NaN right after `DistributedModelParallel`, before any step? | A-01 | pins (adapted) | Runs (`xpu` and `cpu`): 0 NaN/Inf parameters before and after sharding; first forward finite (XPU 0.138142, CPU 0.139062; 82 predictions, 0 NaN). |
| [`probe_nan_tbe_init_alias.py`](probes/probe_nan_tbe_init_alias.py) | Does DENSE TBE init write NaN, or do the fp16 views not alias the real buffer? | A-01 | pins + a torchlib-xpu with the DENSE TBE fix (not in the pins) | **Fails:** same CUDA error as `probe_kernel_tbe_dense_nobag.py`. |
| [`probe_nan_rng_initialisers.py`](probes/probe_nan_rng_initialisers.py) | Are `uniform_` / `normal_` correct for fp16 on XPU across sizes? | A-01 | pins | Runs: nan=0 at every size (same as in the trial). |
| [`probe_nan_shard_weight_alias.py`](probes/probe_nan_shard_weight_alias.py) | Which plan is chosen, and do the named parameters still alias the TBE buffer after sharding? | A-01 | pins (adapted) | Runs: plan is `table_wise` + `fused` for all three tables (no `BatchedDenseEmbedding`); the named parameters are `TableBatchedEmbeddingSlice`s, fp16, 0 NaN. |
| [`probe_nan_init_bounds.py`](probes/probe_nan_init_bounds.py) `cpu\|xpu` | Are TorchRec's `uniform_` init bounds NaN? | A-01 | pins (adapted; also reads the FUSED kernel's `weights_dev`) | Runs (`xpu` and `cpu`): one `BatchedFusedEmbedding`, bounds ±0.001, 0/192M NaN before and after init; parameters still alias the flat buffer. |
| [`probe_nan_corruption_stage.py`](probes/probe_nan_corruption_stage.py) | At which sharding stage does the embedding buffer turn NaN? | A-01 | pins (adapted; also watches the FUSED kernel's `weights_dev`) | Runs: buffer clean (0/192M NaN) after init, around DDP and after `make_optimizer_and_shard`. Without the `weights_dev` fallback it watched nothing and reported "clean" vacuously. |
| [`probe_nan_fp16_adam_eps.py`](probes/probe_nan_fp16_adam_eps.py) | Does a zero-gradient Adam step poison fp16 parameters? | A-01 (reproduction) | pins | Runs: **reproduces** — fp16 with a partly-zero gradient: 1016/1024 NaN on XPU; fp32: 0. |
| [`probe_optim_workaround_matrix.sh`](probes/probe_optim_workaround_matrix.sh) | Are the two XPU optimizer workarounds still needed? | A-01 | earlier branch (`DLRM_V3_FUSE_SPARSE_OPTIMIZER`, `DLRM_V3_OPTIMIZER_INIT_STATE`) | *Not run; not adaptable.* On the pins both workarounds are hard-coded in `make_optimizer_and_shard` (fused params via sharders, `init_state` skipped on XPU) with no switch, so there is nothing to toggle. Logs to `/tmp/wa_*.log`. |
| [`probe_ccl_2rank_transport.sh`](probes/probe_ccl_2rank_transport.sh) | Which oneCCL configuration lets a 2-rank XPU job start under `torchrun`? | [A-03](FINDINGS.md#a-03-two-rank-cross-tile-xccl-took-the-card-off-the-pcie-bus) | pins (standalone) | *Not run* (2 ranks; small collectives only). Logs to `/tmp/xccl_*.log`. |
| [`probe_ccl_2rank_config_matrix.sh`](probes/probe_ccl_2rank_config_matrix.sh) | Which oneCCL configuration lets 2-rank DLRM-v3 training survive? | A-03 | earlier branch | *Not run; not adapted on purpose.* **Runs the cross-tile training workload that disabled the card** (A-03). |
| [`probe_profiler_activity_teardown.py`](probes/probe_profiler_activity_teardown.py) | Which `torch.profiler` activity combinations survive teardown on XPU? | [A-04](FINDINGS.md#a-04-profiler-aborts-the-process-on-teardown) | pins | Runs: `xpu_only` with start/stop survives. |
| [`probe_profiler_sigabrt_fix.py`](probes/probe_profiler_sigabrt_fix.py) | Does the DLRM-v3 profiler wrapper abort on XPU, and does calling `stop()` fix it? | A-04 | pins | Runs: **reproduces** the Kineto `clearActivities` abort without `stop()`; with `stop()` exits cleanly. |
| [`probe_numerics_train_step_cpu_vs_xpu.py`](probes/probe_numerics_train_step_cpu_vs_xpu.py) | Do N training steps on XPU track CPU from bitwise-identical weights, in one process? | [H-10](FINDINGS.md#h-10-seeded-xpu-training-is-not-bit-reproducible) | pins (adapted; FP32 tables through a wrapper around `get_embedding_table_config`, otherwise A-01 masks the comparison) | Runs: 10 steps, worst relative loss difference 0.002%, worst weight difference 1.2e-2 (Adam sign flips on near-zero gradients); verdict "tracks". The harness does the sharded, trend-level version: `run_dlrm_v3_xpu.sh --device cpu` + `compare_loss_curves.py`. |

## Tools

| Tool | Purpose | Status |
| --- | --- | --- |
| [`tool_sync_to_dut.sh`](probes/tool_sync_to_dut.sh) | Move a local branch to a test host as a **git bundle**, for hosts that cannot reach the GitHub remotes (SHAs are preserved). | Host from `DUT=user@host` (required); expects a clone at `~/dlrmv3-xpu/src-<name>`. *Not run.* |

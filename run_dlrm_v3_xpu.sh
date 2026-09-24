#!/usr/bin/env bash
# run_dlrm_v3_xpu.sh — Multi-phase launcher for DLRM-V3 on Intel XPU
#
# Usage:
#   bash run_dlrm_v3_xpu.sh [--phase 1|2|3] [--dataset movielens-1m-xpu] [--mode train-eval]
#
# Phases:
#   1  Single XPU, no DMP (validates model + ops)
#   2  Single XPU with DMP (validates sharding pipeline)
#   3  Multi-XPU with XCCL (validates distributed training)

set -eo pipefail

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
PHASE="${PHASE:-1}"
DATASET="${DATASET:-movielens-1m}"
MODE="${MODE:-train-eval}"
GIN_CONFIG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)   PHASE="$2";   shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --mode)    MODE="$2";    shift 2 ;;
        --gin)     GIN_CONFIG="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------
# Environment activation (set -u disabled: setvars.sh and conda.sh
# reference unset variables internally)
# ------------------------------------------------------------------
echo "=== Activating Intel oneAPI ==="
#source /opt/intel/oneapi/setvars.sh --force 2>/dev/null || true
source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force 2>/dev/null || true

echo "=== Activating conda environment ==="
#source /opt/miniforge/etc/profile.d/conda.sh
#conda activate dlrmV3-v3

# Re-enable nounset now that activation is done
set -u

# ------------------------------------------------------------------
# XPU-specific environment variables
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HSTU_EMBEDDING_DIM="${HSTU_EMBEDDING_DIM:-64}"
export HASH_SIZE="${HASH_SIZE:-1000000}"
export ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-COMPOSITE}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

# fbgemm_xpu (torchlib-xpu) native extensions (_C.so / _C_training.so) have no
# RPATH/RUNPATH baked in and need libtorch.so/libtorch_xpu.so/libtorch_cpu.so
# resolvable at dlopen time. fbgemm_xpu is imported lazily (in utils.py/main.py,
# after torch/torchrec/fbgemm_gpu are already loaded — importing it earlier
# causes a duplicate-schema-registration abort), but its
# native extension still needs libtorch resolvable at that later import time.
# Prepending torch's lib dir here fixes it regardless of when the import happens.
# TODO(upstream): report to torchlib-xpu maintainer — _C/_C_training should
# bake in a proper RPATH so this workaround isn't needed.
_TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))' 2>/dev/null || true)"
if [[ -n "${_TORCH_LIB}" ]]; then
    export LD_LIBRARY_PATH="${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

# CCL transport: use OFI/TCP instead of MPI to avoid Intel MPI 2021.16 segfault
# (MPIDI_GPU_init_mpl_global crashes on single-rank XPU collective init).
# FI_PROVIDER=tcp: force TCP libfabric provider — avoids EBADF/pidfd issues in
# non-container bare-metal environments with certain kernel versions.
# Both vars are required together; individually they don't prevent the crash.
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-ofi}"
export FI_PROVIDER="${FI_PROVIDER:-tcp}"

echo "=== XPU Configuration ==="
echo "  HSTU_EMBEDDING_DIM   = ${HSTU_EMBEDDING_DIM}"
echo "  HASH_SIZE            = ${HASH_SIZE}"
echo "  Phase                = ${PHASE}"
echo "  Dataset              = ${DATASET}"
echo "  Mode                 = ${MODE}"

# ------------------------------------------------------------------
# Detect available XPU devices
# ------------------------------------------------------------------
_DETECTED_XPUS=$(python -c "import torch; print(torch.xpu.device_count() if hasattr(torch,'xpu') and torch.xpu.is_available() else 0)")
NUM_XPUS="${NUM_XPUS:-${_DETECTED_XPUS}}"
echo "  XPU devices found  = ${_DETECTED_XPUS}"
echo "  NUM_XPUS (used)    = ${NUM_XPUS}"

if [[ "${NUM_XPUS}" -eq 0 ]]; then
    echo "ERROR: No XPU devices found. Ensure PyTorch+XPU is installed and Intel GPU drivers are loaded."
    exit 1
fi

# ------------------------------------------------------------------
# Common arguments
# ------------------------------------------------------------------
# Invoke train_ranker.py directly — xpu_launch.py (former wrapper for the
# torch.jit.script no-op / DDP init-sync bypass) was removed in Fase J; those
# patches were later consolidated into xpu_compat.py::apply_xpu_import_time_patches()
# and, as of 2026-07-23, removed entirely after being confirmed dead. HSTU
# config dim overrides are set via gin (get_hstu_configs.* bindings) instead
# of the env-var-driven monkey-patch that used to live in train_ranker.py.
TRAIN_SCRIPT="${SCRIPT_DIR}/generative_recommenders/dlrm_v3/train/train_ranker.py"

# Default gin config for XPU (unless overridden via --gin)
DEFAULT_GIN="${SCRIPT_DIR}/generative_recommenders/dlrm_v3/train/gin/movielens_1m_xpu.gin"

GIN_ARGS=""
if [[ -n "${GIN_CONFIG}" ]]; then
    GIN_ARGS="--gin_config_file ${GIN_CONFIG}"
else
    GIN_ARGS="--gin_config_file ${DEFAULT_GIN}"
fi

# ------------------------------------------------------------------
# Phase execution
# ------------------------------------------------------------------
case "${PHASE}" in
    1)
        echo "=== Phase 1: Single XPU, world_size=1 ==="
        python "${TRAIN_SCRIPT}" \
            --dataset "${DATASET}" \
            --mode "${MODE}" \
            --device_type xpu \
            --world_size 1 \
            ${GIN_ARGS}
        ;;
    2)
        echo "=== Phase 2: Single XPU with DMP ==="
        torchrun \
            --standalone \
            --nproc_per_node=1 \
            "${TRAIN_SCRIPT}" \
            --dataset "${DATASET}" \
            --mode "${MODE}" \
            --device_type xpu \
            ${GIN_ARGS}
        ;;
    3)
        echo "=== Phase 3: Multi-XPU with XCCL (${NUM_XPUS} devices) ==="
        # CCL_ZE_IPC_EXCHANGE: controls Level-Zero IPC buffer exchange mode.
        # - pidfd:   default; crashes with EBADF in container environments but
        #            works on bare-metal. Required for pt2pt (all_to_all) ops.
        # - sockets: avoids pidfd crash but does NOT support pt2pt → RuntimeError.
        # - drmfd:   NOT supported in this version of oneCCL (2021.17).
        # On bare-metal host (dut5044), use pidfd (override with CCL_ZE_IPC_EXCHANGE=sockets
        # only if running inside a container without pidfd_getfd support).
        CCL_ZE_IPC_EXCHANGE="${CCL_ZE_IPC_EXCHANGE:-pidfd}" \
        PYTHONFAULTHANDLER=1 \
        torchrun \
            --standalone \
            --nproc_per_node="${NUM_XPUS}" \
            "${TRAIN_SCRIPT}" \
            --dataset "${DATASET}" \
            --mode "${MODE}" \
            --device_type xpu \
            ${GIN_ARGS}
        ;;
    *)
        echo "ERROR: Unknown phase '${PHASE}'. Use 1, 2, or 3."
        exit 1
        ;;
esac

echo "=== Done (Phase ${PHASE}) ==="

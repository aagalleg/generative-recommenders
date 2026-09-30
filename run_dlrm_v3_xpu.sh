#!/usr/bin/env bash
# run_dlrm_v3_xpu.sh — Multi-phase launcher for DLRM-V3 on Intel XPU
#
# Usage:
#   bash run_dlrm_v3_xpu.sh [--phase 1|2|3] [--dataset movielens-1m] [--mode MODE] [--gin FILE]
#                           [--device xpu|cpu]
#
# Modes:
#   train | eval | train-eval | streaming-train-eval   train_ranker.py (default: train-eval)
#   infer   unquantized inference via inference/main.py (MLPerf LoadGen);
#           phase 1 only; LoadGen's mlperf_log_* files are written to RUN_DIR
#
# Devices:
#   xpu     default
#   cpu     CPU reference for the XPU loss curve: phase 1, training modes only;
#           compare with compare_loss_curves.py
#
# Environment overrides:
#   DATA_DIR     directory containing data/<dataset>/   (default: <repo>/datasets)
#   RUN_DIR      run output: run.gin, run.log, manifest.json, operative_config.gin
#                (default: <repo>/exps/xpu_runs/<ts>-...)
#   ONEAPI_ROOT  oneAPI install sourced if not already active (default: /opt/intel/oneapi)
#
# Phases:
#   1  Single device, one process spawned by train_ranker.py (validates model + ops;
#      the model is still sharded with DMP, world_size=1)
#   2  Single XPU, one process launched by torchrun (validates the torchrun launch path)
#   3  Multi-XPU with XCCL (validates distributed training)

set -eo pipefail

# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------
PHASE="${PHASE:-1}"
DATASET="${DATASET:-movielens-1m}"
MODE="${MODE:-train-eval}"
DEVICE="${DEVICE:-xpu}"
GIN_CONFIG=""
LAUNCH_COMMAND="$0 $*"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase)   PHASE="$2";   shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --mode)    MODE="$2";    shift 2 ;;
        --gin)     GIN_CONFIG="$2"; shift 2 ;;
        --device)  DEVICE="$2";  shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "${DEVICE}" in
    xpu) ;;
    cpu)
        [[ "${PHASE}" == 1 && "${MODE}" != infer ]] ||
            { echo "ERROR: --device cpu supports --phase 1 training modes only."; exit 1; }
        ;;
    *) echo "ERROR: Unknown device '${DEVICE}'. Use xpu or cpu."; exit 1 ;;
esac

# ------------------------------------------------------------------
# Environment activation. The Python environment is whatever `python`
# resolves to (the container's /opt/venv/mybuild is already on PATH).
# set -u stays off until after setvars.sh, which references unset vars.
# ------------------------------------------------------------------
echo "=== Activating Intel oneAPI ==="
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
if [[ "${SETVARS_COMPLETED:-}" == "1" ]]; then
    echo "  already active (SETVARS_COMPLETED=1)"
elif [[ -f "${ONEAPI_ROOT}/setvars.sh" ]]; then
    source "${ONEAPI_ROOT}/setvars.sh" --force > /dev/null ||
        { echo "ERROR: sourcing ${ONEAPI_ROOT}/setvars.sh failed."; exit 1; }
else
    echo "ERROR: ${ONEAPI_ROOT}/setvars.sh not found. Set ONEAPI_ROOT."
    exit 1
fi

set -u

# ------------------------------------------------------------------
# XPU-specific environment variables
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/datasets}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/exps/xpu_runs/$(date +%Y%m%d-%H%M%S)-phase${PHASE}-${MODE}-${DEVICE}}"
mkdir -p "${RUN_DIR}"
exec > >(tee "${RUN_DIR}/run.log") 2>&1
# train_ranker.py writes operative_config.gin here.
export DLRMV3_RUN_DIR="${RUN_DIR}"
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
echo "  Device               = ${DEVICE}"
echo "  DATA_DIR             = ${DATA_DIR}"
echo "  RUN_DIR              = ${RUN_DIR}"

# ------------------------------------------------------------------
# Detect available XPU devices
# ------------------------------------------------------------------
_DETECTED_XPUS=$(python -c "import torch; print(torch.xpu.device_count() if hasattr(torch,'xpu') and torch.xpu.is_available() else 0)")
NUM_XPUS="${NUM_XPUS:-${_DETECTED_XPUS}}"
echo "  XPU devices found  = ${_DETECTED_XPUS}"
echo "  NUM_XPUS (used)    = ${NUM_XPUS}"

if [[ "${DEVICE}" == xpu && "${NUM_XPUS}" -eq 0 ]]; then
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
INFER_SCRIPT="${SCRIPT_DIR}/generative_recommenders/dlrm_v3/inference/main.py"

# Default gin config for XPU (unless overridden via --gin), and the binding
# that carries this host's data path.
if [[ "${MODE}" == infer ]]; then
    DEFAULT_GIN="${SCRIPT_DIR}/generative_recommenders/dlrm_v3/inference/gin/movielens_1m_xpu.gin"
    DATA_BINDING="run.dataset_path_prefix"
else
    DEFAULT_GIN="${SCRIPT_DIR}/generative_recommenders/dlrm_v3/train/gin/movielens_1m_xpu.gin"
    DATA_BINDING="make_train_test_dataloaders.new_path_prefix"
fi

GIN_BASE="$(realpath -e "${GIN_CONFIG:-${DEFAULT_GIN}}")" ||
    { echo "ERROR: gin config not found: ${GIN_CONFIG:-${DEFAULT_GIN}}"; exit 1; }

# The dataset loader resolves <new_path_prefix>/data/<dataset>/..., so the
# prefix must point at a directory containing data/.
if [[ "${DATASET}" == movielens-1m && ! -f "${DATA_DIR}/data/ml-1m/sasrec_format.csv" ]]; then
    echo "ERROR: ${DATA_DIR}/data/ml-1m/sasrec_format.csv not found. Set DATA_DIR."
    exit 1
fi

# Per-run gin: the base config with its data path overridden for this host.
RUN_GIN="${RUN_DIR}/run.gin"
cat > "${RUN_GIN}" <<EOF
include '${GIN_BASE}'
${DATA_BINDING} = "${DATA_DIR}"
EOF
GIN_ARGS="--gin_config_file ${RUN_GIN}"

# Component revisions, runtime and env for this run. Written before training
# so that failed runs are traceable too; a run without it is not started.
echo "=== Run manifest: ${RUN_DIR}/manifest.json ==="
python "${SCRIPT_DIR}/write_run_manifest.py" \
    --out "${RUN_DIR}/manifest.json" \
    --command "${LAUNCH_COMMAND}" \
    --phase "${PHASE}" \
    --mode "${MODE}" \
    --device "${DEVICE}" \
    --dataset "${DATASET}" \
    --gin "${GIN_BASE}" \
    --data-dir "${DATA_DIR}" ||
    { echo "ERROR: writing the run manifest failed."; exit 1; }

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
if [[ "${MODE}" == infer ]]; then
    [[ "${PHASE}" == 1 ]] || { echo "ERROR: --mode infer supports --phase 1 only."; exit 1; }
    echo "=== Inference: single XPU, unquantized ==="
    # WORLD_SIZE=1 selects the single-worker dense path; HSTUModelFamily
    # otherwise uses one worker per visible device. LoadGen writes its logs
    # to the working directory.
    (cd "${RUN_DIR}" && WORLD_SIZE=1 python "${INFER_SCRIPT}" \
        --dataset "${DATASET}" \
        ${GIN_ARGS})
    echo "=== Done (inference) ==="
    exit 0
fi

# ------------------------------------------------------------------
# Phase execution
# ------------------------------------------------------------------
case "${PHASE}" in
    1)
        echo "=== Phase 1: single ${DEVICE}, world_size=1 ==="
        python "${TRAIN_SCRIPT}" \
            --dataset "${DATASET}" \
            --mode "${MODE}" \
            --device_type "${DEVICE}" \
            --world_size 1 \
            ${GIN_ARGS}
        ;;
    2)
        echo "=== Phase 2: single xpu via torchrun, world_size=1 ==="
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

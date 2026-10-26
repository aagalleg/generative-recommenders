#!/usr/bin/env bash
# Find a oneCCL configuration under which the 2-rank DLRM-v3 run survives.
#
# Small collectives already pass on this host, but the training run aborts in
# the driver (drm_neo.cpp:268) at the first heavy collective. Kernel 5.15 has no
# pidfd support, so CCL falls back to drmfd Level Zero IPC exchange, which this
# 2023-era KMD cannot service. These variants try to keep CCL off that path.
set -uo pipefail

cd /workspace/src-gr
export PYTHONPATH=/workspace/src-gr MASTER_ADDR=127.0.0.1
export HSTU_EMBEDDING_DIM=64 HASH_SIZE=1000000 HSTU_MAX_SEQ_LEN=1024
export HSTU_TRANSDUCER_DIM=128 HSTU_ATTN_NUM_LAYERS=3 HSTU_ATTN_LINEAR_DIM=64
export HSTU_ATTN_QK_DIM=64 HSTU_NUM_HEADS=4 HSTU_PREPROCESSOR_HIDDEN_DIM=128
export HSTU_TABLE_DTYPE=FP32 DLRM_V3_SEED=1234

run() {
  local name="$1" log="/tmp/r2_${1}.log"
  shift
  echo "=== ${name}: $* ==="
  env "$@" timeout 600 torchrun --standalone --nproc_per_node=2 \
    generative_recommenders/dlrm_v3/train/train_ranker.py \
    --dataset debug --mode train --device_type xpu --world_size 2 \
    > "$log" 2>&1
  local rc=$?
  local steps
  steps=$(grep -c "train - batch" "$log")
  echo "    exit=${rc} loss_lines=${steps}"
  if [ "$steps" -gt 0 ]; then
    grep "train - batch" "$log" | tail -2 | sed "s/^/      /"
  fi
  grep -oE "Abort was called at [0-9]+ line in file:|Duplicate ranks|drm_neo.cpp" "$log" \
    | sort -u | sed "s/^/      /"
  echo
}

run ze_ipc_sockets   CCL_ZE_IPC_EXCHANGE=sockets
run ze_disabled      CCL_ZE_ENABLE=0
run ze_disabled_sock CCL_ZE_ENABLE=0 CCL_ZE_IPC_EXCHANGE=sockets
run atl_shm_off      CCL_ZE_ENABLE=0 CCL_ATL_SHM=0

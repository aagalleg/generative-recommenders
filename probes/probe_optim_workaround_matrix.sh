#!/usr/bin/env bash
# Re-test the two XPU optimizer workarounds carried by the prior effort.
#
# Both were commented out with "temporarily disabled" notes. This runs the
# 10-step debug config with each re-enabled independently so their current
# status is a measurement rather than an assumption.
set -uo pipefail

cd /workspace/src-gr
export PYTHONPATH=/workspace/src-gr MASTER_ADDR=127.0.0.1
export HSTU_EMBEDDING_DIM=64 HASH_SIZE=1000000 HSTU_MAX_SEQ_LEN=1024
export HSTU_TRANSDUCER_DIM=128 HSTU_ATTN_NUM_LAYERS=3 HSTU_ATTN_LINEAR_DIM=64
export HSTU_ATTN_QK_DIM=64 HSTU_NUM_HEADS=4 HSTU_PREPROCESSOR_HIDDEN_DIM=128

run_case() {
  local name="$1" log="/tmp/wa_${1}.log"
  shift
  echo "=============== ${name} ==============="
  env "$@" timeout 600 python -u \
    generative_recommenders/dlrm_v3/train/train_ranker.py \
    --dataset debug --mode train --device_type xpu --world_size 1 \
    > "$log" 2>&1
  local rc=$?
  local steps
  steps=$(grep -c "train - Step" "$log")
  echo "exit=${rc} steps=${steps}/9"
  grep -E "optimizer groups:" "$log" | tail -1
  grep -E "fusing sparse optimizer|calling init_state on" "$log" | tail -2
  # Final accuracy, to catch a run that completes but stops learning.
  grep "train - Step 10" "$log" \
    | grep -oE "lifetime_accuracy/vvp100.: tensor\([0-9.]+" | tail -1
  if [ "$rc" -ne 0 ]; then
    echo "--- failure ---"
    grep -iE "Error|Traceback|violation|SIGSEGV|signal" "$log" \
      | grep -v "FutureWarning" | head -6
    tail -3 "$log" | cut -c1-200
  fi
  echo
}

run_case baseline                DLRM_V3_DUMMY=0
run_case fuse_sparse_optimizer   DLRM_V3_FUSE_SPARSE_OPTIMIZER=1
run_case optimizer_init_state    DLRM_V3_OPTIMIZER_INIT_STATE=1
run_case both                    DLRM_V3_FUSE_SPARSE_OPTIMIZER=1 DLRM_V3_OPTIMIZER_INIT_STATE=1

#!/usr/bin/env bash
# Find an oneCCL configuration that lets a 2-rank XPU job start under torchrun.
#
# The DLRM-v3 2-rank run aborts in MPI_Group_incl with "Duplicate ranks ...
# value 0", i.e. both ranks initialised MPI as singleton rank 0. setup() exports
# PMI_RANK/PMI_SIZE to help CCL find its rank, but that is also what makes CCL
# choose its MPI transport, and nothing launched MPI here.
set -uo pipefail

cat > /tmp/xccl_min.py <<'PY'
import os
import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
torch.xpu.set_device(rank)
dist.init_process_group("cpu:gloo,xpu:xccl", rank=rank, world_size=world)

t = torch.full((1024,), float(rank + 1), device=f"xpu:{rank}")
dist.all_reduce(t)
torch.xpu.synchronize()
expected = sum(range(1, world + 1))
ok = bool((t == expected).all())
if rank == 0:
    print(f"    all_reduce -> {t[0].item()} (expected {expected}) ok={ok}")
dist.destroy_process_group()
PY

run() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  env "$@" timeout 240 torchrun --standalone --nproc_per_node=2 /tmp/xccl_min.py \
    > "/tmp/xccl_${name}.log" 2>&1
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "    PASS"
    grep "all_reduce ->" "/tmp/xccl_${name}.log"
  else
    echo "    FAIL rc=${rc}"
    grep -iE "Duplicate ranks|Invalid rank|EXCEPTION|Abort|error" "/tmp/xccl_${name}.log" \
      | head -3
  fi
  echo
}

# Baseline: no PMI vars at all, which is how a plain torchrun job looks.
run bare CCL_ZE_IPC_EXCHANGE=sockets

# Reproduce what setup() does today.
run with_pmi CCL_ZE_IPC_EXCHANGE=sockets PMI_RANK=0 PMI_SIZE=2 \
  MPI_LOCALRANKID=0 MPI_LOCALNRANKS=2

# PMI vars present but transport pinned away from MPI.
run pmi_ofi CCL_ZE_IPC_EXCHANGE=sockets CCL_ATL_TRANSPORT=ofi \
  PMI_RANK=0 PMI_SIZE=2 MPI_LOCALRANKID=0 MPI_LOCALNRANKS=2

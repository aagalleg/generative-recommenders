"""Do TorchRec's init_parameters writes reach a FUSED TBE's real weights on XPU?

Fused sibling of probe_nan_tbe_init_alias.py (whose DENSE constructor cannot be
built on XPU, A-06). TorchRec's BaseBatchedEmbedding.init_parameters writes the
tables through param.data.uniform_() on the views returned by
split_embedding_weights(). The question is the same: after that write, are the
weights finite, i.e. do the views alias the TBE's real buffer and does uniform_
work for fp16 on XPU?

Steps 1-2 (plain uniform_ and an uninitialised fp16 tensor) are device-only and
identical to the DENSE probe.
"""

import torch

import fbgemm_gpu  # noqa: F401
import fbgemm_xpu  # noqa: F401
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.config.embedding_config import ComputeDevice


def nan_frac(t):
    t = t.detach().float()
    return int(torch.isnan(t).sum()), t.numel()


print("=== 1. plain uniform_ on xpu ===")
for dtype in (torch.float32, torch.float16):
    for n in (1024, 1_000_000 * 4):
        t = torch.empty(n, dtype=dtype, device="xpu")
        t.uniform_(-0.001, 0.001)
        torch.xpu.synchronize()
        n_nan, numel = nan_frac(t)
        print(
            f"  uniform_ dtype={str(dtype):<16} numel={numel:<9} "
            f"nan={n_nan:<10} min={t.float().min().item():.4g} "
            f"max={t.float().max().item():.4g}"
        )

print("\n=== 2. empty (uninitialised) fp16 on xpu, for reference ===")
t = torch.empty(1_000_000, dtype=torch.float16, device="xpu")
torch.xpu.synchronize()
n_nan, numel = nan_frac(t)
print(f"  empty fp16: nan={n_nan}/{numel}")

print("\n=== 3. SplitTableBatchedEmbeddingBagsCodegen (fused) on xpu ===")
rows, dim = 1_000_000, 64
tbe = SplitTableBatchedEmbeddingBagsCodegen(
    [(rows, dim, EmbeddingLocation.DEVICE, ComputeDevice.XPU) for _ in range(3)],
    pooling_mode=PoolingMode.NONE,
    optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
    weights_precision=SparseType.FP16,
    output_dtype=SparseType.FP32,
    device="xpu:0",
)
flat = tbe.weights_dev
print(f"  tbe.weights_dev: dtype={flat.dtype} device={flat.device} numel={flat.numel()}")
n_nan, numel = nan_frac(flat)
print(f"  weights straight after construction: nan={n_nan}/{numel}")

views = tbe.split_embedding_weights()
print(f"  split_embedding_weights -> {len(views)} views")
flat_ptr = flat.data_ptr()
flat_bytes = flat.numel() * flat.element_size()
for i, v in enumerate(views):
    off = v.data_ptr() - flat_ptr
    inside = 0 <= off < flat_bytes
    print(
        f"    view[{i}] shape={tuple(v.shape)} dtype={v.dtype} "
        f"device={v.device} byte_offset={off} aliases_weights={inside}"
    )

print("\n  writing via the views, as init_parameters does")
for v in views:
    v.data.uniform_(-0.001, 0.001)
torch.xpu.synchronize()

for i, v in enumerate(views):
    n_nan, numel = nan_frac(v)
    print(f"    view[{i}] after uniform_: nan={n_nan}/{numel}")
n_nan, numel = nan_frac(tbe.weights_dev)
print(f"  flat weights after uniform_ through views: nan={n_nan}/{numel}")

print("\n=== 4. writing the flat buffer directly ===")
tbe.weights_dev.data.uniform_(-0.001, 0.001)
torch.xpu.synchronize()
n_nan, numel = nan_frac(tbe.weights_dev)
print(f"  flat weights after direct uniform_: nan={n_nan}/{numel}")
for i, v in enumerate(tbe.split_embedding_weights()):
    n_nan, numel = nan_frac(v)
    print(f"    view[{i}] reads back: nan={n_nan}/{numel}")

"""Find why DENSE TBE embedding weights come out 100% NaN on XPU.

TorchRec's BaseBatchedEmbedding.init_parameters writes the tables through
param.data.uniform_() on the fp16 views returned by split_embedding_weights().
After sharding, all 64M elements of every table read as NaN, so either
uniform_ misbehaves for fp16 on XPU or those views do not alias the TBE's real
weight buffer, leaving it untouched.
"""

import torch

import fbgemm_gpu  # noqa: F401
import fbgemm_xpu  # noqa: F401
from fbgemm_gpu.split_table_batched_embeddings_ops_inference import PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    DenseTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.split_embedding_configs import SparseType


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

print("\n=== 3. DenseTableBatchedEmbeddingBagsCodegen on xpu ===")
rows, dim = 1_000_000, 64
tbe = DenseTableBatchedEmbeddingBagsCodegen(
    [(rows, dim), (rows, dim), (rows, dim)],
    pooling_mode=PoolingMode.NONE,
    use_cpu=False,
    weights_precision=SparseType.FP16,
    output_dtype=SparseType.FP32,
)
print(f"  tbe.weights: dtype={tbe.weights.dtype} device={tbe.weights.device} "
      f"numel={tbe.weights.numel()}")
n_nan, numel = nan_frac(tbe.weights)
print(f"  weights straight after construction: nan={n_nan}/{numel}")

views = tbe.split_embedding_weights()
print(f"  split_embedding_weights -> {len(views)} views")
flat_ptr = tbe.weights.data_ptr()
flat_bytes = tbe.weights.numel() * tbe.weights.element_size()
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
n_nan, numel = nan_frac(tbe.weights)
print(f"  flat weights after uniform_ through views: nan={n_nan}/{numel}")

print("\n=== 4. writing the flat buffer directly ===")
tbe.weights.data.uniform_(-0.001, 0.001)
torch.xpu.synchronize()
n_nan, numel = nan_frac(tbe.weights)
print(f"  flat weights after direct uniform_: nan={n_nan}/{numel}")
for i, v in enumerate(tbe.split_embedding_weights()):
    n_nan, numel = nan_frac(v)
    print(f"    view[{i}] reads back: nan={n_nan}/{numel}")

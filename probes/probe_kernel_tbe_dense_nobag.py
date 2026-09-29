"""Check the DENSE no-bag TBE lookup on XPU, forward and backward.

This is the embedding critical path for DLRM-v3: TorchRec restricts XPU to the
DENSE compute kernel, and EmbeddingCollection (what HSTU uses) is non-pooled, so
lookups arrive with pooling_mode=NONE. PR #127 implements only that route --
pooled SUM/MEAN raises "not implemented yet for SplitLookupFunction_dense_Op".

The reference is plain torch.nn.functional.embedding on CPU rather than the CPU
TBE, for two reasons: it is independent of fbgemm entirely, and the CPU TBE has
no Autograd key registered, so backprop through it is unreliable.
"""

import torch
import torch.nn.functional as F
import fbgemm_gpu  # noqa: F401
import fbgemm_xpu  # noqa: F401
from fbgemm_gpu.split_table_batched_embeddings_ops_common import PoolingMode
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    DenseTableBatchedEmbeddingBagsCodegen,
)

torch.manual_seed(0)

E, D, T, B, L = 1000, 64, 3, 16, 8  # rows/table, dim, tables, batch, indices per bag

# offsets has T*B+1 entries; row i of the no-bag output is table (i // L) // B.
indices = torch.randint(0, E, (T * B * L,), dtype=torch.int64)
offsets = torch.arange(0, T * B * L + 1, L, dtype=torch.int64)
init_weights = torch.randn(T, E, D)
grad_seed = torch.randn(T * B * L, D)


def reference_cpu():
    w = init_weights.clone().requires_grad_(True)
    table_of_row = (torch.arange(T * B * L) // L) // B
    flat = w[table_of_row, indices]  # [T*B*L, D]
    flat.backward(grad_seed)
    return flat.detach(), w.grad.detach()


def tbe_xpu():
    emb = DenseTableBatchedEmbeddingBagsCodegen(
        [(E, D) for _ in range(T)],
        pooling_mode=PoolingMode.NONE,
        use_cpu=False,
    )
    emb = emb.to("xpu:0")
    with torch.no_grad():
        emb.weights.copy_(init_weights.reshape(-1).to("xpu:0"))
    emb.weights.requires_grad_(True)
    out = emb(indices.to("xpu:0"), offsets.to("xpu:0"))
    out.backward(grad_seed.to("xpu:0"))
    torch.xpu.synchronize()
    return out.detach().float().cpu(), emb.weights.grad.detach().float().cpu()


ref_out, ref_grad = reference_cpu()
xpu_out, xpu_grad = tbe_xpu()

print(f"forward  shape ref={tuple(ref_out.shape)} xpu={tuple(xpu_out.shape)}")
fwd = (ref_out - xpu_out).abs().max().item()
print(f"forward  max|diff| = {fwd:.3e}")

ref_grad_flat = ref_grad.reshape(-1)
print(f"backward shape ref={tuple(ref_grad_flat.shape)} xpu={tuple(xpu_grad.shape)}")
bwd = (ref_grad_flat - xpu_grad).abs().max().item()
print(f"backward max|diff| = {bwd:.3e}")
print(f"backward grad nonzero elems = {int((xpu_grad != 0).sum())} / {xpu_grad.numel()}")

ok = fwd < 1e-4 and bwd < 1e-3 and torch.isfinite(xpu_grad).all()
print("RESULT:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)

"""Check the FUSED no-bag TBE lookup on XPU, forward and fused backward.

Fused sibling of probe_kernel_tbe_dense_nobag.py. Training places tables with
SplitTableBatchedEmbeddingBagsCodegen (TW / fused, optimizer applied in the
backward), not the DENSE module, whose constructor calls torch.cuda (A-06).

A fused TBE has no .grad on its weights: backward applies the optimizer in
place. The probe uses EXACT_ROWWISE_ADAGRAD (what training uses; the SGD route
goes through the _pt2 op, which rejects XPU tensors) and compares the updated
weights to a CPU reference of one rowwise-Adagrad step from zero state:
    m = mean(g^2, dim=-1);  w_after = w_before - lr * g / (sqrt(m) + eps)
Each row is looked up at most once per table, so g is the plain gradient.
"""

import torch
import fbgemm_gpu  # noqa: F401
import fbgemm_xpu  # noqa: F401
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    EmbeddingLocation,
    PoolingMode,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from fbgemm_gpu.tbe.config.embedding_config import ComputeDevice

torch.manual_seed(0)
LR, EPS = 0.1, 1e-8

E, D, T, B, L = 1000, 64, 3, 16, 8  # rows/table, dim, tables, batch, indices per bag

# Distinct rows per table, so the SGD update of one row sees one gradient and
# the reference gradient is unambiguous.
indices = torch.stack(
    [torch.randperm(E)[: B * L] for _ in range(T)]
).reshape(-1)
offsets = torch.arange(0, T * B * L + 1, L, dtype=torch.int64)
init_weights = torch.randn(T, E, D)
grad_seed = torch.randn(T * B * L, D)
table_of_row = (torch.arange(T * B * L) // L) // B


def reference_cpu():
    out = init_weights[table_of_row, indices]
    after = init_weights.clone()
    g = grad_seed
    step = LR * g / (g.pow(2).mean(dim=-1, keepdim=True).sqrt() + EPS)
    after[table_of_row, indices] -= step
    return out, after


def tbe_xpu():
    emb = SplitTableBatchedEmbeddingBagsCodegen(
        [(E, D, EmbeddingLocation.DEVICE, ComputeDevice.XPU) for _ in range(T)],
        pooling_mode=PoolingMode.NONE,
        optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        learning_rate=LR,
        eps=EPS,
        device="xpu:0",
    )
    for t, w in enumerate(emb.split_embedding_weights()):
        w.data.copy_(init_weights[t].to(w.device))
    torch.xpu.synchronize()
    out = emb(indices.to("xpu:0"), offsets.to("xpu:0"))
    out.backward(grad_seed.to("xpu:0"))
    torch.xpu.synchronize()
    after = torch.stack(
        [w.detach().float().cpu() for w in emb.split_embedding_weights()]
    )
    return out.detach().float().cpu(), after


ref_out, ref_after = reference_cpu()
xpu_out, xpu_after = tbe_xpu()

print(f"forward  shape ref={tuple(ref_out.shape)} xpu={tuple(xpu_out.shape)}")
fwd = (ref_out - xpu_out).abs().max().item()
print(f"forward  max|diff| = {fwd:.3e}")

print(f"weights  shape ref={tuple(ref_after.shape)} xpu={tuple(xpu_after.shape)}")
bwd = (ref_after - xpu_after).abs().max().item()
print(f"backward (updated weights) max|diff| = {bwd:.3e}")
changed = int((xpu_after != init_weights).sum())
print(f"backward updated elems = {changed} / {xpu_after.numel()}")

ok = fwd < 1e-4 and bwd < 1e-4 and changed > 0 and bool(torch.isfinite(xpu_after).all())
print("RESULT:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)

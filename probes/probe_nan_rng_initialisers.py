"""Test the RNG in-place initialisers on XPU across dtypes and sizes.

DMP._init_parameters replaces meta parameters with torch.empty_like and then
calls module.reset_parameters(); for nn.Embedding that is init.normal_. The
DLRM-v3 embedding tables are fp16 and come out 100% NaN, which is an active
write of NaN rather than uninitialised memory, so the initialiser itself is the
suspect.
"""

import torch


def check(label, t):
    torch.xpu.synchronize() if t.device.type == "xpu" else None
    f = t.detach().float()
    n = int(torch.isnan(f).sum())
    i = int(torch.isinf(f).sum())
    finite = f[torch.isfinite(f)]
    rng = (
        f"min={finite.min().item():.4g} max={finite.max().item():.4g} "
        f"std={finite.std().item():.4g}"
        if finite.numel()
        else "no finite values"
    )
    verdict = "BAD" if n or i else "ok"
    print(f"  {label:<52} nan={n:<10} inf={i:<8} {rng:<48} {verdict}")
    return n == 0 and i == 0


rows, dim = 1_000_000, 64
sizes = [(1024,), (65536,), (rows, dim)]

for dev in ("cpu", "xpu"):
    print(f"=== normal_() on {dev} ===")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for shape in sizes:
            t = torch.empty(shape, dtype=dtype, device=dev)
            t.normal_(0.0, 1.0)
            check(f"normal_ {str(dtype).replace('torch.',''):<10} {shape}", t)

print("\n=== nn.Embedding.reset_parameters() the way DMP calls it ===")
for dev in ("cpu", "xpu"):
    for dtype in (torch.float32, torch.float16):
        emb = torch.nn.Embedding(rows, dim, dtype=dtype, device="meta")
        emb.weight = torch.nn.Parameter(
            torch.empty_like(emb.weight, device=dev),
            requires_grad=True,
        )
        with torch.no_grad():
            emb.reset_parameters()
        check(f"{dev} nn.Embedding dtype={str(dtype).replace('torch.','')}", emb.weight)

print("\n=== does size matter for fp16 normal_ on xpu ===")
for n in (2**16, 2**20, 2**24, 2**25, 2**26, 64_000_000):
    t = torch.empty(n, dtype=torch.float16, device="xpu")
    t.normal_(0.0, 1.0)
    check(f"fp16 normal_ numel={n}", t)

"""Confirm that a zero-gradient Adam step poisons fp16 parameters.

KeyedOptimizer.init_state assigns param.grad = zeros_like(param) and then calls
step() to materialise optimizer state. For an fp16 parameter, Adam's eps=1e-8
is below the smallest fp16 subnormal (~5.96e-8) and rounds to zero, so the
update becomes exp_avg / (sqrt(exp_avg_sq) + 0) = 0/0 = NaN.

This matters on XPU specifically because the XPU path only exposes the DENSE
kernel, so the fp16 embedding tables land in the dense Adam group instead of a
fused sparse optimizer.
"""

import torch

print(f"fp16 smallest subnormal: {torch.finfo(torch.float16).tiny * 2**-10:.3g}")
print(f"fp16 smallest normal:    {torch.finfo(torch.float16).tiny:.3g}")
print(f"eps=1e-8 as fp16:        {torch.tensor(1e-8, dtype=torch.float16).item():.3g}")
print()

print(f"{'device':<7} {'dtype':<10} {'eps':<8} {'nan after zero-grad step':>26}")
for device in ("cpu", "xpu"):
    for dtype in (torch.float32, torch.float16):
        for eps in (1e-8, 1e-4):
            p = torch.nn.Parameter(
                torch.full((1024,), 0.001, dtype=dtype, device=device)
            )
            opt = torch.optim.Adam([p], lr=0.001, betas=(0.95, 0.999), eps=eps)
            p.grad = torch.zeros_like(p)
            opt.step()
            if device == "xpu":
                torch.xpu.synchronize()
            n = int(torch.isnan(p.detach().float()).sum())
            print(
                f"{device:<7} {str(dtype).replace('torch.',''):<10} {eps:<8.0e} "
                f"{f'{n}/{p.numel()}':>26}"
            )

print("\nsame check for a nonzero gradient, which is the normal training case")
for dtype in (torch.float32, torch.float16):
    p = torch.nn.Parameter(torch.full((1024,), 0.001, dtype=dtype, device="xpu"))
    opt = torch.optim.Adam([p], lr=0.001, betas=(0.95, 0.999), eps=1e-8)
    p.grad = torch.full_like(p, 0.01)
    opt.step()
    torch.xpu.synchronize()
    n = int(torch.isnan(p.detach().float()).sum())
    print(f"  xpu {str(dtype).replace('torch.',''):<10} nonzero grad -> nan={n}")

print("\npartially-zero gradient, as a sparsely-updated embedding table sees")
for dtype in (torch.float32, torch.float16):
    p = torch.nn.Parameter(torch.full((1024,), 0.001, dtype=dtype, device="xpu"))
    opt = torch.optim.Adam([p], lr=0.001, betas=(0.95, 0.999), eps=1e-8)
    g = torch.zeros_like(p)
    g[:8] = 0.01  # only a few rows touched by the batch
    p.grad = g
    opt.step()
    torch.xpu.synchronize()
    f = p.detach().float()
    n = int(torch.isnan(f).sum())
    print(
        f"  xpu {str(dtype).replace('torch.',''):<10} 8/1024 nonzero -> nan={n}/{p.numel()}"
    )

"""Check the Stage B jagged ops against their CPU implementations.

These are what HammerKernel.PYTORCH decomposes HSTU attention into, so they are
on the DLRM-v3 model-body critical path.
"""

import torch
import fbgemm_gpu  # noqa: F401
import fbgemm_xpu  # noqa: F401

torch.manual_seed(0)

B, D, MAXL = 8, 16, 12
lengths = torch.randint(1, MAXL + 1, (B,), dtype=torch.int64)
offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
total = int(lengths.sum())
values = torch.randn(total, D)
dense = torch.randn(B, MAXL, D)

results = []


def compare(name, cpu_fn, xpu_fn):
    try:
        c = cpu_fn()
        x = xpu_fn()
        c = c[0] if isinstance(c, (tuple, list)) else c
        x = x[0] if isinstance(x, (tuple, list)) else x
        torch.xpu.synchronize()
        diff = (c.float() - x.float().cpu()).abs().max().item()
        ok = diff < 1e-5
        print(f"  {name:<46} max|diff|={diff:.2e}  {'PASS' if ok else 'FAIL'}")
        results.append(ok)
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:<46} ERROR: {type(exc).__name__}: {str(exc)[:90]}")
        results.append(False)


print("=== Stage B jagged ops, XPU vs CPU ===")

compare(
    "jagged_to_padded_dense",
    lambda: torch.ops.fbgemm.jagged_to_padded_dense(values, [offsets], [MAXL], 0.0),
    lambda: torch.ops.fbgemm.jagged_to_padded_dense(
        values.to("xpu"), [offsets.to("xpu")], [MAXL], 0.0
    ),
)

compare(
    "dense_to_jagged",
    lambda: torch.ops.fbgemm.dense_to_jagged(dense, [offsets]),
    lambda: torch.ops.fbgemm.dense_to_jagged(dense.to("xpu"), [offsets.to("xpu")]),
)

compare(
    "jagged_dense_elementwise_add_jagged_output",
    lambda: torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        values, [offsets], dense
    ),
    lambda: torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        values.to("xpu"), [offsets.to("xpu")], dense.to("xpu")
    ),
)

# jagged_to_padded_dense is used inside autograd in the HSTU fallback, so the
# backward path matters as much as the forward.
print("=== autograd through jagged_to_padded_dense ===")
try:
    v_cpu = values.clone().requires_grad_(True)
    torch.ops.fbgemm.jagged_to_padded_dense(v_cpu, [offsets], [MAXL], 0.0).sum().backward()
    v_xpu = values.clone().to("xpu").requires_grad_(True)
    torch.ops.fbgemm.jagged_to_padded_dense(
        v_xpu, [offsets.to("xpu")], [MAXL], 0.0
    ).sum().backward()
    torch.xpu.synchronize()
    d = (v_cpu.grad - v_xpu.grad.cpu()).abs().max().item()
    ok = d < 1e-5
    print(f"  backward max|diff|={d:.2e}  {'PASS' if ok else 'FAIL'}")
    results.append(ok)
except Exception as exc:  # noqa: BLE001
    print(f"  backward ERROR: {type(exc).__name__}: {str(exc)[:110]}")
    results.append(False)

print(f"\nSUMMARY: {sum(results)}/{len(results)} passed")
raise SystemExit(0 if all(results) else 1)

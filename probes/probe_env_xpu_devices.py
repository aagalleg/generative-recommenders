"""Report what torch sees on the XPU devices in this container."""

import torch

print("torch      :", torch.__version__)
print("xpu avail  :", torch.xpu.is_available())
print("device cnt :", torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    p = torch.xpu.get_device_properties(i)
    print(f"  xpu:{i} {p.name}")
    print(f"        {p.total_memory / 2**30:.0f} GiB | EUs={p.gpu_eu_count} | driver={p.driver_version}")
    print(f"        matrix_mma={p.has_subgroup_matrix_multiply_accumulate} bf16_conv={p.has_bfloat16_conversions}")

a = torch.randn(2048, 2048, device="xpu:0", dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device="xpu:0", dtype=torch.bfloat16)
c = (a @ b).float()
torch.xpu.synchronize()
print(f"matmul bf16 : OK, sum={c.sum().item():.1f}")

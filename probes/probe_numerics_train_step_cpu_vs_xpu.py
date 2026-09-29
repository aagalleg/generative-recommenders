"""Compare N full training steps on XPU against CPU from identical weights.

The sharded training runs cannot be compared point-wise: embedding tables are
initialised by uniform_ on the device, so each device draws different initial
weights, and the planner also gives them different embedding optimizers. Both
effects swamp any kernel difference.

This removes both. One process, one model built on CPU and deep-copied to XPU,
the same batches, the same optimizer on both, and a real forward/backward/step
each iteration -- so the comparison covers gradients and the update, not just
the forward pass.
"""

import copy
import logging
import os

import torch

logging.basicConfig(level=logging.ERROR)

for k, v in {
    "HSTU_EMBEDDING_DIM": "64",
    "HASH_SIZE": "100000",
}.items():
    os.environ.setdefault(k, v)

import gin  # noqa: E402

import fbgemm_xpu  # noqa: F401,E402
from generative_recommenders.common import HammerKernel  # noqa: E402
from generative_recommenders.dlrm_v3.datasets.dataset import collate_fn  # noqa: E402
from generative_recommenders.dlrm_v3 import harness  # noqa: E402
from generative_recommenders.dlrm_v3.train.utils import (  # noqa: E402
    HammerToTorchDataset,
    make_model,
)
from generative_recommenders.dlrm_v3.utils import get_dataset  # noqa: E402

gin.parse_config_file(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../generative_recommenders/dlrm_v3/train/gin/debug.gin",
    )
)
# The model dims this probe was written for, set as gin bindings (movielens_1m_xpu.gin);
# as environment variables they would be silently ignored (A-02).
for k, v in {
    "hstu_transducer_embedding_dim": 128,
    "hstu_attn_num_layers": 3,
    "hstu_attn_linear_dim": 64,
    "hstu_attn_qk_dim": 64,
    "hstu_num_heads": 4,
    "hstu_preprocessor_hidden_dim": 128,
}.items():
    gin.bind_parameter(f"get_hstu_configs.{k}", v)
harness.apply_size_overrides("xpu")

# Cap max_seq_len at 1024: the pins keep the 16384 default, sized for the
# fused Triton kernel; the PYTORCH HSTU fallback is O(N^2) and runs out of memory.
from generative_recommenders.dlrm_v3.train import utils as train_utils  # noqa: E402

_get_hstu = train_utils.get_hstu_configs


def _capped_hstu(dataset):
    cfg = _get_hstu(dataset)
    cfg.max_seq_len = 1024
    return cfg


train_utils.get_hstu_configs = _capped_hstu

# Force FP32 tables: the pins hard-code FP16 tables, and FP16 with
# dense Adam eps=1e-8 turns to NaN on both devices (A-01).
from torchrec.modules.embedding_configs import DataType  # noqa: E402

_get_tables = train_utils.get_embedding_table_config


def _fp32_tables(dataset):
    tables = _get_tables(dataset)
    for t in tables.values():
        t.data_type = DataType.FP32
    return tables


train_utils.get_embedding_table_config = _fp32_tables

STEPS, BATCH, SEED = 10, 16, 1234

torch.manual_seed(SEED)
model, hstu_config, table_configs = make_model()
model.set_hammer_kernel(HammerKernel.PYTORCH)
model.to_empty(device="cpu")
torch.manual_seed(SEED)
with torch.no_grad():
    for p in model.parameters():
        p.normal_(0.0, 0.02) if p.is_floating_point() else p.zero_()
    for b in model.buffers():
        b.normal_(0.0, 0.02) if b.is_floating_point() else b.zero_()
model.train()

model_cpu = model
model_xpu = copy.deepcopy(model_cpu).to("xpu")

# Identical starting weights is the whole point of the exercise; check it.
worst_w = max(
    (a - b.cpu()).abs().max().item()
    for a, b in zip(model_cpu.parameters(), model_xpu.parameters())
)
print(f"max |weight difference| at start: {worst_w:.3e}")
assert worst_w == 0.0, "models did not start identical"

opt_cpu = torch.optim.Adam(model_cpu.parameters(), lr=1e-3, betas=(0.95, 0.999), eps=1e-8)
opt_xpu = torch.optim.Adam(model_xpu.parameters(), lr=1e-3, betas=(0.95, 0.999), eps=1e-8)

dataset_class, kwargs = get_dataset(name="debug")
kwargs["embedding_config"] = table_configs
ds = HammerToTorchDataset(
    dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
)


def batch_for(step, dev):
    torch.manual_seed(9000 + step)
    b = collate_fn([ds[step * BATCH + i] for i in range(BATCH)])
    b.to(dev)
    return b


print(f"\n{'step':>4} {'cpu loss':>12} {'xpu loss':>12} {'abs diff':>11} {'rel':>9} {'max|w diff|':>12}")
worst_rel = 0.0
worst_weight_drift = 0.0
for step in range(STEPS):
    losses = {}
    for tag, m, opt in (("cpu", model_cpu, opt_cpu), ("xpu", model_xpu, opt_xpu)):
        b = batch_for(step, tag)
        opt.zero_grad()
        out = m.forward(b.uih_features_kjt, b.candidates_features_kjt)
        loss = sum(out[2].values())
        loss.backward()
        opt.step()
        losses[tag] = loss.item()
    torch.xpu.synchronize()

    drift = max(
        (a - b.cpu()).abs().max().item()
        for a, b in zip(model_cpu.parameters(), model_xpu.parameters())
    )
    d = abs(losses["cpu"] - losses["xpu"])
    rel = d / max(abs(losses["cpu"]), 1e-12)
    worst_rel = max(worst_rel, rel)
    worst_weight_drift = max(worst_weight_drift, drift)
    print(
        f"{step:>4} {losses['cpu']:>12.6f} {losses['xpu']:>12.6f} "
        f"{d:>11.3e} {rel:>8.3%} {drift:>12.3e}"
    )

print(f"\nworst relative loss divergence over {STEPS} steps: {worst_rel:.3%}")
print(f"worst weight divergence:                        {worst_weight_drift:.3e}")

# Adam normalises by sqrt(exp_avg_sq), so its step size is ~lr regardless of
# gradient magnitude. Where a gradient is near zero, a tiny difference between
# devices can flip its sign and produce full-magnitude opposite updates. That
# shows up as weight divergence without moving the loss, so name the parameters
# involved rather than leaving the number unexplained.
print(f"\n{'most divergent parameters':<58} {'max|w diff|':>12} {'|grad|':>11} {'|w|':>11}")
rows = []
for (n, a), (_, b) in zip(model_cpu.named_parameters(), model_xpu.named_parameters()):
    d = (a - b.cpu()).abs().max().item()
    g = a.grad.abs().max().item() if a.grad is not None else float("nan")
    rows.append((d, n, g, a.abs().max().item()))
for d, n, g, w in sorted(rows, reverse=True)[:5]:
    print(f"{n:<58} {d:>12.3e} {g:>11.3e} {w:>11.3e}")
print(
    "\nverdict: "
    + (
        "XPU tracks CPU through forward, backward and optimizer step"
        if worst_rel < 0.02
        else "DIVERGENT -- investigate"
    )
)

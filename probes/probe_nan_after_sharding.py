"""Check the DLRM-v3 model for NaN immediately after sharding on XPU.

Unsharded, XPU and CPU produce an identical loss, so the nan seen in the real
training run comes from something only the sharded path touches. This mirrors
_main_func up to the first forward and inspects parameters right after
DistributedModelParallel, before any optimizer step, to separate "weights
arrive corrupt" from "the lookup kernel computes garbage".

usage: probe_nan_after_sharding.py <cpu|xpu>
"""

import logging
import os
import sys

import torch

logging.basicConfig(level=logging.WARNING)

DEV = sys.argv[1] if len(sys.argv) > 1 else "xpu"

for k, v in {
    "HSTU_EMBEDDING_DIM": "64",
    "HASH_SIZE": "1000000",
}.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29591")

import gin  # noqa: E402

import fbgemm_xpu  # noqa: F401,E402
from generative_recommenders.dlrm_v3.datasets.dataset import collate_fn  # noqa: E402
from generative_recommenders.dlrm_v3 import harness  # noqa: E402
from generative_recommenders.dlrm_v3.train.utils import (  # noqa: E402
    HammerToTorchDataset,
    make_model,
    make_optimizer_and_shard,
    setup,
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
harness.apply_size_overrides(DEV)

# Cap max_seq_len at 1024: the pins keep the 16384 default, sized for the
# fused Triton kernel; the PYTORCH HSTU fallback is O(N^2) and runs out of memory.
from generative_recommenders.dlrm_v3.train import utils as train_utils  # noqa: E402

_get_hstu = train_utils.get_hstu_configs


def _capped_hstu(dataset):
    cfg = _get_hstu(dataset)
    cfg.max_seq_len = 1024
    return cfg


train_utils.get_hstu_configs = _capped_hstu

device = torch.device("xpu", 0) if DEV == "xpu" else torch.device("cpu")
setup(rank=0, world_size=1, master_port=int(os.environ["MASTER_PORT"]), device=device)

torch.manual_seed(1234)
model, hstu_config, table_configs = make_model()
if DEV == "xpu":
    from generative_recommenders.common import HammerKernel

    model.set_hammer_kernel(HammerKernel.PYTORCH)


def report(tag, module):
    rows = []
    for name, p in module.named_parameters():
        if not p.is_floating_point():
            continue
        t = p.detach()
        if hasattr(t, "local_shards"):  # ShardedTensor
            shards = t.local_shards()
            t = shards[0].tensor if shards else None
            if t is None:
                continue
        if t.device.type == "meta":
            continue
        t = t.float()
        n, i = int(torch.isnan(t).sum()), int(torch.isinf(t).sum())
        if n or i:
            rows.append((name, tuple(p.shape), n, i, t.numel()))
    print(f"\n[{tag}] params containing NaN/Inf: {len(rows)}")
    for name, shape, n, i, numel in rows[:15]:
        print(f"  {name:<62} {str(shape):<16} nan={n}/{numel} inf={i}")
    return rows


report("pre-shard", model)

model, optimizer = make_optimizer_and_shard(
    model=model, device=device, world_size=1
)
if DEV == "xpu":
    torch.xpu.synchronize()

post = report("post-shard", model)

# Embedding weights specifically: these are what the TBE lookup reads.
print("\n[embedding weight stats]")
for name, p in model.named_parameters():
    if "embedding" not in name.lower() or not p.is_floating_point():
        continue
    t = p.detach()
    if hasattr(t, "local_shards"):
        shards = t.local_shards()
        if not shards:
            continue
        t = shards[0].tensor
    t = t.float()
    print(
        f"  {name:<62} shape={tuple(p.shape)} dtype={p.dtype} "
        f"min={t.min().item():.4g} max={t.max().item():.4g} "
        f"mean={t.mean().item():.4g} nan={int(torch.isnan(t).sum())}"
    )

dataset_class, kwargs = get_dataset(name="debug")
kwargs["embedding_config"] = table_configs
ds = HammerToTorchDataset(
    dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
)
torch.manual_seed(777)
batch = collate_fn([ds[i] for i in range(16)])
batch.to(device)

out = model.forward(batch.uih_features_kjt, batch.candidates_features_kjt)
if DEV == "xpu":
    torch.xpu.synchronize()

print("\n[first forward]")
for k, v in out[2].items():
    print(f"  aux_loss {k} = {v.item():.6f}")
preds = out[3]
for k, v in (preds.items() if isinstance(preds, dict) else [("preds", preds)]):
    t = v.detach().float()
    print(
        f"  {k}: nan={int(torch.isnan(t).sum())}/{t.numel()} "
        f"min={t.min().item():.4g} max={t.max().item():.4g}"
    )

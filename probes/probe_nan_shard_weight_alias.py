"""Determine why the sharded DLRM-v3 embedding tables read as 100% NaN on XPU.

Established so far: the DENSE TBE initialises correctly in isolation, normal_
and uniform_ are fine for fp16 on XPU, and uninitialised XPU fp16 memory is
only ~1.5% NaN -- so 100% NaN is neither a broken initialiser nor plain garbage.

This inspects the sharded model itself: the sharding plan actually chosen, the
BatchedDenseEmbedding flat weight buffer, and whether the named parameters
still alias that buffer after DistributedModelParallel has run.
"""

import logging
import os

import torch

logging.basicConfig(level=logging.WARNING)

for k, v in {
    "HSTU_EMBEDDING_DIM": "64",
    "HASH_SIZE": "1000000",
}.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29593")

import gin  # noqa: E402

import fbgemm_xpu  # noqa: F401,E402
from generative_recommenders.common import HammerKernel  # noqa: E402
from generative_recommenders.dlrm_v3 import harness  # noqa: E402
from generative_recommenders.dlrm_v3.train.utils import (  # noqa: E402
    make_model,
    make_optimizer_and_shard,
    setup,
)

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

device = torch.device("xpu", 0)
setup(rank=0, world_size=1, master_port=int(os.environ["MASTER_PORT"]), device=device)

torch.manual_seed(1234)
model, hstu_config, table_configs = make_model()
model.set_hammer_kernel(HammerKernel.PYTORCH)
model, optimizer = make_optimizer_and_shard(model=model, device=device, world_size=1)
torch.xpu.synchronize()

print("=== sharding plan ===")
for path, cfgs in model.plan.plan.items():
    for name, ps in cfgs.items():
        print(f"  {path}.{name}: type={ps.sharding_type} kernel={ps.compute_kernel}")


def nan_of(t):
    f = t.detach().float()
    return int(torch.isnan(f).sum()), f.numel()


print("\n=== BatchedDenseEmbedding modules found ===")
found = 0
for name, mod in model.named_modules():
    emb_module = getattr(mod, "_emb_module", None)
    if emb_module is None or not hasattr(emb_module, "weights"):
        continue
    found += 1
    w = emb_module.weights
    n, numel = nan_of(w)
    print(f"  {name}")
    print(f"    type={type(mod).__name__} emb={type(emb_module).__name__}")
    print(f"    flat weights: dtype={w.dtype} device={w.device} numel={numel} nan={n}")
    base = w.data_ptr()
    span = numel * w.element_size()
    for i, v in enumerate(emb_module.split_embedding_weights()):
        vn, vnumel = nan_of(v)
        off = v.data_ptr() - base
        print(
            f"    view[{i}] shape={tuple(v.shape)} nan={vn}/{vnumel} "
            f"byte_offset={off} aliases_flat={0 <= off < span}"
        )
    print(f"    named_parameters of {name}:")
    for pname, p in mod.named_parameters():
        pn, pnumel = nan_of(p)
        off = p.data_ptr() - base
        print(
            f"      {pname}: dtype={p.dtype} nan={pn}/{pnumel} "
            f"byte_offset={off} aliases_flat={0 <= off < span}"
        )
if not found:
    print("  none -- embeddings are not going through BatchedDenseEmbedding")

print("\n=== top-level embedding params as the train loop sees them ===")
for name, p in model.named_parameters():
    if "embedding_collection" not in name:
        continue
    n, numel = nan_of(p)
    print(
        f"  {name}: dtype={p.dtype} shape={tuple(p.shape)} "
        f"nan={n}/{numel} type={type(p).__name__} ptr={p.data_ptr()}"
    )

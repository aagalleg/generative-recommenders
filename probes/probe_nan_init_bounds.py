"""Instrument TorchRec's embedding init to explain the 100% NaN tables on XPU.

init_parameters fills each table with param.data.uniform_(weight_init_min,
weight_init_max). uniform_ with NaN bounds yields exactly 100% NaN, which
matches what the sharded model shows, so this logs the bounds and the buffer
state either side of the call.

Runs on both devices: the same config trains fine on CPU, so anything that
differs between the two is the lead.

usage: probe_nan_init_bounds.py <cpu|xpu>
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
os.environ.setdefault("MASTER_PORT", "29595")

import gin  # noqa: E402

import fbgemm_xpu  # noqa: F401,E402
from torchrec.distributed import batched_embedding_kernel as bek  # noqa: E402

_orig_init = bek.BaseBatchedEmbedding.init_parameters
seen_modules = []


def traced_init_parameters(self):
    seen_modules.append(self)
    print(f"\n  [init_parameters] {type(self).__name__}")
    print(f"    weight_init_mins = {self._weight_init_mins}")
    print(f"    weight_init_maxs = {self._weight_init_maxs}")
    print(f"    local_rows       = {self._local_rows}")
    print(f"    local_cols       = {self._local_cols}")
    # DENSE kernel: .weights; FUSED kernel (the pins, A-01): .weights_dev
    weights = getattr(self._emb_module, "weights", None)
    if weights is None:
        weights = getattr(self._emb_module, "weights_dev", None)
    if weights is not None:
        f = weights.detach().float()
        print(
            f"    flat weights before: dtype={weights.dtype} device={weights.device} "
            f"numel={f.numel()} nan={int(torch.isnan(f).sum())}"
        )
    _orig_init(self)
    if weights is not None:
        if weights.device.type == "xpu":
            torch.xpu.synchronize()
        f = weights.detach().float()
        finite = f[torch.isfinite(f)]
        print(
            f"    flat weights after : nan={int(torch.isnan(f).sum())}/{f.numel()} "
            + (
                f"min={finite.min().item():.4g} max={finite.max().item():.4g}"
                if finite.numel()
                else "no finite values"
            )
        )


bek.BaseBatchedEmbedding.init_parameters = traced_init_parameters

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
    model.set_hammer_kernel(HammerKernel.PYTORCH)

print(f"=== sharding on {DEV} ===")
model, optimizer = make_optimizer_and_shard(model=model, device=device, world_size=1)
if DEV == "xpu":
    torch.xpu.synchronize()

def describe(t):
    f = t.detach().float()
    finite = f[torch.isfinite(f)]
    body = (
        f"min={finite.min().item():.4g} max={finite.max().item():.4g}"
        if finite.numel()
        else "no finite values"
    )
    return f"nan={int(torch.isnan(f).sum())}/{f.numel()} {body}"


print("\n=== the same TBE buffers, re-read after sharding ===")
for mod in seen_modules:
    weights = getattr(mod._emb_module, "weights", None)
    if weights is None:
        continue
    print(f"  {type(mod).__name__}.weights ptr={weights.data_ptr()}")
    print(f"    {describe(weights)}")
    for i, v in enumerate(mod._emb_module.split_embedding_weights()):
        print(f"    fresh view[{i}] ptr={v.data_ptr()}: {describe(v)}")

print("\n=== embedding params after sharding ===")
for name, p in model.named_parameters():
    if "embedding_collection" not in name:
        continue
    orig = getattr(p, "_original_tensor", None)
    extra = ""
    if orig is not None:
        extra = (
            f"\n      _original_tensor ptr={orig.data_ptr()} "
            f"still_aliases={orig.data_ptr() == p.data_ptr() - p.storage_offset() * p.element_size()}"
            f"\n      _original_tensor: {describe(orig)}"
        )
    print(f"  {name} ptr={p.data_ptr()}\n      {describe(p)}{extra}")

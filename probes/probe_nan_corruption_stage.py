"""Pinpoint the stage of sharding that fills the XPU embedding buffer with NaN.

Established: the DENSE TBE buffer is correctly initialised (nan=0, bounds
[-0.001, 0.001]), the pointer never changes, and by the end of
make_optimizer_and_shard the very same address reads 100% NaN. So something
overwrites all 384 MB in between.

Watches that exact buffer around every DistributedDataParallel construction and
around the sharding calls, and reports the first stage at which it goes bad.
"""

import logging
import os

import torch

logging.basicConfig(level=logging.ERROR)

for k, v in {
    "HSTU_EMBEDDING_DIM": "64",
    "HASH_SIZE": "1000000",
}.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29597")

import gin  # noqa: E402

import fbgemm_xpu  # noqa: F401,E402
from torchrec.distributed import batched_embedding_kernel as bek  # noqa: E402

WATCH = {}
stages = []


def snapshot(label):
    w = WATCH.get("weights")
    if w is None:
        return
    torch.xpu.synchronize()
    f = w.detach().float()
    n = int(torch.isnan(f).sum())
    finite = f[torch.isfinite(f)]
    body = (
        f"min={finite.min().item():.5g} max={finite.max().item():.5g}"
        if finite.numel()
        else "NO FINITE VALUES"
    )
    stages.append((label, n, f.numel()))
    print(f"  {label:<56} nan={n:>10}/{f.numel()} {body}")


_orig_init = bek.BaseBatchedEmbedding.init_parameters


def traced_init(self):
    _orig_init(self)
    # DENSE kernel: .weights; FUSED kernel (the pins, A-01): .weights_dev
    w = getattr(self._emb_module, "weights", None)
    if w is None:
        w = getattr(self._emb_module, "weights_dev", None)
    if w is not None and "weights" not in WATCH:
        WATCH["weights"] = w
        print(f"\nwatching {type(self).__name__}.weights ptr={w.data_ptr()}")
        snapshot("after BatchedDenseEmbedding.init_parameters")


bek.BaseBatchedEmbedding.init_parameters = traced_init

_orig_ddp_init = torch.nn.parallel.DistributedDataParallel.__init__
_ddp_n = 0


def traced_ddp_init(self, *args, **kwargs):
    global _ddp_n
    _ddp_n += 1
    n = _ddp_n
    mod = kwargs.get("module", args[0] if args else None)
    snapshot(f"before DDP #{n} ({type(mod).__name__})")
    _orig_ddp_init(self, *args, **kwargs)
    snapshot(f"after  DDP #{n} ({type(mod).__name__})")


torch.nn.parallel.DistributedDataParallel.__init__ = traced_ddp_init

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

print("=== stages ===")
model, optimizer = make_optimizer_and_shard(model=model, device=device, world_size=1)
snapshot("after make_optimizer_and_shard")

first_bad = next((s for s in stages if s[1] > 0), None)
print(
    f"\nfirst stage with NaN: {first_bad[0] if first_bad else 'none -- buffer stayed clean'}"
)

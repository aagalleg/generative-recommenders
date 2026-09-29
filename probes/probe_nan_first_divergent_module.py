"""Find where the XPU forward pass diverges from CPU in the DLRM-v3 model body.

The 10-step debug run completes and its accuracy metrics move, but the loss is
nan from batch 0 on XPU while CPU gives ~0.14.

Runs both devices in one process from bitwise-identical weights and inputs, and
diffs every module output, so the report is the first module that diverges
rather than the first that happens to contain a NaN.

Unsharded on purpose: DistributedModelParallel and the sharded TBE lookup are
out of the picture, so a divergence here isolates to the HSTU model body.
"""

import copy
import logging
import os

import torch

logging.basicConfig(level=logging.WARNING)

for k, v in {
    "HSTU_EMBEDDING_DIM": "64",
    "HASH_SIZE": "1000000",
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

BATCH, SEED = 16, 1234


def build_model():
    torch.manual_seed(SEED)
    model, hstu_config, table_configs = make_model()
    model.set_hammer_kernel(HammerKernel.PYTORCH)
    # TorchRec builds the embedding modules on meta for sharding, so the
    # parameters have no storage until to_empty().
    model.to_empty(device="cpu")
    torch.manual_seed(SEED)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.02) if p.is_floating_point() else p.zero_()
        for b in model.buffers():
            b.normal_(0.0, 0.02) if b.is_floating_point() else b.zero_()
    model.train()
    return model, hstu_config, table_configs


model_cpu, hstu_config, table_configs = build_model()

dataset_class, kwargs = get_dataset(name="debug")
kwargs["embedding_config"] = table_configs
ds = HammerToTorchDataset(
    dataset=dataset_class(hstu_config=hstu_config, is_inference=False, **kwargs)
)


def make_batch():
    torch.manual_seed(777)
    return collate_fn([ds[i] for i in range(BATCH)])


batch_cpu, batch_xpu = make_batch(), make_batch()
same = torch.equal(
    batch_cpu.uih_features_kjt.values(), batch_xpu.uih_features_kjt.values()
)
print(f"identical input batches: {same}")
assert same, "dataset generation is not reproducible; comparison would be invalid"

model_xpu = copy.deepcopy(model_cpu).to("xpu")
batch_xpu.to("xpu")

captured = {"cpu": {}, "xpu": {}}


def hook(name, tag):
    def fn(mod, args, out):
        t = out[0] if isinstance(out, (tuple, list)) and out else out
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            captured[tag][name] = t.detach().float().cpu()

    return fn


order = []
handles = []
for name, mod in model_cpu.named_modules():
    order.append(name)
    handles.append(mod.register_forward_hook(hook(name, "cpu")))
for name, mod in model_xpu.named_modules():
    handles.append(mod.register_forward_hook(hook(name, "xpu")))

out_cpu = model_cpu.forward(
    batch_cpu.uih_features_kjt, batch_cpu.candidates_features_kjt
)
out_xpu = model_xpu.forward(
    batch_xpu.uih_features_kjt, batch_xpu.candidates_features_kjt
)
torch.xpu.synchronize()
for h in handles:
    h.remove()

print("\naux_losses:")
for k in out_cpu[2]:
    c, x = out_cpu[2][k].item(), out_xpu[2][k].item()
    print(f"  {k:<24} cpu={c:<14.6f} xpu={x:.6f}")

print(f"\n{'module':<52} {'shape':<20} {'max|diff|':>11} {'xpu nan':>9} {'cpu nan':>9}")
first_bad = None
for name in order:
    c, x = captured["cpu"].get(name), captured["xpu"].get(name)
    if c is None or x is None or c.shape != x.shape:
        continue
    xn, cn = int(torch.isnan(x).sum()), int(torch.isnan(c).sum())
    d = (
        float("nan")
        if (xn or cn)
        else (c - x).abs().max().item()
    )
    flag = ""
    if first_bad is None and (xn and not cn):
        first_bad = name
        flag = "  <== first XPU-only NaN"
    print(
        f"{name or '<root>':<52} {str(tuple(x.shape)):<20} {d:>11.3e} {xn:>9} {cn:>9}{flag}"
    )

print(f"\nfirst module with XPU NaN and clean CPU: {first_bad}")

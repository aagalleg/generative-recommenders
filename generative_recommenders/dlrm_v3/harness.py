# pyre-strict
"""Run-harness helpers shared by the train and inference entry points."""

import os
import random
from typing import Dict

import gin
import numpy as np
import torch


@gin.configurable
def seed_everything(seed: int = 0) -> None:
    """Seed python, numpy and torch (CPU and all accelerators)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def record_loss(rank: int, step: int, losses: Dict[str, torch.Tensor]) -> None:
    """Rank 0 appends this step's losses to $DLRMV3_RUN_DIR/losses.csv."""
    run_dir = os.environ.get("DLRMV3_RUN_DIR")
    if rank != 0 or not run_dir:
        return
    path = os.path.join(run_dir, "losses.csv")
    names = sorted(losses)
    values = [float(losses[k].detach().float()) for k in names]
    new_file = not os.path.exists(path)
    with open(path, "a") as f:
        if new_file:
            f.write(",".join(["step", *names, "total"]) + "\n")
        f.write(",".join([str(step), *map(repr, values), repr(sum(values))]) + "\n")


def apply_size_overrides(device_type: str) -> None:
    """Set configs.HSTU_EMBEDDING_DIM / HASH_SIZE from the environment.

    They are shared between get_hstu_configs() and get_embedding_table_config()
    in configs.py, so they are module globals rather than gin bindings.
    Applied on XPU (defaults sized for XPU memory) and, on any other device,
    when either variable is set, so a CPU reference builds the same model.
    """
    if device_type != "xpu" and not {"HSTU_EMBEDDING_DIM", "HASH_SIZE"} & set(os.environ):
        return
    import generative_recommenders.dlrm_v3.configs as configs

    configs.HSTU_EMBEDDING_DIM = int(os.environ.get("HSTU_EMBEDDING_DIM", "64"))
    configs.HASH_SIZE = int(os.environ.get("HASH_SIZE", "1000000"))


def write_operative_config(rank: int, operative: bool = True) -> None:
    """Rank 0 writes the gin bindings actually used, plus the configs.py
    globals set outside gin, to $DLRMV3_RUN_DIR (set by run_dlrm_v3_xpu.sh).
    With operative=False (before the model is built, so that early failures
    leave a record) all parsed bindings are written instead."""
    run_dir = os.environ.get("DLRMV3_RUN_DIR")
    if rank != 0 or not run_dir:
        return
    import generative_recommenders.dlrm_v3.configs as configs

    with open(os.path.join(run_dir, "operative_config.gin"), "w") as f:
        if not operative:
            f.write("# PARTIAL: written before model construction; all parsed\n")
            f.write("# bindings (gin.config_str()), not the operative config.\n\n")
        f.write("# configs.py module globals (not gin bindings), effective values:\n")
        f.write(f"# HSTU_EMBEDDING_DIM = {configs.HSTU_EMBEDDING_DIM}\n")
        f.write(f"# HASH_SIZE = {configs.HASH_SIZE}\n\n")
        f.write(gin.operative_config_str() if operative else gin.config_str())

#!/usr/bin/env python3
"""Write RUN_DIR/manifest.json: the component revisions, runtime and
environment a DLRM-v3 XPU run was started with.

Called by run_dlrm_v3_xpu.sh after its environment is set up, so module
resolution (PYTHONPATH, venv) matches the training process. Components are
located with importlib.util.find_spec, not imported, to avoid the
fbgemm_xpu import-order abort. The operative gin config is written separately
by train_ranker.py (operative_config.gin), since it only exists in-process.
"""

import argparse
import base64
import datetime
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

ENV_PREFIXES = ("HSTU_", "CCL_", "FI_", "ZE_")
ENV_KEYS = ("HASH_SIZE",)
ONEAPI_PIP_PREFIXES = (
    "intel-", "onemkl", "oneccl", "impi", "dpcpp", "tcmlib", "umf", "mkl", "tbb",
)
MAX_LISTED_FILES = 50


def _git(path: Path, *args: str) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], stderr=subprocess.DEVNULL, text=True
        ).rstrip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_info(path: Path) -> Optional[Dict[str, Any]]:
    top = _git(path, "rev-parse", "--show-toplevel")
    if top is None:
        return None
    # Untracked files are not counted: builds generate some (torchrec/version.py).
    modified = _git(Path(top), "status", "--porcelain", "--untracked-files=no") or ""
    modified_files = [line[3:] for line in modified.splitlines()]
    return {
        "path": top,
        "sha": _git(Path(top), "rev-parse", "HEAD"),
        "branch": _git(Path(top), "rev-parse", "--abbrev-ref", "HEAD"),
        "commit_time_utc": _iso(int(_git(Path(top), "log", "-1", "--format=%ct") or 0)),
        "dirty": bool(modified_files),
        "modified_files": modified_files[:MAX_LISTED_FILES],
    }


def _iso(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_record(dist: md.Distribution) -> List[str]:
    """Installed files whose content no longer matches the dist's RECORD,
    i.e. files patched after installation."""
    modified = []
    for f in dist.files or []:
        if f.hash is None or f.hash.mode != "sha256":
            continue
        path = Path(dist.locate_file(f))
        if not path.is_file():
            modified.append(f"{f} (missing)")
            continue
        digest = base64.urlsafe_b64encode(
            bytes.fromhex(_sha256_file(path))
        ).rstrip(b"=").decode()
        if digest != f.hash.value:
            modified.append(str(f))
    return modified


def _direct_url_dir(dist: md.Distribution) -> Optional[Path]:
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    url = json.loads(raw).get("url", "")
    return Path(unquote(urlparse(url).path)) if url.startswith("file://") else None


def _module_dirs(module: str) -> List[Path]:
    spec = importlib.util.find_spec(module)
    if spec is None:
        raise RuntimeError(f"module {module} not found")
    return [Path(p) for p in (spec.submodule_search_locations or [])] or [
        Path(spec.origin).parent
    ]


def source_component(module: str, dist_name: str) -> Dict[str, Any]:
    """Component whose Python sources are imported from a git checkout
    (PYTHONPATH or editable install)."""
    dirs = _module_dirs(module)
    git = next((g for g in (git_info(d) for d in dirs) if g), None)
    if git is None:
        raise RuntimeError(f"{module} is not imported from a git checkout: {dirs}")
    out: Dict[str, Any] = {
        "kind": "git-source",
        "module_path": [str(d) for d in dirs],
        "version": md.version(dist_name),
        "dirty": git["dirty"],
        "git": git,
    }
    # Compiled extensions of an editable install are built once, outside the
    # tree; they carry no revision, so record their identity and build time.
    binaries = sorted(p for d in dirs for p in d.glob("*.so") if not str(p).startswith(git["path"]))
    if binaries:
        built = min(p.stat().st_mtime for p in binaries)
        out["binaries"] = [
            {"path": str(p), "sha256": _sha256_file(p), "mtime_utc": _iso(p.stat().st_mtime)}
            for p in binaries
        ]
        out["binaries_older_than_head_commit"] = _iso(built) < git["commit_time_utc"]
    return out


def installed_component(module: str, dist_name: str) -> Dict[str, Any]:
    """Component imported from site-packages (a wheel or non-editable build)."""
    dist = md.distribution(dist_name)
    record = dist.read_text("RECORD")
    modified = verify_record(dist)
    out: Dict[str, Any] = {
        "kind": "installed",
        "module_path": [str(d) for d in _module_dirs(module)],
        "dist": dist_name,
        "version": dist.version,
        # Content fingerprint of the installed wheel (per-file sha256 list).
        "record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None,
        "dirty": bool(modified),
        "modified_files": modified[:MAX_LISTED_FILES],
    }
    src = _direct_url_dir(dist)
    if src is not None:
        # Built from a local tree; the tree may have moved on since the build.
        out["built_from"] = git_info(src) or {"path": str(src)}
        local = dist.version.partition("+")[2]
        if local and out["built_from"].get("sha"):
            out["build_sha_matches_source_head"] = out["built_from"]["sha"].startswith(local)
    return out


def oneapi_info() -> Dict[str, Any]:
    root = Path(os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi"))
    icpx = None
    try:
        icpx = subprocess.check_output(["icpx", "--version"], text=True).splitlines()[0]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    pip_runtime = {
        d.metadata["Name"]: d.version
        for d in md.distributions()
        if d.metadata["Name"] and d.metadata["Name"].lower().startswith(ONEAPI_PIP_PREFIXES)
    }
    return {
        # What torch loads at run time comes from these pip wheels ...
        "pip_runtime": dict(sorted(pip_runtime.items())),
        # ... while the system install built fbgemm-xpu.
        "system_root": str(root),
        "system_compiler": os.path.realpath(root / "compiler" / "latest"),
        "icpx": icpx,
    }


def torch_info() -> Dict[str, Any]:
    import torch

    info: Dict[str, Any] = {
        "version": torch.__version__,
        "xpu_build": getattr(torch.version, "xpu", None),
        "xpu_available": torch.xpu.is_available(),
        "devices": [],
    }
    if torch.xpu.is_available():
        for i in range(torch.xpu.device_count()):
            p = torch.xpu.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": p.name,
                "driver_version": getattr(p, "driver_version", None),
                "total_memory_gib": round(p.total_memory / 2**30, 1),
            })
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--command", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gin", required=True)
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    manifest = {
        "schema": 1,
        "created_utc": _iso(datetime.datetime.now().timestamp()),
        "host": platform.node(),
        "kernel": platform.release(),
        "launch": {
            "command": args.command,
            "phase": args.phase,
            "mode": args.mode,
            "dataset": args.dataset,
            "gin_base": args.gin,
            "data_dir": args.data_dir,
        },
        "components": {
            "torchlib-xpu": source_component("fbgemm_xpu", "fbgemm-xpu"),
            "fbgemm": installed_component("fbgemm_gpu", "fbgemm-gpu-cpu"),
            "torchrec": installed_component("torchrec", "torchrec"),
            "generative-recommenders": source_component(
                "generative_recommenders", "generative-recommenders"
            ),
        },
        "runtime": {
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch": torch_info(),
            "oneapi": oneapi_info(),
        },
        "env": {
            k: v for k, v in sorted(os.environ.items())
            if k.startswith(ENV_PREFIXES) or k in ENV_KEYS
        },
        "files": {
            "gin": "run.gin",
            "operative_config": "operative_config.gin",
            "log": "run.log",
        },
    }
    Path(args.out).write_text(json.dumps(manifest, indent=2) + "\n")

    for name, c in manifest["components"].items():
        rev = c.get("git", c.get("built_from", {})).get("sha") or c["version"]
        print(f"  {name:24s} {rev[:12]:12s} {'DIRTY' if c['dirty'] else 'clean'}")


if __name__ == "__main__":
    main()

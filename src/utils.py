"""
utils.py

WHAT TO WRITE HERE:
- `set_seed(seed: int)`: seed `random`, `numpy`, and `torch` (+ `torch.cuda`) for
  reproducibility across runs (config.yaml's `seed`).
- `get_device()`: return "cuda" if available else "mps" (Apple Silicon) else "cpu".
- `load_config(path: str) -> dict`: load config.yaml via `yaml.safe_load`.
- `save_checkpoint(model, optimizer, epoch, path)` / `load_checkpoint(model, optimizer, path)`:
  wrap `torch.save`/`torch.load` for model+optimizer state dicts.
- A simple logger setup (or just configure Python's `logging` module once here) so all
  scripts/modules can `import logging; logging.info(...)` consistently instead of
  scattering `print()` calls.
"""

import random
from pathlib import Path
import yaml
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """TODO: implement — seed random, numpy, torch, torch.cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """TODO: implement — return best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_config(path: str = "config.yaml") -> dict:
    """TODO: implement — yaml.safe_load(open(path))."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_checkpoint(model, optimizer, epoch: int, path: str) -> None:
    """Save model + optimizer state and the epoch to `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(model, optimizer, path: str, device: str = "cpu") -> int:
    """Load weights into `model` (and `optimizer` if given); returns the saved epoch."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]

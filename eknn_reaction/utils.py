"""Small shared utilities: seeding, logging, checkpoint I/O, meters."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    ``deterministic=True`` is intended for final benchmark runs.  It disables
    cuDNN autotuning/TF32 and asks PyTorch to warn when an operation lacks a
    deterministic CUDA implementation.  The environment variable is also set
    here for callers that invoke this utility before their first CUDA use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(pref: str = "cpu") -> torch.device:
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str = "eknn_reaction", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s",
                                         "%H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def save_checkpoint(path: str, model: torch.nn.Module, extra: Dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model: torch.nn.Module, map_location: str = "cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    return payload


def dump_json(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

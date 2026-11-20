"""Evaluation metrics for reaction yield prediction.

Beyond average MAE/RMSE we report the *distribution* of per-substrate errors
(mean + tail), because the whole point of individualisation is improving the
worst-served substrates -- a claim that averages can hide.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-8))


def per_group_summary(group_errors: List[float]) -> Dict[str, float]:
    arr = np.asarray(group_errors, dtype=np.float64)
    return dict(
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        worst=float(arr.max()),
    )

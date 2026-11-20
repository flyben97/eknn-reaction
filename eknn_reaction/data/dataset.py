"""Target-value standardisation utility."""

from __future__ import annotations

import numpy as np


class Standardizer:
    """Simple mean/std standardiser for target values.

    Used by GNN training scripts (ekNN and gradient TTT) to normalise
    yield values during training and denormalise predictions during
    evaluation.
    """

    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def normalize(self, y: np.ndarray) -> np.ndarray:
        return (y - self.mean) / max(self.std, 1e-8)

    def denormalize(self, y: np.ndarray) -> np.ndarray:
        return y * self.std + self.mean

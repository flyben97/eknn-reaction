"""Numerically stable distance weighting for residual and label retrieval."""
import numpy as np


def temperature_weights(distances, temperature):
    """Normalize over the last axis, including far neighbours at low temperature."""
    d = np.asarray(distances, dtype=float)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if d.ndim not in (1, 2) or d.shape[-1] == 0 or not np.isfinite(d).all():
        raise ValueError("distances must be finite nonempty vectors or a matrix")
    weights = np.exp(-(d - d.min(axis=-1, keepdims=True)) / temperature)
    return weights / weights.sum(axis=-1, keepdims=True)

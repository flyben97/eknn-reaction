"""Unit tests for the blinded HiTEA target-memory baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "develop_hitea_target_memory_baseline",
    ROOT / "scripts" / "develop_hitea_target_memory_baseline.py",
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_leaf_neighbours_rank_matching_leaf_vectors_first():
    memory = np.asarray([[1, 2, 3], [1, 2, 4], [9, 9, 9]])
    query = np.asarray([[1, 2, 3]])
    top, similarity, weights = runner.leaf_neighbours(query, memory, k=2)
    assert top[0, 0] == 0
    assert similarity[0, 0] == 1.0
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_affine_residual_reduces_to_global_offset_for_constant_base():
    memory_base = np.asarray([2.0, 2.0, 2.0])
    memory_y = np.asarray([3.0, 4.0, 5.0])
    query_base = np.asarray([2.0, 7.0])
    prediction = runner.affine_residual_prediction(memory_base, memory_y, query_base)
    assert np.allclose(prediction, [4.0, 9.0])


def test_weighted_memory_uses_selected_values():
    values = np.asarray([10.0, 20.0, 100.0])
    top = np.asarray([[0, 1]])
    weights = np.asarray([[0.75, 0.25]])
    assert np.allclose(runner.weighted_memory(values, top, weights), [12.5])


"""Regression tests for neighbour weighting and graph cache invalidation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('release_training', ROOT/'scripts/train_gnn_eknn.py')
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


def test_distant_single_neighbour_keeps_its_full_value():
    query = np.array([[1., 0.]])
    memory = np.array([[0., 1.]])
    values = np.array([3.])
    assert np.allclose(training.knn_correct(query, memory, values, 1, .05), [3.])
    assert np.allclose(training.knn_label_predict(query, memory, values, 1, .05), [3.])
    indices, distances = training.prepare_knn_neighbours(query, memory, 1)
    assert np.allclose(training.aggregate_prepared_neighbours(indices, distances, values, 1, .05), [3.])


def test_direct_and_prepared_paths_agree_far_from_memory():
    query = np.array([[1., 0.]], dtype=np.float32)
    memory = np.array([[-1., 0.], [0., -1.], [0., 1.]], dtype=np.float32)
    values = np.array([7., -2., 5.])
    indices, distances = training.prepare_knn_neighbours(query, memory, 3)
    direct = training.knn_correct(query, memory, values, 3, .01)
    cached = training.aggregate_prepared_neighbours(indices, distances, values, 3, .01)
    assert np.allclose(direct, cached)
    assert values.min() <= direct[0] <= values.max()


def test_nonpositive_temperature_rejected():
    from eknn_reaction.retrieval import temperature_weights
    with pytest.raises(ValueError):
        temperature_weights([1., 2.], 0.)


def test_cache_respects_fraction_data_and_legacy_namespace(tmp_path):
    train = pd.DataFrame({'S': ['C', 'CC', 'CCC', 'CO', 'CN', 'CF'], 'Y': np.arange(6.)})
    test = pd.DataFrame({'S': ['CCO', 'CCN', 'CCF', 'CCl'], 'Y': np.arange(6., 10.)})
    train.to_csv(tmp_path/'train.csv', index=False)
    test.to_csv(tmp_path/'test.csv', index=False)
    ds = SimpleNamespace(train_csv=str(tmp_path/'train.csv'), test_csv=str(tmp_path/'test.csv'),
                         smiles_cols=['S'], target_col='Y')
    # Intentionally no use_condition_features: older baseline CLIs omit it.
    args = SimpleNamespace(dataset='demo', random_split=True, random_split_seed=42,
                           random_split_frac=.7, pe_dim=0)
    cache = str(tmp_path/'cache')
    first = training.load_and_build_graphs(ds, args, cache)
    args.random_split_frac = .5
    second = training.load_and_build_graphs(ds, args, cache)
    assert len(first[0]) == 7
    assert len(second[0]) == 5
    train['Y'] += 100
    train.to_csv(tmp_path/'train.csv', index=False)
    third = training.load_and_build_graphs(ds, args, cache)
    assert not np.array_equal(second[1], third[1])
    assert len(list((tmp_path/'cache').glob('*.pkl'))) == 3

#!/usr/bin/env python3
"""Reaction GNN + eKNN residual transfer.

Train a GraphGPS backbone on reaction yield prediction, then evaluate
with ekNN TTT: find k nearest training reactions in embedding space,
temperature-weighted yield aggregation.

Usage:
  python scripts/train_gnn_eknn.py --dataset cpac_test_sub --epochs 5 --set train.device=cpu
  python scripts/train_gnn_eknn.py --dataset bhc_test1 --epochs 100
"""

from __future__ import annotations

import argparse, os, sys, time, copy, math, pickle
import itertools
import hashlib
import json
from pathlib import Path
from importlib.metadata import version

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_script_dir = str(_PROJECT_ROOT / "scripts")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import GroupShuffleSplit

from eknn_reaction.retrieval import temperature_weights
from eknn_reaction.config import ReactionConfig, override_from_dotlist
from eknn_reaction.data import get_dataset
from eknn_reaction.data.datasets import DATASETS, DatasetConfig
from eknn_reaction.data.dataset import Standardizer
from eknn_reaction.metrics import mae, r2
from eknn_reaction import utils
from eknn_reaction.models.graph_encoder import (
    build_reaction_graphs, ReactionGraphDataset, collate_graphs,
    ATOM_FEAT_DIM, BOND_FEAT_DIM,
)
from eknn_reaction.models.reaction_gnn import ReactionGNN

from .cli import parse_args, parse_component_identity_roles
from .inference import encode_reactions, retrieval_vectors
from .neighbors import knn_correct, prepare_knn_neighbours, aggregate_prepared_neighbours, multi_k_ensemble, knn_label_predict
from .splits import make_selection_split

LOG = utils.get_logger()


def release_provenance(ds, args):
    """Identify the corrected numerical implementation and input files."""
    files = {}
    for role, filename in (("train", ds.train_csv), ("test", ds.test_csv)):
        digest = hashlib.sha256()
        with open(filename, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        files[role] = {"path": str(filename), "sha256": digest.hexdigest()}
    code = {}
    root = Path(_script_dir).parent
    for relative in ("scripts/train_gnn_eknn.py", "eknn_reaction/retrieval.py",
                     "eknn_reaction/models/graph_encoder.py", "eknn_reaction/models/reaction_gnn.py",
                     "eknn_reaction/training/runner.py", "eknn_reaction/training/cli.py",
                     "eknn_reaction/training/inference.py", "eknn_reaction/training/neighbors.py",
                     "eknn_reaction/training/splits.py"):
        if (root / relative).is_file():
            code[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    return {"release": "0.3.0", "weighting": "stable_normalized_distance_exponential",
            "argv": sys.argv, "arguments": vars(args), "inputs": files, "code_sha256": code,
            "packages": {name: version(name) for name in
                         ("numpy", "pandas", "torch", "torch-geometric", "rdkit", "scikit-learn")}}

# Grid search space (same as ekNN v2)
K_GRID = [5, 10, 20, 50, 100]
T_GRID = [0.05, 0.1, 0.2, 0.5]


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────





# ──────────────────────────────────────────────────────────────────────
# Graph Encoding Helpers (for ekNN)
# ──────────────────────────────────────────────────────────────────────





def build_eknn_index(
    model,
    train_graphs,
    train_targets,
    device,
    std: Standardizer,
    condition_scale=0.0,
):
    """Build ekNN training index.

    Returns:
        h_train_norm: [N, d] L2-normalised training embeddings
        residual_train: [N] normalised residuals (y_norm - y_base)
        y_base_train: [N] base predictions (normalised)
    """
    y_norm = std.normalize(train_targets)
    h_train, y_base_norm = encode_reactions(model, train_graphs, device)
    residual_train = y_norm - y_base_norm
    h_train_norm = retrieval_vectors(
        h_train, train_graphs, condition_scale=condition_scale
    )
    return h_train_norm, residual_train, y_base_norm










# ──────────────────────────────────────────────────────────────────────
# ekNN Hyperparameter Selection and Fixed Test Evaluation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def select_eknn_hyperparams(model, train_graphs, val_graphs,
                            train_targets, val_targets, device, std,
                            condition_scales=(0.0,),
                            residual_scales=(1.0,),
                            base_alignment_scales=(0.0,)):
    """Select all eKNN retrieval and residual-transfer controls on validation."""
    LOG.info("Selection stage: building ekNN index (%d reactions)...", len(train_graphs))
    y_train_norm = std.normalize(train_targets)
    h_train, y_base_train = encode_reactions(model, train_graphs, device)
    residual_train = y_train_norm - y_base_train
    scales = tuple(float(scale) for scale in condition_scales)
    correction_scales = tuple(float(scale) for scale in residual_scales)
    alignment_scales = tuple(float(scale) for scale in base_alignment_scales)
    n_combos = (
        len(scales) * len(K_GRID) * len(T_GRID)
        * len(correction_scales) * len(alignment_scales)
    )

    LOG.info("Selection stage: grid search on validation set (%d reactions)...", len(val_graphs))
    h_val, y_val_base_norm = encode_reactions(model, val_graphs, device)

    best_val_r2 = float('-inf')
    best_combo = None
    val_grid = {}
    i = 0
    for condition_scale in scales:
        h_train_norm = retrieval_vectors(
            h_train, train_graphs, condition_scale=condition_scale
        )
        h_val_norm = retrieval_vectors(
            h_val, val_graphs, condition_scale=condition_scale
        )
        neighbour_idx, neighbour_distances = prepare_knn_neighbours(
            h_val_norm, h_train_norm, max(K_GRID)
        )
        for k, T in itertools.product(K_GRID, T_GRID):
            corrections = aggregate_prepared_neighbours(
                neighbour_idx,
                neighbour_distances,
                residual_train,
                k,
                T,
            )
            neighbour_base = aggregate_prepared_neighbours(
                neighbour_idx,
                neighbour_distances,
                y_base_train,
                k,
                T,
            )
            base_alignment = neighbour_base - y_val_base_norm
            for residual_scale in correction_scales:
                for base_alignment_scale in alignment_scales:
                    y_final_norm = (
                        y_val_base_norm
                        + residual_scale * corrections
                        + base_alignment_scale * base_alignment
                    )
                    y_final = std.denormalize(y_final_norm)
                    vr2 = float(r2(y_final, val_targets))
                    vmae = float(mae(y_final, val_targets))
                    key = (
                        f"C{str(condition_scale).replace('.', 'p')}_"
                        f"k{k}_T{str(T).replace('.', 'p')}_"
                        f"A{str(residual_scale).replace('.', 'p')}_"
                        f"B{str(base_alignment_scale).replace('.', 'p')}"
                    )
                    val_grid[key] = {
                        "condition_scale": condition_scale,
                        "k": k,
                        "T": T,
                        "residual_scale": residual_scale,
                        "base_alignment_scale": base_alignment_scale,
                        "r2": vr2,
                        "mae": vmae,
                    }
                    if vr2 > best_val_r2:
                        best_val_r2 = vr2
                        best_combo = (
                            condition_scale,
                            k,
                            T,
                            residual_scale,
                            base_alignment_scale,
                        )
                    i += 1
                    if i % 175 == 0 or i == n_combos:
                        LOG.info(
                            "  [val %d/%d] C=%.2f k=%d T=%.2f "
                            "A=%.2f B=%.2f: R²=%.4f",
                            i, n_combos, condition_scale, k, T,
                            residual_scale, base_alignment_scale, vr2,
                        )

    (best_condition_scale,
     best_k,
     best_T,
     best_residual_scale,
     best_base_alignment_scale) = best_combo
    LOG.info(
        "Val-selected eKNN: C=%.2f k=%d T=%.2f A=%.2f B=%.2f "
        "val-R²=%.4f",
        best_condition_scale, best_k, best_T, best_residual_scale,
        best_base_alignment_scale, best_val_r2,
    )

    best_h_train_norm = retrieval_vectors(
        h_train, train_graphs, condition_scale=best_condition_scale
    )
    best_h_val_norm = retrieval_vectors(
        h_val, val_graphs, condition_scale=best_condition_scale
    )
    best_residual_correction = knn_correct(
        best_h_val_norm,
        best_h_train_norm,
        residual_train,
        best_k,
        best_T,
    )
    best_neighbour_base = knn_correct(
        best_h_val_norm,
        best_h_train_norm,
        y_base_train,
        best_k,
        best_T,
    )
    best_prediction_norm = (
        y_val_base_norm
        + best_residual_scale * best_residual_correction
        + best_base_alignment_scale
        * (best_neighbour_base - y_val_base_norm)
    )

    return {
        "condition_scale": best_condition_scale,
        "k": best_k,
        "T": best_T,
        "residual_scale": best_residual_scale,
        "base_alignment_scale": best_base_alignment_scale,
        "val_r2": best_val_r2,
        "_predictions": {
            "target": np.asarray(val_targets, dtype=float),
            "static": np.asarray(
                std.denormalize(y_val_base_norm), dtype=float
            ),
            "eknn": np.asarray(
                std.denormalize(best_prediction_norm), dtype=float
            ),
        },
    }, val_grid


@torch.no_grad()
def evaluate_eknn_fixed(model, train_graphs, test_graphs, train_targets,
                        test_targets, device, std, selected):
    """Evaluate fixed validation-selected k/T once on the independent test set."""
    LOG.info("Final stage: building ekNN index from full training set (%d reactions)...", len(train_graphs))
    condition_scale = float(selected.get("condition_scale", 0.0))
    residual_scale = float(selected.get("residual_scale", 1.0))
    base_alignment_scale = float(
        selected.get("base_alignment_scale", 0.0)
    )
    h_train_norm, residual_train, y_base_train = build_eknn_index(
        model, train_graphs, train_targets, device, std, condition_scale)

    best_k, best_T = int(selected["k"]), float(selected["T"])
    LOG.info(
        "Final stage: test evaluation at fixed C=%.2f k=%d T=%.2f "
        "A=%.2f B=%.2f...",
        condition_scale, best_k, best_T, residual_scale,
        base_alignment_scale,
    )
    h_test, y_base_norm = encode_reactions(model, test_graphs, device)
    y_base = std.denormalize(y_base_norm)
    h_test_norm = retrieval_vectors(
        h_test, test_graphs, condition_scale=condition_scale
    )

    # Static metrics
    static_mae = float(mae(y_base, test_targets))
    static_r2 = float(r2(y_base, test_targets))
    LOG.info("Static: R²=%.4f  MAE=%.4f", static_r2, static_mae)

    # Fixed val-selected k/T on test
    corrections_best = knn_correct(h_test_norm, h_train_norm, residual_train, best_k, best_T)
    neighbour_base_best = knn_correct(
        h_test_norm, h_train_norm, y_base_train, best_k, best_T
    )
    base_alignment_best = neighbour_base_best - y_base_norm
    y_best_norm = (
        y_base_norm
        + residual_scale * corrections_best
        + base_alignment_scale * base_alignment_best
    )
    y_best = std.denormalize(y_best_norm)
    best_test_r2 = float(r2(y_best, test_targets))
    best_test_mae = float(mae(y_best, test_targets))
    best_delta = best_test_r2 - static_r2
    LOG.info(
        "Val-selected (C=%.2f,k=%d,T=%.2f,A=%.2f,B=%.2f) "
        "on test: R²=%.4f ΔR²=%+.4f",
        condition_scale, best_k, best_T, residual_scale,
        base_alignment_scale,
        best_test_r2, best_delta,
    )

    # Multi-k ensemble on test
    LOG.info("Fixed multi-k ensemble (k=5,10,20,50, T=0.1)...")
    ens_corrections = multi_k_ensemble(h_test_norm, h_train_norm, residual_train,
                                       k_list=(5, 10, 20, 50), T=0.1)
    ens_neighbour_base = multi_k_ensemble(
        h_test_norm,
        h_train_norm,
        y_base_train,
        k_list=(5, 10, 20, 50),
        T=0.1,
    )
    y_ens_norm = (
        y_base_norm
        + residual_scale * ens_corrections
        + base_alignment_scale * (ens_neighbour_base - y_base_norm)
    )
    y_ens = std.denormalize(y_ens_norm)
    ens_r2 = float(r2(y_ens, test_targets))
    ens_mae = float(mae(y_ens, test_targets))
    ens_delta = ens_r2 - static_r2

    # Optional reference only: do not use these rows for model or k/T selection.
    LOG.info("Reference: full grid on test (not used for selection)...")
    grid_results = {}
    for k, T in itertools.product(K_GRID, T_GRID):
        corrections = knn_correct(h_test_norm, h_train_norm, residual_train, k, T)
        neighbour_base = knn_correct(
            h_test_norm, h_train_norm, y_base_train, k, T
        )
        y_final_norm = (
            y_base_norm
            + residual_scale * corrections
            + base_alignment_scale * (neighbour_base - y_base_norm)
        )
        y_final = std.denormalize(y_final_norm)
        kr2 = float(r2(y_final, test_targets))
        kmae = float(mae(y_final, test_targets))
        key = f"k{k}_T{str(T).replace('.', 'p')}"
        grid_results[key] = {"k": k, "T": T, "r2": kr2, "mae": kmae, "delta_r2": kr2 - static_r2}

    return {
        "static_mae": static_mae, "static_r2": static_r2,
        "best": {"condition_scale": condition_scale, "k": best_k, "T": best_T,
                 "residual_scale": residual_scale,
                 "base_alignment_scale": base_alignment_scale,
                 "r2": best_test_r2, "mae": best_test_mae, "delta_r2": best_delta,
                 "note": "condition scale/k/T/residual scale/base alignment selected on held-out validation with a selection model, then fixed for final full-train model"},
        "ensemble": {"r2": ens_r2, "mae": ens_mae, "delta_r2": ens_delta,
                     "k_list": [5, 10, 20, 50], "T": 0.1,
                     "residual_scale": residual_scale,
                     "base_alignment_scale": base_alignment_scale},
        "grid_search": grid_results,
        "grid_search_note": "Full grid on test set — for sensitivity analysis only, NOT used for k/T selection",
        "n_train": int(len(train_graphs)),
        "n_test": int(len(test_graphs)),
        "_predictions": {
            "target": np.asarray(test_targets, dtype=float),
            "static": np.asarray(y_base, dtype=float),
            "eknn": np.asarray(y_best, dtype=float),
        },
    }


@torch.no_grad()
def predict_eknn_fixed_unscored(
    model,
    train_graphs,
    query_graphs,
    train_targets,
    device,
    std,
    selected,
):
    """Predict with fixed eKNN controls without accepting query targets.

    This is deliberately separate from ``evaluate_eknn_fixed`` so a locked
    ensemble can materialise and hash outer predictions before the one-time
    scoring step.  The returned arrays contain predictions only.
    """
    condition_scale = float(selected.get("condition_scale", 0.0))
    residual_scale = float(selected.get("residual_scale", 1.0))
    base_alignment_scale = float(
        selected.get("base_alignment_scale", 0.0)
    )
    best_k = int(selected["k"])
    best_T = float(selected["T"])

    LOG.info(
        "Unscored outer prediction: index=%d query=%d "
        "C=%.2f k=%d T=%.2f A=%.2f B=%.2f",
        len(train_graphs),
        len(query_graphs),
        condition_scale,
        best_k,
        best_T,
        residual_scale,
        base_alignment_scale,
    )
    h_train_norm, residual_train, y_base_train = build_eknn_index(
        model,
        train_graphs,
        train_targets,
        device,
        std,
        condition_scale,
    )
    h_query, y_base_query_norm = encode_reactions(
        model, query_graphs, device
    )
    h_query_norm = retrieval_vectors(
        h_query,
        query_graphs,
        condition_scale=condition_scale,
    )
    residual_correction = knn_correct(
        h_query_norm,
        h_train_norm,
        residual_train,
        best_k,
        best_T,
    )
    neighbour_base = knn_correct(
        h_query_norm,
        h_train_norm,
        y_base_train,
        best_k,
        best_T,
    )
    prediction_norm = (
        y_base_query_norm
        + residual_scale * residual_correction
        + base_alignment_scale * (
            neighbour_base - y_base_query_norm
        )
    )
    return {
        "static": np.asarray(
            std.denormalize(y_base_query_norm), dtype=float
        ),
        "eknn": np.asarray(
            std.denormalize(prediction_norm), dtype=float
        ),
    }


# ──────────────────────────────────────────────────────────────────────
# Label kNN Baseline (direct label retrieval, not residual)
# ──────────────────────────────────────────────────────────────────────



@torch.no_grad()
def select_label_knn_hyperparams(model, train_graphs, val_graphs,
                                 train_targets, val_targets, device, std):
    """Select label-kNN k/T on held-out validation."""
    y_train_norm = std.normalize(train_targets)

    h_train, _ = encode_reactions(model, train_graphs, device)
    h_train_norm = h_train / (np.linalg.norm(h_train, axis=1, keepdims=True) + 1e-8)

    h_val, _ = encode_reactions(model, val_graphs, device)
    h_val_norm = h_val / (np.linalg.norm(h_val, axis=1, keepdims=True) + 1e-8)

    # Grid search on val
    best_val_r2 = float('-inf')
    best_combo = None
    for i, (k, T) in enumerate(itertools.product(K_GRID, T_GRID)):
        preds = knn_label_predict(h_val_norm, h_train_norm, y_train_norm, k, T)
        y_final = std.denormalize(preds)
        vr2 = float(r2(y_final, val_targets))
        if vr2 > best_val_r2:
            best_val_r2 = vr2
            best_combo = (k, T)

    best_k, best_T = best_combo
    LOG.info("Label-kNN val-selected: k=%d T=%.2f  val-R²=%.4f", best_k, best_T, best_val_r2)

    return {"k": best_k, "T": best_T, "val_r2": best_val_r2}


@torch.no_grad()
def evaluate_label_knn_fixed(model, train_graphs, test_graphs, train_targets,
                             test_targets, device, std, selected):
    """Evaluate label-kNN with fixed validation-selected k/T."""
    y_train_norm = std.normalize(train_targets)

    h_train, _ = encode_reactions(model, train_graphs, device)
    h_train_norm = h_train / (np.linalg.norm(h_train, axis=1, keepdims=True) + 1e-8)

    h_test, _ = encode_reactions(model, test_graphs, device)
    h_test_norm = h_test / (np.linalg.norm(h_test, axis=1, keepdims=True) + 1e-8)
    best_k, best_T = int(selected["k"]), float(selected["T"])
    preds_test = knn_label_predict(h_test_norm, h_train_norm, y_train_norm, best_k, best_T)
    y_test_pred = std.denormalize(preds_test)
    test_r2 = float(r2(y_test_pred, test_targets))
    test_mae_val = float(mae(y_test_pred, test_targets))

    LOG.info("Label-kNN test: R²=%.4f  MAE=%.4f", test_r2, test_mae_val)

    return {
        "k": best_k, "T": best_T,
        "val_r2": float(selected["val_r2"]),
        "test_r2": test_r2, "test_mae": test_mae_val,
        "note": "k/T selected on held-out validation with a selection model, then fixed for final full-train model",
    }


# ──────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────

def _numeric_condition_array(df, column, *, log1p=False):
    """Read one condition column as float, optionally using log1p values.

    Invalid/missing entries remain NaN here.  They are imputed from the
    training fold only by ``prepare_condition_features`` below.
    """
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
    values[~np.isfinite(values)] = np.nan
    if log1p:
        # Amounts, time and volume are non-negative by definition.  Treat an
        # unexpected negative entry as missing rather than silently applying a
        # transformation with a different physical interpretation.
        invalid = np.isfinite(values) & (values < 0)
        values[invalid] = np.nan
        values = np.log1p(values)
    return values


def prepare_condition_features(df_train, df_test, ds, *, include_categorical=True):
    """Fit fold-safe condition preprocessing and return graph-level features.

    Each raw numeric condition becomes a standardised value plus an explicit
    missingness indicator.  Medians, means and scales are fit *only* on
    ``df_train`` so neither held-out validation rows nor outer-test rows can
    influence the representation.
    """
    numeric_condition_cols = list(ds.condition_cols)
    categorical_condition_cols = (
        list(getattr(ds, "condition_categorical_cols", []))
        if include_categorical else []
    )
    if not numeric_condition_cols and not categorical_condition_cols:
        raise ValueError(
            f"Dataset '{ds.name}' has no registered condition columns; "
            "cannot enable --use-condition-features"
        )
    all_condition_cols = numeric_condition_cols + categorical_condition_cols
    missing_train = [col for col in all_condition_cols if col not in df_train.columns]
    missing_test = [col for col in all_condition_cols if col not in df_test.columns]
    if missing_train or missing_test:
        raise KeyError(
            "Condition columns absent from split CSVs; regenerate the showcase "
            f"data if needed. train missing={missing_train}, test missing={missing_test}"
        )

    log1p_cols = set(ds.condition_log1p_cols)
    train_values, test_values = [], []
    raw_stats = []
    for column in numeric_condition_cols:
        train_raw = _numeric_condition_array(
            df_train, column, log1p=column in log1p_cols)
        test_raw = _numeric_condition_array(
            df_test, column, log1p=column in log1p_cols)
        train_missing = ~np.isfinite(train_raw)
        test_missing = ~np.isfinite(test_raw)

        if train_missing.all():
            # A column absent in every training row carries no numerical
            # variation; keep a deterministic zero-imputed value while its
            # missingness bit still records the state.
            median = 0.0
        else:
            median = float(np.nanmedian(train_raw))
        train_imputed = np.where(train_missing, median, train_raw)
        test_imputed = np.where(test_missing, median, test_raw)
        mean = float(np.mean(train_imputed))
        std = float(np.std(train_imputed))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0

        train_values.append(((train_imputed - mean) / std).astype(np.float32))
        test_values.append(((test_imputed - mean) / std).astype(np.float32))
        raw_stats.append({
            "column": column,
            "log1p": bool(column in log1p_cols),
            "median_train": median,
            "mean_train": mean,
            "std_train": std,
            "missing_train": int(train_missing.sum()),
            "missing_test": int(test_missing.sum()),
        })

    if numeric_condition_cols:
        train_scaled = np.column_stack(train_values).astype(np.float32)
        test_scaled = np.column_stack(test_values).astype(np.float32)
        train_missing_flags = np.column_stack([
            ~np.isfinite(_numeric_condition_array(df_train, col, log1p=col in log1p_cols))
            for col in numeric_condition_cols
        ]).astype(np.float32)
        test_missing_flags = np.column_stack([
            ~np.isfinite(_numeric_condition_array(df_test, col, log1p=col in log1p_cols))
            for col in numeric_condition_cols
        ]).astype(np.float32)
    else:
        train_scaled = np.zeros((len(df_train), 0), dtype=np.float32)
        test_scaled = np.zeros((len(df_test), 0), dtype=np.float32)
        train_missing_flags = np.zeros((len(df_train), 0), dtype=np.float32)
        test_missing_flags = np.zeros((len(df_test), 0), dtype=np.float32)

    def categorical_values(frame, column):
        values = []
        for value in frame[column].tolist():
            if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null"}:
                values.append("__MISSING__")
            else:
                values.append(str(value).strip())
        return np.asarray(values, dtype=object)

    categorical_train_blocks, categorical_test_blocks = [], []
    categorical_feature_columns, categorical_stats = [], []
    for column in categorical_condition_cols:
        train_raw = categorical_values(df_train, column)
        test_raw = categorical_values(df_test, column)
        vocabulary = sorted(set(train_raw.tolist()))
        other_token = "__OTHER__"
        if other_token in vocabulary:
            other_token = "__OTHER_UNSEEN__"
        categories = [*vocabulary, other_token]
        category_to_index = {category: index for index, category in enumerate(categories)}
        train_index = np.asarray([
            category_to_index.get(value, category_to_index[other_token])
            for value in train_raw
        ])
        test_index = np.asarray([
            category_to_index.get(value, category_to_index[other_token])
            for value in test_raw
        ])
        train_block = np.zeros((len(train_raw), len(categories)), dtype=np.float32)
        test_block = np.zeros((len(test_raw), len(categories)), dtype=np.float32)
        train_block[np.arange(len(train_raw)), train_index] = 1.0
        test_block[np.arange(len(test_raw)), test_index] = 1.0
        categorical_train_blocks.append(train_block)
        categorical_test_blocks.append(test_block)
        categorical_feature_columns.extend([
            f"{column}=={category}" for category in categories
        ])
        categorical_stats.append({
            "column": column,
            "vocabulary_train": vocabulary,
            "other_token": other_token,
            "n_categories_train": int(len(vocabulary)),
            "n_unseen_or_missing_test": int((test_index == category_to_index[other_token]).sum()),
        })

    numeric_train_features = np.concatenate([train_scaled, train_missing_flags], axis=1)
    numeric_test_features = np.concatenate([test_scaled, test_missing_flags], axis=1)
    if categorical_train_blocks:
        train_features = np.concatenate(
            [numeric_train_features, *categorical_train_blocks], axis=1
        )
        test_features = np.concatenate(
            [numeric_test_features, *categorical_test_blocks], axis=1
        )
    else:
        train_features = numeric_train_features
        test_features = numeric_test_features
    info = {
        "raw_condition_columns": numeric_condition_cols,
        "categorical_condition_columns": categorical_condition_cols,
        "log1p_columns": sorted(log1p_cols),
        "feature_columns": (
            [f"{column}_z" for column in numeric_condition_cols]
            + [f"{column}_missing" for column in numeric_condition_cols]
            + categorical_feature_columns
        ),
        "n_raw_features": int(len(all_condition_cols)),
        "n_numeric_raw_features": int(len(numeric_condition_cols)),
        "n_categorical_raw_features": int(len(categorical_condition_cols)),
        "n_model_features": int(train_features.shape[1]),
        "fit_statistics": raw_stats,
        "categorical_fit_statistics": categorical_stats,
        "fit_scope": "provided fitting partition only",
    }
    return train_features, test_features, info


def prepare_categorical_condition_ids(df_train, df_test, ds):
    """Fit categorical condition vocabularies on ``df_train`` only.

    Unlike the legacy one-hot option, this returns one integer ID per raw
    categorical field so the condition-token encoder can learn a compact
    field-specific embedding.  The final ID in each vocabulary is reserved
    for missing or unseen values in held-out rows.
    """
    columns = list(getattr(ds, "condition_categorical_cols", []))
    if not columns:
        empty_train = np.zeros((len(df_train), 0), dtype=np.int64)
        empty_test = np.zeros((len(df_test), 0), dtype=np.int64)
        return empty_train, empty_test, [], {
            "columns": [],
            "cardinalities": [],
            "fit_scope": "provided fitting partition only",
        }
    missing_train = [column for column in columns if column not in df_train.columns]
    missing_test = [column for column in columns if column not in df_test.columns]
    if missing_train or missing_test:
        raise KeyError(
            "Categorical condition columns absent from split CSVs: "
            f"train missing={missing_train}, test missing={missing_test}"
        )

    def canonical_values(frame, column):
        values = []
        for value in frame[column].tolist():
            if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null"}:
                values.append("__MISSING__")
            else:
                values.append(str(value).strip())
        return np.asarray(values, dtype=object)

    train_blocks, test_blocks = [], []
    cardinalities, field_info = [], []
    for column in columns:
        train_raw = canonical_values(df_train, column)
        test_raw = canonical_values(df_test, column)
        vocabulary = sorted(set(train_raw.tolist()))
        oov_token = "__OTHER__"
        if oov_token in vocabulary:
            oov_token = "__OTHER_UNSEEN__"
        value_to_id = {value: index for index, value in enumerate(vocabulary)}
        oov_id = len(vocabulary)
        train_ids = np.asarray([value_to_id.get(value, oov_id) for value in train_raw], dtype=np.int64)
        test_ids = np.asarray([value_to_id.get(value, oov_id) for value in test_raw], dtype=np.int64)
        train_blocks.append(train_ids)
        test_blocks.append(test_ids)
        cardinalities.append(len(vocabulary) + 1)
        field_info.append({
            "column": column,
            "vocabulary_train": vocabulary,
            "oov_token": oov_token,
            "cardinality": int(len(vocabulary) + 1),
            "n_oov_or_unseen_train": int((train_ids == oov_id).sum()),
            "n_oov_or_unseen_test": int((test_ids == oov_id).sum()),
        })

    return (
        np.column_stack(train_blocks).astype(np.int64),
        np.column_stack(test_blocks).astype(np.int64),
        cardinalities,
        {
            "columns": columns,
            "cardinalities": [int(value) for value in cardinalities],
            "fields": field_info,
            "fit_scope": "provided fitting partition only",
        },
    )


def load_and_build_graphs(ds, args, cache_dir=None):
    """Load reaction data and build molecular graphs.

    Returns:
        train_graphs, train_targets
        test_graphs, test_targets
        std: Standardizer
    """
    df_train = pd.read_csv(ds.train_csv)
    df_test = pd.read_csv(ds.test_csv)

    # ── Random split mode: merge train+test, shuffle, re-split ──
    if args.random_split:
        df_all = pd.concat([df_train, df_test], ignore_index=True)
        n_total = len(df_all)
        rng = np.random.RandomState(args.random_split_seed)
        perm = rng.permutation(n_total)
        n_train = int(n_total * args.random_split_frac)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        df_train = df_all.iloc[train_idx].reset_index(drop=True)
        df_test = df_all.iloc[test_idx].reset_index(drop=True)
        LOG.info("Random split: total=%d train=%d test=%d (frac=%.1f seed=%d)",
                 n_total, len(df_train), len(df_test),
                 args.random_split_frac, args.random_split_seed)

    LOG.info("Dataset: %s | train=%d test=%d smiles_cols=%s target=%s",
             args.dataset, len(df_train), len(df_test),
             ds.smiles_cols, ds.target_col)

    # Bind cached graphs AND targets to data content, schema, split, code and runtime.
    # Conditions are still attached and fitted after the inner split.
    if cache_dir:
        key = {
            "schema_version": 7,
            "smiles_cols": list(ds.smiles_cols), "target_col": ds.target_col,
            "pe_dim": args.pe_dim, "random_split": args.random_split,
            "random_seed": getattr(args, "random_split_seed", None),
            "random_fraction": getattr(args, "random_split_frac", None),
            "graph_encoder": hashlib.sha256(
                (Path(_script_dir).parent / "eknn_reaction/models/graph_encoder.py").read_bytes()
            ).hexdigest(),
            "runtime": {name: version(name) for name in ("torch", "torch-geometric", "rdkit")},
        }
        for role, filename in (("train", ds.train_csv), ("test", ds.test_csv)):
            digest = hashlib.sha256()
            with open(filename, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            key[role + "_sha256"] = digest.hexdigest()
        cache_id = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        cache_file = os.path.join(cache_dir, f"graphs_v7_{cache_id}.pkl")
        if os.path.exists(cache_file):
            LOG.info("Loading cached graphs from %s", cache_file)
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
            return (cache['train_graphs'], cache['train_targets'],
                    cache['test_graphs'], cache['test_targets'],
                    Standardizer(cache['yield_mean'], cache['yield_std']))

    # Build graphs
    LOG.info("Building reaction graphs (this may take a moment)...")
    # Condition scaling is deliberately deferred until after the inner split.
    # This lets the selection model fit its imputation/scaling statistics on
    # selection-train rows only, while the final model refits them on the full
    # outer training split.  Graph topology itself is condition-independent.
    condition_info = None
    if getattr(args, "use_condition_features", False):
        n_numeric_conditions = len(ds.condition_cols)
        n_categorical_conditions = (
            len(getattr(ds, "condition_categorical_cols", []))
            if getattr(args, "include_categorical_conditions", False) else 0
        )
        if n_numeric_conditions + n_categorical_conditions == 0:
            raise ValueError(
                f"Dataset '{ds.name}' has no registered condition columns; "
                "cannot enable --use-condition-features"
            )
        LOG.info(
            "Condition fusion requested: %d numeric + %d categorical raw columns; fold-safe scaling "
            "will be fit after the validation split",
            n_numeric_conditions, n_categorical_conditions,
        )
    train_graphs, train_targets = build_reaction_graphs(
        df_train, ds.smiles_cols, ds.target_col,
        add_laplacian_pe=True, pe_k=args.pe_dim, verbose=True)
    test_graphs, test_targets = build_reaction_graphs(
        df_test, ds.smiles_cols, ds.target_col,
        add_laplacian_pe=True, pe_k=args.pe_dim, verbose=True)

    # Standardizer
    std = Standardizer(float(np.nanmean(train_targets)), float(np.nanstd(train_targets)))

    # Cache
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'train_graphs': train_graphs,
                'train_targets': train_targets,
                'test_graphs': test_graphs,
                'test_targets': test_targets,
                'yield_mean': std.mean,
                'yield_std': std.std,
                'condition_info': condition_info,
            }, f)
        LOG.info("Cached graphs to %s", cache_file)

    return train_graphs, train_targets, test_graphs, test_targets, std


def filter_valid(graphs, targets):
    """Remove None entries (failed graph builds). Returns (valid_graphs, valid_targets)."""
    valid_idx = [i for i, g in enumerate(graphs) if g is not None]
    return [graphs[i] for i in valid_idx], targets[valid_idx]


def graph_rows_to_frame(graphs, source_frame):
    """Recover source rows in the same order as a valid-graph list."""
    source_indices = [int(graph.reaction_idx) for graph in graphs]
    if any(index < 0 or index >= len(source_frame) for index in source_indices):
        raise IndexError("A graph reaction_idx is outside its source split CSV")
    return source_frame.iloc[source_indices].reset_index(drop=True)


def attach_condition_features(
    graphs,
    features,
    categorical_features=None,
    component_identity_features=None,
):
    """Clone graphs and attach fold-fitted reaction-context arrays.

    ``component_identity_features`` stores one integer ID for each molecular
    role.  Its vocabulary must be fitted on the corresponding training fold;
    the helper below maps any held-out molecular identity to that role's OOV
    token.  This keeps the identity-aware GNN valid under group-disjoint
    validation and external evaluation.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(graphs) != features.shape[0]:
        raise ValueError(
            "Condition feature alignment error: expected [len(graphs), n_features], got "
            f"{features.shape} for {len(graphs)} graphs"
        )
    if categorical_features is not None:
        categorical_features = np.asarray(categorical_features, dtype=np.int64)
        if (categorical_features.ndim != 2
                or len(graphs) != categorical_features.shape[0]):
            raise ValueError(
                "Categorical condition alignment error: expected [len(graphs), n_features], got "
                f"{categorical_features.shape} for {len(graphs)} graphs"
            )
    if component_identity_features is not None:
        component_identity_features = np.asarray(
            component_identity_features, dtype=np.int64
        )
        if (component_identity_features.ndim != 2
                or len(graphs) != component_identity_features.shape[0]):
            raise ValueError(
                "Component identity alignment error: expected [len(graphs), n_features], got "
                f"{component_identity_features.shape} for {len(graphs)} graphs"
            )
    conditioned_graphs = []
    for index, (graph, feature_row) in enumerate(zip(graphs, features)):
        conditioned = graph.clone()
        conditioned.condition_features = torch.from_numpy(
            feature_row.reshape(1, -1)
        )
        if categorical_features is not None:
            conditioned.condition_categorical_features = torch.from_numpy(
                categorical_features[index:index + 1]
            )
        if component_identity_features is not None:
            conditioned.component_identity_features = torch.from_numpy(
                component_identity_features[index:index + 1]
            )
        conditioned_graphs.append(conditioned)
    return conditioned_graphs


def prepare_component_identity_ids(df_train, df_test, ds, columns=None):
    """Fit per-role molecular identity vocabularies on ``df_train`` only.

    The raw component SMILES are valid prediction-time inputs, but their
    integer encodings cannot be built globally: doing so would disclose the
    identities occurring only in validation/test.  For every role, the final
    vocabulary entry is therefore reserved for any held-out OOV value.
    """
    columns = list(ds.smiles_cols if columns is None else columns)
    unknown_columns = [column for column in columns if column not in ds.smiles_cols]
    if unknown_columns:
        raise ValueError(
            "Component identity columns must be registered molecular roles; got "
            f"{unknown_columns}"
        )
    if not columns:
        empty_train = np.zeros((len(df_train), 0), dtype=np.int64)
        empty_test = np.zeros((len(df_test), 0), dtype=np.int64)
        return empty_train, empty_test, [], {
            "columns": [],
            "cardinalities": [],
            "fit_scope": "provided fitting partition only",
        }
    missing_train = [column for column in columns if column not in df_train.columns]
    missing_test = [column for column in columns if column not in df_test.columns]
    if missing_train or missing_test:
        raise KeyError(
            "Component identity columns absent from split CSVs: "
            f"train missing={missing_train}, test missing={missing_test}"
        )

    def canonical_values(frame, column):
        values = []
        for value in frame[column].tolist():
            if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null"}:
                values.append("__MISSING__")
            else:
                values.append(str(value).strip())
        return np.asarray(values, dtype=object)

    train_blocks, test_blocks = [], []
    cardinalities, field_info = [], []
    for column in columns:
        train_raw = canonical_values(df_train, column)
        test_raw = canonical_values(df_test, column)
        vocabulary = sorted(set(train_raw.tolist()))
        oov_token = "__OTHER__"
        if oov_token in vocabulary:
            oov_token = "__OTHER_UNSEEN__"
        value_to_id = {value: index for index, value in enumerate(vocabulary)}
        oov_id = len(vocabulary)
        train_ids = np.asarray(
            [value_to_id.get(value, oov_id) for value in train_raw], dtype=np.int64
        )
        test_ids = np.asarray(
            [value_to_id.get(value, oov_id) for value in test_raw], dtype=np.int64
        )
        train_blocks.append(train_ids)
        test_blocks.append(test_ids)
        cardinalities.append(len(vocabulary) + 1)
        field_info.append({
            "column": column,
            "vocabulary_train": vocabulary,
            "oov_token": oov_token,
            "cardinality": int(len(vocabulary) + 1),
            "n_oov_or_unseen_train": int((train_ids == oov_id).sum()),
            "n_oov_or_unseen_test": int((test_ids == oov_id).sum()),
        })

    return (
        np.column_stack(train_blocks).astype(np.int64),
        np.column_stack(test_blocks).astype(np.int64),
        cardinalities,
        {
            "columns": columns,
            "cardinalities": [int(value) for value in cardinalities],
            "fields": field_info,
            "fit_scope": "provided fitting partition only",
        },
    )




@torch.no_grad()
def evaluate_static(model, graphs, targets, device, std):
    """Evaluate the graph-model prediction without eKNN correction."""
    _, y_base_norm = encode_reactions(model, graphs, device)
    y_base = std.denormalize(y_base_norm)
    return {
        "r2": float(r2(y_base, targets)),
        "mae": float(mae(y_base, targets)),
    }


# ──────────────────────────────────────────────────────────────────────
# Model Construction and Training
# ──────────────────────────────────────────────────────────────────────

def build_model(
    args,
    condition_dim=0,
    condition_categorical_cardinalities=(),
    component_identity_cardinalities=(),
):
    """Build the reaction GNN with the CLI-selected backbone and options."""
    flag = lambda name: bool(getattr(args, name, False))
    return ReactionGNN(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        pe_dim=args.pe_dim,
        condition_dim=condition_dim,
        condition_fusion_mode=args.condition_fusion_mode,
        condition_node_film=args.condition_node_film,
        condition_categorical_cardinalities=condition_categorical_cardinalities,
        condition_token_layers=args.condition_token_layers,
        component_identity_cardinalities=component_identity_cardinalities,
        component_identity_dim=getattr(args, "component_identity_dim", 32),
        backbone=args.backbone,
        use_global_attn=not any([flag("mpnn"), flag("gat"), flag("graph_transformer"),
                                 flag("edge_transformer"), flag("graphormer"),
                                 flag("deep_gps"), flag("light_gps")]) and args.backbone == "gps",
        use_gat=flag("gat"),
        use_graph_transformer=flag("graph_transformer"),
        use_edge_transformer=flag("edge_transformer"),
        use_graphormer=flag("graphormer"),
        use_deep_gps=flag("deep_gps"),
        use_light_gps=flag("light_gps"),
        readout="attn" if flag("attn_pool") else args.readout,
        use_jk=flag("jk"),
        use_virtual_node=flag("virtual_node"),
        use_component_readout=flag("component_readout"),
        use_rolewise_readout=flag("rolewise_readout"),
        use_role_interaction_readout=flag("role_interaction_readout"),
        role_interaction_layers=args.role_interaction_layers,
        use_role_concat_readout=flag("role_concat_readout"),
        use_role_factorized_readout=flag("role_factorized_readout"),
        use_role_factorized_retrieval=flag(
            "role_factorized_retrieval"
        ),
        retrieval_dim=args.retrieval_dim,
    )


def continuous_metric_loss(
    embeddings,
    targets,
    *,
    label_temperature,
    embedding_temperature,
):
    """Match embedding neighbours to a soft distribution of nearby labels."""
    if embeddings.shape[0] < 2:
        return embeddings.sum() * 0.0
    if label_temperature <= 0 or embedding_temperature <= 0:
        raise ValueError("Metric-learning temperatures must be positive")

    z = F.normalize(embeddings, p=2, dim=-1)
    logits = (z @ z.transpose(0, 1)) / embedding_temperature
    label_distance = torch.abs(targets[:, None] - targets[None, :])
    target_logits = -label_distance / label_temperature
    diagonal = torch.eye(
        embeddings.shape[0], dtype=torch.bool, device=embeddings.device
    )
    logits = logits.masked_fill(diagonal, torch.finfo(logits.dtype).min)
    target_logits = target_logits.masked_fill(
        diagonal, torch.finfo(target_logits.dtype).min
    )
    target_probability = torch.softmax(target_logits, dim=1)
    log_probability = torch.log_softmax(logits, dim=1)
    return -(target_probability * log_probability).sum(dim=1).mean()


def train_gnn_model(
    model,
    train_graphs,
    train_targets,
    std,
    device,
    args,
    stage_name,
    validation_graphs=None,
    validation_targets=None,
):
    """Train one GNN stage with the existing full-data SWA recipe."""
    train_dataset = ReactionGraphDataset(train_graphs, train_targets)
    train_loader = PyGDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)

    n_params = model.get_num_params()
    LOG.info("[%s] Model: %d params | train=%d", stage_name, n_params, len(train_dataset))
    model.to(device)

    if args.optimizer == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay)
        LOG.info("[%s] Optimizer: SGD (lr=%.1e momentum=%.1f wd=%.1e)",
                 stage_name, args.lr, args.momentum, args.weight_decay)
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.warmup_epochs > 0:
        def lr_lambda(ep):
            if ep < args.warmup_epochs:
                return (ep + 1) / args.warmup_epochs
            progress = (ep - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
            return 0.01 + 0.99 * (1 + np.cos(np.pi * progress)) / 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    def norm_y(y_raw):
        return (y_raw - std.mean) / (std.std + 1e-8)

    best_train_loss, best_state = float('inf'), None
    use_validation_checkpoint = validation_graphs is not None
    if use_validation_checkpoint and validation_targets is None:
        raise ValueError("validation_targets are required with validation_graphs")
    best_val_r2 = float("-inf")
    best_val_state = None
    best_val_epoch = 0
    final_epoch = 0
    t0 = time.time()

    swa_state = None
    n_swa = 0
    swa_start = args.swa_start if args.swa_start > 0 else max(0, args.epochs - 20)
    LOG.info("[%s] SWA: start=%d (last %d epochs averaged)",
             stage_name, swa_start, args.epochs - swa_start)

    for ep in range(args.epochs):
        model.train()
        train_loss, n_batches = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            y_raw = batch.y.squeeze(-1)
            y_norm = norm_y(y_raw)
            opt.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out["yield"].squeeze(-1), y_norm)
            if args.metric_loss_weight > 0:
                retrieval_loss = continuous_metric_loss(
                    out["embedding"],
                    y_norm,
                    label_temperature=args.metric_label_temperature,
                    embedding_temperature=args.metric_embedding_temperature,
                )
                loss = loss + args.metric_loss_weight * retrieval_loss
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        final_epoch = ep + 1

        avg_train_loss = train_loss / max(n_batches, 1)
        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            best_state = copy.deepcopy(model.state_dict())

        if ep >= swa_start:
            if swa_state is None:
                swa_state = copy.deepcopy(model.state_dict())
                n_swa = 1
            else:
                for k in swa_state:
                    swa_state[k] = (n_swa * swa_state[k] + model.state_dict()[k]) / (n_swa + 1)
                n_swa += 1

        if (ep % 10) == 0 or ep == args.epochs - 1:
            LOG.info("[%s] ep %d: train=%.4f best_train=%.4f%s",
                     stage_name, ep, avg_train_loss, best_train_loss,
                     " [SWA]" if ep >= swa_start else "")

        should_check_val = (
            use_validation_checkpoint
            and final_epoch >= args.min_epochs
            and (
                final_epoch % max(args.val_check_interval, 1) == 0
                or ep == args.epochs - 1
            )
        )
        if should_check_val:
            val_metrics = evaluate_static(
                model, validation_graphs, validation_targets, device, std
            )
            LOG.info(
                "[%s] ep %d validation: R²=%.4f MAE=%.4f best=%.4f",
                stage_name,
                final_epoch,
                val_metrics["r2"],
                val_metrics["mae"],
                best_val_r2,
            )
            if val_metrics["r2"] > best_val_r2 + 1e-5:
                best_val_r2 = val_metrics["r2"]
                best_val_epoch = final_epoch
                best_val_state = copy.deepcopy(model.state_dict())
            elif (
                best_val_epoch > 0
                and final_epoch - best_val_epoch >= args.early_stop_patience
            ):
                LOG.info(
                    "[%s] Early stopping: no validation R² improvement for %d epochs",
                    stage_name,
                    args.early_stop_patience,
                )
                break

    if use_validation_checkpoint and best_val_state is not None:
        model.load_state_dict(best_val_state)
        chosen_epoch = best_val_epoch
        LOG.info(
            "[%s] Loaded validation-selected checkpoint (epoch=%d, R²=%.4f)",
            stage_name, chosen_epoch, best_val_r2,
        )
    elif swa_state is not None and n_swa > 0:
        model.load_state_dict(swa_state)
        chosen_epoch = final_epoch
        LOG.info("[%s] Loaded SWA weights (averaged over %d epochs, start=%d)",
                 stage_name, n_swa, swa_start)
    elif best_state is not None:
        model.load_state_dict(best_state)
        chosen_epoch = final_epoch
    else:
        chosen_epoch = final_epoch

    elapsed = time.time() - t0
    LOG.info(
        "[%s] Trained: best_train=%.4f selected_epoch=%d ran_epochs=%d elapsed=%.0fs",
        stage_name, best_train_loss, chosen_epoch, final_epoch, elapsed,
    )
    return {
        "epoch": int(chosen_epoch),
        "ran_epochs": int(final_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_r2": (
            float(best_val_r2) if use_validation_checkpoint else None
        ),
        "n_params": int(n_params),
        "elapsed_sec": float(elapsed),
    }


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────

def run_one(ds_name, cfg, device, out_dir, args):
    ds = get_dataset(ds_name)
    if args.predict_outer_unscored and not args.validation_only:
        raise ValueError(
            "--predict-outer-unscored requires --validation-only"
        )
    if args.retrieval_dim < 0:
        raise ValueError("--retrieval-dim must be non-negative")
    if args.metric_loss_weight < 0:
        raise ValueError("--metric-loss-weight must be non-negative")
    if args.metric_loss_weight > 0 and args.retrieval_dim < 1:
        raise ValueError(
            "--metric-loss-weight requires a positive --retrieval-dim so the "
            "retrieval objective does not distort the yield representation"
        )
    if not args.eknn_condition_scales:
        raise ValueError("--eknn-condition-scales cannot be empty")
    if any(scale < 0 for scale in args.eknn_condition_scales):
        raise ValueError("--eknn-condition-scales must be non-negative")
    if (
        any(scale > 0 for scale in args.eknn_condition_scales)
        and not args.use_condition_features
    ):
        raise ValueError(
            "A positive --eknn-condition-scales value requires "
            "--use-condition-features"
        )
    if not args.eknn_residual_scales:
        raise ValueError("--eknn-residual-scales cannot be empty")
    if any(scale < 0 for scale in args.eknn_residual_scales):
        raise ValueError("--eknn-residual-scales must be non-negative")
    if not args.eknn_base_alignment_scales:
        raise ValueError("--eknn-base-alignment-scales cannot be empty")
    if any(scale < 0 or scale > 1 for scale in args.eknn_base_alignment_scales):
        raise ValueError(
            "--eknn-base-alignment-scales values must lie in [0, 1]"
        )
    if args.component_identity_dim < 1:
        raise ValueError("--component-identity-dim must be positive")
    if (
        args.role_factorized_retrieval
        and not args.role_factorized_readout
    ):
        raise ValueError(
            "--role-factorized-retrieval requires "
            "--role-factorized-readout"
        )
    if args.component_identity_embeddings and not args.use_condition_features:
        raise ValueError(
            "--component-identity-embeddings currently requires "
            "--use-condition-features"
        )
    component_identity_roles = (
        parse_component_identity_roles(args.component_identity_roles, ds)
        if args.component_identity_embeddings
        else []
    )
    LOG.info("=" * 60)
    LOG.info("  Reaction GNN + eKNN residual transfer | %s", ds_name)
    LOG.info("  GNN layers=%d, d_model=%d, heads=%d, ff_dim=%d",
             args.n_layers, args.d_model, args.n_heads, args.ff_dim)
    LOG.info("=" * 60)

    # Load data
    cache_dir = (
        args.graph_cache_dir or os.path.join(out_dir, ".graph_cache")
    ) if args.cache_graphs else None
    train_graphs, train_targets, test_graphs, test_targets, std = \
        load_and_build_graphs(ds, args, cache_dir)
    condition_info = None

    train_graphs, train_targets = filter_valid(train_graphs, train_targets)
    test_graphs, test_targets = filter_valid(test_graphs, test_targets)
    LOG.info("Valid: train=%d test=%d", len(train_graphs), len(test_graphs))

    condition_dim = 0
    selection_categorical_cardinalities = ()
    final_categorical_cardinalities = ()
    selection_component_identity_cardinalities = ()
    final_component_identity_cardinalities = ()
    source_train_frame = None
    source_test_frame = None
    final_train_frame = None
    final_test_frame = None
    selection_frame = None
    val_frame = None
    selection_condition_info = None
    final_condition_info = None
    selection_component_identity_info = None
    final_component_identity_info = None
    if args.condition_categorical_embeddings and not args.use_condition_features:
        raise ValueError(
            "--condition-categorical-embeddings requires --use-condition-features"
        )
    if (
        args.role_interaction_readout
        or args.role_concat_readout
        or args.role_factorized_readout
    ) and not args.use_condition_features:
        raise ValueError(
            "Conditioned role readouts require --use-condition-features"
        )
    if sum(
        bool(value) for value in (
            args.role_interaction_readout,
            args.role_concat_readout,
            args.role_factorized_readout,
            args.component_readout,
            args.rolewise_readout,
        )
    ) > 1:
        raise ValueError(
            "Choose only one of --role-interaction-readout, "
            "--role-concat-readout, --role-factorized-readout, "
            "--rolewise-readout, and --component-readout"
        )
    if (args.condition_categorical_embeddings
            and args.condition_fusion_mode != "cross_attention"):
        raise ValueError(
            "--condition-categorical-embeddings requires "
            "--condition-fusion-mode cross_attention"
        )
    if args.condition_categorical_embeddings and args.include_categorical_conditions:
        raise ValueError(
            "Choose either learned categorical condition tokens or legacy one-hot "
            "categorical conditions, not both"
        )
    if args.use_condition_features:
        if args.random_split:
            raise ValueError(
                "--use-condition-features is not yet supported with --random-split; "
                "use a registered immutable split so fold-safe source-row alignment is preserved"
            )
        n_numeric_conditions = len(ds.condition_cols)
        n_categorical_conditions = (
            len(getattr(ds, "condition_categorical_cols", []))
            if (args.include_categorical_conditions or args.condition_categorical_embeddings)
            else 0
        )
        if n_numeric_conditions + n_categorical_conditions == 0:
            raise ValueError("Condition fusion requested, but this dataset has no condition columns")
        source_train_frame = pd.read_csv(ds.train_csv)
        source_test_frame = pd.read_csv(ds.test_csv)
        LOG.info(
            "Model variant: GNN + %s condition fusion%s%s (%d numeric + %d categorical raw conditions)",
            args.condition_fusion_mode,
            " + node-FiLM" if args.condition_node_film else "",
            " + categorical tokens" if args.condition_categorical_embeddings else "",
            n_numeric_conditions, n_categorical_conditions,
        )
        if args.component_identity_embeddings:
            LOG.info(
                "Fold-fitted molecular identity tokens enabled for roles=%s dim=%d",
                component_identity_roles,
                args.component_identity_dim,
            )
    else:
        LOG.info("Model variant: pure GNN baseline (graph inputs only)")

    # Two-stage honest validation protocol:
    # 1) train a selection model without validation rows and select k/T on val;
    # 2) retrain the final model on the full training set and evaluate test once
    #    with the fixed k/T selected in stage 1.
    t0_run = time.time()
    train_idx, val_idx, selection_split = make_selection_split(
        train_graphs, ds, args, cfg.train.seed
    )

    val_graphs_list = [train_graphs[i] for i in val_idx]
    val_targets_arr = train_targets[val_idx]
    selection_graphs = [train_graphs[i] for i in train_idx]
    selection_targets = train_targets[train_idx]

    if args.use_condition_features:
        selection_frame = graph_rows_to_frame(selection_graphs, source_train_frame)
        val_frame = graph_rows_to_frame(val_graphs_list, source_train_frame)
        selection_features, val_features, selection_condition_info = \
            prepare_condition_features(
                selection_frame, val_frame, ds,
                include_categorical=args.include_categorical_conditions,
            )
        selection_categorical_features = None
        val_categorical_features = None
        selection_categorical_info = None
        selection_component_identity_features = None
        val_component_identity_features = None
        if args.condition_categorical_embeddings:
            (selection_categorical_features,
             val_categorical_features,
             selection_categorical_cardinalities,
             selection_categorical_info) = prepare_categorical_condition_ids(
                 selection_frame, val_frame, ds
             )
        if args.component_identity_embeddings:
            (selection_component_identity_features,
             val_component_identity_features,
             selection_component_identity_cardinalities,
             selection_component_identity_info) = prepare_component_identity_ids(
                 selection_frame,
                 val_frame,
                 ds,
                 columns=component_identity_roles,
             )
        selection_graphs = attach_condition_features(
            selection_graphs,
            selection_features,
            selection_categorical_features,
            component_identity_features=selection_component_identity_features,
        )
        val_graphs_list = attach_condition_features(
            val_graphs_list,
            val_features,
            val_categorical_features,
            component_identity_features=val_component_identity_features,
        )
        condition_dim = int(selection_features.shape[1])
        LOG.info(
            "Fold-fitted condition feature dimension: %d%s",
            condition_dim,
            (f" + categorical cardinalities={list(selection_categorical_cardinalities)}"
             if selection_categorical_cardinalities else ""),
        )
        condition_info = {
            "selection_preprocessing": selection_condition_info,
            "selection_categorical_embedding_preprocessing": selection_categorical_info,
            "final_preprocessing": None,
            "final_categorical_embedding_preprocessing": None,
            "fit_policy": (
                "selection preprocessing fit on selection-train only; final "
                "preprocessing refit on the full outer training split"
            ),
        }

    selection_std = Standardizer(float(np.nanmean(selection_targets)),
                                 float(np.nanstd(selection_targets)))
    final_std = Standardizer(float(np.nanmean(train_targets)),
                             float(np.nanstd(train_targets)))

    protocol = (
        "two_stage_group_val_select_full_retrain"
        if args.group_val else "two_stage_val_select_full_retrain"
    )
    LOG.info(
        "Protocol: %s | selection_train=%d val=%d final_train=%d | val_split=%s",
        protocol, len(selection_graphs), len(val_graphs_list), len(train_graphs),
        selection_split["mode"],
    )
    if args.group_val:
        LOG.info(
            "Inner validation audit: column=%s groups=%d train=%d val=%d overlap=%d",
            selection_split["group_column"], selection_split["n_groups_total"],
            selection_split["n_groups_train"], selection_split["n_groups_val"],
            selection_split["overlapping_groups"],
        )

    # Stage 1: train only on fit_train and select k/T on held-out validation.
    utils.set_seed(cfg.train.seed, deterministic=args.deterministic)
    selection_model = build_model(
        args,
        condition_dim=condition_dim,
        condition_categorical_cardinalities=selection_categorical_cardinalities,
        component_identity_cardinalities=selection_component_identity_cardinalities,
    )
    selection_info = train_gnn_model(
        selection_model, selection_graphs, selection_targets,
        selection_std, device, args, stage_name="selection",
        validation_graphs=(
            val_graphs_list if args.select_checkpoint_on_val else None
        ),
        validation_targets=(
            val_targets_arr if args.select_checkpoint_on_val else None
        ))

    selected_eknn, val_grid = select_eknn_hyperparams(
        selection_model, selection_graphs, val_graphs_list,
        selection_targets, val_targets_arr, device, selection_std,
        condition_scales=args.eknn_condition_scales,
        residual_scales=args.eknn_residual_scales,
        base_alignment_scales=args.eknn_base_alignment_scales)
    selection_prediction_arrays = selected_eknn.pop("_predictions")
    selected_label_knn = select_label_knn_hyperparams(
        selection_model, selection_graphs, val_graphs_list,
        selection_targets, val_targets_arr, device, selection_std)

    selection_static = evaluate_static(
        selection_model, val_graphs_list, val_targets_arr, device, selection_std
    )
    LOG.info(
        "Selection static validation: R²=%.4f  MAE=%.4f",
        selection_static["r2"], selection_static["mae"],
    )

    if args.validation_only:
        seed = cfg.train.seed
        variant = (
            "gnn_condition_fusion_"
            f"{args.condition_fusion_mode}{'_nodefilm' if args.condition_node_film else ''}"
            f"{'_componentreadout' if args.component_readout else ''}"
            f"{'_rolewise' if args.rolewise_readout else ''}"
            f"{'_roleinteraction' if args.role_interaction_readout else ''}"
            f"{'_roleconcat' if args.role_concat_readout else ''}"
            f"{'_rolefactorized' if args.role_factorized_readout else ''}"
            f"{'_roleretrieval' if args.role_factorized_retrieval else ''}"
            f"{'_roleids' if args.component_identity_embeddings else ''}"
            f"{'_catconds' if args.include_categorical_conditions else ''}"
            f"{'_catembed' if args.condition_categorical_embeddings else ''}"
            if condition_dim else (
                "pure_gnn_componentreadout"
                if args.component_readout
                else ("pure_gnn_rolewise" if args.rolewise_readout else "pure_gnn")
            )
        )
        if args.retrieval_dim > 0:
            variant += (
                f"_retrieval{args.retrieval_dim}"
                f"_metric{str(args.metric_loss_weight).replace('.', 'p')}"
            )
        if args.readout != "mean_max" and not args.attn_pool:
            variant += f"_pool{args.readout}"
        validation_output = {
            "dataset": ds_name,
            "model": "reaction_gnn_validation",
            "variant": variant,
            "protocol": "selection_validation_only",
            "test_evaluated_once": False,
            "backbone": args.backbone,
            "seed": seed,
            "config": {
                "d_model": args.d_model, "n_layers": args.n_layers,
                "n_heads": args.n_heads, "ff_dim": args.ff_dim,
                "dropout": args.dropout, "condition_dim": condition_dim,
                "condition_fusion_mode": args.condition_fusion_mode if condition_dim else None,
                "include_categorical_conditions": bool(args.include_categorical_conditions) if condition_dim else False,
                "condition_categorical_embeddings": bool(args.condition_categorical_embeddings) if condition_dim else False,
                "condition_categorical_cardinalities": list(selection_categorical_cardinalities),
                "condition_token_layers": int(args.condition_token_layers) if condition_dim else None,
                "condition_node_film": bool(args.condition_node_film) if condition_dim else False,
                "component_readout": bool(args.component_readout),
                "rolewise_readout": bool(args.rolewise_readout),
                "role_interaction_readout": bool(args.role_interaction_readout),
                "role_interaction_layers": int(args.role_interaction_layers),
                "role_concat_readout": bool(args.role_concat_readout),
                "role_factorized_readout": bool(
                    args.role_factorized_readout
                ),
                "role_factorized_retrieval": bool(
                    args.role_factorized_retrieval
                ),
                "retrieval_dim": int(args.retrieval_dim),
                "metric_loss_weight": float(args.metric_loss_weight),
                "metric_label_temperature": float(args.metric_label_temperature),
                "metric_embedding_temperature": float(args.metric_embedding_temperature),
                "readout": "attn" if args.attn_pool else args.readout,
                "eknn_condition_scales": [
                    float(value) for value in args.eknn_condition_scales
                ],
                "eknn_residual_scales": [
                    float(value) for value in args.eknn_residual_scales
                ],
                "eknn_base_alignment_scales": [
                    float(value)
                    for value in args.eknn_base_alignment_scales
                ],
                "component_identity_embeddings": bool(args.component_identity_embeddings),
                "component_identity_dim": int(args.component_identity_dim),
                "component_identity_roles": component_identity_roles,
                "component_identity_cardinalities": [
                    int(value)
                    for value in selection_component_identity_cardinalities
                ],
                "select_checkpoint_on_val": bool(args.select_checkpoint_on_val),
                "val_check_interval": int(args.val_check_interval),
            },
            "condition_features": {
                "enabled": bool(condition_dim),
                "preprocessing": condition_info,
            },
            "component_identity_features": {
                "enabled": bool(args.component_identity_embeddings),
                "selection_preprocessing": selection_component_identity_info,
                "final_preprocessing": None,
                "fit_policy": "vocabulary fit on the provided fitting partition only",
            },
            "selection_split": selection_split,
            "selection": {
                "n_train": int(len(selection_graphs)),
                "n_val": int(len(val_graphs_list)),
                "epoch": selection_info["epoch"],
                "best_train_loss": selection_info["best_train_loss"],
                "static": selection_static,
                "eknn": selected_eknn,
                "label_knn": selected_label_knn,
            },
        }
        od = os.path.join(out_dir, ds_name)
        os.makedirs(od, exist_ok=True)
        tag = (
            f"validation_{variant}_{args.backbone}_d{args.d_model}_"
            f"L{args.n_layers}_s{seed}_ep{selection_info['epoch']}"
        )
        validation_prediction_path = os.path.join(
            od, f"{tag}_predictions.csv"
        )
        validation_prediction_frame = pd.DataFrame({
            "target": selection_prediction_arrays["target"],
            "static_prediction": selection_prediction_arrays["static"],
            "eknn_prediction": selection_prediction_arrays["eknn"],
        })
        if val_frame is not None and ds.group_col in val_frame.columns:
            validation_prediction_frame.insert(
                0,
                ds.group_col,
                val_frame[ds.group_col].astype(str).to_numpy(),
            )
        validation_prediction_frame.to_csv(
            validation_prediction_path, index=False
        )
        validation_output["validation_predictions_csv"] = (
            validation_prediction_path
        )
        if args.predict_outer_unscored:
            outer_query_frame = graph_rows_to_frame(
                test_graphs, source_test_frame
            )
            outer_query_graphs = test_graphs
            outer_condition_info = None
            outer_categorical_info = None
            outer_component_identity_info = None
            if args.use_condition_features:
                (
                    _,
                    outer_query_features,
                    outer_condition_info,
                ) = prepare_condition_features(
                    selection_frame,
                    outer_query_frame,
                    ds,
                    include_categorical=args.include_categorical_conditions,
                )
                outer_categorical_features = None
                outer_component_identity_features = None
                if args.condition_categorical_embeddings:
                    (
                        _,
                        outer_categorical_features,
                        outer_categorical_cardinalities,
                        outer_categorical_info,
                    ) = prepare_categorical_condition_ids(
                        selection_frame,
                        outer_query_frame,
                        ds,
                    )
                    if tuple(outer_categorical_cardinalities) != tuple(
                        selection_categorical_cardinalities
                    ):
                        raise RuntimeError(
                            "Categorical cardinalities changed while "
                            "reconstructing selection-fold preprocessing"
                        )
                if args.component_identity_embeddings:
                    (
                        _,
                        outer_component_identity_features,
                        outer_component_identity_cardinalities,
                        outer_component_identity_info,
                    ) = prepare_component_identity_ids(
                        selection_frame,
                        outer_query_frame,
                        ds,
                        columns=component_identity_roles,
                    )
                    if tuple(
                        outer_component_identity_cardinalities
                    ) != tuple(selection_component_identity_cardinalities):
                        raise RuntimeError(
                            "Identity cardinalities changed while "
                            "reconstructing selection-fold preprocessing"
                        )
                outer_query_graphs = attach_condition_features(
                    test_graphs,
                    outer_query_features,
                    outer_categorical_features,
                    component_identity_features=(
                        outer_component_identity_features
                    ),
                )

            outer_prediction_arrays = predict_eknn_fixed_unscored(
                selection_model,
                selection_graphs,
                outer_query_graphs,
                selection_targets,
                device,
                selection_std,
                selected_eknn,
            )
            outer_prediction_path = os.path.join(
                od, f"{tag}_outer_unscored_predictions.csv"
            )
            outer_prediction_frame = pd.DataFrame({
                "outer_row_index": np.asarray(
                    [
                        int(graph.reaction_idx)
                        for graph in test_graphs
                    ],
                    dtype=int,
                ),
                "static_prediction": outer_prediction_arrays["static"],
                "eknn_prediction": outer_prediction_arrays["eknn"],
            })
            if ds.group_col in outer_query_frame.columns:
                outer_prediction_frame.insert(
                    1,
                    ds.group_col,
                    outer_query_frame[ds.group_col]
                    .astype(str)
                    .to_numpy(),
                )
            outer_prediction_frame.to_csv(
                outer_prediction_path, index=False
            )
            validation_output["outer_unscored_prediction"] = {
                "csv": outer_prediction_path,
                "n_rows": int(len(outer_prediction_frame)),
                "target_exported": False,
                "metrics_computed": False,
                "preprocessing": outer_condition_info,
                "categorical_preprocessing": outer_categorical_info,
                "component_identity_preprocessing": (
                    outer_component_identity_info
                ),
            }
        validation_output["release_provenance"] = release_provenance(ds, args)
        utils.dump_json(os.path.join(od, f"{tag}.json"), validation_output)
        LOG.info(
            "Validation-only complete; outer targets were not scored or "
            "exported. Static val R²=%.4f, eKNN val R²=%.4f | saved %s.json",
            selection_static["r2"], selected_eknn["val_r2"], tag,
        )
        del selection_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    del selection_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Stage 2: retrain final model on the full training set and evaluate test once.
    if args.use_condition_features:
        final_train_frame = graph_rows_to_frame(train_graphs, source_train_frame)
        final_test_frame = graph_rows_to_frame(test_graphs, source_test_frame)
        final_train_features, final_test_features, final_condition_info = \
            prepare_condition_features(
                final_train_frame, final_test_frame, ds,
                include_categorical=args.include_categorical_conditions,
            )
        final_train_categorical_features = None
        final_test_categorical_features = None
        final_categorical_info = None
        final_train_component_identity_features = None
        final_test_component_identity_features = None
        if args.condition_categorical_embeddings:
            (final_train_categorical_features,
             final_test_categorical_features,
             final_categorical_cardinalities,
             final_categorical_info) = prepare_categorical_condition_ids(
                 final_train_frame, final_test_frame, ds
             )
        if args.component_identity_embeddings:
            (final_train_component_identity_features,
             final_test_component_identity_features,
             final_component_identity_cardinalities,
             final_component_identity_info) = prepare_component_identity_ids(
                 final_train_frame,
                 final_test_frame,
                 ds,
                 columns=component_identity_roles,
             )
        final_train_graphs = attach_condition_features(
            train_graphs,
            final_train_features,
            final_train_categorical_features,
            component_identity_features=final_train_component_identity_features,
        )
        final_test_graphs = attach_condition_features(
            test_graphs,
            final_test_features,
            final_test_categorical_features,
            component_identity_features=final_test_component_identity_features,
        )
        condition_dim = int(final_train_features.shape[1])
        condition_info["final_preprocessing"] = final_condition_info
        condition_info["final_categorical_embedding_preprocessing"] = final_categorical_info
    else:
        final_train_graphs = train_graphs
        final_test_graphs = test_graphs

    utils.set_seed(cfg.train.seed, deterministic=args.deterministic)
    model = build_model(
        args,
        condition_dim=condition_dim,
        condition_categorical_cardinalities=final_categorical_cardinalities,
        component_identity_cardinalities=final_component_identity_cardinalities,
    )
    final_info = train_gnn_model(
        model, final_train_graphs, train_targets,
        final_std, device, args, stage_name="final")
    final_epoch = final_info["epoch"]
    n_params = final_info["n_params"]

    if args.save_ckpt:
        os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
        torch.save({'model': model.state_dict(), 'epoch': final_epoch,
                     'best_train_loss': final_info["best_train_loss"],
                     'std_mean': final_std.mean, 'std_std': final_std.std,
                     'protocol': protocol,
                     'selected_condition_scale': selected_eknn.get("condition_scale", 0.0),
                     'selected_residual_scale': selected_eknn.get("residual_scale", 1.0),
                     'selected_base_alignment_scale': selected_eknn.get("base_alignment_scale", 0.0),
                     'selected_k': selected_eknn["k"], 'selected_T': selected_eknn["T"]},
                    args.save_ckpt)
        LOG.info("Checkpoint saved: %s", args.save_ckpt)

    results = evaluate_eknn_fixed(
        model, final_train_graphs, final_test_graphs,
        train_targets, test_targets, device, final_std, selected_eknn)
    prediction_arrays = results.pop("_predictions")

    label_knn_results = evaluate_label_knn_fixed(
        model, final_train_graphs, final_test_graphs,
        train_targets, test_targets, device, final_std, selected_label_knn)

    # Save
    seed = cfg.train.seed
    output = {
        "dataset": ds_name,
        "model": "reaction_gnn_eknn",
        "protocol": protocol,
        "test_evaluated_once": True,
        "backbone": args.backbone,
        "split": "random" if args.random_split else "group",
        "split_seed": args.random_split_seed if args.random_split else None,
        "config": {
            "d_model": args.d_model, "n_layers": args.n_layers,
            "n_heads": args.n_heads, "ff_dim": args.ff_dim,
            "dropout": args.dropout, "condition_dim": condition_dim,
            "condition_fusion_mode": args.condition_fusion_mode if condition_dim else None,
            "include_categorical_conditions": bool(args.include_categorical_conditions) if condition_dim else False,
            "condition_categorical_embeddings": bool(args.condition_categorical_embeddings) if condition_dim else False,
            "condition_categorical_cardinalities": list(final_categorical_cardinalities),
            "condition_token_layers": int(args.condition_token_layers) if condition_dim else None,
            "condition_node_film": bool(args.condition_node_film) if condition_dim else False,
            "component_readout": bool(args.component_readout),
            "rolewise_readout": bool(args.rolewise_readout),
            "role_interaction_readout": bool(args.role_interaction_readout),
            "role_interaction_layers": int(args.role_interaction_layers),
            "role_concat_readout": bool(args.role_concat_readout),
            "role_factorized_readout": bool(
                args.role_factorized_readout
            ),
            "role_factorized_retrieval": bool(
                args.role_factorized_retrieval
            ),
            "retrieval_dim": int(args.retrieval_dim),
            "metric_loss_weight": float(args.metric_loss_weight),
            "metric_label_temperature": float(args.metric_label_temperature),
            "metric_embedding_temperature": float(args.metric_embedding_temperature),
            "readout": "attn" if args.attn_pool else args.readout,
            "eknn_condition_scales": [
                float(value) for value in args.eknn_condition_scales
            ],
            "eknn_residual_scales": [
                float(value) for value in args.eknn_residual_scales
            ],
            "eknn_base_alignment_scales": [
                float(value) for value in args.eknn_base_alignment_scales
            ],
            "component_identity_embeddings": bool(args.component_identity_embeddings),
            "component_identity_dim": int(args.component_identity_dim),
            "component_identity_roles": component_identity_roles,
            "component_identity_cardinalities": [
                int(value) for value in final_component_identity_cardinalities
            ],
            "select_checkpoint_on_val": bool(args.select_checkpoint_on_val),
            "val_check_interval": int(args.val_check_interval),
        },
        "condition_features": {
            "enabled": bool(condition_dim),
            "preprocessing": condition_info,
        },
        "component_identity_features": {
            "enabled": bool(args.component_identity_embeddings),
            "selection_preprocessing": selection_component_identity_info,
            "final_preprocessing": final_component_identity_info,
            "fit_policy": "vocabulary fit on the provided fitting partition only",
        },
        "selection_split": selection_split,
        "n_params": n_params,
        "epoch": final_epoch,
        "seed": seed,
        "selection": {
            "n_train": int(len(selection_graphs)),
            "n_val": int(len(val_graphs_list)),
            "epoch": selection_info["epoch"],
            "best_train_loss": selection_info["best_train_loss"],
            "std_mean": selection_std.mean,
            "std_std": selection_std.std,
            "static": selection_static,
            "selected_eknn": selected_eknn,
            "selected_label_knn": selected_label_knn,
        },
        "final": {
            "n_train": int(len(train_graphs)),
            "epoch": final_info["epoch"],
            "best_train_loss": final_info["best_train_loss"],
            "std_mean": final_std.mean,
            "std_std": final_std.std,
        },
        "static": {"r2": results["static_r2"], "mae": results["static_mae"]},
        "label_knn": label_knn_results,
        "val_selected": selected_eknn,
        "val_grid_search": val_grid,
        "grid_search": results["grid_search"],
        "grid_search_note": results["grid_search_note"],
        "best": results["best"],
        "ensemble": results["ensemble"],
        "n_train": results["n_train"],
        "n_val": int(len(val_graphs_list)),
        "selection_n_train": int(len(selection_graphs)),
        "final_n_train": int(len(train_graphs)),
        "n_test": results["n_test"],
    }

    od = os.path.join(out_dir, ds_name)
    os.makedirs(od, exist_ok=True)
    bb = getattr(args, 'backbone', 'gps')
    variant_tag = (
        f"cond_{args.condition_fusion_mode}{'_nodefilm' if args.condition_node_film else ''}"
        f"{'_componentreadout' if args.component_readout else ''}"
        f"{'_rolewise' if args.rolewise_readout else ''}"
        f"{'_roleinteraction' if args.role_interaction_readout else ''}"
        f"{'_roleconcat' if args.role_concat_readout else ''}"
        f"{'_rolefactorized' if args.role_factorized_readout else ''}"
        f"{'_roleretrieval' if args.role_factorized_retrieval else ''}"
        f"{'_roleids' if args.component_identity_embeddings else ''}"
        f"{'_catconds' if args.include_categorical_conditions else ''}"
        f"{'_catembed' if args.condition_categorical_embeddings else ''}"
        if condition_dim else (
            "graphonly_componentreadout"
            if args.component_readout
            else ("graphonly_rolewise" if args.rolewise_readout else "graphonly")
        )
    )
    if args.retrieval_dim > 0:
        variant_tag += (
            f"_retrieval{args.retrieval_dim}"
            f"_metric{str(args.metric_loss_weight).replace('.', 'p')}"
        )
    if args.readout != "mean_max" and not args.attn_pool:
        variant_tag += f"_pool{args.readout}"
    val_tag = "groupval" if args.group_val else "rowval"
    tag = (
        f"gnn_{variant_tag}_{bb}_d{args.d_model}_L{args.n_layers}_"
        f"eknn_twostage_{val_tag}_v{str(args.val_frac).replace(chr(46),chr(112))}_"
        f"s{seed}_ep{final_epoch}"
    )
    prediction_path = os.path.join(od, f"ttt_{tag}_test_predictions.csv")
    validation_prediction_path = os.path.join(
        od, f"ttt_{tag}_validation_predictions.csv"
    )
    validation_prediction_frame = pd.DataFrame({
        "target": selection_prediction_arrays["target"],
        "static_prediction": selection_prediction_arrays["static"],
        "eknn_prediction": selection_prediction_arrays["eknn"],
    })
    if val_frame is not None and ds.group_col in val_frame.columns:
        validation_prediction_frame.insert(
            0,
            ds.group_col,
            val_frame[ds.group_col].astype(str).to_numpy(),
        )
    validation_prediction_frame.to_csv(
        validation_prediction_path, index=False
    )
    prediction_frame = pd.DataFrame({
        "target": prediction_arrays["target"],
        "static_prediction": prediction_arrays["static"],
        "eknn_prediction": prediction_arrays["eknn"],
    })
    if final_test_frame is not None and ds.group_col in final_test_frame.columns:
        prediction_frame.insert(
            0,
            ds.group_col,
            final_test_frame[ds.group_col].astype(str).to_numpy(),
        )
    prediction_frame.to_csv(prediction_path, index=False)
    output["test_predictions_csv"] = prediction_path
    output["validation_predictions_csv"] = validation_prediction_path
    output["release_provenance"] = release_provenance(ds, args)
    utils.dump_json(os.path.join(od, f"ttt_{tag}.json"), output)
    LOG.info("Saved: ttt_%s.json", tag)

    elapsed = time.time() - t0_run
    LOG.info("Done. Static R²=%.4f Fixed(k=%d,T=%.2f) ΔR²=%+.4f Ensemble ΔR²=%+.4f elapsed=%.0fs",
             results['static_r2'], results['best']['k'], results['best']['T'],
             results['best']['delta_r2'], results['ensemble']['delta_r2'], elapsed)


# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = ReactionConfig()
    if args.set:
        cfg = override_from_dotlist(cfg, args.set)
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.batch_size = args.batch_size
    utils.set_seed(cfg.train.seed, deterministic=args.deterministic)
    device = utils.get_device(cfg.train.device)

    custom = [args.train_csv, args.test_csv, args.smiles_cols, args.target_col]
    if any(custom):
        if not all(custom) or "," in args.dataset:
            raise ValueError("Custom CSVs require --train-csv, --test-csv, --smiles-cols, --target-col and one dataset name")
        DATASETS[args.dataset] = DatasetConfig(
            name=args.dataset, dataset="custom", train_csv=args.train_csv,
            test_csv=args.test_csv, smiles_cols=args.smiles_cols,
            target_col=args.target_col, group_col=args.group_col,
            target_name=args.target_col, description="User-provided split",
        )
    failed = False
    for ds in args.dataset.split(","):
        try:
            run_one(ds, cfg, device, args.out_dir, args)
        except Exception as e:
            failed = True
            LOG.error("FAILED: %s — %s", ds, e)
            import traceback
            traceback.print_exc()
    LOG.info("All done.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

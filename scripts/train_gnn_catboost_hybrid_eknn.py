#!/usr/bin/env python3
"""Condition-aware GNN + CatBoost hybrid with eKNN residual correction.

The base reaction GNN consumes all molecular graphs and fold-standardised
numerical reaction conditions.  A second, condition-aware CatBoost branch
uses role-preserving Morgan fingerprints, numerical conditions and registered
categorical conditions.  It can additionally consume the learned GNN
embedding.  The final hybrid retains the GNN eKNN residual correction:

    prediction = alpha * GNN-eKNN + (1 - alpha) * CatBoost(GNN, FP, conditions)

All architecture, CatBoost representation, eKNN k/T and blend weight choices
are made on a reaction-key-disjoint inner validation split.  The outer test is
constructed only after that selection and is evaluated once.

Example:
  python scripts/train_gnn_catboost_hybrid_eknn.py \
      --dataset michael_ddg_random --group-val --device cuda --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Set before importing torch so cuBLAS can honour deterministic workspace
# settings when --deterministic is used below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from catboost import CatBoostRegressor
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# kcal mol^-1 K^-1.  The Michael dataset's ddG values were derived from
# enantiomeric excess using the usual RT log(er) relation.
GAS_CONSTANT_KCAL = 0.00198720425864083

from eknn_reaction import utils
from eknn_reaction.data import get_dataset
from eknn_reaction.metrics import mae, r2

_GNN_SPEC = importlib.util.spec_from_file_location(
    "train_gnn_eknn", ROOT / "scripts" / "train_gnn_eknn.py"
)
train_gnn_eknn = importlib.util.module_from_spec(_GNN_SPEC)
assert _GNN_SPEC.loader is not None
_GNN_SPEC.loader.exec_module(train_gnn_eknn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/michael_gnn_condition_hybrid"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false",
                        help="Allow nondeterministic CUDA kernels (not recommended for final benchmarks)")
    parser.add_argument("--validation-only", action="store_true",
                        help="Select on the inner split only; never evaluate outer-test targets")
    parser.add_argument("--group-val", action="store_true",
                        help="Use the registered reaction-key group column for inner validation")
    parser.add_argument("--val-frac", type=float, default=0.15)

    # GNN trunk.  Numerical reaction conditions are always fused into it.
    parser.add_argument("--backbone", type=str, default="mpnn",
                        choices=["gps", "mpnn", "gat", "graph_transformer",
                                 "edge_transformer", "graphormer", "deep_gps", "light",
                                 "gcn", "sage", "gin", "gatv2", "transformer_conv",
                                 "pna", "general_conv", "gen_conv", "film",
                                 "gated_graph", "res_gated", "egconv"])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--swa-start", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--condition-node-film", action="store_true")
    parser.add_argument(
        "--gnn-target-mode",
        choices=["ddg", "ddg_per_kelvin"],
        default="ddg",
        help=("Target used to train GNN/eKNN. The physical mode predicts "
              "ddG divided by known temperature and is converted back before scoring."),
    )
    parser.add_argument(
        "--no-component-identity-embeddings",
        dest="component_identity_embeddings",
        action="store_false",
        help=("Disable fold-fitted per-role molecular identity tokens in the "
              "condition-aware GNN"),
    )
    parser.add_argument(
        "--component-identity-dim", type=int, default=32,
        help="Embedding width for each fold-fitted molecular-role identity token",
    )
    parser.add_argument(
        "--component-identity-roles",
        type=str,
        default="acceptor_smiles,donor_smiles,catalyst_smiles,ligand_smiles,solvent_smiles,additive_smiles",
        help=("Comma-separated molecular roles given fold-fitted identity tokens; "
              "all roles by default"),
    )
    parser.add_argument("--pe-dim", type=int, default=3)
    parser.add_argument("--no-cache-graphs", dest="cache_graphs", action="store_false")
    parser.add_argument("--graph-cache-dir", type=Path, default=None,
                        help="Optional frozen graph-cache directory shared across runs")
    parser.set_defaults(
        cache_graphs=True,
        deterministic=True,
        component_identity_embeddings=True,
        catboost_component_identities=True,
    )

    # CatBoost condition branch.
    parser.add_argument("--fp-bits", type=int, default=1024)
    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--catboost-iterations", type=int, default=600)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--catboost-lr", type=float, default=0.05)
    parser.add_argument("--catboost-l2", type=float, default=8.0)
    parser.add_argument(
        "--catboost-target-mode",
        choices=["ddg", "ddg_per_kelvin", "ee_percent"],
        default="ddg",
        help=("Regression target for the condition branch. Physical modes use "
              "only known temperature and are inverted to ddG before scoring."),
    )
    parser.add_argument(
        "--no-catboost-component-identities",
        dest="catboost_component_identities",
        action="store_false",
        help="Exclude role-preserving raw molecular identity categories from CatBoost",
    )
    parser.add_argument(
        "--catboost-component-pairs", action="store_true",
        help=("Add valid prediction-time substrate-pair and catalyst-ligand "
              "categories to CatBoost; select this option on inner validation"),
    )
    parser.add_argument(
        "--catboost-representations",
        type=str,
        default="morgan_conditions,gnn_embedding_morgan_conditions",
        help="Comma-separated candidates selected on validation",
    )
    parser.add_argument(
        "--blend-weights",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="GNN-eKNN weights alpha in alpha*GNN-eKNN+(1-alpha)*CatBoost",
    )
    parser.add_argument(
        "--min-gnn-weight",
        type=float,
        default=0.1,
        help="Minimum alpha eligible as the reported hybrid (default: retain at least 10%% GNN-eKNN)",
    )
    return parser.parse_args()


def _parse_csv_values(text: str, cast, name: str) -> list:
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _parse_component_identity_roles(args: argparse.Namespace, ds) -> list[str]:
    roles = _parse_csv_values(
        args.component_identity_roles, str, "--component-identity-roles"
    )
    unknown = sorted(set(roles) - set(ds.smiles_cols))
    if unknown:
        raise ValueError(
            f"Unknown component identity role(s): {unknown}; expected a subset of {ds.smiles_cols}"
        )
    if len(set(roles)) != len(roles):
        raise ValueError("--component-identity-roles must not contain duplicates")
    return roles


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _morgan_component_fingerprint(
    smiles: object, generator: Any, n_bits: int
) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    bit_vector = generator.GetFingerprint(mol)
    output = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, output)
    return output.astype(np.float32)


def reaction_fingerprints(
    frame: pd.DataFrame, smiles_cols: list[str], radius: int, n_bits: int
) -> np.ndarray:
    """Build a role-preserving Morgan vector for every reaction row."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows = []
    for _, row in frame.iterrows():
        rows.append(np.concatenate([
            _morgan_component_fingerprint(row[column], generator, n_bits)
            for column in smiles_cols
        ]))
    return np.vstack(rows).astype(np.float32)


def _cat_values(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(index=np.arange(len(frame)))
    output = pd.DataFrame(index=np.arange(len(frame)))
    for column in columns:
        values = frame[column].astype(object).where(frame[column].notna(), "__MISSING__")
        output[column] = values.astype(str).replace({"nan": "__MISSING__"})
    return output


def _component_category_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return role-preserving molecular categories with non-colliding names."""
    output = pd.DataFrame(index=np.arange(len(frame)))
    for column in columns:
        values = frame[column].astype(object).where(frame[column].notna(), "__MISSING__")
        output[f"component_identity::{column}"] = (
            values.astype(str).replace({"nan": "__MISSING__"})
        )
    return output


def _component_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reaction sub-combinations available before observing the outcome."""
    required = {
        "acceptor_smiles", "donor_smiles", "catalyst_smiles", "ligand_smiles"
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame(index=np.arange(len(frame)))
    values = _component_category_frame(frame, sorted(required))
    output = pd.DataFrame(index=np.arange(len(frame)))
    output["component_pair::acceptor_donor"] = (
        values["component_identity::acceptor_smiles"]
        + "\x1f" + values["component_identity::donor_smiles"]
    )
    output["component_pair::catalyst_ligand"] = (
        values["component_identity::catalyst_smiles"]
        + "\x1f" + values["component_identity::ligand_smiles"]
    )
    return output


def build_catboost_features(
    fit_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    ds,
    *,
    fp_bits: int,
    fp_radius: int,
    fit_embedding: np.ndarray | None = None,
    query_embedding: np.ndarray | None = None,
    include_component_identities: bool = True,
    include_component_pairs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    """Build CatBoost inputs with fold-fitted condition preprocessing only."""
    fit_fp = reaction_fingerprints(fit_frame, ds.smiles_cols, fp_radius, fp_bits)
    query_fp = reaction_fingerprints(query_frame, ds.smiles_cols, fp_radius, fp_bits)
    fp_columns = [f"fp_{index}" for index in range(fit_fp.shape[1])]
    fit_parts = [pd.DataFrame(fit_fp, columns=fp_columns)]
    query_parts = [pd.DataFrame(query_fp, columns=fp_columns)]

    fit_conditions, query_conditions, condition_info = \
        train_gnn_eknn.prepare_condition_features(
            fit_frame, query_frame, ds, include_categorical=False
        )
    condition_columns = [f"condition_{index}" for index in range(fit_conditions.shape[1])]
    fit_parts.append(pd.DataFrame(fit_conditions, columns=condition_columns))
    query_parts.append(pd.DataFrame(query_conditions, columns=condition_columns))

    if fit_embedding is not None or query_embedding is not None:
        if fit_embedding is None or query_embedding is None:
            raise ValueError("Both fit and query GNN embeddings must be supplied together")
        if fit_embedding.shape[0] != len(fit_frame) or query_embedding.shape[0] != len(query_frame):
            raise ValueError("GNN embedding rows do not align with CatBoost feature rows")
        embedding_columns = [f"gnn_embedding_{index}" for index in range(fit_embedding.shape[1])]
        fit_parts.append(pd.DataFrame(fit_embedding, columns=embedding_columns))
        query_parts.append(pd.DataFrame(query_embedding, columns=embedding_columns))
    else:
        embedding_columns = []

    categorical_columns = list(getattr(ds, "condition_categorical_cols", []))
    fit_parts.append(_cat_values(fit_frame, categorical_columns))
    query_parts.append(_cat_values(query_frame, categorical_columns))
    component_identity_columns: list[str] = []
    component_pair_columns: list[str] = []
    if include_component_identities:
        fit_component_ids = _component_category_frame(fit_frame, ds.smiles_cols)
        query_component_ids = _component_category_frame(query_frame, ds.smiles_cols)
        component_identity_columns = list(fit_component_ids.columns)
        fit_parts.append(fit_component_ids)
        query_parts.append(query_component_ids)
    if include_component_pairs:
        fit_component_pairs = _component_pair_frame(fit_frame)
        query_component_pairs = _component_pair_frame(query_frame)
        component_pair_columns = list(fit_component_pairs.columns)
        if list(fit_component_pairs.columns) != list(query_component_pairs.columns):
            raise RuntimeError("Component-pair train/query schemas differ")
        fit_parts.append(fit_component_pairs)
        query_parts.append(query_component_pairs)
    categorical_columns = (
        categorical_columns + component_identity_columns + component_pair_columns
    )
    fit_features = pd.concat(fit_parts, axis=1)
    query_features = pd.concat(query_parts, axis=1)
    if list(fit_features.columns) != list(query_features.columns):
        raise RuntimeError("CatBoost train/query feature schemas differ")
    info = {
        "fingerprint": {
            "radius": int(fp_radius),
            "n_bits_per_component": int(fp_bits),
            "n_components": int(len(ds.smiles_cols)),
        },
        "condition_preprocessing": condition_info,
        "categorical_columns": categorical_columns,
        "component_identity_columns": component_identity_columns,
        "component_pair_columns": component_pair_columns,
        "n_gnn_embedding_features": int(len(embedding_columns)),
        "n_features": int(fit_features.shape[1]),
    }
    return fit_features, query_features, categorical_columns, info


def make_catboost(args: argparse.Namespace) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=args.catboost_iterations,
        depth=args.catboost_depth,
        learning_rate=args.catboost_lr,
        l2_leaf_reg=args.catboost_l2,
        loss_function="RMSE",
        random_seed=args.seed,
        verbose=False,
        thread_count=-1,
    )


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2(y_pred, y_true)),
        "mae": float(mae(y_pred, y_true)),
    }


def _temperature_kelvin(frame: pd.DataFrame) -> np.ndarray:
    """Return the known reaction temperature in K for a physical target map."""
    column = "cond_temperature_c"
    if column not in frame.columns:
        raise KeyError(
            f"{column!r} is required for a temperature-normalised target"
        )
    temperature = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    temperature = temperature + 273.15
    if not np.isfinite(temperature).all() or np.any(temperature <= 0):
        raise ValueError(
            "Physical target transformation requires finite temperatures above absolute zero"
        )
    return temperature.astype(np.float64)


def catboost_training_target(
    ddg: np.ndarray, frame: pd.DataFrame, mode: str
) -> np.ndarray:
    """Map a ddG training label to a physically meaningful alternative target."""
    values = np.asarray(ddg, dtype=np.float64)
    if mode == "ddg":
        return values
    temperature = _temperature_kelvin(frame)
    if mode == "ddg_per_kelvin":
        return values / temperature
    if mode == "ee_percent":
        # ddG = 2RT atanh(ee / 100).  The clipped range protects against
        # tiny numerical excursions beyond the mathematical domain.
        ee = 100.0 * np.tanh(values / (2.0 * GAS_CONSTANT_KCAL * temperature))
        return np.clip(ee, 0.0, 99.999)
    raise ValueError(f"Unknown CatBoost target mode {mode!r}")


def catboost_prediction_to_ddg(
    prediction: np.ndarray, frame: pd.DataFrame, mode: str
) -> np.ndarray:
    """Invert ``catboost_training_target`` using only prediction-time inputs."""
    values = np.asarray(prediction, dtype=np.float64)
    if mode == "ddg":
        return values.astype(np.float32)
    temperature = _temperature_kelvin(frame)
    if mode == "ddg_per_kelvin":
        return (values * temperature).astype(np.float32)
    if mode == "ee_percent":
        ee_fraction = np.clip(values / 100.0, 0.0, 0.99999)
        ddg = 2.0 * GAS_CONSTANT_KCAL * temperature * np.arctanh(ee_fraction)
        return ddg.astype(np.float32)
    raise ValueError(f"Unknown CatBoost target mode {mode!r}")


def gnn_predictions_with_eknn(
    model,
    fit_graphs,
    query_graphs,
    fit_targets_model_units: np.ndarray,
    std,
    device,
    *,
    k: int,
    temperature: float,
    query_frame: pd.DataFrame,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ddG-scale static/eKNN predictions without querying target labels."""
    h_fit_norm, residual_fit, _ = train_gnn_eknn.build_eknn_index(
        model, fit_graphs, fit_targets_model_units, device, std
    )
    h_query, y_query_norm = train_gnn_eknn.encode_reactions(model, query_graphs, device)
    h_query_norm = h_query / (np.linalg.norm(h_query, axis=1, keepdims=True) + 1e-8)
    correction = train_gnn_eknn.knn_correct(
        h_query_norm, h_fit_norm, residual_fit, k, temperature
    )
    static_prediction = catboost_prediction_to_ddg(
        std.denormalize(y_query_norm), query_frame, target_mode
    )
    eknn_prediction = catboost_prediction_to_ddg(
        std.denormalize(y_query_norm + correction), query_frame, target_mode
    )
    return static_prediction, eknn_prediction, h_fit_norm, h_query, h_query_norm


def select_gnn_eknn_for_ddg(
    model,
    fit_graphs,
    val_graphs,
    fit_targets_model_units: np.ndarray,
    val_targets_ddg: np.ndarray,
    val_frame: pd.DataFrame,
    device,
    std,
    *,
    target_mode: str,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Select eKNN k/T on ddG after inverting a possible physical target map."""
    h_fit_norm, residual_fit, _ = train_gnn_eknn.build_eknn_index(
        model, fit_graphs, fit_targets_model_units, device, std
    )
    h_val, y_val_base_norm = train_gnn_eknn.encode_reactions(model, val_graphs, device)
    h_val_norm = h_val / (np.linalg.norm(h_val, axis=1, keepdims=True) + 1e-8)

    best_r2 = float("-inf")
    best_combo: tuple[int, float] | None = None
    grid: dict[str, dict[str, float]] = {}
    for k, temperature in itertools.product(train_gnn_eknn.K_GRID, train_gnn_eknn.T_GRID):
        correction = train_gnn_eknn.knn_correct(
            h_val_norm, h_fit_norm, residual_fit, k, temperature
        )
        prediction = catboost_prediction_to_ddg(
            std.denormalize(y_val_base_norm + correction), val_frame, target_mode
        )
        metrics = score(val_targets_ddg, prediction)
        key = f"k{k}_T{str(temperature).replace('.', 'p')}"
        grid[key] = {"k": int(k), "T": float(temperature), **metrics}
        if metrics["r2"] > best_r2:
            best_r2 = metrics["r2"]
            best_combo = (int(k), float(temperature))
    assert best_combo is not None
    return {"k": best_combo[0], "T": best_combo[1], "val_r2": float(best_r2)}, grid


def select_catboost_and_blend(
    args: argparse.Namespace,
    ds,
    fit_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    fit_targets: np.ndarray,
    val_targets: np.ndarray,
    fit_embedding: np.ndarray,
    val_embedding: np.ndarray,
    gnn_eknn_val: np.ndarray,
) -> tuple[dict[str, Any], CatBoostRegressor, np.ndarray]:
    """Select the CatBoost representation and a GNN-retaining blend weight."""
    representations = _parse_csv_values(
        args.catboost_representations, str, "--catboost-representations"
    )
    allowed = {"morgan_conditions", "gnn_embedding_morgan_conditions"}
    unknown = sorted(set(representations) - allowed)
    if unknown:
        raise ValueError(f"Unknown CatBoost representation(s): {unknown}")
    if not (0.0 <= args.min_gnn_weight <= 1.0):
        raise ValueError("--min-gnn-weight must be in [0, 1]")

    candidate_rows: list[dict[str, Any]] = []
    best: tuple[float, str, CatBoostRegressor, np.ndarray, dict[str, Any]] | None = None
    for representation in representations:
        kwargs = {}
        if representation == "gnn_embedding_morgan_conditions":
            kwargs = {"fit_embedding": fit_embedding, "query_embedding": val_embedding}
        x_fit, x_val, categorical_columns, feature_info = build_catboost_features(
            fit_frame, val_frame, ds,
            fp_bits=args.fp_bits,
            fp_radius=args.fp_radius,
            include_component_identities=args.catboost_component_identities,
            include_component_pairs=args.catboost_component_pairs,
            **kwargs,
        )
        model = make_catboost(args)
        model.fit(
            x_fit,
            catboost_training_target(
                fit_targets, fit_frame, args.catboost_target_mode
            ),
            cat_features=categorical_columns,
        )
        prediction = catboost_prediction_to_ddg(
            model.predict(x_val), val_frame, args.catboost_target_mode
        )
        metrics = score(val_targets, prediction)
        candidate_rows.append({
            "representation": representation,
            "metrics": metrics,
            "feature_info": feature_info,
            "target_mode": args.catboost_target_mode,
        })
        if best is None or metrics["r2"] > best[0]:
            best = (metrics["r2"], representation, model, prediction, feature_info)

    assert best is not None
    _, selected_representation, selected_model, catboost_val, selected_feature_info = best
    blend_weights = _parse_csv_values(args.blend_weights, float, "--blend-weights")
    if any(weight < 0 or weight > 1 for weight in blend_weights):
        raise ValueError("--blend-weights values must be in [0, 1]")
    blend_rows = []
    best_blend: tuple[float, float, dict[str, float]] | None = None
    for alpha in sorted(set(blend_weights)):
        prediction = alpha * gnn_eknn_val + (1.0 - alpha) * catboost_val
        metrics = score(val_targets, prediction)
        row = {"gnn_eknn_weight": float(alpha), **metrics}
        blend_rows.append(row)
        if alpha + 1e-12 >= args.min_gnn_weight:
            candidate = (metrics["r2"], float(alpha), metrics)
            if best_blend is None or candidate[0] > best_blend[0]:
                best_blend = candidate
    if best_blend is None:
        raise ValueError("No --blend-weights value meets --min-gnn-weight")

    return (
        {
            "candidates": candidate_rows,
            "selected_representation": selected_representation,
            "selected_metrics": score(val_targets, catboost_val),
            "selected_feature_info": selected_feature_info,
            "blend_candidates": blend_rows,
            "selected_blend_weight": best_blend[1],
            "selected_blend_metrics": best_blend[2],
        },
        selected_model,
        catboost_val,
    )


def _conditioned_graphs_for_fold(
    fit_graphs,
    query_graphs,
    fit_frame: pd.DataFrame,
    query_frame: pd.DataFrame,
    ds,
    *,
    use_component_identity_embeddings: bool,
    component_identity_roles: list[str],
):
    fit_conditions, query_conditions, info = train_gnn_eknn.prepare_condition_features(
        fit_frame, query_frame, ds, include_categorical=False
    )
    fit_component_ids = None
    query_component_ids = None
    component_cardinalities: list[int] = []
    component_info: dict[str, Any] | None = None
    if use_component_identity_embeddings:
        (fit_component_ids,
         query_component_ids,
         component_cardinalities,
         component_info) = train_gnn_eknn.prepare_component_identity_ids(
             fit_frame, query_frame, ds, columns=component_identity_roles
         )
    return (
        train_gnn_eknn.attach_condition_features(
            fit_graphs,
            fit_conditions,
            component_identity_features=fit_component_ids,
        ),
        train_gnn_eknn.attach_condition_features(
            query_graphs,
            query_conditions,
            component_identity_features=query_component_ids,
        ),
        info,
        component_cardinalities,
        component_info,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not args.group_val:
        raise ValueError("This strict-reaction-key hybrid requires --group-val")
    RDLogger.DisableLog("rdApp.*")
    ds = get_dataset(args.dataset)
    if not ds.condition_cols:
        raise ValueError(f"Dataset '{args.dataset}' has no registered numeric condition columns")
    component_identity_roles = _parse_component_identity_roles(args, ds)

    device = utils.get_device(args.device)
    utils.set_seed(args.seed, deterministic=args.deterministic)
    args.use_condition_features = True
    args.condition_fusion_mode = "gated_add"
    args.include_categorical_conditions = False
    args.condition_categorical_embeddings = False
    args.condition_token_layers = 1
    args.random_split = False
    args.component_readout = False
    args.mpnn = False
    args.gat = False
    args.graph_transformer = False
    args.edge_transformer = False
    args.graphormer = False
    args.deep_gps = False
    args.light_gps = False
    args.attn_pool = False
    args.jk = False
    args.virtual_node = False
    args.early_stop_patience = 0
    args.min_epochs = 0

    cache_root = (
        args.graph_cache_dir
        if args.graph_cache_dir is not None
        else args.out_dir / ".graph_cache"
    )
    cache_dir = str(cache_root) if args.cache_graphs else None
    train_graphs, train_targets, test_graphs, test_targets, _ = \
        train_gnn_eknn.load_and_build_graphs(ds, args, cache_dir)
    train_graphs, train_targets = train_gnn_eknn.filter_valid(train_graphs, train_targets)
    test_graphs, test_targets = train_gnn_eknn.filter_valid(test_graphs, test_targets)
    source_train = pd.read_csv(ds.train_csv)
    source_test = pd.read_csv(ds.test_csv)
    fit_idx, val_idx, split_audit = train_gnn_eknn.make_selection_split(
        train_graphs, ds, args, args.seed
    )
    fit_graphs_raw = [train_graphs[index] for index in fit_idx]
    val_graphs_raw = [train_graphs[index] for index in val_idx]
    fit_targets = train_targets[fit_idx]
    val_targets = train_targets[val_idx]
    fit_frame = train_gnn_eknn.graph_rows_to_frame(fit_graphs_raw, source_train)
    val_frame = train_gnn_eknn.graph_rows_to_frame(val_graphs_raw, source_train)
    (fit_graphs,
     val_graphs,
     selection_condition_info,
     selection_component_cardinalities,
     selection_component_info) = _conditioned_graphs_for_fold(
         fit_graphs_raw,
         val_graphs_raw,
         fit_frame,
         val_frame,
         ds,
         use_component_identity_embeddings=args.component_identity_embeddings,
         component_identity_roles=component_identity_roles,
    )
    condition_dim = int(fit_graphs[0].condition_features.shape[1])
    fit_targets_model_units = catboost_training_target(
        fit_targets, fit_frame, args.gnn_target_mode
    )
    selection_std = train_gnn_eknn.Standardizer(
        float(np.nanmean(fit_targets_model_units)),
        float(np.nanstd(fit_targets_model_units)),
    )

    t0 = time.time()
    utils.set_seed(args.seed, deterministic=args.deterministic)
    selection_model = train_gnn_eknn.build_model(
        args,
        condition_dim=condition_dim,
        component_identity_cardinalities=selection_component_cardinalities,
    )
    selection_train_info = train_gnn_eknn.train_gnn_model(
        selection_model,
        fit_graphs,
        fit_targets_model_units,
        selection_std,
        device,
        args,
        "selection",
    )
    selected_eknn, eknn_grid = select_gnn_eknn_for_ddg(
        selection_model,
        fit_graphs,
        val_graphs,
        fit_targets_model_units,
        val_targets,
        val_frame,
        device,
        selection_std,
        target_mode=args.gnn_target_mode,
    )
    (gnn_static_val, gnn_eknn_val, fit_embedding_norm,
     val_embedding, _) = gnn_predictions_with_eknn(
        selection_model,
        fit_graphs,
        val_graphs,
        fit_targets_model_units,
        selection_std,
        device,
        k=selected_eknn["k"],
        temperature=selected_eknn["T"],
        query_frame=val_frame,
        target_mode=args.gnn_target_mode,
    )
    # CatBoost should see the unnormalised learned embedding, not the
    # retrieval-normalised version, so encode the fitting rows once more.
    fit_embedding, _ = train_gnn_eknn.encode_reactions(selection_model, fit_graphs, device)
    selection = {
        "n_train": int(len(fit_graphs)),
        "n_val": int(len(val_graphs)),
        "split": split_audit,
        "condition_preprocessing": selection_condition_info,
        "component_identity_preprocessing": selection_component_info,
        "gnn_train": selection_train_info,
        "gnn_static": score(val_targets, gnn_static_val),
        "gnn_eknn": {**score(val_targets, gnn_eknn_val), **selected_eknn},
        "eknn_grid": eknn_grid,
    }
    catboost_selection, _, catboost_val = select_catboost_and_blend(
        args,
        ds,
        fit_frame,
        val_frame,
        fit_targets,
        val_targets,
        fit_embedding,
        val_embedding,
        gnn_eknn_val,
    )
    selection["catboost"] = catboost_selection
    alpha = float(catboost_selection["selected_blend_weight"])
    hybrid_val = alpha * gnn_eknn_val + (1.0 - alpha) * catboost_val
    selection["hybrid_eknn"] = score(val_targets, hybrid_val)

    output: dict[str, Any] = {
        "dataset": args.dataset,
        "model": "gnn_catboost_condition_hybrid_eknn",
        "protocol": "two_stage_group_val_select_full_retrain",
        "test_evaluated_once": not args.validation_only,
        "seed": int(args.seed),
        "device": str(device),
        "deterministic": bool(args.deterministic),
        "graph_cache": {
            "directory": str(cache_root) if args.cache_graphs else None,
            "file": (
                str(cache_root / f"{args.dataset}_group_pe{args.pe_dim}_v5_graphs.pkl")
                if args.cache_graphs else None
            ),
            "sha256": (
                _sha256(cache_root / f"{args.dataset}_group_pe{args.pe_dim}_v5_graphs.pkl")
                if args.cache_graphs else None
            ),
        },
        "gnn_config": {
            "backbone": args.backbone,
            "d_model": int(args.d_model),
            "n_layers": int(args.n_layers),
            "n_heads": int(args.n_heads),
            "ff_dim": int(args.ff_dim),
            "dropout": float(args.dropout),
            "condition_fusion_mode": "gated_add",
            "condition_encoding": "fold-standardised numeric values plus missingness indicators",
            "condition_node_film": bool(args.condition_node_film),
            "component_identity_embeddings": bool(args.component_identity_embeddings),
            "component_identity_dim": int(args.component_identity_dim),
            "component_identity_roles": component_identity_roles,
        },
        "catboost_config": {
            "iterations": int(args.catboost_iterations),
            "depth": int(args.catboost_depth),
            "learning_rate": float(args.catboost_lr),
            "l2_leaf_reg": float(args.catboost_l2),
            "target_mode": args.catboost_target_mode,
            "raw_categorical_condition_encoding": True,
            "component_identity_categories": bool(args.catboost_component_identities),
            "component_pair_categories": bool(args.catboost_component_pairs),
        },
        "selection": selection,
    }

    output_dir = args.out_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_path = output_dir / f"hybrid_validation_s{args.seed}.csv"
    pd.DataFrame({
        "source_row": val_frame.get("source_row", pd.Series(np.arange(len(val_frame)))),
        "target": val_targets,
        "gnn_static": gnn_static_val,
        "gnn_eknn": gnn_eknn_val,
        "catboost": catboost_val,
        "hybrid_eknn": hybrid_val,
    }).to_csv(validation_path, index=False)
    output["selection"]["validation_predictions_csv"] = str(validation_path)

    if args.validation_only:
        output["final"] = None
        output["test"] = None
    else:
        del selection_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_train_frame = train_gnn_eknn.graph_rows_to_frame(train_graphs, source_train)
        final_test_frame = train_gnn_eknn.graph_rows_to_frame(test_graphs, source_test)
        (final_train_graphs,
         final_test_graphs,
         final_condition_info,
         final_component_cardinalities,
         final_component_info) = _conditioned_graphs_for_fold(
             train_graphs,
             test_graphs,
             final_train_frame,
             final_test_frame,
             ds,
             use_component_identity_embeddings=args.component_identity_embeddings,
             component_identity_roles=component_identity_roles,
         )
        final_std = train_gnn_eknn.Standardizer(
            float(np.nanmean(train_targets)), float(np.nanstd(train_targets))
        )
        utils.set_seed(args.seed, deterministic=args.deterministic)
        final_model = train_gnn_eknn.build_model(
            args,
            condition_dim=condition_dim,
            component_identity_cardinalities=final_component_cardinalities,
        )
        final_train_info = train_gnn_eknn.train_gnn_model(
            final_model, final_train_graphs, train_targets, final_std, device, args, "final"
        )
        (gnn_static_test, gnn_eknn_test, _, test_embedding,
         _) = gnn_predictions_with_eknn(
            final_model,
            final_train_graphs,
            final_test_graphs,
            train_targets,
            final_std,
            device,
            k=selected_eknn["k"],
            temperature=selected_eknn["T"],
        )
        final_fit_embedding, _ = train_gnn_eknn.encode_reactions(
            final_model, final_train_graphs, device
        )
        selected_representation = catboost_selection["selected_representation"]
        catboost_kwargs = {}
        if selected_representation == "gnn_embedding_morgan_conditions":
            catboost_kwargs = {
                "fit_embedding": final_fit_embedding,
                "query_embedding": test_embedding,
            }
        x_final_train, x_test, categorical_columns, final_feature_info = \
            build_catboost_features(
                final_train_frame,
                final_test_frame,
                ds,
                fp_bits=args.fp_bits,
                fp_radius=args.fp_radius,
                include_component_identities=args.catboost_component_identities,
                include_component_pairs=args.catboost_component_pairs,
                **catboost_kwargs,
        )
        final_catboost = make_catboost(args)
        final_catboost.fit(
            x_final_train,
            catboost_training_target(
                train_targets, final_train_frame, args.catboost_target_mode
            ),
            cat_features=categorical_columns,
        )
        catboost_test = catboost_prediction_to_ddg(
            final_catboost.predict(x_test), final_test_frame, args.catboost_target_mode
        )
        hybrid_test = alpha * gnn_eknn_test + (1.0 - alpha) * catboost_test
        output["final"] = {
            "n_train": int(len(train_targets)),
            "condition_preprocessing": final_condition_info,
            "component_identity_preprocessing": final_component_info,
            "gnn_train": final_train_info,
            "catboost_feature_info": final_feature_info,
            "selected_eknn": selected_eknn,
            "selected_blend_weight": alpha,
        }
        output["test"] = {
            "n_test": int(len(test_targets)),
            "gnn_static": score(test_targets, gnn_static_test),
            "gnn_eknn": score(test_targets, gnn_eknn_test),
            "catboost": score(test_targets, catboost_test),
            "hybrid_eknn": score(test_targets, hybrid_test),
        }
        test_path = output_dir / f"hybrid_test_s{args.seed}.csv"
        pd.DataFrame({
            "source_row": final_test_frame.get("source_row", pd.Series(np.arange(len(final_test_frame)))),
            "target": test_targets,
            "gnn_static": gnn_static_test,
            "gnn_eknn": gnn_eknn_test,
            "catboost": catboost_test,
            "hybrid_eknn": hybrid_test,
        }).to_csv(test_path, index=False)
        output["test_predictions_csv"] = str(test_path)

    json_path = output_dir / f"gnn_catboost_hybrid_{args.dataset}_s{args.seed}.json"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    elapsed = time.time() - t0
    if args.validation_only:
        print(
            f"Validation-only: GNN-eKNN R2={selection['gnn_eknn']['r2']:.4f}; "
            f"CatBoost R2={catboost_selection['selected_metrics']['r2']:.4f}; "
            f"hybrid(alpha={alpha:.2f}) R2={selection['hybrid_eknn']['r2']:.4f}"
        )
    else:
        print(
            f"Test: GNN-eKNN R2={output['test']['gnn_eknn']['r2']:.4f}; "
            f"CatBoost R2={output['test']['catboost']['r2']:.4f}; "
            f"hybrid-eKNN R2={output['test']['hybrid_eknn']['r2']:.4f}"
        )
    print(f"Wrote {json_path} | elapsed={elapsed:.1f}s")
    return output


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

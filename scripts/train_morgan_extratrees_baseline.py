#!/usr/bin/env python3
"""Train a non-neural Morgan-fingerprint ExtraTrees reference baseline.

This script is deliberately separate from the GNN pipeline: the pure GNN and
the GNN+condition-fusion model remain reported as their own models.  It gives
the Michael benchmark a strong, transparent classical reference while using
the same group-disjoint inner-validation discipline as the GNN experiment.

For each candidate representation (Morgan only; Morgan + prediction-time
conditions), the tree leaf size is selected on inner validation.  The selected
setting is then fit once on the complete outer training split and evaluated
once on the untouched outer test split.

Example:
  python scripts/train_morgan_extratrees_baseline.py \
    --dataset michael_ddg_random --group-val
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eknn_reaction.data import get_dataset
from eknn_reaction.metrics import mae, r2

import importlib.util

_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_gnn_eknn", ROOT / "scripts" / "train_gnn_eknn.py"
)
train_gnn_eknn = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(train_gnn_eknn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/michael_condition_tuning"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--group-val", action="store_true",
                        help="Use the registered dataset group column for the inner validation split")
    parser.add_argument("--n-bits", type=int, default=256)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--min-samples-leaf", type=str, default="1,2,4,8",
                        help="Comma-separated leaf-size candidates selected on validation")
    parser.add_argument("--include-categorical-conditions", action="store_true",
                        help="Include registered categorical condition columns in addition to numerical conditions")
    parser.add_argument("--validation-only", action="store_true",
                        help="Select on inner validation and write the leaderboard without fitting/evaluating test")
    return parser.parse_args()


def morgan_component_fingerprint(smiles: object, generator, n_bits: int) -> np.ndarray:
    """Return one binary Morgan vector, using zeros for an invalid optional input."""
    mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    bit_vector = generator.GetFingerprint(mol)
    output = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, output)
    return output.astype(np.float32)


def reaction_fingerprints(frame: pd.DataFrame, smiles_cols: list[str],
                          radius: int, n_bits: int) -> np.ndarray:
    """Concatenate a role-preserving fingerprint for every input component."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    rows = []
    for _, row in frame.iterrows():
        rows.append(np.concatenate([
            morgan_component_fingerprint(row[column], generator, n_bits)
            for column in smiles_cols
        ]))
    return np.vstack(rows).astype(np.float32)


def validation_split(frame: pd.DataFrame, ds, val_frac: float, seed: int,
                     group_val: bool) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    n_rows = len(frame)
    if group_val:
        if not ds.group_col:
            raise ValueError(f"Dataset '{ds.name}' has no group_col")
        groups = frame[ds.group_col].astype(str).to_numpy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        fit_idx, val_idx = next(splitter.split(np.zeros(n_rows), groups=groups))
        fit_groups = set(groups[fit_idx])
        val_groups = set(groups[val_idx])
        if fit_groups & val_groups:
            raise RuntimeError("Inner validation group leakage")
        audit = {
            "mode": "group_disjoint",
            "group_column": ds.group_col,
            "n_groups_total": int(np.unique(groups).size),
            "n_groups_train": int(len(fit_groups)),
            "n_groups_val": int(len(val_groups)),
            "overlapping_groups": 0,
        }
    else:
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n_rows)
        n_val = max(1, int(n_rows * val_frac))
        val_idx = indices[:n_val]
        fit_idx = indices[n_val:]
        audit = {
            "mode": "row_random",
            "group_column": None,
            "n_groups_total": None,
            "n_groups_train": None,
            "n_groups_val": None,
            "overlapping_groups": None,
        }
    return np.asarray(fit_idx), np.asarray(val_idx), audit


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"r2": float(r2(y_pred, y_true)), "mae": float(mae(y_pred, y_true))}


def make_model(n_estimators: int, min_samples_leaf: int, seed: int) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=1.0,
        random_state=seed,
        n_jobs=-1,
    )


def main() -> int:
    args = parse_args()
    ds = get_dataset(args.dataset)
    train_frame = pd.read_csv(ds.train_csv)
    test_frame = pd.read_csv(ds.test_csv)
    y_train = pd.to_numeric(train_frame[ds.target_col], errors="raise").to_numpy(dtype=np.float32)
    y_test = pd.to_numeric(test_frame[ds.target_col], errors="raise").to_numpy(dtype=np.float32)
    fit_idx, val_idx, split_audit = validation_split(
        train_frame, ds, args.val_frac, args.seed, args.group_val
    )
    leaf_grid = [int(value) for value in args.min_samples_leaf.split(",") if value.strip()]
    if not leaf_grid or any(value < 1 for value in leaf_grid):
        raise ValueError("--min-samples-leaf must contain positive integers")

    print(
        f"{args.dataset}: train={len(train_frame)}, valid={len(val_idx)}, test={len(test_frame)} | "
        f"inner split={split_audit['mode']}"
    )
    print("Building Morgan fingerprints on CPU...")
    fp_train = reaction_fingerprints(train_frame, ds.smiles_cols, args.radius, args.n_bits)
    fp_test = None
    if not args.validation_only:
        fp_test = reaction_fingerprints(test_frame, ds.smiles_cols, args.radius, args.n_bits)

    # Fit preprocessing only on fit rows for representation selection.  This
    # matters even for trees because median imputation must not inspect valid.
    representations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        "morgan_only": (fp_train[fit_idx], fp_train[val_idx], fp_test),
    }
    if ds.condition_cols:
        fit_conditions, val_conditions, condition_selection_info = \
            train_gnn_eknn.prepare_condition_features(
                train_frame.iloc[fit_idx].reset_index(drop=True),
                train_frame.iloc[val_idx].reset_index(drop=True),
                ds,
                include_categorical=args.include_categorical_conditions,
            )
        # Test conditions are not needed for model selection; a placeholder
        # keeps the representation bookkeeping uniform and is replaced after
        # the selected feature family is known.
        representations["morgan_plus_conditions"] = (
            np.concatenate([fp_train[fit_idx], fit_conditions], axis=1),
            np.concatenate([fp_train[val_idx], val_conditions], axis=1),
            None,
        )
    else:
        condition_selection_info = None

    leaderboard: list[dict[str, object]] = []
    best: tuple[float, str, int] | None = None
    for representation_name, (x_fit, x_val, _) in representations.items():
        for leaf_size in leaf_grid:
            model = make_model(args.n_estimators, leaf_size, args.seed)
            model.fit(x_fit, y_train[fit_idx])
            metrics = score(y_train[val_idx], model.predict(x_val))
            row = {
                "representation": representation_name,
                "min_samples_leaf": leaf_size,
                "validation_r2": metrics["r2"],
                "validation_mae": metrics["mae"],
            }
            leaderboard.append(row)
            print(
                f"valid {representation_name:>22s} leaf={leaf_size}: "
                f"R2={metrics['r2']:.4f} MAE={metrics['mae']:.4f}"
            )
            candidate = (metrics["r2"], representation_name, leaf_size)
            if best is None or candidate[0] > best[0]:
                best = candidate

    assert best is not None
    _, selected_representation, selected_leaf = best
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "dataset": args.dataset,
        "model": "morgan_extratrees",
        "protocol": "inner_validation_select_full_train_test_once",
        "test_evaluated_once": not args.validation_only,
        "seed": args.seed,
        "fingerprint": {
            "radius": args.radius,
            "n_bits_per_component": args.n_bits,
            "n_components": len(ds.smiles_cols),
        },
        "include_categorical_conditions": bool(args.include_categorical_conditions),
        "selection_split": split_audit,
        "selection": {
            "n_train": int(len(fit_idx)),
            "n_val": int(len(val_idx)),
            "candidates": leaderboard,
            "selected_representation": selected_representation,
            "selected_min_samples_leaf": selected_leaf,
            "condition_preprocessing": condition_selection_info,
        },
    }
    if args.validation_only:
        output["final"] = None
        output["test"] = None
    else:
        if selected_representation == "morgan_only":
            x_train_final, x_test_final = fp_train, fp_test
            final_condition_info = None
        else:
            train_conditions, test_conditions, final_condition_info = \
                train_gnn_eknn.prepare_condition_features(
                    train_frame, test_frame, ds,
                    include_categorical=args.include_categorical_conditions,
                )
            x_train_final = np.concatenate([fp_train, train_conditions], axis=1)
            x_test_final = np.concatenate([fp_test, test_conditions], axis=1)
        # One final fit/evaluation only after validation has selected all choices.
        final_model = make_model(args.n_estimators, selected_leaf, args.seed)
        final_model.fit(x_train_final, y_train)
        test_metrics = score(y_test, final_model.predict(x_test_final))
        output["final"] = {
            "n_train": int(len(train_frame)),
            "condition_preprocessing": final_condition_info,
        }
        output["test"] = test_metrics
    json_path = args.out_dir / f"morgan_extratrees_{args.dataset}_s{args.seed}.json"
    csv_path = args.out_dir / f"morgan_extratrees_{args.dataset}_s{args.seed}_validation.csv"
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(leaderboard).sort_values("validation_r2", ascending=False).to_csv(csv_path, index=False)
    if args.validation_only:
        print(f"Selected {selected_representation}, leaf={selected_leaf}; outer test was not evaluated")
    else:
        print(
            f"Selected {selected_representation}, leaf={selected_leaf}; "
            f"test R2={test_metrics['r2']:.4f}, MAE={test_metrics['mae']:.4f}"
        )
    print(f"Wrote {json_path}\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

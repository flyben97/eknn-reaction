#!/usr/bin/env python3
"""Strict reaction-key extrapolation for classical molecular ML models.

All models receive the same role-preserving Morgan fingerprints and the same
fold-fitted reaction-condition vector.  Hyperparameters are selected by
group-disjoint cross-validation on the immutable outer training split.  Only
after selection is each model refit on all outer-training rows and evaluated
once on the untouched, reaction-key-disjoint outer test split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, ParameterSampler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eknn_reaction.data import get_dataset

_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "train_gnn_eknn", ROOT / "scripts" / "train_gnn_eknn.py"
)
train_gnn_eknn = importlib.util.module_from_spec(_TRAIN_SPEC)
assert _TRAIN_SPEC.loader is not None
_TRAIN_SPEC.loader.exec_module(train_gnn_eknn)


MODEL_NAMES = ("rf", "xgb", "catboost", "lgbm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="michael_ddg_random")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES,
                        default=list(MODEL_NAMES))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/michael_classical_strict"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--catboost-task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--lgbm-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--validation-only", action="store_true",
                        help="Run CV selection without constructing outer-test features")
    return parser.parse_args()


def morgan_component_fingerprint(
    value: object, generator, n_bits: int
) -> np.ndarray:
    molecule = Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
    if molecule is None:
        return np.zeros(n_bits, dtype=np.float32)
    fingerprint = generator.GetFingerprint(molecule)
    output = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, output)
    return output.astype(np.float32)


def reaction_fingerprints(
    frame: pd.DataFrame,
    smiles_columns: list[str],
    radius: int,
    n_bits: int,
) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )
    output = np.empty(
        (len(frame), len(smiles_columns) * n_bits), dtype=np.float32
    )
    for row_index, (_, row) in enumerate(frame.iterrows()):
        output[row_index] = np.concatenate(
            [
                morgan_component_fingerprint(
                    row[column], generator, n_bits
                )
                for column in smiles_columns
            ]
        )
    return output


def parameter_space(model_name: str) -> dict[str, list[Any]]:
    if model_name == "rf":
        return {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 12, 20, 32],
            "max_features": ["sqrt", 0.05, 0.1, 0.2, 0.3],
            "min_samples_leaf": [1, 2, 4, 8],
            "min_samples_split": [2, 5, 10],
            "bootstrap": [False, True],
        }
    if model_name == "xgb":
        return {
            "n_estimators": [300, 600, 900],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.02, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.2, 0.4, 0.7],
            "min_child_weight": [1, 3, 7],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [1.0, 5.0, 10.0],
            "gamma": [0.0, 0.05, 0.2],
        }
    if model_name == "catboost":
        return {
            "iterations": [300, 600, 900],
            "depth": [4, 6, 8, 10],
            "learning_rate": [0.02, 0.05, 0.1],
            "l2_leaf_reg": [3.0, 10.0, 30.0],
            "random_strength": [0.1, 1.0, 3.0],
            "bagging_temperature": [0.0, 0.5, 1.0, 3.0],
        }
    if model_name == "lgbm":
        return {
            "n_estimators": [300, 600, 900],
            "learning_rate": [0.02, 0.05, 0.1],
            "num_leaves": [15, 31, 63, 127],
            "max_depth": [-1, 6, 10, 16],
            "min_child_samples": [5, 10, 20, 40],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.2, 0.4, 0.7],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.0, 1.0, 5.0],
        }
    raise ValueError(f"Unknown model {model_name!r}")


def make_model(
    model_name: str,
    parameters: dict[str, Any],
    args: argparse.Namespace,
):
    common_seed = int(args.seed)
    if model_name == "rf":
        return RandomForestRegressor(
            **parameters,
            random_state=common_seed,
            n_jobs=-1,
        )
    if model_name == "xgb":
        return XGBRegressor(
            **parameters,
            objective="reg:squarederror",
            tree_method="hist",
            device=args.xgb_device,
            random_state=common_seed,
            n_jobs=-1,
            verbosity=0,
        )
    if model_name == "catboost":
        catboost_options = dict(
            **parameters,
            loss_function="RMSE",
            random_seed=common_seed,
            task_type=args.catboost_task_type,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        if args.catboost_task_type == "GPU":
            catboost_options.update(devices="0", gpu_ram_part=0.85)
        return CatBoostRegressor(**catboost_options)
    if model_name == "lgbm":
        return LGBMRegressor(
            **parameters,
            objective="regression",
            random_state=common_seed,
            n_jobs=-1,
            verbosity=-1,
            device_type=args.lgbm_device,
        )
    raise ValueError(f"Unknown model {model_name!r}")


def metric_dict(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(targets, predictions)),
        "mae": float(mean_absolute_error(targets, predictions)),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def outer_split_audit(
    train_frame: pd.DataFrame, test_frame: pd.DataFrame, group_column: str
) -> dict[str, Any]:
    train_groups = set(train_frame[group_column].astype(str))
    test_groups = set(test_frame[group_column].astype(str))
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(
            f"Outer strict split leakage: {len(overlap)} {group_column} values overlap"
        )
    audit: dict[str, Any] = {
        "mode": "reaction_key_group_disjoint",
        "group_column": group_column,
        "n_train_groups": len(train_groups),
        "n_test_groups": len(test_groups),
        "overlapping_groups": 0,
    }
    if "reaction_key" in train_frame.columns and "reaction_key" in test_frame.columns:
        train_keys = set(train_frame["reaction_key"].astype(str))
        test_keys = set(test_frame["reaction_key"].astype(str))
        key_overlap = train_keys & test_keys
        if key_overlap:
            raise RuntimeError(
                f"Outer reaction-key leakage: {len(key_overlap)} keys overlap"
            )
        audit.update({
            "reaction_key_column": "reaction_key",
            "n_train_reaction_keys": len(train_keys),
            "n_test_reaction_keys": len(test_keys),
            "overlapping_reaction_keys": 0,
        })
    return audit


def build_cv_features(
    train_frame: pd.DataFrame,
    fingerprints: np.ndarray,
    ds,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    folds = []
    audits = []
    groups = train_frame[ds.group_col].astype(str).to_numpy()
    for fold_index, (fit_indices, validation_indices) in enumerate(cv_splits):
        fit_frame = train_frame.iloc[fit_indices].reset_index(drop=True)
        validation_frame = train_frame.iloc[validation_indices].reset_index(drop=True)
        if ds.condition_cols or ds.condition_categorical_cols:
            fit_conditions, validation_conditions, condition_info = (
                train_gnn_eknn.prepare_condition_features(
                    fit_frame,
                    validation_frame,
                    ds,
                    include_categorical=True,
                )
            )
        else:
            fit_conditions = np.empty((len(fit_frame), 0), dtype=np.float32)
            validation_conditions = np.empty(
                (len(validation_frame), 0), dtype=np.float32
            )
            condition_info = {"enabled": False, "feature_dim": 0}
        fit_groups = set(groups[fit_indices])
        validation_groups = set(groups[validation_indices])
        overlap = fit_groups & validation_groups
        if overlap:
            raise RuntimeError(
                f"CV fold {fold_index} leaks {len(overlap)} groups"
            )
        folds.append({
            "fit_indices": fit_indices,
            "validation_indices": validation_indices,
            "x_fit": np.concatenate(
                [fingerprints[fit_indices], fit_conditions], axis=1
            ),
            "x_validation": np.concatenate(
                [fingerprints[validation_indices], validation_conditions], axis=1
            ),
        })
        audits.append({
            "fold": fold_index,
            "n_fit": int(len(fit_indices)),
            "n_validation": int(len(validation_indices)),
            "n_fit_groups": int(len(fit_groups)),
            "n_validation_groups": int(len(validation_groups)),
            "overlapping_groups": 0,
            "condition_feature_dim": int(fit_conditions.shape[1]),
            "condition_preprocessing": condition_info,
        })
    return folds, audits


def search_one_model(
    model_name: str,
    folds: list[dict[str, Any]],
    targets: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = list(
        ParameterSampler(
            parameter_space(model_name),
            n_iter=args.n_trials,
            random_state=args.seed,
        )
    )
    leaderboard = []
    best_row = None
    print(
        f"\n[{model_name}] {len(candidates)} candidates x "
        f"{len(folds)} group folds",
        flush=True,
    )
    for candidate_index, parameters in enumerate(candidates, start=1):
        fold_metrics = []
        started = time.time()
        for fold in folds:
            model = make_model(model_name, parameters, args)
            fit_indices = fold["fit_indices"]
            validation_indices = fold["validation_indices"]
            model.fit(fold["x_fit"], targets[fit_indices])
            predictions = model.predict(fold["x_validation"])
            fold_metrics.append(
                metric_dict(targets[validation_indices], predictions)
            )
        r2_values = np.asarray(
            [metrics["r2"] for metrics in fold_metrics], dtype=float
        )
        mae_values = np.asarray(
            [metrics["mae"] for metrics in fold_metrics], dtype=float
        )
        row = {
            "candidate": candidate_index,
            "parameters": jsonable(parameters),
            "fold_metrics": fold_metrics,
            "mean_cv_r2": float(r2_values.mean()),
            "sd_cv_r2": float(r2_values.std(ddof=1)),
            "mean_cv_mae": float(mae_values.mean()),
            "sd_cv_mae": float(mae_values.std(ddof=1)),
            "elapsed_sec": float(time.time() - started),
        }
        leaderboard.append(row)
        if (
            best_row is None
            or row["mean_cv_r2"] > best_row["mean_cv_r2"]
        ):
            best_row = row
        print(
            f"[{model_name}] {candidate_index:02d}/{len(candidates)} "
            f"R2={row['mean_cv_r2']:.4f}±{row['sd_cv_r2']:.4f} "
            f"MAE={row['mean_cv_mae']:.4f} "
            f"({row['elapsed_sec']:.1f}s)",
            flush=True,
        )
    assert best_row is not None
    print(
        f"[{model_name}] selected CV R2={best_row['mean_cv_r2']:.4f} "
        f"params={best_row['parameters']}",
        flush=True,
    )
    return best_row, leaderboard


def main() -> int:
    args = parse_args()
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    if args.n_trials < 1:
        raise ValueError("--n-trials must be positive")

    ds = get_dataset(args.dataset)
    if not ds.group_col:
        raise ValueError(
            f"Dataset {args.dataset!r} has no group column for strict CV"
        )
    train_frame = pd.read_csv(ds.train_csv)
    test_frame = pd.read_csv(ds.test_csv)
    y_train = pd.to_numeric(
        train_frame[ds.target_col], errors="raise"
    ).to_numpy(dtype=np.float32)
    y_test = pd.to_numeric(
        test_frame[ds.target_col], errors="raise"
    ).to_numpy(dtype=np.float32)
    outer_audit = outer_split_audit(
        train_frame, test_frame, ds.group_col
    )
    print(
        f"{args.dataset}: train={len(train_frame)} test={len(test_frame)} | "
        f"outer reaction-key overlap=0",
        flush=True,
    )
    print(
        f"Building {len(ds.smiles_cols)} x {args.n_bits}-bit "
        "role-preserving Morgan features...",
        flush=True,
    )
    fp_train = reaction_fingerprints(
        train_frame, ds.smiles_cols, args.radius, args.n_bits
    )
    fp_test = None
    if not args.validation_only:
        fp_test = reaction_fingerprints(
            test_frame, ds.smiles_cols, args.radius, args.n_bits
        )

    groups = train_frame[ds.group_col].astype(str).to_numpy()
    splitter = GroupKFold(
        n_splits=args.cv_folds, shuffle=True, random_state=args.seed
    )
    cv_splits = list(splitter.split(fp_train, y_train, groups=groups))
    folds, fold_audits = build_cv_features(
        train_frame, fp_train, ds, cv_splits
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common_output = {
        "dataset": args.dataset,
        "protocol": (
            f"{args.cv_folds}fold_group_disjoint_cv_select_"
            "full_outer_train_refit_test_once"
        ),
        "seed": int(args.seed),
        "test_evaluated_once": not args.validation_only,
        "fingerprint": {
            "radius": int(args.radius),
            "n_bits_per_component": int(args.n_bits),
            "n_components": int(len(ds.smiles_cols)),
            "role_preserving_concatenation": True,
        },
        "conditions": {
            "include_numeric": True,
            "include_categorical": True,
            "fold_fitted": True,
        },
        "outer_split_audit": outer_audit,
        "cv_fold_audits": fold_audits,
        "n_trials_per_model": int(args.n_trials),
    }

    selection_results = {}
    final_results = {}
    for model_name in args.models:
        selected, leaderboard = search_one_model(
            model_name, folds, y_train, args
        )
        selection_results[model_name] = {
            "selected": selected,
            "leaderboard": leaderboard,
        }

    final_condition_info = None
    x_train_final = None
    x_test_final = None
    if not args.validation_only:
        assert fp_test is not None
        if ds.condition_cols or ds.condition_categorical_cols:
            train_conditions, test_conditions, final_condition_info = (
                train_gnn_eknn.prepare_condition_features(
                    train_frame,
                    test_frame,
                    ds,
                    include_categorical=True,
                )
            )
        else:
            train_conditions = np.empty(
                (len(train_frame), 0), dtype=np.float32
            )
            test_conditions = np.empty(
                (len(test_frame), 0), dtype=np.float32
            )
            final_condition_info = {"enabled": False, "feature_dim": 0}
        x_train_final = np.concatenate(
            [fp_train, train_conditions], axis=1
        )
        x_test_final = np.concatenate(
            [fp_test, test_conditions], axis=1
        )

    summary_rows = []
    for model_name in args.models:
        selected = selection_results[model_name]["selected"]
        model_output = {
            **common_output,
            "model": model_name,
            "backend": {
                "rf": "cpu",
                "xgb": args.xgb_device,
                "catboost": args.catboost_task_type.lower(),
                "lgbm": args.lgbm_device,
            }[model_name],
            "selection": selection_results[model_name],
            "final": None,
            "test": None,
        }
        test_metrics = None
        if not args.validation_only:
            assert x_train_final is not None and x_test_final is not None
            final_model = make_model(
                model_name, selected["parameters"], args
            )
            started = time.time()
            final_model.fit(x_train_final, y_train)
            test_predictions = final_model.predict(x_test_final)
            test_metrics = metric_dict(y_test, test_predictions)
            model_output["final"] = {
                "n_train": int(len(train_frame)),
                "feature_dim": int(x_train_final.shape[1]),
                "condition_preprocessing": final_condition_info,
                "fit_elapsed_sec": float(time.time() - started),
            }
            model_output["test"] = test_metrics
            final_results[model_name] = {
                "final": model_output["final"],
                "test": test_metrics,
            }
            print(
                f"[{model_name}] OUTER STRICT TEST "
                f"R2={test_metrics['r2']:.4f} "
                f"MAE={test_metrics['mae']:.4f}",
                flush=True,
            )
        output_path = (
            args.out_dir
            / f"{model_name}_{args.dataset}_s{args.seed}.json"
        )
        output_path.write_text(
            json.dumps(jsonable(model_output), indent=2) + "\n",
            encoding="utf-8",
        )
        summary_rows.append({
            "model": model_name,
            "backend": model_output["backend"],
            "cv_r2_mean": selected["mean_cv_r2"],
            "cv_r2_sd": selected["sd_cv_r2"],
            "cv_mae_mean": selected["mean_cv_mae"],
            "selected_parameters": json.dumps(selected["parameters"], sort_keys=True),
            "test_r2": None if test_metrics is None else test_metrics["r2"],
            "test_mae": None if test_metrics is None else test_metrics["mae"],
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        "cv_r2_mean", ascending=False
    )
    summary_path = (
        args.out_dir
        / f"classical_strict_summary_{args.dataset}_s{args.seed}.csv"
    )
    summary.to_csv(summary_path, index=False)
    combined_path = (
        args.out_dir
        / f"classical_strict_all_{args.dataset}_s{args.seed}.json"
    )
    combined_path.write_text(
        json.dumps(
            jsonable({
                **common_output,
                "models": selection_results,
                "final_results": final_results,
                "summary": summary_rows,
            }),
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {combined_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Calibrate a HiTEA LightGBM model with residuals from nearby terminal leaves.

Calibration and development queries use separate rows. Locked target labels stay masked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEED = 20260831
CALIBRATION_BUDGETS = (5, 10, 20)
METHODS = (
    "base",
    "target_mean",
    "global_residual",
    "affine_residual",
    "leaf_label_knn",
    "leaf_residual",
    "leaf_residual_permuted_raw",
    "leaf_residual_shrunk",
    "leaf_residual_permuted",
)

CATEGORICAL_FEATURES = (
    "reaction_class",
    "reaction_keyword",
    "reaction_group",
    "reaction_smiles",
    "product_id",
    "solvent_name",
    "catalyst_1_id_1",
    "catalyst_1_id_2",
    "catalyst_1_short_hand",
    "catalyst_2_id_1",
    "catalyst_2_id_2",
    "catalyst_2_short_hand",
    "reactant_1_id",
    "reactant_2_id",
    "reactant_3_id",
    "reagent_1_id",
    "reagent_1_short_hand",
    "reagent_2_id",
    "reagent_2_short_hand",
    "product_smiles",
    "reactant_1_smiles",
    "reactant_2_smiles",
    "reactant_3_smiles",
    "catalyst_1_smiles_1",
    "catalyst_1_smiles_2",
    "catalyst_2_smiles_1",
    "catalyst_2_smiles_2",
)
NUMERIC_FEATURES = (
    "reaction_temperature_c",
    "reaction_time_h",
    "catalyst_1_equiv",
    "catalyst_2_equiv",
    "reactant_1_equiv",
    "reactant_2_equiv",
    "reactant_3_equiv",
    "reagent_1_equiv",
    "reagent_2_equiv",
)
MISSING_FEATURES = tuple(f"{column}_missing" for column in NUMERIC_FEATURES)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fold_safe_features(
    train: pd.DataFrame, query: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_x = pd.DataFrame(index=train.index)
    query_x = pd.DataFrame(index=query.index)
    cardinalities: dict[str, int] = {}
    for column in CATEGORICAL_FEATURES:
        values = train[column].fillna("<MISSING>").astype(str)
        vocabulary = {
            value: index
            for index, value in enumerate(sorted(values.unique()))
        }
        train_x[column] = values.map(vocabulary).astype(np.int32)
        query_x[column] = (
            query[column].fillna("<MISSING>").astype(str).map(vocabulary).fillna(-1)
        ).astype(np.int32)
        cardinalities[column] = int(len(vocabulary))

    medians: dict[str, float] = {}
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(train[column], errors="coerce")
        median = float(values.median()) if values.notna().any() else 0.0
        medians[column] = median
        train_x[column] = values.fillna(median).astype(np.float32)
        query_x[column] = (
            pd.to_numeric(query[column], errors="coerce").fillna(median).astype(np.float32)
        )
    for column in MISSING_FEATURES:
        train_x[column] = pd.to_numeric(train[column], errors="coerce").fillna(1).astype(np.int8)
        query_x[column] = pd.to_numeric(query[column], errors="coerce").fillna(1).astype(np.int8)
    return train_x, query_x, {
        "categorical_cardinalities": cardinalities,
        "numeric_medians": medians,
        "unknown_category_policy": -1,
    }


def fit_base_model(train_x: pd.DataFrame, train_y: np.ndarray, seed: int):
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=160,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        max_depth=-1,
        subsample=0.90,
        colsample_bytree=0.80,
        reg_alpha=0.10,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        train_x,
        train_y,
        categorical_feature=list(CATEGORICAL_FEATURES),
    )
    return model


def leaf_neighbours(
    query_leaf: np.ndarray,
    memory_leaf: np.ndarray,
    *,
    k: int = 5,
    temperature: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(memory_leaf) == 0:
        raise ValueError("Residual memory is empty")
    similarities = (query_leaf[:, None, :] == memory_leaf[None, :, :]).mean(axis=2)
    k_eff = min(int(k), len(memory_leaf))
    top = np.argpartition(-similarities, kth=k_eff - 1, axis=1)[:, :k_eff]
    top_similarity = np.take_along_axis(similarities, top, axis=1)
    order = np.argsort(-top_similarity, axis=1)
    top = np.take_along_axis(top, order, axis=1)
    top_similarity = np.take_along_axis(top_similarity, order, axis=1)
    logits = (top_similarity - top_similarity.max(axis=1, keepdims=True)) / temperature
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return top, top_similarity, weights


def weighted_memory(values: np.ndarray, top: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(values[top] * weights, axis=1)


def affine_residual_prediction(
    memory_base: np.ndarray,
    memory_y: np.ndarray,
    query_base: np.ndarray,
    *,
    ridge: float = 10.0,
) -> np.ndarray:
    residual = memory_y - memory_base
    centre = float(memory_base.mean())
    x = memory_base - centre
    beta = float(np.dot(x, residual - residual.mean()) / (np.dot(x, x) + ridge))
    return query_base + float(residual.mean()) + beta * (query_base - centre)


def metric_record(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    error = pred - y
    absolute = np.abs(error)
    y_varies = len(y) >= 2 and float(np.ptp(y)) > 1e-12
    pred_varies = len(pred) >= 2 and float(np.ptp(pred)) > 1e-12
    if y_varies:
        r2 = float(r2_score(y, pred))
        pearson = float(stats.pearsonr(y, pred).statistic) if pred_varies else np.nan
        spearman = float(stats.spearmanr(y, pred).statistic) if pred_varies else np.nan
    else:
        r2 = pearson = spearman = np.nan
    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": r2,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "bias": float(error.mean()),
        "p90_absolute_error": float(np.quantile(absolute, 0.90)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
    }


def bootstrap_fold_delta(
    metrics: pd.DataFrame,
    candidate: str,
    comparator: str,
    *,
    budget: int,
    n_bootstrap: int = 10000,
) -> dict[str, float]:
    subset = metrics.loc[
        metrics["subgroup"].eq("all")
        & metrics["budget_percent"].eq(budget)
        & metrics["method"].isin([candidate, comparator])
    ]
    wide = subset.pivot(index="fold_id", columns="method", values="mae").dropna()
    delta = (wide[candidate] - wide[comparator]).to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(delta, size=(n_bootstrap, len(delta)), replace=True).mean(axis=1)
    return {
        "n_folds": int(len(delta)),
        "mean_fold_mae_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
        "fraction_folds_improved": float(np.mean(delta < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/hitea_retrieval_shift/v1")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/hitea_target_memory_development_v1")
    )
    parser.add_argument("--only-folds", nargs="*", default=[])
    args = parser.parse_args()
    started = time.time()

    table_path = args.data_root / "development_modeling_table.csv"
    manifest = json.loads((args.data_root / "materialization_manifest.json").read_text())
    table = pd.read_csv(table_path, low_memory=False)
    locked = pd.read_csv(args.data_root / "locked_test_registry.csv")
    locked_ids = set(locked["record_id"].astype(str))
    if int(table["Yield"].isna().sum()) != len(locked_ids):
        raise AssertionError("Development table does not have exactly the locked labels masked")
    if set(table.loc[table["Yield"].isna(), "record_id"].astype(str)) != locked_ids:
        raise AssertionError("Masked labels and locked test registry differ")

    fold_index = pd.read_csv(args.data_root / "fold_index.csv")
    if args.only_folds:
        fold_index = fold_index.loc[fold_index["fold_id"].isin(args.only_folds)]
    if fold_index.empty:
        raise ValueError("No folds selected")

    args.output.mkdir(parents=True, exist_ok=True)
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    fold_settings: list[dict] = []

    for fold_number, fold in enumerate(fold_index.itertuples(index=False), start=1):
        fold_id = str(fold.fold_id)
        assignment = pd.read_csv(
            args.data_root / "folds" / fold_id / "target_assignment.csv"
        )
        target = table.loc[table["record_id"].isin(assignment["record_id"])].merge(
            assignment,
            on=["record_id", "exact_input_group", "reaction_core_group"],
            validate="one_to_one",
        )
        if not target.loc[target["is_target_test"], "Yield"].isna().all():
            raise AssertionError(f"Locked target labels are visible in {fold_id}")
        target_development = target.loc[~target["is_target_test"]].copy()
        if target_development["Yield"].isna().any():
            raise AssertionError(f"Non-test target labels are unexpectedly missing in {fold_id}")
        fixed_query = target_development.loc[
            ~target_development["is_calibration_20"]
        ].copy()
        if fixed_query.empty:
            raise ValueError(f"No fixed development query rows in {fold_id}")

        target_groups = set(target["exact_input_group"])
        source = table.loc[
            table["Yield"].notna()
            & ~table["screen_id"].eq(str(fold.target_screen_id))
            & ~table["exact_input_group"].isin(target_groups)
        ].copy()
        if set(source["exact_input_group"]) & target_groups:
            raise AssertionError(f"Source-target exact input leakage in {fold_id}")

        train_x, target_x, preprocessing = fold_safe_features(source, target_development)
        model_seed = stable_seed(SEED, fold_id)
        model = fit_base_model(train_x, source["Yield"].to_numpy(float), model_seed)
        target_base = model.predict(target_x).astype(float)
        target_leaf = model.predict(target_x, pred_leaf=True).astype(np.int32)
        target_development["base_prediction"] = target_base
        target_development["_target_x_position"] = np.arange(len(target_development))

        query_positions = fixed_query["record_id"].map(
            target_development.set_index("record_id")["_target_x_position"]
        ).to_numpy(int)
        query_base = target_base[query_positions]
        query_leaf = target_leaf[query_positions]
        query_y = fixed_query["Yield"].to_numpy(float)

        fold_settings.append(
            {
                "fold_id": fold_id,
                "reaction_keyword": str(fold.reaction_keyword),
                "target_screen_id": str(fold.target_screen_id),
                "n_source_train": int(len(source)),
                "n_target_development": int(len(target_development)),
                "n_fixed_query": int(len(fixed_query)),
                "n_locked_test_unread": int(target["is_target_test"].sum()),
                "model_seed": int(model_seed),
                "n_features": int(train_x.shape[1]),
                "n_trees": int(target_leaf.shape[1]),
                "preprocessing": preprocessing,
            }
        )

        for budget in CALIBRATION_BUDGETS:
            memory = target_development.loc[
                target_development[f"is_calibration_{budget:02d}"]
            ].copy()
            if memory.empty:
                raise ValueError(f"Empty {budget}% memory in {fold_id}")
            if set(memory["exact_input_group"]) & set(fixed_query["exact_input_group"]):
                raise AssertionError(f"Memory-query exact input leakage in {fold_id}/{budget}")
            memory_positions = memory["_target_x_position"].to_numpy(int)
            memory_base = target_base[memory_positions]
            memory_leaf = target_leaf[memory_positions]
            memory_y = memory["Yield"].to_numpy(float)
            memory_residual = memory_y - memory_base

            top, top_similarity, weights = leaf_neighbours(query_leaf, memory_leaf)
            local_residual = weighted_memory(memory_residual, top, weights)
            local_label = weighted_memory(memory_y, top, weights)
            global_residual = float(memory_residual.mean())
            # Pseudo-count shrinkage toward the target-global residual.  With
            # normalised neighbour weights, 2.0 gives a 1/3 local contribution.
            shrunk_residual = (local_residual + 2.0 * global_residual) / 3.0
            rng = np.random.default_rng(stable_seed(SEED, fold_id, budget, "permute"))
            permuted_residual = rng.permutation(memory_residual)
            permuted_local = weighted_memory(permuted_residual, top, weights)
            permuted_shrunk = (permuted_local + 2.0 * global_residual) / 3.0

            predictions = {
                "base": query_base,
                "target_mean": np.full(len(query_y), float(memory_y.mean())),
                "global_residual": query_base + global_residual,
                "affine_residual": affine_residual_prediction(
                    memory_base, memory_y, query_base
                ),
                "leaf_label_knn": local_label,
                "leaf_residual": query_base + local_residual,
                "leaf_residual_permuted_raw": query_base + permuted_local,
                "leaf_residual_shrunk": query_base + shrunk_residual,
                "leaf_residual_permuted": query_base + permuted_shrunk,
            }
            diagnostic = pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "reaction_keyword": str(fold.reaction_keyword),
                    "target_screen_id": str(fold.target_screen_id),
                    "budget_percent": budget,
                    "record_id": fixed_query["record_id"].to_numpy(),
                    "exact_input_group": fixed_query["exact_input_group"].to_numpy(),
                    "reaction_core_group": fixed_query["reaction_core_group"].to_numpy(),
                    "Yield": query_y,
                    "base_prediction": query_base,
                    "memory_size": int(len(memory)),
                    "nearest_leaf_similarity": top_similarity[:, 0],
                    "mean_top_leaf_similarity": top_similarity.mean(axis=1),
                    "effective_k": int(top.shape[1]),
                }
            )
            for method, values in predictions.items():
                diagnostic[f"prediction_{method}"] = values
                for subgroup, mask in (
                    ("all", np.ones(len(query_y), dtype=bool)),
                    ("zero", query_y == 0.0),
                    ("nonzero", query_y > 0.0),
                ):
                    if not mask.any():
                        continue
                    metric_rows.append(
                        {
                            "fold_id": fold_id,
                            "reaction_keyword": str(fold.reaction_keyword),
                            "target_screen_id": str(fold.target_screen_id),
                            "budget_percent": budget,
                            "method": method,
                            "subgroup": subgroup,
                            **metric_record(query_y[mask], np.asarray(values)[mask]),
                        }
                    )
            prediction_parts.append(diagnostic)
        print(
            f"[{fold_number:02d}/{len(fold_index):02d}] {fold_id}: "
            f"source={len(source)} dev_query={len(fixed_query)}",
            flush=True,
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(args.output / "development_predictions.csv", index=False)
    metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    json_dump(args.output / "fold_settings.json", {"folds": fold_settings})

    summary_rows: list[dict] = []
    for (budget, method), part in metrics.loc[metrics["subgroup"].eq("all")].groupby(
        ["budget_percent", "method"]
    ):
        pred_column = f"prediction_{method}"
        pred_part = predictions.loc[predictions["budget_percent"].eq(budget)]
        pooled = metric_record(
            pred_part["Yield"].to_numpy(float), pred_part[pred_column].to_numpy(float)
        )
        summary_rows.append(
            {
                "scope": "all_folds_pooled",
                "reaction_keyword": "ALL",
                "budget_percent": int(budget),
                "method": method,
                **pooled,
                "macro_fold_mae": float(part["mae"].mean()),
            }
        )
        for family, family_metrics in part.groupby("reaction_keyword"):
            family_predictions = pred_part.loc[pred_part["reaction_keyword"].eq(family)]
            family_pooled = metric_record(
                family_predictions["Yield"].to_numpy(float),
                family_predictions[pred_column].to_numpy(float),
            )
            summary_rows.append(
                {
                    "scope": "family_pooled",
                    "reaction_keyword": family,
                    "budget_percent": int(budget),
                    "method": method,
                    **family_pooled,
                    "macro_fold_mae": float(family_metrics["mae"].mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    comparisons = {
        comparator: bootstrap_fold_delta(
            metrics,
            "leaf_residual",
            comparator,
            budget=20,
        )
        for comparator in (
            "base",
            "global_residual",
            "affine_residual",
            "leaf_residual_permuted_raw",
            "leaf_label_knn",
        )
    }
    pooled20 = summary.loc[
        summary["scope"].eq("all_folds_pooled")
        & summary["budget_percent"].eq(20)
    ].set_index("method")
    local_mae = float(pooled20.loc["leaf_residual", "mae"])
    base_mae = float(pooled20.loc["base", "mae"])
    global_mae = float(pooled20.loc["global_residual", "mae"])
    decision = {
        "status": "diagnostic_only",
        "locked_target_test_scored": False,
        "candidate": "LightGBM leaf-embedding target residual retrieval (unshrunk)",
        "candidate_selection": (
            "Unshrunk residual retrieval selected over pseudocount-2 shrinkage "
            "on the blinded development queries; now frozen for confirmatory work."
        ),
        "budget_percent": 20,
        "pooled_mae": {
            "base": base_mae,
            "global_residual": global_mae,
            "leaf_residual": local_mae,
        },
        "relative_mae_improvement_vs_base": float((base_mae - local_mae) / base_mae),
        "relative_mae_improvement_vs_global_residual": float(
            (global_mae - local_mae) / global_mae
        ),
        "pre_registered_feasibility_gates": {
            "at_least_5pct_vs_base": bool(local_mae <= 0.95 * base_mae),
            "at_least_2pct_vs_global_residual": bool(local_mae <= 0.98 * global_mae),
            "permutation_control_worse": bool(
                local_mae < float(pooled20.loc["leaf_residual_permuted_raw", "mae"])
            ),
            "direct_label_knn_not_better": bool(
                local_mae <= float(pooled20.loc["leaf_label_knn", "mae"])
            ),
        },
        "fold_block_bootstrap": comparisons,
        "interpretation_rule": (
            "This run can justify proceeding to GNN development, but cannot validate "
            "the GNN eKNN claim or unlock the locked target test."
        ),
    }
    json_dump(args.output / "development_decision.json", decision)
    run_manifest = {
        "run_id": "hitea_target_memory_development_v1",
        "status": "diagnostic_only",
        "data_manifest_sha256": manifest["development_modeling_table_sha256"],
        "locked_test_registry_sha256": manifest["locked_test_registry_sha256"],
        "n_locked_test_labels_available": 0,
        "n_folds": int(len(fold_index)),
        "seed": SEED,
        "base_model": "LightGBM L1 regression",
        "retrieval_embedding": "terminal-leaf identity vector",
        "retrieval": {"k": 5, "temperature": 0.05, "global_shrinkage_pseudocount": 2.0},
        "affine_ridge": 10.0,
        "calibration_budgets_percent": list(CALIBRATION_BUDGETS),
        "fixed_development_query_policy": "non-test rows outside the 20% calibration memory",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "lightgbm": lgb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "wall_time_seconds": float(time.time() - started),
    }
    json_dump(args.output / "run_manifest.json", run_manifest)
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

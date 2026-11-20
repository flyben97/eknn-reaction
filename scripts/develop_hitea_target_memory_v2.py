#!/usr/bin/env python3
"""Select a HiTEA correction using group-blocked calibration validation.

Candidate methods combine tree leaves, molecular fingerprints and reaction conditions.
Fit the base models outside the held-out groups and keep locked target labels masked.
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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

RDLogger.DisableLog("rdApp.*")

SEED = 20260831
CALIBRATION_BUDGETS = (5, 10, 20)

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

SMILES_ROLES = (
    "reactant_1_smiles",
    "reactant_2_smiles",
    "reactant_3_smiles",
    "catalyst_1_smiles_1",
    "catalyst_1_smiles_2",
    "catalyst_2_smiles_1",
    "catalyst_2_smiles_2",
    "product_smiles",
)
CONDITION_CATEGORICAL = ("solvent_name", "reaction_class", "reaction_group")
MORGAN_BITS = 1024
MORGAN_RADIUS = 2

K_GRID = (1, 3, 5, 8, 12)
TEMPERATURE_GRID = (0.05, 0.10, 0.25)
SCALE_GRID = (0.5, 1.0)
BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
COMBINATION_WEIGHTS = (
    (0.5, 0.5, 0.0),
    (0.5, 0.0, 0.5),
    (0.0, 0.5, 0.5),
    (0.34, 0.33, 0.33),
    (0.6, 0.2, 0.2),
    (0.2, 0.6, 0.2),
    (0.2, 0.2, 0.6),
)
BASE_VARIANTS = ("base_v1", "base_zi")
# ε-top-set ensemble of retrieval configurations (see SELECTION_RULE).
ENSEMBLE_REL_TOLERANCE = 0.05
ENSEMBLE_MAX_MEMBERS = 25
# Retrieval must beat the best no-retrieval method by this relative margin to
# be used at all (complexity preference; mirrors the program's 2% gate).
FAMILY_GATE_MARGIN = 0.02
DIAGNOSTIC_SPACES = ("leaf", "morgan", "condition")
# Pre-registered frozen selection rule (order resolves ties: earlier wins).
SELECTION_RULE = (
    "Per fold and budget: score every registered candidate (no-retrieval "
    "methods: each base variant, its global residual offset and its affine "
    "calibration; retrieval grid: base in {v1, zi}; similarity in {leaf, "
    "morgan, condition, combined(w)}; k in {1,3,5,8,12}; temperature in "
    "{0.05,0.1,0.25}; residual scale in {0.5,1.0}; residual/label blend in "
    "{0,.25,.5,.75,1}) by group-blocked leave-one-out MAE on the calibration "
    "memory: the held-out row's whole exact_input_group (replicate experiments) "
    "is excluded, mirroring the memory-query group-disjointness rule, with "
    "self-exclusion as fallback. If the best retrieval configuration does not "
    "beat the best no-retrieval method by at least a 2% relative margin "
    "(complexity preference, mirroring the program's 2% promotion gate), fall "
    "back to that method. Otherwise "
    "average the predictions of every retrieval configuration whose LOO MAE is "
    "within 5% of the best (at most 25 members, ordered by score then name) - "
    "the ensemble dampens selector variance instead of trusting a single "
    "argmin. Ties resolve toward the earlier registered method. Development "
    "query labels are never used for selection."
)


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ── Fold-safe feature space ──────────────────────────────────────────────


def fit_feature_space(train: pd.DataFrame) -> dict:
    vocabulary: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL_FEATURES:
        values = train[column].fillna("<MISSING>").astype(str)
        vocabulary[column] = {
            value: index for index, value in enumerate(sorted(values.unique()))
        }
    medians: dict[str, float] = {}
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(train[column], errors="coerce")
        medians[column] = float(values.median()) if values.notna().any() else 0.0
    return {"vocabulary": vocabulary, "medians": medians}


def apply_feature_space(space: dict, frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in CATEGORICAL_FEATURES:
        mapped = (
            frame[column]
            .fillna("<MISSING>")
            .astype(str)
            .map(space["vocabulary"][column])
            .fillna(-1)
        )
        output[column] = mapped.astype(np.int32)
    for column in NUMERIC_FEATURES:
        output[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(space["medians"][column])
            .astype(np.float32)
        )
    for column in MISSING_FEATURES:
        output[column] = (
            pd.to_numeric(frame[column], errors="coerce").fillna(1).astype(np.int8)
        )
    return output


LGBM_PARAMS = dict(
    n_estimators=160,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    max_depth=-1,
    subsample=0.90,
    colsample_bytree=0.80,
    reg_alpha=0.10,
    reg_lambda=2.0,
    n_jobs=8,
    verbosity=-1,
)


def fit_base_model(train_x: pd.DataFrame, train_y: np.ndarray, seed: int):
    model = lgb.LGBMRegressor(objective="regression_l1", random_state=seed, **LGBM_PARAMS)
    model.fit(train_x, train_y, categorical_feature=list(CATEGORICAL_FEATURES))
    return model


def fit_zero_inflated_model(train_x: pd.DataFrame, train_y: np.ndarray, seed: int):
    """P(y=0) classifier plus an L1 regressor fitted on nonzero rows only."""
    is_zero = train_y <= 0.0
    classifier = lgb.LGBMClassifier(objective="binary", random_state=seed, **LGBM_PARAMS)
    classifier.fit(train_x, is_zero.astype(np.int8), categorical_feature=list(CATEGORICAL_FEATURES))
    nonzero = ~is_zero
    if int(nonzero.sum()) < 10:
        regressor = None
    else:
        regressor = lgb.LGBMRegressor(
            objective="regression_l1", random_state=seed, **LGBM_PARAMS
        )
        regressor.fit(
            train_x.loc[nonzero],
            train_y[nonzero],
            categorical_feature=list(CATEGORICAL_FEATURES),
        )
    return classifier, regressor


def zero_inflated_predict(classifier, regressor, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p_zero = classifier.predict_proba(x)[:, 1]
    if regressor is None:
        return np.zeros(len(x), dtype=float), p_zero
    prediction = (1.0 - p_zero) * regressor.predict(x)
    return np.clip(prediction, 0.0, 100.0), p_zero


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


# ── Retrieval primitives ─────────────────────────────────────────────────


def top_neighbours_from_similarity(similarity: np.ndarray, k: int):
    """Row-wise top-k (index, similarity), sorted descending."""
    k_eff = min(int(k), similarity.shape[1])
    top = np.argpartition(-similarity, kth=k_eff - 1, axis=1)[:, :k_eff]
    top_similarity = np.take_along_axis(similarity, top, axis=1)
    order = np.argsort(-top_similarity, axis=1)
    return (
        np.take_along_axis(top, order, axis=1),
        np.take_along_axis(top_similarity, order, axis=1),
    )


def similarity_weights(top_similarity: np.ndarray, temperature: float) -> np.ndarray:
    logits = (top_similarity - top_similarity.max(axis=1, keepdims=True)) / temperature
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def blend_from_components(
    query_base: np.ndarray,
    local_label: np.ndarray,
    local_residual: np.ndarray,
    *,
    scale: float,
    blend: float,
) -> np.ndarray:
    """(1-blend) * (base + scale * residual kNN) + blend * label kNN."""
    return (1.0 - blend) * (query_base + scale * local_residual) + blend * local_label


def query_neighbour_components(
    similarity: np.ndarray,
    memory_base: np.ndarray,
    memory_y: np.ndarray,
    *,
    k: int,
    temperature: float,
):
    top, top_similarity = top_neighbours_from_similarity(similarity, k)
    weights = similarity_weights(top_similarity, temperature)
    local_label = np.einsum("ij,ij->i", memory_y[top], weights)
    local_residual = np.einsum("ij,ij->i", (memory_y - memory_base)[top], weights)
    nonzero = (memory_y > 0.0).astype(np.float64)
    local_nonzero = np.einsum("ij,ij->i", nonzero[top], weights)
    return local_label, local_residual, local_nonzero


# ── Similarity spaces ────────────────────────────────────────────────────


def morgan_bitvect(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)


class MorganSpace:
    """Role-preserving mean per-role Tanimoto similarity, cached per SMILES."""

    def __init__(self):
        self.cache: dict[str, object] = {}

    def fps_for(self, row) -> list:
        fps = []
        for column in SMILES_ROLES:
            value = row[column]
            key = value if isinstance(value, str) else ""
            if key not in self.cache:
                self.cache[key] = morgan_bitvect(key)
            fps.append(self.cache[key])
        return fps

    def similarity(self, query_frame: pd.DataFrame, memory_frame: pd.DataFrame) -> np.ndarray:
        query_fps = [self.fps_for(row) for row in query_frame.to_dict("records")]
        memory_fps = [self.fps_for(row) for row in memory_frame.to_dict("records")]
        output = np.zeros((len(query_fps), len(memory_fps)), dtype=np.float64)
        for i, q in enumerate(query_fps):
            for j, m in enumerate(memory_fps):
                values = [
                    DataStructs.TanimotoSimilarity(qf, mf)
                    for qf, mf in zip(q, m)
                    if qf is not None and mf is not None
                ]
                output[i, j] = float(np.mean(values)) if values else 0.0
        return output


class ConditionSpace:
    """Deploy-time condition similarity: categorical match fraction plus an
    RBF kernel over standardized numeric conditions (source-fitted)."""

    def __init__(self, source: pd.DataFrame, target: pd.DataFrame):
        self.categorical = CONDITION_CATEGORICAL
        numeric = np.zeros((len(target), len(NUMERIC_FEATURES)), dtype=np.float64)
        self.stds: dict[str, float] = {}
        for position, column in enumerate(NUMERIC_FEATURES):
            values = pd.to_numeric(source[column], errors="coerce")
            mean = float(values.mean()) if values.notna().any() else 0.0
            std = float(values.std()) if values.notna().any() else 0.0
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            self.stds[column] = std
            numeric[:, position] = (
                pd.to_numeric(target[column], errors="coerce").fillna(mean).to_numpy() - mean
            ) / std
        self.numeric = numeric
        self.categorical_values = [
            target[column].fillna("<MISSING>").astype(str).to_numpy()
            for column in self.categorical
        ]
        # RBF band: median pairwise distance on a deterministic subsample.
        rng = np.random.default_rng(SEED)
        sample = numeric[rng.choice(len(numeric), size=min(len(numeric), 500), replace=False)]
        diffs = sample[:, None, :] - sample[None, :, :]
        distances = np.sqrt((diffs ** 2).sum(axis=2)).ravel()
        distances = distances[distances > 0]
        self.band = float(np.median(distances)) if len(distances) else 1.0

    def similarity(self, query_rows: np.ndarray, memory_rows: np.ndarray) -> np.ndarray:
        query_numeric = self.numeric[query_rows]
        memory_numeric = self.numeric[memory_rows]
        squared = ((query_numeric[:, None, :] - memory_numeric[None, :, :]) ** 2).sum(axis=2)
        rbf = np.exp(-squared / (2.0 * max(self.band, 1e-8) ** 2))
        match = np.zeros((len(query_rows), len(memory_rows)), dtype=np.float64)
        for values in self.categorical_values:
            match += values[query_rows][:, None] == values[memory_rows][None, :]
        return 0.5 * rbf + 0.5 * match / len(self.categorical_values)


class FoldSimilarity:
    """Per-fold similarity matrices against the calibration memory and the
    identity-leaf overlap from the v1 base model.  Combined spaces reuse the
    cached leaf / morgan / condition base matrices."""

    def __init__(
        self,
        *,
        target_frame: pd.DataFrame,
        cal_positions: np.ndarray,
        query_positions: np.ndarray,
        leaf_matrix: np.ndarray,
        morgan: MorganSpace,
        condition: ConditionSpace,
    ):
        self.cal_positions = np.asarray(cal_positions, dtype=int)
        self.query_positions = np.asarray(query_positions, dtype=int)
        self.target_frame = target_frame
        self.leaf_matrix = leaf_matrix  # [n_target, n_target] leaf overlap
        self.morgan = morgan
        self.condition = condition
        self._cache: dict[tuple, np.ndarray] = {}

    def matrix_positions(
        self, space: str, rows: np.ndarray, memory_rows: np.ndarray
    ) -> np.ndarray:
        key = (space, tuple(rows.tolist()), tuple(memory_rows.tolist()))
        if key in self._cache:
            return self._cache[key]
        if space == "leaf":
            result = self.leaf_matrix[np.ix_(rows, memory_rows)].astype(np.float64)
        elif space == "morgan":
            result = self.morgan.similarity(
                self.target_frame.iloc[rows], self.target_frame.iloc[memory_rows]
            )
        elif space == "condition":
            result = self.condition.similarity(rows, memory_rows)
        elif space.startswith("combined:"):
            w_leaf, w_morgan, w_condition = (float(v) for v in space.split(":")[1:])
            result = (
                w_leaf * self.matrix_positions("leaf", rows, memory_rows)
                + w_morgan * self.matrix_positions("morgan", rows, memory_rows)
                + w_condition * self.matrix_positions("condition", rows, memory_rows)
            )
        else:
            raise ValueError(f"Unknown base similarity space: {space}")
        self._cache[key] = result
        return result

    def matrix(self, space: str, memory_kind: str) -> np.ndarray:
        rows = self.cal_positions if memory_kind == "cal" else self.query_positions
        return self.matrix_positions(space, rows, self.cal_positions)


SPACES = DIAGNOSTIC_SPACES + tuple(
    f"combined:{w[0]}:{w[1]}:{w[2]}" for w in COMBINATION_WEIGHTS
)


def leaf_overlap_matrix(leaves: np.ndarray) -> np.ndarray:
    """Fraction of matching terminal leaves between every pair of rows (v1)."""
    n, n_trees = leaves.shape
    overlap = np.zeros((n, n), dtype=np.float64)
    for tree in range(n_trees):
        column = leaves[:, tree]
        overlap += column[:, None] == column[None, :]
    return overlap / n_trees


# ── LOO selection on the calibration memory ─────────────────────────────


def simple_loo_mae(
    method: str,
    cal_base: dict[str, np.ndarray],
    cal_y: np.ndarray,
    group_ids: np.ndarray | None = None,
) -> float:
    n_cal = len(cal_y)
    if method in cal_base:
        return float(np.abs(cal_base[method] - cal_y).mean())
    if method.endswith("__global_residual"):
        variant = method[: -len("__global_residual")]
        errors = np.empty(n_cal)
        for index in range(n_cal):
            keep = np.arange(n_cal) != index
            if group_ids is not None:
                blocked = keep & (group_ids != group_ids[index])
                if blocked.sum() >= 1:
                    keep = blocked
            offset = float((cal_y[keep] - cal_base[variant][keep]).mean())
            errors[index] = abs(cal_base[variant][index] + offset - cal_y[index])
        return float(errors.mean())
    if method.endswith("__affine"):
        variant = method[: -len("__affine")]
        errors = np.empty(n_cal)
        for index in range(n_cal):
            keep = np.arange(n_cal) != index
            if group_ids is not None:
                blocked = keep & (group_ids != group_ids[index])
                if blocked.sum() >= 1:
                    keep = blocked
            prediction = affine_residual_prediction(
                cal_base[variant][keep], cal_y[keep], cal_base[variant][index : index + 1]
            )
            errors[index] = abs(prediction[0] - cal_y[index])
        return float(errors.mean())
    raise ValueError(f"Unknown simple LOO method: {method}")


SIMPLE_METHODS = tuple(
    f"{variant}{suffix}"
    for variant in ("base_v1", "base_zi", "base_ge")
    for suffix in ("", "__global_residual", "__affine")
)


# ── Group-blocked LOO selection with ε-top-set ensemble ──────────────────


def loo_neighbour_components(
    similarity: np.ndarray,
    memory_base: np.ndarray,
    memory_y: np.ndarray,
    *,
    k: int,
    temperature: float,
    group_ids: np.ndarray | None = None,
):
    """Leave-one-out local label / residual means for every memory row.

    With ``group_ids`` the blocked variant excludes every memory row sharing
    the row's exact_input_group (replicate rows), mirroring the memory-query
    group-disjointness rule; it falls back to self-exclusion when the blocked
    memory would be empty.
    """
    n = similarity.shape[0]
    local_label = np.empty(n, dtype=np.float64)
    local_residual = np.empty(n, dtype=np.float64)
    for index in range(n):
        keep = np.ones(n, dtype=bool)
        keep[index] = False
        if group_ids is not None:
            blocked = keep & (group_ids != group_ids[index])
            if blocked.sum() >= 1:
                keep = blocked
        top, top_similarity = top_neighbours_from_similarity(similarity[index, keep][None, :], k)
        weights = similarity_weights(top_similarity, temperature)
        local_label[index] = float(memory_y[keep][top[0]] @ weights[0])
        local_residual[index] = float((memory_y - memory_base)[keep][top[0]] @ weights[0])
    return local_label, local_residual


def retrieval_loo_mae(
    local_label: np.ndarray,
    local_residual: np.ndarray,
    cal_base_variant: np.ndarray,
    cal_y: np.ndarray,
) -> dict[float, dict[float, float]]:
    """MAE per (scale, blend) from shared LOO neighbour components."""
    output: dict[float, dict[float, float]] = {}
    for scale in SCALE_GRID:
        per_blend: dict[float, float] = {}
        for blend in BLEND_GRID:
            prediction = blend_from_components(
                cal_base_variant, local_label, local_residual, scale=scale, blend=blend
            )
            per_blend[blend] = float(np.abs(prediction - cal_y).mean())
        output[scale] = per_blend
    return output


def group_split_for_diagnostic(group_ids: np.ndarray, seed: int):
    """Group-disjoint half split, used only for the residual sign-stability
    diagnostic (never for selection)."""
    n = len(group_ids)
    unique_groups, group_of_row = np.unique(group_ids, return_inverse=True)
    sizes = np.bincount(group_of_row, minlength=len(unique_groups))
    order = np.random.default_rng(seed).permutation(len(unique_groups))
    count = 0
    side = np.zeros(len(unique_groups), dtype=bool)
    for group in order:
        if count < n / 2.0:
            side[group] = True
            count += int(sizes[group])
    mask_a = side[group_of_row]
    return np.flatnonzero(mask_a), np.flatnonzero(~mask_a)


# ── Metrics ──────────────────────────────────────────────────────────────


def metric_record(y: np.ndarray, pred: np.ndarray, score: np.ndarray | None = None) -> dict:
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
    record = {
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
    if score is not None and 0.0 < float(np.mean(y > 0)) < 1.0:
        record["nonzero_auroc"] = float(roc_auc_score(y > 0, score))
        record["nonzero_auprc"] = float(average_precision_score(y > 0, score))
    else:
        record["nonzero_auroc"] = np.nan
        record["nonzero_auprc"] = np.nan
    return record


def bootstrap_fold_delta(
    metrics: pd.DataFrame,
    candidate: str,
    comparator: str,
    *,
    budget: int,
    n_bootstrap: int = 10000,
) -> dict:
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


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/hitea_retrieval_shift/v1"))
    parser.add_argument("--output", type=Path, default=Path("results/hitea_target_memory_development_v2"))
    parser.add_argument("--only-folds", nargs="*", default=[])
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.data_root / "materialization_manifest.json").read_text())
    table = pd.read_csv(args.data_root / "development_modeling_table.csv", low_memory=False)
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

    # Global-exclusion base: exclude all 26 target screens AND every
    # exact_input_group that appears in them (the Track 2 GNN protocol).
    all_target_screen_ids = set(fold_index["target_screen_id"].astype(str))
    all_target_groups = set(
        table.loc[table["screen_id"].isin(all_target_screen_ids), "exact_input_group"]
    )
    global_source = table.loc[
        table["Yield"].notna()
        & ~table["screen_id"].isin(all_target_screen_ids)
        & ~table["exact_input_group"].isin(all_target_groups)
    ].copy()
    ge_space = fit_feature_space(global_source)
    ge_model = fit_base_model(
        apply_feature_space(ge_space, global_source),
        global_source["Yield"].to_numpy(float),
        stable_seed(SEED, "global_exclusion"),
    )
    ge_source_size = int(len(global_source))

    morgan = MorganSpace()
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    fold_settings: list[dict] = []
    selection_rows: list[dict] = []

    for fold_number, fold in enumerate(fold_index.itertuples(index=False), start=1):
        fold_id = str(fold.fold_id)
        assignment = pd.read_csv(args.data_root / "folds" / fold_id / "target_assignment.csv")
        target = table.loc[table["record_id"].isin(assignment["record_id"])].merge(
            assignment,
            on=["record_id", "exact_input_group", "reaction_core_group"],
            validate="one_to_one",
        )
        if not target.loc[target["is_target_test"], "Yield"].isna().all():
            raise AssertionError(f"Locked target labels are visible in {fold_id}")
        target_development = target.loc[~target["is_target_test"]].copy().reset_index(drop=True)
        if target_development["Yield"].isna().any():
            raise AssertionError(f"Non-test target labels are unexpectedly missing in {fold_id}")
        fixed_query = target_development.loc[~target_development["is_calibration_20"]].copy()
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

        # Per-fold fold-safe feature space, fitted on the source only.
        fold_space = fit_feature_space(source)
        train_x = apply_feature_space(fold_space, source)
        target_x = apply_feature_space(fold_space, target_development)
        model_seed = stable_seed(SEED, fold_id)
        base_v1 = fit_base_model(train_x, source["Yield"].to_numpy(float), model_seed)
        classifier_zi, regressor_zi = fit_zero_inflated_model(
            train_x, source["Yield"].to_numpy(float), model_seed
        )

        target_base_v1 = np.clip(base_v1.predict(target_x).astype(float), 0.0, 100.0)
        target_base_zi, p_zero_zi = zero_inflated_predict(classifier_zi, regressor_zi, target_x)
        target_base_ge = np.clip(
            ge_model.predict(apply_feature_space(ge_space, target_development)).astype(float),
            0.0,
            100.0,
        )
        target_leaf = base_v1.predict(target_x, pred_leaf=True).astype(np.int32)

        position_of = pd.Series(
            np.arange(len(target_development)), index=target_development["record_id"].to_numpy()
        )
        query_pos = position_of.loc[fixed_query["record_id"].to_numpy()].to_numpy(int)
        query_y = fixed_query["Yield"].to_numpy(float)
        leaf_matrix = leaf_overlap_matrix(target_leaf)
        condition_space = ConditionSpace(source, target_development)

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
                "condition_rbf_band": condition_space.band,
            }
        )

        cal_positions_by_budget: dict[int, np.ndarray] = {}
        for budget in CALIBRATION_BUDGETS:
            memory = target_development.loc[target_development[f"is_calibration_{budget:02d}"]]
            if memory.empty:
                raise ValueError(f"Empty {budget}% memory in {fold_id}")
            if set(memory["exact_input_group"]) & set(fixed_query["exact_input_group"]):
                raise AssertionError(f"Memory-query exact input leakage in {fold_id}/{budget}")
            cal_positions_by_budget[budget] = memory.index.to_numpy(int)

        for budget in CALIBRATION_BUDGETS:
            cal_positions = cal_positions_by_budget[budget]
            cal_y = target_development.iloc[cal_positions]["Yield"].to_numpy(float)
            cal_group_ids = (
                target_development.iloc[cal_positions]["exact_input_group"].to_numpy()
            )
            n_cal = len(cal_positions)

            cal_base = {
                "base_v1": target_base_v1[cal_positions],
                "base_zi": target_base_zi[cal_positions],
                "base_ge": target_base_ge[cal_positions],
            }
            query_base = {
                "base_v1": target_base_v1[query_pos],
                "base_zi": target_base_zi[query_pos],
                "base_ge": target_base_ge[query_pos],
            }
            query_p_zero = p_zero_zi[query_pos]

            sim = FoldSimilarity(
                target_frame=target_development,
                cal_positions=cal_positions,
                query_positions=query_pos,
                leaf_matrix=leaf_matrix,
                morgan=morgan,
                condition=condition_space,
            )

            # ── Group-blocked LOO selection over the frozen registry ──
            simple_candidates: list[dict] = []
            retrieval_candidates: list[dict] = []
            for method in SIMPLE_METHODS:
                simple_candidates.append(
                    {
                        "method": method,
                        "loo_mae": simple_loo_mae(method, cal_base, cal_y, cal_group_ids),
                        "config": {"kind": "simple", "method": method},
                    }
                )
            for base_variant in BASE_VARIANTS:
                for space in SPACES:
                    matrix = sim.matrix(space, memory_kind="cal")
                    for k in K_GRID:
                        if k >= n_cal:
                            continue
                        for temperature in TEMPERATURE_GRID:
                            local_label, local_residual = loo_neighbour_components(
                                matrix,
                                cal_base[base_variant],
                                cal_y,
                                k=k,
                                temperature=temperature,
                                group_ids=cal_group_ids,
                            )
                            per_scale = retrieval_loo_mae(
                                local_label, local_residual, cal_base[base_variant], cal_y
                            )
                            for scale, per_blend in per_scale.items():
                                for blend, loo_mae in per_blend.items():
                                    retrieval_candidates.append(
                                        {
                                            "method": (
                                                f"{base_variant}|{space}|k{k}|T{temperature}"
                                                f"|s{scale}|b{blend}"
                                            ),
                                            "loo_mae": loo_mae,
                                            "config": {
                                                "kind": "retrieval",
                                                "base": base_variant,
                                                "space": space,
                                                "k": k,
                                                "temperature": temperature,
                                                "scale": scale,
                                                "blend": blend,
                                            },
                                        }
                                    )
            best_simple = min(
                simple_candidates, key=lambda c: (c["loo_mae"], c["method"])
            )
            best_retrieval = min(
                retrieval_candidates, key=lambda c: (c["loo_mae"], c["method"])
            )
            use_retrieval = (
                best_retrieval["loo_mae"]
                <= best_simple["loo_mae"] * (1.0 - FAMILY_GATE_MARGIN)
            )
            gate_note = {
                "best_simple_method": best_simple["method"],
                "best_simple_loo_mae": best_simple["loo_mae"],
                "best_retrieval_loo_mae": best_retrieval["loo_mae"],
                "gate_margin": FAMILY_GATE_MARGIN,
            }
            if use_retrieval:
                # ε-top-set ensemble: average every retrieval configuration
                # whose LOO MAE is within ENSEMBLE_REL_TOLERANCE of the best.
                members = sorted(
                    (
                        c
                        for c in retrieval_candidates
                        if c["loo_mae"]
                        <= best_retrieval["loo_mae"] * (1.0 + ENSEMBLE_REL_TOLERANCE)
                    ),
                    key=lambda c: (c["loo_mae"], c["method"]),
                )[:ENSEMBLE_MAX_MEMBERS]
                selected_method = f"ensemble_eps{ENSEMBLE_REL_TOLERANCE}(n={len(members)})"
                selection_rows.append(
                    {
                        "fold_id": fold_id,
                        "budget_percent": budget,
                        "selected_method": selected_method,
                        "family": "retrieval_ensemble",
                        "loo_mae": best_retrieval["loo_mae"],
                        "best_member": best_retrieval["method"],
                        "n_members": len(members),
                        "members": ";".join(c["method"] for c in members),
                        **gate_note,
                    }
                )
            else:
                members = [best_simple]
                selected_method = best_simple["method"]
                selection_rows.append(
                    {
                        "fold_id": fold_id,
                        "budget_percent": budget,
                        "selected_method": selected_method,
                        "family": "simple",
                        "loo_mae": best_simple["loo_mae"],
                        "best_member": best_simple["method"],
                        "n_members": 1,
                        "members": best_simple["method"],
                        **gate_note,
                    }
                )

            # ── Predictions on the fixed development queries ──
            predictions: dict[str, np.ndarray] = {}
            scores: dict[str, np.ndarray | None] = {}
            for variant in ("base_v1", "base_zi", "base_ge"):
                predictions[variant] = query_base[variant]
                scores[variant] = 1.0 - query_p_zero if variant == "base_zi" else None
            for variant in ("base_v1", "base_zi", "base_ge"):
                offset = float((cal_y - cal_base[variant]).mean())
                predictions[f"{variant}__global_residual"] = query_base[variant] + offset
                predictions[f"{variant}__affine"] = affine_residual_prediction(
                    cal_base[variant], cal_y, query_base[variant]
                )
            predictions["target_mean"] = np.full(len(query_y), float(cal_y.mean()))

            # Fixed diagnostic retrievals: every space at k=5, T=0.05, s=1,
            # pure residual (b0) and pure label (b1), on both base variants.
            for space in DIAGNOSTIC_SPACES:
                matrix = sim.matrix(space, memory_kind="query")
                for variant in BASE_VARIANTS:
                    local_label, local_residual, local_nonzero = query_neighbour_components(
                        matrix, cal_base[variant], cal_y, k=5, temperature=0.05
                    )
                    for blend, tag in ((0.0, "b0.0"), (1.0, "b1.0")):
                        name = f"r_{variant}|{space}|k5|T0.05|s1|{tag}"
                        predictions[name] = blend_from_components(
                            query_base[variant],
                            local_label,
                            local_residual,
                            scale=1.0,
                            blend=blend,
                        )
                        scores[name] = local_nonzero if blend == 1.0 else None

            # Frozen selected procedure: ε-top-set ensemble (single member when
            # the simple family wins the gate).
            def member_prediction(config: dict, labels: np.ndarray):
                if config["kind"] == "simple":
                    method = config["method"]
                    if method in cal_base:
                        return query_base[method].copy(), None
                    if method.endswith("__global_residual"):
                        variant = method[: -len("__global_residual")]
                        offset = float((labels - cal_base[variant]).mean())
                        return query_base[variant] + offset, None
                    variant = method[: -len("__affine")]
                    return (
                        affine_residual_prediction(
                            cal_base[variant], labels, query_base[variant]
                        ),
                        None,
                    )
                matrix = sim.matrix(config["space"], memory_kind="query")
                local_label, local_residual, local_nonzero = query_neighbour_components(
                    matrix,
                    cal_base[config["base"]],
                    labels,
                    k=config["k"],
                    temperature=config["temperature"],
                )
                prediction = blend_from_components(
                    query_base[config["base"]],
                    local_label,
                    local_residual,
                    scale=config["scale"],
                    blend=config["blend"],
                )
                score = local_nonzero if config["blend"] >= 0.5 else None
                return prediction, score

            member_predictions = [member_prediction(m["config"], cal_y) for m in members]
            selected = np.mean([p for p, _ in member_predictions], axis=0)
            member_scores = [s for _, s in member_predictions if s is not None]
            scores_member = (
                np.mean(member_scores, axis=0) if len(member_scores) > len(members) / 2 else None
            )
            predictions["v2_selected"] = selected
            scores["v2_selected"] = scores_member

            # Residual-permutation control: same members, permuted labels.
            rng = np.random.default_rng(stable_seed(SEED, fold_id, budget, "v2permute"))
            permuted_y = rng.permutation(cal_y)
            if use_retrieval:
                permuted_members = [member_prediction(m["config"], permuted_y) for m in members]
                predictions["v2_selected_permuted"] = np.mean(
                    [p for p, _ in permuted_members], axis=0
                )
            else:
                predictions["v2_selected_permuted"] = selected.copy()

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
                    "memory_size": int(n_cal),
                    "selected_method": selected_method,
                }
            )
            for method, values in predictions.items():
                diagnostic[f"prediction_{method}"] = np.asarray(values, dtype=float)
                score = scores.get(method)
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
                            **metric_record(
                                query_y[mask],
                                np.asarray(values, dtype=float)[mask],
                                score[mask] if score is not None else None,
                            ),
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
    pd.DataFrame(selection_rows).to_csv(args.output / "selection_log.csv", index=False)
    json_dump(
        args.output / "fold_settings.json",
        {"folds": fold_settings, "selection_rule": SELECTION_RULE},
    )

    summary_rows: list[dict] = []
    for (budget, method), part in metrics.loc[metrics["subgroup"].eq("all")].groupby(
        ["budget_percent", "method"]
    ):
        pred_part = predictions.loc[predictions["budget_percent"].eq(budget)]
        pooled = metric_record(
            pred_part["Yield"].to_numpy(float),
            pred_part[f"prediction_{method}"].to_numpy(float),
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
            summary_rows.append(
                {
                    "scope": "family_pooled",
                    "reaction_keyword": family,
                    "budget_percent": int(budget),
                    "method": method,
                    **metric_record(
                        family_predictions["Yield"].to_numpy(float),
                        family_predictions[f"prediction_{method}"].to_numpy(float),
                    ),
                    "macro_fold_mae": float(family_metrics["mae"].mean()),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    pooled20 = summary.loc[
        summary["scope"].eq("all_folds_pooled") & summary["budget_percent"].eq(20)
    ].set_index("method")
    comparisons = {
        comparator: bootstrap_fold_delta(metrics, "v2_selected", comparator, budget=20)
        for comparator in (
            "base_v1",
            "base_zi",
            "base_zi__affine",
            "r_base_zi|leaf|k5|T0.05|s1|b0.0",
            "r_base_zi|leaf|k5|T0.05|s1|b1.0",
        )
        if comparator in pooled20.index
    }
    decision = {
        "status": "diagnostic_only",
        "locked_target_test_scored": False,
        "candidate": "Per-fold calibration-LOO gated correction (v2 frozen procedure)",
        "selection_rule": SELECTION_RULE,
        "budget_percent": 20,
        "pooled_mae": {
            method: (
                float(pooled20.loc[method, "mae"]) if method in pooled20.index else np.nan
            )
            for method in ("base_v1", "base_zi", "v2_selected", "v2_selected_permuted")
        },
        "pooled_r2": {
            method: (
                float(pooled20.loc[method, "r2"]) if method in pooled20.index else np.nan
            )
            for method in ("base_v1", "base_zi", "v2_selected")
        },
        "fold_block_bootstrap_vs_v2_selected": comparisons,
        "interpretation_rule": (
            "Development-round upgrade over v1. Cannot validate the GNN eKNN claim "
            "or unlock the locked target test."
        ),
    }
    json_dump(args.output / "development_decision_v2.json", decision)
    json_dump(
        args.output / "run_manifest.json",
        {
            "run_id": "hitea_target_memory_development_v2",
            "status": "diagnostic_only",
            "data_manifest_sha256": manifest["development_modeling_table_sha256"],
            "locked_test_registry_sha256": manifest["locked_test_registry_sha256"],
            "n_locked_test_labels_scored": 0,
            "n_folds": int(len(fold_index)),
            "n_global_exclusion_source_rows": ge_source_size,
            "seed": SEED,
            "base_models": [
                "LightGBM L1 (v1)",
                "zero-inflated LightGBM",
                "global-exclusion LightGBM",
            ],
            "similarity_spaces": list(SPACES),
            "retrieval_grid": {
                "k": list(K_GRID),
                "temperature": list(TEMPERATURE_GRID),
                "scale": list(SCALE_GRID),
                "blend": list(BLEND_GRID),
            },
            "selection_rule": SELECTION_RULE,
            "calibration_budgets_percent": list(CALIBRATION_BUDGETS),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
            "wall_time_seconds": float(time.time() - started),
        },
    )
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

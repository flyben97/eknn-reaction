#!/usr/bin/env python3
"""Freeze the high-calibration photoredox affine-eKNN candidate.

This script uses only the already-opened paper-cold-start development target.
It deterministically reallocates 60% of each target paper to calibration,
cross-fits the calibration predictions, and evaluates the pre-specified
chemical/affine/local-1NN blend on the remaining opened rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from develop_photoredox_meta_eknn_r2 import (
    ROOT,
    ReactionFeaturizer,
    base_model,
    deterministic_paper_roles,
    metrics,
    similarity_matrix,
)


CALIBRATION_FRACTION = 0.60
CALIBRATION_WEIGHT = 10.0
SLOPE_PRIOR = 1.0
SLOPE_SHRINKAGE = 150.0
INTERCEPT_SHRINKAGE = 5.0
BASE_WEIGHT = 0.12
AFFINE_WEIGHT = 0.82
PAPER_1NN_WEIGHT = 0.06


def stable_int(value: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{value}".encode()).hexdigest(), 16)


def calibration_folds(frame: pd.DataFrame, calibration: np.ndarray) -> np.ndarray:
    """Assign duplicate groups to five OOF folds inside every paper."""
    folds = np.full(len(frame), -1, dtype=int)
    for paper, group in frame.loc[calibration].groupby("paper_group", sort=True):
        duplicate_groups = []
        for duplicate_group, rows in group.groupby("duplicate_group", sort=True):
            duplicate_groups.append(
                (
                    stable_int(
                        f"{paper}|{duplicate_group}",
                        "affine-calibration-oof-v1",
                    ),
                    rows.index.to_numpy(),
                )
            )
        duplicate_groups.sort(key=lambda item: item[0])
        for rank, (_, rows) in enumerate(duplicate_groups):
            folds[rows] = rank % 5
    if np.any(folds[calibration] < 0):
        raise RuntimeError("Every calibration row must receive an OOF fold")
    return folds


def fit_base(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_fit: np.ndarray,
    *,
    seed: int,
):
    fit_x = np.concatenate([source_x, target_x[target_fit]])
    fit_y = np.concatenate([source_y, target_y[target_fit]])
    weights = np.concatenate(
        [
            np.ones(len(source_y), dtype=float),
            np.full(int(target_fit.sum()), CALIBRATION_WEIGHT, dtype=float),
        ]
    )
    model = base_model(seed)
    model.fit(fit_x, fit_y, sample_weight=weights)
    return model


def crossfit_calibration_predictions(
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
    calibration: np.ndarray,
    folds: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    prediction = np.full(len(target_y), np.nan, dtype=float)
    for fold in range(5):
        heldout = calibration & (folds == fold)
        target_fit = calibration & ~heldout
        model = fit_base(
            source_x,
            source_y,
            target_x,
            target_y,
            target_fit,
            seed=seed + fold,
        )
        prediction[heldout] = np.clip(model.predict(target_x[heldout]), 0, 1)
    if np.any(~np.isfinite(prediction[calibration])):
        raise RuntimeError("Missing cross-fitted calibration prediction")
    return prediction


def paper_affine_parameters(
    frame: pd.DataFrame,
    target_y: np.ndarray,
    calibration: np.ndarray,
    calibration_oof: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Estimate per-paper affine maps, shrunk toward identity and zero offset."""
    parameters: dict[str, dict[str, float]] = {}
    for paper, group in frame.loc[calibration].groupby("paper_group", sort=True):
        rows = group.index.to_numpy()
        x = calibration_oof[rows]
        y = target_y[rows]
        n = len(rows)
        centered_x = x - x.mean()
        denominator = float(np.sum(centered_x**2))
        if denominator <= 1e-12:
            raw_slope = SLOPE_PRIOR
        else:
            raw_slope = float(np.sum(centered_x * (y - y.mean())) / denominator)
        raw_slope = float(np.clip(raw_slope, 0.0, 2.0))
        slope = float(
            (n * raw_slope + SLOPE_SHRINKAGE * SLOPE_PRIOR)
            / (n + SLOPE_SHRINKAGE)
        )
        # Shrink the level correction independently from the slope.  Anchoring
        # the slope term at the paper's calibration mean avoids introducing a
        # systematic offset when the slope is shrunk toward one.
        raw_intercept = float(y.mean() - x.mean())
        intercept = float(n / (n + INTERCEPT_SHRINKAGE) * raw_intercept)
        parameters[str(paper)] = {
            "n": int(n),
            "prediction_mean": float(x.mean()),
            "raw_slope": raw_slope,
            "slope": slope,
            "raw_intercept": raw_intercept,
            "intercept": intercept,
        }
    return parameters


def apply_paper_affine(
    frame: pd.DataFrame,
    rows: np.ndarray,
    prediction: np.ndarray,
    parameters: dict[str, dict[str, float]],
) -> np.ndarray:
    output = np.empty(len(rows), dtype=float)
    for position, row in enumerate(rows):
        paper = str(frame.at[row, "paper_group"])
        parameter = parameters[paper]
        output[position] = (
            prediction[row]
            + parameter["intercept"]
            + (parameter["slope"] - 1.0)
            * (prediction[row] - parameter["prediction_mean"])
        )
    return np.clip(output, 0, 1)


def paper_local_1nn(
    frame: pd.DataFrame,
    normalized: dict[str, np.ndarray],
    query: np.ndarray,
    memory: np.ndarray,
    target_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = similarity_matrix(query, memory, "chem", normalized)
    query_paper = frame.loc[query, "paper_group"].astype(str).to_numpy()
    memory_paper = frame.loc[memory, "paper_group"].astype(str).to_numpy()
    same_paper = query_paper[:, None] == memory_paper[None, :]
    scoped = np.where(same_paper, similarities, -np.inf)
    nearest = np.argmax(scoped, axis=1)
    nearest_similarity = scoped[np.arange(len(query)), nearest]
    if np.any(~np.isfinite(nearest_similarity)):
        raise RuntimeError("Every query row must have paper-local calibration memory")
    return target_y[memory[nearest]], nearest_similarity


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=ROOT / "data/photoredox_ood_full/03_paper_coldstart_confirmatory_7k",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/photoredox_affine_eknn_candidate_development_s42",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_path = args.task_dir / "source_train.csv"
    calibration_path = args.task_dir / "calibration.csv"
    test_path = args.task_dir / "target_test.csv"
    source = pd.read_csv(source_path).reset_index(drop=True)
    target = pd.concat(
        [pd.read_csv(calibration_path), pd.read_csv(test_path)],
        ignore_index=True,
    )
    source_y = source["Yield"].to_numpy(dtype=float)
    target_y = target["Yield"].to_numpy(dtype=float)

    eligible, calibration, _ = deterministic_paper_roles(
        target,
        min_rows=15,
        calibration_fraction=CALIBRATION_FRACTION,
    )
    if not eligible.all():
        raise RuntimeError("Every opened target paper must be eligible")
    query = np.flatnonzero(~calibration)
    memory = np.flatnonzero(calibration)
    folds = calibration_folds(target, calibration)

    featurizer = ReactionFeaturizer()
    source_x, _ = featurizer.transform(source)
    target_x, target_normalized = featurizer.transform(target)
    calibration_oof = crossfit_calibration_predictions(
        source_x,
        source_y,
        target_x,
        target_y,
        calibration,
        folds,
        seed=args.seed + 1000,
    )
    final_model = fit_base(
        source_x,
        source_y,
        target_x,
        target_y,
        calibration,
        seed=args.seed + 2000,
    )
    base_all = np.clip(final_model.predict(target_x), 0, 1)
    parameters = paper_affine_parameters(
        target,
        target_y,
        calibration,
        calibration_oof,
    )
    affine = apply_paper_affine(target, query, base_all, parameters)
    local_1nn, nearest_similarity = paper_local_1nn(
        target,
        target_normalized,
        query,
        memory,
        target_y,
    )
    candidate = np.clip(
        BASE_WEIGHT * base_all[query]
        + AFFINE_WEIGHT * affine
        + PAPER_1NN_WEIGHT * local_1nn,
        0,
        1,
    )
    papers = target.loc[query, "paper_group"].astype(str).to_numpy()
    score = {
        "chemical_base": metrics(target_y[query], base_all[query], papers),
        "paper_affine": metrics(target_y[query], affine, papers),
        "affine_eknn_candidate": metrics(target_y[query], candidate, papers),
    }

    assignments = target[
        ["reaction_id", "paper_group", "catalyst_group", "duplicate_group"]
    ].copy()
    assignments["role"] = np.where(calibration, "calibration", "development_test")
    assignments["calibration_oof_fold"] = folds
    assignments.to_csv(args.output_dir / "split_assignments.csv", index=False)
    calibration_frame = target.loc[
        memory,
        ["reaction_id", "paper_group", "duplicate_group", "Yield"],
    ].copy().reset_index(drop=True)
    calibration_frame["chemical_base_oof"] = calibration_oof[memory]
    calibration_frame["chemical_base_final"] = base_all[memory]
    calibration_frame.to_csv(
        args.output_dir / "calibration_oof_predictions.csv", index=False
    )
    pd.DataFrame.from_dict(parameters, orient="index").rename_axis("paper_group").to_csv(
        args.output_dir / "paper_affine_parameters.csv"
    )
    prediction_frame = target.loc[
        query,
        ["reaction_id", "paper_group", "catalyst_group", "substrate_smiles", "Yield"],
    ].copy().reset_index(drop=True)
    prediction_frame["chemical_base"] = base_all[query]
    prediction_frame["paper_affine"] = affine
    prediction_frame["paper_1nn"] = local_1nn
    prediction_frame["nearest_similarity"] = nearest_similarity
    prediction_frame["affine_eknn_candidate"] = candidate
    prediction_frame.to_csv(args.output_dir / "opened_development_predictions.csv", index=False)

    recipe = {
        "status": "frozen_from_opened_development_only",
        "external_confirmation_labels_loaded": False,
        "seed": args.seed,
        "calibration_fraction": CALIBRATION_FRACTION,
        "opened_target_counts": {
            "calibration": int(calibration.sum()),
            "development_test": int((~calibration).sum()),
            "papers": int(target["paper_group"].nunique()),
        },
        "chemical_model": {
            "family": "LightGBM_on_reaction_Morgan_counts_and_descriptors",
            "target_calibration_weight": CALIBRATION_WEIGHT,
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "reg_lambda": 3.0,
        },
        "paper_affine": {
            "calibration_prediction": "five_fold_duplicate_group_safe_oof",
            "raw_slope_clip": [0.0, 2.0],
            "slope_prior": SLOPE_PRIOR,
            "slope_shrinkage": SLOPE_SHRINKAGE,
            "intercept_prior": 0.0,
            "intercept_shrinkage": INTERCEPT_SHRINKAGE,
        },
        "paper_local_eknn": {
            "k": 1,
            "target": "measured_yield",
            "similarity": {
                "substrate": 0.4,
                "reactants": 0.2,
                "products": 0.2,
                "reagents": 0.1,
                "catalysts": 0.1,
            },
        },
        "final_blend": {
            "chemical_base": BASE_WEIGHT,
            "paper_affine": AFFINE_WEIGHT,
            "paper_1nn": PAPER_1NN_WEIGHT,
        },
        "opened_target_development": score,
        "absolute_target": {"r2": 0.60, "pearson_r": 0.80},
        "input_sha256": {
            "source_train": file_sha256(source_path),
            "opened_calibration": file_sha256(calibration_path),
            "opened_test": file_sha256(test_path),
        },
    }
    recipe_path = args.output_dir / "frozen_recipe.json"
    recipe_path.write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(recipe, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

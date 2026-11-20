#!/usr/bin/env python3
"""Generate unscored PhotoCatDB predictions with the frozen affine-eKNN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from develop_photoredox_meta_eknn_r2 import ROOT, ReactionFeaturizer
from freeze_photoredox_affine_eknn_candidate import (
    AFFINE_WEIGHT,
    BASE_WEIGHT,
    CALIBRATION_FRACTION,
    PAPER_1NN_WEIGHT,
    apply_paper_affine,
    calibration_folds,
    crossfit_calibration_predictions,
    fit_base,
    paper_affine_parameters,
    paper_local_1nn,
)


SEEDS = (42, 123, 2025)
TASK_DEFAULT = ROOT / "data/photoredox_external/photocatdb_r2_confirmatory_v1"
SOURCE_DEFAULT = (
    ROOT
    / "data/photoredox_ood_full/03_paper_coldstart_confirmatory_7k/source_train.csv"
)
RECIPE_DEFAULT = (
    ROOT
    / "results/photoredox_affine_eknn_candidate_development/frozen_ensemble_recipe.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, default=TASK_DEFAULT)
    parser.add_argument("--source-csv", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--candidate-recipe", type=Path, default=RECIPE_DEFAULT)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    materialization_path = args.task_dir / "materialization_manifest.json"
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if sha256(args.candidate_recipe) != materialization["candidate_recipe_sha256"]:
        raise RuntimeError("Frozen ensemble recipe hash does not match the task")
    calibration_path = args.task_dir / "calibration.csv"
    test_path = args.task_dir / "target_test_features.csv"
    if sha256(calibration_path) != materialization["calibration_sha256"]:
        raise RuntimeError("Calibration file changed after materialization")
    if sha256(test_path) != materialization["target_test_features_sha256"]:
        raise RuntimeError("Locked test feature file changed after materialization")

    source = pd.read_csv(args.source_csv).reset_index(drop=True)
    calibration_frame = pd.read_csv(calibration_path).reset_index(drop=True)
    test_frame = pd.read_csv(test_path).reset_index(drop=True)
    target = pd.concat([calibration_frame, test_frame], ignore_index=True)
    n_calibration = len(calibration_frame)
    calibration = np.zeros(len(target), dtype=bool)
    calibration[:n_calibration] = True
    query = np.arange(n_calibration, len(target), dtype=int)
    memory = np.arange(n_calibration, dtype=int)
    target_y = np.full(len(target), np.nan, dtype=float)
    target_y[memory] = calibration_frame["Yield"].to_numpy(dtype=float)
    source_y = source["Yield"].to_numpy(dtype=float)
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
    paper_1nn, nearest_similarity = paper_local_1nn(
        target,
        target_normalized,
        query,
        memory,
        target_y,
    )
    candidate = np.clip(
        BASE_WEIGHT * base_all[query]
        + AFFINE_WEIGHT * affine
        + PAPER_1NN_WEIGHT * paper_1nn,
        0,
        1,
    )
    prediction = test_frame[
        ["reaction_id", "paper_group", "catalyst_group", "substrate_smiles"]
    ].copy()
    prediction["chemical_base"] = base_all[query]
    prediction["paper_affine"] = affine
    prediction["paper_1nn"] = paper_1nn
    prediction["nearest_similarity"] = nearest_similarity
    prediction["affine_eknn_candidate"] = candidate
    prediction_path = args.output_dir / "target_predictions_unscored.csv"
    prediction.to_csv(prediction_path, index=False)

    calibration_output = calibration_frame[
        ["reaction_id", "paper_group", "duplicate_group", "Yield"]
    ].copy()
    calibration_output["chemical_base_oof"] = calibration_oof[memory]
    calibration_output["chemical_base_final"] = base_all[memory]
    calibration_output["oof_fold"] = folds[memory]
    calibration_output.to_csv(
        args.output_dir / "calibration_oof_predictions.csv", index=False
    )
    pd.DataFrame.from_dict(parameters, orient="index").rename_axis("paper_group").to_csv(
        args.output_dir / "paper_affine_parameters.csv"
    )
    manifest = {
        "status": "frozen_predictions_without_loading_locked_test_labels",
        "seed": args.seed,
        "calibration_fraction": CALIBRATION_FRACTION,
        "rows": {
            "source": int(len(source)),
            "calibration": int(len(calibration_frame)),
            "locked_test": int(len(test_frame)),
            "papers": int(test_frame["paper_group"].nunique()),
        },
        "weights": {
            "chemical_base": BASE_WEIGHT,
            "paper_affine": AFFINE_WEIGHT,
            "paper_1nn": PAPER_1NN_WEIGHT,
        },
        "hashes": {
            "source_csv": sha256(args.source_csv),
            "candidate_recipe": sha256(args.candidate_recipe),
            "materialization_manifest": sha256(materialization_path),
            "calibration": sha256(calibration_path),
            "target_test_features": sha256(test_path),
            "prediction": sha256(prediction_path),
            "predictor_script": sha256(Path(__file__)),
        },
    }
    manifest_path = args.output_dir / "prediction_freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

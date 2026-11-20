#!/usr/bin/env python3
"""Freeze the unscored equal-weight PhotoCatDB prediction ensemble."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2025)
OUTPUT = ROOT / "results/photocatdb_external_affine_eknn_ensemble"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    merged = None
    seed_files = {}
    prediction_columns = (
        "chemical_base",
        "paper_affine",
        "paper_1nn",
        "affine_eknn_candidate",
    )
    for seed in SEEDS:
        run = ROOT / f"results/photocatdb_external_affine_eknn_s{seed}"
        manifest_path = run / "prediction_freeze_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "frozen_predictions_without_loading_locked_test_labels":
            raise RuntimeError(f"Seed {seed} prediction is not frozen")
        prediction_path = run / "target_predictions_unscored.csv"
        if sha256(prediction_path) != manifest["hashes"]["prediction"]:
            raise RuntimeError(f"Seed {seed} prediction hash mismatch")
        frame = pd.read_csv(prediction_path)
        renamed = frame.rename(
            columns={column: f"{column}_s{seed}" for column in prediction_columns}
        )
        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(
                renamed.drop(
                    columns=[
                        "paper_group",
                        "catalyst_group",
                        "substrate_smiles",
                    ]
                ),
                on="reaction_id",
                how="inner",
                validate="one_to_one",
            )
        seed_files[str(seed)] = {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256(manifest_path),
            "prediction": str(prediction_path.relative_to(ROOT)),
            "prediction_sha256": sha256(prediction_path),
        }
    assert merged is not None
    for column in prediction_columns:
        merged[column] = merged[
            [f"{column}_s{seed}" for seed in SEEDS]
        ].mean(axis=1)
    output_path = OUTPUT / "target_predictions_ensemble_unscored.csv"
    merged.to_csv(output_path, index=False)
    manifest = {
        "status": "ensemble_frozen_before_locked_labels_are_loaded",
        "seeds": list(SEEDS),
        "seed_weights": {str(seed): 1 / len(SEEDS) for seed in SEEDS},
        "rows": int(len(merged)),
        "seed_files": seed_files,
        "ensemble_prediction": str(output_path.relative_to(ROOT)),
        "ensemble_prediction_sha256": sha256(output_path),
    }
    manifest_path = OUTPUT / "ensemble_freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

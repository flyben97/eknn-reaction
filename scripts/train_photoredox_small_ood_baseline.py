#!/usr/bin/env python3
"""Train a reproducible small-data baseline for a photoredox OOD task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEED = 42
FP_SIZE = 2048
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)


def fingerprint(smiles: str) -> tuple[np.ndarray, DataStructs.ExplicitBitVect]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    bit_vector = MORGAN.GetFingerprint(mol)
    array = np.zeros(FP_SIZE, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array, bit_vector


def featurize(frame: pd.DataFrame) -> tuple[np.ndarray, list[tuple]]:
    rows: list[np.ndarray] = []
    bit_pairs: list[tuple] = []
    for amine, arene in zip(frame["amine_smiles"], frame["arene_smiles"]):
        amine_array, amine_bits = fingerprint(amine)
        arene_array, arene_bits = fingerprint(arene)
        rows.append(np.concatenate([amine_array, arene_array]))
        bit_pairs.append((amine_bits, arene_bits))
    return np.asarray(rows), bit_pairs


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(target, prediction)),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
    }


def pair_nearest_similarity(query_pairs: list[tuple], train_pairs: list[tuple]) -> np.ndarray:
    values = []
    for query_amine, query_arene in query_pairs:
        closest = max(
            (
                DataStructs.TanimotoSimilarity(query_amine, train_amine)
                + DataStructs.TanimotoSimilarity(query_arene, train_arene)
            )
            / 2.0
            for train_amine, train_arene in train_pairs
        )
        values.append(closest)
    return np.asarray(values)


def distribution(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "n": int(len(frame)),
        "yield_mean": float(frame["Yield"].mean()),
        "yield_std": float(frame["Yield"].std()),
        "yield_min": float(frame["Yield"].min()),
        "yield_max": float(frame["Yield"].max()),
    }


def identity_features(
    reference: pd.DataFrame, query: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Build an additive two-role incidence matrix without test-only levels."""
    amines = {value: index for index, value in enumerate(sorted(reference["amine_smiles"].unique()))}
    arenes = {value: index for index, value in enumerate(sorted(reference["arene_smiles"].unique()))}

    def encode(frame: pd.DataFrame) -> np.ndarray:
        result = np.zeros((len(frame), len(amines) + len(arenes)), dtype=float)
        for row_number, row in enumerate(frame.itertuples(index=False)):
            if row.amine_smiles not in amines or row.arene_smiles not in arenes:
                raise ValueError("Held-out identity is absent from the fitting split")
            result[row_number, amines[row.amine_smiles]] = 1.0
            result[row_number, len(amines) + arenes[row.arene_smiles]] = 1.0
        return result

    return encode(reference), encode(query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("data/photoredox_ood_small/02_arene_amination_pair_ood"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.task_dir / "baseline_extratrees"
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {
        split: pd.read_csv(args.task_dir / f"{split}.csv")
        for split in ("train", "valid", "test")
    }
    arrays: dict[str, np.ndarray] = {}
    bit_pairs: dict[str, list[tuple]] = {}
    targets: dict[str, np.ndarray] = {}
    for split, frame in frames.items():
        arrays[split], bit_pairs[split] = featurize(frame)
        targets[split] = frame["Yield"].to_numpy(dtype=float)

    grid = []
    best_model = None
    best_params = None
    best_valid_r2 = -np.inf
    for min_samples_leaf in (1, 2, 4, 8):
        for max_features in (0.25, 0.5, 1.0):
            model = ExtraTreesRegressor(
                n_estimators=800,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=SEED,
                n_jobs=-1,
            )
            model.fit(arrays["train"], targets["train"])
            prediction = model.predict(arrays["valid"])
            score = metrics(targets["valid"], prediction)
            params = {
                "min_samples_leaf": min_samples_leaf,
                "max_features": max_features,
            }
            grid.append({**params, **score})
            if score["r2"] > best_valid_r2:
                best_valid_r2 = score["r2"]
                best_model = model
                best_params = params

    assert best_model is not None and best_params is not None
    valid_prediction = best_model.predict(arrays["valid"])

    identity_train_x, identity_valid_x = identity_features(
        frames["train"], frames["valid"]
    )
    best_identity_alpha = None
    best_identity_valid_r2 = -np.inf
    best_identity_valid_prediction = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha)
        model.fit(identity_train_x, targets["train"])
        prediction = model.predict(identity_valid_x)
        score = r2_score(targets["valid"], prediction)
        if score > best_identity_valid_r2:
            best_identity_valid_r2 = float(score)
            best_identity_alpha = alpha
            best_identity_valid_prediction = prediction
    assert best_identity_alpha is not None
    assert best_identity_valid_prediction is not None

    train_valid_x = np.concatenate([arrays["train"], arrays["valid"]])
    train_valid_y = np.concatenate([targets["train"], targets["valid"]])
    final_model = ExtraTreesRegressor(
        n_estimators=1600,
        random_state=SEED,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(train_valid_x, train_valid_y)
    test_prediction = final_model.predict(arrays["test"])
    train_valid_frame = pd.concat(
        [frames["train"], frames["valid"]], ignore_index=True
    )
    identity_train_valid_x, identity_test_x = identity_features(
        train_valid_frame, frames["test"]
    )
    identity_model = Ridge(alpha=best_identity_alpha)
    identity_model.fit(identity_train_valid_x, train_valid_y)
    identity_test_prediction = identity_model.predict(identity_test_x)
    mean_prediction = np.full(len(frames["test"]), train_valid_y.mean())
    nearest = pair_nearest_similarity(
        bit_pairs["test"], bit_pairs["train"] + bit_pairs["valid"]
    )

    report = {
        "model": "ExtraTrees on concatenated role-specific Morgan fingerprints",
        "fingerprint": {"radius": 2, "n_bits_per_role": FP_SIZE},
        "seed": SEED,
        "selected_params": best_params,
        "split_distributions": {
            split: distribution(frame) for split, frame in frames.items()
        },
        "validation_selected_model": metrics(targets["valid"], valid_prediction),
        "test_retrained_model": metrics(targets["test"], test_prediction),
        "identity_additive_ridge": {
            "selected_alpha": best_identity_alpha,
            "validation": metrics(
                targets["valid"], best_identity_valid_prediction
            ),
            "test_retrained": metrics(
                targets["test"], identity_test_prediction
            ),
        },
        "test_train_mean_baseline": metrics(targets["test"], mean_prediction),
        "test_nearest_pair_similarity": {
            "mean": float(nearest.mean()),
            "median": float(np.median(nearest)),
            "min": float(nearest.min()),
            "max": float(nearest.max()),
        },
        "selection_note": (
            "Hyperparameters were selected only on validation. The final model "
            "was retrained on train+validation and evaluated once on test."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame(grid).sort_values("r2", ascending=False).to_csv(
        output_dir / "validation_grid.csv", index=False
    )
    valid_output = frames["valid"][["reaction_id", "Yield"]].copy()
    valid_output["prediction"] = valid_prediction
    valid_output.to_csv(output_dir / "valid_predictions.csv", index=False)
    test_output = frames["test"][["reaction_id", "Yield"]].copy()
    test_output["prediction"] = test_prediction
    test_output["identity_ridge_prediction"] = identity_test_prediction
    test_output["nearest_pair_similarity"] = nearest
    test_output.to_csv(output_dir / "test_predictions.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

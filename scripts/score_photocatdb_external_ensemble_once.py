#!/usr/bin/env python3
"""Score the frozen PhotoCatDB ensemble exactly once."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "data/photoredox_external/photocatdb_r2_confirmatory_v1"
RESULT = ROOT / "results/photocatdb_external_affine_eknn_ensemble"
SEEDS = (42, 123, 2025)
MODELS = (
    "chemical_base",
    "paper_affine",
    "paper_1nn",
    "affine_eknn_candidate",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(target: np.ndarray, prediction: np.ndarray, papers: np.ndarray) -> dict:
    absolute = np.abs(target - prediction)
    paper_mae = pd.DataFrame({"paper": papers, "absolute": absolute}).groupby(
        "paper"
    )["absolute"].mean()
    return {
        "r2": float(r2_score(target, prediction)),
        "pearson_r": float(np.corrcoef(target, prediction)[0, 1]),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "bias": float(np.mean(prediction - target)),
        "p90_abs_error": float(np.quantile(absolute, 0.90)),
        "macro_paper_mae": float(paper_mae.mean()),
    }


def cluster_bootstrap(
    frame: pd.DataFrame, *, replicates: int = 10000, seed: int = 20260828
) -> dict:
    groups = [group for _, group in frame.groupby("paper_group", sort=True)]
    generator = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=float)
    for index in range(replicates):
        selected = generator.integers(0, len(groups), size=len(groups))
        base_error = []
        candidate_error = []
        for group_index in selected:
            group = groups[int(group_index)]
            target = group["Yield"].to_numpy(dtype=float)
            base_error.append(
                np.abs(target - group["chemical_base"].to_numpy(dtype=float))
            )
            candidate_error.append(
                np.abs(
                    target
                    - group["affine_eknn_candidate"].to_numpy(dtype=float)
                )
            )
        differences[index] = np.concatenate(candidate_error).mean() - np.concatenate(
            base_error
        ).mean()
    return {
        "estimand": "candidate_minus_chemical_base_micro_MAE",
        "replicates": replicates,
        "seed": seed,
        "estimate": float(
            np.mean(
                np.abs(frame["Yield"] - frame["affine_eknn_candidate"])
                - np.abs(frame["Yield"] - frame["chemical_base"])
            )
        ),
        "ci95": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "probability_improves": float(np.mean(differences < 0)),
    }


def main() -> None:
    sentinel = RESULT / "SCORED_ONCE"
    if sentinel.exists():
        raise RuntimeError("Locked PhotoCatDB labels have already been scored")
    freeze_path = RESULT / "ensemble_freeze_manifest.json"
    materialization_path = TASK / "materialization_manifest.json"
    prediction_path = RESULT / "target_predictions_ensemble_unscored.csv"
    label_path = TASK / "target_test_labels.locked.csv"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
    if freeze["status"] != "ensemble_frozen_before_locked_labels_are_loaded":
        raise RuntimeError("Ensemble was not frozen before scoring")
    if sha256(prediction_path) != freeze["ensemble_prediction_sha256"]:
        raise RuntimeError("Ensemble prediction hash mismatch")
    if sha256(label_path) != materialization["target_test_labels_sha256"]:
        raise RuntimeError("Locked label hash mismatch")

    prediction = pd.read_csv(prediction_path)
    labels = pd.read_csv(label_path)
    scored = prediction.merge(labels, on="reaction_id", validate="one_to_one")
    if len(scored) != freeze["rows"]:
        raise RuntimeError("Prediction/label row count mismatch")
    target = scored["Yield"].to_numpy(dtype=float)
    papers = scored["paper_group"].astype(str).to_numpy()
    ensemble_metrics = {
        model: metric(target, scored[model].to_numpy(dtype=float), papers)
        for model in MODELS
    }

    seed_metrics = []
    for seed in SEEDS:
        seed_path = (
            ROOT
            / f"results/photocatdb_external_affine_eknn_s{seed}/target_predictions_unscored.csv"
        )
        seed_prediction = pd.read_csv(seed_path)[
            ["reaction_id", "affine_eknn_candidate"]
        ]
        joined = labels.merge(seed_prediction, on="reaction_id", validate="one_to_one")
        score = metric(
            joined["Yield"].to_numpy(dtype=float),
            joined["affine_eknn_candidate"].to_numpy(dtype=float),
            scored.set_index("reaction_id").loc[
                joined["reaction_id"], "paper_group"
            ].astype(str).to_numpy(),
        )
        seed_metrics.append({"seed": seed, **score})
    pd.DataFrame(seed_metrics).to_csv(RESULT / "seed_metrics.csv", index=False)

    calibration = pd.read_csv(TASK / "calibration.csv")
    seen_keys = set(
        calibration["paper_group"].astype(str)
        + "|"
        + calibration["substrate_smiles"].fillna("").astype(str)
    )
    scored["identity_seen"] = (
        scored["paper_group"].astype(str)
        + "|"
        + scored["substrate_smiles"].fillna("").astype(str)
    ).isin(seen_keys)
    subgroup = {}
    for name, group in scored.groupby("identity_seen", sort=True):
        subgroup["identity_seen" if name else "identity_unseen"] = {
            "n": int(len(group)),
            **metric(
                group["Yield"].to_numpy(dtype=float),
                group["affine_eknn_candidate"].to_numpy(dtype=float),
                group["paper_group"].astype(str).to_numpy(),
            ),
        }

    per_paper_rows = []
    for paper, group in scored.groupby("paper_group", sort=True):
        row = {"paper_group": paper, "n": int(len(group))}
        y = group["Yield"].to_numpy(dtype=float)
        for model in MODELS:
            p = group[model].to_numpy(dtype=float)
            row[f"{model}_mae"] = float(np.mean(np.abs(y - p)))
            row[f"{model}_r2"] = float(r2_score(y, p)) if len(y) >= 2 else np.nan
            row[f"{model}_pearson_r"] = (
                float(np.corrcoef(y, p)[0, 1])
                if len(y) >= 2 and np.std(y) > 0 and np.std(p) > 0
                else np.nan
            )
        per_paper_rows.append(row)
    pd.DataFrame(per_paper_rows).to_csv(RESULT / "per_paper_metrics.csv", index=False)
    bootstrap = cluster_bootstrap(scored)
    base = ensemble_metrics["chemical_base"]
    candidate = ensemble_metrics["affine_eknn_candidate"]
    gates = {
        "r2_at_least_0p60": candidate["r2"] >= 0.60,
        "pearson_r_at_least_0p80": candidate["pearson_r"] >= 0.80,
        "mae_at_most_0p14": candidate["mae"] <= 0.14,
        "mae_improvement_at_least_0p01": base["mae"] - candidate["mae"] >= 0.01,
        "macro_paper_mae_improvement_at_least_0p01": (
            base["macro_paper_mae"] - candidate["macro_paper_mae"] >= 0.01
        ),
        "p90_at_most_0p30": candidate["p90_abs_error"] <= 0.30,
        "bootstrap_upper_below_zero": bootstrap["ci95"][1] < 0,
        "two_seeds_r2_0p55_r_0p75": sum(
            row["r2"] >= 0.55 and row["pearson_r"] >= 0.75
            for row in seed_metrics
        )
        >= 2,
    }
    payload = {
        "status": "locked_external_confirmation_scored_once",
        "dataset": "PhotoCatDB conditional subset; one row per canonical reaction",
        "rows": int(len(scored)),
        "papers": int(scored["paper_group"].nunique()),
        "ensemble_metrics": ensemble_metrics,
        "seed_metrics": seed_metrics,
        "identity_subgroups": subgroup,
        "paper_cluster_bootstrap": bootstrap,
        "gates": gates,
        "absolute_target_pass": bool(
            gates["r2_at_least_0p60"] and gates["pearson_r_at_least_0p80"]
        ),
        "all_promotion_gates_pass": bool(all(gates.values())),
        "hashes": {
            "ensemble_freeze_manifest": sha256(freeze_path),
            "ensemble_prediction": sha256(prediction_path),
            "locked_labels": sha256(label_path),
            "scorer_script": sha256(Path(__file__)),
        },
    }
    scored.to_csv(RESULT / "target_predictions_scored.csv", index=False)
    summary_path = RESULT / "score_summary.json"
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sentinel.write_text(sha256(summary_path) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

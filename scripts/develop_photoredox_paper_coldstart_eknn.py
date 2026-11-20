#!/usr/bin/env python3
"""Freeze a paper-cold-start residual blend on previously opened dev tasks."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from develop_photoredox_domain_gated_eknn import (
    domain_correction,
    residual_statistics,
)


def metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    target = frame["Yield"].to_numpy(dtype=float)
    error = np.abs(prediction - target)
    return {
        "mae": float(error.mean()),
        "p90_abs_error": float(np.quantile(error, 0.90)),
        "macro_paper_mae": float(
            pd.DataFrame(
                {"error": error, "paper_group": frame["paper_group"].astype(str)}
            ).groupby("paper_group")["error"].mean().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/photoredox_paper_coldstart_development"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_specs = [
        ("old_s42", Path("results/photoredox_full_cf_hierarchical_eknn_s42"), "target_global_b112"),
        ("old_s123", Path("results/photoredox_full_cf_hierarchical_eknn_s123"), "target_global_b112"),
        ("old_s2025", Path("results/photoredox_full_cf_hierarchical_eknn_s2025"), "target_global_b112"),
        ("scaffold_confirm_s42", Path("results/photoredox_domain_gated_confirmatory_s42"), "target_global"),
    ]
    runs = []
    for run_name, run_dir, chemistry_column in run_specs:
        calibration = pd.read_csv(
            run_dir / "calibration_leave_identity_out_predictions.csv"
        )
        test = pd.read_csv(run_dir / "target_predictions_scored.csv")
        test = test.loc[test["calibration_paper_seen"].astype(bool)].reset_index(
            drop=True
        )
        runs.append(
            {
                "name": run_name,
                "calibration": calibration,
                "test": test,
                "chemistry_column": chemistry_column,
                "statistics": residual_statistics(calibration),
            }
        )

    rows = []
    for values in itertools.product(
        ("paper", "paper_stratum"),
        (0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
        (1.0, 2.0, 5.0),
        (0.0, 0.25, 0.50, 0.75, 1.0),
        (0.0, 0.25, 0.50, 0.75, 1.0),
    ):
        scope, paper_shrinkage, stratum_shrinkage, paper_weight, chemistry_weight = values
        run_metrics = []
        for run in runs:
            frame = run["test"]
            paper_value, seen = domain_correction(
                frame,
                run["statistics"],
                scope=scope,
                paper_shrinkage=paper_shrinkage,
                stratum_shrinkage=stratum_shrinkage,
            )
            if not seen.all():
                raise RuntimeError("Paper-seen development filter is inconsistent")
            base = frame["base"].to_numpy(dtype=float)
            chemistry = frame[run["chemistry_column"]].to_numpy(dtype=float) - base
            prediction = base + paper_weight * paper_value + chemistry_weight * chemistry
            base_metrics = metrics(frame, base)
            candidate_metrics = metrics(frame, prediction)
            run_metrics.append(
                {
                    "delta_mae": candidate_metrics["mae"] - base_metrics["mae"],
                    "delta_macro_paper_mae": (
                        candidate_metrics["macro_paper_mae"]
                        - base_metrics["macro_paper_mae"]
                    ),
                    "delta_p90": (
                        candidate_metrics["p90_abs_error"]
                        - base_metrics["p90_abs_error"]
                    ),
                }
            )
        result = pd.DataFrame(run_metrics)
        rows.append(
            {
                "scope": scope,
                "paper_shrinkage": paper_shrinkage,
                "stratum_shrinkage": stratum_shrinkage,
                "paper_weight": paper_weight,
                "chemistry_weight": chemistry_weight,
                "mean_delta_mae": float(result["delta_mae"].mean()),
                "mean_delta_macro_paper_mae": float(
                    result["delta_macro_paper_mae"].mean()
                ),
                "mean_delta_p90": float(result["delta_p90"].mean()),
                "improved_runs": int((result["delta_mae"] < 0).sum()),
                "min_run_improvement": float(-result["delta_mae"].max()),
            }
        )
    grid = pd.DataFrame(rows)
    eligible = grid.loc[
        (grid["improved_runs"] == len(runs))
        & (grid["mean_delta_p90"] <= 0.01)
    ].sort_values(
        ["mean_delta_macro_paper_mae", "mean_delta_mae", "mean_delta_p90"]
    )
    if eligible.empty:
        raise RuntimeError("No paper-cold-start blend satisfies development guards")
    selected = eligible.iloc[0].to_dict()
    grid.sort_values(
        ["mean_delta_macro_paper_mae", "mean_delta_mae"]
    ).to_csv(args.output_dir / "development_grid.csv", index=False)

    individual = []
    for run in runs:
        frame = run["test"]
        paper_value, _ = domain_correction(
            frame,
            run["statistics"],
            scope=str(selected["scope"]),
            paper_shrinkage=float(selected["paper_shrinkage"]),
            stratum_shrinkage=float(selected["stratum_shrinkage"]),
        )
        base = frame["base"].to_numpy(dtype=float)
        chemistry = frame[run["chemistry_column"]].to_numpy(dtype=float) - base
        prediction = (
            base
            + float(selected["paper_weight"]) * paper_value
            + float(selected["chemistry_weight"]) * chemistry
        )
        base_metrics = metrics(frame, base)
        candidate_metrics = metrics(frame, prediction)
        individual.append(
            {
                "run": run["name"],
                "n": len(frame),
                **candidate_metrics,
                "delta_mae": candidate_metrics["mae"] - base_metrics["mae"],
                "delta_macro_paper_mae": (
                    candidate_metrics["macro_paper_mae"]
                    - base_metrics["macro_paper_mae"]
                ),
            }
        )
    pd.DataFrame(individual).to_csv(
        args.output_dir / "selected_recipe_run_metrics.csv", index=False
    )
    recipe = {
        "status": "frozen_from_opened_development_sets",
        "confirmation_protocol": "paper-coldstart-HR-eKNN-confirmatory-v1",
        "development_runs": [spec[0] for spec in run_specs],
        "development_scope": "calibration-paper-seen rows only",
        "selected": {
            **selected,
            "chemistry_method": "target_global_b112",
            "unseen_global_weight": 0.0,
        },
        "individual_run_metrics": individual,
        "new_confirmation_test_loaded": False,
    }
    (args.output_dir / "frozen_recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(recipe, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

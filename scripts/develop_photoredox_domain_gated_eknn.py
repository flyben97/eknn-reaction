#!/usr/bin/env python3
"""Select a paper-domain gate using only the opened Photoredox development set."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def metric_row(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, float]:
    target = frame["Yield"].to_numpy(dtype=float)
    absolute = np.abs(prediction - target)
    stratum = (
        frame["substrate_scaffold"].astype(str)
        + "||"
        + frame["catalyst_group"].astype(str)
    )
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "p90_abs_error": float(np.quantile(absolute, 0.90)),
        "macro_stratum_mae": float(
            pd.DataFrame({"error": absolute, "stratum": stratum})
            .groupby("stratum")["error"].mean().mean()
        ),
    }


def residual_statistics(calibration: pd.DataFrame) -> dict:
    work = calibration.copy()
    work["residual"] = work["Yield"] - work["base"]
    paper = work.groupby("paper_group")["residual"].agg(["sum", "count"])
    paper_stratum = work.groupby(
        ["paper_group", "substrate_scaffold", "catalyst_group"]
    )["residual"].agg(["sum", "count"])
    return {
        "global": float(work["residual"].mean()),
        "paper_sum": paper["sum"].to_dict(),
        "paper_count": paper["count"].to_dict(),
        "paper_stratum_sum": paper_stratum["sum"].to_dict(),
        "paper_stratum_count": paper_stratum["count"].to_dict(),
    }


def domain_correction(
    frame: pd.DataFrame,
    statistics: dict,
    *,
    scope: str,
    paper_shrinkage: float,
    stratum_shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    corrections = np.zeros(len(frame), dtype=float)
    seen = np.zeros(len(frame), dtype=bool)
    global_value = statistics["global"]
    for index, (_, row) in enumerate(frame.iterrows()):
        paper = row["paper_group"]
        n_paper = float(statistics["paper_count"].get(paper, 0.0))
        seen[index] = n_paper > 0
        paper_value = (
            statistics["paper_sum"].get(paper, 0.0)
            + paper_shrinkage * global_value
        ) / max(n_paper + paper_shrinkage, 1e-12)
        if not seen[index]:
            paper_value = global_value
        value = paper_value
        if scope == "paper_stratum" and seen[index]:
            key = (
                paper,
                row["substrate_scaffold"],
                row["catalyst_group"],
            )
            n_stratum = float(
                statistics["paper_stratum_count"].get(key, 0.0)
            )
            if n_stratum > 0:
                value = (
                    statistics["paper_stratum_sum"].get(key, 0.0)
                    + stratum_shrinkage * paper_value
                ) / (n_stratum + stratum_shrinkage)
        corrections[index] = value
    return corrections, seen


def candidate_prediction(
    test: pd.DataFrame,
    domain_value: np.ndarray,
    paper_seen: np.ndarray,
    *,
    paper_weight: float,
    chemistry_weight: float,
    chemistry_method: str,
    unseen_global_weight: float,
    global_value: float,
) -> np.ndarray:
    base = test["base"].to_numpy(dtype=float)
    chemistry = test[chemistry_method].to_numpy(dtype=float) - base
    correction = paper_weight * domain_value + chemistry_weight * chemistry
    correction[~paper_seen] = unseen_global_weight * global_value
    return base + correction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2025])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/photoredox_domain_gated_development"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for seed in args.seeds:
        run_dir = args.results_root / f"photoredox_full_cf_hierarchical_eknn_s{seed}"
        calibration = pd.read_csv(
            run_dir / "calibration_leave_identity_out_predictions.csv"
        )
        test = pd.read_csv(run_dir / "target_predictions_scored.csv")
        runs[seed] = {
            "calibration": calibration,
            "test": test,
            "statistics": residual_statistics(calibration),
        }

    grid_rows = []
    parameter_grid = itertools.product(
        ("paper", "paper_stratum"),
        (0.5, 1.0, 2.0, 5.0, 10.0),
        (0.5, 1.0, 2.0, 5.0),
        (0.25, 0.50, 0.75, 1.0),
        (0.0, 0.25, 0.50, 0.75),
        ("target_global_b112", "cf_hierarchical_b112"),
        (0.5, 1.0),
    )
    for parameters in parameter_grid:
        (
            scope,
            paper_shrinkage,
            stratum_shrinkage,
            paper_weight,
            chemistry_weight,
            chemistry_method,
            unseen_global_weight,
        ) = parameters
        seed_metrics = []
        for seed, run in runs.items():
            test = run["test"]
            domain_value, paper_seen = domain_correction(
                test,
                run["statistics"],
                scope=scope,
                paper_shrinkage=paper_shrinkage,
                stratum_shrinkage=stratum_shrinkage,
            )
            prediction = candidate_prediction(
                test,
                domain_value,
                paper_seen,
                paper_weight=paper_weight,
                chemistry_weight=chemistry_weight,
                chemistry_method=chemistry_method,
                unseen_global_weight=unseen_global_weight,
                global_value=run["statistics"]["global"],
            )
            base_metrics = metric_row(test, test["base"].to_numpy(dtype=float))
            metrics = metric_row(test, prediction)
            unseen_metrics = metric_row(
                test.loc[~paper_seen].reset_index(drop=True), prediction[~paper_seen]
            )
            unseen_base = metric_row(
                test.loc[~paper_seen].reset_index(drop=True),
                test.loc[~paper_seen, "base"].to_numpy(dtype=float),
            )
            seed_metrics.append(
                {
                    "seed": seed,
                    **metrics,
                    "delta_mae": metrics["mae"] - base_metrics["mae"],
                    "delta_macro_stratum_mae": (
                        metrics["macro_stratum_mae"]
                        - base_metrics["macro_stratum_mae"]
                    ),
                    "delta_p90": (
                        metrics["p90_abs_error"] - base_metrics["p90_abs_error"]
                    ),
                    "paper_unseen_delta_mae": (
                        unseen_metrics["mae"] - unseen_base["mae"]
                    ),
                }
            )
        summary = pd.DataFrame(seed_metrics)
        grid_rows.append(
            {
                "scope": scope,
                "paper_shrinkage": paper_shrinkage,
                "stratum_shrinkage": stratum_shrinkage,
                "paper_weight": paper_weight,
                "chemistry_weight": chemistry_weight,
                "chemistry_method": chemistry_method,
                "unseen_global_weight": unseen_global_weight,
                "mean_mae": float(summary["mae"].mean()),
                "mean_macro_stratum_mae": float(
                    summary["macro_stratum_mae"].mean()
                ),
                "mean_delta_mae": float(summary["delta_mae"].mean()),
                "mean_delta_macro_stratum_mae": float(
                    summary["delta_macro_stratum_mae"].mean()
                ),
                "mean_delta_p90": float(summary["delta_p90"].mean()),
                "mean_paper_unseen_delta_mae": float(
                    summary["paper_unseen_delta_mae"].mean()
                ),
                "improved_seeds": int((summary["delta_mae"] < 0).sum()),
                "min_seed_improvement": float(-summary["delta_mae"].max()),
            }
        )
    grid = pd.DataFrame(grid_rows)
    eligible = grid.loc[
        (grid["improved_seeds"] == len(args.seeds))
        & (grid["mean_paper_unseen_delta_mae"] <= 0.0)
        & (grid["mean_delta_p90"] <= 0.01)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No domain-gated recipe satisfies development guardrails")
    eligible = eligible.sort_values(
        ["mean_delta_macro_stratum_mae", "mean_delta_mae", "mean_delta_p90"]
    )
    selected = eligible.iloc[0].to_dict()
    grid.sort_values(
        ["mean_delta_macro_stratum_mae", "mean_delta_mae"]
    ).to_csv(args.output_dir / "development_grid.csv", index=False)

    individual_rows = []
    for seed, run in runs.items():
        test = run["test"]
        domain_value, paper_seen = domain_correction(
            test,
            run["statistics"],
            scope=selected["scope"],
            paper_shrinkage=float(selected["paper_shrinkage"]),
            stratum_shrinkage=float(selected["stratum_shrinkage"]),
        )
        prediction = candidate_prediction(
            test,
            domain_value,
            paper_seen,
            paper_weight=float(selected["paper_weight"]),
            chemistry_weight=float(selected["chemistry_weight"]),
            chemistry_method=selected["chemistry_method"],
            unseen_global_weight=float(selected["unseen_global_weight"]),
            global_value=run["statistics"]["global"],
        )
        output = test[
            ["reaction_id", "Yield", "paper_group", "substrate_smiles",
             "substrate_scaffold", "catalyst_group", "base"]
        ].copy()
        output["paper_seen"] = paper_seen
        output["domain_correction"] = domain_value
        output["domain_gated_prediction"] = prediction
        output.to_csv(
            args.output_dir / f"development_predictions_s{seed}.csv", index=False
        )
        base_metrics = metric_row(test, test["base"].to_numpy(dtype=float))
        candidate_metrics = metric_row(test, prediction)
        individual_rows.append(
            {"seed": seed, **candidate_metrics,
             "delta_mae": candidate_metrics["mae"] - base_metrics["mae"],
             "delta_macro_stratum_mae": (
                 candidate_metrics["macro_stratum_mae"]
                 - base_metrics["macro_stratum_mae"]
             )}
        )
    pd.DataFrame(individual_rows).to_csv(
        args.output_dir / "selected_recipe_seed_metrics.csv", index=False
    )
    recipe = {
        "status": "frozen_from_opened_development_set",
        "development_seeds": args.seeds,
        "selection_objective": (
            "minimum mean macro-stratum MAE subject to all seeds improving, "
            "paper-unseen mean MAE non-worsening, and mean p90 delta <= 0.01"
        ),
        "selected": selected,
        "individual_seed_metrics": individual_rows,
        "new_confirmation_test_loaded": False,
    }
    (args.output_dir / "frozen_recipe.json").write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(recipe, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

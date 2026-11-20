#!/usr/bin/env python3
"""Compare GNN, fingerprint and decision-tree leaf retrieval on held-out HiTEA screens.

Select the correction within each calibration fold using group-blocked leave-one-out
validation. Locked target labels stay masked.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPEC_V2 = importlib.util.spec_from_file_location(
    "develop_hitea_target_memory_v2", ROOT / "scripts" / "develop_hitea_target_memory_v2.py"
)
v2 = importlib.util.module_from_spec(SPEC_V2)
assert SPEC_V2.loader is not None
SPEC_V2.loader.exec_module(v2)

SEEDS = (42, 123, 2025)
GNN_ROOT = Path("results/hitea_gnn_development")
OUTPUT = Path("results/hitea_gnn_target_memory_development_v1")
SPACES = ("gnn", "morgan", "leaf")
BASE_VARIANTS = ("base_gnn", "base_zi")
SIMPLE_SUFFIXES = ("", "__global_residual", "__affine")

SELECTION_RULE = (
    "Identical frozen rule as v2 run D: per fold and budget every registered "
    "candidate (no-retrieval: base in {base_gnn, base_zi} x {plain, "
    "global_residual, affine}; retrieval grid: base in {base_gnn, base_zi}, "
    "similarity in {gnn, morgan, leaf}, k in {1,3,5,8,12}, temperature in "
    "{0.05,0.1,0.25}, residual scale in {0.5,1.0}, residual/label blend in "
    "{0,.25,.5,.75,1}) is scored by group-blocked leave-one-out MAE on the "
    "calibration memory (held-out row's whole exact_input_group excluded, "
    "self-exclusion fallback). If the best retrieval configuration does not "
    "beat the best no-retrieval method by at least a 2% relative margin, fall "
    "back to that method. Otherwise average the predictions of every "
    "retrieval configuration whose LOO MAE is within 5% of the best (at most "
    "25 members). Ties resolve toward the lexicographically earlier method "
    "name. Development query labels are never used for selection. The GNN "
    "base and embeddings come from a frozen model trained on the "
    "global-exclusion source (all 26 target screens and their "
    "exact_input_groups removed), one model per seed."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/hitea_retrieval_shift/v1"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.data_root / "materialization_manifest.json").read_text())
    table = pd.read_csv(args.data_root / "development_modeling_table.csv", low_memory=False)
    locked = pd.read_csv(args.data_root / "locked_test_registry.csv")
    locked_ids = set(locked["record_id"].astype(str))
    if int(table["Yield"].isna().sum()) != len(locked_ids) or set(
        table.loc[table["Yield"].isna(), "record_id"].astype(str)
    ) != locked_ids:
        raise AssertionError("Masked labels and locked test registry differ")
    fold_index = pd.read_csv(args.data_root / "fold_index.csv")

    seed_artifacts = {}
    for seed in args.seeds:
        seed_dir = GNN_ROOT / f"s{seed}"
        training_manifest = json.loads(
            (seed_dir / "training_manifest.json").read_text(encoding="utf-8")
        )
        if training_manifest.get("locked_target_test_scored") is not False:
            raise AssertionError("GNN training manifest must record locked labels unscored")
        if training_manifest["protocol"] != "hitea-gnn-global-exclusion-base-v1":
            raise AssertionError("Unexpected GNN training protocol")
        if int(training_manifest["n_target_screens_excluded"]) != 26:
            raise AssertionError("GNN source must exclude all 26 target screens")
        if int(training_manifest["n_target_exact_input_groups_excluded"]) <= 0:
            raise AssertionError("GNN source must exclude target exact_input_groups")
        # record_id is stored as an object array of str, hence allow_pickle.
        npz = np.load(seed_dir / "embeddings.npz", allow_pickle=True)
        record_ids = npz["record_id"].astype(str)
        seed_artifacts[seed] = {
            "embedding": dict(zip(record_ids, npz["embedding"])),
            "base": {record_id: float(value) for record_id, value in zip(record_ids, npz["base_pred"])},
        }
        print(f"loaded seed {seed}: {len(seed_artifacts[seed]['embedding'])} embeddings", flush=True)

    morgan = v2.MorganSpace()
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    selection_rows: list[dict] = []
    fold_settings: list[dict] = []

    for fold_number, fold in enumerate(fold_index.itertuples(index=False), start=1):
        fold_id = str(fold.fold_id)
        assignment = pd.read_csv(args.data_root / "folds" / fold_id / "target_assignment.csv")
        target = table.loc[table["record_id"].isin(assignment["record_id"])].merge(
            assignment, on=["record_id", "exact_input_group", "reaction_core_group"],
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

        fold_space = v2.fit_feature_space(source)
        train_x = v2.apply_feature_space(fold_space, source)
        target_x = v2.apply_feature_space(fold_space, target_development)
        model_seed = v2.stable_seed(v2.SEED, fold_id)
        base_v1 = v2.fit_base_model(train_x, source["Yield"].to_numpy(float), model_seed)
        classifier_zi, regressor_zi = v2.fit_zero_inflated_model(
            train_x, source["Yield"].to_numpy(float), model_seed
        )

        target_base_zi, _ = v2.zero_inflated_predict(classifier_zi, regressor_zi, target_x)
        target_leaf = base_v1.predict(target_x, pred_leaf=True).astype(np.int32)
        leaf_matrix = v2.leaf_overlap_matrix(target_leaf)
        morgan_matrix = morgan.similarity(target_development, target_development)
        position_of = pd.Series(
            np.arange(len(target_development)), index=target_development["record_id"].to_numpy()
        )
        query_pos = position_of.loc[fixed_query["record_id"].to_numpy()].to_numpy(int)
        query_y = fixed_query["Yield"].to_numpy(float)

        seed_states = {}
        for seed in args.seeds:
            dev_ids = target_development["record_id"].astype(str).to_numpy()
            artifacts = seed_artifacts[seed]
            missing = [r for r in dev_ids if r not in artifacts["embedding"]]
            if missing:
                raise AssertionError(f"{len(missing)} target rows lack seed {seed} embeddings")
            seed_states[seed] = {
                "embedding": np.stack([artifacts["embedding"][r] for r in dev_ids]),
                "base_gnn": np.clip(
                    np.array([artifacts["base"][r] for r in dev_ids], dtype=float), 0.0, 100.0
                ),
            }

        fold_settings.append(
            {
                "fold_id": fold_id,
                "reaction_keyword": str(fold.reaction_keyword),
                "target_screen_id": str(fold.target_screen_id),
                "n_source_train": int(len(source)),
                "n_target_development": int(len(target_development)),
                "n_fixed_query": int(len(fixed_query)),
                "n_locked_test_unread": int(target["is_target_test"].sum()),
            }
        )

        for budget in v2.CALIBRATION_BUDGETS:
            memory = target_development.loc[target_development[f"is_calibration_{budget:02d}"]]
            if memory.empty:
                raise ValueError(f"Empty {budget}% memory in {fold_id}")
            if set(memory["exact_input_group"]) & set(fixed_query["exact_input_group"]):
                raise AssertionError(f"Memory-query exact input leakage in {fold_id}/{budget}")
            cal_positions = memory.index.to_numpy(int)
            cal_y = target_development.iloc[cal_positions]["Yield"].to_numpy(float)
            cal_group_ids = (
                target_development.iloc[cal_positions]["exact_input_group"].to_numpy()
            )
            n_cal = len(cal_positions)

            for seed in args.seeds:
                state = seed_states[seed]
                cal_base = {
                    "base_gnn": state["base_gnn"][cal_positions],
                    "base_zi": target_base_zi[cal_positions],
                }
                query_base = {
                    "base_gnn": state["base_gnn"][query_pos],
                    "base_zi": target_base_zi[query_pos],
                }

                def gnn_similarity(rows: np.ndarray, _s=state) -> np.ndarray:
                    return np.clip(_s["embedding"][rows] @ _s["embedding"][cal_positions].T, -1.0, 1.0)

                # Per-space similarity matrices against the calibration memory.
                matrices = {
                    "gnn": {
                        "cal": gnn_similarity(cal_positions),
                        "query": gnn_similarity(query_pos),
                    },
                    "morgan": {
                        "cal": morgan_matrix[np.ix_(cal_positions, cal_positions)],
                        "query": morgan_matrix[np.ix_(query_pos, cal_positions)],
                    },
                    "leaf": {
                        "cal": leaf_matrix[np.ix_(cal_positions, cal_positions)],
                        "query": leaf_matrix[np.ix_(query_pos, cal_positions)],
                    },
                }

                simple_candidates, retrieval_candidates = [], []
                for variant in BASE_VARIANTS:
                    for suffix in SIMPLE_SUFFIXES:
                        method = f"{variant}{suffix}"
                        simple_candidates.append(
                            {
                                "method": method,
                                "loo_mae": v2.simple_loo_mae(
                                    method, cal_base, cal_y, cal_group_ids
                                ),
                                "config": {"kind": "simple", "method": method},
                            }
                        )
                for variant in BASE_VARIANTS:
                    for space in SPACES:
                        cal_matrix = matrices[space]["cal"]
                        for k in v2.K_GRID:
                            if k >= n_cal:
                                continue
                            for temperature in v2.TEMPERATURE_GRID:
                                local_label, local_residual = v2.loo_neighbour_components(
                                    cal_matrix, cal_base[variant], cal_y,
                                    k=k, temperature=temperature, group_ids=cal_group_ids,
                                )
                                per_scale = v2.retrieval_loo_mae(
                                    local_label, local_residual, cal_base[variant], cal_y
                                )
                                for scale, per_blend in per_scale.items():
                                    for blend, loo_mae in per_blend.items():
                                        retrieval_candidates.append(
                                            {
                                                "method": (
                                                    f"{variant}|{space}|k{k}|T{temperature}"
                                                    f"|s{scale}|b{blend}"
                                                ),
                                                "loo_mae": loo_mae,
                                                "config": {
                                                    "kind": "retrieval", "base": variant,
                                                    "space": space, "k": k,
                                                    "temperature": temperature,
                                                    "scale": scale, "blend": blend,
                                                },
                                            }
                                        )
                best_simple = min(simple_candidates, key=lambda c: (c["loo_mae"], c["method"]))
                best_retrieval = min(
                    retrieval_candidates, key=lambda c: (c["loo_mae"], c["method"])
                )
                use_retrieval = (
                    best_retrieval["loo_mae"]
                    <= best_simple["loo_mae"] * (1.0 - v2.FAMILY_GATE_MARGIN)
                )
                gate_note = {
                    "best_simple_method": best_simple["method"],
                    "best_simple_loo_mae": best_simple["loo_mae"],
                    "best_retrieval_loo_mae": best_retrieval["loo_mae"],
                    "gate_margin": v2.FAMILY_GATE_MARGIN,
                }
                if use_retrieval:
                    members = sorted(
                        (
                            c for c in retrieval_candidates
                            if c["loo_mae"]
                            <= best_retrieval["loo_mae"] * (1.0 + v2.ENSEMBLE_REL_TOLERANCE)
                        ),
                        key=lambda c: (c["loo_mae"], c["method"]),
                    )[: v2.ENSEMBLE_MAX_MEMBERS]
                    selected_method = (
                        f"ensemble_eps{v2.ENSEMBLE_REL_TOLERANCE}(n={len(members)})"
                    )
                    selection_rows.append(
                        {
                            "seed": seed, "fold_id": fold_id, "budget_percent": budget,
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
                            "seed": seed, "fold_id": fold_id, "budget_percent": budget,
                            "selected_method": selected_method, "family": "simple",
                            "loo_mae": best_simple["loo_mae"],
                            "best_member": best_simple["method"], "n_members": 1,
                            "members": best_simple["method"],
                            **gate_note,
                        }
                    )

                def member_prediction(config: dict, labels: np.ndarray) -> np.ndarray:
                    if config["kind"] == "simple":
                        method = config["method"]
                        if method in cal_base:
                            return query_base[method].copy()
                        if method.endswith("__global_residual"):
                            variant = method[: -len("__global_residual")]
                            offset = float((labels - cal_base[variant]).mean())
                            return query_base[variant] + offset
                        variant = method[: -len("__affine")]
                        return v2.affine_residual_prediction(
                            cal_base[variant], labels, query_base[variant]
                        )
                    local_label, local_residual, _ = v2.query_neighbour_components(
                        matrices[config["space"]]["query"],
                        cal_base[config["base"]], labels,
                        k=config["k"], temperature=config["temperature"],
                    )
                    return v2.blend_from_components(
                        query_base[config["base"]], local_label, local_residual,
                        scale=config["scale"], blend=config["blend"],
                    )

                member_predictions = [member_prediction(m["config"], cal_y) for m in members]
                selected = np.mean(member_predictions, axis=0)

                rng = np.random.default_rng(
                    v2.stable_seed(v2.SEED, fold_id, budget, seed, "gnnpermute")
                )
                permuted_y = rng.permutation(cal_y)
                if use_retrieval:
                    permuted = np.mean(
                        [member_prediction(m["config"], permuted_y) for m in members],
                        axis=0,
                    )
                else:
                    permuted = selected.copy()

                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "seed": seed,
                            "fold_id": fold_id,
                            "reaction_keyword": str(fold.reaction_keyword),
                            "target_screen_id": str(fold.target_screen_id),
                            "budget_percent": budget,
                            "record_id": fixed_query["record_id"].to_numpy(),
                            "exact_input_group": fixed_query["exact_input_group"].to_numpy(),
                            "Yield": query_y,
                            "memory_size": int(n_cal),
                            "selected_method": selected_method,
                            "prediction_base_gnn": query_base["base_gnn"],
                            "prediction_base_zi": query_base["base_zi"],
                            "prediction_selected": selected,
                            "prediction_selected_permuted": permuted,
                        }
                    )
                )
                for method, values in (
                    ("base_gnn", query_base["base_gnn"]),
                    ("base_zi", query_base["base_zi"]),
                    ("selected", selected),
                    ("selected_permuted", permuted),
                ):
                    values = np.asarray(values, dtype=float)
                    for subgroup, mask in (
                        ("all", np.ones(len(query_y), dtype=bool)),
                        ("zero", query_y == 0.0),
                        ("nonzero", query_y > 0.0),
                    ):
                        if not mask.any():
                            continue
                        metric_rows.append(
                            {
                                "seed": seed, "fold_id": fold_id,
                                "reaction_keyword": str(fold.reaction_keyword),
                                "target_screen_id": str(fold.target_screen_id),
                                "budget_percent": budget, "method": method,
                                "subgroup": subgroup,
                                **v2.metric_record(query_y[mask], values[mask]),
                            }
                        )
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
    v2.json_dump(
        args.output / "fold_settings.json",
        {"folds": fold_settings, "selection_rule": SELECTION_RULE},
    )

    summary_rows = []
    for seed in args.seeds:
        seed_pred = predictions.loc[predictions["seed"].eq(seed)]
        for (budget, method), part in metrics.loc[
            metrics["subgroup"].eq("all") & metrics["seed"].eq(seed)
        ].groupby(["budget_percent", "method"]):
            summary_rows.append(
                {
                    "scope": "seed_pooled", "seed": seed,
                    "reaction_keyword": "ALL", "budget_percent": int(budget),
                    "method": method,
                    **v2.metric_record(
                        seed_pred.loc[seed_pred["budget_percent"].eq(budget), "Yield"].to_numpy(float),
                        seed_pred.loc[
                            seed_pred["budget_percent"].eq(budget),
                            f"prediction_{method}",
                        ].to_numpy(float),
                    ),
                    "macro_fold_mae": float(part["mae"].mean()),
                }
            )
    for (budget, method), part in metrics.loc[
        metrics["subgroup"].eq("all")
        & metrics["method"].isin(["selected", "selected_permuted"])
    ].groupby(["budget_percent", "method"]):
        pooled_rows = []
        for seed in args.seeds:
            pred_part = predictions.loc[
                predictions["budget_percent"].eq(budget) & predictions["seed"].eq(seed)
            ]
            pooled_rows.append(
                (pred_part["Yield"].to_numpy(float),
                 pred_part[f"prediction_{method}"].to_numpy(float))
            )
        all_y = np.concatenate([row[0] for row in pooled_rows])
        all_p = np.concatenate([row[1] for row in pooled_rows])
        summary_rows.append(
            {
                "scope": "seed_concat_pooled", "seed": -1, "reaction_keyword": "ALL",
                "budget_percent": int(budget), "method": method,
                **v2.metric_record(all_y, all_p),
                "macro_fold_mae": float(part["mae"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output / "summary_metrics.csv", index=False)

    pooled20 = summary.loc[
        summary["scope"].eq("seed_concat_pooled") & summary["budget_percent"].eq(20)
    ].set_index("method")
    per_seed_mae = {
        seed: float(
            summary.loc[
                summary["scope"].eq("seed_pooled") & summary["seed"].eq(seed)
                & summary["budget_percent"].eq(20) & summary["method"].eq("selected"),
                "mae",
            ].iloc[0]
        )
        for seed in args.seeds
    }
    comparisons = {}
    for comparator in ("base_gnn", "base_zi", "base_zi__affine"):
        if comparator not in pooled20.index:
            continue
        comparisons[comparator] = {
            seed: v2.bootstrap_fold_delta(
                metrics.loc[metrics["seed"].eq(seed)], "selected", comparator, budget=20
            )
            for seed in args.seeds
        }
    decision = {
        "status": "diagnostic_only",
        "locked_target_test_scored": False,
        "candidate": "GNN-embedding retrieval under the frozen v2 selection rule",
        "selection_rule": SELECTION_RULE,
        "budget_percent": 20,
        "per_seed_pooled_mae_selected": per_seed_mae,
        "pooled_mae_seed_concat": {
            method: (
                float(pooled20.loc[method, "mae"]) if method in pooled20.index else np.nan
            )
            for method in ("base_gnn", "base_zi", "selected", "selected_permuted")
        },
        "pooled_r2_seed_concat": {
            method: (
                float(pooled20.loc[method, "r2"]) if method in pooled20.index else np.nan
            )
            for method in ("base_gnn", "base_zi", "selected")
        },
        "fold_block_bootstrap_by_seed_vs_selected": comparisons,
        "interpretation_rule": (
            "Contract comparison: GNN embedding vs Morgan vs leaf identity under "
            "one identical fold-safe selection rule. Cannot validate the GNN eKNN "
            "claim alone and cannot unlock the locked target test."
        ),
    }
    v2.json_dump(args.output / "development_decision_gnn.json", decision)
    v2.json_dump(
        args.output / "run_manifest.json",
        {
            "run_id": "hitea_gnn_target_memory_development_v1",
            "status": "diagnostic_only",
            "locked_target_test_scored": False,
            "data_manifest_sha256": manifest["development_modeling_table_sha256"],
            "locked_test_registry_sha256": manifest["locked_test_registry_sha256"],
            "n_locked_test_labels_scored": 0,
            "n_folds": int(len(fold_index)),
            "seeds": list(args.seeds),
            "gnn_checkpoints": [
                str(GNN_ROOT / f"s{seed}" / "hitea_gnn_full_source.pt") for seed in args.seeds
            ],
            "similarity_spaces": list(SPACES),
            "base_variants": list(BASE_VARIANTS),
            "selection_rule": SELECTION_RULE,
            "packages": {
                "lightgbm": lgb.__version__, "numpy": np.__version__,
                "pandas": pd.__version__, "scipy": scipy.__version__,
            },
            "python": sys.version,
            "platform": platform.platform(),
            "wall_time_seconds": float(time.time() - started),
        },
    )
    print(json.dumps(decision, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

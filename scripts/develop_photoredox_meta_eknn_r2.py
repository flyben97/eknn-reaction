#!/usr/bin/env python3
"""Develop a leakage-safe meta-eKNN on the opened paper-cold-start task.

The previously scored target is used for development only. Source papers are
cross-fitted: each eligible paper contributes a deterministic 20% calibration
memory, while its remaining rows are predicted by a chemical model that never
sees their labels. A second-stage model learns how to combine chemical
predictions with paper/global/catalyst neighbour evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]


def stable_int(value: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{value}".encode()).hexdigest(), 16)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def metrics(target: np.ndarray, prediction: np.ndarray, papers: np.ndarray) -> dict:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    absolute = np.abs(target - prediction)
    paper_mae = pd.DataFrame(
        {"paper": papers.astype(str), "absolute": absolute}
    ).groupby("paper")["absolute"].mean()
    return {
        "r2": float(r2_score(target, prediction)),
        "pearson_r": float(np.corrcoef(target, prediction)[0, 1]),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "bias": float(np.mean(prediction - target)),
        "p90_abs_error": float(np.quantile(absolute, 0.9)),
        "macro_paper_mae": float(paper_mae.mean()),
    }


def deterministic_paper_roles(
    frame: pd.DataFrame, *, min_rows: int, calibration_fraction: float
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    calibration = np.zeros(len(frame), dtype=bool)
    eligible = np.zeros(len(frame), dtype=bool)
    paper_fold: dict[str, int] = {}
    papers = []
    for paper, group in frame.groupby("paper_group", sort=True):
        if len(group) < min_rows:
            continue
        papers.append(str(paper))
    ordered = sorted(papers, key=lambda value: stable_int(value, "meta-paper-fold-v1"))
    for rank, paper in enumerate(ordered):
        paper_fold[paper] = rank % 5
        indices = frame.index[frame["paper_group"].astype(str) == paper].to_numpy()
        n_calibration = max(2, int(round(calibration_fraction * len(indices))))
        n_calibration = min(n_calibration, len(indices) - 2)
        duplicate_groups = []
        for duplicate_group, group in frame.loc[indices].groupby(
            "duplicate_group", sort=True
        ):
            duplicate_groups.append(
                (
                    stable_int(
                        str(duplicate_group), "meta-paper-calibration-v1"
                    ),
                    group.index.to_numpy(),
                )
            )
        duplicate_groups.sort(key=lambda item: item[0])
        selected = []
        for _, group_indices in duplicate_groups:
            if len(selected) >= n_calibration:
                break
            if len(indices) - len(selected) - len(group_indices) < 2:
                continue
            selected.extend(group_indices.tolist())
        eligible[indices] = True
        calibration[selected] = True
    folds = np.full(len(frame), -1, dtype=int)
    for paper, fold in paper_fold.items():
        folds[frame["paper_group"].astype(str).to_numpy() == paper] = fold
    return eligible, calibration, folds


class ReactionFeaturizer:
    def __init__(self) -> None:
        self.generators = {
            256: rdFingerprintGenerator.GetMorganGenerator(
                radius=2, fpSize=256, countSimulation=False
            ),
            512: rdFingerprintGenerator.GetMorganGenerator(
                radius=2, fpSize=512, countSimulation=False
            ),
        }

    def fingerprint(self, smiles: str, size: int) -> np.ndarray:
        output = np.zeros(size, dtype=np.float32)
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is not None:
            DataStructs.ConvertToNumpyArray(
                self.generators[size].GetCountFingerprint(molecule), output
            )
        return output

    @staticmethod
    def normalized(values: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norm, 1e-8)

    @staticmethod
    def descriptors(smiles: str) -> np.ndarray:
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            return np.zeros(8, dtype=np.float32)
        return np.asarray(
            [
                Descriptors.MolWt(molecule),
                Crippen.MolLogP(molecule),
                Descriptors.TPSA(molecule),
                Descriptors.NumHDonors(molecule),
                Descriptors.NumHAcceptors(molecule),
                Descriptors.NumRotatableBonds(molecule),
                Descriptors.RingCount(molecule),
                Descriptors.FractionCSP3(molecule),
            ],
            dtype=np.float32,
        )

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        raw = {}
        for name, column, size in (
            ("sub", "substrate_smiles", 512),
            ("rea", "reactants_graph_smiles", 512),
            ("pro", "products_graph_smiles", 512),
            ("reg", "reagents_graph_smiles", 256),
            ("cat", "catalysts_graph_smiles", 256),
        ):
            raw[name] = np.stack(
                [self.fingerprint(value, size) for value in frame[column].fillna("")]
            )
        descriptor_block = np.stack(
            [self.descriptors(value) for value in frame["substrate_smiles"].fillna("")]
        )
        numeric = frame[
            [
                "Stages",
                "n_catalysts",
                "substrate_to_product_similarity",
                "substrate_scaffold_retained_in_product",
            ]
        ].fillna(0).astype(float).to_numpy(dtype=np.float32)
        chemical = np.concatenate(
            [raw["sub"], raw["rea"], raw["pro"], raw["reg"], raw["cat"],
             descriptor_block, numeric],
            axis=1,
        )
        raw["diff"] = raw["pro"] - raw["rea"]
        normalized = {name: self.normalized(values) for name, values in raw.items()}
        return chemical, normalized


def base_model(seed: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=10,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=3.0,
        random_state=seed,
        n_jobs=12,
        verbosity=-1,
    )


def similarity_matrix(
    query: np.ndarray, memory: np.ndarray, name: str,
    normalized: dict[str, np.ndarray],
) -> np.ndarray:
    if name == "diff":
        return normalized["diff"][query] @ normalized["diff"][memory].T
    if name == "subdiff":
        return (
            0.25 * normalized["sub"][query] @ normalized["sub"][memory].T
            + 0.75 * normalized["diff"][query] @ normalized["diff"][memory].T
        )
    if name == "chem":
        return (
            0.4 * normalized["sub"][query] @ normalized["sub"][memory].T
            + 0.2 * normalized["rea"][query] @ normalized["rea"][memory].T
            + 0.2 * normalized["pro"][query] @ normalized["pro"][memory].T
            + 0.1 * normalized["reg"][query] @ normalized["reg"][memory].T
            + 0.1 * normalized["cat"][query] @ normalized["cat"][memory].T
        )
    raise ValueError(name)


def build_meta_features(
    frame: pd.DataFrame,
    normalized: dict[str, np.ndarray],
    query: np.ndarray,
    memory: np.ndarray,
    base_prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    query = np.asarray(query, dtype=int)
    memory = np.asarray(memory, dtype=int)
    memory_yield = target[memory]
    memory_residual = memory_yield - base_prediction[memory]
    query_papers = frame.loc[query, "paper_group"].astype(str).to_numpy()
    memory_papers = frame.loc[memory, "paper_group"].astype(str).to_numpy()
    query_catalysts = frame.loc[query, "catalyst_group"].astype(str).to_numpy()
    memory_catalysts = frame.loc[memory, "catalyst_group"].astype(str).to_numpy()
    query_identities = frame.loc[query, "substrate_smiles"].astype(str).to_numpy()
    memory_identities = frame.loc[memory, "substrate_smiles"].astype(str).to_numpy()

    values: dict[str, np.ndarray] = {"base": base_prediction[query]}
    for name in ("diff", "subdiff", "chem"):
        similarities = similarity_matrix(query, memory, name, normalized)
        for scope in ("paper", "catalyst", "global"):
            if scope == "paper":
                eligible = query_papers[:, None] == memory_papers[None, :]
            elif scope == "catalyst":
                eligible = query_catalysts[:, None] == memory_catalysts[None, :]
            else:
                eligible = np.ones_like(similarities, dtype=bool)
            scoped = np.where(eligible, similarities, -1e9)
            order = np.argsort(-scoped, axis=1)[:, :8]
            selected_similarity = np.take_along_axis(scoped, order, axis=1)
            selected_valid = np.take_along_axis(eligible, order, axis=1)
            prefix = f"{scope}_{name}"
            values[f"{prefix}_pool"] = eligible.sum(axis=1).astype(float)
            values[f"{prefix}_nearest_similarity"] = np.where(
                selected_valid[:, 0], selected_similarity[:, 0], -1.0
            )
            for k in (1, 3, 5, 8):
                for temperature in (0.1, 0.2):
                    mask = selected_valid & (np.arange(8)[None, :] < k)
                    logits = (selected_similarity - selected_similarity[:, [0]]) / temperature
                    weights = np.exp(np.clip(logits, -50, 0)) * mask
                    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
                    suffix = f"k{k}_t{str(temperature).replace('.', '')}"
                    values[f"{prefix}_yield_{suffix}"] = np.sum(
                        weights * memory_yield[order], axis=1
                    )
                    values[f"{prefix}_residual_{suffix}"] = (
                        base_prediction[query]
                        + np.sum(weights * memory_residual[order], axis=1)
                    )

    paper_memory: dict[str, np.ndarray] = {}
    for paper in np.unique(memory_papers):
        paper_memory[paper] = np.flatnonzero(memory_papers == paper)
    statistics = {
        "paper_calibration_n": [],
        "paper_yield_mean": [],
        "paper_yield_std": [],
        "paper_residual_mean": [],
        "paper_residual_std": [],
        "paper_identity_seen": [],
    }
    for paper, identity in zip(query_papers, query_identities):
        local = paper_memory.get(paper, np.asarray([], dtype=int))
        if len(local) == 0:
            statistics["paper_calibration_n"].append(0.0)
            statistics["paper_yield_mean"].append(float(memory_yield.mean()))
            statistics["paper_yield_std"].append(float(memory_yield.std()))
            statistics["paper_residual_mean"].append(float(memory_residual.mean()))
            statistics["paper_residual_std"].append(float(memory_residual.std()))
            statistics["paper_identity_seen"].append(0.0)
        else:
            statistics["paper_calibration_n"].append(float(len(local)))
            statistics["paper_yield_mean"].append(float(memory_yield[local].mean()))
            statistics["paper_yield_std"].append(float(memory_yield[local].std()))
            statistics["paper_residual_mean"].append(float(memory_residual[local].mean()))
            statistics["paper_residual_std"].append(float(memory_residual[local].std()))
            statistics["paper_identity_seen"].append(
                float(identity in set(memory_identities[local]))
            )
    values.update({key: np.asarray(value) for key, value in statistics.items()})
    output = pd.DataFrame(values)
    return output, list(output.columns)


def meta_model(seed: int, leaves: int, min_child: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.025,
        num_leaves=leaves,
        min_child_samples=min_child,
        reg_lambda=5.0,
        reg_alpha=0.5,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=12,
        verbosity=-1,
    )


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
        default=ROOT / "results/photoredox_meta_eknn_r2_development",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(args.task_dir / "source_train.csv").reset_index(drop=True)
    calibration = pd.read_csv(args.task_dir / "calibration.csv").reset_index(drop=True)
    test = pd.read_csv(args.task_dir / "target_test.csv").reset_index(drop=True)
    source_y = source["Yield"].to_numpy(dtype=float)

    featurizer = ReactionFeaturizer()
    source_chemical, source_normalized = featurizer.transform(source)
    eligible, source_calibration, source_fold = deterministic_paper_roles(
        source, min_rows=15, calibration_fraction=args.calibration_fraction
    )
    source_query = eligible & ~source_calibration
    source_base = np.full(len(source), np.nan, dtype=float)
    meta_frames = []
    meta_targets = []
    meta_papers = []
    meta_folds = []
    meta_query_indices = []

    assignment = source[["reaction_id", "paper_group", "catalyst_group"]].copy()
    assignment["meta_eligible"] = eligible
    assignment["meta_role"] = np.where(
        source_calibration, "calibration", np.where(source_query, "query", "base_only")
    )
    assignment["meta_fold"] = source_fold
    assignment.to_csv(args.output_dir / "source_meta_split_assignments.csv", index=False)

    for fold in range(5):
        heldout = source_fold == fold
        memory = np.flatnonzero(heldout & source_calibration)
        query = np.flatnonzero(heldout & source_query)
        fit = np.flatnonzero(~heldout | source_calibration)
        weights = np.ones(len(fit), dtype=float)
        weights[source_calibration[fit]] = 10.0
        model = base_model(args.seed + fold)
        model.fit(source_chemical[fit], source_y[fit], sample_weight=weights)
        source_base[query] = np.clip(model.predict(source_chemical[query]), 0, 1)
        source_base[memory] = np.clip(model.predict(source_chemical[memory]), 0, 1)
        features, names = build_meta_features(
            source, source_normalized, query, memory, source_base, source_y
        )
        meta_frames.append(features)
        meta_targets.append(source_y[query])
        meta_papers.extend(source.loc[query, "paper_group"].astype(str))
        meta_folds.extend([fold] * len(query))
        meta_query_indices.extend(query.tolist())

    meta_x = pd.concat(meta_frames, ignore_index=True)
    meta_y = np.concatenate(meta_targets)
    meta_papers_array = np.asarray(meta_papers)
    meta_folds_array = np.asarray(meta_folds, dtype=int)
    if meta_x.columns.tolist() != names:
        raise RuntimeError("Meta feature schema changed across folds")

    selection_rows = []
    selected_prediction = None
    selected = None
    for leaves in (7, 15, 31):
        for min_child in (20, 50, 100):
            prediction = np.zeros(len(meta_y), dtype=float)
            for fold in range(5):
                fit = meta_folds_array != fold
                valid = ~fit
                model = meta_model(args.seed + 100 + fold, leaves, min_child)
                model.fit(meta_x.loc[fit], meta_y[fit])
                prediction[valid] = np.clip(model.predict(meta_x.loc[valid]), 0, 1)
            score = metrics(meta_y, prediction, meta_papers_array)
            selection_rows.append(
                {"leaves": leaves, "min_child": min_child, **score}
            )
            key = (score["r2"], score["pearson_r"], -score["mae"])
            if selected is None or key > selected[0]:
                selected = (key, leaves, min_child, score)
                selected_prediction = prediction.copy()
    selection_frame = pd.DataFrame(selection_rows).sort_values("r2", ascending=False)
    selection_frame.to_csv(args.output_dir / "meta_model_selection.csv", index=False)
    assert selected is not None and selected_prediction is not None
    _, selected_leaves, selected_min_child, source_oof_metrics = selected

    # Select a source-OOF blend between the chemical model and meta-eKNN.
    source_query_indices = np.asarray(meta_query_indices, dtype=int)
    source_query_base = source_base[source_query_indices]
    blend_rows = []
    best_blend = None
    for meta_weight in np.linspace(0, 1, 21):
        prediction = np.clip(
            (1.0 - meta_weight) * source_query_base
            + meta_weight * selected_prediction,
            0,
            1,
        )
        score = metrics(meta_y, prediction, meta_papers_array)
        blend_rows.append({"meta_weight": meta_weight, **score})
        key = (score["r2"], score["pearson_r"], -score["mae"])
        if best_blend is None or key > best_blend[0]:
            best_blend = (key, float(meta_weight), score)
    pd.DataFrame(blend_rows).to_csv(
        args.output_dir / "source_oof_blend_selection.csv", index=False
    )
    assert best_blend is not None
    _, meta_weight, source_blend_metrics = best_blend

    # Fit the selected source meta-model and apply to the opened target only as
    # development evidence.
    final_meta = meta_model(args.seed + 999, selected_leaves, selected_min_child)
    final_meta.fit(meta_x, meta_y)
    target_frame = pd.concat([calibration, test], ignore_index=True)
    target_eligible, target_calibration_mask, _ = deterministic_paper_roles(
        target_frame,
        min_rows=15,
        calibration_fraction=args.calibration_fraction,
    )
    if not target_eligible.all():
        raise RuntimeError("Every opened target paper should be eligible")
    target_query_mask = ~target_calibration_mask
    target_y = target_frame["Yield"].to_numpy(dtype=float)
    target_chemical, target_normalized = featurizer.transform(target_frame)
    fit_chemical = np.concatenate(
        [source_chemical, target_chemical[target_calibration_mask]]
    )
    fit_y = np.concatenate([source_y, target_y[target_calibration_mask]])
    fit_weight = np.concatenate(
        [np.ones(len(source)), np.full(int(target_calibration_mask.sum()), 10.0)]
    )
    final_base = base_model(args.seed + 2000)
    final_base.fit(fit_chemical, fit_y, sample_weight=fit_weight)
    target_base = np.clip(final_base.predict(target_chemical), 0, 1)
    target_query = np.flatnonzero(target_query_mask)
    target_memory = np.flatnonzero(target_calibration_mask)
    target_meta_x, target_names = build_meta_features(
        target_frame,
        target_normalized,
        target_query,
        target_memory,
        target_base,
        target_y,
    )
    if target_names != names:
        raise RuntimeError("Target meta feature schema does not match source")
    target_meta = np.clip(final_meta.predict(target_meta_x), 0, 1)
    target_blend = np.clip(
        (1.0 - meta_weight) * target_base[target_query]
        + meta_weight * target_meta,
        0,
        1,
    )
    target_truth = target_y[target_query]
    target_papers = target_frame.loc[target_query, "paper_group"].astype(str).to_numpy()
    development_metrics = {
        "chemical_base": metrics(
            target_truth, target_base[target_query], target_papers
        ),
        "meta_eknn": metrics(target_truth, target_meta, target_papers),
        "source_selected_blend": metrics(target_truth, target_blend, target_papers),
    }

    predictions = target_frame.loc[
        target_query,
        ["reaction_id", "paper_group", "catalyst_group", "substrate_smiles"],
    ].copy().reset_index(drop=True)
    predictions["Yield"] = target_truth
    predictions["chemical_base"] = target_base[target_query]
    predictions["meta_eknn"] = target_meta
    predictions["source_selected_blend"] = target_blend
    predictions.to_csv(args.output_dir / "opened_development_predictions.csv", index=False)
    meta_oof = pd.DataFrame(
        {
            "paper_group": meta_papers_array,
            "fold": meta_folds_array,
            "Yield": meta_y,
            "chemical_base": source_query_base,
            "meta_eknn_oof": selected_prediction,
        }
    )
    meta_oof.to_csv(args.output_dir / "source_meta_oof_predictions.csv", index=False)
    recipe = {
        "status": "frozen_from_opened_development_only",
        "confirmation_test_loaded": False,
        "source_meta_rows": int(len(meta_y)),
        "source_meta_papers": int(len(np.unique(meta_papers_array))),
        "calibration_fraction": args.calibration_fraction,
        "opened_target_counts": {
            "calibration": int(target_calibration_mask.sum()),
            "test": int(target_query_mask.sum()),
        },
        "chemical_model": {
            "family": "LightGBM_on_reaction_Morgan_and_descriptors",
            "calibration_weight": 10.0,
            "num_leaves": 31,
            "min_child_samples": 10,
            "n_estimators": 600,
            "learning_rate": 0.03,
        },
        "meta_model": {
            "family": "LightGBM_meta_eKNN",
            "num_leaves": selected_leaves,
            "min_child_samples": selected_min_child,
            "meta_weight": meta_weight,
            "feature_names": names,
        },
        "source_meta_oof": {
            "meta_model": source_oof_metrics,
            "selected_blend": source_blend_metrics,
        },
        "opened_target_development": development_metrics,
        "absolute_target": {"r2": 0.60, "pearson_r": 0.80},
    }
    write_json(args.output_dir / "frozen_recipe.json", recipe)
    print(json.dumps(recipe, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

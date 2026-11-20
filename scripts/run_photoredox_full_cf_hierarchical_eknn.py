#!/usr/bin/env python3
"""Train, freeze, and separately score the full Photoredox CF-HR-eKNN case.

The ``train-freeze`` phase never reads the Yield column from target_test.csv.
The ``score`` phase verifies the frozen prediction hash before loading only
reaction_id and Yield from that file.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_gnn_eknn", ROOT / "scripts" / "train_gnn_eknn.py"
)
train = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train)

from eknn_reaction import utils
from eknn_reaction.data.dataset import Standardizer
from eknn_reaction.models.graph_encoder import build_reaction_graphs


ROLE_COLUMNS = [
    "reactants_graph_smiles",
    "products_graph_smiles",
    "reagents_graph_smiles",
    "catalysts_graph_smiles",
]
PRIMARY_METHOD = "cf_hierarchical_b112"
METHODS = [
    "base",
    "global_offset_b112",
    "cf_source_eknn",
    "target_global_b112",
    PRIMARY_METHOD,
    "cf_hierarchical_b48",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def model_args(epochs: int) -> SimpleNamespace:
    return SimpleNamespace(
        d_model=96,
        n_layers=3,
        n_heads=4,
        ff_dim=192,
        dropout=0.10,
        pe_dim=3,
        backbone="mpnn",
        mpnn=False,
        gat=False,
        graph_transformer=False,
        edge_transformer=False,
        graphormer=False,
        deep_gps=False,
        light_gps=False,
        readout="mean_max",
        attn_pool=False,
        jk=False,
        virtual_node=False,
        component_readout=False,
        rolewise_readout=False,
        role_interaction_readout=False,
        role_interaction_layers=2,
        role_concat_readout=False,
        role_factorized_readout=False,
        role_factorized_retrieval=False,
        retrieval_dim=0,
        condition_fusion_mode="gated_add",
        condition_node_film=False,
        condition_token_layers=2,
        component_identity_dim=32,
        batch_size=32,
        lr=1e-3,
        weight_decay=1e-4,
        optimizer="adamw",
        momentum=0.9,
        grad_clip=1.0,
        warmup_epochs=min(5, epochs),
        epochs=epochs,
        min_epochs=min(25, epochs),
        val_check_interval=5,
        early_stop_patience=15,
        swa_start=epochs + 1,
        metric_loss_weight=0.0,
        metric_label_temperature=0.5,
        metric_embedding_temperature=0.1,
    )


def model_config(args: SimpleNamespace) -> dict:
    keys = (
        "d_model", "n_layers", "n_heads", "ff_dim", "dropout", "pe_dim",
        "backbone", "mpnn", "gat", "graph_transformer", "edge_transformer",
        "graphormer", "deep_gps", "light_gps", "readout", "attn_pool", "jk",
        "virtual_node", "component_readout", "rolewise_readout",
        "role_interaction_readout", "role_interaction_layers",
        "role_concat_readout", "role_factorized_readout",
        "role_factorized_retrieval", "retrieval_dim",
        "condition_fusion_mode", "condition_node_film",
        "condition_token_layers", "component_identity_dim",
    )
    return {key: getattr(args, key) for key in keys}


def build_graphs(frame: pd.DataFrame) -> tuple[list, np.ndarray]:
    graphs, targets = build_reaction_graphs(
        frame,
        ROLE_COLUMNS,
        "Yield",
        add_laplacian_pe=True,
        pe_k=3,
        verbose=True,
    )
    failed = [index for index, graph in enumerate(graphs) if graph is None]
    if failed:
        raise RuntimeError(f"Graph construction failed for {len(failed)} rows")
    return list(graphs), targets


def normalise_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def top_neighbours(
    query: np.ndarray, memory: np.ndarray, max_k: int
) -> tuple[np.ndarray, np.ndarray]:
    max_k = min(int(max_k), len(memory))
    similarity = np.clip(query @ memory.T, -1.0, 1.0)
    distance = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * similarity))
    indices = np.argpartition(distance, max_k - 1, axis=1)[:, :max_k]
    local_distance = np.take_along_axis(distance, indices, axis=1)
    order = np.argsort(local_distance, axis=1)
    indices = np.take_along_axis(indices, order, axis=1)
    local_distance = np.take_along_axis(local_distance, order, axis=1)
    return indices, local_distance


def weighted_from_top_neighbours(
    indices: np.ndarray,
    distances: np.ndarray,
    values: np.ndarray,
    *,
    k: int,
    temperature: float,
) -> np.ndarray:
    k = min(int(k), indices.shape[1])
    local_indices = indices[:, :k]
    local_distances = distances[:, :k]
    logits = -(local_distances - local_distances[:, :1]) / float(temperature)
    weights = np.exp(logits)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return np.sum(weights * values[local_indices], axis=1)


def target_memory_correction(
    query_embeddings: np.ndarray,
    query_frame: pd.DataFrame,
    memory_embeddings: np.ndarray,
    memory_residuals: np.ndarray,
    memory_frame: pd.DataFrame,
    *,
    k: int,
    temperature: float,
    scale: float,
    shrinkage: float,
    scope: str,
    exclude_query_identity: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    if scope not in {"global", "hierarchical"}:
        raise ValueError(f"Unknown target-memory scope: {scope}")
    memory_scaffold = memory_frame["substrate_scaffold"].astype(str).to_numpy()
    memory_catalyst = memory_frame["catalyst_group"].astype(str).to_numpy()
    memory_identity = memory_frame["substrate_smiles"].astype(str).to_numpy()
    output = np.zeros(len(query_frame), dtype=np.float64)
    diagnostics = []

    for row_index, (_, row) in enumerate(query_frame.iterrows()):
        eligible = np.ones(len(memory_frame), dtype=bool)
        if exclude_query_identity:
            eligible &= memory_identity != str(row["substrate_smiles"])
        level = "global"
        if scope == "hierarchical":
            exact = eligible.copy()
            exact &= memory_scaffold == str(row["substrate_scaffold"])
            exact &= memory_catalyst == str(row["catalyst_group"])
            scaffold = eligible.copy()
            scaffold &= memory_scaffold == str(row["substrate_scaffold"])
            if exact.any():
                eligible = exact
                level = "stratum"
            elif scaffold.any():
                eligible = scaffold
                level = "scaffold"
        candidate_indices = np.flatnonzero(eligible)
        if len(candidate_indices) == 0:
            diagnostics.append(
                {"target_pool": 0, "target_neighbours": 0,
                 "target_neff": 0.0, "target_scope_level": "none",
                 "target_nearest_distance": np.nan}
            )
            continue
        vector = query_embeddings[row_index]
        distances = np.sqrt(
            np.maximum(
                0.0,
                2.0
                - 2.0
                * np.clip(memory_embeddings[candidate_indices] @ vector, -1.0, 1.0),
            )
        )
        k_effective = min(int(k), len(candidate_indices))
        local = np.argpartition(distances, k_effective - 1)[:k_effective]
        local = local[np.argsort(distances[local])]
        local_distances = distances[local]
        weights = np.exp(
            -(local_distances - local_distances.min()) / float(temperature)
        )
        weights /= max(float(weights.sum()), 1e-12)
        neff = float(1.0 / np.sum(weights ** 2))
        gate = neff / (neff + float(shrinkage)) if shrinkage > 0 else 1.0
        selected = candidate_indices[local]
        output[row_index] = (
            float(scale) * gate * float(np.dot(weights, memory_residuals[selected]))
        )
        diagnostics.append(
            {"target_pool": int(len(candidate_indices)),
             "target_neighbours": int(k_effective), "target_neff": neff,
             "target_scope_level": level,
             "target_nearest_distance": float(local_distances[0])}
        )
    return output, pd.DataFrame(diagnostics)


def paper_domain_correction(
    query_frame: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    calibration_residuals: np.ndarray,
    *,
    scope: str,
    paper_shrinkage: float,
    stratum_shrinkage: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical-Bayes paper/stratum residual with global cold-start fallback."""
    if scope not in {"paper", "paper_stratum"}:
        raise ValueError(f"Unknown paper-domain scope: {scope}")
    work = calibration_frame[
        ["paper_group", "substrate_scaffold", "catalyst_group"]
    ].copy()
    work["residual"] = np.asarray(calibration_residuals, dtype=float)
    global_value = float(work["residual"].mean())
    paper = work.groupby("paper_group")["residual"].agg(["sum", "count"])
    paper_stratum = work.groupby(
        ["paper_group", "substrate_scaffold", "catalyst_group"]
    )["residual"].agg(["sum", "count"])
    correction = np.zeros(len(query_frame), dtype=float)
    paper_seen = np.zeros(len(query_frame), dtype=bool)
    for index, (_, row) in enumerate(query_frame.iterrows()):
        paper_key = str(row["paper_group"])
        if paper_key in paper.index:
            paper_seen[index] = True
            paper_value = (
                float(paper.loc[paper_key, "sum"])
                + float(paper_shrinkage) * global_value
            ) / (float(paper.loc[paper_key, "count"]) + float(paper_shrinkage))
        else:
            paper_value = global_value
        value = paper_value
        if scope == "paper_stratum" and paper_seen[index]:
            stratum_key = (
                paper_key,
                str(row["substrate_scaffold"]),
                str(row["catalyst_group"]),
            )
            if stratum_key in paper_stratum.index:
                value = (
                    float(paper_stratum.loc[stratum_key, "sum"])
                    + float(stratum_shrinkage) * paper_value
                ) / (
                    float(paper_stratum.loc[stratum_key, "count"])
                    + float(stratum_shrinkage)
                )
        correction[index] = value
    return correction, paper_seen


def metric_row(
    target: np.ndarray, prediction: np.ndarray, frame: pd.DataFrame
) -> dict[str, float]:
    absolute = np.abs(target - prediction)
    score = {
        "r2": float(r2_score(target, prediction)) if len(target) > 1 else np.nan,
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "bias": float(np.mean(prediction - target)),
        "p90_abs_error": float(np.quantile(absolute, 0.90)),
    }
    grouped = pd.DataFrame(
        {
            "absolute": absolute,
            "scaffold": frame["substrate_scaffold"].astype(str).to_numpy(),
            "stratum": (
                frame["substrate_scaffold"].astype(str)
                + "||"
                + frame["catalyst_group"].astype(str)
            ).to_numpy(),
        }
    )
    score["macro_scaffold_mae"] = float(
        grouped.groupby("scaffold", sort=False)["absolute"].mean().mean()
    )
    score["macro_stratum_mae"] = float(
        grouped.groupby("stratum", sort=False)["absolute"].mean().mean()
    )
    score["macro_paper_mae"] = float(
        pd.DataFrame(
            {
                "absolute": absolute,
                "paper": frame["paper_group"].astype(str).to_numpy(),
            }
        ).groupby("paper", sort=False)["absolute"].mean().mean()
    )
    return score


def choose_grid(rows: list[dict]) -> dict:
    return min(
        rows,
        key=lambda row: (
            row["macro_stratum_mae"], row["mae"], row["rmse"],
            json.dumps(row, sort_keys=True),
        ),
    )


def deterministic_one_per_stratum(frame: pd.DataFrame) -> np.ndarray:
    work = frame[["reaction_id", "substrate_scaffold", "catalyst_group"]].copy()
    work["stratum"] = (
        work["substrate_scaffold"].astype(str)
        + "||"
        + work["catalyst_group"].astype(str)
    )
    work["rank_key"] = work["reaction_id"].astype(str).map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    chosen = (
        work.sort_values(["stratum", "rank_key"])
        .groupby("stratum", sort=True, as_index=False)
        .head(1)
        .index.to_numpy(dtype=int)
    )
    expected = int(work["stratum"].nunique())
    if len(chosen) != expected:
        raise RuntimeError(
            f"Expected one calibration row for each of {expected} strata, "
            f"found {len(chosen)}"
        )
    return np.sort(chosen)


# Backward-compatible alias for the original 48-stratum experiment and tests.
deterministic_budget_48 = deterministic_one_per_stratum


def add_file_logger(output_dir: Path) -> None:
    handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S")
    )
    train.LOG.addHandler(handler)


def audit_data_contract(
    source: pd.DataFrame,
    calibration: pd.DataFrame,
    target_features: pd.DataFrame,
    *,
    deployment_axis: str = "scaffold",
) -> dict:
    source_scaffolds = set(source["substrate_scaffold"].astype(str))
    calibration_scaffolds = set(calibration["substrate_scaffold"].astype(str))
    target_scaffolds = set(target_features["substrate_scaffold"].astype(str))
    scaffold_overlap = source_scaffolds & (calibration_scaffolds | target_scaffolds)
    calibration_strata = set(
        zip(
            calibration["substrate_scaffold"].astype(str),
            calibration["catalyst_group"].astype(str),
        )
    )
    target_strata = set(
        zip(
            target_features["substrate_scaffold"].astype(str),
            target_features["catalyst_group"].astype(str),
        )
    )
    source_papers = set(source["paper_group"].astype(str))
    calibration_papers = set(calibration["paper_group"].astype(str))
    target_papers = set(target_features["paper_group"].astype(str))
    paper_overlap = source_papers & (calibration_papers | target_papers)
    calibration_paper_strata = set(
        zip(
            calibration["paper_group"].astype(str),
            calibration["catalyst_group"].astype(str),
        )
    )
    target_paper_strata = set(
        zip(
            target_features["paper_group"].astype(str),
            target_features["catalyst_group"].astype(str),
        )
    )
    source_catalysts = set(source["catalyst_group"].astype(str))
    target_catalysts = set(target_features["catalyst_group"].astype(str))
    duplicate_sets = {
        "source": set(source["duplicate_group"].astype(str)),
        "calibration": set(calibration["duplicate_group"].astype(str)),
        "target": set(target_features["duplicate_group"].astype(str)),
    }
    duplicate_overlap = (
        (duplicate_sets["source"] & duplicate_sets["calibration"])
        | (duplicate_sets["source"] & duplicate_sets["target"])
        | (duplicate_sets["calibration"] & duplicate_sets["target"])
    )
    audit = {
        "source_target_scaffold_overlap": int(len(scaffold_overlap)),
        "calibration_target_scaffold_sets_equal": calibration_scaffolds == target_scaffolds,
        "calibration_target_strata_sets_equal": calibration_strata == target_strata,
        "target_catalysts_not_in_source": int(len(target_catalysts - source_catalysts)),
        "cross_split_duplicate_group_overlap": int(len(duplicate_overlap)),
        "target_scaffolds": int(len(target_scaffolds)),
        "target_strata": int(len(target_strata)),
        "source_target_paper_overlap": int(len(paper_overlap)),
        "calibration_target_paper_sets_equal": calibration_papers == target_papers,
        "calibration_target_paper_strata_equal": (
            calibration_paper_strata == target_paper_strata
        ),
        "target_papers": int(len(target_papers)),
        "deployment_axis": deployment_axis,
    }
    if deployment_axis == "paper":
        required = {
            "source_target_paper_overlap": 0,
            "calibration_target_paper_sets_equal": True,
            "calibration_target_paper_strata_equal": True,
            "target_catalysts_not_in_source": 0,
            "cross_split_duplicate_group_overlap": 0,
        }
    elif deployment_axis == "scaffold":
        required = {
            "source_target_scaffold_overlap": 0,
            "calibration_target_scaffold_sets_equal": True,
            "calibration_target_strata_sets_equal": True,
            "target_catalysts_not_in_source": 0,
            "cross_split_duplicate_group_overlap": 0,
        }
    else:
        raise ValueError(f"Unknown deployment axis: {deployment_axis}")
    failures = {
        key: {"observed": audit[key], "required": value}
        for key, value in required.items()
        if audit[key] != value
    }
    if failures:
        raise RuntimeError(f"Frozen data contract failed: {failures}")
    return audit


def train_and_freeze(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    add_file_logger(output_dir)
    task_dir = args.task_dir.resolve()
    domain_recipe = None
    if args.domain_recipe is not None:
        domain_recipe = json.loads(args.domain_recipe.read_text(encoding="utf-8"))
        if domain_recipe.get("new_confirmation_test_loaded") is not False:
            raise RuntimeError(
                "The domain recipe must explicitly record that confirmation "
                "test labels were not loaded"
            )
    if args.skip_source_crossfit and not args.fix_source_scale_zero:
        raise ValueError(
            "--skip-source-crossfit requires --fix-source-scale-zero"
        )

    source_path = task_dir / "source_train.csv"
    calibration_path = task_dir / "calibration.csv"
    target_path = task_dir / "target_test.csv"
    source = pd.read_csv(source_path)
    calibration = pd.read_csv(calibration_path)
    target_columns = pd.read_csv(target_path, nrows=0).columns.tolist()
    if "Yield" not in target_columns:
        raise RuntimeError("Locked target file does not contain the expected Yield column")
    target_features = pd.read_csv(
        target_path, usecols=[column for column in target_columns if column != "Yield"]
    )
    target_for_graphs = target_features.copy()
    target_for_graphs["Yield"] = 0.0
    if not source["reaction_id"].is_unique or not calibration["reaction_id"].is_unique:
        raise RuntimeError("Reaction identifiers must be unique within each split")
    deployment_axis = (
        "paper"
        if domain_recipe is not None
        and domain_recipe.get("confirmation_protocol")
        == "paper-coldstart-HR-eKNN-confirmatory-v1"
        else "scaffold"
    )
    data_contract_audit = audit_data_contract(
        source,
        calibration,
        target_features,
        deployment_axis=deployment_axis,
    )

    print(
        json.dumps(
            {"phase": "train-freeze", "target_yield_loaded": False,
             "counts": {"source": len(source), "calibration": len(calibration),
                        "target_features": len(target_features)}},
            ensure_ascii=False,
        ),
        flush=True,
    )
    source_graphs, source_targets = build_graphs(source)
    calibration_graphs, calibration_targets = build_graphs(calibration)
    target_graphs, _ = build_graphs(target_for_graphs)

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    source_groups = source["substrate_scaffold"].fillna("<NA>").astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    fit_indices, val_indices = next(
        splitter.split(np.zeros(len(source)), groups=source_groups)
    )
    if set(source_groups[fit_indices]) & set(source_groups[val_indices]):
        raise RuntimeError("Source epoch-selection group leakage detected")

    selection_args = model_args(args.max_epochs)
    utils.set_seed(args.seed, deterministic=True)
    selection_model = train.build_model(selection_args).to(device)
    fit_standardizer = Standardizer(
        float(np.mean(source_targets[fit_indices])),
        float(np.std(source_targets[fit_indices])),
    )
    selection_info = train.train_gnn_model(
        selection_model,
        [source_graphs[int(index)] for index in fit_indices],
        source_targets[fit_indices],
        fit_standardizer,
        device,
        selection_args,
        stage_name="mpnn_source_epoch_selection",
        validation_graphs=[source_graphs[int(index)] for index in val_indices],
        validation_targets=source_targets[val_indices],
    )
    source_validation = train.evaluate_static(
        selection_model,
        [source_graphs[int(index)] for index in val_indices],
        source_targets[val_indices],
        device,
        fit_standardizer,
    )
    selected_epoch = int(selection_info["epoch"])
    del selection_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    fixed_args = model_args(selected_epoch)
    oof_prediction = np.full(len(source), np.nan, dtype=np.float64)
    oof_fold = np.full(len(source), -1, dtype=int)
    crossfit_training = []
    if not args.skip_source_crossfit:
        crossfit = GroupKFold(
            n_splits=args.crossfit_folds, shuffle=True, random_state=args.seed
        )
        for fold, (train_indices, held_indices) in enumerate(
            crossfit.split(np.zeros(len(source)), groups=source_groups)
        ):
            if set(source_groups[train_indices]) & set(source_groups[held_indices]):
                raise RuntimeError(f"Cross-fit group leakage detected in fold {fold}")
            fold_standardizer = Standardizer(
                float(np.mean(source_targets[train_indices])),
                float(np.std(source_targets[train_indices])),
            )
            utils.set_seed(args.seed + 100 + fold, deterministic=True)
            fold_model = train.build_model(fixed_args).to(device)
            fold_info = train.train_gnn_model(
                fold_model,
                [source_graphs[int(index)] for index in train_indices],
                source_targets[train_indices],
                fold_standardizer,
                device,
                fixed_args,
                stage_name=f"mpnn_crossfit_fold_{fold}",
            )
            _, held_normalised = train.encode_reactions(
                fold_model,
                [source_graphs[int(index)] for index in held_indices],
                device,
                batch_size=args.encode_batch_size,
            )
            oof_prediction[held_indices] = fold_standardizer.denormalize(held_normalised)
            oof_fold[held_indices] = fold
            crossfit_training.append(
                {"fold": fold, "train_rows": int(len(train_indices)),
                 "held_rows": int(len(held_indices)), **fold_info}
            )
            del fold_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if np.isnan(oof_prediction).any() or (oof_fold < 0).any():
            raise RuntimeError("Cross-fitting did not predict every source reaction")

    full_standardizer = Standardizer(
        float(np.mean(source_targets)), float(np.std(source_targets))
    )
    utils.set_seed(args.seed, deterministic=True)
    final_model = train.build_model(fixed_args).to(device)
    final_info = train.train_gnn_model(
        final_model,
        source_graphs,
        source_targets,
        full_standardizer,
        device,
        fixed_args,
        stage_name="mpnn_full_source",
    )
    source_embedding, source_base_norm = train.encode_reactions(
        final_model, source_graphs, device, batch_size=args.encode_batch_size
    )
    calibration_embedding, calibration_base_norm = train.encode_reactions(
        final_model, calibration_graphs, device, batch_size=args.encode_batch_size
    )
    target_embedding, target_base_norm = train.encode_reactions(
        final_model, target_graphs, device, batch_size=args.encode_batch_size
    )
    source_embedding = normalise_rows(source_embedding)
    calibration_embedding = normalise_rows(calibration_embedding)
    target_embedding = normalise_rows(target_embedding)
    source_base = full_standardizer.denormalize(source_base_norm)
    calibration_base = full_standardizer.denormalize(calibration_base_norm)
    target_base = full_standardizer.denormalize(target_base_norm)
    source_oof_residual = (
        np.zeros(len(source), dtype=float)
        if args.fix_source_scale_zero
        else source_targets - oof_prediction
    )

    source_k = (10, 30, 100)
    source_temperature = (0.05, 0.15, 0.35)
    source_scale = (0.0, 0.25, 0.5, 1.0)
    if args.fix_source_scale_zero:
        best_source = {
            "k": 10,
            "temperature": 0.05,
            "scale": 0.0,
            **metric_row(calibration_targets, calibration_base, calibration),
        }
        source_grid = [best_source]
        calibration_source_correction = np.zeros(len(calibration), dtype=float)
    else:
        cal_source_indices, cal_source_distances = top_neighbours(
            calibration_embedding, source_embedding, max(source_k)
        )
        source_grid = []
        for k, temperature, scale in itertools.product(
            source_k, source_temperature, source_scale
        ):
            correction = scale * weighted_from_top_neighbours(
                cal_source_indices,
                cal_source_distances,
                source_oof_residual,
                k=k,
                temperature=temperature,
            )
            prediction = calibration_base + correction
            source_grid.append(
                {"k": k, "temperature": temperature, "scale": scale,
                 **metric_row(calibration_targets, prediction, calibration)}
            )
        best_source = choose_grid(source_grid)
        calibration_source_correction = best_source["scale"] * weighted_from_top_neighbours(
            cal_source_indices,
            cal_source_distances,
            source_oof_residual,
            k=int(best_source["k"]),
            temperature=float(best_source["temperature"]),
        )
    source_grid_frame = pd.DataFrame(source_grid).sort_values(
        ["macro_stratum_mae", "mae", "rmse"]
    )
    source_grid_frame.to_csv(output_dir / "source_calibration_grid.csv", index=False)
    calibration_source_prediction = calibration_base + calibration_source_correction
    calibration_target_residual = calibration_targets - calibration_source_prediction

    target_k = (1, 3, 5)
    target_temperature = (0.10, 0.30)
    target_scale = (0.25, 0.50, 1.0)
    target_shrinkage = (0.0, 2.0, 5.0)
    target_grid_rows = []
    best_target = {}
    for scope in ("global", "hierarchical"):
        scope_rows = []
        for k, temperature, scale, shrinkage in itertools.product(
            target_k, target_temperature, target_scale, target_shrinkage
        ):
            correction, _ = target_memory_correction(
                calibration_embedding,
                calibration,
                calibration_embedding,
                calibration_target_residual,
                calibration,
                k=k,
                temperature=temperature,
                scale=scale,
                shrinkage=shrinkage,
                scope=scope,
                exclude_query_identity=True,
            )
            prediction = calibration_source_prediction + correction
            row = {
                "scope": scope, "k": k, "temperature": temperature,
                "scale": scale, "shrinkage": shrinkage,
                **metric_row(calibration_targets, prediction, calibration),
            }
            scope_rows.append(row)
            target_grid_rows.append(row)
        best_target[scope] = choose_grid(scope_rows)
    pd.DataFrame(target_grid_rows).sort_values(
        ["scope", "macro_stratum_mae", "mae", "rmse"]
    ).to_csv(output_dir / "target_calibration_grid.csv", index=False)

    if args.fix_source_scale_zero:
        target_source_correction = np.zeros(len(target_features), dtype=float)
    else:
        target_source_indices, target_source_distances = top_neighbours(
            target_embedding, source_embedding, int(best_source["k"])
        )
        target_source_correction = best_source["scale"] * weighted_from_top_neighbours(
            target_source_indices,
            target_source_distances,
            source_oof_residual,
            k=int(best_source["k"]),
            temperature=float(best_source["temperature"]),
        )
    target_source_prediction = target_base + target_source_correction
    global_target = best_target["global"]
    target_global_correction, _ = target_memory_correction(
        target_embedding, target_features, calibration_embedding,
        calibration_target_residual, calibration,
        k=int(global_target["k"]),
        temperature=float(global_target["temperature"]),
        scale=float(global_target["scale"]),
        shrinkage=float(global_target["shrinkage"]),
        scope="global", exclude_query_identity=False,
    )
    hierarchical_target = best_target["hierarchical"]
    target_hierarchical_correction, target_diagnostics = target_memory_correction(
        target_embedding, target_features, calibration_embedding,
        calibration_target_residual, calibration,
        k=int(hierarchical_target["k"]),
        temperature=float(hierarchical_target["temperature"]),
        scale=float(hierarchical_target["scale"]),
        shrinkage=float(hierarchical_target["shrinkage"]),
        scope="hierarchical", exclude_query_identity=False,
    )
    budget_indices = deterministic_budget_48(calibration)
    budget_frame = calibration.iloc[budget_indices].reset_index(drop=True)
    budget_embedding = calibration_embedding[budget_indices]
    budget_residual = calibration_target_residual[budget_indices]
    target_budget_correction, _ = target_memory_correction(
        target_embedding, target_features, budget_embedding, budget_residual,
        budget_frame,
        k=int(hierarchical_target["k"]),
        temperature=float(hierarchical_target["temperature"]),
        scale=float(hierarchical_target["scale"]),
        shrinkage=float(hierarchical_target["shrinkage"]),
        scope="hierarchical", exclude_query_identity=False,
    )

    domain_prediction = None
    paper_eb_prediction = None
    domain_recipe_selected = None
    if domain_recipe is not None:
        domain_recipe_selected = domain_recipe["selected"]
        paper_correction, domain_paper_seen = paper_domain_correction(
            target_features,
            calibration,
            calibration_target_residual,
            scope=str(domain_recipe_selected["scope"]),
            paper_shrinkage=float(
                domain_recipe_selected["paper_shrinkage"]
            ),
            stratum_shrinkage=float(
                domain_recipe_selected["stratum_shrinkage"]
            ),
        )
        paper_eb_prediction = target_source_prediction + paper_correction
        chemistry_method = str(domain_recipe_selected["chemistry_method"])
        if chemistry_method == "cf_hierarchical_b112":
            chemistry_correction = target_hierarchical_correction
        elif chemistry_method == "target_global_b112":
            chemistry_correction = target_global_correction
        else:
            raise ValueError(
                f"Unsupported recipe chemistry method: {chemistry_method}"
            )
        domain_correction = (
            float(domain_recipe_selected["paper_weight"]) * paper_correction
            + float(domain_recipe_selected["chemistry_weight"])
            * chemistry_correction
        )
        domain_correction[~domain_paper_seen] = (
            float(domain_recipe_selected["unseen_global_weight"])
            * float(np.mean(calibration_target_residual))
        )
        domain_prediction = target_source_prediction + domain_correction

    calibration_global_correction, _ = target_memory_correction(
        calibration_embedding, calibration, calibration_embedding,
        calibration_target_residual, calibration,
        k=int(global_target["k"]),
        temperature=float(global_target["temperature"]),
        scale=float(global_target["scale"]),
        shrinkage=float(global_target["shrinkage"]),
        scope="global", exclude_query_identity=True,
    )
    calibration_hierarchical_correction, _ = target_memory_correction(
        calibration_embedding, calibration, calibration_embedding,
        calibration_target_residual, calibration,
        k=int(hierarchical_target["k"]),
        temperature=float(hierarchical_target["temperature"]),
        scale=float(hierarchical_target["scale"]),
        shrinkage=float(hierarchical_target["shrinkage"]),
        scope="hierarchical", exclude_query_identity=True,
    )
    calibration_predictions = calibration[
        ["reaction_id", "Yield", "substrate_smiles", "substrate_scaffold",
         "catalyst_group", "paper_group"]
    ].copy()
    calibration_predictions["base"] = calibration_base
    calibration_predictions["cf_source_eknn"] = calibration_source_prediction
    calibration_predictions["target_global_b112"] = (
        calibration_source_prediction + calibration_global_correction
    )
    calibration_predictions[PRIMARY_METHOD] = (
        calibration_source_prediction + calibration_hierarchical_correction
    )
    calibration_predictions.to_csv(
        output_dir / "calibration_leave_identity_out_predictions.csv", index=False
    )

    calibration_identities = set(calibration["substrate_smiles"].astype(str))
    calibration_papers = set(calibration["paper_group"].astype(str))
    predictions = target_features[
        ["reaction_id", "substrate_smiles", "substrate_scaffold",
         "catalyst_group", "paper_group"]
    ].copy()
    predictions["calibration_identity_seen"] = predictions[
        "substrate_smiles"
    ].astype(str).isin(calibration_identities)
    predictions["calibration_paper_seen"] = predictions[
        "paper_group"
    ].astype(str).isin(calibration_papers)
    predictions["base"] = target_base
    predictions["global_offset_b112"] = target_base + float(
        np.mean(calibration_targets - calibration_base)
    )
    predictions["cf_source_eknn"] = target_source_prediction
    predictions["target_global_b112"] = target_source_prediction + target_global_correction
    predictions[PRIMARY_METHOD] = (
        target_source_prediction + target_hierarchical_correction
    )
    predictions["cf_hierarchical_b48"] = (
        target_source_prediction + target_budget_correction
    )
    if domain_recipe is not None:
        predictions["global_offset"] = target_source_prediction + float(
            np.mean(calibration_target_residual)
        )
        predictions["target_global"] = (
            target_source_prediction + target_global_correction
        )
        predictions["chem_hierarchical"] = (
            target_source_prediction + target_hierarchical_correction
        )
        predictions["paper_eb"] = paper_eb_prediction
        candidate_column = (
            "paper_coldstart_hr_eknn"
            if domain_recipe.get("confirmation_protocol")
            == "paper-coldstart-HR-eKNN-confirmatory-v1"
            else "domain_gated_hr_eknn"
        )
        predictions[candidate_column] = domain_prediction
    predictions = pd.concat([predictions, target_diagnostics], axis=1)
    prediction_path = output_dir / "target_predictions_unscored.csv"
    predictions.to_csv(prediction_path, index=False)
    prediction_hash = sha256_file(prediction_path)

    split_assignments = source[["reaction_id", "substrate_scaffold"]].copy()
    split_assignments["epoch_selection_role"] = "fit"
    split_assignments.loc[val_indices, "epoch_selection_role"] = "validation"
    split_assignments["crossfit_fold"] = oof_fold
    split_assignments.to_csv(output_dir / "source_split_assignments.csv", index=False)
    source_oof = source[["reaction_id", "substrate_scaffold", "Yield"]].copy()
    source_oof["crossfit_fold"] = oof_fold
    source_oof["oof_prediction"] = oof_prediction
    source_oof["oof_residual"] = source_oof_residual
    source_oof["full_source_prediction"] = source_base
    source_oof.to_csv(output_dir / "source_oof_predictions.csv", index=False)
    torch.save(
        {
            "model": final_model.state_dict(),
            "model_config": model_config(fixed_args),
            "epoch": selected_epoch,
            "std_mean": full_standardizer.mean,
            "std_std": full_standardizer.std,
            "source_validation": source_validation,
            "target_test_yield_loaded": False,
        },
        output_dir / "mpnn_full_source.pt",
    )
    if domain_recipe is not None:
        paper_coldstart = (
            domain_recipe.get("confirmation_protocol")
            == "paper-coldstart-HR-eKNN-confirmatory-v1"
        )
        primary_method = (
            "paper_coldstart_hr_eknn" if paper_coldstart
            else "domain_gated_hr_eknn"
        )
        methods = [
            "base", "global_offset", "target_global", "chem_hierarchical",
            "paper_eb", primary_method,
        ]
        protocol_name = (
            "paper-coldstart-HR-eKNN-confirmatory-v1" if paper_coldstart
            else "domain-gated-HR-eKNN-confirmatory-v1"
        )
        protocol_file = (
            ROOT / "experiments" / "photoredox_paper_coldstart_confirmatory_protocol.md"
            if paper_coldstart
            else ROOT / "experiments" / "photoredox_domain_gated_confirmatory_protocol.md"
        )
    else:
        methods = METHODS
        primary_method = PRIMARY_METHOD
        protocol_name = "CF-HR-eKNN-v1"
        protocol_file = (
            ROOT / "experiments" / "photoredox_full_cf_hierarchical_eknn_protocol.md"
        )
    manifest = {
        "protocol": protocol_name,
        "protocol_file": str(protocol_file.resolve()) if protocol_file.is_file() else None,
        "phase": "frozen_unscored",
        "target_test_yield_loaded": False,
        "seed": args.seed,
        "device": str(device),
        "counts": {
            "source": len(source),
            "calibration": len(calibration),
            "one_per_stratum_budget": len(budget_indices),
            "target": len(target_features),
        },
        "data_contract_audit": data_contract_audit,
        "source_validation": source_validation,
        "training": {"selection": selection_info, "selected_epoch": selected_epoch,
                     "source_crossfit_skipped": bool(args.skip_source_crossfit),
                     "crossfit": crossfit_training, "full": final_info},
        "selected_hyperparameters": {
            "source": best_source,
            "target_global": global_target,
            "target_hierarchical": hierarchical_target,
        },
        "domain_recipe": domain_recipe,
        "domain_recipe_file": (
            str(args.domain_recipe.resolve()) if args.domain_recipe is not None else None
        ),
        "methods": methods,
        "primary_method": primary_method,
        "input_hashes": {
            "source_train_sha256": sha256_file(source_path),
            "calibration_sha256": sha256_file(calibration_path),
            "target_features_sha256": feature_hash(target_features),
        },
        "frozen_prediction_file": str(prediction_path),
        "frozen_prediction_sha256": prediction_hash,
    }
    write_json(output_dir / "freeze_manifest.json", manifest)
    print(
        json.dumps(
            {"frozen_prediction_sha256": prediction_hash,
             "selected_epoch": selected_epoch,
             "source_validation": source_validation,
             "selected": manifest["selected_hyperparameters"],
             "target_yield_loaded": False},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def paper_cluster_bootstrap(
    frame: pd.DataFrame,
    target: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    papers = frame["paper_group"].astype(str).to_numpy()
    unique = np.unique(papers)
    counts = np.zeros(len(unique), dtype=float)
    difference_sums = np.zeros(len(unique), dtype=float)
    base_error = np.abs(base - target)
    candidate_error = np.abs(candidate - target)
    for index, paper in enumerate(unique):
        mask = papers == paper
        counts[index] = int(mask.sum())
        difference_sums[index] = float(
            np.sum(candidate_error[mask] - base_error[mask])
        )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(iterations, len(unique)))
    sampled_counts = counts[draws].sum(axis=1)
    changes = difference_sums[draws].sum(axis=1) / sampled_counts
    return {
        "delta_mae_candidate_minus_base": float(
            np.mean(candidate_error - base_error)
        ),
        "bootstrap_mean": float(np.mean(changes)),
        "ci95_low": float(np.quantile(changes, 0.025)),
        "ci95_high": float(np.quantile(changes, 0.975)),
        "iterations": int(iterations),
        "clusters": int(len(unique)),
    }


def score(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_path = Path(manifest["frozen_prediction_file"])
    actual_hash = sha256_file(prediction_path)
    expected_hash = manifest["frozen_prediction_sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Frozen prediction hash mismatch: {actual_hash} != {expected_hash}"
        )
    predictions = pd.read_csv(prediction_path)
    label_path = args.task_dir.resolve() / "target_test.csv"
    labels = pd.read_csv(label_path, usecols=["reaction_id", "Yield"])
    if not labels["reaction_id"].is_unique:
        raise RuntimeError("Target reaction identifiers are not unique")
    scored = predictions.merge(labels, on="reaction_id", how="left", validate="one_to_one")
    if scored["Yield"].isna().any() or len(scored) != len(predictions):
        raise RuntimeError("Frozen predictions and locked target labels do not align")
    target = scored["Yield"].to_numpy(dtype=float)

    subgroup_masks = {
        "all": np.ones(len(scored), dtype=bool),
        "identity_seen": scored["calibration_identity_seen"].astype(bool).to_numpy(),
        "identity_unseen": ~scored["calibration_identity_seen"].astype(bool).to_numpy(),
        "paper_unseen": ~scored["calibration_paper_seen"].astype(bool).to_numpy(),
        "identity_and_paper_unseen": (
            ~scored["calibration_identity_seen"].astype(bool).to_numpy()
            & ~scored["calibration_paper_seen"].astype(bool).to_numpy()
        ),
    }
    rows = []
    for subgroup, mask in subgroup_masks.items():
        subgroup_frame = scored.loc[mask].reset_index(drop=True)
        subgroup_target = target[mask]
        for method in manifest["methods"]:
            if int(mask.sum()) == 0:
                empty_metrics = {
                    "r2": np.nan,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "bias": np.nan,
                    "p90_abs_error": np.nan,
                    "macro_scaffold_mae": np.nan,
                    "macro_stratum_mae": np.nan,
                    "macro_paper_mae": np.nan,
                }
            else:
                empty_metrics = metric_row(
                    subgroup_target,
                    scored.loc[mask, method].to_numpy(dtype=float),
                    subgroup_frame,
                )
            row = {
                "subgroup": subgroup,
                "method": method,
                "n": int(mask.sum()),
                **empty_metrics,
            }
            rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "test_metrics_by_subgroup.csv", index=False)

    bootstrap_rows = []
    base = scored["base"].to_numpy(dtype=float)
    for method in manifest["methods"]:
        if method == "base":
            continue
        bootstrap_rows.append(
            {
                "method": method,
                **paper_cluster_bootstrap(
                    scored,
                    target,
                    base,
                    scored[method].to_numpy(dtype=float),
                    iterations=args.bootstrap_iterations,
                    seed=args.seed,
                ),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(output_dir / "paper_cluster_bootstrap.csv", index=False)

    scored["stratum"] = (
        scored["substrate_scaffold"].astype(str)
        + "||"
        + scored["catalyst_group"].astype(str)
    )
    per_stratum_rows = []
    for stratum, group in scored.groupby("stratum", sort=True):
        row = {"stratum": stratum, "n": int(len(group))}
        group_target = group["Yield"].to_numpy(dtype=float)
        for method in manifest["methods"]:
            row[f"{method}_mae"] = float(
                mean_absolute_error(group_target, group[method].to_numpy(dtype=float))
            )
        per_stratum_rows.append(row)
    pd.DataFrame(per_stratum_rows).to_csv(
        output_dir / "test_metrics_by_stratum.csv", index=False
    )
    scored.to_csv(output_dir / "target_predictions_scored.csv", index=False)

    all_metrics = metrics[metrics["subgroup"] == "all"].set_index("method")
    unseen_metrics = metrics[
        metrics["subgroup"] == "identity_unseen"
    ].set_index("method")
    paper_unseen_metrics = metrics[
        metrics["subgroup"] == "paper_unseen"
    ].set_index("method")
    primary_method = str(manifest["primary_method"])
    primary_bootstrap = bootstrap.set_index("method").loc[primary_method]
    micro_improvement = float(
        all_metrics.loc["base", "mae"] - all_metrics.loc[primary_method, "mae"]
    )
    macro_improvement = float(
        all_metrics.loc["base", "macro_stratum_mae"]
        - all_metrics.loc[primary_method, "macro_stratum_mae"]
    )
    identity_unseen_delta = float(
        unseen_metrics.loc[primary_method, "mae"]
        - unseen_metrics.loc["base", "mae"]
    )
    paper_unseen_delta = (
        float(
            paper_unseen_metrics.loc[primary_method, "mae"]
            - paper_unseen_metrics.loc["base", "mae"]
        )
        if int(paper_unseen_metrics.loc["base", "n"]) > 0
        else float("nan")
    )
    p90_delta = float(
        all_metrics.loc[primary_method, "p90_abs_error"]
        - all_metrics.loc["base", "p90_abs_error"]
    )
    if manifest["protocol"] == "paper-coldstart-HR-eKNN-confirmatory-v1":
        macro_paper_improvement = float(
            all_metrics.loc["base", "macro_paper_mae"]
            - all_metrics.loc[primary_method, "macro_paper_mae"]
        )
        gates = {
            "micro_mae_improvement_at_least_0.005": micro_improvement >= 0.005,
            "macro_paper_improvement_at_least_0.005": macro_paper_improvement >= 0.005,
            "paper_bootstrap_upper_below_zero": float(primary_bootstrap["ci95_high"]) < 0.0,
            "identity_unseen_mae_worsening_at_most_0.002": identity_unseen_delta <= 0.002,
            "p90_worsening_at_most_0.01": p90_delta <= 0.01,
        }
    elif manifest["protocol"] == "domain-gated-HR-eKNN-confirmatory-v1":
        macro_paper_improvement = float("nan")
        gates = {
            "micro_mae_improvement_at_least_0.005": micro_improvement >= 0.005,
            "macro_stratum_improvement_at_least_0.005": macro_improvement >= 0.005,
            "paper_bootstrap_upper_below_zero": float(primary_bootstrap["ci95_high"]) < 0.0,
            "paper_unseen_mae_worsening_at_most_0.002": paper_unseen_delta <= 0.002,
            "p90_worsening_at_most_0.01": p90_delta <= 0.01,
        }
    else:
        macro_paper_improvement = float("nan")
        gates = {
            "macro_stratum_improvement_at_least_0.005": macro_improvement >= 0.005,
            "paper_bootstrap_upper_below_zero": float(primary_bootstrap["ci95_high"]) < 0.0,
            "identity_unseen_mae_worsening_at_most_0.005": identity_unseen_delta <= 0.005,
            "p90_worsening_at_most_0.01": p90_delta <= 0.01,
        }
    if all(gates.values()):
        status = "validated"
    elif macro_improvement > 0 and float(primary_bootstrap["ci95_high"]) >= 0:
        status = "exploratory_signal"
    else:
        status = "not_supported"
    score_summary = {
        "protocol": manifest["protocol"],
        "prediction_sha256_verified_before_label_load": True,
        "prediction_sha256": actual_hash,
        "target_label_file_sha256_after_unlock": sha256_file(label_path),
        "primary_method": primary_method,
        "promotion_status": status,
        "promotion_gates": gates,
        "promotion_statistics": {
            "micro_mae_improvement": micro_improvement,
            "macro_stratum_mae_improvement": macro_improvement,
            "macro_paper_mae_improvement": macro_paper_improvement,
            "paper_bootstrap_delta_mae_ci95": [
                float(primary_bootstrap["ci95_low"]),
                float(primary_bootstrap["ci95_high"]),
            ],
            "identity_unseen_mae_delta": identity_unseen_delta,
            "paper_unseen_mae_delta": paper_unseen_delta,
            "p90_abs_error_delta": p90_delta,
        },
        "all_test_metrics": all_metrics.reset_index().to_dict(orient="records"),
    }
    write_json(output_dir / "score_summary.json", score_summary)
    print(json.dumps(score_summary, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train-freeze", "score"), required=True)
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("data/photoredox_ood_full/01_scaffold_fewshot_full_10k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/photoredox_full_cf_hierarchical_eknn_s42"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--crossfit-folds", type=int, default=3)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument(
        "--domain-recipe",
        type=Path,
        help="Development-frozen domain-gating recipe JSON",
    )
    parser.add_argument(
        "--fix-source-scale-zero",
        action="store_true",
        help="Set the source-residual memory scale to zero",
    )
    parser.add_argument(
        "--skip-source-crossfit",
        action="store_true",
        help="Skip source OOF residual generation when its scale is fixed at zero",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "train-freeze":
        train_and_freeze(args)
    else:
        score(args)


if __name__ == "__main__":
    main()

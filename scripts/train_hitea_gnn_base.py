#!/usr/bin/env python3
"""Fit a HiTEA GNN while excluding target screens and their exact input groups.

The input table masks held-out yields. Graphs come from build_hitea_gnn_graph_cache.py.
The runner saves a checkpoint, embeddings, base predictions and an exclusion audit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "train_gnn_eknn", ROOT / "scripts" / "train_gnn_eknn.py"
)
train = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train)

from eknn_reaction import utils
from eknn_reaction.data.dataset import Standardizer

SPEC_CACHE = importlib.util.spec_from_file_location(
    "build_hitea_gnn_graph_cache", ROOT / "scripts" / "build_hitea_gnn_graph_cache.py"
)
graph_cache = importlib.util.module_from_spec(SPEC_CACHE)
assert SPEC_CACHE.loader is not None
SPEC_CACHE.loader.exec_module(graph_cache)

CONDITION_NUMERIC_COLS = (
    "reaction_temperature_c",
    "reaction_time_h",
    "catalyst_1_equiv",
    "catalyst_2_equiv",
    "reactant_1_equiv",
    "reactant_2_equiv",
    "reactant_3_equiv",
    "reagent_1_equiv",
    "reagent_2_equiv",
)
CONDITION_CATEGORICAL_COLS = ("solvent_name", "reaction_class", "reaction_group")
MAX_EPOCHS = 80
VAL_FRACTION = 0.15


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    def default(value):
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=default)
        + "\n",
        encoding="utf-8",
    )


def model_args(epochs: int) -> SimpleNamespace:
    """Frozen architecture, copied from run_photoredox_full_cf_hierarchical_eknn."""
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


def normalise_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def dataset_shim() -> SimpleNamespace:
    return SimpleNamespace(
        name="hitea_global_exclusion",
        condition_cols=list(CONDITION_NUMERIC_COLS),
        condition_log1p_cols=(),
        condition_categorical_cols=list(CONDITION_CATEGORICAL_COLS),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/hitea_retrieval_shift/v1"))
    parser.add_argument(
        "--graph-cache", type=Path, default=Path("results/hitea_gnn_development/graph_cache")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/hitea_gnn_development")
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encode-batch-size", type=int, default=64)
    args = parser.parse_args()
    started = time.time()
    seed_dir = args.output / f"s{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train.LOG.addHandler(
        logging.FileHandler(seed_dir / "training.log", encoding="utf-8")
    )

    # ── Data, lock audit, global-exclusion source ────────────────────────
    manifest = json.loads((args.data_root / "materialization_manifest.json").read_text())
    table = pd.read_csv(args.data_root / "development_modeling_table.csv", low_memory=False)
    locked = pd.read_csv(args.data_root / "locked_test_registry.csv")
    locked_ids = set(locked["record_id"].astype(str))
    masked = table.loc[table["Yield"].isna(), "record_id"].astype(str)
    if len(masked) != len(locked_ids) or set(masked) != locked_ids:
        raise AssertionError("Masked labels and locked test registry differ")

    fold_index = pd.read_csv(args.data_root / "folds" / "fold_index.csv") if (
        args.data_root / "folds" / "fold_index.csv"
    ).exists() else pd.read_csv(args.data_root / "fold_index.csv")
    all_target_screens = set(fold_index["target_screen_id"].astype(str))
    target_screen_mask = table["screen_id"].astype(str).isin(all_target_screens)
    all_target_groups = set(table.loc[target_screen_mask, "exact_input_group"].astype(str))

    in_source = (
        table["Yield"].notna()
        & ~target_screen_mask
        & ~table["exact_input_group"].astype(str).isin(all_target_groups)
    )
    source = table.loc[in_source].reset_index(drop=True)
    if source["Yield"].isna().any():
        raise AssertionError("Training source contains masked labels")
    if set(source["screen_id"].astype(str)) & all_target_screens:
        raise AssertionError("Target screens leaked into the global source")
    if set(source["exact_input_group"].astype(str)) & all_target_groups:
        raise AssertionError("Target exact_input_groups leaked into the global source")
    for fold in fold_index.itertuples(index=False):
        assignment = pd.read_csv(
            args.data_root / "folds" / str(fold.fold_id) / "target_assignment.csv"
        )
        fold_groups = set(
            table.loc[
                table["record_id"].isin(assignment["record_id"]), "exact_input_group"
            ].astype(str)
        )
        if fold_groups & set(source["exact_input_group"].astype(str)):
            raise AssertionError(f"Fold {fold.fold_id} groups leaked into the global source")

    # ── Graphs from the frozen cache ─────────────────────────────────────
    cache_manifest = json.loads(
        (args.graph_cache / "cache_manifest.json").read_text(encoding="utf-8")
    )
    expected_table_sha = manifest["development_modeling_table_sha256"]
    if cache_manifest["modeling_table_sha256"] != expected_table_sha:
        raise AssertionError("Graph cache was built from a different modeling table")
    if cache_manifest["n_rows_without_graph"]:
        raise AssertionError(
            f"Graph cache has {cache_manifest['n_rows_without_graph']} unencodable rows"
        )
    graphs_by_key = torch.load(
        args.graph_cache / "role_tuple_graphs.pt", weights_only=False
    )
    row_keys = [
        graph_cache.tuple_key(graph_cache.canonical_role_tuple(row))
        for row in table[list(graph_cache.ROLE_COLUMNS)].astype(object).to_dict("records")
    ]
    if any(key not in graphs_by_key for key in set(row_keys)):
        raise AssertionError("Table role tuples missing from the graph cache")

    # Build aligned graph lists (shared objects; clones are made downstream).
    table_position = pd.Series(np.arange(len(table)), index=table["record_id"].to_numpy())
    source_positions = table_position.loc[source["record_id"].to_numpy()].to_numpy(int)
    source_graphs = [graphs_by_key[row_keys[position]] for position in source_positions]
    all_graphs = [graphs_by_key[key] for key in row_keys]
    print(
        f"source={len(source_graphs)} rows from cache of {len(graphs_by_key)} tuples",
        flush=True,
    )

    # ── Fold-safe condition features, fitted on the global source only ──
    ds = dataset_shim()
    source_features, table_features, condition_info = train.prepare_condition_features(
        source, table, ds, include_categorical=True
    )
    if not np.isfinite(source_features).all() or not np.isfinite(table_features).all():
        raise AssertionError("Non-finite condition features")
    source_graphs = train.attach_condition_features(source_graphs, source_features)
    table_graphs = train.attach_condition_features(all_graphs, table_features)
    condition_dim = int(source_features.shape[1])
    print(f"condition_dim={condition_dim}", flush=True)

    # ── Epoch selection on a group-disjoint 15% validation split ────────
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    source_groups = source["exact_input_group"].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=args.seed)
    fit_indices, val_indices = next(
        splitter.split(np.zeros(len(source)), groups=source_groups)
    )
    if set(source_groups[fit_indices]) & set(source_groups[val_indices]):
        raise AssertionError("Epoch-selection group leakage detected")

    selection_args = model_args(MAX_EPOCHS)
    utils.set_seed(args.seed, deterministic=True)
    selection_model = train.build_model(selection_args, condition_dim=condition_dim).to(device)
    targets = source["Yield"].to_numpy(dtype=float)
    fit_standardizer = Standardizer(
        float(np.mean(targets[fit_indices])), float(np.std(targets[fit_indices]))
    )
    selection_info = train.train_gnn_model(
        selection_model,
        [source_graphs[int(index)] for index in fit_indices],
        targets[fit_indices],
        fit_standardizer,
        device,
        selection_args,
        stage_name="hitea_gnn_epoch_selection",
        validation_graphs=[source_graphs[int(index)] for index in val_indices],
        validation_targets=targets[val_indices],
    )
    source_validation = train.evaluate_static(
        selection_model,
        [source_graphs[int(index)] for index in val_indices],
        targets[val_indices],
        device,
        fit_standardizer,
    )
    selected_epoch = int(selection_info["epoch"])
    del selection_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Full-source retrain, frozen (no cross-fit: retrieval-only usage) ─
    fixed_args = model_args(selected_epoch)
    full_standardizer = Standardizer(float(np.mean(targets)), float(np.std(targets)))
    utils.set_seed(args.seed, deterministic=True)
    final_model = train.build_model(fixed_args, condition_dim=condition_dim).to(device)
    final_info = train.train_gnn_model(
        final_model,
        source_graphs,
        targets,
        full_standardizer,
        device,
        fixed_args,
        stage_name="hitea_gnn_full_source",
    )

    embedding, base_norm = train.encode_reactions(
        final_model, table_graphs, device, batch_size=args.encode_batch_size
    )
    base_pred = full_standardizer.denormalize(base_norm)
    embedding = normalise_rows(embedding)

    source_base_norm_check = base_norm[source_positions]
    source_base = full_standardizer.denormalize(source_base_norm_check)
    source_r2 = float(
        1.0
        - np.sum((source_base - targets) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    )
    source_mae = float(np.abs(source_base - targets).mean())

    checkpoint = {
        "model": final_model.state_dict(),
        "model_config": model_config(fixed_args),
        "condition_dim": condition_dim,
        "condition_info": condition_info,
        "epoch": selected_epoch,
        "std_mean": full_standardizer.mean,
        "std_std": full_standardizer.std,
        "selection": selection_info,
        "source_validation": source_validation,
        "max_epochs": MAX_EPOCHS,
        "val_fraction": VAL_FRACTION,
        "exclusion_protocol": "global_exclusion_26_screens_and_their_exact_input_groups",
        "target_test_yield_loaded": False,
    }
    torch.save(checkpoint, seed_dir / "hitea_gnn_full_source.pt")

    np.savez_compressed(
        seed_dir / "embeddings.npz",
        record_id=table["record_id"].astype(str).to_numpy(),
        embedding=embedding.astype(np.float32),
        base_pred=base_pred.astype(np.float64),
        in_global_source=in_source.to_numpy(dtype=bool),
    )

    training_manifest = {
        "protocol": "hitea-gnn-global-exclusion-base-v1",
        "status": "diagnostic_only",
        "locked_target_test_scored": False,
        "seed": args.seed,
        "device": str(device),
        "n_source_train": int(len(source)),
        "n_target_screens_excluded": int(len(all_target_screens)),
        "n_target_exact_input_groups_excluded": int(len(all_target_groups)),
        "n_development_table_rows_encoded": int(len(table)),
        "condition_dim": condition_dim,
        "condition_numeric_cols": list(CONDITION_NUMERIC_COLS),
        "condition_categorical_cols": list(CONDITION_CATEGORICAL_COLS),
        "condition_fit_scope": "global source only",
        "source_validation": source_validation,
        "selected_epoch": selected_epoch,
        "selection": selection_info,
        "full_training": final_info,
        "full_source_insample": {"r2": source_r2, "mae": source_mae},
        "global_source_record_id_sha256": sha256_text(
            "\n".join(sorted(source["record_id"].astype(str)))
        ),
        "input_hashes": {
            "development_modeling_table_sha256": expected_table_sha,
            "locked_test_registry_sha256": manifest["locked_test_registry_sha256"],
            "graph_cache_sha256": cache_manifest["graph_cache_sha256"],
        },
        "output_hashes": {
            "checkpoint_sha256": sha256_file(seed_dir / "hitea_gnn_full_source.pt"),
            "embeddings_sha256": sha256_file(seed_dir / "embeddings.npz"),
        },
        "wall_time_seconds": float(time.time() - started),
    }
    write_json(seed_dir / "training_manifest.json", training_manifest)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "selected_epoch": selected_epoch,
                "source_validation": source_validation,
                "full_source_insample": {"r2": source_r2, "mae": source_mae},
                "target_yield_loaded": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

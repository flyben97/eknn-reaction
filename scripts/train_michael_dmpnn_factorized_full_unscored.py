#!/usr/bin/env python3
"""Train full-data D-MPNN factorized models and export unscored predictions."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eknn_reaction import utils
from eknn_reaction.data import get_dataset
from eknn_reaction.data.dataset import Standardizer
from scripts import train_gnn_eknn as core


def model_args(seed: int, output_root: Path):
    original_argv = sys.argv
    try:
        sys.argv = [
            "train_gnn_eknn.py",
            "--dataset", "michael_ddg_random",
            "--epochs", "70",
            "--batch-size", "32",
            "--d-model", "128",
            "--n-layers", "3",
            "--n-heads", "4",
            "--ff-dim", "384",
            "--dropout", "0.15",
            "--backbone", "dmpnn",
            "--pe-dim", "0",
            "--readout", "mean_max",
            "--role-factorized-readout",
            "--use-condition-features",
            "--deterministic",
            "--graph-cache-dir",
            (
                "results/michael_gnn_xgb_challenge/"
                "role_factorized_cache"
            ),
            "--out-dir", str(output_root),
            "--set", "train.device=cuda", f"train.seed={seed}",
        ]
        return core.parse_args()
    finally:
        sys.argv = original_argv


def selected_result_path(seed: int) -> Path:
    if seed == 42:
        pattern = (
            "results/michael_gnn_xgb_challenge/role_factorized_screen/"
            "dmpnn_factorized_pe0/michael_ddg_random/"
            "validation_*_s42_ep70.json"
        )
    else:
        pattern = (
            "results/michael_gnn_xgb_challenge/"
            f"dmpnn_factorized_multiseed/seed_{seed}/"
            "michael_ddg_random/"
            f"validation_*_s{seed}_ep70.json"
        )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one validation result for seed {seed}, got {matches}"
        )
    return Path(matches[0])


def train_seed(seed: int, output_root: Path) -> None:
    args = model_args(seed, output_root)
    ds = get_dataset(args.dataset)
    train_graphs, train_targets, test_graphs, test_targets, _ = (
        core.load_and_build_graphs(
            ds, args, args.graph_cache_dir
        )
    )
    train_graphs, train_targets = core.filter_valid(
        train_graphs, train_targets
    )
    test_graphs, _ = core.filter_valid(test_graphs, test_targets)

    train_source = pd.read_csv(ds.train_csv)
    test_source = pd.read_csv(ds.test_csv)
    train_frame = core.graph_rows_to_frame(
        train_graphs, train_source
    )
    test_frame = core.graph_rows_to_frame(test_graphs, test_source)
    train_features, test_features, condition_info = (
        core.prepare_condition_features(
            train_frame,
            test_frame,
            ds,
            include_categorical=False,
        )
    )
    train_graphs = core.attach_condition_features(
        train_graphs, train_features
    )
    test_graphs = core.attach_condition_features(
        test_graphs, test_features
    )
    standardizer = Standardizer(
        float(np.nanmean(train_targets)),
        float(np.nanstd(train_targets)),
    )

    utils.set_seed(seed, deterministic=True)
    model = core.build_model(
        args, condition_dim=int(train_features.shape[1])
    )
    device = torch.device("cuda")
    training_info = core.train_gnn_model(
        model,
        train_graphs,
        train_targets,
        standardizer,
        device,
        args,
        stage_name=f"full_seed_{seed}",
    )

    selection_path = selected_result_path(seed)
    selection_result = json.loads(selection_path.read_text())
    selected_eknn = selection_result["selection"]["eknn"]
    predictions = core.predict_eknn_fixed_unscored(
        model,
        train_graphs,
        test_graphs,
        train_targets,
        device,
        standardizer,
        selected_eknn,
    )

    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = (
        seed_dir / "outer_predictions_unscored.csv"
    )
    prediction_frame = pd.DataFrame({
        "outer_row_index": [
            int(graph.reaction_idx) for graph in test_graphs
        ],
        ds.group_col: test_frame[ds.group_col].astype(str),
        "static_prediction": predictions["static"],
        "eknn_prediction": predictions["eknn"],
    })
    prediction_frame.to_csv(prediction_path, index=False)

    checkpoint_path = seed_dir / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "seed": seed,
        "architecture": (
            "unpretrained D-MPNN + conditioned role-factorized readout"
        ),
        "condition_dim": int(train_features.shape[1]),
        "standardizer_mean": standardizer.mean,
        "standardizer_std": standardizer.std,
        "selected_eknn": selected_eknn,
        "training_info": training_info,
    }, checkpoint_path)
    metadata = {
        "seed": seed,
        "outer_targets_scored": False,
        "outer_targets_exported": False,
        "n_train": int(len(train_graphs)),
        "n_outer_query": int(len(test_graphs)),
        "training_info": training_info,
        "condition_preprocessing": condition_info,
        "validation_selection_json": str(selection_path),
        "selected_eknn": selected_eknn,
        "predictions_csv": str(prediction_path),
        "checkpoint": str(checkpoint_path),
    }
    (seed_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    del model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/michael_gnn_xgb_challenge/"
            "dmpnn_factorized_full"
        ),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456]
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this full-data run")
    for seed in args.seeds:
        train_seed(seed, args.output_root)


if __name__ == "__main__":
    main()

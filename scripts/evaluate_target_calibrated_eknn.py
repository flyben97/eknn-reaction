#!/usr/bin/env python3
"""Few-shot target-domain eKNN calibration for a frozen reaction GNN.

Hyperparameters are selected exclusively by leave-one-out predictions inside
the labelled calibration set. The final target-test labels are loaded only for
the single fixed evaluation and never enter neighbour or parameter selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_gnn_eknn", ROOT / "scripts" / "train_gnn_eknn.py"
)
train_gnn_eknn = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(train_gnn_eknn)

from eknn_reaction.data.dataset import Standardizer
from eknn_reaction.models.graph_encoder import build_reaction_graphs


ROLE_COLUMNS = [
    "reactants_graph_smiles",
    "products_graph_smiles",
    "reagents_graph_smiles",
    "catalysts_graph_smiles",
]


def model_args(config: dict | None = None) -> SimpleNamespace:
    args = SimpleNamespace(
        d_model=64,
        n_layers=2,
        n_heads=4,
        ff_dim=128,
        dropout=0.10,
        pe_dim=3,
        condition_fusion_mode="gated_add",
        condition_node_film=False,
        condition_token_layers=2,
        component_identity_dim=32,
        backbone="gps",
        mpnn=True,
        gat=False,
        graph_transformer=False,
        edge_transformer=False,
        graphormer=False,
        deep_gps=False,
        light_gps=False,
        attn_pool=False,
        readout="mean_max",
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
    )
    for key, value in (config or {}).items():
        setattr(args, key, value)
    return args


def metric_dict(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(target, prediction)),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
    }


def build_graphs(frame: pd.DataFrame):
    graphs, targets = build_reaction_graphs(
        frame,
        ROLE_COLUMNS,
        "Yield",
        add_laplacian_pe=True,
        pe_k=3,
        verbose=False,
    )
    valid = [index for index, graph in enumerate(graphs) if graph is not None]
    if len(valid) != len(frame):
        raise RuntimeError(f"Failed to graph {len(frame) - len(valid)} reactions")
    return [graphs[index] for index in valid], targets[valid]


def normalise_rows(values: np.ndarray) -> np.ndarray:
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)


def weighted_residual(
    query: np.ndarray,
    candidates: np.ndarray,
    residuals: np.ndarray,
    *,
    k: int,
    temperature: float,
) -> float:
    if len(candidates) == 0:
        return 0.0
    distances = np.linalg.norm(candidates - query, axis=1)
    k_effective = min(k, len(candidates))
    indices = np.argpartition(distances, k_effective - 1)[:k_effective]
    weights = np.exp(-distances[indices] / temperature)
    weights /= weights.sum() + 1e-12
    return float(np.dot(weights, residuals[indices]))


def corrections(
    query_embeddings: np.ndarray,
    query_scaffolds: np.ndarray,
    query_identities: np.ndarray,
    query_papers: np.ndarray,
    source_embeddings: np.ndarray,
    source_residuals: np.ndarray,
    calibration_embeddings: np.ndarray,
    calibration_residuals: np.ndarray,
    calibration_scaffolds: np.ndarray,
    calibration_identities: np.ndarray,
    calibration_papers: np.ndarray,
    *,
    k_source: int,
    k_target: int,
    temperature: float,
    source_scale: float,
    target_scale: float,
    target_scope: str,
    leave_one_out: bool,
) -> np.ndarray:
    output = np.zeros(len(query_embeddings), dtype=float)
    for index, (embedding, scaffold, identity, paper) in enumerate(
        zip(query_embeddings, query_scaffolds, query_identities, query_papers)
    ):
        source_value = weighted_residual(
            embedding,
            source_embeddings,
            source_residuals,
            k=k_source,
            temperature=temperature,
        )
        mask = np.ones(len(calibration_embeddings), dtype=bool)
        if leave_one_out:
            mask[index] = False
        if target_scope not in {
            "all", "scaffold", "paper", "paper_then_scaffold",
            "identity_then_paper_then_scaffold",
        }:
            raise ValueError(f"Unknown target scope: {target_scope}")
        if target_scope == "identity_then_paper_then_scaffold":
            identity_mask = calibration_identities == identity
            if leave_one_out:
                identity_mask[index] = False
            if identity_mask.any():
                mask &= identity_mask
            else:
                paper_mask = calibration_papers == paper
                if leave_one_out:
                    paper_mask[index] = False
                if paper_mask.any():
                    mask &= paper_mask
                else:
                    scaffold_mask = calibration_scaffolds == scaffold
                    if leave_one_out:
                        scaffold_mask[index] = False
                    if scaffold_mask.any():
                        mask &= scaffold_mask
        elif target_scope == "paper_then_scaffold":
            paper_mask = calibration_papers == paper
            if leave_one_out:
                paper_mask[index] = False
            if paper_mask.any():
                mask &= paper_mask
            else:
                scaffold_mask = calibration_scaffolds == scaffold
                if leave_one_out:
                    scaffold_mask[index] = False
                if scaffold_mask.any():
                    mask &= scaffold_mask
        elif target_scope == "paper":
            paper_mask = calibration_papers == paper
            if leave_one_out:
                paper_mask[index] = False
            if paper_mask.any():
                mask &= paper_mask
        elif target_scope == "scaffold":
            scaffold_mask = calibration_scaffolds == scaffold
            if leave_one_out:
                scaffold_mask[index] = False
            # A singleton scaffold cannot support target-domain LOO. Fall back
            # to the other calibration scaffolds for parameter selection only.
            if scaffold_mask.any():
                mask &= scaffold_mask
        target_value = weighted_residual(
            embedding,
            calibration_embeddings[mask],
            calibration_residuals[mask],
            k=k_target,
            temperature=temperature,
        )
        output[index] = source_scale * source_value + target_scale * target_value
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path(
            "data/photoredox_ood_small/05_heteroarene_scaffold_fewshot_654"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/photoredox_heteroarene_scaffold_654_selection/model.pt"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/photoredox_heteroarene_scaffold_fewshot_654"
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frames = {
        "source_train": pd.read_csv(args.task_dir / "source_train.csv"),
        "calibration": pd.read_csv(args.task_dir / "calibration.csv"),
        "target_test": pd.read_csv(args.task_dir / "target_test.csv"),
    }
    graphs = {}
    targets = {}
    for split, frame in frames.items():
        graphs[split], targets[split] = build_graphs(frame)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    model = train_gnn_eknn.build_model(model_args(model_config)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    standardizer = Standardizer(checkpoint["std_mean"], checkpoint["std_std"])

    embeddings = {}
    base_norm = {}
    base_prediction = {}
    for split in frames:
        embeddings[split], base_norm[split] = train_gnn_eknn.encode_reactions(
            model, graphs[split], device
        )
        embeddings[split] = normalise_rows(embeddings[split])
        base_prediction[split] = standardizer.denormalize(base_norm[split])

    source_residuals = standardizer.normalize(targets["source_train"]) - base_norm[
        "source_train"
    ]
    calibration_residuals = standardizer.normalize(
        targets["calibration"]
    ) - base_norm["calibration"]
    calibration_scaffolds = frames["calibration"][
        "heteroarene_scaffold"
    ].astype(str).to_numpy()
    test_scaffolds = frames["target_test"]["heteroarene_scaffold"].astype(
        str
    ).to_numpy()
    calibration_identities = frames["calibration"][
        "heteroarene_smiles"
    ].astype(str).to_numpy()
    test_identities = frames["target_test"]["heteroarene_smiles"].astype(
        str
    ).to_numpy()
    calibration_papers = frames["calibration"]["paper_group"].astype(str).to_numpy()
    test_papers = frames["target_test"]["paper_group"].astype(str).to_numpy()

    grid = []
    for values in itertools.product(
        (5, 10, 20, 50),
        (1, 2, 3, 5),
        (0.05, 0.1, 0.2, 0.5),
        (0.0, 0.5, 1.0),
        (0.0, 0.25, 0.5, 0.75, 1.0),
        (
            "all", "scaffold", "paper", "paper_then_scaffold",
            "identity_then_paper_then_scaffold",
        ),
    ):
        k_source, k_target, temperature, source_scale, target_scale, scope = values
        correction = corrections(
            embeddings["calibration"],
            calibration_scaffolds,
            calibration_identities,
            calibration_papers,
            embeddings["source_train"],
            source_residuals,
            embeddings["calibration"],
            calibration_residuals,
            calibration_scaffolds,
            calibration_identities,
            calibration_papers,
            k_source=k_source,
            k_target=k_target,
            temperature=temperature,
            source_scale=source_scale,
            target_scale=target_scale,
            target_scope=scope,
            leave_one_out=True,
        )
        prediction = standardizer.denormalize(
            base_norm["calibration"] + correction
        )
        score = metric_dict(targets["calibration"], prediction)
        row = {
            "k_source": k_source,
            "k_target": k_target,
            "temperature": temperature,
            "source_scale": source_scale,
            "target_scale": target_scale,
            "target_scope": scope,
            **score,
        }
        grid.append(row)
    best = max(grid, key=lambda row: (row["r2"], -row["mae"]))

    test_correction = corrections(
        embeddings["target_test"],
        test_scaffolds,
        test_identities,
        test_papers,
        embeddings["source_train"],
        source_residuals,
        embeddings["calibration"],
        calibration_residuals,
        calibration_scaffolds,
        calibration_identities,
        calibration_papers,
        k_source=int(best["k_source"]),
        k_target=int(best["k_target"]),
        temperature=float(best["temperature"]),
        source_scale=float(best["source_scale"]),
        target_scale=float(best["target_scale"]),
        target_scope=str(best["target_scope"]),
        leave_one_out=False,
    )
    target_prediction = standardizer.denormalize(
        base_norm["target_test"] + test_correction
    )

    report = {
        "task": "few_shot_unseen_scaffold_target_adaptation",
        "model": "frozen_source_GNN_plus_target_calibrated_eKNN",
        "backbone": model_config.get("backbone", "gps"),
        "model_config": model_config,
        "counts": {split: int(len(frame)) for split, frame in frames.items()},
        "base_gnn": {
            "calibration": metric_dict(
                targets["calibration"], base_prediction["calibration"]
            ),
            "target_test": metric_dict(
                targets["target_test"], base_prediction["target_test"]
            ),
        },
        "selection": {
            "protocol": "leave-one-out on calibration only",
            "selected": best,
            "n_grid_candidates": int(len(grid)),
        },
        "target_calibrated_eknn": {
            "target_test": metric_dict(targets["target_test"], target_prediction)
        },
        "test_evaluated_once": True,
        "interpretation": (
            "Target scaffold identities are absent from source GNN training "
            "but represented by labelled calibration examples. This is "
            "few-shot target adaptation, not zero-shot scaffold OOD."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(grid).sort_values("r2", ascending=False).to_csv(
        args.output_dir / "calibration_loo_grid.csv", index=False
    )
    output = frames["target_test"][[
        "reaction_id", "heteroarene_scaffold", "heteroarene_smiles", "Yield"
    ]].copy()
    output["base_gnn_prediction"] = base_prediction["target_test"]
    output["target_calibrated_eknn_prediction"] = target_prediction
    output.to_csv(args.output_dir / "target_test_predictions.csv", index=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

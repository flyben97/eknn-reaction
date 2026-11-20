#!/usr/bin/env python3
"""Freeze a paper-held-out Photoredox cold-start task without reading Yield."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from prepare_photoredox_domain_gated_confirmatory import (
    feature_hash,
    materialize,
    sha256_file,
    stable_key,
    write_json,
)


def choose_target_papers(
    frame: pd.DataFrame,
    *,
    candidate_limit: int,
    min_paper_raw_rows: int,
    max_paper_raw_rows: int,
    min_identities: int,
    min_scaffolds: int,
    min_source_catalyst_rows: int,
    min_stratum_rows: int,
    min_supported_paper_rows: int,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    stats = frame.groupby("paper_group").agg(
        n_reactions=("reaction_id", "size"),
        n_identities=("substrate_smiles", "nunique"),
        n_scaffolds=("substrate_scaffold", "nunique"),
    )
    candidates = set(
        stats.index[
            stats["n_reactions"].between(min_paper_raw_rows, max_paper_raw_rows)
            & (stats["n_identities"] >= min_identities)
            & (stats["n_scaffolds"] >= min_scaffolds)
        ]
    )
    selected = set(
        sorted(candidates, key=lambda value: stable_key(str(value)))[
            :candidate_limit
        ]
    )
    previous: set[str] | None = None
    eligible = frame.iloc[0:0].copy()
    while selected != previous:
        previous = set(selected)
        source = frame.loc[~frame["paper_group"].isin(selected)]
        supported_catalysts = set(
            source["catalyst_group"]
            .value_counts()
            .loc[lambda values: values >= min_source_catalyst_rows]
            .index
        )
        eligible = frame.loc[
            frame["paper_group"].isin(selected)
            & frame["catalyst_group"].isin(supported_catalysts)
        ].copy()
        stratum_size = eligible.groupby(
            ["paper_group", "catalyst_group"]
        )["reaction_id"].transform("size")
        eligible = eligible.loc[stratum_size >= min_stratum_rows].copy()
        paper_size = eligible["paper_group"].value_counts()
        selected = set(
            paper_size.index[paper_size >= min_supported_paper_rows]
        )
    source = frame.loc[~frame["paper_group"].isin(selected)].copy()
    eligible = eligible.loc[eligible["paper_group"].isin(selected)].copy()
    return selected, source, eligible


def split_target(
    target_pool: pd.DataFrame, calibration_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_indices: list[int] = []
    for _, part in target_pool.groupby(
        ["paper_group", "catalyst_group"], sort=True
    ):
        n_selected = max(
            2, min(len(part) - 2, round(len(part) * calibration_fraction))
        )
        ordered = sorted(
            (int(index) for index in part.index),
            key=lambda index: stable_key(str(target_pool.at[index, "reaction_id"])),
        )
        calibration_indices.extend(ordered[:n_selected])
    calibration = target_pool.loc[sorted(calibration_indices)].copy()
    target_test = target_pool.drop(index=sorted(calibration_indices)).copy()
    return calibration, target_test


def freeze_split(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    columns = pd.read_csv(args.source, nrows=0).columns.tolist()
    frame = pd.read_csv(
        args.source, usecols=[column for column in columns if column != "Yield"]
    )
    target_papers, source, target_pool = choose_target_papers(
        frame,
        candidate_limit=args.candidate_limit,
        min_paper_raw_rows=args.min_paper_raw_rows,
        max_paper_raw_rows=args.max_paper_raw_rows,
        min_identities=args.min_identities,
        min_scaffolds=args.min_scaffolds,
        min_source_catalyst_rows=args.min_source_catalyst_rows,
        min_stratum_rows=args.min_stratum_rows,
        min_supported_paper_rows=args.min_supported_paper_rows,
    )
    calibration, target_test = split_target(
        target_pool, args.calibration_fraction
    )
    if not target_papers:
        raise RuntimeError("No target paper survives the feature-only contract")
    if set(source["paper_group"]) & set(target_pool["paper_group"]):
        raise RuntimeError("Target paper leakage into source")
    calibration_strata = set(
        zip(calibration["paper_group"], calibration["catalyst_group"])
    )
    test_strata = set(zip(target_test["paper_group"], target_test["catalyst_group"]))
    if calibration_strata != test_strata:
        raise RuntimeError("Calibration and test paper-catalyst support differs")
    if not set(target_pool["catalyst_group"]).issubset(set(source["catalyst_group"])):
        raise RuntimeError("A target catalyst is absent from source")

    assignments = frame[
        ["reaction_id", "substrate_smiles", "substrate_scaffold",
         "catalyst_group", "paper_group", "duplicate_group"]
    ].copy()
    assignments["role"] = "excluded_unsupported_target_context"
    assignments.loc[
        assignments["reaction_id"].isin(set(source["reaction_id"])), "role"
    ] = "source_train"
    assignments.loc[
        assignments["reaction_id"].isin(set(calibration["reaction_id"])), "role"
    ] = "calibration"
    assignments.loc[
        assignments["reaction_id"].isin(set(target_test["reaction_id"])), "role"
    ] = "target_test"
    assignments = assignments.sort_values("reaction_id").reset_index(drop=True)
    assignments_path = args.output / "split_assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    identity_seen = target_test["substrate_smiles"].isin(
        set(calibration["substrate_smiles"])
    )
    counts = assignments["role"].value_counts().to_dict()
    manifest = {
        "protocol": "photoredox_paper_coldstart_confirmatory_v1",
        "phase": "split_frozen_labels_unread",
        "yield_loaded": False,
        "source_universe": str(args.source.resolve()),
        "source_universe_feature_sha256": feature_hash(frame),
        "split_assignments_sha256": sha256_file(assignments_path),
        "counts": {key: int(value) for key, value in counts.items()},
        "target": {
            "papers": int(target_pool["paper_group"].nunique()),
            "paper_catalyst_strata": int(
                target_pool.groupby(["paper_group", "catalyst_group"]).ngroups
            ),
            "scaffolds": int(target_pool["substrate_scaffold"].nunique()),
            "identities": int(target_pool["substrate_smiles"].nunique()),
            "test_paper_seen_rows": int(len(target_test)),
            "test_paper_unseen_rows": 0,
            "test_identity_seen_rows": int(identity_seen.sum()),
            "test_identity_unseen_rows": int((~identity_seen).sum()),
        },
        "parameters": {
            "candidate_limit": args.candidate_limit,
            "calibration_fraction": args.calibration_fraction,
            "min_paper_raw_rows": args.min_paper_raw_rows,
            "max_paper_raw_rows": args.max_paper_raw_rows,
            "min_identities": args.min_identities,
            "min_scaffolds": args.min_scaffolds,
            "min_source_catalyst_rows": args.min_source_catalyst_rows,
            "min_stratum_rows": args.min_stratum_rows,
            "min_supported_paper_rows": args.min_supported_paper_rows,
        },
    }
    write_json(args.output / "split_freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze-split", "materialize"), required=True)
    parser.add_argument(
        "--source", type=Path,
        default=Path(
            "data/photoredox_ood_full/02_domain_gated_confirmatory_8k/source_train.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "data/photoredox_ood_full/03_paper_coldstart_confirmatory_7k"
        ),
    )
    parser.add_argument("--candidate-limit", type=int, default=40)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--min-paper-raw-rows", type=int, default=15)
    parser.add_argument("--max-paper-raw-rows", type=int, default=150)
    parser.add_argument("--min-identities", type=int, default=8)
    parser.add_argument("--min-scaffolds", type=int, default=3)
    parser.add_argument("--min-source-catalyst-rows", type=int, default=20)
    parser.add_argument("--min-stratum-rows", type=int, default=5)
    parser.add_argument("--min-supported-paper-rows", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "freeze-split":
        freeze_split(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()

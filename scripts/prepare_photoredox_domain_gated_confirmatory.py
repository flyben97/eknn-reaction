#!/usr/bin/env python3
"""Freeze and materialize a label-blind confirmatory Photoredox split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


FORCED_SOURCE_SCAFFOLDS = {"<ACYCLIC>", "c1ccccc1"}


def stable_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def choose_target(
    frame: pd.DataFrame,
    *,
    min_scaffold_raw_rows: int,
    max_scaffold_raw_rows: int,
    min_papers: int,
    min_identities: int,
    min_retained_fraction: float,
    min_source_catalyst_rows: int,
    min_stratum_rows: int,
    min_target_rows_per_scaffold: int,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame]:
    stats = frame.groupby("substrate_scaffold").agg(
        n_reactions=("reaction_id", "size"),
        n_papers=("paper_group", "nunique"),
        n_identities=("substrate_smiles", "nunique"),
        retained_fraction=("substrate_scaffold_retained_in_product", "mean"),
    )
    selected = set(
        stats.index[
            stats["n_reactions"].between(
                min_scaffold_raw_rows, max_scaffold_raw_rows
            )
            & (stats["n_papers"] >= min_papers)
            & (stats["n_identities"] >= min_identities)
            & (stats["retained_fraction"] >= min_retained_fraction)
            & (~stats.index.isin(FORCED_SOURCE_SCAFFOLDS))
        ]
    )
    previous: set[str] | None = None
    eligible = frame.iloc[0:0].copy()
    while selected != previous:
        previous = set(selected)
        source = frame.loc[~frame["substrate_scaffold"].isin(selected)]
        supported_catalysts = set(
            source["catalyst_group"]
            .value_counts()
            .loc[lambda values: values >= min_source_catalyst_rows]
            .index
        )
        eligible = frame.loc[
            frame["substrate_scaffold"].isin(selected)
            & frame["catalyst_group"].isin(supported_catalysts)
            & frame["paper_group"].isin(set(source["paper_group"]))
        ].copy()
        stratum_size = eligible.groupby(
            ["substrate_scaffold", "catalyst_group"]
        )["reaction_id"].transform("size")
        eligible = eligible.loc[stratum_size >= min_stratum_rows].copy()
        supported_scaffolds = eligible["substrate_scaffold"].value_counts()
        selected = set(
            supported_scaffolds.index[
                supported_scaffolds >= min_target_rows_per_scaffold
            ]
        )
    source = frame.loc[~frame["substrate_scaffold"].isin(selected)].copy()
    eligible = eligible.loc[eligible["substrate_scaffold"].isin(selected)].copy()
    return selected, source, eligible


def split_target(
    target_pool: pd.DataFrame, calibration_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_indices: list[int] = []
    for _, part in target_pool.groupby(
        ["substrate_scaffold", "catalyst_group"], sort=True
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
    if "Yield" not in columns:
        raise RuntimeError("Expected a Yield column in the source universe")
    frame = pd.read_csv(
        args.source, usecols=[column for column in columns if column != "Yield"]
    )
    required = {
        "reaction_id", "substrate_smiles", "substrate_scaffold",
        "substrate_scaffold_retained_in_product", "paper_group",
        "catalyst_group", "duplicate_group",
    }
    if required - set(frame.columns):
        raise RuntimeError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    if not frame["reaction_id"].is_unique:
        raise RuntimeError("The source universe has duplicate reaction IDs")

    target_scaffolds, source, target_pool = choose_target(
        frame,
        min_scaffold_raw_rows=args.min_scaffold_raw_rows,
        max_scaffold_raw_rows=args.max_scaffold_raw_rows,
        min_papers=args.min_papers,
        min_identities=args.min_identities,
        min_retained_fraction=args.min_retained_fraction,
        min_source_catalyst_rows=args.min_source_catalyst_rows,
        min_stratum_rows=args.min_stratum_rows,
        min_target_rows_per_scaffold=args.min_target_rows_per_scaffold,
    )
    calibration, target_test = split_target(
        target_pool, args.calibration_fraction
    )
    if not target_scaffolds:
        raise RuntimeError("No target scaffold satisfies the structural contract")

    source_scaffolds = set(source["substrate_scaffold"].astype(str))
    target_scaffold_set = set(target_pool["substrate_scaffold"].astype(str))
    if source_scaffolds & target_scaffold_set:
        raise RuntimeError("Target scaffold leakage into source")
    calibration_strata = set(
        zip(calibration["substrate_scaffold"], calibration["catalyst_group"])
    )
    test_strata = set(
        zip(target_test["substrate_scaffold"], target_test["catalyst_group"])
    )
    if calibration_strata != test_strata:
        raise RuntimeError("Calibration and test stratum supports differ")
    if not set(target_pool["catalyst_group"]).issubset(set(source["catalyst_group"])):
        raise RuntimeError("A target catalyst is absent from source")
    if not set(target_pool["paper_group"]).issubset(set(source["paper_group"])):
        raise RuntimeError("A target paper is absent from source")

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

    paper_seen = target_test["paper_group"].isin(set(calibration["paper_group"]))
    identity_seen = target_test["substrate_smiles"].isin(
        set(calibration["substrate_smiles"])
    )
    counts = assignments["role"].value_counts().to_dict()
    manifest = {
        "protocol": "photoredox_domain_gated_confirmatory_v1",
        "phase": "split_frozen_labels_unread",
        "yield_loaded": False,
        "source_universe": str(args.source.resolve()),
        "source_universe_feature_sha256": feature_hash(frame),
        "split_assignments_sha256": sha256_file(assignments_path),
        "counts": {key: int(value) for key, value in counts.items()},
        "target": {
            "scaffolds": int(target_pool["substrate_scaffold"].nunique()),
            "strata": int(
                target_pool.groupby(
                    ["substrate_scaffold", "catalyst_group"]
                ).ngroups
            ),
            "papers": int(target_pool["paper_group"].nunique()),
            "identities": int(target_pool["substrate_smiles"].nunique()),
            "calibration_papers": int(calibration["paper_group"].nunique()),
            "test_paper_seen_rows": int(paper_seen.sum()),
            "test_paper_unseen_rows": int((~paper_seen).sum()),
            "test_identity_seen_rows": int(identity_seen.sum()),
            "test_identity_unseen_rows": int((~identity_seen).sum()),
        },
        "parameters": {
            "calibration_fraction": args.calibration_fraction,
            "min_scaffold_raw_rows": args.min_scaffold_raw_rows,
            "max_scaffold_raw_rows": args.max_scaffold_raw_rows,
            "min_papers": args.min_papers,
            "min_identities": args.min_identities,
            "min_retained_fraction": args.min_retained_fraction,
            "min_source_catalyst_rows": args.min_source_catalyst_rows,
            "min_stratum_rows": args.min_stratum_rows,
            "min_target_rows_per_scaffold": args.min_target_rows_per_scaffold,
        },
    }
    write_json(args.output / "split_freeze_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def materialize(args: argparse.Namespace) -> None:
    manifest_path = args.output / "split_freeze_manifest.json"
    assignments_path = args.output / "split_assignments.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(assignments_path)
    if actual_hash != manifest["split_assignments_sha256"]:
        raise RuntimeError("Split assignment hash mismatch before materialization")
    assignments = pd.read_csv(assignments_path, usecols=["reaction_id", "role"])
    frame = pd.read_csv(args.source)
    if "Yield" not in frame.columns:
        raise RuntimeError("Cannot materialize labels: Yield is absent")
    merged = frame.merge(assignments, on="reaction_id", how="inner", validate="one_to_one")
    if len(merged) != len(assignments):
        raise RuntimeError("Source universe and frozen assignments do not align")
    split_names = {
        "source_train": "source_train.csv",
        "calibration": "calibration.csv",
        "target_test": "target_test.csv",
    }
    for role, filename in split_names.items():
        part = merged.loc[merged["role"] == role].drop(columns="role")
        part = part.sort_values("reaction_id")
        expected = int(manifest["counts"][role])
        if len(part) != expected:
            raise RuntimeError(f"Materialized {role} count mismatch")
        part.to_csv(args.output / filename, index=False)
    shutil.copyfile(args.output / "source_train.csv", args.output / "train.csv")
    shutil.copyfile(args.output / "calibration.csv", args.output / "valid.csv")
    shutil.copyfile(args.output / "target_test.csv", args.output / "test.csv")
    materialized = {
        **manifest,
        "phase": "labels_materialized_test_unscored",
        "yield_loaded_for_materialization_only": True,
        "files": {
            name: {"path": str((args.output / filename).resolve()),
                   "sha256": sha256_file(args.output / filename)}
            for name, filename in split_names.items()
        },
    }
    write_json(args.output / "metadata.json", materialized)
    print(
        json.dumps(
            {"phase": materialized["phase"], "counts": manifest["counts"],
             "split_assignments_sha256_verified": True,
             "target_yield_summarized_or_scored": False},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze-split", "materialize"), required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "data/photoredox_ood_full/01_scaffold_fewshot_full_10k/source_train.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/photoredox_ood_full/02_domain_gated_confirmatory_8k"
        ),
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--min-scaffold-raw-rows", type=int, default=15)
    parser.add_argument("--max-scaffold-raw-rows", type=int, default=300)
    parser.add_argument("--min-papers", type=int, default=2)
    parser.add_argument("--min-identities", type=int, default=4)
    parser.add_argument("--min-retained-fraction", type=float, default=0.60)
    parser.add_argument("--min-source-catalyst-rows", type=int, default=20)
    parser.add_argument("--min-stratum-rows", type=int, default=6)
    parser.add_argument("--min-target-rows-per-scaffold", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "freeze-split":
        freeze_split(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()

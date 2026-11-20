#!/usr/bin/env python3
"""Split a strict scaffold-OOD target pool into calibration and target test.

The source GNN training set remains unchanged. Target-domain reaction rows are
sampled within every held-out scaffold, matching a deployment setting where a
small random batch of target-domain experiments is labelled before predicting
the remaining experiments. Exact reaction contexts remain disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def stable_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def select_rows(frame: pd.DataFrame, fraction: float) -> set[int]:
    n_selected = max(1, min(len(frame) - 1, round(len(frame) * fraction)))
    ordered = sorted(
        (int(index) for index in frame.index),
        key=lambda index: stable_key(str(frame.at[index, "reaction_id"])),
    )
    return set(ordered[:n_selected])


def select_scaffold_paper_coverage_rows(
    frame: pd.DataFrame, fraction: float
) -> set[int]:
    target = max(1, round(len(frame) * fraction))
    selected: set[int] = set()

    # Cover every scaffold first, starting with the rarest. Prefer a row whose
    # paper has not yet been covered so one experiment can satisfy both axes.
    scaffold_sizes = frame["heteroarene_scaffold"].value_counts()
    for scaffold in sorted(scaffold_sizes.index, key=lambda s: (scaffold_sizes[s], s)):
        part = frame[frame["heteroarene_scaffold"].eq(scaffold)]
        covered_papers = set(frame.loc[list(selected), "paper_group"])
        candidates = part[~part["paper_group"].isin(covered_papers)]
        if candidates.empty:
            candidates = part
        index = min(
            (int(value) for value in candidates.index),
            key=lambda value: stable_key(str(frame.at[value, "reaction_id"])),
        )
        selected.add(index)

    # Then cover every target paper that can remain represented in target_test.
    paper_sizes = frame["paper_group"].value_counts()
    for paper in sorted(paper_sizes.index, key=lambda p: (paper_sizes[p], p)):
        if paper in set(frame.loc[list(selected), "paper_group"]):
            continue
        part = frame[frame["paper_group"].eq(paper)]
        index = min(
            (int(value) for value in part.index),
            key=lambda value: stable_key(str(frame.at[value, "reaction_id"])),
        )
        selected.add(index)

    target = max(target, len(selected))
    # Fill remaining slots toward the requested marginal scaffold/paper
    # proportions. This uses structural/domain labels only, never Yield.
    while len(selected) < target:
        scaffold_selected = frame.loc[list(selected), "heteroarene_scaffold"].value_counts()
        paper_selected = frame.loc[list(selected), "paper_group"].value_counts()
        candidates = [int(value) for value in frame.index if int(value) not in selected]
        def priority(index: int) -> tuple[float, str]:
            scaffold = frame.at[index, "heteroarene_scaffold"]
            paper = frame.at[index, "paper_group"]
            scaffold_deficit = fraction * scaffold_sizes[scaffold] - scaffold_selected.get(scaffold, 0)
            paper_deficit = fraction * paper_sizes[paper] - paper_selected.get(paper, 0)
            return (-(scaffold_deficit + paper_deficit), stable_key(str(frame.at[index, "reaction_id"])))
        selected.add(min(candidates, key=priority))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-task-dir",
        type=Path,
        default=Path(
            "data/photoredox_ood_small/04_heteroarene_scaffold_ood_654"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/photoredox_ood_small/05_heteroarene_scaffold_fewshot_654"
        ),
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument(
        "--sampling-mode",
        choices=("scaffold", "scaffold_paper_coverage"),
        default="scaffold",
    )
    args = parser.parse_args()

    if not 0.05 <= args.calibration_fraction <= 0.5:
        raise ValueError("calibration-fraction must be between 0.05 and 0.5")
    source_train = pd.read_csv(args.strict_task_dir / "train.csv")
    target_pool = pd.concat(
        [
            pd.read_csv(args.strict_task_dir / "valid.csv"),
            pd.read_csv(args.strict_task_dir / "test.csv"),
        ],
        ignore_index=True,
    )

    if args.sampling_mode == "scaffold":
        calibration_indices: set[int] = set()
        for scaffold, part in target_pool.groupby("heteroarene_scaffold", sort=True):
            del scaffold
            calibration_indices.update(select_rows(part, args.calibration_fraction))
    else:
        calibration_indices = select_scaffold_paper_coverage_rows(
            target_pool, args.calibration_fraction
        )
    calibration = target_pool.loc[sorted(calibration_indices)].copy()
    target_test = target_pool.drop(index=sorted(calibration_indices)).copy()

    source_scaffolds = set(source_train["heteroarene_scaffold"])
    calibration_scaffolds = set(calibration["heteroarene_scaffold"])
    test_scaffolds = set(target_test["heteroarene_scaffold"])
    if source_scaffolds & (calibration_scaffolds | test_scaffolds):
        raise AssertionError("Target scaffolds leak into source GNN training")
    if calibration_scaffolds != test_scaffolds:
        raise AssertionError("Calibration and target test scaffold supports differ")
    if (
        args.sampling_mode == "scaffold_paper_coverage"
        and not set(target_test["paper_group"]).issubset(
            set(calibration["paper_group"])
        )
    ):
        raise AssertionError("A target-test paper is absent from calibration")
    for left, right in (
        (source_train, calibration),
        (source_train, target_test),
        (calibration, target_test),
    ):
        if set(left["duplicate_group"]) & set(right["duplicate_group"]):
            raise AssertionError("Exact context leakage across partitions")

    args.output.mkdir(parents=True, exist_ok=True)
    source_train.to_csv(args.output / "source_train.csv", index=False)
    calibration.to_csv(args.output / "calibration.csv", index=False)
    target_test.to_csv(args.output / "target_test.csv", index=False)
    # Conventional aliases simplify reuse by generic dataset loaders.
    shutil.copyfile(args.output / "source_train.csv", args.output / "train.csv")
    shutil.copyfile(args.output / "calibration.csv", args.output / "valid.csv")
    shutil.copyfile(args.output / "target_test.csv", args.output / "test.csv")

    scaffold_distribution = []
    for scaffold in sorted(calibration_scaffolds):
        scaffold_distribution.append(
            {
                "scaffold": scaffold,
                "calibration_rows": int(
                    calibration["heteroarene_scaffold"].eq(scaffold).sum()
                ),
                "target_test_rows": int(
                    target_test["heteroarene_scaffold"].eq(scaffold).sum()
                ),
                "calibration_identities": int(
                    calibration.loc[
                        calibration["heteroarene_scaffold"].eq(scaffold),
                        "heteroarene_smiles",
                    ].nunique()
                ),
                "target_test_identities": int(
                    target_test.loc[
                        target_test["heteroarene_scaffold"].eq(scaffold),
                        "heteroarene_smiles",
                    ].nunique()
                ),
            }
        )
    pd.DataFrame(scaffold_distribution).to_csv(
        args.output / "target_scaffold_distribution.csv", index=False
    )

    metadata = {
        "task": "few_shot_unseen_scaffold_target_adaptation",
        "counts": {
            "source_train": int(len(source_train)),
            "calibration": int(len(calibration)),
            "target_test": int(len(target_test)),
            "total": int(len(source_train) + len(target_pool)),
        },
        "calibration_fraction_requested": args.calibration_fraction,
        "calibration_fraction_actual": float(len(calibration) / len(target_pool)),
        "sampling_mode": args.sampling_mode,
        "protocol": (
            "Train/freeze the base GNN on source_train. Use calibration labels "
            "for target-domain eKNN parameter selection and residual indexing. "
            "Evaluate target_test once."
        ),
        "interpretation": (
            "Few-shot scaffold adaptation, not zero-shot scaffold OOD: target "
            "scaffold identities are represented in calibration but not in "
            "source GNN training."
        ),
        "split_rule": (
            "Within each target scaffold, reaction rows were deterministically "
            "hash-sampled to approximate the requested calibration fraction. "
            "Yield was not used."
            if args.sampling_mode == "scaffold"
            else
            "Calibration rows were deterministically selected to cover every "
            "target scaffold and target-test paper, then filled toward the "
            "requested scaffold/paper marginal proportions. Yield was not used."
        ),
        "target_scaffolds": sorted(calibration_scaffolds),
        "scaffold_distribution": scaffold_distribution,
        "leakage_audit": {
            "target_scaffold_in_source_overlap": 0,
            "calibration_test_scaffold_support_equal": True,
            "target_test_papers_covered_by_calibration": bool(
                set(target_test["paper_group"]).issubset(
                    set(calibration["paper_group"])
                )
            ),
            "calibration_test_heteroarene_identity_overlap": int(
                len(
                    set(calibration["heteroarene_smiles"])
                    & set(target_test["heteroarene_smiles"])
                )
            ),
            "exact_duplicate_group_overlap": 0,
            "yield_used_for_split": False,
        },
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

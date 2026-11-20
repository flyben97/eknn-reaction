#!/usr/bin/env python3
"""Build a large few-shot unseen-substrate-scaffold photoredox task.

The source split labels are ignored.  Exact reaction contexts are deduplicated,
the main substrate is chosen by product retention/similarity, and target rows
are restricted to scaffold-by-catalyst strata with enough observations to put
both calibration and test examples in every stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from prepare_photoredox_ood import (
    derive_base_table,
    largest_organic_fragment,
    parse_smiles_list,
    scaffold_smiles,
)


RDLogger.DisableLog("rdApp.*")
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
FORCED_SOURCE_SCAFFOLDS = {"<ACYCLIC>", "c1ccccc1"}


def stable_key(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def product_aligned_substrate(
    reactants_value: object, products_value: object
) -> tuple[str, str, float, bool]:
    reactants = [
        largest_organic_fragment(smiles)
        for smiles in parse_smiles_list(reactants_value)
    ]
    products = [
        largest_organic_fragment(smiles)
        for smiles in parse_smiles_list(products_value)
    ]
    product_fingerprints = [FP_GENERATOR.GetFingerprint(mol) for mol in products]

    candidates = []
    for molecule in reactants:
        scaffold = scaffold_smiles(molecule)
        scaffold_molecule = (
            Chem.MolFromSmiles(scaffold) if scaffold != "<ACYCLIC>" else None
        )
        retained = bool(
            scaffold_molecule is not None
            and scaffold_molecule.GetNumAtoms() > 0
            and any(product.HasSubstructMatch(scaffold_molecule) for product in products)
        )
        fingerprint = FP_GENERATOR.GetFingerprint(molecule)
        similarity = max(
            DataStructs.BulkTanimotoSimilarity(fingerprint, product_fingerprints)
        )
        scaffold_atoms = (
            scaffold_molecule.GetNumHeavyAtoms()
            if scaffold_molecule is not None
            else 0
        )
        score = (
            int(retained),
            float(similarity),
            int(scaffold_atoms),
            int(molecule.GetNumHeavyAtoms()),
        )
        candidates.append((score, molecule, scaffold, retained, similarity))
    _, molecule, scaffold, retained, similarity = max(
        candidates, key=lambda item: item[0]
    )
    return (
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True),
        scaffold,
        float(similarity),
        bool(retained),
    )


def choose_target_scaffolds(
    frame: pd.DataFrame,
    *,
    min_scaffold_rows: int,
    max_scaffold_rows: int,
    min_stratum_rows: int,
    min_target_rows_per_scaffold: int,
    min_source_catalyst_rows: int,
) -> tuple[set[str], pd.DataFrame, pd.DataFrame, dict]:
    scaffold_stats = frame.groupby("substrate_scaffold").agg(
        n_reactions=("reaction_id", "size"),
        n_papers=("paper_group", "nunique"),
        n_catalysts=("catalyst_group", "nunique"),
        n_identities=("substrate_smiles", "nunique"),
        retained_fraction=("substrate_scaffold_retained_in_product", "mean"),
        median_product_similarity=("substrate_to_product_similarity", "median"),
    )
    selected = set(
        scaffold_stats.index[
            scaffold_stats["n_reactions"].between(
                min_scaffold_rows, max_scaffold_rows
            )
            & (scaffold_stats["n_papers"] >= 2)
            & (scaffold_stats["n_identities"] >= 5)
            & (scaffold_stats["retained_fraction"] >= 0.70)
            & (~scaffold_stats.index.isin(FORCED_SOURCE_SCAFFOLDS))
        ]
    )
    n_initial_candidates = len(selected)

    previous: set[str] | None = None
    eligible = frame.iloc[0:0].copy()
    source = frame.copy()
    while selected != previous:
        previous = set(selected)
        source = frame.loc[~frame["substrate_scaffold"].isin(selected)].copy()
        source_catalyst_counts = source["catalyst_group"].value_counts()
        source_papers = set(source["paper_group"])
        supported_catalysts = set(
            source_catalyst_counts.index[
                source_catalyst_counts >= min_source_catalyst_rows
            ]
        )
        eligible = frame.loc[
            frame["substrate_scaffold"].isin(selected)
            & frame["catalyst_group"].isin(supported_catalysts)
            & frame["paper_group"].isin(source_papers)
        ].copy()
        stratum_sizes = eligible.groupby(
            ["substrate_scaffold", "catalyst_group"]
        )["reaction_id"].transform("size")
        eligible = eligible.loc[stratum_sizes >= min_stratum_rows].copy()
        retained_scaffold_counts = eligible["substrate_scaffold"].value_counts()
        selected = set(
            retained_scaffold_counts.index[
                retained_scaffold_counts >= min_target_rows_per_scaffold
            ]
        )

    source = frame.loc[~frame["substrate_scaffold"].isin(selected)].copy()
    eligible = eligible.loc[eligible["substrate_scaffold"].isin(selected)].copy()
    diagnostics = {
        "n_initial_structural_candidate_scaffolds": int(
            n_initial_candidates
        ),
        "n_final_target_scaffolds": int(len(selected)),
        "n_rows_excluded_with_target_scaffold_but_unsupported_context": int(
            frame["substrate_scaffold"].isin(selected).sum() - len(eligible)
        ),
    }
    return selected, source, eligible, diagnostics


def split_target_pool(
    target_pool: pd.DataFrame, calibration_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration_indices: set[int] = set()
    columns = ["substrate_scaffold", "catalyst_group"]
    for _, part in target_pool.groupby(columns, sort=True):
        n_selected = max(
            1, min(len(part) - 1, round(len(part) * calibration_fraction))
        )
        ordered = sorted(
            (int(index) for index in part.index),
            key=lambda index: stable_key(str(target_pool.at[index, "reaction_id"])),
        )
        calibration_indices.update(ordered[:n_selected])
    calibration = target_pool.loc[sorted(calibration_indices)].copy()
    target_test = target_pool.drop(index=sorted(calibration_indices)).copy()
    return calibration, target_test


def yield_summary(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "mean": float(frame["Yield"].mean()),
        "std": float(frame["Yield"].std()),
        "median": float(frame["Yield"].median()),
        "min": float(frame["Yield"].min()),
        "max": float(frame["Yield"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/photoredox/photoredox_scifinder.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/photoredox_ood_full/01_scaffold_fewshot_full_10k"
        ),
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--min-scaffold-rows", type=int, default=20)
    parser.add_argument("--max-scaffold-rows", type=int, default=250)
    parser.add_argument("--min-stratum-rows", type=int, default=10)
    parser.add_argument("--min-target-rows-per-scaffold", type=int, default=20)
    parser.add_argument("--min-source-catalyst-rows", type=int, default=20)
    args = parser.parse_args()
    if not 0.05 <= args.calibration_fraction <= 0.30:
        raise ValueError("calibration-fraction must be between 0.05 and 0.30")

    base = derive_base_table(args.source)
    n_source_rows = len(base)
    base = (
        base.sort_values("reaction_id")
        .drop_duplicates("duplicate_group", keep="first")
        .reset_index(drop=True)
    )
    aligned = [
        product_aligned_substrate(reactants, products)
        for reactants, products in zip(
            base["Reactants_Smiles"], base["Products_Smiles"]
        )
    ]
    base["substrate_smiles"] = [value[0] for value in aligned]
    base["substrate_scaffold"] = [value[1] for value in aligned]
    base["substrate_to_product_similarity"] = [value[2] for value in aligned]
    base["substrate_scaffold_retained_in_product"] = [
        value[3] for value in aligned
    ]

    (
        target_scaffolds,
        source_train,
        target_pool,
        selection_diagnostics,
    ) = choose_target_scaffolds(
        base,
        min_scaffold_rows=args.min_scaffold_rows,
        max_scaffold_rows=args.max_scaffold_rows,
        min_stratum_rows=args.min_stratum_rows,
        min_target_rows_per_scaffold=args.min_target_rows_per_scaffold,
        min_source_catalyst_rows=args.min_source_catalyst_rows,
    )
    calibration, target_test = split_target_pool(
        target_pool, args.calibration_fraction
    )

    source_scaffolds = set(source_train["substrate_scaffold"])
    calibration_scaffolds = set(calibration["substrate_scaffold"])
    test_scaffolds = set(target_test["substrate_scaffold"])
    calibration_catalysts = set(calibration["catalyst_group"])
    test_catalysts = set(target_test["catalyst_group"])
    source_catalysts = set(source_train["catalyst_group"])
    calibration_strata = set(
        zip(calibration["substrate_scaffold"], calibration["catalyst_group"])
    )
    test_strata = set(
        zip(target_test["substrate_scaffold"], target_test["catalyst_group"])
    )
    if source_scaffolds & (calibration_scaffolds | test_scaffolds):
        raise AssertionError("Target substrate scaffold leaks into source train")
    if calibration_scaffolds != test_scaffolds:
        raise AssertionError("Calibration/test scaffold supports differ")
    if calibration_catalysts != test_catalysts:
        raise AssertionError("Calibration/test catalyst supports differ")
    if calibration_strata != test_strata:
        raise AssertionError("Calibration/test scaffold-catalyst supports differ")
    if not test_catalysts.issubset(source_catalysts):
        raise AssertionError("A target catalyst is unseen in source train")
    if not set(target_test["paper_group"]).issubset(set(source_train["paper_group"])):
        raise AssertionError("A target-test paper is unseen in source train")
    for left, right in (
        (source_train, calibration),
        (source_train, target_test),
        (calibration, target_test),
    ):
        if set(left["duplicate_group"]) & set(right["duplicate_group"]):
            raise AssertionError("Exact reaction context leakage across partitions")

    output_columns = [
        "reaction_id",
        "Reactants_Smiles",
        "Products_Smiles",
        "Reagents_Smiles",
        "Catalysts_Smiles",
        "reactants_graph_smiles",
        "products_graph_smiles",
        "reagents_graph_smiles",
        "catalysts_graph_smiles",
        "Stages",
        "Yield",
        "Notes",
        "Title",
        "Type",
        "paper_group",
        "catalyst_group",
        "n_catalysts",
        "substrate_smiles",
        "substrate_scaffold",
        "substrate_to_product_similarity",
        "substrate_scaffold_retained_in_product",
        "duplicate_group",
    ]
    splits = {
        "source_train": source_train,
        "calibration": calibration,
        "target_test": target_test,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame[output_columns].sort_values("reaction_id").to_csv(
            args.output / f"{name}.csv", index=False
        )
    source_train[output_columns].sort_values("reaction_id").to_csv(
        args.output / "train.csv", index=False
    )
    calibration[output_columns].sort_values("reaction_id").to_csv(
        args.output / "valid.csv", index=False
    )
    target_test[output_columns].sort_values("reaction_id").to_csv(
        args.output / "test.csv", index=False
    )

    target_strata = (
        target_pool.groupby(
            ["substrate_scaffold", "catalyst_group"], as_index=False
        )
        .agg(
            n_target_pool=("reaction_id", "size"),
            n_papers=("paper_group", "nunique"),
            n_substrates=("substrate_smiles", "nunique"),
            catalyst_smiles=("Catalysts_Smiles", "first"),
            types=("Type", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
        )
    )
    calibration_counts = calibration.groupby(
        ["substrate_scaffold", "catalyst_group"]
    ).size()
    test_counts = target_test.groupby(
        ["substrate_scaffold", "catalyst_group"]
    ).size()
    target_strata["n_calibration"] = [
        int(calibration_counts[(row.substrate_scaffold, row.catalyst_group)])
        for row in target_strata.itertuples(index=False)
    ]
    target_strata["n_target_test"] = [
        int(test_counts[(row.substrate_scaffold, row.catalyst_group)])
        for row in target_strata.itertuples(index=False)
    ]
    target_strata.to_csv(args.output / "target_strata.csv", index=False)

    catalyst_summary = (
        pd.concat(
            [frame.assign(partition=name) for name, frame in splits.items()],
            ignore_index=True,
        )
        .groupby(["partition", "catalyst_group"], as_index=False)
        .agg(
            n_reactions=("reaction_id", "size"),
            n_scaffolds=("substrate_scaffold", "nunique"),
            n_papers=("paper_group", "nunique"),
            catalyst_smiles=("Catalysts_Smiles", "first"),
            types=("Type", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
        )
    )
    catalyst_summary.to_csv(args.output / "catalyst_summary.csv", index=False)

    metadata = {
        "task": "full_photoredox_fewshot_unseen_substrate_scaffold",
        "source_file": str(args.source),
        "source_rows": int(n_source_rows),
        "deduplicated_rows": int(len(base)),
        "exact_duplicate_rows_removed": int(n_source_rows - len(base)),
        "deduplicated_full_data_audit": {
            "papers": int(base["paper_group"].nunique()),
            "catalyst_systems": int(base["catalyst_group"].nunique()),
            "reactant_bundles": int(base["reactants_graph_smiles"].nunique()),
            "product_bundles": int(base["products_graph_smiles"].nunique()),
            "reagent_bundles": int(base["reagents_graph_smiles"].nunique()),
            "substrate_identities": int(base["substrate_smiles"].nunique()),
            "product_aligned_substrate_scaffolds": int(
                base["substrate_scaffold"].nunique()
            ),
        },
        "counts": {
            name: int(len(frame)) for name, frame in splits.items()
        }
        | {"included_total": int(sum(len(frame) for frame in splits.values()))},
        "chemical_scope": {
            "source_papers": int(source_train["paper_group"].nunique()),
            "source_catalyst_systems": int(
                source_train["catalyst_group"].nunique()
            ),
            "target_papers": int(target_pool["paper_group"].nunique()),
            "target_catalyst_systems": int(target_pool["catalyst_group"].nunique()),
            "target_scaffolds": int(len(target_scaffolds)),
            "target_scaffold_catalyst_strata": int(len(target_strata)),
            "target_substrate_identities": int(
                target_pool["substrate_smiles"].nunique()
            ),
        },
        "selection": {
            "main_substrate_rule": (
                "Reactant selected by retained product scaffold, Morgan "
                "similarity to product, scaffold size, then heavy-atom count."
            ),
            "calibration_fraction_requested": args.calibration_fraction,
            "calibration_fraction_actual": float(
                len(calibration) / len(target_pool)
            ),
            "target_scaffold_frequency_range": [
                args.min_scaffold_rows,
                args.max_scaffold_rows,
            ],
            "minimum_scaffold_catalyst_stratum_rows": args.min_stratum_rows,
            "minimum_target_rows_per_scaffold": args.min_target_rows_per_scaffold,
            "minimum_source_rows_per_target_catalyst": args.min_source_catalyst_rows,
            "yield_used_for_selection": False,
            "source_split_column_used": False,
            **selection_diagnostics,
        },
        "yield_summary_for_post_split_audit_only": {
            name: yield_summary(frame) for name, frame in splits.items()
        },
        "leakage_audit": {
            "target_scaffold_in_source_overlap": 0,
            "calibration_test_scaffold_support_equal": True,
            "calibration_test_catalyst_support_equal": True,
            "calibration_test_joint_scaffold_catalyst_support_equal": True,
            "target_catalysts_unseen_in_source": 0,
            "target_test_papers_unseen_in_source": 0,
            "exact_duplicate_group_overlap": 0,
        },
        "interpretation": (
            "Few-shot target adaptation: target substrate scaffolds are absent "
            "from source training, while every target scaffold-catalyst stratum "
            "has labelled calibration and held-out test reactions."
        ),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

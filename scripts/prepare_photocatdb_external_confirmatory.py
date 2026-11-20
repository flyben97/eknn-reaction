#!/usr/bin/env python3
"""Prepare a label-isolated PhotoCatDB paper-few-shot confirmation task.

`freeze` reads no yield column.  It canonicalizes reactions, removes reactions
already present in the local SciFinder corpus, retains one deterministic row
per external reaction signature, and freezes a 60/40 within-paper split.
`materialize` may run only after the assignment and candidate recipe hashes
exist; it creates labelled calibration data and a separate locked label file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "data/external/photocatdb/extracted/PhotoCatDB_cond.csv"
SOURCE_DEFAULT = ROOT / "data/photoredox/photoredox_scifinder.csv"
RECIPE_DEFAULT = (
    ROOT
    / "results/photoredox_affine_eknn_candidate_development/frozen_ensemble_recipe.json"
)
OUTPUT_DEFAULT = ROOT / "data/photoredox_external/photocatdb_r2_confirmatory_v1"
CALIBRATION_FRACTION = 0.60
MIN_PAPER_ROWS = 15
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
KNOWN_NUMERIC_SHIFT_SOURCES = {
    # In this source table, the numeric value in `time(h)` matches the
    # literature yield while `yield/%` is the constant reaction time (10 h).
    # The source is also entirely removed by the corrected source-overlap
    # filter, but the schema rule is retained for independent reuse.
    "Org. Lett., 2018, 20, 190-193.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(value: str, salt: str) -> int:
    return int(hashlib.sha256(f"{salt}|{value}".encode()).hexdigest(), 16)


def canonical_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def canonical_side(side: str) -> str | None:
    """Canonicalize a dot-separated side or a serialized SMILES list.

    The SciFinder CSV stores reaction sides as Python-list strings such as
    ``['CCO', 'O=C=O']``, whereas PhotoCatDB stores ordinary dot-separated
    reaction SMILES.  Both forms must be normalized before source-overlap
    filtering; treating a serialized list as a single SMILES silently makes
    every SciFinder signature invalid.
    """
    value = str(side).strip()
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        raw_components = [
            component
            for item in parsed
            for component in str(item).split(".")
            if component
        ]
    else:
        raw_components = value.split(".")
    components = []
    for smiles in raw_components:
        canonical = canonical_smiles(smiles)
        if canonical is None:
            return None
        components.append(canonical)
    return ".".join(sorted(components))


def canonical_reaction(reaction: str) -> str | None:
    parts = str(reaction).split(">>")
    if len(parts) != 2:
        return None
    reactants = canonical_side(parts[0])
    products = canonical_side(parts[1])
    if reactants is None or products is None:
        return None
    return f"{reactants}>>{products}"


def substrate_fields(reaction: str) -> tuple[str, str, float, int]:
    reactant_side, product_side = reaction.split(">>")
    reactants = [Chem.MolFromSmiles(value) for value in reactant_side.split(".")]
    products = [Chem.MolFromSmiles(value) for value in product_side.split(".")]
    reactants = [value for value in reactants if value is not None]
    products = [value for value in products if value is not None]
    if not reactants:
        return "", "", 0.0, 0
    product_fingerprints = [GENERATOR.GetFingerprint(value) for value in products]
    ranked = []
    for molecule in reactants:
        fingerprint = GENERATOR.GetFingerprint(molecule)
        similarity = max(
            (DataStructs.TanimotoSimilarity(fingerprint, other)
             for other in product_fingerprints),
            default=0.0,
        )
        ranked.append((similarity, molecule.GetNumHeavyAtoms(), molecule))
    similarity, _, substrate = max(ranked, key=lambda item: (item[0], item[1]))
    substrate_smiles = Chem.MolToSmiles(substrate, canonical=True)
    try:
        scaffold_molecule = MurckoScaffold.GetScaffoldForMol(substrate)
        scaffold = Chem.MolToSmiles(scaffold_molecule, canonical=True)
    except Exception:
        scaffold = ""
    retained = int(
        bool(scaffold)
        and any(
            product.HasSubstructMatch(Chem.MolFromSmiles(scaffold))
            for product in products
        )
    )
    return substrate_smiles, scaffold, float(similarity), retained


def source_reaction_signatures(path: Path) -> set[str]:
    frame = pd.read_csv(path, usecols=["Reactants_Smiles", "Products_Smiles"])
    reactions = (
        frame["Reactants_Smiles"].fillna("").astype(str)
        + ">>"
        + frame["Products_Smiles"].fillna("").astype(str)
    )
    return set(reactions.map(canonical_reaction).dropna())


def feature_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = raw.copy()
    frame["external_row"] = np.arange(len(frame), dtype=int)
    frame["reaction_signature"] = frame["reaction"].map(canonical_reaction)
    invalid = int(frame["reaction_signature"].isna().sum())
    frame = frame.loc[frame["reaction_signature"].notna()].copy()

    # Keep a stable representative and remove reaction-level duplicates
    # globally, so a chemical reaction cannot appear in both split roles.
    frame["representative_rank"] = frame.apply(
        lambda row: stable_int(
            f"{row['reaction_signature']}|{row['source']}|{row['external_row']}",
            "photocatdb-representative-v1",
        ),
        axis=1,
    )
    before_unique = len(frame)
    frame = (
        frame.sort_values("representative_rank")
        .drop_duplicates("reaction_signature", keep="first")
        .sort_values("external_row")
        .copy()
    )
    removed_duplicate_rows = before_unique - len(frame)

    source_signatures = source_reaction_signatures(SOURCE_DEFAULT)
    overlap = frame["reaction_signature"].isin(source_signatures)
    removed_source_overlap = int(overlap.sum())
    frame = frame.loc[~overlap].copy()

    counts = frame.groupby("source")["external_row"].transform("size")
    ineligible = counts < MIN_PAPER_ROWS
    removed_small_paper_rows = int(ineligible.sum())
    frame = frame.loc[~ineligible].copy().reset_index(drop=True)

    substrate = frame["reaction_signature"].map(substrate_fields)
    frame["substrate_smiles"] = substrate.map(lambda value: value[0])
    frame["substrate_scaffold"] = substrate.map(lambda value: value[1])
    frame["substrate_to_product_similarity"] = substrate.map(lambda value: value[2])
    frame["substrate_scaffold_retained_in_product"] = substrate.map(
        lambda value: value[3]
    )
    reaction_parts = frame["reaction_signature"].str.split(">>", n=1, expand=True)
    frame["reaction_id"] = frame["external_row"].map(
        lambda value: f"photocatdb_cond_{int(value):06d}"
    )
    frame["Reactants_Smiles"] = reaction_parts[0]
    frame["Products_Smiles"] = reaction_parts[1]
    frame["Reagents_Smiles"] = ""
    frame["Catalysts_Smiles"] = ""
    frame["reactants_graph_smiles"] = reaction_parts[0]
    frame["products_graph_smiles"] = reaction_parts[1]
    frame["reagents_graph_smiles"] = ""
    frame["catalysts_graph_smiles"] = ""
    frame["Stages"] = 1
    frame["Notes"] = frame["wave_length"].fillna("").astype(str)
    frame["Title"] = frame["source"].astype(str)
    frame["Type"] = frame["cat_class"].fillna(-1).astype(str)
    frame["paper_group"] = frame["source"].astype(str)
    frame["catalyst_group"] = frame["photocatalyst"].fillna("missing").astype(str)
    frame["n_catalysts"] = frame["photocatalyst"].notna().astype(int)
    frame["duplicate_group"] = frame["reaction_signature"].map(
        lambda value: hashlib.sha256(value.encode()).hexdigest()[:20]
    )
    audit = {
        "raw_rows": int(len(raw)),
        "invalid_reaction_rows": invalid,
        "reaction_duplicate_rows_removed": int(removed_duplicate_rows),
        "source_exact_overlap_rows_removed": removed_source_overlap,
        "small_paper_rows_removed": removed_small_paper_rows,
        "eligible_rows": int(len(frame)),
        "eligible_papers": int(frame["paper_group"].nunique()),
    }
    return frame, audit


def assign_roles(frame: pd.DataFrame) -> pd.DataFrame:
    assignment = frame[
        [
            "external_row",
            "reaction_id",
            "paper_group",
            "catalyst_group",
            "duplicate_group",
            "reaction_signature",
        ]
    ].copy()
    assignment["role"] = "locked_test"
    for paper, group in assignment.groupby("paper_group", sort=True):
        ordered = sorted(
            group.index,
            key=lambda index: stable_int(
                str(assignment.at[index, "duplicate_group"]),
                f"photocatdb-calibration-v1|{paper}",
            ),
        )
        n_calibration = max(2, int(round(CALIBRATION_FRACTION * len(ordered))))
        n_calibration = min(n_calibration, len(ordered) - 2)
        assignment.loc[ordered[:n_calibration], "role"] = "calibration"
    return assignment


def freeze(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The yield column is deliberately excluded at parser level.
    raw = pd.read_csv(args.raw_csv, usecols=lambda column: column != "yield/%")
    features, audit = feature_frame(raw)
    assignments = assign_roles(features)
    model_columns = [
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
    feature_path = args.output_dir / "all_features.csv"
    assignment_path = args.output_dir / "split_assignments.csv"
    features[["external_row", *model_columns]].to_csv(feature_path, index=False)
    assignments.to_csv(assignment_path, index=False)
    manifest = {
        "status": "split_frozen_without_reading_yield_column",
        "schema_transform_version": "photocatdb-to-eknn-v1",
        "calibration_fraction": CALIBRATION_FRACTION,
        "minimum_paper_rows": MIN_PAPER_ROWS,
        "audit": audit,
        "role_counts": {
            key: int(value)
            for key, value in assignments["role"].value_counts().items()
        },
        "paper_counts": {
            key: int(value)
            for key, value in assignments.groupby("role")["paper_group"].nunique().items()
        },
        "hashes": {
            "raw_csv": sha256(args.raw_csv),
            "candidate_recipe": sha256(args.candidate_recipe),
            "all_features": sha256(feature_path),
            "split_assignments": sha256(assignment_path),
        },
    }
    manifest_path = args.output_dir / "split_freeze_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def materialize(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "split_freeze_manifest.json"
    assignment_path = args.output_dir / "split_assignments.csv"
    feature_path = args.output_dir / "all_features.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["hashes"]
    if sha256(args.raw_csv) != expected["raw_csv"]:
        raise RuntimeError("Raw PhotoCatDB CSV changed after split freeze")
    if sha256(args.candidate_recipe) != expected["candidate_recipe"]:
        raise RuntimeError("Candidate recipe changed after split freeze")
    if sha256(feature_path) != expected["all_features"]:
        raise RuntimeError("Frozen feature file changed")
    if sha256(assignment_path) != expected["split_assignments"]:
        raise RuntimeError("Frozen assignment file changed")

    labels = pd.read_csv(args.raw_csv, usecols=["time(h)", "yield/%", "source"])
    labels["external_row"] = np.arange(len(labels), dtype=int)
    primary_yield = pd.to_numeric(labels["yield/%"], errors="coerce")
    # Five source tables in the public CSV are shifted by one column: their
    # slash-delimited selectivity ratio occupies `yield/%`, while the numeric
    # percentage yield occupies `time(h)`.  Apply this fallback only when the
    # declared yield is non-numeric and the preceding value is a valid percent.
    preceding_numeric = pd.to_numeric(labels["time(h)"], errors="coerce")
    shifted_slash_ratio = (
        primary_yield.isna()
        & labels["yield/%"].astype(str).str.contains("/", regex=False)
        & preceding_numeric.between(0, 100)
    )
    shifted_numeric = (
        labels["source"].isin(KNOWN_NUMERIC_SHIFT_SOURCES)
        & preceding_numeric.between(0, 100)
    )
    shifted = shifted_slash_ratio | shifted_numeric
    labels["Yield"] = primary_yield.where(~shifted, preceding_numeric) / 100.0
    assignments = pd.read_csv(assignment_path)
    features = pd.read_csv(feature_path)
    selected = assignments.merge(labels, on="external_row", validate="one_to_one")
    if selected["Yield"].isna().any():
        raise RuntimeError("Selected external rows contain missing yield labels")
    if not selected["Yield"].between(0, 1).all():
        raise RuntimeError("External yields must be in [0, 1] after scaling")
    materialized = features.merge(
        selected[["external_row", "role", "Yield"]],
        on="external_row",
        validate="one_to_one",
    )
    model_columns = [
        column for column in features.columns if column != "external_row"
    ]
    calibration_path = args.output_dir / "calibration.csv"
    test_feature_path = args.output_dir / "target_test_features.csv"
    test_label_path = args.output_dir / "target_test_labels.locked.csv"
    materialized.loc[materialized["role"] == "calibration", [*model_columns, "Yield"]].to_csv(
        calibration_path, index=False
    )
    materialized.loc[materialized["role"] == "locked_test", model_columns].to_csv(
        test_feature_path, index=False
    )
    materialized.loc[
        materialized["role"] == "locked_test", ["reaction_id", "Yield"]
    ].to_csv(test_label_path, index=False)
    locked = {
        "status": "labels_materialized_after_split_and_candidate_freeze",
        "candidate_recipe_sha256": expected["candidate_recipe"],
        "split_assignments_sha256": expected["split_assignments"],
        "calibration_sha256": sha256(calibration_path),
        "target_test_features_sha256": sha256(test_feature_path),
        "target_test_labels_sha256": sha256(test_label_path),
        "calibration_rows": int((materialized["role"] == "calibration").sum()),
        "locked_test_rows": int((materialized["role"] == "locked_test").sum()),
        "yield_schema_audit": {
            "declared_numeric_rows": int(primary_yield.notna().sum()),
            "shifted_slash_ratio_rows_recovered": int(shifted_slash_ratio.sum()),
            "shifted_numeric_rows_recovered": int(shifted_numeric.sum()),
            "shifted_total_rows_recovered": int(shifted.sum()),
            "unrecoverable_rows": int(labels["Yield"].isna().sum()),
        },
    }
    (args.output_dir / "materialization_manifest.json").write_text(
        json.dumps(locked, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(locked, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "materialize"))
    parser.add_argument("--raw-csv", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--candidate-recipe", type=Path, default=RECIPE_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()

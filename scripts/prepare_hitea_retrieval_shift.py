#!/usr/bin/env python3
"""Prepare leakage-safe HiTEA screen-shift tasks for residual retrieval.

The script consumes the unmodified author-released HiTEA CSV.  Split
membership is determined only from prediction-time inputs and stable hashes;
target values never participate in screen eligibility or row assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger


EXPECTED_SOURCE_SHA256 = (
    "9d3b35b8606538df7fc8dd9104b6dd43188ddc8dc43e3d9d3972e10a621ef889"
)
DEFAULT_KEYWORDS = (
    "HECK",
    "LEWIS ACID",
    "MITSUNOBU",
    "NEGISHI",
    "SNAR",
    "ULLMANN",
)
CALIBRATION_BUDGETS = (0.05, 0.10, 0.20)
MISSING_CATEGORY = "<MISSING>"
MISSING_SMILES = "[He]"
RDLogger.DisableLog("rdApp.*")

REQUIRED_COLUMNS = {
    "Dataset entry number",
    "ReactionClass",
    "SCREEN_ID",
    "NOTEBOOK_ID",
    "REACTION_ID",
    "KeyWord_STD",
    "ReactionGroup",
    "RXN_SMILES",
    "PCAT_CMPD_ID",
    "PRODUCT_STRUCTURE",
    "Product_Yield_PCT_Area_UV",
    "Solvent_1_Name",
    "Reaction_T",
    "Reaction_Time_hrs",
    "Catalyst_1_eq",
    "Catalyst_1_ID[1]",
    "catalyst_1_ID_1_SMILES",
    "Catalyst_1_ID[2]",
    "catalyst_1_ID_2_SMILES",
    "Catalyst_1_Short_Hand",
    "Catalyst_2_eq",
    "Catalyst_2_ID[1]",
    "catalyst_2_ID_1_SMILES",
    "Catalyst_2_ID[2]",
    "catalyst_2_ID_2_SMILES",
    "Catalyst_2_Short_Hand",
    "Reactant_1_eq",
    "Reactant_1_ID",
    "Reactant_1_SMILES",
    "Reactant_2_eq",
    "Reactant_2_ID",
    "reactant_2_SMILES",
    "Reactant_3_eq",
    "Reactant_3_ID",
    "reactant_3_SMILES",
    "Reagent_1_eq",
    "Reagent_1_ID",
    "Reagent_1_Short_Hand",
    "Reagent_2_eq",
    "Reagent_2_ID",
    "Reagent_2_Short_Hand",
}

STRING_MAPPING = {
    "ReactionClass": "reaction_class",
    "SCREEN_ID": "screen_id",
    "NOTEBOOK_ID": "notebook_id",
    "KeyWord_STD": "reaction_keyword",
    "ReactionGroup": "reaction_group",
    "RXN_SMILES": "reaction_smiles",
    "PCAT_CMPD_ID": "product_id",
    "Solvent_1_Name": "solvent_name",
    "Catalyst_1_ID[1]": "catalyst_1_id_1",
    "Catalyst_1_ID[2]": "catalyst_1_id_2",
    "Catalyst_1_Short_Hand": "catalyst_1_short_hand",
    "Catalyst_2_ID[1]": "catalyst_2_id_1",
    "Catalyst_2_ID[2]": "catalyst_2_id_2",
    "Catalyst_2_Short_Hand": "catalyst_2_short_hand",
    "Reactant_1_ID": "reactant_1_id",
    "Reactant_2_ID": "reactant_2_id",
    "Reactant_3_ID": "reactant_3_id",
    "Reagent_1_ID": "reagent_1_id",
    "Reagent_1_Short_Hand": "reagent_1_short_hand",
    "Reagent_2_ID": "reagent_2_id",
    "Reagent_2_Short_Hand": "reagent_2_short_hand",
}

SMILES_MAPPING = {
    "PRODUCT_STRUCTURE": "product_smiles",
    "Reactant_1_SMILES": "reactant_1_smiles",
    "reactant_2_SMILES": "reactant_2_smiles",
    "reactant_3_SMILES": "reactant_3_smiles",
    "catalyst_1_ID_1_SMILES": "catalyst_1_smiles_1",
    "catalyst_1_ID_2_SMILES": "catalyst_1_smiles_2",
    "catalyst_2_ID_1_SMILES": "catalyst_2_smiles_1",
    "catalyst_2_ID_2_SMILES": "catalyst_2_smiles_2",
}

NUMERIC_MAPPING = {
    "Reaction_T": "reaction_temperature_c",
    "Reaction_Time_hrs": "reaction_time_h",
    "Catalyst_1_eq": "catalyst_1_equiv",
    "Catalyst_2_eq": "catalyst_2_equiv",
    "Reactant_1_eq": "reactant_1_equiv",
    "Reactant_2_eq": "reactant_2_equiv",
    "Reactant_3_eq": "reactant_3_equiv",
    "Reagent_1_eq": "reagent_1_equiv",
    "Reagent_2_eq": "reagent_2_equiv",
}

FORBIDDEN_SOURCE_COLUMNS = {
    "Product_Yield_Mass_Ion_Count",
    "Product_Selectivity",
    "Product_UV_Analysis_Wavelength_nm",
    "yie_react",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_values(values: Iterable[str]) -> str:
    return stable_digest("\n".join(sorted(str(value) for value in values)))


def normalise_string(value: object, *, missing: str = MISSING_CATEGORY) -> str:
    if pd.isna(value):
        return missing
    text = " ".join(str(value).strip().split())
    return text if text else missing


def stable_numeric(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return MISSING_CATEGORY
    return np.format_float_positional(float(numeric), trim="-")


def hash_columns(frame: pd.DataFrame, columns: Sequence[str], prefix: str) -> pd.Series:
    def row_hash(row: pd.Series) -> str:
        payload = "\x1f".join(f"{column}={row[column]}" for column in columns)
        return f"{prefix}_{stable_digest(payload)[:20]}"

    return frame.loc[:, columns].apply(row_hash, axis=1)


def build_modeling_table(source: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(source.columns)
    if missing:
        raise ValueError(f"HiTEA source is missing required columns: {sorted(missing)}")

    target = pd.to_numeric(source["Product_Yield_PCT_Area_UV"], errors="coerce")
    if target.isna().any():
        raise ValueError(f"Primary target has {int(target.isna().sum())} missing values")
    if not target.between(0.0, 100.0).all():
        raise ValueError("Primary target must lie in [0, 100]")

    table = pd.DataFrame(index=source.index)
    table["source_row"] = np.arange(len(source), dtype=np.int64)
    table["dataset_entry_number"] = source["Dataset entry number"].map(
        lambda value: normalise_string(value)
    )
    source_reaction_id = source["REACTION_ID"].map(normalise_string)
    if source_reaction_id.eq(MISSING_CATEGORY).any() or source_reaction_id.duplicated().any():
        table["record_id"] = table["dataset_entry_number"].map(
            lambda value: f"hitea_{stable_digest(value)[:20]}"
        )
    else:
        table["record_id"] = source_reaction_id
    if table["record_id"].duplicated().any():
        raise ValueError("Could not derive a unique record_id")

    for original, derived in STRING_MAPPING.items():
        table[derived] = source[original].map(normalise_string)
    for original, derived in SMILES_MAPPING.items():
        table[derived] = source[original].map(
            lambda value: normalise_string(value, missing=MISSING_SMILES)
        )
    for original, derived in NUMERIC_MAPPING.items():
        table[derived] = pd.to_numeric(source[original], errors="coerce")
        table[f"{derived}_missing"] = table[derived].isna().astype(np.int8)
    table["Yield"] = target.astype(float)

    table["reaction_core_group"] = table["reaction_smiles"].map(
        lambda value: f"core_{stable_digest(value)[:20]}"
    )

    string_predictors = list(STRING_MAPPING.values()) + list(SMILES_MAPPING.values())
    # SCREEN_ID and NOTEBOOK_ID define deployment domains, not reaction inputs.
    string_predictors.remove("screen_id")
    string_predictors.remove("notebook_id")
    numeric_predictors = list(NUMERIC_MAPPING.values())
    exact_frame = table[string_predictors].copy()
    for column in numeric_predictors:
        exact_frame[column] = table[column].map(stable_numeric)
    table["exact_input_group"] = hash_columns(
        exact_frame, string_predictors + numeric_predictors, "input"
    )
    return table


def select_group_prefix(
    group_sizes: pd.Series, target_rows: int, *, salt: str
) -> list[str]:
    if target_rows <= 0 or group_sizes.empty:
        return []
    ordered = sorted(
        (str(group) for group in group_sizes.index),
        key=lambda group: stable_digest(f"{salt}|{group}"),
    )
    selected: list[str] = []
    rows = 0
    for group in ordered:
        if len(selected) >= len(ordered) - 1:
            break
        selected.append(group)
        rows += int(group_sizes.loc[group])
        if rows >= target_rows:
            break
    return selected


def make_target_assignment(
    target: pd.DataFrame,
    *,
    keyword: str,
    screen_id: str,
    test_fraction: float,
    calibration_budgets: Sequence[float] = CALIBRATION_BUDGETS,
) -> pd.DataFrame:
    group_sizes = target.groupby("exact_input_group").size()
    test_groups = set(
        select_group_prefix(
            group_sizes,
            max(1, round(len(target) * test_fraction)),
            salt=f"test|{keyword}|{screen_id}",
        )
    )
    if not test_groups or test_groups == set(group_sizes.index):
        raise ValueError(f"Could not create a non-empty grouped test for {keyword}/{screen_id}")

    remaining_sizes = group_sizes.drop(index=list(test_groups))
    calibration_order = sorted(
        (str(group) for group in remaining_sizes.index),
        key=lambda group: stable_digest(f"calibration|{keyword}|{screen_id}|{group}"),
    )
    cumulative: list[tuple[str, int]] = []
    total = 0
    for group in calibration_order:
        total += int(remaining_sizes.loc[group])
        cumulative.append((group, total))

    assignment = target[
        ["record_id", "exact_input_group", "reaction_core_group"]
    ].copy()
    assignment["is_target_test"] = assignment["exact_input_group"].isin(test_groups)
    for budget in sorted(calibration_budgets):
        target_rows = max(1, round(len(target) * float(budget)))
        selected_groups = {
            group for group, cumulative_rows in cumulative if cumulative_rows <= target_rows
        }
        if not selected_groups and cumulative:
            selected_groups = {cumulative[0][0]}
        elif cumulative and cumulative[-1][1] < target_rows:
            selected_groups = {group for group, _ in cumulative}
        else:
            for group, cumulative_rows in cumulative:
                selected_groups.add(group)
                if cumulative_rows >= target_rows:
                    break
        label = f"is_calibration_{round(100 * budget):02d}"
        assignment[label] = assignment["exact_input_group"].isin(selected_groups)

    calibration_columns = [
        f"is_calibration_{round(100 * budget):02d}"
        for budget in sorted(calibration_budgets)
    ]
    if assignment.loc[assignment["is_target_test"], calibration_columns].any().any():
        raise AssertionError("Target test overlaps a calibration budget")
    for small, large in zip(calibration_columns, calibration_columns[1:]):
        if not assignment.loc[assignment[small], large].all():
            raise AssertionError("Calibration budgets are not nested")
    if assignment["exact_input_group"].duplicated().any():
        grouped = assignment.groupby("exact_input_group")
        for column in ["is_target_test", *calibration_columns]:
            if grouped[column].nunique().gt(1).any():
                raise AssertionError(f"Exact input group split within {column}")
    return assignment


def screen_input_summary(table: pd.DataFrame, keywords: Sequence[str]) -> pd.DataFrame:
    subset = table.loc[table["reaction_keyword"].isin(keywords)]
    return (
        subset.groupby(["reaction_keyword", "screen_id"], as_index=False)
        .agg(
            n_rows=("record_id", "size"),
            n_exact_input_groups=("exact_input_group", "nunique"),
            n_reaction_cores=("reaction_core_group", "nunique"),
            n_notebooks=("notebook_id", "nunique"),
        )
        .sort_values(["reaction_keyword", "n_rows", "screen_id"], ascending=[True, False, True])
        .reset_index(drop=True)
    )


def invalid_smiles_audit(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return invalid unique role values and the number of affected rows.

    Invalid optional components are not silently rewritten: the downstream
    graph builder currently skips an unparseable role while retaining the
    other valid roles.  This audit makes that information loss explicit.
    """
    rows: list[dict] = []
    affected = pd.Series(False, index=table.index)
    for role in SMILES_MAPPING.values():
        counts = table[role].value_counts(dropna=False)
        for value, n_rows in counts.items():
            smiles = str(value)
            if Chem.MolFromSmiles(smiles) is not None:
                continue
            mask = table[role].astype(str).eq(smiles)
            affected |= mask
            rows.append(
                {
                    "role": role,
                    "smiles": smiles,
                    "n_rows": int(n_rows),
                    "reaction_keywords": "|".join(
                        sorted(set(table.loc[mask, "reaction_keyword"]))
                    ),
                    "policy": "role_skipped_by_current_graph_builder",
                }
            )
    columns = ["role", "smiles", "n_rows", "reaction_keywords", "policy"]
    return pd.DataFrame(rows, columns=columns), int(affected.sum())


def json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/external/hitea/8_SEPT_APPROVED_full_dataset.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hitea_retrieval_shift/v1"),
    )
    parser.add_argument("--keywords", nargs="+", default=list(DEFAULT_KEYWORDS))
    parser.add_argument("--min-screen-rows", type=int, default=80)
    parser.add_argument("--min-input-groups", type=int, default=10)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--allow-source-hash-mismatch", action="store_true")
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(
            f"Missing official HiTEA source: {args.source}. Download the author-released "
            "data/8_SEPT_APPROVED_full_dataset.csv first."
        )
    if not 0.10 <= args.test_fraction <= 0.40:
        raise ValueError("test-fraction must lie in [0.10, 0.40]")

    source_hash = sha256_file(args.source)
    if source_hash != EXPECTED_SOURCE_SHA256 and not args.allow_source_hash_mismatch:
        raise ValueError(
            f"Source SHA-256 mismatch: expected {EXPECTED_SOURCE_SHA256}, got {source_hash}"
        )

    raw = pd.read_csv(args.source, low_memory=False)
    table = build_modeling_table(raw)
    if FORBIDDEN_SOURCE_COLUMNS & set(table.columns):
        raise AssertionError("A forbidden post-outcome column reached the modeling table")
    invalid_smiles, n_invalid_smiles_rows = invalid_smiles_audit(table)

    keywords = tuple(dict.fromkeys(str(value).upper() for value in args.keywords))
    summary = screen_input_summary(table, keywords)
    summary["eligible"] = (
        summary["n_rows"].ge(args.min_screen_rows)
        & summary["n_exact_input_groups"].ge(args.min_input_groups)
    )

    args.output.mkdir(parents=True, exist_ok=True)
    fold_root = args.output / "folds"
    fold_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "modeling_table.csv", index=False)
    summary.to_csv(args.output / "screen_input_audit.csv", index=False)
    invalid_smiles.to_csv(args.output / "invalid_smiles_components.csv", index=False)

    fold_rows: list[dict] = []
    locked_test_record_ids: set[str] = set()
    for row in summary.loc[summary["eligible"]].itertuples(index=False):
        keyword = str(row.reaction_keyword)
        screen_id = str(row.screen_id)
        target_mask = table["screen_id"].eq(screen_id)
        target = table.loc[target_mask & table["reaction_keyword"].eq(keyword)].copy()
        # A screen ID should belong to one keyword.  Fail rather than silently
        # retain same-screen rows from another target definition.
        same_screen_keywords = set(table.loc[target_mask, "reaction_keyword"])
        if same_screen_keywords != {keyword}:
            raise ValueError(
                f"SCREEN_ID {screen_id} spans keywords {sorted(same_screen_keywords)}"
            )

        assignment = make_target_assignment(
            target,
            keyword=keyword,
            screen_id=screen_id,
            test_fraction=args.test_fraction,
        )
        target_groups = set(target["exact_input_group"])
        source_train = table.loc[
            ~target_mask & ~table["exact_input_group"].isin(target_groups)
        ]
        purged = table.loc[
            ~target_mask & table["exact_input_group"].isin(target_groups)
        ]
        if set(source_train["exact_input_group"]) & target_groups:
            raise AssertionError("Exact input leakage remains in source train")

        fold_id = f"{keyword.lower().replace(' ', '_')}_{screen_id.lower()}"
        fold_dir = fold_root / fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)
        assignment.sort_values("record_id").to_csv(
            fold_dir / "target_assignment.csv", index=False
        )
        locked_test_record_ids.update(
            assignment.loc[assignment["is_target_test"], "record_id"].astype(str)
        )

        target_with_assignment = target.merge(
            assignment,
            on=["record_id", "exact_input_group", "reaction_core_group"],
            validate="one_to_one",
        )
        # Yield summaries are post-freeze diagnostics only.  They are written
        # after assignment and never used to choose folds or records.
        test_frame = target_with_assignment.loc[target_with_assignment["is_target_test"]]
        cal_counts = {
            f"calibration_{round(100 * budget):02d}": int(
                target_with_assignment[f"is_calibration_{round(100 * budget):02d}"].sum()
            )
            for budget in CALIBRATION_BUDGETS
        }
        manifest = {
            "fold_id": fold_id,
            "reaction_keyword": keyword,
            "target_screen_id": screen_id,
            "split_selection_used_target": False,
            "target_rows": int(len(target)),
            "target_exact_input_groups": int(target["exact_input_group"].nunique()),
            "target_test_rows": int(len(test_frame)),
            "target_test_exact_input_groups": int(test_frame["exact_input_group"].nunique()),
            **cal_counts,
            "source_train_rows": int(len(source_train)),
            "source_purged_overlap_rows": int(len(purged)),
            "source_record_ids_sha256": digest_values(source_train["record_id"]),
            "target_assignment_sha256": sha256_file(fold_dir / "target_assignment.csv"),
            "post_split_target_audit_only": {
                "target_yield_mean": float(target["Yield"].mean()),
                "target_yield_std": float(target["Yield"].std()),
                "target_zero_fraction": float(target["Yield"].eq(0.0).mean()),
                "test_yield_mean": float(test_frame["Yield"].mean()),
                "test_yield_std": float(test_frame["Yield"].std()),
            },
        }
        json_dump(fold_dir / "split_manifest.json", manifest)
        fold_rows.append(manifest)

    if not fold_rows:
        raise ValueError("No eligible target screens under the frozen input-only rules")

    fold_index = pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "post_split_target_audit_only"}
            for row in fold_rows
        ]
    ).sort_values("fold_id")
    fold_index.to_csv(args.output / "fold_index.csv", index=False)
    development_table = table.copy()
    locked_mask = development_table["record_id"].astype(str).isin(
        locked_test_record_ids
    )
    development_table.loc[locked_mask, "Yield"] = np.nan
    development_table.to_csv(
        args.output / "development_modeling_table.csv", index=False
    )
    pd.DataFrame(
        {"record_id": sorted(locked_test_record_ids), "role": "locked_target_test"}
    ).to_csv(args.output / "locked_test_registry.csv", index=False)
    manifest = {
        "task": "hitea_grouped_unseen_screen_target_memory",
        "version": "v1",
        "source_file": str(args.source),
        "source_sha256": source_hash,
        "source_rows": int(len(raw)),
        "modeling_rows": int(len(table)),
        "modeling_table_sha256": sha256_file(args.output / "modeling_table.csv"),
        "development_modeling_table_sha256": sha256_file(
            args.output / "development_modeling_table.csv"
        ),
        "locked_test_registry_sha256": sha256_file(
            args.output / "locked_test_registry.csv"
        ),
        "n_unique_locked_test_records": int(len(locked_test_record_ids)),
        "development_label_policy": (
            "Yield is blank for the union of every fold's locked target-test "
            "records; use this table for all method development."
        ),
        "keywords": list(keywords),
        "screen_eligibility": {
            "uses_target": False,
            "minimum_rows": int(args.min_screen_rows),
            "minimum_exact_input_groups": int(args.min_input_groups),
        },
        "target_test_fraction_requested": float(args.test_fraction),
        "calibration_budgets": list(CALIBRATION_BUDGETS),
        "n_eligible_folds": int(len(fold_index)),
        "smiles_parse_audit": {
            "invalid_unique_role_values": int(len(invalid_smiles)),
            "affected_rows": int(n_invalid_smiles_rows),
            "audit_file": "invalid_smiles_components.csv",
            "policy": "retain row; current graph builder skips only the invalid role",
        },
        "forbidden_post_outcome_columns": sorted(FORBIDDEN_SOURCE_COLUMNS),
        "exact_input_group_columns": [
            column
            for column in list(STRING_MAPPING.values()) + list(SMILES_MAPPING.values())
            if column not in {"screen_id", "notebook_id"}
        ]
        + list(NUMERIC_MAPPING.values()),
    }
    json_dump(args.output / "materialization_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

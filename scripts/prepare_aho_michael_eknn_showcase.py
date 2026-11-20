#!/usr/bin/env python3
"""Prepare reproducible Aho and Michael-addition eKNN showcase benchmarks.

The two source tables are archival reaction records rather than predefined
train/test benchmarks.  This script derives five immutable CSV split pairs:

* ``aho_ddg_random``: reaction-identity-grouped random split;
* ``aho_ddg_ligand_ood``: unseen ligand-identity split;
* ``michael_ddg_random``: reaction-identity-grouped random split;
* ``michael_ddg_row_random``: ordinary row-level random split;
* ``michael_ddg_acceptor_ood``: unseen acceptor-identity split.

All benchmarks use the shared ``ddG`` target.  Product SMILES are deliberately
excluded from model inputs, because a reaction-outcome model should not use the
product structure at prediction time.  Missing optional components are encoded
as ``[He]`` so component positions stay stable in the disconnected reaction
graph used by the GNN.

Run from the repository root:

    python scripts/prepare_aho_michael_eknn_showcase.py

Use ``--overwrite`` only when intentionally regenerating the derived files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
MISSING_SMILES = "[He]"
SPLIT_SEED = 42
TEST_FRACTION = 0.30

AHO_INPUT_COLS = [
    "reactant_smiles",
    "ligand_smiles",
    "metal_smiles",
    "solvent_smiles",
    "additive_smiles",
]
MICHAEL_INPUT_COLS = [
    "acceptor_smiles",
    "donor_smiles",
    "catalyst_smiles",
    "ligand_smiles",
    "solvent_smiles",
    "additive_smiles",
]

# These values are recorded before the reaction outcome is known and are
# therefore available at prediction time.  They deliberately exclude product
# descriptors and any target-derived metadata.  Missingness is preserved by
# the model-side condition preprocessor as an explicit indicator feature.
MICHAEL_CONDITION_SOURCE_COLS = [
    ("cond_temperature_c", "temperature (C)"),
    ("cond_step1_time_h", "step 1 reaction time (h)"),
    ("cond_step2_temperature_c", "step 2 reaction temperature (C)"),
    ("cond_step2_time_h", "step 2 reaction time (h)"),
    ("cond_catalyst_equiv", "catalyst(pre) 1 ammount (eq)"),
    ("cond_ligand_equiv", "ligand 1 ammount (eq)"),
    ("cond_acceptor_equiv", "acceptor ammount (eq)"),
    ("cond_donor_equiv", "donor ammount (eq)"),
    ("cond_additive1_equiv", "additive 1 ammount (eq)"),
    ("cond_additive2_equiv", "additive 2 ammount (eq)"),
    ("cond_solvent1_ml", "solvent 1 ammount (mL)"),
    ("cond_solvent2_ml", "solvent 2 ammount (mL)"),
]
MICHAEL_CONDITION_COLS = [name for name, _ in MICHAEL_CONDITION_SOURCE_COLS]
MICHAEL_CATEGORICAL_CONDITION_SOURCE_COLS = [
    ("cond_metal_label", "Metal"),
    ("cond_ligand_type", "ligand_type"),
    ("cond_michael_bonding_type", "Michael Bonding Type"),
    ("cond_acceptor_class", "acceptor_class_code"),
    ("cond_donor_class", "donor_class_code"),
]
MICHAEL_CATEGORICAL_CONDITION_COLS = [
    name for name, _ in MICHAEL_CATEGORICAL_CONDITION_SOURCE_COLS
]


@dataclass(frozen=True)
class Benchmark:
    """One derived benchmark and its split-defining group."""

    name: str
    dataset: str
    frame: pd.DataFrame
    out_dir: Path
    input_cols: list[str]
    condition_cols: list[str]
    categorical_condition_cols: list[str]
    split_kind: str
    split_group_col: str
    split_strategy: str
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing derived CSV files and manifests.",
    )
    return parser.parse_args()


def _is_missing(value: object) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "na", "n/a"}


def clean_smiles(value: object, *, required: bool, context: str) -> str:
    """Return a valid SMILES, preserving a slot for a missing optional input."""

    if _is_missing(value):
        if required:
            raise ValueError(f"Required SMILES is missing: {context}")
        return MISSING_SMILES

    smiles = str(value).strip()
    if Chem.MolFromSmiles(smiles) is not None:
        return smiles
    if required:
        raise ValueError(f"Required SMILES is not RDKit-parseable: {context}: {smiles!r}")
    return MISSING_SMILES


def metal_to_smiles(value: object, *, context: str) -> str:
    """Represent a tabular elemental-metal label as a valid one-atom SMILES."""

    if _is_missing(value):
        return MISSING_SMILES
    metal = str(value).strip()
    smiles = f"[{metal}]"
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError(f"Unrecognised metal label at {context}: {metal!r}")
    return smiles


def _finite_target(source: pd.DataFrame, column: str, label: str) -> pd.Series:
    target = pd.to_numeric(source[column], errors="coerce")
    keep = np.isfinite(target.to_numpy())
    if not keep.all():
        dropped = int((~keep).sum())
        print(f"[{label}] dropping {dropped} rows with a non-finite target")
    return target


def build_aho_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise Aho inputs while avoiding unparseable metal-complex SMILES."""

    target = _finite_target(source, "ddG", "Aho")
    keep = np.isfinite(target.to_numpy())
    source = source.loc[keep].copy()
    target = target.loc[keep]

    frame = pd.DataFrame({"source_row": source.index.astype(int), "ddG": target.astype(float)})
    frame["reactant_smiles"] = [
        clean_smiles(v, required=True, context=f"Aho row {i}, Reactant SMILES")
        for i, v in source["Reactant SMILES"].items()
    ]
    frame["ligand_smiles"] = [
        clean_smiles(v, required=True, context=f"Aho row {i}, Ligand SMILES")
        for i, v in source["Ligand SMILES"].items()
    ]
    frame["metal_smiles"] = [
        metal_to_smiles(v, context=f"Aho row {i}, Metal")
        for i, v in source["Metal"].items()
    ]
    frame["solvent_smiles"] = [
        clean_smiles(v, required=True, context=f"Aho row {i}, Solvent SMILES")
        for i, v in source["Solvent SMILES"].items()
    ]
    frame["additive_smiles"] = [
        clean_smiles(v, required=False, context=f"Aho row {i}, Additive SMILES")
        for i, v in source["Additive SMILES"].items()
    ]
    frame["ligand_group"] = frame["ligand_smiles"]
    frame["reaction_key"] = frame[AHO_INPUT_COLS].agg("\x1f".join, axis=1)
    return frame


def build_michael_frame(source: pd.DataFrame) -> pd.DataFrame:
    """Canonicalise Michael-addition inputs and omit product-only information."""

    target = _finite_target(source, "ddG (kcal/mol)", "Michael addition")
    keep = np.isfinite(target.to_numpy())
    source = source.loc[keep].copy()
    target = target.loc[keep]

    frame = pd.DataFrame({"source_row": source.index.astype(int), "ddG": target.astype(float)})
    mappings = [
        ("acceptor_smiles", "acceptor_smiles", True),
        ("donor_smiles", "donor_smiles", True),
        ("catalyst_smiles", "catalyst_smiles", True),
        ("ligand_smiles", "ligands_smiles", True),
        ("solvent_smiles", "solvents_smiles", False),
        ("additive_smiles", "additives_smiles", False),
    ]
    for output_col, source_col, required in mappings:
        frame[output_col] = [
            clean_smiles(
                value,
                required=required,
                context=f"Michael-addition row {row_index}, {source_col}",
            )
            for row_index, value in source[source_col].items()
        ]
    for output_col, source_col in MICHAEL_CONDITION_SOURCE_COLS:
        # Keep missing values as NaN.  Imputation/scaling are fitted on each
        # training fold only in train_gnn_eknn.py, so no validation/test rows
        # affect the condition representation.
        frame[output_col] = pd.to_numeric(source[source_col], errors="coerce").astype(float)
    for output_col, source_col in MICHAEL_CATEGORICAL_CONDITION_SOURCE_COLS:
        values = source[source_col]
        frame[output_col] = values.where(~values.isna(), "__MISSING__").astype(str)
    frame["acceptor_group"] = frame["acceptor_smiles"]
    frame["reaction_key"] = frame[MICHAEL_INPUT_COLS].agg("\x1f".join, axis=1)
    return frame


def split_frame(frame: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create a fixed group-disjoint 70/30 split and return its audit record."""

    groups = frame[group_col].astype(str)
    if groups.nunique() < 2:
        raise ValueError(f"Need at least two groups for a split, got {groups.nunique()} in {group_col}")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION,
        random_state=SPLIT_SEED,
    )
    train_idx, test_idx = next(splitter.split(frame, groups=groups))
    train = frame.iloc[train_idx].copy().sort_values("source_row").reset_index(drop=True)
    test = frame.iloc[test_idx].copy().sort_values("source_row").reset_index(drop=True)

    train_groups = set(train[group_col].astype(str))
    test_groups = set(test[group_col].astype(str))
    overlap = train_groups & test_groups
    if overlap:
        raise RuntimeError(f"Group leakage detected for {group_col}: {len(overlap)} overlapping groups")

    train_keys = set(train["reaction_key"])
    test_keys = set(test["reaction_key"])
    key_overlap = train_keys & test_keys
    if key_overlap:
        raise RuntimeError(f"Reaction-identity leakage detected: {len(key_overlap)} overlapping keys")

    audit = {
        "requested_test_fraction": TEST_FRACTION,
        "split_seed": SPLIT_SEED,
        "split_group_column": group_col,
        "n_rows_total": int(len(frame)),
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)),
        "actual_test_fraction": float(len(test) / len(frame)),
        "n_groups_total": int(groups.nunique()),
        "n_groups_train": int(len(train_groups)),
        "n_groups_test": int(len(test_groups)),
        "overlapping_split_groups": 0,
        "overlapping_reaction_keys": 0,
        "target_mean_train": float(train["ddG"].mean()),
        "target_mean_test": float(test["ddG"].mean()),
        "target_sd_train": float(train["ddG"].std(ddof=1)),
        "target_sd_test": float(test["ddG"].std(ddof=1)),
    }
    return train, test, audit


def split_row_random_frame(
    frame: pd.DataFrame, group_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create a fixed ordinary row-level 70/30 split and audit overlap.

    This is intentionally an IID-style control: repeated reaction identities
    are allowed to occur on both sides of the outer split. The manifest records
    the resulting identity overlap explicitly, so this split cannot be
    mistaken for a group-disjoint generalisation test.
    """

    if len(frame) < 2:
        raise ValueError("Need at least two rows for a row-level random split")

    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(len(frame))
    n_train = int(len(frame) * (1.0 - TEST_FRACTION))
    if not 0 < n_train < len(frame):
        raise ValueError(
            f"Invalid row-level split: n_rows={len(frame)}, n_train={n_train}"
        )
    train = frame.iloc[order[:n_train]].copy().sort_values("source_row").reset_index(drop=True)
    test = frame.iloc[order[n_train:]].copy().sort_values("source_row").reset_index(drop=True)

    train_groups = set(train[group_col].astype(str))
    test_groups = set(test[group_col].astype(str))
    group_overlap = train_groups & test_groups
    train_keys = set(train["reaction_key"])
    test_keys = set(test["reaction_key"])
    key_overlap = train_keys & test_keys

    return train, test, {
        "requested_test_fraction": TEST_FRACTION,
        "split_seed": SPLIT_SEED,
        "split_group_column": None,
        "split_strategy": "row_random",
        "n_rows_total": int(len(frame)),
        "n_rows_train": int(len(train)),
        "n_rows_test": int(len(test)),
        "actual_test_fraction": float(len(test) / len(frame)),
        "n_groups_total": int(frame[group_col].astype(str).nunique()),
        "n_groups_train": int(len(train_groups)),
        "n_groups_test": int(len(test_groups)),
        "overlapping_split_groups": int(len(group_overlap)),
        "overlapping_reaction_keys": int(len(key_overlap)),
        "target_mean_train": float(train["ddG"].mean()),
        "target_mean_test": float(test["ddG"].mean()),
        "target_sd_train": float(train["ddG"].std(ddof=1)),
        "target_sd_test": float(test["ddG"].std(ddof=1)),
    }


def prepare_benchmark(benchmark: Benchmark, *, overwrite: bool) -> None:
    """Write one train/test pair plus a transparent split manifest."""

    train_path = benchmark.out_dir / f"{benchmark.name}_train.csv"
    test_path = benchmark.out_dir / f"{benchmark.name}_test.csv"
    manifest_path = benchmark.out_dir / f"{benchmark.name}_manifest.json"
    outputs = (train_path, test_path, manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path.relative_to(ROOT)) for path in existing)
        raise FileExistsError(f"Derived files already exist: {names}. Use --overwrite to replace them.")

    if benchmark.split_strategy == "group_disjoint":
        train, test, audit = split_frame(benchmark.frame, benchmark.split_group_col)
    elif benchmark.split_strategy == "row_random":
        train, test, audit = split_row_random_frame(
            benchmark.frame, benchmark.split_group_col
        )
    else:
        raise ValueError(
            f"Unknown split strategy for {benchmark.name!r}: "
            f"{benchmark.split_strategy!r}"
        )
    train = train.copy()
    test = test.copy()
    train["split_group"] = train[benchmark.split_group_col]
    test["split_group"] = test[benchmark.split_group_col]

    output_cols = [
        "source_row", *benchmark.input_cols, *benchmark.condition_cols,
        *benchmark.categorical_condition_cols,
        "ddG", "split_group", "reaction_key",
    ]
    benchmark.out_dir.mkdir(parents=True, exist_ok=True)
    train.loc[:, output_cols].to_csv(train_path, index=False)
    test.loc[:, output_cols].to_csv(test_path, index=False)

    manifest = {
        "benchmark": benchmark.name,
        "dataset": benchmark.dataset,
        "description": benchmark.description,
        "target": {
            "derived_column": "ddG",
            "source_column": "ddG" if benchmark.dataset == "Aho" else "ddG (kcal/mol)",
        },
        "model_input_columns": benchmark.input_cols,
        "condition_feature_columns": benchmark.condition_cols,
        "categorical_condition_feature_columns": benchmark.categorical_condition_cols,
        "condition_feature_source_columns": {
            output: source
            for output, source in MICHAEL_CONDITION_SOURCE_COLS
            if output in benchmark.condition_cols
        },
        "categorical_condition_feature_source_columns": {
            output: source
            for output, source in MICHAEL_CATEGORICAL_CONDITION_SOURCE_COLS
            if output in benchmark.categorical_condition_cols
        },
        "excluded_columns": ["Product SMILES"] if benchmark.dataset == "Aho" else ["products_smiles"],
        "missing_component_policy": {
            "value": MISSING_SMILES,
            "reason": "Retains a stable component slot for an absent optional reagent.",
        },
        "split_kind": benchmark.split_kind,
        "split_strategy": benchmark.split_strategy,
        "audit": audit,
        "files": {
            "train": str(train_path.relative_to(ROOT)),
            "test": str(test_path.relative_to(ROOT)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[{benchmark.name}] train={audit['n_rows_train']} test={audit['n_rows_test']} "
        f"({audit['actual_test_fraction']:.1%} test), split groups="
        f"{audit['n_groups_train']}/{audit['n_groups_test']}, "
        f"overlap={audit['overlapping_reaction_keys']} reaction identities"
    )


def main() -> int:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")

    aho_source = pd.read_csv(ROOT / "data" / "aho" / "aho.csv")
    michael_source = pd.read_csv(
        ROOT / "data" / "Michael addition" / "feng_published_version.csv"
    )
    aho = build_aho_frame(aho_source)
    michael = build_michael_frame(michael_source)

    benchmarks = [
        Benchmark(
            name="aho_ddg_random",
            dataset="Aho",
            frame=aho,
            out_dir=ROOT / "data" / "aho" / "eknn_showcase",
            input_cols=AHO_INPUT_COLS,
            condition_cols=[],
            categorical_condition_cols=[],
            split_kind="reaction-identity-grouped random split",
            split_group_col="reaction_key",
            split_strategy="group_disjoint",
            description="Aho ddG prediction with duplicate reaction identities kept within one split.",
        ),
        Benchmark(
            name="aho_ddg_ligand_ood",
            dataset="Aho",
            frame=aho,
            out_dir=ROOT / "data" / "aho" / "eknn_showcase",
            input_cols=AHO_INPUT_COLS,
            condition_cols=[],
            categorical_condition_cols=[],
            split_kind="unseen-ligand-identity split",
            split_group_col="ligand_group",
            split_strategy="group_disjoint",
            description="Aho ddG prediction with no ligand identity shared between train and test.",
        ),
        Benchmark(
            name="michael_ddg_random",
            dataset="Michael addition",
            frame=michael,
            out_dir=ROOT / "data" / "Michael addition" / "eknn_showcase",
            input_cols=MICHAEL_INPUT_COLS,
            condition_cols=MICHAEL_CONDITION_COLS,
            categorical_condition_cols=MICHAEL_CATEGORICAL_CONDITION_COLS,
            split_kind="reaction-identity-grouped random split",
            split_group_col="reaction_key",
            split_strategy="group_disjoint",
            description="Michael-addition ddG prediction with duplicate reaction identities kept within one split.",
        ),
        Benchmark(
            name="michael_ddg_row_random",
            dataset="Michael addition",
            frame=michael,
            out_dir=ROOT / "data" / "Michael addition" / "eknn_showcase",
            input_cols=MICHAEL_INPUT_COLS,
            condition_cols=MICHAEL_CONDITION_COLS,
            categorical_condition_cols=MICHAEL_CATEGORICAL_CONDITION_COLS,
            split_kind="ordinary row-level random split",
            split_group_col="reaction_key",
            split_strategy="row_random",
            description=(
                "Michael-addition ddG IID-style control with rows randomly split; "
                "duplicate reaction identities may occur in both train and test."
            ),
        ),
        Benchmark(
            name="michael_ddg_acceptor_ood",
            dataset="Michael addition",
            frame=michael,
            out_dir=ROOT / "data" / "Michael addition" / "eknn_showcase",
            input_cols=MICHAEL_INPUT_COLS,
            condition_cols=MICHAEL_CONDITION_COLS,
            categorical_condition_cols=MICHAEL_CATEGORICAL_CONDITION_COLS,
            split_kind="unseen-acceptor-identity split",
            split_group_col="acceptor_group",
            split_strategy="group_disjoint",
            description="Michael-addition ddG prediction with no acceptor identity shared between train and test.",
        ),
    ]
    for benchmark in benchmarks:
        prepare_benchmark(benchmark, overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

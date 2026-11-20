#!/usr/bin/env python3
"""Build reproducible out-of-distribution splits for the photoredox dataset.

The script deliberately ignores the source ``split`` column.  It canonicalizes
the four molecular-list columns, derives chemically meaningful grouping
variables, and writes one train/valid/test triplet for each OOD task.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")

SOURCE_DROP_COLUMNS = {
    "Unnamed: 0",
    "ori_row_number",
    "split",
    "Catalysts_Labels",
    "Steps",  # Constant (1) in the source dataset.
}
SMILES_LIST_COLUMNS = (
    "Reactants_Smiles",
    "Products_Smiles",
    "Reagents_Smiles",
    "Catalysts_Smiles",
)
SPLIT_ORDER = ("train", "valid", "test")
SEED = 42


class DisjointSet:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def parse_smiles_list(value: object) -> list[str]:
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"Expected a non-empty SMILES list, got: {value!r}")
    return [str(item) for item in parsed]


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonicalize_smiles_list(value: object, *, sort_items: bool = False) -> list[str]:
    items = [canonicalize_smiles(s) for s in parse_smiles_list(value)]
    if sort_items:
        items.sort()
    return items


def serialize_smiles_list(items: Sequence[str]) -> str:
    return repr(list(items))


def normalized_title(title: object) -> str:
    text = unicodedata.normalize("NFKC", str(title)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def largest_organic_fragment(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    fragments = list(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True))
    organic = [
        frag
        for frag in fragments
        if any(atom.GetAtomicNum() == 6 for atom in frag.GetAtoms())
    ]
    candidates = organic or fragments
    if not candidates:
        raise ValueError(f"No molecular fragment found in: {smiles}")
    return max(
        candidates,
        key=lambda frag: (frag.GetNumHeavyAtoms(), frag.GetNumAtoms()),
    )


def scaffold_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return "<ACYCLIC>"
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def short_hash(value: str, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def derive_base_table(source: Path) -> pd.DataFrame:
    raw = pd.read_csv(source)
    required = set(SMILES_LIST_COLUMNS) | {"Yield", "Title", "Type", "Stages"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = raw.drop(
        columns=[column for column in SOURCE_DROP_COLUMNS if column in raw.columns]
    ).copy()
    out.insert(0, "reaction_id", [f"prx_{i:06d}" for i in range(len(out))])

    canonical_lists: dict[str, list[list[str]]] = {}
    for column in SMILES_LIST_COLUMNS:
        sort_items = column == "Catalysts_Smiles"
        canonical_lists[column] = [
            canonicalize_smiles_list(value, sort_items=sort_items)
            for value in raw[column]
        ]
        out[column] = [
            serialize_smiles_list(items) for items in canonical_lists[column]
        ]

    # The source stores a Python-list literal in each molecular role.  The
    # graph encoder expects one RDKit-readable SMILES per role, so expose a
    # disconnected-molecule representation without discarding any component.
    out["reactants_graph_smiles"] = [
        ".".join(items) for items in canonical_lists["Reactants_Smiles"]
    ]
    out["products_graph_smiles"] = [
        ".".join(items) for items in canonical_lists["Products_Smiles"]
    ]
    out["reagents_graph_smiles"] = [
        ".".join(items) for items in canonical_lists["Reagents_Smiles"]
    ]
    out["catalysts_graph_smiles"] = [
        ".".join(items) for items in canonical_lists["Catalysts_Smiles"]
    ]

    paper_keys = [normalized_title(title) for title in raw["Title"]]
    out["paper_group"] = [short_hash(key, "paper") for key in paper_keys]

    catalyst_keys = [
        serialize_smiles_list(items)
        for items in canonical_lists["Catalysts_Smiles"]
    ]
    out["catalyst_group"] = [
        short_hash(key, "catalyst") for key in catalyst_keys
    ]
    out["n_catalysts"] = [len(items) for items in canonical_lists["Catalysts_Smiles"]]

    main_substrates: list[str] = []
    main_scaffolds: list[str] = []
    partner_scaffolds: list[str] = []
    scaffold_pairs: list[str] = []
    main_heavy_atoms: list[int] = []
    n_organic_reactants: list[int] = []

    for reactants in canonical_lists["Reactants_Smiles"]:
        molecules = [largest_organic_fragment(smiles) for smiles in reactants]
        molecules.sort(
            key=lambda mol: (mol.GetNumHeavyAtoms(), mol.GetNumAtoms()),
            reverse=True,
        )
        scaffolds = [scaffold_smiles(mol) for mol in molecules]
        main_substrates.append(
            Chem.MolToSmiles(molecules[0], canonical=True, isomericSmiles=True)
        )
        main_scaffolds.append(scaffolds[0])
        partner_scaffolds.append(scaffolds[1] if len(scaffolds) >= 2 else "<NONE>")
        scaffold_pairs.append(
            " || ".join(sorted(scaffolds[:2])) if len(scaffolds) >= 2 else "<NONE>"
        )
        main_heavy_atoms.append(molecules[0].GetNumHeavyAtoms())
        n_organic_reactants.append(len(molecules))

    out["main_substrate_smiles"] = main_substrates
    out["main_scaffold"] = main_scaffolds
    out["partner_scaffold"] = partner_scaffolds
    out["scaffold_pair"] = scaffold_pairs
    out["main_heavy_atoms"] = main_heavy_atoms
    out["n_organic_reactants"] = n_organic_reactants

    duplicate_keys = []
    for row in out.itertuples(index=False):
        duplicate_keys.append(
            "\x1f".join(
                [
                    getattr(row, "Reactants_Smiles"),
                    getattr(row, "Products_Smiles"),
                    getattr(row, "Reagents_Smiles"),
                    getattr(row, "Catalysts_Smiles"),
                ]
            )
        )
    out["duplicate_group"] = [
        short_hash(key, "duplicate") for key in duplicate_keys
    ]

    # Stable ordering makes all output files byte-reproducible.
    out = out.sort_values("reaction_id").reset_index(drop=True)
    return out


def add_paper_duplicate_components(df: pd.DataFrame) -> pd.DataFrame:
    """Connect rows sharing either a paper or an exact reaction context."""
    dsu = DisjointSet(len(df))
    for column in ("paper_group", "duplicate_group"):
        for indices in df.groupby(column, sort=False).indices.values():
            indices = list(indices)
            first = indices[0]
            for index in indices[1:]:
                dsu.union(first, index)
    roots = [dsu.find(i) for i in range(len(df))]
    root_to_id = {
        root: f"paper_component_{number:04d}"
        for number, root in enumerate(sorted(set(roots)))
    }
    result = df.copy()
    result["paper_duplicate_component"] = [root_to_id[root] for root in roots]
    return result


@dataclass(frozen=True)
class SelectionResult:
    groups: frozenset[str]
    rows: int
    score: float


def _category_matrix(df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    types = df["Type"].fillna("<MISSING>").astype(str)
    yield_bins = pd.cut(
        df["Yield"],
        bins=[-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf],
        labels=["y1", "y2", "y3", "y4", "y5"],
    ).astype(str)
    categories = types + "|" + yield_bins
    matrix = pd.crosstab(df[group_col], categories)
    matrix.insert(0, "__size__", df.groupby(group_col).size())
    reference = matrix.drop(columns="__size__").sum(axis=0).to_numpy(dtype=float)
    reference /= reference.sum()
    return matrix, reference


def select_balanced_groups(
    df: pd.DataFrame,
    group_col: str,
    candidates: Iterable[str],
    target_rows: int,
    *,
    seed: int,
    trials: int = 6000,
    min_groups: int = 1,
) -> SelectionResult:
    """Select a row-balanced, approximately chemistry-balanced group subset."""
    matrix, reference = _category_matrix(df, group_col)
    candidate_list = sorted(set(candidates) & set(matrix.index))
    if not candidate_list:
        raise ValueError(f"No candidate groups available for {group_col}")

    sizes = matrix.loc[candidate_list, "__size__"].to_numpy(dtype=int)
    category_counts = matrix.loc[candidate_list].drop(columns="__size__").to_numpy()
    rng = np.random.default_rng(seed)
    best: SelectionResult | None = None

    for _ in range(trials):
        order = rng.permutation(len(candidate_list))
        selected_positions: list[int] = []
        total = 0
        for position in order:
            size = int(sizes[position])
            if total < target_rows:
                selected_positions.append(int(position))
                total += size
            else:
                break

        if selected_positions:
            without_last = selected_positions[:-1]
            rows_without_last = total - int(sizes[selected_positions[-1]])
            if without_last and abs(rows_without_last - target_rows) < abs(total - target_rows):
                selected_positions = without_last
                total = rows_without_last

        if len(selected_positions) < min_groups or total == 0:
            continue

        selected_counts = category_counts[selected_positions].sum(axis=0).astype(float)
        selected_proportions = selected_counts / selected_counts.sum()
        size_error = abs(total - target_rows) / max(target_rows, 1)
        chemistry_error = float(np.abs(selected_proportions - reference).mean())
        diversity_penalty = 1.0 / math.sqrt(len(selected_positions))
        score = 12.0 * size_error + 3.0 * chemistry_error + 0.03 * diversity_penalty

        if best is None or score < best.score:
            groups = frozenset(candidate_list[pos] for pos in selected_positions)
            best = SelectionResult(groups=groups, rows=total, score=score)

    if best is None:
        raise RuntimeError(f"Could not select groups for {group_col}")
    return best


def labels_from_group_sets(
    df: pd.DataFrame,
    group_col: str,
    valid_groups: set[str] | frozenset[str],
    test_groups: set[str] | frozenset[str],
) -> pd.Series:
    labels = pd.Series("train", index=df.index, dtype=object)
    labels.loc[df[group_col].isin(valid_groups)] = "valid"
    labels.loc[df[group_col].isin(test_groups)] = "test"
    return labels


def split_paper_ood(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    working = add_paper_duplicate_components(df)
    group_col = "paper_duplicate_component"
    target = round(len(working) * 0.15)
    all_groups = set(working[group_col])
    test = select_balanced_groups(
        working, group_col, all_groups, target, seed=SEED + 101, min_groups=20
    )
    valid_candidates = all_groups - set(test.groups)
    valid = select_balanced_groups(
        working,
        group_col,
        valid_candidates,
        target,
        seed=SEED + 102,
        min_groups=20,
    )
    working["task_split"] = labels_from_group_sets(
        working, group_col, set(valid.groups), set(test.groups)
    )
    return working, {
        "description": "New-publication/protocol OOD with exact-reaction duplicates connected to their papers.",
        "primary_group": group_col,
        "constraints": [
            "paper_group is disjoint across splits",
            "duplicate_group is disjoint across splits",
        ],
    }


def split_catalyst_ood(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    group_col = "catalyst_group"
    stats = df.groupby(group_col).agg(
        n=("reaction_id", "size"),
        papers=("paper_group", "nunique"),
    )
    candidates = set(stats.index[(stats["n"] >= 20) & (stats["papers"] >= 2)])
    target = round(len(df) * 0.15)

    test = select_balanced_groups(
        df,
        group_col,
        candidates,
        target,
        seed=SEED + 201,
        min_groups=5,
    )
    valid_candidates = candidates - set(test.groups)
    valid = select_balanced_groups(
        df,
        group_col,
        valid_candidates,
        target,
        seed=SEED + 202,
        min_groups=5,
    )
    working = df.copy()
    working["task_split"] = labels_from_group_sets(
        working, group_col, set(valid.groups), set(test.groups)
    )
    return working, {
        "description": "Exact catalyst-bundle cold start; validation and test catalysts are absent from training.",
        "primary_group": group_col,
        "candidate_rule": "At least 20 reactions and at least 2 normalized papers.",
        "constraints": [
            "catalyst_group is disjoint across splits",
            "duplicate_group is disjoint across splits",
            "Type is retained for auditing but must not be used as a model feature",
        ],
    }


def split_scaffold_ood(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    group_col = "main_scaffold"
    frequencies = df[group_col].value_counts()
    forced_train = {"<ACYCLIC>", "c1ccccc1"}
    candidates = set(
        frequencies.index[
            (frequencies >= 5)
            & (frequencies <= 100)
            & (~frequencies.index.isin(forced_train))
        ]
    )
    target = round(len(df) * 0.15)
    test = select_balanced_groups(
        df,
        group_col,
        candidates,
        target,
        seed=SEED + 301,
        min_groups=20,
    )
    valid_candidates = candidates - set(test.groups)
    valid = select_balanced_groups(
        df,
        group_col,
        valid_candidates,
        target,
        seed=SEED + 302,
        min_groups=20,
    )
    working = df.copy()
    working["task_split"] = labels_from_group_sets(
        working, group_col, set(valid.groups), set(test.groups)
    )
    return working, {
        "description": "Main-substrate Bemis-Murcko scaffold OOD.",
        "primary_group": group_col,
        "candidate_rule": "Held-out scaffolds have 5-100 rows; generic phenyl and acyclic groups remain in training.",
        "constraints": [
            "main_scaffold is disjoint across splits",
            "duplicate_group is disjoint across splits",
        ],
    }


def _pair_component_counts(df: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in df.itertuples(index=False):
        components = {row.main_scaffold, row.partner_scaffold}
        for component in components:
            counts[component] += 1
    return counts


def split_scaffold_pair_ood(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    working = df.loc[df["n_organic_reactants"] >= 2].copy().reset_index(drop=True)
    group_col = "scaffold_pair"
    pair_counts = working[group_col].value_counts()
    component_counts = _pair_component_counts(working)

    candidates: set[str] = set()
    pair_components: dict[str, tuple[str, str]] = {}
    for pair, group in working.groupby(group_col, sort=False):
        first = group.iloc[0]
        components = (first["main_scaffold"], first["partner_scaffold"])
        pair_components[pair] = components
        pair_n = len(group)
        if not (2 <= pair_n <= 100):
            continue
        if all(component_counts[c] - pair_n >= 20 for c in set(components)):
            candidates.add(pair)

    target = round(len(working) * 0.15)
    test = select_balanced_groups(
        working,
        group_col,
        candidates,
        target,
        seed=SEED + 401,
        min_groups=20,
    )
    valid_candidates = candidates - set(test.groups)
    valid = select_balanced_groups(
        working,
        group_col,
        valid_candidates,
        target,
        seed=SEED + 402,
        min_groups=20,
    )
    working["task_split"] = labels_from_group_sets(
        working, group_col, set(valid.groups), set(test.groups)
    )

    # Enforce familiar individual components after both held-out sets are removed.
    train_component_counts = _pair_component_counts(
        working.loc[working["task_split"] == "train"]
    )
    heldout_pairs = set(valid.groups) | set(test.groups)
    violations = {
        pair
        for pair in heldout_pairs
        if any(train_component_counts[component] < 10 for component in set(pair_components[pair]))
    }
    if violations:
        working.loc[working[group_col].isin(violations), "task_split"] = "train"

    return working, {
        "description": "Unseen pair of individually familiar reactant scaffolds.",
        "primary_group": group_col,
        "candidate_rule": "Pair frequency 2-100; every component has at least 20 rows outside that pair.",
        "constraints": [
            "scaffold_pair is disjoint across splits",
            "every held-out component scaffold has at least 10 training rows",
            "duplicate_group is disjoint across splits",
            "single-reactant records are outside the scope of this task",
        ],
        "pairs_returned_to_train_for_support": len(violations),
    }


def split_multicatalyst_ood(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    paper_stats = df.groupby("paper_group")["n_catalysts"].agg(["min", "max"])
    paper_class = pd.Series(index=paper_stats.index, dtype=object)
    paper_class.loc[paper_stats["max"] == 1] = "train"
    paper_class.loc[(paper_stats["min"] == 1) & (paper_stats["max"] >= 2)] = "valid"
    paper_class.loc[paper_stats["min"] >= 2] = "test"

    working = df.copy()
    working["task_split"] = working["paper_group"].map(paper_class)
    if working["task_split"].isna().any():
        raise RuntimeError("Unclassified paper in multi-catalyst split")

    duplicate_spans = working.groupby("duplicate_group")["task_split"].nunique()
    excluded_groups = set(duplicate_spans.index[duplicate_spans > 1])
    excluded_rows = int(working["duplicate_group"].isin(excluded_groups).sum())
    if excluded_groups:
        working = working.loc[
            ~working["duplicate_group"].isin(excluded_groups)
        ].copy()

    return working, {
        "description": "Ordered shift from single-catalyst papers through mixed papers to exclusively multi-catalyst papers.",
        "primary_group": "paper_group",
        "constraints": [
            "train papers contain only one-item catalyst bundles",
            "validation papers contain both one-item and multi-item bundles",
            "test papers contain only multi-item catalyst bundles",
            "paper_group and duplicate_group are disjoint across splits",
        ],
        "excluded_cross_regime_duplicate_groups": len(excluded_groups),
        "excluded_cross_regime_duplicate_rows": excluded_rows,
    }


def split_complexity_ood(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    working = df.copy()
    conditions = [
        working["main_heavy_atoms"] <= 22,
        working["main_heavy_atoms"].between(23, 25),
        working["main_heavy_atoms"] >= 26,
    ]
    working["task_split"] = np.select(
        conditions, ["train", "valid", "test"], default="unassigned"
    )
    if (working["task_split"] == "unassigned").any():
        raise RuntimeError("Unassigned row in complexity split")
    return working, {
        "description": "Ordered molecular-complexity extrapolation using main-substrate heavy-atom count.",
        "primary_group": "main_heavy_atoms",
        "constraints": [
            "train: main_heavy_atoms <= 22",
            "validation: 23 <= main_heavy_atoms <= 25",
            "test: main_heavy_atoms >= 26",
            "duplicate_group is disjoint across splits",
        ],
    }


def pairwise_overlap(df: pd.DataFrame, column: str) -> dict[str, int]:
    values = {
        split: set(df.loc[df["task_split"] == split, column])
        for split in SPLIT_ORDER
    }
    return {
        "train_valid": len(values["train"] & values["valid"]),
        "train_test": len(values["train"] & values["test"]),
        "valid_test": len(values["valid"] & values["test"]),
    }


def catalyst_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["task_split", "catalyst_group"], as_index=False)
        .agg(
            n_reactions=("reaction_id", "size"),
            n_papers=("paper_group", "nunique"),
            catalyst_smiles=("Catalysts_Smiles", "first"),
            types=("Type", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
            mean_yield=("Yield", "mean"),
        )
        .sort_values(["task_split", "n_reactions"], ascending=[True, False])
    )


def group_assignment_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(["task_split", group_col], as_index=False)
        .agg(
            n_reactions=("reaction_id", "size"),
            n_papers=("paper_group", "nunique"),
            mean_yield=("Yield", "mean"),
        )
        .sort_values(["task_split", "n_reactions"], ascending=[True, False])
    )


def audit_task(df: pd.DataFrame, task_name: str, primary_group: str) -> dict:
    split_counts = df["task_split"].value_counts().reindex(SPLIT_ORDER, fill_value=0)
    audit = {
        "task": task_name,
        "n_total": int(len(df)),
        "split_counts": {split: int(split_counts[split]) for split in SPLIT_ORDER},
        "split_fractions": {
            split: round(float(split_counts[split] / len(df)), 6)
            for split in SPLIT_ORDER
        },
        "yield": {
            split: {
                "mean": round(
                    float(df.loc[df["task_split"] == split, "Yield"].mean()), 6
                ),
                "std": round(
                    float(df.loc[df["task_split"] == split, "Yield"].std()), 6
                ),
            }
            for split in SPLIT_ORDER
        },
        "primary_group_overlap": pairwise_overlap(df, primary_group),
        "duplicate_group_overlap": pairwise_overlap(df, "duplicate_group"),
        "paper_group_overlap": pairwise_overlap(df, "paper_group"),
        "catalyst_group_overlap": pairwise_overlap(df, "catalyst_group"),
        "main_scaffold_overlap": pairwise_overlap(df, "main_scaffold"),
        "scaffold_pair_overlap": pairwise_overlap(df, "scaffold_pair"),
    }
    if any(audit["primary_group_overlap"].values()):
        raise AssertionError(f"{task_name}: primary group leakage detected")
    if any(audit["duplicate_group_overlap"].values()):
        raise AssertionError(f"{task_name}: duplicate reaction leakage detected")
    if any(split_counts[split] == 0 for split in SPLIT_ORDER):
        raise AssertionError(f"{task_name}: empty split detected")
    return audit


def write_task(
    root: Path,
    task_name: str,
    df: pd.DataFrame,
    metadata: dict,
) -> dict:
    task_dir = root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    primary_group = str(metadata["primary_group"])
    audit = audit_task(df, task_name, primary_group)

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
        "main_substrate_smiles",
        "main_scaffold",
        "partner_scaffold",
        "scaffold_pair",
        "main_heavy_atoms",
        "n_organic_reactants",
        "duplicate_group",
    ]
    for split in SPLIT_ORDER:
        part = (
            df.loc[df["task_split"] == split, output_columns]
            .sort_values("reaction_id")
            .reset_index(drop=True)
        )
        part.to_csv(task_dir / f"{split}.csv", index=False)

    # Existing two-stage trainers make their own fold-safe validation split
    # from the fitting CSV, then retrain on that entire CSV.  This convenience
    # file therefore contains every non-test row while preserving the external
    # test set untouched.
    train_valid = (
        df.loc[df["task_split"].isin(["train", "valid"]), output_columns]
        .sort_values("reaction_id")
        .reset_index(drop=True)
    )
    train_valid.to_csv(task_dir / "train_valid.csv", index=False)

    assignment = group_assignment_summary(df, primary_group)
    assignment.to_csv(task_dir / "group_assignments.csv", index=False)
    if task_name == "02_catalyst_identity_ood":
        catalyst_group_summary(df).to_csv(
            task_dir / "catalyst_assignments.csv", index=False
        )

    complete_metadata = {
        "source_file": "data/photoredox/photoredox_scifinder.csv",
        "seed": SEED,
        "dropped_source_columns": sorted(SOURCE_DROP_COLUMNS),
        **metadata,
        "audit": audit,
    }
    (task_dir / "metadata.json").write_text(
        json.dumps(complete_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return audit


def write_summary(root: Path, audits: Sequence[dict]) -> None:
    rows = []
    for audit in audits:
        row = {"task": audit["task"], "n_total": audit["n_total"]}
        for split in SPLIT_ORDER:
            row[f"n_{split}"] = audit["split_counts"][split]
            row[f"frac_{split}"] = audit["split_fractions"][split]
            row[f"yield_mean_{split}"] = audit["yield"][split]["mean"]
        row["primary_train_test_overlap"] = audit["primary_group_overlap"][
            "train_test"
        ]
        row["duplicate_train_test_overlap"] = audit["duplicate_group_overlap"][
            "train_test"
        ]
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "split_summary.csv", index=False)


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
        default=Path("data/photoredox_ood"),
    )
    args = parser.parse_args()

    base = derive_base_table(args.source)
    args.output.mkdir(parents=True, exist_ok=True)

    task_builders = [
        ("01_paper_ood", split_paper_ood),
        ("02_catalyst_identity_ood", split_catalyst_ood),
        ("03_main_scaffold_ood", split_scaffold_ood),
        ("04_scaffold_pair_ood", split_scaffold_pair_ood),
        ("05_single_to_multicatalyst_ood", split_multicatalyst_ood),
        ("06_complexity_ood", split_complexity_ood),
    ]

    audits = []
    for task_name, builder in task_builders:
        task_df, metadata = builder(base)
        audits.append(write_task(args.output, task_name, task_df, metadata))
        counts = audits[-1]["split_counts"]
        print(
            f"{task_name}: "
            f"train={counts['train']} valid={counts['valid']} test={counts['test']}"
        )

    write_summary(args.output, audits)


if __name__ == "__main__":
    main()

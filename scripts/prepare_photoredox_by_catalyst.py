#!/usr/bin/env python3
"""Partition the complete photoredox table into exact catalyst-bundle folders."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from prepare_photoredox_full_scaffold_fewshot import product_aligned_substrate
from prepare_photoredox_ood import derive_base_table


OUTPUT_COLUMNS = [
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


def add_product_aligned_substrate(base: pd.DataFrame) -> pd.DataFrame:
    deduplicated = (
        base.sort_values("reaction_id")
        .drop_duplicates("duplicate_group", keep="first")
        .copy()
    )
    aligned = [
        product_aligned_substrate(reactants, products)
        for reactants, products in zip(
            deduplicated["Reactants_Smiles"],
            deduplicated["Products_Smiles"],
        )
    ]
    values_by_duplicate = {
        duplicate_group: values
        for duplicate_group, values in zip(deduplicated["duplicate_group"], aligned)
    }
    result = base.copy()
    result["substrate_smiles"] = [
        values_by_duplicate[value][0] for value in result["duplicate_group"]
    ]
    result["substrate_scaffold"] = [
        values_by_duplicate[value][1] for value in result["duplicate_group"]
    ]
    result["substrate_to_product_similarity"] = [
        values_by_duplicate[value][2] for value in result["duplicate_group"]
    ]
    result["substrate_scaffold_retained_in_product"] = [
        values_by_duplicate[value][3] for value in result["duplicate_group"]
    ]
    return result


def string_labels(values: pd.Series) -> list[str]:
    labels = sorted(set(values.dropna().astype(str)))
    return labels or ["<MISSING>"]


def yield_summary(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "mean": float(frame["Yield"].mean()),
        "std": float(frame["Yield"].std()) if len(frame) > 1 else 0.0,
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
        default=Path("data/photoredox_by_catalyst"),
    )
    args = parser.parse_args()

    base = add_product_aligned_substrate(derive_base_table(args.source))
    deduplicated = (
        base.sort_values("reaction_id")
        .drop_duplicates("duplicate_group", keep="first")
        .copy()
    )
    catalyst_groups = sorted(set(base["catalyst_group"]))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for catalyst_group in catalyst_groups:
        all_records = base.loc[base["catalyst_group"].eq(catalyst_group)].copy()
        modelling_data = deduplicated.loc[
            deduplicated["catalyst_group"].eq(catalyst_group)
        ].copy()
        catalyst_smiles = str(modelling_data.iloc[0]["Catalysts_Smiles"])
        catalyst_items = ast.literal_eval(catalyst_smiles)
        labels = string_labels(all_records["Type"])
        folder = args.output / catalyst_group
        folder.mkdir(parents=True, exist_ok=True)
        all_records[OUTPUT_COLUMNS].sort_values("reaction_id").to_csv(
            folder / "all_records.csv", index=False
        )
        modelling_data[OUTPUT_COLUMNS].sort_values("reaction_id").to_csv(
            folder / "data.csv", index=False
        )
        paper_summary = (
            modelling_data.groupby("paper_group", as_index=False)
            .agg(
                n_reactions=("reaction_id", "size"),
                n_substrates=("substrate_smiles", "nunique"),
                n_scaffolds=("substrate_scaffold", "nunique"),
                title=("Title", "first"),
                mean_yield=("Yield", "mean"),
            )
            .sort_values(["n_reactions", "paper_group"], ascending=[False, True])
        )
        paper_summary.to_csv(folder / "paper_summary.csv", index=False)

        metadata = {
            "catalyst_group": catalyst_group,
            "definition": "Exact canonical Catalysts_Smiles bundle",
            "catalyst_smiles": catalyst_smiles,
            "n_original_catalyst_items": int(len(catalyst_items)),
            "type_labels": labels,
            "counts": {
                "all_records": int(len(all_records)),
                "deduplicated_modelling_records": int(len(modelling_data)),
                "exact_duplicate_rows_removed": int(
                    len(all_records) - len(modelling_data)
                ),
                "papers": int(modelling_data["paper_group"].nunique()),
                "substrate_identities": int(
                    modelling_data["substrate_smiles"].nunique()
                ),
                "substrate_scaffolds": int(
                    modelling_data["substrate_scaffold"].nunique()
                ),
                "reactant_bundles": int(
                    modelling_data["reactants_graph_smiles"].nunique()
                ),
                "product_bundles": int(
                    modelling_data["products_graph_smiles"].nunique()
                ),
                "reagent_bundles": int(
                    modelling_data["reagents_graph_smiles"].nunique()
                ),
            },
            "yield_summary_for_audit_only": yield_summary(modelling_data),
            "source_split_column_included": False,
            "feature_warning": (
                "Type, Title, paper_group, catalyst_group, and audit columns "
                "must not be used as molecular model inputs."
            ),
        }
        (folder / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "catalyst_group": catalyst_group,
                "folder": str(folder),
                "type_labels": "|".join(labels),
                "n_all_records": len(all_records),
                "n_deduplicated": len(modelling_data),
                "n_exact_duplicates_removed": len(all_records) - len(modelling_data),
                "n_papers": modelling_data["paper_group"].nunique(),
                "n_substrate_identities": modelling_data["substrate_smiles"].nunique(),
                "n_substrate_scaffolds": modelling_data["substrate_scaffold"].nunique(),
                "n_reactant_bundles": modelling_data["reactants_graph_smiles"].nunique(),
                "n_product_bundles": modelling_data["products_graph_smiles"].nunique(),
                "n_reagent_bundles": modelling_data["reagents_graph_smiles"].nunique(),
                "mean_yield": modelling_data["Yield"].mean(),
                "catalyst_smiles": catalyst_smiles,
            }
        )

    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["n_deduplicated", "catalyst_group"], ascending=[False, True]
    )
    manifest.to_csv(args.output / "manifest.csv", index=False)
    family_summary = (
        manifest.assign(
            primary_type=manifest["type_labels"].str.split("|").str[0]
        )
        .groupby("primary_type", as_index=False)
        .agg(
            n_exact_catalyst_bundles=("catalyst_group", "size"),
            n_deduplicated_reactions=("n_deduplicated", "sum"),
            n_all_records=("n_all_records", "sum"),
        )
        .sort_values("n_deduplicated_reactions", ascending=False)
    )
    family_summary.to_csv(args.output / "family_summary.csv", index=False)

    readme = f"""# Photoredox 按精确催化体系拆分的数据

该目录由输入的 photoredox 记录生成，共 {len(catalyst_groups)} 个
`catalyst_*` 子目录。每个目录对应一种规范化后的精确 `Catalysts_Smiles` 组合，
不是一篇论文，也不是粗粒度 Ir/Ru 等家族标签。

每个子目录包含：

- `all_records.csv`：属于该催化体系的所有规范化记录，保留精确重复；
- `data.csv`：删除精确反应上下文重复后的推荐建模数据；
- `paper_summary.csv`：该体系在各论文中的数量、底物和骨架统计；
- `metadata.json`：完整催化剂 SMILES、类型标签、规模和审计信息。

顶层 `manifest.csv` 汇总所有催化体系，`family_summary.csv` 给出粗粒度标签统计。
所有 CSV 均删除原始 split、`Catalysts_Labels`、源行号和常数 `Steps`。

总计：`all_records.csv` 合计 {len(base):,} 条；`data.csv` 合计
{len(deduplicated):,} 条，删除 {len(base) - len(deduplicated):,} 条精确重复。

复现命令：

```bash
python \\
  scripts/prepare_photoredox_by_catalyst.py
```
"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")
    report = {
        "source_file": str(args.source),
        "output": str(args.output),
        "n_catalyst_folders": int(len(catalyst_groups)),
        "n_all_records": int(len(base)),
        "n_deduplicated_records": int(len(deduplicated)),
        "n_exact_duplicate_rows_removed": int(len(base) - len(deduplicated)),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

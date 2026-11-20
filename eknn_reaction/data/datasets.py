"""Dataset registry for eKNN reaction-prediction datasets.

Each entry defines:
- train_csv / test_csv: paths relative to the repository ``data/`` directory
- smiles_cols: SMILES column names to encode as input
- target_col: prediction target (Yield or G)
- group_col: column for TTT substrate grouping (None = single group)
- target_name: human-readable target name
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_DATA_DIR = os.environ.get("EKNN_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))


@dataclass
class DatasetConfig:
    name: str                          # short name, e.g. "bhc_test1"
    dataset: str                       # parent dataset: "BHC", "CPAC", etc.
    train_csv: str                     # absolute path to train CSV
    test_csv: str                      # absolute path to test CSV
    smiles_cols: List[str]             # SMILES column names
    target_col: str                    # "Yield" or "G"
    group_col: Optional[str] = None    # TTT grouping column
    condition_cols: List[str] = field(default_factory=list)
    """Prediction-time numerical reaction-condition columns (optional)."""
    condition_log1p_cols: List[str] = field(default_factory=list)
    """Subset of ``condition_cols`` transformed with log1p before scaling."""
    condition_categorical_cols: List[str] = field(default_factory=list)
    """Prediction-time categorical reaction-condition columns (optional)."""
    target_name: str = "Yield"         # human-readable
    description: str = ""


def _p(*parts: str) -> str:
    return os.path.join(_DATA_DIR, *parts)


# =========================================================================== #
# Dataset registry
# =========================================================================== #
DATASETS: Dict[str, DatasetConfig] = {
    # ----------------------------------------------------------------- BHC --
    "bhc_test1": DatasetConfig(
        name="bhc_test1", dataset="BHC",
        train_csv=_p("BHC", "BHC_Test1_with_names_train.csv"),
        test_csv=_p("BHC", "BHC_Test1_with_names_test.csv"),
        smiles_cols=["Ligand_smiles", "Additive_smiles", "Base_smiles", "Aryl halide_smiles"],
        target_col="Yield", group_col="Aryl halide_smiles",
        target_name="Yield", description="BHC Test1 (provided split; substrate identities overlap)"),
    "bhc_test2": DatasetConfig(
        name="bhc_test2", dataset="BHC",
        train_csv=_p("BHC", "BHC_Test2_with_names_train.csv"),
        test_csv=_p("BHC", "BHC_Test2_with_names_test.csv"),
        smiles_cols=["Ligand_smiles", "Additive_smiles", "Base_smiles", "Aryl halide_smiles"],
        target_col="Yield", group_col="Aryl halide_smiles",
        target_name="Yield", description="BHC Test2"),
    "bhc_test3": DatasetConfig(
        name="bhc_test3", dataset="BHC",
        train_csv=_p("BHC", "BHC_Test3_with_names_train.csv"),
        test_csv=_p("BHC", "BHC_Test3_with_names_test.csv"),
        smiles_cols=["Ligand_smiles", "Additive_smiles", "Base_smiles", "Aryl halide_smiles"],
        target_col="Yield", group_col="Aryl halide_smiles",
        target_name="Yield", description="BHC Test3"),
    "bhc_test4": DatasetConfig(
        name="bhc_test4", dataset="BHC",
        train_csv=_p("BHC", "BHC_Test4_with_names_train.csv"),
        test_csv=_p("BHC", "BHC_Test4_with_names_test.csv"),
        smiles_cols=["Ligand_smiles", "Additive_smiles", "Base_smiles", "Aryl halide_smiles"],
        target_col="Yield", group_col="Aryl halide_smiles",
        target_name="Yield", description="BHC Test4"),
    "bhc_full": DatasetConfig(
        name="bhc_full", dataset="BHC",
        train_csv=_p("BHC", "BHC_Full_train.csv"),
        test_csv=_p("BHC", "BHC_Full_test.csv"),
        smiles_cols=["Ligand", "Additive", "Base", "Aryl halide"],
        target_col="Yield", group_col="Aryl halide",
        target_name="Yield", description="BHC Full 3955 rxns, provided 7/3 split, seed=42"),
    "bhc_full_s123": DatasetConfig(
        name="bhc_full_s123", dataset="BHC",
        train_csv=_p("BHC", "BHC_Full_s123_train.csv"),
        test_csv=_p("BHC", "BHC_Full_s123_test.csv"),
        smiles_cols=["Ligand", "Additive", "Base", "Aryl halide"],
        target_col="Yield", group_col="Aryl halide",
        target_name="Yield", description="BHC Full reaction-level 7/3 split, seed=123"),
    "bhc_full_s456": DatasetConfig(
        name="bhc_full_s456", dataset="BHC",
        train_csv=_p("BHC", "BHC_Full_s456_train.csv"),
        test_csv=_p("BHC", "BHC_Full_s456_test.csv"),
        smiles_cols=["Ligand", "Additive", "Base", "Aryl halide"],
        target_col="Yield", group_col="Aryl halide",
        target_name="Yield", description="BHC Full reaction-level 7/3 split, seed=456"),
    "bhc_full_s789": DatasetConfig(
        name="bhc_full_s789", dataset="BHC",
        train_csv=_p("BHC", "BHC_Full_s789_train.csv"),
        test_csv=_p("BHC", "BHC_Full_s789_test.csv"),
        smiles_cols=["Ligand", "Additive", "Base", "Aryl halide"],
        target_col="Yield", group_col="Aryl halide",
        target_name="Yield", description="BHC Full reaction-level 7/3 split, seed=789"),

    # ---------------------------------------------------------------- CPAC --
    "cpac_test_cat": DatasetConfig(
        name="cpac_test_cat", dataset="CPAC",
        train_csv=_p("CPAC", "CPAC_test_cat_with_names_train.csv"),
        test_csv=_p("CPAC", "CPAC_test_cat_with_names_test.csv"),
        smiles_cols=["Catalyst_smiles", "Imine_smiles", "Thiol_smiles"],
        target_col="G", group_col="Catalyst_smiles",
        target_name="G", description="CPAC test-cat (group by catalyst)"),
    "cpac_test_sub-cat": DatasetConfig(
        name="cpac_test_sub-cat", dataset="CPAC",
        train_csv=_p("CPAC", "CPAC_test_sub-cat_with_names_train.csv"),
        test_csv=_p("CPAC", "CPAC_test_sub-cat_with_names_test.csv"),
        smiles_cols=["Catalyst_smiles", "Imine_smiles", "Thiol_smiles"],
        target_col="G", group_col="Imine_smiles",
        target_name="G", description="CPAC test_sub-cat (group by imine)"),
    "cpac_test_sub": DatasetConfig(
        name="cpac_test_sub", dataset="CPAC",
        train_csv=_p("CPAC", "CPAC_test_sub_with_names_train.csv"),
        test_csv=_p("CPAC", "CPAC_test_sub_with_names_test.csv"),
        smiles_cols=["Catalyst_smiles", "Imine_smiles", "Thiol_smiles"],
        target_col="G", group_col="Imine_smiles",
        target_name="G", description="CPAC test_sub (group by imine)"),

    # -------------------------------------------------------------- Ni-ECR --
    "niecr": DatasetConfig(
        name="niecr", dataset="Ni-ECR",
        train_csv=_p("Ni-ECR", "NiECR_with_names_train.csv"),
        test_csv=_p("Ni-ECR", "NiECR_with_names_test.csv"),
        smiles_cols=["Ligand_SMILES", "Product_SMILES"],
        target_col="G", group_col="Ligand_SMILES",
        target_name="G", description="Ni-ECR (group by ligand)"),

    # ------------------------------------------------------------- SHC-oAK --
    "shcoak": DatasetConfig(
        name="shcoak", dataset="SHC-oAK",
        train_csv=_p("SHC-oAK", "SHC-oAK_train.csv"),
        test_csv=_p("SHC-oAK", "SHC-oAK_test.csv"),
        smiles_cols=["S1", "S2"],
        target_col="Yield", group_col="S1",
        target_name="Yield", description="SHC-oAK (group by substrate S1)"),

    # ----------------------------------------------------------------- SMC --
    "smc_borane": DatasetConfig(
        name="smc_borane", dataset="SMC",
        train_csv=_p("SMC", "SMC_Borane_with_names_train.csv"),
        test_csv=_p("SMC", "SMC_Borane_with_names_test.csv"),
        smiles_cols=["SM_1_smiles", "SM_2_smiles", "Product_smiles",
                      "Ligand_smiles", "Base_smiles", "Solvent_smiles"],
        target_col="Yield", group_col="SM_1_smiles",
        target_name="Yield", description="SMC Borane split"),
    "smc_ligand": DatasetConfig(
        name="smc_ligand", dataset="SMC",
        train_csv=_p("SMC", "SMC_Ligand_with_names_train.csv"),
        test_csv=_p("SMC", "SMC_Ligand_with_names_test.csv"),
        smiles_cols=["SM_1_smiles", "SM_2_smiles", "Product_smiles",
                      "Ligand_smiles", "Base_smiles", "Solvent_smiles"],
        target_col="Yield", group_col="Ligand_smiles",
        target_name="Yield", description="SMC Ligand split"),
    "smc_random": DatasetConfig(
        name="smc_random", dataset="SMC",
        train_csv=_p("SMC", "SMC_Random_with_names_train.csv"),
        test_csv=_p("SMC", "SMC_Random_with_names_test.csv"),
        smiles_cols=["SM_1_smiles", "SM_2_smiles", "Product_smiles",
                      "Ligand_smiles", "Base_smiles", "Solvent_smiles"],
        target_col="Yield", group_col="SM_1_smiles",
        target_name="Yield", description="SMC Random split"),

    # ------------------------------------------------------------- SYP-POK --
    "syp_pok": DatasetConfig(
        name="syp_pok", dataset="SYP-POK",
        train_csv=_p("SYP-POK", "SYP-POK_train.csv"),
        test_csv=_p("SYP-POK", "SYP-POK_test.csv"),
        smiles_cols=["sulfoxonium_ylide_smiles", "P(O)OH_smiles",
                      "product_smiles", "Catalyst_smiles", "Solvent_smiles"],
        target_col="yield", group_col="Catalyst_smiles",
        target_name="Yield", description="SYP-POK (group by catalyst)"),

    # ----------------------------------------------------------- Photoredox --
    # The fitting CSV contains the prescribed train+validation partitions.
    # Training scripts create a fold-safe scaffold-group validation subset
    # from it, select eKNN controls there, and retrain on all non-test rows.
    "photoredox_main_scaffold_ood": DatasetConfig(
        name="photoredox_main_scaffold_ood", dataset="Photoredox",
        train_csv=_p(
            "photoredox_ood", "03_main_scaffold_ood", "train_valid.csv"
        ),
        test_csv=_p(
            "photoredox_ood", "03_main_scaffold_ood", "test.csv"
        ),
        smiles_cols=[
            "reactants_graph_smiles", "products_graph_smiles",
            "reagents_graph_smiles", "catalysts_graph_smiles",
        ],
        target_col="Yield", group_col="main_scaffold",
        target_name="Yield",
        description=(
            "Photoredox yield, unseen main-substrate Bemis-Murcko scaffold"
        ),
    ),
    "photoredox_arene_amination_pair_ood": DatasetConfig(
        name="photoredox_arene_amination_pair_ood", dataset="Photoredox",
        train_csv=_p(
            "photoredox_ood_small", "02_arene_amination_pair_ood",
            "train_valid.csv"
        ),
        test_csv=_p(
            "photoredox_ood_small", "02_arene_amination_pair_ood", "test.csv"
        ),
        smiles_cols=["amine_smiles", "arene_smiles"],
        target_col="Yield", group_col="reaction_pair_group",
        target_name="Yield",
        description=(
            "Single-paper Ru arene amination; unseen amine-arene pairs with "
            "both component identities observed during fitting"
        ),
    ),
    "photoredox_arene_amination_pair_ood_full": DatasetConfig(
        name="photoredox_arene_amination_pair_ood_full", dataset="Photoredox",
        train_csv=_p(
            "photoredox_ood_small", "02_arene_amination_pair_ood",
            "train_valid.csv"
        ),
        test_csv=_p(
            "photoredox_ood_small", "02_arene_amination_pair_ood", "test.csv"
        ),
        smiles_cols=[
            "reactants_graph_smiles", "products_graph_smiles",
            "reagents_graph_smiles", "catalysts_graph_smiles",
        ],
        target_col="Yield", group_col="reaction_pair_group",
        target_name="Yield",
        description=(
            "Single-paper Ru arene amination pair OOD with complete reaction "
            "roles known before yield measurement"
        ),
    ),
    "photoredox_condition_completion_650_selection": DatasetConfig(
        name="photoredox_condition_completion_650_selection",
        dataset="Photoredox",
        train_csv=_p(
            "photoredox_ood_small", "03_condition_completion_ood_650",
            "train.csv"
        ),
        test_csv=_p(
            "photoredox_ood_small", "03_condition_completion_ood_650",
            "valid.csv"
        ),
        smiles_cols=[
            "reactants_graph_smiles", "products_graph_smiles",
            "reagents_graph_smiles", "catalysts_graph_smiles",
        ],
        target_col="Yield", group_col="duplicate_group",
        target_name="Yield",
        description=(
            "Selection-only split for the 650-row same-paper near-condition "
            "completion OOD task"
        ),
    ),
    "photoredox_condition_completion_650": DatasetConfig(
        name="photoredox_condition_completion_650", dataset="Photoredox",
        train_csv=_p(
            "photoredox_ood_small", "03_condition_completion_ood_650",
            "train_valid.csv"
        ),
        test_csv=_p(
            "photoredox_ood_small", "03_condition_completion_ood_650",
            "test.csv"
        ),
        smiles_cols=[
            "reactants_graph_smiles", "products_graph_smiles",
            "reagents_graph_smiles", "catalysts_graph_smiles",
        ],
        target_col="Yield", group_col="duplicate_group",
        target_name="Yield",
        description=(
            "650-row same-paper near-condition completion OOD; identical "
            "substrate/product anchor and similar condition observed in train"
        ),
    ),
    "photoredox_heteroarene_scaffold_654_selection": DatasetConfig(
        name="photoredox_heteroarene_scaffold_654_selection",
        dataset="Photoredox",
        train_csv=_p(
            "photoredox_ood_small", "04_heteroarene_scaffold_ood_654",
            "train.csv"
        ),
        test_csv=_p(
            "photoredox_ood_small", "04_heteroarene_scaffold_ood_654",
            "valid.csv"
        ),
        smiles_cols=[
            "reactants_graph_smiles", "products_graph_smiles",
            "reagents_graph_smiles", "catalysts_graph_smiles",
        ],
        target_col="Yield", group_col="heteroarene_scaffold",
        target_name="Yield",
        description=(
            "Selection split for the 654-row expanded heteroarene strict "
            "unseen-scaffold OOD task"
        ),
    ),
    "photoredox_heteroarene_scaffold_654": DatasetConfig(
        name="photoredox_heteroarene_scaffold_654", dataset="Photoredox",
        train_csv=_p(
            "photoredox_ood_small", "04_heteroarene_scaffold_ood_654",
            "train_valid.csv"
        ),
        test_csv=_p(
            "photoredox_ood_small", "04_heteroarene_scaffold_ood_654",
            "test.csv"
        ),
        smiles_cols=[
            "reactants_graph_smiles", "products_graph_smiles",
            "reagents_graph_smiles", "catalysts_graph_smiles",
        ],
        target_col="Yield", group_col="heteroarene_scaffold",
        target_name="Yield",
        description=(
            "654-row expanded Minisci/heteroarene functionalisation task "
            "with strict held-out substrate Murcko scaffolds"
        ),
    ),

    # --------------------------------------------------------- Aho showcase --
    # Derived by scripts/prepare_aho_michael_eknn_showcase.py.  The source
    # table is retained untouched under data/aho/aho.csv.
    "aho_ddg_random": DatasetConfig(
        name="aho_ddg_random", dataset="Aho",
        train_csv=_p("aho", "eknn_showcase", "aho_ddg_random_train.csv"),
        test_csv=_p("aho", "eknn_showcase", "aho_ddg_random_test.csv"),
        smiles_cols=["reactant_smiles", "ligand_smiles", "metal_smiles",
                     "solvent_smiles", "additive_smiles"],
        target_col="ddG", group_col="split_group",
        target_name="ddG", description="Aho ddG, reaction-identity-grouped random split"),
    "aho_ddg_ligand_ood": DatasetConfig(
        name="aho_ddg_ligand_ood", dataset="Aho",
        train_csv=_p("aho", "eknn_showcase", "aho_ddg_ligand_ood_train.csv"),
        test_csv=_p("aho", "eknn_showcase", "aho_ddg_ligand_ood_test.csv"),
        smiles_cols=["reactant_smiles", "ligand_smiles", "metal_smiles",
                     "solvent_smiles", "additive_smiles"],
        target_col="ddG", group_col="split_group",
        target_name="ddG", description="Aho ddG, unseen ligand-identity split"),

    # ---------------------------------------------- Michael-addition showcase --
    # Derived by scripts/prepare_aho_michael_eknn_showcase.py.  Product SMILES
    # are intentionally not model inputs, avoiding outcome leakage.
    "michael_ddg_random": DatasetConfig(
        name="michael_ddg_random", dataset="Michael addition",
        train_csv=_p("Michael addition", "eknn_showcase", "michael_ddg_random_train.csv"),
        test_csv=_p("Michael addition", "eknn_showcase", "michael_ddg_random_test.csv"),
        smiles_cols=["acceptor_smiles", "donor_smiles", "catalyst_smiles",
                     "ligand_smiles", "solvent_smiles", "additive_smiles"],
        target_col="ddG", group_col="split_group",
        condition_cols=[
            "cond_temperature_c", "cond_step1_time_h", "cond_step2_temperature_c",
            "cond_step2_time_h", "cond_catalyst_equiv", "cond_ligand_equiv",
            "cond_acceptor_equiv", "cond_donor_equiv", "cond_additive1_equiv",
            "cond_additive2_equiv", "cond_solvent1_ml", "cond_solvent2_ml",
        ],
        condition_log1p_cols=[
            "cond_step1_time_h", "cond_step2_time_h", "cond_catalyst_equiv",
            "cond_ligand_equiv", "cond_acceptor_equiv", "cond_donor_equiv",
            "cond_additive1_equiv", "cond_additive2_equiv", "cond_solvent1_ml",
            "cond_solvent2_ml",
        ],
        condition_categorical_cols=[
            "cond_metal_label", "cond_ligand_type", "cond_michael_bonding_type",
            "cond_acceptor_class", "cond_donor_class",
        ],
        target_name="ddG", description="Michael-addition ddG, reaction-identity-grouped random split"),
    "michael_ddg_row_random": DatasetConfig(
        name="michael_ddg_row_random", dataset="Michael addition",
        train_csv=_p("Michael addition", "eknn_showcase", "michael_ddg_row_random_train.csv"),
        test_csv=_p("Michael addition", "eknn_showcase", "michael_ddg_row_random_test.csv"),
        smiles_cols=["acceptor_smiles", "donor_smiles", "catalyst_smiles",
                     "ligand_smiles", "solvent_smiles", "additive_smiles"],
        target_col="ddG", group_col="split_group",
        condition_cols=[
            "cond_temperature_c", "cond_step1_time_h", "cond_step2_temperature_c",
            "cond_step2_time_h", "cond_catalyst_equiv", "cond_ligand_equiv",
            "cond_acceptor_equiv", "cond_donor_equiv", "cond_additive1_equiv",
            "cond_additive2_equiv", "cond_solvent1_ml", "cond_solvent2_ml",
        ],
        condition_log1p_cols=[
            "cond_step1_time_h", "cond_step2_time_h", "cond_catalyst_equiv",
            "cond_ligand_equiv", "cond_acceptor_equiv", "cond_donor_equiv",
            "cond_additive1_equiv", "cond_additive2_equiv", "cond_solvent1_ml",
            "cond_solvent2_ml",
        ],
        condition_categorical_cols=[
            "cond_metal_label", "cond_ligand_type", "cond_michael_bonding_type",
            "cond_acceptor_class", "cond_donor_class",
        ],
        target_name="ddG",
        description=("Michael-addition ddG, ordinary row-level random split "
                     "(duplicate reaction identities may cross train/test)")),
    "michael_ddg_acceptor_ood": DatasetConfig(
        name="michael_ddg_acceptor_ood", dataset="Michael addition",
        train_csv=_p("Michael addition", "eknn_showcase", "michael_ddg_acceptor_ood_train.csv"),
        test_csv=_p("Michael addition", "eknn_showcase", "michael_ddg_acceptor_ood_test.csv"),
        smiles_cols=["acceptor_smiles", "donor_smiles", "catalyst_smiles",
                     "ligand_smiles", "solvent_smiles", "additive_smiles"],
        target_col="ddG", group_col="split_group",
        condition_cols=[
            "cond_temperature_c", "cond_step1_time_h", "cond_step2_temperature_c",
            "cond_step2_time_h", "cond_catalyst_equiv", "cond_ligand_equiv",
            "cond_acceptor_equiv", "cond_donor_equiv", "cond_additive1_equiv",
            "cond_additive2_equiv", "cond_solvent1_ml", "cond_solvent2_ml",
        ],
        condition_log1p_cols=[
            "cond_step1_time_h", "cond_step2_time_h", "cond_catalyst_equiv",
            "cond_ligand_equiv", "cond_acceptor_equiv", "cond_donor_equiv",
            "cond_additive1_equiv", "cond_additive2_equiv", "cond_solvent1_ml",
            "cond_solvent2_ml",
        ],
        condition_categorical_cols=[
            "cond_metal_label", "cond_ligand_type", "cond_michael_bonding_type",
            "cond_acceptor_class", "cond_donor_class",
        ],
        target_name="ddG", description="Michael-addition ddG, unseen acceptor-identity split"),
}


def get_dataset(name: str) -> DatasetConfig:
    """Get a dataset config by name."""
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    return DATASETS[name]


def list_datasets() -> List[str]:
    """Return all dataset names."""
    return list(DATASETS.keys())


def list_datasets_by_group() -> Dict[str, List[str]]:
    """Group dataset names by parent dataset."""
    groups: Dict[str, List[str]] = {}
    for name, cfg in DATASETS.items():
        groups.setdefault(cfg.dataset, []).append(name)
    return groups

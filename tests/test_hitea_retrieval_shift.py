"""Tests for the leakage-safe HiTEA retrieval-shift preparation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_hitea_retrieval_shift",
    ROOT / "scripts" / "prepare_hitea_retrieval_shift.py",
)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare)


def test_grouped_target_assignment_is_nested_and_disjoint():
    rows = []
    for group in range(20):
        for replicate in range(1 + group % 3):
            rows.append(
                {
                    "record_id": f"r-{group}-{replicate}",
                    "exact_input_group": f"g-{group}",
                    "reaction_core_group": f"core-{group % 4}",
                }
            )
    target = pd.DataFrame(rows)
    assignment = prepare.make_target_assignment(
        target,
        keyword="HECK",
        screen_id="SCRN_TEST",
        test_fraction=0.20,
    )

    test = assignment["is_target_test"]
    assert test.any()
    assert (~test).any()
    assert not assignment.loc[test, "is_calibration_20"].any()
    assert assignment.loc[assignment["is_calibration_05"], "is_calibration_10"].all()
    assert assignment.loc[assignment["is_calibration_10"], "is_calibration_20"].all()
    grouped = assignment.groupby("exact_input_group")
    for column in [
        "is_target_test",
        "is_calibration_05",
        "is_calibration_10",
        "is_calibration_20",
    ]:
        assert grouped[column].nunique().max() == 1


def test_target_assignment_is_deterministic():
    target = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(30)],
            "exact_input_group": [f"g{i}" for i in range(30)],
            "reaction_core_group": [f"c{i % 3}" for i in range(30)],
        }
    )
    left = prepare.make_target_assignment(
        target, keyword="SNAR", screen_id="S1", test_fraction=0.20
    )
    right = prepare.make_target_assignment(
        target.sample(frac=1.0, random_state=7),
        keyword="SNAR",
        screen_id="S1",
        test_fraction=0.20,
    )
    columns = [
        "record_id",
        "is_target_test",
        "is_calibration_05",
        "is_calibration_10",
        "is_calibration_20",
    ]
    assert left[columns].sort_values("record_id").reset_index(drop=True).equals(
        right[columns].sort_values("record_id").reset_index(drop=True)
    )


def test_exact_input_group_does_not_depend_on_target():
    required = {column: [None, None] for column in prepare.REQUIRED_COLUMNS}
    required.update(
        {
            "Dataset entry number": [1, 2],
            "REACTION_ID": ["r1", "r2"],
            "ReactionClass": ["Pd coupling", "Pd coupling"],
            "SCREEN_ID": ["S1", "S1"],
            "NOTEBOOK_ID": ["N1", "N1"],
            "KeyWord_STD": ["HECK", "HECK"],
            "ReactionGroup": ["Metal Mediated", "Metal Mediated"],
            "RXN_SMILES": ["C>>CO", "C>>CO"],
            "PCAT_CMPD_ID": ["p", "p"],
            "PRODUCT_STRUCTURE": ["CO", "CO"],
            "Product_Yield_PCT_Area_UV": [0.0, 99.0],
            "Reactant_1_SMILES": ["C", "C"],
            "Reaction_T": [25.0, 25.0],
        }
    )
    table = prepare.build_modeling_table(pd.DataFrame(required))
    assert table.loc[0, "exact_input_group"] == table.loc[1, "exact_input_group"]
    assert "yie_react" not in table.columns


def test_invalid_smiles_audit_is_role_specific():
    table = pd.DataFrame(
        {
            "reaction_keyword": ["HECK", "HECK"],
            **{
                role: ["C", "C"]
                for role in prepare.SMILES_MAPPING.values()
            },
        }
    )
    table.loc[1, "catalyst_1_smiles_1"] = "B(F)(F)(F)F"
    invalid, affected = prepare.invalid_smiles_audit(table)
    assert affected == 1
    assert len(invalid) == 1
    assert invalid.iloc[0]["role"] == "catalyst_1_smiles_1"

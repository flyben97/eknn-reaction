"""Focused tests for the Photoredox cross-fitted hierarchical eKNN runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "photoredox_cf_eknn",
    ROOT / "scripts" / "run_photoredox_full_cf_hierarchical_eknn.py",
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_top_neighbours_and_weighting_are_ordered():
    memory = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    query = np.asarray([[1.0, 0.0]])
    indices, distances = runner.top_neighbours(query, memory, max_k=2)
    assert indices.tolist() == [[0, 1]]
    prediction = runner.weighted_from_top_neighbours(
        indices, distances, np.asarray([0.8, 0.2, -1.0]), k=1, temperature=0.1
    )
    assert np.allclose(prediction, [0.8])


def test_hierarchy_prefers_stratum_and_excludes_identity():
    memory_frame = pd.DataFrame(
        {
            "substrate_scaffold": ["A", "A", "A", "B"],
            "catalyst_group": ["C1", "C1", "C2", "C1"],
            "substrate_smiles": ["same", "other", "third", "fourth"],
        }
    )
    query_frame = pd.DataFrame(
        {
            "substrate_scaffold": ["A"],
            "catalyst_group": ["C1"],
            "substrate_smiles": ["same"],
        }
    )
    embeddings = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.8, 0.2], [0.0, 1.0]], dtype=float
    )
    embeddings = runner.normalise_rows(embeddings)
    correction, diagnostics = runner.target_memory_correction(
        np.asarray([[1.0, 0.0]]),
        query_frame,
        embeddings,
        np.asarray([100.0, 0.4, 0.2, -0.5]),
        memory_frame,
        k=3,
        temperature=0.1,
        scale=1.0,
        shrinkage=0.0,
        scope="hierarchical",
        exclude_query_identity=True,
    )
    assert np.allclose(correction, [0.4])
    assert diagnostics.loc[0, "target_scope_level"] == "stratum"
    assert diagnostics.loc[0, "target_pool"] == 1


def test_budget_selects_one_reaction_per_stratum():
    rows = []
    for scaffold in range(8):
        for catalyst in range(6):
            for replicate in range(2):
                rows.append(
                    {
                        "reaction_id": f"r-{scaffold}-{catalyst}-{replicate}",
                        "substrate_scaffold": f"s{scaffold}",
                        "catalyst_group": f"c{catalyst}",
                    }
                )
    frame = pd.DataFrame(rows)
    selected = runner.deterministic_budget_48(frame)
    selected_frame = frame.iloc[selected]
    counts = selected_frame.groupby(
        ["substrate_scaffold", "catalyst_group"]
    ).size()
    assert len(selected) == 48
    assert counts.eq(1).all()


def test_paper_domain_correction_shrinks_and_cold_starts_globally():
    calibration = pd.DataFrame(
        {
            "paper_group": ["p1", "p1", "p2"],
            "substrate_scaffold": ["s1", "s1", "s2"],
            "catalyst_group": ["c1", "c1", "c2"],
        }
    )
    query = pd.DataFrame(
        {
            "paper_group": ["p1", "new"],
            "substrate_scaffold": ["s1", "s3"],
            "catalyst_group": ["c1", "c3"],
        }
    )
    correction, seen = runner.paper_domain_correction(
        query,
        calibration,
        np.asarray([0.3, 0.1, -0.1]),
        scope="paper_stratum",
        paper_shrinkage=2.0,
        stratum_shrinkage=2.0,
    )
    global_value = 0.1
    paper_value = (0.4 + 2.0 * global_value) / 4.0
    expected_seen = (0.4 + 2.0 * paper_value) / 4.0
    assert np.allclose(correction, [expected_seen, global_value])
    assert seen.tolist() == [True, False]


def test_paper_axis_contract_allows_scaffold_overlap_but_not_paper_overlap():
    def frame(reaction, paper, duplicate):
        return pd.DataFrame(
            {
                "reaction_id": [reaction],
                "substrate_scaffold": ["shared_scaffold"],
                "catalyst_group": ["c1"],
                "paper_group": [paper],
                "duplicate_group": [duplicate],
            }
        )

    audit = runner.audit_data_contract(
        frame("source", "source_paper", "d1"),
        frame("calibration", "target_paper", "d2"),
        frame("target", "target_paper", "d3"),
        deployment_axis="paper",
    )
    assert audit["source_target_scaffold_overlap"] == 1
    assert audit["source_target_paper_overlap"] == 0

#!/usr/bin/env python3
"""Unit/integration tests for eKNN reaction prediction (GNN-based)."""

import os
import sys

import torch
import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.join(_script_dir, "..")
sys.path.insert(0, _project_dir)

from eknn_reaction.data import get_dataset, list_datasets, Standardizer
from eknn_reaction.data.tokenizer import SMILESTokenizer, SPECIAL_TOKENS
from eknn_reaction.models import ReactionGNN
from eknn_reaction.utils import set_seed


class TestTokenizer:
    def test_basic_tokenization(self):
        tok = SMILESTokenizer(max_len=64)
        tokens = tok.tokenize_smiles("C1=CC=CC=C1Br")
        assert "C" in tokens
        assert "Br" in tokens
        assert "1" in tokens
        assert "=" in tokens

    def test_bracketed_atoms(self):
        tok = SMILESTokenizer(max_len=64)
        tokens = tok.tokenize_smiles("C[C@@H](C)O")
        assert "[C@@H]" in tokens

    def test_encode_decode(self):
        tok = SMILESTokenizer(max_len=64)
        smiles = "CC(=O)O"
        tok.fit([smiles])
        ids = tok.encode(smiles)
        decoded = tok.decode(ids)
        assert smiles in decoded

    def test_vocab_building(self):
        tok = SMILESTokenizer(max_len=64)
        tok.fit(["CCO", "CCN", "c1ccccc1"])
        assert tok.vocab_size > len(SPECIAL_TOKENS)
        assert tok.pad_idx == 0


class TestStandardizer:
    def test_normalize_denormalize(self):
        std = Standardizer(mean=50.0, std=10.0)
        y = np.array([40.0, 50.0, 60.0], dtype=np.float32)
        y_norm = std.normalize(y)
        assert abs(y_norm[1]) < 0.01  # mean → 0
        assert y_norm[2] > 0  # above mean → positive
        y_back = std.denormalize(y_norm)
        assert np.allclose(y, y_back, atol=0.01)


class TestDatasets:
    def test_dataset_registry(self):
        ds_names = list_datasets()
        assert len(ds_names) > 0
        assert "cpac_test_sub" in ds_names
        ds = get_dataset("cpac_test_sub")
        assert ds.target_col == "G"
        assert len(ds.smiles_cols) == 3


class TestModels:
    def test_reaction_gnn_build(self):
        set_seed(0)
        model = ReactionGNN(d_model=128, n_layers=2, n_heads=2,
                         ff_dim=256, pe_dim=3, backbone="mpnn")
        n = model.get_num_params()
        assert n > 0

    def test_reaction_gnn_backbone_registry(self):
        """All registered backbones should build without error."""
        backbones = ["gps", "mpnn", "gat", "gcn", "sage", "gin", "gatv2",
                     "transformer_conv", "gen_conv", "film", "gated_graph",
                     "res_gated", "dmpnn", "egconv"]
        for bb in backbones:
            try:
                model = ReactionGNN(d_model=64, n_layers=1, n_heads=2,
                                 ff_dim=128, backbone=bb)
                n = model.get_num_params()
                assert n > 0
            except Exception as e:
                if "not available" in str(e) or "has no attribute" in str(e):
                    pass  # skip if PyG conv layer not available
                else:
                    raise

    def test_reaction_gnn_with_virtual_node(self):
        set_seed(0)
        model = ReactionGNN(d_model=128, n_layers=2, backbone="mpnn",
                         use_virtual_node=True)
        assert model.get_num_params() > 0

    def test_reaction_gnn_with_jk(self):
        set_seed(0)
        model = ReactionGNN(d_model=128, n_layers=2, backbone="mpnn",
                         use_jk=True)
        assert model.get_num_params() > 0

    def test_conditioned_role_interaction_forward(self):
        from torch_geometric.data import Batch
        from eknn_reaction.models.graph_encoder import smiles_to_graph

        graph = smiles_to_graph(["CC=O", "CCN", "[Zn]"], pe_k=3)
        graph.condition_features = torch.zeros((1, 10), dtype=torch.float32)
        batch = Batch.from_data_list([graph])
        model = ReactionGNN(
            d_model=32,
            n_layers=1,
            n_heads=2,
            ff_dim=64,
            condition_dim=10,
            backbone="mpnn",
            use_role_interaction_readout=True,
            role_interaction_layers=1,
            retrieval_dim=16,
        )
        output = model(batch)
        assert output["yield"].shape == (1, 1)
        assert output["embedding"].shape == (1, 16)

    def test_conditioned_role_concat_forward(self):
        from torch_geometric.data import Batch
        from eknn_reaction.models.graph_encoder import smiles_to_graph

        graph = smiles_to_graph(["CC=O", "CCN", "[Zn]"], pe_k=3)
        graph.condition_features = torch.zeros((1, 10), dtype=torch.float32)
        batch = Batch.from_data_list([graph])
        model = ReactionGNN(
            d_model=32,
            n_layers=1,
            n_heads=2,
            ff_dim=64,
            condition_dim=10,
            backbone="gcn",
            use_role_concat_readout=True,
        )
        output = model(batch)
        assert output["yield"].shape == (1, 1)
        assert output["embedding"].shape == (1, 64)

    def test_conditioned_role_factorized_forward(self):
        from torch_geometric.data import Batch
        from eknn_reaction.models.graph_encoder import smiles_to_graph

        graph = smiles_to_graph(
            ["CC=O", "CCN", "[Zn]", "P", "O", "Cl"],
            add_laplacian_pe=False,
            pe_k=0,
        )
        graph.condition_features = torch.zeros(
            (1, 10), dtype=torch.float32
        )
        batch = Batch.from_data_list([graph])
        model = ReactionGNN(
            d_model=32,
            n_layers=1,
            n_heads=2,
            ff_dim=64,
            pe_dim=0,
            condition_dim=10,
            backbone="res_gated",
            use_role_factorized_readout=True,
            use_role_factorized_retrieval=True,
        )
        output = model(batch)
        assert output["yield"].shape == (1, 1)
        assert output["embedding"].shape == (1, 256)

    def test_conditioned_rolewise_forward_with_identity_tokens(self):
        from torch_geometric.data import Batch
        from eknn_reaction.models.graph_encoder import smiles_to_graph

        graph = smiles_to_graph(
            ["CC=O", "CCN", "[Zn]", "", "O", "Cl"], pe_k=3
        )
        graph.condition_features = torch.zeros((1, 10), dtype=torch.float32)
        graph.component_identity_features = torch.tensor(
            [[0, 1, 2, 0, 1, 1]], dtype=torch.long
        )
        batch = Batch.from_data_list([graph])
        model = ReactionGNN(
            d_model=32,
            n_layers=1,
            n_heads=2,
            ff_dim=64,
            condition_dim=10,
            component_identity_cardinalities=[2, 3, 4, 2, 3, 3],
            component_identity_dim=8,
            backbone="gcn",
            readout="mean_max_sum",
            use_rolewise_readout=True,
        )
        output = model(batch)
        assert output["yield"].shape == (1, 1)
        assert output["embedding"].shape == (1, 96)


class TestGraphEncoder:
    def test_smiles_to_graph(self):
        """Build a molecular graph from SMILES (requires RDKit)."""
        from eknn_reaction.models.graph_encoder import smiles_to_graph
        smiles_list = ["CCO", "CC(=O)O", "c1ccccc1"]
        g = smiles_to_graph(smiles_list, add_laplacian_pe=True, pe_k=3)
        if g is None:
            return  # RDKit parsing failed (rare)
        assert g.x.shape[0] > 0  # has atoms
        assert g.edge_index.shape[1] > 0  # has bonds
        assert g.x.shape[1] == 161  # ATOM_FEAT_DIM

    def test_bidirectional_bond_features_stay_aligned(self):
        from eknn_reaction.models.graph_encoder import smiles_to_graph

        graph = smiles_to_graph(["C=CC"], add_laplacian_pe=False)
        assert torch.equal(graph.edge_attr[0], graph.edge_attr[1])
        assert torch.equal(graph.edge_attr[2], graph.edge_attr[3])
        assert not torch.equal(graph.edge_attr[0], graph.edge_attr[2])

    def test_missing_component_does_not_shift_role_ids(self):
        from eknn_reaction.models.graph_encoder import smiles_to_graph

        graph = smiles_to_graph(["CC", "", "O"], add_laplacian_pe=False)
        assert set(graph.molecule_ids.tolist()) == {0, 2}


class TestGroupBlockedLooSelector:
    """Pure-function tests for the v2 blind-development selection primitives
    (scripts/develop_hitea_target_memory_v2.py)."""

    @staticmethod
    def _load_v2():
        import importlib.util

        root = os.path.join(_project_dir)
        spec = importlib.util.spec_from_file_location(
            "develop_v2_for_tests",
            os.path.join(root, "scripts", "develop_hitea_target_memory_v2.py"),
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_group_blocked_loo_excludes_whole_group(self):
        """A memory row's neighbours must exclude its whole exact_input_group,
        falling back to self-exclusion only when that empties the memory."""
        import numpy as np

        v2 = self._load_v2()
        # 4 memory rows: rows 0/1 share a group; row 0 is the held-out row.
        similarity = np.array(
            [
                [1.0, 0.99, 0.10, 0.10],
                [0.99, 1.0, 0.10, 0.10],
                [0.10, 0.10, 1.0, 0.50],
                [0.10, 0.10, 0.50, 1.0],
            ]
        )
        base = np.array([10.0, 40.0, 20.0, 20.0])
        y = np.array([15.0, 45.0, 20.0, 20.0])
        groups = np.array(["gA", "gA", "gB", "gC"])

        label, residual = v2.loo_neighbour_components(
            similarity, base, y, k=1, temperature=0.05, group_ids=groups
        )
        # Held-out row 0 must not see row 1 (same group) despite 0.99
        # similarity: its nearest allowed neighbour is row 2 or 3.
        assert label[0] == 20.0
        assert residual[0] == 0.0

        # Self-exclusion fallback: row 0 alone in its group.
        groups_solo = np.array(["gA", "gB", "gC", "gD"])
        label_solo, _ = v2.loo_neighbour_components(
            similarity, base, y, k=1, temperature=0.05, group_ids=groups_solo
        )
        assert label_solo[0] == 45.0

    def test_blend_from_components_endpoints(self):
        """blend=0 is pure scaled-residual kNN; blend=1 ignores the base."""
        v2 = self._load_v2()
        query_base = np.array([10.0])
        local_label = np.array([50.0])
        local_residual = np.array([2.0])
        pure_residual = v2.blend_from_components(
            query_base, local_label, local_residual, scale=1.0, blend=0.0
        )
        pure_label = v2.blend_from_components(
            query_base, local_label, local_residual, scale=1.0, blend=1.0
        )
        assert pure_residual[0] == 12.0
        assert pure_label[0] == 50.0

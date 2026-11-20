"""Reaction Graph Builder — SMILES → PyG Data conversion.

Converts multi-component reaction SMILES to a disjoint molecular graph
suitable for GraphGPS-style encoding + ekNN TTT.

Requires: rdkit, torch, torch_geometric
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdmolops
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.transforms import AddLaplacianEigenvectorPE
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


# ──────────────────────────────────────────────────────────────────────
# Atom & Bond Featurisation
# ──────────────────────────────────────────────────────────────────────

# RDKit atomic number → index (0=padded, 1-118=elements)
ATOM_NUM_OFFSET = 1  # 0 = padding

# Hybridisation mapping
_HYBRID_MAP = {
    Chem.rdchem.HybridizationType.SP:    1,
    Chem.rdchem.HybridizationType.SP2:   2,
    Chem.rdchem.HybridizationType.SP3:   3,
    Chem.rdchem.HybridizationType.SP3D:  4,
    Chem.rdchem.HybridizationType.SP3D2: 5,
    Chem.rdchem.HybridizationType.OTHER: 6,
}

# Chirality mapping
_CHIRAL_MAP = {
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:  1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    Chem.rdchem.ChiralType.CHI_OTHER:           3,
}

# Bond type mapping
_BOND_MAP = {
    Chem.rdchem.BondType.SINGLE:   1,
    Chem.rdchem.BondType.DOUBLE:   2,
    Chem.rdchem.BondType.TRIPLE:   3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

# Bond stereo mapping
_BOND_STEREO_MAP = {
    Chem.rdchem.BondStereo.STEREOANY: 1,
    Chem.rdchem.BondStereo.STEREOZ:   2,
    Chem.rdchem.BondStereo.STEREOE:   3,
}

# Feature dimensions
ATOM_FEAT_DIM = 161   # total one-hot dims
BOND_FEAT_DIM = 13


def _one_hot(value: int, num_classes: int) -> np.ndarray:
    """Create one-hot encoding with clipping."""
    arr = np.zeros(num_classes, dtype=np.float32)
    if 0 <= value < num_classes:
        arr[value] = 1.0
    return arr


def atom_features(mol: Chem.Mol) -> np.ndarray:
    """Extract one-hot atom features for all atoms in a molecule.

    Returns:
        np.ndarray [N_atoms, ATOM_FEAT_DIM] (161-dim one-hot)
    """
    feats = []
    for atom in mol.GetAtoms():
        f = []
        # atomic number: 0-118 (119 dims)
        an = atom.GetAtomicNum()
        f.append(_one_hot(min(an, 118), 119))
        # degree: 0-10 (11 dims)
        f.append(_one_hot(min(atom.GetDegree(), 10), 11))
        # formal charge: -5..+5 → 0..10 (11 dims)
        fc = atom.GetFormalCharge()
        f.append(_one_hot(max(0, min(fc + 5, 10)), 11))
        # chirality: 0-3 (4 dims)
        f.append(_one_hot(_CHIRAL_MAP.get(atom.GetChiralTag(), 0), 4))
        # hybridisation: 0-6 (7 dims)
        f.append(_one_hot(_HYBRID_MAP.get(atom.GetHybridization(), 0), 7))
        # aromatic: 0-1 (2 dims)
        f.append(_one_hot(int(atom.GetIsAromatic()), 2))
        # num hydrogens: 0-4 (5 dims, 4+ → 4)
        f.append(_one_hot(min(atom.GetTotalNumHs(), 4), 5))
        # in ring: 0-1 (2 dims)
        f.append(_one_hot(int(atom.IsInRing()), 2))

        feats.append(np.concatenate(f))

    return np.stack(feats)


def bond_features(mol: Chem.Mol) -> np.ndarray:
    """Extract one-hot bond features for all bonds in a molecule.

    Returns:
        np.ndarray [N_bonds * 2, BOND_FEAT_DIM] (bi-directional, 13-dim one-hot)
    """
    feats = []
    for bond in mol.GetBonds():
        f = []
        # bond type: 0-4 (5 dims)
        f.append(_one_hot(_BOND_MAP.get(bond.GetBondType(), 0), 5))
        # conjugated: 0-1 (2 dims)
        f.append(_one_hot(int(bond.GetIsConjugated()), 2))
        # in ring: 0-1 (2 dims)
        f.append(_one_hot(int(bond.IsInRing()), 2))
        # stereo: 0-3 (4 dims)
        f.append(_one_hot(_BOND_STEREO_MAP.get(bond.GetStereo(), 0), 4))
        feats.append(np.concatenate(f))

    if len(feats) == 0:
        return np.zeros((0, BOND_FEAT_DIM), dtype=np.float32)
    return np.stack(feats)


# ──────────────────────────────────────────────────────────────────────
# Graph Building
# ──────────────────────────────────────────────────────────────────────

def smiles_to_mol(smiles: str, sanitize: bool = True) -> Optional[Chem.Mol]:
    """Parse a single SMILES string to an RDKit molecule."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        if sanitize:
            Chem.SanitizeMol(mol)
        # Add hydrogens explicitly (RDKit by default uses implicit H)
        # We keep implicit for efficiency — degree already accounts for H count
        return mol
    except Exception:
        return None


def smiles_to_graph(
    smiles_list: List[str],
    add_laplacian_pe: bool = True,
    pe_k: int = 3,
) -> Optional[Data]:
    """Convert a list of component SMILES strings to a single disjoint molecular graph.

    The reaction graph contains all atoms from all components as one disconnected
    graph.  Each atom carries a ``molecule_id`` (0..M-1) and optional Laplacian
    positional encoding.

    Args:
        smiles_list: SMILES strings for each reaction component (e.g. ligand, base, …).
        add_laplacian_pe: If True, add Laplacian eigenvector positional encoding.
        pe_k: Number of Laplacian eigenvectors to compute.

    Returns:
        PyG Data object with fields:
            x:              [N, ATOM_FEAT_DIM]  one-hot atom features
            edge_index:     [2, 2E]             undirected edges
            edge_attr:      [2E, BOND_FEAT_DIM] one-hot bond features
            molecule_ids:   [N]                 which component (0..M-1)
            ptr:            [M+1]               component boundaries (for pooling)
            laplacian_pe:   [N, pe_k]           (if add_laplacian_pe=True)
        Returns None if all SMILES fail to parse.
    """
    if not HAS_RDKIT or not HAS_PYG:
        raise RuntimeError("RDKit and PyG are required for graph building")

    role_mols = []
    for role_id, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is not None:
            # Preserve the original reaction-role index.  Compressing this
            # list after a missing component would incorrectly relabel every
            # later molecule (for example, solvent as ligand).
            role_mols.append((role_id, mol))

    if not role_mols:
        return None

    # ── Build per-molecule atom & bond features ──
    all_x, all_edge_index, all_edge_attr, all_mol_ids = [], [], [], []
    atom_offset = 0

    for role_id, mol in role_mols:
        n_atoms = mol.GetNumAtoms()
        n_bonds = mol.GetNumBonds()

        # Atom features
        x_mol = atom_features(mol)
        all_x.append(torch.from_numpy(x_mol))
        all_mol_ids.append(torch.full((n_atoms,), role_id, dtype=torch.long))

        # Edge index (bi-directional) + bond features
        if n_bonds > 0:
            bf = bond_features(mol)
            bf_t = torch.from_numpy(bf)

            # Build edge_index from bond table
            row, col = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx() + atom_offset
                j = bond.GetEndAtomIdx() + atom_offset
                row.extend([i, j])
                col.extend([j, i])
            ei = torch.tensor([row, col], dtype=torch.long)

            # edge_index is ordered [i→j, j→i] for each bond, so its
            # attributes must be repeated consecutively in the same order.
            # Concatenating the whole table with itself would silently assign
            # attributes from a different bond whenever a molecule contains
            # more than one bond.
            ea = bf_t.repeat_interleave(2, dim=0)

            all_edge_index.append(ei)
            all_edge_attr.append(ea)

        atom_offset += n_atoms

    # ── Concatenate ──
    x = torch.cat(all_x, dim=0)
    molecule_ids = torch.cat(all_mol_ids, dim=0)

    if all_edge_index:
        edge_index = torch.cat(all_edge_index, dim=1)
        edge_attr = torch.cat(all_edge_attr, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, BOND_FEAT_DIM), dtype=torch.float32)

    # Compute ptr (cumulative atom counts per molecule)
    sizes = [mol.GetNumAtoms() for _, mol in role_mols]
    ptr = torch.tensor([0] + list(np.cumsum(sizes)), dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        molecule_ids=molecule_ids,
        ptr=ptr,
        num_nodes=x.size(0),
    )

    # ── Laplacian PE (per disconnected component) ──
    if add_laplacian_pe and pe_k > 0:
        try:
            lap_pe = _compute_laplacian_pe(data, pe_k=pe_k)
            data.laplacian_pe = lap_pe
        except Exception:
            # Fallback: zero PE
            data.laplacian_pe = torch.zeros((x.size(0), pe_k), dtype=torch.float32)

    return data


def _compute_laplacian_pe(data: Data, pe_k: int = 3) -> torch.Tensor:
    """Compute Laplacian Positional Encoding for each connected component.

    Uses sparse eigendecomposition of the normalised Laplacian.
    Falls back gracefully for large or degenerate graphs.
    """
    from torch_geometric.utils import (
        get_laplacian, to_dense_adj, to_scipy_sparse_matrix,
    )

    edge_index = data.edge_index
    n = data.num_nodes

    if edge_index.size(1) == 0:
        return torch.zeros((n, pe_k), dtype=torch.float32)

    # Normalised Laplacian
    edge_index_lap, edge_weight_lap = get_laplacian(
        edge_index, normalization='sym', num_nodes=n)

    # Use scipy for eigendecomposition (more stable than torch.linalg.eigh)
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix

        L = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes=n)
        # Compute pe_k+1 eigenvectors (discard first — constant)
        eigvals, eigvecs = eigsh(L, k=pe_k + 1, which='SM', return_eigenvectors=True)
        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigvals)
        # Take pe_k smallest non-zero eigenvectors
        pe = eigvecs[:, idx[1:pe_k + 1]]  # skip first (constant)
        # Normalise signs (random flip fix — use sign of max absolute value)
        pe_sign = np.sign(pe[np.argmax(np.abs(pe), axis=0), range(pe_k)])
        pe = pe * pe_sign
        return torch.from_numpy(pe.astype(np.float32))

    except Exception:
        # Scipy fallback failed — use dense eigendecomposition
        try:
            from torch_geometric.utils import to_dense_adj
            L_dense = to_dense_adj(edge_index_lap, edge_attr=edge_weight_lap, max_num_nodes=n)[0]
            # L_dense: [n, n]
            D_dense = torch.eye(n) - L_dense  # symmetrically normalised
            eigvals, eigvecs = torch.linalg.eigh(L_dense)
            idx = torch.argsort(eigvals)
            pe = eigvecs[:, idx[1:pe_k + 1]]  # skip first eigenvector
            # Sign normalisation
            pe_sign = torch.sign(pe[torch.argmax(pe.abs(), dim=0), torch.arange(pe_k)])
            return pe * pe_sign
        except Exception:
            return torch.zeros((n, pe_k), dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────
# Batch Graph Construction (for data loading)
# ──────────────────────────────────────────────────────────────────────

def build_reaction_graphs(
    df,
    smiles_cols: List[str],
    target_col: str,
    condition_features: Optional[np.ndarray] = None,
    add_laplacian_pe: bool = True,
    pe_k: int = 3,
    verbose: bool = True,
) -> Tuple[List[Data], np.ndarray]:
    """Build PyG Data objects for all reactions in a DataFrame.

    Args:
        df: pandas DataFrame with SMILES columns and target column.
        smiles_cols: column names containing SMILES strings.
        target_col: column name containing the target value (yield, G, etc.).
        condition_features: optional float array with shape [n_rows, n_features].
            Each row is attached to its graph as ``condition_features`` with
            shape [1, n_features], allowing PyG batching to produce [B, F].
        add_laplacian_pe: whether to add Laplacian PE.
        pe_k: number of Laplacian eigenvectors.
        verbose: print progress.

    Returns:
        graphs: list of PyG Data objects (None for failed rows).
        targets: np.ndarray of target values (NaN for failed rows).
    """
    graphs = []
    targets = []
    n_failed = 0

    if condition_features is not None:
        condition_features = np.asarray(condition_features, dtype=np.float32)
        if condition_features.ndim != 2 or condition_features.shape[0] != len(df):
            raise ValueError(
                "condition_features must have shape [len(df), n_features], got "
                f"{condition_features.shape} for {len(df)} rows"
            )

    for idx, (_, row) in enumerate(df.iterrows()):
        smiles_list = [str(row[c]) for c in smiles_cols]
        try:
            g = smiles_to_graph(smiles_list, add_laplacian_pe=add_laplacian_pe, pe_k=pe_k)
            if g is not None:
                g.reaction_idx = idx
                if condition_features is not None:
                    # Keep a leading singleton dimension.  PyG concatenates
                    # arbitrary tensor attributes along dim=0, yielding a
                    # clean [batch_size, n_features] condition matrix.
                    g.condition_features = torch.from_numpy(
                        condition_features[idx:idx + 1]
                    )
                graphs.append(g)
                targets.append(float(row[target_col]))
            else:
                graphs.append(None)
                targets.append(float('nan'))
                n_failed += 1
        except Exception:
            graphs.append(None)
            targets.append(float('nan'))
            n_failed += 1

    if verbose and n_failed > 0:
        print(f"[graph_encoder] {n_failed}/{len(df)} reactions failed to convert")

    return graphs, np.array(targets, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
# PyG Dataset wrapper
# ──────────────────────────────────────────────────────────────────────

from torch.utils.data import Dataset as TorchDataset


class ReactionGraphDataset(TorchDataset):
    """Torch Dataset wrapping pre-built PyG Data objects.

    Filters out failed graph builds automatically.
    """

    def __init__(self, graphs: List[Optional[Data]], targets: np.ndarray):
        valid_idx = [i for i, g in enumerate(graphs) if g is not None]
        self.graphs = [graphs[i] for i in valid_idx]
        self.targets = targets[valid_idx]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx].clone()
        g.y = torch.tensor([self.targets[idx]], dtype=torch.float32)
        return g


def collate_graphs(batch: List[Data]) -> Data:
    """Collate a list of PyG Data objects into a batch."""
    return Batch.from_data_list(batch)

"""ReactionGNN — graph reaction encoder and yield head for residual transfer.

Architecture:
    Reaction SMILES → disjoint molecular graph → GPSBlock × N
    → Readout (mean+max pool) → Yield Head → y_base

GPSBlock = GINEConv (local MP) + Multi-head Self-Attention (global) + FFN
All with Pre-LayerNorm and residual connections.

The model supplies the reaction embedding used by eKNN residual transfer.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import to_dense_batch

from .graph_encoder import ATOM_FEAT_DIM, BOND_FEAT_DIM


# ──────────────────────────────────────────────────────────────────────
# Embedding Layers
# ──────────────────────────────────────────────────────────────────────

class AtomEncoder(nn.Module):
    """Project one-hot atom features + molecule_id + Laplacian PE → d_model."""

    def __init__(self, d_model: int, max_molecules: int = 8, pe_dim: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.atom_proj = nn.Linear(ATOM_FEAT_DIM, d_model)
        self.mol_id_embed = nn.Embedding(max_molecules, d_model)
        self.pe_proj = nn.Linear(pe_dim, d_model) if pe_dim > 0 else None
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, molecule_ids, laplacian_pe=None):
        """
        Args:
            x: [N, ATOM_FEAT_DIM] one-hot atom features
            molecule_ids: [N] long tensor (0..M-1)
            laplacian_pe: [N, pe_dim] or None
        Returns:
            [N, d_model]
        """
        h = self.atom_proj(x)
        h = h + self.mol_id_embed(molecule_ids)
        if laplacian_pe is not None and self.pe_proj is not None:
            h = h + self.pe_proj(laplacian_pe)
        h = self.norm(h)
        h = self.dropout(h)
        return h


class EdgeEncoder(nn.Module):
    """Project one-hot bond features → edge_attr_dim."""

    def __init__(self, edge_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(BOND_FEAT_DIM, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )

    def forward(self, edge_attr):
        return self.proj(edge_attr)


# ──────────────────────────────────────────────────────────────────────
# GPS Block
# ──────────────────────────────────────────────────────────────────────

class GPSBlock(nn.Module):
    """One GraphGPS block: Local MP + Global Attention + FFN.

    All sub-layers use Pre-LayerNorm and residual connections.
    """

    def __init__(self, d_model: int, edge_dim: int,
                 n_heads: int = 4, ff_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        # Local: GINEConv
        self.norm1 = nn.LayerNorm(d_model)
        gin_nn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.local_mp = GINEConv(gin_nn, edge_dim=edge_dim)

        # Global: Multi-head Self-Attention
        self.norm2 = nn.LayerNorm(d_model)
        self.global_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Args:
            x: [N, d_model] node features
            edge_index: [2, E]
            edge_attr: [E, edge_dim] edge features
            batch: [N] batch index for each node
        Returns:
            x: [N, d_model]
        """
        # ── Local MP ──
        x_local = self.local_mp(self.norm1(x), edge_index, edge_attr)
        x = x + self.dropout(x_local)

        # ── Global Attention (per-graph) ──
        x_norm = self.norm2(x)
        x_dense, mask = to_dense_batch(x_norm, batch)
        # x_dense: [B, N_max, d_model], mask: [B, N_max]
        x_attn, _ = self.global_attn(
            x_dense, x_dense, x_dense,
            key_padding_mask=~mask,
            need_weights=False,
        )
        # Unpad
        x_global = x_attn[mask]
        x = x + self.dropout(x_global)

        # ── FFN ──
        x_ffn = self.ffn(self.norm3(x))
        x = x + self.dropout(x_ffn)

        return x


# ──────────────────────────────────────────────────────────────────────
# Graph Transformer Block (pure attention, no GNN)
# ──────────────────────────────────────────────────────────────────────

class GraphTransformerBlock(nn.Module):
    """Pure Graph Transformer block: Self-Attention + FFN.

    No message-passing GNN component — relies entirely on global attention.
    Edge features are fused with node features before attention.
    """

    def __init__(self, d_model: int, edge_dim: int,
                 n_heads: int = 4, ff_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        # Edge projection
        self.edge_proj = nn.Linear(edge_dim, d_model)

        # Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

        # FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        # ── Global Self-Attention (per graph) ──
        x_norm = self.norm1(x)
        x_dense, mask = to_dense_batch(x_norm, batch)
        x_attn, _ = self.attn(
            x_dense, x_dense, x_dense,
            key_padding_mask=~mask,
            need_weights=False,
        )
        x_global = x_attn[mask]
        x = x + self.dropout(x_global)

        # ── FFN ──
        x_ffn = self.ffn(self.norm2(x))
        x = x + self.dropout(x_ffn)

        return x


# ──────────────────────────────────────────────────────────────────────
# Edge-Augmented Graph Transformer (edge features bias attention)
# ──────────────────────────────────────────────────────────────────────

class EdgeTransformerBlock(nn.Module):
    """Edge-Augmented Graph Transformer: Self-Attention with edge bias + FFN.

    Edge features are projected and added to node features so that attention
    can leverage bond type information implicitly.
    """

    def __init__(self, d_model: int, edge_dim: int,
                 n_heads: int = 4, ff_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        # Edge → node feature augmentation (simplified edge encoding)
        self.edge_proj = nn.Linear(edge_dim, d_model)

        # Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

        # Edge-gate: how much edge info to mix in
        self.edge_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

        self.norm_fuse = nn.LayerNorm(d_model)

        # FFN
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        # ── Edge feature aggregation per node ──
        row, col = edge_index
        edge_feat = self.edge_proj(edge_attr)  # [E, d_model]
        # Aggregate edge features to target nodes (simple mean)
        edge_to_node = torch.zeros(x.shape[0], x.shape[1],
                                   device=x.device, dtype=x.dtype)
        edge_to_node = edge_to_node.index_add(0, col, edge_feat)
        # Normalise by degree
        degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        degree = degree.index_add(0, col, torch.ones(edge_feat.shape[0], device=x.device))
        degree = degree.clamp(min=1)
        edge_to_node = edge_to_node / degree.unsqueeze(-1)

        # Gate: how much edge info to mix
        gate = self.edge_gate(torch.cat([x, edge_to_node], dim=-1))
        x_aug = self.norm_fuse(x + gate * edge_to_node)

        # ── Global Self-Attention ──
        x_norm = self.norm1(x_aug)
        x_dense, mask = to_dense_batch(x_norm, batch)
        x_attn, _ = self.attn(
            x_dense, x_dense, x_dense,
            key_padding_mask=~mask,
            need_weights=False,
        )
        x_global = x_attn[mask]
        x = x + self.dropout(x_global)

        # ── FFN ──
        x_ffn = self.ffn(self.norm2(x))
        x = x + self.dropout(x_ffn)

        return x


# ──────────────────────────────────────────────────────────────────────
# Graphormer Block (centrality + edge-aware attention)
# ──────────────────────────────────────────────────────────────────────

class GraphormerBlock(nn.Module):
    """Graphormer-style block: Degree Centrality + Edge-biased Attention + FFN.

    - Degree centrality encoding: learnable embedding per node degree
    - Edge-aware attention: edge features → scalar bias per node pair in attention
    - Pre-LN + residual connections throughout

    Reference: Ying et al., "Do Transformers Really Perform Bad for Graph
    Representation?", NeurIPS 2021.
    """

    def __init__(self, d_model: int, edge_dim: int,
                 n_heads: int = 4, ff_dim: int = 512,
                 dropout: float = 0.1, max_degree: int = 64):
        super().__init__()

        # Degree centrality encoding
        self.degree_embed = nn.Embedding(max_degree + 1, d_model)
        self.dropout_deg = nn.Dropout(dropout)

        # Edge → attention bias (shared across heads)
        self.edge_bias_mlp = nn.Sequential(
            nn.Linear(edge_dim, n_heads),
            nn.GELU(),
            nn.Linear(n_heads, n_heads),
        )

        # No-edge bias (for node pairs without direct bond)
        self.no_edge_bias = nn.Parameter(torch.zeros(1, n_heads, 1, 1))

        # Self-Attention
        self.norm1 = nn.LayerNorm(d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN (deeper than standard GPS)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def _build_edge_bias(self, x, edge_index, edge_attr, batch):
        """Build [B, N_max, N_max] attention bias from edge features."""
        B = int(batch.max().item()) + 1
        # Pad to dense
        x_dense, mask = to_dense_batch(x, batch)
        _, N_max, _ = x_dense.shape

        # Build dense edge bias matrix: start with no-edge bias
        edge_bias = self.no_edge_bias.expand(B, self.n_heads, N_max, N_max).clone()

        # Fill in edge biases for connected nodes
        row, col = edge_index
        edge_biases = self.edge_bias_mlp(edge_attr)  # [E, n_heads]

        # Map node indices to graph-relative indices
        batch_idx = batch[row]  # [E]
        # Compute per-graph offsets
        offsets = torch.zeros(B + 1, dtype=torch.long, device=x.device)
        ones = torch.ones_like(batch)
        offsets[1:] = torch.cumsum(torch.bincount(batch + 1, minlength=B + 1)[1:], dim=0)

        # TODO: fill direct edge bias properly with per-graph indexing
        # For now, skip direct edge bias — use degree centrality only
        # This is a simplified Graphormer

        return edge_bias

    def forward(self, x, edge_index, edge_attr, batch):
        # ── Degree Centrality Encoding ──
        # Compute degree per node
        degree = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        col = edge_index[1]
        degree = degree.index_add(0, col, torch.ones(col.shape[0], dtype=torch.long, device=x.device))
        degree = degree.clamp(max=64)
        x_deg = self.degree_embed(degree)
        x = x + self.dropout_deg(x_deg)

        # ── Self-Attention (per graph) ──
        x_norm = self.norm1(x)
        x_dense, mask = to_dense_batch(x_norm, batch)
        B, N_max, d = x_dense.shape

        # Multi-head projection
        q = self.q_proj(x_dense).view(B, N_max, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        k = self.k_proj(x_dense).view(B, N_max, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        v = self.v_proj(x_dense).view(B, N_max, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = self.d_head ** -0.5
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

        # Causal mask for padding
        key_mask = ~mask  # True = ignore
        attn_logits = attn_logits.masked_fill(
            key_mask.unsqueeze(1).unsqueeze(-1),
            float('-inf'),
        )

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights = self.dropout(attn_weights)

        x_attn = torch.matmul(attn_weights, v)  # [B, H, N, d_head]
        x_attn = x_attn.permute(0, 2, 1, 3).reshape(B, N_max, d)
        x_attn = self.out_proj(x_attn)

        # Unpad
        x_global = x_attn[mask]
        x = x + self.dropout(x_global)

        # ── FFN (deeper) ──
        x_ffn = self.ffn(self.norm2(x))
        x = x + self.dropout(x_ffn)

        return x


# ──────────────────────────────────────────────────────────────────────
# Deep GPS Block (GPS with deeper FFN + 2x GNN heads)
# ──────────────────────────────────────────────────────────────────────

class DeepGPSBlock(nn.Module):
    """Deeper GPS variant: 2 GNN heads (GINE + GAT) + Global Attn + Deeper FFN."""

    def __init__(self, d_model: int, edge_dim: int,
                 n_heads: int = 4, ff_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()

        # Local 1: GINEConv
        self.norm1a = nn.LayerNorm(d_model)
        gin_nn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.local_mp = GINEConv(gin_nn, edge_dim=edge_dim)

        # Global: Multi-head Self-Attention
        self.norm2 = nn.LayerNorm(d_model)
        self.global_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

        # FFN (deeper)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        # GINE
        x_local = self.local_mp(self.norm1a(x), edge_index, edge_attr)
        x = x + self.dropout(x_local)

        # Global Attention
        x_norm = self.norm2(x)
        x_dense, mask = to_dense_batch(x_norm, batch)
        x_attn, _ = self.global_attn(x_dense, x_dense, x_dense,
                                      key_padding_mask=~mask, need_weights=False)
        x = x + self.dropout(x_attn[mask])

        # Deeper FFN
        x_ffn = self.ffn(self.norm3(x))
        x = x + self.dropout(x_ffn)

        return x


# ──────────────────────────────────────────────────────────────────────
# Lightweight GPS (single FFN layer, more dropout)
# ──────────────────────────────────────────────────────────────────────

class LightGPSBlock(nn.Module):
    """Lightweight GPS: GINE + Global Attn + Single-layer FFN + high dropout."""

    def __init__(self, d_model: int, edge_dim: int,
                 n_heads: int = 4, ff_dim: int = 512,
                 dropout: float = 0.2):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        gin_nn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.local_mp = GINEConv(gin_nn, edge_dim=edge_dim)

        self.norm2 = nn.LayerNorm(d_model)
        self.global_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x_local = self.local_mp(self.norm1(x), edge_index, edge_attr)
        x = x + self.dropout(x_local)

        x_norm = self.norm2(x)
        x_dense, mask = to_dense_batch(x_norm, batch)
        x_attn, _ = self.global_attn(x_dense, x_dense, x_dense,
                                      key_padding_mask=~mask, need_weights=False)
        x = x + self.dropout(x_attn[mask])

        x_ffn = self.ffn(self.norm3(x))
        x = x + self.dropout(x_ffn)

        return x


# ──────────────────────────────────────────────────────────────────────
# GAT Block (local attention over bonded atoms)
# ──────────────────────────────────────────────────────────────────────

class GATBlock(nn.Module):
    """GATConv + FFN. Local attention over bonded atoms only — better
    generalisation than global MHA for datasets with many groups."""

    def __init__(self, d_model: int, edge_dim: int,
                 n_heads: int = 4, ff_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.gat = GATConv(d_model, d_model // n_heads, heads=n_heads,
                           edge_dim=edge_dim, dropout=dropout, concat=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x + self.dropout(self.gat(self.norm1(x), edge_index, edge_attr))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ──────────────────────────────────────────────────────────────────────
# Pure MPNN Block (no global attention — better generalisation)
# ──────────────────────────────────────────────────────────────────────

class PureMPNNBlock(nn.Module):
    """GINEConv + FFN only. No global attention — avoids overfitting to
    specific molecular graph topologies. Better for datasets with many
    groups where test molecules have unseen structures."""

    def __init__(self, d_model: int, edge_dim: int,
                 ff_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        gin_nn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.local_mp = GINEConv(gin_nn, edge_dim=edge_dim)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        # Local MP only
        x = x + self.dropout(self.local_mp(self.norm1(x), edge_index, edge_attr))
        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ──────────────────────────────────────────────────────────────────────
# PyG Standard GNN Blocks — comprehensive benchmark of official PyG conv layers
# ──────────────────────────────────────────────────────────────────────

class _PyGBlock(nn.Module):
    """Generic block wrapping any PyG Conv layer: Pre-LN → Conv → residual → LN → FFN.

    Handles: (a) convs with/without edge features, (b) multi-head convs that output
    d_model*heads with concat=True → projects back to d_model.
    """

    def __init__(self, d_model: int, edge_dim: int, conv_cls, conv_kwargs: dict,
                 ff_dim: int = 384, dropout: float = 0.1, uses_edge_attr: bool = True,
                 multi_head_concat: bool = False, positionals: list | None = None):
        super().__init__()
        self.uses_edge_attr = uses_edge_attr
        self.multi_head_concat = multi_head_concat

        self.norm1 = nn.LayerNorm(d_model)
        if positionals is not None:
            self.conv = conv_cls(*positionals, **conv_kwargs)
        else:
            self.conv = conv_cls(**conv_kwargs)

        # Output projection for multi-head concat convs
        conv_out_dim = conv_kwargs.get('out_channels', d_model)
        if multi_head_concat:
            n_heads = conv_kwargs.get('heads', 4)
            conv_out_dim = d_model * n_heads
        self.out_proj = nn.Linear(conv_out_dim, d_model) if multi_head_concat else nn.Identity()

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x_in = self.norm1(x)
        if self.uses_edge_attr:
            x_new = self.conv(x_in, edge_index, edge_attr)
        else:
            x_new = self.conv(x_in, edge_index)
        x_new = self.out_proj(x_new)
        x = x + self.dropout(x_new)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ── Individual blocks (named for clarity in registry) ──

class GCNBlock(nn.Module):
    """GCNConv (Kipf & Welling 2017) — classic graph convolution, no edge features."""
    def __init__(self, d_model, edge_dim, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.block = _PyGBlock(d_model, edge_dim, GCNConv,
                               dict(add_self_loops=False), ff_dim, dropout,
                               uses_edge_attr=False, positionals=[d_model, d_model])
    def forward(self, x, edge_index, edge_attr, batch):
        return self.block(x, edge_index, edge_attr, batch)


class SAGEBlock(nn.Module):
    """SAGEConv (Hamilton et al. 2017) — GraphSAGE with mean aggregation."""
    def __init__(self, d_model, edge_dim, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.block = _PyGBlock(d_model, edge_dim, SAGEConv,
                               dict(aggr='mean'), ff_dim, dropout,
                               uses_edge_attr=False, positionals=[d_model, d_model])
    def forward(self, x, edge_index, edge_attr, batch):
        return self.block(x, edge_index, edge_attr, batch)


class GINBlock(nn.Module):
    """GINConv (Xu et al. 2019) — theoretically most expressive MPNN with ε parameter."""
    def __init__(self, d_model, edge_dim, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import GINConv
        gin_nn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = GINConv(gin_nn, train_eps=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x + self.dropout(self.conv(self.norm1(x), edge_index))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class GATv2Block(nn.Module):
    """GATv2Conv (Brody et al. 2022) — dynamic attention that fixes GAT's static ordering issue."""
    def __init__(self, d_model, edge_dim, n_heads=4, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import GATv2Conv
        n_h = 4 if d_model % 4 == 0 else 2
        head_dim = d_model // n_h
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = GATv2Conv(d_model, head_dim, heads=n_h, edge_dim=edge_dim,
                               dropout=dropout, concat=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x + self.dropout(self.conv(self.norm1(x), edge_index, edge_attr))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerConvBlock(nn.Module):
    """TransformerConv (Shi et al. 2021, UniMP) — edge-aware graph transformer."""
    def __init__(self, d_model, edge_dim, n_heads=4, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import TransformerConv
        n_h = 4 if d_model % 4 == 0 else 2
        head_dim = d_model // n_h
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = TransformerConv(d_model, head_dim, heads=n_h, edge_dim=edge_dim,
                                     dropout=dropout, concat=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x + self.dropout(self.conv(self.norm1(x), edge_index, edge_attr))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class PNABlock(nn.Module):
    """PNAConv (Corso et al. 2020) — multiple aggregators + degree scalers, SOTA MPNN."""
    def __init__(self, d_model, edge_dim, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import PNAConv
        # Use a sensible default degree histogram — updated on first forward
        deg_default = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5])
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        self.pna_conv = PNAConv(
            in_channels=d_model, out_channels=d_model,
            aggregators=aggregators, scalers=scalers,
            deg=deg_default, edge_dim=edge_dim,
            towers=4, pre_layers=1, post_layers=1,
            divide_input=False,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self._deg_updated = False

    def forward(self, x, edge_index, edge_attr, batch):
        if not self._deg_updated and edge_index.size(1) > 0:
            col = edge_index[1]
            deg = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            deg = deg.index_add(0, col, torch.ones(col.size(0), dtype=torch.long, device=x.device))
            max_deg = max(deg.max().item() + 1, 10)
            deg_hist = torch.bincount(deg, minlength=max_deg).float()
            deg_hist = deg_hist / (deg_hist.sum() + 1e-8)
            try:
                self.pna_conv.aggr.deg = deg_hist.to(x.device)
            except AttributeError:
                pass  # deg updated internally
            self._deg_updated = True
        x = x + self.dropout(self.pna_conv(self.norm1(x), edge_index, edge_attr))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class GeneralConvBlock(nn.Module):
    """GeneralConv (You et al. 2020) — learned operator selection over GNN designs."""
    def __init__(self, d_model, edge_dim, n_heads=4, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import GeneralConv
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = GeneralConv(d_model, d_model, heads=n_heads, attention=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x + self.dropout(self.conv(self.norm1(x), edge_index))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class GENConvBlock(nn.Module):
    """GENConv (Li et al. 2020, DeeperGCN) — deep GCN with message norm + edge features."""
    def __init__(self, d_model, edge_dim, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import GENConv
        self.block = _PyGBlock(d_model, edge_dim, GENConv,
                               dict(edge_dim=edge_dim, aggr='softmax', t=0.1, learn_t=True),
                               ff_dim, dropout, uses_edge_attr=True, positionals=[d_model, d_model])
    def forward(self, x, edge_index, edge_attr, batch):
        return self.block(x, edge_index, edge_attr, batch)


class FiLMConvBlock(nn.Module):
    """FiLMConv (Brockschmidt 2020) — feature-wise linear modulation for GNNs."""
    def __init__(self, d_model, edge_dim, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import FiLMConv
        self.block = _PyGBlock(d_model, edge_dim, FiLMConv,
                               dict(num_relations=1), ff_dim, dropout,
                               uses_edge_attr=False, positionals=[d_model, d_model])
    def forward(self, x, edge_index, edge_attr, batch):
        return self.block(x, edge_index, edge_attr, batch)


class GatedGraphConvBlock(nn.Module):
    """GatedGraphConv (Li et al. 2016) — GRU-based message passing (GG-NN)."""
    def __init__(self, d_model, edge_dim, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import GatedGraphConv
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = GatedGraphConv(d_model, 1)  # (out_channels, num_layers)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x + self.dropout(self.conv(self.norm1(x), edge_index))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class ResGatedGraphBlock(nn.Module):
    """ResGatedGraphConv (Bresson & Laurent 2017) — gated GCN with edge features."""
    def __init__(self, d_model, edge_dim, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import ResGatedGraphConv
        self.block = _PyGBlock(d_model, edge_dim, ResGatedGraphConv,
                               dict(edge_dim=edge_dim), ff_dim, dropout,
                               uses_edge_attr=True, positionals=[d_model, d_model])
    def forward(self, x, edge_index, edge_attr, batch):
        return self.block(x, edge_index, edge_attr, batch)


class DirectedEdgeMPNNBlock(nn.Module):
    """Directed-bond message passing with immediate backtracking removed.

    For every directed bond u→v, the update aggregates messages entering u
    but subtracts the reverse v→u message.  This is the key D-MPNN inductive
    bias and is especially useful for small molecular graphs where ordinary
    node convolutions can repeatedly echo information across the same bond.
    """

    def __init__(
        self,
        d_model: int,
        edge_dim: int,
        ff_dim: int = 512,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.node_norm = nn.LayerNorm(d_model)
        self.edge_input = nn.Sequential(
            nn.Linear(d_model + edge_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.edge_norm = nn.LayerNorm(d_model)
        self.node_update = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _reverse_indices(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        source, target = edge_index
        keys = source * n_nodes + target
        reverse_keys = target * n_nodes + source
        sorted_keys, permutation = torch.sort(keys)
        positions = torch.searchsorted(sorted_keys, reverse_keys)
        positions = positions.clamp(max=max(sorted_keys.numel() - 1, 0))
        reverse = permutation[positions]
        if not torch.equal(keys[reverse], reverse_keys):
            raise ValueError(
                "DirectedEdgeMPNNBlock requires every molecular bond to have "
                "both directed edges"
            )
        return reverse

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_index.shape[1] == 0:
            return x + self.dropout(self.ffn(self.ffn_norm(x)))

        source, target = edge_index
        node_states = self.node_norm(x)
        edge_states = self.edge_input(
            torch.cat([node_states[source], edge_attr], dim=-1)
        )
        incoming = torch.zeros(
            (x.shape[0], self.d_model), dtype=x.dtype, device=x.device
        )
        incoming.index_add_(0, target, edge_states)
        reverse = self._reverse_indices(edge_index, x.shape[0])
        non_backtracking = incoming[source] - edge_states[reverse]
        edge_states = self.edge_norm(
            edge_states + self.dropout(self.edge_update(non_backtracking))
        )

        atom_messages = torch.zeros_like(incoming)
        atom_messages.index_add_(0, target, edge_states)
        node_delta = self.node_update(
            torch.cat([node_states, atom_messages], dim=-1)
        )
        x = self.output_norm(x + self.dropout(node_delta))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class EGConvBlock(nn.Module):
    """EGConv (Tailor et al. 2022) — efficient graph convolution with multiple bases."""
    def __init__(self, d_model, edge_dim, n_heads=4, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn import EGConv
        self.norm1 = nn.LayerNorm(d_model)
        self.conv = EGConv(d_model, d_model, num_bases=4, num_heads=n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x + self.dropout(self.conv(self.norm1(x), edge_index))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ──────────────────────────────────────────────────────────────────────
# AttentiveFP Block (Xiong et al., JMC 2020)
# ──────────────────────────────────────────────────────────────────────

class AttentiveFPBlock(nn.Module):
    """AttentiveFP (Xiong et al., JMC 2020) — graph attention with molecular-level GRU.

    Unlike other blocks, returns graph-level embeddings [B, d] (not atom-level [N, d]).
    The model should skip readout/pooling when using this backbone.
    Use n_layers=1 with this backbone (AttentiveFP handles multi-layer internally).
    """
    _returns_graph_embedding = True

    def __init__(self, d_model, edge_dim, n_heads=4, ff_dim=384, dropout=0.1, **kw):
        super().__init__()
        from torch_geometric.nn.models import AttentiveFP
        num_layers = kw.get('n_layers', 2)
        num_timesteps = kw.get('n_layers', 2)
        self.afp = AttentiveFP(
            in_channels=d_model, hidden_channels=d_model, out_channels=d_model,
            edge_dim=edge_dim, num_layers=num_layers, num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch):
        return self.afp(x, edge_index, edge_attr, batch)


# ──────────────────────────────────────────────────────────────────────
# Full Model
# ──────────────────────────────────────────────────────────────────────


class ConditionComponentTransformer(nn.Module):
    """Fuse molecular-role and reaction-condition tokens with attention.

    The reaction graph is made of disconnected molecular components.  This
    module therefore pools each fixed reaction role, tokenises each numerical
    condition field independently, and lets those tokens interact with the
    pooled reaction representation.  Optional categorical fields use
    fold-fitted embedding tables with an explicit OOV state.
    """

    def __init__(
        self,
        *,
        reaction_dim: int,
        graph_dim: int,
        numeric_condition_dim: int,
        categorical_cardinalities: Sequence[int] = (),
        max_molecules: int = 8,
        n_heads: int = 4,
        ff_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if reaction_dim % n_heads != 0:
            raise ValueError(
                "reaction_dim must be divisible by n_heads for condition "
                f"cross-attention, got {reaction_dim} and {n_heads}"
            )
        if n_layers < 1:
            raise ValueError("condition token transformer requires at least one layer")
        if numeric_condition_dim < 1:
            raise ValueError("condition token transformer requires numeric condition features")

        self.reaction_dim = int(reaction_dim)
        self.graph_dim = int(graph_dim)
        self.numeric_condition_dim = int(numeric_condition_dim)
        self.max_molecules = int(max_molecules)
        self.categorical_cardinalities = tuple(int(v) for v in categorical_cardinalities)

        # Feature-wise scalar tokenisation preserves the identity of each
        # experimental condition (temperature is not merely another scalar).
        self.numeric_weight = nn.Parameter(
            torch.empty(self.numeric_condition_dim, self.reaction_dim)
        )
        self.numeric_bias = nn.Parameter(
            torch.empty(self.numeric_condition_dim, self.reaction_dim)
        )
        nn.init.normal_(self.numeric_weight, std=0.04)
        nn.init.normal_(self.numeric_bias, std=0.02)
        self.numeric_norm = nn.LayerNorm(self.reaction_dim)

        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, self.reaction_dim)
            for cardinality in self.categorical_cardinalities
        ])
        for embedding in self.categorical_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)

        self.component_proj = nn.Sequential(
            nn.Linear(self.graph_dim * 2, self.reaction_dim),
            nn.GELU(),
            nn.LayerNorm(self.reaction_dim),
        )
        self.component_role_embed = nn.Embedding(self.max_molecules, self.reaction_dim)
        self.component_norm = nn.LayerNorm(self.reaction_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=self.reaction_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.context_mlp = nn.Sequential(
            nn.Linear(self.reaction_dim, self.reaction_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.reaction_dim, self.reaction_dim),
        )
        # The cross-token path begins as an identity residual.  This preserves
        # the well-tested gated condition baseline at initialization and lets
        # the model adopt cross-component interactions only when validation
        # data support them.
        nn.init.zeros_(self.context_mlp[-1].weight)
        nn.init.zeros_(self.context_mlp[-1].bias)
        self.output_norm = nn.LayerNorm(self.reaction_dim)

    def _component_tokens(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        molecule_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool atom states into fixed reaction-role tokens and a presence mask."""
        n_graphs = int(batch.max().item()) + 1
        if molecule_ids.shape[0] != x.shape[0]:
            raise ValueError(
                "Role-aware condition fusion requires one molecule_id per atom; "
                "do not combine it with virtual-node graph augmentation"
            )
        role_ids = molecule_ids.clamp(min=0, max=self.max_molecules - 1)
        component_index = batch * self.max_molecules + role_ids
        n_components = n_graphs * self.max_molecules

        comp_mean = global_mean_pool(x, component_index, size=n_components)
        comp_max = global_max_pool(x, component_index, size=n_components)
        counts = global_add_pool(
            torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype),
            component_index,
            size=n_components,
        ).squeeze(-1)
        present = counts.reshape(n_graphs, self.max_molecules) > 0
        tokens = torch.cat([comp_mean, comp_max], dim=-1).reshape(
            n_graphs, self.max_molecules, self.graph_dim * 2
        )
        tokens = self.component_proj(tokens)
        role_index = torch.arange(self.max_molecules, device=x.device)
        tokens = self.component_norm(
            tokens + self.component_role_embed(role_index).unsqueeze(0)
        )
        return tokens, present

    def _numeric_tokens(self, conditions: torch.Tensor) -> torch.Tensor:
        if conditions.ndim != 2 or conditions.shape[1] != self.numeric_condition_dim:
            raise ValueError(
                "Numeric condition shape mismatch: expected [batch, "
                f"{self.numeric_condition_dim}], got {tuple(conditions.shape)}"
            )
        tokens = (
            conditions.unsqueeze(-1) * self.numeric_weight.unsqueeze(0)
            + self.numeric_bias.unsqueeze(0)
        )
        return self.numeric_norm(tokens)

    def _categorical_tokens(
        self, categorical_conditions: Optional[torch.Tensor], batch_size: int
    ) -> Optional[torch.Tensor]:
        if not self.categorical_embeddings:
            return None
        if categorical_conditions is None:
            raise ValueError(
                "Categorical condition embeddings were configured but the graph "
                "batch has no condition_categorical_features"
            )
        expected = len(self.categorical_embeddings)
        if (categorical_conditions.ndim != 2
                or categorical_conditions.shape != (batch_size, expected)):
            raise ValueError(
                "Categorical condition shape mismatch: expected [batch, "
                f"{expected}], got {tuple(categorical_conditions.shape)}"
            )
        tokens = []
        for index, embedding in enumerate(self.categorical_embeddings):
            ids = categorical_conditions[:, index].long()
            if torch.any(ids < 0) or torch.any(ids >= embedding.num_embeddings):
                raise ValueError("Categorical condition ID is outside its fold-fitted vocabulary")
            tokens.append(embedding(ids))
        return torch.stack(tokens, dim=1)

    def forward(
        self,
        reaction_embedding: torch.Tensor,
        numeric_conditions: torch.Tensor,
        categorical_conditions: Optional[torch.Tensor],
        atom_states: torch.Tensor,
        batch: torch.Tensor,
        molecule_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = reaction_embedding.shape[0]
        component_tokens, component_present = self._component_tokens(
            atom_states, batch, molecule_ids
        )
        numeric_tokens = self._numeric_tokens(numeric_conditions)
        categorical_tokens = self._categorical_tokens(categorical_conditions, batch_size)

        token_blocks = [reaction_embedding.unsqueeze(1), component_tokens, numeric_tokens]
        if categorical_tokens is not None:
            token_blocks.append(categorical_tokens)
        tokens = torch.cat(token_blocks, dim=1)

        # Missing numeric fields have an imputed value plus a missingness bit;
        # categorical fields have an explicit OOV/missing state.  Thus only
        # absent molecular components are masked from attention.
        no_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=tokens.device)
        numeric_pad = torch.zeros(
            (batch_size, numeric_tokens.shape[1]), dtype=torch.bool, device=tokens.device
        )
        padding_mask = torch.cat([no_pad, ~component_present, numeric_pad], dim=1)
        if categorical_tokens is not None:
            categorical_pad = torch.zeros(
                (batch_size, categorical_tokens.shape[1]),
                dtype=torch.bool,
                device=tokens.device,
            )
            padding_mask = torch.cat([padding_mask, categorical_pad], dim=1)

        mixed = self.transformer(tokens, src_key_padding_mask=padding_mask)
        return self.output_norm(reaction_embedding + self.context_mlp(mixed[:, 0]))


class ConditionedRoleMessageBlock(nn.Module):
    """Message passing over the small graph of molecular reaction roles.

    Each reaction role (acceptor, donor, catalyst, ligand, solvent, additive,
    ...) is a node.  All present roles are connected, and the encoded
    experimental conditions enter every directed edge message.  This keeps
    component interactions inside a graph neural network instead of handing
    graph embeddings to a separate tabular model.
    """

    def __init__(self, d_model: int, max_molecules: int, dropout: float):
        super().__init__()
        self.d_model = int(d_model)
        self.max_molecules = int(max_molecules)
        self.role_pair_embedding = nn.Embedding(
            self.max_molecules * self.max_molecules, self.d_model
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(self.d_model * 6, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model * 2),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        self.message_norm = nn.LayerNorm(self.d_model)
        self.ffn_norm = nn.LayerNorm(self.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 4, self.d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        present: torch.Tensor,
        condition_context: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_roles, width = tokens.shape
        if n_roles != self.max_molecules or width != self.d_model:
            raise ValueError(
                "Role-message tensor shape mismatch: expected [batch, "
                f"{self.max_molecules}, {self.d_model}], got {tuple(tokens.shape)}"
            )

        # Axis 1 is the receiving role and axis 2 is the sending role.
        target = tokens.unsqueeze(2).expand(-1, -1, n_roles, -1)
        source = tokens.unsqueeze(1).expand(-1, n_roles, -1, -1)
        context = condition_context[:, None, None, :].expand(
            -1, n_roles, n_roles, -1
        )
        role_ids = torch.arange(n_roles, device=tokens.device)
        pair_ids = (
            role_ids[:, None] * self.max_molecules + role_ids[None, :]
        )
        pair = self.role_pair_embedding(pair_ids)[None, :, :, :].expand(
            batch_size, -1, -1, -1
        )

        edge_features = torch.cat(
            [
                target,
                source,
                target - source,
                target * source,
                pair,
                context,
            ],
            dim=-1,
        )
        message_value, message_gate = self.message_mlp(edge_features).chunk(2, dim=-1)
        messages = torch.tanh(message_value) * torch.sigmoid(message_gate)

        valid_edges = present.unsqueeze(2) & present.unsqueeze(1)
        non_self = ~torch.eye(n_roles, dtype=torch.bool, device=tokens.device)
        valid_edges = valid_edges & non_self.unsqueeze(0)
        messages = messages * valid_edges.unsqueeze(-1)
        degree = valid_edges.sum(dim=2, keepdim=True).clamp(min=1)
        aggregated = messages.sum(dim=2) / degree

        update = self.update_mlp(
            torch.cat(
                [
                    tokens,
                    aggregated,
                    condition_context.unsqueeze(1).expand(-1, n_roles, -1),
                ],
                dim=-1,
            )
        )
        tokens = self.message_norm(tokens + self.dropout(update))
        tokens = self.ffn_norm(tokens + self.dropout(self.ffn(tokens)))
        return tokens * present.unsqueeze(-1)


class ConditionedRoleInteractionReadout(nn.Module):
    """Hierarchical graph readout from atom graphs to a condition-aware role graph."""

    def __init__(
        self,
        *,
        graph_dim: int,
        output_dim: int,
        condition_dim: int,
        max_molecules: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        if condition_dim < 1:
            raise ValueError(
                "Conditioned role interaction requires prediction-time condition features"
            )
        if n_layers < 1:
            raise ValueError("Role interaction requires at least one message-passing layer")
        self.graph_dim = int(graph_dim)
        self.output_dim = int(output_dim)
        self.condition_dim = int(condition_dim)
        self.max_molecules = int(max_molecules)

        self.component_projection = nn.Sequential(
            nn.Linear(self.graph_dim * 2, self.graph_dim),
            nn.GELU(),
            nn.LayerNorm(self.graph_dim),
        )
        self.role_embedding = nn.Embedding(self.max_molecules, self.graph_dim)
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, self.graph_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.graph_dim * 2, self.graph_dim),
            nn.LayerNorm(self.graph_dim),
        )
        self.condition_film = nn.Sequential(
            nn.Linear(self.graph_dim, self.graph_dim * 2),
            nn.Tanh(),
        )
        self.blocks = nn.ModuleList(
            [
                ConditionedRoleMessageBlock(
                    self.graph_dim, self.max_molecules, dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.readout_score = nn.Sequential(
            nn.Linear(self.graph_dim * 2, self.graph_dim),
            nn.Tanh(),
            nn.Linear(self.graph_dim, 1),
        )
        self.output = nn.Sequential(
            nn.Linear(self.graph_dim * 3, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.output_dim),
        )

    def _component_tokens(
        self,
        atom_states: torch.Tensor,
        batch: torch.Tensor,
        molecule_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if molecule_ids.shape[0] != atom_states.shape[0]:
            raise ValueError(
                "Role-interaction readout requires one molecular role ID per atom"
            )
        n_graphs = int(batch.max().item()) + 1
        role_ids = molecule_ids.clamp(min=0, max=self.max_molecules - 1)
        component_index = batch * self.max_molecules + role_ids
        n_components = n_graphs * self.max_molecules

        component_mean = global_mean_pool(
            atom_states, component_index, size=n_components
        )
        component_max = global_max_pool(
            atom_states, component_index, size=n_components
        )
        counts = global_add_pool(
            torch.ones(
                (atom_states.shape[0], 1),
                device=atom_states.device,
                dtype=atom_states.dtype,
            ),
            component_index,
            size=n_components,
        ).squeeze(-1)
        present = counts.reshape(n_graphs, self.max_molecules) > 0
        tokens = torch.cat([component_mean, component_max], dim=-1).reshape(
            n_graphs, self.max_molecules, self.graph_dim * 2
        )
        tokens = self.component_projection(tokens)
        role_index = torch.arange(self.max_molecules, device=atom_states.device)
        tokens = tokens + self.role_embedding(role_index).unsqueeze(0)
        return tokens * present.unsqueeze(-1), present

    def forward(
        self,
        atom_states: torch.Tensor,
        batch: torch.Tensor,
        molecule_ids: torch.Tensor,
        conditions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, present = self._component_tokens(
            atom_states, batch, molecule_ids
        )
        condition_context = self.condition_encoder(conditions)
        gamma, beta = self.condition_film(condition_context).chunk(2, dim=-1)
        tokens = (
            tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        ) * present.unsqueeze(-1)

        for block in self.blocks:
            tokens = block(tokens, present, condition_context)

        context_tokens = condition_context.unsqueeze(1).expand(
            -1, self.max_molecules, -1
        )
        scores = self.readout_score(
            torch.cat([tokens, context_tokens], dim=-1)
        ).squeeze(-1)
        scores = scores.masked_fill(~present, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        weighted_summary = (tokens * weights.unsqueeze(-1)).sum(dim=1)

        masked_tokens = tokens.masked_fill(
            ~present.unsqueeze(-1), torch.finfo(tokens.dtype).min
        )
        max_summary = masked_tokens.max(dim=1).values
        return self.output(
            torch.cat([weighted_summary, max_summary, condition_context], dim=-1)
        )


class ConditionedRoleConcatReadout(nn.Module):
    """Role-preserving molecular GNN readout with explicit condition vectors.

    Unlike a global reaction pool, this readout never averages acceptor,
    donor, catalyst, ligand, solvent and additive together.  Fixed-order role
    embeddings and chemically relevant pair products retain the new
    combinations needed by reaction-key-disjoint evaluation.
    """

    def __init__(
        self,
        *,
        graph_dim: int,
        output_dim: int,
        condition_dim: int,
        max_molecules: int,
        dropout: float,
    ):
        super().__init__()
        if condition_dim < 1:
            raise ValueError(
                "Conditioned role concatenation requires condition features"
            )
        self.graph_dim = int(graph_dim)
        self.output_dim = int(output_dim)
        self.condition_dim = int(condition_dim)
        self.max_molecules = int(max_molecules)

        self.role_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.graph_dim * 3 + 1, self.graph_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.graph_dim),
                )
                for _ in range(self.max_molecules)
            ]
        )
        self.role_embedding = nn.Embedding(self.max_molecules, self.graph_dim)
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, self.graph_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.graph_dim * 2, self.graph_dim),
            nn.LayerNorm(self.graph_dim),
        )
        self.condition_film = nn.Linear(self.graph_dim, self.graph_dim * 2)

        # Key chemistry interactions: substrate pair, catalyst/ligand pair,
        # substrate–catalyst/ligand cross pairs, and condition-bearing
        # solvent/additive interactions.  Roles beyond the six registered
        # Michael components are still retained in the flattened token block.
        candidate_pairs = [
            (0, 1), (2, 3),
            (0, 2), (0, 3), (1, 2), (1, 3),
            (0, 4), (1, 4), (2, 4), (3, 4),
            (0, 5), (1, 5), (2, 5), (3, 5),
        ]
        self.role_pairs = tuple(
            pair for pair in candidate_pairs
            if max(pair) < self.max_molecules
        )
        input_dim = (
            self.graph_dim * (
                self.max_molecules + len(self.role_pairs) + 1
            )
            + self.condition_dim
        )
        self.output = nn.Sequential(
            nn.Linear(input_dim, self.graph_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.graph_dim * 4, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )

    def forward(
        self,
        atom_states: torch.Tensor,
        batch: torch.Tensor,
        molecule_ids: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        if molecule_ids.shape[0] != atom_states.shape[0]:
            raise ValueError(
                "Role-concat readout requires one molecular role ID per atom"
            )
        n_graphs = int(batch.max().item()) + 1
        role_ids = molecule_ids.clamp(min=0, max=self.max_molecules - 1)
        component_index = batch * self.max_molecules + role_ids
        n_components = n_graphs * self.max_molecules
        component_mean = global_mean_pool(
            atom_states, component_index, size=n_components
        ).reshape(n_graphs, self.max_molecules, self.graph_dim)
        component_max = global_max_pool(
            atom_states, component_index, size=n_components
        ).reshape(n_graphs, self.max_molecules, self.graph_dim)
        component_sum = global_add_pool(
            atom_states, component_index, size=n_components
        ).reshape(n_graphs, self.max_molecules, self.graph_dim)
        counts = global_add_pool(
            torch.ones(
                (atom_states.shape[0], 1),
                device=atom_states.device,
                dtype=atom_states.dtype,
            ),
            component_index,
            size=n_components,
        ).reshape(n_graphs, self.max_molecules)
        present = counts > 0

        pooled = torch.cat(
            [
                component_mean,
                component_max,
                component_sum,
                torch.log1p(counts).unsqueeze(-1),
            ],
            dim=-1,
        )
        tokens = torch.stack(
            [
                projection(pooled[:, role_index])
                for role_index, projection in enumerate(self.role_projections)
            ],
            dim=1,
        )
        role_index = torch.arange(self.max_molecules, device=atom_states.device)
        tokens = tokens + self.role_embedding(role_index).unsqueeze(0)
        condition_context = self.condition_encoder(conditions)
        gamma, beta = self.condition_film(condition_context).chunk(2, dim=-1)
        tokens = (
            tokens * (1.0 + torch.tanh(gamma).unsqueeze(1))
            + beta.unsqueeze(1)
        ) * present.unsqueeze(-1)

        pair_blocks = []
        for first, second in self.role_pairs:
            pair_present = (present[:, first] & present[:, second]).unsqueeze(-1)
            pair_blocks.append(
                tokens[:, first] * tokens[:, second] * pair_present
            )
        pair_features = torch.cat(pair_blocks, dim=-1)
        features = torch.cat(
            [
                tokens.reshape(n_graphs, -1),
                pair_features,
                condition_context,
                conditions,
            ],
            dim=-1,
        )
        return self.output(features)


class ConditionedRoleFactorizedReadout(nn.Module):
    """Compact role-main-effect and low-rank pair-interaction GNN readout.

    Reaction-key extrapolation asks the model to predict new combinations of
    mostly familiar molecular components.  This module retains one learned
    GNN token per molecular role, adds role-specific main effects, and
    represents chemically relevant pair effects with low-rank elementwise
    products.  Reaction conditions explicitly FiLM-modulate every role token
    and also contribute a direct context term.
    """

    def __init__(
        self,
        *,
        graph_dim: int,
        output_dim: int,
        condition_dim: int,
        max_molecules: int,
        dropout: float,
        factor_dim: int = 32,
    ):
        super().__init__()
        if condition_dim < 1:
            raise ValueError(
                "Role-factorized readout requires condition features"
            )
        self.graph_dim = int(graph_dim)
        self.output_dim = int(output_dim)
        self.condition_dim = int(condition_dim)
        self.max_molecules = int(max_molecules)
        self.factor_dim = min(int(factor_dim), self.graph_dim)

        candidate_pairs = [
            (0, 1), (2, 3),
            (0, 2), (0, 3), (1, 2), (1, 3),
            (0, 4), (1, 4), (2, 4), (3, 4),
            (0, 5), (1, 5), (2, 5), (3, 5),
        ]
        self.role_pairs = tuple(
            pair for pair in candidate_pairs
            if max(pair) < self.max_molecules
        )

        role_input_dim = self.graph_dim * 3 + 1
        self.role_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(role_input_dim, self.graph_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.graph_dim * 2, self.graph_dim),
                nn.LayerNorm(self.graph_dim),
            )
            for _ in range(self.max_molecules)
        ])
        self.condition_encoder = nn.Sequential(
            nn.Linear(self.condition_dim, self.graph_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.graph_dim * 2, self.graph_dim),
            nn.LayerNorm(self.graph_dim),
        )
        self.condition_film = nn.ModuleList([
            nn.Linear(self.graph_dim, self.graph_dim * 2)
            for _ in range(self.max_molecules)
        ])
        self.main_effects = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.graph_dim, self.output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(self.max_molecules)
        ])
        self.role_factors = nn.ModuleList([
            nn.Linear(self.graph_dim, self.factor_dim, bias=False)
            for _ in range(self.max_molecules)
        ])
        self.pair_effects = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.factor_dim, self.graph_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.graph_dim, self.output_dim, bias=False),
            )
            for _ in self.role_pairs
        ])
        self.condition_effect = nn.Sequential(
            nn.Linear(self.graph_dim, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_residual = nn.Sequential(
            nn.Linear(self.graph_dim * 2, self.output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        atom_states: torch.Tensor,
        batch: torch.Tensor,
        molecule_ids: torch.Tensor,
        conditions: torch.Tensor,
    ) -> torch.Tensor:
        if molecule_ids.shape[0] != atom_states.shape[0]:
            raise ValueError(
                "Role-factorized readout requires one molecular role ID "
                "per atom"
            )
        n_graphs = int(batch.max().item()) + 1
        role_ids = molecule_ids.clamp(
            min=0, max=self.max_molecules - 1
        )
        component_index = batch * self.max_molecules + role_ids
        n_components = n_graphs * self.max_molecules
        mean = global_mean_pool(
            atom_states, component_index, size=n_components
        ).reshape(n_graphs, self.max_molecules, self.graph_dim)
        maximum = global_max_pool(
            atom_states, component_index, size=n_components
        ).reshape(n_graphs, self.max_molecules, self.graph_dim)
        total = global_add_pool(
            atom_states, component_index, size=n_components
        ).reshape(n_graphs, self.max_molecules, self.graph_dim)
        counts = global_add_pool(
            torch.ones(
                (atom_states.shape[0], 1),
                device=atom_states.device,
                dtype=atom_states.dtype,
            ),
            component_index,
            size=n_components,
        ).reshape(n_graphs, self.max_molecules)
        present = counts > 0
        size_scaled_total = (
            total / counts.clamp(min=1).sqrt().unsqueeze(-1)
        )
        pooled = torch.cat(
            [
                mean,
                maximum,
                size_scaled_total,
                torch.log1p(counts).unsqueeze(-1),
            ],
            dim=-1,
        )

        condition_context = self.condition_encoder(conditions)
        tokens = []
        for role_index, encoder in enumerate(self.role_encoders):
            token = encoder(pooled[:, role_index])
            gamma, beta = self.condition_film[role_index](
                condition_context
            ).chunk(2, dim=-1)
            token = (
                token * (1.0 + 0.25 * torch.tanh(gamma))
                + 0.25 * beta
            )
            tokens.append(
                token * present[:, role_index].unsqueeze(-1)
            )
        tokens = torch.stack(tokens, dim=1)

        main_summary = torch.zeros(
            (n_graphs, self.output_dim),
            dtype=atom_states.dtype,
            device=atom_states.device,
        )
        factors = []
        for role_index in range(self.max_molecules):
            main_summary = main_summary + self.main_effects[role_index](
                tokens[:, role_index]
            ) * present[:, role_index].unsqueeze(-1)
            factors.append(
                self.role_factors[role_index](tokens[:, role_index])
            )

        pair_summary = torch.zeros_like(main_summary)
        for pair_index, (first, second) in enumerate(self.role_pairs):
            pair_present = (
                present[:, first] & present[:, second]
            ).unsqueeze(-1)
            pair_summary = pair_summary + self.pair_effects[pair_index](
                factors[first] * factors[second]
            ) * pair_present
        if self.role_pairs:
            pair_summary = pair_summary / len(self.role_pairs) ** 0.5

        global_summary = self.global_residual(
            torch.cat(
                [
                    global_mean_pool(atom_states, batch),
                    global_max_pool(atom_states, batch),
                ],
                dim=-1,
            )
        )
        output = (
            main_summary / max(self.max_molecules, 1) ** 0.5
            + pair_summary
            + self.condition_effect(condition_context)
            + global_summary
        )
        # Fixed-order, per-role-normalised GNN tokens preserve the meaning of
        # every molecular slot for eKNN.  Normalising each present role before
        # concatenation prevents a large catalyst/solvent token from drowning
        # out substrate similarity.  Conditions remain an explicit, separately
        # weighted part of the retrieval vector in ``retrieval_vectors``.
        retrieval_tokens = F.normalize(tokens, p=2, dim=-1)
        retrieval_tokens = (
            retrieval_tokens * present.unsqueeze(-1)
        ).reshape(n_graphs, -1)
        retrieval_tokens = retrieval_tokens / max(
            self.max_molecules, 1
        ) ** 0.5
        return self.output_norm(output), retrieval_tokens


class ReactionGNN(nn.Module):
    """Graph reaction encoder with a scalar yield/outcome prediction head.

    Forward returns a dict with ``yield`` and ``embedding`` keys,
    compatible with the ekNN evaluation pipeline.

    Args:
        d_model: hidden dimension (default 256)
        n_layers: number of GPS blocks (default 3)
        n_heads: attention heads (default 4, ignored if use_global_attn=False)
        ff_dim: FFN hidden dimension (default 512)
        edge_dim: edge attribute dimension (default 64)
        dropout: dropout rate (default 0.1)
        max_molecules: max components per reaction (default 8)
        pe_dim: Laplacian PE dimension (default 3)
        readout: pooling mode — "mean_max", "mean_max_sum", "mean", "sum", "attn"
        use_global_attn: if False, use PureMPNNBlock (GINE only, no MHA)
        use_gat: if True, use GATBlock (local attention over neighbours)
        use_jk: if True, use Jumping Knowledge (concat all layer outputs)
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 3,
        n_heads: int = 4,
        ff_dim: int = 512,
        edge_dim: int = 64,
        dropout: float = 0.1,
        max_molecules: int = 8,
        pe_dim: int = 3,
        condition_dim: int = 0,
        condition_fusion_mode: str = "gated_add",
        condition_node_film: bool = False,
        condition_categorical_cardinalities: Optional[Sequence[int]] = None,
        condition_token_layers: int = 2,
        component_identity_cardinalities: Optional[Sequence[int]] = None,
        component_identity_dim: int = 32,
        readout: str = "mean_max",
        use_global_attn: bool = True,
        use_gat: bool = False,
        use_virtual_node: bool = False,
        use_jk: bool = False,
        use_component_readout: bool = False,
        use_rolewise_readout: bool = False,
        use_role_interaction_readout: bool = False,
        role_interaction_layers: int = 2,
        use_role_concat_readout: bool = False,
        use_role_factorized_readout: bool = False,
        use_role_factorized_retrieval: bool = False,
        retrieval_dim: int = 0,
        use_graph_transformer: bool = False,
        use_edge_transformer: bool = False,
        use_graphormer: bool = False,
        use_deep_gps: bool = False,
        use_light_gps: bool = False,
        backbone: str = "",
    ):
        super().__init__()
        self.d_model = d_model
        self.readout_mode = readout
        self.use_jk = use_jk
        self.use_virtual_node = use_virtual_node
        self.use_component_readout = bool(use_component_readout)
        self.use_rolewise_readout = bool(use_rolewise_readout)
        self.use_role_interaction_readout = bool(use_role_interaction_readout)
        self.use_role_concat_readout = bool(use_role_concat_readout)
        self.use_role_factorized_readout = bool(
            use_role_factorized_readout
        )
        self.use_role_factorized_retrieval = bool(
            use_role_factorized_retrieval
        )
        if (
            self.use_role_factorized_retrieval
            and not self.use_role_factorized_readout
        ):
            raise ValueError(
                "Role-factorized retrieval requires the role-factorized "
                "readout"
            )
        self.retrieval_dim = int(retrieval_dim)
        self.condition_dim = int(condition_dim)
        self.condition_fusion_mode = condition_fusion_mode
        self.use_condition_node_film = bool(condition_node_film)
        self.condition_categorical_cardinalities = tuple(
            int(value) for value in (condition_categorical_cardinalities or ())
        )
        self.condition_token_layers = int(condition_token_layers)
        self.component_identity_cardinalities = tuple(
            int(value) for value in (component_identity_cardinalities or ())
        )
        self.component_identity_dim = int(component_identity_dim)
        if self.component_identity_cardinalities and (
            len(self.component_identity_cardinalities) > max_molecules
        ):
            raise ValueError(
                "component_identity_cardinalities cannot contain more fields "
                "than max_molecules"
            )
        if self.component_identity_cardinalities and self.component_identity_dim < 1:
            raise ValueError("component_identity_dim must be positive")

        # ── Virtual node embedding ──
        if use_virtual_node:
            self.vn_embed = nn.Embedding(1, d_model)
            self.vn_norm = nn.LayerNorm(d_model)
        if self.use_component_readout and use_virtual_node:
            raise ValueError(
                "use_component_readout is incompatible with use_virtual_node "
                "because virtual nodes do not have a molecular-component id"
            )
        hierarchical_readouts = (
            self.use_component_readout,
            self.use_rolewise_readout,
            self.use_role_interaction_readout,
            self.use_role_concat_readout,
            self.use_role_factorized_readout,
        )
        if sum(bool(value) for value in hierarchical_readouts) > 1:
            raise ValueError(
                "Choose only one hierarchical reaction readout"
            )
        if (
            self.use_rolewise_readout
            or self.use_role_interaction_readout
            or self.use_role_concat_readout
            or self.use_role_factorized_readout
        ) and use_virtual_node:
            raise ValueError(
                "Role-aware readouts are incompatible with use_virtual_node"
            )
        if (
            self.use_role_interaction_readout
            or self.use_role_concat_readout
            or self.use_role_factorized_readout
        ) and use_jk:
            raise ValueError(
                "Role-aware readouts use the final atom states directly "
                "and is incompatible with Jumping Knowledge"
            )
        if (
            self.use_role_interaction_readout
            or self.use_role_concat_readout
            or self.use_role_factorized_readout
        ) and self.condition_dim < 1:
            raise ValueError(
                "Conditioned role readouts require condition_dim > 0"
            )

        # ── Backbone registry (all supported block types) ──
        _BACKBONE_MAP = {
            "gps": GPSBlock, "mpnn": PureMPNNBlock, "gat": GATBlock,
            "graph_transformer": GraphTransformerBlock,
            "edge_transformer": EdgeTransformerBlock,
            "graphormer": GraphormerBlock, "deep_gps": DeepGPSBlock,
            "light": LightGPSBlock,
            "gcn": GCNBlock, "sage": SAGEBlock, "gin": GINBlock,
            "gatv2": GATv2Block, "transformer_conv": TransformerConvBlock,
            "pna": PNABlock, "general_conv": GeneralConvBlock,
            "gen_conv": GENConvBlock, "film": FiLMConvBlock,
            "gated_graph": GatedGraphConvBlock, "res_gated": ResGatedGraphBlock,
            "dmpnn": DirectedEdgeMPNNBlock, "egconv": EGConvBlock,
        }
        _NO_HEADS = {
            "mpnn", "gcn", "sage", "gin", "film", "gated_graph", "dmpnn"
        }

        # Resolve block class: --backbone takes priority, else fall back to legacy flags
        if backbone and backbone in _BACKBONE_MAP:
            BlockCls = _BACKBONE_MAP[backbone]
        elif use_graphormer:
            BlockCls = GraphormerBlock
        elif use_graph_transformer:
            BlockCls = GraphTransformerBlock
        elif use_edge_transformer:
            BlockCls = EdgeTransformerBlock
        elif use_deep_gps:
            BlockCls = DeepGPSBlock
        elif use_light_gps:
            BlockCls = LightGPSBlock
        elif use_gat:
            BlockCls = GATBlock
        elif use_global_attn:
            BlockCls = GPSBlock
        else:
            BlockCls = PureMPNNBlock

        # Embedding
        self.atom_encoder = AtomEncoder(d_model, max_molecules, pe_dim, dropout)
        self.edge_encoder = EdgeEncoder(edge_dim)

        block_kwargs = dict(d_model=d_model, edge_dim=edge_dim, ff_dim=ff_dim, dropout=dropout)
        if backbone not in _NO_HEADS and (use_global_attn or use_gat or backbone not in ("", "mpnn")):
            block_kwargs['n_heads'] = n_heads

        self.blocks = nn.ModuleList([
            BlockCls(**block_kwargs)
            for _ in range(n_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(d_model)

        # Readout → yield (with LayerNorm to stabilise pooled features)
        if readout == "attn":
            readout_dim = d_model
            self.attn_pool = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.Tanh(),
                nn.Linear(d_model // 2, 1),
            )
        else:
            if readout == "mean_max":
                readout_dim = d_model * 2
            elif readout == "mean_max_sum":
                readout_dim = d_model * 3
            else:
                readout_dim = d_model

        # JK: concat all layer outputs → multiply readout dim
        if use_jk:
            readout_dim = readout_dim * (n_layers + 1)  # +1 for input embedding

        # Optional hierarchical reaction readout.  Molecules are disconnected
        # in the reaction graph, so global mean/max pooling can blur which
        # atoms came from acceptor, donor, catalyst, ligand, etc.  This module
        # first pools each component, then lets attention combine role-aware
        # component tokens.  It is still entirely a graph representation.
        if self.use_component_readout:
            if d_model % n_heads != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by n_heads={n_heads} "
                    "when use_component_readout=True"
                )
            self.component_token_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model),
            )
            self.component_role_embed = nn.Embedding(max_molecules, d_model)
            self.component_attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            )
            self.component_norm = nn.LayerNorm(d_model)
            self.component_out = nn.Sequential(
                nn.Linear(d_model, readout_dim),
                nn.GELU(),
            )
        else:
            self.component_token_proj = None
            self.component_role_embed = None
            self.component_attn = None
            self.component_norm = None
            self.component_out = None

        # A compact role-preserving readout.  The reaction graph contains
        # disconnected molecular components, so a single global pool can
        # erase the distinction between acceptor, donor, catalyst, ligand,
        # solvent and additive.  This branch pools mean/max/sum statistics
        # inside each fixed role, encodes every role with a shared graph
        # projection plus a learned role token, and concatenates the role
        # summaries before a residual projection.  It remains a pure GNN
        # representation: no fingerprints or external/pretrained features.
        if self.use_rolewise_readout:
            role_atom_dim = d_model * (n_layers + 1 if use_jk else 1)
            role_input_dim = role_atom_dim * 3 + 1
            role_hidden_dim = d_model
            role_output_hidden_dim = d_model * 2
            self.rolewise_encoder = nn.Sequential(
                nn.Linear(role_input_dim, role_hidden_dim),
                nn.GELU(),
                nn.LayerNorm(role_hidden_dim),
                nn.Dropout(dropout),
            )
            self.rolewise_role_embed = nn.Embedding(max_molecules, role_hidden_dim)
            self.rolewise_output = nn.Sequential(
                nn.Linear(
                    max_molecules * role_hidden_dim,
                    role_output_hidden_dim,
                ),
                nn.GELU(),
                nn.LayerNorm(role_output_hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(role_output_hidden_dim, readout_dim),
            )
            # Begin as the original global readout and learn a role-preserving
            # residual only when supported by the training data.
            nn.init.zeros_(self.rolewise_output[-1].weight)
            nn.init.zeros_(self.rolewise_output[-1].bias)
        else:
            self.rolewise_encoder = None
            self.rolewise_role_embed = None
            self.rolewise_output = None

        self.readout_norm = nn.LayerNorm(readout_dim)
        if self.use_role_interaction_readout:
            self.role_interaction_readout = ConditionedRoleInteractionReadout(
                graph_dim=d_model,
                output_dim=readout_dim,
                condition_dim=self.condition_dim,
                max_molecules=max_molecules,
                n_layers=role_interaction_layers,
                dropout=dropout,
            )
        else:
            self.role_interaction_readout = None
        if self.use_role_concat_readout:
            self.role_concat_readout = ConditionedRoleConcatReadout(
                graph_dim=d_model,
                output_dim=readout_dim,
                condition_dim=self.condition_dim,
                max_molecules=max_molecules,
                dropout=dropout,
            )
        else:
            self.role_concat_readout = None
        if self.use_role_factorized_readout:
            self.role_factorized_readout = (
                ConditionedRoleFactorizedReadout(
                    graph_dim=d_model,
                    output_dim=readout_dim,
                    condition_dim=self.condition_dim,
                    max_molecules=max_molecules,
                    dropout=dropout,
                )
            )
        else:
            self.role_factorized_readout = None

        # Optional, fold-fitted component identity tokens.  A reaction-key
        # group split deliberately permits an individual substrate/catalyst/
        # ligand to reappear in a new reaction combination.  The GNN still
        # encodes every molecular graph, while these learned role-specific
        # tokens let it retain stable information for components observed in
        # the fitting fold.  Values unseen in that fold are mapped to a
        # dedicated OOV token by the data pipeline, so no held-out identity
        # leaks into training.
        if self.component_identity_cardinalities:
            if any(cardinality < 2 for cardinality in self.component_identity_cardinalities):
                raise ValueError(
                    "Each component identity vocabulary must contain at least "
                    "one value plus an OOV token"
                )
            self.component_identity_embeddings = nn.ModuleList([
                nn.Embedding(cardinality, self.component_identity_dim)
                for cardinality in self.component_identity_cardinalities
            ])
            identity_width = (
                len(self.component_identity_cardinalities) * self.component_identity_dim
            )
            self.component_identity_encoder = nn.Sequential(
                nn.Linear(identity_width, readout_dim),
                nn.GELU(),
                nn.LayerNorm(readout_dim),
                nn.Dropout(dropout),
                nn.Linear(readout_dim, readout_dim),
                nn.GELU(),
            )
            self.component_identity_gate = nn.Sequential(
                nn.Linear(readout_dim * 2, readout_dim),
                nn.Sigmoid(),
            )
            self.component_identity_norm = nn.LayerNorm(readout_dim)
        else:
            self.component_identity_embeddings = None
            self.component_identity_encoder = None
            self.component_identity_gate = None
            self.component_identity_norm = None

        # Optional late, gated fusion of numerical reaction conditions.  The
        # pure-GNN baseline has condition_dim=0 and therefore executes the
        # original graph-only path with no additional parameters.  When
        # enabled, the graph trunk remains intact; only its pooled reaction
        # representation is modulated by prediction-time condition features.
        if self.condition_dim > 0:
            if self.condition_fusion_mode not in {
                "gated_add", "concat_residual", "cross_attention"
            }:
                raise ValueError(
                    "condition_fusion_mode must be 'gated_add', 'concat_residual', "
                    f"or 'cross_attention', got {self.condition_fusion_mode!r}"
                )
            if self.condition_fusion_mode == "cross_attention":
                if use_virtual_node:
                    raise ValueError(
                        "condition_fusion_mode='cross_attention' is incompatible "
                        "with use_virtual_node because role IDs must align with atom states"
                    )
                self.condition_component_fusion = ConditionComponentTransformer(
                    reaction_dim=readout_dim,
                    graph_dim=d_model,
                    numeric_condition_dim=self.condition_dim,
                    categorical_cardinalities=self.condition_categorical_cardinalities,
                    max_molecules=max_molecules,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    n_layers=self.condition_token_layers,
                    dropout=dropout,
                )
                # Keep a direct gated condition path as a stable base.  The
                # component/condition transformer then learns only a residual
                # interaction over that representation.
                self.condition_encoder = nn.Sequential(
                    nn.Linear(self.condition_dim, d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, readout_dim),
                    nn.GELU(),
                )
                self.condition_gate = nn.Sequential(
                    nn.Linear(readout_dim, readout_dim),
                    nn.Sigmoid(),
                )
                self.condition_fusion = None
                self.condition_fusion_norm = nn.LayerNorm(readout_dim)
            else:
                self.condition_component_fusion = None
                self.condition_encoder = nn.Sequential(
                    nn.Linear(self.condition_dim, d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, readout_dim),
                    nn.GELU(),
                )
                if self.condition_fusion_mode == "gated_add":
                    self.condition_gate = nn.Sequential(
                        nn.Linear(readout_dim, readout_dim),
                        nn.Sigmoid(),
                    )
                    self.condition_fusion = None
                else:
                    self.condition_gate = None
                    self.condition_fusion = nn.Sequential(
                        nn.Linear(readout_dim * 2, readout_dim * 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(readout_dim * 2, readout_dim),
                    )
                self.condition_fusion_norm = nn.LayerNorm(readout_dim)
            if self.use_condition_node_film:
                self.condition_node_film = nn.Sequential(
                    nn.Linear(self.condition_dim, d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model * 2),
                )
                # Start as an identity modulation.  This makes the FiLM model
                # initially equivalent to the unmodulated graph trunk and lets
                # training learn conditional effects only when useful.
                nn.init.zeros_(self.condition_node_film[-1].weight)
                nn.init.zeros_(self.condition_node_film[-1].bias)
            else:
                self.condition_node_film = None
        else:
            self.condition_encoder = None
            self.condition_gate = None
            self.condition_fusion = None
            self.condition_fusion_norm = None
            self.condition_component_fusion = None
            self.condition_node_film = None

        self.yield_head = nn.Sequential(
            nn.Linear(readout_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        if self.retrieval_dim > 0:
            self.retrieval_head = nn.Sequential(
                nn.Linear(readout_dim, readout_dim),
                nn.GELU(),
                nn.LayerNorm(readout_dim),
                nn.Linear(readout_dim, self.retrieval_dim),
            )
        else:
            self.retrieval_head = None

    def _pool(self, x, batch):
        """Pool node embeddings into graph-level embedding."""
        if self.readout_mode == "attn":
            B = int(batch.max().item()) + 1
            attn_raw = self.attn_pool(x)
            h = torch.zeros(B, self.d_model, device=x.device)
            for b_idx in range(B):
                mask = (batch == b_idx)
                scores_b = attn_raw[mask].squeeze(-1)
                weights_b = torch.softmax(scores_b, dim=0)
                h[b_idx] = (weights_b.unsqueeze(-1) * x[mask]).sum(dim=0)
            return h
        elif self.readout_mode == "mean_max":
            h_mean = global_mean_pool(x, batch)
            h_max = global_max_pool(x, batch)
            return torch.cat([h_mean, h_max], dim=-1)
        elif self.readout_mode == "mean_max_sum":
            h_mean = global_mean_pool(x, batch)
            h_max = global_max_pool(x, batch)
            h_sum = global_add_pool(x, batch)
            return torch.cat([h_mean, h_max, h_sum], dim=-1)
        elif self.readout_mode == "sum":
            return global_add_pool(x, batch)
        else:
            return global_mean_pool(x, batch)

    def _augment_vn(self, x, edge_index, edge_attr, batch):
        """Add virtual nodes: one per graph, connected to all atoms bidirectionally."""
        B = int(batch.max().item()) + 1
        device = x.device
        N = x.shape[0]

        # Virtual node embeddings [B, d_model]
        vn_idx = torch.arange(B, device=device)
        vn_x = self.vn_embed(torch.zeros(B, dtype=torch.long, device=device))
        vn_x = self.vn_norm(vn_x)

        # Concatenate node features
        x_aug = torch.cat([x, vn_x], dim=0)  # [N+B, d_model]

        # Virtual node ↔ all atoms edges
        atom_idx = torch.arange(N, device=device)
        vn_global_idx = N + batch[atom_idx]  # [N] — which VN each atom connects to
        vn_to_atom = torch.stack([vn_global_idx, atom_idx], dim=0)  # [2, N]
        atom_to_vn = torch.stack([atom_idx, vn_global_idx], dim=0)  # [2, N]

        edge_index_aug = torch.cat([edge_index, vn_to_atom, atom_to_vn], dim=1)  # [2, E+2N]

        # Virtual edge features: zeros (special "virtual bond" type)
        vn_edge = torch.zeros(2 * N, edge_attr.shape[1], device=device, dtype=edge_attr.dtype)
        edge_attr_aug = torch.cat([edge_attr, vn_edge], dim=0)

        # Extended batch
        batch_aug = torch.cat([batch, vn_idx], dim=0)  # [N+B]

        return x_aug, edge_index_aug, edge_attr_aug, batch_aug

    def _condition_matrix(self, data, batch_size: int) -> torch.Tensor:
        """Return the validated [batch, condition_dim] condition matrix."""
        if self.condition_dim <= 0:
            raise ValueError("No condition matrix is available when condition_dim is zero")
        if not hasattr(data, "condition_features"):
            raise ValueError(
                "ReactionGNN was constructed with condition_dim > 0 but the "
                "input graph batch has no condition_features attribute"
            )

        conditions = data.condition_features
        if conditions.ndim == 1:
            conditions = conditions.reshape(1, -1)
        if conditions.ndim != 2 or conditions.shape[1] != self.condition_dim:
            raise ValueError(
                "condition_features shape mismatch: expected [batch, "
                f"{self.condition_dim}], got {tuple(conditions.shape)}"
            )
        if conditions.shape[0] != batch_size:
            raise ValueError(
                "condition_features batch size does not match graph embeddings: "
                f"{conditions.shape[0]} vs {batch_size}"
            )
        return conditions

    def _condition_categorical_matrix(self, data, batch_size: int) -> Optional[torch.Tensor]:
        """Return fold-fitted categorical condition IDs when configured."""
        if not self.condition_categorical_cardinalities:
            return None
        if not hasattr(data, "condition_categorical_features"):
            raise ValueError(
                "ReactionGNN was constructed with categorical condition embeddings "
                "but the input graph batch has no condition_categorical_features"
            )
        categorical = data.condition_categorical_features
        if categorical.ndim == 1:
            categorical = categorical.reshape(1, -1)
        expected_width = len(self.condition_categorical_cardinalities)
        if categorical.shape != (batch_size, expected_width):
            raise ValueError(
                "condition_categorical_features shape mismatch: expected [batch, "
                f"{expected_width}], got {tuple(categorical.shape)}"
            )
        return categorical

    def _component_identity_matrix(
        self, data, batch_size: int
    ) -> Optional[torch.Tensor]:
        """Return fold-fitted role identity IDs when this fusion is enabled."""
        if not self.component_identity_cardinalities:
            return None
        if not hasattr(data, "component_identity_features"):
            raise ValueError(
                "ReactionGNN was constructed with component identity embeddings "
                "but the input graph batch has no component_identity_features"
            )
        identities = data.component_identity_features
        if identities.ndim == 1:
            identities = identities.reshape(1, -1)
        expected_width = len(self.component_identity_cardinalities)
        if identities.shape != (batch_size, expected_width):
            raise ValueError(
                "component_identity_features shape mismatch: expected [batch, "
                f"{expected_width}], got {tuple(identities.shape)}"
            )
        return identities

    def _fuse_component_identities(self, h: torch.Tensor, data) -> torch.Tensor:
        """Fuse role-specific molecular identity tokens into a graph embedding."""
        identities = self._component_identity_matrix(data, h.shape[0])
        if identities is None:
            return h
        tokens = []
        for field_index, embedding in enumerate(self.component_identity_embeddings):
            ids = identities[:, field_index].long()
            if torch.any(ids < 0) or torch.any(ids >= embedding.num_embeddings):
                raise ValueError(
                    "Component identity ID is outside its fold-fitted vocabulary"
                )
            tokens.append(embedding(ids))
        identity_h = self.component_identity_encoder(torch.cat(tokens, dim=-1))
        gate = self.component_identity_gate(torch.cat([h, identity_h], dim=-1))
        return self.component_identity_norm(h + gate * identity_h)

    def _fuse_conditions(
        self,
        h: torch.Tensor,
        data,
        atom_states: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        molecule_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse standardised numeric conditions into a pooled graph embedding."""
        if self.condition_dim <= 0:
            return h

        conditions = self._condition_matrix(data, h.shape[0])
        if self.condition_fusion_mode == "cross_attention":
            if atom_states is None or batch is None or molecule_ids is None:
                raise ValueError("Cross-attention condition fusion needs atom states and role IDs")
            condition_h = self.condition_encoder(conditions)
            gate = self.condition_gate(condition_h)
            h = self.condition_fusion_norm(h + gate * condition_h)
            categorical = self._condition_categorical_matrix(data, h.shape[0])
            return self.condition_component_fusion(
                h,
                conditions,
                categorical,
                atom_states,
                batch,
                molecule_ids,
            )
        condition_h = self.condition_encoder(conditions)
        if self.condition_fusion_mode == "gated_add":
            gate = self.condition_gate(condition_h)
            fused_delta = gate * condition_h
        else:
            fused_delta = self.condition_fusion(torch.cat([h, condition_h], dim=-1))
        return self.condition_fusion_norm(h + fused_delta)

    def _component_embedding(self, x: torch.Tensor, batch: torch.Tensor,
                             molecule_ids: torch.Tensor) -> torch.Tensor:
        """Role-aware pooled representation of the disconnected components."""
        n_graphs = int(batch.max().item()) + 1
        role_ids = molecule_ids.clamp(min=0, max=self.component_role_embed.num_embeddings - 1)
        component_index = batch * self.component_role_embed.num_embeddings + role_ids
        n_components = n_graphs * self.component_role_embed.num_embeddings

        comp_mean = global_mean_pool(x, component_index, size=n_components)
        comp_max = global_max_pool(x, component_index, size=n_components)
        counts = global_add_pool(
            torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype),
            component_index, size=n_components,
        ).squeeze(-1)
        present = counts.reshape(n_graphs, -1) > 0
        tokens = torch.cat([comp_mean, comp_max], dim=-1).reshape(
            n_graphs, -1, self.d_model * 2
        )
        tokens = self.component_token_proj(tokens)
        role_index = torch.arange(tokens.shape[1], device=x.device)
        tokens = tokens + self.component_role_embed(role_index).unsqueeze(0)
        attn_tokens, _ = self.component_attn(
            tokens, tokens, tokens, key_padding_mask=~present, need_weights=False
        )
        tokens = self.component_norm(tokens + attn_tokens)
        denom = present.sum(dim=1, keepdim=True).clamp(min=1)
        summary = (tokens * present.unsqueeze(-1)).sum(dim=1) / denom
        return self.component_out(summary)

    def _rolewise_embedding(
        self,
        x: torch.Tensor,
        batch: torch.Tensor,
        molecule_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return a fixed-order, role-preserving reaction representation."""
        n_graphs = int(batch.max().item()) + 1
        n_roles = self.rolewise_role_embed.num_embeddings
        role_ids = molecule_ids.clamp(min=0, max=n_roles - 1)
        role_index = batch * n_roles + role_ids
        n_role_slots = n_graphs * n_roles

        role_mean = global_mean_pool(x, role_index, size=n_role_slots)
        role_max = global_max_pool(x, role_index, size=n_role_slots)
        role_sum = global_add_pool(x, role_index, size=n_role_slots)
        counts = global_add_pool(
            torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype),
            role_index,
            size=n_role_slots,
        )
        present = counts > 0
        role_features = torch.cat(
            [role_mean, role_max, role_sum, torch.log1p(counts)],
            dim=-1,
        )
        role_tokens = self.rolewise_encoder(role_features).reshape(
            n_graphs, n_roles, -1
        )
        role_positions = torch.arange(n_roles, device=x.device)
        role_tokens = (
            role_tokens
            + self.rolewise_role_embed(role_positions).unsqueeze(0)
        )
        role_tokens = role_tokens * present.reshape(
            n_graphs, n_roles, 1
        ).to(role_tokens.dtype)
        return self.rolewise_output(role_tokens.reshape(n_graphs, -1))

    def get_embedding(
        self, data, return_structured_retrieval: bool = False
    ):
        """Get reaction-level embedding (for ekNN retrieval)."""
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)

        edge_attr_enc = self.edge_encoder(data.edge_attr)
        x = self.atom_encoder(
            data.x, data.molecule_ids,
            getattr(data, 'laplacian_pe', None),
        )

        if self.condition_node_film is not None:
            n_graphs = int(batch.max().item()) + 1
            conditions = self._condition_matrix(data, n_graphs)
            gamma, beta = self.condition_node_film(conditions).chunk(2, dim=-1)
            x = x * (1.0 + gamma[batch]) + beta[batch]

        # Virtual node augmentation
        if self.use_virtual_node:
            x, edge_index, edge_attr_enc, batch = self._augment_vn(
                x, data.edge_index, edge_attr_enc, batch)
        else:
            edge_index = data.edge_index

        # GPS/MPNN blocks (with optional JK)
        xs = [x]  # input embedding
        for blk in self.blocks:
            x = blk(x, edge_index, edge_attr_enc, batch)
            if self.use_jk:
                xs.append(x)

        x = self.final_norm(x)
        rolewise_atom_states = x
        if self.use_jk:
            xs[-1] = x  # use normalized version for last layer
            # Pool each layer's output and concatenate
            hs = []
            rolewise_layers = []
            for layer_x in xs:
                layer_x_norm = self.final_norm(layer_x)
                hs.append(self._pool(layer_x_norm, batch))
                rolewise_layers.append(layer_x_norm)
            h = torch.cat(hs, dim=-1)
            rolewise_atom_states = torch.cat(rolewise_layers, dim=-1)
        else:
            # Readout
            h = self._pool(x, batch)

        structured_retrieval = None
        if self.use_role_interaction_readout:
            conditions = self._condition_matrix(
                data, int(batch.max().item()) + 1
            )
            h = self.role_interaction_readout(
                x, batch, data.molecule_ids, conditions
            )
        elif self.use_role_concat_readout:
            conditions = self._condition_matrix(
                data, int(batch.max().item()) + 1
            )
            h = self.role_concat_readout(
                x, batch, data.molecule_ids, conditions
            )
        elif self.use_role_factorized_readout:
            conditions = self._condition_matrix(
                data, int(batch.max().item()) + 1
            )
            h, structured_retrieval = self.role_factorized_readout(
                x, batch, data.molecule_ids, conditions
            )
        elif self.use_rolewise_readout:
            h = h + self._rolewise_embedding(
                rolewise_atom_states, batch, data.molecule_ids
            )
        elif self.use_component_readout:
            h = h + self._component_embedding(x, batch, data.molecule_ids)
        h = self.readout_norm(h)
        h = self._fuse_component_identities(h, data)

        h = self._fuse_conditions(
            h, data, x, batch, data.molecule_ids
        )
        if return_structured_retrieval:
            return h, structured_retrieval
        return h

    def forward(self, data) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            data: PyG Data/Batch with same fields as get_embedding.

        Returns:
            dict with:
                "yield": [B, 1] predicted yield
                "embedding": [B, readout_dim] reaction embedding
        """
        h, structured_retrieval = self.get_embedding(
            data, return_structured_retrieval=True
        )
        y = self.yield_head(h)
        if self.use_role_factorized_retrieval:
            retrieval_embedding = structured_retrieval
        else:
            retrieval_embedding = (
                self.retrieval_head(h)
                if self.retrieval_head is not None
                else h
            )
        return {"yield": y, "embedding": retrieval_embedding}

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Kept for compatibility with legacy notebooks and frozen experiment scripts.
SERMvGNN = ReactionGNN

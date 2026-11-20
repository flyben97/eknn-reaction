"""Shared inference helpers; numerics preserved from corrected v0.2.1."""
import numpy as np
import torch
from torch_geometric.data import Batch

@torch.no_grad()
def encode_reactions(model, graphs, device, batch_size=32):
    """Encode a list of PyG Data objects into reaction embeddings.

    Args:
        model: ReactionGNN
        graphs: list of PyG Data
        device: torch device
        batch_size: batch size for encoding

    Returns:
        h: [N, readout_dim] numpy array of reaction embeddings
        y_base: [N] numpy array of base predictions
    """
    model.eval()
    h_list, y_list = [], []

    for i in range(0, len(graphs), batch_size):
        batch_graphs = graphs[i:i + batch_size]
        batch = Batch.from_data_list(batch_graphs).to(device)
        out = model(batch)
        h_list.append(out["embedding"].cpu().numpy())
        y_list.append(out["yield"].squeeze(-1).cpu().numpy())

    return np.concatenate(h_list, axis=0), np.concatenate(y_list, axis=0)


def retrieval_vectors(embeddings, graphs, condition_scale=0.0):
    """Build unit retrieval vectors from GNN embeddings and conditions."""
    graph_part = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    )
    if condition_scale <= 0:
        return graph_part
    condition_rows = []
    for graph in graphs:
        if not hasattr(graph, "condition_features"):
            raise ValueError(
                "A positive eKNN condition scale requires condition_features "
                "on every reaction graph"
            )
        condition_rows.append(
            graph.condition_features.detach().cpu().numpy().reshape(1, -1)
        )
    conditions = np.concatenate(condition_rows, axis=0).astype(np.float32)
    condition_part = conditions / np.sqrt(max(conditions.shape[1], 1))
    combined = np.concatenate(
        [graph_part, float(condition_scale) * condition_part], axis=1
    )
    return combined / (
        np.linalg.norm(combined, axis=1, keepdims=True) + 1e-8
    )



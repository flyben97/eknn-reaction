"""Shared neighbors helpers; numerics preserved from corrected v0.2.1."""
import numpy as np
from eknn_reaction.retrieval import temperature_weights

def knn_correct(h_test_norm, h_train_norm, residual_train, k, T):
    """k-NN residual correction with numerically normalized weights."""
    M = h_test_norm.shape[0]
    N_train = h_train_norm.shape[0]
    if N_train == 0:
        raise ValueError("Cannot run kNN correction with an empty training index")
    k_eff = min(k, N_train)
    corrections = np.zeros(M)

    for j in range(M):
        hq = h_test_norm[j]
        dists = np.linalg.norm(h_train_norm - hq, axis=1)
        nn_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        nn_dists = dists[nn_idx]
        nn_residuals = residual_train[nn_idx]
        weights = temperature_weights(nn_dists, T)
        corrections[j] = np.dot(weights, nn_residuals)

    return corrections


def prepare_knn_neighbours(h_query_norm, h_train_norm, max_k):
    """Cache nearest indices/distances for unit-normalised retrieval vectors."""
    if h_train_norm.shape[0] == 0:
        raise ValueError("Cannot prepare neighbours from an empty index")
    k_eff = min(int(max_k), h_train_norm.shape[0])
    query = np.asarray(h_query_norm, dtype=np.float32)
    train = np.asarray(h_train_norm, dtype=np.float32)
    # Compute the same explicit Euclidean norm used by ``knn_correct``.
    # The formerly used 2-2*cos shortcut is mathematically equivalent for
    # unit vectors, but float32 cancellation near zero could reorder tied
    # neighbours and make the selected validation metric differ from the
    # subsequently saved prediction.  Small query batches keep memory bounded.
    distances = np.empty(
        (query.shape[0], train.shape[0]), dtype=np.float32
    )
    distance_batch_size = 16
    for start in range(0, query.shape[0], distance_batch_size):
        stop = min(start + distance_batch_size, query.shape[0])
        delta = query[start:stop, None, :] - train[None, :, :]
        distances[start:stop] = np.linalg.norm(delta, axis=2)
    neighbour_idx = np.argpartition(
        distances, kth=k_eff - 1, axis=1
    )[:, :k_eff]
    neighbour_distances = np.take_along_axis(
        distances, neighbour_idx, axis=1
    )
    order = np.argsort(neighbour_distances, axis=1)
    return (
        np.take_along_axis(neighbour_idx, order, axis=1),
        np.take_along_axis(neighbour_distances, order, axis=1),
    )


def aggregate_prepared_neighbours(
    neighbour_idx,
    neighbour_distances,
    train_values,
    k,
    T,
):
    """Temperature-weighted aggregation using cached nearest neighbours."""
    if T <= 0:
        raise ValueError("eKNN temperature must be positive")
    k_eff = min(int(k), neighbour_idx.shape[1])
    idx = neighbour_idx[:, :k_eff]
    distances = neighbour_distances[:, :k_eff]
    weights = temperature_weights(distances, T)
    values = np.asarray(train_values)[idx]
    return np.sum(weights * values, axis=1)


def multi_k_ensemble(h_test_norm, h_train_norm, residual_train,
                     k_list=(5, 10, 20, 50), T=0.1):
    """Multi-k ensemble (identical to ekNN v2)."""
    corrections = []
    for k in k_list:
        c = knn_correct(h_test_norm, h_train_norm, residual_train, k, T)
        corrections.append(c)
    return np.mean(corrections, axis=0)


def knn_label_predict(h_test_norm, h_train_norm, y_train_norm, k, T):
    """k-NN temperature-weighted label prediction (baseline)."""
    M = h_test_norm.shape[0]
    N_train = h_train_norm.shape[0]
    if N_train == 0:
        raise ValueError("Cannot run label-kNN with an empty training index")
    k_eff = min(k, N_train)
    preds = np.zeros(M)
    for j in range(M):
        dists = np.linalg.norm(h_train_norm - h_test_norm[j], axis=1)
        nn_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
        nn_dists = dists[nn_idx]
        nn_labels = y_train_norm[nn_idx]
        weights = temperature_weights(nn_dists, T)
        preds[j] = np.dot(weights, nn_labels)
    return preds



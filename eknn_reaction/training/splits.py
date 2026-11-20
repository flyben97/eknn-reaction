"""Shared splits helpers; numerics preserved from corrected v0.2.1."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def make_selection_split(train_graphs, ds, args, seed):
    """Return fit/validation indices and an audit record for the inner split."""
    n_train_total = len(train_graphs)
    if n_train_total < 2:
        raise ValueError("Need at least two valid training graphs for a validation split")

    if not args.group_val:
        n_val = max(1, int(n_train_total * args.val_frac))
        if n_val >= n_train_total:
            raise ValueError("Validation split consumed all training rows; reduce --val-frac")
        indices = np.random.RandomState(seed).permutation(n_train_total)
        val_idx = indices[:n_val].tolist()
        val_set = set(val_idx)
        train_idx = [i for i in range(n_train_total) if i not in val_set]
        return train_idx, val_idx, {
            "mode": "row_random",
            "group_column": None,
            "n_groups_total": None,
            "n_groups_train": None,
            "n_groups_val": None,
            "overlapping_groups": None,
        }

    if args.random_split:
        raise ValueError(
            "--group-val is not supported with --random-split because that mode "
            "does not retain a stable source-row/group mapping"
        )
    if not ds.group_col:
        raise ValueError(
            f"Dataset '{ds.name}' has no group_col; cannot use --group-val"
        )

    source_train = pd.read_csv(ds.train_csv)
    group_values = []
    for graph in train_graphs:
        source_idx = int(graph.reaction_idx)
        if source_idx < 0 or source_idx >= len(source_train):
            raise IndexError(
                f"Graph reaction_idx={source_idx} is outside train CSV with {len(source_train)} rows"
            )
        group_values.append(str(source_train.iloc[source_idx][ds.group_col]))
    group_values = np.asarray(group_values, dtype=object)
    n_groups_total = int(np.unique(group_values).size)
    if n_groups_total < 2:
        raise ValueError(
            f"Need at least two groups in '{ds.group_col}' for group validation; got {n_groups_total}"
        )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.val_frac, random_state=seed
    )
    train_idx_arr, val_idx_arr = next(
        splitter.split(np.zeros(n_train_total), groups=group_values)
    )
    train_idx = train_idx_arr.tolist()
    val_idx = val_idx_arr.tolist()
    train_groups = set(group_values[train_idx])
    val_groups = set(group_values[val_idx])
    overlap = train_groups & val_groups
    if overlap:
        raise RuntimeError(
            f"Inner group-validation leakage in '{ds.group_col}': {len(overlap)} groups overlap"
        )
    return train_idx, val_idx, {
        "mode": "group_disjoint",
        "group_column": ds.group_col,
        "n_groups_total": n_groups_total,
        "n_groups_train": int(len(train_groups)),
        "n_groups_val": int(len(val_groups)),
        "overlapping_groups": 0,
    }



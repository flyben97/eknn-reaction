#!/usr/bin/env python3
"""Build the role-tuple graph cache for the HiTEA GNN development track.

Every row of the frozen HiTEA development modeling table is converted to a
disjoint molecular reaction graph over the 8 role SMILES columns, exactly as
``eknn_reaction.models.graph_encoder.smiles_to_graph`` does (Laplacian PE on
the assembled disjoint graph, pe_k=3).  Rows share role tuples whenever their
chemistry is identical and only the conditions differ, so graphs are cached
per unique tuple and the cache stores:

- ``role_tuple_graphs.pt``  key -> PyG Data (the only graph source downstream)
- ``role_tuple_index.csv``  key, role columns, rows using the tuple
- ``cache_manifest.json``   SHA-256 inputs, counts, protocol guards

The cache contains reaction structure only - no labels are read and none are
stored.  Rows whose role SMILES all fail to parse cannot be encoded; the
builder reports the count and the downstream training script asserts it is 0
for the frozen table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eknn_reaction.models.graph_encoder import smiles_to_graph

ROLE_COLUMNS = (
    "reactant_1_smiles",
    "reactant_2_smiles",
    "reactant_3_smiles",
    "catalyst_1_smiles_1",
    "catalyst_1_smiles_2",
    "catalyst_2_smiles_1",
    "catalyst_2_smiles_2",
    "product_smiles",
)
PE_K = 3


def canonical_role_tuple(row) -> tuple[str, ...]:
    """Missing or blank SMILES become the empty string (skipped by the
    graph builder while preserving the original role index)."""
    values = []
    for column in ROLE_COLUMNS:
        value = row[column]
        values.append(value if isinstance(value, str) and value.strip() else "")
    return tuple(values)


def tuple_key(roles: tuple[str, ...]) -> str:
    payload = "\x1f".join(roles)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_one(payload: tuple[str, tuple[str, ...]]):
    key, roles = payload
    graph = smiles_to_graph(list(roles), add_laplacian_pe=True, pe_k=PE_K)
    return key, graph


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/hitea_retrieval_shift/v1")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/hitea_gnn_development/graph_cache"),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Smoke-test only: use the first N table rows",
    )
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    table_path = args.data_root / "development_modeling_table.csv"
    table = pd.read_csv(table_path, low_memory=False)
    if args.limit_rows:
        table = table.head(args.limit_rows).copy()
    missing = [column for column in ROLE_COLUMNS if column not in table.columns]
    if missing:
        raise KeyError(f"Role columns absent from the modeling table: {missing}")
    if "Yield" not in table.columns or "record_id" not in table.columns:
        raise KeyError("Modeling table must carry record_id and Yield columns")

    tuples = [
        canonical_role_tuple(row)
        for row in table[list(ROLE_COLUMNS)].astype(object).to_dict("records")
    ]
    key_of = {tuple_key(roles): roles for roles in tuples}
    if len(key_of) != len(set(tuples)):
        raise AssertionError("Tuple-key collision detected")
    keys = [tuple_key(roles) for roles in tuples]
    table["_graph_key"] = keys

    unique_payload = sorted(key_of.items())
    print(f"{len(table)} rows, {len(unique_payload)} unique role tuples", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        built = list(pool.map(build_one, unique_payload, chunksize=16))
    graphs = dict(built)
    failed = [key for key, graph in graphs.items() if graph is None]
    rows_without_graph = int(table["_graph_key"].isin(failed).sum())
    if rows_without_graph:
        print(
            f"WARNING: {rows_without_graph} rows have no parseable role SMILES "
            f"({len(failed)} tuples); they cannot be GNN-encoded",
            flush=True,
        )

    graph_path = args.output / "role_tuple_graphs.pt"
    torch.save(graphs, graph_path)

    index = (
        table.groupby("_graph_key", sort=True)
        .size()
        .rename("n_rows")
        .reset_index()
        .rename(columns={"_graph_key": "graph_key"})
    )
    role_frame = pd.DataFrame(
        sorted({t for t in tuples}),
        columns=list(ROLE_COLUMNS),
    )
    role_frame["graph_key"] = [tuple_key(roles) for roles in role_frame.itertuples(index=False)]
    index = index.merge(role_frame, on="graph_key", validate="one_to_one")
    index_path = args.output / "role_tuple_index.csv"
    index.to_csv(index_path, index=False)

    manifest = {
        "protocol": "hitea-gnn-role-tuple-graph-cache-v1",
        "contains_labels": False,
        "data_root": str(args.data_root),
        "modeling_table_sha256": sha256_file(table_path),
        "n_rows": int(len(table)),
        "n_unique_role_tuples": int(len(role_frame)),
        "n_tuples_with_graph": int(len(graphs) - len(failed)),
        "n_tuples_without_graph": int(len(failed)),
        "n_rows_without_graph": rows_without_graph,
        "role_columns": list(ROLE_COLUMNS),
        "laplacian_pe_k": PE_K,
        "graph_builder": "eknn_reaction.models.graph_encoder.smiles_to_graph",
        "graph_cache_sha256": sha256_file(graph_path),
        "role_tuple_index_sha256": sha256_file(index_path),
        "wall_time_seconds": float(time.time() - started),
    }
    (args.output / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

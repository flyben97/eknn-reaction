"""Shared cli helpers; numerics preserved from corrected v0.2.1."""
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Reaction GNN + eKNN residual transfer")
    p.add_argument("--dataset", type=str, default="cpac_test_sub")
    p.add_argument("--train-csv", help="Custom training CSV; use with --test-csv and --smiles-cols")
    p.add_argument("--test-csv", help="Custom held-out CSV")
    p.add_argument("--smiles-cols", nargs="+", help="Input SMILES columns for custom CSVs")
    p.add_argument("--target-col", help="Measured target column for custom CSVs")
    p.add_argument("--group-col", help="Optional custom grouping column")
    p.add_argument("--save-ckpt", type=str, default="",
                   help="Path to save checkpoint after training")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--min-epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="Linear LR warmup epochs (0 = no warmup)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "sgd"],
                   help="Optimizer: adamw or sgd (SGD+momentum often generalizes better on small data)")
    p.add_argument("--momentum", type=float, default=0.9,
                   help="SGD momentum (only used with --optimizer sgd)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--select-checkpoint-on-val", action="store_true",
                   help="Select/early-stop the selection-stage GNN checkpoint on held-out validation R2")
    p.add_argument("--val-check-interval", type=int, default=5,
                   help="Epoch interval for selection-stage validation checkpointing")
    p.add_argument("--swa-start", type=int, default=0,
                   help="Epoch to start Stochastic Weight Averaging (0 = disabled). SWA finds flatter minima for better generalization.")
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--ff-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mpnn", action="store_true",
                   help="Use pure MPNN (GINE only, no global MHA)")
    p.add_argument("--attn-pool", action="store_true",
                   help="Use attention pooling instead of mean+max")
    p.add_argument("--readout", type=str, default="mean_max",
                   choices=["mean_max", "mean_max_sum", "mean", "sum", "attn"],
                   help="Graph pooling used before condition fusion")
    p.add_argument("--jk", action="store_true",
                   help="Use Jumping Knowledge (concat all layer outputs)")
    p.add_argument("--gat", action="store_true",
                   help="Use GATConv (local attention) instead of GPS/MPNN")
    p.add_argument("--graph-transformer", action="store_true",
                   help="Use pure Graph Transformer (no GNN, global attn only)")
    p.add_argument("--edge-transformer", action="store_true",
                   help="Use Edge-Augmented Transformer (edge features gate node features)")
    p.add_argument("--graphormer", action="store_true",
                   help="Use Graphormer (degree centrality + edge-aware attention)")
    p.add_argument("--deep-gps", action="store_true",
                   help="Use Deep GPS (GINE + Global Attn + deeper FFN)")
    p.add_argument("--light-gps", action="store_true",
                   help="Use Light GPS (high dropout + single FFN)")
    p.add_argument("--backbone", type=str, default="gps",
                   choices=["gps","mpnn","gat","graph_transformer","edge_transformer",
                            "graphormer","deep_gps","light",
                            "gcn","sage","gin","gatv2","transformer_conv",
                            "pna","general_conv","gen_conv","film",
                            "gated_graph","res_gated","dmpnn","egconv"],
                   help="GNN backbone type (default: gps)")
    p.add_argument("--virtual-node", action="store_true",
                   help="Add virtual node connected to all atoms (global info without attention)")
    p.add_argument("--pe-dim", type=int, default=3,
                   help="Laplacian PE dimension (default: 3)")
    p.add_argument("--cache-graphs", action="store_true", default=True,
                   help="Cache built graphs to disk (default: True)")
    p.add_argument("--no-cache-graphs", dest="cache_graphs", action="store_false")
    p.add_argument("--graph-cache-dir", type=str, default="",
                   help="Optional shared graph-cache directory for reproducible comparisons")
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic PyTorch/CUDA kernels where available")
    p.add_argument("--random-split", action="store_true",
                   help="Merge train+test CSVs, shuffle randomly, re-split 70/30")
    p.add_argument("--random-split-seed", type=int, default=42,
                   help="Seed for random split (default: 42)")
    p.add_argument("--random-split-frac", type=float, default=0.7,
                   help="Train fraction for random split (default: 0.7)")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of training data held out for validation (default: 0.15)")
    p.add_argument("--group-val", action="store_true",
                   help="Make the inner validation split group-disjoint using the dataset group_col")
    p.add_argument("--use-condition-features", "--condition-fusion",
                   dest="use_condition_features", action="store_true",
                   help="Fuse dataset-defined numerical reaction conditions with the pooled GNN embedding")
    p.add_argument("--condition-fusion-mode", type=str, default="gated_add",
                   choices=["gated_add", "concat_residual", "cross_attention"],
                   help="How to combine graph and condition embeddings (default: gated_add)")
    p.add_argument("--include-categorical-conditions", action="store_true",
                   help="Also one-hot encode registered categorical condition columns (off by default; select on validation)")
    p.add_argument("--condition-categorical-embeddings", action="store_true",
                   help="Encode registered categorical conditions as fold-fitted learned tokens; requires --condition-fusion-mode cross_attention")
    p.add_argument("--condition-token-layers", type=int, default=2,
                   help="Number of component/condition Transformer layers for cross_attention fusion (default: 2)")
    p.add_argument("--condition-node-film", action="store_true",
                   help="FiLM-modulate atom embeddings with condition features before GNN message passing")
    p.add_argument("--component-readout", action="store_true",
                   help="Use role-aware per-component pooling before the reaction-level readout")
    p.add_argument("--rolewise-readout", action="store_true",
                   help="Add a compact fixed-role mean/max/sum GNN readout to the global graph pool")
    p.add_argument("--role-interaction-readout", action="store_true",
                   help="Replace global pooling with a condition-modulated molecular-role GNN")
    p.add_argument("--role-interaction-layers", type=int, default=2,
                   help="Reaction-level role-message-passing layers (default: 2)")
    p.add_argument("--role-concat-readout", action="store_true",
                   help="Use fixed-role molecular GNN tokens, pair interactions, and an explicit condition vector")
    p.add_argument(
        "--role-factorized-readout",
        action="store_true",
        help=(
            "Use compact role-specific main effects and low-rank molecular "
            "pair interactions explicitly modulated by conditions"
        ),
    )
    p.add_argument(
        "--role-factorized-retrieval",
        action="store_true",
        help=(
            "Use the fixed-order concatenation of per-role GNN tokens as "
            "the eKNN reaction embedding; requires --role-factorized-readout"
        ),
    )
    p.add_argument("--component-identity-embeddings", action="store_true",
                   help="Fuse fold-fitted per-role molecular identity tokens into the GNN embedding")
    p.add_argument("--component-identity-dim", type=int, default=32,
                   help="Embedding width for each fold-fitted molecular identity token")
    p.add_argument(
        "--component-identity-roles",
        type=str,
        default="",
        help=("Comma-separated molecular roles receiving fold-fitted identity "
              "tokens (default when enabled: every registered SMILES role)"),
    )
    p.add_argument("--retrieval-dim", type=int, default=0,
                   help="Optional learned GNN projection dimension used by eKNN (0 = pooled embedding)")
    p.add_argument("--metric-loss-weight", type=float, default=0.0,
                   help="Weight of continuous supervised contrastive loss on the eKNN embedding")
    p.add_argument("--metric-label-temperature", type=float, default=0.5,
                   help="Target-distance temperature for metric learning")
    p.add_argument("--metric-embedding-temperature", type=float, default=0.1,
                   help="Cosine-similarity temperature for metric learning")
    p.add_argument("--eknn-condition-scales", type=float, nargs="+", default=[0.0],
                   help="Validation grid for the explicit condition-vector weight in eKNN distance")
    p.add_argument("--eknn-residual-scales", type=float, nargs="+", default=[1.0],
                   help="Validation grid for the magnitude of the transferred eKNN residual")
    p.add_argument(
        "--eknn-base-alignment-scales",
        type=float,
        nargs="+",
        default=[0.0],
        help=("Validation grid for aligning the query baseline to the "
              "neighbour-weighted training baseline (0 = ordinary residual eKNN)"),
    )
    p.add_argument("--validation-only", action="store_true",
                   help="Train/evaluate only the selection split; never construct or evaluate an outer test model")
    p.add_argument(
        "--predict-outer-unscored",
        action="store_true",
        help=(
            "With --validation-only, use the validation-selected model and "
            "fixed eKNN controls to write outer-query predictions without "
            "computing any outer-test metric or exporting its target"
        ),
    )
    return p.parse_args()


def parse_component_identity_roles(raw_roles, ds):
    """Validate a comma-separated subset of registered molecular roles."""
    if not raw_roles:
        return list(ds.smiles_cols)
    roles = [value.strip() for value in str(raw_roles).split(",") if value.strip()]
    unknown = [value for value in roles if value not in ds.smiles_cols]
    if unknown:
        raise ValueError(
            "--component-identity-roles contains unregistered roles: "
            f"{unknown}; available={list(ds.smiles_cols)}"
        )
    if len(roles) != len(set(roles)):
        raise ValueError("--component-identity-roles must not contain duplicates")
    return roles



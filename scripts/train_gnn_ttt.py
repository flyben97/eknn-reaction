#!/usr/bin/env python3
"""Train a GNN and adapt its residual predictions with a per-substrate MLP.

Each adapter is fitted on training reactions from the corresponding substrate group.
"""
from __future__ import annotations
import argparse, os, sys, time, copy
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as PyGDataLoader

from eknn_reaction.config import ReactionConfig
from eknn_reaction.data import get_dataset
from eknn_reaction.data.dataset import Standardizer
from eknn_reaction.metrics import mae, r2
from eknn_reaction import utils
from eknn_reaction.models.graph_encoder import (build_reaction_graphs, ReactionGraphDataset, collate_graphs)
from eknn_reaction.models.reaction_gnn import ReactionGNN

LOG = utils.get_logger()

# ── TTT Adapter ──
class TTTAdapter(nn.Module):
    """Small MLP adapter for residual correction."""
    def __init__(self, d_model=256, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
    def forward(self, h): return self.net(h).squeeze(-1)
    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="bhc_test1")
    p.add_argument("--backbone", type=str, default="gcn")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--ff-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--ttt-lr", type=float, default=0.01)
    p.add_argument("--ttt-steps", type=int, default=500)
    p.add_argument("--ttt-hidden", type=int, default=64)
    p.add_argument("--out-dir", type=str, default="results_gnn_ttt")
    return p.parse_args()


@torch.no_grad()
def get_embeddings(model, graphs, device, batch_size=128):
    """Extract GNN embeddings (batched — preserves input order)."""
    model.eval(); hs, ys = [], []
    loader = PyGDataLoader(ReactionGraphDataset(graphs, np.zeros(len(graphs))),
                           batch_size=batch_size, shuffle=False)
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        hs.append(out["embedding"].cpu().numpy())
        ys.append(out["yield"].squeeze(-1).cpu().numpy())
    return np.concatenate(hs), np.concatenate(ys)


def run_one(ds_name, args, device, out_dir, cfg):
    ds = get_dataset(ds_name)
    LOG.info("="*60)
    LOG.info("  GNN + Residual TTT | %s backbone=%s d=%d L=%d seed=%d",
             ds_name, args.backbone, args.d_model, args.n_layers, args.seed)

    # Load data
    df_train = pd.read_csv(ds.train_csv); df_test = pd.read_csv(ds.test_csv)
    LOG.info("Data: train=%d test=%d group_col=%s", len(df_train), len(df_test), ds.group_col)

    # Build graphs (with caching)
    cache_dir = os.path.join(out_dir, ".graph_cache")
    cache_file = os.path.join(cache_dir, f"{ds_name}_group_graphs.pkl")
    if os.path.exists(cache_file):
        import pickle
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        train_graphs, train_targets = cache['train_graphs'], cache['train_targets']
        test_graphs, test_targets = cache['test_graphs'], cache['test_targets']
        std = Standardizer(cache['yield_mean'], cache['yield_std'])
    else:
        train_graphs, train_targets = build_reaction_graphs(df_train, ds.smiles_cols, ds.target_col, add_laplacian_pe=True, pe_k=3, verbose=False)
        test_graphs, test_targets = build_reaction_graphs(df_test, ds.smiles_cols, ds.target_col, add_laplacian_pe=True, pe_k=3, verbose=False)
        # filter None
        valid_tr = [i for i,g in enumerate(train_graphs) if g is not None]
        valid_te = [i for i,g in enumerate(test_graphs) if g is not None]
        train_graphs = [train_graphs[i] for i in valid_tr]; train_targets = train_targets[valid_tr]
        test_graphs = [test_graphs[i] for i in valid_te]; test_targets = test_targets[valid_te]
        std = Standardizer(float(np.nanmean(train_targets)), float(np.nanstd(train_targets)))
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            import pickle
            with open(cache_file, 'wb') as f:
                pickle.dump({'train_graphs':train_graphs,'train_targets':train_targets,'test_graphs':test_graphs,'test_targets':test_targets,'yield_mean':std.mean,'yield_std':std.std}, f)

    # Also filter df to match
    df_train = df_train.iloc[valid_tr].reset_index(drop=True) if 'valid_tr' in dir() else df_train
    df_test = df_test.iloc[valid_te].reset_index(drop=True) if 'valid_te' in dir() else df_test
    LOG.info("Valid: train=%d test=%d", len(train_graphs), len(test_graphs))

    # Train/val split
    n_train = len(train_graphs); n_val = max(1, int(n_train*0.15))
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_train)
    val_idx_set = set(indices[:n_val].tolist()); val_idx_list = indices[:n_val].tolist()
    train_idx = [i for i in range(n_train) if i not in val_idx_set]

    train_subset = ReactionGraphDataset([train_graphs[i] for i in train_idx], train_targets[train_idx])
    val_subset = ReactionGraphDataset([train_graphs[i] for i in val_idx_list], train_targets[val_idx_list])
    full_train_dataset = ReactionGraphDataset(train_graphs, train_targets)
    train_loader = PyGDataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    # Model
    model = ReactionGNN(d_model=args.d_model, n_layers=args.n_layers, n_heads=4,
                     ff_dim=args.ff_dim, dropout=args.dropout, pe_dim=3,
                     backbone=args.backbone, use_global_attn=False, readout="mean_max")
    model.to(device); n_params = sum(p.numel() for p in model.parameters())
    LOG.info("Params: %d", n_params)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    def lr_lambda(ep):
        if ep < args.warmup_epochs: return (ep+1)/args.warmup_epochs
        p = (ep-args.warmup_epochs)/max(args.epochs-args.warmup_epochs,1)
        return 0.01+0.99*(1+np.cos(np.pi*p))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def norm_y(y): return (y-std.mean)/(std.std+1e-8)

    best_val, best_state = float('inf'), None; patience, final_epoch = 0, 0
    for ep in range(args.epochs):
        model.train(); tl, nb = 0.0, 0
        for batch in train_loader:
            batch=batch.to(device); yn=norm_y(batch.y.squeeze(-1))
            opt.zero_grad(); loss=F.mse_loss(model(batch)["yield"].squeeze(-1), yn)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step(); tl+=loss.item(); nb+=1
        scheduler.step()
        model.eval(); vl, nv = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch=batch.to(device); yn=norm_y(batch.y.squeeze(-1))
                vl+=F.mse_loss(model(batch)["yield"].squeeze(-1),yn,reduction='sum').item(); nv+=batch.num_graphs
        val_loss=vl/max(nv,1)
        if val_loss<best_val: best_val=val_loss; best_state=copy.deepcopy(model.state_dict()); patience=0
        else: patience+=1
        final_epoch=ep+1
        if ep%20==0: LOG.info("ep %d: train=%.4f val=%.4f",ep,tl/max(nb,1),val_loss)
        if patience>=20 and final_epoch>=30: break

    model.load_state_dict(best_state); LOG.info("Trained: val=%.4f ep=%d",best_val,final_epoch)

    # Static evaluation
    h_test, yp_test = get_embeddings(model, test_graphs, device)
    yp_denorm = std.denormalize(yp_test)
    static_r2 = float(r2(yp_denorm, test_targets)); static_mae = float(mae(yp_denorm, test_targets))
    LOG.info("Static: R²=%.4f MAE=%.4f", static_r2, static_mae)

    # Determine actual embedding dimension from model output
    embed_dim = h_test.shape[1]
    LOG.info("Embedding dim: %d", embed_dim)

    # ── Residual TTT evaluation (per-substrate: train ONE adapter per group,
    #    apply to all test reactions of that group — the correct per-substrate
    #    TTT and ~Nx faster than the old per-reaction retrain) ──
    group_col = ds.group_col
    # cache ALL train embeddings once (batched), avoid per-group re-encoding
    h_train_all, yp_train_all = get_embeddings(model, train_graphs, device)
    resid_train_all = std.normalize(train_targets) - yp_train_all
    tr_groups = df_train[group_col].to_numpy()
    te_groups = df_test[group_col].to_numpy()

    n_ttt_nomatch, diverged_count = 0, 0
    n_test_corrected, total_substrates = 0, 0
    all_yp_ttt = yp_test.copy()

    for substrate in np.unique(te_groups):
        test_pos = np.where(te_groups == substrate)[0]
        train_match = np.where(tr_groups == substrate)[0]
        train_match = train_match[train_match < len(train_graphs)]
        total_substrates += 1
        if len(train_match) == 0:
            n_ttt_nomatch += len(test_pos); continue

        h_warm = torch.from_numpy(h_train_all[train_match].astype(np.float32)).to(device)
        res_warm = torch.from_numpy(resid_train_all[train_match].astype(np.float32)).to(device)

        adapter = TTTAdapter(d_model=embed_dim, hidden=args.ttt_hidden).to(device)
        adapter.reset()
        opt_ttt = torch.optim.SGD(adapter.parameters(), lr=args.ttt_lr, momentum=0.9)

        diverged = False
        for step in range(args.ttt_steps):
            opt_ttt.zero_grad()
            loss = F.mse_loss(adapter(h_warm), res_warm)
            if torch.isnan(loss) or torch.isinf(loss):
                diverged = True; break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt_ttt.step()

        if diverged:
            diverged_count += 1; continue  # leave static prediction for this group

        h_test_grp = torch.from_numpy(h_test[test_pos].astype(np.float32)).to(device)
        with torch.no_grad():
            corr = adapter(h_test_grp).cpu().numpy()
        all_yp_ttt[test_pos] = yp_test[test_pos] + corr
        n_test_corrected += len(test_pos)
    n_ttt_ok = n_test_corrected

    yp_ttt_denorm = std.denormalize(all_yp_ttt)
    ttt_r2 = float(r2(yp_ttt_denorm, test_targets))
    ttt_mae = float(mae(yp_ttt_denorm, test_targets))
    ttt_delta = ttt_r2 - static_r2
    LOG.info("TTT: R²=%.4f Δ=%.4f | %d/%d substrates ok, %d diverged, %d no-match",
             ttt_r2, ttt_delta, n_ttt_ok, total_substrates, diverged_count, n_ttt_nomatch)

    # Save
    od = os.path.join(out_dir, ds_name); os.makedirs(od, exist_ok=True)
    tag = f"gnn_ttt_{args.backbone}_s{args.seed}_d{args.d_model}"
    utils.dump_json(os.path.join(od, f"{tag}.json"), {
        "dataset": ds_name, "model": "gnn_residual_ttt", "backbone": args.backbone,
        "config": {"d_model":args.d_model,"n_layers":args.n_layers,"ff_dim":args.ff_dim,"dropout":args.dropout},
        "ttt_config": {"lr":args.ttt_lr,"steps":args.ttt_steps,"hidden":args.ttt_hidden},
        "n_params": n_params, "epoch": final_epoch, "seed": args.seed,
        "static": {"r2": static_r2, "mae": static_mae},
        "ttt": {"r2": ttt_r2, "mae": ttt_mae, "delta_r2": ttt_delta},
        "ttt_stats": {"n_ok": n_ttt_ok, "n_diverged": diverged_count, "n_nomatch": n_ttt_nomatch, "n_substrates": total_substrates},
    })
    LOG.info("Done! static=%.4f ttt=%.4f Δ=%.4f", static_r2, ttt_r2, ttt_delta)


def main():
    args = parse_args()
    cfg = ReactionConfig()
    cfg.train.epochs = args.epochs; cfg.train.lr = args.lr; cfg.train.batch_size = args.batch_size
    cfg.train.seed = args.seed
    utils.set_seed(cfg.train.seed)
    device = utils.get_device(cfg.train.device)
    for ds in args.dataset.split(","):
        try: run_one(ds, args, device, args.out_dir, cfg)
        except Exception as e: LOG.error("FAILED %s: %s", ds, e); import traceback; traceback.print_exc()
    LOG.info("All done.")

if __name__ == "__main__": sys.exit(main())

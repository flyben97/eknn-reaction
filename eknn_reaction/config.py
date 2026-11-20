"""Central configuration objects for eKNN reaction prediction.

Adapted from SEDM's config.py.  Uses frozen-ish dataclasses with YAML + CLI support.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List

try:
    import yaml
except Exception:
    yaml = None


@dataclass
class DataConfig:
    """Reaction data settings."""
    csv_path: str = "../Buchwald-Hartwig-Coupling.csv"
    max_len: int = 256                    # max SMILES token sequence length
    batch_size: int = 32
    test_frac: float = 0.3
    val_frac: float = 0.15
    seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture settings.  Shares TTT parameters with SEDM."""
    # v1 params (kept for backward compat)
    vocab_size: int = 200
    pad_idx: int = 0
    n_backbone_layers: int = 2
    # v2/v3 compat
    d_model: int = 256
    mol_dim: int = 768
    enc_layers: int = 6          # Transformer encoder layers (v4)
    enc_heads: int = 8           # Transformer encoder heads (v4)
    enc_ffn: int = 1024          # Transformer encoder FFN dim (v4)
    adapter_heads: int = 4
    adapter_layers: int = 2
    adapter_ffn: int = 256
    # TTT params (shared)
    ttt_heads: int = 4
    ttt_base_lr: float = 1.0
    forget_gamma: float = 0.05
    trust_region: float = 1.0
    dropout: float = 0.1
    use_ttt: bool = True
    # v5 Transformer backbone (no TTTLayer — uses residual adapter instead)
    v5_enc_layers: int = 3       # Transformer encoder layers
    v5_enc_heads: int = 4        # attention heads (d_model must be divisible)
    v5_enc_ffn: int = 512        # FFN hidden dim
    v5_dropout: float = 0.15     # dropout (stronger reg for small-data regime)
    # v6 Multi-Scale Conv1D backbone
    v6_n_blocks: int = 3         # multi-scale Conv1D blocks
    v6_dropout: float = 0.1      # dropout in multi-scale blocks


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.05
    device: str = "cpu"
    amp: bool = False
    log_every: int = 10
    # loss weights
    w_yield: float = 1.0
    w_ssl: float = 0.5
    seed: int = 42


@dataclass
class EvalConfig:
    tta_steps: int = 1
    report_per_group: bool = True


@dataclass
class ReactionConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    out_dir: str = "runs/eknn_reaction"
    exp_name: str = "default"

    def sync(self) -> "ReactionConfig":
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReactionConfig":
        return cls(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            eval=EvalConfig(**d.get("eval", {})),
            out_dir=d.get("out_dir", "runs/eknn_reaction"),
            exp_name=d.get("exp_name", "default"),
        ).sync()

    @classmethod
    def from_yaml(cls, path: str) -> "ReactionConfig":
        if yaml is None:
            raise RuntimeError("pyyaml is required to load YAML configs")
        with open(path, "r") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d or {})

    def save_yaml(self, path: str) -> None:
        if yaml is None:
            raise RuntimeError("pyyaml is required to save YAML configs")
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)


def override_from_dotlist(cfg: ReactionConfig, dotlist: list) -> ReactionConfig:
    """Apply ``a.b=c`` style CLI overrides onto a config in place."""
    d = cfg.to_dict()
    for item in dotlist:
        if "=" not in item:
            raise ValueError(f"override '{item}' is not in key=value form")
        key, val = item.split("=", 1)
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        cur = node.get(parts[-1])
        node[parts[-1]] = _coerce(val, cur)
    return ReactionConfig.from_dict(d)


def _coerce(val: str, like: Any) -> Any:
    if isinstance(like, bool):
        return val.lower() in ("1", "true", "yes", "y")
    if isinstance(like, int):
        return int(val)
    if isinstance(like, float):
        return float(val)
    return val


# Compatibility alias for historical scripts.  New code should use
# ReactionConfig, which describes the actual graph-reaction pipeline.
SERMConfig = ReactionConfig

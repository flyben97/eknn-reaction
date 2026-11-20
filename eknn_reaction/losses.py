"""Loss functions for reaction-outcome prediction.

Supports both v1 (yield + yield_next SSL) and v2 (yield only).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def reaction_loss(
    out: Dict[str, torch.Tensor],
    batch_target: torch.Tensor,
    batch_yield_next: Optional[torch.Tensor] = None,
    w_yield: float = 1.0,
    w_ssl: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Unified loss supporting both v1 and v2 calling conventions.

    v1: reaction_loss(out, y, yn, w_yield, w_ssl)
    v2: reaction_loss(out, y)
    """
    y = F.mse_loss(out["yield"], batch_target)
    if batch_yield_next is not None and "yield_next" in out:
        ssl = F.mse_loss(out["yield_next"], batch_yield_next)
        total = w_yield * y + w_ssl * ssl
        return total, {"yield_loss": y.item(), "ssl": ssl.item(), "total": total.item()}
    return y, {"yield_loss": y.item(), "total": y.item()}


# Compatibility alias for pre-rename callers.
serm_loss = reaction_loss

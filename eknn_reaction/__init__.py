"""eKNN residual-transfer tools for reaction prediction.

The public API reflects the current method: a reaction GNN provides an
embedding and base prediction, and eKNN transfers nearby training residuals.
"""

from .config import ReactionConfig, DataConfig, ModelConfig, TrainConfig, EvalConfig
from .models import ReactionGNN
from .utils import set_seed, get_device, get_logger, save_checkpoint, load_checkpoint

# Backward-compatible aliases for notebooks and checkpoints created before the
# project was renamed from SERM.  New code should use ReactionConfig and
# ReactionGNN.
SERMConfig = ReactionConfig
SERMvGNN = ReactionGNN

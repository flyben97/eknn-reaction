"""Compatibility namespace for the former ``serm`` package name.

The project is now named eKNN Reaction / eKNN residual transfer for reaction
prediction.  New code should import :mod:`eknn_reaction`.  This module maps
the legacy nested import paths to the renamed package so prior notebooks and
frozen scripts continue to run.
"""

from __future__ import annotations

import importlib
import sys

from eknn_reaction import *  # noqa: F401,F403

_MODULES = {
    "config": "eknn_reaction.config",
    "losses": "eknn_reaction.losses",
    "metrics": "eknn_reaction.metrics",
    "utils": "eknn_reaction.utils",
    "data": "eknn_reaction.data",
    "data.dataset": "eknn_reaction.data.dataset",
    "data.datasets": "eknn_reaction.data.datasets",
    "data.tokenizer": "eknn_reaction.data.tokenizer",
    "models": "eknn_reaction.models",
    "models.graph_encoder": "eknn_reaction.models.graph_encoder",
    "models.serm_gnn": "eknn_reaction.models.reaction_gnn",
}

for legacy, canonical in _MODULES.items():
    module = importlib.import_module(canonical)
    sys.modules[f"{__name__}.{legacy}"] = module
    if "." not in legacy:
        setattr(sys.modules[__name__], legacy, module)

#!/usr/bin/env python3
"""Compatibility entry point; canonical implementation is eknn_reaction.training.runner."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eknn_reaction.training import runner as _runner
# Preserve file-loader access as well as normal import/monkeypatch compatibility.
globals().update({k: v for k, v in vars(_runner).items() if not k.startswith("__")})
if __name__ == "__main__":
    raise SystemExit(_runner.main())
else:
    sys.modules[__name__] = _runner

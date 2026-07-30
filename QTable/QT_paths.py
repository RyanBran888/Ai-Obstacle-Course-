"""Makes the repo root and `Architecture/` importable from this folder.

`DQN/env_bridge.py` does this inline. Here several modules need it and import
order is not guaranteed, so the bootstrap lives in one place. Importing this
module has the side effect; that is the whole point of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHITECTURE = ROOT / "Architecture"
DQN_DIR = ROOT / "DQN"

# `DQN/` itself goes on the path because the DQN's own modules import
# `env_bridge` and `room_manifest` flat while importing `DQN.DQN_model` and
# `DQN.DQN_train` as a package. Matching that convention exactly matters: if
# this package imported `DQN.env_bridge` instead, Python would load the bridge a
# second time under a second name, and `CoopEnvBridge` here would be a different
# class object from the one `curriculum.py` uses.
for path in (ROOT, ARCHITECTURE, DQN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

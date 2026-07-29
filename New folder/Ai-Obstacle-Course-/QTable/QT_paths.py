"""Makes `Architecture/` importable from this folder.

`QN/env_bridge.py` does this inline, which works because it is the only module
in that folder that touches `coop_env`. Here two modules need it and import
order is not guaranteed, so the bootstrap lives in one place and both import it
first. Importing this module has the side effect; that is the whole point of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ARCHITECTURE = Path(__file__).resolve().parent.parent / "Architecture"

if str(ARCHITECTURE) not in sys.path:
    sys.path.insert(0, str(ARCHITECTURE))

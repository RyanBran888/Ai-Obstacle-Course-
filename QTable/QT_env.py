"""Tabular view of the DQN's environment.

This is a three-line subclass, and that is the entire design decision.

`DQN/env_bridge.py` is 1695 lines of dynamics: movement, crate pushing, hold
switches, timed doors with re-arming, temporary bridges, reset zones, wipeout
balls, per-agent key ownership, role assignment, BFS navigation, and a reward
function with sixteen constants. An earlier version of this package hand-copied
that logic so the tabular stack could stay torch-free. Within a week the copy
had drifted -- it was still on nine actions when the bridge had moved to six,
still on `R_HAZARD = -1.0` when the bridge said -0.5 -- and a drifted copy does
not produce a comparison, it produces two unrelated numbers.

So the bridge is inherited, not reimplemented. Identical rooms, identical
rewards, identical dynamics; the only thing that changes is what `_obs` hands
back. Where the DQN gets 1325 floats, the table gets a hashable tuple, and the
expensive vector is never built at all.
"""

from __future__ import annotations

from typing import Any, Callable

import QT_paths  # noqa: F401  -- puts the repo root and Architecture on sys.path

from env_bridge import CoopEnvBridge, N_AGENTS  # noqa: E402
from DQN.DQN_model import N_ACTIONS  # noqa: E402

from QT_encoder import StateEncoder  # noqa: E402


class TabularBridge(CoopEnvBridge):
    """`CoopEnvBridge` that emits discrete state keys instead of float vectors.

    Every reward, every mechanic, every room comes from the parent unchanged.
    Only the observation is swapped, because a table cannot index 1325 floats.
    """

    def __init__(
        self,
        *args: Any,
        encoder: StateEncoder | None = None,
        accepts: Callable[[Any], bool] | None = None,
        max_draws: int = 400,
        **kwargs: Any,
    ) -> None:
        # Set before super().__init__ so the first _obs call has an encoder.
        self.encoder = encoder or StateEncoder()
        self.accepts = accepts
        self.max_draws = max_draws
        self.rejected = 0
        super().__init__(*args, **kwargs)

    def reset(self, seed=None):
        """Draw a fresh procedural room, redrawing until `accepts` is satisfied.

        Curriculum stages are lessons, not levels: `stage.config` gets the rough
        shape right and `stage.accepts` insists the room actually contains the
        mechanic being taught. Generating a new room every episode means the
        tables never get to memorise a layout -- every episode is a room they
        have not seen.

        The rejected draws only cost `sess.reset()`, not a full
        `_begin_episode()`, so the expensive cache and navigation build happens
        once per *accepted* room rather than once per attempt.
        """
        if self.accepts is None or seed is not None or self.micro is not None:
            return super().reset(seed)

        for _ in range(self.max_draws):
            self.sess.reset()
            if self.accepts(self.sess.room):
                return self._begin_episode()
            self.rejected += 1

        raise RuntimeError(
            f"no generated room passed accepts() in {self.max_draws} draws -- "
            "the stage config and its predicate disagree"
        )

    def _obs(self, i: int) -> tuple:
        return self.encoder.encode(self, i)

    # `obs_dim` is meaningless for a table; keep the attribute so anything that
    # introspects the bridge does not trip over its absence.
    obs_dim = 0
    n_actions = N_ACTIONS
    n_agents = N_AGENTS

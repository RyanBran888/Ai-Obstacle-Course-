"""The Q-table and the agent that owns one.

`DQN/DQN_model.py` approximates Q(s,a) with a dueling network -- a shared trunk
of (256, 128, 64) splitting into a state value and an action advantage -- trained
by gradient descent on batches drawn from a grouped replay buffer. This file
does the same job with a dictionary and arithmetic.

No torch here. The bridge drags torch in, so this package is not torch-free
overall, but the learner itself is a `defaultdict` of small float lists, and
keeping it plain makes the update rule readable against the textbook version.

What deliberately matches the DQN:

    action masking      Action 4 is illegal where `can_interact` is false, per
                        `POLICY_CONTRACT["mask_invalid_interact"]`.
    route bias          `ROUTE_Q_BIAS` is added to the route action before the
                        argmax, imported from `DQN_model` so the two cannot
                        drift apart.
    Agent surface       act / remember / learn_batch / sync / save / load, so
                        either family can be driven by the same loop.

What is deliberately absent, and why:

    replay buffer       A table has no cross-state interference. Replay exists
                        in DQN so new gradients do not wreck old fits; there are
                        no shared weights here to wreck. `remember()` therefore
                        performs the update immediately and `learn_batch()` is a
                        no-op, kept only so the API lines up.
    target network      Same reason. Nothing shared to destabilise, so
                        bootstrapping off the live table is fine. `sync()` is a
                        no-op.
    dueling heads       Value/advantage decomposition is a generalisation trick
                        for function approximators. A table stores every entry
                        independently, so there is nothing to share.

Three update rules are selectable, because "the Q-table one" is not a single
algorithm:

    q_learning      off-policy. Targets the best next action whether or not the
                    agent takes it. Learns the optimal path, then walks it right
                    alongside the lava, because exploration is not its problem.
    sarsa           on-policy. Targets the action actually taken next, so the
                    cost of exploring is priced in. Gives the cautious policy.
    expected_sarsa  on-policy, averaged over the epsilon-greedy distribution
                    rather than sampled from it. Same bias as SARSA with less
                    variance; the best default of the three.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import Any, Hashable, Iterable

import QT_paths  # noqa: F401  -- puts the repo root and Architecture on sys.path

from DQN.DQN_model import N_ACTIONS  # noqa: E402

from QT_encoder import policy_scores, valid_actions  # noqa: E402

UPDATE_RULES = ("q_learning", "sarsa", "expected_sarsa")


class QTable:
    """Sparse action-value table.

    Unvisited entries are not stored; they read as `init` on demand. Setting
    `init` above any achievable return ("optimistic initialisation") is a cheap
    explorer -- every untried action looks better than every tried one, so the
    agent sweeps its options before settling, with no schedule to tune.
    """

    __slots__ = ("n_actions", "init", "q", "_rng")

    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        init: float = 0.0,
        rng: random.Random | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.init = float(init)
        self.q: defaultdict[Hashable, list[float]] = defaultdict(self._row)
        self._rng = rng or random.Random()

    def _row(self) -> list[float]:
        return [self.init] * self.n_actions

    # -- reading -----------------------------------------------------------

    def values(self, key: Hashable) -> list[float]:
        return self.q[key]

    def value(self, key: Hashable, action: int) -> float:
        return self.q[key][action]

    def best_action(self, key: tuple) -> int:
        """Argmax over route-biased, mask-filtered scores. Ties broken randomly.

        The tie-break is not cosmetic. Every row starts uniform, so a
        deterministic argmax would send every agent north out of every unvisited
        state and quietly destroy exploration.
        """
        scores = policy_scores(self.q[key], key)
        top = max(scores)
        best = [a for a, s in enumerate(scores) if s == top]
        return best[0] if len(best) == 1 else self._rng.choice(best)

    def best_value(self, key: tuple) -> float:
        """Max Q over *legal* actions, without the route bias.

        The bias shapes behaviour; it must not leak into the learning target or
        the table would be bootstrapping off a number it can never actually
        receive.
        """
        legal = valid_actions(key)
        row = self.q[key]
        return max(row[a] for a in legal)

    def act(self, key: tuple, eps: float) -> int:
        if self._rng.random() < eps:
            return self._rng.choice(valid_actions(key))
        return self.best_action(key)

    def policy_probs(self, key: tuple, eps: float) -> dict[int, float]:
        """Epsilon-greedy distribution over legal actions -- what expected SARSA averages."""
        legal = valid_actions(key)
        scores = policy_scores(self.q[key], key)
        top = max(scores[a] for a in legal)
        greedy = [a for a in legal if scores[a] == top]

        share = eps / len(legal)
        probs = {a: share for a in legal}
        for a in greedy:
            probs[a] += (1.0 - eps) / len(greedy)
        return probs

    # -- writing -----------------------------------------------------------

    def update(self, key: Hashable, action: int, delta: float) -> None:
        self.q[key][action] += delta

    # -- housekeeping ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.q)

    @property
    def entries(self) -> int:
        return len(self.q) * self.n_actions

    def coverage(self) -> float:
        if not self.q:
            return 0.0
        touched = sum(1 for row in self.q.values() for v in row if v != self.init)
        return touched / self.entries


class TabularAgent:
    """One Q-table bot.

    The public surface matches `DQN_train.Agent` so a training loop can drive
    either family without knowing which it has.
    """

    def __init__(
        self,
        n_actions: int = N_ACTIONS,
        alpha: float = 0.25,
        gamma: float = 0.99,
        rule: str = "expected_sarsa",
        init: float = 0.0,
        lambda_: float = 0.8,
        alpha_min: float = 0.02,
        alpha_decay: float = 0.9995,
        rng: random.Random | None = None,
    ) -> None:
        if rule not in UPDATE_RULES:
            raise ValueError(f"rule must be one of {UPDATE_RULES}, got {rule!r}")

        self.n_actions = n_actions
        self.gamma = gamma
        self.rule = rule
        self.lambda_ = lambda_

        # Eligibility traces multiply the effective step size by roughly
        # 1/(1 - gamma*lambda). At gamma=0.99, lambda=0.8 that is 4.8x, so an
        # alpha of 0.3 becomes an effective 1.44 -- above 1.0, which overshoots
        # every target and oscillates instead of converging.
        #
        # This cost real debugging time in an earlier version: on one fixed
        # room, alpha_decay 0.9995 scored 100% and 0.9998 scored 0%. Neither
        # number was wrong; the faster decay simply dropped under the stability
        # threshold sooner, by luck. Cap it where the maths says it belongs
        # rather than leaving the landmine in the config.
        self.alpha_cap = 0.9 * (1.0 - gamma * lambda_) if lambda_ > 0.0 else 1.0
        self.alpha = min(alpha, self.alpha_cap)
        self.alpha_min = min(alpha_min, self.alpha_cap)
        self.alpha_decay = alpha_decay
        self.clamped = self.alpha < alpha

        # Robbins-Monro: a constant step size never converges, it hovers. With
        # traces the hovering is large enough to destroy the policy outright, so
        # refuse the setting rather than fail mysteriously.
        if alpha_decay >= 1.0 and lambda_ > 0.0:
            raise ValueError(
                "alpha_decay must be < 1.0 when lambda_ > 0; a constant step "
                "size does not converge under eligibility traces. Try 0.9995."
            )

        self.rng = rng or random.Random()
        self.table = QTable(n_actions, init=init, rng=self.rng)
        self._traces: dict[Hashable, list[float]] = {}
        self.updates = 0

    # -- acting ------------------------------------------------------------

    def act(self, key: tuple, eps: float = 0.0) -> int:
        return self.table.act(key, eps)

    def best_actions(self, keys: Iterable[tuple]) -> list[int]:
        return [self.table.best_action(k) for k in keys]

    # -- learning ----------------------------------------------------------

    def start_episode(self) -> None:
        self._traces.clear()

    def remember(
        self,
        key: tuple,
        action: int,
        reward: float,
        next_key: tuple,
        done: bool,
        next_action: int | None = None,
        eps: float = 0.0,
    ) -> float:
        """One temporal-difference update. Returns |TD error|.

        Named `remember` to match `DQN_train.Agent`, but a table learns online,
        so the update happens here rather than being queued for a later batch.
        `next_action` is read only by `sarsa`, `eps` only by `expected_sarsa`;
        passing both always is harmless and lets the caller switch rules without
        changing its call site.
        """
        bootstrap = 0.0 if done else self._bootstrap(next_key, next_action, eps)
        target = reward + self.gamma * bootstrap
        error = target - self.table.value(key, action)

        if self.lambda_ > 0.0:
            self._trace_update(key, action, error, next_key, next_action)
        else:
            self.table.update(key, action, self.alpha * error)

        self.updates += 1
        return abs(error)

    #: Kept so `TabularAgent` is drop-in for `DQN_train.Agent`. A table has no
    #: replay buffer and no target network, so these have nothing to do.
    def learn_batch(self, batch_size: int = 0) -> None:
        return None

    def sync(self) -> None:
        return None

    def _bootstrap(self, next_key: tuple, next_action: int | None, eps: float) -> float:
        if self.rule == "q_learning":
            return self.table.best_value(next_key)
        if self.rule == "sarsa":
            if next_action is None:
                return self.table.best_value(next_key)
            return self.table.value(next_key, next_action)

        probs = self.table.policy_probs(next_key, eps)
        row = self.table.values(next_key)
        return sum(p * row[a] for a, p in probs.items())

    def _trace_update(
        self,
        key: Hashable,
        action: int,
        error: float,
        next_key: tuple,
        next_action: int | None,
    ) -> None:
        """Watkins-style replacing traces.

        Credit flows back to everything recently visited instead of one step at
        a time, which is what makes a single reward at the exit reachable in a
        sensible number of episodes -- tabular's answer to the problem the
        replay buffer solves for the DQN. The trace is cut when the agent takes a
        non-greedy action under `q_learning`, since past that point the return
        no longer reflects the greedy policy the rule is estimating.
        """
        row = self._traces.setdefault(key, [0.0] * self.n_actions)
        row[action] = 1.0  # replacing, not accumulating

        decay = self.gamma * self.lambda_
        dead: list[Hashable] = []
        for state, trace in self._traces.items():
            values = self.table.values(state)
            for a, e in enumerate(trace):
                if e > 1e-4:
                    values[a] += self.alpha * error * e
                    trace[a] = e * decay
                else:
                    trace[a] = 0.0
            if not any(trace):
                dead.append(state)
        for state in dead:
            del self._traces[state]

        if self.rule == "q_learning" and next_action is not None:
            if next_action != self.table.best_action(next_key):
                self._traces.clear()

    def decay_alpha(self) -> None:
        self.alpha = max(self.alpha_min, min(self.alpha_cap, self.alpha * self.alpha_decay))

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "n_actions": self.n_actions,
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "rule": self.rule,
                    "lambda": self.lambda_,
                    "init": self.table.init,
                    "updates": self.updates,
                    "q": dict(self.table.q),
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, path: str) -> "TabularAgent":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        self.n_actions = blob["n_actions"]
        self.alpha = blob["alpha"]
        self.gamma = blob["gamma"]
        self.rule = blob["rule"]
        self.lambda_ = blob.get("lambda", 0.0)
        self.updates = blob.get("updates", 0)
        self.table = QTable(self.n_actions, init=blob["init"], rng=self.rng)
        self.table.q = defaultdict(self.table._row, blob["q"])
        return self

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        capped = " (capped)" if self.clamped else ""
        return (
            f"TabularAgent(rule={self.rule}, alpha={self.alpha:.3f}{capped}, "
            f"gamma={self.gamma}, lambda={self.lambda_}, "
            f"states={len(self.table)}, updates={self.updates})"
        )


def build_agents(n: int = 2, shared_table: bool = False, **kwargs: Any) -> list[TabularAgent]:
    """Make `n` bots. Mirrors `DQN_train.build_agents`.

    `shared_table=True` points every bot at one table. The roles are near
    symmetric apart from spawn position, so sharing doubles the data per entry
    and roughly halves the time to fill it. The cost is that the bots can no
    longer specialise, which matters exactly when one needs to hold a switch
    while the other walks through. Off by default, matching the DQN's two
    independent networks.
    """
    agents = [TabularAgent(**kwargs) for _ in range(n)]
    if shared_table and agents:
        for agent in agents[1:]:
            agent.table = agents[0].table
    return agents


def table_stats(agents: Iterable[TabularAgent]) -> dict[str, Any]:
    agents = list(agents)
    return {
        "states": [len(a.table) for a in agents],
        "coverage": [round(a.table.coverage(), 3) for a in agents],
        "updates": [a.updates for a in agents],
    }

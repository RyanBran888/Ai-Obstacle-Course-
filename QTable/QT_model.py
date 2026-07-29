"""The Q-table and the agent that owns one.

`QN_model.QNetwork` approximates Q(s,a) with a stack of `nn.Linear` layers and
learns by gradient descent. This file does the same job with a dictionary and
arithmetic. That is the whole difference, and it is a big one:

    network                             table
    ------------------------------      ------------------------------
    generalises to unseen states        every state learned separately
    lossy -- may never fit exactly      exact, converges to optimal Q*
    fixed memory                        memory grows with states visited
    needs a target net + replay to      stable on its own
      stay stable
    input must be a float vector        input must be hashable

No torch, no numpy. A `defaultdict` of small float lists is genuinely the right
data structure here, and keeping it plain makes the update rule easy to read
against the textbook version.

Three update rules are selectable, because "the Q-table one" is not one
algorithm:

    q_learning      off-policy. Targets the best next action, whether or not
                    the agent takes it. Learns the optimal path and then
                    happily walks it next to the lava, because epsilon-greedy
                    exploration is not its problem.
    sarsa           on-policy. Targets the action actually taken next, so the
                    cost of exploring is baked into the value. Gives the
                    cautious policy that keeps its distance from hazards.
    expected_sarsa  on-policy, but averages over the epsilon-greedy
                    distribution instead of sampling it. Same bias as SARSA
                    with less variance; usually the best default of the three.
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import Any, Hashable, Iterable

UPDATE_RULES = ("q_learning", "sarsa", "expected_sarsa")


class QTable:
    """Sparse action-value table.

    Unvisited entries are not stored; they read as `init` on demand. Setting
    `init` above any achievable return ("optimistic initialisation") is a cheap
    and surprisingly effective explorer: every untried action looks better than
    every tried one, so the agent sweeps its options before settling. It costs
    nothing and it does not need a schedule.
    """

    __slots__ = ("n_actions", "init", "q", "_rng")

    def __init__(
        self,
        n_actions: int,
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

    def values(self, state: Hashable) -> list[float]:
        return self.q[state]

    def value(self, state: Hashable, action: int) -> float:
        return self.q[state][action]

    def best_value(self, state: Hashable) -> float:
        return max(self.q[state])

    def best_action(self, state: Hashable) -> int:
        """Argmax, ties broken at random.

        The tie-break matters more than it looks. Every row starts uniform, so
        a deterministic argmax would send every agent north out of every
        unvisited state and quietly destroy exploration.
        """
        row = self.q[state]
        top = max(row)
        best = [i for i, v in enumerate(row) if v == top]
        return best[0] if len(best) == 1 else self._rng.choice(best)

    def act(self, state: Hashable, eps: float) -> int:
        if self._rng.random() < eps:
            return self._rng.randrange(self.n_actions)
        return self.best_action(state)

    def policy_probs(self, state: Hashable, eps: float) -> list[float]:
        """Epsilon-greedy action distribution -- what expected SARSA averages over."""
        row = self.q[state]
        top = max(row)
        greedy = [i for i, v in enumerate(row) if v == top]
        share = eps / self.n_actions
        probs = [share] * self.n_actions
        for i in greedy:
            probs[i] += (1.0 - eps) / len(greedy)
        return probs

    # -- writing -----------------------------------------------------------

    def update(self, state: Hashable, action: int, delta: float) -> None:
        self.q[state][action] += delta

    # -- housekeeping ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.q)

    @property
    def entries(self) -> int:
        return len(self.q) * self.n_actions

    def coverage(self) -> float:
        """Fraction of stored entries that have moved off their initial value."""
        if not self.q:
            return 0.0
        touched = sum(1 for row in self.q.values() for v in row if v != self.init)
        return touched / self.entries

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(
                {"n_actions": self.n_actions, "init": self.init, "q": dict(self.q)},
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, path: str) -> "QTable":
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        self.n_actions = blob["n_actions"]
        self.init = blob["init"]
        self.q = defaultdict(self._row, blob["q"])
        return self


class TabularAgent:
    """One Q-table bot.

    The public surface deliberately matches `QN_train.Agent` -- `act`, `learn`,
    `save`, `load` -- so a training loop can drive either family without caring
    which it has. `learn` returns the absolute TD error, which plays the same
    role in the logs that the network's loss does.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.2,
        gamma: float = 0.99,
        rule: str = "expected_sarsa",
        init: float = 0.0,
        lambda_: float = 0.0,
        alpha_min: float = 0.02,
        alpha_decay: float = 1.0,
        rng: random.Random | None = None,
    ) -> None:
        if rule not in UPDATE_RULES:
            raise ValueError(f"rule must be one of {UPDATE_RULES}, got {rule!r}")

        self.n_actions = n_actions
        self.gamma = gamma
        self.rule = rule
        self.lambda_ = lambda_

        # Eligibility traces multiply the effective step size. A state revisited
        # along a trajectory accumulates roughly 1/(1 - gamma*lambda) worth of
        # update, so a step size that is stable at lambda=0 can diverge outright
        # at lambda=0.8: alpha=0.3 with gamma=0.99 gives an effective 1.44, and
        # anything at or above 1.0 overshoots the target every time and
        # oscillates instead of converging.
        #
        # This bit me during development. Alpha 0.3 with alpha_decay 0.9995
        # scored 100% on a fixed room; the same run at 0.9998 scored 0%. Nothing
        # was wrong with either number -- the faster decay simply fell under the
        # stability threshold sooner, by luck. Rather than leave that landmine
        # in the config, cap alpha where the maths says it has to be.
        self.alpha_cap = 0.9 * (1.0 - gamma * lambda_) if lambda_ > 0.0 else 1.0
        self.alpha = min(alpha, self.alpha_cap)
        self.alpha_min = min(alpha_min, self.alpha_cap)
        self.alpha_decay = alpha_decay
        self.clamped = self.alpha < alpha

        # Robbins-Monro: a constant step size never converges, it hovers around
        # the answer forever. With traces on, that hovering is large enough to
        # destroy the policy outright -- alpha pinned at the cap scored 0% on a
        # fixed room that the same agent solves 100% of the time once alpha is
        # allowed to decay. Refuse the setting rather than fail mysteriously.
        if alpha_decay >= 1.0 and lambda_ > 0.0:
            raise ValueError(
                "alpha_decay must be < 1.0 when lambda_ > 0; a constant step "
                "size does not converge under eligibility traces. Try 0.9995."
            )

        self.rng = rng or random.Random()
        self.table = QTable(n_actions, init=init, rng=self.rng)

        #: Eligibility traces, only allocated when lambda_ > 0.
        self._traces: dict[Hashable, list[float]] = {}
        self.updates = 0

    # -- acting ------------------------------------------------------------

    def act(self, state: Hashable, eps: float) -> int:
        return self.table.act(state, eps)

    def greedy(self, state: Hashable) -> int:
        return self.table.best_action(state)

    # -- learning ----------------------------------------------------------

    def start_episode(self) -> None:
        self._traces.clear()

    def learn(
        self,
        state: Hashable,
        action: int,
        reward: float,
        next_state: Hashable,
        done: bool,
        next_action: int | None = None,
        eps: float = 0.0,
    ) -> float:
        """One temporal-difference update. Returns |TD error|.

        `next_action` is only read by the `sarsa` rule; `eps` only by
        `expected_sarsa`. Passing both always is harmless and keeps the caller
        free to switch rules without changing its call site.
        """
        bootstrap = 0.0 if done else self._bootstrap(next_state, next_action, eps)
        target = reward + self.gamma * bootstrap
        error = target - self.table.value(state, action)

        if self.lambda_ > 0.0:
            self._trace_update(state, action, error, next_state, next_action, eps)
        else:
            self.table.update(state, action, self.alpha * error)

        self.updates += 1
        return abs(error)

    def _bootstrap(self, next_state: Hashable, next_action: int | None, eps: float) -> float:
        if self.rule == "q_learning":
            return self.table.best_value(next_state)
        if self.rule == "sarsa":
            if next_action is None:
                return self.table.best_value(next_state)
            return self.table.value(next_state, next_action)
        probs = self.table.policy_probs(next_state, eps)
        row = self.table.values(next_state)
        return sum(p * v for p, v in zip(probs, row))

    def _trace_update(
        self,
        state: Hashable,
        action: int,
        error: float,
        next_state: Hashable,
        next_action: int | None,
        eps: float,
    ) -> None:
        """Watkins-style replacing traces.

        Credit flows backwards to everything recently visited instead of one
        step at a time, which is what makes a single sparse reward at the exit
        reachable in a sensible number of episodes. The trace is cut when the
        agent takes a non-greedy action under `q_learning`, because past that
        point the return no longer reflects the greedy policy the rule is
        estimating.
        """
        row = self._traces.setdefault(state, [0.0] * self.n_actions)
        row[action] = 1.0  # replacing, not accumulating

        decay = self.gamma * self.lambda_
        dead: list[Hashable] = []
        for s, trace in self._traces.items():
            values = self.table.values(s)
            for a, e in enumerate(trace):
                if e > 1e-4:
                    values[a] += self.alpha * error * e
                    trace[a] = e * decay
                else:
                    trace[a] = 0.0
            if not any(trace):
                dead.append(s)
        for s in dead:
            del self._traces[s]

        if self.rule == "q_learning" and next_action is not None:
            if next_action != self.table.best_action(next_state):
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
        self.table = QTable(self.n_actions, init=blob["init"], rng=self.rng)
        self.table.q = defaultdict(self.table._row, blob["q"])
        return self

    # -- reporting ---------------------------------------------------------

    def describe(self) -> str:
        return (
            f"TabularAgent(rule={self.rule}, alpha={self.alpha:.3f}"
            f"{' (capped)' if self.clamped else ''}, "
            f"gamma={self.gamma}, lambda={self.lambda_}, "
            f"states={len(self.table)}, updates={self.updates})"
        )


def build_agents(
    n: int = 2, shared_table: bool = False, **kwargs: Any
) -> list[TabularAgent]:
    """Make `n` bots.

    `shared_table=True` points every bot at one table. The two roles are
    symmetric apart from where they spawn, so sharing doubles the data per
    entry and roughly halves the time to fill it in. It also means the bots
    cannot specialise, which matters as soon as a room needs one of them to
    stand on a hold switch while the other walks through the door. Off by
    default, to match the two independent networks in `QN_train`.
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
        "entries": [a.table.entries for a in agents],
        "coverage": [round(a.table.coverage(), 3) for a in agents],
        "updates": [a.updates for a in agents],
    }

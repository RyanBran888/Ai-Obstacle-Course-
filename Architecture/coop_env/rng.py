"""Seeded randomness with stable, independent sub-streams.

Reproducibility is a hard requirement, so every random draw in the project goes
through `SeededRandom`. The important property is `derive()`: each subsystem
(layout, hazards, mechanisms, ...) pulls from its own named stream, so adding a
draw inside the hazard pass cannot shift the numbers the layout pass sees. Named
streams are hashed with BLAKE2b rather than `hash()` because the built-in string
hash is salted per-process and would break reproducibility across runs.
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable, Sequence, TypeVar

T = TypeVar("T")

MAX_SEED = 2**63 - 1


def normalize_seed(seed: int | str | None) -> int:
    """Coerce any seed-ish value into a non-negative 63-bit integer."""
    if seed is None:
        return random.SystemRandom().randrange(MAX_SEED)
    if isinstance(seed, int):
        return abs(seed) % MAX_SEED
    digest = hashlib.blake2b(seed.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % MAX_SEED


def derive_seed(seed: int, label: str) -> int:
    """Deterministically mix a label into a seed to produce a child seed."""
    payload = f"{seed}:{label}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % MAX_SEED


class SeededRandom:
    """A deterministic random source that can spawn independent sub-streams."""

    __slots__ = ("seed", "label", "_rng", "_children")

    def __init__(self, seed: int | str | None = None, label: str = "root") -> None:
        self.seed = normalize_seed(seed)
        self.label = label
        self._rng = random.Random(self.seed)
        self._children: dict[str, "SeededRandom"] = {}

    def derive(self, label: str) -> "SeededRandom":
        """Return a stable child stream for `label`, created on first request."""
        if label not in self._children:
            child = SeededRandom.__new__(SeededRandom)
            child.seed = derive_seed(self.seed, label)
            child.label = f"{self.label}.{label}"
            child._rng = random.Random(child.seed)
            child._children = {}
            self._children[label] = child
        return self._children[label]

    def fork(self, label: str) -> "SeededRandom":
        """Like `derive`, but always a fresh stream (used for retry attempts)."""
        child = SeededRandom.__new__(SeededRandom)
        child.seed = derive_seed(self.seed, label)
        child.label = f"{self.label}.{label}"
        child._rng = random.Random(child.seed)
        child._children = {}
        return child

    # -- draws -----------------------------------------------------------

    def randint(self, low: int, high: int) -> int:
        """Inclusive integer draw; tolerates an inverted range."""
        if low > high:
            low, high = high, low
        return self._rng.randint(low, high)

    def in_range(self, span: tuple[int, int]) -> int:
        return self.randint(span[0], span[1])

    def uniform(self, low: float, high: float) -> float:
        return self._rng.uniform(low, high)

    def random(self) -> float:
        return self._rng.random()

    def chance(self, probability: float) -> bool:
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return self._rng.random() < probability

    def choice(self, options: Sequence[T]) -> T:
        if not options:
            raise IndexError("cannot choose from an empty sequence")
        return options[self._rng.randrange(len(options))]

    def sample(self, options: Sequence[T], count: int) -> list[T]:
        count = max(0, min(count, len(options)))
        return self._rng.sample(list(options), count)

    def shuffled(self, options: Iterable[T]) -> list[T]:
        items = list(options)
        self._rng.shuffle(items)
        return items

    def weighted_choice(self, weights: dict[T, float]) -> T:
        """Pick a key with probability proportional to its weight.

        Keys are sorted before drawing so the result depends only on the seed,
        never on dict insertion order.
        """
        items = [(k, w) for k, w in sorted(weights.items(), key=lambda kv: str(kv[0])) if w > 0]
        if not items:
            raise ValueError("weighted_choice requires at least one positive weight")
        total = sum(w for _, w in items)
        roll = self._rng.random() * total
        upto = 0.0
        for key, weight in items:
            upto += weight
            if roll <= upto:
                return key
        return items[-1][0]

    def __repr__(self) -> str:
        return f"SeededRandom(seed={self.seed}, label={self.label!r})"

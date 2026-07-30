"""Convert tabular Q-learning pickles into DQN-format .pt checkpoints.

    python DQN/convert_qtable.py qtable_curriculum_agent0.pkl \
                                 qtable_curriculum_agent1.pkl \
                                 --out runs/converted

Writes ``agent_0.pt``, ``agent_1.pt`` and ``curriculum.pt`` into ``--out``, in
the exact format ``DQN_train.Agent.save`` produces, so ``load_model.load_agent``
and ``load_model.load_policy`` accept them unchanged.

WHAT THIS ACTUALLY DOES
-----------------------
A Q-table is a dictionary; a DQN is a function. There are no weights to copy
across, so this is not a format translation -- it is a *distillation*. The
table's rows become supervised regression targets and the network is fitted to
reproduce them. The output is only as good as that fit, so the tool measures
how often the fitted network picks the same action as the table
(``--min-agreement``) and refuses to write a checkpoint that failed to learn
its source.

The harder half of the problem is the state keys. The network eats the
1325-float observation ``env_bridge`` emits; the table is keyed on whatever the
tabular agent used. Three cases, in descending order of trustworthiness:

1. ``--key-fn module:function`` (best). You supply the same discretiser the
   tabular agent used, as ``f(obs, agent_index, bridge) -> hashable key``. Real
   episodes are rolled out, so every training pair is a genuine
   (real observation, table row) match. Use this whenever the table is keyed on
   anything compact -- coordinates, tuples, bitmasks.
2. Keys already are full observations (length 1325). Decoded directly, exact.
3. ``--allow-lossy``. Keys are numeric but the wrong length, so they are padded
   into an observation-shaped vector at an arbitrary offset. The checkpoint
   loads and the numbers are real, but the input layout does not match what the
   environment produces, so it will NOT behave sensibly in a live episode. It
   is stamped ``qtable_lossy: true`` for that reason. Treat it as a format
   demo, not a working agent.

Start here if you do not know which case you are in:

    python DQN/convert_qtable.py qtable_curriculum_agent0.pkl --inspect

which loads the pickle, reports its structure, key shapes and action-row width,
and writes nothing.

SECURITY: unpickling runs arbitrary code. Only point this at files you or a
collaborator produced.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import pickle
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

# DQN/ so sibling modules import, the repo root so `DQN.DQN_model` resolves,
# and Architecture/ so coop_env resolves.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _path in (str(_HERE), str(_ROOT), str(_ROOT / "Architecture")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from DQN.DQN_model import ACTIONS, N_ACTIONS, OBS_DIM  # noqa: E402

#: Action rows narrower than the network's are widened to this many columns by
#: giving the missing actions a value below every real one, so distillation can
#: never learn to pick an action the table never scored.
_UNSCORED_MARGIN = 1.0

#: Common attribute names a pickled agent object might keep its table under.
_TABLE_ATTRS = ("q", "Q", "q_table", "qtable", "Q_table", "table", "values")


# ----------------------------------------------------------------- loading


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def extract_table(payload: Any, path: Path) -> dict[Any, np.ndarray]:
    """Normalise whatever was pickled into ``{state_key: action_values}``."""
    table = _find_table(payload)
    if table is None:
        raise ValueError(
            f"{path.name}: could not find a Q-table in a "
            f"{type(payload).__name__}. Run with --inspect to see its shape."
        )

    if isinstance(table, np.ndarray):
        return _table_from_array(table, path)
    if not isinstance(table, Mapping):
        raise ValueError(
            f"{path.name}: Q-table is a {type(table).__name__}, "
            "expected a mapping or an array"
        )

    rows: dict[Any, np.ndarray] = {}
    for key, value in table.items():
        row = _as_row(value)
        if row is None:
            raise ValueError(
                f"{path.name}: entry for key {key!r} is a "
                f"{type(value).__name__}, not a row of action values"
            )
        rows[key] = row
    if not rows:
        raise ValueError(f"{path.name}: Q-table is empty")
    return rows


def _find_table(payload: Any) -> Any:
    """Dig the table out of a bare dict, a wrapper dict, or an agent object."""
    if isinstance(payload, Mapping):
        for name in _TABLE_ATTRS:
            inner = payload.get(name)
            if isinstance(inner, (Mapping, np.ndarray)) and len(inner):
                return inner
        # A bare table: keys are states, values are action rows.
        if payload and all(_as_row(v) is not None for v in payload.values()):
            return payload
        return None
    if isinstance(payload, np.ndarray):
        return payload
    for name in _TABLE_ATTRS:
        inner = getattr(payload, name, None)
        if isinstance(inner, (Mapping, np.ndarray)) and len(inner):
            return inner
    return None


def _as_row(value: Any) -> np.ndarray | None:
    """Coerce one table value into a 1-D float row, or return None."""
    if isinstance(value, np.ndarray):
        return value.astype(np.float64).reshape(-1) if value.size else None
    if isinstance(value, Mapping):
        # {action: value}; ordered by action index or by ACTIONS name.
        try:
            order = sorted(value, key=_action_sort_key)
        except TypeError:
            return None
        return np.asarray([float(value[k]) for k in order], dtype=np.float64)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            return np.asarray([float(v) for v in value], dtype=np.float64)
        except (TypeError, ValueError):
            return None
    return None


def _action_sort_key(key: Any) -> tuple[int, Any]:
    if isinstance(key, str) and key in ACTIONS:
        return (0, ACTIONS.index(key))
    if isinstance(key, (int, np.integer)):
        return (0, int(key))
    return (1, str(key))


def _table_from_array(array: np.ndarray, path: Path) -> dict[Any, np.ndarray]:
    """A dense ndarray table: leading axes index the state, last axis actions."""
    if array.ndim < 2:
        raise ValueError(
            f"{path.name}: array Q-table needs at least 2 dimensions, got {array.ndim}"
        )
    flat = array.reshape(-1, array.shape[-1]).astype(np.float64)
    shape = array.shape[:-1]
    return {
        tuple(int(i) for i in np.unravel_index(index, shape)): flat[index]
        for index in range(flat.shape[0])
    }


# ----------------------------------------------------------------- keys


def decode_key(key: Any) -> np.ndarray | None:
    """Turn a state key into a float vector, or None if it cannot be read."""
    if isinstance(key, np.ndarray):
        return key.astype(np.float32).reshape(-1)
    if isinstance(key, bytes):
        if len(key) % 4 == 0:
            return np.frombuffer(key, dtype=np.float32).copy()
        return None
    if isinstance(key, (int, float, np.integer, np.floating)):
        return np.asarray([float(key)], dtype=np.float32)
    if isinstance(key, str):
        try:
            parsed = ast.literal_eval(key)
        except (ValueError, SyntaxError):
            return None
        return None if isinstance(parsed, str) else decode_key(parsed)
    if isinstance(key, Sequence):
        flat = _flatten_numbers(key)
        return None if flat is None else np.asarray(flat, dtype=np.float32)
    return None


def _flatten_numbers(value: Any) -> list[float] | None:
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, Sequence):
        out: list[float] = []
        for item in value:
            part = _flatten_numbers(item)
            if part is None:
                return None
            out.extend(part)
        return out
    return None


# ----------------------------------------------------------------- datasets


class Dataset:
    """Observations, their table rows, and the two masks over those rows."""

    def __init__(
        self,
        obs: np.ndarray,
        targets: np.ndarray,
        weight: np.ndarray,
        scored: np.ndarray,
        lossy: bool,
        source: str,
        transform: dict[str, Any] | None = None,
    ) -> None:
        self.obs = obs
        self.targets = targets
        self.weight = weight
        self.scored = scored
        self.lossy = lossy
        self.source = source
        #: Affine input transform baked into `obs`, recorded so callers can
        #: reproduce it. Only ever set on the lossy path.
        self.transform = transform

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def widen_rows(
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad or trim action rows to N_ACTIONS.

    Returns ``(targets, weight, scored)``, three arrays of the same shape:

    ``targets``  what the network should output.
    ``weight``   which entries take part in the regression.
    ``scored``   which entries the table genuinely rated, used when comparing
                 greedy actions.

    The two masks differ in both directions. An entry left at exactly 0.0 is a
    Q-table's way of saying "never updated", so it is dropped from the loss and
    from the comparison. A column the table did not have at all is *trained*,
    towards a value below every real entry in its row, so the network can never
    prefer an action the table never rated -- but it is not scored, because
    there is no table opinion to agree with.
    """
    count, width = rows.shape
    keep = min(width, N_ACTIONS)

    targets = np.empty((count, N_ACTIONS), dtype=np.float32)
    targets[:, :keep] = rows[:, :keep]

    scored = np.zeros((count, N_ACTIONS), dtype=bool)
    scored[:, :keep] = rows[:, :keep] != 0.0
    weight = scored.copy()

    if width < N_ACTIONS:
        targets[:, width:] = (rows.min(axis=1) - _UNSCORED_MARGIN)[:, None]
        weight[:, width:] = True

    return targets, weight, scored


def _encode_padded(raw: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Lay compact keys out inside an observation-shaped vector.

    Only used on the ``--allow-lossy`` path, where the layout is arbitrary by
    definition, so the encoding is chosen purely for how well it fits rather
    than for any resemblance to a real observation.

    Dropping a handful of raw numbers into a 1325-wide field of zeros fits
    badly for two compounding reasons: the first layer is initialised for
    fan-in 1325, so the signal barely reaches the hidden units, and an MLP
    asked to fit hundreds of unrelated values across a 3-D coordinate input
    runs straight into spectral bias. One-hot encoding each discrete column
    sidesteps both -- the states become near-orthogonal and the fit is easy.
    Continuous columns fall back to standardisation.
    """
    columns = [np.unique(raw[:, i]) for i in range(raw.shape[1])]
    discrete = [
        len(values) <= 512 and np.all(values == np.round(values))
        for values in columns
    ]
    width = sum(
        len(values) if is_discrete else 1
        for values, is_discrete in zip(columns, discrete)
    )

    obs = np.zeros((raw.shape[0], OBS_DIM), dtype=np.float32)
    if not all(discrete) or width > OBS_DIM:
        used = min(raw.shape[1], OBS_DIM)
        obs[:, :used] = raw[:, :used]
        mean = obs[:, :used].mean(axis=0)
        scale = obs[:, :used].std(axis=0)
        scale[scale < 1e-6] = 1.0
        obs[:, :used] = (obs[:, :used] - mean) / scale
        return obs, {
            "encoding": "standardised",
            "columns": used,
            "mean": [float(v) for v in mean],
            "scale": [float(v) for v in scale],
        }

    offset = 0
    levels: list[list[float]] = []
    for index, values in enumerate(columns):
        lookup = {float(v): position for position, v in enumerate(values)}
        for row in range(raw.shape[0]):
            obs[row, offset + lookup[float(raw[row, index])]] = 1.0
        levels.append([float(v) for v in values])
        offset += len(values)
    return obs, {
        "encoding": "one_hot",
        "width": int(width),
        "levels": levels,
    }


def dataset_from_keys(
    table: dict[Any, np.ndarray],
    allow_lossy: bool,
    label: str,
) -> Dataset:
    """Build training pairs by reading observations straight out of the keys."""
    keys = list(table)
    decoded = [decode_key(key) for key in keys]
    undecodable = [k for k, d in zip(keys, decoded) if d is None]
    if undecodable:
        sample = ", ".join(repr(k) for k in undecodable[:3])
        raise ValueError(
            f"{label}: {len(undecodable)} state key(s) are not numeric and "
            f"cannot be turned into observations (e.g. {sample}). "
            "Supply --key-fn so real observations can be sampled from the "
            "environment instead."
        )

    widths = Counter(int(d.size) for d in decoded if d is not None)
    if len(widths) > 1:
        raise ValueError(
            f"{label}: state keys have mixed widths {dict(widths)}; "
            "a single observation layout is required"
        )
    width = next(iter(widths))

    rows = np.stack([np.asarray(table[key], dtype=np.float64) for key in keys])
    row_widths = {rows.shape[1]}
    if len(row_widths) > 1:  # pragma: no cover - stack would have raised
        raise ValueError(f"{label}: action rows have mixed widths")

    lossy = width != OBS_DIM
    if lossy and not allow_lossy:
        raise ValueError(
            f"{label}: state keys are {width} wide but the network expects "
            f"{OBS_DIM}. Pass --key-fn to sample real observations (correct), "
            "or --allow-lossy to pad them into shape (loads, but will not play "
            "the game)."
        )

    raw = np.stack([np.asarray(d, dtype=np.float32) for d in decoded])
    if lossy:
        obs, transform = _encode_padded(raw)
    else:
        obs = np.zeros((len(keys), OBS_DIM), dtype=np.float32)
        obs[:, :width] = raw
        transform = None

    targets, weight, scored = widen_rows(rows)
    return Dataset(
        obs,
        targets,
        weight,
        scored,
        lossy,
        "keys(padded)" if lossy else "keys",
        transform,
    )


def dataset_from_environment(
    table: dict[Any, np.ndarray],
    key_fn,
    agent_index: int,
    episodes: int,
    max_steps: int,
    seed: int,
    label: str,
) -> Dataset:
    """Roll out real episodes and pair each observation with its table row."""
    from env_bridge import CoopEnvBridge

    rng = np.random.default_rng(seed)
    bridge = CoopEnvBridge(seed=seed, max_steps=max_steps)

    seen: dict[Any, np.ndarray] = {}
    borrowed: dict[Any, np.ndarray] = {}
    misses = 0
    for episode in range(episodes):
        observations = bridge.reset(seed=seed + episode)
        for _ in range(max_steps):
            for index in range(len(observations)):
                obs = np.asarray(observations[index], dtype=np.float32)
                key = key_fn(obs, index, bridge)
                if key not in table:
                    misses += 1
                elif index == agent_index:
                    seen.setdefault(key, obs)
                else:
                    # This agent's own view of the state is preferred; a
                    # teammate's is only used for keys it never reaches
                    # itself, since the observation is agent-relative.
                    borrowed.setdefault(key, obs)
            actions = [
                int(rng.integers(N_ACTIONS)) for _ in range(len(observations))
            ]
            observations, _, done, cut, _ = bridge.step(actions)
            if done or cut:
                break

    for key, obs in borrowed.items():
        seen.setdefault(key, obs)

    if not seen:
        raise ValueError(
            f"{label}: rolled out {episodes} episode(s) and no observation "
            "produced a key present in the table. Check that --key-fn matches "
            "the discretiser the tabular agent used."
        )

    keys = list(seen)
    obs = np.stack([seen[key] for key in keys]).astype(np.float32)
    rows = np.stack([np.asarray(table[key], dtype=np.float64) for key in keys])
    targets, weight, scored = widen_rows(rows)
    coverage = len(seen) / len(table)
    print(
        f"  sampled {len(seen)} of {len(table)} table states "
        f"({coverage:.1%} coverage, {misses} unmatched lookups) "
        f"for agent {agent_index}"
    )
    return Dataset(obs, targets, weight, scored, False, "environment")


def count_conflicts(data: Dataset) -> tuple[int, int]:
    """Observations that appear twice wanting different actions.

    One network cannot satisfy both, so a shared fit is capped by however many
    of these there are. Returns ``(conflicting, duplicated)``.
    """
    floor = np.finfo(np.float32).min
    greedy = np.where(data.scored, data.targets, floor).argmax(axis=1)
    eligible = data.scored.any(axis=1)

    best: dict[bytes, int] = {}
    duplicated = 0
    conflicting = 0
    for row in np.flatnonzero(eligible):
        digest = data.obs[row].tobytes()
        previous = best.get(digest)
        if previous is None:
            best[digest] = int(greedy[row])
            continue
        duplicated += 1
        if previous != int(greedy[row]):
            conflicting += 1
    return conflicting, duplicated


def resolve_key_fn(spec: str):
    module_name, _, attribute = spec.partition(":")
    if not attribute:
        raise ValueError("--key-fn must look like 'module:function'")
    function = getattr(importlib.import_module(module_name), attribute)
    if not callable(function):
        raise ValueError(f"{spec} is not callable")
    return function


def merge(datasets: Sequence[Dataset]) -> Dataset:
    """Stack per-agent datasets into the shared-policy training set."""
    transforms = [d.transform for d in datasets]
    if any(t != transforms[0] for t in transforms):
        # Each lossy table standardises against its own column statistics, so
        # stacking two of them would put the halves on different input scales.
        raise ValueError(
            "cannot merge tables that were padded with different input "
            "transforms; convert them separately"
        )
    return Dataset(
        np.concatenate([d.obs for d in datasets]),
        np.concatenate([d.targets for d in datasets]),
        np.concatenate([d.weight for d in datasets]),
        np.concatenate([d.scored for d in datasets]),
        any(d.lossy for d in datasets),
        "+".join(sorted({d.source for d in datasets})),
        transforms[0],
    )


# ----------------------------------------------------------------- fitting


def distil(
    data: Dataset,
    device: torch.device,
    epochs: int,
    batch: int,
    lr: float,
    holdout: float,
    seed: int,
    quiet: bool,
):
    """Fit a fresh Agent's network to the table, and report how well it fits."""
    from DQN.DQN_train import Agent

    torch.manual_seed(seed)
    agent = Agent(device=device, replay_capacity=1, lr=lr)

    order = np.random.default_rng(seed).permutation(len(data))
    cut = max(1, int(len(order) * (1.0 - holdout))) if len(order) > 1 else 1
    train_index, test_index = order[:cut], order[cut:]
    if not len(test_index):
        test_index = train_index

    obs = torch.as_tensor(data.obs, device=device)
    targets = torch.as_tensor(data.targets, device=device)
    weights = torch.as_tensor(data.weight.astype(np.float32), device=device)
    scored = torch.as_tensor(data.scored, device=device)
    train = torch.as_tensor(train_index.copy(), device=device)
    test = torch.as_tensor(test_index.copy(), device=device)

    loss_fn = torch.nn.SmoothL1Loss(reduction="none")
    # Selection is on *fidelity*: a Q-table is a lookup, so the checkpoint's
    # job is to reproduce the rows it was given, not to generalise past them.
    # Held-out agreement is reported alongside as information about whether the
    # network found any structure, and is never the thing optimised for.
    best = {
        "fidelity": -1.0,
        "holdout": float("nan"),
        "state": None,
        "epoch": 0,
        "loss": float("nan"),
    }

    for epoch in range(1, epochs + 1):
        agent.net.train(True)
        shuffled = train[torch.randperm(len(train), device=device)]
        total = 0.0
        for start in range(0, len(shuffled), batch):
            rows = shuffled[start : start + batch]
            weight = weights[rows]
            raw = loss_fn(agent.net(obs[rows]), targets[rows]) * weight
            denominator = weight.sum().clamp(min=1.0)
            loss = raw.sum() / denominator
            agent.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.net.parameters(), agent.clip)
            agent.opt.step()
            total += float(loss.item()) * len(rows)

        fidelity = _agreement(agent, obs, targets, scored, train)
        holdout = _agreement(agent, obs, targets, scored, test)
        if fidelity > best["fidelity"]:
            best = {
                "fidelity": fidelity,
                "holdout": holdout,
                "state": {
                    k: v.detach().cpu().clone()
                    for k, v in agent.net.state_dict().items()
                },
                "epoch": epoch,
                "loss": total / max(1, len(shuffled)),
            }
        if not quiet and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            print(
                f"  epoch {epoch:4d}  loss {total / max(1, len(shuffled)):.5f}"
                f"  fidelity {fidelity:.1%}  held-out {holdout:.1%}"
            )

    if best["state"] is not None:
        agent.net.load_state_dict(best["state"])
    agent.sync()
    agent.net.train(False)
    agent.target.train(False)

    report = {
        "states": len(data),
        "train_states": int(len(train_index)),
        "holdout_states": int(len(test_index)),
        "best_epoch": int(best["epoch"]),
        "train_loss": float(best["loss"]),
        "fidelity": float(best["fidelity"]),
        "holdout_agreement": float(best["holdout"]),
        "source": data.source,
        "lossy": bool(data.lossy),
        "input_transform": data.transform,
    }
    return agent, report


@torch.no_grad()
def _agreement(agent, obs, targets, scored, rows) -> float:
    """How often the network's best action matches the table's best action.

    Restricted to the entries the table actually rated, so a padded column
    can never count as a match or a miss.
    """
    if not len(rows):
        return float("nan")
    agent.net.train(False)
    eligible_actions = scored[rows]
    eligible = eligible_actions.any(dim=1)
    if not bool(eligible.any()):
        return float("nan")
    floor = torch.finfo(torch.float32).min
    wanted = targets[rows].masked_fill(~eligible_actions, floor).argmax(dim=1)
    got = (
        agent.net(obs[rows]).masked_fill(~eligible_actions, floor).argmax(dim=1)
    )
    return float((wanted[eligible] == got[eligible]).float().mean().item())


# ----------------------------------------------------------------- output


def save_checkpoint(agent, path: Path, provenance: dict[str, Any]) -> str:
    """Write via Agent.save, then stamp provenance in, atomically."""
    temporary = path.with_name(f".{path.name}.tmp")
    agent.save(str(temporary))
    checkpoint = torch.load(temporary, map_location="cpu")
    checkpoint["qtable_source"] = provenance
    checkpoint["qtable_lossy"] = bool(provenance.get("lossy"))
    with temporary.open("wb") as handle:
        torch.save(checkpoint, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return str(path)


def verify(path: Path, device: str) -> None:
    """Prove the file we just wrote loads through the normal front doors."""
    from DQN.load_model import load_agent, load_policy

    load_agent(path, device=device)
    load_policy(path, device=device)


# ----------------------------------------------------------------- inspect


def _abbreviate(key: Any, limit: int = 8) -> str:
    """Keys can be 1325 floats long; show enough to recognise the layout."""
    if isinstance(key, (Sequence, np.ndarray)) and not isinstance(key, (str, bytes)):
        items = list(key)
        head = ", ".join(f"{float(v):g}" if _is_number(v) else repr(v)
                         for v in items[:limit])
        tail = f", ... +{len(items) - limit} more" if len(items) > limit else ""
        return f"({head}{tail})"
    return repr(key)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def inspect(path: Path) -> None:
    payload = load_pickle(path)
    print(f"\n{path}")
    print(f"  pickled type: {type(payload).__name__}")
    if isinstance(payload, Mapping):
        print(f"  top-level entries: {len(payload)}")
    elif hasattr(payload, "__dict__"):
        print(f"  attributes: {sorted(vars(payload))}")

    try:
        table = extract_table(payload, path)
    except ValueError as error:
        print(f"  !! {error}")
        return

    keys = list(table)
    rows = np.stack([table[key] for key in keys[: min(len(keys), 5000)]])
    widths = Counter(len(table[key]) for key in keys)
    key_types = Counter(type(key).__name__ for key in keys)
    decoded = [decode_key(key) for key in keys[: min(len(keys), 5000)]]
    key_widths = Counter(
        int(d.size) if d is not None else "undecodable" for d in decoded
    )

    print(f"  states: {len(table)}")
    print(f"  key types: {dict(key_types)}")
    print(f"  key widths: {dict(key_widths)}  (network needs {OBS_DIM})")
    print(f"  action-row widths: {dict(widths)}  (network needs {N_ACTIONS})")
    print(
        f"  values: min {rows.min():.4f}  max {rows.max():.4f}  "
        f"mean {rows.mean():.4f}  zero entries {(rows == 0).mean():.1%}"
    )
    for key in keys[:3]:
        print(f"  example: {_abbreviate(key)} -> {np.round(table[key], 4).tolist()}")

    if key_widths.get(OBS_DIM):
        print("  => keys are full observations; convert directly.")
    elif "undecodable" in key_widths:
        print("  => keys are opaque; you must pass --key-fn.")
    else:
        print("  => keys are compact; pass --key-fn, or --allow-lossy to force.")


# ----------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python DQN/convert_qtable.py qtable_curriculum_agent0.pkl --inspect\n"
            "  python DQN/convert_qtable.py qtable_curriculum_agent0.pkl "
            "qtable_curriculum_agent1.pkl --out runs/converted\n"
        ),
    )
    parser.add_argument("tables", nargs="+", help="qtable_*.pkl, agent 0 first")
    parser.add_argument("--out", default=".", help="directory for the .pt files")
    parser.add_argument("--inspect", action="store_true", help="report and exit")
    parser.add_argument(
        "--shared",
        action="store_true",
        help="write one policy distilled from every table under all three "
             "names, matching run_curriculum's shared-policy export",
    )
    parser.add_argument("--key-fn", help="module:function state discretiser")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--allow-lossy", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=0.9,
        help="refuse to write if the fit reproduces fewer than this fraction "
             "of the table's own greedy actions on the states it covers "
             "(0 disables the gate)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    paths = [Path(p) for p in args.tables]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"no such file: {path}")

    if args.inspect:
        for path in paths:
            inspect(path)
        return 0

    if len(paths) > 2:
        raise SystemExit("expected at most two tables (agent 0 and agent 1)")
    if not 0.0 <= args.holdout < 1.0:
        raise SystemExit("--holdout must be in [0, 1)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    key_fn = resolve_key_fn(args.key_fn) if args.key_fn else None

    datasets: list[Dataset] = []
    for index, path in enumerate(paths):
        print(f"reading {path}")
        try:
            table = extract_table(load_pickle(path), path)
            print(f"  {len(table)} states")
            if key_fn is not None:
                data = dataset_from_environment(
                    table,
                    key_fn,
                    index,
                    args.episodes,
                    args.max_steps,
                    args.seed,
                    path.name,
                )
            else:
                data = dataset_from_keys(table, args.allow_lossy, path.name)
        except ValueError as error:
            # These are the expected "your table does not fit" outcomes, not
            # crashes, so report them as messages rather than tracebacks.
            raise SystemExit(f"\n{error}") from None
        datasets.append(data)

    # agent_N.pt keeps agent N's own table; curriculum.pt is the shared policy
    # fitted to both. --shared collapses all three onto that shared policy,
    # which is what run_curriculum's export bundle produces.
    if args.shared or len(datasets) == 1:
        shared = merge(datasets) if len(datasets) > 1 else datasets[0]
        plan = [("shared", shared, ["agent_0.pt", "agent_1.pt", "curriculum.pt"])]
    else:
        plan = [
            (f"agent {i}", data, [f"agent_{i}.pt"])
            for i, data in enumerate(datasets)
        ]
        plan.append(("shared", merge(datasets), ["curriculum.pt"]))

    # Every fit is checked before anything is written, so a failure late in the
    # plan cannot leave half a bundle on disk next to a "nothing was written".
    report: dict[str, Any] = {"tables": [str(p) for p in paths], "fits": {}}
    pending: list[tuple[Any, dict[str, Any], list[str]]] = []
    for label, data, names in plan:
        print(f"distilling {label} ({len(data)} states, source: {data.source})")
        conflicting, duplicated = count_conflicts(data)
        ceiling = 1.0 - conflicting / max(1, len(data))
        if conflicting:
            print(
                f"  {conflicting} of {duplicated} repeated observation(s) "
                f"disagree about the best action, capping fidelity near "
                f"{ceiling:.1%}"
            )
        agent, fit = distil(
            data,
            device,
            args.epochs,
            args.batch,
            args.lr,
            args.holdout,
            args.seed,
            args.quiet,
        )
        print(
            f"  best epoch {fit['best_epoch']}  "
            f"fidelity {fit['fidelity']:.1%}  "
            f"held-out {fit['holdout_agreement']:.1%}"
        )

        fit["conflicting_states"] = int(conflicting)
        fit["fidelity_ceiling"] = float(ceiling)

        fidelity = fit["fidelity"]
        if args.min_agreement > 0 and not fidelity >= args.min_agreement:
            if ceiling < args.min_agreement:
                remedy = (
                    f"The tables disagree with each other on {conflicting} "
                    "state(s), so no single network can reach the floor -- "
                    f"{ceiling:.1%} is the arithmetic ceiling. curriculum.pt "
                    "is one policy fitted to both tables, so it can only be "
                    "as consistent as they are. Lower --min-agreement to "
                    "accept the compromise, or convert each table on its own "
                    "if the agents are meant to behave differently."
                )
            else:
                remedy = (
                    "Raise --epochs, lower --batch, or lower the floor if a "
                    "rough fit is acceptable."
                )
            raise SystemExit(
                f"\n{label}: the fitted network reproduces only "
                f"{fidelity:.1%} of the table's greedy actions, below the "
                f"--min-agreement floor of {args.min_agreement:.0%}. Nothing "
                f"was written. {remedy}"
            )
        if fit["holdout_agreement"] < 0.5:
            print(
                f"  note: held-out agreement is {fit['holdout_agreement']:.1%}, "
                "so the network memorised the table rather than finding "
                "structure in it. Expect table-like behaviour on states the "
                "table covers and arbitrary behaviour elsewhere."
            )

        provenance = {
            "converted_from": [str(p) for p in paths],
            "role": label,
            **fit,
        }
        pending.append((agent, provenance, names))
        report["fits"][label] = fit

    written: list[str] = []
    for agent, provenance, names in pending:
        for name in names:
            written.append(save_checkpoint(agent, out / name, provenance))
            verify(out / name, args.device)

    (out / "conversion_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nwrote:")
    for path_string in written:
        print(f"  {path_string}")
    print(f"  {out / 'conversion_report.json'}")
    if any(data.lossy for _, data, _ in plan):
        print(
            "\nWARNING: keys were padded to fit the observation layout, so "
            "these checkpoints load but will not play the game correctly. "
            "Re-run with --key-fn for a usable agent."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

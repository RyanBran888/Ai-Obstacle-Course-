from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Architecture"))

import coop_env
from coop_env import (
    EntityKind,
    GenerationConfig,
    Room,
    RoomGenerator,
    RoomShape,
    SwitchMode,
    Tile,
    WipeoutBallSize,
)
from coop_env.generation import GateKind
from coop_env.rng import derive_seed
from coop_env.tiles import HAZARD_TILES, tile_name


class RoomStage(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def config(self) -> GenerationConfig: ...

    @property
    def accepts(self) -> Callable[[Room], bool]: ...

    @property
    def required_features(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class RoomRecord:
    seed: int
    geometry_sha256: str
    navigation_sha256: str
    task_sha256: str
    attempts: int
    width: int
    height: int
    shape: str
    counts: tuple[tuple[str, int], ...]
    features: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "geometry_sha256": self.geometry_sha256,
            "navigation_sha256": self.navigation_sha256,
            "task_sha256": self.task_sha256,
            "attempts": self.attempts,
            "width": self.width,
            "height": self.height,
            "shape": self.shape,
            "counts": dict(self.counts),
            "features": list(self.features),
        }


@dataclass(frozen=True, slots=True)
class StageRoomManifest:
    stage: str
    config_json: str
    config_sha256: str
    train: tuple[RoomRecord, ...]
    validation: tuple[RoomRecord, ...]
    test: tuple[RoomRecord, ...]

    @property
    def config(self) -> GenerationConfig:
        return GenerationConfig.from_dict(json.loads(self.config_json))

    def records(self, split: str) -> tuple[RoomRecord, ...]:
        if split == "train":
            return self.train
        if split == "validation":
            return self.validation
        if split == "test":
            return self.test
        raise KeyError(f"unknown room split {split!r}")

    def seeds(self, split: str) -> tuple[int, ...]:
        return tuple(record.seed for record in self.records(split))

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "config": json.loads(self.config_json),
            "config_sha256": self.config_sha256,
            "splits": {
                split: [record.as_dict() for record in self.records(split)]
                for split in ("train", "validation", "test")
            },
        }


@dataclass(frozen=True, slots=True)
class CurriculumRoomManifest:
    data_seed: int
    stages: tuple[StageRoomManifest, ...]
    generator_version: str = coop_env.__version__
    schema_version: int = 3

    def stage(self, name: str) -> StageRoomManifest:
        for stage in self.stages:
            if stage.stage == name:
                return stage
        raise KeyError(f"stage {name!r} is not in the room manifest")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generator": "Architecture/coop_env/RoomGenerator",
            "generator_version": self.generator_version,
            "data_seed": self.data_seed,
            "stages": [stage.as_dict() for stage in self.stages],
        }

    @property
    def sha256(self) -> str:
        return _hash_json(self.as_dict())


def room_fingerprints(room: Room) -> tuple[str, str, str]:
    geometry = {
        "width": room.width,
        "height": room.height,
        "terrain": room.terrain.to_list(),
    }
    navigation = {
        "geometry": geometry,
        "spawns": [
            {"index": spawn.index, "pos": list(spawn.pos)}
            for spawn in room.spawns
        ],
        "exit": {
            "id": room.exit.id,
            "pos": list(room.exit.pos),
        },
    }
    task = {
        "navigation": navigation,
        "entities": [
            _canonical(entity)
            for entity in sorted(room.entities, key=lambda item: item.id)
        ],
        "topology": {
            "regions": [
                {
                    "id": region_id,
                    "tiles": [
                        list(pos)
                        for pos in sorted(
                            region.tiles, key=lambda item: (item.y, item.x)
                        )
                    ],
                }
                for region_id, region in sorted(room.topology.regions.items())
            ],
            "edges": [list(edge) for edge in room.topology.graph.edges()],
            "portals": [
                {
                    "edge": list(edge),
                    "tiles": [list(pos) for pos in positions],
                }
                for edge, positions in sorted(room.topology.portals.items())
            ],
            "spawn_regions": list(room.topology.spawn_regions),
            "exit_region": room.topology.exit_region,
            "depths": [
                [region_id, depth]
                for region_id, depth in sorted(room.topology.depths.items())
            ],
        },
        "exit_requires_both_agents": room.config.exit_requires_both_agents,
    }
    return _hash_json(geometry), _hash_json(navigation), _hash_json(task)


def build_manifest_suite(
    stages: Sequence[RoomStage],
    *,
    data_seed: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    progress: Callable[[str], None] | None = None,
) -> CurriculumRoomManifest:
    if min(train_size, validation_size, test_size) < 1:
        raise ValueError("room split sizes must be positive")
    if not stages:
        raise ValueError("at least one curriculum stage is required")
    if len({stage.name for stage in stages}) != len(stages):
        raise ValueError("curriculum stage names must be unique")

    buckets: dict[str, dict[str, list[RoomRecord]]] = {
        stage.name: {"train": [], "validation": [], "test": []}
        for stage in stages
    }
    used_seeds: set[int] = set()
    used_navigation: set[str] = set()
    used_tasks: set[str] = set()

    for stage in stages:
        generator = RoomGenerator(stage.config)
        for split, count in (
            ("test", test_size),
            ("validation", validation_size),
            ("train", train_size),
        ):
            required_features = set(stage.required_features)
            missing_features = (
                set(required_features)
                if count >= len(required_features)
                else set()
            )
            accepted = 0
            attempt = 0
            attempt_limit = max(1_000, count * 200)
            while accepted < count:
                if attempt >= attempt_limit:
                    raise RuntimeError(
                        f"{stage.name} {split} accepted {accepted}/{count} "
                        f"rooms after {attempt_limit} candidates"
                    )
                seed = derive_seed(
                    data_seed,
                    f"curriculum-manifest:v2:{stage.name}:{split}:{attempt}",
                )
                attempt += 1
                if seed in used_seeds:
                    continue

                outcome = generator.generate_with_report(seed)
                room = outcome.room
                if (
                    not outcome.report.ok
                    or outcome.fallback
                    or bool(room.metadata.get("fallback"))
                    or not stage.accepts(room)
                ):
                    continue
                geometry, navigation, task = room_fingerprints(room)
                if navigation in used_navigation or task in used_tasks:
                    continue
                features = room_features(room)
                if missing_features and not missing_features.intersection(features):
                    continue

                record = RoomRecord(
                    seed=seed,
                    geometry_sha256=geometry,
                    navigation_sha256=navigation,
                    task_sha256=task,
                    attempts=outcome.attempts,
                    width=room.width,
                    height=room.height,
                    shape=room.shape.value,
                    counts=tuple(room.counts().items()),
                    features=features,
                )
                buckets[stage.name][split].append(record)
                used_seeds.add(seed)
                used_navigation.add(navigation)
                used_tasks.add(task)
                missing_features.difference_update(features)
                accepted += 1
            if progress is not None:
                progress(f"  {stage.name}: {split} {accepted}/{count}")

    stage_manifests = tuple(
        StageRoomManifest(
            stage=stage.name,
            config_json=_config_json(stage.config),
            config_sha256=_hash_text(_config_json(stage.config)),
            train=tuple(buckets[stage.name]["train"]),
            validation=tuple(buckets[stage.name]["validation"]),
            test=tuple(buckets[stage.name]["test"]),
        )
        for stage in stages
    )
    manifest = CurriculumRoomManifest(data_seed=data_seed, stages=stage_manifests)
    assert_disjoint(manifest)
    return manifest


def assert_disjoint(manifest: CurriculumRoomManifest) -> None:
    if len({stage.stage for stage in manifest.stages}) != len(manifest.stages):
        raise ValueError("room manifest contains duplicate stage names")

    seed_owner: dict[int, str] = {}
    navigation_owner: dict[str, str] = {}
    task_owner: dict[str, str] = {}
    for stage in manifest.stages:
        if _hash_text(stage.config_json) != stage.config_sha256:
            raise ValueError(f"{stage.stage} config hash does not match its snapshot")
        for split in ("train", "validation", "test"):
            records = stage.records(split)
            seeds = tuple(record.seed for record in records)
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"{stage.stage} {split} contains duplicate seeds")
            navigations = [record.navigation_sha256 for record in records]
            tasks = [record.task_sha256 for record in records]
            if len(navigations) != len(set(navigations)):
                raise ValueError(f"{stage.stage} {split} contains duplicate navigation")
            if len(tasks) != len(set(tasks)):
                raise ValueError(f"{stage.stage} {split} contains duplicate tasks")
            for record in records:
                owner = f"{stage.stage}:{split}"
                _claim(seed_owner, record.seed, owner, "seed")
                _claim(
                    navigation_owner,
                    record.navigation_sha256,
                    owner,
                    "navigation",
                )
                _claim(task_owner, record.task_sha256, owner, "task")


def verify_manifest(
    manifest: CurriculumRoomManifest,
    stages: Sequence[RoomStage],
    *,
    splits: Sequence[str] = ("train", "validation", "test"),
) -> None:
    if manifest.generator_version != coop_env.__version__:
        raise ValueError("room manifest generator version does not match")
    assert_disjoint(manifest)
    if tuple(stage.name for stage in stages) != tuple(
        stage.stage for stage in manifest.stages
    ):
        raise ValueError("curriculum stages do not match the room manifest")
    for split in splits:
        sizes = {len(stage.records(split)) for stage in manifest.stages}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) < 1:
            raise ValueError(f"{split} split sizes are incomplete or inconsistent")

    for stage in stages:
        saved = manifest.stage(stage.name)
        if _hash_text(_config_json(stage.config)) != saved.config_sha256:
            raise ValueError(f"{stage.name} config changed after room staging")
        generator = RoomGenerator(saved.config)
        for split in splits:
            covered = {
                feature
                for record in saved.records(split)
                for feature in record.features
            }
            required_features = set(stage.required_features)
            missing = (
                sorted(required_features - covered)
                if len(saved.records(split)) >= len(required_features)
                else []
            )
            if missing:
                raise ValueError(
                    f"{stage.name} {split} lacks required features: "
                    + ", ".join(missing)
                )
            for record in saved.records(split):
                outcome = generator.generate_with_report(record.seed)
                room = outcome.room
                if (
                    not outcome.report.ok
                    or outcome.fallback
                    or bool(room.metadata.get("fallback"))
                    or not stage.accepts(room)
                ):
                    raise ValueError(
                        f"{stage.name} {split} seed {record.seed} is no longer valid"
                    )
                fingerprints = room_fingerprints(room)
                expected = (
                    record.geometry_sha256,
                    record.navigation_sha256,
                    record.task_sha256,
                )
                if fingerprints != expected:
                    raise ValueError(
                        f"{stage.name} {split} seed {record.seed} changed"
                    )
                if (
                    outcome.attempts != record.attempts
                    or room.width != record.width
                    or room.height != record.height
                    or room.shape.value != record.shape
                    or tuple(room.counts().items()) != record.counts
                    or room_features(room) != record.features
                ):
                    raise ValueError(
                        f"{stage.name} {split} seed {record.seed} metadata changed"
                    )


def room_features(room: Room) -> tuple[str, ...]:
    features = {
        f"shape:{room.shape.value}",
        *(f"entity:{entity.kind.name.lower()}" for entity in room.entities),
        *(
            f"tile:{tile_name(tile)}"
            for tile in Tile
            if room.terrain.count(tile) > 0
        ),
        *(
            f"switch_mode:{switch.mode.value}"
            for switch in room.switches
        ),
        *(
            f"wipeout_size:{ball.size.value}"
            for ball in room.wipeout_balls
        ),
        *(
            f"gate:{gate.get('kind')}"
            for gate in room.metadata.get("gates", ())
        ),
    }
    for name in (
        "require_wipeout_crossing",
        "require_bridge_crossing",
        "require_reset_detour",
        "require_key_for_each_agent",
        "require_combined_course",
    ):
        if getattr(room.config, name):
            features.add(f"contract:{name.removeprefix('require_')}")
    course = room.metadata.get("combined_course")
    if isinstance(course, dict):
        required_size = course.get("required_ball_size")
        if required_size in {"normal", "big"}:
            features.add(f"contract:required_wipeout:{required_size}")
    return tuple(sorted(features))


def verify_training_coverage(
    manifest: CurriculumRoomManifest,
    train_limits: Mapping[str, int],
    *,
    require_all: bool = True,
) -> tuple[str, ...]:
    features: set[str] = set()
    for stage in manifest.stages:
        limit = train_limits.get(stage.stage)
        if limit is None or limit < 1:
            raise ValueError(f"missing positive train limit for {stage.stage}")
        for record in stage.train[:limit]:
            features.update(record.features)

    required = {
        *(f"entity:{kind.name.lower()}" for kind in EntityKind),
        *(f"shape:{shape.value}" for shape in RoomShape),
        *(f"switch_mode:{mode.value}" for mode in SwitchMode),
        *(f"wipeout_size:{size.value}" for size in WipeoutBallSize),
        *(f"gate:{kind.value}" for kind in GateKind),
        "tile:obstacle",
        *(f"tile:{tile_name(tile)}" for tile in HAZARD_TILES),
        "contract:wipeout_crossing",
        "contract:bridge_crossing",
        "contract:reset_detour",
        "contract:key_for_each_agent",
        "contract:combined_course",
        "contract:required_wipeout:normal",
        "contract:required_wipeout:big",
    }
    missing = sorted(required - features)
    if require_all and missing:
        raise ValueError(
            "training rooms do not cover generated mechanics: "
            + ", ".join(missing)
        )
    return tuple(sorted(features))


def save_manifest(manifest: CurriculumRoomManifest, path: str | Path) -> Path:
    target = Path(path)
    text = json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"{target} already contains a different room manifest"
            )
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target


def _claim(
    owners: dict[Any, str],
    value: Any,
    claimant: str,
    label: str,
) -> None:
    owner = owners.setdefault(value, claimant)
    if owner != claimant:
        raise ValueError(f"{label} appears in both {owner} and {claimant}")


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        _canonical(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _config_json(config: GenerationConfig) -> str:
    return json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__name__,
            "fields": {
                field.name: _canonical(getattr(value, field.name))
                for field in fields(value)
                if not field.name.startswith("_")
            },
        }
    if isinstance(value, Mapping):
        pairs = [
            [_canonical(key), _canonical(item)]
            for key, item in value.items()
        ]
        return {
            "mapping": sorted(
                pairs,
                key=lambda pair: json.dumps(
                    pair[0], sort_keys=True, separators=(",", ":")
                ),
            )
        }
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            ),
        )
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    raise TypeError(f"cannot fingerprint value of type {type(value).__name__}")

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
from coop_env import GenerationConfig, Room, RoomGenerator
from coop_env.rng import derive_seed


class RoomStage(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def config(self) -> GenerationConfig: ...

    @property
    def accepts(self) -> Callable[[Room], bool]: ...


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
    schema_version: int = 1

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
) -> CurriculumRoomManifest:
    if min(train_size, validation_size, test_size) < 1:
        raise ValueError("room split sizes must be positive")
    if not stages:
        raise ValueError("at least one curriculum stage is required")
    if len({stage.name for stage in stages}) != len(stages):
        raise ValueError("curriculum stage names must be unique")

    generators = [RoomGenerator(stage.config) for stage in stages]
    buckets: dict[str, dict[str, list[RoomRecord]]] = {
        stage.name: {"train": [], "validation": [], "test": []}
        for stage in stages
    }
    used_seeds: set[int] = set()
    used_geometry: set[str] = set()
    used_tasks: set[str] = set()

    for split, count in (
        ("test", test_size),
        ("validation", validation_size),
        ("train", train_size),
    ):
        accepted = 0
        attempt = 0
        while accepted < count:
            if attempt >= count * 1_000:
                raise RuntimeError(f"could not generate enough {split} room families")
            seed = derive_seed(data_seed, f"curriculum-manifest:v1:{split}:{attempt}")
            attempt += 1
            if seed in used_seeds:
                continue

            candidate: list[tuple[str, RoomRecord]] = []
            valid = True
            for stage, generator in zip(stages, generators, strict=True):
                outcome = generator.generate_with_report(seed)
                room = outcome.room
                if (
                    not outcome.report.ok
                    or outcome.fallback
                    or bool(room.metadata.get("fallback"))
                    or not stage.accepts(room)
                ):
                    valid = False
                    break
                geometry, navigation, task = room_fingerprints(room)
                candidate.append(
                    (
                        stage.name,
                        RoomRecord(
                            seed=seed,
                            geometry_sha256=geometry,
                            navigation_sha256=navigation,
                            task_sha256=task,
                            attempts=outcome.attempts,
                            width=room.width,
                            height=room.height,
                            shape=room.shape.value,
                            counts=tuple(room.counts().items()),
                        ),
                    )
                )

            geometries = {record.geometry_sha256 for _, record in candidate}
            tasks = {record.task_sha256 for _, record in candidate}
            if (
                not valid
                or geometries & used_geometry
                or tasks & used_tasks
            ):
                continue

            for stage_name, record in candidate:
                buckets[stage_name][split].append(record)
            used_seeds.add(seed)
            used_geometry.update(geometries)
            used_tasks.update(tasks)
            accepted += 1

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
    geometry_owner: dict[str, str] = {}
    task_owner: dict[str, str] = {}
    expected_seeds: dict[str, tuple[int, ...]] = {}

    for stage in manifest.stages:
        if _hash_text(stage.config_json) != stage.config_sha256:
            raise ValueError(f"{stage.stage} config hash does not match its snapshot")
        for split in ("train", "validation", "test"):
            records = stage.records(split)
            seeds = tuple(record.seed for record in records)
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"{stage.stage} {split} contains duplicate seeds")
            if split in expected_seeds and seeds != expected_seeds[split]:
                raise ValueError(f"{stage.stage} does not share the {split} seed families")
            expected_seeds.setdefault(split, seeds)
            geometries = [record.geometry_sha256 for record in records]
            tasks = [record.task_sha256 for record in records]
            if len(geometries) != len(set(geometries)):
                raise ValueError(f"{stage.stage} {split} contains duplicate geometry")
            if len(tasks) != len(set(tasks)):
                raise ValueError(f"{stage.stage} {split} contains duplicate tasks")
            for record in records:
                _claim(seed_owner, record.seed, split, "seed")
                _claim(
                    geometry_owner,
                    record.geometry_sha256,
                    split,
                    "geometry",
                )
                _claim(task_owner, record.task_sha256, split, "task")


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

    for stage in stages:
        saved = manifest.stage(stage.name)
        if _hash_text(_config_json(stage.config)) != saved.config_sha256:
            raise ValueError(f"{stage.name} config changed after room staging")
        generator = RoomGenerator(saved.config)
        for split in splits:
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
                ):
                    raise ValueError(
                        f"{stage.name} {split} seed {record.seed} metadata changed"
                    )


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
    split: str,
    label: str,
) -> None:
    owner = owners.setdefault(value, split)
    if owner != split:
        raise ValueError(f"{label} appears in both {owner} and {split}")


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

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


ROOM_SPLITS = ("train", "validation", "test")
SELECTION_ALGORITHM = "derive-seed-v2-global-fingerprint-lazy-v2"


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
    candidate_index: int
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
            "candidate_index": self.candidate_index,
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
    selection_cursors: tuple[tuple[str, int], ...] = ()
    feature_targets: tuple[tuple[str, int], ...] = ()

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

    def selection_cursor(self, split: str) -> int:
        if split not in ROOM_SPLITS:
            raise KeyError(f"unknown room split {split!r}")
        return dict(self.selection_cursors).get(split, 0)

    def feature_target(self, split: str) -> int:
        if split not in ROOM_SPLITS:
            raise KeyError(f"unknown room split {split!r}")
        return dict(self.feature_targets).get(split, 0)

    def seeds(self, split: str) -> tuple[int, ...]:
        return tuple(record.seed for record in self.records(split))

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "config": json.loads(self.config_json),
            "config_sha256": self.config_sha256,
            "splits": {
                split: [record.as_dict() for record in self.records(split)]
                for split in ROOM_SPLITS
            },
            "selection_cursors": {
                split: self.selection_cursor(split) for split in ROOM_SPLITS
            },
            "feature_targets": {
                split: self.feature_target(split) for split in ROOM_SPLITS
            },
        }


@dataclass(frozen=True, slots=True)
class CurriculumRoomManifest:
    data_seed: int
    stages: tuple[StageRoomManifest, ...]
    generator_version: str = coop_env.__version__
    schema_version: int = 4
    selection_algorithm: str = SELECTION_ALGORITHM

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
            "selection_algorithm": self.selection_algorithm,
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


class LazyRoomManifestBuilder:
    """Materialize deterministic room pools only when they are requested."""

    def __init__(
        self,
        stages: Sequence[RoomStage],
        *,
        data_seed: int,
        initial: CurriculumRoomManifest | None = None,
    ) -> None:
        if not stages:
            raise ValueError("at least one curriculum stage is required")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("curriculum stage names must be unique")

        self.stages = tuple(stages)
        self.data_seed = data_seed
        self._stage_by_name = {stage.name: stage for stage in self.stages}
        self._config_json = {
            stage.name: _config_json(stage.config) for stage in self.stages
        }
        self._generators = {
            stage.name: RoomGenerator(stage.config) for stage in self.stages
        }
        self._records: dict[str, dict[str, list[RoomRecord]]] = {
            stage.name: {split: [] for split in ROOM_SPLITS}
            for stage in self.stages
        }
        self._cursors: dict[str, dict[str, int]] = {
            stage.name: {split: 0 for split in ROOM_SPLITS}
            for stage in self.stages
        }
        self._feature_targets: dict[str, dict[str, int | None]] = {
            stage.name: {split: None for split in ROOM_SPLITS}
            for stage in self.stages
        }
        self._generated_rooms: dict[
            str, dict[str, list[tuple[RoomRecord, Room]]]
        ] = {
            stage.name: {split: [] for split in ROOM_SPLITS}
            for stage in self.stages
        }
        self._used_seeds: set[int] = set()
        self._used_navigation: set[str] = set()
        self._used_tasks: set[str] = set()

        if initial is not None:
            self._load(initial)

    def ensure(
        self,
        stage_name: str,
        split: str,
        count: int,
        *,
        feature_target: int | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[RoomRecord, ...]:
        """Return a deterministic prefix, generating only missing records."""
        stage = self._stage(stage_name)
        _check_split(split)
        if count < 0:
            raise ValueError("room count must be nonnegative")
        quota_target = count if feature_target is None else feature_target
        if quota_target < count:
            raise ValueError("feature_target must be at least count")
        saved_target = self._feature_targets[stage_name][split]
        if saved_target is None:
            self._feature_targets[stage_name][split] = quota_target
        elif saved_target != quota_target:
            raise ValueError(
                f"{stage_name} {split} feature_target changed from "
                f"{saved_target} to {quota_target}"
            )

        records = self._records[stage_name][split]
        required_features = set(stage.required_features)
        covered = {
            feature
            for record in records[:count]
            for feature in record.features
        }
        missing_features = (
            required_features - covered
            if quota_target >= len(required_features)
            else set()
        )
        remaining_slots = count - len(records)
        coverage_due = count >= len(required_features)
        if len(records) >= count:
            if coverage_due and missing_features:
                raise ValueError(
                    f"{stage_name} {split} lacks required features: "
                    + ", ".join(sorted(missing_features))
                )
            return tuple(records[:count])
        if coverage_due and len(missing_features) > remaining_slots:
            raise ValueError(
                f"{stage_name} {split} cannot cover {len(missing_features)} "
                f"missing features in {remaining_slots} rooms"
            )

        start_cursor = self._cursors[stage_name][split]
        candidate_limit = start_cursor + max(
            1_000,
            max(count, quota_target) * 200,
        )
        generator = self._generators[stage_name]
        while len(records) < count:
            candidate_index = self._cursors[stage_name][split]
            if candidate_index >= candidate_limit:
                raise RuntimeError(
                    f"{stage_name} {split} accepted {len(records)}/{count} "
                    f"rooms after {candidate_index - start_cursor} candidates"
                )
            seed = _candidate_seed(
                self.data_seed,
                stage_name,
                split,
                candidate_index,
            )
            self._cursors[stage_name][split] = candidate_index + 1
            if seed in self._used_seeds:
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
            if (
                navigation in self._used_navigation
                or task in self._used_tasks
            ):
                continue
            features = room_features(room)
            if (
                missing_features
                and not missing_features.intersection(features)
            ):
                continue

            record = RoomRecord(
                seed=seed,
                candidate_index=candidate_index,
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
            records.append(record)
            self._generated_rooms[stage_name][split].append((record, room))
            self._used_seeds.add(seed)
            self._used_navigation.add(navigation)
            self._used_tasks.add(task)
            missing_features.difference_update(features)

        if coverage_due and missing_features:
            raise RuntimeError(
                f"{stage_name} {split} lacks required features: "
                + ", ".join(sorted(missing_features))
            )
        if progress is not None:
            progress(f"  {stage_name}: {split} {len(records)}/{count}")
        return tuple(records[:count])

    def take_rooms(
        self,
        stage_name: str,
        split: str,
    ) -> tuple[tuple[RoomRecord, Room], ...]:
        """Take newly generated rooms without storing them in the manifest."""
        self._stage(stage_name)
        _check_split(split)
        rooms = tuple(self._generated_rooms[stage_name][split])
        self._generated_rooms[stage_name][split].clear()
        return rooms

    def snapshot(self) -> CurriculumRoomManifest:
        """Return an immutable manifest; unmaterialized splits stay empty."""
        stages = tuple(
            StageRoomManifest(
                stage=stage.name,
                config_json=self._config_json[stage.name],
                config_sha256=_hash_text(self._config_json[stage.name]),
                train=tuple(self._records[stage.name]["train"]),
                validation=tuple(self._records[stage.name]["validation"]),
                test=tuple(self._records[stage.name]["test"]),
                selection_cursors=tuple(
                    (split, self._cursors[stage.name][split])
                    for split in ROOM_SPLITS
                ),
                feature_targets=tuple(
                    (
                        split,
                        self._feature_targets[stage.name][split] or 0,
                    )
                    for split in ROOM_SPLITS
                ),
            )
            for stage in self.stages
        )
        manifest = CurriculumRoomManifest(
            data_seed=self.data_seed,
            stages=stages,
        )
        assert_disjoint(manifest)
        return manifest

    def _stage(self, name: str) -> RoomStage:
        try:
            return self._stage_by_name[name]
        except KeyError as error:
            raise KeyError(f"unknown curriculum stage {name!r}") from error

    def _load(self, manifest: CurriculumRoomManifest) -> None:
        if manifest.data_seed != self.data_seed:
            raise ValueError("initial manifest data seed does not match")
        verify_manifest_structure(manifest, self.stages)
        for saved in manifest.stages:
            for split in ROOM_SPLITS:
                records = saved.records(split)
                self._records[saved.stage][split].extend(records)
                self._cursors[saved.stage][split] = saved.selection_cursor(split)
                target = saved.feature_target(split)
                self._feature_targets[saved.stage][split] = (
                    target if target > 0 else None
                )
                for record in records:
                    self._used_seeds.add(record.seed)
                    self._used_navigation.add(record.navigation_sha256)
                    self._used_tasks.add(record.task_sha256)


def build_manifest_suite(
    stages: Sequence[RoomStage],
    *,
    data_seed: int,
    train_size: int,
    validation_size: int,
    test_size: int,
    progress: Callable[[str], None] | None = None,
) -> CurriculumRoomManifest:
    """Build a complete suite using the legacy stage/split order."""
    if min(train_size, validation_size, test_size) < 1:
        raise ValueError("room split sizes must be positive")
    builder = LazyRoomManifestBuilder(stages, data_seed=data_seed)
    for stage in stages:
        for split, count in (
            ("test", test_size),
            ("validation", validation_size),
            ("train", train_size),
        ):
            builder.ensure(
                stage.name,
                split,
                count,
                feature_target=count,
                progress=progress,
            )
            builder.take_rooms(stage.name, split)
    return builder.snapshot()


def verify_manifest_structure(
    manifest: CurriculumRoomManifest,
    stages: Sequence[RoomStage],
) -> None:
    """Verify a complete or partial manifest without regenerating rooms."""
    if manifest.schema_version != 4:
        raise ValueError("room manifest schema does not match")
    if manifest.selection_algorithm != SELECTION_ALGORITHM:
        raise ValueError("room manifest selection algorithm does not match")
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


def assert_disjoint(manifest: CurriculumRoomManifest) -> None:
    if len({stage.stage for stage in manifest.stages}) != len(manifest.stages):
        raise ValueError("room manifest contains duplicate stage names")

    seed_owner: dict[int, str] = {}
    navigation_owner: dict[str, str] = {}
    task_owner: dict[str, str] = {}
    for stage in manifest.stages:
        if _hash_text(stage.config_json) != stage.config_sha256:
            raise ValueError(f"{stage.stage} config hash does not match its snapshot")
        cursor_names = tuple(name for name, _ in stage.selection_cursors)
        if (
            len(cursor_names) != len(ROOM_SPLITS)
            or set(cursor_names) != set(ROOM_SPLITS)
        ):
            raise ValueError(f"{stage.stage} selection cursors are incomplete")
        if any(cursor < 0 for _, cursor in stage.selection_cursors):
            raise ValueError(f"{stage.stage} has a negative selection cursor")
        target_names = tuple(name for name, _ in stage.feature_targets)
        if (
            len(target_names) != len(ROOM_SPLITS)
            or set(target_names) != set(ROOM_SPLITS)
        ):
            raise ValueError(f"{stage.stage} feature targets are incomplete")
        if any(target < 0 for _, target in stage.feature_targets):
            raise ValueError(f"{stage.stage} has a negative feature target")
        for split in ROOM_SPLITS:
            records = stage.records(split)
            seeds = tuple(record.seed for record in records)
            if len(seeds) != len(set(seeds)):
                raise ValueError(f"{stage.stage} {split} contains duplicate seeds")
            candidate_indexes = tuple(
                record.candidate_index for record in records
            )
            if (
                any(index < 0 for index in candidate_indexes)
                or len(candidate_indexes) != len(set(candidate_indexes))
                or tuple(sorted(candidate_indexes)) != candidate_indexes
            ):
                raise ValueError(
                    f"{stage.stage} {split} has invalid candidate indexes"
                )
            cursor = stage.selection_cursor(split)
            if candidate_indexes and cursor <= max(candidate_indexes):
                raise ValueError(
                    f"{stage.stage} {split} selection cursor is stale"
                )
            target = stage.feature_target(split)
            if records and target < len(records):
                raise ValueError(
                    f"{stage.stage} {split} feature target is too small"
                )
            navigations = [record.navigation_sha256 for record in records]
            tasks = [record.task_sha256 for record in records]
            if len(navigations) != len(set(navigations)):
                raise ValueError(f"{stage.stage} {split} contains duplicate navigation")
            if len(tasks) != len(set(tasks)):
                raise ValueError(f"{stage.stage} {split} contains duplicate tasks")
            for record in records:
                if record.seed != _candidate_seed(
                    manifest.data_seed,
                    stage.stage,
                    split,
                    record.candidate_index,
                ):
                    raise ValueError(
                        f"{stage.stage} {split} seed does not match its "
                        "candidate index"
                    )
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
    expected_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> None:
    verify_manifest_structure(manifest, stages)
    for split in splits:
        _check_split(split)
        if expected_counts is None:
            sizes = {len(stage.records(split)) for stage in manifest.stages}
            if len(sizes) != 1 or not sizes or next(iter(sizes)) < 1:
                raise ValueError(
                    f"{split} split sizes are incomplete or inconsistent"
                )
        else:
            split_counts = expected_counts.get(split)
            if split_counts is None:
                raise ValueError(f"missing expected counts for {split}")
            for stage in manifest.stages:
                expected = split_counts.get(stage.stage)
                if expected is None or expected < 0:
                    raise ValueError(
                        f"invalid expected count for {stage.stage} {split}"
                    )
                if len(stage.records(split)) != expected:
                    raise ValueError(
                        f"{stage.stage} {split} has "
                        f"{len(stage.records(split))}/{expected} rooms"
                    )

    for stage in stages:
        saved = manifest.stage(stage.name)
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
        if limit is None:
            raise ValueError(f"missing positive train limit for {stage.stage}")
        if limit < 1:
            if require_all:
                raise ValueError(f"missing positive train limit for {stage.stage}")
            continue
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


def _check_split(split: str) -> None:
    if split not in ROOM_SPLITS:
        raise KeyError(f"unknown room split {split!r}")


def _candidate_seed(
    data_seed: int,
    stage: str,
    split: str,
    candidate_index: int,
) -> int:
    return derive_seed(
        data_seed,
        f"curriculum-manifest:v2:{stage}:{split}:{candidate_index}",
    )


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

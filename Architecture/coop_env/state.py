"""Mutable per-episode state.

`EpisodeState` holds every value that can change while a room is being played:
which keys are gone, which switches are thrown, where the crates ended up, and
what the clock says. It is rebuilt from the immutable `Room` on every reset,
which is why reset is exact by construction -- there is no accumulated drift to
undo.

Scope note: this module implements *mechanism* behaviour only -- how a door
reacts to its own requirement, where a platform is at tick N, whether a plate
under a crate reads as pressed. It contains no agents, no movement, no
decision-making, and nothing that chooses to press anything. The mutator
methods (`collect_key`, `set_switch`, ...) are the seams a future control layer
would call; this project never calls them on its own behalf.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .entities import (
    LockedDoor,
    MovingPlatform,
    PressurePlate,
    PushableBlock,
    Switch,
    SwitchMode,
)
from .room import Room
from .tiles import Tile, is_hazard, is_walkable
from .utils.geometry import Vec2


@dataclass(slots=True)
class EpisodeState:
    """Live state for one episode of one room."""

    room: Room
    tick: int = 0

    keys_collected: set[str] = field(default_factory=set)
    switches_active: dict[str, bool] = field(default_factory=dict)
    checkpoints_reached: set[str] = field(default_factory=set)
    block_positions: dict[str, Vec2] = field(default_factory=dict)

    doors_open: dict[str, bool] = field(default_factory=dict)
    exit_open: bool = False

    # internal bookkeeping
    _plates_external: set[str] = field(default_factory=set, repr=False)
    _plates_pressed: dict[str, bool] = field(default_factory=dict, repr=False)
    _door_latched: dict[str, bool] = field(default_factory=dict, repr=False)
    _door_timers: dict[str, int] = field(default_factory=dict, repr=False)
    _door_spent: set[str] = field(default_factory=set, repr=False)
    """Timed doors whose window has closed; they re-arm when the trigger releases."""

    # -- construction ------------------------------------------------------

    @classmethod
    def from_room(cls, room: Room) -> "EpisodeState":
        state = cls(room=room)
        state.reset()
        return state

    def reset(self) -> "EpisodeState":
        """Return every mechanism to its blueprint default."""
        self.tick = 0
        self.keys_collected = set()
        self.checkpoints_reached = set()
        self.switches_active = {s.id: False for s in self.room.switches}
        self.block_positions = {b.id: b.pos for b in self.room.blocks}
        self._plates_external = set()
        self._plates_pressed = {p.id: False for p in self.room.plates}
        self._door_latched = {d.id: False for d in self.room.doors}
        self._door_timers = {}
        self._door_spent = set()
        self.doors_open = {d.id: False for d in self.room.doors}
        self.exit_open = False
        self.refresh()
        return self

    # -- requirement StateView protocol ------------------------------------

    def is_key_collected(self, key_id: str) -> bool:
        return key_id in self.keys_collected

    def is_switch_active(self, switch_id: str) -> bool:
        return self.switches_active.get(switch_id, False)

    def is_plate_pressed(self, plate_id: str) -> bool:
        return self._plates_pressed.get(plate_id, False)

    def is_checkpoint_reached(self, checkpoint_id: str) -> bool:
        return checkpoint_id in self.checkpoints_reached

    def is_door_open(self, door_id: str) -> bool:
        return self.doors_open.get(door_id, False)

    # -- mechanism mutators ------------------------------------------------
    # Seams for a future control layer. Nothing in this project calls them
    # except tests and the mechanism self-check tool.

    def collect_key(self, key_id: str) -> None:
        if self.room.find(key_id) is None:
            raise KeyError(f"no key {key_id!r} in this room")
        self.keys_collected.add(key_id)
        self.refresh()

    def set_switch(self, switch_id: str, active: bool) -> None:
        switch = self.room.find(switch_id)
        if not isinstance(switch, Switch):
            raise KeyError(f"no switch {switch_id!r} in this room")
        if switch.mode is SwitchMode.ONESHOT and self.switches_active.get(switch_id):
            return  # one-shot switches never turn back off
        self.switches_active[switch_id] = active
        self.refresh()

    def set_plate(self, plate_id: str, pressed: bool) -> None:
        plate = self.room.find(plate_id)
        if not isinstance(plate, PressurePlate):
            raise KeyError(f"no pressure plate {plate_id!r} in this room")
        if pressed:
            self._plates_external.add(plate_id)
        else:
            self._plates_external.discard(plate_id)
        self.refresh()

    def reach_checkpoint(self, checkpoint_id: str) -> None:
        if self.room.find(checkpoint_id) is None:
            raise KeyError(f"no checkpoint {checkpoint_id!r} in this room")
        self.checkpoints_reached.add(checkpoint_id)
        self.refresh()

    def place_block(self, block_id: str, pos: Vec2) -> None:
        block = self.room.find(block_id)
        if not isinstance(block, PushableBlock):
            raise KeyError(f"no pushable block {block_id!r} in this room")
        self.block_positions[block_id] = pos
        self.refresh()

    # -- time --------------------------------------------------------------

    def advance(self, ticks: int = 1) -> "EpisodeState":
        """Step the environment clock.

        Only time-driven mechanics respond: platform tracks, temporary bridges,
        and timed-door countdowns. Nothing moves through the room as a result.
        """
        if ticks < 0:
            raise ValueError("cannot advance time backwards")
        for _ in range(ticks):
            self.tick += 1
            for door_id in list(self._door_timers):
                remaining = self._door_timers[door_id] - 1
                if remaining <= 0:
                    del self._door_timers[door_id]
                    self._door_latched[door_id] = False
                    # Stay shut until the trigger is released and thrown again,
                    # otherwise a still-active switch would re-open it instantly.
                    self._door_spent.add(door_id)
                else:
                    self._door_timers[door_id] = remaining
            self.refresh()
        return self

    # -- derived state -----------------------------------------------------

    def refresh(self) -> None:
        """Recompute everything that follows from the primary state."""
        self._refresh_plates()
        self._refresh_doors()
        self.exit_open = self.room.exit.requirement.is_satisfied(self)

    def _refresh_plates(self) -> None:
        weighted = {
            pos for bid, pos in self.block_positions.items() if self._block_is_weight(bid)
        }
        for plate in self.room.plates:
            pressed = plate.id in self._plates_external
            if plate.accepts_block and plate.pos in weighted:
                pressed = True
            self._plates_pressed[plate.id] = pressed

    def _block_is_weight(self, block_id: str) -> bool:
        block = self.room.find(block_id)
        return isinstance(block, PushableBlock)

    def _refresh_doors(self) -> None:
        for door in self.room.doors:
            satisfied = door.requirement.is_satisfied(self)
            if door.latching:
                if not satisfied:
                    self._door_spent.discard(door.id)  # trigger released: re-arm
                if (
                    satisfied
                    and not self._door_latched[door.id]
                    and door.id not in self._door_spent
                ):
                    self._door_latched[door.id] = True
                    if door.timer:
                        self._door_timers[door.id] = door.timer
                self.doors_open[door.id] = self._door_latched[door.id]
            else:
                self.doors_open[door.id] = satisfied

    def platform_position(self, platform_id: str) -> Vec2:
        platform = self.room.find(platform_id)
        if not isinstance(platform, MovingPlatform):
            raise KeyError(f"no moving platform {platform_id!r} in this room")
        return platform.position_at(self.tick)

    def platform_positions(self) -> dict[str, Vec2]:
        return {p.id: p.position_at(self.tick) for p in self.room.platforms}

    def solid_bridge_tiles(self) -> set[Vec2]:
        tiles: set[Vec2] = set()
        for bridge in self.room.bridges:
            if bridge.is_solid_at(self.tick):
                tiles.update(bridge.tiles)
        return tiles

    def supported_hazard_tiles(self) -> set[Vec2]:
        """Hazard tiles currently made crossable by a platform or bridge."""
        supported = set(self.platform_positions().values())
        supported |= self.solid_bridge_tiles()
        return supported

    # -- occupancy queries -------------------------------------------------

    def blocking_entity_at(self, pos: Vec2) -> str | None:
        """Id of whatever currently blocks `pos`, or None."""
        for entity in self.room.entities_at(pos):
            if isinstance(entity, LockedDoor) and not self.doors_open.get(entity.id, False):
                return entity.id
        for block_id, block_pos in self.block_positions.items():
            if block_pos == pos:
                return block_id
        return None

    def is_walkable(self, pos: Vec2) -> bool:
        """Can this tile currently be stood on?

        Terrain rules first, then dynamic overrides: closed doors and crates
        block otherwise-open floor, platforms and bridges make hazard tiles
        temporarily crossable.
        """
        if not self.room.terrain.in_bounds(pos):
            return False
        tile = Tile(self.room.terrain[pos])
        if is_walkable(tile):
            return self.blocking_entity_at(pos) is None
        if is_hazard(tile):
            return pos in self.supported_hazard_tiles()
        return False

    def is_hazardous(self, pos: Vec2) -> bool:
        """Hazard terrain with nothing currently covering it."""
        if not self.room.terrain.in_bounds(pos):
            return False
        if not is_hazard(self.room.terrain[pos]):
            return False
        return pos not in self.supported_hazard_tiles()

    def walkable_tiles(self) -> set[Vec2]:
        return {p for p in self.room.terrain.positions() if self.is_walkable(p)}

    # -- progress readout --------------------------------------------------

    def objectives_remaining(self) -> list[str]:
        """Which exit prerequisites are still outstanding, as readable strings."""
        if self.exit_open:
            return []
        outstanding: list[str] = []
        for entity_id in sorted(self.room.exit.requirement.referenced_ids()):
            entity = self.room.find(entity_id)
            if entity is None:
                continue
            done = (
                self.is_key_collected(entity_id)
                or self.is_switch_active(entity_id)
                or self.is_plate_pressed(entity_id)
                or self.is_checkpoint_reached(entity_id)
            )
            if not done:
                outstanding.append(entity_id)
        return outstanding

    # -- persistence -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Serialisable copy of the live state (excluding the room blueprint)."""
        return {
            "tick": self.tick,
            "keys_collected": sorted(self.keys_collected),
            "switches_active": dict(sorted(self.switches_active.items())),
            "checkpoints_reached": sorted(self.checkpoints_reached),
            "block_positions": {k: tuple(v) for k, v in sorted(self.block_positions.items())},
            "plates_external": sorted(self._plates_external),
            "door_latched": dict(sorted(self._door_latched.items())),
            "door_timers": dict(sorted(self._door_timers.items())),
            "door_spent": sorted(self._door_spent),
        }

    def restore(self, snapshot: dict[str, Any]) -> "EpisodeState":
        self.tick = int(snapshot["tick"])
        self.keys_collected = set(snapshot["keys_collected"])
        self.switches_active = dict(snapshot["switches_active"])
        self.checkpoints_reached = set(snapshot["checkpoints_reached"])
        self.block_positions = {
            k: Vec2(*v) for k, v in snapshot["block_positions"].items()
        }
        self._plates_external = set(snapshot["plates_external"])
        self._door_latched = dict(snapshot["door_latched"])
        self._door_timers = dict(snapshot["door_timers"])
        self._door_spent = set(snapshot.get("door_spent", ()))
        self.refresh()
        return self

    def describe(self) -> str:
        open_doors = sum(1 for v in self.doors_open.values() if v)
        return (
            f"EpisodeState(tick={self.tick}, keys={len(self.keys_collected)}/"
            f"{len(self.room.keys)}, doors_open={open_doors}/{len(self.room.doors)}, "
            f"exit_open={self.exit_open})"
        )


def tiles_of(entities: Iterable[Any]) -> set[Vec2]:
    out: set[Vec2] = set()
    for entity in entities:
        out.update(entity.footprint())
    return out

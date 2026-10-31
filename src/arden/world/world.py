"""Headless world container, in-process event bus, creator API, and save/load."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


from arden.environment.body import (
    BodyConfig,
    BodyStateSystem,
    DriveState,
    Drives,
    resolve_body_config,
)
from arden.environment.daynight import DayNightConfig, DayNightCycle

from arden.world.objects import (
    Composite,
    ObjectFactory,
    Position,
    Velocity,
    WorldObject,
    object_radius,
    validate_world_object,
)

from arden.world.physics import (
    COLLISION_ITERATIONS,
    COLLISION_RESTITUTION,
    DEFAULT_OCCLUSION_TOLERANCE,
    MAX_SUBSTEP_DISTANCE,
    MAX_SUBSTEPS,
    SEPARATION_SLOP,
    SHAPE_RETENTION,
    STOP_SPEED_EPSILON,
    WATER_DRIFT_STRENGTH,
    WATER_FLOAT_MAX_WEIGHT,
    WATER_SINK_DRAG,
    TerrainGrid,
    Tile,
    TileType,
    _set_active_world,
    can_bind,
    compute_impulse_delta,
    compute_strike_force,
    is_occluded as _is_occluded_helper,
    is_stack_stable,
    max_carry_step,
    object_top_height,
    should_break_on_strike,
    stack_stability_score,
)

_OBJECT_ID = re.compile(r"^object_[0-9]+$")
_LEVEL = re.compile(r"^(?:ground|raised|on_top_of:object_[0-9]+|inside:object_[0-9]+)$")

EventType = Literal[
    "object_spawned",
    "object_despawned",
    "object_moved",
    "object_changed",
    "spoke",
    "time_of_day_changed",
    "cycle_frozen",
    "tick",
]
EVENT_TYPES: frozenset[str] = frozenset(
    {
        "object_spawned",
        "object_despawned",
        "object_moved",
        "object_changed",
        "spoke",
        "time_of_day_changed",
        "cycle_frozen",
        "tick",
    }
)
_EVENT_TYPE_ALIASES: dict[str, str] = {
    "time_of_day_set": "time_of_day_changed",
    "set_time_of_day": "time_of_day_changed",
    "time_changed": "time_of_day_changed",
    "freeze_cycle": "cycle_frozen",
    "cycle_freeze_changed": "cycle_frozen",
    "daynight_frozen": "cycle_frozen",
}


def _coerce_position(pos: Position | Mapping[str, Any] | tuple[float, float] | list[float]) -> Position:
    """Convert a Position, mapping, or (x, y) sequence into a validated Position."""
    if isinstance(pos, Position):
        return pos
    if isinstance(pos, Mapping):
        return Position.model_validate(dict(pos))
    if isinstance(pos, tuple | list) and len(pos) == 2:
        return Position(x=pos[0], y=pos[1])
    raise TypeError("position must be a Position, {'x', 'y'} mapping, or (x, y) pair")


class AgentBody(BaseModel):
    """Physical state of the agent's body in the world."""

    model_config = ConfigDict(extra="forbid")

    position: Position = Field(default_factory=lambda: Position(x=0.0, y=0.0))
    level: str = "ground"
    energy: float = Field(default=1.0, ge=0.0, le=1.0)
    comfort: float = Field(default=1.0, ge=0.0, le=1.0)
    holding: str | None = None

    @field_validator("level")
    @classmethod
    def level_is_known(cls, value: str) -> str:
        if _LEVEL.fullmatch(value) is None:
            raise ValueError(
                "level must be ground, raised, on_top_of:object_<n>, or inside:object_<n>"
            )
        return value

    @field_validator("energy", "comfort")
    @classmethod
    def values_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("body state values must be finite")
        return value

    @field_validator("holding")
    @classmethod
    def holding_is_object_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _OBJECT_ID.fullmatch(value) is None:
            raise ValueError("holding must be None or an object_<n> id")
        return value


class WorldEvent(BaseModel):
    """Base typed event emitted on the world event bus."""

    model_config = ConfigDict(extra="forbid")

    type: EventType
    tick: int = Field(ge=0)
    source: str = "world"
    object_id: str | None = None
    position: Position | None = None
    previous_position: Position | None = None
    changes: dict[str, Any] | None = None
    object: dict[str, Any] | None = None
    text: str | None = None
    time_of_day: float | None = None
    previous_time_of_day: float | None = None
    frozen: bool | None = None
    light_level: float | None = None
    ambient_temperature: float | None = None

    @property
    def kind(self) -> EventType:
        """Alias for ``type``."""
        return self.type

    @property
    def event_type(self) -> EventType:
        """Alias for ``type``."""
        return self.type

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


class ObjectSpawnedEvent(WorldEvent):
    """Emitted when an object is added to the world."""

    type: Literal["object_spawned"] = "object_spawned"
    object_id: str = Field(pattern=_OBJECT_ID.pattern)
    position: Position
    object: dict[str, Any]


class ObjectDespawnedEvent(WorldEvent):
    """Emitted when an object is removed from the world."""

    type: Literal["object_despawned"] = "object_despawned"
    object_id: str = Field(pattern=_OBJECT_ID.pattern)
    position: Position
    object: dict[str, Any]


class ObjectMovedEvent(WorldEvent):
    """Emitted when an object changes position."""

    type: Literal["object_moved"] = "object_moved"
    object_id: str = Field(pattern=_OBJECT_ID.pattern)
    previous_position: Position
    position: Position

    @property
    def from_position(self) -> Position:
        """Alias for ``previous_position``."""
        return self.previous_position

    @property
    def to_position(self) -> Position:
        """Alias for ``position``."""
        return self.position


class ObjectChangedEvent(WorldEvent):
    """Emitted when an object's properties or state change."""

    type: Literal["object_changed"] = "object_changed"
    object_id: str = Field(pattern=_OBJECT_ID.pattern)
    changes: dict[str, Any]
    object: dict[str, Any]


class TickEvent(WorldEvent):
    """Emitted at the end of each simulation tick."""

    type: Literal["tick"] = "tick"


class SpokeEvent(WorldEvent):
    """Emitted when the agent vocalizes via ``speak(text)``."""

    type: Literal["spoke"] = "spoke"
    text: str


class TimeOfDayChangedEvent(WorldEvent):
    """Emitted when the creator sets the world's time of day directly."""

    type: Literal["time_of_day_changed"] = "time_of_day_changed"
    source: str = "creator"
    previous_time_of_day: float
    time_of_day: float
    light_level: float
    ambient_temperature: float

    @property
    def fraction(self) -> float:
        """Alias for ``time_of_day``."""
        return self.time_of_day

    @property
    def light(self) -> float:
        """Alias for ``light_level``."""
        return self.light_level

    @property
    def temperature(self) -> float:
        """Alias for ``ambient_temperature``."""
        return self.ambient_temperature


TimeOfDayEvent = TimeOfDayChangedEvent


class CycleFrozenEvent(WorldEvent):
    """Emitted when the creator freezes or unfreezes the day/night cycle."""

    type: Literal["cycle_frozen"] = "cycle_frozen"
    source: str = "creator"
    frozen: bool
    time_of_day: float


CycleFreezeEvent = CycleFrozenEvent


EventHandler = Callable[[WorldEvent], None]


class EventBus:
    """Synchronous in-process pub/sub bus for typed world events."""

    def __init__(self) -> None:
        self._all_subscribers: list[EventHandler] = []
        self._internal_listeners: list[EventHandler] = []
        self._typed_subscribers: dict[str, list[EventHandler]] = {
            kind: [] for kind in EVENT_TYPES
        }

    def subscribe(
        self,
        handler_or_type: EventHandler | EventType,
        handler: EventHandler | None = None,
        *,
        event_type: EventType | None = None,
    ) -> Callable[[], None]:
        """Register a subscriber for all events or a specific event type."""
        if isinstance(handler_or_type, str):
            if handler is None or not callable(handler):
                raise TypeError("handler must be callable when event_type is given first")
            target_type: str | None = handler_or_type
            callback = handler
        elif callable(handler_or_type):
            callback = handler_or_type
            target_type = event_type if event_type is not None else (
                handler if isinstance(handler, str) else None
            )
        else:
            raise TypeError("subscribe expects a callable handler or (event_type, handler)")

        if target_type is not None:
            target_type = _EVENT_TYPE_ALIASES.get(target_type, target_type)
            if target_type not in EVENT_TYPES:
                allowed = ", ".join(sorted(EVENT_TYPES))
                raise ValueError(f"unknown event type {target_type!r}; expected one of: {allowed}")
            bucket = self._typed_subscribers[target_type]
            bucket.append(callback)

            def _unsubscribe_typed() -> None:
                if callback in bucket:
                    bucket.remove(callback)

            return _unsubscribe_typed

        self._all_subscribers.append(callback)

        def _unsubscribe_all() -> None:
            if callback in self._all_subscribers:
                self._all_subscribers.remove(callback)

        return _unsubscribe_all

    def unsubscribe(
        self,
        handler: EventHandler,
        event_type: EventType | None = None,
    ) -> None:
        """Remove a previously registered handler."""
        if event_type is not None:
            resolved_type = _EVENT_TYPE_ALIASES.get(event_type, event_type)
            if resolved_type not in EVENT_TYPES:
                allowed = ", ".join(sorted(EVENT_TYPES))
                raise ValueError(f"unknown event type {event_type!r}; expected one of: {allowed}")
            bucket = self._typed_subscribers[resolved_type]
            if handler in bucket:
                bucket.remove(handler)
            return

        if handler in self._all_subscribers:
            self._all_subscribers.remove(handler)
        for bucket in self._typed_subscribers.values():
            if handler in bucket:
                bucket.remove(handler)

    def emit(self, event: WorldEvent) -> None:
        """Deliver ``event`` to matching subscribers in registration order."""
        for callback in list(self._internal_listeners):
            callback(event)
        for callback in list(self._typed_subscribers.get(event.type, ())):
            callback(event)
        for callback in list(self._all_subscribers):
            callback(event)


class _RestoreDescriptor:
    """Allow ``restore`` to be called on a ``World`` instance or on the ``World`` class."""

    def __get__(
        self,
        instance: World | None,
        owner: type[World],
    ) -> Callable[[str | bytes | Mapping[str, Any]], World]:
        if instance is None:
            return owner._restore_from_cls
        return instance._restore_from_instance


class World:
    """Headless simulation container holding objects, the agent body, tick, and RNG."""

    def __init__(
        self,
        rng: random.Random | int | None = None,
        *,
        seed: int | None = None,
        body: AgentBody | Mapping[str, Any] | None = None,
        objects: Iterable[WorldObject | Mapping[str, Any]]
        | Mapping[str, WorldObject | Mapping[str, Any]]
        | None = None,
        terrain: TerrainGrid | Mapping[str, Any] | None = None,
        daynight: DayNightCycle | DayNightConfig | Mapping[str, Any] | None = None,
        config: Any = None,
        body_config: BodyConfig | Mapping[str, Any] | str | None = None,
        preset: str | None = None,
        drives: DriveState | Mapping[str, Any] | None = None,
        weather: Any = "clear",
    ) -> None:
        if isinstance(rng, random.Random):
            if seed is not None:
                raise ValueError("pass either rng or seed, not both")
            self.rng: random.Random = rng
        elif isinstance(rng, int):
            if seed is not None:
                raise ValueError("pass either rng or seed, not both")
            self.rng = random.Random(rng)
        elif rng is None:
            self.rng = random.Random(0 if seed is None else seed)
        else:
            raise TypeError("rng must be a random.Random instance, an int seed, or None")

        if body is None:
            self.body: AgentBody = AgentBody()
        elif isinstance(body, AgentBody):
            self.body = body.model_copy(deep=True)
        elif isinstance(body, Mapping):
            self.body = AgentBody.model_validate(dict(body))
        else:
            raise TypeError("body must be an AgentBody, a mapping, or None")

        if terrain is None:
            self.terrain: TerrainGrid = TerrainGrid()
        elif isinstance(terrain, TerrainGrid):
            self.terrain = TerrainGrid.from_dict(terrain.to_dict())
        elif isinstance(terrain, Mapping):
            self.terrain = TerrainGrid.from_dict(terrain)
        else:
            raise TypeError("terrain must be a TerrainGrid, a mapping, or None")

        self.config: Any = config
        if daynight is None and config is not None:
            dn_cfg = getattr(config, "daynight", None)
            if dn_cfg is not None:
                daynight = dn_cfg

        if daynight is None:
            self.daynight: DayNightCycle = DayNightCycle()
        elif isinstance(daynight, DayNightCycle):
            self.daynight = DayNightCycle.from_dict(daynight.to_dict())
        elif isinstance(daynight, DayNightConfig):
            self.daynight = DayNightCycle(config=daynight)
        elif isinstance(daynight, Mapping):
            if "config" in daynight or "time_of_day" in daynight or "frozen" in daynight:
                self.daynight = DayNightCycle.from_dict(daynight)
            else:
                self.daynight = DayNightCycle(config=daynight)
        else:
            raise TypeError("daynight must be a DayNightCycle, DayNightConfig, mapping, or None")

        self.tick: int = 0
        self._factory = ObjectFactory()
        self.objects: dict[str, WorldObject] = {}
        self.events = EventBus()
        self._manual_body_modifiers: Any = None
        self.weather: Any = weather
        self.weather_intensity: float = 1.0
        self._sheltered_override: float | bool | None = None
        resolved_body_cfg = (
            BodyConfig.from_preset(preset)
            if preset is not None and body_config is None
            else resolve_body_config(body_config, self)
        )
        self._body_state_system: BodyStateSystem = BodyStateSystem(
            self,
            config=resolved_body_cfg,
            drives=drives,
        )
        self._opaque_actions: dict[str, dict[str, Callable[..., Any]]] = {}
        self._perception_event_log: list[WorldEvent] = []
        self._has_been_perceived: bool = False
        self._auto_held_composite_id: str | None = None
        self.events._internal_listeners.append(self._record_perception_event)

        if objects is not None:
            items: Iterable[WorldObject | Mapping[str, Any]]
            if isinstance(objects, Mapping):
                items = objects.values()
            else:
                items = objects
            for item in items:
                if isinstance(item, WorldObject):
                    created = self._factory.create(item)
                elif isinstance(item, Mapping):
                    created = self._factory.create(item)
                else:
                    raise TypeError("objects must contain WorldObject instances or mappings")
                self.objects[created.id] = created

        _set_active_world(self)

    def _record_perception_event(self, event: WorldEvent) -> None:
        if event.type == "tick":
            return
        if event.type == "object_spawned" and not self._has_been_perceived:
            return
        self._perception_event_log.append(event)

    @property
    def agent_body(self) -> AgentBody:
        """Alias for ``body``."""
        return self.body

    @property
    def event_bus(self) -> EventBus:
        """Alias for ``events``."""
        return self.events

    @property
    def day_night(self) -> DayNightCycle:
        """Alias for ``daynight``."""
        return self.daynight

    @property
    def time_of_day(self) -> float:
        """Current time-of-day fraction in ``[0.0, 1.0)`` (0.0=midnight, 0.5=noon)."""
        return self.daynight.time_of_day

    @property
    def cycle_frozen(self) -> bool:
        """Return True when the day/night cycle is frozen."""
        return self.daynight.frozen

    @property
    def is_cycle_frozen(self) -> bool:
        """Alias for ``cycle_frozen``."""
        return self.daynight.frozen

    @property
    def light_level(self) -> float:
        """Current smooth light level in ``[0.0, 1.0]``."""
        return self.daynight.light

    @property
    def light(self) -> float:
        """Alias for ``light_level``."""
        return self.daynight.light

    @property
    def coarse_light_level(self) -> float:
        """Current coarse numeric light level in ``[0.0, 1.0]``."""
        return self.daynight.coarse_light

    @property
    def coarse_light(self) -> float:
        """Alias for ``coarse_light_level``."""
        return self.daynight.coarse_light

    @property
    def visibility_factor(self) -> float:
        """Current perception ``visibility_factor`` driven by the day/night cycle."""
        return self.daynight.visibility_factor

    @property
    def ambient_temperature(self) -> float:
        """Current ambient temperature exposed for the body-state model."""
        return self.daynight.ambient_temperature

    @property
    def temperature(self) -> float:
        """Alias for ``ambient_temperature``."""
        return self.daynight.ambient_temperature

    @property
    def body_state(self) -> BodyStateSystem:
        """Stateful body energy, comfort, side-effect, and drive manager."""
        return self._body_state_system

    @property
    def body_config(self) -> BodyConfig:
        """Current ``BodyConfig`` under ``[body]``."""
        return self._body_state_system.config

    @body_config.setter
    def body_config(self, value: BodyConfig | Mapping[str, Any] | str) -> None:
        self._body_state_system.config = resolve_body_config(value, self)

    @property
    def drives(self) -> DriveState:
        """Intrinsic boredom and curiosity drive state."""
        return self._body_state_system.drives

    @drives.setter
    def drives(self, value: DriveState | Mapping[str, Any]) -> None:
        if isinstance(value, DriveState):
            self._body_state_system.drives = value
        elif isinstance(value, Mapping):
            self._body_state_system.drives = DriveState.model_validate(dict(value))
        else:
            raise TypeError("drives must be a DriveState or mapping")

    @property
    def boredom(self) -> float:
        """Current boredom drive level in ``[0.0, 1.0]``."""
        return self._body_state_system.drives.boredom

    @boredom.setter
    def boredom(self, value: float) -> None:
        self._body_state_system.drives.boredom = max(0.0, min(1.0, float(value)))

    @property
    def curiosity(self) -> float:
        """Current accumulated curiosity reward."""
        return self._body_state_system.drives.curiosity

    @curiosity.setter
    def curiosity(self, value: float) -> None:
        self._body_state_system.drives.curiosity = max(0.0, float(value))

    @property
    def curiosity_reward(self) -> float:
        """Alias for ``curiosity``."""
        return self._body_state_system.drives.curiosity

    @curiosity_reward.setter
    def curiosity_reward(self, value: float) -> None:
        self._body_state_system.drives.curiosity = max(0.0, float(value))

    @property
    def body_modifiers(self) -> Any:
        """Current ``BodyModifiers`` derived from energy, comfort, cold, and wet."""
        if self._manual_body_modifiers is not None:
            return self._manual_body_modifiers
        return self._body_state_system.modifiers

    @body_modifiers.setter
    def body_modifiers(self, value: Any) -> None:
        self._manual_body_modifiers = value

    @property
    def modifiers(self) -> Any:
        """Alias for ``body_modifiers``."""
        return self.body_modifiers

    @property
    def wet(self) -> float:
        """Current agent wetness level in ``[0.0, 1.0]``."""
        return self._body_state_system.wet

    @wet.setter
    def wet(self, value: float | bool | None) -> None:
        self._body_state_system.wet = value

    @property
    def cold(self) -> float:
        """Current agent cold exposure level in ``[0.0, 1.0]``."""
        return self._body_state_system.cold

    @cold.setter
    def cold(self, value: float | bool | None) -> None:
        self._body_state_system.cold = value

    @property
    def sheltered(self) -> float:
        """Current shelter level in ``[0.0, 1.0]``."""
        return self._body_state_system.current_shelter_level()

    @sheltered.setter
    def sheltered(self, value: float | bool | None) -> None:
        self._sheltered_override = value

    @property
    def memory_salience(self) -> float:
        """Current episodic memory salience score from drives and body state."""
        return self._body_state_system.memory_salience

    def gate_triggers(self) -> list[str]:
        """Return drive/body triggers currently exceeding gate wake thresholds."""
        return self._body_state_system.gate_triggers()

    @property
    def should_wake_gate(self) -> bool:
        """Return True if any drive or body threshold warrants waking the LLM."""
        return self._body_state_system.should_wake_gate

    def record_activity(
        self,
        verb: str,
        *,
        force: float | None = None,
        speed: float | None = None,
    ) -> float:
        """Drain energy immediately for an executed action, weighted by ``verb`` and ``force``."""
        return self._body_state_system.record_activity(verb, force=force, speed=speed)

    def rest(self, ticks: int = 1) -> tuple[float, float]:
        """Restore energy and comfort via rest (idle). Returns ``(energy, comfort)``."""
        return self._body_state_system.rest(ticks)

    def eat(self, target: str | WorldObject | None = None) -> Any:
        """Eat an object with an ``edible`` rule within reach to restore energy."""
        return self._body_state_system.eat(target)

    def set_time_of_day(
        self,
        fraction: float,
        *,
        source: str = "creator",
    ) -> float:
        """Set the world's time of day to ``fraction`` in ``[0.0, 1.0]``, emitting ``time_of_day_changed``."""
        _set_active_world(self)
        prev_tod = self.daynight.time_of_day
        new_tod = self.daynight.set_time_of_day(fraction)
        self.events.emit(
            TimeOfDayChangedEvent(
                tick=self.tick,
                source=source,
                previous_time_of_day=prev_tod,
                time_of_day=new_tod,
                light_level=self.daynight.light,
                ambient_temperature=self.daynight.ambient_temperature,
            )
        )
        return new_tod

    def freeze_cycle(
        self,
        frozen: bool = True,
        *,
        source: str = "creator",
    ) -> bool:
        """Freeze or unfreeze the day/night cycle, emitting ``cycle_frozen``."""
        _set_active_world(self)
        new_frozen = self.daynight.freeze_cycle(frozen)
        self.events.emit(
            CycleFrozenEvent(
                tick=self.tick,
                source=source,
                frozen=new_frozen,
                time_of_day=self.daynight.time_of_day,
            )
        )
        return new_frozen

    def subscribe(
        self,
        handler_or_type: EventHandler | EventType,
        handler: EventHandler | None = None,
        *,
        event_type: EventType | None = None,
    ) -> Callable[[], None]:
        """Register an event subscriber on the world's event bus."""
        return self.events.subscribe(handler_or_type, handler, event_type=event_type)

    def unsubscribe(
        self,
        handler: EventHandler,
        event_type: EventType | None = None,
    ) -> None:
        """Remove an event subscriber from the world's event bus."""
        self.events.unsubscribe(handler, event_type=event_type)

    def spawn(
        self,
        props: Mapping[str, Any] | WorldObject,
        pos: Position | Mapping[str, Any] | tuple[float, float] | list[float] | None = None,
        *,
        source: str = "creator",
    ) -> WorldObject:
        """Create and add an object to the world, emitting ``object_spawned``."""
        _set_active_world(self)
        if isinstance(props, WorldObject):
            data = props.model_dump()
            if data.get("id") == "object_0" or data.get("id") in self._factory._used:
                data.pop("id", None)
        elif isinstance(props, Mapping):
            data = dict(props)
        else:
            raise TypeError("props must be a mapping or WorldObject")

        if pos is not None:
            position = _coerce_position(pos)
            data["position"] = {"x": position.x, "y": position.y}

        obj = self._factory.create(data)
        self.objects[obj.id] = obj
        self.events.emit(
            ObjectSpawnedEvent(
                tick=self.tick,
                source=source,
                object_id=obj.id,
                position=obj.position,
                object=obj.model_dump(mode="json"),
            )
        )
        return obj

    def _restore_part_object(
        self,
        part: WorldObject,
        pos: Position,
        level: str,
        *,
        source: str = "agent",
    ) -> WorldObject:
        """Restore a constituent ``part`` back into ``self.objects`` with its original id."""
        _set_active_world(self)
        part_id = part.id
        if part_id in self.objects or part_id == "object_0":
            part_id = self._factory._peek_next()

        data = part.model_dump()
        data["id"] = part_id
        data["position"] = {"x": pos.x, "y": pos.y}
        data["level"] = level
        data["velocity"] = {"x": 0.0, "y": 0.0}
        restored = validate_world_object(data)
        self._factory._claim(restored.id)
        self.objects[restored.id] = restored
        self.events.emit(
            ObjectSpawnedEvent(
                tick=self.tick,
                source=source,
                object_id=restored.id,
                position=restored.position,
                object=restored.model_dump(mode="json"),
            )
        )
        return restored

    def despawn(self, object_id: str, *, source: str = "creator") -> WorldObject:
        """Remove an object by id, emitting ``object_despawned``."""
        _set_active_world(self)
        if object_id not in self.objects:
            raise KeyError(f"unknown object id: {object_id}")
        obj = self.objects.pop(object_id)
        self._opaque_actions.pop(object_id, None)
        if self.body.holding == object_id:
            self.body.holding = None
        if self._auto_held_composite_id == object_id:
            self._auto_held_composite_id = None
        if self.body.level == f"on_top_of:{object_id}":
            self.body.level = "raised" if self.terrain.is_ledge(self.body.position) else "ground"
        self.events.emit(
            ObjectDespawnedEvent(
                tick=self.tick,
                source=source,
                object_id=obj.id,
                position=obj.position,
                object=obj.model_dump(mode="json"),
            )
        )
        # Topple any objects that were resting on top of or inside the despawned object
        for upper_id in sorted(self.objects):
            upper = self.objects.get(upper_id)
            if upper is None:
                continue
            if upper.level == f"on_top_of:{object_id}" or upper.level == f"inside:{object_id}":
                self._topple_object(upper.id, support=obj, move_dx=0.0, move_dy=0.0, source=source)
        return obj

    def move(
        self,
        object_id: str,
        pos: Position | Mapping[str, Any] | tuple[float, float] | list[float],
        *,
        source: str = "creator",
    ) -> WorldObject:
        """Move an existing object to ``pos``, emitting ``object_moved``."""
        _set_active_world(self)
        if object_id not in self.objects:
            raise KeyError(f"unknown object id: {object_id}")
        new_pos = _coerce_position(pos)
        current = self.objects[object_id]
        previous_pos = current.position
        updated = current.model_copy(update={"position": new_pos})
        self.objects[object_id] = updated
        self.events.emit(
            ObjectMovedEvent(
                tick=self.tick,
                source=source,
                object_id=object_id,
                previous_position=previous_pos,
                position=new_pos,
            )
        )
        self._propagate_support_movement(
            object_id,
            previous_pos=previous_pos,
            new_pos=new_pos,
            source=source,
        )
        return self.objects[object_id]

    def update_object(
        self,
        object_id: str,
        changes: Mapping[str, Any],
        *,
        source: str = "world",
    ) -> WorldObject:
        """Update properties or state on an object, emitting ``object_changed``."""
        _set_active_world(self)
        if object_id not in self.objects:
            raise KeyError(f"unknown object id: {object_id}")
        if not isinstance(changes, Mapping):
            raise TypeError("changes must be a mapping")
        if "id" in changes and changes["id"] != object_id:
            raise ValueError("cannot change an object's id")

        current = self.objects[object_id]
        merged = {**current.model_dump(), **dict(changes)}
        updated = validate_world_object(merged)
        self.objects[object_id] = updated

        serialized_changes = json.loads(json.dumps(dict(changes)))
        self.events.emit(
            ObjectChangedEvent(
                tick=self.tick,
                source=source,
                object_id=object_id,
                changes=serialized_changes,
                object=updated.model_dump(mode="json"),
            )
        )
        return updated

    def set_tile(
        self,
        x_or_pos: int | float | Position | Mapping[str, Any] | tuple[float, float] | list[float],
        y_or_tile: int | float | Tile | TileType | Mapping[str, Any] | None = None,
        tile: Tile | TileType | Mapping[str, Any] | None = None,
        *,
        flow: Velocity | Position | Mapping[str, Any] | tuple[float, float] | list[float] | None = None,
    ) -> Tile:
        """Set a terrain tile on ``self.terrain``."""
        return self.terrain.set_tile(x_or_pos, y_or_tile, tile, flow=flow)

    def tile_at(
        self,
        x_or_pos: float | Position | Mapping[str, Any] | tuple[float, float] | list[float],
        y: float | None = None,
    ) -> Tile:
        """Return the terrain ``Tile`` at ``pos``."""
        return self.terrain.get_tile(x_or_pos, y)

    def can_reach(
        self,
        target_pos: Position | Mapping[str, Any] | tuple[float, float] | list[float],
        *,
        from_pos: Position | Mapping[str, Any] | tuple[float, float] | list[float] | None = None,
        from_level: str | None = None,
    ) -> bool:
        """Return True if ``target_pos`` is reachable from ``from_pos`` and ``from_level``."""
        origin_pos = self.body.position if from_pos is None else _coerce_position(from_pos)
        origin_level = self.body.level if from_level is None else from_level
        return self.terrain.can_reach(
            target_pos,
            from_pos=origin_pos,
            from_level=origin_level,
            objects=self.objects,
        )

    can_reach_tile = can_reach

    def climb_onto(self, object_id: str) -> bool:
        """Climb the agent body onto ``object_id`` if reachable and tall enough."""
        _set_active_world(self)
        if object_id not in self.objects:
            raise KeyError(f"unknown object id: {object_id}")
        target = self.objects[object_id]
        if self.body.holding == object_id or target.height <= 0.0:
            return False
        if target.level == "raised" and self.body.level == "ground":
            return False
        dist = math.hypot(
            target.position.x - self.body.position.x,
            target.position.y - self.body.position.y,
        )
        if dist > target.radius + 1.5:
            return False
        self.body.position = target.position.model_copy()
        self.body.level = f"on_top_of:{object_id}"
        return True

    def move_agent(
        self,
        target_pos: Position | Mapping[str, Any] | tuple[float, float] | list[float],
    ) -> bool:
        """Move the agent body to ``target_pos`` if terrain level rules allow it."""
        _set_active_world(self)
        dest = _coerce_position(target_pos)
        if not self.can_reach(dest):
            return False
        dest_tile = self.terrain.get_tile(dest)
        self.body.position = dest
        self.body.level = "raised" if dest_tile.type == "ledge" else "ground"
        return True

    def apply_impulse(
        self,
        object_id: str,
        direction: Position | Velocity | Mapping[str, Any] | tuple[float, float] | list[float] | str | int | float,
        force: float,
    ) -> WorldObject:
        """Apply a directional impulse of ``force`` to ``object_id``."""
        _set_active_world(self)
        if object_id not in self.objects:
            raise KeyError(f"unknown object id: {object_id}")
        obj = self.objects[object_id]
        dv = compute_impulse_delta(obj, direction, force)
        if dv.speed > 0.0:
            new_vel = Velocity(x=obj.velocity.x + dv.x, y=obj.velocity.y + dv.y)
            self.objects[object_id] = obj.model_copy(update={"velocity": new_vel})
        return self.objects[object_id]

    def push(
        self,
        object_id: str,
        direction: Position | Velocity | Mapping[str, Any] | tuple[float, float] | list[float] | str | int | float = (1.0, 0.0),
        force: float | None = None,
    ) -> WorldObject:
        """Push ``object_id`` with ``force`` along ``direction``."""
        if force is None:
            if isinstance(direction, int | float) and not isinstance(direction, bool):
                return self.apply_impulse(object_id, (1.0, 0.0), float(direction))
            return self.apply_impulse(object_id, direction, 0.5)
        return self.apply_impulse(object_id, direction, float(force))

    def strike(
        self,
        object_id: str,
        force_or_direction: Any = 0.5,
        force: float | None = None,
        *,
        direction: Any = (1.0, 0.0),
        tool: str | WorldObject | None = None,
        source: str = "world",
    ) -> WorldObject:
        """Strike ``object_id`` with ``force``; fragile objects break, movable ones receive an impulse."""
        _set_active_world(self)
        if object_id not in self.objects:
            raise KeyError(f"unknown object id: {object_id}")

        if force is None:
            if isinstance(force_or_direction, int | float) and not isinstance(force_or_direction, bool):
                force_val = float(force_or_direction)
                dir_val = direction
            else:
                force_val = 0.5
                dir_val = force_or_direction
        else:
            force_val = float(force)
            dir_val = force_or_direction

        tool_obj: WorldObject | None = None
        if isinstance(tool, WorldObject):
            tool_obj = tool
        elif isinstance(tool, str):
            tool_obj = self.objects.get(tool)
        elif self.body.holding is not None and self.body.holding != object_id:
            tool_obj = self.objects.get(self.body.holding)

        effective_impact = compute_strike_force(force_val, tool_obj)

        obj = self.objects[object_id]
        if should_break_on_strike(obj, effective_impact):
            new_state = {**obj.state, "broken": True}
            return self.update_object(object_id, {"state": new_state}, source=source)

        if not obj.fixed and effective_impact > 0.0:
            self.apply_impulse(object_id, dir_val, effective_impact)
        return self.objects[object_id]

    def bind(
        self,
        first_id: str | WorldObject,
        second_id: str | WorldObject,
        binder: str | WorldObject | None = None,
        *,
        binder_id: str | None = None,
        source: str = "world",
    ) -> Any:
        """Bind ``first_id`` and ``second_id`` using a binding object.

        When called with ``binder_id=...`` (legacy property check), updates
        ``state["bound_to"]`` and returns ``bool``. When called with positional
        ``binder`` (building API), joins the objects into a ``Composite``.
        """
        _set_active_world(self)
        fid = first_id.id if isinstance(first_id, WorldObject) else str(first_id)
        sid = second_id.id if isinstance(second_id, WorldObject) else str(second_id)

        if binder is None and binder_id is not None:
            if fid not in self.objects:
                raise KeyError(f"unknown object id: {fid}")
            if sid not in self.objects:
                raise KeyError(f"unknown object id: {sid}")
            if fid == sid:
                raise ValueError("cannot bind an object to itself")
            if binder_id not in self.objects:
                raise KeyError(f"unknown object id: {binder_id}")
            first_obj = self.objects[fid]
            second_obj = self.objects[sid]
            binder_obj = self.objects[binder_id]
            if not can_bind(binder_obj):
                return False

            first_bound = list(first_obj.state.get("bound_to", []))
            if sid not in first_bound:
                first_bound.append(sid)
            second_bound = list(second_obj.state.get("bound_to", []))
            if fid not in second_bound:
                second_bound.append(fid)

            self.update_object(
                fid,
                {"state": {**first_obj.state, "bound_to": first_bound}},
                source=source,
            )
            self.update_object(
                sid,
                {"state": {**second_obj.state, "bound_to": second_bound}},
                source=source,
            )
            return True

        from arden.actions.verbs import bind as _verb_bind

        res = _verb_bind(self, first_id, second_id, binder)
        if res.ok and res.composite is not None:
            return res.composite
        return False

    def _top_of_stack(self, base_id: str, *, exclude_id: str | None = None) -> str:
        """Find the top-most object id currently resting in the stack above ``base_id``."""
        curr_id = base_id
        visited: set[str] = {base_id}
        if exclude_id is not None:
            visited.add(exclude_id)
        while True:
            next_id: str | None = None
            for oid in sorted(self.objects):
                if oid in visited:
                    continue
                if self.objects[oid].level == f"on_top_of:{curr_id}":
                    next_id = oid
                    break
            if next_id is None:
                break
            visited.add(next_id)
            curr_id = next_id
        return curr_id

    def stack(
        self,
        upper_id: str | WorldObject,
        lower_id: str | WorldObject,
        *,
        source: str = "world",
    ) -> WorldObject:
        """Place ``upper_id`` on top of ``lower_id``, toppling immediately if unstable."""
        _set_active_world(self)
        uid = upper_id.id if isinstance(upper_id, WorldObject) else str(upper_id)
        lid = lower_id.id if isinstance(lower_id, WorldObject) else str(lower_id)
        if uid not in self.objects:
            raise KeyError(f"unknown object id: {uid}")
        if lid not in self.objects:
            raise KeyError(f"unknown object id: {lid}")
        if uid == lid:
            raise ValueError("cannot stack an object on top of itself")

        support_id = self._top_of_stack(lid, exclude_id=uid)
        lower = self.objects[support_id]
        upper = self.objects[uid]
        if self.body.holding == uid:
            self.body.holding = None
        if self._auto_held_composite_id == uid:
            self._auto_held_composite_id = None

        prev_pos = upper.position
        target_level = f"on_top_of:{support_id}"
        score = stack_stability_score(upper, lower, self.objects)
        new_state = {**upper.state, "made_by_agent": True, "stability_score": score}
        updated = upper.model_copy(
            update={
                "position": lower.position.model_copy(),
                "level": target_level,
                "velocity": Velocity(x=0.0, y=0.0),
                "state": new_state,
            }
        )
        self.objects[uid] = updated
        if prev_pos != lower.position:
            self.events.emit(
                ObjectMovedEvent(
                    tick=self.tick,
                    source=source,
                    object_id=uid,
                    previous_position=prev_pos,
                    position=lower.position,
                )
            )
            self._propagate_container_contents(uid, new_pos=lower.position, source=source)
        self.events.emit(
            ObjectChangedEvent(
                tick=self.tick,
                source=source,
                object_id=uid,
                changes={"level": target_level},
                object=updated.model_dump(mode="json"),
            )
        )
        if not is_stack_stable(updated, lower, self.objects):
            self._topple_object(uid, support=lower, move_dx=0.0, move_dy=0.0, source=source)
            self.resolve_collisions()
        return self.objects[uid]

    def insert(
        self,
        held_id: str | WorldObject,
        target_id: str | WorldObject,
    ) -> Any:
        """Insert ``held_id`` into hollow ``target_id``."""
        from arden.actions.verbs import insert as _verb_insert

        return _verb_insert(self, held_id, target_id)

    def separate(
        self,
        target_id: str | WorldObject,
    ) -> Any:
        """Separate a composite, container, or stack back into its parts."""
        from arden.actions.verbs import separate as _verb_separate

        return _verb_separate(self, target_id)

    def is_occluded(
        self,
        viewer_pos: Any,
        target: Any,
        *,
        tolerance: float = DEFAULT_OCCLUSION_TOLERANCE,
    ) -> bool:
        """Return True when a taller object lies close to the line between ``viewer_pos`` and ``target``."""
        _set_active_world(self)
        return _is_occluded_helper(
            viewer_pos,
            target,
            self.objects,
            tolerance=tolerance,
            world=self,
        )

    def object_top_height(self, object_id: str) -> float:
        """Return the top elevation (base elevation + height) of ``object_id``."""
        if object_id not in self.objects:
            raise KeyError(f"unknown object id: {object_id}")
        return object_top_height(self.objects[object_id], self.objects, self.terrain)

    def register_opaque_action(
        self,
        object_id: str | WorldObject,
        action_or_handler: str | Callable[..., Any],
        handler: Callable[..., Any] | None = None,
    ) -> str:
        """Register an opaque ``action_<n>`` with a handler on ``object_id``."""
        from arden.actions.verbs import register_opaque_action as _reg

        return _reg(self, object_id, action_or_handler, handler)

    register_action = register_opaque_action

    def available_opaque_actions(
        self,
        *,
        target: str | WorldObject | None = None,
        modifiers: Any = None,
    ) -> list[str]:
        """Return ``action_<n>`` ids offered by objects currently within reach."""
        from arden.actions.verbs import available_opaque_actions as _avail

        return _avail(self, target=target, modifiers=modifiers)

    available_actions = available_opaque_actions

    def invoke_opaque_action(
        self,
        action_id: str,
        *args: Any,
        target: str | WorldObject | None = None,
        modifiers: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke a registered opaque ``action_<n>`` if its object is within reach."""
        from arden.actions.verbs import invoke_opaque_action as _invoke

        return _invoke(self, action_id, *args, target=target, modifiers=modifiers, **kwargs)

    def _propagate_container_contents(
        self,
        container_id: str,
        *,
        new_pos: Position,
        source: str = "world",
        _visited: set[str] | None = None,
    ) -> None:
        """Move any objects with ``level == 'inside:<container_id>'`` along with ``container_id``."""
        if _visited is None:
            _visited = set()
        if container_id in _visited:
            return
        _visited.add(container_id)

        for child_id in sorted(self.objects):
            child = self.objects.get(child_id)
            if child is None or child.level != f"inside:{container_id}":
                continue
            prev_pos = child.position
            if prev_pos != new_pos:
                updated_child = child.model_copy(update={"position": new_pos.model_copy()})
                self.objects[child_id] = updated_child
                self.events.emit(
                    ObjectMovedEvent(
                        tick=self.tick,
                        source=source,
                        object_id=child_id,
                        previous_position=prev_pos,
                        position=new_pos.model_copy(),
                    )
                )
            self._propagate_container_contents(
                child_id,
                new_pos=new_pos,
                source=source,
                _visited=_visited,
            )

    def _topple_object(
        self,
        upper_id: str,
        *,
        support: WorldObject,
        move_dx: float,
        move_dy: float,
        source: str = "world",
        _visited: set[str] | None = None,
    ) -> None:
        """Topple ``upper_id`` (and anything stacked on it) off ``support`` onto the plane."""
        if _visited is None:
            _visited = set()
        if upper_id in _visited or upper_id not in self.objects:
            return
        _visited.add(upper_id)

        upper = self.objects[upper_id]
        prev_pos = upper.position

        # Determine fall direction (opposite to support movement or deterministic from world RNG)
        move_len = math.hypot(move_dx, move_dy)
        if move_len > 1e-9:
            dir_x = -move_dx / move_len
            dir_y = -move_dy / move_len
        else:
            angle = self.rng.uniform(0.0, 2.0 * math.pi)
            dir_x = math.cos(angle)
            dir_y = math.sin(angle)

        sep_dist = support.radius + upper.radius + 0.05
        candidate_pos = Position(
            x=support.position.x + dir_x * sep_dist,
            y=support.position.y + dir_y * sep_dist,
        )
        landing_level = "raised" if self.terrain.is_ledge(candidate_pos) else "ground"

        updated = upper.model_copy(
            update={
                "position": candidate_pos,
                "level": landing_level,
                "velocity": Velocity(x=0.0, y=0.0),
            }
        )
        self.objects[upper_id] = updated

        if prev_pos != candidate_pos:
            self.events.emit(
                ObjectMovedEvent(
                    tick=self.tick,
                    source=source,
                    object_id=upper_id,
                    previous_position=prev_pos,
                    position=candidate_pos,
                )
            )
            self._propagate_container_contents(
                upper_id,
                new_pos=candidate_pos,
                source=source,
            )
        if upper.level != landing_level:
            self.events.emit(
                ObjectChangedEvent(
                    tick=self.tick,
                    source=source,
                    object_id=upper_id,
                    changes={"level": landing_level},
                    object=updated.model_dump(mode="json"),
                )
            )

        # Cascade topple to anything stacked on top of upper_id
        for child_id in sorted(self.objects):
            child = self.objects.get(child_id)
            if child is not None and child.level == f"on_top_of:{upper_id}":
                self._topple_object(
                    child_id,
                    support=updated,
                    move_dx=move_dx,
                    move_dy=move_dy,
                    source=source,
                    _visited=_visited,
                )

    def _propagate_support_movement(
        self,
        lower_id: str,
        *,
        previous_pos: Position,
        new_pos: Position,
        source: str = "world",
        _visited: set[str] | None = None,
    ) -> bool:
        """Carry or topple objects resting ``on_top_of`` or ``inside`` ``lower_id``."""
        if _visited is None:
            _visited = set()
        if lower_id in _visited or lower_id not in self.objects:
            return False
        _visited.add(lower_id)

        lower = self.objects[lower_id]
        dx = new_pos.x - previous_pos.x
        dy = new_pos.y - previous_pos.y
        step_dist = math.hypot(dx, dy)
        any_toppled = False

        # 1. Objects inside lower_id always move with the container
        self._propagate_container_contents(lower_id, new_pos=lower.position, source=source)

        # 2. Objects stacked on_top_of lower_id are carried if stable, else topple
        for upper_id in sorted(self.objects):
            upper = self.objects.get(upper_id)
            if upper is None or upper.level != f"on_top_of:{lower_id}":
                continue

            if is_stack_stable(upper, lower, self.objects) and step_dist <= max_carry_step(upper, lower):
                prev_upper_pos = upper.position
                carried_pos = Position(x=lower.position.x, y=lower.position.y)
                if prev_upper_pos != carried_pos:
                    updated_upper = upper.model_copy(update={"position": carried_pos})
                    self.objects[upper_id] = updated_upper
                    self.events.emit(
                        ObjectMovedEvent(
                            tick=self.tick,
                            source=source,
                            object_id=upper_id,
                            previous_position=prev_upper_pos,
                            position=carried_pos,
                        )
                    )
                if self._propagate_support_movement(
                    upper_id,
                    previous_pos=prev_upper_pos,
                    new_pos=carried_pos,
                    source=source,
                    _visited=_visited,
                ):
                    any_toppled = True
            else:
                self._topple_object(
                    upper_id,
                    support=lower,
                    move_dx=dx,
                    move_dy=dy,
                    source=source,
                    _visited=_visited,
                )
                any_toppled = True

        if self.body.level == f"on_top_of:{lower_id}":
            self.body.position = lower.position.model_copy()

        return any_toppled

    def _active_layer_ids(self, level: str) -> list[str]:
        """Return sorted object IDs participating in 2D collisions on ``level``."""
        held = self.body.holding
        return [
            obj_id
            for obj_id in sorted(self.objects)
            if obj_id != held and self.objects[obj_id].level == level
        ]

    def _resolve_layer_collisions(
        self,
        layer_ids: list[str],
        *,
        apply_velocity_response: bool = True,
        prevent_ledge_entry: set[str] | None = None,
    ) -> None:
        """Apply circle-circle velocity collision response and non-overlap projection for one level."""
        n = len(layer_ids)
        if n < 2:
            return

        if apply_velocity_response:
            for i in range(n):
                id_a = layer_ids[i]
                for j in range(i + 1, n):
                    id_b = layer_ids[j]
                    a = self.objects[id_a]
                    b = self.objects[id_b]
                    if a.fixed and b.fixed:
                        continue
                    dx = b.position.x - a.position.x
                    dy = b.position.y - a.position.y
                    dist = math.hypot(dx, dy)
                    min_dist = a.radius + b.radius
                    if dist >= min_dist:
                        continue

                    if dist > 1e-9:
                        nx, ny = dx / dist, dy / dist
                    else:
                        num_a = int(id_a.removeprefix("object_"))
                        num_b = int(id_b.removeprefix("object_"))
                        angle = ((num_a * 31 + num_b * 17) % 360) * (math.pi / 180.0)
                        nx, ny = math.cos(angle), math.sin(angle)

                    rvx = a.velocity.x - b.velocity.x
                    rvy = a.velocity.y - b.velocity.y
                    vel_along_normal = rvx * nx + rvy * ny
                    if vel_along_normal <= 0.0:
                        continue

                    if a.fixed:
                        imp = (1.0 + COLLISION_RESTITUTION) * vel_along_normal
                        new_vb = Velocity(
                            x=b.velocity.x + imp * nx,
                            y=b.velocity.y + imp * ny,
                        )
                        self.objects[id_b] = b.model_copy(update={"velocity": new_vb})
                    elif b.fixed:
                        imp = (1.0 + COLLISION_RESTITUTION) * vel_along_normal
                        new_va = Velocity(
                            x=a.velocity.x - imp * nx,
                            y=a.velocity.y - imp * ny,
                        )
                        self.objects[id_a] = a.model_copy(update={"velocity": new_va})
                    else:
                        ma = max(0.05, a.weight)
                        mb = max(0.05, b.weight)
                        inv_ma = 1.0 / ma
                        inv_mb = 1.0 / mb
                        j_mag = (1.0 + COLLISION_RESTITUTION) * vel_along_normal / (inv_ma + inv_mb)
                        new_va = Velocity(
                            x=a.velocity.x - (j_mag * inv_ma) * nx,
                            y=a.velocity.y - (j_mag * inv_ma) * ny,
                        )
                        new_vb = Velocity(
                            x=b.velocity.x + (j_mag * inv_mb) * nx,
                            y=b.velocity.y + (j_mag * inv_mb) * ny,
                        )
                        self.objects[id_a] = a.model_copy(update={"velocity": new_va})
                        self.objects[id_b] = b.model_copy(update={"velocity": new_vb})

        # Iterative positional projection
        for _ in range(COLLISION_ITERATIONS):
            any_overlap = False
            for i in range(n):
                id_a = layer_ids[i]
                for j in range(i + 1, n):
                    id_b = layer_ids[j]
                    a = self.objects[id_a]
                    b = self.objects[id_b]
                    if a.fixed and b.fixed:
                        continue
                    dx = b.position.x - a.position.x
                    dy = b.position.y - a.position.y
                    dist = math.hypot(dx, dy)
                    min_dist = a.radius + b.radius
                    if dist >= min_dist - 1e-7:
                        continue

                    any_overlap = True
                    if dist > 1e-9:
                        nx, ny = dx / dist, dy / dist
                    else:
                        num_a = int(id_a.removeprefix("object_"))
                        num_b = int(id_b.removeprefix("object_"))
                        angle = ((num_a * 31 + num_b * 17) % 360) * (math.pi / 180.0)
                        nx, ny = math.cos(angle), math.sin(angle)

                    overlap = (min_dist + SEPARATION_SLOP) - dist
                    if a.fixed:
                        cand_b = Position(
                            x=b.position.x + overlap * nx,
                            y=b.position.y + overlap * ny,
                        )
                        if prevent_ledge_entry and id_b in prevent_ledge_entry and self.terrain.is_ledge(cand_b):
                            cand_b = self._clamp_out_of_ledge(b.position, cand_b)
                        self.objects[id_b] = b.model_copy(update={"position": cand_b})
                    elif b.fixed:
                        cand_a = Position(
                            x=a.position.x - overlap * nx,
                            y=a.position.y - overlap * ny,
                        )
                        if prevent_ledge_entry and id_a in prevent_ledge_entry and self.terrain.is_ledge(cand_a):
                            cand_a = self._clamp_out_of_ledge(a.position, cand_a)
                        self.objects[id_a] = a.model_copy(update={"position": cand_a})
                    else:
                        ma = max(0.05, a.weight)
                        mb = max(0.05, b.weight)
                        wa = mb / (ma + mb)
                        wb = ma / (ma + mb)
                        cand_a = Position(
                            x=a.position.x - overlap * wa * nx,
                            y=a.position.y - overlap * wa * ny,
                        )
                        cand_b = Position(
                            x=b.position.x + overlap * wb * nx,
                            y=b.position.y + overlap * wb * ny,
                        )
                        if prevent_ledge_entry and id_a in prevent_ledge_entry and self.terrain.is_ledge(cand_a):
                            cand_a = self._clamp_out_of_ledge(a.position, cand_a)
                        if prevent_ledge_entry and id_b in prevent_ledge_entry and self.terrain.is_ledge(cand_b):
                            cand_b = self._clamp_out_of_ledge(b.position, cand_b)
                        self.objects[id_a] = a.model_copy(update={"position": cand_a})
                        self.objects[id_b] = b.model_copy(update={"position": cand_b})
            if not any_overlap:
                break

        # Fallback de-wedging pass if any movable object is still squeezed into an overlap
        for i in range(n):
            id_a = layer_ids[i]
            a = self.objects[id_a]
            if a.fixed:
                continue
            has_overlap = False
            for j in range(n):
                if i == j:
                    continue
                b = self.objects[layer_ids[j]]
                if math.hypot(a.position.x - b.position.x, a.position.y - b.position.y) < a.radius + b.radius - 1e-6:
                    has_overlap = True
                    break
            if not has_overlap:
                continue

            base_x, base_y = a.position.x, a.position.y
            placed = False
            for ring in range(1, 80):
                dist_step = ring * 0.05
                for step_idx in range(16):
                    angle = step_idx * (2.0 * math.pi / 16.0)
                    cx = base_x + math.cos(angle) * dist_step
                    cy = base_y + math.sin(angle) * dist_step
                    cand = Position(x=cx, y=cy)
                    if prevent_ledge_entry and id_a in prevent_ledge_entry and self.terrain.is_ledge(cand):
                        continue
                    clear = True
                    for j in range(n):
                        if i == j:
                            continue
                        b = self.objects[layer_ids[j]]
                        if math.hypot(cx - b.position.x, cy - b.position.y) < a.radius + b.radius + SEPARATION_SLOP:
                            clear = False
                            break
                    if clear:
                        self.objects[id_a] = a.model_copy(update={"position": cand})
                        placed = True
                        break
                if placed:
                    break

    def _clamp_out_of_ledge(self, safe_pos: Position, candidate_pos: Position) -> Position:
        """Return a position that does not enter a ledge tile starting from ``safe_pos``."""
        only_x = Position(x=candidate_pos.x, y=safe_pos.y)
        only_y = Position(x=safe_pos.x, y=candidate_pos.y)
        if not self.terrain.is_ledge(only_x):
            return only_x
        if not self.terrain.is_ledge(only_y):
            return only_y
        return safe_pos

    def resolve_collisions(self) -> None:
        """Resolve all circle overlaps on ``ground`` and ``raised`` levels."""
        for level in ("ground", "raised"):
            ids = self._active_layer_ids(level)
            prevent = (
                {oid for oid in ids if not self.terrain.is_ledge(self.objects[oid].position)}
                if level == "ground"
                else None
            )
            self._resolve_layer_collisions(
                ids,
                apply_velocity_response=False,
                prevent_ledge_entry=prevent,
            )
            for oid in ids:
                if oid in self.objects:
                    self._propagate_container_contents(
                        oid,
                        new_pos=self.objects[oid].position,
                        source="world",
                    )

    def _step_physics(self) -> None:
        """Execute one deterministic tick of 2.5D physics."""
        if not self.objects:
            return

        start_positions: dict[str, Position] = {
            obj_id: obj.position for obj_id, obj in self.objects.items()
        }

        # 1. Check stacked objects (missing support, static instability, or impulse on upper object)
        for obj_id in sorted(self.objects):
            obj = self.objects.get(obj_id)
            if obj is None or not obj.level.startswith("on_top_of:"):
                continue
            support_id = obj.level.removeprefix("on_top_of:")
            support = self.objects.get(support_id)
            if support is None:
                landing_level = "raised" if self.terrain.is_ledge(obj.position) else "ground"
                updated = obj.model_copy(update={"level": landing_level})
                self.objects[obj_id] = updated
                self.events.emit(
                    ObjectChangedEvent(
                        tick=self.tick,
                        source="world",
                        object_id=obj_id,
                        changes={"level": landing_level},
                        object=updated.model_dump(mode="json"),
                    )
                )
            elif not is_stack_stable(obj, support, self.objects) or obj.velocity.speed > 0.0:
                self._topple_object(
                    obj_id,
                    support=support,
                    move_dx=-obj.velocity.x,
                    move_dy=-obj.velocity.y,
                    source="world",
                )

        # 2. Apply terrain water effects (wetness, floating, downstream drift)
        held = self.body.holding
        for obj_id in sorted(self.objects):
            if obj_id == held:
                continue
            obj = self.objects[obj_id]
            if obj.level != "ground":
                continue
            tile = self.terrain.get_tile(obj.position)
            if tile.type == "water":
                floats = not obj.fixed and obj.weight <= WATER_FLOAT_MAX_WEIGHT
                state_changed = False
                new_state = dict(obj.state)
                if new_state.get("wet") is not True:
                    new_state["wet"] = True
                    state_changed = True
                if new_state.get("floating") != floats:
                    new_state["floating"] = floats
                    state_changed = True

                new_vel = obj.velocity
                if floats:
                    drift_scale = WATER_DRIFT_STRENGTH / (1.0 + 0.5 * obj.weight)
                    new_vel = Velocity(
                        x=obj.velocity.x + tile.flow.x * drift_scale,
                        y=obj.velocity.y + tile.flow.y * drift_scale,
                    )
                elif not obj.fixed and obj.velocity.speed > 0.0:
                    new_vel = Velocity(
                        x=obj.velocity.x * WATER_SINK_DRAG,
                        y=obj.velocity.y * WATER_SINK_DRAG,
                    )

                if state_changed:
                    updated = obj.model_copy(update={"state": new_state, "velocity": new_vel})
                    self.objects[obj_id] = updated
                    self.events.emit(
                        ObjectChangedEvent(
                            tick=self.tick,
                            source="world",
                            object_id=obj_id,
                            changes={"state": new_state},
                            object=updated.model_dump(mode="json"),
                        )
                    )
                elif new_vel != obj.velocity:
                    self.objects[obj_id] = obj.model_copy(update={"velocity": new_vel})

        # 3. Substepped plane movement, ledge boundaries, and circle collisions
        ground_ids = self._active_layer_ids("ground")
        raised_ids = self._active_layer_ids("raised")
        active_ids = ground_ids + raised_ids
        non_ledge_ground_ids = {
            oid for oid in ground_ids if not self.terrain.is_ledge(self.objects[oid].position)
        }

        max_speed = max(
            (self.objects[oid].velocity.speed for oid in active_ids if not self.objects[oid].fixed),
            default=0.0,
        )

        if max_speed > 0.0:
            num_substeps = min(MAX_SUBSTEPS, max(1, math.ceil(max_speed / MAX_SUBSTEP_DISTANCE)))
            dt_sub = 1.0 / num_substeps
            for _ in range(num_substeps):
                for oid in active_ids:
                    obj = self.objects[oid]
                    if obj.fixed or obj.velocity.speed == 0.0:
                        continue
                    old_pos = obj.position
                    cand_x = old_pos.x + obj.velocity.x * dt_sub
                    cand_y = old_pos.y + obj.velocity.y * dt_sub
                    cand_pos = Position(x=cand_x, y=cand_y)
                    vx, vy = obj.velocity.x, obj.velocity.y

                    # Ground objects outside a ledge cannot move onto a ledge tile
                    if oid in non_ledge_ground_ids and self.terrain.is_ledge(cand_pos):
                        x_only = Position(x=cand_x, y=old_pos.y)
                        y_only = Position(x=old_pos.x, y=cand_y)
                        if self.terrain.is_ledge(x_only):
                            cand_x = old_pos.x
                            vx = -vx * COLLISION_RESTITUTION
                        if self.terrain.is_ledge(y_only):
                            cand_y = old_pos.y
                            vy = -vy * COLLISION_RESTITUTION
                        cand_pos = Position(x=cand_x, y=cand_y)
                        if self.terrain.is_ledge(cand_pos):
                            cand_pos = old_pos
                            vx = -obj.velocity.x * COLLISION_RESTITUTION
                            vy = -obj.velocity.y * COLLISION_RESTITUTION

                    self.objects[oid] = obj.model_copy(
                        update={
                            "position": cand_pos,
                            "velocity": Velocity(x=vx, y=vy),
                        }
                    )

                self._resolve_layer_collisions(
                    self._active_layer_ids("ground"),
                    apply_velocity_response=True,
                    prevent_ledge_entry=non_ledge_ground_ids,
                )
                self._resolve_layer_collisions(
                    self._active_layer_ids("raised"),
                    apply_velocity_response=True,
                )
        else:
            # Even with zero velocity, ensure no objects overlap after a step
            self._resolve_layer_collisions(
                ground_ids,
                apply_velocity_response=False,
                prevent_ledge_entry=non_ledge_ground_ids,
            )
            self._resolve_layer_collisions(
                raised_ids,
                apply_velocity_response=False,
            )

        # 4. Raised objects that moved off a ledge drop to ground level
        for oid in list(raised_ids):
            obj = self.objects.get(oid)
            if obj is not None and obj.level == "raised" and not self.terrain.is_ledge(obj.position):
                updated = obj.model_copy(update={"level": "ground"})
                self.objects[oid] = updated
                self.events.emit(
                    ObjectChangedEvent(
                        tick=self.tick,
                        source="world",
                        object_id=oid,
                        changes={"level": "ground"},
                        object=updated.model_dump(mode="json"),
                    )
                )

        # 5. Apply per-tick friction damping based on shape
        for oid in active_ids:
            obj = self.objects.get(oid)
            if obj is None or obj.fixed or obj.velocity.speed == 0.0:
                continue
            retention = SHAPE_RETENTION.get(obj.shape, 0.60)
            nvx = obj.velocity.x * retention
            nvy = obj.velocity.y * retention
            if math.hypot(nvx, nvy) < STOP_SPEED_EPSILON:
                nvx, nvy = 0.0, 0.0
            self.objects[oid] = obj.model_copy(update={"velocity": Velocity(x=nvx, y=nvy)})

        # 6. Propagate carry or topple to objects stacked on top of or inside moved supports
        any_toppled = False
        for oid in active_ids:
            obj = self.objects.get(oid)
            prev_pos = start_positions.get(oid)
            if obj is None or prev_pos is None or obj.position == prev_pos:
                continue
            if self._propagate_support_movement(
                oid,
                previous_pos=prev_pos,
                new_pos=obj.position,
                source="world",
            ):
                any_toppled = True

        if any_toppled:
            self.resolve_collisions()

        # 7. Emit ObjectMovedEvent for any active object whose position changed this tick
        for oid in sorted(self.objects):
            prev_pos = start_positions.get(oid)
            obj = self.objects[oid]
            if (
                prev_pos is not None
                and obj.position != prev_pos
                and not obj.level.startswith("on_top_of:")
                and not obj.level.startswith("inside:")
            ):
                self.events.emit(
                    ObjectMovedEvent(
                        tick=self.tick,
                        source="world",
                        object_id=oid,
                        previous_position=prev_pos,
                        position=obj.position,
                    )
                )

    def step(
        self,
        *,
        verb: str | None = None,
        force: float | None = None,
        speed: float | None = None,
        resting: bool | None = None,
        cold: float | bool | None = None,
        wet: float | bool | None = None,
        storm: float | bool | None = None,
        sheltered: float | bool | None = None,
        dry: bool | None = None,
        warm: float | bool | None = None,
        temperature: float | None = None,
        weather: Any = None,
        novelty: float | int | bool = 0.0,
        surprise: float | int | bool = 0.0,
    ) -> int:
        """Advance the simulation by exactly one tick."""
        _set_active_world(self)
        self.tick += 1
        # Advance the simulation RNG once per tick so tick progression is
        # bound to the seeded generator and verified across save/restore.
        self.rng.random()
        self.daynight.step(1)
        self._step_physics()
        self._body_state_system.step(
            verb=verb,
            force=force,
            speed=speed,
            resting=resting,
            cold=cold,
            wet=wet,
            storm=storm,
            sheltered=sheltered,
            dry=dry,
            warm=warm,
            temperature=temperature,
            weather=weather,
            novelty=novelty,
            surprise=surprise,
        )
        self.events.emit(TickEvent(tick=self.tick, source="world"))
        return self.tick

    def snapshot(self) -> str:
        """Serialize the full world state (including RNG, terrain, daynight, and factory state) to JSON."""
        version, internal_state, gauss_next = self.rng.getstate()
        payload = {
            "tick": self.tick,
            "body": self.body.model_dump(mode="json"),
            "objects": {
                obj_id: self.objects[obj_id].model_dump(mode="json")
                for obj_id in sorted(self.objects)
            },
            "terrain": self.terrain.to_dict(),
            "daynight": self.daynight.to_dict(),
            "body_state": self._body_state_system.to_dict(),
            "weather": self.weather,
            "factory": {
                "next": self._factory._next,
                "used": sorted(self._factory._used),
            },
            "rng_state": [version, list(internal_state), gauss_next],
        }
        return json.dumps(payload, sort_keys=True)

    def _restore_from_instance(self, snapshot: str | bytes | Mapping[str, Any]) -> World:
        """Restore world state in-place from a ``snapshot()`` payload."""
        if isinstance(snapshot, str | bytes):
            raw = json.loads(snapshot)
        elif isinstance(snapshot, Mapping):
            raw = dict(snapshot)
        else:
            raise TypeError("snapshot must be a JSON string, bytes, or mapping")

        if not isinstance(raw, Mapping):
            raise ValueError("snapshot payload must be a JSON object")  # noqa: TRY004

        tick = raw.get("tick")
        if not isinstance(tick, int) or isinstance(tick, bool) or tick < 0:
            raise ValueError("snapshot tick must be a non-negative integer")

        body = AgentBody.model_validate(raw.get("body", {}))

        raw_objects = raw.get("objects")
        if not isinstance(raw_objects, Mapping):
            raise ValueError("snapshot objects must be a mapping")  # noqa: TRY004

        restored_objects: dict[str, WorldObject] = {}
        for obj_id, obj_data in raw_objects.items():
            obj = validate_world_object(obj_data)
            if obj.id != obj_id:
                raise ValueError(f"object key {obj_id!r} does not match object.id {obj.id!r}")
            restored_objects[obj.id] = obj

        raw_terrain = raw.get("terrain")
        if isinstance(raw_terrain, Mapping):
            terrain = TerrainGrid.from_dict(raw_terrain)
        else:
            terrain = TerrainGrid()

        raw_daynight = raw.get("daynight")
        if isinstance(raw_daynight, Mapping):
            daynight = DayNightCycle.from_dict(raw_daynight)
        else:
            daynight = DayNightCycle()

        factory = ObjectFactory()
        raw_factory = raw.get("factory")
        if isinstance(raw_factory, Mapping):
            next_id = raw_factory.get("next", 1)
            used_ids = raw_factory.get("used", [])
            if not isinstance(next_id, int) or isinstance(next_id, bool) or next_id < 1:
                raise ValueError("snapshot factory.next must be a positive integer")
            factory._next = next_id
            factory._used = {str(item) for item in used_ids}
            for obj_id in restored_objects:
                factory._claim(obj_id)
        else:
            for obj_id in restored_objects:
                factory._claim(obj_id)

        rng_state = raw.get("rng_state")
        if not isinstance(rng_state, list | tuple) or len(rng_state) != 3:
            raise ValueError("snapshot rng_state must be a 3-element sequence")
        version, internal_state, gauss_next = rng_state
        if not isinstance(internal_state, list | tuple):
            raise ValueError("snapshot rng_state internal state must be a sequence")  # noqa: TRY004
        self.rng.setstate((int(version), tuple(int(x) for x in internal_state), gauss_next))

        self.tick = tick
        self.body = body
        self.objects = restored_objects
        self.terrain = terrain
        self.daynight = daynight
        raw_body_state = raw.get("body_state")
        if isinstance(raw_body_state, Mapping):
            self._body_state_system = BodyStateSystem.from_dict(raw_body_state, world=self)
        else:
            self._body_state_system = BodyStateSystem(self, config=self.body_config)
        if "weather" in raw:
            self.weather = raw["weather"]
        self._factory = factory
        _set_active_world(self)
        return self

    @classmethod
    def _restore_from_cls(cls, snapshot: str | bytes | Mapping[str, Any]) -> World:
        """Create a new ``World`` restored from ``snapshot``."""
        world = cls()
        return world._restore_from_instance(snapshot)

    restore = _RestoreDescriptor()

    @classmethod
    def from_snapshot(cls, snapshot: str | bytes | Mapping[str, Any]) -> World:
        """Create a new ``World`` restored from ``snapshot``."""
        return cls._restore_from_cls(snapshot)

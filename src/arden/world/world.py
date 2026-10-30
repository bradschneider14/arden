"""Headless world container, in-process event bus, creator API, and save/load."""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from arden.world.objects import ObjectFactory, Position, WorldObject

_OBJECT_ID = re.compile(r"^object_[0-9]+$")
_LEVEL = re.compile(r"^(?:ground|raised|on_top_of:object_[0-9]+|inside:object_[0-9]+)$")

EventType = Literal[
    "object_spawned",
    "object_despawned",
    "object_moved",
    "object_changed",
    "tick",
]
EVENT_TYPES: frozenset[str] = frozenset(
    {
        "object_spawned",
        "object_despawned",
        "object_moved",
        "object_changed",
        "tick",
    }
)


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

    @property
    def kind(self) -> EventType:
        """Alias for ``type``."""
        return self.type

    @property
    def event_type(self) -> EventType:
        """Alias for ``type``."""
        return self.type


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


EventHandler = Callable[[WorldEvent], None]


class EventBus:
    """Synchronous in-process pub/sub bus for typed world events."""

    def __init__(self) -> None:
        self._all_subscribers: list[EventHandler] = []
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
        """Register a subscriber for all events or a specific event type.

        Supports both ``subscribe(callback)``, ``subscribe("tick", callback)``,
        and ``subscribe(callback, event_type="tick")``. Returns an unsubscribe
        callable.
        """
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
            if event_type not in EVENT_TYPES:
                allowed = ", ".join(sorted(EVENT_TYPES))
                raise ValueError(f"unknown event type {event_type!r}; expected one of: {allowed}")
            bucket = self._typed_subscribers[event_type]
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

        self.tick: int = 0
        self._factory = ObjectFactory()
        self.objects: dict[str, WorldObject] = {}
        self.events = EventBus()

        if objects is not None:
            items: Iterable[WorldObject | Mapping[str, Any]]
            if isinstance(objects, Mapping):
                items = objects.values()
            else:
                items = objects
            for item in items:
                if isinstance(item, WorldObject):
                    created = self._factory.create(item.model_dump())
                elif isinstance(item, Mapping):
                    created = self._factory.create(item)
                else:
                    raise TypeError("objects must contain WorldObject instances or mappings")
                self.objects[created.id] = created

    @property
    def agent_body(self) -> AgentBody:
        """Alias for ``body``."""
        return self.body

    @property
    def event_bus(self) -> EventBus:
        """Alias for ``events``."""
        return self.events

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
        if isinstance(props, WorldObject):
            data = props.model_dump()
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

    def despawn(self, object_id: str, *, source: str = "creator") -> WorldObject:
        """Remove an object by id, emitting ``object_despawned``."""
        if object_id not in self.objects:
            raise KeyError(f"unknown object id: {object_id}")
        obj = self.objects.pop(object_id)
        if self.body.holding == object_id:
            self.body.holding = None
        self.events.emit(
            ObjectDespawnedEvent(
                tick=self.tick,
                source=source,
                object_id=obj.id,
                position=obj.position,
                object=obj.model_dump(mode="json"),
            )
        )
        return obj

    def move(
        self,
        object_id: str,
        pos: Position | Mapping[str, Any] | tuple[float, float] | list[float],
        *,
        source: str = "creator",
    ) -> WorldObject:
        """Move an existing object to ``pos``, emitting ``object_moved``."""
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
        return updated

    def update_object(
        self,
        object_id: str,
        changes: Mapping[str, Any],
        *,
        source: str = "world",
    ) -> WorldObject:
        """Update properties or state on an object, emitting ``object_changed``."""
        if object_id not in self.objects:
            raise KeyError(f"unknown object id: {object_id}")
        if not isinstance(changes, Mapping):
            raise TypeError("changes must be a mapping")
        if "id" in changes and changes["id"] != object_id:
            raise ValueError("cannot change an object's id")

        current = self.objects[object_id]
        merged = {**current.model_dump(), **dict(changes)}
        updated = WorldObject.model_validate(merged)
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

    def step(self) -> int:
        """Advance the simulation by exactly one tick."""
        self.tick += 1
        # Advance the simulation RNG once per tick so tick progression is
        # bound to the seeded generator and verified across save/restore.
        self.rng.random()
        self.events.emit(TickEvent(tick=self.tick, source="world"))
        return self.tick

    def snapshot(self) -> str:
        """Serialize the full world state (including RNG and factory state) to JSON."""
        version, internal_state, gauss_next = self.rng.getstate()
        payload = {
            "tick": self.tick,
            "body": self.body.model_dump(mode="json"),
            "objects": {
                obj_id: self.objects[obj_id].model_dump(mode="json")
                for obj_id in sorted(self.objects)
            },
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
            obj = WorldObject.model_validate(obj_data)
            if obj.id != obj_id:
                raise ValueError(f"object key {obj_id!r} does not match object.id {obj.id!r}")
            restored_objects[obj.id] = obj

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
        self._factory = factory
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

"""Body-level verbs, building verbs, parameter validation, body modifiers, and opaque actions.

Objects decide what each verb does via their hidden ``BehaviorRule`` entries
plus deterministic 2.5D physics; the verb code stays generic.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from arden.world.objects import (
     _COMPOSITE_EXTRA_KEYS,
    BehaviorRule,
    Composite,
    Position,
    Velocity,
    WorldObject,
    validate_world_object,
)
from arden.world.physics import (
    LEDGE_HEIGHT,
    _get_active_world,
    _object_elevation,
    _set_active_world,
    apply_impulse,
    can_bind,
    compute_strike_force,
    extract_xy,
    is_occluded,
    is_stack_stable,
    normalize_direction,
    stack_stability_score,
    strike_impact_multiplier,
)
from arden.world.world import (
    ObjectChangedEvent,
    ObjectDespawnedEvent,
    ObjectMovedEvent,
    ObjectSpawnedEvent,
    SpokeEvent,
    World,
    WorldEvent,
)

_OBJECT_ID = re.compile(r"^object_[0-9]+$")
_OPAQUE_ACTION_ID = re.compile(r"^action_[0-9]+$")

OUT_OF_REACH: Literal["out_of_reach"] = "out_of_reach"
TOO_HEAVY: Literal["too_heavy"] = "too_heavy"
HANDS_FULL: Literal["hands_full"] = "hands_full"
NOTHING_HAPPENED: Literal["nothing_happened"] = "nothing_happened"
INVALID_PARAMS: Literal["invalid_params"] = "invalid_params"

REASON_CODES: tuple[str, ...] = (
    OUT_OF_REACH,
    TOO_HEAVY,
    HANDS_FULL,
    NOTHING_HAPPENED,
)

ReasonCode = Literal[
    "out_of_reach",
    "too_heavy",
    "hands_full",
    "nothing_happened",
    "invalid_params",
]

DEFAULT_REACH: float = 1.5
MAX_GRAB_WEIGHT: float = 5.0
MAX_PUSH_WEIGHT: float = 50.0
DEFAULT_FRAGILE_BREAK_THRESHOLD: float = 0.5
BASE_MOVE_STEP: float = 1.0


class BodyModifiers(BaseModel):
    """Physical side-effect modifiers applied by every verb.

    Defaults are neutral; M7 populates these from energy, cold, and wet.
    """

    model_config = ConfigDict(extra="forbid")

    speed: float = Field(default=1.0, ge=0.0)
    force: float = Field(default=1.0, ge=0.0)
    dexterity: float = Field(default=1.0, ge=0.0)
    grab_fail_chance: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("speed", "force", "dexterity", "grab_fail_chance")
    @classmethod
    def _values_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("modifier values must be finite")
        return value


class ActionResult(BaseModel):
    """Outcome of executing a body verb or opaque action."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    reason: str | None = None
    events: list[WorldEvent] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_fields(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        d = dict(values)
        if "ok" not in d and "status" in d:
            status = d.pop("status")
            if isinstance(status, str):
                d["ok"] = status.lower() in ("ok", "true", "success")
            else:
                d["ok"] = bool(status)
        elif "status" in d:
            d.pop("status")
        if "reason" not in d and "reason_code" in d:
            d["reason"] = d.pop("reason_code")
        elif "reason_code" in d:
            d.pop("reason_code")
        return d

    @property
    def failed(self) -> bool:
        """True when the action did not succeed."""
        return not self.ok

    @property
    def success(self) -> bool:
        """Alias for ``ok``."""
        return self.ok

    @property
    def status(self) -> Literal["ok", "failed"]:
        """String status ``'ok'`` or ``'failed'``."""
        return "ok" if self.ok else "failed"

    @property
    def reason_code(self) -> str | None:
        """Alias for ``reason``."""
        return self.reason

    @property
    def composite(self) -> Composite | None:
        """Return the ``Composite`` object produced by this action, if any."""
        comp = self.data.get("composite")
        if isinstance(comp, Composite):
            return comp
        return None

    @property
    def object(self) -> WorldObject | None:
        """Return the primary ``WorldObject`` produced or affected by this action, if any."""
        for key in ("composite", "object"):
            val = self.data.get(key)
            if isinstance(val, WorldObject):
                return val
        return None

    @property
    def id(self) -> str | None:
        """Return the primary object/composite ID from ``data``, if any."""
        for key in ("composite_id", "id", "object_id", "stacked", "inserted", "holding"):
            val = self.data.get(key)
            if isinstance(val, str):
                return val
        obj = self.object
        return obj.id if obj is not None else None

    @property
    def parts(self) -> list[Any]:
        """Return separated or constituent parts from ``data``, if any."""
        if "part_objects" in self.data and isinstance(self.data["part_objects"], list):
            return self.data["part_objects"]
        if "parts" in self.data and isinstance(self.data["parts"], list):
            return self.data["parts"]
        comp = self.composite
        return list(comp.parts) if comp is not None else []

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        data_dict = self.__dict__.get("data", {})
        if isinstance(data_dict, Mapping):
            for key in ("composite", "object"):
                obj = data_dict.get(key)
                if isinstance(obj, WorldObject) and hasattr(obj, name):
                    return getattr(obj, name)
            if name in data_dict:
                return data_dict[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __bool__(self) -> bool:
        return self.ok


OpaqueHandler = Callable[..., Any]

# Global fallback registry for opaque actions registered without an explicit World.
_FALLBACK_OPAQUE_ACTIONS: dict[str, dict[str, OpaqueHandler]] = {}
_FALLBACK_NEXT_ACTION_INDEX: int = 1


def _world_opaque_registry(world: World) -> dict[str, dict[str, OpaqueHandler]]:
    reg = getattr(world, "_opaque_actions", None)
    if reg is None:
        reg = {}
        setattr(world, "_opaque_actions", reg)
    return reg


def _next_opaque_action_id(world: World | None) -> str:
    global _FALLBACK_NEXT_ACTION_INDEX
    used: set[str] = set()
    if world is not None:
        reg = _world_opaque_registry(world)
        for actions in reg.values():
            used.update(actions.keys())
        for obj in world.objects.values():
            for rule in obj.behavior:
                if _OPAQUE_ACTION_ID.fullmatch(rule.verb):
                    used.add(rule.verb)
        idx = int(getattr(world, "_next_opaque_action_idx", 1))
        while f"action_{idx}" in used:
            idx += 1
        setattr(world, "_next_opaque_action_idx", idx + 1)
        return f"action_{idx}"

    for actions in _FALLBACK_OPAQUE_ACTIONS.values():
        used.update(actions.keys())
    while f"action_{_FALLBACK_NEXT_ACTION_INDEX}" in used:
        _FALLBACK_NEXT_ACTION_INDEX += 1
    action_id = f"action_{_FALLBACK_NEXT_ACTION_INDEX}"
    _FALLBACK_NEXT_ACTION_INDEX += 1
    return action_id


def _resolve_world(explicit_world: World | None = None) -> World:
    if explicit_world is not None:
        _set_active_world(explicit_world)
        return explicit_world
    world = _get_active_world()
    if world is None:
        raise RuntimeError("no active World found; pass world explicitly or instantiate a World first")
    return world


def _resolve_modifiers(
    world: World,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> BodyModifiers:
    if isinstance(modifiers, BodyModifiers):
        return modifiers
    if isinstance(modifiers, Mapping):
        return BodyModifiers.model_validate(dict(modifiers))
    world_mods = getattr(world, "body_modifiers", None)
    if isinstance(world_mods, BodyModifiers):
        return world_mods
    if isinstance(world_mods, Mapping):
        return BodyModifiers.model_validate(dict(world_mods))
    body_mods = getattr(world.body, "modifiers", None)
    if isinstance(body_mods, BodyModifiers):
        return body_mods
    if isinstance(body_mods, Mapping):
        return BodyModifiers.model_validate(dict(body_mods))
    return BodyModifiers()


def _validate_unit_interval(value: Any) -> float | None:
    """Return float in ``[0.0, 1.0]`` or ``None`` if invalid."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    fval = float(value)
    if not math.isfinite(fval) or fval < 0.0 or fval > 1.0:
        return None
    return fval


def _resolve_target_object(
    world: World,
    target: str | WorldObject | ActionResult | None,
) -> WorldObject | None:
    if target is None:
        return None
    if isinstance(target, ActionResult):
        tid = target.id
        return world.objects.get(tid) if tid is not None else target.object
    if isinstance(target, WorldObject):
        return world.objects.get(target.id, target if target.id in world.objects else None)
    if isinstance(target, str):
        return world.objects.get(target)
    return None


def _agent_elevation(world: World) -> float:
    level = world.body.level
    if level == "raised":
        return LEDGE_HEIGHT
    if level == "ground":
        if world.terrain.is_ledge(world.body.position):
            return LEDGE_HEIGHT
        return 0.0
    if level.startswith("on_top_of:"):
        support_id = level.removeprefix("on_top_of:")
        support = world.objects.get(support_id)
        if support is not None:
            return _object_elevation(support, world.objects, world.terrain) + support.height
    return 0.0


def in_reach(
    world_or_target: World | str | WorldObject | Position | Mapping[str, Any] | tuple[float, float] | list[float],
    target: str | WorldObject | Position | Mapping[str, Any] | tuple[float, float] | list[float] | None = None,
    *,
    reach: float = DEFAULT_REACH,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> bool:
    """Return True if ``target`` (an object or location) is within reach of the agent body."""
    if isinstance(world_or_target, World):
        world = world_or_target
        actual_target = target
    else:
        world = _resolve_world()
        actual_target = world_or_target

    _ = _resolve_modifiers(world, modifiers)
    if actual_target is None:
        return False

    body_pos = world.body.position
    agent_elev = _agent_elevation(world)

    # Check if actual_target is an object or object id
    obj: WorldObject | None = None
    if isinstance(actual_target, ActionResult):
        obj = _resolve_target_object(world, actual_target)
    elif isinstance(actual_target, WorldObject):
        obj = world.objects.get(actual_target.id, actual_target)
    elif isinstance(actual_target, str):
        obj = world.objects.get(actual_target)
        if obj is None:
            return False

    if obj is not None:
        if world.body.holding == obj.id:
            return True
        dist = math.hypot(obj.position.x - body_pos.x, obj.position.y - body_pos.y)
        if dist > reach + obj.radius + 1e-6:
            return False
        obj_elev = _object_elevation(obj, world.objects, world.terrain)
        # An object on a raised ledge is out of reach from the ground unless the agent can reach that level
        if obj.level == "raised" and world.body.level == "ground" and agent_elev < LEDGE_HEIGHT:
            if not world.can_reach(obj.position):
                return False
        if abs(obj_elev - agent_elev) > reach + 1e-6:
            return False
        return True

    # Location reach check
    try:
        tx, ty = extract_xy(actual_target)
    except (TypeError, ValueError, KeyError):
        return False
    if not (math.isfinite(tx) and math.isfinite(ty)):
        return False
    dist = math.hypot(tx - body_pos.x, ty - body_pos.y)
    if dist > reach + 1e-6:
        return False
    if not world.can_reach((tx, ty)):
        return False
    return True


def _rule_extras(rule: BehaviorRule) -> dict[str, Any]:
    return dict(rule.__pydantic_extra__ or {})


def _rule_matches_force(rule: BehaviorRule, effective_force: float) -> bool:
    extras = _rule_extras(rule)
    above = extras.get("above_force")
    if isinstance(above, int | float) and not isinstance(above, bool):
        threshold = float(above)
        if threshold >= 1.0:
            if effective_force < threshold:
                return False
        elif effective_force <= threshold:
            return False

    min_f = extras.get("min_force", extras.get("threshold", extras.get("force_threshold")))
    if isinstance(min_f, int | float) and not isinstance(min_f, bool):
        if effective_force < float(min_f):
            return False

    below = extras.get("below_force", extras.get("max_force"))
    if isinstance(below, int | float) and not isinstance(below, bool):
        if effective_force > float(below):
            return False

    return True


def _apply_rule_state_updates(
    world: World,
    obj: WorldObject,
    rule: BehaviorRule,
) -> bool:
    """Apply any state or property updates specified on ``rule``. Returns True if updated."""
    extras = _rule_extras(rule)
    changes: dict[str, Any] = {}

    if isinstance(extras.get("changes"), Mapping):
        changes.update(dict(extras["changes"]))
    if isinstance(extras.get("update"), Mapping):
        changes.update(dict(extras["update"]))

    state_patch: dict[str, Any] = {}
    for key in ("state", "set_state", "state_changes"):
        val = extras.get(key)
        if isinstance(val, Mapping):
            state_patch.update(dict(val))

    if state_patch:
        current = world.objects.get(obj.id, obj)
        base_state = dict(changes.get("state", current.state))
        base_state.update(state_patch)
        changes["state"] = base_state

    if not changes or obj.id not in world.objects:
        return False

    world.update_object(obj.id, changes, source="agent")
    return True


def _detach_from_container(world: World, child_obj: WorldObject) -> None:
    """If ``child_obj`` is currently ``inside:<container_id>``, update the container's contents and weight."""
    if not child_obj.level.startswith("inside:"):
        return
    container_id = child_obj.level.removeprefix("inside:")
    container = world.objects.get(container_id)
    if container is None:
        return

    remaining = [
        o for o in world.objects.values()
        if o.id != child_obj.id and o.level == f"inside:{container_id}"
    ]
    remaining_ids = [o.id for o in remaining]

    if isinstance(container, Composite) and container.base_object is not None:
        base_obj = WorldObject.model_validate(container.base_object)
        if not remaining:
            restored_state = dict(base_obj.state)
            restored_state.pop("contents", None)
            restored = base_obj.model_copy(
                update={
                    "position": container.position.model_copy(),
                    "level": container.level,
                    "velocity": container.velocity.model_copy(),
                    "state": restored_state,
                }
            )
            world.objects[container_id] = restored
            world.events.emit(
                ObjectChangedEvent(
                    tick=world.tick,
                    source="agent",
                    object_id=container_id,
                    changes={"weight": restored.weight, "state": restored_state},
                    object=restored.model_dump(mode="json"),
                )
            )
            return

        new_weight = round(base_obj.weight + sum(o.weight for o in remaining), 6)
        new_state = {**container.state, "contents": remaining_ids, "made_by_agent": True}
        updated_comp = container.model_copy(
            update={
                "weight": new_weight,
                "parts": [base_obj, *remaining],
                "state": new_state,
            }
        )
        world.objects[container_id] = updated_comp
        world.events.emit(
            ObjectChangedEvent(
                tick=world.tick,
                source="agent",
                object_id=container_id,
                changes={"weight": new_weight, "state": new_state},
                object=updated_comp.model_dump(mode="json"),
            )
        )
    else:
        new_weight = max(0.0, round(container.weight - child_obj.weight, 6))
        new_state = dict(container.state)
        if remaining_ids:
            new_state["contents"] = remaining_ids
        else:
            new_state.pop("contents", None)
        world.update_object(
            container_id,
            {"weight": new_weight, "state": new_state},
            source="agent",
        )


def _separate_bound_composite(
    world: World,
    comp: Composite,
    *,
    source: str = "agent",
) -> list[WorldObject]:
    """Despawn bound ``comp`` and restore its original constituent parts to ``world.objects``."""
    original_parts = comp.original_parts
    base_pos = comp.position.model_copy()
    base_level = "raised" if world.terrain.is_ledge(base_pos) else "ground"

    if comp.id in world.objects:
        world.despawn(comp.id, source=source)

    restored: list[WorldObject] = []
    count = max(1, len(original_parts))
    offset = max(0.10, comp.radius * 0.6)
    for idx, part in enumerate(original_parts):
        angle = idx * (2.0 * math.pi / count)
        px = base_pos.x + (math.cos(angle) * offset if count > 1 else 0.0)
        py = base_pos.y + (math.sin(angle) * offset if count > 1 else 0.0)
        part_pos = Position(x=px, y=py)
        part_level = "raised" if world.terrain.is_ledge(part_pos) else base_level
        spawned = world._restore_part_object(part, part_pos, part_level, source=source)
        restored.append(spawned)

    if restored:
        world.resolve_collisions()
    return [world.objects[p.id] for p in restored if p.id in world.objects]


def _break_object(
    world: World,
    target_obj: WorldObject,
    rule: BehaviorRule | None = None,
) -> list[WorldObject]:
    """Despawn ``target_obj`` and optionally spawn smaller pieces."""
    extras = _rule_extras(rule) if rule is not None else {}
    response = str(extras.get("response", "break")).lower()

    # Despawn the broken object first
    if target_obj.id in world.objects:
        world.despawn(target_obj.id, source="agent")

    # Determine whether to spawn smaller pieces or only remove
    if response in ("remove", "destroy", "consume"):
        return []

    raw_pieces = extras.get("pieces", target_obj.state.get("pieces"))
    if raw_pieces == 0 or raw_pieces is False:
        return []

    base_level = "raised" if world.terrain.is_ledge(target_obj.position) else "ground"
    spawned_pieces: list[WorldObject] = []

    if isinstance(raw_pieces, list):
        count = len(raw_pieces)
        for idx, piece_spec in enumerate(raw_pieces):
            if not isinstance(piece_spec, Mapping):
                continue
            angle = idx * (2.0 * math.pi / max(1, count))
            offset = max(0.08, target_obj.radius * 0.6)
            px = target_obj.position.x + math.cos(angle) * offset
            py = target_obj.position.y + math.sin(angle) * offset
            props = {
                "shape": target_obj.shape,
                "height": max(0.01, target_obj.height * 0.5),
                "volume": max(0.0001, target_obj.volume / max(2, count)),
                "weight": max(0.01, target_obj.weight / max(2, count)),
                "rigidity": target_obj.rigidity,
                "binding": target_obj.binding,
                "hollow": False,
                "fragile": False,
                "color": target_obj.color,
                "fixed": False,
                "level": base_level,
                "state": {**target_obj.state, "piece_of": target_obj.id},
                **dict(piece_spec),
            }
            piece = world.spawn(props, Position(x=px, y=py), source="agent")
            spawned_pieces.append(piece)
        if spawned_pieces:
            world.resolve_collisions()
        return spawned_pieces

    num_pieces = int(raw_pieces) if isinstance(raw_pieces, int) and raw_pieces > 0 else 2
    if target_obj.volume <= 0.0 and target_obj.weight <= 0.0:
        return []

    piece_volume = target_obj.volume / num_pieces
    piece_weight = target_obj.weight / num_pieces
    piece_height = target_obj.height * 0.5
    offset = max(0.08, target_obj.radius * 0.6)

    for idx in range(num_pieces):
        angle = idx * (2.0 * math.pi / num_pieces)
        px = target_obj.position.x + math.cos(angle) * offset
        py = target_obj.position.y + math.sin(angle) * offset
        piece_state = dict(target_obj.state)
        piece_state.pop("pieces", None)
        piece_state["piece_of"] = target_obj.id
        piece_props: dict[str, Any] = {
            "shape": target_obj.shape,
            "height": piece_height,
            "volume": piece_volume,
            "weight": piece_weight,
            "rigidity": target_obj.rigidity,
            "binding": target_obj.binding,
            "hollow": False,
            "fragile": False,
            "color": target_obj.color,
            "fixed": False,
            "level": base_level,
            "state": piece_state,
        }
        piece = world.spawn(piece_props, Position(x=px, y=py), source="agent")
        spawned_pieces.append(piece)

    if spawned_pieces:
        world.resolve_collisions()
    return spawned_pieces


def _direction_from_agent_to_object(world: World, obj: WorldObject) -> tuple[float, float]:
    dx = obj.position.x - world.body.position.x
    dy = obj.position.y - world.body.position.y
    dist = math.hypot(dx, dy)
    if dist <= 1e-9:
        return (1.0, 0.0)
    return (dx / dist, dy / dist)


def move(
    world_or_direction: Any,
    direction_or_speed: Any = 1.0,
    speed: float | None = None,
    *,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Move the agent body in ``direction`` at ``speed`` (in ``0..1``)."""
    if isinstance(world_or_direction, World):
        resolved_world = _resolve_world(world_or_direction)
        direction = direction_or_speed
        raw_speed = 1.0 if speed is None else speed
    else:
        resolved_world = _resolve_world(world)
        direction = world_or_direction
        raw_speed = direction_or_speed if speed is None else speed

    mods = _resolve_modifiers(resolved_world, modifiers)
    valid_speed = _validate_unit_interval(raw_speed)
    if valid_speed is None:
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    # Allow moving toward an object id if direction is an object_<n> string in world.objects
    if isinstance(direction, str) and direction in resolved_world.objects:
        target_obj = resolved_world.objects[direction]
        ux, uy = _direction_from_agent_to_object(resolved_world, target_obj)
    else:
        try:
            ux, uy = normalize_direction(direction)
        except (TypeError, ValueError):
            return ActionResult(ok=False, reason=INVALID_PARAMS)

    effective_speed = valid_speed * mods.speed * BASE_MOVE_STEP
    if effective_speed <= 0.0 or (ux == 0.0 and uy == 0.0):
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    prev_pos = resolved_world.body.position
    dest = Position(
        x=prev_pos.x + ux * effective_speed,
        y=prev_pos.y + uy * effective_speed,
    )

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        if not resolved_world.move_agent(dest):
            return ActionResult(ok=False, reason=OUT_OF_REACH)

        # Keep held item (and anything inside it) moving with the agent body
        held_id = resolved_world.body.holding
        if held_id is not None and held_id in resolved_world.objects:
            held_obj = resolved_world.objects[held_id]
            if held_obj.position != dest or held_obj.level != resolved_world.body.level:
                old_obj_pos = held_obj.position
                updated_held = held_obj.model_copy(
                    update={
                        "position": dest.model_copy(),
                        "level": resolved_world.body.level,
                    }
                )
                resolved_world.objects[held_id] = updated_held
                if old_obj_pos != dest:
                    resolved_world.events.emit(
                        ObjectMovedEvent(
                            tick=resolved_world.tick,
                            source="agent",
                            object_id=held_id,
                            previous_position=old_obj_pos,
                            position=dest,
                        )
                    )
                    resolved_world._propagate_container_contents(
                        held_id,
                        new_pos=dest,
                        source="agent",
                    )
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={
            "previous_position": {"x": prev_pos.x, "y": prev_pos.y},
            "position": {"x": dest.x, "y": dest.y},
            "level": resolved_world.body.level,
        },
    )


def look(
    world_or_target: World | str | WorldObject,
    target: str | WorldObject | None = None,
    *,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Focus perception on ``target`` within reach and return its perceptible properties."""
    if isinstance(world_or_target, World):
        resolved_world = _resolve_world(world_or_target)
        actual_target = target
    else:
        resolved_world = _resolve_world(world)
        actual_target = world_or_target

    mods = _resolve_modifiers(resolved_world, modifiers)
    target_obj = _resolve_target_object(resolved_world, actual_target)
    if target_obj is None:
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if not in_reach(resolved_world, target_obj, modifiers=mods):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if (
        resolved_world.body.holding != target_obj.id
        and is_occluded(resolved_world.body.position, target_obj, world=resolved_world)
    ):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        for rule in target_obj.behavior:
            if rule.verb == "look":
                _apply_rule_state_updates(resolved_world, target_obj, rule)
        refreshed = resolved_world.objects.get(target_obj.id, target_obj)
        perceptible = refreshed.perceptible()
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data=perceptible,
    )


def push(
    world_or_target: World | str | WorldObject,
    target_or_force: Any = 0.5,
    force: float | None = None,
    *,
    direction: Any = None,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Push ``target`` with ``force`` (in ``0..1``), applying physics impulse and behavior rules."""
    if isinstance(world_or_target, World):
        resolved_world = _resolve_world(world_or_target)
        actual_target = target_or_force
        raw_force = 0.5 if force is None else force
    else:
        resolved_world = _resolve_world(world)
        actual_target = world_or_target
        raw_force = target_or_force if force is None else force

    mods = _resolve_modifiers(resolved_world, modifiers)
    valid_force = _validate_unit_interval(raw_force)
    if valid_force is None:
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    target_obj = _resolve_target_object(resolved_world, actual_target)
    if target_obj is None:
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if not in_reach(resolved_world, target_obj, modifiers=mods):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    effective_force = valid_force * mods.force
    if effective_force <= 0.0:
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    push_rules = [r for r in target_obj.behavior if r.verb == "push"]

    # Check if any push rule explicitly blocks or requires higher force
    for rule in push_rules:
        extras = _rule_extras(rule)
        resp = str(extras.get("response", "")).lower()
        if resp in ("none", "ignore", "noop", "blocked") or extras.get("scale") == 0:
            reason = str(extras.get("reason", NOTHING_HAPPENED))
            return ActionResult(ok=False, reason=reason)
        if not _rule_matches_force(rule, effective_force):
            return ActionResult(ok=False, reason=TOO_HEAVY)

    if direction is not None:
        try:
            push_dir = normalize_direction(direction)
        except (TypeError, ValueError):
            return ActionResult(ok=False, reason=INVALID_PARAMS)
    else:
        push_dir = _direction_from_agent_to_object(resolved_world, target_obj)

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        state_mutated = False
        for rule in push_rules:
            if _apply_rule_state_updates(resolved_world, target_obj, rule):
                state_mutated = True

        current_obj = resolved_world.objects.get(target_obj.id, target_obj)
        if current_obj.fixed:
            if state_mutated:
                return ActionResult(ok=True, reason=None, events=captured)
            reason = TOO_HEAVY if current_obj.weight > MAX_GRAB_WEIGHT else NOTHING_HAPPENED
            return ActionResult(ok=False, reason=reason)

        if current_obj.weight > MAX_PUSH_WEIGHT * max(0.1, effective_force):
            return ActionResult(ok=False, reason=TOO_HEAVY)

        before_vel = current_obj.velocity
        updated_obj = apply_impulse(resolved_world, current_obj.id, push_dir, effective_force)
        if updated_obj.velocity == before_vel and not state_mutated:
            return ActionResult(ok=False, reason=NOTHING_HAPPENED)

        if updated_obj.velocity != before_vel:
            resolved_world.events.emit(
                ObjectChangedEvent(
                    tick=resolved_world.tick,
                    source="agent",
                    object_id=updated_obj.id,
                    changes={"velocity": updated_obj.velocity.model_dump(mode="json")},
                    object=updated_obj.model_dump(mode="json"),
                )
            )
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={
            "object_id": target_obj.id,
            "velocity": resolved_world.objects[target_obj.id].velocity.model_dump(mode="json"),
        },
    )


def _is_object_ref(val: Any) -> bool:
    return isinstance(val, str | WorldObject | ActionResult)


def strike(
    world_or_target: Any,
    target_or_force: Any = 0.5,
    force: Any = None,
    fourth_arg: Any = None,
    *,
    direction: Any = None,
    tool: str | WorldObject | ActionResult | None = None,
    with_object: str | WorldObject | ActionResult | None = None,
    held: str | WorldObject | ActionResult | None = None,
    using: str | WorldObject | ActionResult | None = None,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Strike ``target`` with ``force`` (in ``0..1``).

    When the agent holds an object (or a ``tool`` is given), the impact scales
    with the tool's derived ``weight``, ``length``/``height``, ``rigidity``,
    and ``center_of_mass``. Bound composites fail at the joint when struck or
    swung above their ``joint_strength``.
    """
    explicit_tool = tool or with_object or held or using
    if isinstance(world_or_target, World):
        resolved_world = _resolve_world(world_or_target)
        if _is_object_ref(target_or_force) and _is_object_ref(force):
            explicit_tool = explicit_tool or target_or_force
            actual_target = force
            raw_force = 0.5 if fourth_arg is None else fourth_arg
        else:
            actual_target = target_or_force
            raw_force = 0.5 if force is None else force
    else:
        resolved_world = _resolve_world(world)
        if _is_object_ref(world_or_target) and _is_object_ref(target_or_force):
            explicit_tool = explicit_tool or world_or_target
            actual_target = target_or_force
            raw_force = 0.5 if force is None else force
        else:
            actual_target = world_or_target
            raw_force = target_or_force if force is None else force

    mods = _resolve_modifiers(resolved_world, modifiers)
    valid_force = _validate_unit_interval(raw_force)
    if valid_force is None:
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    target_obj = _resolve_target_object(resolved_world, actual_target)
    if target_obj is None:
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    tool_obj: WorldObject | None = None
    if explicit_tool is not None:
        tool_obj = _resolve_target_object(resolved_world, explicit_tool)
        if tool_obj is None or not in_reach(resolved_world, tool_obj, modifiers=mods):
            return ActionResult(ok=False, reason=OUT_OF_REACH)
    elif resolved_world.body.holding is not None and resolved_world.body.holding != target_obj.id:
        tool_obj = resolved_world.objects.get(resolved_world.body.holding)

    # A held long tool extends reach slightly by its length
    reach_bonus = max(0.0, tool_obj.length * 0.5) if tool_obj is not None else 0.0
    if not in_reach(resolved_world, target_obj, reach=DEFAULT_REACH + reach_bonus, modifiers=mods):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    effective_force = valid_force * mods.force
    if effective_force <= 0.0:
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    if direction is not None:
        try:
            strike_dir = normalize_direction(direction)
        except (TypeError, ValueError):
            return ActionResult(ok=False, reason=INVALID_PARAMS)
    else:
        strike_dir = _direction_from_agent_to_object(resolved_world, target_obj)

    tool_mult = strike_impact_multiplier(tool_obj)
    effective_impact = compute_strike_force(effective_force, tool_obj)

    strike_rules = [r for r in target_obj.behavior if r.verb == "strike"]
    break_rules = [
        r
        for r in strike_rules
        if str(_rule_extras(r).get("response", "")).lower()
        in ("break", "shatter", "remove", "destroy", "consume")
    ]

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        # Check if target itself is a bound Composite whose joint fails under load
        if (
            isinstance(target_obj, Composite)
            and target_obj.join_type == "bind"
            and target_obj.parts
            and effective_impact > target_obj.joint_strength
        ):
            restored = _separate_bound_composite(resolved_world, target_obj, source="agent")
            return ActionResult(
                ok=True,
                reason=None,
                events=captured,
                data={
                    "broken": True,
                    "joint_failed": True,
                    "object_id": target_obj.id,
                    "separated_parts": [p.id for p in restored],
                    "piece_ids": [p.id for p in restored],
                    "effective_force": effective_impact,
                    "impact_force": effective_impact,
                },
            )

        # Check if held/used bound Composite tool fails at its joint under load
        joint_failed = False
        separated_tool_parts: list[str] = []
        if (
            isinstance(tool_obj, Composite)
            and tool_obj.join_type == "bind"
            and tool_obj.parts
            and effective_force > tool_obj.joint_strength
        ):
            joint_failed = True

        # 1. Check if target object breaks (via explicit break rule or fragile property)
        matched_break_rule: BehaviorRule | None = None
        for b_rule in break_rules:
            if _rule_matches_force(b_rule, effective_impact):
                matched_break_rule = b_rule
                break

        should_break = matched_break_rule is not None
        if not should_break and target_obj.fragile and not break_rules:
            raw_thresh = target_obj.state.get("break_threshold", DEFAULT_FRAGILE_BREAK_THRESHOLD)
            thresh = (
                float(raw_thresh)
                if isinstance(raw_thresh, int | float) and not isinstance(raw_thresh, bool)
                else DEFAULT_FRAGILE_BREAK_THRESHOLD
            )
            if effective_impact > thresh or (thresh >= 1.0 and effective_impact >= 1.0):
                should_break = True

        if should_break:
            pieces = _break_object(resolved_world, target_obj, matched_break_rule)
            if joint_failed and isinstance(tool_obj, Composite) and tool_obj.id in resolved_world.objects:
                restored_parts = _separate_bound_composite(resolved_world, tool_obj, source="agent")
                separated_tool_parts = [p.id for p in restored_parts]
            return ActionResult(
                ok=True,
                reason=None,
                events=captured,
                data={
                    "broken": True,
                    "joint_failed": joint_failed,
                    "separated_parts": separated_tool_parts,
                    "object_id": target_obj.id,
                    "piece_ids": [p.id for p in pieces],
                    "effective_force": effective_impact,
                    "impact_force": effective_impact,
                    "tool_multiplier": tool_mult,
                },
            )

        # 2. Apply non-break strike rules (bounce, impulse, state change, etc.)
        any_effect = False
        impulse_applied = False
        for rule in strike_rules:
            if rule in break_rules:
                continue
            if not _rule_matches_force(rule, effective_impact):
                continue
            extras = _rule_extras(rule)
            resp = str(extras.get("response", "")).lower()
            if resp in ("none", "ignore", "noop", "blocked"):
                return ActionResult(ok=False, reason=NOTHING_HAPPENED)

            if _apply_rule_state_updates(resolved_world, target_obj, rule):
                any_effect = True

            current_obj = resolved_world.objects.get(target_obj.id, target_obj)
            if resp == "bounce":
                scale = float(extras.get("scale", 4.0)) / 4.0
                new_state = {**current_obj.state, "bounced": True}
                resolved_world.update_object(
                    current_obj.id,
                    {"state": new_state},
                    source="agent",
                )
                if not current_obj.fixed:
                    apply_impulse(
                        resolved_world,
                        current_obj.id,
                        strike_dir,
                        effective_impact * scale,
                    )
                impulse_applied = True
                any_effect = True
            elif resp in ("roll", "slide", "move", "impulse", "knock") or "scale" in extras:
                scale = float(extras.get("scale", 4.0)) / 4.0
                if not current_obj.fixed:
                    before_vel = current_obj.velocity
                    after_obj = apply_impulse(
                        resolved_world,
                        current_obj.id,
                        strike_dir,
                        effective_impact * scale,
                    )
                    if after_obj.velocity != before_vel:
                        resolved_world.events.emit(
                            ObjectChangedEvent(
                                tick=resolved_world.tick,
                                source="agent",
                                object_id=after_obj.id,
                                changes={"velocity": after_obj.velocity.model_dump(mode="json")},
                                object=after_obj.model_dump(mode="json"),
                            )
                        )
                        impulse_applied = True
                        any_effect = True

        # 3. Default physical strike impulse on movable objects
        current_obj = resolved_world.objects.get(target_obj.id)
        if current_obj is not None and not impulse_applied and not current_obj.fixed:
            before_vel = current_obj.velocity
            after_obj = apply_impulse(resolved_world, current_obj.id, strike_dir, effective_impact)
            if after_obj.velocity != before_vel:
                resolved_world.events.emit(
                    ObjectChangedEvent(
                        tick=resolved_world.tick,
                        source="agent",
                        object_id=after_obj.id,
                        changes={"velocity": after_obj.velocity.model_dump(mode="json")},
                        object=after_obj.model_dump(mode="json"),
                    )
                )
                any_effect = True

        if joint_failed and isinstance(tool_obj, Composite) and tool_obj.id in resolved_world.objects:
            restored_parts = _separate_bound_composite(resolved_world, tool_obj, source="agent")
            separated_tool_parts = [p.id for p in restored_parts]
            any_effect = True

        if not any_effect:
            return ActionResult(ok=False, reason=NOTHING_HAPPENED)
    finally:
        unsub()

    vel_dump = (
        resolved_world.objects[target_obj.id].velocity.model_dump(mode="json")
        if target_obj.id in resolved_world.objects
        else {"x": 0.0, "y": 0.0}
    )
    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={
            "broken": False,
            "joint_failed": joint_failed,
            "separated_parts": separated_tool_parts,
            "object_id": target_obj.id,
            "effective_force": effective_impact,
            "impact_force": effective_impact,
            "tool_multiplier": tool_mult,
            "velocity": vel_dump,
        },
    )


def grab(
    world_or_target: World | str | WorldObject,
    target: str | WorldObject | None = None,
    *,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Grab ``target`` if hands are free, target is within reach, and not too heavy."""
    if isinstance(world_or_target, World):
        resolved_world = _resolve_world(world_or_target)
        actual_target = target
    else:
        resolved_world = _resolve_world(world)
        actual_target = world_or_target

    mods = _resolve_modifiers(resolved_world, modifiers)
    target_obj = _resolve_target_object(resolved_world, actual_target)

    if resolved_world.body.holding is not None:
        if target_obj is not None and resolved_world.body.holding == target_obj.id:
            resolved_world._auto_held_composite_id = None
            return ActionResult(
                ok=True,
                reason=None,
                events=[],
                data={"holding": target_obj.id},
            )
        if resolved_world._auto_held_composite_id == resolved_world.body.holding:
            resolved_world.body.holding = None
            resolved_world._auto_held_composite_id = None
        else:
            return ActionResult(ok=False, reason=HANDS_FULL)

    if target_obj is None:
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if not in_reach(resolved_world, target_obj, modifiers=mods):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    grab_rules = [r for r in target_obj.behavior if r.verb == "grab"]
    for rule in grab_rules:
        extras = _rule_extras(rule)
        resp = str(extras.get("response", "")).lower()
        if resp in ("none", "ignore", "noop", "blocked", "deny"):
            reason = str(extras.get("reason", NOTHING_HAPPENED))
            return ActionResult(ok=False, reason=reason)
        if resp == "too_heavy":
            return ActionResult(ok=False, reason=TOO_HEAVY)

    max_weight = MAX_GRAB_WEIGHT * mods.force
    if target_obj.fixed or target_obj.weight > max_weight:
        return ActionResult(ok=False, reason=TOO_HEAVY)

    # Check grab failure probability from grab_fail_chance and dexterity
    clamped_fail = max(0.0, min(1.0, mods.grab_fail_chance))
    clamped_dex = max(0.0, min(1.0, mods.dexterity))
    fail_prob = 1.0 - (1.0 - clamped_fail) * clamped_dex
    if fail_prob >= 1.0:
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)
    if fail_prob > 0.0 and resolved_world.rng.random() < fail_prob:
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        if target_obj.level.startswith("inside:"):
            _detach_from_container(resolved_world, target_obj)

        # Topple anything that was stacked on top of the grabbed object
        for upper_id in sorted(resolved_world.objects):
            upper = resolved_world.objects.get(upper_id)
            if upper is not None and upper.level == f"on_top_of:{target_obj.id}":
                resolved_world._topple_object(
                    upper.id,
                    support=target_obj,
                    move_dx=0.0,
                    move_dy=0.0,
                    source="agent",
                )

        for rule in grab_rules:
            _apply_rule_state_updates(resolved_world, target_obj, rule)

        current_obj = resolved_world.objects[target_obj.id]
        prev_pos = current_obj.position
        agent_pos = resolved_world.body.position.model_copy()
        agent_level = (
            "raised"
            if resolved_world.terrain.is_ledge(agent_pos)
            else "ground"
        )
        updated_obj = current_obj.model_copy(
            update={
                "position": agent_pos,
                "level": agent_level,
                "velocity": Velocity(x=0.0, y=0.0),
            }
        )
        resolved_world.objects[target_obj.id] = updated_obj
        resolved_world.body.holding = target_obj.id
        resolved_world._auto_held_composite_id = None

        if prev_pos != agent_pos:
            resolved_world.events.emit(
                ObjectMovedEvent(
                    tick=resolved_world.tick,
                    source="agent",
                    object_id=target_obj.id,
                    previous_position=prev_pos,
                    position=agent_pos,
                )
            )
            resolved_world._propagate_container_contents(
                target_obj.id,
                new_pos=agent_pos,
                source="agent",
            )
        resolved_world.events.emit(
            ObjectChangedEvent(
                tick=resolved_world.tick,
                source="agent",
                object_id=target_obj.id,
                changes={"held": True, "level": agent_level},
                object=updated_obj.model_dump(mode="json"),
            )
        )
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={"holding": target_obj.id},
    )


def drop(
    world_or_target: World | str | WorldObject | None = None,
    target: str | WorldObject | None = None,
    *,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Drop the currently held object at the agent's position."""
    if isinstance(world_or_target, World):
        resolved_world = _resolve_world(world_or_target)
        explicit_target = target
    else:
        resolved_world = _resolve_world(world)
        explicit_target = world_or_target

    _ = _resolve_modifiers(resolved_world, modifiers)

    held_id = resolved_world.body.holding
    if held_id is None or held_id not in resolved_world.objects:
        resolved_world.body.holding = None
        resolved_world._auto_held_composite_id = None
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    if explicit_target is not None:
        target_id = explicit_target.id if isinstance(explicit_target, WorldObject) else str(explicit_target)
        if target_id != held_id:
            return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    held_obj = resolved_world.objects[held_id]
    drop_pos = resolved_world.body.position.model_copy()
    landing_level = "raised" if resolved_world.terrain.is_ledge(drop_pos) else "ground"

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        resolved_world.body.holding = None
        resolved_world._auto_held_composite_id = None
        prev_pos = held_obj.position
        updated_obj = held_obj.model_copy(
            update={
                "position": drop_pos,
                "level": landing_level,
                "velocity": Velocity(x=0.0, y=0.0),
            }
        )
        resolved_world.objects[held_id] = updated_obj

        if prev_pos != drop_pos:
            resolved_world.events.emit(
                ObjectMovedEvent(
                    tick=resolved_world.tick,
                    source="agent",
                    object_id=held_id,
                    previous_position=prev_pos,
                    position=drop_pos,
                )
            )
            resolved_world._propagate_container_contents(
                held_id,
                new_pos=drop_pos,
                source="agent",
            )
        resolved_world.events.emit(
            ObjectChangedEvent(
                tick=resolved_world.tick,
                source="agent",
                object_id=held_id,
                changes={"held": False, "level": landing_level},
                object=updated_obj.model_dump(mode="json"),
            )
        )

        drop_rules = [r for r in updated_obj.behavior if r.verb == "drop"]
        for rule in drop_rules:
            extras = _rule_extras(rule)
            resp = str(extras.get("response", "")).lower()
            if resp in ("break", "shatter", "remove", "destroy"):
                _break_object(resolved_world, updated_obj, rule)
                break
            _apply_rule_state_updates(resolved_world, updated_obj, rule)

        if held_id in resolved_world.objects:
            resolved_world.resolve_collisions()
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={"dropped": held_id, "position": {"x": drop_pos.x, "y": drop_pos.y}},
    )


def place(
    world_or_location: Any,
    location: Any = None,
    *,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Place the currently held item at ``location`` within reach."""
    if isinstance(world_or_location, World):
        resolved_world = _resolve_world(world_or_location)
        actual_location = location
    else:
        resolved_world = _resolve_world(world)
        actual_location = world_or_location

    mods = _resolve_modifiers(resolved_world, modifiers)

    held_id = resolved_world.body.holding
    if held_id is None or held_id not in resolved_world.objects:
        resolved_world.body.holding = None
        resolved_world._auto_held_composite_id = None
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    if actual_location is None:
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    # Resolve location (Position, (x, y), {'x', 'y'}, or target object)
    if isinstance(actual_location, str):
        target_obj = resolved_world.objects.get(actual_location)
        if target_obj is None:
            return ActionResult(ok=False, reason=OUT_OF_REACH)
        dest_pos = target_obj.position.model_copy()
    elif isinstance(actual_location, WorldObject):
        dest_pos = actual_location.position.model_copy()
    else:
        try:
            lx, ly = extract_xy(actual_location)
            dest_pos = Position(x=lx, y=ly)
        except (TypeError, ValueError, KeyError):
            return ActionResult(ok=False, reason=INVALID_PARAMS)

    if not in_reach(resolved_world, dest_pos, modifiers=mods):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    held_obj = resolved_world.objects[held_id]
    landing_level = "raised" if resolved_world.terrain.is_ledge(dest_pos) else "ground"

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        resolved_world.body.holding = None
        resolved_world._auto_held_composite_id = None
        prev_pos = held_obj.position
        updated_obj = held_obj.model_copy(
            update={
                "position": dest_pos,
                "level": landing_level,
                "velocity": Velocity(x=0.0, y=0.0),
            }
        )
        resolved_world.objects[held_id] = updated_obj

        if prev_pos != dest_pos:
            resolved_world.events.emit(
                ObjectMovedEvent(
                    tick=resolved_world.tick,
                    source="agent",
                    object_id=held_id,
                    previous_position=prev_pos,
                    position=dest_pos,
                )
            )
            resolved_world._propagate_container_contents(
                held_id,
                new_pos=dest_pos,
                source="agent",
            )
        resolved_world.events.emit(
            ObjectChangedEvent(
                tick=resolved_world.tick,
                source="agent",
                object_id=held_id,
                changes={"held": False, "level": landing_level},
                object=updated_obj.model_dump(mode="json"),
            )
        )

        place_rules = [r for r in updated_obj.behavior if r.verb == "place"]
        for rule in place_rules:
            _apply_rule_state_updates(resolved_world, updated_obj, rule)

        resolved_world.resolve_collisions()
    finally:
        unsub()

    final_pos = resolved_world.objects[held_id].position
    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={"placed": held_id, "position": {"x": final_pos.x, "y": final_pos.y}},
    )


def stack(
    world_or_held: Any = None,
    held_or_target: Any = None,
    target: Any = None,
    *,
    held: Any = None,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Stack ``held`` onto ``target``.

    Uses size, shape, and weight for a stability score; unstable stacks topple
    deterministically given the world seed.
    """
    if isinstance(world_or_held, World):
        resolved_world = _resolve_world(world_or_held)
        if target is not None:
            raw_held = held if held is not None else held_or_target
            raw_target = target
        elif held is not None:
            raw_held = held
            raw_target = held_or_target
        else:
            raw_held = resolved_world.body.holding
            raw_target = held_or_target
    else:
        resolved_world = _resolve_world(world)
        if held_or_target is not None:
            raw_held = world_or_held
            raw_target = held_or_target
        elif held is not None:
            raw_held = held
            raw_target = world_or_held
        else:
            raw_held = resolved_world.body.holding
            raw_target = world_or_held

    mods = _resolve_modifiers(resolved_world, modifiers)

    if raw_target is None:
        return ActionResult(ok=False, reason=INVALID_PARAMS)
    if raw_held is None:
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    held_obj = _resolve_target_object(resolved_world, raw_held)
    target_obj = _resolve_target_object(resolved_world, raw_target)
    if held_obj is None or target_obj is None:
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if held_obj.id == target_obj.id:
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    if (
        resolved_world.body.holding is not None
        and resolved_world.body.holding != held_obj.id
        and resolved_world._auto_held_composite_id != resolved_world.body.holding
    ):
        return ActionResult(ok=False, reason=HANDS_FULL)

    if not in_reach(resolved_world, held_obj, modifiers=mods) or not in_reach(
        resolved_world, target_obj, modifiers=mods
    ):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    max_weight = MAX_GRAB_WEIGHT * mods.force
    if held_obj.fixed or held_obj.weight > max_weight:
        return ActionResult(ok=False, reason=TOO_HEAVY)

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        if held_obj.level.startswith("inside:"):
            _detach_from_container(resolved_world, held_obj)

        support_id = resolved_world._top_of_stack(target_obj.id, exclude_id=held_obj.id)
        support_obj = resolved_world.objects[support_id]
        score = stack_stability_score(held_obj, support_obj, resolved_world.objects)

        if not support_obj.made_by_agent:
            resolved_world.update_object(
                support_obj.id,
                {"state": {**support_obj.state, "made_by_agent": True}},
                source="agent",
            )

        stacked_obj = resolved_world.stack(held_obj.id, support_id, source="agent")
        stable = stacked_obj.level == f"on_top_of:{support_id}"
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={
            "stacked": stacked_obj.id,
            "target": target_obj.id,
            "on_top_of": support_id,
            "stable": stable,
            "toppled": not stable,
            "stability_score": score,
            "stability": score,
            "made_by_agent": True,
            "level": stacked_obj.level,
            "position": {"x": stacked_obj.position.x, "y": stacked_obj.position.y},
            "object": stacked_obj,
        },
    )


def bind(
    world_or_held: Any = None,
    held_or_target: Any = None,
    target_or_binder: Any = None,
    binder: Any = None,
    *,
    held: Any = None,
    target: Any = None,
    binder_id: Any = None,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Bind ``held`` and ``target`` using ``binder`` (which requires ``binding=True``).

    Consumes ``binder`` into the joint and produces a ``Composite`` whose
    properties (weight, length/height, rigidity, center of mass, shape, and
    joint strength) are derived from its parts and binder.
    """
    explicit_binder = binder if binder is not None else binder_id
    if isinstance(world_or_held, World):
        resolved_world = _resolve_world(world_or_held)
        if explicit_binder is not None and target is None and held is None:
            if target_or_binder is not None:
                raw_held = held_or_target
                raw_target = target_or_binder
                raw_binder = explicit_binder
            else:
                raw_held = resolved_world.body.holding
                raw_target = held_or_target
                raw_binder = explicit_binder
        else:
            raw_held = held if held is not None else held_or_target
            raw_target = target if target is not None else target_or_binder
            raw_binder = explicit_binder
    else:
        resolved_world = _resolve_world(world)
        if target_or_binder is not None and explicit_binder is None:
            raw_held = world_or_held
            raw_target = held_or_target
            raw_binder = target_or_binder
        elif explicit_binder is not None:
            if held_or_target is not None and world_or_held is not None:
                raw_held = world_or_held
                raw_target = held_or_target
                raw_binder = explicit_binder
            else:
                raw_held = held if held is not None else resolved_world.body.holding
                raw_target = target if target is not None else world_or_held
                raw_binder = explicit_binder
        else:
            raw_held = held if held is not None else resolved_world.body.holding
            raw_target = target if target is not None else world_or_held
            raw_binder = held_or_target

    mods = _resolve_modifiers(resolved_world, modifiers)

    if raw_binder is None or raw_target is None:
        return ActionResult(ok=False, reason=INVALID_PARAMS)
    if raw_held is None:
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    held_obj = _resolve_target_object(resolved_world, raw_held)
    target_obj = _resolve_target_object(resolved_world, raw_target)
    binder_obj = _resolve_target_object(resolved_world, raw_binder)

    if held_obj is None or target_obj is None or binder_obj is None:
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if not (
        in_reach(resolved_world, held_obj, modifiers=mods)
        and in_reach(resolved_world, target_obj, modifiers=mods)
        and in_reach(resolved_world, binder_obj, modifiers=mods)
    ):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if not can_bind(binder_obj):
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    if held_obj.id == target_obj.id or binder_obj.id in (held_obj.id, target_obj.id):
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    if held_obj.fixed or target_obj.fixed or binder_obj.fixed:
        return ActionResult(ok=False, reason=TOO_HEAVY)

    # Dexterity check (cold reduces dexterity so binds can fail)
    clamped_dex = max(0.0, min(1.0, mods.dexterity))
    if clamped_dex <= 0.0:
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)
    if clamped_dex < 1.0 and resolved_world.rng.random() < (1.0 - clamped_dex):
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    was_holding = resolved_world.body.holding in (held_obj.id, target_obj.id, binder_obj.id)
    comp_pos = target_obj.position.model_copy()
    comp_level = "raised" if resolved_world.terrain.is_ledge(comp_pos) else "ground"

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        for obj in (held_obj, target_obj, binder_obj):
            if obj.level.startswith("inside:"):
                _detach_from_container(resolved_world, obj)

        # Despawn binder (consumed into the joint) and constituent parts
        resolved_world.despawn(binder_obj.id, source="agent")
        resolved_world.despawn(held_obj.id, source="agent")
        resolved_world.despawn(target_obj.id, source="agent")

        comp_id = resolved_world._factory._peek_next()
        composite = Composite.from_parts(
            [held_obj, target_obj],
            binder=binder_obj,
            id=comp_id,
            position=comp_pos,
            level=comp_level,
            join_type="bind",
            made_by_agent=True,
        )
        resolved_world._factory._claim(composite.id)
        resolved_world.objects[composite.id] = composite
        resolved_world.events.emit(
            ObjectSpawnedEvent(
                tick=resolved_world.tick,
                source="agent",
                object_id=composite.id,
                position=composite.position,
                object=composite.model_dump(mode="json"),
            )
        )

        if composite.weight <= MAX_GRAB_WEIGHT * mods.force and not composite.fixed:
            resolved_world.body.holding = composite.id
            resolved_world._auto_held_composite_id = None if was_holding else composite.id
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={
            "composite_id": composite.id,
            "id": composite.id,
            "composite": composite,
            "object": composite,
            "parts": [held_obj.id, target_obj.id],
            "binder": binder_obj.id,
            "binder_id": binder_obj.id,
            "joint_strength": composite.joint_strength,
            "center_of_mass": float(composite.center_of_mass),
            "made_by_agent": True,
        },
    )


def insert(
    world_or_held: Any = None,
    held_or_target: Any = None,
    target: Any = None,
    *,
    held: Any = None,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Insert ``held`` into hollow ``target`` if it has enough available volume."""
    if isinstance(world_or_held, World):
        resolved_world = _resolve_world(world_or_held)
        if target is not None:
            raw_held = held if held is not None else held_or_target
            raw_target = target
        elif held is not None:
            raw_held = held
            raw_target = held_or_target
        else:
            raw_held = resolved_world.body.holding
            raw_target = held_or_target
    else:
        resolved_world = _resolve_world(world)
        if held_or_target is not None:
            raw_held = world_or_held
            raw_target = held_or_target
        elif held is not None:
            raw_held = held
            raw_target = world_or_held
        else:
            raw_held = resolved_world.body.holding
            raw_target = world_or_held

    mods = _resolve_modifiers(resolved_world, modifiers)

    if raw_target is None:
        return ActionResult(ok=False, reason=INVALID_PARAMS)
    if raw_held is None:
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    held_obj = _resolve_target_object(resolved_world, raw_held)
    target_obj = _resolve_target_object(resolved_world, raw_target)
    if held_obj is None or target_obj is None:
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if held_obj.id == target_obj.id:
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    if (
        resolved_world.body.holding is not None
        and resolved_world.body.holding != held_obj.id
        and resolved_world._auto_held_composite_id != resolved_world.body.holding
    ):
        return ActionResult(ok=False, reason=HANDS_FULL)

    if not in_reach(resolved_world, held_obj, modifiers=mods) or not in_reach(
        resolved_world, target_obj, modifiers=mods
    ):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    max_weight = MAX_GRAB_WEIGHT * mods.force
    if held_obj.fixed or held_obj.weight > max_weight:
        return ActionResult(ok=False, reason=TOO_HEAVY)

    if not target_obj.hollow:
        return ActionResult(
            ok=False,
            reason=NOTHING_HAPPENED,
            data={"reason_detail": "not_hollow"},
        )

    current_contents = [
        o for o in resolved_world.objects.values()
        if o.level == f"inside:{target_obj.id}" and o.id != held_obj.id
    ]
    used_volume = sum(o.volume for o in current_contents)
    raw_cap = target_obj.state.get("capacity", target_obj.volume)
    capacity = (
        float(raw_cap)
        if isinstance(raw_cap, int | float) and not isinstance(raw_cap, bool)
        else target_obj.volume
    )
    if capacity <= 0.0 or held_obj.volume > (capacity - used_volume) + 1e-9:
        return ActionResult(
            ok=False,
            reason=NOTHING_HAPPENED,
            data={"reason_detail": "insufficient_volume"},
        )

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        if resolved_world.body.holding == held_obj.id:
            resolved_world.body.holding = None
            resolved_world._auto_held_composite_id = None

        if held_obj.level.startswith("inside:"):
            _detach_from_container(resolved_world, held_obj)

        for upper_id in sorted(resolved_world.objects):
            upper = resolved_world.objects.get(upper_id)
            if upper is not None and upper.level == f"on_top_of:{held_obj.id}":
                resolved_world._topple_object(
                    upper.id,
                    support=held_obj,
                    move_dx=0.0,
                    move_dy=0.0,
                    source="agent",
                )

        prev_pos = held_obj.position
        dest_pos = target_obj.position.model_copy()
        inside_level = f"inside:{target_obj.id}"
        updated_held = held_obj.model_copy(
            update={
                "position": dest_pos,
                "level": inside_level,
                "velocity": Velocity(x=0.0, y=0.0),
                "state": {**held_obj.state, "made_by_agent": True},
            }
        )
        resolved_world.objects[held_obj.id] = updated_held

        if prev_pos != dest_pos:
            resolved_world.events.emit(
                ObjectMovedEvent(
                    tick=resolved_world.tick,
                    source="agent",
                    object_id=held_obj.id,
                    previous_position=prev_pos,
                    position=dest_pos,
                )
            )
            resolved_world._propagate_container_contents(
                held_obj.id,
                new_pos=dest_pos,
                source="agent",
            )
        resolved_world.events.emit(
            ObjectChangedEvent(
                tick=resolved_world.tick,
                source="agent",
                object_id=held_obj.id,
                changes={"level": inside_level, "state": updated_held.state},
                object=updated_held.model_dump(mode="json"),
            )
        )

        # Upgrade container to a Composite whose weight includes its contents
        refreshed_target = resolved_world.objects[target_obj.id]
        if isinstance(refreshed_target, Composite) and refreshed_target.base_object is not None:
            base_dump = dict(refreshed_target.base_object)
            base_container = WorldObject.model_validate(base_dump)
        else:
            base_dump = refreshed_target.model_dump(exclude=set(_COMPOSITE_EXTRA_KEYS))
            base_container = WorldObject.model_validate(base_dump)

        all_contents = [*current_contents, updated_held]
        content_ids = [o.id for o in all_contents]
        total_weight = round(base_container.weight + sum(o.weight for o in all_contents), 6)
        new_container_state = {
            **refreshed_target.state,
            "made_by_agent": True,
            "contents": content_ids,
        }
        container_comp = Composite.model_validate(
            {
                **base_dump,
                "position": refreshed_target.position.model_copy(),
                "level": refreshed_target.level,
                "velocity": refreshed_target.velocity.model_copy(),
                "weight": total_weight,
                "state": new_container_state,
                "parts": [base_container, *all_contents],
                "join_type": "insert",
                "base_object": base_dump,
                "made_by_agent": True,
            }
        )
        resolved_world.objects[target_obj.id] = container_comp
        resolved_world.events.emit(
            ObjectChangedEvent(
                tick=resolved_world.tick,
                source="agent",
                object_id=target_obj.id,
                changes={"weight": total_weight, "state": new_container_state},
                object=container_comp.model_dump(mode="json"),
            )
        )
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={
            "inserted": held_obj.id,
            "into": target_obj.id,
            "container_id": target_obj.id,
            "composite_id": target_obj.id,
            "composite": resolved_world.objects[target_obj.id],
            "object": resolved_world.objects[target_obj.id],
            "contents": content_ids,
            "made_by_agent": True,
        },
    )


def separate(
    world_or_target: Any = None,
    target: Any = None,
    *,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Take a composite, filled container, or stack apart and restore its original parts."""
    if isinstance(world_or_target, World):
        resolved_world = _resolve_world(world_or_target)
        raw_target = target if target is not None else resolved_world.body.holding
    else:
        resolved_world = _resolve_world(world)
        raw_target = world_or_target if world_or_target is not None else (
            target if target is not None else resolved_world.body.holding
        )

    mods = _resolve_modifiers(resolved_world, modifiers)
    if raw_target is None:
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    target_obj = _resolve_target_object(resolved_world, raw_target)
    if target_obj is None:
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if not in_reach(resolved_world, target_obj, modifiers=mods):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        # Case 1: Bound Composite
        if isinstance(target_obj, Composite) and target_obj.join_type == "bind" and target_obj.parts:
            restored = _separate_bound_composite(resolved_world, target_obj, source="agent")
            part_ids = [p.id for p in restored]
            return ActionResult(
                ok=True,
                reason=None,
                events=captured,
                data={
                    "separated": target_obj.id,
                    "parts": part_ids,
                    "part_ids": part_ids,
                    "part_objects": restored,
                },
            )

        # Case 2: Target is inside a container, or target is a container holding objects
        container_obj: WorldObject | None = None
        items_to_remove: list[WorldObject] = []
        if target_obj.level.startswith("inside:"):
            cid = target_obj.level.removeprefix("inside:")
            container_obj = resolved_world.objects.get(cid)
            items_to_remove = [target_obj]
        else:
            inside_items = [
                o for o in resolved_world.objects.values()
                if o.level == f"inside:{target_obj.id}"
            ]
            if inside_items:
                container_obj = target_obj
                items_to_remove = inside_items

        if container_obj is not None and items_to_remove:
            base_pos = container_obj.position
            landing_level = "raised" if resolved_world.terrain.is_ledge(base_pos) else "ground"
            count = len(items_to_remove)
            for idx, item in enumerate(items_to_remove):
                _detach_from_container(resolved_world, item)
                angle = idx * (2.0 * math.pi / max(1, count))
                sep_dist = container_obj.radius + item.radius + 0.08
                new_pos = Position(
                    x=base_pos.x + math.cos(angle) * sep_dist,
                    y=base_pos.y + math.sin(angle) * sep_dist,
                )
                prev_pos = item.position
                updated_item = item.model_copy(
                    update={
                        "position": new_pos,
                        "level": landing_level,
                        "velocity": Velocity(x=0.0, y=0.0),
                    }
                )
                resolved_world.objects[item.id] = updated_item
                if prev_pos != new_pos:
                    resolved_world.events.emit(
                        ObjectMovedEvent(
                            tick=resolved_world.tick,
                            source="agent",
                            object_id=item.id,
                            previous_position=prev_pos,
                            position=new_pos,
                        )
                    )
                resolved_world.events.emit(
                    ObjectChangedEvent(
                        tick=resolved_world.tick,
                        source="agent",
                        object_id=item.id,
                        changes={"level": landing_level},
                        object=updated_item.model_dump(mode="json"),
                    )
                )
            resolved_world.resolve_collisions()
            part_ids = [container_obj.id, *[o.id for o in items_to_remove]]
            part_objs = [resolved_world.objects[pid] for pid in part_ids if pid in resolved_world.objects]
            return ActionResult(
                ok=True,
                reason=None,
                events=captured,
                data={
                    "separated": container_obj.id,
                    "parts": part_ids,
                    "part_ids": part_ids,
                    "part_objects": part_objs,
                },
            )

        # Case 3: Stacked objects (target is on_top_of something or has objects on_top_of it)
        stacked_above = [
            o for o in resolved_world.objects.values()
            if o.level == f"on_top_of:{target_obj.id}"
        ]
        if target_obj.level.startswith("on_top_of:") or stacked_above:
            if target_obj.level.startswith("on_top_of:"):
                sup_id = target_obj.level.removeprefix("on_top_of:")
                sup = resolved_world.objects.get(sup_id, target_obj)
                resolved_world._topple_object(
                    target_obj.id,
                    support=sup,
                    move_dx=0.0,
                    move_dy=0.0,
                    source="agent",
                )
                resolved_world.resolve_collisions()
                part_ids = [sup.id, target_obj.id]
            else:
                part_ids = [target_obj.id]
                for upper in stacked_above:
                    part_ids.append(upper.id)
                    resolved_world._topple_object(
                        upper.id,
                        support=target_obj,
                        move_dx=0.0,
                        move_dy=0.0,
                        source="agent",
                    )
                resolved_world.resolve_collisions()
            part_objs = [resolved_world.objects[pid] for pid in part_ids if pid in resolved_world.objects]
            return ActionResult(
                ok=True,
                reason=None,
                events=captured,
                data={
                    "separated": target_obj.id,
                    "parts": part_ids,
                    "part_ids": part_ids,
                    "part_objects": part_objs,
                },
            )
    finally:
        unsub()

    return ActionResult(ok=False, reason=NOTHING_HAPPENED)


def speak(
    world_or_text: World | str,
    text: str | None = None,
    *,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> ActionResult:
    """Vocalize ``text``, emitting a ``spoke`` event."""
    if isinstance(world_or_text, World):
        resolved_world = _resolve_world(world_or_text)
        actual_text = text
    else:
        resolved_world = _resolve_world(world)
        actual_text = world_or_text

    _ = _resolve_modifiers(resolved_world, modifiers)

    if not isinstance(actual_text, str):
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        resolved_world.events.emit(
            SpokeEvent(
                tick=resolved_world.tick,
                source="agent",
                text=actual_text,
                position=resolved_world.body.position.model_copy(),
            )
        )
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={"text": actual_text},
    )


def register_opaque_action(
    world_or_obj: World | str | WorldObject,
    obj_or_action: str | WorldObject | OpaqueHandler | None = None,
    action_or_handler: str | OpaqueHandler | None = None,
    handler: OpaqueHandler | None = None,
    *,
    world: World | None = None,
) -> str:
    """Register an opaque action ``action_<n>`` with ``handler`` on an object."""
    resolved_world: World | None
    if isinstance(world_or_obj, World):
        resolved_world = world_or_obj
        _set_active_world(resolved_world)
        target_arg = obj_or_action
        third_arg = action_or_handler
        fourth_arg = handler
    else:
        resolved_world = world if world is not None else _get_active_world()
        target_arg = world_or_obj
        third_arg = obj_or_action
        fourth_arg = action_or_handler

    if isinstance(target_arg, WorldObject):
        object_id = target_arg.id
    elif isinstance(target_arg, str):
        object_id = target_arg
    else:
        raise TypeError("object target must be an object_<n> id or WorldObject")

    if _OBJECT_ID.fullmatch(object_id) is None:
        raise ValueError(f"object_id must match 'object_<n>', got {object_id!r}")

    if callable(third_arg) and fourth_arg is None:
        action_id = _next_opaque_action_id(resolved_world)
        fn = third_arg
    elif isinstance(third_arg, str) and callable(fourth_arg):
        action_id = third_arg
        fn = fourth_arg
    elif callable(fourth_arg) and third_arg is None:
        action_id = _next_opaque_action_id(resolved_world)
        fn = fourth_arg
    else:
        raise TypeError(
            "register_opaque_action requires a callable handler and optional 'action_<n>' id"
        )

    if _OPAQUE_ACTION_ID.fullmatch(action_id) is None:
        raise ValueError(f"opaque action id must match 'action_<n>', got {action_id!r}")

    if resolved_world is not None:
        reg = _world_opaque_registry(resolved_world)
        reg.setdefault(object_id, {})[action_id] = fn
    _FALLBACK_OPAQUE_ACTIONS.setdefault(object_id, {})[action_id] = fn
    return action_id


register_action = register_opaque_action


def _object_opaque_actions(world: World, obj: WorldObject) -> dict[str, OpaqueHandler | None]:
    actions: dict[str, OpaqueHandler | None] = {}
    if obj.id in _FALLBACK_OPAQUE_ACTIONS:
        actions.update(_FALLBACK_OPAQUE_ACTIONS[obj.id])
    reg = _world_opaque_registry(world)
    if obj.id in reg:
        actions.update(reg[obj.id])
    for rule in obj.behavior:
        if _OPAQUE_ACTION_ID.fullmatch(rule.verb) and rule.verb not in actions:
            actions[rule.verb] = None
    return actions


def available_opaque_actions(
    world: World | None = None,
    *,
    target: str | WorldObject | None = None,
    reach: float = DEFAULT_REACH,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
) -> list[str]:
    """Return sorted ``action_<n>`` ids offered by objects currently within reach."""
    resolved_world = _resolve_world(world)
    mods = _resolve_modifiers(resolved_world, modifiers)

    offered: set[str] = set()
    if target is not None:
        obj = _resolve_target_object(resolved_world, target)
        if obj is None or not in_reach(resolved_world, obj, reach=reach, modifiers=mods):
            return []
        offered.update(_object_opaque_actions(resolved_world, obj).keys())
    else:
        for obj_id in sorted(resolved_world.objects):
            obj = resolved_world.objects[obj_id]
            if not in_reach(resolved_world, obj, reach=reach, modifiers=mods):
                continue
            offered.update(_object_opaque_actions(resolved_world, obj).keys())

    return sorted(offered, key=lambda name: int(name.removeprefix("action_")))


get_opaque_actions = available_opaque_actions
offered_opaque_actions = available_opaque_actions
available_actions = available_opaque_actions


def _invoke_handler(
    handler: OpaqueHandler,
    world: World,
    obj: WorldObject,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Invoke a registered opaque action handler, adapting to its parameter signature."""
    try:
        sig = inspect.signature(handler)
        params = list(sig.parameters.values())
    except (TypeError, ValueError):
        return handler(world, obj, *args, **kwargs)

    has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    positional_params = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    if has_var_positional:
        return handler(world, obj, *args, **kwargs if has_var_keyword else {})

    n_pos = len(positional_params)
    if n_pos == 0:
        return handler(**kwargs) if has_var_keyword else handler()

    first_name = positional_params[0].name.lower()
    if first_name in ("world", "w", "sim"):
        candidate_args = (world, obj, *args) if n_pos >= 2 else (world,)
    elif first_name in ("obj", "object", "target", "self", "item", "object_id"):
        first_val = obj.id if first_name == "object_id" else obj
        candidate_args = (first_val, *args)
    elif args and n_pos == len(args):
        candidate_args = args
    elif n_pos == 1 and not args:
        candidate_args = (obj,)
    else:
        candidate_args = (world, obj, *args)

    trimmed_args = candidate_args[:n_pos]
    accepted_kwargs = (
        kwargs
        if has_var_keyword
        else {k: v for k, v in kwargs.items() if k in sig.parameters}
    )
    return handler(*trimmed_args, **accepted_kwargs)


def invoke_opaque_action(
    world_or_action: World | str,
    action_or_arg: Any = None,
    *args: Any,
    target: str | WorldObject | None = None,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ActionResult:
    """Execute a registered ``action_<n>`` if the providing object is within reach."""
    if isinstance(world_or_action, World):
        resolved_world = _resolve_world(world_or_action)
        action_id = action_or_arg
        call_args = args
    else:
        resolved_world = _resolve_world(world)
        action_id = world_or_action
        call_args = (() if action_or_arg is None else (action_or_arg, *args))

    mods = _resolve_modifiers(resolved_world, modifiers)

    if not isinstance(action_id, str) or _OPAQUE_ACTION_ID.fullmatch(action_id) is None:
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    # Locate candidate objects in the world that provide action_id
    candidates: list[tuple[WorldObject, OpaqueHandler | None]] = []
    if target is not None:
        target_obj = _resolve_target_object(resolved_world, target)
        if target_obj is None:
            return ActionResult(ok=False, reason=OUT_OF_REACH)
        obj_actions = _object_opaque_actions(resolved_world, target_obj)
        if action_id not in obj_actions:
            return ActionResult(ok=False, reason=NOTHING_HAPPENED)
        candidates.append((target_obj, obj_actions[action_id]))
    else:
        for obj_id in sorted(resolved_world.objects):
            obj = resolved_world.objects[obj_id]
            obj_actions = _object_opaque_actions(resolved_world, obj)
            if action_id in obj_actions:
                candidates.append((obj, obj_actions[action_id]))

    if not candidates:
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    reachable_candidates = [
        (obj, fn)
        for obj, fn in candidates
        if in_reach(resolved_world, obj, modifiers=mods)
    ]
    if not reachable_candidates:
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    provider_obj, handler = reachable_candidates[0]
    captured: list[WorldEvent] = []
    unsub = resolved_world.subscribe(captured.append)
    try:
        # Apply any BehaviorRule with verb == action_id
        for rule in provider_obj.behavior:
            if rule.verb == action_id:
                _apply_rule_state_updates(resolved_world, provider_obj, rule)

        if handler is not None:
            outcome = _invoke_handler(handler, resolved_world, provider_obj, call_args, kwargs)
            if isinstance(outcome, ActionResult):
                merged_events = list(captured)
                for ev in outcome.events:
                    if ev not in merged_events:
                        merged_events.append(ev)
                return outcome.model_copy(update={"events": merged_events})
            if isinstance(outcome, bool):
                return ActionResult(
                    ok=outcome,
                    reason=None if outcome else NOTHING_HAPPENED,
                    events=captured,
                    data={"action": action_id, "object_id": provider_obj.id},
                )
            data_payload: dict[str, Any] = {
                "action": action_id,
                "object_id": provider_obj.id,
            }
            if isinstance(outcome, Mapping):
                data_payload.update(dict(outcome))
            elif outcome is not None:
                data_payload["result"] = outcome
            return ActionResult(ok=True, reason=None, events=captured, data=data_payload)
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={"action": action_id, "object_id": provider_obj.id},
    )


execute_opaque_action = invoke_opaque_action
run_opaque_action = invoke_opaque_action
opaque_action = invoke_opaque_action


def execute_action(
    world_or_verb: World | str,
    verb_or_target: str | None = None,
    target: Any = None,
    params: Mapping[str, Any] | None = None,
    *,
    world: World | None = None,
    modifiers: BodyModifiers | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ActionResult:
    """Dispatch a body verb or opaque ``action_<n>`` by name."""
    if isinstance(world_or_verb, World):
        resolved_world = _resolve_world(world_or_verb)
        verb = str(verb_or_target or "")
        actual_target = target
        actual_params = dict(params or {})
    else:
        resolved_world = _resolve_world(world)
        verb = world_or_verb
        actual_target = verb_or_target
        actual_params = dict(target) if isinstance(target, Mapping) and params is None else dict(params or {})

    actual_params.update(kwargs)

    if verb == "move":
        direction = actual_params.get("direction", actual_target)
        speed = actual_params.get("speed", 1.0)
        return move(resolved_world, direction, speed, modifiers=modifiers)
    if verb == "look":
        tgt = actual_target if actual_target is not None else actual_params.get("target")
        return look(resolved_world, tgt, modifiers=modifiers)
    if verb == "push":
        tgt = actual_target if actual_target is not None else actual_params.get("target")
        force = actual_params.get("force", 0.5)
        direction = actual_params.get("direction")
        return push(resolved_world, tgt, force, direction=direction, modifiers=modifiers)
    if verb == "strike":
        tgt = actual_target if actual_target is not None else actual_params.get("target")
        force = actual_params.get("force", 0.5)
        direction = actual_params.get("direction")
        tool = actual_params.get("tool", actual_params.get("held", actual_params.get("with_object")))
        return strike(resolved_world, tgt, force, direction=direction, tool=tool, modifiers=modifiers)
    if verb == "grab":
        tgt = actual_target if actual_target is not None else actual_params.get("target")
        return grab(resolved_world, tgt, modifiers=modifiers)
    if verb == "drop":
        tgt = actual_target if actual_target is not None else actual_params.get("target")
        return drop(resolved_world, tgt, modifiers=modifiers)
    if verb == "place":
        loc = actual_params.get("location", actual_params.get("position", actual_target))
        return place(resolved_world, loc, modifiers=modifiers)
    if verb == "stack":
        tgt = actual_target if actual_target is not None else actual_params.get("target")
        held_arg = actual_params.get("held", actual_params.get("held_item"))
        return stack(resolved_world, held_arg, tgt, modifiers=modifiers)
    if verb == "bind":
        tgt = actual_target if actual_target is not None else actual_params.get("target")
        held_arg = actual_params.get("held", actual_params.get("held_item"))
        binder_arg = actual_params.get("binder", actual_params.get("binder_id"))
        return bind(resolved_world, held_arg, tgt, binder_arg, modifiers=modifiers)
    if verb == "insert":
        tgt = actual_target if actual_target is not None else actual_params.get("target")
        held_arg = actual_params.get("held", actual_params.get("held_item"))
        return insert(resolved_world, held_arg, tgt, modifiers=modifiers)
    if verb == "separate":
        tgt = actual_target if actual_target is not None else actual_params.get("target")
        return separate(resolved_world, tgt, modifiers=modifiers)
    if verb == "speak":
        text = actual_params.get("text", actual_target)
        return speak(resolved_world, text, modifiers=modifiers)
    if _OPAQUE_ACTION_ID.fullmatch(verb):
        return invoke_opaque_action(
            resolved_world,
            verb,
            target=actual_target,
            modifiers=modifiers,
            **actual_params,
        )

    return ActionResult(ok=False, reason=NOTHING_HAPPENED)


__all__ = [
    "DEFAULT_FRAGILE_BREAK_THRESHOLD",
    "DEFAULT_REACH",
    "HANDS_FULL",
    "INVALID_PARAMS",
    "MAX_GRAB_WEIGHT",
    "MAX_PUSH_WEIGHT",
    "NOTHING_HAPPENED",
    "OUT_OF_REACH",
    "REASON_CODES",
    "TOO_HEAVY",
    "ActionResult",
    "BodyModifiers",
    "Composite",
    "OpaqueHandler",
    "ReasonCode",
    "available_actions",
    "available_opaque_actions",
    "bind",
    "drop",
    "execute_action",
    "execute_opaque_action",
    "get_opaque_actions",
    "grab",
    "in_reach",
    "insert",
    "invoke_opaque_action",
    "look",
    "move",
    "offered_opaque_actions",
    "opaque_action",
    "place",
    "push",
    "register_action",
    "register_opaque_action",
    "run_opaque_action",
    "separate",
    "speak",
    "stack",
    "strike",
]

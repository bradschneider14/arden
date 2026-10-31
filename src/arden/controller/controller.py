"""Scripted low-level controller: carries out an Intent's steps tick by tick.

The controller is scripted (not reinforcement-learned): it navigates on the
2D tile grid with A* obstacle avoidance (and a greedy steering fallback),
dispatches body verbs in order, retries a failed step at most once, emits an
``intent_ended`` event with the final observation when an intent finishes, and
idles (rests) when no intent is active.
"""

from __future__ import annotations

import heapq
import math
import re
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

import arden.world as _world_pkg
import arden.world.world as _world_mod
from arden.actions.verbs import (
     BASE_MOVE_STEP,
    DEFAULT_REACH,
    INVALID_PARAMS,
    NOTHING_HAPPENED,
    OUT_OF_REACH,
    ActionResult,
    BodyModifiers,
    _agent_elevation,
    _resolve_modifiers,
    execute_action,
    in_reach,
    move as move_verb,
)
from arden.llm.schemas import Intent, Step
from arden.perception.feed import Percept, perceive
from arden.world.objects import Position, WorldObject
from arden.world.physics import (
    LEDGE_HEIGHT,
    _get_active_world,
    _object_elevation,
    _set_active_world,
    extract_xy,
    is_occluded,
)
from arden.world.world import EventBus, World, WorldEvent

RUNNING: Literal["running"] = "running"
SUCCEEDED: Literal["succeeded"] = "succeeded"
FAILED: Literal["failed"] = "failed"
IDLE: Literal["idle"] = "idle"

NO_PATH: Literal["no_path"] = "no_path"
INTENT_ENDED: Literal["intent_ended"] = "intent_ended"

ControllerState = Literal["running", "succeeded", "failed", "idle"]

_OBJECT_ID_RE = re.compile(r"^object_[0-9]+$")
_TOWARD_OBJECT_RE = re.compile(r"^(?:toward|towards|to)\s+(object_[0-9]+)$", re.IGNORECASE)
_COORD_STRING_RE = re.compile(
    r"^\s*\(?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)?\s*$"
)

# Register "intent_ended" on the world event bus so world.subscribe("intent_ended", fn) works.
if "intent_ended" not in _world_mod.EVENT_TYPES:
    _world_mod.EVENT_TYPES = frozenset({*_world_mod.EVENT_TYPES, "intent_ended"})

_orig_event_bus_subscribe = EventBus.subscribe
_orig_event_bus_unsubscribe = EventBus.unsubscribe
_orig_record_perception_event = World._record_perception_event


def _patched_event_bus_subscribe(
    self: EventBus,
    handler_or_type: Any,
    handler: Any = None,
    *,
    event_type: Any = None,
) -> Callable[[], None]:
    self._typed_subscribers.setdefault("intent_ended", [])
    return _orig_event_bus_subscribe(self, handler_or_type, handler, event_type=event_type)


def _patched_event_bus_unsubscribe(
    self: EventBus,
    handler: Any,
    event_type: Any = None,
) -> None:
    self._typed_subscribers.setdefault("intent_ended", [])
    return _orig_event_bus_unsubscribe(self, handler, event_type=event_type)


def _patched_record_perception_event(self: World, event: WorldEvent) -> None:
    if event.type == "intent_ended":
        return
    _orig_record_perception_event(self, event)


if getattr(EventBus.subscribe, "__name__", "") != "_patched_event_bus_subscribe":
    EventBus.subscribe = _patched_event_bus_subscribe  # type: ignore[method-assign]
    EventBus.unsubscribe = _patched_event_bus_unsubscribe  # type: ignore[method-assign]
    World._record_perception_event = _patched_record_perception_event  # type: ignore[method-assign]


class IntentEndedEvent(WorldEvent):
    """Emitted when the controller completes or fails an ``Intent``."""

    type: Literal["intent_ended"] = "intent_ended"  # type: ignore[assignment]
    source: str = "controller"
    status: Literal["succeeded", "failed"]
    intent: Intent
    reason: str | None = None
    step: Step | None = None
    step_index: int | None = None
    observation: Percept

    @property
    def ok(self) -> bool:
        """True when the intent succeeded."""
        return self.status == SUCCEEDED

    @property
    def success(self) -> bool:
        """Alias for ``ok``."""
        return self.status == SUCCEEDED

    @property
    def succeeded(self) -> bool:
        """True when the intent succeeded."""
        return self.status == SUCCEEDED

    @property
    def failed(self) -> bool:
        """True when the intent failed."""
        return self.status == FAILED

    @property
    def reason_code(self) -> str | None:
        """Alias for ``reason``."""
        return self.reason

    @property
    def failed_step(self) -> Step | None:
        """Alias for ``step``."""
        return self.step

    @property
    def percept(self) -> Percept:
        """Alias for ``observation``."""
        return self.observation

    @property
    def final_observation(self) -> Percept:
        """Alias for ``observation``."""
        return self.observation

    @property
    def goal(self) -> str:
        """Goal string of the completed intent."""
        return self.intent.goal

    @property
    def expect(self) -> str:
        """Predicted outcome from the completed intent."""
        return self.intent.expect

    @property
    def confidence(self) -> float:
        """Confidence score from the completed intent."""
        return self.intent.confidence


setattr(_world_mod, "IntentEndedEvent", IntentEndedEvent)
setattr(_world_pkg, "IntentEndedEvent", IntentEndedEvent)


class ControllerStatus(BaseModel):
    """Result of a single ``Controller.tick(world)`` call.

    Can be compared directly to ``"running"``, ``"succeeded"``, ``"failed"``,
    ``"idle"``, ``"failed(<reason>)"``, or another ``ControllerStatus``.
    """

    model_config = ConfigDict(extra="forbid")

    state: ControllerState
    reason: str | None = None
    step: Step | None = None
    step_index: int | None = None
    intent: Intent | None = None
    observation: Percept | None = None
    event: IntentEndedEvent | None = None
    resting: bool = False

    RUNNING: ClassVar[str] = RUNNING
    SUCCEEDED: ClassVar[str] = SUCCEEDED
    FAILED: ClassVar[str] = FAILED
    IDLE: ClassVar[str] = IDLE

    @classmethod
    def running(
        cls,
        *,
        intent: Intent | None = None,
        step: Step | None = None,
        step_index: int | None = None,
    ) -> ControllerStatus:
        """Construct a ``running`` controller status."""
        return cls(
            state=RUNNING,
            intent=intent,
            step=step,
            step_index=step_index,
            resting=False,
        )

    @classmethod
    def succeeded(
        cls,
        *,
        intent: Intent | None = None,
        observation: Percept | None = None,
        event: IntentEndedEvent | None = None,
    ) -> ControllerStatus:
        """Construct a ``succeeded`` controller status."""
        return cls(
            state=SUCCEEDED,
            intent=intent,
            observation=observation,
            event=event,
            resting=False,
        )

    @classmethod
    def failed(
        cls,
        reason: str | None = None,
        *,
        step: Step | None = None,
        step_index: int | None = None,
        intent: Intent | None = None,
        observation: Percept | None = None,
        event: IntentEndedEvent | None = None,
    ) -> ControllerStatus:
        """Construct a ``failed(reason)`` controller status."""
        return cls(
            state=FAILED,
            reason=reason,
            step=step,
            step_index=step_index,
            intent=intent,
            observation=observation,
            event=event,
            resting=False,
        )

    @classmethod
    def idle(cls) -> ControllerStatus:
        """Construct an ``idle`` (resting) controller status."""
        return cls(state=IDLE, resting=True)

    @property
    def status(self) -> ControllerState:
        """Alias for ``state``."""
        return self.state

    @property
    def kind(self) -> ControllerState:
        """Alias for ``state``."""
        return self.state

    @property
    def reason_code(self) -> str | None:
        """Alias for ``reason``."""
        return self.reason

    @property
    def failed_step(self) -> Step | None:
        """Alias for ``step``."""
        return self.step

    @property
    def percept(self) -> Percept | None:
        """Alias for ``observation``."""
        return self.observation

    @property
    def final_observation(self) -> Percept | None:
        """Alias for ``observation``."""
        return self.observation

    @property
    def is_running(self) -> bool:
        return self.state == RUNNING

    @property
    def is_succeeded(self) -> bool:
        return self.state == SUCCEEDED

    @property
    def is_failed(self) -> bool:
        return self.state == FAILED

    @property
    def is_idle(self) -> bool:
        return self.state == IDLE

    @property
    def is_resting(self) -> bool:
        return self.resting or self.state == IDLE

    @property
    def ok(self) -> bool:
        return self.state == SUCCEEDED

    def __str__(self) -> str:
        if self.state == FAILED and self.reason:
            return f"failed({self.reason})"
        return self.state

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            if other == self.state:
                return True
            if self.state == FAILED and self.reason and other == f"failed({self.reason})":
                return True
            return False
        if isinstance(other, ControllerStatus):
            if self.state != other.state:
                return False
            if self.state == FAILED:
                if self.reason is not None and other.reason is not None and self.reason != other.reason:
                    return False
                if self.step is not None and other.step is not None and self.step != other.step:
                    return False
            return True
        return super().__eq__(other)


def _snapshot_observation(world: World) -> Percept:
    """Capture a full ``Percept`` without clearing ``world._perception_event_log``."""
    was_perceived = world._has_been_perceived
    pending_events = list(world._perception_event_log)
    obs = perceive(world, events=pending_events)
    world._has_been_perceived = was_perceived
    return obs


def _parse_move_target(
    step: Step,
) -> tuple[Literal["object", "position", "direction"], Any]:
    """Classify a ``move`` step into object navigation, position navigation, or directional move."""
    params = step.params or {}

    # 1. Check explicit target field on Step
    if isinstance(step.target, str):
        raw_t = step.target.strip()
        if _OBJECT_ID_RE.fullmatch(raw_t):
            return ("object", raw_t)
        m_toward = _TOWARD_OBJECT_RE.fullmatch(raw_t)
        if m_toward is not None:
            return ("object", m_toward.group(1))
        m_coord = _COORD_STRING_RE.fullmatch(raw_t)
        if m_coord is not None:
            return ("position", Position(x=float(m_coord.group(1)), y=float(m_coord.group(2))))

    # 2. Check object/position navigation keys in step.params
    for key in ("target", "object_id", "to", "position", "location", "destination", "goal"):
        if key not in params or params[key] is None:
            continue
        val = params[key]
        if isinstance(val, WorldObject):
            return ("object", val.id)
        if isinstance(val, str):
            raw_s = val.strip()
            if _OBJECT_ID_RE.fullmatch(raw_s):
                return ("object", raw_s)
            m_toward = _TOWARD_OBJECT_RE.fullmatch(raw_s)
            if m_toward is not None:
                return ("object", m_toward.group(1))
            m_coord = _COORD_STRING_RE.fullmatch(raw_s)
            if m_coord is not None:
                return ("position", Position(x=float(m_coord.group(1)), y=float(m_coord.group(2))))
        try:
            px, py = extract_xy(val)
            return ("position", Position(x=px, y=py))
        except (TypeError, ValueError, KeyError):
            return ("direction", val)

    # 3. Check top-level ("x", "y") in step.params when "direction" is absent
    if "x" in params and "y" in params and "direction" not in params:
        try:
            return ("position", Position(x=float(params["x"]), y=float(params["y"])))
        except (TypeError, ValueError):
            return ("direction", None)

    # 4. Check if params["direction"] references an object id ("object_7" or "toward object_7")
    direction = params.get("direction", step.target)
    if isinstance(direction, WorldObject):
        return ("object", direction.id)
    if isinstance(direction, str):
        raw_d = direction.strip()
        if _OBJECT_ID_RE.fullmatch(raw_d):
            return ("object", raw_d)
        m_toward = _TOWARD_OBJECT_RE.fullmatch(raw_d)
        if m_toward is not None:
            return ("object", m_toward.group(1))

    return ("direction", direction)


def _point_passable(
    world: World,
    wx: float,
    wy: float,
    *,
    from_pos: Position,
    from_level: str,
    target_object_id: str | None = None,
    ignore_movable: bool = False,
    clearance_margin: float = 0.15,
) -> tuple[bool, str]:
    """Return ``(passable, resulting_level)`` for stepping from ``from_pos`` to ``(wx, wy)``."""
    cand_pos = Position(x=wx, y=wy)
    if not world.terrain.can_reach(
        cand_pos,
        from_pos=from_pos,
        from_level=from_level,
        objects=world.objects,
    ):
        return (False, from_level)

    dest_tile = world.terrain.get_tile(cand_pos)
    next_level = "raised" if dest_tile.type == "ledge" else "ground"

    held_id = world.body.holding
    for obj in world.objects.values():
        if obj.id == held_id or obj.id == target_object_id:
            continue
        if obj.level != next_level:
            continue
        if ignore_movable and not obj.fixed:
            continue
        min_clear = obj.radius + clearance_margin
        dist_cand = math.hypot(wx - obj.position.x, wy - obj.position.y)
        if dist_cand < min_clear:
            dist_from = math.hypot(from_pos.x - obj.position.x, from_pos.y - obj.position.y)
            # Allow moving strictly away if the agent started slightly inside the margin
            if dist_cand <= dist_from + 1e-6:
                return (False, next_level)

    return (True, next_level)


def _segment_passable(
    world: World,
    p0: Position,
    level0: str,
    p1: Position,
    *,
    target_object_id: str | None = None,
    ignore_movable: bool = False,
    clearance_margin: float = 0.15,
) -> tuple[bool, str]:
    """Check whether the straight segment ``p0 -> p1`` is free of terrain and object obstacles."""
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    dist = math.hypot(dx, dy)
    if dist <= 1e-9:
        return _point_passable(
            world,
            p1.x,
            p1.y,
            from_pos=p0,
            from_level=level0,
            target_object_id=target_object_id,
            ignore_movable=ignore_movable,
            clearance_margin=clearance_margin,
        )

    steps = max(1, math.ceil(dist / 0.2))
    curr_pos = p0
    curr_level = level0
    for i in range(1, steps + 1):
        t = i / steps
        sx = p0.x + dx * t
        sy = p0.y + dy * t
        ok, next_level = _point_passable(
            world,
            sx,
            sy,
            from_pos=curr_pos,
            from_level=curr_level,
            target_object_id=target_object_id,
            ignore_movable=ignore_movable,
            clearance_margin=clearance_margin,
        )
        if not ok:
            return (False, curr_level)
        curr_pos = Position(x=sx, y=sy)
        curr_level = next_level

    return (True, curr_level)


def _can_reach_object_from(
    world: World,
    pos: Position,
    level: str,
    target_obj: WorldObject,
    *,
    reach: float = DEFAULT_REACH,
) -> bool:
    """Return True if ``target_obj`` would be in reach and unoccluded from ``(pos, level)``."""
    dist = math.hypot(target_obj.position.x - pos.x, target_obj.position.y - pos.y)
    if dist > reach + target_obj.radius - 0.02:
        return False
    if dist < target_obj.radius + 0.10:
        return False

    agent_elev = LEDGE_HEIGHT if (level == "raised" or world.terrain.is_ledge(pos)) else 0.0
    obj_elev = _object_elevation(target_obj, world.objects, world.terrain)
    if target_obj.level == "raised" and level == "ground" and agent_elev < LEDGE_HEIGHT:
        if not world.terrain.can_reach(
            target_obj.position,
            from_pos=pos,
            from_level=level,
            objects=world.objects,
        ):
            return False
    if abs(obj_elev - agent_elev) > reach + 1e-6:
        return False
    if is_occluded(pos, target_obj, world=world):
        return False
    return True


def _astar_search(
    world: World,
    goal_pos: Position,
    *,
    target_obj: WorldObject | None = None,
    ignore_movable: bool = False,
    max_expansions: int = 3500,
) -> tuple[list[Position] | None, bool]:
    """Run A* on a half-tile lattice toward ``goal_pos`` (or within reach of ``target_obj``).

    Returns ``(waypoints, exhausted_search_space)``:
    - If a path is found, returns ``(waypoints, False)``.
    - If the open set becomes empty without reaching the goal (the agent or goal
      is enclosed by obstacles), returns ``(None, True)``.
    - If ``max_expansions`` was hit before exhausting the open set, returns
      ``(None, False)`` so the caller can fall back to greedy steering.
    """
    start_pos = world.body.position
    start_level = world.body.level
    tile_size = world.terrain.tile_size
    grid_step = max(0.25, tile_size * 0.5)
    target_id = target_obj.id if target_obj is not None else None

    # Compute search bounding box including nearby terrain tiles and objects
    margin = max(8.0, 6.0 * tile_size)
    min_x = min(start_pos.x, goal_pos.x) - margin
    max_x = max(start_pos.x, goal_pos.x) + margin
    min_y = min(start_pos.y, goal_pos.y) - margin
    max_y = max(start_pos.y, goal_pos.y) + margin

    for tx, ty in world.terrain._tiles:
        wx = (tx + 0.5) * tile_size
        wy = (ty + 0.5) * tile_size
        if abs(wx - start_pos.x) <= 30.0 and abs(wy - start_pos.y) <= 30.0:
            min_x = min(min_x, wx - 2.0 * tile_size)
            max_x = max(max_x, wx + 2.0 * tile_size)
            min_y = min(min_y, wy - 2.0 * tile_size)
            max_y = max(max_y, wy + 2.0 * tile_size)

    for obj in world.objects.values():
        if abs(obj.position.x - start_pos.x) <= 30.0 and abs(obj.position.y - start_pos.y) <= 30.0:
            min_x = min(min_x, obj.position.x - 2.0 * tile_size)
            max_x = max(max_x, obj.position.x + 2.0 * tile_size)
            min_y = min(min_y, obj.position.y - 2.0 * tile_size)
            max_y = max(max_y, obj.position.y + 2.0 * tile_size)

    def to_node(x: float, y: float) -> tuple[int, int]:
        return (round((x - start_pos.x) / grid_step), round((y - start_pos.y) / grid_step))

    def to_world(nx: int, ny: int) -> Position:
        return Position(x=round(start_pos.x + nx * grid_step, 6), y=round(start_pos.y + ny * grid_step, 6))

    start_node = (0, 0, start_level)
    g_score: dict[tuple[int, int, str], float] = {start_node: 0.0}
    came_from: dict[tuple[int, int, str], tuple[int, int, str]] = {}

    def heuristic(wx: float, wy: float) -> float:
        return math.hypot(goal_pos.x - wx, goal_pos.y - wy)

    counter = 0
    open_heap: list[tuple[float, int, int, int, str]] = [
        (heuristic(start_pos.x, start_pos.y), counter, 0, 0, start_level)
    ]
    closed: set[tuple[int, int, str]] = set()

    neighbors = (
        (1, 0, grid_step),
        (-1, 0, grid_step),
        (0, 1, grid_step),
        (0, -1, grid_step),
        (1, 1, grid_step * math.sqrt(2.0)),
        (1, -1, grid_step * math.sqrt(2.0)),
        (-1, 1, grid_step * math.sqrt(2.0)),
        (-1, -1, grid_step * math.sqrt(2.0)),
    )

    expansions = 0
    while open_heap:
        _, _, nx, ny, lvl = heapq.heappop(open_heap)
        state = (nx, ny, lvl)
        if state in closed:
            continue
        closed.add(state)

        curr_pos = to_world(nx, ny)

        # Check goal condition
        if target_obj is not None:
            if _can_reach_object_from(world, curr_pos, lvl, target_obj):
                path: list[Position] = [curr_pos]
                cur = state
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(to_world(cur[0], cur[1]))
                path.reverse()
                return (path, False)
        else:
            dist_to_goal = math.hypot(goal_pos.x - curr_pos.x, goal_pos.y - curr_pos.y)
            if dist_to_goal <= grid_step * 0.75:
                seg_ok, _ = _segment_passable(
                    world,
                    curr_pos,
                    lvl,
                    goal_pos,
                    target_object_id=None,
                    ignore_movable=ignore_movable,
                )
                if seg_ok:
                    path = [goal_pos]
                    if dist_to_goal > 1e-6:
                        path.append(curr_pos)
                    cur = state
                    while cur in came_from:
                        cur = came_from[cur]
                        path.append(to_world(cur[0], cur[1]))
                    path.reverse()
                    return (path, False)

        expansions += 1
        if expansions >= max_expansions:
            return (None, False)

        base_g = g_score[state]
        for dx, dy, step_cost in neighbors:
            nnx = nx + dx
            nny = ny + dy
            cand_pos = to_world(nnx, nny)
            if cand_pos.x < min_x or cand_pos.x > max_x or cand_pos.y < min_y or cand_pos.y > max_y:
                continue

            seg_ok, next_lvl = _segment_passable(
                world,
                curr_pos,
                lvl,
                cand_pos,
                target_object_id=target_id,
                ignore_movable=ignore_movable,
            )
            if not seg_ok:
                continue

            # Slightly penalize water tiles so the agent prefers dry land when available
            tile_penalty = 0.25 * step_cost if world.terrain.is_water(cand_pos) else 0.0
            tentative_g = base_g + step_cost + tile_penalty
            next_state = (nnx, nny, next_lvl)
            if tentative_g + 1e-9 < g_score.get(next_state, math.inf):
                g_score[next_state] = tentative_g
                came_from[next_state] = state
                counter += 1
                f_score = tentative_g + heuristic(cand_pos.x, cand_pos.y)
                heapq.heappush(open_heap, (f_score, counter, nnx, nny, next_lvl))

    return (None, True)


def _select_next_move_vector(
    world: World,
    goal_pos: Position,
    *,
    max_step_dist: float,
    target_obj: WorldObject | None = None,
) -> tuple[tuple[float, float] | None, float, str | None]:
    """Compute ``(unit_direction, step_distance, failure_reason)`` for one navigation tick."""
    start_pos = world.body.position
    start_level = world.body.level
    target_id = target_obj.id if target_obj is not None else None

    # Check if a target object is on a raised ledge that cannot be reached from ground
    if target_obj is not None and target_obj.level == "raised" and start_level == "ground":
        agent_elev = _agent_elevation(world)
        if agent_elev < LEDGE_HEIGHT and not world.can_reach(target_obj.position):
            # Check if any climbable object exists on ground adjacent to the ledge
            has_climb_support = any(
                o.id != target_obj.id
                and o.level == "ground"
                and o.height >= 0.15
                and world.terrain.can_reach(
                    target_obj.position,
                    from_pos=o.position,
                    from_level=f"on_top_of:{o.id}",
                    objects=world.objects,
                )
                for o in world.objects.values()
            )
            if not has_climb_support:
                return (None, 0.0, OUT_OF_REACH)

    # Check if a target position is on an unreachable ledge tile or inside a fixed obstacle
    if target_obj is None:
        if world.terrain.is_ledge(goal_pos) and start_level == "ground":
            if not world.can_reach(goal_pos):
                return (None, 0.0, OUT_OF_REACH)
        dest_level = "raised" if world.terrain.is_ledge(goal_pos) else "ground"
        for obj in world.objects.values():
            if obj.fixed and obj.level == dest_level:
                if math.hypot(goal_pos.x - obj.position.x, goal_pos.y - obj.position.y) < obj.radius:
                    return (None, 0.0, NO_PATH)

    # Run A* avoiding all non-target objects first, then fall back to avoiding only fixed objects
    path, exhausted = _astar_search(
        world,
        goal_pos,
        target_obj=target_obj,
        ignore_movable=False,
    )
    used_ignore_movable = False
    if path is None:
        path, exhausted = _astar_search(
            world,
            goal_pos,
            target_obj=target_obj,
            ignore_movable=True,
        )
        used_ignore_movable = True

    if path is not None and len(path) >= 1:
        # String-pull along the path to find the furthest visible point within max_step_dist
        # Build dense sample points along the polyline path
        dense_points: list[Position] = [start_pos]
        prev = start_pos
        for wp in path:
            seg_len = math.hypot(wp.x - prev.x, wp.y - prev.y)
            if seg_len <= 1e-6:
                continue
            n_sub = max(1, math.ceil(seg_len / 0.1))
            for i in range(1, n_sub + 1):
                t = i / n_sub
                dense_points.append(
                    Position(
                        x=prev.x + (wp.x - prev.x) * t,
                        y=prev.y + (wp.y - prev.y) * t,
                    )
                )
            prev = wp

        chosen_pt: Position | None = None
        for cand in reversed(dense_points[1:]):
            d = math.hypot(cand.x - start_pos.x, cand.y - start_pos.y)
            if d > max_step_dist + 1e-6 or d <= 1e-6:
                continue
            ok, _ = _segment_passable(
                world,
                start_pos,
                start_level,
                cand,
                target_object_id=target_id,
                ignore_movable=used_ignore_movable,
            )
            if ok and world.can_reach(cand):
                chosen_pt = cand
                break

        if chosen_pt is None and len(path) >= 2:
            # Step toward the first non-start waypoint
            first_wp = path[1]
            dx = first_wp.x - start_pos.x
            dy = first_wp.y - start_pos.y
            d = math.hypot(dx, dy)
            if d > 1e-6:
                step_d = min(max_step_dist, d)
                chosen_pt = Position(
                    x=start_pos.x + (dx / d) * step_d,
                    y=start_pos.y + (dy / d) * step_d,
                )

        if chosen_pt is not None:
            dx = chosen_pt.x - start_pos.x
            dy = chosen_pt.y - start_pos.y
            dist = math.hypot(dx, dy)
            if dist > 1e-6:
                # If approaching a target object, do not step closer than (radius + 0.25)
                if target_obj is not None:
                    min_stop = target_obj.radius + 0.25
                    dist_to_obj = math.hypot(
                        target_obj.position.x - chosen_pt.x,
                        target_obj.position.y - chosen_pt.y,
                    )
                    if dist_to_obj < min_stop:
                        full_to_obj = math.hypot(
                            target_obj.position.x - start_pos.x,
                            target_obj.position.y - start_pos.y,
                        )
                        allowed = max(0.0, full_to_obj - min_stop)
                        dist = min(dist, allowed)
                if dist <= 1e-6:
                    return ((1.0, 0.0), 0.0, None)
                return ((dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)), dist, None)

    if exhausted:
        return (None, 0.0, NO_PATH)

    # Greedy fallback when the target is very far (beyond A* expansion cap)
    base_dx = goal_pos.x - start_pos.x
    base_dy = goal_pos.y - start_pos.y
    base_dist = math.hypot(base_dx, base_dy)
    if base_dist <= 1e-6:
        return ((1.0, 0.0), 0.0, None)

    base_angle = math.atan2(base_dy, base_dx)
    step_d = min(max_step_dist, base_dist)
    if target_obj is not None:
        step_d = min(step_d, max(0.0, base_dist - (target_obj.radius + 0.25)))
    if step_d <= 1e-6:
        return ((1.0, 0.0), 0.0, None)

    for deg_offset in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150):
        ang = base_angle + math.radians(deg_offset)
        ux, uy = math.cos(ang), math.sin(ang)
        cand = Position(x=start_pos.x + ux * step_d, y=start_pos.y + uy * step_d)
        ok, _ = _segment_passable(
            world,
            start_pos,
            start_level,
            cand,
            target_object_id=target_id,
            ignore_movable=False,
        )
        if ok and world.can_reach(cand):
            return ((ux, uy), step_d, None)

    return (None, 0.0, NO_PATH)


class Controller:
    """Scripted low-level controller that executes an ``Intent``'s steps each tick."""

    def __init__(
        self,
        world: World | None = None,
        *,
        intent: Intent | Mapping[str, Any] | None = None,
        modifiers: BodyModifiers | Mapping[str, Any] | None = None,
        max_retries: int = 1,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.world = world
        self.modifiers = modifiers
        self.max_retries: int = max_retries

        self._intent: Intent | None = None
        self._active: bool = False
        self._step_index: int = 0
        self._retries: int = 0
        self._move_ticks_on_step: int = 0
        self._status: ControllerStatus = ControllerStatus.idle()
        self._last_event: IntentEndedEvent | None = None
        self._emitted_events: list[IntentEndedEvent] = []
        self._subscribers: list[Callable[[IntentEndedEvent], None]] = []

        if intent is not None:
            self.set_intent(intent)

    @property
    def intent(self) -> Intent | None:
        """Currently assigned ``Intent`` (or ``None`` if none has been set)."""
        return self._intent

    @property
    def status(self) -> ControllerStatus:
        """Most recent ``ControllerStatus``."""
        return self._status

    @property
    def step_index(self) -> int:
        """Index of the currently executing step within ``intent.steps``."""
        return self._step_index

    @property
    def current_step(self) -> Step | None:
        """The ``Step`` currently being executed, or ``None`` if not running."""
        if self._intent is None or not self._active:
            return None
        if 0 <= self._step_index < len(self._intent.steps):
            return self._intent.steps[self._step_index]
        return None

    @property
    def retries(self) -> int:
        """Number of retries used so far on the current step."""
        return self._retries

    @property
    def retry_count(self) -> int:
        """Alias for ``retries``."""
        return self._retries

    @property
    def is_idle(self) -> bool:
        """True when the controller has no active intent and is resting."""
        return not self._active and self._status.state == IDLE

    @property
    def is_resting(self) -> bool:
        """True when the agent is idling with no active intent (treated as rest by M7)."""
        return not self._active

    @property
    def resting(self) -> bool:
        """Alias for ``is_resting``."""
        return self.is_resting

    @property
    def is_running(self) -> bool:
        """True when an intent is currently executing."""
        return self._active

    @property
    def last_event(self) -> IntentEndedEvent | None:
        """Most recently emitted ``IntentEndedEvent``."""
        return self._last_event

    @property
    def events(self) -> list[IntentEndedEvent]:
        """All ``IntentEndedEvent`` instances emitted by this controller."""
        return list(self._emitted_events)

    def subscribe(
        self,
        handler: Callable[[IntentEndedEvent], None],
    ) -> Callable[[], None]:
        """Subscribe ``handler`` to ``intent_ended`` events from this controller."""
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._subscribers.append(handler)

        def _unsubscribe() -> None:
            if handler in self._subscribers:
                self._subscribers.remove(handler)

        return _unsubscribe

    on_intent_ended = subscribe

    def unsubscribe(self, handler: Callable[[IntentEndedEvent], None]) -> None:
        """Remove a previously subscribed ``intent_ended`` handler."""
        if handler in self._subscribers:
            self._subscribers.remove(handler)

    def clear_intent(self) -> None:
        """Clear any active intent and transition the controller to ``idle``."""
        self._intent = None
        self._active = False
        self._step_index = 0
        self._retries = 0
        self._move_ticks_on_step = 0
        self._status = ControllerStatus.idle()

    cancel_intent = clear_intent

    def set_intent(self, intent: Intent | Mapping[str, Any] | None) -> None:
        """Set a new ``Intent`` to carry out (or ``None`` to idle)."""
        if intent is None:
            self.clear_intent()
            return
        if isinstance(intent, Intent):
            validated = intent
        elif isinstance(intent, Mapping):
            validated = Intent.model_validate(dict(intent))
        else:
            raise TypeError("intent must be an Intent, mapping, or None")

        self._intent = validated
        self._active = True
        self._step_index = 0
        self._retries = 0
        self._move_ticks_on_step = 0
        first_step = validated.steps[0] if validated.steps else None
        self._status = ControllerStatus.running(
            intent=validated,
            step=first_step,
            step_index=0 if first_step is not None else None,
        )

    def _resolve_world(self, world: World | None) -> World:
        if world is not None:
            self.world = world
            _set_active_world(world)
            return world
        if self.world is not None:
            _set_active_world(self.world)
            return self.world
        active = _get_active_world()
        if active is not None:
            self.world = active
            return active
        raise RuntimeError("Controller.tick requires a World instance")

    def _emit_intent_ended(
        self,
        world: World,
        *,
        status: Literal["succeeded", "failed"],
        reason: str | None = None,
        step: Step | None = None,
        step_index: int | None = None,
    ) -> tuple[IntentEndedEvent, Percept]:
        assert self._intent is not None
        observation = _snapshot_observation(world)
        event = IntentEndedEvent(
            tick=world.tick,
            source="controller",
            status=status,
            intent=self._intent,
            reason=reason,
            step=step,
            step_index=step_index,
            observation=observation,
        )
        self._last_event = event
        self._emitted_events.append(event)
        world.events._typed_subscribers.setdefault("intent_ended", [])
        world.events.emit(event)
        for callback in list(self._subscribers):
            callback(event)
        return (event, observation)

    def _execute_move(
        self,
        world: World,
        step: Step,
        mods: BodyModifiers,
    ) -> tuple[Literal["completed", "in_progress", "failed"], str | None]:
        """Execute one tick of a ``move`` step.

        Returns ``("completed", None)``, ``("in_progress", None)``, or ``("failed", reason)``.
        """
        raw_speed = step.params.get("speed", 1.0)
        if isinstance(raw_speed, bool) or not isinstance(raw_speed, int | float):
            return ("failed", INVALID_PARAMS)
        speed_val = float(raw_speed)
        if not math.isfinite(speed_val) or speed_val < 0.0 or speed_val > 1.0:
            return ("failed", INVALID_PARAMS)

        effective_max_step = speed_val * mods.speed * BASE_MOVE_STEP
        if effective_max_step <= 0.0:
            return ("failed", NOTHING_HAPPENED)

        mode, target_val = _parse_move_target(step)

        if mode == "object":
            target_id: str = target_val
            target_obj = world.objects.get(target_id)
            if target_obj is None:
                return ("failed", OUT_OF_REACH)

            dist_before = math.hypot(
                target_obj.position.x - world.body.position.x,
                target_obj.position.y - world.body.position.y,
            )
            min_stop = target_obj.radius + 0.25
            already_in_reach = in_reach(world, target_obj, modifiers=mods) and not is_occluded(
                world.body.position, target_obj, world=world
            )
            if already_in_reach and (self._move_ticks_on_step > 0 or dist_before <= min_stop + 1e-6):
                return ("completed", None)

            unit_dir, step_dist, fail_reason = _select_next_move_vector(
                world,
                target_obj.position,
                max_step_dist=effective_max_step,
                target_obj=target_obj,
            )
            if fail_reason is not None or unit_dir is None:
                return ("failed", fail_reason or NO_PATH)

            if step_dist <= 1e-6:
                if in_reach(world, target_obj, modifiers=mods):
                    return ("completed", None)
                return ("failed", OUT_OF_REACH)

            call_speed = min(speed_val, step_dist / (mods.speed * BASE_MOVE_STEP))
            res = move_verb(world, unit_dir, call_speed, modifiers=mods)
            if not res.ok:
                return ("failed", res.reason or OUT_OF_REACH)

            self._move_ticks_on_step += 1
            refreshed_obj = world.objects.get(target_id, target_obj)
            if in_reach(world, refreshed_obj, modifiers=mods) and not is_occluded(
                world.body.position, refreshed_obj, world=world
            ):
                return ("completed", None)
            return ("in_progress", None)

        if mode == "position":
            goal_pos: Position = target_val
            dist_before = math.hypot(
                goal_pos.x - world.body.position.x,
                goal_pos.y - world.body.position.y,
            )
            if dist_before <= 1e-3:
                return ("completed", None)

            unit_dir, step_dist, fail_reason = _select_next_move_vector(
                world,
                goal_pos,
                max_step_dist=effective_max_step,
                target_obj=None,
            )
            if fail_reason is not None or unit_dir is None:
                return ("failed", fail_reason or NO_PATH)

            if step_dist <= 1e-6:
                return ("completed", None)

            call_speed = min(speed_val, step_dist / (mods.speed * BASE_MOVE_STEP))
            res = move_verb(world, unit_dir, call_speed, modifiers=mods)
            if not res.ok:
                return ("failed", res.reason or OUT_OF_REACH)

            self._move_ticks_on_step += 1
            dist_after = math.hypot(
                goal_pos.x - world.body.position.x,
                goal_pos.y - world.body.position.y,
            )
            if dist_after <= 1e-3:
                return ("completed", None)
            return ("in_progress", None)

        # Single-step directional move
        res = move_verb(world, target_val, speed_val, modifiers=mods)
        if not res.ok:
            return ("failed", res.reason or OUT_OF_REACH)
        self._move_ticks_on_step += 1
        return ("completed", None)

    def _execute_step(
        self,
        world: World,
        step: Step,
        mods: BodyModifiers,
    ) -> tuple[Literal["completed", "in_progress", "failed"], str | None]:
        """Execute a single ``Step`` for the current tick."""
        if step.verb == "move":
            return self._execute_move(world, step, mods)

        res: ActionResult = execute_action(
            world,
            step.verb,
            step.target,
            step.params,
            modifiers=mods,
        )
        if res.ok:
            return ("completed", None)
        return ("failed", res.reason or NOTHING_HAPPENED)

    def tick(
        self,
        world: World | None = None,
        *,
        modifiers: BodyModifiers | Mapping[str, Any] | None = None,
    ) -> ControllerStatus:
        """Execute one tick of the active ``Intent`` (or idle if no intent is active)."""
        resolved_world = self._resolve_world(world)

        if not self._active or self._intent is None:
            self._status = ControllerStatus.idle()
            return self._status

        active_intent = self._intent
        steps = active_intent.steps

        # Empty steps list succeeds immediately
        if not steps or self._step_index >= len(steps):
            self._active = False
            event, obs = self._emit_intent_ended(
                resolved_world,
                status=SUCCEEDED,
            )
            self._status = ControllerStatus.succeeded(
                intent=active_intent,
                observation=obs,
                event=event,
            )
            return self._status

        effective_mods = _resolve_modifiers(
            resolved_world,
            modifiers if modifiers is not None else self.modifiers,
        )
        step = steps[self._step_index]
        outcome, reason = self._execute_step(resolved_world, step, effective_mods)

        if outcome == "in_progress":
            # Step made forward progress (e.g. multi-tick navigation); reset step retry counter
            self._retries = 0
            self._status = ControllerStatus.running(
                intent=active_intent,
                step=step,
                step_index=self._step_index,
            )
            return self._status

        if outcome == "completed":
            self._step_index += 1
            self._retries = 0
            self._move_ticks_on_step = 0
            if self._step_index >= len(steps):
                self._active = False
                event, obs = self._emit_intent_ended(
                    resolved_world,
                    status=SUCCEEDED,
                    step=step,
                    step_index=len(steps) - 1,
                )
                self._status = ControllerStatus.succeeded(
                    intent=active_intent,
                    observation=obs,
                    event=event,
                )
                return self._status

            next_step = steps[self._step_index]
            self._status = ControllerStatus.running(
                intent=active_intent,
                step=next_step,
                step_index=self._step_index,
            )
            return self._status

        # Step failed on this tick: retry at most max_retries times (default 1)
        failure_reason = reason or NOTHING_HAPPENED
        if self._retries < self.max_retries:
            self._retries += 1
            self._status = ControllerStatus.running(
                intent=active_intent,
                step=step,
                step_index=self._step_index,
            )
            return self._status

        # Retries exhausted -> intent fails with the failed step and reason
        failed_idx = self._step_index
        self._active = False
        event, obs = self._emit_intent_ended(
            resolved_world,
            status=FAILED,
            reason=failure_reason,
            step=step,
            step_index=failed_idx,
        )
        self._status = ControllerStatus.failed(
            failure_reason,
            step=step,
            step_index=failed_idx,
            intent=active_intent,
            observation=obs,
            event=event,
        )
        return self._status

    step = tick


__all__ = [
    "FAILED",
    "IDLE",
    "INTENT_ENDED",
    "NO_PATH",
    "OUT_OF_REACH",
    "RUNNING",
    "SUCCEEDED",
    "Controller",
    "ControllerState",
    "ControllerStatus",
    "IntentEndedEvent",
]

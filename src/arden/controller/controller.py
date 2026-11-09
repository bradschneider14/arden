"""Scripted low-level controller for executing multi-step Intents tick-by-tick.

Per ``docs/DESIGN.md`` Section 7.2 and Open Question #3 (recorded in
``docs/decisions.md``), the low-level controller is scripted (not RL):
- ``Controller.set_intent(intent)`` installs a new ``Intent`` (or clears it when
  ``None``).
- ``Controller.tick(world) -> ControllerStatus`` advances execution by one tick,
  returning ``running``, ``succeeded``, ``failed(reason)``, or ``idle``.
- Navigation steps (``move``) walk toward a target object id, position, or
  direction with tile-grid obstacle avoidance (A* with greedy fallback) while
  respecting ``world.body_modifiers``.
- Steps execute in order, retrying a failed step at most once before reporting
  ``failed`` with the failed step and reason.
- When an intent ends (``succeeded`` or ``failed``), an ``IntentEndedEvent``
  (``type="intent_ended"``) is emitted with the final ``Percept`` observation
  for the wake gate and surprise checks.
- With no active intent, the controller idles and the agent rests.
"""

from __future__ import annotations

import heapq
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from arden.actions import verbs as action_verbs
from arden.actions.verbs import (
    BASE_MOVE_STEP,
    DEFAULT_REACH,
    INVALID_PARAMS,
    NO_PATH,
    NOTHING_HAPPENED,
    OUT_OF_REACH,
    ActionResult,
    BodyModifiers,
    in_reach,
)

INTERACTION_REACH: float = DEFAULT_REACH
CLIMB_REACH: float = 1.25
from arden.llm.schemas import Intent, Step
from arden.perception.feed import Percept, perceive
from arden.world.objects import Position, WorldObject
from arden.world.physics import (
    LEDGE_HEIGHT,
    TerrainGrid,
    _get_active_world,
    extract_xy,
    normalize_direction,
)
from arden.world.world import EventBus, IntentEndedEvent, World

ControllerState = Literal["idle", "running", "succeeded", "failed"]

IDLE: Literal["idle"] = "idle"
RUNNING: Literal["running"] = "running"
SUCCEEDED: Literal["succeeded"] = "succeeded"
FAILED: Literal["failed"] = "failed"

_OBJECT_ID_RE = re.compile(r"\b(object_[0-9]+)\b", re.IGNORECASE)
_COORD_PAIR_RE = re.compile(
    r"^\s*\(?\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)?\s*$"
)

DEFAULT_ARRIVAL_TOLERANCE: float = 0.35
DEFAULT_OBJECT_STOP_DISTANCE: float = 0.75
DEFAULT_MAX_RETRIES: int = 1
DEFAULT_SEARCH_PADDING: int = 16
DEFAULT_MAX_ASTAR_NODES: int = 4096


class ControllerStatus:
    """Status returned by ``Controller.tick(world)``.

    Supports both object attribute access (``status.state``, ``status.reason``,
    ``status.step``, ``status.observation``) and equality comparison against
    state strings (``status == "running"``, ``status == "succeeded"``,
    ``status == "failed"``, ``status == "idle"``, ``status == "failed(out_of_reach)"``)
    or other ``ControllerStatus`` instances.
    """

    def __init__(
        self,
        state: ControllerState,
        reason: str | None = None,
        *,
        step: Step | None = None,
        step_index: int | None = None,
        retries: int = 0,
        intent: Intent | None = None,
        observation: Percept | None = None,
        result: ActionResult | None = None,
        event: IntentEndedEvent | None = None,
    ) -> None:
        if state not in (IDLE, RUNNING, SUCCEEDED, FAILED):
            raise ValueError(f"unknown controller status state: {state!r}")
        self.state: ControllerState = state
        self.reason: str | None = reason
        self.step: Step | None = step
        self.step_index: int | None = step_index
        self.retries: int = int(retries)
        self.intent: Intent | None = intent
        self.observation: Percept | None = observation
        self.result: ActionResult | None = result
        self.event: IntentEndedEvent | None = event

    @classmethod
    def idle(
        cls,
        *,
        result: ActionResult | None = None,
        observation: Percept | None = None,
    ) -> ControllerStatus:
        return cls(IDLE, result=result, observation=observation)

    @classmethod
    def running(
        cls,
        *,
        step: Step | None = None,
        step_index: int | None = None,
        retries: int = 0,
        intent: Intent | None = None,
        result: ActionResult | None = None,
        observation: Percept | None = None,
    ) -> ControllerStatus:
        return cls(
            RUNNING,
            step=step,
            step_index=step_index,
            retries=retries,
            intent=intent,
            result=result,
            observation=observation,
        )

    @classmethod
    def succeeded(
        cls,
        *,
        step: Step | None = None,
        step_index: int | None = None,
        retries: int = 0,
        intent: Intent | None = None,
        observation: Percept | None = None,
        result: ActionResult | None = None,
        event: IntentEndedEvent | None = None,
    ) -> ControllerStatus:
        return cls(
            SUCCEEDED,
            step=step,
            step_index=step_index,
            retries=retries,
            intent=intent,
            observation=observation,
            result=result,
            event=event,
        )

    @classmethod
    def failed(
        cls,
        reason: str = NOTHING_HAPPENED,
        *,
        step: Step | None = None,
        step_index: int | None = None,
        retries: int = 0,
        intent: Intent | None = None,
        observation: Percept | None = None,
        result: ActionResult | None = None,
        event: IntentEndedEvent | None = None,
    ) -> ControllerStatus:
        return cls(
            FAILED,
            reason=reason,
            step=step,
            step_index=step_index,
            retries=retries,
            intent=intent,
            observation=observation,
            result=result,
            event=event,
        )

    @property
    def status(self) -> ControllerState:
        """Alias for ``state``."""
        return self.state

    @property
    def kind(self) -> ControllerState:
        """Alias for ``state``."""
        return self.state

    @property
    def value(self) -> ControllerState:
        """Alias for ``state``."""
        return self.state

    @property
    def name(self) -> str:
        """Uppercase name of ``state``."""
        return self.state.upper()

    @property
    def ok(self) -> bool:
        """True unless ``state == 'failed'``."""
        return self.state != FAILED

    @property
    def is_idle(self) -> bool:
        return self.state == IDLE

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
    def done(self) -> bool:
        """True when an intent has finished (``succeeded`` or ``failed``)."""
        return self.state in (SUCCEEDED, FAILED)

    @property
    def reason_code(self) -> str | None:
        """Alias for ``reason``."""
        return self.reason

    @property
    def failed_step(self) -> Step | None:
        """Return the step that caused failure when ``state == 'failed'``."""
        return self.step if self.state == FAILED else None

    @property
    def percept(self) -> Percept | None:
        """Alias for ``observation``."""
        return self.observation

    @property
    def final_observation(self) -> Percept | None:
        """Alias for ``observation``."""
        return self.observation

    def __call__(
        self,
        reason: str | None = None,
        *,
        step: Step | None = None,
        step_index: int | None = None,
        retries: int | None = None,
        intent: Intent | None = None,
        observation: Percept | None = None,
        result: ActionResult | None = None,
        event: IntentEndedEvent | None = None,
    ) -> ControllerStatus:
        """Allow calling ``ControllerStatus.FAILED('out_of_reach')`` as a factory."""
        return ControllerStatus(
            self.state,
            reason=self.reason if reason is None else reason,
            step=self.step if step is None else step,
            step_index=self.step_index if step_index is None else step_index,
            retries=self.retries if retries is None else retries,
            intent=self.intent if intent is None else intent,
            observation=self.observation if observation is None else observation,
            result=self.result if result is None else result,
            event=self.event if event is None else event,
        )

    def __str__(self) -> str:
        if self.state == FAILED and self.reason:
            return f"failed({self.reason})"
        return self.state

    def __repr__(self) -> str:
        parts = [f"state={self.state!r}"]
        if self.reason is not None:
            parts.append(f"reason={self.reason!r}")
        if self.step is not None:
            parts.append(f"step={self.step.verb!r}")
        if self.step_index is not None:
            parts.append(f"step_index={self.step_index!r}")
        return f"ControllerStatus({', '.join(parts)})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ControllerStatus):
            if self.state != other.state:
                return False
            if self.reason is not None and other.reason is not None:
                return self.reason == other.reason
            return True
        if isinstance(other, str):
            norm = other.strip().lower()
            if norm == self.state:
                return True
            if self.state == FAILED and self.reason:
                if norm == f"failed({self.reason.lower()})" or norm == f"failed:{self.reason.lower()}":
                    return True
            return False
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.state)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc


# Convenience status singletons / factories on ControllerStatus
ControllerStatus.IDLE = ControllerStatus(IDLE)  # type: ignore[attr-defined]
ControllerStatus.RUNNING = ControllerStatus(RUNNING)  # type: ignore[attr-defined]
ControllerStatus.SUCCEEDED = ControllerStatus(SUCCEEDED)  # type: ignore[attr-defined]
ControllerStatus.FAILED = ControllerStatus(FAILED)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _MoveTargetSpec:
    """Resolved destination or direction for a ``move`` step."""

    kind: Literal["object", "position", "direction", "invalid"]
    object_id: str | None = None
    position: Position | None = None
    direction: tuple[float, float] | None = None
    speed: float = 1.0
    ticks: int | None = None
    reason: str | None = None


def _coerce_intent(
    intent: Intent | Mapping[str, Any] | Sequence[Step | Mapping[str, Any]] | Step | None,
) -> Intent | None:
    """Coerce an ``Intent``, mapping, or sequence of ``Step`` objects into a validated ``Intent``."""
    if intent is None:
        return None
    if isinstance(intent, Intent):
        return intent
    if isinstance(intent, Step):
        return Intent(
            goal=f"{intent.verb} {intent.target or ''}".strip(),
            steps=[intent],
            wake_if=[],
            expect="step completed",
            thought="hand-built single-step intent",
            confidence=1.0,
        )
    if isinstance(intent, Mapping):
        payload = dict(intent)
        payload.setdefault("goal", "hand-built intent")
        payload.setdefault("steps", [])
        payload.setdefault("wake_if", [])
        payload.setdefault("expect", "intent completed")
        payload.setdefault("thought", "hand-built intent")
        payload.setdefault("confidence", 1.0)
        return Intent.model_validate(payload)
    if isinstance(intent, Sequence) and not isinstance(intent, str | bytes):
        steps = [s if isinstance(s, Step) else Step.model_validate(dict(s)) for s in intent]
        return Intent(
            goal="hand-built step sequence",
            steps=steps,
            wake_if=[],
            expect="steps completed",
            thought="hand-built step sequence",
            confidence=1.0,
        )
    raise TypeError(
        "intent must be an Intent, mapping, sequence of Steps, Step, or None"
    )


def _cell_center(terrain: TerrainGrid, cell: tuple[int, int]) -> Position:
    return Position(
        x=(cell[0] + 0.5) * terrain.tile_size,
        y=(cell[1] + 0.5) * terrain.tile_size,
    )


def _cell_in_grid_bounds(terrain: TerrainGrid, cell: tuple[int, int]) -> bool:
    tx, ty = cell
    if terrain.width is not None and not (0 <= tx < terrain.width):
        return False
    if terrain.height is not None and not (0 <= ty < terrain.height):
        return False
    return True


def _point_in_grid_bounds(terrain: TerrainGrid, pos: Position) -> bool:
    if not (math.isfinite(pos.x) and math.isfinite(pos.y)):
        return False
    if terrain.width is not None:
        max_x = terrain.width * terrain.tile_size
        if not (0.0 <= pos.x <= max_x):
            return False
    if terrain.height is not None:
        max_y = terrain.height * terrain.tile_size
        if not (0.0 <= pos.y <= max_y):
            return False
    return True


def _obstacle_objects_for_navigation(
    world: World,
    *,
    exclude_ids: set[str] | None = None,
    goal_pos: Position | None = None,
) -> list[WorldObject]:
    """Return objects on the agent's plane that act as navigation obstacles."""
    excluded = set(exclude_ids or ())
    if world.body.holding is not None:
        excluded.add(world.body.holding)
    if world.body.level.startswith("on_top_of:"):
        excluded.add(world.body.level.removeprefix("on_top_of:"))

    agent_plane = "raised" if world.body.level == "raised" else "ground"
    start_pos = world.body.position
    obstacles: list[WorldObject] = []

    for obj_id in sorted(world.objects):
        if obj_id in excluded:
            continue
        obj = world.objects[obj_id]
        if obj.level.startswith("inside:") or obj.level.startswith("on_top_of:"):
            continue
        obj_plane = (
            "raised"
            if (obj.level == "raised" or world.terrain.is_ledge(obj.position))
            else "ground"
        )
        if obj_plane != agent_plane:
            continue
        # Do not block the agent's current starting point if it begins overlapping an object
        if math.hypot(obj.position.x - start_pos.x, obj.position.y - start_pos.y) <= obj.radius + 0.05:
            continue
        # If navigating to a raw coordinate where a non-fixed object sits right on the goal,
        # allow approaching it unless it is a fixed obstacle.
        if (
            goal_pos is not None
            and not obj.fixed
            and math.hypot(obj.position.x - goal_pos.x, obj.position.y - goal_pos.y) <= obj.radius + 0.1
        ):
            continue
        obstacles.append(obj)

    return obstacles


def _cell_blocked_by_objects(
    terrain: TerrainGrid,
    cell: tuple[int, int],
    obstacles: Sequence[WorldObject],
) -> bool:
    """Return True if any obstacle object blocks traversal through ``cell``."""
    if not obstacles:
        return False
    center = _cell_center(terrain, cell)
    half = terrain.tile_size * 0.5
    for obj in obstacles:
        if terrain.world_to_cell(obj.position) == cell:
            return True
        # Circle-box overlap check with a small clearance margin inside the tile
        dx = max(0.0, abs(obj.position.x - center.x) - half * 0.7)
        dy = max(0.0, abs(obj.position.y - center.y) - half * 0.7)
        if math.hypot(dx, dy) < obj.radius:
            return True
    return False


def _point_blocked_by_objects(
    pos: Position,
    obstacles: Sequence[WorldObject],
    *,
    clearance: float = 0.05,
) -> bool:
    for obj in obstacles:
        if math.hypot(pos.x - obj.position.x, pos.y - obj.position.y) < obj.radius + clearance:
            return True
    return False


def _can_traverse_cell(
    world: World,
    from_cell: tuple[int, int],
    from_level: str,
    to_cell: tuple[int, int],
    obstacles: Sequence[WorldObject],
    *,
    allow_goal_cell: tuple[int, int] | None = None,
) -> tuple[bool, str]:
    """Return ``(ok, next_level)`` for stepping from ``from_cell`` to ``to_cell``."""
    terrain = world.terrain
    if not _cell_in_grid_bounds(terrain, to_cell):
        return (False, from_level)

    from_center = _cell_center(terrain, from_cell)
    to_center = _cell_center(terrain, to_cell)
    if not terrain.can_reach(
        to_center,
        from_pos=from_center,
        from_level=from_level,
        objects=world.objects,
    ):
        return (False, from_level)

    if to_cell != allow_goal_cell and _cell_blocked_by_objects(terrain, to_cell, obstacles):
        return (False, from_level)

    dest_tile = terrain.get_tile(to_center)
    next_level = "raised" if dest_tile.type == "ledge" else "ground"
    return (True, next_level)


def _segment_is_passable(
    world: World,
    start: Position,
    end: Position,
    obstacles: Sequence[WorldObject],
) -> bool:
    """Sample points along ``start -> end`` to verify terrain and object clearance."""
    dist = math.hypot(end.x - start.x, end.y - start.y)
    if dist <= 1e-9:
        return True
    terrain = world.terrain
    steps = max(2, int(math.ceil(dist / (terrain.tile_size * 0.25))))
    prev_pos = start
    prev_level = world.body.level
    for i in range(1, steps + 1):
        t = i / steps
        sample = Position(
            x=start.x + (end.x - start.x) * t,
            y=start.y + (end.y - start.y) * t,
        )
        if not _point_in_grid_bounds(terrain, sample):
            return False
        if not terrain.can_reach(
            sample,
            from_pos=prev_pos,
            from_level=prev_level,
            objects=world.objects,
        ):
            return False
        if _point_blocked_by_objects(sample, obstacles):
            return False
        tile = terrain.get_tile(sample)
        prev_pos = sample
        prev_level = "raised" if tile.type == "ledge" else "ground"
    return True


def find_grid_path(
    world: World,
    goal_pos: Position,
    *,
    target_object: WorldObject | None = None,
    exclude_ids: set[str] | None = None,
    max_nodes: int = DEFAULT_MAX_ASTAR_NODES,
) -> list[Position] | None:
    """Find a tile-grid A* path of waypoint positions from ``world.body.position`` to ``goal_pos``.

    Returns a list of ``Position`` waypoints (excluding the starting position), or
    ``None`` when no valid path exists.
    """
    terrain = world.terrain
    start_pos = world.body.position
    start_cell = terrain.world_to_cell(start_pos)
    goal_cell = terrain.world_to_cell(goal_pos)

    obstacles = _obstacle_objects_for_navigation(
        world,
        exclude_ids=exclude_ids,
        goal_pos=goal_pos if target_object is None else None,
    )

    # Determine valid goal cells on the grid.
    goal_cells: set[tuple[int, int]] = set()
    if target_object is not None:
        # Any passable cell whose center is within interaction/climb reach of target_object
        # and has an unobstructed reach line to target_object
        reach_limit = min(INTERACTION_REACH, CLIMB_REACH) + target_object.radius
        search_r = max(1, int(math.ceil(reach_limit / terrain.tile_size)))
        for dx in range(-search_r, search_r + 1):
            for dy in range(-search_r, search_r + 1):
                cand = (goal_cell[0] + dx, goal_cell[1] + dy)
                if not _cell_in_grid_bounds(terrain, cand):
                    continue
                center = _cell_center(terrain, cand)
                if math.hypot(
                    center.x - target_object.position.x,
                    center.y - target_object.position.y,
                ) > reach_limit + 1e-6:
                    continue
                if _cell_blocked_by_objects(terrain, cand, obstacles):
                    continue
                if cand != goal_cell:
                    # Do not allow reaching diagonally through a sealed corner of two blocked cardinal neighbors
                    if abs(dx) == 1 and abs(dy) == 1:
                        c1 = (goal_cell[0] + dx, goal_cell[1])
                        c2 = (goal_cell[0], goal_cell[1] + dy)
                        c1_ok = (
                            _cell_in_grid_bounds(terrain, c1)
                            and terrain.can_reach(
                                _cell_center(terrain, c1),
                                from_pos=center,
                                from_level=world.body.level,
                                objects=world.objects,
                            )
                            and not _cell_blocked_by_objects(terrain, c1, obstacles)
                        )
                        c2_ok = (
                            _cell_in_grid_bounds(terrain, c2)
                            and terrain.can_reach(
                                _cell_center(terrain, c2),
                                from_pos=center,
                                from_level=world.body.level,
                                objects=world.objects,
                            )
                            and not _cell_blocked_by_objects(terrain, c2, obstacles)
                        )
                        if not (c1_ok or c2_ok):
                            continue
                    if not _segment_is_passable(
                        world, center, target_object.position, obstacles
                    ):
                        continue
                goal_cells.add(cand)
    else:
        if _cell_in_grid_bounds(terrain, goal_cell) and not _cell_blocked_by_objects(
            terrain, goal_cell, obstacles
        ):
            goal_cells.add(goal_cell)

    if not goal_cells:
        return None

    if start_cell in goal_cells:
        if target_object is None:
            if _segment_is_passable(world, start_pos, goal_pos, obstacles):
                return [goal_pos]
        else:
            if _segment_is_passable(world, start_pos, target_object.position, obstacles):
                return [target_object.position]

    # Compute dynamic search bounds when terrain width/height are unbounded
    xs = [start_cell[0], goal_cell[0]]
    ys = [start_cell[1], goal_cell[1]]
    for tx, ty in terrain._tiles:
        xs.append(tx)
        ys.append(ty)
    for obj in obstacles:
        ox, oy = terrain.world_to_cell(obj.position)
        xs.append(ox)
        ys.append(oy)

    min_x = 0 if terrain.width is not None else min(xs) - DEFAULT_SEARCH_PADDING
    max_x = (
        terrain.width - 1
        if terrain.width is not None
        else max(xs) + DEFAULT_SEARCH_PADDING
    )
    min_y = 0 if terrain.height is not None else min(ys) - DEFAULT_SEARCH_PADDING
    max_y = (
        terrain.height - 1
        if terrain.height is not None
        else max(ys) + DEFAULT_SEARCH_PADDING
    )

    def _in_search_box(c: tuple[int, int]) -> bool:
        return min_x <= c[0] <= max_x and min_y <= c[1] <= max_y

    def _heuristic(c: tuple[int, int]) -> float:
        return math.hypot(c[0] - goal_cell[0], c[1] - goal_cell[1])

    start_state = (start_cell[0], start_cell[1], world.body.level)
    open_heap: list[tuple[float, int, tuple[int, int, str]]] = [
        (_heuristic(start_cell), 0, start_state)
    ]
    g_score: dict[tuple[int, int, str], float] = {start_state: 0.0}
    came_from: dict[tuple[int, int, str], tuple[int, int, str]] = {}
    closed: set[tuple[int, int, str]] = set()
    push_count = 1
    expanded = 0

    neighbor_offsets = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    found_state: tuple[int, int, str] | None = None

    while open_heap and expanded < max_nodes:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1

        cx, cy, clevel = current
        curr_cell = (cx, cy)
        if curr_cell in goal_cells:
            # Verify the goal cell can reach the target level if target is on a ledge
            if target_object is not None:
                curr_center = _cell_center(terrain, curr_cell)
                if (
                    target_object.level == "raised"
                    or terrain.is_ledge(target_object.position)
                ) and clevel == "ground":
                    if not terrain.can_reach(
                        target_object.position,
                        from_pos=curr_center,
                        from_level=clevel,
                        objects=world.objects,
                    ):
                        continue
            elif terrain.is_ledge(goal_pos) and clevel == "ground":
                curr_center = _cell_center(terrain, curr_cell)
                if not terrain.can_reach(
                    goal_pos,
                    from_pos=curr_center,
                    from_level=clevel,
                    objects=world.objects,
                ):
                    continue
            found_state = current
            break

        for dx, dy in neighbor_offsets:
            nx, ny = cx + dx, cy + dy
            next_cell = (nx, ny)
            if not _in_search_box(next_cell):
                continue

            # Prevent diagonal corner-cutting between blocked cardinal cells
            if dx != 0 and dy != 0:
                ok_x, _ = _can_traverse_cell(
                    world, curr_cell, clevel, (cx + dx, cy), obstacles
                )
                ok_y, _ = _can_traverse_cell(
                    world, curr_cell, clevel, (cx, cy + dy), obstacles
                )
                if not (ok_x and ok_y):
                    continue

            ok, next_level = _can_traverse_cell(
                world,
                curr_cell,
                clevel,
                next_cell,
                obstacles,
            )
            if not ok:
                continue

            step_cost = math.hypot(dx, dy)
            # Slight preference against water tiles when dry land is available
            if terrain.get_tile(_cell_center(terrain, next_cell)).type == "water":
                step_cost += 0.15

            next_state = (nx, ny, next_level)
            tentative_g = g_score[current] + step_cost
            if tentative_g + 1e-9 < g_score.get(next_state, math.inf):
                came_from[next_state] = current
                g_score[next_state] = tentative_g
                f_score = tentative_g + _heuristic(next_cell)
                push_count += 1
                heapq.heappush(open_heap, (f_score, push_count, next_state))

    if found_state is None:
        return None

    cell_path: list[tuple[int, int]] = []
    node: tuple[int, int, str] | None = found_state
    while node is not None:
        cell_path.append((node[0], node[1]))
        node = came_from.get(node)
    cell_path.reverse()

    waypoints: list[Position] = [
        _cell_center(terrain, cell) for cell in cell_path[1:]
    ]
    final_cell = cell_path[-1]
    if target_object is None:
        if final_cell == goal_cell:
            if waypoints:
                waypoints[-1] = goal_pos
            else:
                waypoints.append(goal_pos)
    else:
        if final_cell == goal_cell and not _point_blocked_by_objects(
            target_object.position, obstacles
        ):
            if waypoints:
                waypoints[-1] = target_object.position
            else:
                waypoints.append(target_object.position)
        elif not waypoints:
            waypoints.append(_cell_center(terrain, final_cell))

    return waypoints


def _greedy_avoidance_direction(
    world: World,
    desired_dir: tuple[float, float],
    step_dist: float,
    obstacles: Sequence[WorldObject],
) -> tuple[float, float] | None:
    """Fallback steering: try ``desired_dir`` first, then angular offsets up to +/-120 deg."""
    ux, uy = desired_dir
    if ux == 0.0 and uy == 0.0:
        return None
    base_angle = math.atan2(uy, ux)
    offsets_deg = (0.0, 30.0, -30.0, 45.0, -45.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0)
    start = world.body.position
    probe_dist = max(step_dist, world.terrain.tile_size * 0.5)

    for deg in offsets_deg:
        rad = base_angle + math.radians(deg)
        cand_ux = math.cos(rad)
        cand_uy = math.sin(rad)
        dest = Position(x=start.x + cand_ux * step_dist, y=start.y + cand_uy * step_dist)
        probe = Position(x=start.x + cand_ux * probe_dist, y=start.y + cand_uy * probe_dist)
        if not _point_in_grid_bounds(world.terrain, dest):
            continue
        if not world.can_reach(dest):
            continue
        if _point_blocked_by_objects(dest, obstacles) or _point_blocked_by_objects(probe, obstacles):
            continue
        if _segment_is_passable(world, start, dest, obstacles):
            return (cand_ux, cand_uy)
    return None


def _extract_position_value(value: Any) -> Position | None:
    """Try to parse a Position from a Position, mapping, (x, y) sequence, or 'x, y' string."""
    if value is None:
        return None
    if isinstance(value, Position):
        return value
    if isinstance(value, Mapping):
        if "x" in value and "y" in value:
            try:
                return Position(x=float(value["x"]), y=float(value["y"]))
            except (TypeError, ValueError):
                return None
        if "position" in value:
            return _extract_position_value(value["position"])
    if isinstance(value, tuple | list) and len(value) == 2:
        try:
            return Position(x=float(value[0]), y=float(value[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        m = _COORD_PAIR_RE.match(value)
        if m is not None:
            try:
                return Position(x=float(m.group(1)), y=float(m.group(2)))
            except (TypeError, ValueError):
                return None
    return None


def _resolve_move_target(step: Step, world: World) -> _MoveTargetSpec:
    """Inspect ``step.target`` and ``step.params`` to determine how ``move`` should navigate."""
    params = dict(step.params or {})
    raw_speed = params.get("speed", 1.0)
    try:
        speed = float(raw_speed)
    except (TypeError, ValueError):
        return _MoveTargetSpec(kind="invalid", reason=INVALID_PARAMS)
    if not math.isfinite(speed) or speed <= 0.0 or speed > 1.0:
        return _MoveTargetSpec(kind="invalid", reason=INVALID_PARAMS)

    raw_ticks = params.get("ticks", params.get("steps", params.get("duration")))
    ticks: int | None = None
    if raw_ticks is not None:
        try:
            ticks = max(1, int(raw_ticks))
        except (TypeError, ValueError):
            ticks = None

    # 1. Check for an object_<n> reference in step.target or step.params
    candidates_for_obj: list[Any] = [
        step.target,
        params.get("target"),
        params.get("object"),
        params.get("object_id"),
        params.get("to"),
        params.get("destination"),
        params.get("goal"),
        params.get("direction"),
    ]
    for cand in candidates_for_obj:
        if isinstance(cand, WorldObject):
            return _MoveTargetSpec(
                kind="object", object_id=cand.id, speed=speed, ticks=ticks
            )
        if isinstance(cand, str):
            match = _OBJECT_ID_RE.search(cand)
            if match is not None:
                return _MoveTargetSpec(
                    kind="object",
                    object_id=match.group(1).lower(),
                    speed=speed,
                    ticks=ticks,
                )

    # 2. Check for explicit target coordinates in params or step.target
    if "x" in params and "y" in params:
        pos = _extract_position_value({"x": params["x"], "y": params["y"]})
        if pos is not None:
            return _MoveTargetSpec(kind="position", position=pos, speed=speed, ticks=ticks)

    for key in ("position", "pos", "to", "target", "destination", "location", "goal", "point"):
        if key in params:
            pos = _extract_position_value(params[key])
            if pos is not None:
                return _MoveTargetSpec(
                    kind="position", position=pos, speed=speed, ticks=ticks
                )

    if step.target is not None:
        pos = _extract_position_value(step.target)
        if pos is not None:
            return _MoveTargetSpec(kind="position", position=pos, speed=speed, ticks=ticks)

    # 3. Check for a pure direction (plus optional distance)
    raw_dir = params.get("direction", step.target)
    if raw_dir is not None:
        # Maybe raw_dir is a coordinate pair only when passed via position keys above;
        # here treat it as a direction vector / cardinal string / angle.
        try:
            ux, uy = normalize_direction(raw_dir)
        except (TypeError, ValueError):
            pos = _extract_position_value(raw_dir)
            if pos is not None:
                return _MoveTargetSpec(
                    kind="position", position=pos, speed=speed, ticks=ticks
                )
            return _MoveTargetSpec(kind="invalid", reason=INVALID_PARAMS)
        if ux == 0.0 and uy == 0.0:
            return _MoveTargetSpec(kind="invalid", reason=NOTHING_HAPPENED)
        raw_distance = params.get("distance")
        if raw_distance is not None:
            try:
                dist_val = float(raw_distance)
            except (TypeError, ValueError):
                return _MoveTargetSpec(kind="invalid", reason=INVALID_PARAMS)
            if not math.isfinite(dist_val) or dist_val <= 0.0:
                return _MoveTargetSpec(kind="invalid", reason=INVALID_PARAMS)
            dest = Position(
                x=world.body.position.x + ux * dist_val,
                y=world.body.position.y + uy * dist_val,
            )
            return _MoveTargetSpec(kind="position", position=dest, speed=speed, ticks=ticks)
        return _MoveTargetSpec(
            kind="direction", direction=(ux, uy), speed=speed, ticks=ticks
        )

    return _MoveTargetSpec(kind="invalid", reason=INVALID_PARAMS)


class Controller:
    """Scripted low-level intent controller.

    Executes an ``Intent``'s ``steps`` in order across ticks, retrying a failed
    step at most once before failing the intent and emitting ``IntentEndedEvent``.
    When no intent is active, ``tick(world)`` idles (calling ``rest``).
    """

    def __init__(
        self,
        intent: Intent | Mapping[str, Any] | Sequence[Step | Mapping[str, Any]] | Step | None = None,
        *,
        world: World | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        arrival_tolerance: float = DEFAULT_ARRIVAL_TOLERANCE,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if arrival_tolerance <= 0.0 or not math.isfinite(arrival_tolerance):
            raise ValueError("arrival_tolerance must be a positive finite float")

        self.world: World | None = world
        self.max_retries: int = int(max_retries)
        self.arrival_tolerance: float = float(arrival_tolerance)
        self.events: EventBus = EventBus()

        self._intent: Intent | None = None
        self._step_index: int = 0
        self._step_retries: int = 0
        self._move_ticks_For_step: int = 0
        self._waypoints: list[Position] | None = None
        self._last_status: ControllerStatus = ControllerStatus.idle()
        self._last_event: IntentEndedEvent | None = None
        self.event_history: list[IntentEndedEvent] = []

        if intent is not None:
            self.set_intent(intent)

    @property
    def intent(self) -> Intent | None:
        """Currently active ``Intent``, or ``None`` when idle."""
        return self._intent

    @property
    def current_intent(self) -> Intent | None:
        """Alias for ``intent``."""
        return self._intent

    @property
    def step_index(self) -> int:
        """Index of the currently executing step within ``intent.steps``."""
        return self._step_index

    @property
    def current_step(self) -> Step | None:
        """Currently executing ``Step``, or ``None`` if no intent is active."""
        if self._intent is None:
            return None
        if 0 <= self._step_index < len(self._intent.steps):
            return self._intent.steps[self._step_index]
        return None

    @property
    def retries(self) -> int:
        """Number of retries used so far on the current step."""
        return self._step_retries

    @property
    def status(self) -> ControllerStatus:
        """Most recent ``ControllerStatus``."""
        return self._last_status

    @property
    def last_event(self) -> IntentEndedEvent | None:
        """Most recent ``IntentEndedEvent`` emitted by the controller."""
        return self._last_event

    @property
    def is_idle(self) -> bool:
        return self._intent is None

    def subscribe(
        self,
        handler_or_type: Callable[[ Any], None] | str,
        handler: Callable[[Any], None] | None = None,
        *,
        event_type: str | None = None,
    ) -> Callable[[], None]:
        """Subscribe to controller events (such as ``intent_ended``)."""
        return self.events.subscribe(handler_or_type, handler, event_type=event_type)  # type: ignore[arg-type]

    def set_intent(
        self,
        intent: Intent | Mapping[str, Any] | Sequence[Step | Mapping[str, Any]] | Step | None,
    ) -> Intent | None:
        """Set or replace the active ``Intent`` (or clear to idle when ``None``)."""
        coerced = _coerce_intent(intent)
        self._intent = coerced
        self._step_index = 0
        self._step_retries = 0
        self._move_ticks_For_step = 0
        self._waypoints = None
        if coerced is None:
            self._last_status = ControllerStatus.idle()
        else:
            step = coerced.steps[0] if coerced.steps else None
            self._last_status = ControllerStatus.running(
                step=step,
                step_index=0 if step is not None else None,
                retries=0,
                intent=coerced,
            )
        return self._intent

    def clear_intent(self) -> None:
        """Clear any active intent and return the controller to ``idle``."""
        self.set_intent(None)

    def interrupt(
        self,
        world: World | None = None,
        *,
        reason: str = "interrupted",
    ) -> ControllerStatus:
        """Interrupt the active intent, if any, and return to ``idle``."""
        if self._intent is None:
            self._last_status = ControllerStatus.idle()
            return self._last_status
        resolved_world = self._resolve_world(world)
        step = self.current_step
        idx = self._step_index
        retries = self._step_retries
        active_intent = self._intent
        self._intent = None
        self._step_index = 0
        self._step_retries = 0
        self._move_ticks_For_step = 0
        self._waypoints = None
        obs = perceive(resolved_world) if resolved_world is not None else None
        event: IntentEndedEvent | None = None
        if resolved_world is not None:
            event = self._emit_intent_ended(
                resolved_world,
                status="failed",
                intent=active_intent,
                reason=reason,
                step=step,
                step_index=idx,
                retries=retries,
                observation=obs,
                result= None,
            )
        self._last_status = ControllerStatus.failed(
            reason,
            step=step,
            step_index=idx,
            retries=retries,
            intent=active_intent,
            observation=obs,
            event=event,
        )
        return self._last_status

    def _resolve_world(self, world: World | None) -> World | None:
        if world is not None:
            self.world = world
            return world
        if self.world is not None:
            return self.world
        return _get_active_world()

    def _emit_intent_ended(
        self,
        world: World,
        *,
        status: Literal["succeeded", "failed"],
        intent: Intent,
        reason: str | None,
        step: Step | None,
        step_index: int | None,
        retries: int,
        observation: Percept,
        result: ActionResult | None,
    ) -> IntentEndedEvent:
        event = IntentEndedEvent(
            tick=world.tick,
            source="controller",
            status=status,
            ok=(status == "succeeded"),
            intent=intent,
            goal=intent.goal,
            expect=intent.expect,
            confidence=intent.confidence,
            reason=reason,
            step=step,
            step_index=step_index,
            retries=retries,
            observation=observation,
            result=result,
        )
        self._last_event = event
        self.event_history.append(event)
        self.events.emit(event)
        world.events.emit(event)
        return event

    def _emit_action_result(
        self,
        world: World,
        *,
        step: Step,
        result: ActionResult,
    ) -> None:
        from arden.world.world import ActionResultEvent

        target_str = step.target if isinstance(step.target, str) else None
        params_dict = dict(step.params or {})
        obj_id = (
            target_str
            if target_str is not None and _OBJECT_ID_RE.fullmatch(target_str)
            else (result.id if result.id and _OBJECT_ID_RE.fullmatch(result.id) else None)
        )
        clean_data: dict[str, Any] = {}
        for k, v in (result.data or {}).items():
            if hasattr(v, "model_dump"):
                clean_data[k] = v.model_dump(mode="json")
            elif isinstance(v, list):
                clean_data[k] = [
                    x.model_dump(mode="json") if hasattr(x, "model_dump") else x
                    for x in v
                ]
            else:
                clean_data[k] = v
        clean_events = [
            ev.model_dump(mode="json") if hasattr(ev, "model_dump") else ev
            for ev in (result.events or [])
        ]
        ev = ActionResultEvent(
            tick=world.tick,
            source="agent",
            verb=step.verb,
            target=target_str,
            object_id=obj_id,
            params=params_dict,
            ok=result.ok,
            reason=result.reason,
            events=clean_events,
            data=clean_data,
            result=result,
        )
        self.events.emit(ev)
        world.events.emit(ev)

    def _finish_intent(
        self,
        world: World,
        *,
        status: Literal["succeeded", "failed"],
        reason: str | None = None,
        step: Step | None = None,
        step_index: int | None = None,
        retries: int = 0,
        result: ActionResult | None = None,
    ) -> ControllerStatus:
        assert self._intent is not None
        finished_intent = self._intent
        obs = perceive(world, _emit_event=False)
        event = self._emit_intent_ended(
            world,
            status=status,
            intent=finished_intent,
            reason=reason,
            step=step,
            step_index=step_index,
            retries=retries,
            observation=obs,
            result=result,
        )
        self._intent = None
        self._step_index = 0
        self._step_retries = 0
        self._move_ticks_For_step = 0
        self._waypoints = None

        if status == "succeeded":
            self._last_status = ControllerStatus.succeeded(
                step=step,
                step_index=step_index,
                retries=retries,
                intent=finished_intent,
                observation=obs,
                result=result,
                event=event,
            )
        else:
            self._last_status = ControllerStatus.failed(
                reason or NOTHING_HAPPENED,
                step=step,
                step_index=step_index,
                retries=retries,
                intent=finished_intent,
                observation=obs,
                result=result,
                event=event,
            )
        return self._last_status

    def _handle_step_failure(
        self,
        world: World,
        step: Step,
        reason: str,
        *,
        result: ActionResult | None = None,
    ) -> ControllerStatus:
        """Retry a failed step at most ``self.max_retries`` times, then fail the intent."""
        if result is not None and not getattr(self, "_action_emitted_for_tick", False):
            self._action_emitted_for_tick = True
            self._emit_action_result(world, step=step, result=result)
        self._waypoints = None
        if self._step_retries < self.max_retries:
            self._step_retries += 1
            self._last_status = ControllerStatus.running(
                step=step,
                step_index=self._step_index,
                retries=self._step_retries,
                intent=self._intent,
                result=result,
            )
            return self._last_status

        return self._finish_intent(
            world,
            status="failed",
            reason=reason,
            step=step,
            step_index=self._step_index,
            retries=self._step_retries,
            result=result,
        )

    def _advance_step(
        self,
        world: World,
        completed_step: Step,
        *,
        result: ActionResult | None = None,
    ) -> ControllerStatus:
        """Advance to the next step or finish with ``succeeded`` when all steps complete."""
        if result is not None and not getattr(self, "_action_emitted_for_tick", False):
            self._action_emitted_for_tick = True
            self._emit_action_result(world, step=completed_step, result=result)
        assert self._intent is not None
        completed_idx = self._step_index
        completed_retries = self._step_retries
        self._step_index += 1
        self._step_retries = 0
        self._move_ticks_For_step = 0
        self._waypoints = None

        if self._step_index >= len(self._intent.steps):
            return self._finish_intent(
                world,
                status="succeeded",
                step=completed_step,
                step_index=completed_idx,
                retries=completed_retries,
                result=result,
            )

        next_step = self._intent.steps[self._step_index]
        self._last_status = ControllerStatus.running(
            step=next_step,
            step_index=self._step_index,
            retries=0,
            intent=self._intent,
            result=result,
        )
        return self._last_status

    def tick(self, world: World | None = None) -> ControllerStatus:
        """Execute one tick of the active intent, or rest if idle."""
        resolved_world = self._resolve_world(world)
        if resolved_world is None:
            raise RuntimeError("Controller.tick requires a World instance")

        if self._intent is None:
            setattr(resolved_world, "_in_controller_tick", True)
            try:
                rest_res = action_verbs.rest(resolved_world, 1)
            finally:
                setattr(resolved_world, "_in_controller_tick", False)
            self._last_status = ControllerStatus.idle(result=rest_res)
            return self._last_status

        if not self._intent.steps or self._step_index >= len(self._intent.steps):
            return self._finish_intent(resolved_world, status="succeeded")

        step = self._intent.steps[self._step_index]
        verb_name = step.verb.strip().lower()

        self._action_emitted_for_tick = False
        setattr(resolved_world, "_in_controller_tick", True)
        try:
            if verb_name == "move":
                status_out = self._tick_move_step(resolved_world, step)
                if status_out.result is not None and not self._action_emitted_for_tick:
                    self._action_emitted_for_tick = True
                    self._emit_action_result(resolved_world, step=step, result=status_out.result)
                return status_out

            res = self._execute_action_step(resolved_world, step)
            if not self._action_emitted_for_tick:
                self._action_emitted_for_tick = True
                self._emit_action_result(resolved_world, step=step, result=res)
        finally:
            setattr(resolved_world, "_in_controller_tick", False)

        if res.ok:
            return self._advance_step(resolved_world, step, result=res)

        reason = res.reason or NOTHING_HAPPENED
        return self._handle_step_failure(resolved_world, step, reason, result=res)

    step = tick

    def _object_arrived(
        self,
        world: World,
        target_obj: WorldObject,
        mods: BodyModifiers,
        *,
        stop_margin: float = DEFAULT_OBJECT_STOP_DISTANCE,
    ) -> bool:
        """Return True if the agent has arrived within comfortable interaction distance of ``target_obj``."""
        if not in_reach(world, target_obj, modifiers=mods):
            return False
        dist = math.hypot(
            target_obj.position.x - world.body.position.x,
            target_obj.position.y - world.body.position.y,
        )
        max_stop = min(INTERACTION_REACH, CLIMB_REACH) + target_obj.radius
        desired_stop = min(max_stop, target_obj.radius + stop_margin)
        return dist <= desired_stop + 1e-6

    def _tick_move_step(self, world: World, step: Step) -> ControllerStatus:
        """Execute one tick of a ``move`` step with tile-grid navigation and obstacle avoidance."""
        mods = world.body_modifiers
        spec = _resolve_move_target(step, world)
        if spec.kind == "invalid":
            reason = spec.reason or INVALID_PARAMS
            return self._handle_step_failure(
                world,
                step,
                reason,
                result=ActionResult(ok=False, reason=reason),
            )

        # Case 1: Pure direction step (no target object or target position)
        if spec.kind == "direction":
            assert spec.direction is not None
            obstacles = _obstacle_objects_for_navigation(world)
            step_dist = spec.speed * mods.speed * BASE_MOVE_STEP
            steer = _greedy_avoidance_direction(
                world, spec.direction, step_dist, obstacles
            )
            chosen_dir = steer if steer is not None else spec.direction
            res = action_verbs.move(world, chosen_dir, spec.speed, modifiers=mods)
            if not res.ok:
                return self._handle_step_failure(
                    world,
                    step,
                    res.reason or OUT_OF_REACH,
                    result=res,
                )
            self._move_ticks_For_step += 1
            if spec.ticks is None or self._move_ticks_For_step >= spec.ticks:
                return self._advance_step(world, step, result=res)
            self._step_retries = 0
            self._last_status = ControllerStatus.running(
                step=step,
                step_index=self._step_index,
                retries=0,
                intent=self._intent,
                result=res,
            )
            return self._last_status

        # Case 2: Target object or target position
        target_obj: WorldObject | None = None
        exclude_ids: set[str] = set()
        if spec.kind == "object":
            assert spec.object_id is not None
            target_obj = world.objects.get(spec.object_id)
            if target_obj is None:
                return self._handle_step_failure(
                    world,
                    step,
                    OUT_OF_REACH,
                    result=ActionResult(
                        ok=False, reason=OUT_OF_REACH, data={"target": spec.object_id}
                    ),
                )
            goal_pos = target_obj.position
            exclude_ids.add(target_obj.id)

            # Check if target object is on an unreachable raised ledge
            agent_elev = (
                LEDGE_HEIGHT if world.body.level == "raised" else 0.0
            )
            if (
                (target_obj.level == "raised" or world.terrain.is_ledge(goal_pos))
                and world.body.level == "ground"
                and agent_elev < LEDGE_HEIGHT
                and not world.can_reach(goal_pos)
            ):
                return self._handle_step_failure(
                    world,
                    step,
                    OUT_OF_REACH,
                    result=ActionResult(
                        ok=False, reason=OUT_OF_REACH, data={"target": spec.object_id}
                    ),
                )
        else:
            assert spec.position is not None
            goal_pos = spec.position
            if not _point_in_grid_bounds(world.terrain, goal_pos):
                return self._handle_step_failure(
                    world,
                    step,
                    OUT_OF_REACH,
                    result=ActionResult(ok=False, reason=OUT_OF_REACH),
                )
            if world.terrain.is_ledge(goal_pos) and not world.can_reach(goal_pos):
                return self._handle_step_failure(
                    world,
                    step,
                    OUT_OF_REACH,
                    result=ActionResult(ok=False, reason=OUT_OF_REACH),
                )
            # If the next step in the intent targets an object at goal_pos, exclude it from obstacles
            if self._intent is not None and self._step_index + 1 < len(self._intent.steps):
                next_target = self._intent.steps[self._step_index + 1].target
                if isinstance(next_target, str) and next_target in world.objects:
                    exclude_ids.add(next_target)

        # Check if already arrived before moving
        if target_obj is not None:
            if self._object_arrived(world, target_obj, mods):
                return self._advance_step(
                    world,
                    step,
                    result=ActionResult(
                        ok=True,
                        data={
                            "target": target_obj.id,
                            "position": {
                                "x": world.body.position.x,
                                "y": world.body.position.y,
                            },
                        },
                    ),
                )
        else:
            dist_to_goal = math.hypot(
                goal_pos.x - world.body.position.x,
                goal_pos.y - world.body.position.y,
            )
            if dist_to_goal <= self.arrival_tolerance:
                return self._advance_step(
                    world,
                    step,
                    result=ActionResult(
                        ok=True,
                        data={
                            "position": {
                                "x": world.body.position.x,
                                "y": world.body.position.y,
                            }
                        },
                    ),
                )

        # Compute or refresh A* waypoints
        waypoints = find_grid_path(
            world,
            goal_pos,
            target_object=target_obj,
            exclude_ids=exclude_ids,
        )
        obstacles = _obstacle_objects_for_navigation(
            world,
            exclude_ids=exclude_ids,
            goal_pos=goal_pos if target_obj is None else None,
        )

        max_step_dist = spec.speed * mods.speed * BASE_MOVE_STEP
        if max_step_dist <= 1e-9:
            return self._handle_step_failure(
                world,
                step,
                NOTHING_HAPPENED,
                result=ActionResult(ok=False, reason=NOTHING_HAPPENED),
            )

        move_dir: tuple[float, float] | None = None
        step_speed = spec.speed

        if waypoints:
            # Drop any leading waypoints the agent has already reached
            while len(waypoints) > 1 and (
                math.hypot(
                    waypoints[0].x - world.body.position.x,
                    waypoints[0].y - world.body.position.y,
                )
                <= min(self.arrival_tolerance, max_step_dist * 0.5)
            ):
                waypoints.pop(0)

            # Smooth to the next waypoint if the direct segment is clear
            if len(waypoints) >= 2 and _segment_is_passable(
                world, world.body.position, waypoints[1], obstacles
            ):
                waypoints.pop(0)

            wp = waypoints[0]
            dx = wp.x - world.body.position.x
            dy = wp.y - world.body.position.y
            wp_dist = math.hypot(dx, dy)

            if wp_dist > 1e-9:
                ux, uy = dx / wp_dist, dy / wp_dist
                # If approaching the final waypoint (or target object), scale speed to avoid overshooting
                if len(waypoints) == 1:
                    if target_obj is not None:
                        desired_stop = target_obj.radius + DEFAULT_OBJECT_STOP_DISTANCE * 0.6
                        rem = max(0.05, wp_dist - desired_stop) if wp == target_obj.position else wp_dist
                    else:
                        rem = wp_dist
                    if rem < max_step_dist and mods.speed * BASE_MOVE_STEP > 1e-9:
                        step_speed = max(
                            1e-4,
                            min(spec.speed, rem / (mods.speed * BASE_MOVE_STEP)),
                        )
                elif wp_dist < max_step_dist and mods.speed * BASE_MOVE_STEP > 1e-9:
                    step_speed = max(
                        1e-4,
                        min(spec.speed, wp_dist / (mods.speed * BASE_MOVE_STEP)),
                    )

                dest_probe = Position(
                    x=world.body.position.x + ux * (step_speed * mods.speed * BASE_MOVE_STEP),
                    y=world.body.position.y + uy * (step_speed * mods.speed * BASE_MOVE_STEP),
                )
                if world.can_reach(dest_probe) and not _point_blocked_by_objects(
                    dest_probe, obstacles
                ):
                    move_dir = (ux, uy)
                else:
                    move_dir = _greedy_avoidance_direction(
                        world, (ux, uy), step_speed * mods.speed * BASE_MOVE_STEP, obstacles
                    )
        else:
            # If A* found no path on the grid, fail with no_path.
            return self._handle_step_failure(
                world,
                step,
                NO_PATH,
                result=ActionResult(
                    ok=False,
                    reason=NO_PATH,
                    data={"target": target_obj.id} if target_obj is not None else {},
                ),
            )

        if move_dir is None:
            return self._handle_step_failure(
                world,
                step,
                NO_PATH,
                result=ActionResult(
                    ok=False,
                    reason=NO_PATH,
                    data={"target": target_obj.id} if target_obj is not None else {},
                ),
            )

        res = action_verbs.move(world, move_dir, step_speed, modifiers=mods)
        if not res.ok:
            return self._handle_step_failure(
                world,
                step,
                res.reason or OUT_OF_REACH,
                result=res,
            )

        # Progress succeeded on this tick -> reset step retry counter
        self._step_retries = 0
        self._move_ticks_For_step += 1

        # Check if we have now arrived at the destination
        if target_obj is not None:
            target_now = world.objects.get(target_obj.id, target_obj)
            if self._object_arrived(world, target_now, mods) or (
                len(waypoints) == 1
                and math.hypot(
                    waypoints[0].x - world.body.position.x,
                    waypoints[0].y - world.body.position.y,
                )
                <= self.arrival_tolerance
                and in_reach(world, target_now, modifiers=mods)
            ):
                return self._advance_step(world, step, result=res)
        else:
            if (
                math.hypot(
                    goal_pos.x - world.body.position.x,
                    goal_pos.y - world.body.position.y,
                )
                <= self.arrival_tolerance
            ):
                return self._advance_step(world, step, result=res)

        self._last_status = ControllerStatus.running(
            step=step,
            step_index=self._step_index,
            retries=0,
            intent=self._intent,
            result=res,
        )
        return self._last_status

    def _execute_action_step(self, world: World, step: Step) -> ActionResult:
        """Dispatch a non-navigation ``Step`` to the corresponding verb in ``arden.actions.verbs``."""
        verb = step.verb.strip().lower()
        params = dict(step.params or {})
        target = step.target
        mods = world.body_modifiers

        if verb in ("rest", "idle"):
            ticks = int(params.get("ticks", params.get("duration", 1)))
            return action_verbs.rest(world, max(1, ticks))

        if verb in ("grab", "pickup"):
            actual_target = target if target is not None else params.get("target")
            return action_verbs.grab(world, actual_target, modifiers=mods)

        if verb == "drop":
            actual_target = target if target is not None else params.get("target")
            return action_verbs.drop(world, actual_target, modifiers=mods)

        if verb == "push":
            actual_target = target if target is not None else params.get("target")
            direction = params.get("direction")
            force = params.get("force", 0.5)
            res = action_verbs.push(
                world,
                actual_target,
                force,
                direction=direction,
                modifiers=mods,
            )
            if res.ok and isinstance(actual_target, str) and actual_target in world.objects:
                obj = world.objects[actual_target]
                if obj.velocity.speed > 0.0:
                    world._step_physics()
            return res

        if verb == "strike":
            actual_target = target if target is not None else params.get("target")
            force = params.get("force", 0.5)
            direction = params.get("direction")
            tool = params.get("tool", params.get("held", params.get("with_object")))
            res = action_verbs.strike(
                world,
                actual_target,
                force,
                direction=direction,
                tool=tool,
                modifiers=mods,
            )
            if res.ok and isinstance(actual_target, str) and actual_target in world.objects:
                obj = world.objects[actual_target]
                if obj.velocity.speed > 0.0:
                    world._step_physics()
            return res

        if verb in ("place", "put"):
            loc = (
                params.get("location")
                or params.get("position")
                or params.get("destination")
                or params.get("to")
                or target
            )
            return action_verbs.place(world, loc, modifiers=mods)

        if verb == "stack":
            held = params.get("held", params.get("held_item", world.body.holding))
            actual_target = target if target is not None else params.get("target")
            return action_verbs.stack(world, held, actual_target, modifiers=mods)

        if verb == "insert":
            held = params.get("held", params.get("held_item", world.body.holding))
            actual_target = target if target is not None else params.get("target")
            return action_verbs.insert(world, held, actual_target, modifiers=mods)

        if verb == "separate":
            actual_target = target if target is not None else params.get("target")
            return action_verbs.separate(world, actual_target, modifiers=mods)

        if verb == "bind":
            held = params.get("held", params.get("held_item", world.body.holding))
            other = (
                target
                if target is not None
                else params.get("target", params.get("second", params.get("with")))
            )
            binder = params.get(
                "binder", params.get("binder_id", params.get("using"))
            )
            return action_verbs.bind(
                world,
                held,
                other,
                binder,
                modifiers=mods,
            )

        if verb in ("look", "inspect"):
            actual_target = target if target is not None else params.get("target")
            return action_verbs.look(world, actual_target, modifiers=mods)

        if verb == "speak":
            text = (
                params.get("text")
                or params.get("message")
                or params.get("utterance")
                or (target if isinstance(target, str) else "")
            )
            return action_verbs.speak(world, text, modifiers=mods)

        if verb in ("eat", "edible"):
            actual_target = target if target is not None else params.get("target")
            return action_verbs.eat(world, actual_target, modifiers=mods)

        return action_verbs.execute_action(
            world,
            verb,
            target=target,
            params=params,
            modifiers=mods,
        )

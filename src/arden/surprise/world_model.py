"""WorldModel interface, NullWorldModel, and simple baseline predictor (T16).

Rules from docs/DESIGN.md (Sections 10.4, 10.5, 10.6) and Ticket T16:
- ``WorldModel`` defines ``predict(state, action) -> embedding`` and ``error(pred, actual) -> float``.
- ``NullWorldModel`` always returns ``0.0`` prediction error.
- ``VelocityWorldModel`` (also exported as ``BaselineWorldModel``, ``ExtrapolationWorldModel``,
  and ``LastVelocityWorldModel``) implements a simple last-velocity linear extrapolation baseline
  so physical surprise works before Phase 2.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


class StateEmbedding(list[float]):
    """Embedding vector returned by ``WorldModel.predict(state, action)``.

    Subclasses ``list[float]`` so callers can treat it as a plain numeric
    sequence while also carrying per-object predicted 2D positions and
    velocities for multi-object states (``World``, ``Percept``, or object dicts).
    """

    def __init__(
        self,
        values: Sequence[float] = (),
        *,
        positions: Mapping[str, tuple[float, float]] | None = None,
        velocities: Mapping[str, tuple[float, float]] | None = None,
        primary_id: str | None = None,
    ) -> None:
        super().__init__(float(v) for v in values)
        self.positions: dict[str, tuple[float, float]] = dict(positions or {})
        self.velocities: dict[str, tuple[float, float]] = dict(velocities or {})
        self.primary_id: str | None = primary_id

    @property
    def vector(self) -> list[float]:
        """Plain list of embedding floats."""
        return list(self)

    def to_list(self) -> list[float]:
        """Return a copy of the embedding floats as a standard ``list``."""
        return list(self)


def _extract_xy_and_vel(
    item: Any,
    default_id: str = "object_0",
) -> tuple[str, tuple[float, float], tuple[float, float] | None] | None:
    """Extract ``(object_id, (x, y), (vx, vy) | None)`` from a single object or coordinate container."""
    if item is None:
        return None

    # Plain numeric sequence [x, y] or [x, y, vx, vy]
    if isinstance(item, tuple | list) and not isinstance(item, StateEmbedding):
        if len(item) >= 2 and all(
            isinstance(v, int | float) and not isinstance(v, bool) for v in item[:2]
        ):
            x, y = float(item[0]), float(item[1])
            if len(item) >= 4 and all(
                isinstance(v, int | float) and not isinstance(v, bool) for v in item[2:4]
            ):
                return default_id, (x, y), (float(item[2]), float(item[3]))
            return default_id, (x, y), None

    if isinstance(item, Mapping):
        oid = str(item.get("id", item.get("object_id", default_id)))
        pos = item.get("position")
        if isinstance(pos, Mapping) and "x" in pos and "y" in pos:
            x, y = float(pos["x"]), float(pos["y"])
        elif hasattr(pos, "x") and hasattr(pos, "y"):
            x, y = float(pos.x), float(pos.y)
        elif isinstance(pos, tuple | list) and len(pos) >= 2:
            x, y = float(pos[0]), float(pos[1])
        elif "x" in item and "y" in item:
            x, y = float(item["x"]), float(item["y"])
        else:
            return None

        vel: tuple[float, float] | None = None
        raw_vel = item.get("velocity")
        if isinstance(raw_vel, Mapping) and ("x" in raw_vel or "vx" in raw_vel):
            vx = float(raw_vel.get("x", raw_vel.get("vx", 0.0)))
            vy = float(raw_vel.get("y", raw_vel.get("vy", 0.0)))
            vel = (vx, vy)
        elif hasattr(raw_vel, "x") and hasattr(raw_vel, "y"):
            vel = (float(raw_vel.x), float(raw_vel.y))
        elif isinstance(raw_vel, tuple | list) and len(raw_vel) >= 2:
            vel = (float(raw_vel[0]), float(raw_vel[1]))
        elif "vx" in item or "vy" in item:
            vel = (float(item.get("vx", 0.0)), float(item.get("vy", 0.0)))

        return oid, (x, y), vel

    # Object with .position or (.x, .y)
    oid = str(getattr(item, "id", getattr(item, "object_id", default_id)))
    pos_attr = getattr(item, "position", None)
    if pos_attr is not None:
        if hasattr(pos_attr, "x") and hasattr(pos_attr, "y"):
            x, y = float(pos_attr.x), float(pos_attr.y)
        elif isinstance(pos_attr, Mapping) and "x" in pos_attr and "y" in pos_attr:
            x, y = float(pos_attr["x"]), float(pos_attr["y"])
        elif isinstance(pos_attr, tuple | list) and len(pos_attr) >= 2:
            x, y = float(pos_attr[0]), float(pos_attr[1])
        else:
            return None
    elif hasattr(item, "x") and hasattr(item, "y"):
        x, y = float(item.x), float(item.y)
    else:
        return None

    vel_attr = getattr(item, "velocity", None)
    vel_tuple: tuple[float, float] | None = None
    if vel_attr is not None:
        if hasattr(vel_attr, "x") and hasattr(vel_attr, "y"):
            vel_tuple = (float(vel_attr.x), float(vel_attr.y))
        elif isinstance(vel_attr, Mapping):
            vel_tuple = (
                float(vel_attr.get("x", vel_attr.get("vx", 0.0))),
                float(vel_attr.get("y", vel_attr.get("vy", 0.0))),
            )
        elif isinstance(vel_attr, tuple | list) and len(vel_attr) >= 2:
            vel_tuple = (float(vel_attr[0]), float(vel_attr[1]))
    elif hasattr(item, "vx") or hasattr(item, "vy"):
        vel_tuple = (float(getattr(item, "vx", 0.0)), float(getattr(item, "vy", 0.0)))

    return oid, (x, y), vel_tuple


def extract_state_objects(
    state: Any,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float] | None], str | None]:
    """Extract ``(positions_by_id, explicit_velocities_by_id, primary_id)`` from ``state``."""
    if isinstance(state, StateEmbedding) and state.positions:
        vels: dict[str, tuple[float, float] | None] = {
            k: v for k, v in state.velocities.items()
        }
        return dict(state.positions), vels, state.primary_id

    if state is None:
        return {}, {}, None

    # 1. World or Percept with .objects
    objs_attr = getattr(state, "objects", None)
    if objs_attr is not None and not isinstance(state, Mapping):
        positions: dict[str, tuple[float, float]] = {}
        velocities: dict[str, tuple[float, float] | None] = {}
        items = objs_attr.values() if isinstance(objs_attr, Mapping) else objs_attr
        for idx, obj in enumerate(items):
            extracted = _extract_xy_and_vel(obj, default_id=f"object_{idx}")
            if extracted is not None:
                oid, pos, vel = extracted
                positions[oid] = pos
                velocities[oid] = vel
        primary = next(iter(sorted(positions)), None) if len(positions) == 1 else None
        return positions, velocities, primary

    # 2. Mapping container
    if isinstance(state, Mapping):
        if "objects" in state and isinstance(state["objects"], Mapping | Sequence):
            positions = {}
            velocities = {}
            raw_objs = state["objects"]
            items = raw_objs.values() if isinstance(raw_objs, Mapping) else raw_objs
            for idx, obj in enumerate(items):
                extracted = _extract_xy_and_vel(obj, default_id=f"object_{idx}")
                if extracted is not None:
                    oid, pos, vel = extracted
                    positions[oid] = pos
                    velocities[oid] = vel
            primary = next(iter(sorted(positions)), None) if len(positions) == 1 else None
            return positions, velocities, primary

        # Single object mapping (has x/y or position)
        single = _extract_xy_and_vel(state, default_id="object_0")
        if single is not None:
            oid, pos, vel = single
            return {oid: pos}, {oid: vel}, oid

        # Mapping of {object_id: object_or_pos}
        positions = {}
        velocities = {}
        for k, v in state.items():
            extracted = _extract_xy_and_vel(v, default_id=str(k))
            if extracted is not None:
                oid, pos, vel = extracted
                key_id = str(k) if str(k).startswith("object_") else oid
                positions[key_id] = pos
                velocities[key_id] = vel
        if positions:
            primary = next(iter(sorted(positions)), None) if len(positions) == 1 else None
            return positions, velocities, primary

    # 3. Single object, Position, or numeric vector
    single = _extract_xy_and_vel(state, default_id="object_0")
    if single is not None:
        oid, pos, vel = single
        return {oid: pos}, {oid: vel}, oid

    # 4. Sequence of objects
    if isinstance(state, Sequence) and not isinstance(state, str | bytes):
        positions = {}
        velocities = {}
        for idx, item in enumerate(state):
            extracted = _extract_xy_and_vel(item, default_id=f"object_{idx}")
            if extracted is not None:
                oid, pos, vel = extracted
                positions[oid] = pos
                velocities[oid] = vel
        if positions:
            primary = next(iter(sorted(positions)), None) if len(positions) == 1 else None
            return positions, velocities, primary

    return {}, {}, None


def _extract_action_delta(
    action: Any,
    target_id: str | None = None,
) -> tuple[float, float] | None:
    """Extract an explicit ``(dx, dy)`` velocity/delta from ``action`` when provided directly."""
    if action is None:
        return None
    if isinstance(action, tuple | list) and len(action) == 2:
        if all(isinstance(v, int | float) and not isinstance(v, bool) for v in action):
            return (float(action[0]), float(action[1]))
    if isinstance(action, Mapping):
        act_target = action.get("target", action.get("object_id"))
        if act_target is not None and target_id is not None and str(act_target) != target_id:
            return None
        for key in ("delta", "velocity", "displacement"):
            val = action.get(key)
            if isinstance(val, Mapping) and "x" in val and "y" in val:
                return (float(val["x"]), float(val["y"]))
            if isinstance(val, tuple | list) and len(val) >= 2:
                return (float(val[0]), float(val[1]))
        if "dx" in action or "dy" in action:
            return (float(action.get("dx", 0.0)), float(action.get("dy", 0.0)))
    return None


@runtime_checkable
class WorldModel(Protocol):
    """Interface for predicting next state embeddings and measuring physical prediction error."""

    def predict(self, state: Any, action: Any = None) -> StateEmbedding:
        """Predict the next state embedding given current ``state`` and ``action``."""
        ...

    def error(self, pred: Any, actual: Any) -> float:
        """Compute non-negative prediction error between ``pred`` and ``actual``."""
        ...


class NullWorldModel:
    """Trivial ``WorldModel`` baseline whose prediction error is always ``0.0``."""

    def predict(self, state: Any = None, action: Any = None) -> StateEmbedding:
        """Return a zero embedding regardless of ``state`` and ``action``."""
        positions, _, primary = extract_state_objects(state)
        zero_positions = {oid: (0.0, 0.0) for oid in positions}
        vec = [0.0] * max(2, 2 * len(positions))
        return StateEmbedding(vec, positions=zero_positions, primary_id=primary)

    def error(self, pred: Any = None, actual: Any = None) -> float:
        """Always return ``0.0`` error."""
        del pred, actual
        return 0.0

    def reset(self) -> None:
        """No-op reset for interface compatibility."""


class VelocityWorldModel:
    """Simple baseline ``WorldModel`` using last-velocity linear extrapolation.

    For each tracked object at position ``(x, y)`` with velocity ``(vx, vy)``
    (taken from the state's explicit velocity or inferred from ``pos(t) - pos(t - 1)``),
    predicts next position ``(x + vx, y + vy)``.
    """

    def __init__(self, dt: float = 1.0) -> None:
        self.dt: float = float(dt)
        self._last_positions: dict[str, tuple[float, float]] = {}
        self._last_velocities: dict[str, tuple[float, float]] = {}
        self.last_object_errors: dict[str, float] = {}

    def reset(self) -> None:
        """Clear tracked object positions and velocities."""
        self._last_positions.clear()
        self._last_velocities.clear()
        self.last_object_errors.clear()

    def encode(self, state: Any) -> StateEmbedding:
        """Encode ``state`` into a ``StateEmbedding`` of current object positions."""
        positions, velocities, primary = extract_state_objects(state)
        if not positions:
            if isinstance(state, Sequence) and not isinstance(state, str | bytes):
                nums = [
                    float(v)
                    for v in state
                    if isinstance(v, int | float) and not isinstance(v, bool)
                ]
                if nums:
                    return StateEmbedding(nums)
            return StateEmbedding([0.0, 0.0])

        flat: list[float] = []
        clean_vels: dict[str, tuple[float, float]] = {}
        for oid in sorted(positions):
            px, py = positions[oid]
            flat.extend([px, py])
            v = velocities.get(oid)
            if v is not None:
                clean_vels[oid] = v
            elif oid in self._last_velocities:
                clean_vels[oid] = self._last_velocities[oid]
            else:
                clean_vels[oid] = (0.0, 0.0)

        return StateEmbedding(
            flat,
            positions=positions,
            velocities=clean_vels,
            primary_id=primary,
        )

    def predict(self, state: Any, action: Any = None) -> StateEmbedding:
        """Predict next state embedding by extrapolating ``position + last_velocity``."""
        positions, explicit_vels, primary = extract_state_objects(state)
        if not positions:
            if isinstance(state, Sequence) and not isinstance(state, str | bytes):
                nums = [
                    float(v)
                    for v in state
                    if isinstance(v, int | float) and not isinstance(v, bool)
                ]
                if nums:
                    return StateEmbedding(nums)
            return StateEmbedding([0.0, 0.0])

        pred_positions: dict[str, tuple[float, float]] = {}
        pred_velocities: dict[str, tuple[float, float]] = {}
        flat: list[float] = []

        for oid in sorted(positions):
            x, y = positions[oid]
            exp_v = explicit_vels.get(oid)
            prev_pos = self._last_positions.get(oid)

            # Determine velocity:
            # 1. If explicit velocity is non-zero, use it.
            # 2. Else if we have a previous position for this object, infer velocity from (pos - prev_pos).
            # 3. Else if explicit velocity is provided (0, 0), use (0, 0).
            # 4. Else fall back to last known velocity or (0, 0).
            if exp_v is not None and (abs(exp_v[0]) > 1e-9 or abs(exp_v[1]) > 1e-9):
                vx, vy = exp_v
            elif prev_pos is not None:
                vx = x - prev_pos[0]
                vy = y - prev_pos[1]
            elif exp_v is not None:
                vx, vy = exp_v
            else:
                vx, vy = self._last_velocities.get(oid, (0.0, 0.0))

            act_delta = _extract_action_delta(action, target_id=oid)
            if act_delta is not None:
                vx += act_delta[0]
                vy += act_delta[1]

            pred_x = round(x + vx * self.dt, 6)
            pred_y = round(y + vy * self.dt, 6)

            pred_positions[oid] = (pred_x, pred_y)
            pred_velocities[oid] = (vx, vy)
            self._last_positions[oid] = (x, y)
            self._last_velocities[oid] = (vx, vy)
            flat.extend([pred_x, pred_y])

        return StateEmbedding(
            flat,
            positions=pred_positions,
            velocities=pred_velocities,
            primary_id=primary,
        )

    def error_by_object(self, pred: Any, actual: Any) -> dict[str, float]:
        """Return per-object Euclidean prediction errors for objects present in ``pred`` and ``actual``."""
        pred_positions, _, pred_primary = extract_state_objects(pred)
        act_positions, _, act_primary = extract_state_objects(actual)

        per_obj: dict[str, float] = {}
        if pred_positions and act_positions:
            shared_ids = sorted(set(pred_positions) & set(act_positions))
            if shared_ids:
                for oid in shared_ids:
                    px, py = pred_positions[oid]
                    ax, ay = act_positions[oid]
                    per_obj[oid] = round(math.hypot(px - ax, py - ay), 6)
            elif len(pred_positions) == 1 and len(act_positions) == 1:
                oid = act_primary or pred_primary or "object_0"
                px, py = next(iter(pred_positions.values()))
                ax, ay = next(iter(act_positions.values()))
                per_obj[oid] = round(math.hypot(px - ax, py - ay), 6)

        self.last_object_errors = per_obj
        return per_obj

    def error(self, pred: Any, actual: Any) -> float:
        """Compute Euclidean prediction error between ``pred`` and ``actual``."""
        if pred is None or actual is None:
            return 0.0

        per_obj = self.error_by_object(pred, actual)
        if per_obj:
            return round(math.sqrt(sum(d * d for d in per_obj.values())), 6)

        # Fallback: direct numeric vector comparison
        pred_vec = (
            list(pred)
            if isinstance(pred, Sequence) and not isinstance(pred, str | bytes)
            else list(self.encode(pred))
        )
        act_vec = (
            list(actual)
            if isinstance(actual, Sequence) and not isinstance(actual, str | bytes)
            else list(self.encode(actual))
        )
        if not pred_vec or not act_vec:
            return 0.0
        n = min(len(pred_vec), len(act_vec))
        dist = math.sqrt(sum((float(pred_vec[i]) - float(act_vec[i])) ** 2 for i in range(n)))
        return round(dist, 6)


# Standard aliases for the simple baseline world model
LastVelocityWorldModel = VelocityWorldModel
ExtrapolationWorldModel = VelocityWorldModel
BaselineWorldModel = VelocityWorldModel
SimpleWorldModel = VelocityWorldModel
LinearWorldModel = VelocityWorldModel

__all__ = [
    "BaselineWorldModel",
    "ExtrapolationWorldModel",
    "LastVelocityWorldModel",
    "LinearWorldModel",
    "NullWorldModel",
    "SimpleWorldModel",
    "StateEmbedding",
    "VelocityWorldModel",
    "WorldModel",
    "extract_state_objects",
]

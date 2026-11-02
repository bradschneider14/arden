"""Body state (energy, comfort), physical side effects (BodyModifiers), and intrinsic drives.

Rules from docs/DESIGN.md (Section 9 & 9.1):
- No hunger drive. Energy and comfort are plain numeric values in ``[0.0, 1.0]`` with no labels.
- Energy drains slowly over time and faster with activity (weighted by verb and force);
  restored by rest (idle) and optionally by eating objects with an ``edible`` rule.
- Comfort is lowered by cold, wet, night temperature, storms, and low energy;
  restored by shelter, dryness, warmth, and rest.
- Side effects fill ``BodyModifiers`` with gradual curves:
  - low energy -> slower movement and weaker push/strike
  - cold -> grab/bind failure chance
  - wet -> slower movement and held items can slip
  - very low comfort -> smaller sensing radius
- Effect strengths are configurable under ``[body]`` with ``gentle`` and ``harsh`` presets;
  nothing ever forces sleep or death.
- Drives: boredom rises when no novelty or surprise occurs and falls when it does;
  curiosity reward accumulates from surprise and novelty. Both are exposed to the
  attention gate (thresholds) and to memory salience.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from arden.actions.verbs import ActionResult, BodyModifiers
    from arden.world.objects import BehaviorRule, Position, Velocity, WorldObject
    from arden.world.world import AgentBody, World

PresetName = Literal["default", "normal", "gentle", "harsh"]

DEFAULT_VERB_ENERGY_WEIGHTS: dict[str, float] = {
    "push": 1.0,
    "strike": 1.0,
    "stack": 0.75,
    "bind": 0.65,
    "separate": 0.60,
    "move": 0.60,
    "insert": 0.50,
    "grab": 0.35,
    "place": 0.30,
    "drop": 0.15,
    "speak": 0.10,
    "look": 0.05,
    "idle": 0.0,
    "rest": 0.0,
    "eat": 0.0,
    "edible": 0.0,
}

_PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "default": {},
    "normal": {},
    "gentle": {
        "energy_time_drain": 0.001,
        "energy_activity_drain": 0.010,
        "energy_rest_recovery": 0.020,
        "edible_energy_restore": 0.20,
        "comfort_cold_drain": 0.010,
        "comfort_wet_drain": 0.010,
        "comfort_night_drain": 0.006,
        "comfort_storm_drain": 0.014,
        "comfort_low_energy_drain": 0.010,
        "comfort_shelter_recovery": 0.020,
        "comfort_dry_recovery": 0.007,
        "comfort_warm_recovery": 0.008,
        "comfort_rest_recovery": 0.020,
        "low_energy_speed_effect": 0.20,
        "low_energy_force_effect": 0.22,
        "cold_grab_fail_effect": 0.15,
        "cold_bind_fail_effect": 0.15,
        "wet_speed_effect": 0.10,
        "wet_slip_effect": 0.10,
        "low_comfort_sensing_effect": 0.20,
        "boredom_rise_rate": 0.015,
    },
    "harsh": {
        "energy_time_drain": 0.004,
        "energy_activity_drain": 0.035,
        "energy_rest_recovery": 0.010,
        "edible_energy_restore": 0.10,
        "comfort_cold_drain": 0.030,
        "comfort_wet_drain": 0.030,
        "comfort_night_drain": 0.022,
        "comfort_storm_drain": 0.048,
        "comfort_low_energy_drain": 0.032,
        "comfort_shelter_recovery": 0.010,
        "comfort_dry_recovery": 0.003,
        "comfort_warm_recovery": 0.004,
        "comfort_rest_recovery": 0.010,
        "low_energy_speed_effect": 0.60,
        "low_energy_force_effect": 0.65,
        "cold_grab_fail_effect": 0.55,
        "cold_bind_fail_effect": 0.55,
        "wet_speed_effect": 0.35,
        "wet_slip_effect": 0.45,
        "low_comfort_sensing_effect": 0.60,
        "boredom_rise_rate": 0.030,
    },
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _smooth_curve(x: float) -> float:
    """Map ``x`` in ``[0.0, 1.0]`` onto a gradual monotonic curve in ``[0.0, 1.0]``."""
    u = _clamp(x, 0.0, 1.0)
    return 0.5 * u + 0.5 * (u * u)


def _fmt_num(value: float | int) -> str:
    rounded = round(float(value), 4)
    if rounded == 0.0:
        return "0"
    return f"{rounded:.4f}".rstrip("0").rstrip(".")


class BodyConfig(BaseModel):
    """Configuration for body energy, comfort, physical side effects, and drives under ``[body]``."""

    model_config = ConfigDict(extra="forbid")

    preset: PresetName = "default"

    # Energy rates
    energy_time_drain: float = Field(default=0.002, ge=0.0, le=1.0)
    energy_activity_drain: float = Field(default=0.020, ge=0.0, le=1.0)
    energy_rest_recovery: float = Field(default=0.015, ge=0.0, le=1.0)
    edible_energy_restore: float = Field(default=0.15, ge=0.0, le=1.0)
    verb_energy_weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_VERB_ENERGY_WEIGHTS)
    )
    low_energy_effect_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    # Comfort drain rates
    comfort_cold_drain: float = Field(default=0.018, ge=0.0, le=1.0)
    comfort_wet_drain: float = Field(default=0.018, ge=0.0, le=1.0)
    comfort_night_drain: float = Field(default=0.012, ge=0.0, le=1.0)
    comfort_storm_drain: float = Field(default=0.028, ge=0.0, le=1.0)
    comfort_low_energy_drain: float = Field(default=0.020, ge=0.0, le=1.0)

    # Comfort recovery rates
    comfort_shelter_recovery: float = Field(default=0.015, ge=0.0, le=1.0)
    comfort_dry_recovery: float = Field(default=0.005, ge=0.0, le=1.0)
    comfort_warm_recovery: float = Field(default=0.006, ge=0.0, le=1.0)
    comfort_rest_recovery: float = Field(default=0.015, ge=0.0, le=1.0)
    comfort_neutral_temperature: float = Field(default=0.85)
    low_energy_comfort_threshold: float = Field(default=0.50, ge=0.0, le=1.0)

    # Physical side-effect strengths (never 1.0 on speed/force/sensing so nothing forces sleep or death)
    low_energy_speed_effect: float = Field(default=0.40, ge=0.0, lt=1.0)
    low_energy_force_effect: float = Field(default=0.45, ge=0.0, lt=1.0)
    cold_grab_fail_effect: float = Field(default=0.35, ge=0.0, le=1.0)
    cold_bind_fail_effect: float = Field(default=0.35, ge=0.0, le=1.0)
    wet_speed_effect: float = Field(default=0.20, ge=0.0, lt=1.0)
    wet_slip_effect: float = Field(default=0.25, ge=0.0, le=1.0)
    low_comfort_sensing_effect: float = Field(default=0.40, ge=0.0, lt=1.0)
    very_low_comfort_threshold: float = Field(default=0.35, ge=0.0, le=1.0)

    # Drives and gate thresholds
    boredom_rise_rate: float = Field(default=0.020, ge=0.0, le=1.0)
    boredom_drop_scale: float = Field(default=0.50, ge=0.0)
    curiosity_novelty_weight: float = Field(default=0.30, ge=0.0)
    curiosity_surprise_weight: float = Field(default=0.50, ge=0.0)
    low_energy_wake_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    low_comfort_wake_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    boredom_wake_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    curiosity_wake_threshold: float = Field(default=0.60, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def _apply_preset_defaults(cls, values: Any) -> Any:
        if isinstance(values, str):
            values = {"preset": values}
        if not isinstance(values, Mapping):
            return values
        raw = dict(values)
        preset_raw = raw.get("preset", "default")
        if isinstance(preset_raw, str):
            preset_key = preset_raw.strip().lower()
            if preset_key not in _PRESET_OVERRIDES:
                allowed = ", ".join(sorted(_PRESET_OVERRIDES))
                raise ValueError(f"preset must be one of: {allowed}")
            raw["preset"] = preset_key
            preset_defaults = _PRESET_OVERRIDES[preset_key]
            for k, v in preset_defaults.items():
                if k not in raw:
                    raw[k] = v
        return raw

    @field_validator("verb_energy_weights")
    @classmethod
    def _merge_verb_weights(cls, value: dict[str, float]) -> dict[str, float]:
        merged = dict(DEFAULT_VERB_ENERGY_WEIGHTS)
        for k, v in value.items():
            fval = float(v)
            if not math.isfinite(fval) or fval < 0.0:
                raise ValueError(f"verb energy weight for {k!r} must be a finite number >= 0")
            merged[str(k)] = fval
        return merged

    @classmethod
    def from_preset(cls, preset: PresetName | str, **overrides: Any) -> BodyConfig:
        """Build a ``BodyConfig`` from ``preset`` (``'default'``, ``'gentle'``, or ``'harsh'``)."""
        return cls.model_validate({"preset": preset, **overrides})

    @classmethod
    def gentle(cls, **overrides: Any) -> BodyConfig:
        """Return the ``gentle`` preset configuration."""
        return cls.from_preset("gentle", **overrides)

    @classmethod
    def harsh(cls, **overrides: Any) -> BodyConfig:
        """Return the ``harsh`` preset configuration."""
        return cls.from_preset("harsh", **overrides)


BodySettings = BodyConfig

DEFAULT_BODY_CONFIG = BodyConfig()
GENTLE_BODY_CONFIG = BodyConfig.gentle()
HARSH_BODY_CONFIG = BodyConfig.harsh()
BODY_PRESETS: dict[str, BodyConfig] = {
    "default": DEFAULT_BODY_CONFIG,
    "normal": DEFAULT_BODY_CONFIG,
    "gentle": GENTLE_BODY_CONFIG,
    "harsh": HARSH_BODY_CONFIG,
}


def resolve_body_config(
    config: BodyConfig | Mapping[str, Any] | str | None = None,
    world: World | None = None,
) -> BodyConfig:
    """Resolve a ``BodyConfig`` from an explicit argument, ``world``, or defaults."""
    if isinstance(config, BodyConfig):
        return config
    if isinstance(config, str):
        return BodyConfig.from_preset(config)
    if isinstance(config, Mapping):
        if "body" in config and isinstance(config["body"], Mapping | BodyConfig | str):
            return resolve_body_config(config["body"])
        return BodyConfig.model_validate(dict(config))
    if world is not None:
        w_body_cfg = getattr(world, "body_config", None)
        if isinstance(w_body_cfg, BodyConfig):
            return w_body_cfg
        w_cfg = getattr(world, "config", None)
        if w_cfg is not None:
            sub = getattr(w_cfg, "body", None)
            if isinstance(sub, BodyConfig):
                return sub
            if isinstance(sub, Mapping | str):
                return resolve_body_config(sub)
    return DEFAULT_BODY_CONFIG


class DriveThresholds(BaseModel):
    """Thresholds exposed to the attention gate for drive-triggered wakes."""

    model_config = ConfigDict(extra="forbid")

    low_energy: float = Field(default=0.25, ge=0.0, le=1.0)
    low_comfort: float = Field(default=0.25, ge=0.0, le=1.0)
    boredom: float = Field(default=0.70, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.60, ge=0.0)

    @property
    def boredom_threshold(self) -> float:
        return self.boredom

    @property
    def curiosity_threshold(self) -> float:
        return self.curiosity

    @property
    def low_energy_threshold(self) -> float:
        return self.low_energy

    @property
    def low_comfort_threshold(self) -> float:
        return self.low_comfort


class DriveState(BaseModel):
    """Intrinsic boredom and curiosity drive state exposed to the gate and memory salience."""

    model_config = ConfigDict(extra="forbid")

    boredom: float = Field(default=0.0, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.0, ge=0.0)
    last_curiosity_reward: float = Field(default=0.0, ge=0.0)
    idle_ticks: int = Field(default=0, ge=0)

    @field_validator("boredom", "curiosity", "last_curiosity_reward")
    @classmethod
    def _values_are_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("drive values must be finite")
        return value

    @property
    def curiosity_reward(self) -> float:
        """Alias for accumulated ``curiosity`` reward."""
        return self.curiosity

    @curiosity_reward.setter
    def curiosity_reward(self, value: float) -> None:
        self.curiosity = max(0.0, float(value))

    def step(
        self,
        *,
        novelty: float | int | bool = 0.0,
        surprise: float | int | bool = 0.0,
        config: BodyConfig | Mapping[str, Any] | str | None = None,
    ) -> DriveState:
        """Advance boredom and curiosity for one step or observation.

        - Boredom rises when no novelty or surprise occurs, and falls when it does.
        - Curiosity reward accumulates from surprise and novelty.
        """
        cfg = resolve_body_config(config)
        nov = 1.0 if novelty is True else (0.0 if novelty is False else max(0.0, float(novelty)))
        surp = 1.0 if surprise is True else (0.0 if surprise is False else max(0.0, float(surprise)))
        if not (math.isfinite(nov) and math.isfinite(surp)):
            raise ValueError("novelty and surprise must be finite numbers")

        signal = nov + surp
        if signal > 0.0:
            self.boredom = _clamp(self.boredom - cfg.boredom_drop_scale * signal, 0.0, 1.0)
            reward = cfg.curiosity_novelty_weight * nov + cfg.curiosity_surprise_weight * surp
            self.last_curiosity_reward = reward
            self.curiosity = max(0.0, self.curiosity + reward)
            self.idle_ticks = 0
        else:
            self.boredom = _clamp(self.boredom + cfg.boredom_rise_rate, 0.0, 1.0)
            self.last_curiosity_reward = 0.0
            self.idle_ticks += 1
        return self

    update = step

    def reset_curiosity(self) -> None:
        """Reset accumulated curiosity reward after the gate wakes or consumes it."""
        self.curiosity = 0.0
        self.last_curiosity_reward = 0.0

    def thresholds(
        self,
        config: BodyConfig | Mapping[str, Any] | str | None = None,
    ) -> DriveThresholds:
        """Return gate thresholds for drive-based wake checks."""
        cfg = resolve_body_config(config)
        return DriveThresholds(
            low_energy=cfg.low_energy_wake_threshold,
            low_comfort=cfg.low_comfort_wake_threshold,
            boredom=cfg.boredom_wake_threshold,
            curiosity=cfg.curiosity_wake_threshold,
        )

    def gate_triggers(
        self,
        body: AgentBody | Mapping[str, Any] | None = None,
        *,
        energy: float | None = None,
        comfort: float | None = None,
        config: BodyConfig | Mapping[str, Any] | str | None = None,
    ) -> list[str]:
        """Return the list of drive triggers currently exceeding gate wake thresholds."""
        thresh = self.thresholds(config)
        triggers: list[str] = []

        if self.boredom >= thresh.boredom:
            triggers.append("boredom")
        if self.curiosity >= thresh.curiosity:
            triggers.append("curiosity")

        e_val = energy
        c_val = comfort
        if body is not None:
            if isinstance(body, Mapping):
                if e_val is None and "energy" in body:
                    e_val = float(body["energy"])
                if c_val is None and "comfort" in body:
                    c_val = float(body["comfort"])
            else:
                if e_val is None and hasattr(body, "energy"):
                    e_val = float(body.energy)
                if c_val is None and hasattr(body, "comfort"):
                    c_val = float(body.comfort)

        if e_val is not None and e_val <= thresh.low_energy:
            triggers.append("low_energy")
        if c_val is not None and c_val <= thresh.low_comfort:
            triggers.append("low_comfort")

        return triggers

    def should_wake_gate(
        self,
        body: AgentBody | Mapping[str, Any] | None = None,
        *,
        energy: float | None = None,
        comfort: float | None = None,
        config: BodyConfig | Mapping[str, Any] | str | None = None,
    ) -> bool:
        """Return True if any drive or body threshold warrants waking the LLM."""
        return bool(
            self.gate_triggers(body, energy=energy, comfort=comfort, config=config)
        )

    def memory_salience(
        self,
        body: AgentBody | Mapping[str, Any] | None = None,
        *,
        comfort: float | None = None,
        energy: float | None = None,
        surprise: float = 0.0,
        novelty: float = 0.0,
    ) -> float:
        """Compute memory salience score in ``[0.0, 1.0]`` from drives and body comfort/energy."""
        return compute_memory_salience(
            drives=self,
            body=body,
            comfort=comfort,
            energy=energy,
            surprise=surprise,
            novelty=novelty,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize drive state to a JSON-compatible dictionary."""
        return {
            "boredom": self.boredom,
            "curiosity": self.curiosity,
            "last_curiosity_reward": self.last_curiosity_reward,
            "idle_ticks": self.idle_ticks,
        }


Drives = DriveState


def compute_memory_salience(
    drives: DriveState | Mapping[str, Any] | None = None,
    body: AgentBody | Mapping[str, Any] | None = None,
    *,
    comfort: float | None = None,
    energy: float | None = None,
    boredom: float | None = None,
    curiosity: float | None = None,
    surprise: float = 0.0,
    novelty: float = 0.0,
) -> float:
    """Compute episodic memory salience in ``[0.0, 1.0]`` from drives and body state.

    - Accumulates salience from curiosity reward, surprise, and novelty.
    - Boredom contributes drive salience.
    - Lower comfort (discomfort) and lower energy increase salience so uncomfortable
      episodes are remembered more strongly (docs/DESIGN.md 9.1).
    """
    b_val = 0.0 if boredom is None else float(boredom)
    cur_val = 0.0 if curiosity is None else float(curiosity)
    if drives is not None:
        if isinstance(drives, DriveState):
            if boredom is None:
                b_val = drives.boredom
            if curiosity is None:
                cur_val = drives.curiosity
        elif isinstance(drives, Mapping):
            if boredom is None:
                b_val = float(drives.get("boredom", 0.0))
            if curiosity is None:
                cur_val = float(drives.get("curiosity", drives.get("curiosity_reward", 0.0)))

    c_val = 1.0 if comfort is None else float(comfort)
    e_val = 1.0 if energy is None else float(energy)
    if body is not None:
        if isinstance(body, Mapping):
            if comfort is None:
                c_val = float(body.get("comfort", 1.0))
            if energy is None:
                e_val = float(body.get("energy", 1.0))
        else:
            if comfort is None and hasattr(body, "comfort"):
                c_val = float(body.comfort)
            if energy is None and hasattr(body, "energy"):
                e_val = float(body.energy)

    discomfort = _clamp(1.0 - c_val, 0.0, 1.0)
    fatigue = _clamp(1.0 - e_val, 0.0, 1.0)
    curiosity_term = 1.0 - math.exp(-max(0.0, cur_val + 0.5 * max(0.0, float(surprise)) + 0.3 * max(0.0, float(novelty))))
    boredom_term = _clamp(b_val, 0.0, 1.0)

    raw_salience = (
        0.10
        + 0.45 * curiosity_term
        + 0.25 * discomfort
        + 0.10 * fatigue
        + 0.10 * boredom_term
    )
    return _clamp(raw_salience, 0.0, 1.0)


memory_salience = compute_memory_salience


def activity_energy_cost(
    verb: str | None,
    *,
    force: float | None = None,
    speed: float | None = None,
    config: BodyConfig | Mapping[str, Any] | str | None = None,
) -> float:
    """Compute energy drain for executing ``verb`` with ``force`` or ``speed``."""
    if verb is None:
        return 0.0
    cfg = resolve_body_config(config)
    v_name = str(verb).strip().lower()
    if v_name in ("idle", "rest", "eat", "edible", "none", ""):
        return 0.0

    weight = cfg.verb_energy_weights.get(
        v_name,
        0.25 if v_name.startswith("action_") else 0.25,
    )
    if weight <= 0.0:
        return 0.0

    if force is not None:
        intensity = _clamp(float(force), 0.0, 1.0)
    elif speed is not None:
        intensity = _clamp(float(speed), 0.0, 1.0)
    elif v_name in ("push", "strike"):
        intensity = 0.5
    else:
        intensity = 1.0

    if intensity <= 0.0:
        return 0.0

    return cfg.energy_activity_drain * weight * (0.25 + 0.75 * intensity)


def compute_shelter_level(world: World) -> float:
    """Detect how sheltered the agent body is in ``world`` (in ``[0.0, 1.0]``)."""
    explicit = getattr(world, "_sheltered_override", None)
    if explicit is not None:
        if isinstance(explicit, bool):
            return 1.0 if explicit else 0.0
        return _clamp(float(explicit), 0.0, 1.0)

    body = world.body
    if body.level.startswith("inside:"):
        cid = body.level.removeprefix("inside:")
        if cid in world.objects:
            return 1.0

    best = 0.0
    for obj in world.objects.values():
        if obj.id == body.holding:
            continue
        dist = math.hypot(obj.position.x - body.position.x, obj.position.y - body.position.y)
        if dist > 1.5 + obj.radius:
            continue
        if obj.state.get("shelter") is True:
            best = max(best, 1.0)
        elif isinstance(obj.state.get("shelter"), int | float) and not isinstance(
            obj.state.get("shelter"), bool
        ):
            best = max(best, _clamp(float(obj.state["shelter"]), 0.0, 1.0))
        elif obj.hollow and (obj.volume >= 0.04 or obj.height >= 0.4):
            # Hollow log or built hollow shelter
            best = max(best, 0.85)
        elif obj.made_by_agent and obj.height >= 0.4:
            # Agent-built wall, roof, or stacked shelter
            best = max(best, 0.80)
        elif obj.fixed and obj.height >= 1.5:
            # Large tree casts partial shelter from rain
            best = max(best, 0.50)

    return _clamp(best, 0.0, 1.0)


def extract_weather_conditions(
    world: World | None = None,
    *,
    weather: Any = None,
) -> tuple[str, float]:
    """Return ``(weather_type, intensity)`` from ``weather`` or ``world.weather``."""
    raw = weather if weather is not None else (getattr(world, "weather", None) if world is not None else None)
    if raw is None:
        return ("clear", 1.0)
    if isinstance(raw, str):
        w_type = raw.strip().lower() or "clear"
        intensity = float(getattr(world, "weather_intensity", 1.0)) if world is not None else 1.0
        return (w_type, _clamp(intensity, 0.0, 1.0))
    if isinstance(raw, Mapping):
        w_type = str(raw.get("type", raw.get("current", raw.get("weather", "clear")))).strip().lower()
        intensity = float(raw.get("intensity", 1.0))
        return (w_type, _clamp(intensity, 0.0, 1.0))
    w_type = str(
        getattr(raw, "type", getattr(raw, "current", getattr(raw, "weather_type", "clear")))
    ).strip().lower()
    intensity = float(getattr(raw, "intensity", 1.0))
    return (w_type, _clamp(intensity, 0.0, 1.0))


def compute_body_modifiers(
    body_or_energy: AgentBody | World | Mapping[str, Any] | float | None = None,
    *,
    energy: float | None = None,
    comfort: float | None = None,
    cold: float | bool | None = None,
    wet: float | bool | None = None,
    config: BodyConfig | Mapping[str, Any] | str | None = None,
    preset: PresetName | str | None = None,
    world: World | None = None,
    ignore_energy_threshold: bool = False,
) -> BodyModifiers:
    """Compute ``BodyModifiers`` from ``energy``, ``comfort``, ``cold``, and ``wet`` using gradual curves.

    - Low energy -> slower movement (``speed``) and weaker push/strike (``force``).
    - Cold -> grab/bind failure chance (``grab_fail_chance``, ``bind_fail_chance``, ``dexterity``).
    - Wet -> slower movement (``speed``) and held items can slip (``slip_chance``), plus slippery grip.
    - Very low comfort -> smaller sensing radius (``sensing_radius_scale``).
    """
    from arden.actions.verbs import BodyModifiers
    from arden.world.world import AgentBody, World as WorldCls

    resolved_world = world
    e_val: float = 1.0
    c_val: float = 1.0

    if isinstance(body_or_energy, WorldCls):
        resolved_world = body_or_energy
        e_val = resolved_world.body.energy
        c_val = resolved_world.body.comfort
    elif isinstance(body_or_energy, AgentBody):
        e_val = body_or_energy.energy
        c_val = body_or_energy.comfort
    elif isinstance(body_or_energy, Mapping):
        e_val = float(body_or_energy.get("energy", 1.0))
        c_val = float(body_or_energy.get("comfort", 1.0))
        if cold is None and "cold" in body_or_energy:
            cold = body_or_energy["cold"]
        if wet is None and "wet" in body_or_energy:
            wet = body_or_energy["wet"]
    elif isinstance(body_or_energy, int | float) and not isinstance(body_or_energy, bool):
        e_val = float(body_or_energy)
        ignore_energy_threshold = True

    if energy is not None:
        e_val = float(energy)
        if resolved_world is None:
            ignore_energy_threshold = True
    if comfort is not None:
        c_val = float(comfort)

    e_val = _clamp(e_val, 0.0, 1.0)
    c_val = _clamp(c_val, 0.0, 1.0)

    if preset is not None and config is None:
        cfg = BodyConfig.from_preset(preset)
    else:
        cfg = resolve_body_config(config, resolved_world)

    # Resolve cold and wet levels from world when not explicitly given
    if cold is None:
        if resolved_world is not None:
            bs = getattr(resolved_world, "_body_state_system", None)
            if bs is not None:
                cold_level = bs.current_cold_level()
            else:
                cold_level = 0.0
        else:
            cold_level = 0.0
    elif isinstance(cold, bool):
        cold_level = 1.0 if cold else 0.0
    else:
        cold_level = _clamp(float(cold), 0.0, 1.0)

    if wet is None:
        if resolved_world is not None:
            bs = getattr(resolved_world, "_body_state_system", None)
            if bs is not None:
                wet_level = bs.current_wet_level()
            else:
                wet_level = 0.0
        else:
            wet_level = 0.0
    elif isinstance(wet, bool):
        wet_level = 1.0 if wet else 0.0
    else:
        wet_level = _clamp(float(wet), 0.0, 1.0)

    # 1. Low energy -> slower movement and weaker push/strike force
    thresh = 1.0 if ignore_energy_threshold else cfg.low_energy_effect_threshold
    if e_val >= thresh or thresh <= 0.0:
        energy_curve = 0.0
    else:
        energy_deficit = (thresh - e_val) / thresh
        energy_curve = _smooth_curve(energy_deficit)

    energy_speed_mult = 1.0 - cfg.low_energy_speed_effect * energy_curve
    force_mult = 1.0 - cfg.low_energy_force_effect * energy_curve

    # 2. Wet -> slower movement and held items can slip (plus slippery grip)
    wet_curve = _smooth_curve(wet_level)
    wet_speed_mult = 1.0 - cfg.wet_speed_effect * wet_curve
    speed_mult = max(0.10, round(energy_speed_mult * wet_speed_mult, 6))
    force_mult = max(0.10, round(force_mult, 6))
    slip_chance = round(_clamp(cfg.wet_slip_effect * wet_curve, 0.0, 1.0), 6)

    # 3. Cold -> grab and bind failure chance (and wet + cold compounds grab failures)
    cold_curve = _smooth_curve(cold_level)
    cold_bind_fail = _clamp(cfg.cold_bind_fail_effect * cold_curve, 0.0, 1.0)
    dexterity = round(_clamp(1.0 - cold_bind_fail, 0.05, 1.0), 6)
    bind_fail_chance = round(cold_bind_fail, 6)

    cold_grab_fail = _clamp(cfg.cold_grab_fail_effect * cold_curve, 0.0, 1.0)
    wet_grab_fail = _clamp(0.6 * cfg.wet_slip_effect * wet_curve, 0.0, 1.0)
    combined_grab_fail = 1.0 - (1.0 - cold_grab_fail) * (1.0 - wet_grab_fail)
    grab_fail_chance = round(_clamp(combined_grab_fail, 0.0, 1.0), 6)

    # 4. Very low comfort -> smaller sensing radius
    v_thresh = cfg.very_low_comfort_threshold
    if c_val >= v_thresh or v_thresh <= 0.0:
        comfort_curve = 0.0
    else:
        comfort_deficit = (v_thresh - c_val) / v_thresh
        comfort_curve = _smooth_curve(comfort_deficit)

    sensing_radius_scale = round(
        _clamp(1.0 - cfg.low_comfort_sensing_effect * comfort_curve, 0.10, 1.0),
        6,
    )

    return BodyModifiers(
        speed=speed_mult,
        force=force_mult,
        dexterity=dexterity,
        grab_fail_chance=grab_fail_chance,
        bind_fail_chance=bind_fail_chance,
        slip_chance=slip_chance,
        sensing_radius_scale=sensing_radius_scale,
    )


def format_numeric_body_state(body: AgentBody | Mapping[str, Any]) -> str:
    """Format body state as plain numbers only, with no evaluative words."""
    from arden.world.objects import Position

    if isinstance(body, Mapping):
        pos_raw = body.get("position", {"x": 0.0, "y": 0.0})
        pos = Position.model_validate(pos_raw) if not isinstance(pos_raw, Position) else pos_raw
        level = str(body.get("level", "ground"))
        energy = float(body.get("energy", 1.0))
        comfort = float(body.get("comfort", 1.0))
        holding = body.get("holding")
    else:
        pos = body.position
        level = body.level
        energy = body.energy
        comfort = body.comfort
        holding = body.holding

    held_str = holding if holding is not None else "none"
    return (
        f"pos=({_fmt_num(pos.x)},{_fmt_num(pos.y)}) "
        f"level={level} "
        f"energy={_fmt_num(energy)} "
        f"comfort={_fmt_num(comfort)} "
        f"holding={held_str}"
    )


def _find_edible_rule(obj: WorldObject) -> BehaviorRule | None:
    """Return the first ``BehaviorRule`` on ``obj`` that marks it as edible, if any."""
    for rule in obj.behavior:
        extras = rule.__pydantic_extra__ or {}
        if rule.verb in ("edible", "eat"):
            return rule
        if extras.get("edible") is True or isinstance(extras.get("edible"), int | float):
            return rule
        resp = str(extras.get("response", "")).strip().lower()
        if resp in ("eat", "edible"):
            return rule
    return None


def is_edible_object(obj: WorldObject) -> bool:
    """Return True if ``obj`` has an ``edible`` rule in ``behavior`` or ``state``."""
    if _find_edible_rule(obj) is not None:
        return True
    ed_state = obj.state.get("edible")
    return ed_state is True or (isinstance(ed_state, int | float) and not isinstance(ed_state, bool) and ed_state > 0)


def _extract_edible_restore_amount(obj: WorldObject, config: BodyConfig) -> float:
    rule = _find_edible_rule(obj)
    if rule is not None:
        extras = rule.__pydantic_extra__ or {}
        for key in ("energy", "energy_restore", "restore", "restore_energy", "amount", "edible"):
            val = extras.get(key)
            if isinstance(val, int | float) and not isinstance(val, bool) and float(val) > 0.0:
                return float(val)
    for key in ("energy_restore", "edible_energy", "edible"):
        val = obj.state.get(key)
        if isinstance(val, int | float) and not isinstance(val, bool) and float(val) > 0.0:
            return float(val)
    return config.edible_energy_restore


def eat_object(
    world: World,
    target: str | WorldObject | None = None,
    *,
    config: BodyConfig | Mapping[str, Any] | str | None = None,
) -> ActionResult:
    """Eat an object with an ``edible`` rule (such as berries) within reach to restore energy."""
    from arden.actions.verbs import (
        INVALID_PARAMS,
        NOTHING_HAPPENED,
        OUT_OF_REACH,
        ActionResult,
        in_reach,
    )
    from arden.world.objects import WorldObject

    cfg = resolve_body_config(config, world)
    raw_target = target if target is not None else world.body.holding
    if raw_target is None:
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    target_obj: WorldObject | None
    if isinstance(raw_target, WorldObject):
        target_obj = world.objects.get(raw_target.id, raw_target if raw_target.id in world.objects else None)
    elif isinstance(raw_target, str):
        target_obj = world.objects.get(raw_target)
    else:
        return ActionResult(ok=False, reason=INVALID_PARAMS)

    if target_obj is None:
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if not in_reach(world, target_obj):
        return ActionResult(ok=False, reason=OUT_OF_REACH)

    if not is_edible_object(target_obj):
        return ActionResult(ok=False, reason=NOTHING_HAPPENED)

    restore_amount = _extract_edible_restore_amount(target_obj, cfg)
    prev_energy = world.body.energy
    new_energy = _clamp(prev_energy + restore_amount, 0.0, 1.0)
    world.body.energy = new_energy

    rule = _find_edible_rule(target_obj)
    extras = (rule.__pydantic_extra__ or {}) if rule is not None else {}
    should_consume = extras.get("consume", True) is not False and not target_obj.fixed

    captured: list[Any] = []
    unsub = world.subscribe(captured.append)
    try:
        charges = target_obj.state.get("berries", target_obj.state.get("charges", target_obj.state.get("count")))
        if isinstance(charges, int) and not isinstance(charges, bool) and charges > 1:
            key = "berries" if "berries" in target_obj.state else ("charges" if "charges" in target_obj.state else "count")
            world.update_object(
                target_obj.id,
                {"state": {**target_obj.state, key: charges - 1}},
                source="agent",
            )
        elif should_consume and target_obj.id in world.objects:
            world.despawn(target_obj.id, source="agent")
    finally:
        unsub()

    return ActionResult(
        ok=True,
        reason=None,
        events=captured,
        data={
            "eaten": target_obj.id,
            "object_id": target_obj.id,
            "previous_energy": prev_energy,
            "energy": new_energy,
            "energy_restored": round(new_energy - prev_energy, 6),
        },
    )


class BodyStateSystem:
    """Stateful manager for agent energy, comfort, environmental exposure, side effects, and drives."""

    def __init__(
        self,
        world: World | None = None,
        *,
        config: BodyConfig | Mapping[str, Any] | str | None = None,
        drives: DriveState | Mapping[str, Any] | None = None,
        wetness: float = 0.0,
        coldness: float = 0.0,
        resting: bool = False,
    ) -> None:
        self.world: World | None = world
        self.config: BodyConfig = resolve_body_config(config, world)
        if drives is None:
            self.drives = DriveState()
        elif isinstance(drives, DriveState):
            self.drives = drives.model_copy(deep=True)
        elif isinstance(drives, Mapping):
            self.drives = DriveState.model_validate(dict(drives))
        else:
            raise TypeError("drives must be a DriveState, mapping, or None")

        self.wetness: float = _clamp(wetness, 0.0, 1.0)
        self.coldness: float = _clamp(coldness, 0.0, 1.0)
        self.resting: bool = bool(resting)
        self._explicit_wet: float | None = None
        self._explicit_cold: float | None = None
        self._activity_this_tick: float = 0.0

    @property
    def preset(self) -> PresetName:
        return self.config.preset

    def set_preset(self, preset: PresetName | str) -> BodyConfig:
        """Switch `[body]` preset to ``'gentle'``, ``'harsh'``, or ``'default'``."""
        self.config = BodyConfig.from_preset(preset)
        return self.config

    @property
    def cold(self) -> float:
        return self.current_cold_level()

    @cold.setter
    def cold(self, value: float | bool | None) -> None:
        if value is None:
            self._explicit_cold = None
        elif isinstance(value, bool):
            self._explicit_cold = 1.0 if value else 0.0
            self.coldness = self._explicit_cold
        else:
            self._explicit_cold = _clamp(float(value), 0.0, 1.0)
            self.coldness = self._explicit_cold

    @property
    def wet(self) -> float:
        return self.current_wet_level()

    @wet.setter
    def wet(self, value: float | bool | None) -> None:
        if value is None:
            self._explicit_wet = None
        elif isinstance(value, bool):
            self._explicit_wet = 1.0 if value else 0.0
            self.wetness = self._explicit_wet
        else:
            self._explicit_wet = _clamp(float(value), 0.0, 1.0)
            self.wetness = self._explicit_wet

    def current_shelter_level(self, *, sheltered: float | bool | None = None) -> float:
        if sheltered is not None:
            if isinstance(sheltered, bool):
                return 1.0 if sheltered else 0.0
            return _clamp(float(sheltered), 0.0, 1.0)
        if self.world is not None:
            return compute_shelter_level(self.world)
        return 0.0

    def current_wet_level(
        self,
        *,
        wet: float | bool | None = None,
        sheltered: float | bool | None = None,
        weather: Any = None,
    ) -> float:
        if wet is not None:
            return (1.0 if wet else 0.0) if isinstance(wet, bool) else _clamp(float(wet), 0.0, 1.0)
        if self._explicit_wet is not None:
            return self._explicit_wet

        level = self.wetness
        if self.world is not None:
            shelter = self.current_shelter_level(sheltered=sheltered)
            tile = self.world.terrain.get_tile(self.world.body.position)
            if tile.type == "water" and self.world.body.level == "ground":
                level = max(level, 1.0)
            w_type, w_int = extract_weather_conditions(self.world, weather=weather)
            if w_type == "storm":
                level = max(level, w_int * (1.0 - 0.85 * shelter))
            elif w_type == "rain":
                level = max(level, 0.8 * w_int * (1.0 - 0.85 * shelter))
            held_id = self.world.body.holding
            if held_id is not None and held_id in self.world.objects:
                if self.world.objects[held_id].state.get("wet") is True:
                    level = max(level, 0.35)
        return _clamp(level, 0.0, 1.0)

    def current_cold_level(
        self,
        *,
        cold: float | bool | None = None,
        temperature: float | None = None,
        sheltered: float | bool | None = None,
        weather: Any = None,
    ) -> float:
        if cold is not None:
            return (1.0 if cold else 0.0) if isinstance(cold, bool) else _clamp(float(cold), 0.0, 1.0)
        if self._explicit_cold is not None:
            return self._explicit_cold

        shelter = self.current_shelter_level(sheltered=sheltered)
        temp = temperature
        if temp is None and self.world is not None:
            temp = self.world.ambient_temperature
        if temp is None:
            temp = 1.0

        level = self.coldness
        w_type, w_int = extract_weather_conditions(self.world, weather=weather)
        if w_type == "cold":
            level = max(level, w_int * (1.0 - 0.65 * shelter))
        elif w_type == "storm":
            level = max(level, 0.6 * w_int * (1.0 - 0.65 * shelter))

        if temp < self.config.comfort_neutral_temperature:
            temp_cold = (self.config.comfort_neutral_temperature - temp) / 0.50
            level = max(level, _clamp(temp_cold, 0.0, 1.0) * (1.0 - 0.60 * shelter))

        return _clamp(level, 0.0, 1.0)

    @property
    def modifiers(self) -> BodyModifiers:
        """Current ``BodyModifiers`` derived from energy, comfort, cold, and wet."""
        energy = self.world.body.energy if self.world is not None else 1.0
        comfort = self.world.body.comfort if self.world is not None else 1.0
        return compute_body_modifiers(
            energy=energy,
            comfort=comfort,
            cold=self.current_cold_level(),
            wet=self.current_wet_level(),
            config=self.config,
            world=self.world,
        )

    def record_activity(
        self,
        verb: str,
        *,
        force: float | None = None,
        speed: float | None = None,
    ) -> float:
        """Drain energy immediately for an executed action, weighted by ``verb`` and ``force``."""
        cost = activity_energy_cost(verb, force=force, speed=speed, config=self.config)
        if cost > 0.0 and self.world is not None:
            self.resting = False
            self.world.body.energy = _clamp(self.world.body.energy - cost, 0.0, 1.0)
            self._activity_this_tick += cost
        return cost

    def rest(self, ticks: int = 1) -> tuple[float, float]:
        """Restore energy and comfort via rest (idle). Returns ``(energy, comfort)``."""
        if ticks < 1:
            raise ValueError("ticks must be >= 1")
        if self.world is None:
            return (1.0, 1.0)
        self.resting = True
        for _ in range(ticks):
            self.world.body.energy = _clamp(
                self.world.body.energy + self.config.energy_rest_recovery,
                0.0,
                1.0,
            )
            self.world.body.comfort = _clamp(
                self.world.body.comfort + self.config.comfort_rest_recovery,
                0.0,
                1.0,
            )
        return (self.world.body.energy, self.world.body.comfort)

    def eat(self, target: str | WorldObject | None = None) -> ActionResult:
        """Eat an object with an ``edible`` rule in ``self.world``."""
        if self.world is None:
            raise RuntimeError("BodyStateSystem.eat requires an attached World")
        return eat_object(self.world, target, config=self.config)

    def maybe_slip_held_item(
        self,
        *,
        modifiers: BodyModifiers | None = None,
    ) -> str | None:
        """If wet (``slip_chance > 0``) and holding an item, probabilistically drop it."""
        if self.world is None:
            return None
        held_id = self.world.body.holding
        if held_id is None or held_id not in self.world.objects:
            return None
        mods = modifiers if modifiers is not None else self.modifiers
        if mods.slip_chance <= 0.0:
            return None
        if self.world.rng.random() < mods.slip_chance:
            from arden.actions.verbs import drop as _verb_drop

            res = _verb_drop(self.world, held_id, modifiers=mods)
            if res.ok:
                return held_id
        return None

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
    ) -> BodyModifiers:
        """Advance energy, comfort, wetness, drives, and side effects by one tick."""
        if self.world is None:
            self.drives.step(novelty=novelty, surprise=surprise, config=self.config)
            return self.modifiers

        body = self.world.body
        cfg = self.config

        is_rest = (
            resting
            if resting is not None
            else (verb in ("rest", "idle") or self.resting)
        )

        # 1. Energy update: rest restores energy; otherwise passive time drain + any verb activity drain
        if is_rest and (verb is None or verb in ("rest", "idle")):
            body.energy = _clamp(body.energy + cfg.energy_rest_recovery, 0.0, 1.0)
        else:
            extra_activity = 0.0
            if verb is not None and verb not in ("rest", "idle"):
                extra_activity = activity_energy_cost(
                    verb,
                    force=force,
                    speed=speed,
                    config=cfg,
                )
            body.energy = _clamp(
                body.energy - cfg.energy_time_drain - extra_activity,
                0.0,
                1.0,
            )
        self._activity_this_tick = 0.0

        # 2. Environmental conditions for comfort
        shelter_level = self.current_shelter_level(sheltered=sheltered)
        w_type, w_int = extract_weather_conditions(self.world, weather=weather)

        if storm is not None:
            storm_level = (1.0 if storm else 0.0) if isinstance(storm, bool) else _clamp(float(storm), 0.0, 1.0)
        elif w_type == "storm":
            storm_level = w_int
        else:
            storm_level = 0.0
        storm_exposure = storm_level * (1.0 - 0.75 * shelter_level)

        if dry is True and wet is None:
            wet_level = 0.0
            self.wetness = max(0.0, self.wetness - 0.25)
        else:
            wet_level = self.current_wet_level(wet=wet, sheltered=sheltered, weather=weather)
            if wet is None and self._explicit_wet is None:
                tile = self.world.terrain.get_tile(body.position)
                in_water = tile.type == "water" and body.level == "ground"
                in_rain = w_type in ("rain", "storm") and shelter_level < 0.85
                if in_water or in_rain:
                    self.wetness = wet_level
                else:
                    self.wetness = max(0.0, self.wetness - 0.15 * (1.0 + shelter_level))
                    wet_level = self.wetness

        eff_temp = temperature if temperature is not None else self.world.ambient_temperature
        if warm is True:
            eff_temp = max(eff_temp, cfg.comfort_neutral_temperature + 0.15)
        elif isinstance(warm, int | float) and not isinstance(warm, bool) and float(warm) > 0.0:
            eff_temp = max(eff_temp, cfg.comfort_neutral_temperature + 0.15 * _clamp(float(warm), 0.0, 1.0))

        cold_level = self.current_cold_level(
            cold=cold,
            temperature=eff_temp,
            sheltered=sheltered,
            weather=weather,
        )

        # Night temperature factor
        light = self.world.light_level
        night_factor = _clamp(1.0 - light, 0.0, 1.0) * (1.0 - 0.70 * shelter_level)
        if eff_temp >= cfg.comfort_neutral_temperature and warm is not None:
            night_factor = 0.0

        # Low energy factor on comfort
        if body.energy < cfg.low_energy_comfort_threshold and cfg.low_energy_comfort_threshold > 0.0:
            low_energy_factor = (
                cfg.low_energy_comfort_threshold - body.energy
            ) / cfg.low_energy_comfort_threshold
        else:
            low_energy_factor = 0.0

        # Comfort drains
        comfort_drain = (
            cfg.comfort_cold_drain * cold_level
            + cfg.comfort_wet_drain * wet_level
            + cfg.comfort_night_drain * night_factor
            + cfg.comfort_storm_drain * storm_exposure
            + cfg.comfort_low_energy_drain * low_energy_factor
        )

        # Comfort recoveries (shelter, dryness, warmth, rest)
        dryness_factor = _clamp(1.0 - wet_level, 0.0, 1.0)
        if warm is True:
            warmth_factor = 1.0
        elif isinstance(warm, int | float) and not isinstance(warm, bool):
            warmth_factor = _clamp(float(warm), 0.0, 1.0)
        elif eff_temp >= cfg.comfort_neutral_temperature and cold_level == 0.0 and storm_exposure == 0.0:
            warmth_factor = _clamp(
                (eff_temp - cfg.comfort_neutral_temperature) / 0.15,
                0.25,
                1.0,
            )
        else:
            warmth_factor = 0.0

        comfort_recovery = (
            cfg.comfort_shelter_recovery * shelter_level
            + cfg.comfort_dry_recovery * dryness_factor
            + cfg.comfort_warm_recovery * warmth_factor
            + (cfg.comfort_rest_recovery if is_rest else 0.0)
        )

        body.comfort = _clamp(body.comfort - comfort_drain + comfort_recovery, 0.0, 1.0)

        # 3. Drives update
        self.drives.step(novelty=novelty, surprise=surprise, config=cfg)

        # 4. Compute modifiers and check wet slip
        mods = compute_body_modifiers(
            energy=body.energy,
            comfort=body.comfort,
            cold=cold_level,
            wet=wet_level,
            config=cfg,
            world=self.world,
        )
        if wet_level > 0.0 and body.holding is not None:
            self.maybe_slip_held_item(modifiers=mods)

        return mods

    @property
    def memory_salience(self) -> float:
        """Current episodic memory salience score combining drives, comfort, and energy."""
        body = self.world.body if self.world is not None else None
        return compute_memory_salience(drives=self.drives, body=body)

    def gate_triggers(self) -> list[str]:
        """List of drive/body triggers currently exceeding gate thresholds."""
        body = self.world.body if self.world is not None else None
        return self.drives.gate_triggers(body, config=self.config)

    @property
    def should_wake_gate(self) -> bool:
        """Return True when any drive or body threshold warrants waking the LLM."""
        return bool(self.gate_triggers())

    def to_dict(self) -> dict[str, Any]:
        """Serialize body system state for snapshots."""
        return {
            "config": self.config.model_dump(mode="json"),
            "drives": self.drives.to_dict(),
            "wetness": self.wetness,
            "coldness": self.coldness,
            "resting": self.resting,
            "explicit_wet": self._explicit_wet,
            "explicit_cold": self._explicit_cold,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        world: World | None = None,
    ) -> BodyStateSystem:
        """Restore a ``BodyStateSystem`` from ``to_dict()`` output."""
        cfg = BodyConfig.model_validate(data.get("config", {}))
        drives = DriveState.model_validate(data.get("drives", {}))
        inst = cls(
            world=world,
            config=cfg,
            drives=drives,
            wetness=float(data.get("wetness", 0.0)),
            coldness=float(data.get("coldness", 0.0)),
            resting=bool(data.get("resting", False)),
        )
        raw_ew = data.get("explicit_wet")
        raw_ec = data.get("explicit_cold")
        inst._explicit_wet = None if raw_ew is None else float(raw_ew)
        inst._explicit_cold = None if raw_ec is None else float(raw_ec)
        return inst


BodyStateModel = BodyStateSystem
BodyModel = BodyStateSystem


def __getattr__(name: str) -> Any:
    if name == "BodyModifiers":
        from arden.actions.verbs import BodyModifiers

        return BodyModifiers
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BODY_PRESETS",
    "DEFAULT_BODY_CONFIG",
    "DEFAULT_VERB_ENERGY_WEIGHTS",
    "GENTLE_BODY_CONFIG",
    "HARSH_BODY_CONFIG",
    "BodyConfig",
    "BodyModel",
    "BodySettings",
    "BodyStateModel",
    "BodyStateSystem",
    "DriveState",
    "DriveThresholds",
    "Drives",
    "PresetName",
    "activity_energy_cost",
    "compute_body_modifiers",
    "compute_memory_salience",
    "compute_shelter_level",
    "eat_object",
    "extract_weather_conditions",
    "format_numeric_body_state",
    "is_edible_object",
    "memory_salience",
    "resolve_body_config",
]

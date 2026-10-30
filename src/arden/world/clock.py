"""Fixed-tick world runner with pause, resume, speed scaling, and single-step."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from arden.world.world import World

_BUDGET_EPSILON = 1e-12


def _validate_positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    as_float = float(value)
    if not math.isfinite(as_float) or as_float <= 0.0:
        raise ValueError(f"{name} must be a finite number greater than 0")
    return as_float


class WorldClock:
    """Runner that drives ``World.step()`` at ``tick_rate * speed``.

    Can be driven synchronously in tests with no sleeping via ``step_once()``,
    ``run(ticks)``, or ``advance(dt)``.
    """

    def __init__(
        self,
        world: World,
        tick_rate: float = 10.0,
        *,
        speed: float = 1.0,
        paused: bool = False,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.world = world
        self.tick_rate = _validate_positive_finite(tick_rate, "tick_rate")
        self.speed = _validate_positive_finite(speed, "speed")
        self.paused = bool(paused)
        self._sleep_fn = sleep_fn
        self._tick_budget = 0.0

    @property
    def effective_tick_rate(self) -> float:
        """Target ticks per real-time second after applying ``speed``."""
        return self.tick_rate * self.speed

    @property
    def tick_interval(self) -> float:
        """Real-time seconds between ticks at the current ``tick_rate`` and ``speed``."""
        return 1.0 / (self.tick_rate * self.speed)

    def pause(self) -> None:
        """Pause automatic stepping."""
        self.paused = True
        self._tick_budget = 0.0

    def resume(self) -> None:
        """Resume automatic stepping."""
        self.paused = False

    def set_speed(self, speed: float) -> None:
        """Set the simulation speed multiplier (must be finite and > 0)."""
        self.speed = _validate_positive_finite(speed, "speed")

    def step_once(self) -> int:
        """Advance the world by exactly one tick, even while paused."""
        return self.world.step()

    def tick(self) -> int | None:
        """Advance one tick when unpaused; do nothing and return ``None`` when paused."""
        if self.paused:
            return None
        return self.world.step()

    def advance(self, dt: float) -> int:
        """Advance simulated real time by ``dt`` seconds with no sleeping.

        Returns the number of ticks executed. Does nothing while paused.
        """
        if isinstance(dt, bool) or not isinstance(dt, int | float):
            raise TypeError("dt must be a number")
        dt_float = float(dt)
        if not math.isfinite(dt_float) or dt_float < 0.0:
            raise ValueError("dt must be a finite non-negative number")
        if self.paused or dt_float == 0.0:
            return 0

        self._tick_budget += dt_float * self.tick_rate * self.speed
        steps = 0
        while not self.paused and self._tick_budget + _BUDGET_EPSILON >= 1.0:
            self._tick_budget = max(0.0, self._tick_budget - 1.0)
            self.world.step()
            steps += 1
        if self.paused:
            self._tick_budget = 0.0
        return steps

    def run(self, ticks: int = 1, *, sleep: bool = False) -> int:
        """Run up to ``ticks`` steps while not paused.

        By default runs synchronously with no sleeping unless ``sleep=True`` or
        a custom ``sleep_fn`` was provided. Returns the number of ticks executed.
        """
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 0:
            raise ValueError("ticks must be a non-negative integer")

        sleeper = self._sleep_fn if self._sleep_fn is not None else (time.sleep if sleep else None)
        steps = 0
        for _ in range(ticks):
            if self.paused:
                break
            self.world.step()
            steps += 1
            if sleeper is not None and not self.paused:
                sleeper(self.tick_interval)
        return steps

    def run_ticks(self, ticks: int) -> int:
        """Run up to ``ticks`` steps synchronously with no sleeping."""
        return self.run(ticks, sleep=False)


Clock = WorldClock

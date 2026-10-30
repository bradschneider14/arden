"""Intent JSON the LLM must return, and the body verbs a step may use.

The verb list is the body-level actions in the design doc (section 6).
``grab / drop`` is two verbs. Opaque actions such as ``action_3`` are not
part of this closed list.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

VERBS = (
    "move",
    "look",
    "push",
    "strike",
    "grab",
    "drop",
    "place",
    "stack",
    "bind",
    "insert",
    "separate",
    "speak",
)

Verb = Literal[
    "move",
    "look",
    "push",
    "strike",
    "grab",
    "drop",
    "place",
    "stack",
    "bind",
    "insert",
    "separate",
    "speak",
]


class Step(BaseModel):
    """One body verb the controller can run without another LLM call."""

    model_config = ConfigDict(extra="forbid")

    verb: Verb
    target: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Intent(BaseModel):
    """Structured intent the controller carries out until the next wake.

    ``confidence`` is how sure the agent is of ``expect``. The attention
    gate weights expectation surprise by this value. ``0`` and ``1`` are
    allowed.
    """

    model_config = ConfigDict(extra="forbid")

    goal: str
    steps: list[Step]
    wake_if: list[str]
    expect: str
    thought: str
    confidence: float = Field(ge=0.0, le=1.0)

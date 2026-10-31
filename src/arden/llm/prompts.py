"""Build compact decide, reflect, and name prompts with a token-budget guard.

Rules enforced by this module:
- Prompts never describe objects' semantic meanings or names.
- Prompts never say whether body values (energy, comfort) are good or bad.
- When a prompt exceeds the token budget, retrieved memories are truncated first.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from arden.actions.verbs import ActionResult
from arden.llm.base import require_role
from arden.llm.schemas import VERBS
from arden.perception.feed import BodyPercept, PerceivedObject, Percept
from arden.world.objects import Position, WorldObject
from arden.world.world import AgentBody

DEFAULT_MAX_TOKENS = 1024

_VERB_LIST_STR = ", ".join(VERBS)

DECIDE_SYSTEM_PROMPT = (
    "You are an embodied agent in a physical environment. "
    "You receive structured perception updates, your numeric body state, your self-model, "
    "retrieved memories, and the result of your previous intent. "
    f"Choose your next intent using only the body verbs ({_VERB_LIST_STR}). "
    "Reply with a single JSON object matching the Intent schema: "
    "goal, steps, wake_if, expect, thought, and confidence (0.0 to 1.0)."
)

REFLECT_SYSTEM_PROMPT = (
    "You are an embodied agent reflecting on recent observations, actions, and outcomes. "
    "Summarize salient episodes, update your theories from evidence, and refine your self-model. "
    "Reply with a single JSON object matching the requested schema."
)

NAME_SYSTEM_PROMPT = (
    "You are an embodied agent assigning your own working label to an object or pattern "
    "based only on its perceived physical properties and observed behavior. "
    "Reply with a single JSON object matching the requested schema."
)

_SYSTEM_PROMPTS: dict[str, str] = {
    "decide": DECIDE_SYSTEM_PROMPT,
    "reflect": REFLECT_SYSTEM_PROMPT,
    "name": NAME_SYSTEM_PROMPT,
}


@dataclass(frozen=True, slots=True)
class BuiltPrompt:
    """System and user prompt pair produced for one LLM role."""

    role: str
    system: str
    user: str

    def __iter__(self):
        """Allow unpacking as ``system, user = build_decide_prompt(...)``."""
        yield self.system
        yield self.user


def estimate_tokens(text: str) -> int:
    """Estimate token count for ``text`` without external tokenizer dependencies.

    Uses ``max(word_count, ceil(len(text) / 4))``, a standard conservative heuristic
    for English/JSON text with Qwen/tiktoken subword tokenizers.
    """
    if not text:
        return 0
    char_est = math.ceil(len(text) / 4)
    word_est = len(text.split())
    return max(char_est, word_est)


def _fmt_num(value: float) -> str:
    rounded = round(float(value), 4)
    if rounded == 0.0:
        return "0"
    return f"{rounded:.4f}".rstrip("0").rstrip(".")


def _fmt_pos(pos: Position | Mapping[str, Any] | Sequence[float]) -> str:
    if isinstance(pos, Position):
        return f"({_fmt_num(pos.x)},{_fmt_num(pos.y)})"
    if isinstance(pos, Mapping):
        return f"({_fmt_num(pos.get('x', 0.0))},{_fmt_num(pos.get('y', 0.0))})"
    if isinstance(pos, Sequence) and len(pos) >= 2:
        return f"({_fmt_num(pos[0])},{_fmt_num(pos[1])})"
    return "(0,0)"


def format_body_state(
    body_state: BodyPercept | AgentBody | Mapping[str, Any] | str | None,
) -> str:
    """Format body state as plain numbers and ids, with no evaluative words."""
    if body_state is None:
        return "none"
    if isinstance(body_state, str):
        stripped = body_state.strip()
        if stripped.startswith("body:"):
            stripped = stripped.removeprefix("body:").strip()
        return stripped or "none"
    if isinstance(body_state, BodyPercept | AgentBody):
        held = body_state.holding if body_state.holding is not None else "none"
        return (
            f"pos={_fmt_pos(body_state.position)} "
            f"level={body_state.level} "
            f"energy={_fmt_num(body_state.energy)} "
            f"comfort={_fmt_num(body_state.comfort)} "
            f"holding={held}"
        )
    if isinstance(body_state, Mapping):
        pos = body_state.get("position", {"x": 0.0, "y": 0.0})
        level = body_state.get("level", "ground")
        energy = body_state.get("energy", 1.0)
        comfort = body_state.get("comfort", 1.0)
        holding = body_state.get("holding")
        held = holding if holding is not None else "none"
        return (
            f"pos={_fmt_pos(pos)} "
            f"level={level} "
            f"energy={_fmt_num(energy)} "
            f"comfort={_fmt_num(comfort)} "
            f"holding={held}"
        )
    raise TypeError("body_state must be a BodyPercept, AgentBody, mapping, str, or None")


def format_percept(percept: Percept | PerceivedObject | str | None) -> str:
    """Format percept delta text without duplicating or labeling semantics."""
    if percept is None:
        return "none"
    if isinstance(percept, Percept):
        text = percept.to_text().strip()
        return text if text else "no changes"
    if isinstance(percept, PerceivedObject):
        return percept.to_text()
    if isinstance(percept, str):
        stripped = percept.strip()
        return stripped if stripped else "no changes"
    raise TypeError("percept must be a Percept, PerceivedObject, str, or None")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, WorldObject):
        return obj.id
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    return str(obj)


def format_previous_intent_result(
    previous_intent_result: ActionResult | Mapping[str, Any] | str | None,
) -> str:
    """Format the outcome of the previous intent or step."""
    if previous_intent_result is None:
        return "none"
    if isinstance(previous_intent_result, str):
        stripped = previous_intent_result.strip()
        return stripped if stripped else "none"
    if isinstance(previous_intent_result, ActionResult):
        parts = [f"ok={str(previous_intent_result.ok).lower()}"]
        if previous_intent_result.reason is not None:
            parts.append(f"reason={previous_intent_result.reason}")
        if previous_intent_result.data:
            parts.append(
                f"data={json.dumps(previous_intent_result.data, sort_keys=True, separators=(',', ':'), default=_json_default)}"
            )
        return " ".join(parts)
    if isinstance(previous_intent_result, Mapping):
        return json.dumps(
            dict(previous_intent_result),
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        )
    raise TypeError("previous_intent_result must be an ActionResult, mapping, str, or None")


def _format_memory_entry(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, Mapping):
        if "text" in entry and isinstance(entry["text"], str):
            return entry["text"].strip()
        return json.dumps(dict(entry), sort_keys=True, separators=(",", ":"))
    return str(entry).strip()


def _assemble_user_prompt(
    *,
    role: str,
    percept_str: str,
    body_str: str,
    self_model_str: str,
    memory_items: Sequence[str],
    prev_result_str: str,
    target_id: str | None = None,
) -> str:
    sections: list[str] = [f"role: {role}"]
    if target_id is not None:
        sections.append(f"target: {target_id}")
    sections.append(f"body_state:\n{body_str}")
    sections.append(f"percept_delta:\n{percept_str}")
    sections.append(f"previous_intent_result:\n{prev_result_str}")
    sections.append(f"self_model:\n{self_model_str}")
    if memory_items:
        mem_lines = "\n".join(f"- {item}" for item in memory_items)
        sections.append(f"memories:\n{mem_lines}")
    else:
        sections.append("memories: none")
    return "\n\n".join(sections)


def build_prompt(
    role: str,
    *,
    percept: Percept | PerceivedObject | str | None = None,
    self_model: str | None = None,
    memories: Sequence[Any] | None = None,
    body_state: BodyPercept | AgentBody | Mapping[str, Any] | str | None = None,
    previous_intent_result: ActionResult | Mapping[str, Any] | str | None = None,
    target: str | PerceivedObject | WorldObject | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BuiltPrompt:
    """Build a ``(system, user)`` prompt for ``decide``, ``reflect``, or ``name``.

    Includes:
    - the percept text (delta)
    - the self-model
    - a handful of retrieved memories (empty list until M5)
    - current body state (plain numbers, no good/bad labels)
    - the previous intent result

    Token-budget guard:
    If the total estimated tokens (system + user) exceed ``max_tokens``, memories are
    truncated from the end first until the prompt fits or no memories remain.
    """
    require_role(role)
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")

    if body_state is None and isinstance(percept, Percept):
        body_state = percept.body

    target_id: str | None = None
    if isinstance(target, PerceivedObject):
        target_id = target.id
        if percept is None:
            percept = target
    elif isinstance(target, WorldObject):
        target_id = target.id
    elif isinstance(target, str) and target.strip():
        target_id = target.strip()

    system = _SYSTEM_PROMPTS[role]
    percept_str = format_percept(percept)
    body_str = format_body_state(body_state)
    self_model_str = self_model.strip() if isinstance(self_model, str) and self_model.strip() else "none"
    prev_result_str = format_previous_intent_result(previous_intent_result)

    raw_memories = [_format_memory_entry(m) for m in (memories or ()) if _format_memory_entry(m)]

    # Token-budget guard: truncate memories from the tail first
    kept_memories = list(raw_memories)
    user = _assemble_user_prompt(
        role=role,
        percept_str=percept_str,
        body_str=body_str,
        self_model_str=self_model_str,
        memory_items=kept_memories,
        prev_result_str=prev_result_str,
        target_id=target_id,
    )

    while kept_memories and (estimate_tokens(system) + estimate_tokens(user) > max_tokens):
        kept_memories.pop()
        user = _assemble_user_prompt(
            role=role,
            percept_str=percept_str,
            body_str=body_str,
            self_model_str=self_model_str,
            memory_items=kept_memories,
            prev_result_str=prev_result_str,
            target_id=target_id,
        )

    return BuiltPrompt(role=role, system=system, user=user)


def build_decide_prompt(
    percept: Percept | PerceivedObject | str | None = None,
    self_model: str | None = None,
    memories: Sequence[Any] | None = None,
    body_state: BodyPercept | AgentBody | Mapping[str, Any] | str | None = None,
    previous_intent_result: ActionResult | Mapping[str, Any] | str | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BuiltPrompt:
    """Build the ``decide`` prompt."""
    return build_prompt(
        "decide",
        percept=percept,
        self_model=self_model,
        memories=memories,
        body_state=body_state,
        previous_intent_result=previous_intent_result,
        max_tokens=max_tokens,
    )


def build_reflect_prompt(
    percept: Percept | PerceivedObject | str | None = None,
    self_model: str | None = None,
    memories: Sequence[Any] | None = None,
    body_state: BodyPercept | AgentBody | Mapping[str, Any] | str | None = None,
    previous_intent_result: ActionResult | Mapping[str, Any] | str | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BuiltPrompt:
    """Build the ``reflect`` prompt."""
    return build_prompt(
        "reflect",
        percept=percept,
        self_model=self_model,
        memories=memories,
        body_state=body_state,
        previous_intent_result=previous_intent_result,
        max_tokens=max_tokens,
    )


def build_name_prompt(
    percept: Percept | PerceivedObject | str | None = None,
    self_model: str | None = None,
    memories: Sequence[Any] | None = None,
    body_state: BodyPercept | AgentBody | Mapping[str, Any] | str | None = None,
    previous_intent_result: ActionResult | Mapping[str, Any] | str | None = None,
    *,
    target: str | PerceivedObject | WorldObject | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BuiltPrompt:
    """Build the ``name`` prompt."""
    return build_prompt(
        "name",
        percept=percept,
        self_model=self_model,
        memories=memories,
        body_state=body_state,
        previous_intent_result=previous_intent_result,
        target=target,
        max_tokens=max_tokens,
    )

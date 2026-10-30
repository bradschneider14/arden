"""The one interface every LLM call goes through."""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

LLM_ROLES = ("decide", "reflect", "name")


class LLMOutputError(Exception):
    """The backend returned text that is not valid for the requested schema.

    ``raw`` is the text the backend produced, before parsing. It is kept
    even when that text is not JSON.
    """

    def __init__(self, raw: str, detail: str) -> None:
        self.raw = raw
        super().__init__(detail)


@runtime_checkable
class LLMBackend(Protocol):
    """Structured completion used for deciding, reflecting, and naming.

    Implementations parse their own text and validate it against ``schema``.
    Invalid output raises ``LLMOutputError`` carrying that text. ``role`` is
    one of ``decide``, ``reflect``, or ``name``: one model, three prompts.
    """

    def complete_json(
        self,
        role: str,
        system: str,
        user: str,
        schema: dict,
        timeout_s: float,
    ) -> dict:
        """Return a JSON object for ``role`` that validates against ``schema``."""


def require_role(role: str) -> None:
    """Reject a role this interface does not define."""
    if role not in LLM_ROLES:
        allowed = ", ".join(LLM_ROLES)
        raise ValueError(f"role must be one of: {allowed}")


def parse_json_output(raw: str, schema: dict) -> dict:
    """Parse ``raw`` and validate the object against a JSON Schema.

    The schema is the dict passed to ``complete_json``. This checks the subset
    Pydantic emits for the intent models: ``type``, ``enum``, ``required``,
    ``properties``, ``additionalProperties``, ``items``, ``minimum``,
    ``maximum``, ``anyOf``, and local ``$ref``.
    """
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(raw, f"LLM output is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise LLMOutputError(raw, "LLM output must be a JSON object")
    problems = _validate(value, schema, schema, "$")
    if problems:
        detail = "LLM output does not match the schema: " + "; ".join(problems)
        raise LLMOutputError(raw, detail)
    return value


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported schema $ref: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"Schema $ref not found: {ref}")
        node = node[part]
    if not isinstance(node, dict):
        raise TypeError(f"Schema $ref does not point at an object: {ref}")
    return node


def _is_type(instance: Any, type_name: str) -> bool:
    if type_name == "object":
        return isinstance(instance, dict)
    if type_name == "array":
        return isinstance(instance, list)
    if type_name == "string":
        return isinstance(instance, str)
    if type_name == "boolean":
        return isinstance(instance, bool)
    if type_name == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if type_name == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    if type_name == "null":
        return instance is None
    raise ValueError(f"Unsupported JSON Schema type: {type_name}")


def _validate(instance: Any, schema: dict[str, Any], root: dict[str, Any], path: str) -> list[str]:
    if "$ref" in schema:
        return _validate(instance, _resolve_ref(root, schema["$ref"]), root, path)

    problems: list[str] = []
    if "anyOf" in schema:
        branches = schema["anyOf"]
        if not any(not _validate(instance, branch, root, path) for branch in branches):
            return [f"{path} matched none of the allowed schemas"]

    if "enum" in schema and instance not in schema["enum"]:
        allowed = ", ".join(repr(item) for item in schema["enum"])
        problems.append(f"{path} must be one of: {allowed}")

    type_names: list[str] = []
    if "type" in schema:
        declared = schema["type"]
        type_names = [declared] if isinstance(declared, str) else list(declared)
        if not any(_is_type(instance, name) for name in type_names):
            problems.append(f"{path} must be of type {declared}")
            return problems

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and instance < minimum:
            problems.append(f"{path} must be >= {minimum}")
        if maximum is not None and instance > maximum:
            problems.append(f"{path} must be <= {maximum}")

    if isinstance(instance, dict) and (
        "object" in type_names or "properties" in schema or "additionalProperties" in schema
    ):
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in instance:
                problems.append(f"{path}.{name} is required")
        additional = schema.get("additionalProperties", True)
        for name, value in instance.items():
            if name in properties:
                problems.extend(_validate(value, properties[name], root, f"{path}.{name}"))
            elif additional is False:
                problems.append(f"{path}.{name} is not allowed")
            elif isinstance(additional, dict):
                problems.extend(_validate(value, additional, root, f"{path}.{name}"))

    if isinstance(instance, list) and isinstance(schema.get("items"), dict):
        item_schema = schema["items"]
        for index, value in enumerate(instance):
            problems.extend(_validate(value, item_schema, root, f"{path}[{index}]"))

    return problems

"""OpenAI-compatible / Ollama HTTP backend with JSON Schema enforcement and repair retries."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any, Literal

from arden.llm.base import LLMOutputError, parse_json_output, require_role

TransportFn = Callable[[str, dict[str, Any], float], dict[str, Any] | str]
SchemaMode = Literal["json_schema", "ollama"]


def _default_transport(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any] | str:
    """POST ``payload`` as JSON to ``url`` and return the decoded JSON response."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw_bytes = response.read()
    except urllib.error.HTTPError as exc:
        err_text = ""
        try:
            err_text = exc.read().decode("utf-8", errors="replace")
        except OSError:
            err_text = str(exc)
        raise LLMOutputError(
            err_text,
            f"HTTP {exc.code} from LLM endpoint {url}: {exc.reason}",
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise LLMOutputError(
            "",
            f"Failed to reach LLM endpoint {url}: {exc}",
        ) from exc

    text = raw_bytes.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(
            text,
            f"HTTP response from {url} is not valid JSON: {exc.msg}",
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMOutputError(text, f"HTTP response from {url} must be a JSON object")
    return parsed


def _extract_assistant_text(response: dict[str, Any] | str) -> str:
    """Extract the assistant message text from an OpenAI or Ollama chat response."""
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return response
        if not isinstance(parsed, dict):
            return response
        response = parsed

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(first.get("text"), str):
                return first["text"]

    message = response.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]

    if isinstance(response.get("response"), str):
        return response["response"]

    if isinstance(response.get("content"), str):
        return response["content"]

    raw_dump = json.dumps(response, sort_keys=True)
    raise LLMOutputError(raw_dump, "LLM response is missing assistant message content")


def _build_repair_message(error_detail: str) -> str:
    """Construct the user repair prompt sent after an invalid model response."""
    return (
        f"Your previous response was invalid: {error_detail}. "
        "Reply with ONLY a valid JSON object that strictly conforms to the requested schema."
    )


class OpenAICompatLLM:
    """Local LLM backend speaking OpenAI-compatible ``/v1/chat/completions`` or Ollama ``/api/chat``.

    Supports:
    - OpenAI-compatible ``response_format`` with ``{"type": "json_schema", ...}`` (default)
    - Ollama native ``format`` carrying the JSON schema directly (when ``schema_mode="ollama"``
      or when ``base_url`` ends with ``/api`` or ``/api/chat``).
    - Up to ``max_retries`` repair attempts when output is malformed JSON or violates ``schema``.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434/v1",
        model: str = "qwen2.5:7b-instruct",
        temperature: float = 0.2,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        schema_mode: SchemaMode | None = None,
        transport: TransportFn | None = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not model or not model.strip():
            raise ValueError("model must be a non-empty string")
        if temperature < 0.0 or temperature > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = float(temperature)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self._transport: TransportFn = transport if transport is not None else _default_transport

        if schema_mode is not None:
            if schema_mode not in ("json_schema", "ollama"):
                raise ValueError("schema_mode must be 'json_schema' or 'ollama'")
            self.schema_mode: SchemaMode = schema_mode
        elif self.base_url.endswith("/api") or self.base_url.endswith("/api/chat"):
            self.schema_mode = "ollama"
        else:
            self.schema_mode = "json_schema"

    @property
    def endpoint_url(self) -> str:
        """Full chat completions URL derived from ``base_url``."""
        if self.base_url.endswith("/chat/completions") or self.base_url.endswith("/api/chat"):
            return self.base_url
        if self.base_url.endswith("/api"):
            return f"{self.base_url}/chat"
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        if self.schema_mode == "ollama":
            return f"{self.base_url}/api/chat"
        return f"{self.base_url}/v1/chat/completions"

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        role: str,
    ) -> dict[str, Any]:
        schema_name = str(schema.get("title") or role or "response")
        if self.schema_mode == "ollama":
            return {
                "model": self.model,
                "messages": list(messages),
                "stream": False,
                "format": schema,
                "options": {"temperature": self.temperature},
            }
        return {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        }

    def complete_json(
        self,
        role: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Request a structured JSON completion and retry with a repair prompt on invalid output."""
        require_role(role)
        effective_timeout = self.timeout_s if timeout_s is None or timeout_s <= 0 else float(timeout_s)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error: LLMOutputError | None = None
        total_attempts = 1 + self.max_retries

        for _attempt in range(total_attempts):
            payload = self._build_payload(messages, schema, role)
            response = self._transport(self.endpoint_url, payload, effective_timeout)
            raw_text = _extract_assistant_text(response)
            try:
                return parse_json_output(raw_text, schema)
            except LLMOutputError as exc:
                last_error = exc
                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": _build_repair_message(str(exc))})

        assert last_error is not None
        raise last_error


OllamaLLM = OpenAICompatLLM

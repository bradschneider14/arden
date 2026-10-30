"""Build the LLM backend named in config."""

from __future__ import annotations

from arden.config import Settings
from arden.llm.base import LLMBackend
from arden.llm.mock import MockLLM


def make_backend(settings: Settings) -> LLMBackend:
    """Return the backend selected by ``settings.llm.backend``.

    ``mock`` is the only backend in this milestone. It is seeded from
    ``settings.world.seed``. Any other name raises ``ValueError``.
    """
    name = settings.llm.backend
    if name == "mock":
        return MockLLM(seed=settings.world.seed)
    raise ValueError(
        f"Unknown LLM backend {name!r}. The only backend in this milestone is 'mock'."
    )

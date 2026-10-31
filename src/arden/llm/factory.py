"""Build the LLM backend named in config."""

from __future__ import annotations

from arden.config import Settings
from arden.llm.base import LLMBackend
from arden.llm.mock import MockLLM
from arden.llm.openai_compat import OpenAICompatLLM

_OPENAI_COMPAT_BACKENDS = frozenset({"ollama", "openai_compat", "openai-compat", "llama_cpp", "llama.cpp"})


def make_backend(settings: Settings) -> LLMBackend:
    """Return the backend selected by ``settings.llm.backend``.

    - ``mock`` returns ``MockLLM`` seeded from ``settings.world.seed``.
    - ``ollama`` (or ``openai_compat`` / ``llama_cpp``) returns ``OpenAICompatLLM``
      configured from ``settings.llm``.
    - Any other name raises ``ValueError``.
    """
    name = settings.llm.backend
    if name == "mock":
        return MockLLM(seed=settings.world.seed)
    if name in _OPENAI_COMPAT_BACKENDS:
        return OpenAICompatLLM(
            base_url=settings.llm.base_url,
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            timeout_s=settings.llm.timeout_s,
            max_retries=settings.llm.max_retries,
        )
    raise ValueError(
        f"Unknown LLM backend {name!r}. Supported backends are 'mock' and 'ollama' ('openai_compat')."
    )

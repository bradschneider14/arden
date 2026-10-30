# Agent conventions

`docs/DESIGN.md` is the source of truth for Arden. Read the sections that apply to the ticket before changing code. If a ticket is ambiguous or conflicts with that document, stop and report it rather than guessing.

## Language and layout

- Python 3.11+.
- The installable package lives under `src/arden/`.
- Use type hints.
- Tests use pytest.

## Cross-platform

The project runs on Windows and Linux. Use `pathlib` for filesystem paths. Do not add shell-specific scripts.

## Tests

Tests never need a GPU, a network connection, or a running LLM. Use the mock LLM backend.

## Provisional decisions

Provisional decisions made in tickets are recorded in `docs/decisions.md`.

# Decisions

Provisional decisions made while implementing tickets are recorded here.

## T02 — Config, CLI, and RNG

The design doc does not specify a config schema. These defaults are provisional and easy to widen later.

- Package version is `0.1.0` until a release process exists.
- Settings live in `arden.toml` in the working directory. If that file is absent, `load_config()` returns built-in defaults. `arden.example.toml` records those defaults. An explicit path that does not exist raises `FileNotFoundError`.
- `[world].tick_rate` is simulation ticks per real-time second. Default `10.0`. It must be greater than 0. The loop may still run slower than real time when the LLM is thinking; this value is the target rate.
- `[world].seed` is an integer. Default `0`. `make_rng(settings)` is the only place that constructs the simulation `random.Random`. Callers pass that instance through the run. The process-wide `random` module is not seeded or sampled.
- `[llm].backend` is a non-empty string. Default `"mock"`. Other names are accepted so a later backend is a config change. Only the mock backend is implemented in this milestone.
- `[logging].level` is a standard-library level name (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`), matched case-insensitively and stored uppercased. Default `INFO`. This is process logging, separate from the ground-truth observer log.
- Unknown keys are rejected.
- `python -m arden run` loads config, builds the seeded RNG, and passes both into `run_simulation`. The world loop is not implemented yet, so the command prints `not implemented yet`.

## T03 — LLM backend, intent schema, and mock

The hardware section says the model replies with "intent, target, parameters, a short thought". That is treated as a summary of the intent contract in section 10.3. The schema implemented here is the ticket's `Intent` model, which follows that contract and adds `confidence`.

- Body verbs are the twelve names in section 6. The table cell `grab / drop` is two verbs, `grab` and `drop`. Opaque actions such as `action_3` are not in this enum. Widening the list for those is left to a later ticket.
- `Step.target` defaults to `None` and `Step.params` defaults to `{}`. Some verbs, such as `move`, `place`, and `speak`, do not take a target. Extra fields on `Intent` and `Step` are rejected. Keys inside `params` are open, because the ticket types `params` as a dict rather than a per-verb object.
- `confidence` is required and must be from 0 to 1 inclusive. Section 10.4 weights expectation surprise by confidence. The section 10.3 table does not name the field; the ticket does.
- `make_backend(settings)` reads `settings.llm.backend`. `mock` returns `MockLLM` seeded from `settings.world.seed`. Any other name raises `ValueError`. Config may still store a future name; the factory is what rejects it until that backend exists.
- `MockLLM` uses a private `random.Random` for fixture draws. It does not consume the simulation RNG from `make_rng`. `system`, `user`, and `timeout_s` do not affect the choice, and the mock returns immediately.
- A script queue of raw strings is consumed from the front before any fixture draw. Those strings may be invalid JSON. A scripted call does not advance the seed. Fixture files are `src/arden/llm/fixtures/*.json`, sorted by filename.
- Every response is parsed as JSON and checked against the `schema` argument. The checker covers the JSON Schema subset Pydantic emits for these models (`type`, `enum`, `required`, `properties`, `additionalProperties`, `items`, `minimum`, `maximum`, `anyOf`, and local `$ref`). Invalid output raises `LLMOutputError` with the raw text on `.raw`. No schema library was added.
- `role` must be `decide`, `reflect`, or `name`. Anything else raises `ValueError` and does not consume a scripted response. Reflect and name do not have their own models yet. Callers pass whatever schema those roles should return. The canned fixtures are intents.

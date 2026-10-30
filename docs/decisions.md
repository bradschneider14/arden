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

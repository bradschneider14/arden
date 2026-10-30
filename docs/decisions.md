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

## T04 — World object schema

The ticket names the fields. These choices fill in representation details the design doc leaves open.

- An id is `object_<n>` with `n` a non-negative integer (`object_0`, `object_1`, …). `ObjectFactory` starts at `object_1`. It accepts an explicit id in that form, remembers ids it has issued, and rejects a duplicate. Constructing a `WorldObject` directly does not consult the factory.
- Position is `{"x": float, "y": float}` on the 2D plane. Coordinates must be finite. There is no world bound in this schema.
- `level` is the string `ground`, `raised`, `on_top_of:object_<n>`, or `inside:object_<n>`. The schema checks that shape only. It does not check that the referenced object exists. `inside` is included because the ticket requires it and section 4.1 allows putting things inside.
- `shape` is `sphere`, `cube`, `cylinder`, `flat`, or `long`. `long` is required by the ticket and matches the composite description in section 7.2.
- Height, volume, and weight are finite and `>= 0`. Rigidity is finite and in `0..1` inclusive. Zero weight is allowed; only negatives are rejected. The design doc states the rigidity range and forbids a negative weight; the other bounds keep physical magnitudes in the same family.
- `color`, when set, is a non-empty string. No palette is enforced. Unset color is left out of `perceptible()`.
- `state` is a free JSON object and is perceptible. Hidden rules do not go there.
- Hidden behavior is `behavior`: a list of `BehaviorRule` values, so two responses can share a verb (bounce and break on strike). Each rule has a `verb` key matching `^[a-z][a-z0-9_]*$` and any other JSON-valued fields as data (`scale`, `response`, `above_force`, and so on). Verbs are not limited to the current body-verb list. There is one rule model, not a subclass per object kind.
- `perceptible()` returns the physical fields: id, position, level, shape, height, volume, weight, rigidity, binding, hollow, fragile, color (if set), fixed, and state. It omits `behavior`. The id is included as a referent, as in the intent examples that say `object_7`. It is not turned into a label. Qualitative words such as "round" or "light" are not derived here; those belong to a later perception rendering if they are wanted. Section 2 says physical parameters are exposed and their effects are learned.
- `fixed` is a perceptible flag, separate from hidden behavior. The first-run tree is described as fixed.
- Unknown keys, including `name` or `label`, are rejected so a human label cannot be stored on the object.

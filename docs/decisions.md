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

## T05 — World state, fixed tick loop, clock, and save/load

These choices fill in container, event, clock, and snapshot details not fixed by the design doc.

- `AgentBody` holds `position` (default `(0.0, 0.0)`), `level` (default `"ground"`), `energy` (in `0..1`, default `1.0`), `comfort` (in `0..1`, default `1.0`), and `holding` (`None` or an `object_<n>` id). Despawning a held object clears `holding`.
- `World` stores objects by id in `world.objects`, the agent body on `world.body` (aliased as `world.agent_body`), the tick counter on `world.tick`, the seeded `random.Random` on `world.rng`, and an internal `ObjectFactory` so creator spawns issue sequential `object_<n>` ids. `step()` increments `tick` by 1, advances `rng` once per tick, and emits a `tick` event.
- `EventBus` (`world.events`) delivers typed `WorldEvent` subclasses (`ObjectSpawnedEvent`, `ObjectDespawnedEvent`, `ObjectMovedEvent`, `ObjectChangedEvent`, `TickEvent`) synchronously in registration order. Subscribers can listen to all events (`world.subscribe(fn)`) or a single event type (`world.subscribe("object_spawned", fn)`).
- Creator operations `world.spawn(props, pos)`, `world.despawn(id)`, and `world.move(id, pos)` default to `source="creator"` on the emitted event. `world.update_object(id, changes)` updates properties/state and emits `object_changed` (default `source="world"`).
- `WorldClock` (`Clock`) wraps a `World` and `tick_rate`, supporting `pause()`, `resume()`, `set_speed(x)` (`x > 0`), `step_once()` (steps 1 tick even while paused), `advance(dt)` (accumulates simulated seconds without sleeping), and `run(ticks)` (runs up to `ticks` steps while unpaused, defaulting to no sleeping for synchronous tests).
- `snapshot()` serializes `tick`, `body`, `objects` (including hidden `behavior` rules), `factory` id state, and `rng.getstate()` to a deterministic JSON string (`sort_keys=True`). `restore()` works on both an existing `World` instance and on the `World` class.

## T06 — 2D physics: movement, collisions, pushing, rolling, and height levels

Section 16 (open question 6) leaves open whether to use simple property rules or a light physics engine. We choose deterministic property-driven rules on the 2D plane with discrete height levels rather than an external physics engine:

- **Property-driven rules over a physics engine**: Movement, rolling, collisions, stacking stability, water drift, and occlusion are computed directly from `WorldObject` properties (`shape`, `weight`, `height`, `volume`, `rigidity`, `fixed`, `level`) with no external physics dependency. Physics does not consume draws from `world.rng`, preserving exact tick-for-tick RNG determinism.
- **Terrain grid (`TerrainGrid`, `Tile`)**: Unbounded sparse grid of cells of size `tile_size` (default `1.0`), defaulting to `"grass"`. Supports `"grass"`, `"water"` (wet; default flow `(1.0, 0.0)` unless overridden; movable objects with `weight <= 1.0` float with `state["floating"] = True` and drift downstream each tick; all ground objects in water get `state["wet"] = True`), and `"ledge"` (`level = "raised"`, elevation `1.0`; reachable only from an adjacent raised tile or from `on_top_of:object_<n>` when the support object is adjacent and has `height >= 0.15`; blocks ground-level objects from sliding onto the ledge).
- **Velocity, impulse, and shape friction**: `WorldObject.velocity` (`Velocity(x, y)`, omitted from `perceptible()`) updates positions each tick and decays by shape-dependent retention (`sphere: 0.85`, `cylinder: 0.80`, `long: 0.65`, `cube: 0.55`, `flat: 0.50`, zeroed below `0.005`). `apply_impulse(id, direction, force)` adds velocity `(force * shape_factor * rule_scale) / (1.0 + weight)` along the normalized direction (`shape_factor`: `sphere: 1.2`, `cylinder: 1.1`, `long: 0.85`, `cube: 0.7`, `flat: 0.6`), so round objects roll farther than flat/cube shapes and heavier objects move less.
- **Circle collisions**: Each object's collision radius (`obj.radius` / `object_radius(obj)`) is derived from `shape`, `height`, and `volume` (clamped to `[0.05, 2.0]`, or overridden by `state["radius"]`). Fast movement is substepped, momentum is exchanged on impact (`fixed` objects never move), and iterative positional projection with a fallback de-wedging pass guarantees no two active objects on the same level overlap after a step.
- **Stacking (`on_top_of`) stability**: An object with `level = "on_top_of:object_<n>"` is statically stable when the lower object is not a `"sphere"`, `upper.radius <= lower.radius * 1.5`, `upper.weight <= lower.weight * 2.0` (unless `lower.fixed`), and `not (lower.rigidity < 0.2 and upper.weight > lower.weight)`. When the lower object moves, `upper` is carried along if the step displacement is within `max_carry_step(upper, lower)`; otherwise (or if statically unstable) `upper` topples beside the support onto the plane without overlapping.
- **Approximate occlusion (`is_occluded`)**: `is_occluded(viewer_pos, target)` returns `True` when another object has a strictly greater top height (`elevation + height`) than both `target` and the viewer's base elevation, projects strictly between `viewer_pos` and `target` ($0 < t < 1$), and lies within `max(blocker.radius, tolerance)` (default `tolerance = 0.5`) of the line segment.

## T07 — Perception feed: nearby objects, events, body state, affordances, and deltas

These choices fill in configuration defaults, compass bearings, visibility scaling, delta tracking, and text formatting for the perception feed:

- **Perception config (`PerceptionSettings`)**: Added `[perception]` settings to `Settings` (and `arden.example.toml`) with `sensing_radius = 10.0`, `reach_radius = 1.5`, `move_threshold = 0.1`, and `color_visibility_threshold = 0.75`.
- **Compass bearings**: `compass_bearing(from_pos, to_pos)` maps relative 2D offsets to 8-wind compass bearings (`N`, `NE`, `E`, `SE`, `S`, `SW`, `W`, `NW`, with `+y` as `N` and `+x` as `E`) and `"here"` when distance is `0`.
- **Visibility factor and detail filtering**: Effective sensing radius is `sensing_radius * visibility_factor` (`visibility_factor` in `[0.0, 1.0]`, default `1.0`). Occluded objects (`world.is_occluded(body, obj)`) are omitted. When `visibility_factor < 1.0`, fine detail (`color`) is dropped for objects beyond `sensing_radius * (visibility_factor ** 2)` and dropped for all objects when `visibility_factor <= color_visibility_threshold` (`0.75`), while coarse geometry (`shape`, `height`, `volume`, `weight`, `level`, `fixed`) remains until the object leaves the shrunk sensing radius.
- **Affordances within reach**: For perceived objects within `reach_radius` (or currently held), `Percept.affordances` maps `object_<n>` to sorted verb/action tokens. Standard physical verbs derive from object properties and `behavior` rules; non-physical verbs in `behavior` are exposed only as opaque `action_<n>` tokens with all hidden rule parameters and descriptions stripped.
- **Sanitized events and `PerceptionTracker` deltas**: Events in `Percept.events` strip `behavior`, `velocity`, and `source` (`"creator"`). `PerceptionTracker` tracks visible objects across calls, returning a full percept on its first call (or on `full_refresh()` / `perceive(full=True)`) and delta percepts on subsequent calls containing only `appeared`, `disappeared`, `moved` (beyond `move_threshold`), and `changed` objects. When nothing changes, the delta `Percept` has `is_empty = True`, `bool(percept) = False`, and `to_text() == ""`.
- **`look(target)` and `Percept.to_text()`**: `look` returns a `PerceivedObject` with all perceptible properties for a single target object (`look(target)` or `look(world, target)`). `Percept.to_text()` renders a deterministic, token-cheap multi-line format locked by `tests/fixtures/percept_golden.txt`.


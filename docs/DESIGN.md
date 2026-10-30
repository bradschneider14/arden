# Arden: Design Doc

> System and architecture decisions for a locally run virtual world where an embodied agent explores, learns about itself, and communicates with its creator, with no assigned objective.
>
> Source of truth: the "Arden: Design Doc" Page. This Markdown copy is what coding agents read; keep the two in sync.

## Contents
1. Vision and goals
2. Decisions so far
3. System architecture
4. World and objects (2.5D world, weather, day and night)
5. Perception
6. Actions
7. Building
8. First-run environment
9. Cognition and drives (body state and comfort)
10. Attention and gating (principles, wake triggers, intent contract, surprise, world model, rollout)
11. Memory
12. Creator interface and the mailbox
13. Observability
14. Web interface
15. Hardware and local models
16. Open questions

## 1. Vision and goals

The project is called Arden, after the forest in Shakespeare's *As You Like It*, where people who leave ordinary life behind find out who they are. It is a sandbox, something like a pet simulator, where a virtually embodied agent is left to its own devices. It has no objective. The point is to observe what it becomes: a builder that requests tools, an explorer, a hobbyist, or something unexpected.

- Behavior should emerge from drives, perception, and memory, never from scripted goals.
- The agent learns about itself and its world through experimentation, not given knowledge.
- The creator shapes the world by spawning and despawning objects and answering messages.
- Everything runs locally on the creator's hardware.

## 2. Decisions so far

| Area | Decision |
|---|---|
| Mind | Hybrid: a fast low-level controller for movement and action execution, a local LLM for higher-level thinking, naming, and reflection. |
| Runtime | Fully local. Perception is structured text, not image-based vision. |
| World | 2.5D: movement and physics on a 2D plane with a few discrete height levels; objects are described in 3D terms; top-down or isometric visuals. |
| Creator stance | Responds at will and is sometimes silent. Avoids giving away semantics or too much information. Treated as experimental. |
| Model family | Qwen instruct models, chosen for strong tool calling and structured output. |
| Building | Fully open: no recipes. Materials share one set of physical properties, composites derive theirs from parts and joins, and the agent invents the purposes. |
| Weather | Set manually by the creator, or run automatically from a probabilistic configuration. Both modes can be combined. |
| Day and night | A continuous cycle that lowers visibility and temperature at night, and combines with weather. |
| Body needs | No hunger. Energy and comfort are numeric body-state values with no labels; their meaning is discovered through mild physical side effects. |
| First run | A small natural clearing with about a dozen object types; new things only arrive through the mailbox or creator spawns. |
| Interface | Web-based: a headless simulation server with a browser client for watching and creator controls, built local-first and remote-ready. |
| Hardware target | Start accessible: one 7B to 8B model on a 16 GB consumer GPU, scaling up only once the loop proves itself. |
| Object knowledge | Objects expose perceptible properties, not semantic names. The agent names things itself. |
| Actions | A fixed set of body-level verbs. Objects decide which verbs do something and what happens. |
| Parameters | Physical parameters with known ranges are exposed; their effects must be learned. |
| Novel affordances | Things with no physical analogue (the mailbox) appear as opaque actions with no explanation. |
| Memory | Capped short-term buffer that forces reflection into episodic and semantic tiers. |
| Attention | A cheap gate decides when the LLM thinks, never what the agent does. It errs toward waking the LLM. |
| World model | A small learned predictor whose prediction error is the surprise signal, used for both gating and curiosity. No planning at first. |

## 3. System architecture

```
Creator console --edits--> World simulation --(dashed)--> Observer log (full ground truth)
World simulation --> Perception layer (structured text deltas)
Perception layer --> Low-level controller (navigation, action execution)
Perception layer --> World model (predicts next state, measures surprise)
World model --surprise--> Attention gate (continue or wake the LLM)
Attention gate --wake--> LLM cognition (decide, name, reflect)
LLM cognition --wake conditions (dashed)--> Attention gate
LLM cognition <--> Memory system (short-term, episodic, beliefs)
LLM cognition --intents--> Low-level controller
Low-level controller --actions--> World simulation
```

The world runs on a fixed tick. The controller acts every tick; the attention gate checks every tick and wakes the LLM only on a salient event or a slow heartbeat (see Attention and gating). It issues intents such as "go to the round thing and push it", which the controller carries out.

The simulation is 2.5D (see World and objects), which keeps it light enough to run beside the LLM on one machine. The simulation runs headless as a server, separate from rendering, and the visuals are drawn in a browser client (see Web interface). That favors a custom Python simulation over a game engine, and keeps the visuals swappable without touching the agent.

## 4. World and objects

Each object is a bundle of perceptible properties plus hidden behavior rules that respond to body verbs. The agent sees properties; it must discover behavior.

| Field | Example |
|---|---|
| Internal id | `object_7` (never shown as a name) |
| Perceptible | round, small, light, red |
| Hidden behavior | push with force f moves it roughly f times 4 meters; strike makes it bounce |
| Affordances | push, grab, strike respond; bind and speak do nothing |

Objects can be spawned and despawned by the creator at any time. Composite objects (things the agent builds) inherit and combine properties of their parts.

### 4.1 2.5D world model

Full 3D simulation is too heavy for the local target, but a flat world makes objects hard to tell apart. The compromise is to simulate in 2D and describe in 3D.

- Movement and physics happen on a 2D plane (x, y), which keeps navigation, collisions, and pushing simple.
- Each object has a height and a small number of discrete levels (ground, raised, on top of), enough for stacking, climbing onto, and putting things inside.
- Objects are described with 3D properties: shape (sphere, cube, cylinder, flat), height, volume, weight, and whether they are hollow, so a ball and a coin are easy to tell apart.
- Occlusion is approximate: a tall object can hide smaller ones behind it, which keeps exploring worthwhile without real line-of-sight geometry.
- Visualization is top-down or isometric for the creator only. The agent never sees it, so the art style never affects behavior.

If 3D is wanted later, the same object schema and perception feed can drive a 3D renderer or physics engine without changing the agent.

### 4.2 Weather and world events

Weather is a world event that changes conditions the agent can sense, such as light, wetness, temperature, and wind, and that can act on objects, like wind moving light things or rain making surfaces slippery. It has two control modes.

| Mode | How it works |
|---|---|
| Manual | The creator sets the weather from the console, now or at a scheduled time, with an intensity and duration. |
| Automated | A configuration drives weather probabilistically: each weather type has a chance of starting per period, a duration range, an intensity range, and allowed transitions. |
| Combined | Automation runs by default; a manual event overrides it for its duration, then automation resumes. Automation can be paused. |

| Setting | Meaning |
|---|---|
| types | clear, rain, wind, cold, storm |
| chance | Probability of each type starting, per in-world hour |
| duration | Minimum and maximum length per type |
| intensity | Range per type, from 0 to 1 |
| transitions | Which types can follow which, such as rain to storm but never clear to storm |
| seed | Random seed, so an experiment's weather can be replayed exactly |

Every weather change is written to the observer log with whether it was manual or automated, so the agent's reactions can be tied to their cause. The same two-mode pattern can be reused later for resource regrowth.

### 4.3 Day and night

The world runs a continuous day and night cycle. Light and temperature follow it smoothly rather than switching at fixed moments, so dusk and dawn are gradual.

| Effect | Day | Night |
|---|---|---|
| Visibility | Full sensing radius | Sensing radius shrinks; distant objects lose detail first, then disappear |
| Detail | All perceptible properties | Fine properties like color fade; shape and size remain up close |
| Temperature | Warmer | Cooler, lowering comfort over time |

- Darkness changes perception, not the world: objects are still there, so the agent's memory of where things are becomes more valuable at night.
- Weather stacks on top of the cycle: a storm darkens the day, and cold weather at night is harsher.
- Nothing forces the agent to sleep. Lower comfort and reduced sensing may lead it to rest, which is also when reflection runs, so a daily rhythm can emerge on its own.
- Night creates its own friction for building: shelter that holds warmth, or later, light sources if fire is added.

| Setting | Meaning |
|---|---|
| day_length | Length of a full cycle in world ticks |
| daylight_share | Fraction of the cycle that is daytime |
| min_visibility | Sensing radius at darkest, as a fraction of the daytime radius |
| temperature_swing | How much colder night is than day |
| time | The creator can set the time of day directly, or freeze the cycle |

## 5. Perception

Perception is a compact structured text feed, sent as deltas since the last update to keep local inference cheap.

- Nearby objects with their perceptible properties, distance, and direction, within a sensing radius.
- Events: things that moved, appeared, disappeared, or changed state.
- Body state: position, energy, comfort, and what it is holding, given as plain numbers with no labels like good or bad.
- Available affordances for objects within reach, including opaque ones like `action_3`.

> No image-based vision. If richer perception is wanted later, add coarse senses (light level, sound direction) as text rather than pixels.

## 6. Actions

| Verb | Parameters | Notes |
|---|---|---|
| move | direction, speed | Handled by the controller |
| look | target | Focuses perception on one thing for more detail |
| push | target, force 0 to 1 | |
| strike | target, force 0 to 1 | |
| grab / drop | target | Single held item to start |
| place | location | Puts the held item somewhere |
| stack | held item, target | Balanced by gravity; can topple |
| bind | held item, target, binder | Needs something binding, such as a vine |
| insert | held item, target | Wedges into or places inside something hollow |
| separate | target | Takes a composite apart again |
| speak | free text | Vocalizing; nobody may be listening |

Opaque actions are added to the menu only when the agent is near the object that provides them, with no name, description, or hint.

## 7. Building

Building is as open as possible: there are no recipes and no list of buildable things. The world only enforces physics, and the agent decides what anything is for.

### 7.1 Materials as properties

Every object, natural or built, uses the same small set of physical properties instead of material types like wood or stone.

| Property | Effect |
|---|---|
| Rigidity | Whether it bends, holds weight, or transfers force |
| Weight | How hard it is to carry and how hard it hits |
| Size and shape | Reach, stability when stacked, whether it rolls |
| Binding | Whether it can tie or stick other things together |
| Hollow | Whether things can be put inside |
| Fragile | Whether it breaks when struck or dropped |
| Flammable | Optional, for later: fire as a world event |

### 7.2 Composites

Joining things produces a composite object whose properties are computed from its parts and how they are joined. A stick bound to a heavy flat rock becomes a long, rigid object that is heavy at one end; the simulation never calls it a hammer, it only knows that striking with it hits harder. Composites are real objects: they persist, can be joined again, and can be taken apart.

- Stack: balanced by gravity using size, shape, and weight; unstable stacks topple.
- Bind: needs a binding material; the joint has a strength that can fail under load.
- Insert: places or wedges an object into something hollow, which makes containers possible.

These three joins are enough for shelters, containers, tools, walls, bridges, and towers without designing any of them.

### 7.3 Why it would build

The agent is never told to build. It builds when the world gets in the way of what it already wants.

| Friction | Drive it touches | What building solves |
|---|---|---|
| Rain or cold | Comfort | Shelter |
| Something out of reach | Curiosity, energy | Stacking or a long tool |
| Water or a gap | Curiosity | A bridge to the unexplored side |
| Objects drifting or scattering | Keeping what it values | A container or a pile |
| Nothing new happening | Boredom | Composites are new objects, so building creates novelty |
| A known mailbox | Any of the above | Requests for tools or materials |

Built things are remembered as things the agent made, and reflection can carry that into its self-model. That is where a builder identity would come from, if one forms.

> The creator sets the pace: start sparse with a few loose materials, then introduce friction, like a rainy spell or an out-of-reach object, to see how the agent responds.

## 8. First-run environment

The first run is a small natural clearing: a meadow with a few trees, a stream along one side, a rock ledge, and a mailbox at the edge of a path. It is small enough that the agent meets everything within its first few days, and natural materials give real friction and varied properties without inventing content.

| Group | Object | Key properties | What it enables |
|---|---|---|---|
| Terrain | Grass | Open ground | Movement, a neutral baseline |
| Terrain | Stream | Wet, flowing; light things float | A barrier to cross, objects drifting away |
| Terrain | Rock ledge | Raised level | Height, things out of reach |
| Terrain | Large tree | Tall, fixed; casts shade | Partial shelter from rain, occlusion |
| Material | Sticks | Long, rigid, light | Tools, frames, bridges |
| Material | Flat stones | Heavy, flat, stackable | Stacking, weights, walls |
| Material | Vines | Binding, flexible | Joining things together |
| Material | Leaves | Very light, blown by wind | Roofing, weather cues |
| Curiosity | Ball | Round, light, rolls | Play, early cause and effect |
| Curiosity | Hollow log | Hollow, heavy | A container and a small shelter |
| Curiosity | Berry bush | Berries are small and fragile | Optional food; restores a little energy |
| Opaque | Mailbox | Unexplained action when nearby | Contact with the creator |

- Every material property is covered: rigid, binding, hollow, heavy, light, round, fragile, and wet.
- Stacking, binding, and inserting are all possible from the start, so a shelter, a bridge, or a long tool can emerge without being designed.
- Nothing new appears except through the mailbox or creator spawns, so it is clear what the agent found alone and what the creator introduced.
- Visuals are a simple top-down tile style; the agent never sees them.
- The object names in this table are for humans only. The agent sees only properties.

## 9. Cognition and drives

With no objective, behavior comes from intrinsic drives. The mix chosen here will largely decide what kind of agent emerges.

- Curiosity: reward for surprise, meaning a prediction that turned out wrong, and for novel objects.
- Boredom: builds when nothing new happens, pushing it to explore or experiment.
- Mild needs: energy and comfort create a daily rhythm without dictating behavior.
- Rest: periods of low activity, which also trigger reflection.

### 9.1 Body state and comfort

There is no hunger drive. A strong, recurring need with a single solution would dominate the agent's time and push it toward foraging. Instead, two quantified values make up the body state.

| Value | Lowered by | Restored by |
|---|---|---|
| Energy | Activity, slowly over time | Rest; optionally some discoverable objects that restore a little when eaten |
| Comfort | Cold, wet, night, storms, low energy | Shelter, warmth, dryness, rest |

Food can exist as something to discover, but the agent never needs it to survive.

> The agent sees these only as numbers, such as `comfort 0.3`. Nothing tells it that low is bad. It has to learn what low comfort means from how its body behaves.

| Condition | Physical side effect |
|---|---|
| Low energy | Slower movement; weaker push and strike force |
| Cold | Reduced dexterity: grabs and binds sometimes fail |
| Wet | Slightly slower; held items slip more often |
| Very low comfort | Shorter sensing radius, as if distracted |

- Effects are mild and scale gradually with the value, so they nudge behavior rather than punish it.
- The agent can push through discomfort to finish something it cares about; whether it does is itself worth observing.
- Internally, comfort still feeds the drive system and memory salience, so uncomfortable episodes are remembered more strongly.
- Effect strengths are configurable, so experiments can compare a harsh world with a gentle one.

## 10. Attention and gating

Most ticks only need the agent to keep doing what it is doing. A cheap attention layer handles those, and the LLM is woken only when thought could change something. This keeps local inference affordable without letting a small model limit what the agent becomes.

> The gate decides when the agent thinks, never what it thinks or does. Every intent, name, belief, and message comes from the LLM, so the agent's possibilities are never capped by the cheap layer.

### 10.1 Design principles

- Errors lean toward waking. A needless wake costs time; a missed wake makes the agent duller. Thresholds are tuned with that asymmetry in mind.
- The LLM sets its own wake conditions. Each intent it issues includes what should interrupt it, so the agent's own mind decides what it pays attention to.
- A heartbeat always fires. Even in routine stretches the LLM regularly gets the chance to change its mind, so the agent never runs on autopilot for long.
- Every gate decision is logged with its reason, so the creator can audit missed or wasted wakes.
- The gate sits behind one interface, so rules, a learned model, or a classifier can be swapped in without touching the rest.

### 10.2 Wake triggers

| Trigger | Source | Example |
|---|---|---|
| Surprise | World model, intent, beliefs | The pushed object moved much farther than predicted |
| Novelty | Perception | A never-seen object or a new affordance comes in range |
| Intent ended | Controller | Reached the target, or the action failed or was blocked |
| Agent-set condition | LLM | "Wake me if the red thing moves" |
| Drive threshold | Body state | Energy is low, or boredom has built up |
| Creator event | World | A letter appears in the mailbox, or an object is spawned nearby |
| Heartbeat | Timer | Every N ticks with no other wake |

### 10.3 The intent contract

When the LLM is awake it returns a structured intent the controller can carry out on its own until the next wake.

| Field | Example / meaning |
|---|---|
| goal | `push object_7 toward the tree` |
| steps | A short plan in body verbs the controller executes |
| wake_if | `object_7 stops moving`, `new object in range` |
| expect | What it predicts will happen, for checking surprise against its own beliefs |
| thought | One line of reasoning, kept for memory and the observer log |

The `expect` field gives a second, LLM-level surprise signal: when the outcome contradicts what the agent itself predicted, that is a strong wake and a strong memory.

### 10.4 What counts as surprise

Surprise is the gap between what the agent expected and what happened, weighted by how confident the expectation was. A confident prediction that fails is very surprising; a wrong guess made with no confidence barely registers.

| Kind | Measured as | When | Example |
|---|---|---|---|
| Physical | World model prediction error in embedding space | Every tick | The ball moved twice as far as predicted |
| Expectation | Intent `expect` field compared to the outcome | When an intent ends | It said the ball would roll to the tree; it stopped short |
| Belief | Observation contradicts a semantic belief, scaled by the belief's confidence | On retrieval of related beliefs | A round thing that will not roll |
| Novelty | No history for this object, property mix, or affordance | On first perception | The mailbox's opaque action appears |

Expectation and belief surprises outrank physical ones, because they mean one of the agent's own theories may be wrong. Novelty is not a failed prediction but the absence of one, so it is tracked as its own trigger rather than folded into the error score.

```
surprise = (error × confidence) / (usual error for this context) × habituation
```

- Normalized per context. Error is compared to the typical error for that object or situation, so things that are noisy but familiar, like a wind-blown leaf, stop registering.
- Habituation. A surprise that keeps repeating decays toward zero and should be turned into a belief during reflection instead.
- Explained versus unexplained. Once reflection forms a belief that predicts a surprise, it is resolved. An unexplained surprise keeps a small lasting pull that draws the agent back, which is one way hobbies and fixations can emerge.
- One score, kept by kind. The gate sees a single combined score; memory records which kind of surprise it was, since a broken theory is a richer memory than an odd movement.

### 10.5 World model

A small learned predictor, in the spirit of JEPA (joint embedding predictive architecture), takes the current state and action and predicts the next state in an abstract embedding space. Because perception is structured state rather than pixels, it can be a small MLP or transformer over state vectors, trained online as the agent acts.

- Prediction error is the surprise signal. It feeds the gate and doubles as the curiosity reward.
- Surprise is normalized per object and situation, so a chaotic object stops being interesting once its chaos is expected.
- It is used only for surprise at first. Learned planning is a later step once the agent's behavior is understood.
- It runs on the GPU beside the LLM or on the CPU, and is tiny next to the LLM.

### 10.6 Rollout

1. **Phase 1: rules.** A rule-based gate over the wake triggers, plus LLM-set wake conditions and the heartbeat. Novelty stands in for surprise.
2. **Phase 2: world model.** Add the learned predictor and switch the surprise trigger and curiosity drive to its prediction error.
3. **Phase 3: learned gate.** Only if rules prove too crude: train a small local classifier on the logged gate decisions and whether each wake changed the agent's intent.

> Future thought: a hosted System One classifier such as Jev from TypeSafe AI could serve as the gate in phase 3. It is left out of the design for now because it is not local.

## 11. Memory

1. **Short-term buffer.** A fixed window of recent observations and actions, around 30 to 50 entries. Filling up, or resting, triggers reflection.
2. **Episodic memory.** Reflection compresses the buffer into short episodes with a salience score. Low-salience episodes decay unless recalled.
3. **Semantic beliefs.** Repeated episodes harden into facts and theories with confidence scores that rise with evidence and fall when contradicted.
4. **Self-model.** A short, evolving self-description updated during reflection. This is where personality and hobbies should show up.

Retrieval is by relevance rather than loading everything: when the agent meets a familiar object, pull the beliefs and episodes linked to it, scored by recency, importance, and relevance, as in the Generative Agents paper (https://arxiv.org/abs/2304.03442). A small local embedding model or tag matching is enough. A hard cap on beliefs forces real forgetting, which reveals what the agent cares about.

## 12. Creator interface and the mailbox

The mailbox is a physical object in the world. Near it, the agent gains an opaque action that accepts free text. It is never told what it does or who receives it. Messages reach the creator console, and the creator can reply, spawn an object in response, or stay silent.

> Creator stance: respond at will, sometimes not at all, with no acknowledgement required. Avoid giving away semantics, such as names for objects or explanations of how things work. This part is deliberately experimental.

- Responses can be indirect: spawning a requested tool with no note, or a vague letter, rather than a clear answer.
- Every creator action is logged with a timestamp, so changes in the agent's beliefs about the mailbox can be traced back to what the creator did.
- It may be worth keeping private creator notes for each experiment, since different response styles are effectively different experiments.
- Creator tools: spawn, despawn, and move objects; read and answer mail; set or schedule weather and toggle its automation; set or freeze the time of day; pause and fast-forward time.
- Replies could arrive as a physical letter in the mailbox, itself an object to discover.

## 13. Observability

The agent only knows its compressed memories, but the system keeps a complete ground-truth log of every perception, decision, action, and reflection. Comparing what happened with what the agent remembers, including mistaken theories, is a core part of the experiment.

- A timeline view of the agent's thoughts and actions.
- A browser for current beliefs, their confidence, and how they changed over time.
- The agent's own names for objects mapped to internal ids.

## 14. Web interface

The interface is web-based from the start. It runs on the local machine first and can be opened remotely later without a rewrite.

1. **Simulation server.** A headless Python process runs the world, agent loop, and memory, and keeps running whether or not anyone is watching.
2. **API.** A web API (for example FastAPI) for creator commands, plus a WebSocket that streams world state, agent thoughts, and events live.
3. **Browser client.** Draws the tile world on an HTML canvas (for example with PixiJS) and holds the creator console, mailbox inbox, and observer views.

- Views: the live world, a thought and action timeline, the belief browser, the agent's names for objects, the inbox and reply panel, and weather and time controls.
- Watching never affects the agent: the client only reads state and sends creator commands, so closing the browser changes nothing.
- For remote access, put it behind a private network tunnel such as Tailscale or a reverse proxy with login, rather than exposing it directly. Creator controls need authentication even if viewing is shared.
- A read-only spectator mode could let others watch an experiment without touching it.

## 15. Hardware and local models

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4060 Ti, 16 GB VRAM |
| System RAM | 64 GB |
| CPU | Intel i7 |

Accessibility comes first: the baseline should run on a single mid-range GPU, so other people can reproduce it and so we learn whether the loop works before spending more compute.

| Component | Starting choice | Why |
|---|---|---|
| Runtime | Ollama or llama.cpp | Easy install, OpenAI-style local API, easy to swap models |
| Main model | Qwen instruct, 7B to 8B, 4- or 5-bit quantized | Fits in VRAM with plenty of room for context and fast enough for frequent thinking |
| Upgrade path | 12B to 14B, 4-bit | Still fits in 16 GB if reasoning quality is too weak |
| Embeddings | Small local embedding model | Memory retrieval; light enough to share the GPU or run on CPU |
| Simulation | CPU | A 2D tile world and scripted controller barely touch the GPU |

- One model serves every role (deciding, naming, reflecting) with different prompts, so only one model sits in VRAM.
- The LLM always replies in a strict JSON schema (intent, target, parameters, a short thought), using the runtime's structured-output mode so small models stay reliable.
- Reflection runs while the agent rests, when it can take longer without stalling behavior.
- The world can run slower than real time or pause while the LLM thinks, so model speed never changes the agent's behavior.
- Keep prompts small: perception deltas, the self-model, and a handful of retrieved memories, not the full history.

> The model sits behind one interface, so swapping in a bigger or different model later is a config change, not a rewrite.

## 16. Open questions

1. Which Qwen variant and size to start with; worth a quick bake-off on JSON reliability, tool calling, and reasoning about cause and effect.
2. How the simulation, LLM calls, and web server share one machine: a single process, or separate processes talking over a local queue.
3. Whether the low-level controller is scripted or learned with reinforcement learning.
4. Starting heartbeat interval and surprise threshold, to tune once the loop runs.
5. Whether the agent can die, get hurt, or be reset, and what persists across resets.
6. How detailed composite physics should be: simple property rules, or a light physics engine for stability and joint strength.
7. Whether materials can be consumed or regrow.

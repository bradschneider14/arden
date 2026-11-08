"""Surprise scoring: physical, expectation, belief, and novelty surprise with normalization and habituation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from arden.surprise.world_model import (
    NullWorldModel,
    StateEmbedding,
    VelocityWorldModel,
    WorldModel,
)

SurpriseKind = Literal["physical", "expectation", "belief", "novelty"]
SURPRISE_KINDS: tuple[SurpriseKind, ...] = (
    "physical",
    "expectation",
    "belief",
    "novelty",
)


class SurpriseConfig(BaseModel):
    """Configuration for surprise scoring, normalization, habituation, and pull under ``[surprise]``."""

    model_config = ConfigDict(extra="forbid")

    use_llm_judge: bool = False
    initial_usual_error: float = Field(default=0.1, gt=0.0)
    min_usual_error: float = Field(default=0.1, gt=0.0)
    error_ema_alpha: float = Field(default=0.25, gt=0.0, le=1.0)
    habituation_decay: float = Field(default=0.65, gt=0.0, le=1.0)
    unexplained_pull: float = Field(default=0.2, ge=0.0, le=1.0)
    movement_tolerance: float = Field(default=0.25, ge=0.0)


SurpriseSettings = SurpriseConfig


def compute_surprise_score(
    error: float,
    confidence: float = 1.0,
    usual_error: float = 0.1,
    habituation: float = 1.0,
    *,
    min_usual_error: float = 1e-6,
) -> float:
    """Compute normalized, habituated surprise score.

    Formula:
        ``score = error * confidence / usual_error(context) * habituation``
    """

    safe_error = max(0.0, float(error))
    safe_conf = max(0.0, float(confidence))
    safe_hab = max(0.0, float(habituation))
    denom = max(float(min_usual_error), float(usual_error))
    return (safe_error * safe_conf / denom) * safe_hab


@dataclass
class SurpriseSignal:
    """Output signal from ``SurpriseScorer``: ``SurpriseSignal(score, kind, details)``."""

    score: float
    kind: SurpriseKind = "physical"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = float(self.score)
        if self.kind not in SURPRISE_KINDS:
            raise ValueError(
                f"Invalid surprise kind {self.kind!r}; expected one of {SURPRISE_KINDS}."
            )
        if self.details is None:
            self.details = {}
        elif not isinstance(self.details, dict):
            self.details = dict(self.details)

    def __iter__(self):
        """Allow tuple unpacking: ``score, kind, details = signal``."""

        yield self.score
        yield self.kind
        yield self.details

    @property
    def error(self) -> float:
        """Raw prediction/contradiction error (0.0 for novelty)."""

        if self.kind == "novelty":
            return 0.0
        return float(self.details.get("error", 0.0))

    @property
    def error_score(self) -> float:
        """Error-based surprise score; always 0.0 when ``kind == 'novelty'``."""

        if self.kind == "novelty":
            return 0.0
        return float(self.details.get("error_score", self.score))

    @property
    def novelty_score(self) -> float:
        """Novelty score when ``kind == 'novelty'`` (or embedded novelty details)."""

        if self.kind == "novelty":
            return float(self.score)
        return float(self.details.get("novelty_score", 0.0))

    @property
    def confidence(self) -> float:
        return float(self.details.get("confidence", 1.0))

    @property
    def usual_error(self) -> float:
        return float(self.details.get("usual_error", 0.1))

    @property
    def habituation(self) -> float:
        return float(self.details.get("habituation", 1.0))

    @property
    def context(self) -> str:
        return str(self.details.get("context", self.details.get("object_id", "global")))

    @property
    def object_id(self) -> str | None:
        val = self.details.get("object_id")
        return str(val) if val is not None else None

    @property
    def pull(self) -> float:
        return float(self.details.get("pull", 0.0))

    @property
    def explained(self) -> bool:
        return bool(self.details.get("explained", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "kind": self.kind,
            "details": dict(self.details),
        }


# Stopwords and tokenization helpers for cheap expectation & belief checks
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "would",
    }
)

_NEGATION_WORDS = frozenset(
    {
        "no",
        "not",
        "never",
        "none",
        "nothing",
        "neither",
        "without",
        "cannot",
        "cant",
        "didnt",
        "doesnt",
        "dont",
        "isnt",
        "wasnt",
        "werent",
        "wont",
        "wouldnt",
    }
)

_MOVEMENT_WORDS = frozenset(
    {
        "move",
        "moves",
        "moved",
        "moving",
        "roll",
        "rolls",
        "rolled",
        "rolling",
        "slide",
        "slides",
        "slid",
        "sliding",
        "shift",
        "shifts",
        "shifted",
        "shifting",
        "displace",
        "displaced",
        "push",
        "pushed",
        "fall",
        "falls",
        "fell",
        "falling",
        "bounce",
        "bounces",
        "bounced",
        "closer",
        "further",
        "away",
        "toward",
        "towards",
        "north",
        "south",
        "east",
        "west",
        "left",
        "right",
        "up",
        "down",
        "forward",
        "backward",
    }
)

_STATIONARY_WORDS = frozenset(
    {
        "still",
        "stationary",
        "stay",
        "stays",
        "stayed",
        "remain",
        "remains",
        "remained",
        "unmoved",
        "unchanged",
        "immovable",
        "fixed",
        "static",
        "stuck",
        "heavy",
        "too_heavy",
    }
)

_DIRECTION_VECTORS: dict[str, tuple[float, float]] = {
    "east": (1.0, 0.0),
    "right": (1.0, 0.0),
    "west": (-1.0, 0.0),
    "left": (-1.0, 0.0),
    "south": (0.0, 1.0),
    "down": (0.0, 1.0),
    "north": (0.0, -1.0),
    "up": (0.0, -1.0),
}

_ANTONYM_GROUPS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (
        frozenset({"move", "moves", "moved", "moving", "roll", "rolls", "rolled", "rolling", "slide", "slides", "slid", "mobile", "movable", "pushable"}),
        frozenset({"stationary", "still", "unmoved", "immovable", "static", "fixed", "stuck", "stay", "stays", "stayed", "too_heavy"}),
    ),
    (
        frozenset({"wet", "soaked", "damp"}),
        frozenset({"dry", "arid", "unwet"}),
    ),
    (
        frozenset({"broken", "shattered", "cracked", "damaged", "destroyed", "fragile", "breaks", "broke"}),
        frozenset({"intact", "unbroken", "solid", "sturdy", "indestructible", "unbreakable", "whole"}),
    ),
    (
        frozenset({"heavy", "too_heavy", "ungrabbable", "massive"}),
        frozenset({"light", "lightweight", "grabbable", "portable", "lifted", "picked", "grabbed", "carried", "holding"}),
    ),
    (
        frozenset({"hot", "warm", "burning"}),
        frozenset({"cold", "cool", "freezing", "frozen"}),
    ),
    (
        frozenset({"open", "opened", "unlocked"}),
        frozenset({"closed", "shut", "locked"}),
    ),
    (
        frozenset({"succeed", "succeeded", "success", "completed", "reached", "arrived"}),
        frozenset({"fail", "failed", "failure", "aborted", "blocked", "unreachable", "timeout", "timed_out"}),
    ),
)

_COLOR_WORDS = frozenset(
    {"red", "blue", "green", "yellow", "orange", "purple", "black", "white", "gray", "grey", "brown", "pink", "cyan"}
)
_SHAPE_WORDS = frozenset(
    {"sphere", "cube", "cylinder", "cone", "box", "ball", "block", "pyramid", "ring", "slab", " puddle"}
)


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"['’]", "", text.lower())
    return re.findall(r"[a-z0-9_]+", cleaned)


def _has_negation_near(tokens: Sequence[str], target_words: frozenset[str], window: int = 3) -> bool:
    for idx, tok in enumerate(tokens):
        if tok in target_words:
            start = max(0, idx - window)
            if any(tokens[j] in _NEGATION_WORDS for j in range(start, idx)):
                return True
    return False


class SurpriseScorer:
    """Computes physical, expectation, belief, and novelty surprise signals."""

    def __init__(
        self,
        config: SurpriseConfig | Mapping[str, Any] | None = None,
        *,
        world_model: WorldModel | None = None,
        llm: Any | None = None,
        belief_store: Any | None = None,
        world: Any | None = None,
        use_llm_judge: bool | None = None,
        initial_usual_error: float | None = None,
        min_usual_error: float | None = None,
        error_ema_alpha: float | None = None,
        habituation_decay: float | None = None,
        unexplained_pull: float | None = None,
        movement_tolerance: float | None = None,
    ) -> None:
        if isinstance(config, SurpriseConfig):
            base_cfg = config
        elif isinstance(config, Mapping):
            base_cfg = SurpriseConfig(
                use_llm_judge=bool(config.get("use_llm_judge", False)),
                initial_usual_error=float(config.get("initial_usual_error", 0.1)),
                min_usual_error=float(config.get("min_usual_error", 0.1)),
                error_ema_alpha=float(config.get("error_ema_alpha", 0.25)),
                habituation_decay=float(config.get("habituation_decay", 0.65)),
                unexplained_pull=float(config.get("unexplained_pull", 0.2)),
                movement_tolerance=float(config.get("movement_tolerance", 0.25)),
            )
        elif config is not None and hasattr(config, "surprise"):
            sub = getattr(config, "surprise")
            base_cfg = SurpriseConfig(
                use_llm_judge=bool(getattr(sub, "use_llm_judge", False)),
                initial_usual_error=float(getattr(sub, "initial_usual_error", 0.1)),
                min_usual_error=float(getattr(sub, "min_usual_error", 0.1)),
                error_ema_alpha=float(getattr(sub, "error_ema_alpha", 0.25)),
                habituation_decay=float(getattr(sub, "habituation_decay", 0.65)),
                unexplained_pull=float(getattr(sub, "unexplained_pull", 0.2)),
                movement_tolerance=float(getattr(sub, "movement_tolerance", 0.25)),
            )
        else:
            base_cfg = SurpriseConfig()

        self.config = SurpriseConfig(
            use_llm_judge=base_cfg.use_llm_judge if use_llm_judge is None else bool(use_llm_judge),
            initial_usual_error=(
                base_cfg.initial_usual_error
                if initial_usual_error is None
                else float(initial_usual_error)
            ),
            min_usual_error=(
                base_cfg.min_usual_error
                if min_usual_error is None
                else float(min_usual_error)
            ),
            error_ema_alpha=(
                base_cfg.error_ema_alpha
                if error_ema_alpha is None
                else float(error_ema_alpha)
            ),
            habituation_decay=(
                base_cfg.habituation_decay
                if habituation_decay is None
                else float(habituation_decay)
            ),
            unexplained_pull=(
                base_cfg.unexplained_pull
                if unexplained_pull is None
                else float(unexplained_pull)
            ),
            movement_tolerance=(
                base_cfg.movement_tolerance
                if movement_tolerance is None
                else float(movement_tolerance)
            ),
        )

        self.world_model: WorldModel = (
            world_model if world_model is not None else VelocityWorldModel()
        )
        self.llm = llm
        self.belief_store = belief_store
        self.world = world

        # Per-context running average of error: usual_error(context)
        self._usual_errors: dict[str, float] = {}
        self._context_counts: dict[str, int] = {}

        # Habituation counts per surprise signature
        self._habituation_counts: dict[str, int] = {}

        # Novelty tracking sets
        self._seen_object_ids: set[str] = set()
        self._seen_property_combos: set[tuple[tuple[str, Any], ...]] = set()
        self._seen_affordances: set[str] = set()
        self._seen_object_affordances: set[tuple[str, str]] = set()

        # Unexplained surprises and persistent pull values per object_id
        self._unexplained_surprises: dict[str, list[SurpriseSignal]] = {}
        self._object_pulls: dict[str, float] = {}

        # Previous state tracking for single-arg physical scoring
        self._last_state: Any | None = None

    @property
    def use_llm_judge(self) -> bool:
        return self.config.use_llm_judge

    @use_llm_judge.setter
    def use_llm_judge(self, value: bool) -> None:
        self.config = SurpriseConfig(
            use_llm_judge=bool(value),
            initial_usual_error=self.config.initial_usual_error,
            min_usual_error=self.config.min_usual_error,
            error_ema_alpha=self.config.error_ema_alpha,
            habituation_decay=self.config.habituation_decay,
            unexplained_pull=self.config.unexplained_pull,
            movement_tolerance=self.config.movement_tolerance,
        )

    @property
    def pulls(self) -> dict[str, float]:
        """Mapping of object_id -> persistent pull value for unexplained surprises."""

        return dict(self._object_pulls)

    @property
    def object_pulls(self) -> dict[str, float]:
        return dict(self._object_pulls)

    @property
    def unexplained_surprises(self) -> dict[str, list[SurpriseSignal]]:
        return {k: list(v) for k, v in self._unexplained_surprises.items() if v}

    def usual_error(self, context: str = "global") -> float:
        """Return the current running average error ``usual_error(context)``."""

        key = str(context or "global")
        return float(
            self._usual_errors.get(
                key, max(self.config.min_usual_error, self.config.initial_usual_error)
            )
        )

    def get_usual_error(self, context: str = "global") -> float:
        return self.usual_error(context)

    def _update_usual_error(self, context: str, error: float) -> float:
        """Update and return the per-context running average of error."""

        key = str(context or "global")
        safe_error = max(0.0, float(error))
        prev = self._usual_errors.get(
            key, max(self.config.min_usual_error, self.config.initial_usual_error)
        )
        alpha = self.config.error_ema_alpha
        updated = (1.0 - alpha) * prev + alpha * safe_error
        denom = max(self.config.min_usual_error, updated)
        self._usual_errors[key] = denom
        self._context_counts[key] = self._context_counts.get(key, 0) + 1
        return denom

    def habituation_factor(self, signature: str, *, increment: bool = False) -> float:
        """Return habituation factor in ``(0, 1]`` for ``signature``.

        Starts at 1.0 on the first occurrence and decays monotonically toward 0
        on repeated identical surprises.
        """

        key = str(signature or "default")
        prior_count = self._habituation_counts.get(key, 0)
        factor = float(self.config.habituation_decay ** prior_count)
        if increment:
            self._habituation_counts[key] = prior_count + 1
        return factor

    def get_habituation(self, signature: str) -> float:
        """Return the habituation factor that will apply on the next occurrence of ``signature``."""

        return self.habituation_factor(signature, increment=False)

    def _compute_normalized_score(
        self,
        *,
        error: float,
        confidence: float,
        context: str,
        signature: str,
        habituate: bool | None = None,
    ) -> tuple[float, float, float]:
        """Update running error average and habituation, returning ``(score, usual_error, habituation)``."""

        safe_error = max(0.0, float(error))
        safe_conf = max(0.0, float(confidence))
        usual = self._update_usual_error(context, safe_error)
        should_habituate = (safe_error > 1e-6) if habituate is None else bool(habituate)
        hab = self.habituation_factor(signature, increment=should_habituate)
        score = compute_surprise_score(
            safe_error,
            safe_conf,
            usual,
            hab,
            min_usual_error=self.config.min_usual_error,
        )
        return score, usual, hab

    # ------------------------------------------------------------------
    # Persistent pull tracking for unexplained surprises
    # ------------------------------------------------------------------

    def get_pull(self, object_id: str, *, world: Any | None = None) -> float:
        """Return the persistent unexplained-surprise pull stored on ``object_id``."""

        target_world = world if world is not None else self.world
        if target_world is not None:
            obj = self._find_world_object(target_world, object_id)
            if obj is not None:
                if hasattr(obj, "pull"):
                    obj_pull = float(getattr(obj, "pull"))
                    if obj_pull > 0.0:
                        return obj_pull
                if hasattr(obj, "state") and isinstance(obj.state, Mapping):
                    raw = obj.state.get("unexplained_pull", obj.state.get("pull", 0.0))
                    if float(raw) > 0.0:
                        return float(raw)
        return float(self._object_pulls.get(str(object_id), 0.0))

    def get_object_pull(self, object_id: str, *, world: Any | None = None) -> float:
        return self.get_pull(object_id, world=world)

    def set_pull(
        self,
        object_id: str,
        pull: float,
        *,
        world: Any | None = None,
        target_object: Any | None = None,
    ) -> float:
        """Store or clear persistent pull value on ``object_id`` and any bound ``WorldObject``."""

        oid = str(object_id)
        val = max(0.0, float(pull))
        if val > 0.0:
            self._object_pulls[oid] = val
        else:
            self._object_pulls.pop(oid, None)

        target_world = world if world is not None else self.world
        obj = target_object
        if obj is None and target_world is not None:
            obj = self._find_world_object(target_world, oid)
        if obj is not None:
            self._write_pull_to_object(obj, val)
        return val

    @staticmethod
    def _find_world_object(world: Any, object_id: str) -> Any | None:
        if world is None:
            return None
        objs = getattr(world, "objects", None)
        if isinstance(objs, Mapping):
            return objs.get(object_id)
        if isinstance(objs, Sequence):
            for item in objs:
                item_id = getattr(item, "id", None)
                if item_id is None and isinstance(item, Mapping):
                    item_id = item.get("id")
                if item_id == object_id:
                    return item
        if isinstance(world, Mapping):
            if object_id in world:
                return world[object_id]
            inner = world.get("objects")
            if isinstance(inner, Mapping):
                return inner.get(object_id)
        return None

    @staticmethod
    def _write_pull_to_object(obj: Any, pull_value: float) -> None:
        val = max(0.0, float(pull_value))
        if hasattr(obj, "set_pull") and callable(obj.set_pull):
            obj.set_pull(val)
            return
        if hasattr(obj, "state") and isinstance(obj.state, dict):
            if val > 0.0:
                obj.state["unexplained_pull"] = val
                obj.state["pull"] = val
            else:
                obj.state.pop("unexplained_pull", None)
                obj.state.pop("pull", None)
        elif isinstance(obj, dict):
            if "state" in obj and isinstance(obj["state"], dict):
                if val > 0.0:
                    obj["state"]["unexplained_pull"] = val
                    obj["state"]["pull"] = val
                else:
                    obj["state"].pop("unexplained_pull", None)
                    obj["state"].pop("pull", None)
            obj["unexplained_pull"] = val
            obj["pull"] = val

    def _record_surprise_pull(
        self,
        signal: SurpriseSignal,
        *,
        object_id: str | None = None,
        beliefs: Sequence[Any] | None = None,
        world: Any | None = None,
        target_object: Any | None = None,
    ) -> None:
        oid = object_id or signal.object_id
        if not oid:
            return
        if signal.kind == "novelty" or signal.score <= 0.0 or signal.error <= 0.0:
            return

        candidate_beliefs = self._collect_beliefs(beliefs, query=oid)
        explained = self._is_signal_explained_by_beliefs(signal, candidate_beliefs)
        signal.details["explained"] = explained
        if not explained:
            self._unexplained_surprises.setdefault(oid, []).append(signal)
            pull_val = self.set_pull(
                oid,
                self.config.unexplained_pull,
                world=world,
                target_object=target_object,
            )
            signal.details["pull"] = pull_val
        else:
            self._unexplained_surprises.pop(oid, None)
            self.set_pull(oid, 0.0, world=world, target_object=target_object)
            signal.details["pull"] = 0.0

    def resolve_with_beliefs(
        self,
        beliefs: Sequence[Any] | Any | None = None,
        *,
        world: Any | None = None,
    ) -> dict[str, float]:
        """After reflection, check whether any unexplained surprises are now predicted by beliefs.

        Surprises that remain unexplained keep their persistent pull value on the
        related ``object_id``. Surprises explained by a belief have their pull cleared.
        """

        all_beliefs = self._collect_beliefs(beliefs)
        target_world = world if world is not None else self.world

        for oid in list(self._unexplained_surprises.keys()):
            signals = self._unexplained_surprises.get(oid, [])
            remaining: list[SurpriseSignal] = []
            for sig in signals:
                if not self._is_signal_explained_by_beliefs(sig, all_beliefs):
                    sig.details["explained"] = False
                    remaining.append(sig)
                else:
                    sig.details["explained"] = True
                    sig.details["pull"] = 0.0
            if remaining:
                self._unexplained_surprises[oid] = remaining
                self.set_pull(oid, self.config.unexplained_pull, world=target_world)
            else:
                self._unexplained_surprises.pop(oid, None)
                self.set_pull(oid, 0.0, world=target_world)

        return dict(self._object_pulls)

    def on_reflection(
        self,
        beliefs: Sequence[Any] | Any | None = None,
        *,
        world: Any | None = None,
    ) -> dict[str, float]:
        """Alias for ``resolve_with_beliefs`` called after reflection."""

        return self.resolve_with_beliefs(beliefs, world=world)

    def update_after_reflection(
        self,
        beliefs: Sequence[Any] | Any | None = None,
        *,
        world: Any | None = None,
    ) -> dict[str, float]:
        """Alias for ``resolve_with_beliefs``."""

        return self.resolve_with_beliefs(beliefs, world=world)

    def _collect_beliefs(
        self,
        beliefs: Sequence[Any] | Any | None = None,
        *,
        query: str | None = None,
    ) -> list[Any]:
        collected: list[Any] = []
        if beliefs is not None:
            if hasattr(beliefs, "all_active") and callable(beliefs.all_active):
                collected.extend(beliefs.all_active())
            elif hasattr(beliefs, "all") and callable(beliefs.all):
                collected.extend(beliefs.all())
            elif hasattr(beliefs, "retrieve") and callable(beliefs.retrieve) and query:
                collected.extend(beliefs.retrieve(query))
            elif isinstance(beliefs, (str, Mapping)) or hasattr(beliefs, "statement"):
                collected.append(beliefs)
            elif isinstance(beliefs, Sequence):
                collected.extend(beliefs)

        if self.belief_store is not None and beliefs is not self.belief_store:
            if hasattr(self.belief_store, "all_active") and callable(self.belief_store.all_active):
                collected.extend(self.belief_store.all_active())
            elif hasattr(self.belief_store, "all") and callable(self.belief_store.all):
                collected.extend(self.belief_store.all())
            elif query and hasattr(self.belief_store, "retrieve") and callable(self.belief_store.retrieve):
                collected.extend(self.belief_store.retrieve(query))
        return collected

    def _is_signal_explained_by_beliefs(
        self,
        signal: SurpriseSignal,
        beliefs: Sequence[Any],
    ) -> bool:
        """Check whether at least one active belief predicts/explains ``signal``."""

        if not beliefs:
            return False

        oid = (signal.object_id or "").lower()
        obs_text = str(
            signal.details.get("actual_summary")
            or signal.details.get("observation")
            or signal.details.get("reason")
            or ""
        ).lower()
        obs_tokens = set(_tokenize(obs_text))
        sig_tags = {
            str(t).lower()
            for t in signal.details.get("observed_tags", [])
            if t is not None
        }
        all_signal_tokens = obs_tokens | sig_tags

        for belief in beliefs:
            stmt, tags, conf, status = self._extract_belief_fields(belief)
            if status != "active" or conf <= 0.0:
                continue

            # If this belief itself was the contradicted belief, it does NOT explain the surprise!
            contradicted_id = signal.details.get("belief_id")
            belief_id = getattr(belief, "id", None) or (
                belief.get("id") if isinstance(belief, Mapping) else None
            )
            if contradicted_id is not None and belief_id == contradicted_id:
                continue

            stmt_lower = stmt.lower()
            stmt_tokens = set(_tokenize(stmt_lower))
            tag_tokens = set()
            for t in tags:
                tag_tokens.update(_tokenize(str(t)))
            belief_vocab = stmt_tokens | tag_tokens

            # Must relate to the object or context if an object_id is known
            if oid:
                oid_parts = set(_tokenize(oid))
                oid_mentioned = (
                    oid in stmt_lower
                    or any(oid == str(t).lower() for t in tags)
                    or bool(oid_parts & belief_vocab)
                )
                if not oid_mentioned:
                    continue

            # If the belief contradicts the signal's observation, it does not explain it
            if obs_text:
                contradiction, _ = self._check_single_belief_contradiction(
                    observation=obs_text,
                    obs_tokens=list(_tokenize(obs_text)),
                    obs_dict=signal.details.get("observation_dict"),
                    belief_statement=stmt,
                    belief_tags=tags,
                    object_id=oid or None,
                )
                if contradiction > 0.0:
                    continue

            # Check if the belief's predicate/tags positively predict the observed phenomenon
            if all_signal_tokens:
                meaningful_signal = all_signal_tokens - _STOPWORDS - set(_tokenize(oid))
                meaningful_belief = belief_vocab - _STOPWORDS - set(_tokenize(oid))
                if meaningful_signal & meaningful_belief:
                    return True
                # Synonym group match (e.g. observation "moved", belief "rolls" / "slides")
                for group_a, group_b in _ANTONYM_GROUPS:
                    if (meaningful_signal & group_a and meaningful_belief & group_a) or (
                        meaningful_signal & group_b and meaningful_belief & group_b
                    ):
                        return True
            else:
                # For physical surprise with no text summary, a belief mentioning the object and
                # any dynamic/causal predicate (e.g., moves, bounces, teleports, noisy, jittery, wind, slope)
                # explains it.
                explanatory_words = (
                    _MOVEMENT_WORDS
                    | _STATIONARY_WORDS
                    | {
                        "noisy",
                        "jitter",
                        "jittery",
                        "random",
                        "wind",
                        "current",
                        "slope",
                        "gravity",
                        "magnet",
                        "magnetic",
                        "teleport",
                        "teleports",
                        "accelerates",
                        "decelerates",
                        "friction",
                        "oscillate",
                        "oscillates",
                        "vibrates",
                        "autonomous",
                        "alive",
                    }
                )
                if belief_vocab & explanatory_words or len(belief_vocab - set(_tokenize(oid)) - _STOPWORDS) >= 2:
                    return True

        return False

    # ------------------------------------------------------------------
    # 1. Physical surprise (world-model prediction error)
    # ------------------------------------------------------------------

    def score_physical(
        self,
        prev_state_or_pred: Any,
        actual_state: Any | None = None,
        action: Any | None = None,
        *,
        object_id: str | None = None,
        context: str | None = None,
        confidence: float = 1.0,
        world_model: WorldModel | None = None,
        beliefs: Sequence[Any] | None = None,
        world: Any | None = None,
    ) -> SurpriseSignal:
        """Score physical surprise using ``WorldModel.predict`` and ``WorldModel.error``.

        Supports:
        - ``score_physical(prev_state, actual_state, action=...)``
        - ``score_physical(predicted_embedding, actual_embedding)``
        - ``score_physical(current_state)`` (uses stored previous state)
        """

        model = world_model if world_model is not None else self.world_model

        if actual_state is None:
            # Single-state call: compare against self._last_state
            current_state = prev_state_or_pred
            if self._last_state is None:
                self._last_state = current_state
                ctx = str(context or object_id or "physical")
                return SurpriseSignal(
                    score=0.0,
                    kind="physical",
                    details={
                        "error": 0.0,
                        "confidence": float(confidence),
                        "usual_error": self.usual_error(ctx),
                        "habituation": 1.0,
                        "context": ctx,
                        "object_id": object_id,
                        "reason": "initial_state",
                    },
                )
            prev_state = self._last_state
            self._last_state = current_state
            actual_state = current_state
        else:
            prev_state = prev_state_or_pred
            self._last_state = actual_state

        # If NullWorldModel is active, error is always 0.0
        if isinstance(model, NullWorldModel):
            ctx = str(context or object_id or "physical")
            usual = self._update_usual_error(ctx, 0.0)
            return SurpriseSignal(
                score=0.0,
                kind="physical",
                details={
                    "error": 0.0,
                    "confidence": float(confidence),
                    "usual_error": usual,
                    "habituation": 1.0,
                    "context": ctx,
                    "object_id": object_id,
                },
            )

        # If object_id is not explicitly specified and states have multi-object dicts,
        # score per-object and return the highest-scoring object's physical surprise.
        prev_objs = self._extract_object_map(prev_state)
        curr_objs = self._extract_object_map(actual_state)
        if object_id is None and len(prev_objs) > 0 and len(curr_objs) > 0:
            # Only compare objects that existed in BOTH prev_state and curr_state
            # so brand-new objects (novelty) are never folded into physical error!
            shared_ids = sorted(set(prev_objs.keys()) & set(curr_objs.keys()))
            if not shared_ids:
                ctx = str(context or "physical")
                return SurpriseSignal(
                    score=0.0,
                    kind="physical",
                    details={
                        "error": 0.0,
                        "confidence": float(confidence),
                        "usual_error": self.usual_error(ctx),
                        "habituation": 1.0,
                        "context": ctx,
                        "object_id": None,
                    },
                )

            best_signal: SurpriseSignal | None = None
            for oid in shared_ids:
                sig = self._score_single_physical(
                    prev_objs[oid],
                    curr_objs[oid],
                    action=action,
                    object_id=oid,
                    context=context or oid,
                    confidence=confidence,
                    model=model,
                    beliefs=beliefs,
                    world=world,
                )
                if best_signal is None or sig.score > best_signal.score:
                    best_signal = sig
            assert best_signal is not None
            return best_signal

        inferred_oid = object_id or self._infer_object_id(actual_state) or self._infer_object_id(prev_state)
        return self._score_single_physical(
            prev_state,
            actual_state,
            action=action,
            object_id=inferred_oid,
            context=context or inferred_oid or "physical",
            confidence=confidence,
            model=model,
            beliefs=beliefs,
            world=world,
        )

    def _score_single_physical(
        self,
        prev_state: Any,
        actual_state: Any,
        *,
        action: Any | None,
        object_id: str | None,
        context: str,
        confidence: float,
        model: WorldModel,
        beliefs: Sequence[Any] | None,
        world: Any | None,
    ) -> SurpriseSignal:
        if isinstance(prev_state, StateEmbedding):
            pred_emb = prev_state
        else:
            pred_emb = model.predict(prev_state, action)

        actual_emb = (
            actual_state
            if isinstance(actual_state, StateEmbedding)
            else model.predict(actual_state, None)
            if isinstance(model, NullWorldModel)
            else actual_state
        )
        raw_error = float(model.error(pred_emb, actual_emb))

        # Build a coarse signature for habituation so identical physical anomalies habituate
        err_bucket = round(raw_error, 1)
        signature = f"physical:{context}:{err_bucket}"
        score, usual, hab = self._compute_normalized_score(
            error=raw_error,
            confidence=confidence,
            context=context,
            signature=signature,
            habituate=raw_error > 1e-6,
        )

        signal = SurpriseSignal(
            score=score,
            kind="physical",
            details={
                "error": raw_error,
                "error_score": score,
                "confidence": float(confidence),
                "usual_error": usual,
                "habituation": hab,
                "context": context,
                "object_id": object_id,
                "signature": signature,
                "predicted": list(pred_emb),
            },
        )
        target_obj = actual_state if hasattr(actual_state, "state") else None
        self._record_surprise_pull(
            signal,
            object_id=object_id,
            beliefs=beliefs,
            world=world,
            target_object=target_obj,
        )
        return signal

    @staticmethod
    def _extract_object_map(state: Any) -> dict[str, Any]:
        if state is None or isinstance(state, (StateEmbedding, list, tuple)):
            return {}
        if hasattr(state, "objects"):
            objs = getattr(state, "objects")
            if isinstance(objs, Mapping):
                return {str(k): v for k, v in objs.items()}
            if isinstance(objs, Sequence):
                result: dict[str, Any] = {}
                for item in objs:
                    oid = getattr(item, "id", None) or (
                        item.get("id") if isinstance(item, Mapping) else None
                    )
                    if oid is not None:
                        result[str(oid)] = item
                return result
        if isinstance(state, Mapping) and "objects" in state:
            objs = state["objects"]
            if isinstance(objs, Mapping):
                return {str(k): v for k, v in objs.items()}
            if isinstance(objs, Sequence):
                result = {}
                for item in objs:
                    oid = getattr(item, "id", None) or (
                        item.get("id") if isinstance(item, Mapping) else None
                    )
                    if oid is not None:
                        result[str(oid)] = item
                return result
        return {}

    @staticmethod
    def _infer_object_id(state: Any) -> str | None:
        if state is None:
            return None
        if hasattr(state, "id") and isinstance(getattr(state, "id"), str):
            return str(getattr(state, "id"))
        if hasattr(state, "object_id") and getattr(state, "object_id") is not None:
            return str(getattr(state, "object_id"))
        if isinstance(state, Mapping):
            for key in ("object_id", "id", "target", "target_id"):
                if state.get(key) is not None:
                    return str(state[key])
        return None

    # ------------------------------------------------------------------
    # 2. Expectation surprise (when an intent ends, compare expect vs outcome)
    # ------------------------------------------------------------------

    def score_expectation(
        self,
        intent: Any,
        outcome: Any = None,
        *,
        before_state: Any | None = None,
        after_state: Any | None = None,
        object_id: str | None = None,
        context: str | None = None,
        use_llm_judge: bool | None = None,
        llm: Any | None = None,
        beliefs: Sequence[Any] | None = None,
        world: Any | None = None,
    ) -> SurpriseSignal:
        """Score expectation surprise when an intent ends.

        1. Performs a cheap structured comparison of predicted vs actual object
           movement/position/status where possible.
        2. If the structured movement check cannot resolve the comparison, falls
           back to an optional LLM ``judge`` call when ``use_llm_judge`` is enabled,
           or a cheap lexical comparison otherwise.
        3. Weights by the intent's ``confidence``.
        """

        expect_text, intent_conf, intent_goal, intent_target = self._extract_intent_fields(
            intent
        )
        inferred_oid = (
            object_id
            or intent_target
            or self._infer_object_id(outcome)
            or self._infer_object_id(after_state)
            or self._infer_object_id(before_state)
        )
        ctx = str(context or inferred_oid or intent_goal or "expectation")

        # Step 1: Cheap structured check (predicted vs actual object movement / displacement / position)
        cheap_resolved, cheap_error, cheap_reason, cheap_meta = self._cheap_expectation_check(
            expect_text=expect_text,
            intent=intent,
            outcome=outcome,
            before_state=before_state,
            after_state=after_state,
            object_id=inferred_oid,
        )

        allow_llm = self.config.use_llm_judge if use_llm_judge is None else bool(use_llm_judge)
        active_llm = llm if llm is not None else self.llm
        used_llm_judge = False

        if cheap_resolved:
            raw_error = cheap_error
            reason = cheap_reason
            method = "cheap_structured"
        elif allow_llm and active_llm is not None:
            raw_error, reason = self._llm_judge_expectation(
                active_llm,
                expect_text=expect_text,
                intent_goal=intent_goal,
                outcome=outcome,
                before_state=before_state,
                after_state=after_state,
            )
            used_llm_judge = True
            method = "llm_judge"
        else:
            raw_error, reason = self._cheap_text_expectation_fallback(
                expect_text=expect_text,
                outcome=outcome,
                before_state=before_state,
                after_state=after_state,
            )
            method = "cheap_text"

        outcome_summary = self._summarize_outcome(outcome, before_state, after_state)
        signature = f"expectation:{ctx}:{expect_text.strip().lower()}:{round(raw_error, 1)}"
        score, usual, hab = self._compute_normalized_score(
            error=raw_error,
            confidence=intent_conf,
            context=ctx,
            signature=signature,
            habituate=raw_error > 1e-6,
        )

        details: dict[str, Any] = {
            "error": raw_error,
            "error_score": score,
            "confidence": intent_conf,
            "usual_error": usual,
            "habituation": hab,
            "context": ctx,
            "object_id": inferred_oid,
            "expect": expect_text,
            "goal": intent_goal,
            "actual_summary": outcome_summary,
            "reason": reason,
            "method": method,
            "used_llm_judge": used_llm_judge,
            "signature": signature,
            **cheap_meta,
        }
        signal = SurpriseSignal(score=score, kind="expectation", details=details)
        self._record_surprise_pull(
            signal,
            object_id=inferred_oid,
            beliefs=beliefs,
            world=world,
        )
        return signal

    @staticmethod
    def _extract_intent_fields(intent: Any) -> tuple[str, float, str, str | None]:
        if intent is None:
            return "", 1.0, "", None
        if isinstance(intent, str):
            return intent, 1.0, intent, None
        if isinstance(intent, Mapping):
            expect = str(intent.get("expect", intent.get("expected", "")))
            conf = float(intent.get("confidence", 1.0))
            goal = str(intent.get("goal", intent.get("intent", "")))
            target = intent.get("target") or intent.get("object_id") or intent.get("target_id")
            if target is None:
                target = SurpriseScorer._extract_target_from_text(f"{goal} {expect}")
            return expect, conf, goal, (str(target) if target is not None else None)

        expect = str(getattr(intent, "expect", getattr(intent, "expected", "")))
        conf = float(getattr(intent, "confidence", 1.0))
        goal = str(getattr(intent, "goal", ""))
        target = (
            getattr(intent, "target", None)
            or getattr(intent, "object_id", None)
            or getattr(intent, "target_id", None)
        )
        if target is None:
            target = SurpriseScorer._extract_target_from_text(f"{goal} {expect}")
        return expect, conf, goal, (str(target) if target is not None else None)

    @staticmethod
    def _extract_target_from_text(text: str) -> str | None:
        match = re.search(r"\b([a-zA-Z]+_[0-9]+|[a-zA-Z]+-[0-9]+)\b", text)
        if match:
            return match.group(1)
        return None

    def _cheap_expectation_check(
        self,
        *,
        expect_text: str,
        intent: Any,
        outcome: Any,
        before_state: Any | None,
        after_state: Any | None,
        object_id: str | None,
    ) -> tuple[bool, float, str, dict[str, Any]]:
        """Attempt a cheap structured comparison of predicted vs actual object movement."""

        meta: dict[str, Any] = {}

        # 1. Check explicit structured fields on intent / outcome if present
        pred_delta = self._extract_vector(intent, ("predicted_delta", "expected_delta", "predicted_movement", "expected_movement"))
        if pred_delta is None and isinstance(outcome, Mapping):
            pred_delta = self._extract_vector(outcome, ("predicted_delta", "expected_delta", "predicted_movement", "expected_movement"))

        actual_delta = self._extract_vector(outcome, ("actual_delta", "delta", "displacement", "movement", "actual_movement"))
        pred_pos = self._extract_vector(intent, ("predicted_position", "expected_position", "target_position"))
        if pred_pos is None and isinstance(outcome, Mapping):
            pred_pos = self._extract_vector(outcome, ("predicted_position", "expected_position"))

        before_pos, after_pos, actual_moved_flag = self._extract_movement_from_states(
            outcome=outcome,
            before_state=before_state,
            after_state=after_state,
            object_id=object_id,
        )

        if actual_delta is None and before_pos is not None and after_pos is not None:
            actual_delta = (after_pos[0] - before_pos[0], after_pos[1] - before_pos[1])

        if after_pos is None and before_pos is not None and actual_delta is not None:
            after_pos = (before_pos[0] + actual_delta[0], before_pos[1] + actual_delta[1])

        if pred_delta is not None and actual_delta is not None:
            dist = math.hypot(pred_delta[0] - actual_delta[0], pred_delta[1] - actual_delta[1])
            scale = max(1.0, math.hypot(pred_delta[0], pred_delta[1]), math.hypot(actual_delta[0], actual_delta[1]))
            err = 0.0 if dist <= self.config.movement_tolerance else min(1.0, dist / scale)
            meta["predicted_delta"] = pred_delta
            meta["actual_delta"] = actual_delta
            return True, err, "structured_delta_comparison", meta

        if pred_pos is not None and after_pos is not None:
            dist = math.hypot(pred_pos[0] - after_pos[0], pred_pos[1] - after_pos[1])
            err = 0.0 if dist <= self.config.movement_tolerance else min(1.0, dist / max(1.0, dist))
            meta["predicted_position"] = pred_pos
            meta["actual_position"] = after_pos
            return True, err, "structured_position_comparison", meta

        # 2. Extract movement expectation from `expect_text` (or structured `expected_moved` bool)
        expect_moved, expect_dir, expect_target_pos = self._parse_movement_expectation(
            expect_text, intent
        )

        if expect_target_pos is not None and after_pos is not None:
            dist = math.hypot(expect_target_pos[0] - after_pos[0], expect_target_pos[1] - after_pos[1])
            err = 0.0 if dist <= self.config.movement_tolerance else min(1.0, dist / max(2.0, dist))
            meta["predicted_position"] = expect_target_pos
            meta["actual_position"] = after_pos
            return True, err, "parsed_position_comparison", meta

        # Determine whether the object actually moved from coordinates or explicit flags
        actual_moved: bool | None = actual_moved_flag
        actual_dist: float | None = None
        if actual_delta is not None:
            actual_dist = math.hypot(actual_delta[0], actual_delta[1])
            actual_moved = actual_dist > self.config.movement_tolerance
            meta["actual_delta"] = actual_delta
            meta["actual_distance"] = actual_dist

        if expect_moved is not None and actual_moved is not None:
            meta["expected_moved"] = expect_moved
            meta["actual_moved"] = actual_moved
            if expect_moved != actual_moved:
                return True, 1.0, "movement_mismatch", meta

            # Both moved: check direction if a direction was predicted
            if expect_moved and expect_dir is not None and actual_delta is not None and actual_dist and actual_dist > 1e-6:
                dot = (
                    (actual_delta[0] / actual_dist) * expect_dir[0]
                    + (actual_delta[1] / actual_dist) * expect_dir[1]
                )
                if dot < 0.2:
                    err = min(1.0, (1.0 - dot) / 2.0)
                    meta["expected_direction"] = expect_dir
                    return True, err, "direction_mismatch", meta

            return True, 0.0, "movement_matched", meta

        return False, 0.0, "structured_inconclusive", meta

    @staticmethod
    def _extract_vector(obj: Any, keys: Sequence[str]) -> tuple[float, float] | None:
        if obj is None:
            return None
        for key in keys:
            val = getattr(obj, key, None)
            if val is None and isinstance(obj, Mapping):
                val = obj.get(key)
            if isinstance(val, (tuple, list)) and len(val) >= 2:
                return (float(val[0]), float(val[1]))
            if isinstance(val, Mapping) and ("x" in val or "dx" in val):
                x = val.get("x", val.get("dx", 0.0))
                y = val.get("y", val.get("dy", 0.0))
                return (float(x), float(y))
        return None

    def _extract_movement_from_states(
        self,
        *,
        outcome: Any,
        before_state: Any | None,
        after_state: Any | None,
        object_id: str | None,
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None, bool | None]:
        before_pos: tuple[float, float] | None = None
        after_pos: tuple[float, float] | None = None
        moved_flag: bool | None = None

        if isinstance(outcome, Mapping):
            if "moved" in outcome and isinstance(outcome["moved"], bool):
                moved_flag = outcome["moved"]
            elif "object_moved" in outcome and isinstance(outcome["object_moved"], bool):
                moved_flag = outcome["object_moved"]
            before_pos = self._extract_vector(
                outcome, ("before_position", "prev_position", "start_position", "initial_position", "from_position")
            )
            after_pos = self._extract_vector(
                outcome, ("after_position", "current_position", "end_position", "final_position", "to_position", "position")
            )
            if before_state is None:
                before_state = outcome.get("before") or outcome.get("prev_state") or outcome.get("before_state")
            if after_state is None:
                after_state = outcome.get("after") or outcome.get("current_state") or outcome.get("after_state")
        elif outcome is not None:
            if hasattr(outcome, "moved") and isinstance(getattr(outcome, "moved"), bool):
                moved_flag = bool(getattr(outcome, "moved"))
            before_pos = self._extract_vector(
                outcome, ("before_position", "prev_position", "start_position", "initial_position")
            )
            after_pos = self._extract_vector(
                outcome, ("after_position", "current_position", "end_position", "final_position")
            )
            if before_state is None:
                before_state = getattr(outcome, "before", getattr(outcome, "before_state", None))
            if after_state is None:
                after_state = getattr(outcome, "after", getattr(outcome, "after_state", None))

        if before_pos is None and before_state is not None:
            before_pos = self._extract_object_position(before_state, object_id)
        if after_pos is None and after_state is not None:
            after_pos = self._extract_object_position(after_state, object_id)

        return before_pos, after_pos, moved_flag

    def _extract_object_position(
        self, state: Any, object_id: str | None
    ) -> tuple[float, float] | None:
        if state is None:
            return None
        objs = self._extract_object_map(state)
        if objs:
            if object_id and object_id in objs:
                return self._extract_object_position(objs[object_id], None)
            if len(objs) == 1:
                only_obj = next(iter(objs.values()))
                return self._extract_object_position(only_obj, None)
            return None

        if hasattr(state, "x") and hasattr(state, "y"):
            return (float(getattr(state, "x")), float(getattr(state, "y")))
        if isinstance(state, Mapping):
            if "x" in state and "y" in state:
                return (float(state["x"]), float(state["y"]))
            if "position" in state and isinstance(state["position"], (tuple, list)) and len(state["position"]) >= 2:
                return (float(state["position"][0]), float(state["position"][1]))
        if isinstance(state, (tuple, list)) and len(state) >= 2 and all(isinstance(v, (int, float)) for v in state[:2]):
            return (float(state[0]), float(state[1]))
        return None

    @staticmethod
    def _parse_movement_expectation(
        expect_text: str, intent: Any
    ) -> tuple[bool | None, tuple[float, float] | None, tuple[float, float] | None]:
        if isinstance(intent, Mapping) and "expected_moved" in intent:
            return bool(intent["expected_moved"]), None, None
        if hasattr(intent, "expected_moved") and getattr(intent, "expected_moved") is not None:
            return bool(getattr(intent, "expected_moved")), None, None

        tokens = _tokenize(expect_text)
        if not tokens:
            return None, None, None

        # Coordinate target in expect text, e.g. "(5, 6)" or "x=5, y=6"
        coord_match = re.search(
            r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", expect_text
        )
        target_pos: tuple[float, float] | None = None
        if coord_match:
            target_pos = (float(coord_match.group(1)), float(coord_match.group(2)))

        direction: tuple[float, float] | None = None
        for tok in tokens:
            if tok in _DIRECTION_VECTORS:
                direction = _DIRECTION_VECTORS[tok]
                break

        token_set = set(tokens)
        has_move_word = bool(token_set & _MOVEMENT_WORDS)
        has_stationary_word = bool(token_set & _STATIONARY_WORDS)

        if has_stationary_word and not _has_negation_near(tokens, _STATIONARY_WORDS):
            return False, None, target_pos

        if has_move_word:
            if _has_negation_near(tokens, _MOVEMENT_WORDS):
                return False, None, target_pos
            return True, direction, target_pos

        return None, direction, target_pos

    def _cheap_text_expectation_fallback(
        self,
        *,
        expect_text: str,
        outcome: Any,
        before_state: Any | None,
        after_state: Any | None,
    ) -> tuple[float, str]:
        outcome_text = self._summarize_outcome(outcome, before_state, after_state)
        if not expect_text.strip() or not outcome_text.strip():
            return 0.0, "empty_expectation_or_outcome"

        exp_tokens = _tokenize(expect_text)
        out_tokens = _tokenize(outcome_text)
        exp_set = set(exp_tokens)
        out_set = set(out_tokens)

        # Check explicit outcome failure status
        if isinstance(outcome, Mapping):
            if outcome.get("success") is False or outcome.get("status") in {"failed", "aborted", "blocked", "timeout"}:
                if not (exp_set & {"fail", "failed", "blocked", "impossible"}):
                    return 1.0, "outcome_status_failed"
            if outcome.get("success") is True and (exp_set & {"fail", "failed", "blocked"}):
                return 1.0, "outcome_status_unexpected_success"

        # Check antonym / negation contradiction between expect_text and outcome_text
        for group_a, group_b in _ANTONYM_GROUPS:
            exp_in_a = bool(exp_set & group_a) and not _has_negation_near(exp_tokens, group_a)
            exp_in_b = bool(exp_set & group_b) or _has_negation_near(exp_tokens, group_a)
            out_in_a = bool(out_set & group_a) and not _has_negation_near(out_tokens, group_a)
            out_in_b = bool(out_set & group_b) or _has_negation_near(out_tokens, group_a)

            if (exp_in_a and out_in_b) or (exp_in_b and out_in_a):
                return 1.0, "contradictory_outcome_predicates"

        # Check negated shared predicate
        content_exp = exp_set - _STOPWORDS - _NEGATION_WORDS
        content_out = out_set - _STOPWORDS - _NEGATION_WORDS
        shared_content = content_exp & content_out
        if shared_content:
            exp_neg = any(t in _NEGATION_WORDS for t in exp_tokens)
            out_neg = any(t in _NEGATION_WORDS for t in out_tokens)
            if exp_neg != out_neg:
                return 1.0, "negation_mismatch"
            return 0.0, "shared_predicate_match"

        return 0.0, "no_contradiction_detected"

    def _llm_judge_expectation(
        self,
        llm: Any,
        *,
        expect_text: str,
        intent_goal: str,
        outcome: Any,
        before_state: Any | None,
        after_state: Any | None,
    ) -> tuple[float, str]:
        outcome_text = self._summarize_outcome(outcome, before_state, after_state)
        user_prompt = (
            f"Intent goal: {intent_goal}\n"
            f"Expected outcome: {expect_text}\n"
            f"Actual outcome: {outcome_text}\n"
            "Return JSON with fields: {\"error\": float in [0, 1], \"surprised\": bool, \"reason\": str}."
        )
        system = (
            "You are Arden's expectation judge. Compare the expected outcome of an intent "
            "with the actual observed outcome and rate the discrepancy error from 0.0 (matched) "
            "to 1.0 (completely contradicted)."
        )
        judge_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "error": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "surprised": {"type": "boolean"},
                "reason": {"type": "string"},
            },
            "additionalProperties": True,
        }

        payload: Any = None
        if hasattr(llm, "complete_json") and callable(llm.complete_json):
            try:
                payload = llm.complete_json("judge", system, user_prompt, judge_schema, 10.0)
            except TypeError:
                try:
                    payload = llm.complete_json("judge", user_prompt, system=system)
                except TypeError:
                    payload = llm.complete_json("judge", user_prompt)
        elif hasattr(llm, "complete") and callable(llm.complete):
            try:
                raw = llm.complete("judge", user_prompt, system=system)
            except TypeError:
                raw = llm.complete("judge", user_prompt)
            import json

            payload = json.loads(raw) if isinstance(raw, str) else raw
        elif callable(llm):
            try:
                payload = llm("judge", user_prompt)
            except TypeError:
                payload = llm(user_prompt)

        if isinstance(payload, Mapping):
            if "error" in payload:
                err = max(0.0, min(1.0, float(payload["error"])))
            elif "score" in payload:
                err = max(0.0, min(1.0, float(payload["score"])))
            elif "surprised" in payload:
                err = 1.0 if bool(payload["surprised"]) else 0.0
            else:
                err = 0.0
            reason = str(payload.get("reason", "llm_judge"))
            return err, reason

        return 0.0, "llm_judge_empty"

    @staticmethod
    def _summarize_outcome(
        outcome: Any, before_state: Any | None = None, after_state: Any | None = None
    ) -> str:
        if isinstance(outcome, str):
            return outcome
        if isinstance(outcome, Mapping):
            for key in ("summary", "outcome", "description", "observation", "text", "status"):
                if key in outcome and isinstance(outcome[key], str):
                    return outcome[key]
            return str(dict(outcome))
        if outcome is not None:
            for attr in ("summary", "outcome", "description", "observation", "text", "status"):
                val = getattr(outcome, attr, None)
                if isinstance(val, str):
                    return val
            return str(outcome)
        if after_state is not None:
            return f"before={before_state!r} after={after_state!r}"
        return ""

    # ------------------------------------------------------------------
    # 3. Belief surprise (when observation contradicts retrieved belief's tags/predicate)
    # ------------------------------------------------------------------

    def score_belief(
        self,
        observation: Any,
        beliefs: Sequence[Any] | Any | None = None,
        *,
        object_id: str | None = None,
        context: str | None = None,
        world: Any | None = None,
    ) -> SurpriseSignal:
        """Score belief surprise when an observation contradicts a retrieved belief's tags or predicate.

        Weights the surprise score by the contradicted belief's ``confidence``.
        """

        obs_text, obs_tokens, obs_dict, inferred_oid = self._normalize_observation(
            observation, object_id=object_id
        )
        candidate_beliefs = self._collect_beliefs(
            beliefs, query=inferred_oid or obs_text
        )

        best_error = 0.0
        best_conf = 1.0
        best_belief: Any | None = None
        best_reason = "no_contradiction"

        for belief in candidate_beliefs:
            stmt, tags, conf, status = self._extract_belief_fields(belief)
            if status != "active" or conf <= 0.0:
                continue

            err, reason = self._check_single_belief_contradiction(
                observation=obs_text,
                obs_tokens=obs_tokens,
                obs_dict=obs_dict,
                belief_statement=stmt,
                belief_tags=tags,
                object_id=inferred_oid,
            )
            if err * conf > best_error * best_conf:
                best_error = err
                best_conf = conf
                best_belief = belief
                best_reason = reason

        ctx = str(context or inferred_oid or "belief")
        belief_id = (
            getattr(best_belief, "id", None)
            if best_belief is not None
            else None
        ) or (
            best_belief.get("id")
            if isinstance(best_belief, Mapping)
            else None
        )
        signature = f"belief:{ctx}:{belief_id or best_reason}:{round(best_error, 1)}"
        score, usual, hab = self._compute_normalized_score(
            error=best_error,
            confidence=best_conf if best_error > 0.0 else 1.0,
            context=ctx,
            signature=signature,
            habituate=best_error > 1e-6,
        )

        stmt, tags, conf, _ = (
            self._extract_belief_fields(best_belief)
            if best_belief is not None
            else ("", (), 1.0, "active")
        )
        details: dict[str, Any] = {
            "error": best_error,
            "error_score": score,
            "confidence": best_conf if best_error > 0.0 else 1.0,
            "usual_error": usual,
            "habituation": hab,
            "context": ctx,
            "object_id": inferred_oid,
            "belief_id": belief_id,
            "belief_statement": stmt,
            "belief_tags": list(tags),
            "observation": obs_text,
            "observation_dict": obs_dict,
            "reason": best_reason,
            "signature": signature,
        }
        signal = SurpriseSignal(score=score, kind="belief", details=details)
        target_obj = observation if hasattr(observation, "state") else None
        self._record_surprise_pull(
            signal,
            object_id=inferred_oid,
            beliefs=candidate_beliefs,
            world=world,
            target_object=target_obj,
        )
        return signal

    @staticmethod
    def _extract_belief_fields(belief: Any) -> tuple[str, tuple[str, ...], float, str]:
        if isinstance(belief, str):
            return belief, (), 1.0, "active"
        if isinstance(belief, Mapping):
            stmt = str(belief.get("statement", belief.get("predicate", belief.get("text", ""))))
            raw_tags = belief.get("tags", ())
            tags = tuple(str(t) for t in raw_tags) if isinstance(raw_tags, Sequence) and not isinstance(raw_tags, str) else (str(raw_tags),) if raw_tags else ()
            conf = float(belief.get("confidence", 1.0))
            status = str(belief.get("status", "active"))
            return stmt, tags, conf, status

        stmt = str(getattr(belief, "statement", getattr(belief, "predicate", "")))
        raw_tags = getattr(belief, "tags", ()) or ()
        tags = tuple(str(t) for t in raw_tags)
        conf = float(getattr(belief, "confidence", 1.0))
        status = str(getattr(belief, "status", "active"))
        return stmt, tags, conf, status

    def _normalize_observation(
        self, observation: Any, *, object_id: str | None = None
    ) -> tuple[str, list[str], dict[str, Any], str | None]:
        obs_dict: dict[str, Any] = {}
        inferred_oid = object_id or self._infer_object_id(observation)

        if isinstance(observation, str):
            if inferred_oid is None:
                inferred_oid = self._extract_target_from_text(observation)
            return observation, _tokenize(observation), obs_dict, inferred_oid

        if isinstance(observation, Mapping):
            obs_dict = dict(observation)
        elif hasattr(observation, "__dict__") or hasattr(observation, "__slots__"):
            for key in (
                "id",
                "object_id",
                "kind",
                "x",
                "y",
                "vx",
                "vy",
                "moved",
                "wet",
                "broken",
                "heavy",
                "grabbable",
                "properties",
                "affordances",
                "state",
                "tags",
                "summary",
                "text",
                "observation",
            ):
                if hasattr(observation, key):
                    obs_dict[key] = getattr(observation, key)

        parts: list[str] = []
        if inferred_oid:
            parts.append(inferred_oid)

        for text_key in ("summary", "text", "observation", "description", "predicate", "event"):
            if text_key in obs_dict and isinstance(obs_dict[text_key], str):
                parts.append(obs_dict[text_key])

        if "tags" in obs_dict and isinstance(obs_dict["tags"], Sequence) and not isinstance(obs_dict["tags"], str):
            parts.extend(str(t) for t in obs_dict["tags"])

        props = obs_dict.get("properties")
        if isinstance(props, Mapping):
            for k, v in props.items():
                if isinstance(v, bool):
                    parts.append(str(k) if v else f"not {k}")
                else:
                    parts.append(f"{k} {v}")

        st = obs_dict.get("state")
        if isinstance(st, Mapping):
            for k, v in st.items():
                if k in ("pull", "unexplained_pull"):
                    continue
                if isinstance(v, bool):
                    parts.append(str(k) if v else f"not {k}")
                else:
                    parts.append(f"{k} {v}")

        # Velocity / movement flags in structured observation
        vx = float(obs_dict.get("vx", 0.0) or 0.0)
        vy = float(obs_dict.get("vy", 0.0) or 0.0)
        delta = self._extract_vector(obs_dict, ("delta", "displacement", "movement", "actual_delta"))
        moved = obs_dict.get("moved")
        if moved is True or math.hypot(vx, vy) > 1e-3 or (delta is not None and math.hypot(delta[0], delta[1]) > 1e-3):
            parts.append("moved moving")
            obs_dict["moved"] = True
        elif moved is False:
            parts.append("stationary unmoved")
            obs_dict["moved"] = False

        for bool_key, pos_word, neg_word in (
            ("wet", "wet", "dry"),
            ("broken", "broken", "intact"),
            ("intact", "intact", "broken"),
            ("stationary", "stationary", "moved"),
            ("immovable", "immovable", "moved"),
            ("grabbable", "grabbable", "ungrabbable"),
        ):
            if bool_key in obs_dict and isinstance(obs_dict[bool_key], bool):
                parts.append(pos_word if obs_dict[bool_key] else neg_word)

        obs_text = " ".join(parts) if parts else str(observation)
        return obs_text, _tokenize(obs_text), obs_dict, inferred_oid

    def _check_single_belief_contradiction(
        self,
        *,
        observation: str,
        obs_tokens: Sequence[str],
        obs_dict: Mapping[str, Any] | None,
        belief_statement: str,
        belief_tags: Sequence[str],
        object_id: str | None,
    ) -> tuple[float, str]:
        stmt_tokens = _tokenize(belief_statement)
        tag_tokens: list[str] = []
        for tag in belief_tags:
            tag_tokens.extend(_tokenize(str(tag)))
        all_belief_tokens = stmt_tokens + tag_tokens
        belief_set = set(all_belief_tokens)
        obs_set = set(obs_tokens)

        # If an object_id or subject is specified in the belief, ensure relevance when multiple objects exist
        belief_oid = self._extract_target_from_text(belief_statement)
        if belief_oid is None:
            for tag in belief_tags:
                belief_oid = self._extract_target_from_text(str(tag))
                if belief_oid:
                    break
        if object_id and belief_oid and object_id.lower() != belief_oid.lower():
            return 0.0, "different_object"

        # 1. Check tag-level negation or opposite tags (e.g., tag "stationary" vs obs "moved", or tag "not_wet" vs "wet")
        for raw_tag in belief_tags:
            tag_str = str(raw_tag).lower().strip()
            t_tokens = _tokenize(tag_str)
            if tag_str.startswith(("not_", "non_", "un", "im")) and len(tag_str) > 3:
                for prefix in ("not_", "non_", "un_", "un", "im"):
                    if tag_str.startswith(prefix) and len(tag_str) > len(prefix) + 1:
                        stem = tag_str[len(prefix) :]
                        if stem in obs_set and not _has_negation_near(obs_tokens, frozenset({stem})):
                            return 1.0, f"tag_contradiction:{tag_str}"
            if t_tokens:
                t_set = frozenset(t_tokens)
                if bool(obs_set & t_set) and _has_negation_near(obs_tokens, t_set):
                    return 1.0, f"negated_tag:{tag_str}"

        # 2. Check antonym groups across belief (tags + statement) and observation
        for group_a, group_b in _ANTONYM_GROUPS:
            belief_in_a = (
                bool(belief_set & group_a)
                and not _has_negation_near(stmt_tokens, group_a)
                and not _has_negation_near(tag_tokens, group_a)
            )
            belief_in_b = (
                bool(belief_set & group_b)
                and not _has_negation_near(stmt_tokens, group_b)
                and not _has_negation_near(tag_tokens, group_b)
            ) or _has_negation_near(stmt_tokens, group_a)
            if not belief_in_a and _has_negation_near(stmt_tokens, group_b):
                belief_in_a = True

            obs_in_a = bool(obs_set & group_a) and not _has_negation_near(obs_tokens, group_a)
            obs_in_b = (
                bool(obs_set & group_b) and not _has_negation_near(obs_tokens, group_b)
            ) or _has_negation_near(obs_tokens, group_a)
            if not obs_in_a and _has_negation_near(obs_tokens, group_b):
                obs_in_a = True

            if (belief_in_a and obs_in_b) or (belief_in_b and obs_in_a):
                return 1.0, "antonym_predicate_contradiction"

        # 3. Check mutually exclusive property values (colors, shapes)
        belief_colors = (belief_set & _COLOR_WORDS)
        obs_colors = (obs_set & _COLOR_WORDS)
        if belief_colors and obs_colors and belief_colors.isdisjoint(obs_colors):
            return 1.0, f"color_contradiction:{sorted(belief_colors)}!={sorted(obs_colors)}"

        belief_shapes = (belief_set & _SHAPE_WORDS)
        obs_shapes = (obs_set & _SHAPE_WORDS)
        if belief_shapes and obs_shapes and belief_shapes.isdisjoint(obs_shapes):
            return 1.0, f"shape_contradiction:{sorted(belief_shapes)}!={sorted(obs_shapes)}"

        # 4. Check direct predicate negation mismatch (e.g. belief: "ball_1 rolls on push" vs obs: "ball_1 did not roll")
        oid_tokens = set(_tokenize(object_id or belief_oid or ""))
        content_belief = belief_set - _STOPWORDS - _NEGATION_WORDS - oid_tokens
        content_obs = obs_set - _STOPWORDS - _NEGATION_WORDS - oid_tokens
        shared_predicates = content_belief & content_obs
        if shared_predicates:
            belief_neg = _has_negation_near(stmt_tokens, frozenset(shared_predicates)) or any(
                t in _NEGATION_WORDS for t in stmt_tokens
            )
            obs_neg = _has_negation_near(obs_tokens, frozenset(shared_predicates)) or any(
                t in _NEGATION_WORDS for t in obs_tokens
            )
            if belief_neg != obs_neg:
                return 1.0, "predicate_negation_contradiction"

        return 0.0, "consistent"

    # ------------------------------------------------------------------
    # 4. Novelty (reported as its own kind and never folded into error score)
    # ------------------------------------------------------------------

    def score_novelty(
        self,
        observation_or_object: Any,
        *,
        object_id: str | None = None,
        context: str | None = None,
    ) -> SurpriseSignal:
        """Score novelty as its own kind (``kind='novelty'``) without folding into error score."""

        objs = self._extract_object_list(observation_or_object)
        if len(objs) > 1 and object_id is None:
            best_sig: SurpriseSignal | None = None
            for obj in objs:
                sig = self._score_single_novelty(obj, context=context)
                if best_sig is None or sig.score > best_sig.score:
                    best_sig = sig
            assert best_sig is not None
            return best_sig

        target = objs[0] if len(objs) == 1 else observation_or_object
        return self._score_single_novelty(target, object_id=object_id, context=context)

    def _score_single_novelty(
        self,
        item: Any,
        *,
        object_id: str | None = None,
        context: str | None = None,
    ) -> SurpriseSignal:
        oid = object_id or self._infer_object_id(item)
        kind_name = getattr(item, "kind", None) or (
            item.get("kind") if isinstance(item, Mapping) else None
        )
        props = getattr(item, "properties", None) or (
            item.get("properties") if isinstance(item, Mapping) else None
        )
        affordances = getattr(item, "affordances", None) or (
            item.get("affordances") if isinstance(item, Mapping) else None
        )

        prop_items: list[tuple[str, Any]] = []
        if kind_name:
            prop_items.append(("kind", str(kind_name)))
        if isinstance(props, Mapping):
            for k in sorted(props.keys()):
                v = props[k]
                if isinstance(v, (list, dict, set)):
                    v = str(v)
                prop_items.append((str(k), v))
        prop_combo = tuple(prop_items)

        aff_list: list[str] = []
        if isinstance(affordances, (Sequence, set, frozenset)) and not isinstance(affordances, str):
            aff_list = sorted(str(a) for a in affordances)

        new_object = False
        new_property_combo = False
        new_affordances: list[str] = []

        if oid is not None and oid not in self._seen_object_ids:
            new_object = True
            self._seen_object_ids.add(oid)

        if prop_combo and prop_combo not in self._seen_property_combos:
            new_property_combo = True
            self._seen_property_combos.add(prop_combo)

        for aff in aff_list:
            if aff not in self._seen_affordances:
                new_affordances.append(aff)
                self._seen_affordances.add(aff)
            elif oid is not None and (oid, aff) not in self._seen_object_affordances:
                new_affordances.append(aff)
            if oid is not None:
                self._seen_object_affordances.add((oid, aff))

        # Raw novelty components (kept completely separate from prediction error)
        novelty_components = 0.0
        if new_object:
            novelty_components += 0.5
        if new_property_combo:
            novelty_components += 0.35
        if new_affordances:
            novelty_components += min(0.35, 0.2 * len(new_affordances))
        if not new_object and not new_property_combo and not new_affordances and oid is None and isinstance(item, str):
            # String novelty fallback
            key = item.strip().lower()
            if key and key not in self._seen_object_ids:
                self._seen_object_ids.add(key)
                novelty_components = 1.0

        raw_novelty = min(1.0, novelty_components)
        ctx = str(context or oid or kind_name or "novelty")
        sig_key = f"novelty:{oid or ''}:{prop_combo}:{tuple(aff_list)}"
        hab = self.habituation_factor(sig_key, increment=True)
        score = raw_novelty * hab

        return SurpriseSignal(
            score=score,
            kind="novelty",
            details={
                "error": 0.0,
                "error_score": 0.0,
                "novelty": raw_novelty,
                "novelty_score": score,
                "confidence": 1.0,
                "usual_error": self.usual_error(ctx),
                "habituation": hab,
                "context": ctx,
                "object_id": oid,
                "new_object": new_object,
                "new_property_combo": new_property_combo,
                "new_affordances": new_affordances,
                "signature": sig_key,
            },
        )

    @staticmethod
    def _extract_object_list(percept_or_obj: Any) -> list[Any]:
        if percept_or_obj is None:
            return []
        if hasattr(percept_or_obj, "objects"):
            objs = getattr(percept_or_obj, "objects")
            if isinstance(objs, Mapping):
                return list(objs.values())
            if isinstance(objs, Sequence) and not isinstance(objs, (str, bytes)):
                return list(objs)
        if isinstance(percept_or_obj, Mapping) and "objects" in percept_or_obj:
            objs = percept_or_obj["objects"]
            if isinstance(objs, Mapping):
                return list(objs.values())
            if isinstance(objs, Sequence) and not isinstance(objs, (str, bytes)):
                return list(objs)
        if isinstance(percept_or_obj, Sequence) and not isinstance(percept_or_obj, (str, bytes, StateEmbedding)):
            return list(percept_or_obj)
        return [percept_or_obj]

    # ------------------------------------------------------------------
    # Unified entry points: score() / evaluate() / __call__()
    # ------------------------------------------------------------------

    def score(
        self,
        *,
        kind: SurpriseKind | None = None,
        error: float | None = None,
        confidence: float = 1.0,
        context: str | None = None,
        signature: str | None = None,
        object_id: str | None = None,
        prev_state: Any | None = None,
        actual_state: Any | None = None,
        action: Any | None = None,
        intent: Any | None = None,
        outcome: Any | None = None,
        observation: Any | None = None,
        beliefs: Sequence[Any] | Any | None = None,
        novel_item: Any | None = None,
        world: Any | None = None,
        use_llm_judge: bool | None = None,
    ) -> SurpriseSignal:
        """Unified surprise scoring method.

        Can be called with:
        - Explicit ``kind`` and ``error`` to apply the normalization + habituation formula directly
        - ``intent`` / ``outcome`` for expectation surprise
        - ``observation`` / ``beliefs`` for belief surprise
        - ``prev_state`` / ``actual_state`` for physical surprise
        - ``novel_item`` (or ``kind='novelty'``) for novelty surprise
        """

        if kind == "novelty" or novel_item is not None:
            target = novel_item if novel_item is not None else (observation if observation is not None else actual_state)
            return self.score_novelty(target, object_id=object_id, context=context)

        if error is not None:
            resolved_kind: SurpriseKind = kind if kind is not None else "physical"
            ctx = str(context or object_id or resolved_kind)
            sig = str(signature or f"{resolved_kind}:{ctx}:{round(float(error), 2)}")
            val, usual, hab = self._compute_normalized_score(
                error=float(error),
                confidence=float(confidence),
                context=ctx,
                signature=sig,
                habituate=float(error) > 1e-6,
            )
            signal = SurpriseSignal(
                score=val,
                kind=resolved_kind,
                details={
                    "error": float(error),
                    "error_score": val,
                    "confidence": float(confidence),
                    "usual_error": usual,
                    "habituation": hab,
                    "context": ctx,
                    "object_id": object_id,
                    "signature": sig,
                },
            )
            self._record_surprise_pull(
                signal,
                object_id=object_id,
                beliefs=self._collect_beliefs(beliefs),
                world=world,
            )
            return signal

        if kind == "expectation" or (intent is not None and kind is None):
            return self.score_expectation(
                intent,
                outcome,
                before_state=prev_state,
                after_state=actual_state,
                object_id=object_id,
                context=context,
                use_llm_judge=use_llm_judge,
                beliefs=self._collect_beliefs(beliefs),
                world=world,
            )

        if kind == "belief" or (
            observation is not None and (beliefs is not None or self.belief_store is not None) and kind is None
        ):
            return self.score_belief(
                observation,
                beliefs,
                object_id=object_id,
                context=context,
                world=world,
            )

        if kind == "physical" or prev_state is not None or actual_state is not None:
            return self.score_physical(
                prev_state if prev_state is not None else actual_state,
                actual_state if prev_state is not None else None,
                action=action,
                object_id=object_id,
                context=context,
                confidence=confidence,
                beliefs=self._collect_beliefs(beliefs),
                world=world,
            )

        if observation is not None:
            return self.score_novelty(observation, object_id=object_id, context=context)

        ctx = str(context or object_id or "global")
        return SurpriseSignal(
            score=0.0,
            kind=kind or "physical",
            details={
                "error": 0.0,
                "error_score": 0.0,
                "confidence": float(confidence),
                "usual_error": self.usual_error(ctx),
                "habituation": 1.0,
                "context": ctx,
                "object_id": object_id,
            },
        )

    def evaluate_all(
        self,
        *,
        prev_state: Any | None = None,
        actual_state: Any | None = None,
        action: Any | None = None,
        intent: Any | None = None,
        outcome: Any | None = None,
        observation: Any | None = None,
        beliefs: Sequence[Any] | Any | None = None,
        object_id: str | None = None,
        context: str | None = None,
        world: Any | None = None,
        include_novelty: bool = True,
    ) -> list[SurpriseSignal]:
        """Evaluate all applicable surprise kinds and return their signals.

        Novelty is always returned as its own ``SurpriseSignal(kind='novelty')``
        and never added into any error score.
        """

        signals: list[SurpriseSignal] = []

        if intent is not None:
            signals.append(
                self.score_expectation(
                    intent,
                    outcome,
                    before_state=prev_state,
                    after_state=actual_state,
                    object_id=object_id,
                    context=context,
                    beliefs=self._collect_beliefs(beliefs),
                    world=world,
                )
            )

        obs_target = observation if observation is not None else actual_state
        if obs_target is not None and (beliefs is not None or self.belief_store is not None):
            signals.append(
                self.score_belief(
                    obs_target,
                    beliefs,
                    object_id=object_id,
                    context=context,
                    world=world,
                )
            )

        if prev_state is not None and actual_state is not None:
            signals.append(
                self.score_physical(
                    prev_state,
                    actual_state,
                    action=action,
                    object_id=object_id,
                    context=context,
                    beliefs=self._collect_beliefs(beliefs),
                    world=world,
                )
            )

        if include_novelty and obs_target is not None:
            signals.append(
                self.score_novelty(obs_target, object_id=object_id, context=context)
            )

        return signals

    def __call__(self, *args: Any, **kwargs: Any) -> SurpriseSignal:
        if len(args) == 1 and not kwargs:
            return self.score_physical(args[0])
        if len(args) == 2 and not kwargs:
            return self.score_physical(args[0], args[1])
        return self.score(**kwargs)

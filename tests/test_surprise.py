"""Tests for surprise scoring (T16): physical, expectation, belief, novelty, normalization, habituation, WorldModel, and unexplained pull."""

from __future__ import annotations

import json
import math
import pytest

from arden.agent import AgentLoop
from arden.config import Settings, load_config
from arden.llm.mock import MockLLM
from arden.llm.schemas import Intent, Step
from arden.memory.beliefs import Belief, BeliefStore
from arden.memory.short_term import ShortTermBuffer, reflect
from arden.surprise import (
    BaselineWorldModel,
    NullWorldModel,
    SURPRISE_KINDS,
    StateEmbedding,
    SurpriseConfig,
    SurpriseScorer,
    SurpriseSignal,
    VelocityWorldModel,
    WorldModel,
    compute_surprise_score,
)
from arden.world.objects import Position, Velocity, WorldObject
from arden.world.world import World


def _make_intent(
    *,
    goal: str = "push object_3 east",
    expect: str = "object_3 moves east",
    confidence: float = 0.8,
    target: str = "object_3",
) -> Intent:
    return Intent(
        goal=goal,
        steps=[Step(verb="push", target=target, params={"direction": "east"})],
        wake_if=[],
        expect=expect,
        thought="testing push",
        confidence=confidence,
    )


def _make_sphere(
    object_id: str = "object_3",
    x: float = 1.0,
    y: float = 1.0,
    vx: float = 0.0,
    vy: float = 0.0,
) -> WorldObject:
    return WorldObject(
        id=object_id,
        position=Position(x=x, y=y),
        shape="sphere",
        height=0.3,
        volume=0.01,
        weight=0.5,
        rigidity=0.8,
        velocity=Velocity(x=vx, y=vy),
    )


class TestSurpriseSignalAndFormula:
    """Verify SurpriseSignal structure and the score formula."""

    def test_surprise_signal_fields_and_unpacking(self) -> None:
        sig = SurpriseSignal(0.75, "expectation", {"object_id": "object_3", "error": 0.5})
        assert sig.score == pytest.approx(0.75)
        assert sig.kind == "expectation"
        assert sig.details["object_id"] == "object_3"
        assert sig.object_id == "object_3"

        score, kind, details = sig
        assert score == pytest.approx(0.75)
        assert kind == "expectation"
        assert details["error"] == pytest.approx(0.5)

    def test_invalid_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid surprise kind"):
            SurpriseSignal(0.5, "unknown_kind", {})  # type: ignore[arg-type]

    def test_all_four_kinds_supported(self) -> None:
        assert set(SURPRISE_KINDS) == {"physical", "expectation", "belief", "novelty"}
        for kind in SURPRISE_KINDS:
            sig = SurpriseSignal(0.2, kind, {})
            assert sig.kind == kind

    def test_exact_formula_computation(self) -> None:
        # score = error * confidence / usual_error(context) * habituation
        err = 0.6
        conf = 0.8
        usual = 0.2
        hab = 0.5
        expected = (err * conf / usual) * hab
        assert compute_surprise_score(err, conf, usual, hab) == pytest.approx(expected)

        scorer = SurpriseScorer()
        sig = scorer.score(
            kind="expectation",
            error=err,
            confidence=conf,
            context="ctx_formula",
            signature="sig_formula",
        )
        # Immediately after scoring, scorer.usual_error(context) matches the denominator used
        usual_after = scorer.usual_error("ctx_formula")
        hab_used = sig.details["habituation"]
        assert sig.score == pytest.approx(err * conf / usual_after * hab_used)


class TestWorldModel:
    """Verify WorldModel protocol, NullWorldModel, and VelocityWorldModel baseline."""

    def test_null_world_model_returns_zero_error(self) -> None:
        model = NullWorldModel()
        assert isinstance(model, WorldModel)

        s0 = _make_sphere("object_1", x=0.0, y=0.0, vx=1.0, vy=-1.0)
        s1 = _make_sphere("object_1", x=10.0, y=-5.0, vx=0.0, vy=0.0)
        pred = model.predict(s0, action="push")
        assert isinstance(pred, StateEmbedding)
        assert model.error(pred, s1) == 0.0

        scorer = SurpriseScorer(world_model=NullWorldModel())
        sig = scorer.score_physical(s0, s1)
        assert sig.kind == "physical"
        assert sig.score == 0.0
        assert sig.error == 0.0

    def test_velocity_world_model_extrapolates_last_velocity(self) -> None:
        model = VelocityWorldModel(dt=1.0)
        assert isinstance(model, WorldModel)
        assert BaselineWorldModel is VelocityWorldModel

        # Object at (2, 3) moving with velocity (0.5, -0.25) -> predicted at (2.5, 2.75)
        s0 = _make_sphere("object_1", x=2.0, y=3.0, vx=0.5, vy=-0.25)
        pred = model.predict(s0, action=None)
        assert pred[0] == pytest.approx(2.5)
        assert pred[1] == pytest.approx(2.75)

        # If actual state matches linear velocity extrapolation, error is 0.0
        s1_exact = _make_sphere("object_1", x=2.5, y=2.75, vx=0.5, vy=-0.25)
        assert model.error(pred, s1_exact) == pytest.approx(0.0)

        # If object unexpectedly jumps to (4.5, 2.75), Euclidean distance error is 2.0
        s1_surprising = _make_sphere("object_1", x=4.5, y=2.75, vx=0.5, vy=-0.25)
        assert model.error(pred, s1_surprising) == pytest.approx(2.0)

    def test_physical_surprise_scoring_from_world_model(self) -> None:
        scorer = SurpriseScorer(world_model=VelocityWorldModel(dt=1.0))
        s0 = _make_sphere("object_1", x=0.0, y=0.0, vx=1.0, vy=0.0)
        s1_expected = _make_sphere("object_1", x=1.0, y=0.0, vx=1.0, vy=0.0)
        s1_unexpected = _make_sphere("object_1", x=0.0, y=2.0, vx=0.0, vy=2.0)

        sig_ok = scorer.score_physical(s0, s1_expected)
        assert sig_ok.kind == "physical"
        assert sig_ok.score == pytest.approx(0.0)

        sig_surprise = scorer.score_physical(s0, s1_unexpected)
        assert sig_surprise.kind == "physical"
        assert sig_surprise.score > 0.0
        assert sig_surprise.error == pytest.approx(math.hypot(1.0 - 0.0, 0.0 - 2.0))


class TestExpectationSurprise:
    """Verify expectation surprise: cheap movement check first, optional LLM judge behind config flag, weighted by confidence."""

    def test_cheap_movement_check_matches_expected_movement(self) -> None:
        llm = MockLLM()
        scorer = SurpriseScorer(llm=llm, use_llm_judge=False)
        intent = _make_intent(expect="object_3 moves east", confidence=0.9)

        before = _make_sphere("object_3", x=1.0, y=1.0)
        after = _make_sphere("object_3", x=2.0, y=1.0)

        sig = scorer.score_expectation(intent, before_state=before, after_state=after)
        assert sig.kind == "expectation"
        assert sig.score == pytest.approx(0.0)
        assert sig.details["method"] == "cheap_structured"
        assert sig.details["used_llm_judge"] is False

    def test_cheap_movement_check_detects_lack_of_predicted_movement(self) -> None:
        scorer = SurpriseScorer(use_llm_judge=False)
        intent = _make_intent(expect="object_3 moves east", confidence=0.8)

        before = _make_sphere("object_3", x=1.0, y=1.0)
        after = _make_sphere("object_3", x=1.0, y=1.0)  # did not move

        sig = scorer.score_expectation(intent, before_state=before, after_state=after)
        assert sig.kind == "expectation"
        assert sig.error == pytest.approx(1.0)
        assert sig.score > 0.0
        assert sig.details["method"] == "cheap_structured"
        assert sig.details["used_llm_judge"] is False

    def test_cheap_movement_check_detects_wrong_direction(self) -> None:
        scorer = SurpriseScorer(use_llm_judge=False)
        intent = _make_intent(expect="object_3 moves east", confidence=0.9)

        before = _make_sphere("object_3", x=1.0, y=1.0)
        after = _make_sphere("object_3", x=0.0, y=1.0)  # moved west instead of east

        sig = scorer.score_expectation(intent, before_state=before, after_state=after)
        assert sig.kind == "expectation"
        assert sig.error > 0.5
        assert sig.score > 0.0
        assert sig.details["method"] == "cheap_structured"

    def test_expectation_surprise_weighted_by_intent_confidence(self) -> None:
        before = _make_sphere("object_3", x=1.0, y=1.0)
        after = _make_sphere("object_3", x=1.0, y=1.0)

        scorer_high = SurpriseScorer()
        intent_high = _make_intent(expect="object_3 moves", confidence=0.9)
        sig_high = scorer_high.score_expectation(intent_high, before_state=before, after_state=after)

        scorer_low = SurpriseScorer()
        intent_low = _make_intent(expect="object_3 moves", confidence=0.3)
        sig_low = scorer_low.score_expectation(intent_low, before_state=before, after_state=after)

        assert sig_high.score == pytest.approx(3.0 * sig_low.score)

    def test_optional_llm_judge_behind_config_flag(self) -> None:
        calls: list[str] = []

        class TrackingLLM:
            def complete_json(
                self,
                role: str,
                system: str,
                user: str,
                schema: dict,
                timeout_s: float,
            ) -> dict:
                calls.append(role)
                return {
                    "error": 0.8,
                    "surprised": True,
                    "reason": "Expected warm glow, observed cold silence.",
                }

        llm = TrackingLLM()
        intent = _make_intent(
            goal="inspect object_3",
            expect="object_3 emits a warm chime",
            confidence=0.75,
        )
        outcome = {"summary": "object_3 remained silent and cold"}

        # 1. With use_llm_judge=False (default), LLM is NOT called
        scorer_off = SurpriseScorer(llm=llm, use_llm_judge=False)
        sig_off = scorer_off.score_expectation(intent, outcome)
        assert calls == []
        assert sig_off.details["used_llm_judge"] is False

        # 2. With use_llm_judge=True, inconclusive structured movement falls back to LLM judge
        scorer_on = SurpriseScorer(llm=llm, use_llm_judge=True)
        sig_on = scorer_on.score_expectation(intent, outcome)
        assert calls == ["judge"]
        assert sig_on.details["used_llm_judge"] is True
        assert sig_on.details["method"] == "llm_judge"
        assert sig_on.error == pytest.approx(0.8)
        assert sig_on.confidence == pytest.approx(0.75)
        assert sig_on.score > 0.0

        # 3. Even with use_llm_judge=True, a clear structured movement comparison uses the cheap check first without calling LLM
        calls.clear()
        move_intent = _make_intent(expect="object_3 moves east", confidence=0.8)
        b = _make_sphere("object_3", x=0.0, y=0.0)
        a = _make_sphere("object_3", x=1.5, y=0.0)
        sig_cheap = scorer_on.score_expectation(move_intent, before_state=b, after_state=a)
        assert calls == []
        assert sig_cheap.details["used_llm_judge"] is False
        assert sig_cheap.score == pytest.approx(0.0)

    def test_mock_llm_supports_judge_role(self) -> None:
        mock_llm = MockLLM(
            script=[json.dumps({"error": 0.9, "surprised": True, "reason": "contradicted"})]
        )
        scorer = SurpriseScorer(llm=mock_llm, use_llm_judge=True)
        intent = _make_intent(expect="object_3 hums", confidence=0.6)
        sig = scorer.score_expectation(intent, {"summary": "object_3 was silent"})
        assert sig.kind == "expectation"
        assert sig.details["used_llm_judge"] is True
        assert sig.error == pytest.approx(0.9)


class TestBeliefSurprise:
    """Verify belief surprise when an observation contradicts a retrieved belief's tags or predicate, weighted by belief confidence."""

    def test_belief_predicate_contradiction_and_confidence_weighting(self) -> None:
        belief_high = Belief(
            id=1,
            statement="object_2 is stationary and immovable",
            tags=["object_2", "stationary", "immovable"],
            confidence=0.9,
            created_tick=1,
            last_updated_tick=1,
        )
        belief_low = Belief(
            id=2,
            statement="object_2 is stationary and immovable",
            tags=["object_2", "stationary", "immovable"],
            confidence=0.3,
            created_tick=1,
            last_updated_tick=1,
        )

        scorer_high = SurpriseScorer()
        sig_high = scorer_high.score_belief("object_2 moved east across the ground", [belief_high])
        assert sig_high.kind == "belief"
        assert sig_high.error == pytest.approx(1.0)
        assert sig_high.confidence == pytest.approx(0.9)
        assert sig_high.score > 0.0

        scorer_low = SurpriseScorer()
        sig_low = scorer_low.score_belief("object_2 moved east across the ground", [belief_low])
        assert sig_low.kind == "belief"
        assert sig_low.error == pytest.approx(1.0)
        assert sig_low.confidence == pytest.approx(0.3)
        assert sig_high.score == pytest.approx(3.0 * sig_low.score)

    def test_belief_tag_contradiction(self) -> None:
        belief = Belief(
            id=3,
            statement="object_5 stays dry under cover",
            tags=["object_5", "dry", "not_wet"],
            confidence=0.8,
            created_tick=1,
            last_updated_tick=1,
        )
        scorer = SurpriseScorer()
        sig = scorer.score_belief(
            {"object_id": "object_5", "wet": True, "summary": "object_5 is wet"},
            [belief],
        )
        assert sig.kind == "belief"
        assert sig.score > 0.0
        assert sig.confidence == pytest.approx(0.8)

    def test_consistent_observation_produces_zero_belief_surprise(self) -> None:
        belief = Belief(
            id=4,
            statement="object_3 rolls when pushed",
            tags=["object_3", "rolls", "movable"],
            confidence=0.85,
            created_tick=1,
            last_updated_tick=1,
        )
        scorer = SurpriseScorer()
        sig = scorer.score_belief("object_3 rolled forward when pushed", [belief])
        assert sig.kind == "belief"
        assert sig.score == pytest.approx(0.0)
        assert sig.error == pytest.approx(0.0)


class TestNoveltySurprise:
    """Verify novelty is reported as its own kind and never folded into the error score."""

    def test_novelty_is_reported_as_own_kind_and_never_folded_into_error_score(self) -> None:
        scorer = SurpriseScorer()
        initial_usual = scorer.usual_error("object_99")

        novel_obj = {
            "id": "object_99",
            "kind": "crystal_prism",
            "properties": {"color": "cyan", "shape": "pyramid"},
            "affordances": ["push", "refract"],
        }
        sig = scorer.score_novelty(novel_obj)
        assert sig.kind == "novelty"
        assert sig.score > 0.0
        # Novelty must never be folded into error or error_score, nor alter usual_error
        assert sig.error == 0.0
        assert sig.error_score == 0.0
        assert sig.details["error"] == 0.0
        assert sig.details["error_score"] == 0.0
        assert scorer.usual_error("object_99") == pytest.approx(initial_usual)

        # Evaluating all signals keeps novelty in its own SurpriseSignal(kind="novelty")
        # and does not add novelty into physical/expectation/belief error scores.
        s0 = {"objects": {"object_1": _make_sphere("object_1", x=0.0, y=0.0)}}
        s1 = {
            "objects": {
                "object_1": _make_sphere("object_1", x=0.0, y=0.0),
                "object_99": _make_sphere("object_99", x=3.0, y=4.0),
            }
        }
        phys_sig = scorer.score_physical(s0, s1)
        assert phys_sig.kind == "physical"
        assert phys_sig.score == pytest.approx(0.0)
        assert phys_sig.error == pytest.approx(0.0)


class TestNormalizationAndHabituation:
    """Verify normalization (noisy familiar object stops scoring high) and habituation (5th identical surprise < 1st)."""

    def test_noisy_familiar_object_stops_scoring_high(self) -> None:
        # Use habituation_decay=1.0 to isolate pure normalization via usual_error(context)
        scorer = SurpriseScorer(
            initial_usual_error=0.1,
            min_usual_error=0.05,
            error_ema_alpha=0.35,
            habituation_decay=1.0,
        )

        # A noisy object produces ~1.0 prediction error on every tick with distinct signatures
        scores: list[float] = []
        for i in range(15):
            sig = scorer.score(
                kind="physical",
                error=1.0,
                confidence=1.0,
                context="noisy_leaf_object_4",
                signature=f"noisy_tick_{i}",
            )
            scores.append(sig.score)

        # First surprise on the unfamiliar noisy object is high (> 2.0)
        assert scores[0] > 2.0
        # As usual_error("noisy_leaf_object_4") adapts to ~1.0, the score drops toward ~1.0
        assert scores[-1] < scores[0] * 0.45
        assert scores[-1] == pytest.approx(1.0, rel=0.05)
        assert scorer.usual_error("noisy_leaf_object_4") == pytest.approx(1.0, rel=0.05)

    def test_habituation_fifth_identical_surprise_scores_lower_than_first(self) -> None:
        scorer = SurpriseScorer(habituation_decay=0.65)
        intent = _make_intent(expect="object_3 moves east", confidence=0.85)
        before = _make_sphere("object_3", x=1.0, y=1.0)
        after = _make_sphere("object_3", x=1.0, y=1.0)  # identical failure each time

        signals = [
            scorer.score_expectation(intent, before_state=before, after_state=after)
            for _ in range(5)
        ]
        first_score = signals[0].score
        fifth_score = signals[4].score

        assert first_score > 0.0
        assert fifth_score < first_score
        # Habituation factor itself decays monotonically on each repetition
        assert signals[0].habituation == pytest.approx(1.0)
        assert signals[4].habituation == pytest.approx(0.65**4)
        for idx in range(1, 5):
            assert signals[idx].score < signals[idx - 1].score


class TestUnexplainedSurprisePull:
    """Verify unexplained surprises keep a persistent pull on the related object_id after reflection."""

    def test_unexplained_surprise_keeps_pull_and_clears_when_belief_explains_it(
        self, tmp_path
    ) -> None:
        world = World(
            seed=42,
            objects=[
                {
                    "id": "object_3",
                    "position": {"x": 1.0, "y": 1.0},
                    "level": "ground",
                    "shape": "sphere",
                    "height": 0.3,
                    "volume": 0.01,
                    "weight": 0.5,
                    "rigidity": 0.8,
                }
            ],
        )
        bstore = BeliefStore(db_path=tmp_path / "mem.db", world=world)
        scorer = SurpriseScorer(belief_store=bstore, world=world, unexplained_pull=0.2)

        # Agent expects object_3 to move when pushed, but object_3 stays stationary
        intent = _make_intent(
            goal="push object_3",
            expect="object_3 moves east",
            confidence=0.9,
            target="object_3",
        )
        before = _make_sphere("object_3", x=1.0, y=1.0)
        after = _make_sphere("object_3", x=1.0, y=1.0)

        sig = scorer.score_expectation(
            intent,
            {"summary": "object_3 stayed stationary and immovable"},
            before_state=before,
            after_state=after,
            world=world,
        )
        assert sig.score > 0.0
        assert sig.pull == pytest.approx(0.2)
        assert scorer.get_pull("object_3") == pytest.approx(0.2)
        assert world.get_pull("object_3") == pytest.approx(0.2)
        assert world.objects["object_3"].pull == pytest.approx(0.2)

        # Reflection 1: LLM forms an unrelated belief about object_1 (does NOT explain object_3)
        unrelated_reflection = {
            "episodes": [
                {
                    "summary": "Pushed object_3 but it did not move.",
                    "tags": ["object_3", "push"],
                    "salience": 0.7,
                    "surprise_kind": "expectation",
                    "made_by_agent": [],
                }
            ],
            "beliefs": [
                {
                    "action": "propose",
                    "statement": "object_1 is a tall green cylinder",
                    "tags": ["object_1", "cylinder"],
                    "confidence": 0.7,
                }
            ],
            "self_model": "I am exploring objects in the clearing.",
        }
        llm_1 = MockLLM(script=[json.dumps(unrelated_reflection)])
        buf = ShortTermBuffer(world=world, belief_store=bstore)
        buf.record(
            tick=1,
            observation="object_3 stayed stationary",
            action="push object_3",
            surprise=sig.score,
            surprise_kind=sig.kind,
        )
        reflect(buf, llm_1, belief_store=bstore, world=world, surprise_scorer=scorer)

        # Because no belief predicts/explains object_3's surprise after reflection, pull persists!
        assert scorer.get_pull("object_3") == pytest.approx(0.2)
        assert world.get_pull("object_3") == pytest.approx(0.2)
        assert world.objects["object_3"].pull == pytest.approx(0.2)

        # Reflection 2: LLM proposes a belief that explains object_3 staying stationary
        explaining_reflection = {
            "episodes": [
                {
                    "summary": "object_3 is fixed to the ground and stays stationary.",
                    "tags": ["object_3", "stationary"],
                    "salience": 0.8,
                    "surprise_kind": "expectation",
                    "made_by_agent": [],
                }
            ],
            "beliefs": [
                {
                    "action": "propose",
                    "statement": "object_3 is fixed and stays stationary when pushed",
                    "tags": ["object_3", "stationary", "immovable"],
                    "confidence": 0.8,
                }
            ],
            "self_model": "I learned object_3 is stationary.",
        }
        llm_2 = MockLLM(script=[json.dumps(explaining_reflection)])
        buf.record(tick=2, observation="object_3 is stationary", action="push object_3")
        reflect(buf, llm_2, belief_store=bstore, world=world, surprise_scorer=scorer)

        # Now that a belief predicts object_3's behavior, the unexplained pull is cleared
        assert scorer.get_pull("object_3") == pytest.approx(0.0)
        assert world.get_pull("object_3") == pytest.approx(0.0)
        assert world.objects["object_3"].pull == pytest.approx(0.0)


class TestSurpriseConfigIntegration:
    """Verify SurpriseSettings in Settings and arden.example.toml."""

    def test_settings_and_example_toml_include_surprise(self) -> None:
        cfg = Settings()
        assert isinstance(cfg.surprise, SurpriseConfig)
        assert cfg.surprise.use_llm_judge is False
        assert load_config() == cfg

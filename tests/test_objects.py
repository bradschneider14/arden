"""World object schema: properties, hidden rules, JSON, and the factory."""

from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from arden.world.objects import (
    PERCEPTIBLE_STATE_KEYS,
    BehaviorRule,
    ObjectFactory,
    WorldObject,
)

_PHYSICAL = {
    "position": {"x": 1.5, "y": -2.0},
    "shape": "sphere",
    "height": 0.2,
    "volume": 0.004,
    "weight": 0.3,
    "rigidity": 0.4,
    "binding": False,
    "hollow": False,
    "fragile": False,
    "color": "red",
    "fixed": False,
    "state": {"wet": False},
}


def _rules() -> list[dict[str, object]]:
    return [
        {"verb": "push", "scale": 4},
        {"verb": "strike", "response": "bounce"},
        {"verb": "strike", "response": "break", "above_force": 0.7},
    ]


def test_factory_builds_from_property_dicts_and_assigns_ids() -> None:
    factory = ObjectFactory()
    first = factory.create(dict(_PHYSICAL))
    named_id = factory.create({**_PHYSICAL, "id": "object_7", "shape": "flat", "fixed": True})
    after = factory.create({**_PHYSICAL, "shape": "long", "level": "on_top_of:object_7"})

    assert first.id == "object_1"
    assert named_id.id == "object_7"
    assert after.id == "object_8"
    assert after.level == "on_top_of:object_7"
    assert all(isinstance(rule, BehaviorRule) for rule in first.behavior)


def test_behavior_rules_are_data_keyed_by_verb() -> None:
    obj = ObjectFactory().create({**_PHYSICAL, "behavior": _rules()})
    assert [type(rule) for rule in obj.behavior] == [BehaviorRule, BehaviorRule, BehaviorRule]
    strike = [rule for rule in obj.behavior if rule.verb == "strike"]
    assert len(strike) == 2
    assert strike[0].model_dump()["response"] == "bounce"
    assert strike[1].model_dump()["above_force"] == 0.7
    assert obj.behavior[0].model_dump()["scale"] == 4


def test_json_round_trip_keeps_hidden_rules() -> None:
    original = ObjectFactory().create(
        {
            **_PHYSICAL,
            "level": "inside:object_4",
            "shape": "cylinder",
            "hollow": True,
            "binding": True,
            "fragile": True,
            "behavior": _rules(),
            "state": {"count": 2, "tags": ["a", "b"]},
        }
    )
    restored = WorldObject.from_json(original.to_json())
    assert restored == original
    assert json.loads(original.to_json())["behavior"][1]["response"] == "bounce"


def test_perceptible_omits_hidden_rules_and_human_labels() -> None:
    hidden_label = "human_label_ball"
    obj = ObjectFactory().create(
        {
            **_PHYSICAL,
            "behavior": [
                {"verb": "push", "scale": 4, "note": hidden_label},
                {"verb": "strike", "response": "bounce"},
            ],
        }
    )
    view = obj.perceptible()
    encoded = json.dumps(view)

    assert "behavior" not in view
    assert hidden_label not in encoded
    assert "bounce" not in encoded
    assert "scale" not in encoded
    assert "name" not in view
    assert "label" not in view
    assert view["id"] == "object_1"
    assert "Object 1" not in encoded
    assert view["shape"] == "sphere"
    assert view["color"] == "red"
    assert view["position"] == {"x": 1.5, "y": -2.0}
    assert obj.behavior[0].model_dump()["note"] == hidden_label


def test_perceptible_omits_unset_color() -> None:
    props = dict(_PHYSICAL)
    del props["color"]
    view = ObjectFactory().create(props).perceptible()
    assert "color" not in view
    assert "behavior" not in view


def test_perceptible_keeps_explicit_nulls_in_state() -> None:
    view = ObjectFactory().create({**_PHYSICAL, "state": {"wet": None}}).perceptible()
    assert view["state"] == {"wet": None}


def test_perceptible_state_allowlist_filters_unallowed_keys() -> None:
    assert "wet" in PERCEPTIBLE_STATE_KEYS
    assert "radius" not in PERCEPTIBLE_STATE_KEYS
    assert "floating" not in PERCEPTIBLE_STATE_KEYS

    obj = ObjectFactory().create(
        {
            **_PHYSICAL,
            "state": {
                "wet": True,
                "radius": 0.45,
                "floating": True,
                "label": "ball",
                "note": "secret",
            },
        }
    )
    # Internal state retains all keys for physics and serialization
    assert obj.state["radius"] == 0.45
    assert obj.state["floating"] is True
    assert obj.radius == 0.45

    view = obj.perceptible()
    encoded = json.dumps(view)
    assert view["state"] == {"wet": True}
    assert "radius" not in view["state"]
    assert "floating" not in view["state"]
    assert "label" not in view["state"]
    assert "note" not in view["state"]
    assert "floating" not in encoded
    assert "secret" not in encoded


def test_schema_rejects_a_human_label() -> None:
    with pytest.raises(ValidationError):
        ObjectFactory().create({**_PHYSICAL, "name": "ball"})
    with pytest.raises(ValidationError):
        ObjectFactory().create({**_PHYSICAL, "id": "ball"})


def test_validation_rejects_out_of_range_values() -> None:
    factory = ObjectFactory()
    with pytest.raises(ValidationError):
        factory.create({**_PHYSICAL, "weight": -0.01})
    with pytest.raises(ValidationError):
        factory.create({**_PHYSICAL, "rigidity": 1.01})
    with pytest.raises(ValidationError):
        factory.create({**_PHYSICAL, "rigidity": -0.01})
    with pytest.raises(ValidationError):
        factory.create({**_PHYSICAL, "height": -1})
    with pytest.raises(ValidationError):
        factory.create({**_PHYSICAL, "volume": -1})
    with pytest.raises(ValidationError):
        factory.create({**_PHYSICAL, "shape": "ball"})
    with pytest.raises(ValidationError):
        factory.create({**_PHYSICAL, "level": "on_top_of:ball"})
    with pytest.raises(ValidationError):
        factory.create({**_PHYSICAL, "weight": math.inf})
    with pytest.raises(ValidationError):
        factory.create({**_PHYSICAL, "color": "  "})

    edges = factory.create({**_PHYSICAL, "weight": 0, "rigidity": 0, "height": 0, "volume": 0})
    top = factory.create({**_PHYSICAL, "rigidity": 1, "level": "raised"})
    assert edges.weight == 0
    assert edges.rigidity == 0
    assert top.rigidity == 1
    assert top.level == "raised"


def test_factory_rejects_duplicate_ids() -> None:
    factory = ObjectFactory()
    factory.create({**_PHYSICAL, "id": "object_2"})
    with pytest.raises(ValueError, match="duplicate object id"):
        factory.create({**_PHYSICAL, "id": "object_2"})
    nxt = factory.create(dict(_PHYSICAL))
    assert nxt.id == "object_3"


def test_direct_constructor_validates_level_and_inside() -> None:
    obj = WorldObject.model_validate(
        {**_PHYSICAL, "id": "object_0", "level": "inside:object_9", "shape": "cube"}
    )
    assert obj.id == "object_0"
    assert obj.level == "inside:object_9"
    assert obj.to_json()
    assert WorldObject.from_json(obj.to_json()).level == "inside:object_9"

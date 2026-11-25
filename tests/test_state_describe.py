"""State.describe 坐标系装配与 Frame 一致性。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from common.exception import CoordinateFrameException
from envs.games.kaz.kaz_state import KazEntity, KazState
from envs.games.kaz.snapshot import _sprite_position
from envs.state.base import CoordinateSystem, Entity, State, StateDescribe
from envs.state.spatial import CGCS2000Frame, LocalFrame, WGS84Frame


def _entity(name: str, position) -> Entity:
    return Entity(name=name, type="aircraft", position=position, speed=0.0)


def test_missing_describe_defaults_to_local_frame() -> None:
    state = State.from_dict({
        "entities": [{
            "name": "a",
            "type": "unit",
            "position": {"x": 3.0, "y": 4.0, "z": 0.0},
            "speed": 1.0,
        }],
    })
    assert state.describe.coordinate_system is CoordinateSystem.LOCAL
    assert type(state.entities[0].position) is LocalFrame
    assert state.entities[0].position.x == pytest.approx(3.0)

    with pytest.raises((ValueError, CoordinateFrameException)):
        State.from_dict({
            "entities": [{
                "name": "a",
                "type": "unit",
                "position": {"lon": 118.0, "lat": 32.0, "alt": 0.0},
                "speed": 1.0,
            }],
        })


def test_from_dict_uses_describe_not_field_names() -> None:
    wgs = State.from_dict({
        "describe": {"coordinate_system": "wgs84"},
        "entities": [{
            "name": "a",
            "type": "unit",
            "position": {"lon": 118.8, "lat": 32.1, "alt": 12.0},
            "speed": 0.0,
        }],
    })
    assert type(wgs.entities[0].position) is WGS84Frame
    assert wgs.entities[0].position.longitude == pytest.approx(118.8)

    cgcs = State.from_dict({
        "describe": {"coordinate_system": "cgcs2000"},
        "entities": [{
            "name": "a",
            "type": "unit",
            "position": {"longitude": 118.8, "latitude": 32.1, "ellipsoidal_height": 12.0},
            "speed": 0.0,
        }],
    })
    assert type(cgcs.entities[0].position) is CGCS2000Frame

    local = State.from_dict({
        "describe": {"coordinate_system": "local"},
        "entities": [{
            "name": "a",
            "type": "unit",
            "position": {"x": 1.0, "y": 2.0, "z": 3.0},
            "speed": 0.0,
        }],
    })
    assert type(local.entities[0].position) is LocalFrame


def test_state_roundtrip_keeps_describe_and_position_type() -> None:
    state = State(
        entities=[_entity("a", LocalFrame(x=1.0, y=2.0, z=3.0))],
        describe=StateDescribe(),
    )
    payload = state.to_dict()
    assert payload["describe"]["coordinate_system"] == "local"
    assert "lon" not in payload["entities"][0]["position"]
    restored = State.from_dict(payload)
    assert restored.describe.coordinate_system is CoordinateSystem.LOCAL
    assert type(restored.entities[0].position) is LocalFrame
    assert restored.entities[0].position.z == pytest.approx(3.0)


def test_platforms_migrate_and_follow_describe() -> None:
    state = State.from_dict({
        "describe": {"coordinate_system": "wgs84"},
        "platforms": [{
            "name": "p1",
            "type": "radar",
            "position": {"longitude": 120.0, "latitude": 30.0, "ellipsoidal_height": 0.0},
            "speed": 0.0,
        }],
    })
    assert len(state.entities) == 1
    assert type(state.entities[0].position) is WGS84Frame


def test_direct_construct_rejects_mixed_frames() -> None:
    with pytest.raises(CoordinateFrameException):
        State(
            entities=[
                _entity("a", WGS84Frame(longitude=0.0, latitude=0.0)),
            ],
        )
    with pytest.raises(CoordinateFrameException):
        State(
            entities=[
                _entity("a", LocalFrame(x=0.0, y=0.0)),
                _entity("b", WGS84Frame(longitude=1.0, latitude=1.0)),
            ],
            describe=StateDescribe(),
        )
    State(
        entities=[_entity("a", WGS84Frame(longitude=1.0, latitude=2.0))],
        describe=StateDescribe(coordinate_system=CoordinateSystem.WGS84),
    )




def test_kaz_state_uses_local_frame() -> None:
    entity = KazEntity(
        name="archer_0",
        type="archer",
        position=LocalFrame(x=10.0, y=20.0, z=0.0),
        speed=0.0,
        side="defender",
        base_type="player",
    )
    state = KazState(entities=[entity])
    assert state.describe.coordinate_system is CoordinateSystem.LOCAL
    sprite = SimpleNamespace(rect=SimpleNamespace(x=7.0, y=9.0))
    position = _sprite_position(sprite)
    assert type(position) is LocalFrame
    assert position.x == pytest.approx(7.0)
    assert position.y == pytest.approx(9.0)


def test_empty_scene_states_keep_default_local() -> None:
    scene = State(entities=[])
    assert scene.describe.coordinate_system is CoordinateSystem.LOCAL
    assert scene.entities == []

"""The framework does not choose a subject's environment or recorded state model."""
import pytest

from envs import get_env_class
from envs.base import BaseEnvironment
from envs.sense import get_env_class as get_sense_class
from portal.plugins.registry import PROJECT_PLUGIN_REGISTRY


def test_migrated_subject_is_not_a_builtin_environment():
    assert get_env_class("7v3") is None
    assert get_sense_class("7v3") is None
    assert "7v3" not in PROJECT_PLUGIN_REGISTRY.environment


@pytest.mark.parametrize("payload", [{"entities": []}, {"platforms": []}, {"position": [0, 0]}])
def test_recorded_state_requires_an_environment_owned_converter(payload):
    with pytest.raises(NotImplementedError, match="_state_from_record"):
        BaseEnvironment._state_from_record(None, payload)

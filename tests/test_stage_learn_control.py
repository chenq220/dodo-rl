import pytest

from config import EnvConfig, ModelManagerConfig, MonteCarloMethodConfig, ReplayBufferConfig, StageTaskConfig
from roles.stage_learn_control import StageLearnRole


def test_stage_learn_role_requires_stages():
    with pytest.raises(AssertionError, match="requires StageTaskConfig.stages"):
        StageLearnRole(
            role_id="learn",
            stage_task_config=StageTaskConfig(stages=[]),
            env_config=EnvConfig(use_sim=False),
            sim_config=None,
            replay_buffer_config=ReplayBufferConfig(),
            model_manager_config=ModelManagerConfig(),
            sim_manager_config=None,
            monte_carlo_method_config=MonteCarloMethodConfig(),
            task_manager_address="127.0.0.1:1",
            replay_buffer_address="127.0.0.1:2",
            model_manager_address="127.0.0.1:3",
            sim_manager_address="127.0.0.1:4",
            monte_carlo_method_address="127.0.0.1:5",
        )

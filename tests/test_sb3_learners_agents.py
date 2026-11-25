from types import SimpleNamespace

import numpy as np

from utils.transition_hash import transition_batch_hash
from algos.map import AlogsMap
from config import AgentConfig, LearnerConfig


def test_sb3_map_binds_config_types():
    for algo in (AlogsMap.a2c, AlogsMap.ppo, AlogsMap.dqn):
        learner_config = LearnerConfig(name="solo", algo_type=algo.learner)
        agent_config = AgentConfig(name="solo", agent_type=algo.agent.sb3)
        assert isinstance(learner_config.algo_config, algo.learner.config_type)
        assert agent_config.agent_type == algo.agent.sb3


def test_sb3_agent_and_learner_plugins_registered():
    from algos.modules.agents.sb3_agent import SB3Agent
    from algos.training.sb3_a2c_learner import SB3A2CLearner
    from algos.training.sb3_dqn_learner import SB3DQNLearner
    from algos.training.sb3_ppo_learner import SB3PPOLearner

    assert SB3Agent.__name__ == "SB3Agent"
    assert SB3A2CLearner.__name__ == "SB3A2CLearner"
    assert SB3PPOLearner.__name__ == "SB3PPOLearner"
    assert SB3DQNLearner.__name__ == "SB3DQNLearner"

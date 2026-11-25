from __future__ import annotations

from types import SimpleNamespace

import pytest

from algos.map import AlogsMap
from algos.training.coma_learner import COMALearner
from algos.training.mappo_learner import MAPPOActorCriticLearner
from algos.training.qmix_learner import QMixerLearner
from algos.training.sb3_a2c_learner import SB3A2CLearner
from algos.training.sb3_dqn_learner import SB3DQNLearner
from algos.training.sb3_ppo_learner import SB3PPOLearner
from config import LearnerConfig


@pytest.mark.parametrize('name,node,learner_class,agent_types', [
    ('coma', AlogsMap.coma, COMALearner, ('policy_rnn',)),
    ('qmix', AlogsMap.qmix, QMixerLearner, ('rnn', 'rnn_ns', 'q_rnn')),
    ('mappo', AlogsMap.mappo, MAPPOActorCriticLearner, ('policy_rnn',)),
    ('ppo', AlogsMap.ppo, SB3PPOLearner, ('sb3',)),
    ('a2c', AlogsMap.a2c, SB3A2CLearner, ('sb3',)),
    ('dqn', AlogsMap.dqn, SB3DQNLearner, ('sb3',)),
])
def test_algorithm_agent_compatibility(name, node, learner_class, agent_types):
    assert AlogsMap.compatible_agent_types(node) == agent_types
    config = LearnerConfig(name='compat-test', algo_type=node.learner)
    assert config.algo_type == name
    assert isinstance(config.algo_config, node.learner.config_type)
    with pytest.raises((AssertionError, ValueError), match='supports agent types|requires a policy_rnn|supports only'):
        learner_class(name, config, SimpleNamespace(agent_type='unsupported'), None, 'cpu')

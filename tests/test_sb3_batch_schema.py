"""SB3 Learners consume schema batches without changing replay deduplication."""

import pytest

from algos.learners import Learner
from common.creator import create_agent, create_learner
from common.plugins import PluginsConfig, activate_plugins
from config import AgentConfig, LearnerConfig
from envs.action.model import ActionParameterDefined, StepActionDefined
from portal.plugins.registry import project_plugin_registry
from roles.common.config_alignment import align_agent_config_with_learner_config
from algos.learners.batch_schema import SINGLE_AGENT_SCHEMA
from test_a2c import _build_episode


@pytest.mark.parametrize('algo', ['ppo', 'a2c', 'dqn'])
def test_sb3_schema_update_preserves_duplicate_semantics(algo):
    activate_plugins(PluginsConfig(), default_plugins=project_plugin_registry())
    learner_config = LearnerConfig(name='solo', algo_type=algo, device='cpu')
    cfg = learner_config.algo_config
    if algo == 'ppo':
        cfg.n_steps, cfg.n_epochs, cfg.minibatch_size = 4, 1, 4
    if algo == 'dqn':
        cfg.buffer_size, cfg.learning_starts, cfg.train_batch_size = 32, 0, 2
    agent_config = AgentConfig(name='solo', agent_type='sb3', agent_names=['solo-0'],
                               obs_dim=3, device='cpu', define_action=[
        StepActionDefined(action_key='choose',
                          parameters=[ActionParameterDefined(key='mode', values=[0, 1])],
                          add_target=False),
    ])
    align_agent_config_with_learner_config(agent_config, learner_config)
    agent = create_agent(agent_type='sb3', agent_config=agent_config, agent_net_source='blanco',
                         agent_list=agent_config.agent_names, device='cpu')
    learner = create_learner(learner_config=learner_config, agent=agent, env=None,
                             device='cpu', net_source='blanco')
    assert learner.batch_schema is SINGLE_AGENT_SCHEMA
    assert learner.batch_schema is not Learner.batch_schema
    episodes = [_build_episode(1, terminated_at_end=True), _build_episode(2, terminated_at_end=False)]
    first = learner.learn_update(episodes)
    second = learner.learn_update(episodes)
    if algo == 'dqn':
        assert first['inserted'] == 8 and first['duplicate_skipped'] == 0
        assert second['inserted'] == 0 and second['duplicate_skipped'] == 8
        assert second['replay_size'] == 8
    else:
        assert first['update_cnt'] == 1
        assert second['skip_update'] and second['skip_reason'] == 'same_sample_hash'

from __future__ import annotations

import copy
import numpy as np
import pytest
import torch
from torch import nn
from algos.modules.agents.policy_agent import PolicyRNNAgent
from algos.builtin_plugins import BUILTIN_AGENT_PLUGINS
from algos.modules.agents.q_agent import QAgent
from algos.training.mappo_learner import MAPPOActorCriticLearner
from algos.learners import Learner
from algos.map import AlogsMap
from config import AgentConfig, LearnerConfig, MappoRnnAgentConfig, PolicyRnnAgentConfig, QAgentConfig
from test_action_spaces import spaces


def q_agent(num_layers=1):
    config = AgentConfig(name='q', agent_type='q_rnn', agent_names=['a', 'b'],
                         obs_dim=2, hidden_dim=8, num_layers=num_layers,
                         obs_last_action=True, action_space_spec=spaces()[0].to_dict())
    return QAgent('q_rnn', config, 'blanco', 'cpu')


def observations():
    return {'a': np.array([1., 0.], np.float32), 'b': np.array([0., 1.], np.float32)}


class FixedQ(nn.Module):
    def forward(self, inputs, hidden):
        return torch.tensor([-4., 9., 2.]).expand(len(inputs), -1), hidden + 1


def test_fixed_roles_and_legacy_policy_alias():
    assert AgentConfig(agent_type='q_rnn', agent_output_type='pi_logits').agent_output_type == 'q'
    assert AgentConfig(agent_type='policy_rnn', agent_output_type='q').agent_output_type == 'pi_logits'
    assert AgentConfig(agent_type='mappo_rnn').agent_output_type == 'pi_logits'
    assert isinstance(AgentConfig(agent_type='q_rnn').algo_config, QAgentConfig)
    assert isinstance(AgentConfig(agent_type=AlogsMap.mappo.agent.rnn).algo_config, PolicyRnnAgentConfig)
    assert MappoRnnAgentConfig is PolicyRnnAgentConfig
    assert BUILTIN_AGENT_PLUGINS["mappo_rnn"] == BUILTIN_AGENT_PLUGINS["policy_rnn"]
    assert AlogsMap.compatible_agent_types(AlogsMap.mappo) == ('policy_rnn',)
    assert AlogsMap.compatible_agent_types(AlogsMap.qmix) == ('rnn', 'rnn_ns', 'q_rnn')


def test_learner_cannot_change_a_fixed_agent_role():
    agent = q_agent()
    with pytest.raises(ValueError, match='fixed output'):
        Learner._sync_paired_agent_config(LearnerConfig(algo_type='mappo'), agent)
    assert agent.config.agent_output_type == 'q'
    with pytest.raises(AssertionError, match='supports agent types'):
        MAPPOActorCriticLearner('mappo', LearnerConfig(algo_type='mappo'), agent, None, 'cpu')


def test_q_values_are_not_probabilities_and_epsilon_only_uses_valid_actions():
    agent = q_agent()
    agent.agent_net = FixedQ()
    hidden = agent.get_init_hidden_state()
    values, next_hidden = agent.get_q_values(observations(), ['a', 'b'], hidden)
    torch.testing.assert_close(values['a'], torch.tensor([-4., 9., 2.]))
    masks = {'a': [1, 0, 1], 'b': [0, 1, 0]}
    for evaluate, deterministic in ((False, False), (True, False), (False, True)):
        actions, _ = agent.get_actions(observations(), ['a', 'b'], hidden, masks,
                                       epsilon=0, evaluate=evaluate, deterministic=deterministic)
        assert actions['a'].action_idx == 2 and actions['b'].action_idx == 1
    found = set()
    np.random.seed(12)
    for _ in range(100):
        actions, _ = agent.get_actions(observations(), ['a', 'b'], hidden, masks, epsilon=1)
        found.add(actions['a'].action_idx)
        assert actions['b'].action_idx == 1
    assert found == {0, 2}
    actions, _ = agent.get_actions(observations(), ['a', 'b'], hidden, masks, epsilon=1, evaluate=True)
    assert actions['a'].action_idx == 2
    assert all(torch.count_nonzero(value) == 0 for value in hidden.values())
    assert all(torch.all(value == 1) for value in next_hidden.values())
    with pytest.raises(ValueError, match='no available action'):
        agent.get_actions(observations(), ['a'], hidden, {'a': [0, 0, 0]})
    with pytest.raises(ValueError, match='epsilon'):
        agent.get_actions(observations(), ['a'], hidden, epsilon=1.1)


@pytest.mark.parametrize('num_layers', [1, 2])
def test_q_recurrent_states_subsets_and_checkpoint_reload(num_layers, tmp_path):
    torch.manual_seed(10)
    agent = q_agent(num_layers)
    hidden = agent.get_init_hidden_state()
    first, next_hidden = agent.get_q_values(observations(), ['a', 'b'], hidden)
    again, _ = agent.get_q_values(observations(), ['a'], hidden)
    torch.testing.assert_close(first['a'], again['a'])
    assert next_hidden['a'].shape == (num_layers, 8)
    assert not torch.equal(next_hidden['a'], next_hidden['b'])
    assert torch.count_nonzero(hidden['a']) == 0
    path = tmp_path/'q.pth'
    torch.save(agent.agent_net.state_dict(), path)
    restored = QAgent('q_rnn', copy.deepcopy(agent.config), 'file', 'cpu', model_file_path=str(path))
    values, h = restored.get_q_values(observations(), ['a', 'b'], hidden)
    for name in first:
        torch.testing.assert_close(values[name], first[name])
        torch.testing.assert_close(h[name], next_hidden[name])
    replacement = copy.deepcopy(restored.agent_net)
    with torch.no_grad():
        replacement.fc2.bias.add_(1)
    restored.reset_net(replacement)
    updated, _ = restored.get_q_values(observations(), ['a', 'b'], hidden)
    torch.testing.assert_close(updated['a'], first['a'] + 1)


@pytest.mark.parametrize('space', spaces()[1:], ids=lambda space: space.kind)
def test_q_agent_rejects_unimplemented_action_layouts(space):
    cfg = AgentConfig(agent_type='q_rnn', agent_names=['a'], obs_dim=2,
                      action_space_spec=space.to_dict())
    with pytest.raises(ValueError, match='supported'):
        QAgent('q_rnn', cfg, 'blanco', 'cpu')


@pytest.mark.parametrize('num_layers', [1, 2])
def test_qmix_trains_q_rnn_from_a_replay_window_with_burn_in(num_layers):
    from types import SimpleNamespace
    from algos.training.qmix_learner import QMixerLearner
    from config import ReplayBufferConfig
    from envs.sense.action_space_examples import ActionSpaceCounterParallelEnvironment
    from sample.replay_buffer import ReplayBuffer
    from scheduler.model import ActionParam, AgentCompose, EpisodeInfo
    from scheduler.rollout_sample import RolloutSampleWorker
    from utils.module_utils import inject_env_dims

    agent = q_agent(num_layers)
    space = spaces()[0]
    env = ActionSpaceCounterParallelEnvironment({'a': space, 'b': space}, max_steps=4)
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={'q': 8}, batch_size=1))
    try:
        inject_env_dims(env, [agent])
        config = LearnerConfig(name='q', algo_type='qmix', standardise_rewards=False)
        learner = QMixerLearner('qmix', config, agent, env, 'cpu')
        worker = RolloutSampleWorker(env, SimpleNamespace(accept_sample_task_result=lambda *a, **kw: True),
                                    replay, AgentCompose(step_agent=[agent]), [agent])
        info = EpisodeInfo(episode_num=1, task_id='q-rnn', epsilon=0, deterministic=False, temperature=1,
                           actions_param={'q': ActionParam(epsilon=0, deterministic=False, temperature=1)},
                           seed=3, start_datetime=None, end_datetime=None, metadata={})
        episode = worker.rolling_parallel_episode(info, step_limit=3)['q'][0]
        replay.push({'q': [episode]}, notify_learning=False)
        window = replay._materialize_episode(replay.buffer['q'][0], 1, 3, sliced=True)
        assert window.burn_in.seq_len == 1
        before = copy.deepcopy(learner.agent_net.state_dict())
        metrics = learner.learn_update([window])
        assert np.isfinite(metrics['loss'])
        assert any(not torch.equal(before[name], value) for name, value in learner.agent_net.state_dict().items())
    finally:
        replay.close()
        env.close()

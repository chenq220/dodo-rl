import copy

import pytest
import torch

from algos.training.coma_learner import COMALearner
from algos.modules.agents.policy_agent import PolicyRNNAgent
from config import LearnerConfig, ReplayBufferConfig
from envs.sense.action_space_examples import ActionSpaceCounterAECEnvironment, ActionSpaceCounterParallelEnvironment
from sample.replay_buffer import ReplayBuffer
from test_action_spaces import spaces
from test_mappo_action_spaces import rollout, setup


@pytest.mark.parametrize("space", [spaces()[0], spaces()[2], spaces()[3]], ids=lambda value: value.kind)
@pytest.mark.parametrize("env_class", [ActionSpaceCounterParallelEnvironment, ActionSpaceCounterAECEnvironment])
def test_coma_updates_actor_and_critic_and_restores_snapshot(space, env_class, tmp_path):
    torch.manual_seed(123)
    agent, env, _ = setup(space, env_class)
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={'policy': 8}, batch_size=2))
    try:
        config = LearnerConfig(name="policy", algo_type="coma", standardise_rewards=False)
        config.algo_config.counterfactual_samples = 3
        learner = COMALearner("coma", config, agent, env, "cpu")
        episodes = [rollout(agent, env, 2), rollout(agent, env, 3)]
        replay.push({'policy': episodes}, notify_learning=False)
        episodes = replay.sample('policy', batch_size=2)
        before = copy.deepcopy(learner.snapshot_state())
        metrics = learner.learn_update(episodes)
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        for name, model in learner.models.items():
            assert any(not torch.equal(before[name][key], value) for key, value in model.state_dict().items())
        agent.reset_net(learner.agent_net)
        path = tmp_path / 'coma-actor.pth'
        torch.save(learner.agent_net.state_dict(), path)
        restored_agent = PolicyRNNAgent('policy_rnn', copy.deepcopy(agent.config), 'file', 'cpu',
                                        model_file_path=str(path))
        for name, value in agent.agent_net.state_dict().items():
            torch.testing.assert_close(restored_agent.agent_net.state_dict()[name], value)
        learner.restore_state(before)
        for name, model in learner.models.items():
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name][key])
    finally:
        replay.close()
        env.close()


def test_coma_rejects_stale_policy_samples():
    agent, env, _ = setup(spaces()[0])
    try:
        config = LearnerConfig(name="policy", algo_type="coma", standardise_rewards=False)
        learner = COMALearner("coma", config, agent, env, "cpu")
        episode = rollout(agent, env, 2)
        episode.metadata[0]["behavior_log_probs"]["a"] += 1
        with pytest.raises(ValueError, match="current policy"):
            learner.learn_update([episode])
    finally:
        env.close()


def test_discrete_counterfactual_baseline_enumerates_only_focal_action():
    from types import SimpleNamespace

    learner = object.__new__(COMALearner)
    learner.agent = SimpleNamespace(action_space=SimpleNamespace(kind='discrete'))
    learner.critic_action_dim = 2
    learner.critic = lambda state, actions, active: actions[..., 0, 1] + 10 * actions[..., 1, 1]
    state = torch.zeros(1, 1)
    actions = torch.tensor([[[1., 0.], [0., 1.]]])
    active = torch.ones(1, 2, dtype=torch.bool)
    probs = torch.tensor([[[0.25, 0.75], [0.60, 0.40]]])
    distribution = SimpleNamespace(categoricals=[SimpleNamespace(probs=probs)])
    result = learner._counterfactual_baseline(state, actions, active, distribution)
    torch.testing.assert_close(result, torch.tensor([[10.75, 4.0]]))


def test_continuous_counterfactual_replaces_one_complete_action_vector():
    from types import SimpleNamespace

    learner = object.__new__(COMALearner)
    learner.agent = SimpleNamespace(action_space=SimpleNamespace(kind='multi_continuous'))
    learner.config = SimpleNamespace(algo_config=SimpleNamespace(counterfactual_samples=2))
    seen = []

    def critic(state, actions, active):
        seen.append(actions.clone())
        return actions.sum(dim=(-1, -2))

    learner.critic = critic
    actions = torch.tensor([[[1., 2.], [3., 4.]]])
    samples = torch.tensor([[[5., 6.], [7., 8.]]])
    distribution = SimpleNamespace(sample=lambda: samples)
    result = learner._counterfactual_baseline(torch.zeros(1, 1), actions,
                                              torch.ones(1, 2), distribution)
    torch.testing.assert_close(result, torch.tensor([[18., 18.]]))
    for candidate in seen[:2]:
        torch.testing.assert_close(candidate[:, 0], samples[:, 0])
        torch.testing.assert_close(candidate[:, 1], actions[:, 1])
    for candidate in seen[2:]:
        torch.testing.assert_close(candidate[:, 0], actions[:, 0])
        torch.testing.assert_close(candidate[:, 1], samples[:, 1])


def test_coma_nstep_bootstraps_truncation_but_not_termination():
    from types import SimpleNamespace

    learner = object.__new__(COMALearner)
    learner.config = SimpleNamespace(gamma=0.9, algo_config=SimpleNamespace(q_nstep=3))
    rewards = torch.tensor([[1.], [2.], [0.]])
    valid = torch.tensor([[True], [True], [False]])
    truncation = torch.tensor([[False], [True], [False]])
    next_values = torch.tensor([[7.], [5.], [0.]])
    terminal = torch.zeros_like(valid)
    targets = learner._nstep_targets(rewards, terminal, truncation, valid, next_values)
    torch.testing.assert_close(targets[:2, 0], torch.tensor([6.85, 6.5]))
    terminal[1] = True
    targets = learner._nstep_targets(rewards, terminal, truncation, valid, next_values)
    torch.testing.assert_close(targets[:2, 0], torch.tensor([2.8, 2.]))


def test_coma_team_terminal_accounts_for_previously_dead_agents():
    terminated = torch.tensor([[True, False], [True, False], [True, True], [False, False]])
    truncated = torch.tensor([[False, False], [False, True], [False, True], [False, False]])
    no_next_actor = torch.tensor([True, True, True, False])
    result = COMALearner._team_terminal(terminated, truncated, no_next_actor)
    torch.testing.assert_close(result, torch.tensor([True, False, True, False]))


def test_coma_rejects_repeated_updates_and_invalid_sampling_counts():
    from scheduler.model import resolve_tunable_param_enum

    agent, env, _ = setup(spaces()[0])
    try:
        config = LearnerConfig(name='policy', algo_type='coma', standardise_rewards=False)
        config.algo_config.learn_times_per_episode = 2
        with pytest.raises(ValueError, match='learn_times_per_episode=1'):
            COMALearner('coma', config, agent, env, 'cpu')
        config.algo_config.learn_times_per_episode = 1
        learner = COMALearner('coma', config, agent, env, 'cpu')
        with pytest.raises(ValueError, match='learn_times_per_episode=1'):
            learner.dynamic_modify_config({'learn_times_per_episode': 2})
        assert learner.config.algo_config.learn_times_per_episode == 1
        with pytest.raises(ValueError, match='counterfactual_samples must be a positive integer'):
            learner.dynamic_modify_config({'counterfactual_samples': 1.5})
        assert 'learn_times_per_episode' not in {
            item.value.key for item in resolve_tunable_param_enum('coma')
        }
    finally:
        env.close()

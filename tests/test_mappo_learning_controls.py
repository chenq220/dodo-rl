"""Independent value fitting must preserve PPO policy and checkpoint semantics."""
from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch

from algos.training.mappo_learner import MAPPOActorCriticLearner
from algos.training.mappo_batch import build_batch, gae
from algos.utils.policy_batch import actor_outputs
from config import MappoConfig, ReplayBufferConfig
from sample.replay_buffer import ReplayBuffer
from test_action_spaces import spaces
from test_mappo_action_spaces import rollout, setup


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configured_learner(**overrides):
    torch.manual_seed(42)
    agent, env, original = setup(next(s for s in spaces() if s.kind == 'continuous'))
    config = copy.deepcopy(original.config)
    for key, value in overrides.items():
        setattr(config.algo_config, key, value)
    return agent, env, MAPPOActorCriticLearner('mappo', config, agent, env, 'cpu')


def assert_same_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same_state(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same_state(left, right)
    else:
        assert actual == expected


def test_legacy_defaults_keep_both_learning_rates_and_epochs():
    cfg = MappoConfig()
    assert cfg.actor_lr is None and cfg.critic_lr is None
    assert cfg.actor_epochs is None and cfg.critic_epochs is None
    assert cfg.critic_warmup_updates == 0 and not cfg.warmup_use_mc_returns
    agent, env, learner = configured_learner(lr=3e-4, epochs=3, num_mini_batch=1)
    try:
        assert learner.actor_optimizer.param_groups[0]['lr'] == 3e-4
        assert learner.critic_optimizer.param_groups[0]['lr'] == 3e-4
        metrics = learner.update([rollout(agent, env, 3)])
        assert metrics['actor_updates'] == metrics['critic_updates'] == 3
        assert metrics['critic_warmup_active'] == 0
        assert metrics['mc_returns_used'] == 0
        learner.dynamic_modify_config({'lr': 2e-4})
        assert learner.actor_optimizer.param_groups[0]['lr'] == 2e-4
        assert learner.critic_optimizer.param_groups[0]['lr'] == 2e-4
    finally:
        env.close()


def test_independent_learning_rates_epochs_and_dynamic_fallback():
    agent, env, learner = configured_learner(lr=7e-4, actor_lr=1e-5, critic_lr=1e-3,
                                            actor_epochs=1, critic_epochs=4, num_mini_batch=2)
    try:
        assert learner.actor_optimizer.param_groups[0]['lr'] == 1e-5
        assert learner.critic_optimizer.param_groups[0]['lr'] == 1e-3
        metrics = learner.update([rollout(agent, env, 2), rollout(agent, env, 3)])
        assert metrics['actor_updates'] == 2
        assert metrics['critic_updates'] == 8
        learner.dynamic_modify_config({'lr': 5e-4, 'critic_lr': 2e-3})
        assert learner.actor_optimizer.param_groups[0]['lr'] == 1e-5
        assert learner.critic_optimizer.param_groups[0]['lr'] == 2e-3
        learner.dynamic_modify_config({'actor_lr': None, 'critic_lr': None,
                                      'actor_epochs': None, 'critic_epochs': None, 'epochs': 2})
        assert learner.actor_optimizer.param_groups[0]['lr'] == 5e-4
        assert learner.critic_optimizer.param_groups[0]['lr'] == 5e-4
        metrics = learner.update([rollout(agent, env, 2), rollout(agent, env, 3)])
        assert metrics['actor_updates'] == metrics['critic_updates'] == 4
    finally:
        env.close()


def test_warmup_freezes_actor_and_existing_adam_state_and_resumes_after_restore():
    agent, env, learner = configured_learner(actor_epochs=1, critic_epochs=3, num_mini_batch=1)
    try:
        episodes = [rollout(agent, env, 3)]
        learner.update(episodes)
        assert learner.actor_optimizer.state_dict()['state']  # exercise populated Adam moments
        learner.dynamic_modify_config({'critic_warmup_updates': 3})
        learner.actor.eval()
        before_gradients = [None if parameter.grad is None else parameter.grad.clone()
                            for parameter in learner.actor.parameters()]
        before = copy.deepcopy(learner.snapshot_state())
        for expected_count in (2, 3):
            metrics = learner.update(episodes)
            assert metrics['critic_warmup_active'] == 1
            assert metrics['actor_updates'] == 0 and metrics['critic_updates'] == 3
            assert learner.update_cnt == expected_count
            assert not learner.actor.training
            assert_same_state(learner.actor.state_dict(), before['agent_net'])
            assert_same_state(learner.actor_optimizer.state_dict(), before['actor_optimizer'])
            for parameter, gradient in zip(learner.actor.parameters(), before_gradients):
                assert_same_state(parameter.grad, gradient)
        assert any(not torch.equal(value, before['critic_net'][key])
                   for key, value in learner.critic.state_dict().items())
        # A restored count must resume the remaining warmup, rather than restart or skip it.
        learner.restore_state(before)
        assert learner.update_cnt == 1
        assert learner.update(episodes)['critic_warmup_active'] == 1
        assert learner.update(episodes)['critic_warmup_active'] == 1
        metrics = learner.update(episodes)
        assert metrics['critic_warmup_active'] == 0 and metrics['actor_updates'] == 1
        assert any(not torch.equal(value, before['agent_net'][key])
                   for key, value in learner.actor.state_dict().items())
    finally:
        env.close()


def test_rejected_batch_does_not_spend_a_warmup_update():
    agent, env, learner = configured_learner(critic_warmup_updates=1)
    try:
        episodes = [rollout(agent, env, 3)]
        invalid = copy.deepcopy(episodes)
        invalid[0].metadata[0].pop('behavior_log_probs')
        with pytest.raises(ValueError, match='behavior log probabilities'):
            learner.update(invalid)
        assert learner.update_cnt == 0
        metrics = learner.update(episodes)
        assert learner.update_cnt == 1
        assert metrics['critic_warmup_active'] == 1 and metrics['actor_updates'] == 0
        assert learner.update(episodes)['critic_warmup_active'] == 0
    finally:
        env.close()


@pytest.mark.parametrize('updates', [
    {'actor_lr': 0., 'critic_lr': 0.01},
    {'critic_lr': float('inf')},
    {'actor_epochs': 0},
    {'critic_epochs': 1.5},
    {'critic_warmup_updates': -1},
    {'warmup_use_mc_returns': 'true'},
])
def test_invalid_dynamic_controls_do_not_partially_change_live_optimizers(updates):
    _, env, learner = configured_learner(actor_lr=1e-4, critic_lr=2e-4)
    try:
        before = copy.deepcopy(learner.config.algo_config)
        with pytest.raises(ValueError):
            learner.dynamic_modify_config(updates)
        assert learner.config.algo_config == before
        assert learner.actor_optimizer.param_groups[0]['lr'] == 1e-4
        assert learner.critic_optimizer.param_groups[0]['lr'] == 2e-4
    finally:
        env.close()


def test_mc_targets_are_only_used_during_requested_warmup():
    agent, env, learner = configured_learner(critic_warmup_updates=1, warmup_use_mc_returns=True,
                                            gae_lambda=0.37, num_mini_batch=1)
    try:
        episodes = [rollout(agent, env, 3)]
        with patch('algos.training.mappo_learner.gae', wraps=gae) as calculate:
            metrics = learner.update(episodes)
            assert calculate.call_args_list[0].args[-1] == 1.0
            assert metrics['mc_returns_used'] == 1
        with patch('algos.training.mappo_learner.gae', wraps=gae) as calculate:
            metrics = learner.update(episodes)
            assert calculate.call_args_list[0].args[-1] == 0.37
            assert metrics['mc_returns_used'] == 0
    finally:
        env.close()


def test_lambda_one_returns_do_not_cross_done_or_padding_and_bootstrap_cutoffs():
    # Row 1 ends an episode; the following row is another episode. Row 3 is padding.
    rewards = torch.tensor([1., 2., 100., 999.]).reshape(4, 1, 1, 1)
    values = torch.tensor([10., 20., 30., 40., 999.]).reshape(5, 1, 1, 1)
    valid = torch.tensor([1., 1., 1., 0.]).reshape_as(rewards)
    terminated = torch.tensor([0., 1., 0., 0.]).reshape_as(rewards)
    truncated = torch.zeros_like(rewards)
    _, returns = gae(rewards, values, terminated, truncated, valid, 0.5, 1.)
    torch.testing.assert_close(returns[:3].flatten(), torch.tensor([2., 2., 120.]))
    # A time limit cuts recursion but bootstraps its final observation's value.
    terminated[1] = 0
    truncated[1] = 1
    _, returns = gae(rewards, values, terminated, truncated, valid, 0.5, 1.)
    torch.testing.assert_close(returns[:3].flatten(), torch.tensor([9.5, 17., 120.]))


def test_post_update_kl_describes_final_policy_after_the_last_optimizer_step():
    agent, env, learner = configured_learner(actor_lr=0.02, actor_epochs=1,
                                            critic_epochs=2, num_mini_batch=1)
    try:
        episodes = [rollout(agent, env, 3)]
        batch = build_batch(agent, episodes, 'cpu')
        metrics = learner.update(episodes)
        with torch.no_grad():
            outputs = actor_outputs(learner.actor, batch)[:-1]
            distribution = learner.actor.distribution(outputs, batch.get('action_masks'), batch['temperature'])
            log_ratio = distribution.log_prob(batch['actions']) - batch['old_log_probs']
            mask = batch['valid'] * batch['active'][:-1]
            expected_kl = (((log_ratio.exp() - 1) - log_ratio) * mask).sum() / mask.sum()
            expected_clip = (((log_ratio.exp() - 1).abs() > learner.config.algo_config.eps_clip)
                             .float() * mask).sum() / mask.sum()
        assert metrics['pre_approx_kl'] == pytest.approx(0., abs=1e-6)
        assert metrics['post_approx_kl'] == pytest.approx(expected_kl.item(), abs=1e-6)
        assert metrics['post_clip_fraction'] == pytest.approx(expected_clip.item(), abs=1e-6)
        assert metrics['post_approx_kl'] > 1e-4  # first/last step cannot be invisible
    finally:
        env.close()


def test_raw_advantage_and_value_diagnostics_exclude_padding():
    agent, env, learner = configured_learner(critic_warmup_updates=1, num_mini_batch=1)
    try:
        learner.config.common_reward = False
        episodes = [rollout(agent, env, 2), rollout(agent, env, 4)]
        batch = build_batch(agent, episodes, 'cpu')
        with torch.no_grad():
            before_values = learner.critic(learner.critic_inputs(batch))
            advantages, returns = gae(batch['rewards'], before_values, batch['terminated'],
                                      batch['truncated'], batch['valid'], learner.config.gamma,
                                      learner.config.algo_config.gae_lambda)
            selected_advantages = advantages[(batch['valid'] * batch['active'][:-1]).bool()]
            mask = batch['valid'].bool()
            selected_returns, selected_values = returns[mask], before_values[:-1][mask]
            expected_mse = (selected_returns - selected_values).square().mean().item()
            expected_ev = 1 - (selected_returns - selected_values).var(unbiased=False) / selected_returns.var(unbiased=False)
        metrics = learner.update(episodes)
        assert metrics['raw_advantage_mean'] == pytest.approx(selected_advantages.mean().item(), abs=1e-6)
        assert metrics['raw_advantage_std'] == pytest.approx(selected_advantages.std(unbiased=False).item(), abs=1e-6)
        assert metrics['return_mean'] == pytest.approx(selected_returns.mean().item(), abs=1e-6)
        assert metrics['return_std'] == pytest.approx(selected_returns.std(unbiased=False).item(), abs=1e-6)
        assert metrics['value_mean'] == pytest.approx(selected_values.mean().item(), abs=1e-6)
        assert metrics['batch_value_mse_pre'] == pytest.approx(expected_mse, abs=1e-6)
        assert metrics['batch_value_explained_variance_pre'] == pytest.approx(expected_ev.item(), abs=1e-5)
        assert metrics['batch_value_explained_variance_valid'] == 1
    finally:
        env.close()


def test_independent_mc_value_diagnostic_uses_only_complete_raw_reward_episodes():
    agent, env, learner = configured_learner(num_mini_batch=1)
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={'policy': 32}, batch_size=1))
    try:
        learner.config.common_reward = False
        complete = rollout(agent, env, 4)
        # The counter environment only has time limits: make this recorded final
        # transition a genuine environment terminal for the diagnostic contract.
        for name in agent.agent_list:
            complete.terminations[name][-1] = True
            complete.truncations[name][-1] = False
        partial_cutoff = rollout(agent, env, 2)
        batch = build_batch(agent, [complete], 'cpu')
        with torch.no_grad():
            values = learner.critic(learner.critic_inputs(batch))
            _, returns = gae(batch['rewards'], values, batch['terminated'], batch['truncated'],
                             batch['valid'], learner.config.gamma, 1.)
            selected = returns[batch['valid'].bool()]
            expected_mse = (selected - values[:-1][batch['valid'].bool()]).square().mean().item()
        metrics = learner.update([complete, partial_cutoff])
        assert metrics['mc_complete_samples'] == complete.seq_len * agent.n_agents
        assert metrics['mc_return_mean'] == pytest.approx(selected.mean().item(), abs=1e-6)
        assert metrics['mc_return_std'] == pytest.approx(selected.std(unbiased=False).item(), abs=1e-6)
        assert metrics['mc_value_mse_pre'] == pytest.approx(expected_mse, abs=1e-6)
        # A terminal-ending cursor fragment still misses the episode prefix.
        replay.push({'policy': [complete]}, notify_learning=False)
        replay.sample('policy', batch_size=2, strategy='on_policy_cursor_steps')
        fragment = replay.sample('policy', batch_size=2, strategy='on_policy_cursor_steps')
        assert fragment[0].burn_in is not None
        assert learner.update(fragment)['mc_complete_samples'] == 0
        learner.config.standardise_rewards = True
        metrics = learner.update([complete])
        assert metrics['mc_complete_samples'] == metrics['mc_value_explained_variance_valid'] == 0
    finally:
        replay.close()
        env.close()

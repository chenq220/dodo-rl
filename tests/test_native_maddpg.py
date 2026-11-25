import copy
import math

import numpy as np
import pytest
import toml
import torch

from common.creator import create_agent, create_learner
from common.plugins import activate_plugins
from config import ReplayBufferConfig, TrainConfig
from envs.sense.action_space_examples import (
    ActionSpaceCounterAECEnvironment, ActionSpaceCounterParallelEnvironment,
)
from portal.plugins.registry import project_plugin_registry
from roles.common.config_alignment import align_agent_config_with_learner_config
from algos.learners.batch_schema import assemble_episodes
from sample.model import EpisodeBatch
from sample.replay_buffer import ReplayBuffer
from scheduler.model import AgentCompose
from scheduler.rollout_sample import RolloutSampleWorker
from scheduler.rollout_learn import RolloutLearnWorker
from scheduler.train_task import TrainTaskManager
from test_action_spaces import spaces
from test_mappo_action_spaces import rollout
from utils.module_utils import inject_env_dims
from utils.toml_config import TomlTrainingConfig


def _setup(space, algo_config=None):
    registry = activate_plugins(default_plugins=project_plugin_registry())
    config = TomlTrainingConfig(config_text=toml.dumps({
        "agents": {"0": {
            "name": "policy", "agent_names": ["a", "b"],
            "obs_dim": 2, "action_space_spec": space.to_dict(),
        }},
        "learners": {"0": {
            "name": "policy", "algo_type": "maddpg", "common_reward": False,
            "standardise_rewards": False,
            "algo_config": {"hidden_dim": 16, **(algo_config or {})},
        }},
    }), plugin_registry=registry)
    agent_config = config.agents_config[0]
    learner_config = config.learners_config[0]
    align_agent_config_with_learner_config(agent_config, learner_config)
    agent = create_agent(agent_config.agent_type, agent_config, "blanco",
                         agent_config.agent_names, "cpu")
    env = ActionSpaceCounterParallelEnvironment({"a": space, "b": space}, max_steps=4)
    inject_env_dims(env, [agent])
    return agent, env, create_learner(learner_config, agent, env, "cpu", "blanco")


def test_maddpg_training_rollout_explores_before_the_first_update(monkeypatch):
    agent, env, learner = _setup(spaces()[3])
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={agent.name: 8}, batch_size=2))
    task_manager = TrainTaskManager(
        config=TrainConfig(
            max_episode=2, seed=42, start_epsilon=0.0, end_epsilon=0.0,
            mid_eval_enabled=False, end_eval_episodes=0, sample_learn_max_lag=None,
        ),
        policy_agent_configs=[agent.config],
        learner_configs=[learner.config],
    )
    try:
        rollout_stds = []
        sample_actions = agent.agent_net.sample_actions

        def record_rollout_std(mean, log_std, *, exploration_std=None):
            rollout_stds.append(exploration_std)
            return sample_actions(mean, log_std, exploration_std=exploration_std)

        monkeypatch.setattr(agent.agent_net, "sample_actions", record_rollout_std)
        # The unperturbed actor produces zero in every normalized action dimension.
        with torch.no_grad():
            for param in agent.agent_net.parameters():
                param.zero_()

        worker = RolloutSampleWorker(
            env, task_manager, replay, AgentCompose(step_agent=[agent]), [agent],
        )
        first = worker.rollout_one_episode(step_limit=1)[agent.name][0]
        second = worker.rollout_one_episode(step_limit=1)[agent.name][0]
        assert worker.last_episode_info.epsilon == 0.0
        actions = assemble_episodes(
            [first, second], learner.batch_schema, agent=agent, device="cpu",
        ).data["actions"]
        assert torch.count_nonzero(actions[0]) > 0
        assert torch.count_nonzero(actions[1]) > 0
        assert not torch.equal(actions[0], actions[1])
        assert rollout_stds == [agent.config.algo_config.exploration_std] * 2
        assert worker.last_episode_info.actions_param[agent.name].epsilon == 0.0
    finally:
        task_manager.close()
        replay.close()
        env.close()


@pytest.mark.parametrize("space", spaces()[2:], ids=lambda space: space.kind)
def test_maddpg_assembles_encoded_actions_in_the_learner(space, monkeypatch):
    from envs.action.spaces import ActionSpace

    agent, env, learner = _setup(space)
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={agent.name: 8}, batch_size=2))
    try:
        episodes = [rollout(agent, env, step_limit=2), rollout(agent, env, step_limit=3)]
        replay.push({agent.name: episodes}, notify_learning=False)
        trajectories = replay.sample(agent.name, strategy="uniform", batch_size=2, seed=5)
        assert all(not hasattr(episode, "_validated_action_records") for episode in trajectories)
        expected = assemble_episodes(
            trajectories, learner.batch_schema, agent=agent, device="cpu",
        ).data

        def unexpected_restore(*args, **kwargs):
            pytest.fail("MADDPG should consume recorded vectors without restoring business actions")

        def capture_batch(batch):
            assert batch.data.keys() == expected.keys()
            assert all(torch.equal(batch.data[name], expected[name]) for name in expected)
            return {"captured": 1}

        monkeypatch.setattr(ActionSpace, "restore", staticmethod(unexpected_restore))
        monkeypatch.setattr(learner, "update", capture_batch)
        assert learner.learn_update(trajectories) == {"captured": 1}

        for episode in trajectories:
            for field in (episode.actions, episode.last_actions):
                for aid, records in field.items():
                    field[aid] = [None if record is None else record["value"] for record in records]
        assert learner.learn_update(trajectories) == {"captured": 1}
    finally:
        replay.close()
        env.close()


@pytest.mark.parametrize('field', ['active_agent_mask', 'next_active_agent_mask'])
def test_maddpg_rejects_partial_activity_masks(field):
    agent, env, learner = _setup(spaces()[2])
    try:
        episode = rollout(agent, env, step_limit=2)
        del getattr(episode, field)['b']
        with pytest.raises(ValueError, match=field):
            learner.assemble_batch([episode])
    finally:
        env.close()


@pytest.mark.parametrize("space", spaces()[2:], ids=lambda space: space.kind)
def test_maddpg_rejects_mismatched_action_records(space):
    agent, env, learner = _setup(space)
    try:
        episode = rollout(agent, env, step_limit=2)
        episode.actions[agent.agent_list[0]][0]["space_signature"] = "mismatched"
        with pytest.raises(ValueError, match="signature mismatch"):
            learner.assemble_batch([episode])
    finally:
        env.close()


def test_maddpg_generation_restores_snapshot_before_assembling(monkeypatch):
    agent, env, learner = _setup(spaces()[2])
    try:
        episode = rollout(agent, env, step_limit=2)
        snapshot = copy.deepcopy(learner.actor.state_dict())
        with torch.no_grad():
            next(learner.actor.parameters()).add_(1)

        def capture_batch(batch):
            assert learner.generation == 1
            assert all(torch.equal(value, snapshot[name])
                       for name, value in learner.actor.state_dict().items())
            return {"captured": 1}

        monkeypatch.setattr(learner, "update", capture_batch)
        assert learner.learn_update([episode], generation=1) == {"captured": 1}
    finally:
        env.close()


@pytest.mark.parametrize("space", spaces()[2:], ids=lambda space: space.kind)
def test_maddpg_trains_off_policy_from_joint_replay_and_loads_models(space, tmp_path):
    torch.manual_seed(123)
    agent, env, learner = _setup(space)
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={"policy": 8}, batch_size=2))
    try:
        # Exploration scale belongs to the rollout behavior policy, not the DPG objective.
        learner.config.algo_config.reg = 0.0
        # Generate replay with a different behavior policy so replacing every
        # agent's action would fail the focal-action check below.
        with torch.no_grad():
            for policy in agent.agent_net.policies:
                policy[-1].bias.add_(0.5)
        if space.kind == "multi_continuous":
            learner.config.common_reward = True
            learner.config.standardise_rewards = True
            learner.config.standardise_returns = True
        episodes = [rollout(agent, env, step_limit=2), rollout(agent, env, step_limit=3)]
        replay.push({"policy": episodes}, notify_learning=False)
        assert learner.config.algo_config.sample_strategy == "uniform_steps"
        trajectories = replay.sample("policy", batch_size=2, strategy="uniform_steps", seed=5)
        assert len(trajectories) == 2 and all(item.seq_len == 1 for item in trajectories)
        fields = assemble_episodes(trajectories, learner.batch_schema, agent=agent, device="cpu").data
        # MADDPG 使用最小 schema：不组装 burn-in、集中状态和旧策略统计。
        assert not learner.batch_schema.uses_source("burn_in")
        assert "old_log_probs" not in fields
        assert "burn_in" not in fields
        assert "state" not in fields
        before = copy.deepcopy(learner.snapshot_state())
        critic_inputs = [[], []]
        handles = [
            critic.register_forward_pre_hook(
                lambda _, inputs, slot=slot: critic_inputs[slot].append(inputs[0].detach().clone())
            )
            for slot, critic in enumerate(learner.critic)
        ]
        try:
            metrics = learner.learn_update(trajectories)
        finally:
            for handle in handles:
                handle.remove()
        assert all(math.isfinite(value) for value in metrics.values())
        for name, model in learner.models.items():
            assert any(not torch.equal(value, before[name][key])
                       for key, value in model.state_dict().items())
        for slot, policy in enumerate(learner.actor.policies):
            torch.testing.assert_close(
                policy[-1].log_std.weight, before["agent_net"][f"policies.{slot}.4.log_std.weight"],
            )
            torch.testing.assert_close(
                policy[-1].log_std.bias, before["agent_net"][f"policies.{slot}.4.log_std.bias"],
            )
        assert any(not torch.equal(value, before["target_actor"][key])
                   for key, value in learner.target_actor.state_dict().items())

        # A focal actor may change only its own slice of the critic's joint action input.
        width = space.layout["value_dim"]
        action_start = agent.n_agents * agent.config.obs_dim
        for slot, (replayed, actor_input) in enumerate(critic_inputs):
            own_start = action_start + slot * width
            own = slice(own_start, own_start + width)
            other_start = action_start + (1 - slot) * width
            other = slice(other_start, other_start + width)
            assert not torch.equal(actor_input[:, own], replayed[:, own])
            torch.testing.assert_close(actor_input[:, other], replayed[:, other])

        snapshot = copy.deepcopy(learner.snapshot_state())
        torch.manual_seed(321)
        expected = learner.learn_update(trajectories)
        learner.restore_state(snapshot)
        torch.manual_seed(321)
        assert learner.learn_update(trajectories) == pytest.approx(expected)

        weights = {name: tmp_path / f"{name}.pth" for name in learner.models}
        for name, path in weights.items():
            torch.save(learner.models[name].state_dict(), path)
        restored = create_agent("maddpg_mlp", copy.deepcopy(agent.config), "file",
                                agent.agent_list, "cpu", model_file_path=weights["agent_net"])
        loaded = create_learner(learner.config, restored, env, "cpu", "file", model_file_path=weights)
        obs = {key: np.array([1.0, -1.0]) for key in agent.agent_list}
        agent.reset_net(learner.agent_net)
        expected_actions, _ = agent.get_actions(obs, ["b"], {}, evaluate=True, epsilon=1)
        actual_actions, _ = restored.get_actions(obs, ["b"], {}, deterministic=True, epsilon=1)
        torch.testing.assert_close(
            loaded.critic.state_dict(), learner.critic.state_dict(),
        )
        np.testing.assert_array_equal(space.encode(expected_actions["b"]), space.encode(actual_actions["b"]))
        explored, _ = restored.get_actions(obs, ["b"], {}, epsilon=0)
        assert not np.array_equal(space.encode(explored["b"]), space.encode(actual_actions["b"]))

        torch.save({key: value for key, value in learner.actor.state_dict().items()
                    if ".log_std." not in key}, weights["agent_net"])
        legacy = create_agent(agent.config.agent_type, copy.deepcopy(agent.config), "file",
                              agent.agent_list, "cpu", model_file_path=weights["agent_net"])
        legacy_learner = create_learner(learner.config, legacy, env, "cpu", "file", model_file_path=weights)
        legacy_actions, _ = legacy.get_actions(obs, ["b"], {}, deterministic=True)
        np.testing.assert_array_equal(space.encode(legacy_actions["b"]), space.encode(actual_actions["b"]))
        for policy in legacy_learner.actor.policies:
            scale = policy[-1].log_std
            torch.testing.assert_close(scale.weight, torch.zeros_like(scale.weight))
            torch.testing.assert_close(
                scale.bias, torch.full_like(scale.bias, math.log(agent.config.algo_config.exploration_std)),
            )
    finally:
        replay.close()
        env.close()


def test_maddpg_bootstraps_truncations_and_rejects_aec(monkeypatch):
    space = spaces()[2]
    agent, env, learner = _setup(space)
    aec = ActionSpaceCounterAECEnvironment({"a": space, "b": space}, max_steps=4)
    try:
        with pytest.raises(ValueError, match="Parallel joint transitions"):
            create_learner(learner.config, agent, aec, "cpu", "blanco")
        episode = rollout(agent, env, step_limit=2)
        fields = assemble_episodes([episode], learner.batch_schema, agent=agent, device="cpu").data
        fields["active"][0, -1] = False
        fields["terminated"][0, -1] = torch.tensor([True, False])
        fields["truncated"][0, -1] = torch.tensor([False, True])
        fields["rewards"].fill_(1.0)
        with torch.no_grad():
            for critic in learner.target_critic:
                for param in critic.parameters():
                    param.zero_()
                critic[-1].bias.fill_(2.0)
        calculate_targets = learner._targets

        def check_targets(*args):
            targets = calculate_targets(*args)
            torch.testing.assert_close(targets[-1], torch.tensor([1., 1. + 2 * learner.config.gamma]))
            return targets

        monkeypatch.setattr(learner, "_targets", check_targets)
        learner.update(EpisodeBatch(fields))
    finally:
        env.close()
        aec.close()


def test_maddpg_learner_uses_deterministic_actions_while_rollout_explores(monkeypatch):
    space = spaces()[2]
    agent, env, learner = _setup(space)
    try:
        episodes = [rollout(agent, env, step_limit=2), rollout(agent, env, step_limit=3)]
        calls = {"actor": 0, "target": 0}

        def reject_stochastic_actions(*_args, **_kwargs):
            raise AssertionError("learner updates must not sample the rollout behavior policy")

        def actor_actions(mean):
            calls["actor"] += 1
            return mean.tanh()

        def target_actions(mean):
            calls["target"] += 1
            return mean.tanh()

        monkeypatch.setattr(learner.actor, "sample_actions", reject_stochastic_actions)
        monkeypatch.setattr(learner.target_actor, "sample_actions", reject_stochastic_actions)
        monkeypatch.setattr(learner.actor, "deterministic_actions", actor_actions)
        monkeypatch.setattr(learner.target_actor, "deterministic_actions", target_actions)

        metrics = learner.learn_update(episodes)
        assert all(math.isfinite(value) for value in metrics.values())
        assert calls == {"actor": 1, "target": 1}

        obs = {key: np.array([1.0, -1.0]) for key in agent.agent_list}
        first, _ = agent.get_actions(obs, ["a"], {}, epsilon=0.0)
        second, _ = agent.get_actions(obs, ["a"], {}, epsilon=0.0)
        assert not np.array_equal(space.encode(first["a"]), space.encode(second["a"]))
    finally:
        env.close()


def test_maddpg_actor_warmup_freezes_actor_and_uses_separate_lrs():
    space = spaces()[2]
    agent, env, learner = _setup(space, algo_config={
        "actor_lr": 1e-5, "critic_lr": 1e-3, "actor_warmup_updates": 2,
    })
    try:
        assert learner.actor_optimizer.param_groups[0]["lr"] == 1e-5
        assert learner.critic_optimizer.param_groups[0]["lr"] == 1e-3
        episodes = [rollout(agent, env, step_limit=2), rollout(agent, env, step_limit=3)]
        actor_before = copy.deepcopy(learner.actor.state_dict())
        critic_before = copy.deepcopy(learner.critic.state_dict())
        for _ in range(2):
            metrics = learner.learn_update(episodes)
            assert metrics["actor_warmup"] == 1.0
            assert metrics["pg_loss"] == 0.0 and metrics["agent_grad_norm"] == 0.0
            assert all(torch.equal(value, actor_before[key])
                       for key, value in learner.actor.state_dict().items())
        assert any(not torch.equal(value, critic_before[key])
                   for key, value in learner.critic.state_dict().items())
        metrics = learner.learn_update(episodes)
        assert metrics["actor_warmup"] == 0.0
        assert any(not torch.equal(value, actor_before[key])
                   for key, value in learner.actor.state_dict().items())
    finally:
        env.close()


@pytest.mark.parametrize("configured,actual,expected_decisions", [
    ("uniform_steps", "uniform", 3),
    ("uniform", "uniform_steps", 1),
])
def test_nstep_offline_override_uses_actual_sampling_strategy(
        configured, actual, expected_decisions, monkeypatch):
    agent, env, learner = _setup(spaces()[2], algo_config={
        "n_step": 3, "sample_strategy": configured, "batch_size": 1,
    })
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={agent.name: 4}))
    try:
        replay.push({agent.name: [rollout(agent, env, step_limit=3)]}, notify_learning=False)
        original_sample = replay.sample

        def sample_from_first_step(*args, **kwargs):
            kwargs["seed"] = 1
            episodes = original_sample(*args, **kwargs)
            assert episodes[0].seq_len == 3
            assert episodes.sample_strategy == actual
            return episodes

        monkeypatch.setattr(replay, "sample", sample_from_first_step)
        worker = RolloutLearnWorker([learner], TrainConfig(), replay, None)
        observed_strategies = []
        original_update = learner.update

        def record_update(batch):
            observed_strategies.append(batch.sample_strategy)
            return original_update(batch)

        monkeypatch.setattr(learner, "update", record_update)
        decision_counts = []
        handle = learner.critic[0].register_forward_pre_hook(
            lambda _, inputs: decision_counts.append(inputs[0].shape[0]))
        try:
            result = worker._update_one_learner_by_config(
                learner, learner.config.algo_config, sample_strategy=actual)
        finally:
            handle.remove()
        assert observed_strategies == [actual]
        assert decision_counts[0] == expected_decisions
        assert all(math.isfinite(value) for value in result.values())
        assert learner.config.algo_config.sample_strategy == configured
    finally:
        replay.close()
        env.close()

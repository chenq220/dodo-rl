"""多组框架收口：真实暂存峰值、明细保存、切片初态和模型兼容性。"""
import copy
import pickle
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from algos.modules.agents.rnn_agent import RNNAgent
from algos.training.qmix_learner import QMixerLearner
from common.exception import ModelCheckpointCompatibilityException, ReplayBatchShapeException
from common.info_bridge import info_bridge
from config import EnvConfig, LearnerConfig, LocalStorageConfig, QmixConfig, ReplayBufferConfig
from sample.replay.replay_payload import encode_replay_push_payload
from sample.replay_buffer import ReplayBuffer
from sample.sample_transfer import EpisodeTransition
from test_agent_grouping import AGENT_NAME, GROUPS, FLAT_IDS, _build_rnn_agent, _ma_group_episode


@pytest.fixture(autouse=True)
def replay_budget_context():
    with info_bridge.scope(toml_config={
        "env_config": SimpleNamespace(max_step_per_episode=3), "stage_task_config": None,
    }):
        yield


def _replay(tmp_path, pool="memory", capacities=None):
    return ReplayBuffer(ReplayBufferConfig(
        agent_capacities=capacities or {AGENT_NAME: 2}, tensor_pool=pool,
        storage=LocalStorageConfig(data_path=str(tmp_path)), batch_size=1,
    ))


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
@pytest.mark.parametrize("wire", [False, True])
def test_full_pool_oversized_group_push_and_snapshot(tmp_path, pool, wire):
    groups = GROUPS + (("uav_12", "uav_13", "uav_14", "uav_15"),)
    replay = _replay(tmp_path / "source", pool)
    restored = None
    calls = []
    replay.learn_batches_update_caller = lambda state, episode: calls.append(state)
    try:
        for episode_num in (1, 2):
            episodes = [_ma_group_episode(group, episode_num) for group in groups]
            for index, episode in enumerate(episodes):
                episode.group_index = index
                if wire:
                    episode.prepare_for_replay()
            payload = encode_replay_push_payload({AGENT_NAME: episodes}) if wire else {AGENT_NAME: episodes}
            replay.push(payload, notify_learning=True, episode_num=episode_num)
        assert len(calls) == 2
        assert replay.write_count[AGENT_NAME] == 8
        assert len(replay.tensor_pool.records) == 6
        assert [tuple(ep.agent_ids) for ep in replay.ordered_buffer(AGENT_NAME)] == list(groups[-2:])
        assert all(ep.episode_num == 2 for ep in replay.ordered_buffer(AGENT_NAME))
        path = replay.save_snapshot(tmp_path / "grouped.bin")
        restored = _replay(tmp_path / "restored", pool)
        restored.load_snapshot(path)
        restored.push({AGENT_NAME: [_ma_group_episode(group, 3) for group in groups]}, notify_learning=False)
        assert len(restored.tensor_pool.records) == 6
        assert all(ep.episode_num == 3 for ep in restored.ordered_buffer(AGENT_NAME))
    finally:
        if restored is not None:
            restored.close()
        replay.close()


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_cross_partition_full_batch_failure_preserves_old_rows(tmp_path, pool, monkeypatch):
    replay = _replay(tmp_path, pool, {AGENT_NAME: 2, "other": 1})
    try:
        replay.push({AGENT_NAME: [_ma_group_episode(g, 1) for g in GROUPS],
                     "other": _ma_group_episode(GROUPS[0], 1)}, notify_learning=False)
        before = {index: {key: value.clone() for key, value in row.items()}
                  for index, row in replay.tensor_pool.records.items()}
        old_sequences = dict(replay.write_count)
        old_ids = dict(replay._next_step_meta_id)
        store = replay._store_transition
        calls = 0

        def fail_third(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("injected third-entry failure")
            return store(**kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(replay, "_store_transition", fail_third)
            with pytest.raises(RuntimeError, match="third-entry"):
                replay.push({AGENT_NAME: [_ma_group_episode(g, 2) for g in GROUPS],
                             "other": _ma_group_episode(GROUPS[0], 2)}, notify_learning=False)
        assert replay.write_count == old_sequences
        assert replay._next_step_meta_id == old_ids
        after = replay.tensor_pool.records
        assert after.keys() == before.keys()
        for index, row in before.items():
            for key, tensor in row.items():
                torch.testing.assert_close(after[index][key], tensor)
        replay.push({AGENT_NAME: [_ma_group_episode(g, 3) for g in GROUPS],
                     "other": _ma_group_episode(GROUPS[0], 3)}, notify_learning=False)
        assert all(ep.episode_num == 3 for ep in replay.ordered_buffer(AGENT_NAME))
        assert replay.ordered_buffer("other")[0].episode_num == 3
    finally:
        replay.close()


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
@pytest.mark.parametrize("wire", [False, True])
def test_details_save_all_groups_before_capacity_eviction(tmp_path, pool, wire):
    replay = _replay(tmp_path / "source", pool, {AGENT_NAME: 1})
    restored = _replay(tmp_path / "restored", pool, {AGENT_NAME: 8})
    saved = []
    replay._save_to_backend_storage = saved.append
    try:
        episodes = [_ma_group_episode(group, 11) for group in GROUPS]
        for index, episode in enumerate(episodes):
            episode.group_index = index
            if wire:
                episode.prepare_for_replay()
        payload = encode_replay_push_payload({AGENT_NAME: episodes}) if wire else {AGENT_NAME: episodes}
        replay.push(payload, save=True, notify_learning=False)
        assert replay.agent_buffer_size(AGENT_NAME) == 1
        assert len(saved) == 1
        restored.load_replay_buffer(iter(saved))
        loaded = restored.ordered_buffer(AGENT_NAME)
        assert [tuple(ep.agent_ids) for ep in loaded] == list(GROUPS)
        assert [ep.group_index for ep in loaded] == [0, 1, 2]
        for original, actual in zip(episodes, loaded):
            assert actual.actions == original.actions
            assert actual.avail_actions == original.avail_actions
            np.testing.assert_array_equal(actual.states, original.states)
        replay.push(_ma_group_episode(GROUPS[0], 12), agent_name=AGENT_NAME,
                    save=True, notify_learning=False)
        assert isinstance(replay._cover_data_from_dict(saved[-1])[AGENT_NAME], EpisodeTransition)
    finally:
        restored.close()
        replay.close()


def _learner(groups=GROUPS, *, legacy=False, standardise=False):
    config = copy.deepcopy(_build_rnn_agent(groups, obs_dim=4).config)
    config.state_shape = 4
    if legacy:
        config.agent_names = list(groups[0])
        config.agent_groups = []
    agent = RNNAgent(agent_type="rnn", config=config, agent_net_source="blanco", device="cpu")
    learner_config = LearnerConfig(
        name=AGENT_NAME, algo_type="qmix", state_shape=4, algo_config=QmixConfig(),
        standardise_rewards=standardise, standardise_returns=standardise,
    )
    env = SimpleNamespace(
        config=EnvConfig(max_step_per_episode=6),
        state_space=lambda: gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
    )
    return QMixerLearner(algo_type="qmix", config=learner_config, agent=agent, env=env, device="cpu")


def _episodes():
    episodes = [_ma_group_episode(group, 15, steps=6) for group in GROUPS]
    for index, episode in enumerate(episodes):
        episode.group_index = index
        for aid in episode.agent_ids:
            episode.actions[aid] = [(t + index) % 2 for t in range(6)]
            for field in (episode.obs, episode.next_obs):
                field[aid] = [value + index * 10 for value in field[aid]]
        episode.states = [value + index * 10 for value in episode.states]
        episode.next_states = [value + index * 10 for value in episode.next_states]
    return episodes


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_slice_burn_in_restores_both_macs_and_previous_actions(tmp_path, pool):
    learner = _learner()
    replay = _replay(tmp_path, pool, capacities={AGENT_NAME: 8})
    try:
        episodes = _episodes()
        replay.push({AGENT_NAME: episodes}, notify_learning=False)
        starts = [1, 3, 0]
        chunks = [replay._materialize_episode(meta, start, start + 2, sliced=True)
                  for meta, start in zip(replay.buffer[AGENT_NAME], starts)]
        chunks = pickle.loads(pickle.dumps(chunks))
        assert [chunk.seq_len for chunk in chunks] == [2, 2, 2]
        assert [chunk.burn_in.seq_len for chunk in chunks[:2]] == [1, 3]
        assert chunks[0].burn_in.agent_ids == chunks[0].agent_ids
        window = EpisodeTransition.to_batch(learner.agent.agent_list, chunks, expected_n_agents=4, action_dim=2)
        assert window["filled"].sum().item() == 6
        with torch.no_grad():
            for param in learner.target_mac.agent.parameters():
                param.add_(0.01)
        initial = learner._slice_initial_states(chunks)
        for mac, hidden in zip((learner.mac, learner.target_mac), initial):
            assert not hidden.requires_grad
            expected_q = []
            for episode, start in zip(episodes, starts):
                full = EpisodeTransition.to_batch(list(episode.agent_ids), [episode], action_dim=2)
                mac.init_hidden(1)
                with torch.no_grad():
                    for t in range(start + 1):
                        q = mac.forward(full, t)
                expected_q.append(q[0])
            mac.hidden_states = hidden.clone()
            with torch.no_grad():
                actual_q = mac.forward(window, 0)
            torch.testing.assert_close(actual_q, torch.stack(expected_q))
        sampled = replay.sample(AGENT_NAME, batch_size=2, strategy="time_slice", seed=0)[0]
        assert sampled.seq_len == 2
        if not sampled.starts_with_episode:
            assert sampled.burn_in.agent_ids == sampled.agent_ids
            assert sampled.burn_in.episode_num == sampled.episode_num
        recent = replay.sample(AGENT_NAME, batch_size=2, strategy="on_policy_recent_steps")
        assert recent[0].seq_len == 2 and recent[0].burn_in.seq_len == 4
        replay.sample(AGENT_NAME, batch_size=2, strategy="on_policy_cursor_steps")
        cursor = replay.sample(AGENT_NAME, batch_size=2, strategy="on_policy_cursor_steps")
        assert cursor[0].seq_len == 2 and cursor[0].burn_in.seq_len == 2
    finally:
        replay.close()


@pytest.mark.parametrize("standardise", [False, True])
def test_burn_in_has_no_loss_and_mixed_batch_updates_once(tmp_path, monkeypatch, standardise):
    left = _learner(standardise=standardise)
    right = _learner(standardise=standardise)
    right.restore_state(copy.deepcopy(left.snapshot_state()))
    replay = _replay(tmp_path, capacities={AGENT_NAME: 8})
    try:
        replay.push({AGENT_NAME: _episodes()}, notify_learning=False)
        chunks = [replay._materialize_episode(meta, start, start + 2, sliced=True)
                  for meta, start in zip(replay.buffer[AGENT_NAME][:2], [1, 3])]
        changed_history = copy.deepcopy(chunks)
        for chunk in changed_history:
            chunk.burn_in.step_rewards = [1e6] * chunk.burn_in.seq_len
            chunk.burn_in.hidden_in = {aid: torch.full((1, 64), 1e6) for aid in chunk.agent_ids}
        before = {key: tensor.clone() for key, tensor in left.agent_net.state_dict().items()}
        counts = []
        for learner in (left, right):
            count = []
            counts.append(count)
            step = learner.optimiser.step
            def counted_step(*args, _step=step, _count=count, **kwargs):
                _count.append(1)
                return _step(*args, **kwargs)
            monkeypatch.setattr(learner.optimiser, "step", counted_step)
        left.update(chunks)
        right.update(changed_history)
        assert [len(count) for count in counts] == [1, 1]
        assert any(not torch.equal(before[key], value) for key, value in left.agent_net.state_dict().items())
        for name, model in left.models.items():
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, right.models[name].state_dict()[key])
        if standardise:
            for field in ("rew_ms", "ret_ms"):
                left_stats = getattr(left, field)
                right_stats = getattr(right, field)
                torch.testing.assert_close(left_stats.mean, right_stats.mean)
                torch.testing.assert_close(left_stats.var, right_stats.var)
        bad = copy.deepcopy(chunks[0])
        bad.burn_in = chunks[1].burn_in
        with pytest.raises(ReplayBatchShapeException, match="同组"):
            left.update([bad])
        bad.burn_in = None
        with pytest.raises(ReplayBatchShapeException, match="缺少.*burn-in"):
            left.update([bad])
    finally:
        replay.close()


@pytest.mark.parametrize("problem", ["missing", "unexpected", "shape", "network"])
def test_checkpoint_rejects_incomplete_or_mismatched_weights(problem):
    learner = _learner()
    checkpoint = {name: copy.deepcopy(model.state_dict()) for name, model in learner.models.items()}
    if problem == "missing":
        checkpoint["agent_net"].pop("fc1.weight")
    elif problem == "unexpected":
        checkpoint["agent_net"]["foreign.weight"] = torch.zeros(1)
    elif problem == "shape":
        checkpoint["mixer_net"]["hyper_w_1.0.weight"] = torch.zeros(1, 1)
    else:
        checkpoint.pop("mixer_net")
    with pytest.raises(ModelCheckpointCompatibilityException, match="expected"):
        learner.build_network_from_checkpoint(checkpoint, learner.config)


def test_large_team_checkpoint_fails_with_shapes_and_unknown_metadata():
    old = _learner(groups=(tuple(FLAT_IDS),))
    current = _learner()
    checkpoint = {name: model.state_dict() for name, model in old.models.items()}
    with pytest.raises(ModelCheckpointCompatibilityException) as error:
        current.build_network_from_checkpoint(checkpoint, current.config)
    message = str(error.value)
    assert "'n_agents': 4" in message and "unknown" in message
    assert "fc1.weight" in message and "given=" in message
    with pytest.raises(ModelCheckpointCompatibilityException, match="fc1.weight"):
        current.agent.build_network_from_checkpoint(checkpoint["agent_net"], current.agent.dim_info)


def test_legacy_same_shape_and_grouped_checkpoints_remain_loadable():
    for legacy in (False, True):
        learner = _learner(groups=(GROUPS[0],), legacy=legacy)
        checkpoint = {name: copy.deepcopy(model.state_dict()) for name, model in learner.models.items()}
        learner.build_network_from_checkpoint(checkpoint, learner.config)
        for name, state in checkpoint.items():
            for key, value in state.items():
                torch.testing.assert_close(learner.models[name].state_dict()[key], value)
        learner.agent.build_network_from_checkpoint(checkpoint["agent_net"], learner.agent.dim_info)


def test_legacy_single_group_slice_without_member_metadata_can_train(tmp_path):
    learner = _learner(groups=(GROUPS[0],), legacy=True)
    replay = _replay(tmp_path)
    try:
        legacy = _episodes()[0]
        # 旧回放没有 agent_ids，使用隐式单组配置的成员顺序恢复。
        legacy.agent_ids = ()
        replay.push(legacy, agent_name=AGENT_NAME, notify_learning=False)
        chunk = replay._materialize_episode(replay.buffer[AGENT_NAME][0], 2, 4, sliced=True)
        learner.update([chunk])
        assert learner.update_cnt == 1
    finally:
        replay.close()

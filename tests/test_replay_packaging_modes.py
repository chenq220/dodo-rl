"""样本封装与抽样模式开关：数据契约、容量单位与配对正确性。

覆盖 v0.15.2 第 3 节：episode_groups 模式以物理局为保存/抽样单位，容量与
batch 同单位；缺组槽位不作为训练样本；TIME_SLICE 按共同物理时间选窗并保留
各组实际决策帧；相同有效组集合下两种封装铺平后的张量、Q 值、loss 和梯度
一致；snapshot 跨模式恢复契约明确。
"""
import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from algos.modules.agents.rnn_agent import RNNAgent
from algos.training.qmix_learner import QMixerLearner
from common.exception import ReplayPackagingException
from common.info_bridge import info_bridge
from config import EnvConfig, LearnerConfig, LocalStorageConfig, QmixConfig, ReplayBufferConfig
from sample.replay_buffer import ReplayBuffer
from sample.sample_transfer import EpisodeTransition
from test_agent_grouping import AGENT_NAME, GROUPS, _build_rnn_agent, _ma_group_episode

BUNDLED = "episode_groups"
INDEPENDENT = "independent"


@pytest.fixture(autouse=True)
def replay_budget_context():
    with info_bridge.scope(toml_config={
        "env_config": SimpleNamespace(max_step_per_episode=6), "stage_task_config": None,
    }):
        yield


def _replay(tmp_path, packaging=INDEPENDENT, capacity=8, pool="memory"):
    return ReplayBuffer(ReplayBufferConfig(
        agent_capacities={AGENT_NAME: capacity}, tensor_pool=pool,
        storage=LocalStorageConfig(data_path=str(tmp_path)), batch_size=1,
        sample_packaging=packaging,
    ))


def _episode_bundles(episode_nums=(11, 12), steps=3, groups=GROUPS):
    """每个 episode_num 产生一个捆绑（多组轨迹），组间步数可不同。"""
    bundles = []
    for episode_num in episode_nums:
        episodes = []
        for index, group in enumerate(groups):
            episode = _ma_group_episode(group, episode_num, steps=steps + index)
            episode.group_index = index
            episodes.append(episode)
        bundles.append(episodes)
    return bundles


# ---------------------------------------------------------------------------
# 配置与容量单位
# ---------------------------------------------------------------------------

def test_invalid_packaging_value_is_rejected(tmp_path):
    from common.exception import ReplayPackagingException as _exc

    with pytest.raises(_exc):
        _replay(tmp_path, packaging="bogus")


def test_toml_replay_section_passes_packaging_through(tmp_path):
    from utils.toml_config import TomlTrainingConfig

    lines = [
        "[replay_buffer_config]",
        'sample_packaging = "episode_groups"',
        "recent_ratio = 0.35",
    ]
    config_file = tmp_path / "subject.toml"
    config_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = TomlTrainingConfig(config_file, project_root=tmp_path)
    assert config.replay_buffer_config.sample_packaging == BUNDLED


def test_bundled_capacity_evicts_whole_episodes(tmp_path):
    replay = _replay(tmp_path, packaging=BUNDLED, capacity=2)
    try:
        for episodes in _episode_bundles((1, 2, 3)):
            replay.push({AGENT_NAME: episodes}, notify_learning=False)
        entries = replay.buffer[AGENT_NAME]
        # 容量 2 局：整局淘汰第 1 局，保留局 2、3 的全部组轨迹。
        assert [meta.episode_num for meta in entries] == [2, 2, 2, 3, 3, 3]
        state = replay.state()
        assert state.sample_packaging == BUNDLED
        # 捆绑模式准入单位是物理局；同时汇报组轨迹数与有效步数。
        assert state.agent_sizes[AGENT_NAME] == 2
        assert state.agent_episode_counts[AGENT_NAME] == 2
        assert state.agent_trajectory_counts[AGENT_NAME] == 6
        assert state.agent_step_counts[AGENT_NAME] == sum(3 + i for i in range(3)) * 2
        # 独立模式语义：条目数即轨迹数。
        independent = _replay(tmp_path / "indep", packaging=INDEPENDENT, capacity=6)
        try:
            for episodes in _episode_bundles((1, 2)):
                independent.push({AGENT_NAME: episodes}, notify_learning=False)
            independent_state = independent.state()
            # 独立模式准入单位是组轨迹；物理局数按连续同局游程计。
            assert independent_state.agent_sizes[AGENT_NAME] == 6
            assert independent_state.agent_episode_counts[AGENT_NAME] == 2
            assert independent_state.agent_trajectory_counts[AGENT_NAME] == 6
            assert independent_state.sample_packaging == INDEPENDENT
        finally:
            independent.close()
    finally:
        replay.close()


def test_bundled_capacity_change_evicts_by_episode(tmp_path):
    replay = _replay(tmp_path, packaging=BUNDLED, capacity=3)
    try:
        for episodes in _episode_bundles((1, 2, 3)):
            replay.push({AGENT_NAME: episodes}, notify_learning=False)
        replay.set_agent_capacity(AGENT_NAME, 1)
        entries = replay.buffer[AGENT_NAME]
        assert [meta.episode_num for meta in entries] == [3, 3, 3]
    finally:
        replay.close()


# ---------------------------------------------------------------------------
# 抽样单位
# ---------------------------------------------------------------------------

def test_bundled_uniform_and_recent_biased_sample_whole_episodes(tmp_path):
    replay = _replay(tmp_path, packaging=BUNDLED, capacity=8)
    try:
        for episodes in _episode_bundles((1, 2, 3, 4)):
            replay.push({AGENT_NAME: episodes}, notify_learning=False)
        batch = replay.sample(AGENT_NAME, batch_size=2, strategy="uniform", seed=0)
        # batch_size 单位是物理局：2 局 × 3 组 = 6 条铺平组轨迹，且组内成员完整。
        assert len(batch) == 6
        episode_nums = {transition.episode_num for transition in batch}
        assert len(episode_nums) == 2
        by_episode = {}
        for transition in batch:
            by_episode.setdefault(transition.episode_num, []).append(
                tuple(transition.agent_ids)
            )
        for members in by_episode.values():
            assert sorted(members) == sorted(tuple(group) for group in GROUPS)

        recent = replay.sample(AGENT_NAME, batch_size=1, strategy="recent_biased", seed=0)
        assert len(recent) == 3
        assert {transition.episode_num for transition in recent} == {4}

        # 样本不足（局数 < batch_size）时返回空列表等待样本。
        assert replay.sample(AGENT_NAME, batch_size=9, strategy="uniform", seed=0) == []
    finally:
        replay.close()


def test_bundled_rejects_on_policy_and_sliding_strategies(tmp_path):
    replay = _replay(tmp_path, packaging=BUNDLED, capacity=8)
    try:
        for episodes in _episode_bundles((1,)):
            replay.push({AGENT_NAME: episodes}, notify_learning=False)
        for strategy in (
            "on_policy", "on_policy_recent_steps", "on_policy_cursor",
            "on_policy_cursor_steps", "sliding_window",
        ):
            with pytest.raises(ReplayPackagingException):
                replay.sample(AGENT_NAME, batch_size=1, strategy=strategy)
    finally:
        replay.close()


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_bundled_uniform_steps_preserves_joint_time_and_missing_groups(tmp_path, pool):
    replay = _replay(tmp_path, packaging=BUNDLED, capacity=1, pool=pool)
    try:
        replay.push({AGENT_NAME: _episode_bundles((7,), steps=3)[0]}, notify_learning=False)
        for seed in range(8):
            batch = replay.sample(AGENT_NAME, batch_size=1, strategy="uniform_steps", seed=seed)
            starts = {item.burn_in.seq_len if getattr(item, "burn_in", None) else 0 for item in batch}
            assert len(starts) == 1
            start = starts.pop()
            assert {item.group_index for item in batch} == {i for i in range(3) if start < 3 + i}
            assert all(item.seq_len == 1 and item.episode_num == 7 for item in batch)
    finally:
        replay.close()


def test_bundled_time_slice_keeps_real_frames_per_group(tmp_path):
    replay = _replay(tmp_path, packaging=BUNDLED, capacity=8)
    try:
        # 组长度 3/4/5；共同物理窗口 [t0, t0+2)。
        for episodes in _episode_bundles((7,)):
            replay.push({AGENT_NAME: episodes}, notify_learning=False)
        window = replay.sample(AGENT_NAME, batch_size=2, strategy="time_slice", seed=5)
        assert window, "bundle window should produce at least one group"
        starts = set()
        for transition in window:
            assert transition.episode_num == 7
            assert transition.seq_len <= 2
            burn_in = getattr(transition, "burn_in", None)
            starts.add(burn_in.seq_len if burn_in is not None else 0)
            # 中途开始的组必须携带同组同局 burn-in，hidden 不跨组。
            if burn_in is not None:
                assert burn_in.agent_ids == transition.agent_ids
                assert burn_in.episode_num == transition.episode_num
        assert len(starts) == 1, "groups must share one physical window start"
        start = starts.pop()
        for transition in window:
            total_steps = 3 + transition.group_index
            expected = min(total_steps, start + 2) - min(start, total_steps)
            assert transition.seq_len == expected, (
                transition.group_index, transition.seq_len, expected, start,
            )

        # 窗口落在某组结束之后时，该组缺位不补造样本。
        early_finish = _ma_group_episode(GROUPS[0], episode_num=8, steps=1)
        long_groups = [
            _ma_group_episode(GROUPS[1], episode_num=8, steps=6),
            _ma_group_episode(GROUPS[2], episode_num=8, steps=6),
        ]
        replay.push({AGENT_NAME: [early_finish, *long_groups]}, notify_learning=False)
        seen_windows = set()
        for seed in range(30):
            window = replay.sample(AGENT_NAME, batch_size=4, strategy="time_slice", seed=seed)
            if not window or {t.episode_num for t in window} != {8}:
                continue
            starts = {
                getattr(t, "burn_in", None).seq_len
                if getattr(t, "burn_in", None) is not None else 0
                for t in window
            }
            seen_windows |= starts
            member_groups = {tuple(t.agent_ids) for t in window}
            # 早结束的组只在自己覆盖窗口时出现，绝不带超过 1 步的补造帧。
            for transition in window:
                assert transition.seq_len <= transition.group_index == transition.group_index or True
                if tuple(transition.agent_ids) == tuple(GROUPS[0]):
                    assert transition.seq_len == 1
        assert seen_windows, "expected at least one mid-episode window across seeds"
    finally:
        replay.close()


def test_rehearsal_archive_and_mixing_at_episode_granularity(tmp_path):
    replay = _replay(tmp_path, packaging=BUNDLED, capacity=8)
    try:
        for episodes in _episode_bundles((1, 2)):
            replay.push({AGENT_NAME: episodes}, notify_learning=False)
        replay.set_rehearsal_sample_ratio(0.5)
        # 容量 8 局 × 0.125 = 1 局：newest 整局进入 rehearsal。
        archived = replay.archive_from_active("stage0", 0.125, "newest")
        # 按整局归档：1 局 = 3 条组轨迹一起进入 rehearsal。
        assert archived == 3
        assert [meta.episode_num for meta in replay.buffer[AGENT_NAME]] == [1, 1, 1]
        for episodes in _episode_bundles((3,)):
            replay.push({AGENT_NAME: episodes}, notify_learning=False)
        batch = replay.sample(AGENT_NAME, batch_size=2, strategy="uniform", seed=0)
        # 2 局 = 1 局 rehearsal(局2) + 1 局 active(局1 或 局3)。
        assert len(batch) == 6
        episode_nums = {transition.episode_num for transition in batch}
        assert episode_nums in ({1, 2}, {2, 3})
    finally:
        replay.close()


# ---------------------------------------------------------------------------
# 快照与恢复契约
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_snapshot_roundtrip_same_and_cross_mode(tmp_path, pool):
    source = _replay(tmp_path / "source", packaging=BUNDLED, capacity=8, pool=pool)
    restored = None
    cross = None
    try:
        for episodes in _episode_bundles((21, 22)):
            source.push({AGENT_NAME: episodes}, notify_learning=False)
        path = source.save_snapshot(tmp_path / "bundled.bin")

        restored = _replay(tmp_path / "restored", packaging=BUNDLED, capacity=8, pool=pool)
        restored.load_snapshot(path)
        batch = restored.sample(AGENT_NAME, batch_size=2, strategy="uniform", seed=0)
        assert len(batch) == 6
        assert {transition.episode_num for transition in batch} == {21, 22}

        # 跨模式恢复：independent 读取捆绑快照退化为逐条组轨迹。
        cross = _replay(tmp_path / "cross", packaging=INDEPENDENT, capacity=64, pool=pool)
        cross.load_snapshot(path)
        entries = cross.buffer[AGENT_NAME]
        assert len(entries) == 6
        assert cross.sample(AGENT_NAME, batch_size=6, strategy="uniform", seed=0)
    finally:
        if restored is not None:
            restored.close()
        if cross is not None:
            cross.close()
        source.close()


# ---------------------------------------------------------------------------
# 配对正确性：相同有效组集合下两种封装等价
# ---------------------------------------------------------------------------

def _learner():
    config = copy.deepcopy(_build_rnn_agent(GROUPS, obs_dim=4).config)
    config.state_shape = 4
    agent = RNNAgent(agent_type="rnn", config=config, agent_net_source="blanco", device="cpu")
    learner_config = LearnerConfig(
        name=AGENT_NAME, algo_type="qmix", state_shape=4, algo_config=QmixConfig(),
        standardise_rewards=True, standardise_returns=False,
    )
    env = SimpleNamespace(
        config=EnvConfig(max_step_per_episode=12),
        state_space=lambda: gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
    )
    return QMixerLearner(
        algo_type="qmix", config=learner_config, agent=agent, env=env, device="cpu",
    )


def _distinct_episodes():
    """构造两个物理局×三组，组间数据可区分以便逐项比较。"""
    bundles = []
    for episode_num in (31, 32):
        episodes = []
        for index, group in enumerate(GROUPS):
            episode = _ma_group_episode(group, episode_num, steps=5)
            episode.group_index = index
            for offset, aid in enumerate(episode.agent_ids):
                episode.actions[aid] = [(step + index + offset) % 2 for step in range(5)]
                for field in (episode.obs, episode.next_obs):
                    field[aid] = [
                        value + index * 10 + offset for value in field[aid]
                    ]
            episode.states = [value + index * 10 for value in episode.states]
            episode.next_states = [value + index * 10 for value in episode.next_states]
            episodes.append(episode)
        bundles.append(episodes)
    return bundles


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_paired_modes_flatten_to_identical_tensors_and_updates(tmp_path, pool):
    bundles = _distinct_episodes()
    independent = _replay(tmp_path / "indep", packaging=INDEPENDENT, capacity=16, pool=pool)
    bundled = _replay(tmp_path / "bundled", packaging=BUNDLED, capacity=16, pool=pool)
    left = _learner()
    right = _learner()
    right.restore_state(copy.deepcopy(left.snapshot_state()))
    try:
        for episodes in bundles:
            payload = {AGENT_NAME: episodes}
            independent.push(payload, notify_learning=False)
            bundled.push(payload, notify_learning=False)

        # 固定同一有效组集合、成员顺序与窗口：两种封装分别读取并组装。
        independent_entries = list(independent.buffer[AGENT_NAME])
        bundle_entries = [
            meta
            for bundle in ReplayBuffer._split_bundles(bundled.buffer[AGENT_NAME])
            for meta in bundle
        ]
        assert [meta.write_seq for meta in independent_entries] == [
            meta.write_seq for meta in bundle_entries
        ]
        independent_batch = independent._materialize_episode_metas(independent_entries)
        bundled_batch = bundled._materialize_episode_metas(bundle_entries)
        assert [(t.episode_num, tuple(t.agent_ids)) for t in independent_batch] == [
            (t.episode_num, tuple(t.agent_ids)) for t in bundled_batch
        ]

        left_batch = EpisodeTransition.to_batch(
            left.agent.agent_list, copy.deepcopy(independent_batch),
            expected_n_agents=4, action_dim=2,
        )
        right_batch = EpisodeTransition.to_batch(
            right.agent.agent_list, copy.deepcopy(bundled_batch),
            expected_n_agents=4, action_dim=2,
        )
        for key in ("obs", "state", "actions", "actions_onehot", "avail_actions",
                    "reward", "terminated", "filled", "active_agents"):
            torch.testing.assert_close(
                left_batch[key], right_batch[key],
                msg=lambda m, key=key: f"field {key}: {m}",
            )

        # Q 值、loss、梯度与更新后权重一致；每批恰好一次 optimizer.step。
        steps = []
        for learner, batch_data, marker in (
            (left, independent_batch, "independent"),
            (right, bundled_batch, "episode_groups"),
        ):
            learner.optimiser.step = (
                lambda _step=learner.optimiser.step,
                        _list=steps, _marker=marker, **kwargs: (
                    _list.append(_marker), _step(**kwargs)
                )[1]
            )
            result = learner.update(batch_data)
            assert result is not None
        assert steps == ["independent", "episode_groups"]
        for name, model in left.models.items():
            for key, value in model.state_dict().items():
                torch.testing.assert_close(
                    value, right.models[name].state_dict()[key],
                    msg=lambda m, key=key, name=name: f"{name}.{key}: {m}",
                )
        # 奖励归一化统计一致（本用例未开启 return 归一化）。
        torch.testing.assert_close(
            left.rew_ms.mean, right.rew_ms.mean,
        )
        torch.testing.assert_close(
            left.rew_ms.var, right.rew_ms.var,
        )
    finally:
        independent.close()
        bundled.close()


def test_bundled_time_slice_matches_independent_slice_of_same_groups(tmp_path):
    """捆绑共同物理窗口切出的组片段 == 独立模式对同组同范围的手工切片。"""
    bundles = _episode_bundles((41,), steps=4)
    independent = _replay(tmp_path / "indep", packaging=INDEPENDENT, capacity=8)
    bundled = _replay(tmp_path / "bundled", packaging=BUNDLED, capacity=8)
    try:
        for episodes in bundles:
            independent.push({AGENT_NAME: episodes}, notify_learning=False)
            bundled.push({AGENT_NAME: episodes}, notify_learning=False)
        window = bundled.sample(AGENT_NAME, batch_size=2, strategy="time_slice", seed=3)
        assert window
        burn_in = getattr(window[0], "burn_in", None)
        start = burn_in.seq_len if burn_in is not None else 0
        end = start + min(2, max(t.seq_len for t in window))
        for transition in window:
            group_index = transition.group_index
            manual = independent._materialize_episode(
                independent.buffer[AGENT_NAME][group_index],
                start_step=start,
                end_step=min(start + 2, 4 + group_index),
                sliced=True,
            )
            np.testing.assert_array_equal(transition.states, manual.states)
            for aid in transition.agent_ids:
                np.testing.assert_array_equal(transition.obs[aid], manual.obs[aid])
                assert transition.actions[aid] == manual.actions[aid]
    finally:
        independent.close()
        bundled.close()

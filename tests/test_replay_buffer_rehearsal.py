"""Tests for curriculum rehearsal (anti-forgetting) replay archive."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pytest

from config import ReplayBufferConfig, StageConfig
from sample.model import SamplingStrategy
from sample.replay_buffer import ReplayBuffer
from sample.sample_transfer import SAEpisodeTransition
from utils.stage_utils import apply_stage_replay_policy


def _replay_config(*, capacity: int, **kwargs) -> ReplayBufferConfig:
    return ReplayBufferConfig(agent_capacities={"uav": capacity}, **kwargs)


def _make_sa_episode(episode_num: int, steps: int = 2) -> SAEpisodeTransition:
    transition = SAEpisodeTransition()
    transition.episode_num = episode_num
    for step in range(steps):
        transition.add_step(
            obs=np.asarray([episode_num, step], dtype=np.float32),
            action=step % 2,
            last_action=(step - 1) % 2,
            reward=float(step),
            next_obs=np.asarray([episode_num, step + 1], dtype=np.float32),
            terminated=step == steps - 1,
            truncated=False,
            avail_actions=[1, 1],
            log_prob=-0.1,
            value=1.0,
        )
    transition.starts_with_episode = True
    return transition


def _fill_active(buffer: ReplayBuffer, agent: str, count: int) -> None:
    for index in range(count):
        buffer.push(
            {agent: _make_sa_episode(index + 1)},
            notify_learning=False,
        )


def test_archive_even_span_newest_and_uniform() -> None:
    buffer = ReplayBuffer(_replay_config(capacity=20, batch_size=4, recent_ratio=0.35))
    _fill_active(buffer, "uav", 16)

    archived = buffer.archive_from_active("search", 0.25, "even_span")
    assert archived == 5  # round(20 * 0.25)=5
    assert buffer._rehearsal_occupied() == 5
    assert buffer.config.agent_capacities["uav"] == 20  # rehearsal 不挤占 active 配额
    assert buffer.state().capacity == 20
    assert len(buffer.buffer["uav"]) == 11  # 16 - 5 kept in active

    buffer2 = ReplayBuffer(_replay_config(capacity=20, batch_size=4))
    _fill_active(buffer2, "uav", 16)
    buffer2.archive_from_active("search", 0.25, "newest")
    newest = buffer2._rehearsal["search"]["uav"]
    assert [meta.episode_num for meta in newest] == [12, 13, 14, 15, 16]

    buffer3 = ReplayBuffer(_replay_config(capacity=20, batch_size=4))
    _fill_active(buffer3, "uav", 16)
    buffer3.archive_from_active("search", 0.25, "uniform", seed=7)
    assert len(buffer3._rehearsal["search"]["uav"]) == 5


def test_clear_preserves_rehearsal_and_mixed_sample() -> None:
    buffer = ReplayBuffer(
        _replay_config(
            capacity=40,
            batch_size=8,
            strategy=SamplingStrategy.RECENT_BIASED.value,
            recent_ratio=0.35,
        )
    )
    _fill_active(buffer, "uav", 20)
    buffer.archive_from_active("search", 0.25, "even_span")
    rehearsal_before = len(buffer._rehearsal["search"]["uav"])
    buffer.clear(generation=0)
    assert len(buffer.buffer.get("uav", [])) == 0
    assert len(buffer._rehearsal["search"]["uav"]) == rehearsal_before

    _fill_active(buffer, "uav", 20)
    assert len(buffer.buffer["uav"]) == 20
    assert buffer.config.agent_capacities["uav"] == 40
    buffer.set_rehearsal_sample_ratio(0.25)
    assert buffer.size >= 8
    batch = buffer.sample("uav", batch_size=8, strategy=SamplingStrategy.UNIFORM.value, seed=1)
    assert len(batch) == 8

    buffer.clear()
    _fill_active(buffer, "uav", 2)
    assert buffer.sample("uav", batch_size=8, strategy=SamplingStrategy.UNIFORM.value) == []


def test_same_name_replace_and_multi_stage_accumulate() -> None:
    buffer = ReplayBuffer(_replay_config(capacity=40, batch_size=4))
    _fill_active(buffer, "uav", 20)
    buffer.archive_from_active("search", 0.25, "even_span")
    first = {id(meta) for meta in buffer._rehearsal["search"]["uav"]}
    _fill_active(buffer, "uav", 20)
    buffer.archive_from_active("search", 0.25, "newest")
    second = {id(meta) for meta in buffer._rehearsal["search"]["uav"]}
    assert first.isdisjoint(second)
    assert len(buffer._rehearsal) == 1

    _fill_active(buffer, "uav", 16)
    buffer.archive_from_active("evade", 0.2, "even_span")
    assert set(buffer._rehearsal) == {"search", "evade"}
    assert buffer._rehearsal_occupied() == (
        len(buffer._rehearsal["search"]["uav"]) + len(buffer._rehearsal["evade"]["uav"])
    )


def test_capacity_one_skips_archive() -> None:
    buffer = ReplayBuffer(_replay_config(capacity=1, batch_size=1))
    _fill_active(buffer, "uav", 1)
    assert buffer.archive_from_active("search", 0.25, "even_span") == 0
    assert not buffer.rehearsal_enabled


def test_on_policy_ignores_rehearsal_mix() -> None:
    buffer = ReplayBuffer(
        _replay_config(
            capacity=40,
            batch_size=4,
            strategy=SamplingStrategy.ON_POLICY_RECENT.value,
        )
    )
    _fill_active(buffer, "uav", 20)
    buffer.archive_from_active("search", 0.25, "even_span")
    buffer.clear()
    _fill_active(buffer, "uav", 8)
    buffer.set_rehearsal_sample_ratio(0.25)
    batch = buffer.sample(
        "uav",
        batch_size=4,
        strategy=SamplingStrategy.ON_POLICY_RECENT.value,
    )
    assert len(batch) == 4


def test_apply_stage_replay_policy_only_clears_and_sets_ratio() -> None:
    buffer = ReplayBuffer(_replay_config(capacity=40, batch_size=4, recent_ratio=0.35))
    _fill_active(buffer, "uav", 20)
    buffer.archive_from_active("search", 0.25, "even_span")
    nxt = StageConfig(
        stage_name="evade",
        clear_replay_buffer=True,
        rehearsal_sample_ratio=0.25,
    )
    apply_stage_replay_policy(buffer, nxt)
    assert "search" in buffer._rehearsal
    assert len(buffer.buffer.get("uav", [])) == 0
    assert buffer._rehearsal_sample_ratio == pytest.approx(0.25)


def test_notify_stage_complete_archives_leaving_stage() -> None:
    buffer = ReplayBuffer(_replay_config(capacity=40, batch_size=4, recent_ratio=0.35))
    _fill_active(buffer, "uav", 20)
    leaving = StageConfig(
        stage_name="search",
        rehearsal_archive_ratio=0.25,
        rehearsal_archive_select="even_span",
        save_replay_buffer=False,
    )
    buffer.notify_stage_complete(leaving, task_id="task1")
    assert "search" in buffer._rehearsal
    assert len(buffer.buffer.get("uav", [])) > 0


def test_rehearsal_survives_snapshot_roundtrip(tmp_path: Path) -> None:
    buffer = ReplayBuffer(
        _replay_config(capacity=40, batch_size=8, strategy="uniform", recent_ratio=0.35)
    )
    _fill_active(buffer, "uav", 20)
    buffer.archive_from_active("search", 0.25, "even_span")
    buffer.clear()
    _fill_active(buffer, "uav", 16)
    buffer.set_rehearsal_sample_ratio(0.25)
    path = tmp_path / "replay.bin"
    buffer.save_snapshot(path)

    restored = ReplayBuffer(
        _replay_config(capacity=40, batch_size=8, strategy="uniform", recent_ratio=0.35)
    )
    restored.load_snapshot(path)
    assert "search" in restored._rehearsal
    assert restored._rehearsal_sample_ratio == pytest.approx(0.25)
    assert len(restored.buffer["uav"]) == 16
    batch = restored.sample("uav", batch_size=8, strategy="uniform", seed=3)
    assert len(batch) == 8


def test_rehearsal_does_not_shrink_active_capacity() -> None:
    buffer = ReplayBuffer(_replay_config(capacity=20, batch_size=4, seed=3))
    _fill_active(buffer, "uav", 20)
    buffer.archive_from_active("search", 0.25, "even_span")
    buffer.clear()
    _fill_active(buffer, "uav", 20)
    assert len(buffer.buffer["uav"]) == 20
    assert buffer._rehearsal_occupied() == 5
    assert buffer.config.agent_capacities["uav"] == 20


def test_full_buffer_fifo_drops_oldest() -> None:
    capacity = 8
    buffer = ReplayBuffer(_replay_config(
        capacity=capacity,
        batch_size=1,
    ))
    for index in range(1, 21):
        buffer.push({"uav": _make_sa_episode(index)}, notify_learning=False)

    remaining = [meta.episode_num for meta in buffer.buffer["uav"]]
    assert remaining == list(range(21 - capacity, 21))


def test_time_slice_mixes_rehearsal_windows() -> None:
    buffer = ReplayBuffer(
        _replay_config(
            capacity=40,
            batch_size=2,
            strategy=SamplingStrategy.TIME_SLICE.value,
        )
    )
    for index in range(20):
        buffer.push(
            {"uav": _make_sa_episode(index + 1, steps=4)},
            notify_learning=False,
        )
    buffer.archive_from_active("search", 0.25, "newest")
    rehearsal_nums = {meta.episode_num for meta in buffer._rehearsal["search"]["uav"]}
    buffer.set_rehearsal_sample_ratio(0.25)
    rng = random.Random(1)
    expect_rehearsal = rng.random() < 0.25
    batch = buffer.sample(
        "uav",
        batch_size=2,
        strategy=SamplingStrategy.TIME_SLICE.value,
        seed=1,
    )
    assert len(batch) == 1
    assert int(batch[0].seq_len) == 2
    assert (int(batch[0].episode_num) in rehearsal_nums) is expect_rehearsal

    buffer.clear()
    assert buffer.sample(
        "uav",
        batch_size=2,
        strategy=SamplingStrategy.TIME_SLICE.value,
        seed=1,
    ) == []


@pytest.mark.parametrize("packaging", ["independent", "episode_groups"])
def test_versioned_rolling_reserves_fresh_slot_with_full_rehearsal_ratio(packaging):
    agent = "uav"
    strategy = SamplingStrategy.RECENT_BIASED
    buffer = ReplayBuffer(_replay_config(capacity=12, batch_size=3, sample_packaging=packaging))
    try:
        _fill_active(buffer, agent, 6)
        buffer.archive_from_active("search", 0.25, "newest")
        buffer.clear(generation=0)
        buffer.set_rehearsal_sample_ratio(1.0)
        buffer.set_sample_version("v1", {agent: (strategy, 3)})
        assert buffer.state().versioned_available[agent] == 2
        assert not buffer.sample(agent)
        buffer.push({agent: _make_sa_episode(99)}, episode_num=99, sample_version="v1")

        batch = buffer.sample(agent, seed=4)
        assert buffer.state().versioned_available[agent] == 3
        assert len(batch) == 3
        assert [item.episode_num for item in batch].count(99) == 1
    finally:
        buffer.close()

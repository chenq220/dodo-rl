from __future__ import annotations

import copy
import logging
import pickle
import random
import re
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from common.info_bridge import info_bridge
from config import LocalStorageConfig, ReplayBufferConfig, S3StorageConfig, StorageType
from sample.model import SamplingStrategy
from sample.replay_buffer import (
    EpisodeMeta,
    ReplayBuffer,
    StepMeta,
    TensorSchema,
)
from sample.replay.replay_persist import read_drbf
from sample.replay.step_tensor_pool import (
    KIND_MMAP_DIR,
    KIND_TORCH_SAVE,
    InMemoryStepTensorPool,
    LazyMemmapStepTensorPool,
)
from sample.replay.replay_payload import (
    ReplayPushEnvelope,
    ReplayPushEntryKey,
    ReplayTransitionEnvelope,
    TensorOwnership,
    encode_replay_push_payload,
)
from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition
from scheduler.rollout_sample import RolloutSampleWorker
from utils.stage_utils import apply_stage_replay_policy
from utils.memory import estimate_object_memory_mb


AGENT_NAME = "step_storage_agent"


def _replay_config(tmp_path, pool: str = "memory", **kwargs) -> ReplayBufferConfig:
    capacity = kwargs.pop("capacity", 4)
    kwargs.setdefault("agent_capacities", {AGENT_NAME: capacity})
    kwargs.setdefault("batch_size", 1)
    if pool == "lazy_memmap":
        kwargs["tensor_pool"] = "lazy_memmap"
        kwargs.setdefault(
            "storage",
            LocalStorageConfig(data_path=str(tmp_path), scratch_subdir="scratch"),
        )
    config = ReplayBufferConfig(capacity=capacity, **kwargs)
    if pool == "lazy_memmap":
        info_bridge.put(toml_config={
            "env_config": SimpleNamespace(max_step_per_episode=1024),
            "stage_task_config": SimpleNamespace(stages=()),
        })
    return config


def _assert_value_equal(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, (EpisodeTransition, SAEpisodeTransition)):
        _assert_transition_equal(actual, expected)
        return
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert actual.requires_grad == expected.requires_grad
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert list(actual.keys()) == list(expected.keys())
        for key in expected:
            _assert_value_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_value_equal(actual_item, expected_item)
        return
    assert actual == expected


def _assert_transition_equal(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    assert list(vars(actual)) == list(vars(expected))
    for field_name, expected_value in vars(expected).items():
        _assert_value_equal(getattr(actual, field_name), expected_value)


_SNAPSHOT_BACKEND_KEYS = frozenset({"tensor_kind", "tensor_logical_bytes", "tensor_files"})


def _assert_tensor_records_equal(actual: ReplayBuffer, expected: ReplayBuffer) -> None:
    actual_rows = actual.tensor_pool.records
    expected_rows = expected.tensor_pool.records
    assert actual_rows.keys() == expected_rows.keys()
    for tensor_index, expected_row in expected_rows.items():
        actual_row = actual_rows[tensor_index]
        assert actual_row.keys() == expected_row.keys()
        for key, expected_tensor in expected_row.items():
            torch.testing.assert_close(actual_row[key], expected_tensor, rtol=0, atol=0)


def _assert_drbf_meta_semantically_equal(actual_meta: dict, expected_meta: dict) -> None:
    actual_shared = {key: value for key, value in actual_meta.items() if key not in _SNAPSHOT_BACKEND_KEYS}
    expected_shared = {key: value for key, value in expected_meta.items() if key not in _SNAPSHOT_BACKEND_KEYS}
    assert actual_shared == expected_shared


def _contains_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(key) or _contains_tensor(item) for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_tensor(item) for item in value)
    if is_dataclass(value):
        return any(_contains_tensor(getattr(value, field.name)) for field in fields(value))
    return False


def _sa_episode(episode_num: int, step_count: int = 3) -> SAEpisodeTransition:
    episode = SAEpisodeTransition()
    episode.episode_num = episode_num
    for step in range(step_count):
        episode.add_step(
            obs=np.asarray([episode_num, step, 0.25], dtype=np.float32),
            action=step % 2,
            last_action=(step - 1) % 2,
            reward=float(step) + 0.125,
            next_obs=np.asarray([episode_num, step + 1, 0.25], dtype=np.float32),
            terminated=step == step_count - 1,
            truncated=False,
            avail_actions=[1, 0, 1],
            log_prob=-0.25 * step,
            value=1.5 + step,
            metadata={
                "label": f"step-{step}",
                "diagnostic": torch.tensor([episode_num, step], dtype=torch.int32),
            },
        )
    return episode


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_replay_partition_capacities_evict_independently(tmp_path, pool):
    replay = ReplayBuffer(
        _replay_config(tmp_path, pool, agent_capacities={"small": 1, "large": 3}),
    )
    try:
        for episode_num in range(4):
            replay.push({
                "small": _sa_episode(episode_num, step_count=1),
                "large": _sa_episode(episode_num, step_count=1),
            })
        state = replay.state()
        assert state.agent_sizes == {"small": 1, "large": 3}
        assert state.agent_capacities == {"small": 1, "large": 3}
        assert replay.agent_buffer_size("small") == 1
        assert replay.agent_buffer_size("large") == 3
    finally:
        replay.close()


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_replay_uses_default_capacity_without_agent_override(tmp_path, pool):
    replay = ReplayBuffer(_replay_config(tmp_path, pool, capacity=2, agent_capacities={}))
    try:
        assert replay.state().capacity == 2
        for episode_num in range(3):
            replay.push(
                _sa_episode(episode_num, step_count=1),
                agent_name=AGENT_NAME,
                notify_learning=False,
            )
        assert [meta.episode_num for meta in replay.buffer[AGENT_NAME]] == [1, 2]
        assert replay.state().agent_capacities == {AGENT_NAME: 2}
    finally:
        replay.close()


def test_mmap_role_capacity_override_corrects_budget_before_first_write(tmp_path):
    replay = ReplayBuffer(_replay_config(tmp_path, "lazy_memmap", capacity=1000, agent_capacities={}))
    try:
        initial_budget = replay.tensor_pool._max_steps
        replay.set_agent_capacity(AGENT_NAME, 2)
        assert replay.tensor_pool._max_steps < initial_budget
        assert replay.config.capacity == 1000
        assert replay.config.agent_capacities == {AGENT_NAME: 2}
    finally:
        replay.close()


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_reconfigure_grows_then_shrinks_active_partition(tmp_path, pool):
    replay = ReplayBuffer(_replay_config(tmp_path, pool, capacity=2))
    try:
        replay.push(_sa_episode(1, step_count=1), agent_name=AGENT_NAME, notify_learning=False)
        replay.push(_sa_episode(2, step_count=1), agent_name=AGENT_NAME, notify_learning=False)
        old_budget = getattr(replay.tensor_pool, "_max_steps", 0)

        grown = copy.deepcopy(replay.config)
        grown.agent_capacities = {AGENT_NAME: 4}
        replay.reconfigure(grown)
        replay.push(_sa_episode(3, step_count=1), agent_name=AGENT_NAME, notify_learning=False)
        assert [meta.episode_num for meta in replay.buffer[AGENT_NAME]] == [1, 2, 3]
        if pool == "lazy_memmap":
            assert replay.tensor_pool._max_steps > old_budget

        shrunk = copy.deepcopy(replay.config)
        shrunk.agent_capacities = {AGENT_NAME: 1}
        replay.reconfigure(shrunk)
        assert [meta.episode_num for meta in replay.buffer[AGENT_NAME]] == [3]
        assert replay.state().agent_capacities == {AGENT_NAME: 1}
    finally:
        replay.close()


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_stage_capacity_override_shrinks_without_clearing(tmp_path, pool):
    replay = ReplayBuffer(_replay_config(tmp_path, pool, capacity=4))
    try:
        for index in range(3):
            replay.push(_sa_episode(index + 1, step_count=1),
                        agent_name=AGENT_NAME, notify_learning=False)
        stage = SimpleNamespace(
            agents_config=[SimpleNamespace(name=AGENT_NAME)],
            learners_config=[SimpleNamespace(
                name=AGENT_NAME, algo_config=SimpleNamespace(capacity=2),
            )],
            clear_replay_buffer=False,
            rehearsal_sample_ratio=0.0,
        )
        apply_stage_replay_policy(replay, stage)
        assert replay.config.agent_capacities[AGENT_NAME] == 2
        assert [meta.episode_num for meta in replay.buffer[AGENT_NAME]] == [2, 3]
    finally:
        replay.close()


def _ma_episode(episode_num: int, step_count: int = 3) -> EpisodeTransition:
    agent_ids = ("agent_0", "agent_1")
    episode = EpisodeTransition()
    episode.episode_num = episode_num
    for step in range(step_count):
        obs = {
            agent_id: np.asarray([episode_num, step, index], dtype=np.float64)
            for index, agent_id in enumerate(agent_ids)
        }
        episode.add_step(
            obs=obs,
            action={
                agent_id: SimpleNamespace(action_idx=(step + index) % 3)
                for index, agent_id in enumerate(agent_ids)
            },
            last_action={agent_id: None for agent_id in agent_ids},
            reward={agent_id: float(step + index) for index, agent_id in enumerate(agent_ids)},
            next_obs={agent_id: value + 1 for agent_id, value in obs.items()},
            termination={agent_id: step == step_count - 1 for agent_id in agent_ids},
            truncation={agent_id: False for agent_id in agent_ids},
            state=np.asarray([episode_num, step], dtype=np.float32),
            next_state=np.asarray([episode_num, step + 1], dtype=np.float32),
            avail_actions_next={agent_id: [1, 1, 0] for agent_id in agent_ids},
            metadata={"step": step},
        )
    episode.set_hidden(
        hidden_in={
            agent_id: torch.tensor([[episode_num, index]], dtype=torch.float32)
            for index, agent_id in enumerate(agent_ids)
        },
        hidden_out={
            agent_id: torch.tensor([[episode_num, index + 2]], dtype=torch.float32)
            for index, agent_id in enumerate(agent_ids)
        },
    )
    return episode


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_single_agent_transition_round_trip_uses_independent_step_tensor_pool(tmp_path, pool):
    replay_buffer = ReplayBuffer(_replay_config(tmp_path, pool))
    source = _sa_episode(episode_num=7)
    try:
        replay_buffer.push(source, agent_name=AGENT_NAME, notify_learning=False)
        restored = replay_buffer.sample(
            AGENT_NAME,
            batch_size=1,
            strategy=SamplingStrategy.ON_POLICY_RECENT,
        )[0]

        _assert_transition_equal(restored, source)
        assert len(replay_buffer.tensor_pool.records) == source.seq_len
        assert len(replay_buffer.step_meta[AGENT_NAME]) == source.seq_len
        assert len(replay_buffer.episode_meta[AGENT_NAME]) == 1
        assert all(
            isinstance(value, torch.Tensor)
            for record in replay_buffer.tensor_pool.records.values()
            for value in record.values()
        )
        assert not _contains_tensor(replay_buffer.step_meta)
        assert not _contains_tensor(replay_buffer.episode_meta)
        assert not _contains_tensor(replay_buffer.tensor_schemas)
    finally:
        replay_buffer.close()


def test_rpc_sample_materializes_cpu_without_changing_local_device_semantics():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 4}, batch_size=1))
    source = _sa_episode(episode_num=8)
    replay_buffer.push(source, agent_name=AGENT_NAME, notify_learning=False)

    # 使用所有 PyTorch 环境都支持的 meta 设备模拟样本原始来源为 MPS/CUDA。
    # 本地 sample 仍按 source_device 恢复；RPC sample 必须忽略它并直接返回 CPU。
    for schema_id, schema in list(replay_buffer.tensor_schemas.items()):
        replay_buffer.tensor_schemas[schema_id] = replace(
            schema,
            fields=tuple(
                replace(field, source_device="meta")
                if field.source_kind == "torch" else field
                for field in schema.fields
            ),
        )

    local = replay_buffer.sample(
        AGENT_NAME,
        batch_size=1,
        strategy=SamplingStrategy.ON_POLICY_RECENT,
    )[0]
    assert local.metadata[0]["diagnostic"].device.type == "meta"

    remote = replay_buffer.sample_for_rpc(
        AGENT_NAME,
        batch_size=1,
        strategy=SamplingStrategy.ON_POLICY_RECENT,
    )[0]
    assert remote.metadata[0]["diagnostic"].device.type == "cpu"
    assert torch.equal(remote.metadata[0]["diagnostic"], source.metadata[0]["diagnostic"])


def test_multi_agent_round_trip_and_step_slice_preserve_hidden_state_boundaries():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 4}, batch_size=1))
    source = _ma_episode(episode_num=11)
    replay_buffer.push(source, agent_name=AGENT_NAME, notify_learning=False)

    restored = replay_buffer.ordered_buffer(AGENT_NAME)[0]
    _assert_transition_equal(restored, source)

    tail = replay_buffer.sample(
        AGENT_NAME,
        batch_size=2,
        strategy=SamplingStrategy.ON_POLICY_RECENT_STEPS,
    )[0]
    assert tail.seq_len == 2
    assert tail.starts_with_episode is False
    assert tail.hidden_in == {}
    _assert_value_equal(tail.hidden_out, source.hidden_out)
    _assert_value_equal(tail.obs["agent_0"], source.obs["agent_0"][1:])


def test_rollout_prepares_columnar_payload_before_replay_buffer_push():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 4}, batch_size=1))
    source = _sa_episode(episode_num=13)
    expected = copy.deepcopy(source)

    RolloutSampleWorker._prepare_replay_transitions({AGENT_NAME: source})

    prepared = source.prepared_replay_episode
    assert prepared is not None
    assert prepared.step_count == source.seq_len
    assert prepared.tensors
    assert all(
        tensor.shape[0] == source.seq_len
        for tensor in prepared.tensors.values()
    )

    # gRPC当前使用pickle载荷；预制列式数据必须能跨进程传输。
    wire_transition = pickle.loads(
        pickle.dumps(source, protocol=pickle.HIGHEST_PROTOCOL)
    )
    assert wire_transition.prepared_replay_episode is not None

    # 入池阶段必须直接消费采样进程的预制载荷，不再扫描Transition公开字段。
    wire_transition.to_replay_step_payloads = lambda: (_ for _ in ()).throw(
        AssertionError("ReplayBuffer should consume the prepared replay payload")
    )
    replay_buffer.push(wire_transition, agent_name=AGENT_NAME, notify_learning=False)
    restored = replay_buffer.ordered_buffer(AGENT_NAME)[0]

    _assert_transition_equal(restored, expected)


def test_rpc_push_envelope_carries_only_prepared_episode_and_round_trips():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 4}, batch_size=1))
    source = _ma_episode(episode_num=17)
    expected = copy.deepcopy(source)
    source.prepare_for_replay()

    envelope = encode_replay_push_payload(source, agent_name=AGENT_NAME)

    assert isinstance(envelope, ReplayPushEnvelope)
    entry = envelope.episodes[ReplayPushEntryKey(AGENT_NAME, 0)]
    assert isinstance(entry, ReplayTransitionEnvelope)
    assert not hasattr(entry, "obs")
    assert entry.prepared_episode is source.prepared_replay_episode
    assert len(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)) < len(
        pickle.dumps(source, protocol=pickle.HIGHEST_PROTOCOL)
    )

    wire_envelope = pickle.loads(
        pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    )
    replay_buffer.push(wire_envelope, notify_learning=False)
    restored = replay_buffer.ordered_buffer(AGENT_NAME)[0]
    _assert_transition_equal(restored, expected)


def test_rpc_push_envelope_keeps_unprepared_third_party_values_on_compatibility_path():
    raw = object()
    prepared = _sa_episode(episode_num=19)
    prepared.prepare_for_replay()

    envelope = encode_replay_push_payload(
        {"prepared": prepared, "raw": raw}
    )

    assert isinstance(envelope, ReplayPushEnvelope)
    assert isinstance(
        envelope.episodes[ReplayPushEntryKey("prepared", 0)],
        ReplayTransitionEnvelope,
    )
    assert envelope.episodes["raw"] is raw


def test_rpc_push_envelope_rejects_unknown_version_without_mutating_buffer():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 4}, batch_size=1))
    source = _sa_episode(episode_num=23)
    source.prepare_for_replay()
    envelope = encode_replay_push_payload(source, agent_name=AGENT_NAME)
    invalid = ReplayPushEnvelope(
        version=ReplayPushEnvelope.CURRENT_VERSION + 1,
        episodes=envelope.episodes,
    )

    with pytest.raises(ValueError, match="unsupported ReplayPushEnvelope version"):
        replay_buffer.push(invalid, notify_learning=False)

    assert replay_buffer.size == 0
    assert replay_buffer.tensor_pool.memory_bytes == 0


def test_rpc_push_envelope_schema_failure_rolls_back_all_staged_agents():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(
        agent_capacities={"valid": 4, "invalid": 4}, batch_size=1,
    ))
    valid = _sa_episode(episode_num=29)
    valid.prepare_for_replay()
    encoded = encode_replay_push_payload(valid, agent_name="valid")
    valid_entry = encoded.episodes[ReplayPushEntryKey("valid", 0)]
    invalid_prepared = copy.deepcopy(valid_entry.prepared_episode)
    invalid_prepared.inline_rows = invalid_prepared.inline_rows[:-1]
    invalid_entry = ReplayTransitionEnvelope(
        transition_type=valid_entry.transition_type,
        episode_num=31,
        starts_with_episode=valid_entry.starts_with_episode,
        has_starts_with_episode=valid_entry.has_starts_with_episode,
        prepared_episode=invalid_prepared,
    )
    envelope = ReplayPushEnvelope(
        version=ReplayPushEnvelope.CURRENT_VERSION,
        episodes={"valid": valid_entry, "invalid": invalid_entry},
    )

    with pytest.raises(ValueError, match="inline_rows length"):
        replay_buffer.push(envelope, notify_learning=False)

    assert replay_buffer.size == 0
    assert replay_buffer.tensor_pool.memory_bytes == 0
    assert not replay_buffer.step_meta.get("valid")
    assert replay_buffer._next_step_meta_id == {}
    assert replay_buffer.tensor_schemas == {}
    assert replay_buffer._next_schema_id == 1


def test_rpc_transferred_cpu_tensor_chunk_is_adopted_without_clone():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 4}, batch_size=1))
    source = _sa_episode(episode_num=37)
    source.prepare_for_replay()
    envelope = encode_replay_push_payload(source, agent_name=AGENT_NAME)
    transferred = next(iter(
        envelope.episodes[ReplayPushEntryKey(AGENT_NAME, 0)].prepared_episode.tensors.values()
    ))

    replay_buffer.push(envelope, notify_learning=False)

    stored = next(iter(replay_buffer.tensor_pool._chunks.values())).tensors
    matching = next(
        tensor for tensor in stored.values()
        if tuple(tensor.shape) == tuple(transferred.shape)
        and tensor.dtype == transferred.dtype
    )
    assert matching.data_ptr() == transferred.data_ptr()
    assert matching.requires_grad is False


def test_local_borrowed_transition_still_clones_tensor_chunk():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 4}, batch_size=1))
    source = _sa_episode(episode_num=41)
    source.prepare_for_replay()
    borrowed = next(iter(source.prepared_replay_episode.tensors.values()))

    replay_buffer.push(
        source,
        agent_name=AGENT_NAME,
        notify_learning=False,
    )

    stored = next(iter(replay_buffer.tensor_pool._chunks.values())).tensors
    matching = next(
        tensor for tensor in stored.values()
        if tuple(tensor.shape) == tuple(borrowed.shape)
        and tensor.dtype == borrowed.dtype
    )
    assert matching.data_ptr() != borrowed.data_ptr()


def test_transferred_noncontiguous_tensor_falls_back_to_copy():
    pool = InMemoryStepTensorPool()
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    assert not source.is_contiguous()

    indices = pool.append_batch(
        schema_id=1,
        step_tensors={"value": source},
        ownership=TensorOwnership.TRANSFERRED,
    )
    stored = pool.get_batch(indices)["value"]

    assert stored.is_contiguous()
    assert stored.data_ptr() != source.data_ptr()
    assert torch.equal(stored, source)


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_episode_eviction_releases_only_its_step_tensor_and_metadata_ranges(tmp_path, pool):
    replay_buffer = ReplayBuffer(_replay_config(tmp_path, pool, capacity=2, batch_size=1))
    first = _sa_episode(1, step_count=2)
    second = _sa_episode(2, step_count=3)
    third = _sa_episode(3, step_count=4)
    try:
        for episode in (first, second, third):
            replay_buffer.push(episode, agent_name=AGENT_NAME, notify_learning=False)

        assert [episode.episode_num for episode in replay_buffer.ordered_buffer(AGENT_NAME)] == [2, 3]
        assert len(replay_buffer.tensor_pool.records) == second.seq_len + third.seq_len
        assert len(replay_buffer.step_meta[AGENT_NAME]) == second.seq_len + third.seq_len
        active_episode_sequences = {
            step_meta.episode_seq for step_meta in replay_buffer.step_meta[AGENT_NAME].values()
        }
        assert active_episode_sequences == {2, 3}
    finally:
        replay_buffer.close()


def test_step_meta_and_episode_meta_reference_only_adjacent_index_layer():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 2}, batch_size=1))
    replay_buffer.push(_sa_episode(5), agent_name=AGENT_NAME, notify_learning=False)

    episode_meta = replay_buffer.episode_meta[AGENT_NAME][0]
    assert isinstance(episode_meta, EpisodeMeta)
    assert not hasattr(episode_meta, "tensor_index")

    for step_meta_id in range(episode_meta.step_meta_start, episode_meta.step_meta_end):
        step_meta = replay_buffer.step_meta[AGENT_NAME][step_meta_id]
        assert isinstance(step_meta, StepMeta)
        assert step_meta.tensor_index in replay_buffer.tensor_pool.records
        assert isinstance(replay_buffer.tensor_schemas[step_meta.schema_id], TensorSchema)


def test_in_memory_tensor_pool_checkpoint_and_restore_preserves_tensor_records(tmp_path):
    pool = InMemoryStepTensorPool()
    indices = pool.append([
        {"obs": torch.tensor([1.0, 2.0]), "reward": torch.tensor(3.0)},
        {"obs": torch.tensor([4.0, 5.0]), "reward": torch.tensor(6.0)},
    ])
    ckpt = tmp_path / "step-pool"
    info = pool.checkpoint(ckpt)
    assert info.kind == "torch_save"

    restored = InMemoryStepTensorPool()
    restored.restore(ckpt)

    assert indices == [0, 1]
    restored_records = restored.get(indices)
    assert torch.equal(restored_records[0]["obs"], torch.tensor([1.0, 2.0]))
    assert torch.equal(restored_records[1]["reward"], torch.tensor(6.0))


def test_episode_sampling_strategies_keep_write_order_and_seed_behavior():
    replay_buffer = ReplayBuffer(
        ReplayBufferConfig(agent_capacities={AGENT_NAME: 5}, batch_size=3, recent_ratio=1 / 3)
    )
    for episode_num in range(1, 6):
        replay_buffer.push(
            _sa_episode(episode_num, step_count=1),
            agent_name=AGENT_NAME,
            notify_learning=False,
        )

    chronological = [1, 2, 3, 4, 5]
    assert [episode.episode_num for episode in replay_buffer.ordered_buffer(AGENT_NAME)] == (
        chronological
    )

    uniform_seed = 19
    expected_uniform = random.Random(uniform_seed).sample(chronological, 3)
    actual_uniform = replay_buffer.sample(
        AGENT_NAME,
        batch_size=3,
        strategy=SamplingStrategy.UNIFORM,
        seed=uniform_seed,
    )
    assert [episode.episode_num for episode in actual_uniform] == expected_uniform

    recent_seed = 23
    expected_recent = [5] + random.Random(recent_seed).sample([1, 2, 3, 4], 2)
    actual_recent = replay_buffer.sample(
        AGENT_NAME,
        batch_size=3,
        strategy=SamplingStrategy.RECENT_BIASED,
        seed=recent_seed,
    )
    assert [episode.episode_num for episode in actual_recent] == expected_recent

    actual_on_policy = replay_buffer.sample(
        AGENT_NAME,
        batch_size=3,
        strategy=SamplingStrategy.ON_POLICY_RECENT,
    )
    assert [episode.episode_num for episode in actual_on_policy] == [3, 4, 5]

    first_cursor_batch = replay_buffer.sample(
        AGENT_NAME,
        batch_size=2,
        strategy=SamplingStrategy.ON_POLICY_CURSOR,
    )
    second_cursor_batch = replay_buffer.sample(
        AGENT_NAME,
        batch_size=2,
        strategy=SamplingStrategy.ON_POLICY_CURSOR,
    )
    assert [episode.episode_num for episode in first_cursor_batch] == [1, 2]
    assert [episode.episode_num for episode in second_cursor_batch] == [3, 4]


def test_step_cursor_strategy_reads_step_meta_and_preserves_episode_boundaries():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 4}, batch_size=4))
    replay_buffer.push(
        _sa_episode(1, step_count=2),
        agent_name=AGENT_NAME,
        notify_learning=False,
    )
    replay_buffer.push(
        _sa_episode(2, step_count=3),
        agent_name=AGENT_NAME,
        notify_learning=False,
    )

    first_batch = replay_buffer.sample(
        AGENT_NAME,
        batch_size=4,
        strategy=SamplingStrategy.ON_POLICY_CURSOR_STEPS,
    )
    assert [(episode.episode_num, episode.seq_len) for episode in first_batch] == [(1, 2), (2, 2)]
    assert first_batch[0].starts_with_episode is True
    assert first_batch[1].starts_with_episode is True
    assert replay_buffer.on_policy_step_cursor[AGENT_NAME] == (2, 2)

    # 只剩Episode 2的最后一步，不足一个完整batch时游标不移动。
    assert replay_buffer.sample(
        AGENT_NAME,
        batch_size=4,
        strategy=SamplingStrategy.ON_POLICY_CURSOR_STEPS,
    ) == []
    assert replay_buffer.on_policy_step_cursor[AGENT_NAME] == (2, 2)


@pytest.mark.parametrize(
    "strategy",
    [
        SamplingStrategy.ON_POLICY_RECENT_STEPS,
        SamplingStrategy.ON_POLICY_CURSOR_STEPS,
    ],
)
def test_insufficient_step_sample_uses_metadata_preflight_without_materialization(
        strategy,
):
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 4}, batch_size=560))
    replay_buffer.push(
        _sa_episode(1, step_count=3),
        agent_name=AGENT_NAME,
        notify_learning=False,
    )
    original_cursor = dict(replay_buffer.on_policy_step_cursor)

    def fail_materialization(*_args, **_kwargs):
        raise AssertionError("insufficient sample must not materialize Transition")

    replay_buffer._materialize_episode = fail_materialization

    assert replay_buffer.sample(
        AGENT_NAME,
        batch_size=560,
        strategy=strategy,
    ) == []
    assert replay_buffer.on_policy_step_cursor == original_cursor


def test_lazy_memmap_get_batch_copy_out_and_release(tmp_path):
    pool = LazyMemmapStepTensorPool(
        max_steps=16,
        scratch_dir=tmp_path / "scratch",
    )
    try:
        values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        indices = pool.append_batch(schema_id=1, step_tensors={"value": values})
        batch = pool.get_batch(indices)
        mmap_tensor = torch.from_numpy(np.asarray(pool._columns[1]["value"].mmap[:3]))
        assert batch["value"].data_ptr() != mmap_tensor.data_ptr()
        assert torch.equal(batch["value"], values)

        pool.release(indices[:1])
        with pytest.raises(KeyError):
            pool.get([indices[0]])
        remaining = pool.get_batch(indices[1:])
        assert torch.equal(remaining["value"], values[1:])
    finally:
        pool.close()


def test_lazy_memmap_gather_preserves_order_duplicates_and_copy_out(tmp_path):
    pool = LazyMemmapStepTensorPool(max_steps=8, scratch_dir=tmp_path)
    try:
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        flags = torch.tensor([True, False, True, False])
        indices = pool.append_batch(1, {"value": values, "flags": flags})
        selected = [indices[3], indices[0], indices[3], indices[1]]
        batch = pool.get_batch(selected)
        assert torch.equal(batch["value"], values[[3, 0, 3, 1]])
        assert torch.equal(batch["flags"], flags[[3, 0, 3, 1]])
        batch["value"][0].fill_(-1)
        assert torch.equal(batch["value"][2], values[3])
        assert torch.equal(pool.get_batch([indices[3]])["value"][0], values[3])

        pool.release(indices)
        pool.append_batch(1, {"value": torch.zeros_like(values), "flags": ~flags})
        assert torch.equal(batch["value"][1], values[0])
        with pytest.raises(ValueError, match="one TensorSchema"):
            other = pool.append_batch(2, {"other": torch.zeros(1)})
            pool.get_batch([selected[0], other[0]])
    finally:
        pool.close()


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
@pytest.mark.parametrize("factory", [_sa_episode, _ma_episode], ids=["single_agent", "multi_agent"])
@pytest.mark.parametrize("include_burn_in", [False, True])
@pytest.mark.parametrize("sequence_length", [1, 3])
def test_uniform_windows_match_individual_materialization(
        tmp_path, pool, factory, include_burn_in, sequence_length, monkeypatch,
):
    replay = ReplayBuffer(_replay_config(tmp_path, pool, capacity=8))
    try:
        for num in range(1, 7):
            episode = factory(num, step_count=1 + num)
            # Mix schemas both between episodes and inside individual windows.
            if num % 2:
                episode.metadata[1]["extra_tensor"] = torch.tensor([num], dtype=torch.float32)
            replay.push({AGENT_NAME: episode}, notify_learning=False)
        replay.archive_from_active("previous", 0.5, "even_span")
        replay.set_rehearsal_sample_ratio(0.5)
        kwargs = dict(agent_name=AGENT_NAME, batch_size=32, strategy="uniform_steps",
                      seed=17, include_burn_in=include_burn_in, sequence_length=sequence_length)
        actual = replay.sample(**kwargs)
        remote = replay.sample_for_rpc(**kwargs)

        def individual_windows(windows):
            return [replay._materialize_episode(meta, start, end, sliced=True)
                    for meta, start, end in windows]

        monkeypatch.setattr(replay, "_materialize_episode_windows", individual_windows)
        expected = replay.sample(**kwargs)
        for local_item, remote_item, old_item in zip(actual, remote, expected):
            _assert_transition_equal(local_item, old_item)
            _assert_transition_equal(remote_item, old_item)
        assert len(actual) == len(remote) == len(expected) == 32
        if factory is _ma_episode:
            actual[0].obs[next(iter(actual[0].obs))][0].fill(-999)
        else:
            actual[0].obs[0].fill(-999)
        fresh = replay.sample(**kwargs)
        _assert_transition_equal(fresh[0], expected[0])
    finally:
        replay.close()


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_uniform_windows_read_each_schema_once(tmp_path, pool, monkeypatch):
    replay = ReplayBuffer(_replay_config(tmp_path, pool))
    try:
        for num in (1, 2):
            episode = _sa_episode(num, step_count=5)
            for metadata in episode.metadata:
                metadata["diagnostic"] = torch.tensor([num], dtype=torch.int32 if num == 1 else torch.int64)
            replay.push({AGENT_NAME: episode}, notify_learning=False)
        calls = []
        get_batch = replay.tensor_pool.get_batch

        def track_batch(indices):
            calls.append(list(indices))
            return get_batch(indices)

        monkeypatch.setattr(replay.tensor_pool, "get_batch", track_batch)
        result = replay.sample(AGENT_NAME, batch_size=64, strategy="uniform_steps",
                               seed=23, sequence_length=3, include_burn_in=False)
        assert len(calls) == 2
        assert sum(map(len, calls)) == sum(episode.seq_len for episode in result)
        assert any(len(set(indices)) < len(indices) for indices in calls)
    finally:
        replay.close()


def test_lazy_memmap_rejects_exhausted_max_steps(tmp_path):
    pool = LazyMemmapStepTensorPool(max_steps=2, scratch_dir=tmp_path / "scratch")
    try:
        pool.append_batch(schema_id=1, step_tensors={"value": torch.zeros(2, 3)})
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.append_batch(schema_id=1, step_tensors={"value": torch.ones(1, 3)})
    finally:
        pool.close()


def test_lazy_memmap_scratch_lives_under_storage_subdir(tmp_path):
    replay_buffer = ReplayBuffer(_replay_config(tmp_path, "lazy_memmap"))
    try:
        replay_buffer.push(_sa_episode(1, step_count=2), agent_name=AGENT_NAME, notify_learning=False)
        scratch = Path(replay_buffer.tensor_pool.scratch_dir)
        task_scratch = scratch.parent
        assert scratch.is_relative_to(tmp_path / "scratch")
    finally:
        replay_buffer.close()
    assert not scratch.exists()
    assert not task_scratch.exists()


def test_lazy_memmap_construct_does_not_create_empty_scratch(tmp_path):
    pool = LazyMemmapStepTensorPool(max_steps=2, scratch_dir=tmp_path / "scratch")
    try:
        assert not (tmp_path / "scratch").exists()
        assert pool.created_scratch_dir is None
    finally:
        pool.close()
    assert not (tmp_path / "scratch").exists()


def test_replay_buffer_close_without_push_removes_empty_scratch_shell(tmp_path):
    from utils.task_id import task_id

    leftover = tmp_path / "scratch" / str(task_id) / "step_pool_empty"
    leftover.mkdir(parents=True)
    replay_buffer = ReplayBuffer(_replay_config(tmp_path, "lazy_memmap"))
    replay_buffer.close()
    assert not leftover.exists()
    assert not leftover.parent.exists()


def test_keep_scratch_does_not_keep_empty_shell(tmp_path):
    from utils.task_id import task_id

    leftover = tmp_path / "scratch" / str(task_id) / "step_pool_empty"
    leftover.mkdir(parents=True)
    replay_buffer = ReplayBuffer(_replay_config(tmp_path, "lazy_memmap", keep_scratch=True))
    replay_buffer.close()
    assert not leftover.parent.exists()


def test_keep_scratch_keeps_local_without_s3_upload(tmp_path, monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(
        "sample.replay.replay_persist.build_replay_s3_client",
        lambda storage, *, ensure_bucket=False: client,
    )
    monkeypatch.setattr(
        "sample.replay.step_tensor_pool.resolve_mmap_scratch_dir",
        lambda _config: tmp_path / "scratch",
    )
    replay_buffer = ReplayBuffer(_replay_config(
        tmp_path,
        "lazy_memmap",
        storage_type=StorageType.S3.value,
        storage=S3StorageConfig(bucket_name="replay-buffer"),
        keep_scratch=True,
    ))
    replay_buffer.push(_sa_episode(1, step_count=2), agent_name=AGENT_NAME, notify_learning=False)
    scratch = Path(replay_buffer.tensor_pool.scratch_dir)
    replay_buffer.close()
    assert scratch.exists()
    assert not client.upload_file.called


def test_memory_and_mmap_uniform_sample_are_bit_identical(tmp_path):
    episode_count = 36
    batch_size = 8
    memory = ReplayBuffer(_replay_config(tmp_path, "memory", capacity=64, batch_size=batch_size))
    mmap = ReplayBuffer(_replay_config(tmp_path / "mmap", "lazy_memmap", capacity=64, batch_size=batch_size))
    restored_memory = ReplayBuffer(
        _replay_config(tmp_path / "memory_restored", "memory", capacity=64, batch_size=batch_size)
    )
    restored_mmap = ReplayBuffer(
        _replay_config(tmp_path / "mmap_restored", "lazy_memmap", capacity=64, batch_size=batch_size)
    )
    try:
        for episode_num in range(1, episode_count + 1):
            episode = _sa_episode(episode_num, step_count=3)
            memory.push(episode, agent_name=AGENT_NAME, notify_learning=False)
            mmap.push(copy.deepcopy(episode), agent_name=AGENT_NAME, notify_learning=False)

        _assert_tensor_records_equal(mmap, memory)

        first_memory = memory.sample(
            AGENT_NAME, batch_size=batch_size, strategy=SamplingStrategy.UNIFORM, seed=19,
        )
        first_mmap = mmap.sample(
            AGENT_NAME, batch_size=batch_size, strategy=SamplingStrategy.UNIFORM, seed=19,
        )
        assert len(first_memory) == len(first_mmap) == batch_size
        for actual, expected in zip(first_mmap, first_memory):
            _assert_transition_equal(actual, expected)

        second_memory = memory.sample(
            AGENT_NAME, batch_size=batch_size, strategy=SamplingStrategy.UNIFORM, seed=23,
        )
        second_mmap = mmap.sample(
            AGENT_NAME, batch_size=batch_size, strategy=SamplingStrategy.UNIFORM, seed=23,
        )
        for actual, expected in zip(second_mmap, second_memory):
            _assert_transition_equal(actual, expected)
        assert [ep.episode_num for ep in first_memory] != [ep.episode_num for ep in second_memory]

        memory_path = tmp_path / "memory_replay.bin"
        mmap_path = tmp_path / "mmap_replay.bin"
        memory.save_snapshot(memory_path)
        mmap.save_snapshot(mmap_path)
        memory_meta, _, memory_tensor_len = read_drbf(memory_path)
        mmap_meta, _, mmap_tensor_len = read_drbf(mmap_path)
        assert memory_tensor_len > 0 and mmap_tensor_len > 0
        assert memory_meta["tensor_kind"] == KIND_TORCH_SAVE
        assert mmap_meta["tensor_kind"] == KIND_MMAP_DIR
        _assert_drbf_meta_semantically_equal(mmap_meta, memory_meta)

        restored_memory.load_snapshot(memory_path)
        restored_mmap.load_snapshot(mmap_path)
        _assert_tensor_records_equal(restored_mmap, restored_memory)
        restored_memory_sample = restored_memory.sample(
            AGENT_NAME, batch_size=batch_size, strategy=SamplingStrategy.UNIFORM, seed=19,
        )
        restored_mmap_sample = restored_mmap.sample(
            AGENT_NAME, batch_size=batch_size, strategy=SamplingStrategy.UNIFORM, seed=19,
        )
        for actual, expected in zip(restored_mmap_sample, restored_memory_sample):
            _assert_transition_equal(actual, expected)
        for actual, expected in zip(restored_memory_sample, first_memory):
            _assert_transition_equal(actual, expected)
    finally:
        memory.close()
        mmap.close()
        restored_memory.close()
        restored_mmap.close()


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_replay_buffer_snapshot_roundtrip_keeps_tensor_values(tmp_path, pool):
    config = _replay_config(tmp_path, pool, capacity=16, batch_size=2)
    source = ReplayBuffer(config)
    restored = ReplayBuffer(_replay_config(tmp_path / "restored", pool, capacity=16, batch_size=2))
    try:
        source.push(_sa_episode(1), agent_name=AGENT_NAME, notify_learning=False)
        source.push(_sa_episode(2, step_count=2), agent_name=AGENT_NAME, notify_learning=False)
        path = tmp_path / "replay.bin"
        source.save_snapshot(path)
        restored.load_snapshot(path)
        assert restored.agent_buffer_size(AGENT_NAME) == 2
        source_rows = source.tensor_pool.records
        restored_rows = restored.tensor_pool.records
        assert source_rows.keys() == restored_rows.keys()
        for index, expected_row in source_rows.items():
            for key, expected in expected_row.items():
                torch.testing.assert_close(restored_rows[index][key], expected, rtol=0, atol=0)
    finally:
        source.close()
        restored.close()


def test_mmap_snapshot_refuses_to_load_into_memory_pool(tmp_path):
    mmap = ReplayBuffer(_replay_config(tmp_path, "lazy_memmap", capacity=8, batch_size=1))
    memory = ReplayBuffer(_replay_config(tmp_path, "memory", capacity=8, batch_size=1))
    try:
        mmap.push(_sa_episode(4), agent_name=AGENT_NAME, notify_learning=False)
        path = tmp_path / "mmap.bin"
        mmap.save_snapshot(path)
        with pytest.raises(ValueError, match="tensor_kind"):
            memory.load_snapshot(path)
    finally:
        mmap.close()
        memory.close()


_MEMORY_USAGE_LOG = re.compile(
    r"ReplayBuffer memory usage: "
    r"(?:tensors|tensors_on_disk)=(?P<tensors>[\d,.]+) MB"
    r"(?:, process_rss=(?P<rss>[\d,.]+) MB)?"
    r", metadata=(?P<metadata>[\d,.]+) MB, "
    r"size=(?P<size>\d+), writes=(?P<writes>\d+)"
)


def _parse_mb(text: str) -> float:
    return float(text.replace(",", ""))


def _wait_until(predicate, *, timeout: float = 3.0, message: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(message)


def _memory_usage_records(caplog) -> list:
    return [
        record for record in caplog.records
        if "ReplayBuffer memory usage:" in record.getMessage()
    ]


@pytest.mark.parametrize("pool", ["memory", "lazy_memmap"])
def test_memory_usage_logs_every_interval_after_full_and_matches_occupancy(
        tmp_path, caplog, pool,
):
    """满池后仍按写入局数打点；日志里的 tensors/metadata 与当时真实占用一致。"""
    capacity = 3
    interval = 2
    push_count = 8
    replay_buffer = ReplayBuffer(_replay_config(
        tmp_path, pool, capacity=capacity, batch_size=1,
    ))
    replay_buffer.memory_report_size_interval = interval
    caplog.set_level(logging.INFO, logger="replay_buffer")

    reports = []
    try:
        for episode_num in range(1, push_count + 1):
            replay_buffer.push(
                _sa_episode(episode_num, step_count=4),
                agent_name=AGENT_NAME,
                notify_learning=False,
            )
            if episode_num % interval != 0:
                continue
            expected_count = episode_num // interval
            _wait_until(
                lambda count=expected_count: len(_memory_usage_records(caplog)) >= count,
                message=f"memory usage log was not emitted at writes={episode_num}",
            )
            parsed = _MEMORY_USAGE_LOG.search(
                _memory_usage_records(caplog)[-1].getMessage()
            )
            assert parsed is not None, _memory_usage_records(caplog)[-1].getMessage()

            logged_tensors = _parse_mb(parsed.group("tensors"))
            logged_metadata = _parse_mb(parsed.group("metadata"))
            live_tensors = replay_buffer.tensor_pool.memory_bytes / (1024 ** 2)
            live_metadata = estimate_object_memory_mb({
                "episode_meta": replay_buffer.buffer,
                "step_meta": replay_buffer.step_meta,
                "tensor_schemas": replay_buffer.tensor_schemas,
            })
            reports.append({
                "writes": int(parsed.group("writes")),
                "size": int(parsed.group("size")),
                "live_bytes": replay_buffer.tensor_pool.memory_bytes,
                "live_metadata": live_metadata,
                "logged_metadata": logged_metadata,
            })
            assert int(parsed.group("writes")) == episode_num
            assert int(parsed.group("size")) == min(episode_num, capacity)
            assert logged_tensors == pytest.approx(live_tensors, abs=1e-3)
            assert logged_metadata == pytest.approx(live_metadata, abs=1e-3)
            if pool == "lazy_memmap":
                assert parsed.group("rss") is not None

        assert [item["writes"] for item in reports] == [2, 4, 6, 8]
        assert [item["size"] for item in reports] == [2, 3, 3, 3]
        if pool == "memory":
            assert reports[0]["live_bytes"] < reports[1]["live_bytes"]
            assert reports[1]["live_bytes"] == reports[-1]["live_bytes"]
            assert reports[0]["live_metadata"] < reports[1]["live_metadata"]
            assert reports[1]["logged_metadata"] == pytest.approx(
                reports[-1]["logged_metadata"], abs=1e-3,
            )
    finally:
        replay_buffer.close()

"""ReplayBuffer DRBF 阶段快照往返测试。"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from algos.modules.agents.random_agent import RandomAgent
from common.exception import S3ProtocolException
from config import (
    LocalStorageConfig,
    MonteCarloAgentConfig,
    ReplayBufferConfig,
    S3StorageConfig,
    StageConfig,
    StageTaskConfig,
    StorageType,
    TrainConfig,
)
from envs.sense.target_selection_env import TargetSelectionEnv
from scheduler.model import AgentCompose
from scheduler.rollout_sample import RolloutSampleWorker
from sample.replay_buffer import ReplayBuffer
from sample.replay.replay_persist import (
    extract_drbf_tensors,
    read_drbf,
    resolve_replay_snapshot_path,
    serialize_persist_value,
    write_drbf,
)
from sample.sample_transfer import SAEpisodeTransition
from scheduler.stage_task import StageTask
from utils.stage_utils import (
    apply_stage_to_resources,
    load_skip_stage_replay_if_needed,
    save_stage_replay_if_needed,
)
from utils.toml_config import TomlTrainingConfig


def _replay_config(*, capacity: int, **kwargs) -> ReplayBufferConfig:
    return ReplayBufferConfig(
        agent_capacities={"agent_0": capacity, "selector": capacity}, **kwargs,
    )


def _make_sa_episode(episode_num: int, steps: int = 3) -> SAEpisodeTransition:
    transition = SAEpisodeTransition()
    transition.episode_num = episode_num
    for step in range(steps):
        transition.add_step(
            obs=np.asarray([episode_num, step, 0.25], dtype=np.float32),
            action=step % 2,
            last_action=(step - 1) % 2,
            reward=float(step) + 0.125,
            next_obs=np.asarray([episode_num, step + 1, 0.25], dtype=np.float32),
            terminated=step == steps - 1,
            truncated=False,
            avail_actions=[1, 0, 1],
            log_prob=-0.25 * step,
            value=1.5 + step,
            metadata={"label": f"step-{step}"},
        )
    transition.starts_with_episode = True
    return transition


def test_replay_snapshot_roundtrip(tmp_path: Path) -> None:
    config = _replay_config(capacity=16, batch_size=2)
    buffer = ReplayBuffer(config)
    buffer.push({"agent_0": _make_sa_episode(1)}, save=False, notify_learning=False)
    buffer.push({"agent_0": _make_sa_episode(2, steps=2)}, save=False, notify_learning=False)
    before = buffer.sample("agent_0", batch_size=2)
    assert len(before) == 2

    path = tmp_path / "replay_agent0.bin"
    buffer.save_snapshot(path)
    assert path.exists()
    meta, _offset, tensor_len = read_drbf(path)
    assert meta["format_version"] == 1
    assert tensor_len > 0
    assert meta["tensor_kind"] == buffer.tensor_pool.kind

    restored = ReplayBuffer(_replay_config(capacity=16, batch_size=2))
    restored.load_snapshot(path)
    assert restored.generation == 0
    assert restored.on_policy_cursor == {}
    assert restored.sliding_idx == {}
    assert restored.agent_buffer_size("agent_0") == 2
    after = restored.sample("agent_0", batch_size=2)
    assert len(after) == 2
    assert after[0].seq_len == before[0].seq_len or after[1].seq_len == before[0].seq_len
    rewards = sorted(float(sum(ep.rewards)) for ep in after)
    expected = sorted(float(sum(ep.rewards)) for ep in before)
    assert rewards == expected


def test_snapshot_rejects_smaller_partition_capacity(tmp_path: Path) -> None:
    source = ReplayBuffer(_replay_config(capacity=3, batch_size=1))
    try:
        source.push({"agent_0": _make_sa_episode(1)}, notify_learning=False)
        path = source.save_snapshot(tmp_path / "capacity.bin")
        target = ReplayBuffer(_replay_config(capacity=2, batch_size=1))
        try:
            with pytest.raises(ValueError, match="exceeds current capacity"):
                target.load_snapshot(path)
        finally:
            target.close()
    finally:
        source.close()


def test_load_legacy_scalar_capacity_snapshot(tmp_path: Path) -> None:
    source = ReplayBuffer(_replay_config(capacity=3, batch_size=1))
    try:
        source.push({"agent_0": _make_sa_episode(1)}, notify_learning=False)
        current = source.save_snapshot(tmp_path / "current.bin")
        metadata, tensor_blob = extract_drbf_tensors(current, tmp_path / "tensor_blob")
        metadata.pop("agent_capacities")
        metadata["capacity"] = 3
        metadata["active_capacity"] = 3
        legacy = write_drbf(tmp_path / "legacy.bin", metadata, tensor_blob)

        target = ReplayBuffer(_replay_config(capacity=3, batch_size=1))
        try:
            target.load_snapshot(legacy)
            assert target.agent_buffer_size("agent_0") == 1
        finally:
            target.close()
    finally:
        source.close()


def test_save_stage_replay_if_needed_respects_flag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "utils.stage_utils.samples_dir_from_config",
        lambda _config=None: str(tmp_path),
    )
    buffer = ReplayBuffer(_replay_config(capacity=8, batch_size=1))
    buffer.push({"agent_0": _make_sa_episode(1)}, save=False, notify_learning=False)
    stage = StageConfig(stage_name="s0", save_replay_buffer=False)
    assert save_stage_replay_if_needed(stage, buffer, "task1") is None
    stage.save_replay_buffer = True
    written = save_stage_replay_if_needed(stage, buffer, "task1")
    assert written is not None
    assert written.exists()
    assert written.name.startswith("replay_s0_task1")


def test_notify_stage_complete_uploads_s3(tmp_path: Path, monkeypatch) -> None:
    uploaded: list[Path] = []

    def fake_sync(path, config) -> None:
        uploaded.append(Path(path))

    monkeypatch.setattr("sample.replay.replay_persist.sync_snapshot_to_storage", fake_sync)
    monkeypatch.setattr(
        "sample.replay.replay_persist.samples_dir_from_config",
        lambda _config=None: str(tmp_path),
    )
    buffer = ReplayBuffer(
        _replay_config(
            capacity=8,
            batch_size=1,
            storage_type=StorageType.S3.value,
            storage=S3StorageConfig(bucket_name="replay-buffer"),
        )
    )
    try:
        buffer.push({"agent_0": _make_sa_episode(1)}, save=False, notify_learning=False)
        stage = StageConfig(stage_name="s0", save_replay_buffer=True)
        written = buffer.notify_stage_complete(stage, task_id="task1")
        assert written is not None
        assert written.exists()
        assert uploaded == [written]
    finally:
        buffer.close()


def test_notify_stage_complete_local_skips_s3_upload(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple] = []
    monkeypatch.setattr(
        "sample.replay.replay_persist.sync_snapshot_to_storage",
        lambda *args, **kwargs: calls.append(args),
    )
    monkeypatch.setattr(
        "sample.replay.replay_persist.samples_dir_from_config",
        lambda _config=None: str(tmp_path),
    )
    buffer = ReplayBuffer(
        _replay_config(
            capacity=8,
            batch_size=1,
            storage=LocalStorageConfig(data_path=str(tmp_path)),
        )
    )
    try:
        buffer.push({"agent_0": _make_sa_episode(1)}, save=False, notify_learning=False)
        stage = StageConfig(stage_name="s0", save_replay_buffer=True)
        written = buffer.notify_stage_complete(stage, task_id="task1")
        assert written is not None
        assert calls == []
    finally:
        buffer.close()


def test_notify_stage_complete_saves_rehearsal_before_snapshot(tmp_path: Path) -> None:
    buffer = ReplayBuffer(
        _replay_config(
            capacity=8,
            batch_size=1,
            storage=LocalStorageConfig(data_path=str(tmp_path)),
        )
    )
    restored = ReplayBuffer(_replay_config(capacity=8, batch_size=1))
    try:
        for episode_num in range(4):
            buffer.push(
                {"agent_0": _make_sa_episode(episode_num)},
                save=False,
                notify_learning=False,
            )
        stage = StageConfig(
            stage_name="s0",
            save_replay_buffer=True,
            rehearsal_archive_ratio=0.5,
        )
        written = buffer.notify_stage_complete(stage, task_id="task1")
        assert written is not None

        restored.load_snapshot(written)
        assert "s0" in restored._rehearsal
        assert restored._rehearsal["s0"]["agent_0"]
    finally:
        restored.close()
        buffer.close()


def test_notify_stage_complete_s3_failure_raises_protocol_error(
        tmp_path: Path, monkeypatch,
) -> None:
    def fail_sync(*args, **kwargs) -> None:
        raise RuntimeError("upload denied")

    monkeypatch.setattr("sample.replay.replay_persist.sync_snapshot_to_storage", fail_sync)
    monkeypatch.setattr(
        "sample.replay.replay_persist.samples_dir_from_config",
        lambda _config=None: str(tmp_path),
    )
    buffer = ReplayBuffer(
        _replay_config(
            capacity=8,
            batch_size=1,
            storage_type=StorageType.S3.value,
            storage=S3StorageConfig(bucket_name="replay-buffer"),
        )
    )
    try:
        buffer.push({"agent_0": _make_sa_episode(1)}, save=False, notify_learning=False)
        stage = StageConfig(stage_name="s0", save_replay_buffer=True)
        with pytest.raises(S3ProtocolException, match="upload denied"):
            buffer.notify_stage_complete(stage, task_id="task1")
    finally:
        buffer.close()


def test_load_skip_stage_replay_only_when_start_stage_gt_zero(tmp_path: Path) -> None:
    source = ReplayBuffer(_replay_config(capacity=8, batch_size=1))
    source.push({"agent_0": _make_sa_episode(9)}, save=False, notify_learning=False)
    path = tmp_path / "skip.bin"
    source.save_snapshot(path)

    target = ReplayBuffer(_replay_config(capacity=8, batch_size=1))
    cfg = StageTaskConfig(start_stage=0, skip_stage_replay_path=str(path))
    assert load_skip_stage_replay_if_needed(cfg, target) is False
    assert target.size == 0

    cfg.start_stage = 1
    assert load_skip_stage_replay_if_needed(cfg, target) is True
    assert target.agent_buffer_size("agent_0") == 1
    assert load_skip_stage_replay_if_needed(cfg, target, already_loaded=True) is False


def test_stage_task_applies_skip_stage_model_fields() -> None:
    from config import AgentConfig

    agent = AgentConfig(name="agent_0")
    stages = [
        StageConfig(
            stage_name="a",
            agents_config=[agent],
            train_config=TrainConfig(sample_learn_max_lag=None),
        ),
        StageConfig(
            stage_name="b",
            agents_config=[agent],
            train_config=TrainConfig(
                model_load_dir="/tmp/models",
                model_version="old",
                sample_learn_max_lag=None,
            ),
        ),
    ]
    task = StageTask(
        StageTaskConfig(
            stages=stages,
            start_stage=1,
            skip_stage_model_load_dir="/data/models",
            skip_stage_model_version=" final_v1 ",
        )
    )
    try:
        assert task.starting_stage.train_config.model_version == "final_v1"
        assert task.starting_stage.train_config.model_load_dir == "/data/models"
    finally:
        task.close()

def test_resolve_replay_snapshot_path_relative(tmp_path: Path) -> None:
    resolved = resolve_replay_snapshot_path("foo.bin", samples_dir=str(tmp_path))
    assert resolved == tmp_path / "foo.bin"
    abs_path = tmp_path / "abs.bin"
    assert resolve_replay_snapshot_path(abs_path) == abs_path


def _snapshot_meta(buffer: ReplayBuffer) -> dict:
    """与 ReplayBuffer.save_snapshot 的内存元数据汇总保持同一字段集。"""
    return {
        "format_version": 1,
        "agent_capacities": {
            name: buffer._capacity_for_agent(name) for name in buffer.agent_names()
        },
        "strategy": str(buffer.config.strategy),
        "batch_size": int(buffer.config.batch_size),
        "recent_ratio": float(buffer.config.recent_ratio),
        "rehearsal_sample_ratio": float(buffer._rehearsal_sample_ratio),
        "buffer": serialize_persist_value(buffer.buffer),
        "rehearsal": serialize_persist_value(buffer._rehearsal),
        "step_meta": serialize_persist_value(buffer.step_meta),
        "tensor_schemas": serialize_persist_value(buffer.tensor_schemas),
        "position": serialize_persist_value(dict(buffer.position)),
        "write_count": serialize_persist_value(dict(buffer.write_count)),
        "sample_seq": serialize_persist_value(dict(buffer.sample_seq)),
        "next_step_meta_id": serialize_persist_value(dict(buffer._next_step_meta_id)),
        "next_schema_id": int(buffer._next_schema_id),
        "tensor_kind": getattr(buffer.tensor_pool, "kind", "torch_save"),
    }


def _assert_tensor_pool_equal(expected: ReplayBuffer, actual: ReplayBuffer) -> None:
    """逐索引逐字段比较张量池，覆盖值、shape、dtype 与索引布局。"""
    expected_records = expected.tensor_pool.records
    actual_records = actual.tensor_pool.records
    assert actual_records.keys() == expected_records.keys()
    for tensor_index, expected_row in expected_records.items():
        actual_row = actual_records[tensor_index]
        assert actual_row.keys() == expected_row.keys()
        for field, expected_tensor in expected_row.items():
            actual_tensor = actual_row[field]
            assert actual_tensor.dtype == expected_tensor.dtype
            assert actual_tensor.shape == expected_tensor.shape
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_real_staged_subject_snapshot_file_and_skip_entry_roundtrip(tmp_path: Path) -> None:
    """真实 target_selection 课程采样后，验证 DRBF 文件和跳过阶段入口恢复完全等价。"""
    root = Path(__file__).resolve().parents[1]
    subject = TomlTrainingConfig(
        root / "subjects/shaping_test/target_selection.toml", project_root=root,
    )

    stage0 = copy.deepcopy(subject.stage_task_config.stages[0])
    stage0.stage_name = "selection_collect"
    stage0.train_config.max_episode = 3
    # 本测试只走采样与持久化，不消费学习令牌；关闭训练滞后背压。
    stage0.train_config.sample_learn_max_lag = None
    stage0.train_config.mid_eval_enabled = False
    stage0.train_config.mid_eval_episode_interval = 0
    stage0.train_config.model_save_dir = str(tmp_path / "models")
    stage0.train_config.result_save_dir = str(tmp_path / "results")
    stage0.save_replay_buffer = True
    # 采样使用随机策略，但沿用科目本身的真实环境、动作定义与阶段调度。
    stage0.agents_config[0].agent_type = "monte_carlo"
    stage0.agents_config[0].algo_config = MonteCarloAgentConfig()

    stage1 = copy.deepcopy(stage0)
    stage1.stage_name = "selection_resume"
    stage1.clear_replay_buffer = False
    stage1.save_replay_buffer = False
    stages = [stage0, stage1]
    replay_config = _replay_config(
        capacity=8,
        batch_size=4,
        storage=LocalStorageConfig(data_path=str(tmp_path)),
    )
    stage_task = StageTask(StageTaskConfig(stages=stages), env_name=subject.env_config.sense_name)
    env = TargetSelectionEnv(copy.deepcopy(subject.env_config), connect_sim=False)
    source = ReplayBuffer(replay_config)
    try:
        task_manager = next(stage_task)
        apply_stage_to_resources(stage0, env, learners_by_name={}, replay_buffer=source)
        policy_agent = RandomAgent(
            agent_type="monte_carlo",
            config=stage0.agents_config[0],
            agent_net_source="blanco",
            device="cpu",
        )
        worker = RolloutSampleWorker(
            env=env,
            task_manager=task_manager,
            replay_buffer=source,
            agent_compose=AgentCompose(compose_type="order", step_agent=[policy_agent]),
            policy_agents=[policy_agent],
        )
        for _ in range(3):
            assert worker.rollout_one_episode() is not None

        # 条件 1：文件中的每个元数据字段和张量段都与保存瞬间的内存池一致。
        expected_meta = _snapshot_meta(source)
        snapshot_path = save_stage_replay_if_needed(stage0, source, task_manager.task_id)
        assert snapshot_path is not None
        file_meta, _offset, tensor_len = read_drbf(snapshot_path)
        assert tensor_len > 0
        for key, value in expected_meta.items():
            assert file_meta[key] == value

        # 条件 2：新建 ReplayBuffer 后经 start_stage=1 跳过入口加载，所有数据元数据恢复一致。
        restored = ReplayBuffer(replay_config)
        restored.set_generation(99)
        resume_config = StageTaskConfig(
            stages=stages,
            start_stage=1,
            skip_stage_replay_path=str(snapshot_path),
        )
        resume_task = StageTask(resume_config, env_name=subject.env_config.sense_name)
        try:
            assert resume_task.starting_stage.stage_name == "selection_resume"
            assert next(resume_task).stage_name == "selection_resume"
            # 与 SerialStageTrainRole._apply_stage 相同的先应用阶段、再按 skip 配置灌池顺序。
            apply_stage_to_resources(
                resume_task.starting_stage,
                env,
                learners_by_name={},
                replay_buffer=restored,
            )
            assert load_skip_stage_replay_if_needed(resume_config, restored) is True
            assert restored.generation == 0
            assert _snapshot_meta(restored) == expected_meta
            _assert_tensor_pool_equal(source, restored)
            assert load_skip_stage_replay_if_needed(
                resume_config, restored, already_loaded=True,
            ) is False
        finally:
            resume_task.close()
    finally:
        env.close()
        stage_task.close()

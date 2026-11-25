from __future__ import annotations

import threading
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from common.info_bridge import info_bridge, publish_training_info
from config import (
    AgentConfig,
    AgentType,
    AlgorithmType,
    LearnerConfig,
    LocalStorageConfig,
    PpoConfig,
    ReplayBufferConfig,
    StageConfig,
    StageTaskConfig,
    TrainConfig,
)
from rpc.dispatch import build_train_task_manager_methods
from sample.replay_buffer import ReplayBuffer
from scheduler.stage_task import StageTask, StageTaskManagerProxy
from scheduler.train_task import TrainTaskManager


def _agent_config(name: str = "agent_a") -> AgentConfig:
    return AgentConfig(name=name, agent_type=AgentType.SB3.value, agent_names=[name])


def _learner_config(name: str = "agent_a") -> LearnerConfig:
    learner_config = LearnerConfig(name=name, algo_type=AlgorithmType.PPO.value)
    assert isinstance(learner_config.algo_config, PpoConfig)
    learner_config.algo_config.batch_size = 1
    learner_config.algo_config.learn_times_per_episode = 1
    return learner_config


def test_put_get_and_missing_key() -> None:
    info_bridge.put(foo=1, bar="x")
    assert info_bridge.get("foo") == 1
    assert info_bridge.get("bar") == "x"
    assert info_bridge.get("missing", None) is None
    with pytest.raises(KeyError):
        info_bridge.get("missing")


def test_scope_restores_previous_store() -> None:
    info_bridge.put(foo=1)
    with info_bridge.scope(foo=2, extra=3):
        assert info_bridge.get("foo") == 2
        assert info_bridge.get("extra") == 3
    assert info_bridge.get("foo") == 1
    assert info_bridge.get("extra", None) is None


def test_cross_thread_get_sees_role_put() -> None:
    info_bridge.put(toml_config={"env_config": SimpleNamespace(max_step_per_episode=600)})
    seen: list[int] = []

    def worker() -> None:
        seen.append(int(info_bridge.get("toml_config")["env_config"].max_step_per_episode))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert seen == [600]


def test_snapshot_is_shallow_copy() -> None:
    info_bridge.put(items={"a": 1})
    snap = info_bridge.snapshot()
    snap["items"]["a"] = 9
    snap["other"] = 2
    assert info_bridge.get("items")["a"] == 9
    assert info_bridge.get("other", None) is None


def test_train_task_manager_push_pull_matches_bridge() -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=1, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config()],
        learner_configs=[_learner_config()],
    )
    try:
        task_manager.push_info({"k": 42})
        assert task_manager.pull_info("k") == 42
        assert task_manager.pull_info("missing") is None
        assert task_manager.pull_info()["k"] == 42
        assert info_bridge.get("k") == 42
    finally:
        task_manager.close()


def test_stage_proxy_push_pull_without_task_manager() -> None:
    stage_task = StageTask(StageTaskConfig(stages=[
        StageConfig(
            stage_name="stage_0",
            agents_config=[_agent_config()],
            learners_config=[_learner_config()],
        ),
    ]))
    try:
        proxy = StageTaskManagerProxy(stage_task)
        assert stage_task.current_task_manager is None
        proxy.push_info({"mmap": 7})
        assert proxy.pull_info("mmap") == 7
        assert proxy.pull_info("missing") is None
        assert info_bridge.get("mmap") == 7
    finally:
        stage_task.close()


def test_dispatch_allowlist_includes_push_pull() -> None:
    methods = build_train_task_manager_methods(MagicMock())
    assert "push_info" in methods
    assert "pull_info" in methods


def test_publish_training_info_puts_all_configs() -> None:
    conf = MagicMock()
    publish_training_info(conf)
    packed = info_bridge.get("toml_config")
    assert packed["env_config"] is conf.env_config
    assert packed["stage_task_config"] is conf.stage_task_config
    assert packed["train_config"] is conf.train_config
    assert packed["replay_buffer_config"] is conf.replay_buffer_config
    assert packed["agents_config"] is conf.agents_config
    assert packed["learners_config"] is conf.learners_config


def test_public_training_info_excludes_private_keys() -> None:
    from common.info_bridge import INFO_IS_MASTER, INFO_RAW_CONFIG_FILE, public_training_info

    info_bridge.put(**{INFO_IS_MASTER: True, INFO_RAW_CONFIG_FILE: "secret"})
    assert public_training_info(INFO_IS_MASTER) is None
    assert public_training_info(INFO_RAW_CONFIG_FILE) is None
    snap = public_training_info()
    assert INFO_IS_MASTER not in snap
    assert INFO_RAW_CONFIG_FILE not in snap


def test_lazy_memmap_reads_budget_from_bridge(tmp_path) -> None:
    config = ReplayBufferConfig(
        agent_capacities={"agent": 2},
        batch_size=1,
        tensor_pool="lazy_memmap",
        storage=LocalStorageConfig(data_path=str(tmp_path), scratch_subdir="scratch"),
    )
    with info_bridge.scope(toml_config={
        "env_config": SimpleNamespace(max_step_per_episode=50),
        "stage_task_config": SimpleNamespace(
            stages=(SimpleNamespace(rehearsal_archive_ratio=0.5),),
        ),
    }):
        buffer = ReplayBuffer(config)
        try:
            assert buffer.tensor_pool._max_steps == 200
        finally:
            buffer.close()


def test_lazy_memmap_requires_bridge_budget(tmp_path) -> None:
    config = ReplayBufferConfig(
        agent_capacities={"agent": 2},
        batch_size=1,
        tensor_pool="lazy_memmap",
        storage=LocalStorageConfig(data_path=str(tmp_path), scratch_subdir="scratch"),
    )
    with pytest.raises(ValueError, match="info_bridge toml_config"):
        ReplayBuffer(config)

"""Role 接线的录制回归：实际写入 MsgPack，并隔离历史输入与评估输出。"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import (
    AgentConfig, EnvConfig, EvalConfig, LearnerConfig, MonteCarloMethodConfig,
    ReplayBufferConfig, SampleLearningEvalConfig, SampleLearningTasksConfig, TrainConfig,
)
from envs.base import BaseEnvironment
from roles import sample_learning_evaluation as eval_module
from roles import serial_train as serial_module
from scheduler.model import AgentCompose


@pytest.fixture
def components(monkeypatch):
    agent = MagicMock()
    agent.name = "solo"
    agent.agent_list = ["solo"]
    learner = MagicMock()
    learner.name = "solo"
    env = SimpleNamespace(
        record_caller=None, possible_agents=["solo"], sim=None,
        episode_num=1, step_num=1, max_step_per_episode=1,
        current_actions={"solo": {"move": 1}},
        post_state=SimpleNamespace(to_dict=lambda: {"position": 1}),
        close=MagicMock(),
    )
    manager = MagicMock()
    manager.task_id = "record-test"
    manager.stage_name = "train"
    for module in (serial_module, eval_module):
        monkeypatch.setattr(module, "TrainTaskManager", lambda **kwargs: manager)
        monkeypatch.setattr(module, "create_learner", lambda **kwargs: learner)
    monkeypatch.setattr(eval_module, "create_agent", lambda **kwargs: agent)
    monkeypatch.setattr(eval_module, "create_env", lambda **kwargs: env)
    monkeypatch.setattr(eval_module, "align_agent_config_with_learner_config", lambda *args: None)
    monkeypatch.setattr(serial_module, "build_policy_agents", lambda *args: [agent])
    monkeypatch.setattr(serial_module, "build_env", lambda **kwargs: env)
    monkeypatch.setattr(serial_module, "align_agent_configs", lambda *args: None)
    monkeypatch.setattr(
        serial_module, "build_agent_compose",
        lambda *args: AgentCompose(compose_type="order", step_agent=[agent]),
    )
    return env, manager


def _configs():
    return dict(
        env_config=EnvConfig(use_sim=False, agent_names=["solo"], state_dim=1, max_step_per_episode=1),
        sim_config=None,
        replay_buffer_config=ReplayBufferConfig(batch_size=1),
    )


def _record_episode(env):
    BaseEnvironment.record_sample(env, reset=True)
    BaseEnvironment.record_sample(env)


@pytest.mark.parametrize("record_sample", [False, True])
def test_serial_train_records_environment_samples(tmp_path, components, record_sample):
    env, _ = components
    mc_config = MonteCarloMethodConfig(tmp_local_dir=str(tmp_path), write_enabled=False)
    role = serial_module.SerialTrainRole(
        config=TrainConfig(record_sample=record_sample, monte_carlo_method_config=mc_config),
        step_agents_config=[AgentConfig(name="solo", agent_names=["solo"], state_shape=1)],
        learners_config=[LearnerConfig(name="solo", state_shape=1)],
        **_configs(),
    )
    try:
        _record_episode(env)
    finally:
        role.close()
    if record_sample:
        records = list(role.monte_carlo_method.local_file_reader())
        assert [record["step"] for record in records] == [0, 1]
        assert records[1]["actions"] == {"solo": {"move": 1}}
        assert role.monte_carlo_method.file.closed
    else:
        assert env.record_caller is None
        assert list(tmp_path.iterdir()) == []
    assert mc_config.write_enabled is False


def _evaluation_role(tmp_path, record_sample, explicit_config):
    source = tmp_path / "history.bin"
    source.write_bytes(b"history must remain unchanged")
    read_config = MonteCarloMethodConfig(
        tmp_local_dir=str(tmp_path), load_sample_name=source.name,
        load_file_path=str(source), write_enabled=True,
    )
    # 模拟 TOML 顶层 MC 配置同时被 train/eval 继承；写入器必须隔离读路径。
    eval_config = EvalConfig(
        record_sample=record_sample, max_episode=1, output_dir=str(tmp_path / "eval"),
        monte_carlo_method_config=read_config if explicit_config else None,
    )
    role = eval_module.SampleLearningEvaluationRole(
        config=SampleLearningEvalConfig(save_model=False),
        agent_config=AgentConfig(name="solo", agent_names=["solo"], state_shape=1),
        learner_config=LearnerConfig(name="solo", state_shape=1),
        train_config=TrainConfig(), eval_config=eval_config,
        monte_carlo_method_config=read_config,
        sample_learning_config=SampleLearningTasksConfig(),
        **_configs(),
    )
    return role, source, read_config


@pytest.mark.parametrize("record_sample", [False, True])
@pytest.mark.parametrize("explicit_config", [False, True])
def test_sample_learning_eval_records_to_separate_file(
        tmp_path, monkeypatch, components, record_sample, explicit_config,
):
    env, manager = components
    role, source, read_config = _evaluation_role(tmp_path, record_sample, explicit_config)
    manager.is_sample_completed.side_effect = [False, True]
    monkeypatch.setattr(role, "sample_learning", lambda: [])
    monkeypatch.setattr(role, "_draw_rewards_png", lambda: None)
    monkeypatch.setattr(role, "eval_ep_end_log", lambda episode: None)
    monkeypatch.setattr(
        role.eval_rollout_worker, "rollout_one_episode",
        lambda **kwargs: (_record_episode(env), object())[1],
    )
    assert role.run() == []
    assert env.record_caller is None
    assert source.read_bytes() == b"history must remain unchanged"
    assert read_config.write_enabled is True
    assert read_config.load_file_path == str(source)
    assert role.sample_monte_carlo_method.write_enabled is False
    if record_sample:
        recorder = role.eval_monte_carlo_method
        assert Path(recorder.local_file) != source
        assert [r["step"] for r in recorder.local_file_reader()] == [0, 1]
        assert recorder.file.closed
        entries = role.artifacts_manager.list_entries(manager.task_id, manager.stage_name)
        assert any(entry.producer_path == recorder.local_file for entry in entries)
    else:
        assert role.eval_monte_carlo_method is None
        assert list(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("failure_phase", ["learn", "eval"])
def test_sample_learning_eval_closes_recording_on_failure(tmp_path, monkeypatch, components, failure_phase):
    env, manager = components
    role, _, _ = _evaluation_role(tmp_path, True, False)
    manager.is_sample_completed.return_value = False
    monkeypatch.setattr(role, "sample_learning", lambda: [])

    def fail(**kwargs):
        raise RuntimeError("test failure")

    if failure_phase == "learn":
        monkeypatch.setattr(role, "sample_learning", fail)
    else:
        monkeypatch.setattr(role.eval_rollout_worker, "rollout_one_episode", fail)
    with pytest.raises(RuntimeError, match="test failure"):
        role.run()
    assert role.eval_monte_carlo_method.file.closed
    assert env.record_caller is None

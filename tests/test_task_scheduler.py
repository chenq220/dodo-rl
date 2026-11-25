"""
任务调度单元测试：TrainTaskManager 与分阶段训练（StageTask）。

覆盖三个基本用例：
1. 没有 StageTask 时，TrainTaskManager 可以独立初始化（普通训练就是单阶段 train）。
2. 两个配置不同的阶段：StageTask 为每个阶段实例化独立的 TrainTaskManager，
   实例 ID 不同、环境与 Learner 的 stage_name 不同、Learner 超参不同。
3. 两个阶段：阶段一 AgentA 固定、AgentB 学习；阶段二 AgentB 固定、AgentA 学习。
   两阶段所有 Agent 都参与采样，仅通过是否配置 Learner 区分固定/学习。
"""
from __future__ import annotations

import logging
import csv
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from scheduler.artifact import Artifact, ArtifactType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    AgentConfig,
    AgentType,
    AlgorithmType,
    EvalConfig,
    LearnerConfig,
    PpoConfig,
    StageConfig,
    StageTaskConfig,
    TrainConfig,
)
from scheduler.stage_task import StageTask, StageTaskManagerProxy
from scheduler.model import AutoMLAdvice, LearnTaskResult
from scheduler.train_task import LEASE_KIND_SAMPLE, TrainTaskManager
from sample.model import ReplayBufferState
from utils.random_seed import derive_seed
from utils.stage_utils import apply_stage_to_resources


class _FakeEnv:
    """轻量假环境，只承载 stage_name 属性，模拟一个环境实例描述所有阶段。"""

    def __init__(self):
        self.stage_name = "train"
        self.env_metadata = {}

    def apply_env_metadata(self, overrides: dict) -> None:
        self.env_metadata.update(overrides)


class _FakeLearner:
    """轻量假 Learner，记录 name/env/超参，验证阶段切换对 Learner 的影响。"""

    def __init__(self, name: str, env: _FakeEnv):
        self.name = name
        self.env = env
        self.config = SimpleNamespace(algo_config=None)
        self.models: dict = {}

    def dynamic_modify_config(self, updates: dict) -> None:
        if isinstance(updates, dict):
            self.config.algo_config = updates

    def reset_generation(self) -> None:
        """匹配真实 Learner 的阶段切换契约。"""
        return None


class _GateTaskManager:
    """只实现阶段就绪门闩测试需要的TaskManager接口。"""

    def __init__(self, token: str, sample_completed: bool = False):
        self.token = token
        self.claim_count = 0
        self.closed = False
        self.reports = []
        self.sample_completed = sample_completed

    def get_sample_task_info(self, **_kwargs):
        self.claim_count += 1
        return self.token

    def sample_backpressure_state(self, sample_role_name: str = "unknown"):
        return {"enabled": False, "reason": None, "sample_role_name": sample_role_name}

    def is_sample_completed(self):
        return self.sample_completed

    def get_learn_task_info(self):
        return f"learn-{self.token}"

    def accept_sample_task_result(self, episode_num: int, lease_id: str) -> bool:
        return bool(episode_num and lease_id)

    def report_sample_result(self, episode_num: int, **kwargs):
        self.reports.append((episode_num, kwargs))

    def close(self):
        self.closed = True


def _agent_config(name: str) -> AgentConfig:
    return AgentConfig(name=name, agent_type=AgentType.SB3.value, agent_names=[name])


def _learner_config(
        name: str,
        lr: float,
        *,
        batch_size: int = 10,
        learn_times: int = 1,
) -> LearnerConfig:
    learner_config = LearnerConfig(name=name, algo_type=AlgorithmType.PPO.value)
    assert isinstance(learner_config.algo_config, PpoConfig)
    learner_config.algo_config.lr = lr
    learner_config.algo_config.learn_times_per_episode = learn_times
    learner_config.algo_config.batch_size = batch_size
    return learner_config


def _report_episode(task_manager, episode_num, **kwargs):
    """测试辅助：未带租约时按生产协议为主路令牌签发租约后再上报。"""
    if "lease_id" not in kwargs:
        with task_manager._lock:
            task = task_manager.sample_task_seq[episode_num - 1]
            kwargs["lease_id"] = task_manager._issue_lease_locked(
                task, kind=LEASE_KIND_SAMPLE,
            )
    task_manager.report_sample_result(episode_num, **kwargs)


@pytest.mark.parametrize("fixed_agent_type", [AgentType.SB3.value, AgentType.MADDPG_MLP.value])
def test_fixed_mode_agent_uses_eval_action_params_in_training_token(fixed_agent_type: str) -> None:
    fixed_agent = AgentConfig(
        name="fixed_agent", agent_type=fixed_agent_type, agent_names=["fixed_agent"], fixed_mode=True,
    )
    learning_agent = _agent_config("learning_agent")
    config = TrainConfig(
        max_episode=1,
        start_epsilon=0.8,
        end_epsilon=0.2,
        epsilon_decay_episodes=100,
    )
    task_manager = TrainTaskManager(
        config=config,
        policy_agent_configs=[fixed_agent, learning_agent],
        learner_configs=[_learner_config("learning_agent", lr=1e-3)],
    )
    try:
        token = task_manager.get_sample_task_info(sample_role_name="sample-0")
        assert token is not None
        assert token.deterministic is False
        assert token.epsilon > 0
        assert token.actions_param["fixed_agent"].epsilon == 0.0
        assert token.actions_param["fixed_agent"].deterministic is True
        assert token.actions_param["learning_agent"].epsilon == token.epsilon
        assert token.actions_param["learning_agent"].deterministic is False
    finally:
        task_manager.close()


def test_task_manager_publishes_stage_and_episode_progress(tmp_path: Path, monkeypatch) -> None:
    """阶段和完成局数既写入本进程环境，也镜像给分布式启动脚本。"""
    progress_file = tmp_path / "training_progress.env"
    monkeypatch.setenv("DODO_PROGRESS_FILE", str(progress_file))
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=2, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
        stage_name="stage_alpha",
    )
    try:
        assert progress_file.read_text(encoding="utf-8") == (
            "DODO_TRAINING_STAGE=stage_alpha\n"
            "DODO_TRAINING_EPISODE=0/2\n"
        )
        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0})
        assert progress_file.read_text(encoding="utf-8") == (
            "DODO_TRAINING_STAGE=stage_alpha\n"
            "DODO_TRAINING_EPISODE=1/2\n"
        )
    finally:
        task_manager.close()


def test_stage_proxy_keeps_completed_state_while_switching() -> None:
    """ready 门闩关闭后，旧 TM 的完成状态仍要透出，供 sample role 提交产物。"""
    first = _GateTaskManager("stage-0-token", sample_completed=True)
    stage_task = StageTask(StageTaskConfig(stages=[
        StageConfig(
            stage_name="stage_0",
            agents_config=[_agent_config("agent_a")],
            learners_config=[_learner_config("agent_a", lr=1e-3)],
        ),
    ]))
    stage_task._build_task_manager = lambda _stage: first
    assert next(stage_task) is first
    proxy = StageTaskManagerProxy(stage_task)
    stage_task.mark_current_stage_unready(first)
    assert stage_task.ready_task_manager is None
    assert proxy.is_sample_completed() is True


def test_stage_task_proxy_blocks_tokens_until_current_stage_is_ready() -> None:
    first = _GateTaskManager("stage-0-token")
    second = _GateTaskManager("stage-1-token")
    stage_task = StageTask(StageTaskConfig(stages=[
        StageConfig(
            stage_name="stage_0",
            agents_config=[_agent_config("agent_a")],
            learners_config=[_learner_config("agent_a", lr=1e-3)],
        ),
        StageConfig(
            stage_name="stage_1",
            agents_config=[_agent_config("agent_a")],
            learners_config=[_learner_config("agent_a", lr=1e-3)],
        ),
    ]))
    managers = iter([first, second])
    stage_task._build_task_manager = lambda _stage: next(managers)
    assert next(stage_task) is first
    proxy = StageTaskManagerProxy(stage_task)

    try:
        # 中心RPC已启动但阶段资源尚未应用完成时，不允许领取采样或学习令牌。
        assert proxy.get_sample_task_info(sample_role_name="sample-0") is None
        blocked_learn = proxy.get_learn_task_info()
        assert blocked_learn.learn_info is None
        assert blocked_learn.task_finished is False
        assert proxy.is_sample_completed() is False
        assert proxy.sample_backpressure_state()["stage_ready"] is False
        assert proxy.accept_sample_task_result(1, "lease-1") is False
        assert first.claim_count == 0

        # 就绪门闩属于StageTask：准备上下文内保持关闭，完整退出后自动开放。
        with stage_task.preparing_current_stage(first):
            assert stage_task.ready_task_manager is None
            assert proxy.get_sample_task_info(sample_role_name="sample-0") is None
        assert stage_task.ready_task_manager is first
        assert proxy.get_sample_task_info(sample_role_name="sample-0") == "stage-0-token"
        assert proxy.accept_sample_task_result(1, "lease-1") is True
        proxy.report_sample_result(
            1,
            total_rewards={"agent_a": 1.0},
            episode_rewards_details={"bonus": 2.0},
            lease_id="lease-1",
        )
        assert first.reports == [(
            1,
            {
                "total_rewards": {"agent_a": 1.0},
                "success": True,
                "artifacts": None,
                "episode_metrics": None,
                "episode_rewards_details": {"bonus": 2.0},
                "lease_id": "lease-1",
                "step_count": 0,
            },
        )]

        # 推进阶段会先关闭旧实例并清除ready状态，新阶段不会继承旧阶段令牌入口。
        assert next(stage_task) is second
        assert first.closed is True
        assert stage_task.ready_task_manager is None
        assert proxy.get_sample_task_info(sample_role_name="sample-0") is None
        assert second.claim_count == 0

        # 资源准备异常时StageTask必须继续保持关闭。
        with pytest.raises(RuntimeError, match="prepare failed"):
            with stage_task.preparing_current_stage(second):
                raise RuntimeError("prepare failed")
        assert stage_task.ready_task_manager is None

        with stage_task.preparing_current_stage(second):
            pass
        assert proxy.get_sample_task_info(sample_role_name="sample-0") == "stage-1-token"
        stage_task.mark_current_stage_unready(second)
        assert proxy.get_sample_task_info(sample_role_name="sample-0") is None
    finally:
        stage_task.close()


def test_stage_task_proxy_only_finishes_learning_after_final_stage() -> None:
    class _CompletionTaskManager(_GateTaskManager):
        def __init__(self, token: str):
            super().__init__(token)
            self.sample_finished = False
            self.learning_finished = False

        def get_learn_task_info(self):
            return LearnTaskResult(task_finished=self.learning_finished)

        def is_task_finished(self):
            return self.sample_finished

        def is_sample_completed(self):
            return self.sample_finished

        def is_learning_completed(self):
            return self.learning_finished

    first = _CompletionTaskManager("stage-0-token")
    second = _CompletionTaskManager("stage-1-token")
    stage_task = StageTask(StageTaskConfig(stages=[
        StageConfig(
            stage_name="stage_0",
            agents_config=[_agent_config("agent_a")],
            learners_config=[_learner_config("agent_a", lr=1e-3)],
        ),
        StageConfig(
            stage_name="stage_1",
            agents_config=[_agent_config("agent_a")],
            learners_config=[_learner_config("agent_a", lr=1e-3)],
        ),
    ]))
    managers = iter([first, second])
    stage_task._build_task_manager = lambda _stage: next(managers)
    proxy = StageTaskManagerProxy(stage_task)

    try:
        assert next(stage_task) is first
        with stage_task.preparing_current_stage(first):
            pass
        first.sample_finished = True
        first.learning_finished = True
        assert proxy.is_learning_completed() is False
        assert proxy.get_learn_task_info().task_finished is False

        assert next(stage_task) is second
        with stage_task.preparing_current_stage(second):
            pass
        second.learning_finished = True
        assert proxy.is_learning_completed() is False
        assert proxy.get_learn_task_info().task_finished is False

        second.sample_finished = True
        assert proxy.is_learning_completed() is True
        assert proxy.get_learn_task_info().task_finished is True
    finally:
        stage_task.close()


def test_task_manager_initializes_without_stage_task() -> None:
    """用例1：没有 StageTask，TrainTaskManager 独立初始化，默认阶段 train。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=4, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        assert task_manager.stage_name == "train"
        assert task_manager.sample_task_seq[0].stage_name == "train"
        learn_info = _issue_and_claim_learn_token(task_manager, 1)
        assert learn_info.stage_name == "train"
        assert task_manager.is_task_finished() is False
        print(
            f"[case1] task_id={task_manager.task_id}, stage_name={task_manager.stage_name}, "
            f"max_episode={task_manager.max_episode}"
        )
    finally:
        task_manager.close()


def test_task_env_metadata_is_copied_to_train_and_mid_eval_tokens() -> None:
    """普通采样和盘中评估都使用当前阶段 metadata，且不会被后续配置修改污染。"""
    config = TrainConfig(
        max_episode=3,
        env_metadata={"reward_weights": {"navigation": 0.3}},
        mid_eval_enabled=True,
        mid_eval_episode_interval=1,
        sample_learn_max_lag=None,
    )
    task_manager = TrainTaskManager(
        config=config,
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
        stage_name="deploy_hold_archer",
    )
    try:
        assert task_manager.sample_task_seq[0].stage_name == "deploy_hold_archer"
        assert task_manager.sample_task_seq[0].env_metadata == config.env_metadata
        config.env_metadata["reward_weights"]["navigation"] = 0.9
        assert task_manager.sample_task_seq[0].env_metadata["reward_weights"]["navigation"] == 0.3

        task_manager.sample_task_completed = 1
        mid_eval = task_manager._maybe_build_mid_eval_task()
        assert mid_eval is not None
        assert mid_eval.stage_name == "deploy_hold_archer"
        assert mid_eval.env_metadata == config.env_metadata
        assert mid_eval.env_metadata.get("default_stage") != "course01_evaluation"
    finally:
        task_manager.close()


def test_automl_restart_applies_nested_env_metadata_to_new_tokens() -> None:
    config = TrainConfig(
        max_episode=2,
        env_metadata={"reward_weights": {"kill": 1.0, "navigation": 0.2}},
        sample_learn_max_lag=None,
    )
    task_manager = TrainTaskManager(
        config=config,
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        result = task_manager.restart_training_task(
            reason="env_search",
            advice=AutoMLAdvice(
                decision="restart",
                env_metadata_overrides={"reward_weights": {"navigation": 0.45}},
            ),
        )
        assert result["pending_restart"] is False
        assert config.env_metadata == {
            "reward_weights": {"kill": 1.0, "navigation": 0.45},
        }
        assert task_manager.sample_task_seq[0].env_metadata == config.env_metadata
        snapshot = task_manager._collect_training_snapshot_locked()
        assert snapshot.current_env_metadata_overrides == {
            "reward_weights": {"navigation": 0.45},
        }
    finally:
        task_manager.close()


def test_stage_task_bootstraps_fixed_policy_learner() -> None:
    """课程可只训练一个角色，但仍构造另一个角色的继承策略 Learner。"""
    agent_a = _agent_config("agent_a")
    agent_b = _agent_config("agent_b")
    learner_a = _learner_config("agent_a", lr=1e-3)
    learner_b = _learner_config("agent_b", lr=2e-3)
    stage_task = StageTask(StageTaskConfig(
        stages=[StageConfig(
            stage_name="only_b_learns",
            train_config=TrainConfig(max_episode=2),
            agents_config=[agent_a, agent_b],
            learners_config=[learner_b],
        )],
        bootstrap_learners_config=[learner_a, learner_b],
    ))
    try:
        assert {config.name for config in stage_task.all_learner_configs} == {"agent_a", "agent_b"}
        stage_task.start_next_stage()
        assert [config.name for config in stage_task.current_stage.learners_config] == ["agent_b"]
    finally:
        stage_task.close()


def test_task_manager_skips_single_point_reward_curve(tmp_path: Path) -> None:
    """单个 reward 点不落盘，避免生成看起来像空白图的退化曲线。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=4, sample_learn_max_lag=None, result_save_dir=str(tmp_path)),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0})
        assert task_manager.save_reward_curve() is None
        assert not list(tmp_path.glob("reward_*.png"))

        _report_episode(task_manager,2, total_rewards={"agent_a": 2.0})
        image_path = task_manager.save_reward_curve()
        assert image_path is not None
        assert Path(image_path).exists()
        csv_path = tmp_path / "raw_data" / (Path(image_path).stem + ".csv")
        with csv_path.open(encoding="utf-8", newline="") as stream:
            assert list(csv.DictReader(stream)) == [
                {"episode": "1", "series": "agent_a", "reward": "1.0"},
                {"episode": "2", "series": "agent_a", "reward": "2.0"},
            ]
        assert task_manager._reward_curve_file is not None
        assert not task_manager._reward_curve_file.closed
        reward_curve_file = task_manager._reward_curve_file

        _report_episode(task_manager,3, total_rewards={"agent_a": 3.0})
        assert task_manager.save_reward_curve() == image_path
        with csv_path.open(encoding="utf-8", newline="") as stream:
            assert len(list(csv.DictReader(stream))) == 3
        assert task_manager._reward_curve_file is reward_curve_file
    finally:
        reward_curve_file = task_manager._reward_curve_file
        task_manager.close()
        if reward_curve_file is not None:
            assert reward_curve_file.closed


@pytest.mark.parametrize("plot_method", ["save_reward_curve", "save_reward_details"])
def test_task_completion_waits_for_reward_plot_registration(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plot_method: str,
) -> None:
    """结束归档必须等在途奖励图登记完成，完成后不得再次写图。"""
    import threading
    from config import ArtifactsManagerConfig
    from scheduler.artifact import ArtifactsManager

    monkeypatch.setattr(TrainTaskManager, "_start_monitors", lambda self: None)
    artifacts = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    task_manager = TrainTaskManager(
        config=EvalConfig(
            max_episode=2,
            result_save_dir=str(tmp_path / "results"),
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[],
        artifacts_manager=artifacts,
    )
    for episode in (1, 2):
        _report_episode(
            task_manager, episode,
            total_rewards={"agent_a": float(episode)},
            episode_rewards_details={"bonus": float(episode)},
        )

    registration_started = threading.Event()
    allow_registration = threading.Event()
    closing_started = threading.Event()
    handles_closed = threading.Event()
    failures = []
    outputs = []
    produced_bytes = []
    original_put = task_manager.put_runtime_file
    original_close = task_manager._close_result_artifact_handles

    def delayed_put(*args, **kwargs):
        produced_bytes.append(Path(args[1]).read_bytes())
        registration_started.set()
        assert allow_registration.wait(10)
        return original_put(*args, **kwargs)

    def close_handles():
        closing_started.set()
        original_close()
        handles_closed.set()

    def run(callback):
        try:
            outputs.append(callback())
        except BaseException as exc:
            failures.append(exc)

    monkeypatch.setattr(task_manager, "put_runtime_file", delayed_put)
    monkeypatch.setattr(task_manager, "_close_result_artifact_handles", close_handles)
    producer = threading.Thread(target=run, args=(getattr(task_manager, plot_method),), daemon=True)
    completion = threading.Thread(target=run, args=(task_manager.notify_task_complete,), daemon=True)
    try:
        producer.start()
        assert registration_started.wait(10)
        completion.start()
        assert closing_started.wait(10)
        # 登记尚未完成，收口不能越过绘图锁关闭文件并归档。
        assert not handles_closed.wait(0.1)
    finally:
        allow_registration.set()
        producer.join(10)
        if completion.ident is not None:
            completion.join(10)
        task_manager.close()

    assert not producer.is_alive() and not completion.is_alive()
    assert not failures
    image_path = next(Path(value) for value in outputs if isinstance(value, str))
    archived = tmp_path / "archives" / task_manager.task_id / "results/train" / image_path.name
    assert archived.read_bytes() == produced_bytes[0]


def test_completed_task_never_restarts_reward_plotting(tmp_path: Path, monkeypatch) -> None:
    from config import ArtifactsManagerConfig
    from scheduler.artifact import ArtifactsManager

    monkeypatch.setattr(TrainTaskManager, "_start_monitors", lambda self: None)
    task_manager = TrainTaskManager(
        config=EvalConfig(
            max_episode=2, result_save_dir=str(tmp_path / "results"),
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[],
        artifacts_manager=ArtifactsManager(
            ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")),
        ),
    )
    try:
        for episode in (1, 2):
            _report_episode(
                task_manager, episode,
                total_rewards={"agent_a": float(episode)},
                episode_rewards_details={"bonus": float(episode)},
            )
        task_manager.notify_task_complete()
        assert task_manager.save_reward_curve() is None
        assert task_manager.save_reward_details() is None
        assert task_manager._reward_curve_file is None
        assert task_manager._reward_details_file is None

        def unexpected_plot():
            pytest.fail("completed task attempted another background plot")

        monkeypatch.setattr(task_manager, "_save_reward_plots", unexpected_plot)
        task_manager._reward_curve_monitor_thread()
    finally:
        task_manager.close()


def test_task_manager_reward_plot_interfaces_forward_custom_attributes(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 本例只测显式绘图参数，后台刷新不能覆盖 capture 中的最后一次调用。
    monkeypatch.setattr(TrainTaskManager, "_start_monitors", lambda self: None)
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=3, sample_learn_max_lag=None, result_save_dir=str(tmp_path)),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    curve_kwargs = {}
    details_kwargs = {}

    def capture_curve(*args, **kwargs):
        curve_kwargs.update(kwargs)

    def capture_details(*args, **kwargs):
        details_kwargs.update(kwargs)

    monkeypatch.setattr("scheduler.train_task.save_reward_curve", capture_curve)
    monkeypatch.setattr("scheduler.train_task.save_reward_details", capture_details)
    try:
        _report_episode(task_manager,
            1,
            total_rewards={"agent_a": 1.0},
            episode_rewards_details={"bonus": 1.0},
        )
        _report_episode(task_manager,
            2,
            total_rewards={"agent_a": 2.0},
            episode_rewards_details={"bonus": 2.0},
        )

        assert task_manager.save_reward_curve(
            y_axis_tick_interval=5,
            color_sequence=["#123456"],
        ) is not None
        assert task_manager.save_reward_details(
            y_axis_tick_interval=0.5,
            color_sequence=["#abcdef"],
        ) is not None
        assert curve_kwargs["y_axis_tick_interval"] == 5
        assert curve_kwargs["color_sequence"] == ["#123456"]
        assert details_kwargs["y_axis_tick_interval"] == 0.5
        assert details_kwargs["color_sequence"] == ["#abcdef"]
    finally:
        task_manager.close()


def test_task_manager_saves_episode_reward_details_by_real_episode_number(tmp_path: Path) -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=4, sample_learn_max_lag=None, result_save_dir=str(tmp_path)),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        _report_episode(task_manager,
            3,
            total_rewards={"agent_a": 3.0},
            episode_rewards_details={"explore": 4.0, "cost": -1.0},
        )
        assert task_manager.save_reward_details() is None
        _report_episode(task_manager,
            1,
            total_rewards={"agent_a": 1.0},
            episode_rewards_details={"explore": 2.0, "swing": -0.5},
        )

        image_path = task_manager.save_reward_details()

        assert image_path is not None
        assert Path(image_path).exists()
        assert Path(image_path).name.startswith("reward_details_")
        csv_path = tmp_path / "raw_data" / (Path(image_path).stem + ".csv")
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows == [
            {"episode": "1", "series": "explore", "reward": "2.0"},
            {"episode": "1", "series": "swing", "reward": "-0.5"},
            {"episode": "3", "series": "explore", "reward": "4.0"},
            {"episode": "3", "series": "cost", "reward": "-1.0"},
        ]
        assert task_manager.episode_rewards_details == [
            {"explore": 2.0, "swing": -0.5},
            None,
            {"explore": 4.0, "cost": -1.0},
            None,
        ]
        assert task_manager._reward_details_file is not None
        assert not task_manager._reward_details_file.closed
        snapshot = task_manager._collect_training_snapshot_locked()
        assert snapshot.artifact_paths["reward_details"] == [image_path]
    finally:
        reward_details_file = task_manager._reward_details_file
        task_manager.close()
        if reward_details_file is not None:
            assert reward_details_file.closed


def test_unsupported_reward_details_removes_only_details_image(tmp_path: Path) -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=3, sample_learn_max_lag=None, result_save_dir=str(tmp_path)),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        _report_episode(task_manager,
            1,
            total_rewards={"agent_a": 1.0},
            episode_rewards_details={"bonus": 1.0},
        )
        _report_episode(task_manager,
            2,
            total_rewards={"agent_a": 2.0},
            episode_rewards_details={"bonus": 2.0},
        )
        details_path = task_manager.save_reward_details()
        reward_path = task_manager.save_reward_curve()
        assert details_path is not None and Path(details_path).exists()
        assert reward_path is not None and Path(reward_path).exists()

        _report_episode(task_manager,
            3,
            total_rewards={"agent_a": 3.0},
            episode_rewards_details=None,
        )

        assert task_manager._reward_details_capability == "unsupported"
        assert task_manager.save_reward_details() is None
        assert not Path(details_path).exists()
        assert not (tmp_path / "raw_data" / (Path(details_path).stem + ".csv")).exists()
        assert Path(reward_path).exists()
    finally:
        task_manager.close()


def test_missing_reward_details_produces_only_reward_curve_and_raw_data(tmp_path: Path) -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=3, sample_learn_max_lag=None, result_save_dir=str(tmp_path)),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        for episode in (1, 2):
            _report_episode(task_manager, episode, total_rewards={"agent_a": float(episode)},
                            episode_rewards_details=None)
        reward_path = task_manager.save_reward_curve()
        assert reward_path is not None
        assert (tmp_path / "raw_data" / (Path(reward_path).stem + ".csv")).exists()
        assert task_manager.save_reward_details() is None
        assert not list(tmp_path.glob("reward_details_*.png"))
        assert not list((tmp_path / "raw_data").glob("reward_details_*.csv"))
    finally:
        task_manager.close()


def test_unsupported_reward_details_remove_same_run_stale_image(tmp_path: Path) -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=2, sample_learn_max_lag=None, result_save_dir=str(tmp_path)),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        stale_image_path = Path(
            task_manager._reward_details_image_path(task_manager.generation)
        )
        stale_image_path.parent.mkdir(parents=True, exist_ok=True)
        stale_image_path.write_bytes(b"stale")
        assert stale_image_path.exists()

        _report_episode(task_manager,
            1,
            total_rewards={"agent_a": 1.0},
            episode_rewards_details=None,
        )

        assert task_manager._reward_details_capability == "unsupported"
        assert not stale_image_path.exists()
        assert task_manager.save_reward_details() is None
    finally:
        task_manager.close()


def test_restart_resets_episode_reward_details_capability() -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=2, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        _report_episode(task_manager,
            1,
            total_rewards={"agent_a": 1.0},
            episode_rewards_details={"bonus": 1.0},
        )
        assert task_manager._reward_details_capability == "supported"

        task_manager.restart_training_task(reason="test")

        assert task_manager._reward_details_capability == "unknown"
        assert task_manager.episode_rewards_details == [None, None]
        assert task_manager._last_reward_details_path is None
    finally:
        task_manager.close()


# 单测超时用 5 秒；回收靠注入 now，不真睡。
_TEST_SAMPLE_TIMEOUT_SECONDS = 5
_TEST_SAMPLE_TIMEOUT_EXPIRED = timedelta(seconds=6)


def test_sample_task_timeout_reclaims_lease_and_discards_late_result() -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(
            max_episode=1,
            sample_learn_max_lag=None,
            sample_task_timeout=_TEST_SAMPLE_TIMEOUT_SECONDS,
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        first_claim = task_manager.get_sample_task_info()
        assert first_claim is not None
        assert first_claim.lease_id
        assert first_claim is not task_manager.sample_task_seq[0]

        reclaimed = task_manager._reclaim_timed_out_leases(
            now=first_claim.start_datetime + _TEST_SAMPLE_TIMEOUT_EXPIRED,
        )
        assert reclaimed == 1
        assert task_manager.sample_task_seq[0].start_datetime is None
        assert task_manager.sample_task_seq[0].failed_count == 1
        assert not task_manager.accept_sample_task_result(1, first_claim.lease_id)

        second_claim = task_manager.get_sample_task_info()
        assert second_claim is not None
        assert second_claim.lease_id != first_claim.lease_id

        _report_episode(task_manager,
            1,
            total_rewards={"agent_a": 99.0},
            lease_id=first_claim.lease_id,
        )
        assert task_manager.sample_task_completed == 0
        assert task_manager.sample_rewards["agent_a"] == [None]

        assert task_manager.accept_sample_task_result(1, second_claim.lease_id)
        _report_episode(task_manager,
            1,
            total_rewards={"agent_a": 1.0},
            lease_id=second_claim.lease_id,
        )
        assert task_manager.sample_task_completed == 1
        assert task_manager.sample_rewards["agent_a"] == [1.0]
    finally:
        task_manager.close()


def test_task_config_sample_timeout_defaults_to_300_seconds() -> None:
    assert TrainConfig().sample_task_timeout == 300.0


def test_sample_task_timeout_is_validated_by_task_manager() -> None:
    config = TrainConfig(sample_task_timeout=0)
    assert config.sample_task_timeout == 0
    with pytest.raises(AssertionError, match="sample_task_timeout"):
        TrainTaskManager(
            config=config,
            policy_agent_configs=[_agent_config("agent_a")],
            learner_configs=[_learner_config("agent_a", lr=1e-3)],
        )


def test_sample_timeout_monitor_skips_fresh_leases_and_reclaims_only_expired() -> None:
    """独立监控线程会启动；同一扫描里只回收超过时限、尚未结束的占用令牌。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(
            max_episode=3,
            sample_learn_max_lag=None,
            sample_task_timeout=_TEST_SAMPLE_TIMEOUT_SECONDS,
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        timer = task_manager._sample_task_monitor_timer
        assert timer is not None
        assert timer.name.startswith("sample_task_timeout_monitor_")

        claimed = [task_manager.get_sample_task_info() for _ in range(3)]
        assert [token.episode_num for token in claimed] == [1, 2, 3]
        now = claimed[2].start_datetime
        # 租约表是唯一在途视图：老化直接改 issued_at；第 3 张按已结算处理（租约随回报移除）。
        task_manager._lease_table[claimed[0].lease_id].issued_at = now - _TEST_SAMPLE_TIMEOUT_EXPIRED
        task_manager._lease_table[claimed[1].lease_id].issued_at = now
        task_manager.sample_task_seq[1].start_datetime = now
        task_manager._lease_table.pop(claimed[2].lease_id)
        task_manager.sample_task_seq[2].end_datetime = now

        reclaimed = task_manager._reclaim_timed_out_leases(now=now)
        assert reclaimed == 1
        assert claimed[0].lease_id not in task_manager._lease_table
        assert task_manager.sample_task_seq[0].start_datetime is None
        assert task_manager.sample_task_seq[0].failed_count == 1
        assert task_manager.sample_task_seq[1].start_datetime == now
        assert task_manager.sample_task_seq[2].end_datetime == now

        retry = task_manager.get_sample_task_info()
        assert retry is not None
        assert retry.episode_num == 1
        assert retry.lease_id != claimed[0].lease_id
    finally:
        task_manager.close()


def test_failed_sample_token_is_returned_and_nearest_available_hole_is_researched() -> None:
    """仿真失败 success=False 归还令牌后，TM 在序列中搜索最近可用槽，而不是只按游标递增。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=3, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        claimed = [task_manager.get_sample_task_info() for _ in range(3)]
        assert [token.episode_num for token in claimed] == [1, 2, 3]
        assert task_manager.get_sample_task_info() is None

        _report_episode(task_manager,
            claimed[1].episode_num,
            total_rewards={},
            success=False,
            lease_id=claimed[1].lease_id,
        )
        hole = task_manager.sample_task_seq[1]
        assert hole.start_datetime is None
        assert hole.end_datetime is None
        assert hole.failed_count == 1
        assert task_manager.sample_task_completed == 0
        assert task_manager.sample_rewards["agent_a"] == [None, None, None]

        retry = task_manager.get_sample_task_info()
        assert retry is not None
        assert retry.episode_num == 2
        assert retry.lease_id != claimed[1].lease_id
        assert retry is not hole

        _report_episode(task_manager,
            retry.episode_num,
            total_rewards={"agent_a": 1.5},
            lease_id=retry.lease_id,
        )
        assert task_manager.sample_task_completed == 1
        assert task_manager.sample_rewards["agent_a"][1] == 1.5
        assert task_manager.get_sample_task_info() is None
    finally:
        task_manager.close()


def test_stale_failure_report_does_not_release_reissued_lease() -> None:
    """超时回收并改发给别人后，旧租约的失败回报必须丢弃，不能把新令牌再打回可用。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(
            max_episode=1,
            sample_learn_max_lag=None,
            sample_task_timeout=_TEST_SAMPLE_TIMEOUT_SECONDS,
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        first = task_manager.get_sample_task_info()
        task_manager._reclaim_timed_out_leases(
            now=first.start_datetime + _TEST_SAMPLE_TIMEOUT_EXPIRED,
        )
        second = task_manager.get_sample_task_info()
        assert second.lease_id != first.lease_id

        _report_episode(task_manager,
            first.episode_num,
            total_rewards={},
            success=False,
            lease_id=first.lease_id,
        )
        assert task_manager.sample_task_seq[0].start_datetime is not None
        assert task_manager.sample_task_seq[0].lease_id == second.lease_id
        assert task_manager.get_sample_task_info() is None
    finally:
        task_manager.close()


def test_dual_sample_rewards_tables_default_to_learners() -> None:
    """全场写入 sample_rewards；下游默认读 learn_sample_rewards。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=4, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a"), _agent_config("agent_b")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        assert set(task_manager.sample_rewards) == {"agent_a", "agent_b"}
        assert set(task_manager.learn_sample_rewards) == {"agent_a"}
        _report_episode(task_manager,
            1,
            total_rewards={"agent_a": 1.0, "agent_b": 9.0},
        )
        assert task_manager.sample_rewards["agent_a"][0] == 1.0
        assert task_manager.sample_rewards["agent_b"][0] == 9.0
        assert task_manager.learn_sample_rewards["agent_a"][0] == 1.0
        assert "agent_b" not in task_manager.learn_sample_rewards

        downstream = task_manager._downstream_sample_rewards()
        assert set(downstream) == {"agent_a"}
        assert downstream["agent_a"] == [1.0, None, None, None]
    finally:
        task_manager.close()


def test_task_manager_records_episode_business_metrics() -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=2, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        _report_episode(task_manager,
            1,
            total_rewards={"agent_a": 0.5},
            episode_metrics={"stage_success": 1.0},
        )
        _report_episode(task_manager,2, total_rewards={"agent_a": 0.0})
        snapshot = task_manager._collect_training_snapshot_locked()
        assert snapshot.episode_metrics == {"stage_success": [1.0, None]}
    finally:
        task_manager.close()


def test_stage_success_rate_is_collected_from_standard_infos() -> None:
    from types import SimpleNamespace
    from config import AutoMLConfig

    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=1, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        task_manager._auto_ml = SimpleNamespace(config=AutoMLConfig(
            subject_name="success", metric_primary="stage_success_rate",
        ))
        payload = task_manager._metric_outputs_payload()["stage_success"]
        assert payload["info_path"] == "stage_success"
        assert payload["episode_reduce"] == "latest"
    finally:
        task_manager.close()


def test_downstream_uses_all_agents_when_monitor_flag_enabled() -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=2, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a"), _agent_config("agent_b")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        task_manager.monitor_all_agent_rewards = True
        _report_episode(task_manager,
            1,
            total_rewards={"agent_a": 1.0, "agent_b": 2.0},
        )
        downstream = task_manager._downstream_sample_rewards()
        assert set(downstream) == {"agent_a", "agent_b"}
        assert downstream["agent_b"][0] == 2.0
    finally:
        task_manager.close()


def test_two_stages_produce_distinct_task_managers() -> None:
    """用例2：两阶段配置不同 -> TaskManager 实例不同、stage_name 不同、超参不同。"""
    stage_task_config = StageTaskConfig(stages=[
        StageConfig(
            stage_name="explore",
            train_config=TrainConfig(max_episode=4, sample_learn_max_lag=None),
            agents_config=[_agent_config("agent_a")],
            learners_config=[_learner_config("agent_a", lr=1e-3)],
        ),
        StageConfig(
            stage_name="exploit",
            train_config=TrainConfig(max_episode=6, sample_learn_max_lag=None),
            agents_config=[_agent_config("agent_a")],
            learners_config=[_learner_config("agent_a", lr=5e-4)],
        ),
    ])
    stage_task = StageTask(stage_task_config)
    tm_one = next(iter(stage_task))
    tm_two = next(iter(stage_task))
    try:
        assert tm_one is not tm_two
        assert tm_one.stage_name == "explore"
        assert tm_two.stage_name == "exploit"
        assert tm_one.max_episode == 4
        assert tm_two.max_episode == 6
        print(f"[case2] tm_one={id(tm_one)}, tm_two={id(tm_two)}")
    finally:
        stage_task.close()


def test_stage_task_starts_from_configured_stage_and_fills_model_versions() -> None:
    stages = [
        StageConfig(
            stage_name=f"stage_{index}",
            train_config=TrainConfig(
                max_episode=index + 1,
                model_version=f"original_{index}",
                sample_learn_max_lag=None,
            ),
            agents_config=[_agent_config("agent_a")],
            learners_config=[_learner_config("agent_a", lr=1e-3)],
        )
        for index in range(3)
    ]
    config = StageTaskConfig(
        stages=stages,
        start_stage=1,
        stage_model_versions=[" model_1 ", "model_2"],
    )

    # config.py只声明配置结构；构造StageTask时才校验并填充运行期配置。
    assert [stage.train_config.model_version for stage in stages] == [
        "original_0", "original_1", "original_2",
    ]

    stage_task = StageTask(config)
    try:
        assert [stage.train_config.model_version for stage in stages] == [
            "original_0", "model_1", "model_2",
        ]
        assert stage_task.starting_stage is stages[1]

        first_task_manager = next(stage_task)
        assert stage_task.current_stage_index == 1
        assert stage_task.current_stage.stage_name == "stage_1"
        assert first_task_manager.task_config.model_version == "model_1"
        assert next(stage_task).stage_name == "stage_2"
        with pytest.raises(StopIteration):
            next(stage_task)
    finally:
        stage_task.close()


def test_stage_task_empty_model_version_clears_stage_value() -> None:
    stage = StageConfig(
        train_config=TrainConfig(model_version="original", sample_learn_max_lag=None),
        agents_config=[_agent_config("agent_a")],
        learners_config=[_learner_config("agent_a", lr=1e-3)],
    )

    StageTask(StageTaskConfig(stages=[stage], stage_model_versions=[" "])).close()

    assert stage.train_config.model_version is None


@pytest.mark.parametrize("start_stage", [-1, 2, True, "1"])
def test_stage_task_rejects_invalid_start_stage_at_runtime(start_stage) -> None:
    config = StageTaskConfig(
        stages=[StageConfig(), StageConfig(stage_name="stage1")],
        start_stage=start_stage,
    )

    with pytest.raises((TypeError, ValueError), match="start_stage"):
        StageTask(config)


def test_stage_task_rejects_invalid_stage_model_versions_at_runtime() -> None:
    stage = StageConfig(
        agents_config=[_agent_config("agent_a")],
        learners_config=[_learner_config("agent_a", lr=1e-3)],
    )

    with pytest.raises(ValueError, match="more entries than runnable stages"):
        StageTask(StageTaskConfig(stages=[stage], stage_model_versions=["v0", "v1"]))
    with pytest.raises(TypeError, match=r"stage_model_versions\[0\]"):
        StageTask(StageTaskConfig(stages=[stage], stage_model_versions=[1]))


def test_stage_switch_updates_learner_hyperparams() -> None:
    """用例3：阶段一 AgentB 学习；阶段二 AgentA 学习；Agent 都参与采样。"""
    env = _FakeEnv()
    learner_a = _FakeLearner("agent_a", env)
    learner_b = _FakeLearner("agent_b", env)
    learners_by_name = {"agent_a": learner_a, "agent_b": learner_b}

    stage_one = StageConfig(
        stage_name="stage_a_learns",
        train_config=TrainConfig(max_episode=3, sample_learn_max_lag=None),
        agents_config=[_agent_config("agent_a"), _agent_config("agent_b")],
        learners_config=[_learner_config("agent_b", lr=2e-3)],
    )
    stage_two = StageConfig(
        stage_name="stage_b_learns",
        train_config=TrainConfig(max_episode=5, sample_learn_max_lag=None),
        agents_config=[_agent_config("agent_a"), _agent_config("agent_b")],
        learners_config=[_learner_config("agent_a", lr=3e-3)],
    )

    apply_stage_to_resources(stage_one, env=env, learners_by_name=learners_by_name)
    assert env.stage_name == "stage_a_learns"
    assert learner_b.config.algo_config["lr"] == 2e-3

    apply_stage_to_resources(stage_two, env=env, learners_by_name=learners_by_name)
    assert env.stage_name == "stage_b_learns"
    assert learner_a.config.algo_config["lr"] == 3e-3
    print("[case3] stage switch hyperparams updated")


def test_stage_task_active_learners_follow_stage_config() -> None:
    """阶段切换后，学习令牌只发给当前阶段的 learners_config。"""
    stage_task_config = StageTaskConfig(stages=[
        StageConfig(
            stage_name="stage_one",
            train_config=TrainConfig(max_episode=2, sample_learn_max_lag=None),
            agents_config=[_agent_config("agent_a"), _agent_config("agent_b")],
            learners_config=[_learner_config("agent_b", lr=1e-3)],
        ),
        StageConfig(
            stage_name="stage_two",
            train_config=TrainConfig(max_episode=2, sample_learn_max_lag=None),
            agents_config=[_agent_config("agent_a"), _agent_config("agent_b")],
            learners_config=[_learner_config("agent_a", lr=2e-3)],
        ),
    ])
    stage_task = StageTask(stage_task_config)
    active_learners_per_stage = []
    for task_manager in stage_task:
        stage = stage_task.current_stage
        token_active = _issue_and_claim_learn_token(task_manager, 1).active_learners
        active_learners_per_stage.append({
            "stage_name": stage.stage_name,
            "stage_learners": [c.name for c in stage.learners_config],
            "token_active_learners": list(token_active),
        })
        print(f"[case3] {active_learners_per_stage[-1]}")
    stage_task.close()

    stage_one, stage_two = active_learners_per_stage
    assert stage_one["token_active_learners"] == ["agent_b"]
    assert stage_two["token_active_learners"] == ["agent_a"]


def test_restart_rebuilds_both_reward_tables() -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=2, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a"), _agent_config("agent_b")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0, "agent_b": 2.0})
        task_manager.restart_training_task(reason="test")
        assert task_manager.sample_rewards["agent_a"][0] is None
        assert task_manager.sample_rewards["agent_b"][0] is None
        assert set(task_manager.learn_sample_rewards) == {"agent_a"}
        assert task_manager.learn_sample_rewards["agent_a"][0] is None
    finally:
        task_manager.close()


def test_two_stages_swap_fixed_and_learning_agents() -> None:
    """用例3：阶段一 AgentB 学习；阶段二 AgentA 学习；所有 Agent 都参与采样。"""
    all_agent_names = ["agent_a", "agent_b"]
    stage_task_config = StageTaskConfig(stages=[
        StageConfig(
            stage_name="train_b",
            train_config=TrainConfig(max_episode=4, sample_learn_max_lag=None),
            agents_config=[_agent_config("agent_a"), _agent_config("agent_b")],
            learners_config=[_learner_config("agent_b", lr=1e-3)],
        ),
        StageConfig(
            stage_name="train_a",
            train_config=TrainConfig(max_episode=4, sample_learn_max_lag=None),
            agents_config=[_agent_config("agent_a"), _agent_config("agent_b")],
            learners_config=[_learner_config("agent_a", lr=1e-3)],
        ),
    ])
    stage_task = StageTask(stage_task_config)

    assert {cfg.name for cfg in stage_task.all_learner_configs} == {"agent_b", "agent_a"}
    assert {cfg.name for cfg in stage_task.all_agent_configs} == {"agent_a", "agent_b"}

    active_learners_per_stage = []
    try:
        for task_manager in stage_task:
            stage = stage_task.current_stage
            policy_agent_names = sorted(cfg.name for cfg in task_manager.policy_agent_configs)
            assert policy_agent_names == all_agent_names, (
                f"All agents must participate in sampling for stage={stage.stage_name}, "
                f"given={policy_agent_names}"
            )
            token_active = _issue_and_claim_learn_token(task_manager, 1).active_learners
            active_learners_per_stage.append({
                "stage_name": stage.stage_name,
                "policy_agents": policy_agent_names,
                "stage_learners": [c.name for c in stage.learners_config],
                "token_active_learners": list(token_active),
            })
            print(f"[case3] {active_learners_per_stage[-1]}")
    finally:
        stage_task.close()

    stage_one, stage_two = active_learners_per_stage
    assert stage_one["policy_agents"] == all_agent_names
    assert stage_two["policy_agents"] == all_agent_names
    assert stage_one["token_active_learners"] == ["agent_b"]
    assert stage_two["token_active_learners"] == ["agent_a"]
    assert stage_one["token_active_learners"] != stage_two["token_active_learners"]


def test_mid_eval_requires_enabled_flag() -> None:
    """仅配置 interval 而未开启 mid_eval_enabled 时，不应触发盘中评估。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(
            max_episode=800,
            sample_learn_max_lag=None,
            mid_eval_enabled=False,
            mid_eval_episode_interval=100,
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        task_manager.sample_task_completed = 100
        assert task_manager.should_trigger_mid_eval() is False
    finally:
        task_manager.close()


def test_returned_sample_token_can_bypass_backpressure_once() -> None:
    """失败令牌可穿透背压被领取，领取后其他采样进程重新受背压限制。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=10, sample_learn_max_lag=1),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        now = datetime.now()
        for index in (0, 2, 3, 4, 5, 6):
            task_manager.sample_task_seq[index].end_datetime = now
        returned = task_manager.sample_task_seq[1]
        returned.failed_count = 1
        returned.start_datetime = None
        returned.end_datetime = None
        task_manager.sample_task_completed = 6
        task_manager.current_sample_task = 1
        task_manager.learn_batches_update(_ready_buffer(size=100), 1)

        state = task_manager.sample_backpressure_state(sample_role_name="sample-0")
        assert state["lag"] == 6
        assert state["retry_episode"] == 2
        assert state["enabled"] is False

        retry = task_manager.get_sample_task_info()
        assert retry is not returned
        assert retry.episode_num == 2
        assert retry.lease_id == returned.lease_id

        claimed_state = task_manager.sample_backpressure_state(sample_role_name="sample-1")
        assert claimed_state["retry_episode"] is None
        assert claimed_state["enabled"] is True
    finally:
        task_manager.close()


def test_mid_eval_does_not_miss_milestones_when_parallel_reports_skip_exact_count() -> None:
    """并行采样从199跳到201时仍补发200里程碑，后续400、600同样不漏发。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(
            max_episode=1000,
            sample_learn_max_lag=None,
            mid_eval_enabled=True,
            mid_eval_episode_interval=200,
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        for completed, expected_milestone in ((201, 200), (402, 400), (603, 600)):
            task_manager.sample_task_completed = completed
            assert task_manager.should_trigger_mid_eval() is True
            mid_eval = task_manager._maybe_build_mid_eval_task()
            assert mid_eval is not None
            assert mid_eval.episode_num == expected_milestone
            assert mid_eval.metadata["replay_write"] is False
            assert mid_eval.metadata["record_render"] is True
            assert "mid_eval" not in mid_eval.metadata
            assert task_manager.should_trigger_mid_eval() is False
    finally:
        task_manager.close()


def test_mid_eval_filename_uses_system_local_time() -> None:
    """GIF 文件名由 TM 在签发旁路令牌时生成，使用本机本地时区。"""
    before = datetime.now().astimezone()
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=4, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
        stage_name="search_with_threats",
    )
    try:
        task_manager.task_id = "4f7fd96a"
        filename = task_manager._bypass_record_name(train_episode=1000, dispatch_count=5)
        after = datetime.now().astimezone()
    finally:
        task_manager.close()

    match = re.fullmatch(
        r"mid_eval_4f7fd96a_(\d{14})_search_with_threats_episode_1000_eval_5\.gif",
        filename,
    )
    assert match is not None
    file_time = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(
        tzinfo=before.tzinfo,
    )
    assert abs((file_time - before).total_seconds()) <= 2
    assert abs((after - file_time).total_seconds()) <= 2


def test_eval_gif_requires_enabled_flag() -> None:
    """仅配置 interval 而未开启 render_gif_enabled 时，评估令牌不录 GIF。"""
    task_manager = TrainTaskManager(
        config=EvalConfig(
            max_episode=25,
            render_gif_enabled=False,
            render_gif_episode_interval=10,
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[],
    )
    try:
        assert len(task_manager.sample_task_seq) == 25
        for task in task_manager.sample_task_seq:
            assert not bool((task.metadata or {}).get("record_render"))
            assert not (task.metadata or {}).get("record_name")
    finally:
        task_manager.close()


def test_eval_gif_stamps_milestone_tokens() -> None:
    """评估 GIF 由 TM 在签发令牌时打标；interval=10、max_episode=25 只录 10、20。"""
    task_manager = TrainTaskManager(
        config=EvalConfig(
            max_episode=25,
            render_gif_enabled=True,
            render_gif_episode_interval=10,
            render_video_dir="/tmp/eval-gifs",
            render_gif_fps=12,
            render_gif_frame_stride=2,
            render_gif_max_frames=30,
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[],
        stage_name="eval_stage",
    )
    try:
        recorded = []
        for task in task_manager.sample_task_seq:
            metadata = task.metadata or {}
            if bool(metadata.get("record_render")):
                recorded.append(task.episode_num)
                assert metadata["record_output_dir"] == "/tmp/eval-gifs"
                assert metadata["record_fps"] == 12
                assert metadata["record_frame_stride"] == 2
                assert metadata["record_max_frames"] == 30
                assert metadata["record_name"]
                assert str(metadata["record_name"]).endswith(
                    f"episode_{task.episode_num:04d}.gif"
                )
            else:
                assert not metadata.get("record_name")
        assert recorded == [10, 20]
    finally:
        task_manager.close()


def test_eval_gif_filename_uses_system_local_time() -> None:
    """评估 GIF 文件名由 TM 在签发令牌时生成，使用本机本地时区。"""
    before = datetime.now().astimezone()
    task_manager = TrainTaskManager(
        config=EvalConfig(max_episode=1, render_gif_enabled=True, render_gif_episode_interval=1),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[],
        stage_name="search_with_threats",
    )
    try:
        task_manager.task_id = "4f7fd96a"
        filename = task_manager._eval_record_name(episode_num=20)
        after = datetime.now().astimezone()
    finally:
        task_manager.close()

    match = re.fullmatch(
        r"eval_4f7fd96a_(\d{14})_search_with_threats_episode_0020\.gif",
        filename,
    )
    assert match is not None
    file_time = datetime.strptime(match.group(1), "%Y%m%d%H%M%S").replace(
        tzinfo=before.tzinfo,
    )
    assert abs((file_time - before).total_seconds()) <= 2
    assert abs((after - file_time).total_seconds()) <= 2


def test_restart_resets_automl_restart_episode_counter() -> None:
    """restart 后须清零 _last_automl_restart_episode，否则新一代监控会被永久跳过。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=800, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        task_manager.sample_task_completed = 575
        task_manager._last_automl_restart_episode = 575
        task_manager._last_automl_check_episode = 575

        with task_manager._lock:
            task_manager._restart_training_task_locked("reward_collapse")

        assert task_manager.generation == 1
        assert task_manager.sample_task_completed == 0
        assert task_manager._last_automl_restart_episode == 0
        assert task_manager._last_automl_check_episode == 0

        min_between = 10 if task_manager._auto_ml is None else task_manager._auto_ml.config.min_episodes_between_restart
        completed = 282
        last_restart = task_manager._last_automl_restart_episode
        blocked = last_restart > 0 and completed - last_restart < min_between
        assert blocked is False
    finally:
        task_manager.close()


def _ready_buffer(generation: int = 0, size: int = 100) -> ReplayBufferState:
    return ReplayBufferState(generation=generation, size=size)


def _issue_and_claim_learn_token(task_manager: TrainTaskManager, episode_num: int = 1, size: int = 100):
    task_manager.learn_batches_update(_ready_buffer(task_manager.generation, size=size), episode_num)
    result = task_manager.get_learn_task_info()
    assert result.learn_info is not None
    return result.learn_info


def test_warmup_does_not_issue_empty_permit_or_enable_backpressure() -> None:
    """ReplayBuffer 尚未就绪时不造空令牌，也不让 warmup 被学习背压锁死。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=4, sample_learn_max_lag=1),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3, batch_size=2, learn_times=3)],
    )
    try:
        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0})
        _report_episode(task_manager,2, total_rewards={"agent_a": 1.0})
        task_manager.learn_batches_update(_ready_buffer(size=0), 2)
        completed, expected = task_manager.learning_process
        assert completed == 0
        assert expected == 0
        result = task_manager.get_learn_task_info()
        assert result.learn_info is None
        assert result.task_finished is False
        backpressure = task_manager.sample_backpressure_state()
        assert backpressure["lag"] == 2
        assert backpressure["enabled"] is False
    finally:
        task_manager.close()


def test_multiple_replay_updates_coalesce_into_one_ready_permit() -> None:
    """多个并发 push 只形成一张 ready，不按 episode 积累学习债务。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=3, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3, batch_size=2, learn_times=1)],
    )
    try:
        for episode_num in (3, 1, 2):
            task_manager.learn_batches_update(_ready_buffer(size=3), episode_num)
        assert task_manager.learn_count == 1
        first = task_manager.get_learn_task_info()
        assert first.learn_info is not None
        assert first.learn_info.learn_id == 1
        assert task_manager.learn_count == 0
        blocked_by_inflight = task_manager.get_learn_task_info()
        assert blocked_by_inflight.learn_info is None
    finally:
        task_manager.close()


def test_one_sample_update_issues_one_learn_token() -> None:
    """一次样本更新只发一张学习令牌；learn_times_per_episode 是令牌内更新次数，不是额外令牌。"""
    learn_times = 2
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=4, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3, batch_size=2, learn_times=learn_times)],
    )
    try:
        task_manager.learn_batches_update(_ready_buffer(size=2), 1)
        assert task_manager.learn_count == 1
        first = task_manager.get_learn_task_info()
        assert first.learn_info is not None
        assert first.learn_info.learn_id == 1
        assert first.learn_info.algo_configs["agent_a"].learn_times_per_episode == learn_times
        assert task_manager.learn_count == 0
        assert task_manager.get_learn_task_info().learn_info is None
        assert task_manager.learner_state["agent_a"]["dispatched_learning_count"] == learn_times

        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0})
        task_manager.report_learning_result(
            {"agent_a": [{"loss": 0.2}, {"loss": 0.1}]},
            episode_num=first.learn_info.episode_num,
            learn_id=first.learn_info.learn_id,
        )
        info = task_manager.get_task_manager_info()
        assert info["learn_ready"] is False
        assert info["inflight_learning_count"] == 0
        assert info["completed_learning_steps"] == learn_times
        assert task_manager.get_learn_task_info().learn_info is None

        task_manager.learn_batches_update(_ready_buffer(size=3), 2)
        _report_episode(task_manager,2, total_rewards={"agent_a": 1.0})
        assert task_manager.learn_count == 1
        second = task_manager.get_learn_task_info()
        assert second.learn_info is not None
        assert second.learn_info.learn_id == 2
        assert second.learn_info.algo_configs["agent_a"].learn_times_per_episode == learn_times
        assert task_manager.get_learn_task_info().learn_info is None
        assert task_manager.learner_state["agent_a"]["dispatched_learning_count"] == learn_times * 2
    finally:
        task_manager.close()


def test_learning_result_does_not_rearm_without_new_sample() -> None:
    """学习完成后不自动续发；没有新的 ReplayBuffer.push 就领不到下一张令牌。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=5, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3, batch_size=2, learn_times=1)],
    )
    try:
        task_manager.learn_batches_update(_ready_buffer(size=2), 1)
        first = task_manager.get_learn_task_info()
        assert first.learn_info is not None
        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0})
        task_manager.report_learning_result(
            {"agent_a": [{"loss": 0.1}]},
            episode_num=first.learn_info.episode_num,
            learn_id=first.learn_info.learn_id,
        )
        idle = task_manager.get_learn_task_info()
        assert idle.learn_info is None
        assert idle.task_finished is False

        task_manager.learn_batches_update(_ready_buffer(size=2), 2)
        _report_episode(task_manager,2, total_rewards={"agent_a": 1.0})
        second = task_manager.get_learn_task_info()
        assert second.learn_info is not None
        assert second.learn_info.learn_id == first.learn_info.learn_id + 1
    finally:
        task_manager.close()


def test_first_learn_at_each_sample_watermark_keeps_legacy_seed() -> None:
    """每个新采样水位的首次更新沿用原 episode seed；同水位因学习期间再 push 才派生子 seed。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=3, sample_learn_max_lag=None, seed=42),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3, batch_size=1, learn_times=1)],
    )
    try:
        first = _issue_and_claim_learn_token(task_manager, 1, size=1)
        assert first.seed == derive_seed(42, "train", "learn", 1)
        task_manager.learn_batches_update(_ready_buffer(size=1), 1)
        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0})
        task_manager.report_learning_result(
            {"agent_a": [{"loss": 0.1}]},
            episode_num=first.episode_num,
            learn_id=first.learn_id,
        )

        repeated = task_manager.get_learn_task_info().learn_info
        assert repeated is not None
        assert repeated.seed == derive_seed(42, "train", "learn", 1, 1)
        task_manager.report_learning_result(
            {"agent_a": [{"loss": 0.1}]},
            episode_num=repeated.episode_num,
            learn_id=repeated.learn_id,
        )
        assert task_manager.get_learn_task_info().learn_info is None

        task_manager.learn_batches_update(_ready_buffer(size=2), 2)
        _report_episode(task_manager,2, total_rewards={"agent_a": 1.0})
        next_sample = task_manager.get_learn_task_info().learn_info
        assert next_sample is not None
        assert next_sample.seed == derive_seed(42, "train", "learn", 2)
    finally:
        task_manager.close()


def test_empty_learning_result_does_not_advance_watermark_or_rearm() -> None:
    """没有实际 Learner.update 时只结算 in-flight，不伪造学习水位或续发许可。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=3, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3, batch_size=1, learn_times=1)],
    )
    try:
        learn_info = _issue_and_claim_learn_token(task_manager, 1, size=1)
        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0})
        task_manager.report_learning_result(
            {},
            episode_num=learn_info.episode_num,
            learn_id=learn_info.learn_id,
        )
        info = task_manager.get_task_manager_info()
        assert info["completed_learning_episode"] == 0
        assert info["completed_learning_steps"] == 0
        assert info["learn_ready"] is False
        assert task_manager.get_learn_task_info().learn_info is None
    finally:
        task_manager.close()


def test_final_sample_keeps_exactly_one_tail_permit() -> None:
    """串行末局在采样关闭后仍消费一次已就绪许可，之后不再续发。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=1, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3, batch_size=1, learn_times=1)],
    )
    try:
        task_manager.learn_batches_update(_ready_buffer(size=1), 1)
        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0})
        assert task_manager.is_learning_completed() is False
        tail = task_manager.get_learn_task_info()
        assert tail.learn_info is not None
        task_manager.report_learning_result(
            {"agent_a": [{"loss": 0.1}]},
            episode_num=tail.learn_info.episode_num,
            learn_id=tail.learn_info.learn_id,
        )
        assert task_manager.is_learning_completed() is True
        finished = task_manager.get_learn_task_info()
        assert finished.task_finished is True
        assert finished.learn_info is None
    finally:
        task_manager.close()


def test_inflight_and_one_ready_are_the_only_distributed_tail() -> None:
    """采样关闭时只保留一张 in-flight 和一张合并后的尾部许可。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=2, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3, batch_size=1, learn_times=1)],
    )
    try:
        first = _issue_and_claim_learn_token(task_manager, 1, size=1)
        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0})
        task_manager.learn_batches_update(_ready_buffer(size=2), 2)
        _report_episode(task_manager,2, total_rewards={"agent_a": 2.0})
        assert task_manager.learn_count == 1
        assert task_manager.get_learn_task_info().learn_info is None

        task_manager.report_learning_result(
            {"agent_a": [{"loss": 0.2}]},
            episode_num=first.episode_num,
            learn_id=first.learn_id,
        )
        tail = task_manager.get_learn_task_info()
        assert tail.learn_info is not None
        assert tail.learn_info.learn_id == 2
        task_manager.report_learning_result(
            {"agent_a": [{"loss": 0.1}]},
            episode_num=tail.learn_info.episode_num,
            learn_id=tail.learn_info.learn_id,
        )
        assert task_manager.is_learning_completed() is True
    finally:
        task_manager.close()


def test_restart_zeros_continuous_permit_state() -> None:
    """restart 后采样、许可编号、in-flight 和学习水位全部回到本代次初值。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=1, sample_learn_max_lag=None),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3, batch_size=1, learn_times=1)],
    )
    try:
        learn_info = _issue_and_claim_learn_token(task_manager, 1, size=1)
        _report_episode(task_manager,1, total_rewards={"agent_a": 1.0})
        task_manager.report_learning_result(
            {"agent_a": [{"loss": 0.1}]},
            episode_num=learn_info.episode_num,
            learn_id=learn_info.learn_id,
        )
        task_manager.restart_training_task(reason="test")
        info = task_manager.get_task_manager_info()
        assert info["generation"] == 1
        assert info["completed_sample_episode"] == 0
        assert info["completed_learning_episode"] == 0
        assert info["completed_learning_steps"] == 0
        assert info["learn_ready"] is False
        assert info["inflight_learning_count"] == 0
        _, expected = task_manager.learning_process
        assert expected == 0
        next_info = _issue_and_claim_learn_token(task_manager, 1, size=1)
        assert next_info.learn_id == 1
    finally:
        task_manager.close()


def _complete_main_sample_episodes(task_manager: TrainTaskManager, count: int) -> None:
    for episode_num in range(1, count + 1):
        _report_episode(task_manager,episode_num, total_rewards={"agent_a": float(episode_num)})


def test_end_eval_zero_does_not_block_completion_or_draw_chart(tmp_path: Path) -> None:
    """end_eval_episodes=0 时主路跑满即结束，不生成结束评估奖励图。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(
            max_episode=2,
            sample_learn_max_lag=None,
            end_eval_episodes=0,
            result_save_dir=str(tmp_path),
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        _complete_main_sample_episodes(task_manager, 2)
        assert task_manager.is_sample_completed() is True
        assert task_manager._end_eval_dispatch_count == 0
        assert task_manager._last_end_eval_reward_path is None
        assert list(tmp_path.glob("end_eval_reward_*.png")) == []
    finally:
        task_manager.close()


def test_end_eval_tokens_block_completion_until_all_reported(tmp_path: Path) -> None:
    """主路消费完后签发结束评估令牌；全部回报前不得宣告采样结束，并录 GIF。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(
            max_episode=2,
            sample_learn_max_lag=None,
            end_eval_episodes=3,
            result_save_dir=str(tmp_path),
            render_video_dir=str(tmp_path),
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
        stage_name="search_with_threats",
    )
    try:
        _complete_main_sample_episodes(task_manager, 2)
        assert task_manager.is_sample_completed() is False
        assert task_manager._end_eval_dispatch_count == 3

        tokens = [task_manager.get_sample_task_info() for _ in range(3)]
        assert all(token is not None for token in tokens)
        for index, token in enumerate(tokens, start=1):
            assert token.metadata["replay_write"] is False
            assert token.metadata["record_render"] is True
            assert token.metadata["record_name"].startswith("end_eval_")
            assert token.metadata["record_name"].endswith(f"_eval_{index}.gif")
            assert token.deterministic is True
        assert task_manager.get_sample_task_info() is None
        assert task_manager.is_sample_completed() is False

        for index, token in enumerate(tokens, start=1):
            _report_episode(task_manager,
                token.episode_num,
                total_rewards={"agent_a": float(index)},
                lease_id=token.lease_id,
                artifacts=[Artifact(ArtifactType.RENDER_GIF, str(tmp_path / f"end_eval_{index}.gif"))],
            )
            if index < 3:
                assert task_manager.is_sample_completed() is False

        assert task_manager.is_sample_completed() is True
        assert task_manager._end_eval_completed_count == 3
        assert task_manager._last_end_eval_reward_path is not None
        chart_path = Path(task_manager._last_end_eval_reward_path)
        assert chart_path.exists()
        assert chart_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert task_manager._end_eval_gif_paths == [
            str(tmp_path / "end_eval_1.gif"),
            str(tmp_path / "end_eval_2.gif"),
            str(tmp_path / "end_eval_3.gif"),
        ]
        assert task_manager._end_eval_reward_details == [
            {"total": 1.0},
            {"total": 2.0},
            {"total": 3.0},
        ]
    finally:
        task_manager.close()


def test_end_eval_chart_uses_reward_details_when_env_reports_them(tmp_path: Path) -> None:
    """环境提供奖励构成时，结束评估柱状图按分量堆叠。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(
            max_episode=1,
            sample_learn_max_lag=None,
            end_eval_episodes=2,
            result_save_dir=str(tmp_path),
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        _complete_main_sample_episodes(task_manager, 1)
        tokens = [task_manager.get_sample_task_info() for _ in range(2)]
        _report_episode(task_manager,
            tokens[0].episode_num,
            total_rewards={"agent_a": 3.0},
            episode_rewards_details={"bonus": 4.0, "cost": -1.0},
            lease_id=tokens[0].lease_id,
        )
        _report_episode(task_manager,
            tokens[1].episode_num,
            total_rewards={"agent_a": 1.0},
            episode_rewards_details={"bonus": 2.0, "cost": -1.0},
            lease_id=tokens[1].lease_id,
        )
        assert task_manager.is_sample_completed() is True
        assert task_manager._end_eval_reward_details == [
            {"bonus": 4.0, "cost": -1.0},
            {"bonus": 2.0, "cost": -1.0},
        ]
        assert Path(task_manager._last_end_eval_reward_path).exists()
    finally:
        task_manager.close()


def test_end_eval_failed_token_is_retried_before_completion(tmp_path: Path) -> None:
    """结束评估失败会补发令牌，成功满额后才结束。"""
    task_manager = TrainTaskManager(
        config=TrainConfig(
            max_episode=1,
            sample_learn_max_lag=None,
            end_eval_episodes=1,
            result_save_dir=str(tmp_path),
        ),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        _complete_main_sample_episodes(task_manager, 1)
        first = task_manager.get_sample_task_info()
        _report_episode(task_manager,
            first.episode_num,
            total_rewards={},
            success=False,
            lease_id=first.lease_id,
        )
        assert task_manager.is_sample_completed() is False
        retry = task_manager.get_sample_task_info()
        assert retry is not None
        assert retry.lease_id != first.lease_id
        _report_episode(task_manager,
            retry.episode_num,
            total_rewards={"agent_a": 5.0},
            lease_id=retry.lease_id,
        )
        assert task_manager.is_sample_completed() is True
        assert task_manager._end_eval_completed_count == 1
    finally:
        task_manager.close()


def test_task_manager_close_logs_reward_plot_paths(caplog) -> None:
    task_manager = TrainTaskManager(
        config=TrainConfig(max_episode=1),
        policy_agent_configs=[_agent_config("agent_a")],
        learner_configs=[_learner_config("agent_a", lr=1e-3)],
    )
    try:
        caplog.set_level(logging.INFO, logger="train_task")
        task_manager.close()
        assert "Save reward curve:" in caplog.text
        assert "Save reward details:" in caplog.text
    finally:
        task_manager.close()


if __name__ == "__main__":
    test_task_manager_initializes_without_stage_task()
    test_two_stages_produce_distinct_task_managers()
    test_two_stages_swap_fixed_and_learning_agents()
    print("\nAll task scheduler unit tests passed.")


def test_versioned_episode_leases_wait_for_the_next_exact_model(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch

    from common.exception import ModelVersionNotFound
    from config import ModelManagerConfig
    from models.model_manager import ModelManager
    from roles.sample import SampleRole
    from sample.model import SamplingStrategy
    from utils.module_utils import publish_learners_to_model_manager

    name = "agent_a"
    config = TrainConfig(versioned=True, max_episode=5, save_optimo_models=False)
    learner_config = _learner_config(name, lr=1e-3, batch_size=2)
    learner_config.algo_config.sample_strategy = SamplingStrategy.ON_POLICY_CURSOR.value
    task_manager = TrainTaskManager(
        config=config, policy_agent_configs=[_agent_config(name)], learner_configs=[learner_config],
        supports_versioned_sampling=True,
    )
    pool = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path)))
    model = SimpleNamespace(name=name, algo_type="ppo", models={"agent_net": torch.nn.Linear(1, 1)})
    replay = SimpleNamespace(set_sample_version=lambda version, requests, **kwargs: None)

    def publish(learning_batch):
        publish_learners_to_model_manager(
            [model], pool, task_manager.task_id, task_manager=task_manager,
            train_config=config, is_learning_batch=learning_batch, replay_buffer=replay,
        )

    def replay_progress(token, available):
        task_manager.learn_batches_update(
            ReplayBufferState(
                sample_version=token.required_model_versions[name], versioned_writes=available,
                versioned_available={name: available},
            ),
            token.episode_num,
        )

    def finish(token, success=True):
        task_manager.report_sample_result(
            token.episode_num, {name: 0.0}, success=success, lease_id=token.lease_id,
        )

    try:
        publish(False)
        first = task_manager.get_sample_task_info(sample_role_name="sample-1")
        second = task_manager.get_sample_task_info(sample_role_name="sample-2")
        version = first.required_model_versions[name]
        assert second.required_model_versions[name] == version
        assert pool.get_model(name, version=version, output_type="info").version == version
        assert task_manager.get_sample_task_info(sample_role_name="sample-3") is None

        replay_progress(first, 1)
        finish(first)
        finish(second, success=False)
        retry = task_manager.get_sample_task_info(sample_role_name="sample-3")
        assert retry.episode_num == second.episode_num
        assert retry.lease_id != second.lease_id
        assert retry.required_model_versions == second.required_model_versions

        replay_progress(retry, 2)
        replay_progress(first, 1)  # A delayed callback cannot erase newer progress.
        assert task_manager.get_learn_task_info().learn_info is None
        finish(retry)
        learn = task_manager.get_learn_task_info().learn_info
        assert learn is not None

        future = task_manager.get_sample_task_info(sample_role_name="sample-1")
        future_version = future.required_model_versions[name]
        assert future_version != version
        with pytest.raises(ModelVersionNotFound):
            pool.get_model(name, version=future_version, output_type="info")

        task_manager.report_learning_result({name: {"loss": 1.0}}, learn.episode_num, learn_id=learn.learn_id)
        sampler = SampleRole.__new__(SampleRole)
        sampler.role_id = "sample-1"
        sampler.task_manager = task_manager
        sampler.model_manager = pool
        sampler.policy_agents = [SimpleNamespace(name=name, agent_net=torch.nn.Linear(1, 1), device="cpu")]
        sampler._trainable_agent_names = {name}
        sampler._synced_model_versions = {name: version}
        waits = []

        def publish_on_wait(seconds):
            assert not waits
            waits.append(seconds)
            publish(True)

        monkeypatch.setattr("roles.sample.time.sleep", publish_on_wait)
        sampler._sync_episode_models(future)
        assert waits
        assert sampler._synced_model_versions[name] == future_version
        assert pool.get_model(name, version=future_version, output_type="info").version == future_version
        assert task_manager.get_sample_task_info(sample_role_name="sample-2").required_model_versions == (
            future.required_model_versions
        )
    finally:
        task_manager.close()


@pytest.mark.parametrize(
    "strategy,initial_tickets,progress",
    [
        ("on_policy", 4, [1, 2, 3, 4]),
        ("on_policy_cursor", 4, [1, 2, 3, 4]),
        ("on_policy_recent_steps", 4, [2, 4, 6, 8]),
        ("on_policy_cursor_steps", 4, [2, 4, 6, 8]),
        ("time_slice", 1, [1]),
        ("recent_biased", 4, [1, 2, 3, 4]),
        ("sliding_window", 4, [1, 2, 3, 4]),
        ("uniform", 4, [1, 2, 3, 4]),
    ],
)
def test_versioned_task_manager_loads_rule_from_learner_strategy(strategy, initial_tickets, progress):
    name = "agent_a"
    learner = _learner_config(name, lr=1e-3, batch_size=4)
    learner.algo_config.sample_strategy = strategy
    manager = TrainTaskManager(
        config=TrainConfig(versioned=True, max_episode=10, save_optimo_models=False),
        policy_agent_configs=[_agent_config(name)],
        learner_configs=[learner],
        supports_versioned_sampling=True,
    )
    try:
        tickets = [manager.get_sample_task_info(sample_role_name=str(i)) for i in range(initial_tickets)]
        assert all(tickets)
        version = tickets[0].required_model_versions[name]
        assert {ticket.required_model_versions[name] for ticket in tickets} == {version}
        assert manager.get_sample_task_info(sample_role_name="other") is None
        for i, (ticket, available) in enumerate(zip(tickets, progress), 1):
            manager.learn_batches_update(
                ReplayBufferState(
                    sample_version=version, versioned_writes=i, versioned_available={name: available},
                    versioned_fresh_agents={name},
                ),
                ticket.episode_num,
            )
            manager.report_sample_result(ticket.episode_num, {}, lease_id=ticket.lease_id)
            if i < len(tickets):
                assert manager.get_learn_task_info().learn_info is None
        assert manager.get_learn_task_info().learn_info is not None
        assert manager.get_sample_task_info(sample_role_name="next").required_model_versions[name] != version
    finally:
        manager.close()

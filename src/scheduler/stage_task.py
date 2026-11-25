from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, TYPE_CHECKING

from common.info_bridge import info_bridge, public_training_info
from scheduler.artifact import Artifact
from common.logger import init_logger
from common.process_shutdown import process_close
from common.registry import extensible
from config import AgentConfig, LearnerConfig, StageConfig, StageTaskConfig, TrainConfig
from scheduler.model import LearnTaskResult
from scheduler.train_task import TrainTaskManager
from utils.stage_utils import union_stage_agent_configs, union_stage_learner_configs
from utils.task_id import task_id

if TYPE_CHECKING:
    from scheduler.automl import AutoML
    from scheduler.artifact import ArtifactsManager

LOG = init_logger("stage_task")


@extensible
class StageTask:
    """
    分阶段训练的阶段任务生命周期管理器。

    职责边界（只管 TaskManager，其余交给 Role）：
    - StageTask 是一个迭代器，按 StageTaskConfig.stages 顺序，为每个阶段实例化一个独立的
      TrainTaskManager 并返回；阶段推进和旧实例回收由 StageTask 负责。
    - 清空回放池、写入 Environment.stage_name、重置 Learner 超参、阶段模型保存与继承等
      “除 TaskManager 之外”的切换动作，全部由调用方 Role 完成。
    - Role 初始化时应通过 all_agent_configs / all_learner_configs 收集跨阶段全集，
      一次性构造 Agent/Learner 实例；换阶段时只对当前阶段上场的 Learner 调用
      dynamic_modify_config，不重新构造 Learner。

    普通训练可视为只有一个阶段（stage_name=train 或单元素 stages）的 StageTask；
    StageTaskConfig.stages 为空时视为单阶段 train，便于把基础训练统一到同一条调度路径上。
    """

    def __init__(
            self,
            stage_task_config: StageTaskConfig,
            auto_ml_by_stage: Optional[Dict[str, Optional["AutoML"]]] = None,
            env_name: str = "",
            artifacts_manager: Optional["ArtifactsManager"] = None,
            *, supports_versioned_sampling: bool = False,
    ):
        # 阶段计划；为空时退化为单阶段 train，保证普通训练也走同一套阶段调度。
        self._stages: List[StageConfig] = list(stage_task_config.stages) or [StageConfig(stage_name="train")]
        self._start_stage = self._validate_start_stage(stage_task_config.start_stage)
        self._stage_task_config = stage_task_config
        self._fill_stage_model_versions(stage_task_config.stage_model_versions)
        self._apply_skip_stage_model_load(stage_task_config)
        # Role 预创建的 AutoML 实例；key=stage_name，未启用阶段为 None。
        self._auto_ml_by_stage: Dict[str, Optional[AutoML]] = dict(auto_ml_by_stage or {})
        self._env_name = env_name
        self._artifacts_manager = artifacts_manager
        self._supports_versioned_sampling = supports_versioned_sampling
        self._assert_unique_stage_names()
        self._assert_train_configs()
        self._assert_consistent_archive_dirs()
        self._assert_consistent_agents()
        # 跨阶段 Agent/Learner 全集；Role 初始化时一次性构造，换阶段只更新配置。
        self._all_agent_configs = union_stage_agent_configs(self._stages)
        self._all_learner_configs = (
            list(stage_task_config.bootstrap_learners_config)
            or union_stage_learner_configs(self._stages)
        )
        self._assert_learner_agent_alignment()
        if self._artifacts_manager is not None:
            self._artifacts_manager.plan_task_stages(
                str(task_id),
                (stage.stage_name for stage in self._stages[self._start_stage:]),
                archive_save_dir=self._stages[self._start_stage].train_config.artifacts_archive_save_dir,
            )
        # 当前阶段下标；首次推进后进入start_stage，之前的阶段不会创建TaskManager。
        self._current_index = self._start_stage - 1
        # 当前阶段的 TrainTaskManager 实例；阶段推进时会被新实例替换。
        self._current_task_manager: Optional[TrainTaskManager] = None
        # current只表示TaskManager已经创建；ready表示环境、回放池、Learner和初始模型也已完成切换。
        # RPC入口只能读取ready_task_manager，避免阶段切换窗口内提前发放新阶段令牌。
        self._lifecycle_lock = threading.RLock()
        self._ready_task_manager: Optional[TrainTaskManager] = None
        self._closed = False

    def _validate_start_stage(self, start_stage: int) -> int:
        """校验课程入口下标；配置类只声明结构，运行期约束由StageTask负责。"""
        if isinstance(start_stage, bool) or not isinstance(start_stage, int):
            raise TypeError(
                f"StageTaskConfig.start_stage must be int, given={type(start_stage).__name__}"
            )
        if not 0 <= start_stage < len(self._stages):
            raise ValueError(
                f"StageTaskConfig.start_stage is out of range, "
                f"given={start_stage}, stages={len(self._stages)}"
            )
        return start_stage

    def _fill_stage_model_versions(self, model_versions: List[str]) -> None:
        """从start_stage起依次写入本次运行各阶段的入口模型版本。"""
        if not isinstance(model_versions, list):
            raise TypeError(
                f"StageTaskConfig.stage_model_versions must be list, "
                f"given={type(model_versions).__name__}"
            )
        runnable_stage_count = len(self._stages) - self._start_stage
        if len(model_versions) > runnable_stage_count:
            raise ValueError(
                f"StageTaskConfig.stage_model_versions has more entries than runnable stages, "
                f"versions={len(model_versions)}, runnable_stages={runnable_stage_count}, "
                f"start_stage={self._start_stage}"
            )
        for offset, model_version in enumerate(model_versions):
            if not isinstance(model_version, str):
                raise TypeError(
                    f"StageTaskConfig.stage_model_versions[{offset}] must be str, "
                    f"given={type(model_version).__name__}"
                )
            stage_index = self._start_stage + offset
            self._stages[stage_index].train_config.model_version = model_version.strip() or None

    def _apply_skip_stage_model_load(self, stage_task_config: StageTaskConfig) -> None:
        """
        跳过前置阶段时，把 skip_stage_model_* 写入 starting_stage.train_config。

        仅 start_stage>0 生效；若同时配置了旧字段 stage_model_versions，以 skip_* 覆盖 starting stage。
        """
        if self._start_stage <= 0:
            return
        version = stage_task_config.skip_stage_model_version
        load_dir = stage_task_config.skip_stage_model_load_dir
        # 兼容：未显式配置 skip_stage_model_version 时，沿用 stage_model_versions[0]
        if version is None and stage_task_config.stage_model_versions:
            version = stage_task_config.stage_model_versions[0]
            LOG.warning(
                "StageTaskConfig.stage_model_versions is deprecated for skip-stage bootstrap; "
                "prefer skip_stage_model_version. Mapped versions[0]=%r to starting stage.",
                version,
            )
        if version is None and load_dir is None:
            return
        starting = self._stages[self._start_stage]
        if version is not None:
            if not isinstance(version, str):
                raise TypeError(
                    f"StageTaskConfig.skip_stage_model_version must be str|None, "
                    f"given={type(version).__name__}"
                )
            starting.train_config.model_version = version.strip() or None
        if load_dir is not None:
            if not isinstance(load_dir, str):
                raise TypeError(
                    f"StageTaskConfig.skip_stage_model_load_dir must be str|None, "
                    f"given={type(load_dir).__name__}"
                )
            stripped = load_dir.strip()
            if stripped:
                starting.train_config.model_load_dir = stripped

    def _assert_unique_stage_names(self) -> None:
        names = [stage.stage_name for stage in self._stages]
        assert len(set(names)) == len(names), f"StageConfig.stage_name must be unique, given={names}"

    def _assert_train_configs(self) -> None:
        """StageTask 只支持训练任务"""
        for stage in self._stages:
            assert isinstance(stage.train_config, TrainConfig), (
                f"StageTask only supports training stages, stage={stage.stage_name} "
                f"train_config must be TrainConfig, given={type(stage.train_config).__name__}"
            )

    def _assert_consistent_archive_dirs(self) -> None:
        """同一次分阶段任务必须增量写入同一个 TASK_ID 归档根目录。"""
        archive_dirs = {
            stage.train_config.artifacts_archive_save_dir
            for stage in self._stages
        }
        if len(archive_dirs) != 1:
            raise ValueError(
                "All stages must use the same artifacts_archive_save_dir, "
                f"given={sorted(archive_dirs)}"
            )

    def _assert_consistent_agents(self) -> None:
        """各阶段参与 Agent 名单必须完全一致（Agent 是跨阶段共享的训练对象）。"""
        agent_name_sets = [
            frozenset(agent_config.name for agent_config in stage.agents_config)
            for stage in self._stages
        ]
        assert agent_name_sets, "StageTaskConfig stages must declare agents_config."
        baseline = agent_name_sets[0]
        for index, names in enumerate(agent_name_sets[1:], start=2):
            assert names == baseline, (
                f"All stages must declare the same agents, stage1={sorted(baseline)}, "
                f"stage{index}={sorted(names)}"
            )

    def _assert_learner_agent_alignment(self) -> None:
        """每个可能出现的 Learner 都必须有同名 Agent 参与采样。"""
        agent_names = {agent_config.name for agent_config in self._all_agent_configs}
        for learner_config in self._all_learner_configs:
            assert learner_config.name in agent_names, (
                f"Learner {learner_config.name} has no matching Agent in all_agent_configs, "
                f"agents={sorted(agent_names)}"
            )

    @property
    def all_agent_configs(self) -> List[AgentConfig]:
        """跨阶段参与采样的 Agent 全集；Role 初始化时据此构造 Agent 实例（只建一次）。"""
        return self._all_agent_configs

    @property
    def all_learner_configs(self) -> List[LearnerConfig]:
        """课程启动Learner全集；包含可能全程固定、仅提供策略的Learner。"""
        return self._all_learner_configs

    @property
    def total_stages(self) -> int:
        return len(self._stages)

    @property
    def starting_stage(self) -> StageConfig:
        """本次运行首个执行的阶段配置。"""
        return self._stages[self._start_stage]

    @property
    def start_stage(self) -> int:
        return self._start_stage

    @property
    def stage_task_config(self) -> StageTaskConfig:
        return self._stage_task_config

    @property
    def current_stage_index(self) -> int:
        return self._current_index

    @property
    def current_stage(self) -> Optional[StageConfig]:
        if 0 <= self._current_index < len(self._stages):
            return self._stages[self._current_index]
        return None

    @property
    def current_task_manager(self) -> Optional[TrainTaskManager]:
        with self._lifecycle_lock:
            return self._current_task_manager

    @property
    def ready_task_manager(self) -> Optional[TrainTaskManager]:
        """返回已完成外部资源切换、可安全提供RPC令牌的当前TaskManager。"""
        with self._lifecycle_lock:
            if self._ready_task_manager is self._current_task_manager:
                return self._ready_task_manager
            return None

    def mark_current_stage_ready(self, task_manager: TrainTaskManager) -> None:
        """原子开放当前阶段的令牌入口。"""
        with self._lifecycle_lock:
            assert self._current_task_manager is task_manager, (
                "Only the current stage TaskManager can be marked ready."
            )
            self._ready_task_manager = task_manager

    def mark_current_stage_unready(self, task_manager: Optional[TrainTaskManager] = None) -> None:
        """关闭令牌入口；指定旧实例时不会误关闭已经就绪的新阶段。"""
        with self._lifecycle_lock:
            if task_manager is None or self._ready_task_manager is task_manager:
                self._ready_task_manager = None

    @contextmanager
    def preparing_current_stage(self, task_manager: TrainTaskManager):
        """在阶段外部资源准备期间关闭令牌入口，全部成功后再原子开放。"""
        self.mark_current_stage_unready()
        try:
            yield task_manager
        except BaseException:
            self.mark_current_stage_unready(task_manager)
            raise
        else:
            self.mark_current_stage_ready(task_manager)

    def has_next_stage(self) -> bool:
        """是否还有未开始的后续阶段。"""
        return self._current_index + 1 < len(self._stages)

    def is_all_stages_completed(self) -> bool:
        """是否已经走完最后一个阶段且其任务也已结束。"""
        if self._current_index < len(self._stages) - 1:
            return False
        task_manager = self._current_task_manager
        return task_manager is not None and task_manager.is_task_finished()

    def __iter__(self) -> Iterator[TrainTaskManager]:
        return self

    def __next__(self) -> TrainTaskManager:
        """推进到下一个阶段并返回其全新的 TrainTaskManager；没有后续阶段时结束迭代。"""
        if not self.has_next_stage():
            raise StopIteration
        return self.start_next_stage()

    def start_next_stage(self) -> TrainTaskManager:
        """推进到下一个阶段：回收旧 TaskManager，按新阶段配置实例化新的 TaskManager。"""
        assert self.has_next_stage(), "StageTask has no next stage to start."
        self._close_current_task_manager()
        self._current_index += 1
        task_manager = self._build_task_manager(self._stages[self._current_index])
        with self._lifecycle_lock:
            self._current_task_manager = task_manager
            # 新实例创建后默认仍不可领取，必须完成preparing_current_stage上下文。
            self._ready_task_manager = None
        return task_manager

    def _build_task_manager(self, stage: StageConfig) -> TrainTaskManager:
        """按单个阶段配置实例化 TrainTaskManager；AutoML 由 Role 预创建后注入。"""
        return TrainTaskManager(
            config=stage.train_config,
            policy_agent_configs=stage.agents_config,
            learner_configs=stage.learners_config,
            stage_name=stage.stage_name,
            env_name=self._env_name,
            auto_ml=self._auto_ml_by_stage.get(stage.stage_name),
            artifacts_manager=self._artifacts_manager,
            supports_versioned_sampling=self._supports_versioned_sampling,
        )

    def _close_current_task_manager(self) -> None:
        # 先在锁内撤销RPC可见性，再在锁外关闭后台线程，避免慢close阻塞远端查询。
        with self._lifecycle_lock:
            task_manager = self._current_task_manager
            self._ready_task_manager = None
            self._current_task_manager = None
        if task_manager is not None:
            task_manager.close()

    @process_close
    def close(self) -> None:
        """训练结束时回收当前 TaskManager 的后台线程。"""
        if self._closed:
            return
        self._closed = True
        self._close_current_task_manager()


class StageTaskManagerProxy:
    """把稳定的RPC服务入口转发到StageTask当前已就绪的TaskManager。"""

    def __init__(self, stage_task: StageTask):
        self._stage_task = stage_task

    def _ready_tm(self):
        return self._stage_task.ready_task_manager

    @property
    def _tm(self):
        task_manager = self._stage_task.current_task_manager
        assert task_manager is not None, "StageTask has no active TrainTaskManager yet."
        return task_manager

    def get_sample_task_info(self, **kwargs):
        task_manager = self._ready_tm()
        if task_manager is None:
            return None
        return task_manager.get_sample_task_info(**kwargs)

    def sample_backpressure_state(self, sample_role_name: str = "unknown"):
        task_manager = self._ready_tm()
        if task_manager is None:
            return {
                "enabled": False,
                "sample_completed": 0,
                "learn_completed_episode": 0,
                "lag": 0,
                "max_lag": None,
                "retry_episode": None,
                "reason": None,
                "stage_ready": False,
            }
        return task_manager.sample_backpressure_state(sample_role_name=sample_role_name)

    def sample_gate_state(self, sample_role_name: str = "unknown"):
        task_manager = self._ready_tm()
        if task_manager is None:
            return {
                "blocked": True,
                "reason": "stage_switching",
                "retry_after_seconds": 0.5,
                "stage_ready": False,
            }
        return task_manager.sample_gate_state(sample_role_name=sample_role_name)

    def report_sample_result(
            self,
            episode_num: int,
            total_rewards: dict[str, float],
            success: bool = True,
            artifacts: Optional[list[Artifact]] = None,
            episode_metrics: Optional[dict[str, float]] = None,
            episode_rewards_details: Optional[dict[str, float]] = None,
            lease_id: str = "",
            step_count: int = 0,
    ):
        return self._tm.report_sample_result(
            episode_num,
            total_rewards=total_rewards,
            success=success,
            artifacts=artifacts,
            episode_metrics=episode_metrics,
            episode_rewards_details=episode_rewards_details,
            lease_id=lease_id,
            step_count=step_count,
        )

    def accept_sample_task_result(self, episode_num: int, lease_id: str) -> bool:
        task_manager = self._ready_tm()
        if task_manager is None:
            return False
        return task_manager.accept_sample_task_result(episode_num, lease_id)

    def save_reward_curve(self):
        return self._tm.save_reward_curve()

    def is_sample_completed(self):
        # 阶段切换时 ready 门闩先关，但旧 TM 的采样完成状态仍是 true。
        # 必须把 true 透出去，sample role 才会提交本阶段产物；否则控制端
        # notify_stage_complete 会一直等 producer，形成阶段切换死锁。
        task_manager = self._ready_tm()
        if task_manager is None:
            task_manager = self._stage_task.current_task_manager
            if task_manager is None:
                return False
        return task_manager.is_sample_completed()

    def get_learn_task_info(self):
        task_manager = self._ready_tm()
        if task_manager is None:
            return LearnTaskResult()
        result = task_manager.get_learn_task_info()
        # 单阶段学习结束不等于整场课程结束。阶段控制器会先排空当前阶段，再切换
        # TaskManager；切换窗口和中间阶段都只返回“暂时无令牌”，避免远端 Learner 退出。
        if getattr(result, "task_finished", False) and not self._stage_task.is_all_stages_completed():
            return LearnTaskResult()
        return result

    def report_learning_result(
            self,
            learn_result: dict,
            episode_num: int,
            learn_id: int = 0,
    ):
        return self._tm.report_learning_result(
            learn_result,
            episode_num,
            learn_id=learn_id,
        )

    def is_learning_completed(self):
        task_manager = self._ready_tm()
        if task_manager is None:
            return False
        return (
            self._stage_task.is_all_stages_completed()
            and task_manager.is_learning_completed()
        )

    def get_task_manager_info(self):
        return self._tm.get_task_manager_info()

    @property
    def sample_task_process(self):
        return self._tm.sample_task_process

    @property
    def learning_process(self):
        return self._tm.learning_process

    def custom_episode(self, episode_info, **kwargs):
        task_manager = self._ready_tm()
        if task_manager is None:
            return episode_info
        return task_manager.custom_episode(episode_info, **kwargs)

    def learn_batches_update(self, replay_buffer_state, episode_num: int):
        return self._tm.learn_batches_update(replay_buffer_state, episode_num)

    def restart_training_task(self, reason: str = "", advice=None):
        return self._tm.restart_training_task(reason=reason, advice=advice)

    def push_info(self, attrs: dict) -> None:
        """推拉打进程桥，不依赖当前阶段 TM 存活。"""
        info_bridge.put(**dict(attrs or {}))

    def pull_info(self, key: str | None = None):
        return public_training_info(key)

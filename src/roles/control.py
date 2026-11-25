from __future__ import annotations

from common.simulation import close_sim_manager_on_init_error, create_sim_manager

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimManagerConfig
    from dodo_sim.sim_manager import SimManager


import copy
import time
from typing import Optional

from common.logger import init_logger
from config import (
    AgentConfig,
    GrpcConfig,
    LearnerConfig,
    MonteCarloMethodConfig,
    ReplayBufferConfig,
    TrainConfig,
    SampleType,
)
from roles.common.config_alignment import (
    inherit_replay_buffer_seed,
    apply_learner_capacities,
    apply_replay_capacity_advice,
)
from roles.common.artifact_wiring import (
    resolve_artifacts_manager,
    bind_local_artifact_callers,
    attach_artifact_producer,
    complete_artifact_producers,
    declare_role_log_producer,
    complete_role_logs,
)
from rpc.dispatch import ARTIFACTS_MANAGER, MONTE_CARLO_METHOD, REPLAY_BUFFER, SIM_MANAGER, TRAIN_TASK_MANAGER
from rpc.factory import RpcConfiguration, RpcServiceHost
from sample.monte_carlo import MonteCarloMethod
from sample.replay_buffer import ReplayBuffer
from scheduler.artifact import ArtifactsManager
from scheduler.train_task import TrainTaskManager
from utils.random_seed import apply_train_seed

LOG = init_logger("control")


class ControlRole:
    """控制进程角色：持有任务、样本、仿真和中心录制模块，并分别暴露为独立RPC服务。"""

    @close_sim_manager_on_init_error
    def __init__(
            self,
            train_config: TrainConfig,
            replay_buffer_config: ReplayBufferConfig,
            policy_agent_configs: list[AgentConfig],
            learner_configs: list[LearnerConfig],
            task_manager_address: str,
            replay_buffer_address: str,
            sim_manager_config: SimManagerConfig,
            sim_manager_address: str,
            monte_carlo_method_config: MonteCarloMethodConfig,
            monte_carlo_method_address: str,
            rpc_config: RpcConfiguration | None = None,
            artifacts_manager: Optional[ArtifactsManager] = None,
    ):
        apply_train_seed(
            train_config,
            role_id="control",
            context="ControlRole",
        )
        self.role_id = "control"
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
        self.task_manager_address = task_manager_address
        self.replay_buffer_address = replay_buffer_address
        self.sim_manager_address = sim_manager_address
        self.monte_carlo_method_address = monte_carlo_method_address
        self.task_manager = TrainTaskManager(
            config=train_config,
            policy_agent_configs=policy_agent_configs,
            learner_configs=learner_configs,
            artifacts_manager=self.artifacts_manager,
        )
        inherit_replay_buffer_seed(replay_buffer_config, train_config)
        self.replay_buffer = ReplayBuffer(config=replay_buffer_config)
        apply_learner_capacities(self.replay_buffer, learner_configs)
        self.sim_manager = create_sim_manager(config=sim_manager_config, sample_type=train_config.sample_type)
        record_config = copy.deepcopy(train_config.monte_carlo_method_config or monte_carlo_method_config)
        # 中心录制模块始终初始化；是否真正写入由TrainConfig.record_sample控制。
        record_config.write_enabled = bool(
            train_config.record_sample and train_config.sample_type == SampleType.ONLINE.value
        )
        self.monte_carlo_method = MonteCarloMethod(
            task_id=self.task_manager.task_id,
            config=record_config,
        )
        if train_config.sample_type == SampleType.OFFLINE.value:
            count = self.monte_carlo_method.prepare_offline_index(
                stage_name=self.task_manager.stage_name,
                expected_agents=[config.name for config in policy_agent_configs],
            )
            self.task_manager.prepare_offline_sample_plan(count)
        # 控制进程内部同内存通知，RPC边界仍然保持TrainTaskManager/ReplayBuffer一模块一服务。
        self.replay_buffer.learn_batches_update_caller = self.task_manager.learn_batches_update
        self.task_manager.add_restart_callback(self._on_task_restart)
        register, put, complete_for = bind_local_artifact_callers(
            self.artifacts_manager,
            lambda: self.task_manager.task_id,
            lambda: self.task_manager.stage_name,
        )
        attach_artifact_producer(
            self.replay_buffer, "replay_buffer", register, put, complete_for,
            self.artifacts_manager,
            lambda: self.task_manager.task_id,
            lambda: self.task_manager.stage_name,
        )
        attach_artifact_producer(
            self.monte_carlo_method, "monte_carlo", register, put, complete_for,
            self.artifacts_manager,
            lambda: self.task_manager.task_id,
            lambda: self.task_manager.stage_name,
        )
        declare_role_log_producer(
            self.artifacts_manager,
            lambda: self.task_manager.task_id,
            lambda: self.task_manager.stage_name,
            self.role_id,
        )
        self.rpc_config = rpc_config or GrpcConfig(
            task_manager_address=task_manager_address,
            replay_buffer_address=replay_buffer_address,
            sim_manager_address=sim_manager_address,
            monte_carlo_method_address=monte_carlo_method_address,
        )
        self._rpc_host = RpcServiceHost(
            self.rpc_config,
            "control-0",
            {
                TRAIN_TASK_MANAGER: self.task_manager,
                REPLAY_BUFFER: self.replay_buffer,
                SIM_MANAGER: self.sim_manager,
                MONTE_CARLO_METHOD: self.monte_carlo_method,
                ARTIFACTS_MANAGER: self.artifacts_manager,
            },
        )
        LOG.info("ControlRole initialized modules:")
        LOG.info("  TrainTaskManager")
        LOG.info(
            f"  ReplayBuffer(agent_capacities={self.replay_buffer.config.agent_capacities}, "
            f"batch_size={self.replay_buffer.config.batch_size})"
        )
        LOG.info(
            f"  SimManager(config_file={self.sim_manager.config.config_file if self.sim_manager is not None else 'disabled'}, "
            f"resources={len(self.sim_manager.get_sim_list()) if self.sim_manager is not None else 0})"
        )
        LOG.info(
            f"  MonteCarloMethod(write_enabled={self.monte_carlo_method.write_enabled}, "
            f"local_file={self.monte_carlo_method.local_file})"
        )
        LOG.info(f"  policy_agent_configs({[config.name for config in policy_agent_configs]})")
        LOG.info(f"  learner_configs({[config.name for config in learner_configs]})")

    def _on_task_restart(self, generation: int, advice=None) -> None:
        """TaskManager 重启训练时，控制进程按代次同步中心回放池。"""
        self.replay_buffer.clear(generation=generation)
        apply_replay_capacity_advice(self.replay_buffer, advice)
        LOG.warning(f"ControlRole handled task restart: generation={generation}")

    def run(self) -> None:
        self._rpc_host.start()
        LOG.info(
            f"ControlRole started, task_manager={self.task_manager_address}, "
            f"replay_buffer={self.replay_buffer_address}, sim_manager={self.sim_manager_address}, "
            f"monte_carlo_method={self.monte_carlo_method_address}"
        )
        completed = False
        try:
            # 管理角色自己判定训练结束：采样与学习都完成后自然退出，中心RPC服务随之关闭，
            # 消费进程（LearnRole/SampleRole）按固定间隔重试连不上即自行优雅退出。
            # 不再向消费进程广播“任务结束”状态，以兼容分阶段训练换阶段时仍需继续采样。
            while not (self.task_manager.is_sample_completed() and self.task_manager.is_learning_completed()):
                time.sleep(1.0)
            completed = True
            LOG.info(f"ControlRole finished: sample and learning tasks completed, task_id={self.task_manager.task_id}.")
        finally:
            self._rpc_host.stop()
            if self.sim_manager is not None:
                self.sim_manager.close()
            if (
                    self.monte_carlo_method.write_enabled
                    and self.monte_carlo_method.config.storage_type == "s3"
                    and self.monte_carlo_method.current_sample_count > 0
            ):
                self.monte_carlo_method.upload_to_oss()
            self.monte_carlo_method.close()
        if completed:
            complete_artifact_producers(self.replay_buffer, self.monte_carlo_method)
            complete_role_logs(
                self.artifacts_manager,
                self.task_manager.task_id,
                self.task_manager.stage_name,
                self.role_id,
            )
            self.task_manager.notify_task_complete()

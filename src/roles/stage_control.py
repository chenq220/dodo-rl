from __future__ import annotations

from common.simulation import close_sim_manager_on_init_error, create_sim_manager, apply_stage_sim_metadata

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimManagerConfig
    from dodo_sim.sim_manager import SimManager


import copy
import time
from typing import Optional

from common.logger import init_logger
from config import (
    EnvConfig,
    GrpcConfig,
    MonteCarloMethodConfig,
    SampleType,
    ReplayBufferConfig,
    StageConfig,
    StageTaskConfig,
)
from roles.common.automl_integration import build_auto_ml_by_stage
from roles.common.config_alignment import (
    assert_state_shape_matches_env,
    inherit_replay_buffer_seed,
    apply_learner_capacities,
    apply_replay_capacity_advice,
)
from roles.common.artifact_wiring import (
    resolve_artifacts_manager,
    bind_local_artifact_callers,
    bind_artifact_producer_callers,
    declare_stage_artifact_producer,
    complete_artifact_producers,
    declare_role_log_producer,
    complete_role_logs,
)
from rpc.dispatch import ARTIFACTS_MANAGER, MONTE_CARLO_METHOD, REPLAY_BUFFER, SIM_MANAGER, TRAIN_TASK_MANAGER
from rpc.factory import RpcConfiguration, RpcServiceHost
from sample.monte_carlo import MonteCarloMethod
from sample.replay_buffer import ReplayBuffer
from scheduler.artifact import ArtifactsManager
from scheduler.stage_task import StageTask, StageTaskManagerProxy
from utils.random_seed import apply_train_seed
from utils.stage_utils import (
    apply_stage_replay_policy,
    load_skip_stage_replay_if_needed,
)
from utils.task_id import task_id

LOG = init_logger("stage_control")


class StageControlRole:
    """分阶段完整拓扑的控制角色，不持有 Learner 或 ModelManager。"""

    @close_sim_manager_on_init_error
    def __init__(
            self,
            role_id: str,
            stage_task_config: StageTaskConfig,
            env_config: EnvConfig,
            replay_buffer_config: ReplayBufferConfig,
            task_manager_address: str,
            replay_buffer_address: str,
            sim_manager_config: SimManagerConfig,
            sim_manager_address: str,
            monte_carlo_method_config: MonteCarloMethodConfig,
            monte_carlo_method_address: str,
            rpc_config: RpcConfiguration | None = None,
            artifacts_manager: Optional[ArtifactsManager] = None,
    ) -> None:
        assert stage_task_config.stages, "StageControlRole requires StageTaskConfig.stages."

        self.role_id = role_id
        self.env_config = env_config
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
        self.stage_task = StageTask(
            stage_task_config,
            auto_ml_by_stage=build_auto_ml_by_stage(stage_task_config.stages),
            env_name=env_config.sense_name,
            artifacts_manager=self.artifacts_manager,
        )
        self.task_manager_proxy = StageTaskManagerProxy(self.stage_task)
        assert_state_shape_matches_env(
            env_config,
            self.stage_task.all_agent_configs,
            self.stage_task.all_learner_configs,
        )
        inherit_replay_buffer_seed(replay_buffer_config, self.stage_task.starting_stage.train_config)
        self.replay_buffer = ReplayBuffer(config=replay_buffer_config)
        apply_learner_capacities(self.replay_buffer, self.stage_task.starting_stage.learners_config)
        self.replay_buffer.learn_batches_update_caller = self._current_learn_batches_update
        self.sim_manager = create_sim_manager(
            config=sim_manager_config,
            sample_type="offline" if all(s.train_config.sample_type == "offline"
                                         for s in stage_task_config.stages) else "online",
        )

        record_config = copy.deepcopy(
            self.stage_task.starting_stage.train_config.monte_carlo_method_config
            or monte_carlo_method_config
        )
        record_config.write_enabled = any(
            stage.train_config.record_sample for stage in stage_task_config.stages
        )
        self.monte_carlo_method = MonteCarloMethod(task_id=role_id, config=record_config)
        register, put, complete_for = bind_local_artifact_callers(
            self.artifacts_manager,
            lambda: (
                self.stage_task.current_task_manager.task_id
                if self.stage_task.current_task_manager is not None
                else str(task_id)
            ),
            lambda: (
                self.stage_task.current_task_manager.stage_name
                if self.stage_task.current_task_manager is not None
                else "train"
            ),
        )
        bind_artifact_producer_callers(
            self.replay_buffer, "replay_buffer", register, put, complete_for,
        )
        bind_artifact_producer_callers(
            self.monte_carlo_method, "monte_carlo", register, put, complete_for,
        )

        self.rpc_config = rpc_config or GrpcConfig(
            task_manager_address=task_manager_address,
            replay_buffer_address=replay_buffer_address,
            sim_manager_address=sim_manager_address,
            monte_carlo_method_address=monte_carlo_method_address,
        )
        self._rpc_host = RpcServiceHost(
            self.rpc_config,
            role_id,
            {
                TRAIN_TASK_MANAGER: self.task_manager_proxy,
                REPLAY_BUFFER: self.replay_buffer,
                SIM_MANAGER: self.sim_manager,
                MONTE_CARLO_METHOD: self.monte_carlo_method,
                ARTIFACTS_MANAGER: self.artifacts_manager,
            },
        )
        self._closed = False
        self._skip_stage_replay_loaded = False
        LOG.info(
            f"StageControlRole initialized: role_id={role_id}, "
            f"stages={self.stage_task.total_stages}"
        )

    def _current_learn_batches_update(self, replay_buffer_state, episode_num: int) -> None:
        """把回放池更新通知转发给当前阶段的任务管理器。"""
        task_manager = self.stage_task.current_task_manager
        if task_manager is not None:
            task_manager.learn_batches_update(replay_buffer_state, episode_num)

    def run(self) -> None:
        """启动中心服务并依次调度各阶段，直到采样和学习均完成。"""
        self._rpc_host.start()
        LOG.info(f"StageControlRole {self.role_id} started.")
        try:
            for task_manager in self.stage_task:
                stage = self.stage_task.current_stage
                assert stage is not None
                self._run_stage(task_manager, stage)
            LOG.info(
                f"StageControlRole {self.role_id} finished: "
                f"all stages completed, task_id={task_id}."
            )
        finally:
            self.close()

    def _run_stage(self, task_manager, stage: StageConfig) -> None:
        """准备一个阶段，并在安全排空远端学习任务后推进下一阶段。"""
        LOG.info(f"StageControlRole enter stage: stage_name={stage.stage_name}")
        try:
            with self.stage_task.preparing_current_stage(task_manager):
                apply_stage_sim_metadata(getattr(self, "sim_manager", None), stage.train_config)
                apply_train_seed(
                    stage.train_config,
                    role_id=f"{self.role_id}:{stage.stage_name}",
                    context=f"StageControlRole:{self.role_id}:{stage.stage_name}",
                )
                self._activate_stage_monte_carlo(stage, task_manager)
                if load_skip_stage_replay_if_needed(
                        self.stage_task.stage_task_config,
                        self.replay_buffer,
                        already_loaded=self._skip_stage_replay_loaded,
                ):
                    self._skip_stage_replay_loaded = True
                apply_stage_replay_policy(
                    self.replay_buffer,
                    stage,
                )
                task_manager.add_restart_callback(self._on_task_restart)

            self._declare_stage_artifact_producers(task_manager)
            LOG.info(f"StageControlRole stage ready: stage_name={stage.stage_name}")
            # RPC Learner 的结果通过稳定代理回写。必须等当前阶段学习任务排空后再替换
            # TrainTaskManager，否则迟到的 report_learning_result 可能被写入下一阶段。
            while not (
                    task_manager.is_sample_completed()
                    and task_manager.is_learning_completed()
            ):
                time.sleep(0.2)

            self.stage_task.mark_current_stage_unready(task_manager)
            self.monte_carlo_method.close()
            replay_path = self.replay_buffer.notify_stage_complete(
                stage, task_id=task_manager.task_id,
            )
            complete_artifact_producers(self.replay_buffer, self.monte_carlo_method)
            complete_role_logs(
                self.artifacts_manager, task_manager.task_id, task_manager.stage_name, self.role_id,
            )
            task_manager.notify_stage_complete()
            LOG.info(f"StageControlRole finished stage: stage_name={stage.stage_name}")
        finally:
            self.stage_task.mark_current_stage_unready(task_manager)

    def _declare_stage_artifact_producers(self, task_manager) -> None:
        manager = getattr(self, "artifacts_manager", None)
        if manager is None:
            return
        def current_task_id() -> str:
            return task_manager.task_id

        def current_stage_name() -> str:
            return task_manager.stage_name

        declare_stage_artifact_producer(
            self.replay_buffer, "replay_buffer",
            self.artifacts_manager, current_task_id, current_stage_name,
        )
        declare_stage_artifact_producer(
            self.monte_carlo_method, "monte_carlo",
            self.artifacts_manager, current_task_id, current_stage_name,
        )
        declare_role_log_producer(
            self.artifacts_manager, current_task_id, current_stage_name, self.role_id,
        )

    def _on_task_restart(self, generation: int, advice=None) -> None:
        """当前阶段重启时同步回放池代次；远端 Learner 由令牌代次自行恢复。"""
        self.replay_buffer.clear(generation=generation)
        apply_replay_capacity_advice(self.replay_buffer, advice)
        LOG.warning(
            f"StageControlRole handled task restart: generation={generation}, advice={advice}"
        )

    def _activate_stage_monte_carlo(self, stage: StageConfig, task_manager) -> None:
        monte_carlo_method = self.monte_carlo_method
        config = (
            stage.train_config.monte_carlo_method_config
            or monte_carlo_method.config
        )
        config = copy.deepcopy(config)
        sample_type = stage.train_config.sample_type
        record_sample = stage.train_config.record_sample
        config.write_enabled = bool(
            record_sample and sample_type == SampleType.ONLINE.value
        )
        monte_carlo_method.configure(config)
        if sample_type == SampleType.OFFLINE.value:
            count = monte_carlo_method.prepare_offline_index(
                stage_name=stage.stage_name,
                expected_agents=(list(self.env_config.agent_names)
                                 or [agent.name for agent in stage.agents_config]),
            )
            task_manager.prepare_offline_sample_plan(count)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._rpc_host.stop()
        self.stage_task.close()
        try:
            self.replay_buffer.close()
        finally:
            if self.sim_manager is not None:
                self.sim_manager.close()
            if (
                    self.monte_carlo_method.write_enabled
                    and self.monte_carlo_method.config.storage_type == "s3"
                    and self.monte_carlo_method.current_sample_count > 0
            ):
                self.monte_carlo_method.upload_to_oss()
            self.monte_carlo_method.close()

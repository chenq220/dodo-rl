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
    EnvConfig,
    EvalConfig,
    GrpcConfig,
    ModelManagerConfig,
    MonteCarloMethodConfig,
    ReplayBufferConfig,
)
from envs.base import BaseEnvironment
from models import ModelManager
from rpc.dispatch import (
    ARTIFACTS_MANAGER,
    MODEL_MANAGER,
    MONTE_CARLO_METHOD,
    REPLAY_BUFFER,
    SIM_MANAGER,
    TRAIN_TASK_MANAGER,
)
from rpc.factory import RpcConfiguration, RpcServiceHost
from sample.monte_carlo import MonteCarloMethod
from sample.replay_buffer import ReplayBuffer
from roles.common.config_alignment import assert_state_shape_matches_env
from roles.common.eval_reports import EvalEpisodeReportMixin
from roles.common.artifact_wiring import (
    resolve_artifacts_manager,
    bind_local_artifact_callers,
    attach_artifact_producer,
    complete_role_artifact_producers,
    declare_role_log_producer,
    complete_role_logs,
)
from scheduler.artifact import ArtifactsManager
from scheduler.train_task import TrainTaskManager
from utils.random_seed import apply_train_seed

LOG = init_logger("eval_control")


class EvalControlRole(EvalEpisodeReportMixin):
    """
    分布式评估中心：托管任务总线、回放池、模型池和仿真资源，不分发学习令牌。

    Sample 进程仍通过原有 RPC 领取评估 episode、写入样本并拉取（或不拉）模型。
    """

    @close_sim_manager_on_init_error
    def __init__(
            self,
            role_id: str,
            eval_config: EvalConfig,
            env_config: EnvConfig,
            step_agents_config: list[AgentConfig],
            replay_buffer_config: ReplayBufferConfig,
            model_manager_config: ModelManagerConfig,
            sim_manager_config: SimManagerConfig,
            monte_carlo_method_config: MonteCarloMethodConfig,
            task_manager_address: str,
            replay_buffer_address: str,
            model_manager_address: str,
            sim_manager_address: str,
            monte_carlo_method_address: str,
            rpc_config: RpcConfiguration | None = None,
            artifacts_manager: Optional[ArtifactsManager] = None,
    ):
        self.role_id = role_id
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
        merged_env_metadata = BaseEnvironment._deep_merge_metadata(
            env_config.metadata or {}, eval_config.env_metadata or {},
        )
        env_config.metadata = merged_env_metadata
        eval_config.env_metadata = BaseEnvironment._deep_merge_metadata(
            eval_config.env_metadata or {},
            {key: value for key, value in merged_env_metadata.items()
             if key in ("render_mode", "render_every_step")},
        )
        self.eval_config = eval_config
        assert_state_shape_matches_env(env_config, step_agents_config)
        apply_train_seed(
            eval_config,
            role_id=role_id,
            context=f"EvalControlRole:{role_id}",
        )
        self.rpc_config = rpc_config or GrpcConfig(
            task_manager_address=task_manager_address,
            replay_buffer_address=replay_buffer_address,
            model_manager_address=model_manager_address,
            sim_manager_address=sim_manager_address,
            monte_carlo_method_address=monte_carlo_method_address,
        )
        self.task_manager = TrainTaskManager(
            config=eval_config,
            policy_agent_configs=step_agents_config,
            learner_configs=[],
            env_name=env_config.sense_name,
            artifacts_manager=self.artifacts_manager,
        )
        buffer_config = copy.deepcopy(replay_buffer_config)
        buffer_config.batch_size = 1
        self.replay_buffer = ReplayBuffer(config=buffer_config)
        for agent in step_agents_config:
            self.replay_buffer.set_agent_capacity(agent.name, max(1, eval_config.max_episode))
        self.sim_manager = create_sim_manager(config=sim_manager_config)
        self.model_manager = ModelManager(config=model_manager_config)
        record_config = copy.deepcopy(monte_carlo_method_config)
        record_config.write_enabled = bool(eval_config.record_sample)
        self.monte_carlo_method = MonteCarloMethod(
            task_id=self.task_manager.task_id,
            config=record_config,
        )
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
        attach_artifact_producer(
            self.model_manager, "model_manager", register, put, complete_for,
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
        self._rpc_host = RpcServiceHost(
            self.rpc_config,
            role_id,
            {
                TRAIN_TASK_MANAGER: self.task_manager,
                REPLAY_BUFFER: self.replay_buffer,
                MODEL_MANAGER: self.model_manager,
                SIM_MANAGER: self.sim_manager,
                MONTE_CARLO_METHOD: self.monte_carlo_method,
                ARTIFACTS_MANAGER: self.artifacts_manager,
            },
        )
        self._closed = False
        self.episode_results: list[dict] = []
        LOG.info(f"EvalControlRole initialized modules: role_id={role_id}")
        LOG.info("  TrainTaskManager")
        LOG.info(
            f"  ReplayBuffer(agent_capacities={self.replay_buffer.config.agent_capacities}, "
            f"batch_size={self.replay_buffer.config.batch_size})"
        )
        LOG.info(
            f"  ModelManager(tmp_local_dir={model_manager_config.tmp_local_dir}, "
            f"storage_type={model_manager_config.storage_type})"
        )
        LOG.info(f"  SimManager(resources={len(self.sim_manager.get_sim_list()) if self.sim_manager is not None else 0})")
        LOG.info(
            f"  MonteCarloMethod(write_enabled={self.monte_carlo_method.write_enabled}, "
            f"local_file={self.monte_carlo_method.local_file})"
        )

    def run(self) -> None:
        self._rpc_host.start()
        LOG.info(f"EvalControlRole {self.role_id} started.")
        try:
            while True:
                if self.task_manager.is_sample_completed():
                    LOG.info(
                        f"EvalControlRole {self.role_id} sample tasks completed, "
                        f"task_id={self.task_manager.task_id}."
                    )
                    break
                time.sleep(0.2)
            self._collect_eval_episode_results()
            self._log_summary()
            report_outputs = self.finished_report()
            self.task_manager.put_report_outputs(report_outputs)
            complete_role_artifact_producers(
                self, "replay_buffer", "monte_carlo_method", "model_manager",
            )
            complete_role_logs(
                self.artifacts_manager,
                self.task_manager.task_id,
                self.task_manager.stage_name,
                self.role_id,
            )
            self.task_manager.notify_task_complete()
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._rpc_host.stop()
        try:
            if self.sim_manager is not None:
                self.sim_manager.close()
        finally:
            if (
                    self.monte_carlo_method.write_enabled
                    and self.monte_carlo_method.config.storage_type == "s3"
                    and self.monte_carlo_method.current_sample_count > 0
            ):
                self.monte_carlo_method.upload_to_oss()
            self.monte_carlo_method.close()

from __future__ import annotations

from common.simulation import close_sim_manager_on_init_error, create_sim_manager

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimConfig, SimManagerConfig
    from dodo_sim.sim_manager import SimManager


import copy
import time
from typing import Optional

from common.logger import init_logger
from common.types import AgentName
from envs.base import BaseEnvironment
from config import (
    AgentConfig,
    EnvConfig,
    GrpcConfig,
    LearnerConfig,
    ModelManagerConfig,
    MonteCarloMethodConfig,
    ReplayBufferConfig,
    TrainConfig,
    SampleType,
)
from models import ModelManager
from scheduler.rollout_learn import RolloutLearnWorker
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
from scheduler.artifact import ArtifactsManager
from roles.common.automl_integration import (
    create_auto_ml_if_enabled,
    env_apply_restart_metadata,
    learners_apply_restart_advice,
)
from roles.common.config_alignment import (
    assert_state_shape_matches_env,
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
from scheduler.train_task import TrainTaskManager
from utils.module_utils import (
    align_agent_configs,
    build_env,
    build_learners,
    build_policy_agents,
    latest_metrics_by_learner,
    prepare_env_agent_names,
    publish_learners_to_model_manager,
    updated_learners_from_result,
)
from utils.random_seed import apply_train_seed

LOG = init_logger("learn_control")


class LearnControlRole:
    """
    简化分布式训练的中心角色，把ControlRole、LearnRole和ModelRole合并到同一进程。

    中心模块之间直接使用本地对象；SampleRole仍通过原有RPC接口领取任务、写入样本、
    申请仿真并拉取ModelManager发布的完整模型快照。
    """

    @close_sim_manager_on_init_error
    def __init__(
            self,
            role_id: str,
            train_config: TrainConfig,
            env_config: EnvConfig,
            sim_config: SimConfig,
            step_agents_config: list[AgentConfig],
            learners_config: list[LearnerConfig],
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
        # RPC 监听地址：各中心模块对外暴露的 gRPC endpoint（SampleRole 远程连接）。
        self.role_id = role_id
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
        env_config.metadata = BaseEnvironment._deep_merge_metadata(
            env_config.metadata or {}, train_config.env_metadata,
        )
        self.train_config = train_config
        self.task_manager_address = task_manager_address
        self.replay_buffer_address = replay_buffer_address
        self.model_manager_address = model_manager_address
        self.sim_manager_address = sim_manager_address
        self.monte_carlo_method_address = monte_carlo_method_address
        self.rpc_config = rpc_config or GrpcConfig(
            task_manager_address=task_manager_address,
            replay_buffer_address=replay_buffer_address,
            model_manager_address=model_manager_address,
            sim_manager_address=sim_manager_address,
            monte_carlo_method_address=monte_carlo_method_address,
        )
        # 全局种子：写入 TrainConfig，供采样/学习令牌派生 episode 级 seed。
        apply_train_seed(
            train_config,
            role_id=role_id,
            context=f"LearnControlRole:{role_id}",
        )

        # ---- ControlRole：任务总线、回放池、仿真资源与可选样本录制 ----
        # AutoML：Role 预创建；None 时 TrainTaskManager 不启监控线程。
        self.auto_ml = create_auto_ml_if_enabled(
            train_config.auto_ml_config,
            learners_config,
        )
        # 任务总线：采样/学习令牌权威；SampleRole 经 RPC 领取 episode / 上报结果。
        self.task_manager = TrainTaskManager(
            config=train_config,
            policy_agent_configs=step_agents_config,
            learner_configs=learners_config,
            env_name=env_config.sense_name,
            auto_ml=self.auto_ml,
            artifacts_manager=self.artifacts_manager,
            supports_versioned_sampling=True,
        )
        # 回放池：SampleRole RPC 写入 episode，本进程 LearnRole 本地读取。
        inherit_replay_buffer_seed(replay_buffer_config, train_config)
        self.replay_buffer = ReplayBuffer(config=replay_buffer_config)
        apply_learner_capacities(self.replay_buffer, learners_config)
        # 回放池攒够 batch 后回调 TM，由 TM 决定是否释放对应 episode 的学习令牌。
        self.replay_buffer.learn_batches_update_caller = self.task_manager.learn_batches_update
        # TM 代次切换时清空回放池、恢复 Learner 快照、应用 AutoML 超参并重新发布模型。
        self.task_manager.add_restart_callback(self._on_task_restart)
        self.task_manager.register_learner_save_caller(self._learner_save_caller)
        # 仿真资源池：SampleRole RPC 申请/释放仿真连接。
        self.sim_manager = create_sim_manager(config=sim_manager_config, sample_type=train_config.sample_type)
        # 中心样本录制：是否落盘由 train_config.record_sample 决定。
        record_config = copy.deepcopy(train_config.monte_carlo_method_config or monte_carlo_method_config)
        record_config.write_enabled = bool(
            train_config.record_sample and train_config.sample_type == SampleType.ONLINE.value
        )
        self.monte_carlo_method = MonteCarloMethod(
            task_id=self.task_manager.task_id,
            config=record_config,
        )

        # ---- ModelRole：Learner 更新与 Sample 拉取之间的模型快照边界 ----
        self.model_manager = ModelManager(config=model_manager_config)

        # ---- LearnRole：本进程消费学习令牌，不直连仿真 ----
        prepare_env_agent_names(env_config, step_agents_config)
        align_agent_configs(step_agents_config, learners_config)
        assert_state_shape_matches_env(env_config, step_agents_config, learners_config)
        self.policy_agents = build_policy_agents(step_agents_config)
        # 学习环境桩：仅提供 obs/action 维度，connect_sim=False。
        self.env = build_env(env_config, sim_config, self.policy_agents, connect_sim=False)
        if train_config.sample_type == SampleType.OFFLINE.value:
            count = self.monte_carlo_method.prepare_offline_index(
                stage_name=self.task_manager.stage_name,
                expected_agents=list(self.env.possible_agents),
            )
            self.task_manager.prepare_offline_sample_plan(count)
        self.learners = build_learners(
            train_config=train_config,
            env=self.env,
            policy_agents=self.policy_agents,
            learners_config=learners_config,
            model_manager=self.model_manager,
        )
        # 滚动学习：从 ReplayBuffer 取 batch，向 TM 领取/回报学习令牌。
        self.rollout_worker = RolloutLearnWorker(
            learners=self.learners,
            train_config=train_config,
            replay_buffer=self.replay_buffer,
            task_manager=self.task_manager,
        )
        # 首版模型发布：SampleRole 启动后必须先能拉到结构正确的 Agent 网络。
        publish_learners_to_model_manager(
            learners=self.learners,
            model_manager=self.model_manager,
            task_id=self.task_manager.task_id,
            reason="simplified_distributed_init",
            train_config=train_config,
            task_manager=self.task_manager,
            replay_buffer=self.replay_buffer,
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
        for learner in self.learners:
            attach_artifact_producer(
                learner, f"learner:{learner.name}", register, put, complete_for,
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
        self._log_initialized_modules(model_manager_config)

    def run(self) -> None:
        """启动全部中心RPC服务，并在当前进程持续消费在线学习令牌。"""
        self._start_servers()
        LOG.info(f"LearnControlRole {self.role_id} started.")
        completed = False
        try:
            while True:
                sample_completed = self.task_manager.is_sample_completed()
                learning_completed = self.task_manager.is_learning_completed()
                if sample_completed and learning_completed:
                    LOG.info(
                        f"LearnControlRole {self.role_id} finished: "
                        f"sample and learning tasks completed, task_id={self.task_manager.task_id}."
                    )
                    completed = True
                    break
                if learning_completed:
                    time.sleep(0.2)
                    continue

                # 学习一个批次
                learn_results = self.rollout_worker.rollout_after_episode_updated()
                if not learn_results:
                    time.sleep(0.2)
                    continue
                # 筛选本批次实际更新的 Learner
                updated_learners = updated_learners_from_result(self.learners, learn_results)
                # Learner 更新后的模型提交到 ModelManager
                publish_learners_to_model_manager(
                    learners=updated_learners,
                    model_manager=self.model_manager,
                    task_id=self.task_manager.task_id,
                    metrics=latest_metrics_by_learner(learn_results),
                    is_learning_batch=True,
                    reason="simplified_distributed_online_train",
                    learn_info=self.rollout_worker.last_learn_info,
                    train_config=self.train_config,
                    task_manager=self.task_manager,
                    replay_buffer=self.replay_buffer,
                )
            if completed:
                # 录制器在 close 时 flush + put 样本文件，必须先于阶段归档。
                self._close_sample_recording()
                complete_artifact_producers(
                    self.replay_buffer,
                    self.monte_carlo_method,
                    self.model_manager,
                    *self.learners,
                )
                complete_role_logs(
                    self.artifacts_manager,
                    self.task_manager.task_id,
                    self.task_manager.stage_name,
                    self.role_id,
                )
                # 归档等待采样产物 complete 时，RPC 必须仍可接受 worker 回传。
                self.task_manager.notify_stage_complete()
        finally:
            self.close()

    def _start_servers(self) -> None:
        self._rpc_host.start()

    def _log_initialized_modules(self, model_manager_config: ModelManagerConfig) -> None:
        LOG.info(f"LearnControlRole initialized modules: role_id={self.role_id}")
        LOG.info("  TrainTaskManager")
        LOG.info(
            f"  ReplayBuffer(agent_capacities={self.replay_buffer.config.agent_capacities}, "
            f"batch_size={self.replay_buffer.config.batch_size})"
        )
        LOG.info(f"  RolloutLearnWorker(learners={[learner.name for learner in self.learners]})")
        LOG.info(f"  Agents({[agent.name for agent in self.policy_agents]})")
        LOG.info(f"  Environment({type(self.env).__name__}, connect_sim=False)")
        LOG.info(
            f"  ModelManager(tmp_local_dir={model_manager_config.tmp_local_dir}, "
            f"storage_type={model_manager_config.storage_type})"
        )
        LOG.info(f"  SimManager(resources={len(self.sim_manager.get_sim_list()) if self.sim_manager is not None else 0})")
        LOG.info(
            f"  MonteCarloMethod(write_enabled={self.monte_carlo_method.write_enabled}, "
            f"local_file={self.monte_carlo_method.local_file})"
        )

    def _on_task_restart(self, generation: int, advice=None) -> None:
        """TaskManager 重启训练时，按代次恢复Learner快照并重新发布给采样进程。"""
        self.replay_buffer.clear(generation=generation)
        env_apply_restart_metadata(self.env, self.task_manager)
        for learner in self.learners:
            learner.restore_models_snapshot_for_generation(generation)
        learners_apply_restart_advice({learner.name: learner for learner in self.learners}, advice)
        apply_replay_capacity_advice(self.replay_buffer, advice)
        publish_learners_to_model_manager(
            learners=self.learners,
            model_manager=self.model_manager,
            task_id=self.task_manager.task_id,
            metrics={},
            promote=True,
            reason=f"task_restart:generation_{generation}",
            train_config=self.train_config,
            task_manager=self.task_manager,
            replay_buffer=self.replay_buffer,
        )
        LOG.warning(f"LearnControlRole handled task restart: generation={generation}")

    def _learner_save_caller(self, name_prefix: str) -> list[str]:
        paths: list[str] = []
        for learner in self.learners:
            filenames = {
                model_name: self.task_manager.resolve_model_filename(
                    name_prefix, learner.name, model_name,
                )
                for model_name in learner.models
            }
            paths.extend(
                learner.save_models(self.train_config.model_save_dir, filenames)
            )
        return paths

    def close(self) -> None:
        """停止RPC服务并释放学习环境、仿真管理和样本录制资源。"""
        if self._closed:
            return
        self._closed = True
        self._rpc_host.stop()
        try:
            self.env.close()
        finally:
            if self.sim_manager is not None:
                self.sim_manager.close()
            self._close_sample_recording()

    def _close_sample_recording(self) -> None:
        recorder = self.monte_carlo_method
        if (
                recorder.write_enabled
                and recorder.config.storage_type == "s3"
                and recorder.current_sample_count > 0
                and recorder.file is not None
                and not recorder.file.closed
        ):
            recorder.upload_to_oss()
        recorder.close()

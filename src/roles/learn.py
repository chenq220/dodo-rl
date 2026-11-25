from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimConfig


import time
from typing import TYPE_CHECKING, Optional

from common.logger import init_logger
from envs.base import BaseEnvironment
from config import AgentConfig, EnvConfig, GrpcConfig, LearnerConfig, TorchRpcConfig, TrainConfig
from scheduler.rollout_learn import RolloutLearnWorker
from common.exception import RpcServerGone
from rpc.dispatch import ARTIFACTS_MANAGER, MODEL_MANAGER, REPLAY_BUFFER, TRAIN_TASK_MANAGER
from rpc.factory import (
    RpcConfiguration,
    create_rpc_client,
    init_rpc_client_runtime,
    shutdown_rpc_client_runtime,
)
from roles.common.config_alignment import assert_state_shape_matches_env
from roles.common.artifact_wiring import (
    bind_local_artifact_callers,
    bind_artifact_producer_callers,
    complete_role_logs,
    declare_role_log_producer,
)
from scheduler.artifact import ArtifactsManager
from utils.random_seed import apply_train_seed
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

if TYPE_CHECKING:
    from models import ModelManager
    from sample.replay_buffer import ReplayBuffer
    from scheduler.train_task import TrainTaskManager

LOG = init_logger("learn")


class LearnRole:
    """学习进程角色：持有Learner/RolloutLearnWorker，通过RPC消费学习令牌、采样batch并提交模型。"""

    def __init__(
            self,
            role_id: str,
            train_config: TrainConfig,
            env_config: EnvConfig,
            sim_config: SimConfig,
            step_agents_config: list[AgentConfig],
            learners_config: list[LearnerConfig],
            task_manager_address: str,
            replay_buffer_address: str,
            model_manager_address: str,
            rpc_config: RpcConfiguration | None = None,
            artifacts_manager: Optional[ArtifactsManager] = None,
    ):
        self.role_id = role_id
        self.train_config = train_config
        apply_train_seed(
            train_config,
            role_id=role_id,
            context=f"LearnRole:{role_id}",
        )
        env_config.metadata = BaseEnvironment._deep_merge_metadata(
            env_config.metadata or {}, train_config.env_metadata,
        )
        self.rpc_config = rpc_config or GrpcConfig(
            task_manager_address=task_manager_address,
            replay_buffer_address=replay_buffer_address,
            model_manager_address=model_manager_address,
        )
        init_rpc_client_runtime(self.rpc_config, role_id)
        self.task_manager: TrainTaskManager = create_rpc_client(self.rpc_config, TRAIN_TASK_MANAGER)
        self.replay_buffer: ReplayBuffer = create_rpc_client(self.rpc_config, REPLAY_BUFFER)
        self.model_manager: ModelManager = create_rpc_client(self.rpc_config, MODEL_MANAGER)
        if artifacts_manager is None:
            self.artifacts_manager = create_rpc_client(self.rpc_config, ARTIFACTS_MANAGER)
            self._bind_local_learners = False
        else:
            self.artifacts_manager = artifacts_manager
            self._bind_local_learners = True

        prepare_env_agent_names(env_config, step_agents_config)
        align_agent_configs(step_agents_config, learners_config)
        assert_state_shape_matches_env(env_config, step_agents_config, learners_config)
        self.policy_agents = build_policy_agents(step_agents_config)
        # Learner只需要环境维度和业务接口，不在学习进程连接真实仿真。
        self.env = build_env(env_config, sim_config, self.policy_agents, connect_sim=False)
        self.learners = build_learners(
            train_config=train_config,
            env=self.env,
            policy_agents=self.policy_agents,
            learners_config=learners_config,
            model_manager=self.model_manager,
        )
        register, put, complete_for = bind_local_artifact_callers(
            self.artifacts_manager,
            self._task_id,
            lambda: "train",
        )
        if self._bind_local_learners:
            for learner in self.learners:
                bind_artifact_producer_callers(
                    learner, f"learner:{learner.name}", register, put, complete_for,
                )
                learner.register_stage_intents()
        self.rollout_worker = RolloutLearnWorker(
            learners=self.learners,
            train_config=train_config,
            replay_buffer=self.replay_buffer,
            task_manager=self.task_manager,
        )
        # 首版模型先提交到模型池，采样进程启动后即可拉取非随机结构权重。
        publish_learners_to_model_manager(
            learners=self.learners,
            model_manager=self.model_manager,
            task_id=self._task_id(),
            reason="distributed_init",
        )
        LOG.info(f"LearnRole initialized modules: role_id={self.role_id}")
        LOG.info("  RolloutLearnWorker")
        LOG.info(f"  Learners({[learner.name for learner in self.learners]})")
        LOG.info(f"  Agents({[agent.name for agent in self.policy_agents]})")
        LOG.info(f"  Environment({type(self.env).__name__}, connect_sim=False)")
        LOG.info(f"  TrainTaskManagerRpcClient({task_manager_address})")
        LOG.info(f"  ReplayBufferRpcClient({replay_buffer_address})")
        LOG.info(f"  ModelManagerRpcClient({model_manager_address})")

    def run(self) -> None:
        LOG.info(f"LearnRole {self.role_id} started.")
        submitted_log_stages: set[str] = set()
        current_log_stage: str | None = None
        try:
            while True:
                task_id_value, stage = self._task_id(), self._stage_name()
                if stage and current_log_stage != stage:
                    if current_log_stage and current_log_stage not in submitted_log_stages:
                        complete_role_logs(
                            self.artifacts_manager, task_id_value, current_log_stage, self.role_id,
                        )
                        submitted_log_stages.add(current_log_stage)
                    declare_role_log_producer(
                        self.artifacts_manager, task_id_value, stage, self.role_id,
                    )
                    current_log_stage = stage
                if self.task_manager.is_learning_completed():
                    if current_log_stage and current_log_stage not in submitted_log_stages:
                        complete_role_logs(
                            self.artifacts_manager, task_id_value, current_log_stage, self.role_id,
                        )
                        submitted_log_stages.add(current_log_stage)
                    if isinstance(self.rpc_config, TorchRpcConfig):
                        time.sleep(1.0)
                        continue
                    LOG.info(f"LearnRole {self.role_id} finished: learning task completed, task_id={self._task_id()}.")
                    return
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
                    task_id=self._task_id(),
                    metrics=latest_metrics_by_learner(learn_results),
                    is_learning_batch=True,
                    reason="distributed_online_train",
                    learn_info=self.rollout_worker.last_learn_info,
                )
        except RpcServerGone as exc:
            if current_log_stage and current_log_stage not in submitted_log_stages:
                try:
                    complete_role_logs(
                        self.artifacts_manager, self._task_id(), current_log_stage, self.role_id,
                    )
                except Exception:
                    LOG.exception("LearnRole log ingest failed after server gone")
            LOG.info(f"LearnRole {self.role_id} finished: central RPC server gone, task_id={self._task_id()}. {exc}")
        finally:
            self.close()

    def close(self) -> None:
        try:
            self.env.close()
        finally:
            for client in (self.model_manager, self.replay_buffer, self.task_manager):
                close = getattr(client, "close", None)
                if callable(close):
                    close()
            shutdown_rpc_client_runtime(self.rpc_config)

    def _task_id(self) -> str:
        try:
            return self.task_manager.get_task_manager_info()["task_id"]
        except Exception:
            return self.role_id

    def _stage_name(self) -> str:
        try:
            return str(self.task_manager.get_task_manager_info().get("stage_name") or "")
        except Exception:
            return ""

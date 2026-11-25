from __future__ import annotations

from common.simulation import close_sim_manager_on_init_error, create_sim_manager, apply_stage_sim_metadata

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimConfig, SimManagerConfig
    from dodo_sim.sim_manager import SimManager


import copy
import time
from typing import Optional

from common.logger import init_logger
from envs.base import BaseEnvironment
from common.types import AgentName
from config import (
    EnvConfig,
    GrpcConfig,
    ModelManagerConfig,
    MonteCarloMethodConfig,
    ReplayBufferConfig,
    StageConfig,
    StageTaskConfig,
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
from scheduler.stage_task import StageTask, StageTaskManagerProxy
from roles.common.automl_integration import (
    build_auto_ml_by_stage,
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
    bind_artifact_producer_callers,
    declare_stage_artifact_producer,
    complete_artifact_producers,
    declare_role_log_producer,
    complete_role_logs,
)
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
from utils.stage_utils import (
    apply_stage_to_resources,
    load_skip_stage_replay_if_needed,
)
from utils.task_id import task_id

LOG = init_logger("stage_learn_control")


class StageLearnRole:
    """
    简化分布式（1+N）分阶段训练中心角色，是 LearnControlRole 的分阶段版本。

    与 LearnControlRole 的关系：LearnControlRole 等价于只有一个阶段的 StageLearnRole。
    本角色把“阶段任务生命周期”交给 StageTask（只实例化/回收 TrainTaskManager），自己负责
    StageTask 之外的全部阶段切换动作：写入 env.stage_name、按需清空回放池、重置在场 Learner
    超参、阶段模型保存、阶段结束 Learner 快照保存与跨阶段继承（经 ModelManager）。

    中心 TrainTaskManager RPC 通过 StageTaskManagerProxy 始终服务当前已就绪阶段，采样进程
    （SampleRole）按采样令牌携带的 stage_name 自行切换其本地环境阶段；全部阶段训练结束后
    本角色自然退出，中心 RPC 服务关闭，采样进程重试若干次连不上即自行优雅退出。
    """

    @close_sim_manager_on_init_error
    def __init__(
            self,
            role_id: str,
            stage_task_config: StageTaskConfig,
            env_config: EnvConfig,
            sim_config: SimConfig,
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
        # ---- 入口约束 ----
        assert stage_task_config.stages, "StageLearnRole requires StageTaskConfig.stages."
        # RPC 监听地址：各中心模块对外暴露的 gRPC endpoint（SampleRole 远程连接）。
        self.role_id = role_id
        self.env_config = env_config
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
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

        # ---- 阶段计划：StageTask 按序实例化/回收各阶段 TrainTaskManager ----
        # 各阶段 AutoML 由 Role 预创建；StageTask 只按 stage_name 注入，不再自行构造。
        self.auto_ml_by_stage = build_auto_ml_by_stage(stage_task_config.stages)
        self.stage_task = StageTask(
            stage_task_config,
            auto_ml_by_stage=self.auto_ml_by_stage,
            env_name=env_config.sense_name,
            artifacts_manager=self.artifacts_manager,
            supports_versioned_sampling=True,
        )
        # TM RPC 代理：SampleRole 始终连同一 TM 代理地址，阶段切换时转发到 current_task_manager。
        self.task_manager_proxy = StageTaskManagerProxy(self.stage_task)
        # 全阶段 Agent/Learner 并集：跨阶段只构造一次，换阶段时只改超参与令牌。
        union_agents_config = self.stage_task.all_agent_configs
        union_learners_config = self.stage_task.all_learner_configs
        assert union_learners_config, "StageLearnRole requires at least one learner across stages."
        # 全局种子：以配置的入口阶段TrainConfig初始化，各阶段进入时再按stage覆写。
        apply_train_seed(
            self.stage_task.starting_stage.train_config,
            role_id=role_id,
            context=f"StageLearnRole:{role_id}:init",
        )
        env_config.metadata = BaseEnvironment._deep_merge_metadata(
            env_config.metadata or {}, self.stage_task.starting_stage.train_config.env_metadata,
        )

        # ---- ControlRole：回放池、仿真资源与中心样本录制（TM 按阶段在 StageTask 内创建）----
        inherit_replay_buffer_seed(replay_buffer_config, self.stage_task.starting_stage.train_config)
        self.replay_buffer = ReplayBuffer(config=replay_buffer_config)
        apply_learner_capacities(self.replay_buffer, self.stage_task.starting_stage.learners_config)
        # 回放池 RPC 写入后，经 _current_learn_batches_update 转发到「当前阶段」TM 释放学习令牌。
        self.replay_buffer.learn_batches_update_caller = self._current_learn_batches_update
        # 仿真资源池：SampleRole RPC 申请/释放仿真连接。
        self.sim_manager = create_sim_manager(
            config=sim_manager_config,
            sample_type="offline" if all(s.train_config.sample_type == "offline"
                                         for s in stage_task_config.stages) else "online",
        )
        # 中心样本录制：分阶段默认关闭；task_id 用 role_id 命名。
        record_config = copy.deepcopy(
            self.stage_task.starting_stage.train_config.monte_carlo_method_config
            or monte_carlo_method_config
        )
        record_config.write_enabled = False
        self.monte_carlo_method = MonteCarloMethod(task_id=role_id, config=record_config)

        # ---- ModelRole：Learner 更新与 Sample 拉取之间的模型快照边界 ----
        self.model_manager = ModelManager(config=model_manager_config)

        # ---- LearnRole：本进程消费学习令牌，不直连仿真 ----
        prepare_env_agent_names(env_config, union_agents_config)
        align_agent_configs(union_agents_config, union_learners_config)
        assert_state_shape_matches_env(env_config, union_agents_config, union_learners_config)
        self.policy_agents = build_policy_agents(union_agents_config)
        # 学习环境桩：单实例描述所有阶段，差异由 env.stage_name 驱动（在 _apply_stage 写入）。
        self.env = build_env(env_config, sim_config, self.policy_agents, connect_sim=False)
        # Learner 按并集构造，网络跨阶段不重置；换阶段时 apply_stage_to_resources 更新超参。
        self.learners = build_learners(
            train_config=self.stage_task.starting_stage.train_config,
            env=self.env,
            policy_agents=self.policy_agents,
            learners_config=union_learners_config,
            model_manager=self.model_manager,
        )
        self.learners_by_name = {learner.name: learner for learner in self.learners}
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
        bind_artifact_producer_callers(
            self.model_manager, "model_manager", register, put, complete_for,
        )
        for learner in self.learners:
            bind_artifact_producer_callers(
                learner, f"learner:{learner.name}", register, put, complete_for,
            )

        # gRPC 服务列表；run() 时逐个 bind 并 start。
        # 注意：restart / model_save 回调在 _run_stage 按阶段挂接到当前 TM。
        self._rpc_host = RpcServiceHost(
            self.rpc_config,
            role_id,
            {
                TRAIN_TASK_MANAGER: self.task_manager_proxy,
                REPLAY_BUFFER: self.replay_buffer,
                MODEL_MANAGER: self.model_manager,
                SIM_MANAGER: self.sim_manager,
                MONTE_CARLO_METHOD: self.monte_carlo_method,
                ARTIFACTS_MANAGER: self.artifacts_manager,
            },
        )
        self._closed = False
        self._current_task_manager = None
        self._skip_stage_replay_loaded = False
        LOG.info(
            f"StageLearnRole initialized: role_id={role_id}, "
            f"stages={len(stage_task_config.stages)}"
        )
        LOG.info(f"  Agents({[agent.name for agent in self.policy_agents]})")
        LOG.info(f"  Learners({[learner.name for learner in self.learners]})")
        LOG.info(f"  Environment({type(self.env).__name__}, connect_sim=False)")

    def _current_learn_batches_update(self, replay_buffer_state, episode_num: int):
        """把回放池学习通知转发到当前阶段的任务总线。"""
        task_manager = self.stage_task.current_task_manager
        if task_manager is None:
            return
        task_manager.learn_batches_update(replay_buffer_state, episode_num)

    def run(self) -> None:
        """启动中心RPC服务，按阶段顺序消费学习令牌；全部阶段完成后自然退出。"""
        self._start_servers()
        LOG.info(f"StageLearnRole {self.role_id} started.")
        try:
            for task_manager in self.stage_task:
                stage = self.stage_task.current_stage
                self._run_stage(task_manager, stage)
            LOG.info(f"StageLearnRole {self.role_id} finished: all stages completed, task_id={task_id}.")
        finally:
            self.close()

    def _run_stage(self, task_manager, stage: StageConfig) -> None:
        """运行单个阶段的在线学习，直到该阶段采样完成。"""
        self._current_task_manager = task_manager
        LOG.info(f"StageLearnRole enter stage: stage_name={stage.stage_name}")
        try:
            # StageTask统一控制切换门闩：上下文内禁止领取，资源全部准备成功后自动开放。
            with self.stage_task.preparing_current_stage(task_manager):
                apply_stage_sim_metadata(getattr(self, "sim_manager", None), stage.train_config)
                apply_train_seed(
                    stage.train_config,
                    role_id=f"{self.role_id}:{stage.stage_name}",
                    context=f"StageLearnRole:{self.role_id}:{stage.stage_name}",
                )
                # 写入 env.stage_name、清空回放池、同步在场 Learner 超参。
                self._apply_stage(stage)
                # 当前阶段 TM 代次切换时清空回放池、恢复快照、应用 AutoML 超参并重新发布模型。
                task_manager.add_restart_callback(self._on_task_restart)
                task_manager.register_learner_save_caller(
                    lambda prefix: self._learner_save_caller(prefix, stage, task_manager),
                )
                # 阶段起点发布全量 Learner，SampleRole 拉取本阶段初始策略（含跨阶段继承网络）。
                self._publish_stage_init(stage, task_manager)
                # 学习滚动流程：挂接当前阶段 TM 与共享 ReplayBuffer。
                rollout_worker = self._build_learn_worker(task_manager, stage)
                self._declare_stage_artifact_producers(task_manager)

            LOG.info(f"StageLearnRole stage ready for sampling: stage_name={stage.stage_name}")

            while True:
                if (
                        task_manager.is_task_finished()
                        and task_manager.is_learning_completed()
                ):
                    # 业务完成仍以采样结束为准；资源切换前再排空在途与尾部许可。
                    break
                # 学习一个批次
                learn_results = rollout_worker.rollout_after_episode_updated()
                if not learn_results:
                    time.sleep(0.2)
                    continue
                # 筛选本批次实际更新的 Learner
                updated_learners = updated_learners_from_result(self.learners, learn_results)
                # Learner 更新后的模型提交到 ModelManager
                publish_learners_to_model_manager(
                    learners=updated_learners,
                    model_manager=self.model_manager,
                    task_id=task_manager.task_id,
                    metrics=latest_metrics_by_learner(learn_results),
                    is_learning_batch=True,
                    reason=f"stage_online_train:{stage.stage_name}",
                    learn_info=rollout_worker.last_learn_info,
                    train_config=stage.train_config,
                    task_manager=task_manager,
                    replay_buffer=self.replay_buffer,
                )

            # 本阶段跑满后先关闭入口，再执行模型保存和后续阶段切换。
            self.stage_task.mark_current_stage_unready(task_manager)
            self.monte_carlo_method.close()
            replay_path = self.replay_buffer.notify_stage_complete(
                stage, task_id=task_manager.task_id,
            )
            complete_artifact_producers(
                self.replay_buffer,
                self.monte_carlo_method,
                self.model_manager,
                *self.learners,
            )
            complete_role_logs(
                self.artifacts_manager, task_manager.task_id, task_manager.stage_name, self.role_id,
            )
            # 保存 final 模型、通知 AutoML，并自动归档本阶段已登记产物。
            task_manager.notify_stage_complete()
            self._clone_learner_snapshots(stage)
            LOG.info(f"StageLearnRole finished stage: stage_name={stage.stage_name}")
        finally:
            # 初始化、训练或收尾任一步骤异常时，都不能继续发放该阶段令牌。
            self.stage_task.mark_current_stage_unready(task_manager)

    def _apply_stage(self, stage: StageConfig) -> None:
        """阶段切换：env.stage_name、按需清空回放池、重置在场 Learner 超参。"""
        task_manager = self.stage_task.current_task_manager
        config = copy.deepcopy(stage.train_config.monte_carlo_method_config or self.monte_carlo_method.config)
        config.write_enabled = False
        self.monte_carlo_method.configure(config)
        if stage.train_config.sample_type == SampleType.OFFLINE.value:
            count = self.monte_carlo_method.prepare_offline_index(
                stage_name=stage.stage_name,
                expected_agents=(list(self.env_config.agent_names)
                                 or [agent.name for agent in stage.agents_config]),
            )
            task_manager = self.stage_task.current_task_manager
            assert task_manager is not None
            task_manager.prepare_offline_sample_plan(count)
        if load_skip_stage_replay_if_needed(
                self.stage_task.stage_task_config,
                self.replay_buffer,
                already_loaded=self._skip_stage_replay_loaded,
        ):
            self._skip_stage_replay_loaded = True
        apply_stage_to_resources(
            stage=stage,
            env=self.env,
            learners_by_name=self.learners_by_name,
            replay_buffer=self.replay_buffer,
        )

    def _declare_stage_artifact_producers(self, task_manager) -> None:
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
        declare_stage_artifact_producer(
            self.model_manager, "model_manager",
            self.artifacts_manager, current_task_id, current_stage_name,
        )
        for learner in self.learners:
            declare_stage_artifact_producer(
                learner, f"learner:{learner.name}",
                self.artifacts_manager, current_task_id, current_stage_name,
            )
        declare_role_log_producer(
            self.artifacts_manager, current_task_id, current_stage_name, self.role_id,
        )

    def _publish_stage_init(self, stage: StageConfig, task_manager) -> None:
        """阶段起点把当前网络发布到模型池，让采样进程拿到本阶段初始策略（含跨阶段继承的网络）。"""
        publish_learners_to_model_manager(
            learners=self.learners,
            model_manager=self.model_manager,
            task_id=task_manager.task_id,
            reason=f"stage_init:{stage.stage_name}",
            train_config=stage.train_config,
            task_manager=task_manager,
            replay_buffer=self.replay_buffer,
        )

    def _on_task_restart(self, generation: int, advice=None) -> None:
        """TaskManager 被外部要求重启后，按代次恢复Learner快照并重新发布给采样进程。"""
        self.replay_buffer.clear(generation=generation)
        env_apply_restart_metadata(self.env, self.stage_task.current_task_manager)
        for learner in self.learners:
            learner.restore_models_snapshot_for_generation(generation)
        learners_apply_restart_advice({learner.name: learner for learner in self.learners}, advice)
        apply_replay_capacity_advice(self.replay_buffer, advice)
        task_manager = self.stage_task.current_task_manager
        if task_manager is None:
            return
        train_config = (
            task_manager.task_config
            if isinstance(task_manager.task_config, TrainConfig)
            else None
        )
        publish_learners_to_model_manager(
            learners=self.learners,
            model_manager=self.model_manager,
            task_id=task_manager.task_id,
            metrics={},
            promote=True,
            reason=f"stage_task_restart:generation_{generation}",
            train_config=train_config,
            task_manager=task_manager,
            replay_buffer=self.replay_buffer,
        )
        LOG.warning(f"StageLearnRole handled task restart: generation={generation}")

    def _learner_save_caller(
            self,
            name_prefix: str,
            stage: StageConfig,
            task_manager,
    ) -> list[str]:
        paths: list[str] = []
        for learner in self.learners:
            filenames = {
                model_name: task_manager.resolve_model_filename(
                    name_prefix, learner.name, model_name,
                )
                for model_name in learner.models
            }
            paths.extend(
                learner.save_models(stage.train_config.model_save_dir, filenames)
            )
        return paths

    def _build_learn_worker(self, task_manager, stage: StageConfig) -> RolloutLearnWorker:
        """按当前阶段 TM 构造学习滚动流程；Learner/ReplayBuffer 跨阶段共享。"""
        return RolloutLearnWorker(
            learners=self.learners,
            train_config=stage.train_config,
            replay_buffer=self.replay_buffer,
            task_manager=task_manager,
        )

    def _clone_learner_snapshots(self, stage: StageConfig) -> None:
        """阶段完成后刷新默认快照，后续任务重启可回到最近一个已完成阶段的模型状态。"""
        for learner in self.learners:
            learner.clone_models_snapshot()
        LOG.info(f"StageLearnRole cloned learner snapshots after stage: stage_name={stage.stage_name}")

    def _start_servers(self) -> None:
        self._rpc_host.start()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._rpc_host.stop()
        self.stage_task.close()
        try:
            self.replay_buffer.close()
        finally:
            try:
                self.env.close()
            finally:
                if self.sim_manager is not None:
                    self.sim_manager.close()
                self.monte_carlo_method.close()

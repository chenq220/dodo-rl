from __future__ import annotations

from common.simulation import close_sim_manager_on_init_error, create_sim_manager

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimConfig, SimManagerConfig
    from dodo_sim.sim_manager import SimManager


import time
import copy
from typing import Optional

from algos.agents import Agent
from algos.learners import Learner
from common.creator import create_learner
from common.exception import SampleTaskReturned
from common.logger import init_logger
from common.types import AgentName
from config import EnvConfig, ModelManagerConfig, ReplayBufferConfig, StageConfig, StageTaskConfig, SampleType
from models import ModelManager
from scheduler.rollout_learn import RolloutLearnWorker
from scheduler.rollout_sample import RolloutSampleWorker
from sample.replay_buffer import ReplayBuffer
from sample.monte_carlo import MonteCarloMethod
from scheduler.stage_task import StageTask
from roles.common.automl_integration import (
    build_auto_ml_by_stage,
    env_apply_restart_metadata,
    learners_apply_restart_advice,
)
from roles.common.archive_model_source import (
    required_model_names,
    resolve_learner_archive_model_files,
)
from roles.common.config_alignment import (
    assert_state_shape_matches_env,
    inherit_replay_buffer_seed,
    apply_learner_capacities,
    apply_replay_capacity_advice,
)
from roles.common.simulation_logging import log_simulation_middleware
from roles.common.artifact_wiring import (
    resolve_artifacts_manager,
    bind_local_artifact_callers,
    bind_artifact_producer_callers,
    declare_stage_artifact_producer,
    complete_artifact_producers,
    complete_role_gifs,
    complete_role_logs,
    declare_role_gif_producer,
    declare_role_log_producer,
    put_role_render_gif,
)
from scheduler.artifact import ArtifactsManager
from utils.module_utils import (
    align_agent_configs,
    build_agent_compose,
    build_env,
    build_policy_agents,
    latest_metrics_by_learner,
    prepare_env_agent_names,
    publish_learners_to_model_manager,
    sync_agent_from_model_manager,
    sync_agents_from_model_manager,
    updated_learners_from_result,
)
from utils.stage_utils import (
    apply_stage_to_resources,
    load_skip_stage_replay_if_needed,
)
from utils.random_seed import apply_train_seed
from utils.task_id import task_id

LOG = init_logger("serial_stage_train")


class SerialStageTrainRole:
    """
    生产用单进程串行分阶段训练角色，是 SerialTrainRole 的分阶段版本。

    与 SerialTrainRole 的关系：SerialTrainRole 等价于只有一个阶段的 SerialStageTrainRole。本角色把
    “阶段任务生命周期”交给 StageTask（只负责实例化/回收 TrainTaskManager），自己负责
    StageTask 之外的全部阶段切换动作：写入 env.stage_name、按需清空回放池、重置在场 Learner
    超参、阶段模型保存、阶段结束 Learner 快照保存。

    资源共享原则（参考需求约束）：
    - Agent 是训练对象，跨阶段只构造一次、不重新初始化。
    - 可选 ModelManagerConfig：学习后 submit_model 推送权重，Agent 从 ModelManager 拉 champion。
    - Learner 按全阶段并集构造一次并跨阶段复用（网络不重置），每阶段是否学习由该阶段
      TrainTaskManager 下发的学习令牌决定；某阶段没有对应 Learner 的 Agent 即固定策略。
    - Environment 一个实例描述所有阶段，差异由 env.stage_name 驱动。
    - ReplayBuffer 是缓存通道，阶段切换按配置直接清空，不重建对象。
    """

    @close_sim_manager_on_init_error
    def __init__(
            self,
            stage_task_config: StageTaskConfig,
            env_config: EnvConfig,
            sim_config: Optional[SimConfig],
            replay_buffer_config: ReplayBufferConfig,
            model_manager_config: Optional[ModelManagerConfig] = None,
            sim_manager_config: Optional[SimManagerConfig] = None,
            artifacts_manager: Optional[ArtifactsManager] = None,
    ):
        assert stage_task_config.stages, "SerialStageTrainRole requires StageTaskConfig.stages."

        # 运行元数据：产物命名时间戳、环境配置引用、关闭标志。
        self.timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.env_config = env_config
        self.role_id = "serial_stage_train"
        self._closed = False
        self.monte_carlo_method = None
        self._current_task_manager = None
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
        self.model_manager = (
            ModelManager(config=model_manager_config)
            if model_manager_config is not None
            else None
        )
        # manager 模式时，单进程串行角色本地持有资源池；Environment/Simulation 申请并释放资源。
        self.sim_manager = (
            create_sim_manager(
                config=sim_manager_config,
                sample_type="offline" if all(s.train_config.sample_type == "offline"
                                             for s in stage_task_config.stages) else "online",
            )
            if sim_manager_config is not None
            else None
        )

        # ---- 阶段计划：StageTask 按序实例化/回收各阶段 TrainTaskManager ----
        # 各阶段 AutoML 由 Role 预创建；StageTask 只按 stage_name 注入，不再自行构造。
        self.auto_ml_by_stage = build_auto_ml_by_stage(stage_task_config.stages)
        self.stage_task = StageTask(
            stage_task_config,
            auto_ml_by_stage=self.auto_ml_by_stage,
            env_name=env_config.sense_name,
            artifacts_manager=self.artifacts_manager,
        )
        # ---- 入口约束：模型版本填充完成后，再校验各阶段TrainConfig。 ----
        for stage in stage_task_config.stages:
            self._assert_stage_supported(
                stage, env_config, sim_config, model_manager_config, sim_manager_config,
            )
        # 全阶段 Agent/Learner 并集：跨阶段只构造一次，换阶段时只改超参与令牌。
        union_agents_config = self.stage_task.all_agent_configs
        union_learners_config = self.stage_task.all_learner_configs
        assert union_learners_config, "SerialStageTrainRole requires at least one learner across stages."
        self._bootstrap_train_config = self.stage_task.starting_stage.train_config

        # ---- 执行侧 Agent / 环境：全阶段并集；Learner 延至首阶段种子确定后再构造 ----
        prepare_env_agent_names(env_config, union_agents_config)
        align_agent_configs(union_agents_config, union_learners_config)
        assert_state_shape_matches_env(env_config, union_agents_config, union_learners_config)
        self.policy_agents = build_policy_agents(union_agents_config)
        self.agents_by_name = self._index_agents(self.policy_agents)
        self.learners: list[Learner] = []
        self.learners_by_name: dict[AgentName, Learner] = {}
        # 单环境实例：阶段差异由 env.stage_name 驱动（在 _apply_stage 写入）。
        self.env = build_env(
            env_config=env_config,
            sim_config=sim_config,
            policy_agents=self.policy_agents,
            connect_sim=any(s.train_config.sample_type != "offline" for s in stage_task_config.stages),
            sim_manager=self.sim_manager,
            role_id="serial-stage-train",
        )
        # Learner 在首阶段 _run_stage 中、TrainTaskManager 确定种子后再构造。

        # ---- 回放池：跨阶段共享同一实例，阶段切换时按配置清空缓存 ----
        # 注意：learn_batches_update_caller / restart / model_save 回调在 _apply_stage 按阶段挂接。
        inherit_replay_buffer_seed(replay_buffer_config, self._bootstrap_train_config)
        self.replay_buffer = ReplayBuffer(config=replay_buffer_config)
        apply_learner_capacities(self.replay_buffer, self.stage_task.starting_stage.learners_config)
        # 本机产物 register/put/complete 闭包；阶段 TM 切换后仍读当前 task_id/stage_name。
        self._register_artifact, self._put_artifact, self._complete_artifact_for = bind_local_artifact_callers(
            self.artifacts_manager,
            lambda: (
                self._current_task_manager.task_id
                if self._current_task_manager is not None
                else str(task_id)
            ),
            lambda: (
                self._current_task_manager.stage_name
                if self._current_task_manager is not None
                else "train"
            ),
        )
        bind_artifact_producer_callers(
            self.replay_buffer, "replay_buffer",
            self._register_artifact, self._put_artifact, self._complete_artifact_for,
        )
        if self.model_manager is not None:
            bind_artifact_producer_callers(
                self.model_manager, "model_manager",
                self._register_artifact, self._put_artifact, self._complete_artifact_for,
            )
        self._skip_stage_replay_loaded = False

        LOG.info("SerialStageTrainRole initialized:")
        LOG.info(f"  Stages({stage_task_config.stages.__len__()})")
        LOG.info(f"  Agents({list(self.agents_by_name)})")
        LOG.info(f"  Learners(pending union={list({lc.name for lc in union_learners_config})})")
        LOG.info(f"  Environment({type(self.env).__name__})")
        if self.model_manager is not None:
            LOG.info("  ModelManager(enabled)")
        if self.sim_manager is not None:
            LOG.info(f"  SimManager(file={self.sim_manager.config.config_file if self.sim_manager is not None else 'disabled'})")

    @staticmethod
    def _assert_stage_supported(
            stage: StageConfig,
            env_config: EnvConfig,
            sim_config: Optional[SimConfig],
            model_manager_config: Optional[ModelManagerConfig] = None,
            sim_manager_config: Optional[SimManagerConfig] = None,
    ) -> None:
        config = stage.train_config
        if str(config.model_version or "").strip() and not str(config.model_load_dir or "").strip():
            raise ValueError(
                f"Stage {stage.stage_name} configures model_version={config.model_version!r} "
                f"but model_load_dir is empty."
            )
        if env_config.use_sim:
            assert sim_config is not None, "SerialStageTrainRole requires sim_config when EnvConfig.use_sim=True."
            assert sim_config.config_source in {"local", "manager"}, (
                "SerialStageTrainRole SimConfig.config_source must be 'local' or 'manager'."
            )
            if sim_config.config_source == "manager":
                assert sim_manager_config is not None and sim_manager_config.use_sim, (
                    "SerialStageTrainRole with SimConfig.config_source='manager' requires an enabled SimManagerConfig."
                )

    def run(self) -> None:
        """按阶段顺序串行训练；每个阶段内部采样一局、学习一个批次、同步策略。"""
        LOG.info("SerialStageTrainRole started.")
        try:
            for task_manager in self.stage_task:
                stage = self.stage_task.current_stage
                self._run_stage(task_manager, stage)
            LOG.info(f"SerialStageTrainRole finished: process run completed, task_id={task_id}.")
        finally:
            self.close()

    def _run_stage(self, task_manager, stage: StageConfig) -> None:
        """运行单个阶段，直到本阶段采样完成。"""
        self._current_task_manager = task_manager
        LOG.info(f"SerialStageTrainRole enter stage: stage_name={stage.stage_name}")
        apply_train_seed(
            stage.train_config,
            role_id=f"serial-stage-train:{stage.stage_name}",
            context=f"SerialStageTrainRole:{stage.stage_name}",
        )
        # Learner 首次构造前先让无头环境看到任务级 metadata；后续阶段由
        # apply_stage_to_resources 对同一环境实例继续做增量覆盖。
        self.env.apply_env_metadata(stage.train_config.env_metadata)
        self._ensure_learners_built()
        # 阶段切换：env.stage_name、回放池、Learner 超参，并重挂 TM 回调与 learn_batches_update。
        self._apply_stage(stage, task_manager)
        self._declare_stage_artifact_producers()
        # 按当前阶段 TM 构造采样/学习 worker（仅 task_manager / train_config 随阶段变化）。
        sample_worker, learn_worker = self._build_stage_workers(task_manager, stage)

        if not getattr(self, "_simulation_middleware_logged", False):
            self._simulation_middleware_logged = log_simulation_middleware(self.env, LOG)

        while True:
            if task_manager.is_task_finished():
                break
            try:
                # 开始一局采样
                episode_transitions = sample_worker.rollout_one_episode()
                if episode_transitions is None:
                    break
                # 学习一个批次
                learn_results = learn_worker.rollout_after_episode_updated()
                # 筛选本批次实际更新的 Learner
                updated_learners = updated_learners_from_result(self.learners, learn_results)
                if self.model_manager is not None:
                    # 有ModelManager的情况
                    # Learner 更新后的模型提交到 ModelManager
                    publish_learners_to_model_manager(
                        learners=updated_learners,
                        model_manager=self.model_manager,
                        task_id=task_manager.task_id,
                        metrics=latest_metrics_by_learner(learn_results),
                        is_learning_batch=True,
                        reason="online_train",
                        learn_info=learn_worker.last_learn_info,
                        train_config=stage.train_config,
                        task_manager=task_manager,
                    )
                    # Agent 拉取当前 champion 策略网络
                    for learner in updated_learners:
                        champion = self.model_manager.get_champion(learner.name)
                        if champion is None:
                            LOG.warning(
                                f"ModelManager[{learner.name}] has no champion, skip agent sync."
                            )
                            continue
                        sync_agent_from_model_manager(
                            self.agents_by_name[learner.name],
                            self.model_manager,
                            champion.version,
                        )
                else:
                    # Agent 直接同步 Learner 策略网络
                    for learner in updated_learners:
                        self.agents_by_name[learner.name].reset_net(learner.agent_net)
                # 打印本局结果
                self._log_episode_result(stage, episode_transitions)
            except SampleTaskReturned as exc:
                LOG.warning(f"Sample task returned, skip this learning cycle: {exc}")
                continue

        # producer 先完成样本/回放池落盘，再进入 TaskManager 统一结束点。
        if self.monte_carlo_method is not None:
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
        complete_role_gifs(
            self.artifacts_manager, task_manager.task_id, task_manager.stage_name, self.role_id,
        )
        task_manager.notify_stage_complete()
        self._clone_learner_snapshots(stage)
        LOG.info(
            f"SerialStageTrainRole finished stage: stage_name={stage.stage_name}, "
            f"generation={task_manager.generation}, seed={task_manager.run_seed}"
        )

    def _ensure_learners_built(self) -> None:
        """首阶段进入且种子已写入 TrainConfig 后，再初始化 Learner 网络。"""
        if self.learners:
            return
        union_learners_config = self.stage_task.all_learner_configs
        self.learners = self._build_learners(union_learners_config)
        self.learners_by_name = self._index_learners(self.learners)
        for learner in self.learners:
            bind_artifact_producer_callers(
                learner, f"learner:{learner.name}",
                self._register_artifact, self._put_artifact, self._complete_artifact_for,
            )
        if self.model_manager is None:
            self._sync_agents_from_learners()
        else:
            task_manager = self._current_task_manager
            assert task_manager is not None
            publish_learners_to_model_manager(
                learners=self.learners,
                model_manager=self.model_manager,
                task_id=task_manager.task_id,
                metrics={},
                reason="init",
                promote=True,
                train_config=task_manager.task_config,
                task_manager=task_manager,
            )
            sync_agents_from_model_manager(
                self.policy_agents,
                self.model_manager,
                learner_names=self.learners_by_name,
            )

    def _apply_stage(self, stage: StageConfig, task_manager) -> None:
        """阶段切换动作：env.stage_name、按需清空回放池、重置在场 Learner 超参、重连 TM 回调。"""
        mc_config = stage.train_config.monte_carlo_method_config
        if stage.train_config.sample_type == SampleType.OFFLINE.value and mc_config is None:
            raise ValueError(f"offline stage {stage.stage_name!r} requires monte_carlo_method_config")
        if self.monte_carlo_method is not None:
            self.monte_carlo_method.close()
            self.monte_carlo_method = None
        if mc_config is not None and (stage.train_config.record_sample or stage.train_config.sample_type == SampleType.OFFLINE.value):
            mc_config = copy.deepcopy(mc_config)
            mc_config.write_enabled = bool(
                stage.train_config.record_sample
                and stage.train_config.sample_type == SampleType.ONLINE.value
            )
            self.monte_carlo_method = MonteCarloMethod(task_id=task_manager.task_id, config=mc_config)
            bind_artifact_producer_callers(
                self.monte_carlo_method, "monte_carlo",
                self._register_artifact, self._put_artifact, self._complete_artifact_for,
            )
            if stage.train_config.sample_type == SampleType.OFFLINE.value:
                valid_count = self.monte_carlo_method.prepare_offline_index(
                    stage_name=stage.stage_name,
                    expected_agents=[str(agent) for agent in self.env.possible_agents],
                )
                task_manager.prepare_offline_sample_plan(valid_count)
            elif stage.train_config.record_sample:
                self.env.record_caller = self.monte_carlo_method.receiver
        else:
            self.env.record_caller = None
        if load_skip_stage_replay_if_needed(
                self.stage_task.stage_task_config,
                self.replay_buffer,
                already_loaded=self._skip_stage_replay_loaded,
        ):
            self._skip_stage_replay_loaded = True
        # 写入 env.stage_name、按阶段配置清空 ReplayBuffer、同步在场 Learner 超参。
        apply_stage_to_resources(
            stage=stage,
            env=self.env,
            learners_by_name=self.learners_by_name,
            replay_buffer=self.replay_buffer,
        )
        # 回放池攒够 batch 后回调「当前阶段」TM，由该 TM 决定是否释放学习令牌。
        self.replay_buffer.learn_batches_update_caller = task_manager.learn_batches_update
        # 当前阶段 TM 代次切换（AutoML restart 等）时清空回放池、恢复快照并应用新超参。
        task_manager.add_restart_callback(self._on_task_restart)
        task_manager.register_learner_save_caller(
            lambda prefix: self._learner_save_caller(prefix, stage, task_manager),
        )

    def _declare_stage_artifact_producers(self) -> None:
        """当前阶段 TM 已就绪：expect 本阶段产物模块并立刻 register 意图。"""

        def current_task_id() -> str:
            if self._current_task_manager is not None:
                return self._current_task_manager.task_id
            return str(task_id)

        def current_stage_name() -> str:
            if self._current_task_manager is not None:
                return self._current_task_manager.stage_name
            return "train"

        declare_stage_artifact_producer(
            self.replay_buffer, "replay_buffer",
            self.artifacts_manager, current_task_id, current_stage_name,
        )
        if self.model_manager is not None:
            declare_stage_artifact_producer(
                self.model_manager, "model_manager",
                self.artifacts_manager, current_task_id, current_stage_name,
            )
        for learner in self.learners:
            declare_stage_artifact_producer(
                learner, f"learner:{learner.name}",
                self.artifacts_manager, current_task_id, current_stage_name,
            )
        if self.monte_carlo_method is not None:
            declare_stage_artifact_producer(
                self.monte_carlo_method, "monte_carlo",
                self.artifacts_manager, current_task_id, current_stage_name,
            )
        declare_role_log_producer(
            self.artifacts_manager, current_task_id, current_stage_name, self.role_id,
        )
        declare_role_gif_producer(
            self.artifacts_manager, current_task_id, current_stage_name, self.role_id,
        )

    def _on_task_restart(self, generation: int, advice=None) -> None:
        """TaskManager 重启当前阶段训练时，按代次恢复Learner快照并同步执行Agent。"""
        self.replay_buffer.clear(generation=generation)
        env_apply_restart_metadata(self.env, self._current_task_manager)
        for learner in self.learners:
            learner.restore_models_snapshot_for_generation(generation)
        learners_apply_restart_advice(self.learners_by_name, advice)
        apply_replay_capacity_advice(self.replay_buffer, advice)
        if self.model_manager is None:
            self._sync_agents_from_learners()
        else:
            task_manager = self._current_task_manager
            assert task_manager is not None
            publish_learners_to_model_manager(
                learners=self.learners,
                model_manager=self.model_manager,
                task_id=task_manager.task_id,
                metrics={},
                reason=f"task_restart:generation_{generation}",
                promote=True,
                train_config=task_manager.task_config,
                task_manager=task_manager,
            )
            sync_agents_from_model_manager(
                self.policy_agents,
                self.model_manager,
                learner_names=self.learners_by_name,
            )
        LOG.warning(f"SerialStageTrainRole handled task restart: generation={generation}")

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

    def _build_stage_workers(self, task_manager, stage: StageConfig):
        """按当前阶段的 TrainTaskManager 构造采样/学习滚动流程；仅 task_manager 随阶段变化。"""
        # 当前阶段的Agent顺序规则来源，供 RolloutSampleWorker 消费。
        agent_compose = build_agent_compose(stage.train_config, self.policy_agents)
        # 采样：向当前阶段 TM 领取 episode 令牌，episode 写入共享 ReplayBuffer。
        sample_worker = RolloutSampleWorker(
            policy_agents=self.policy_agents,
            agent_compose=agent_compose,
            env=self.env,
            task_manager=task_manager,
            replay_buffer=self.replay_buffer,
            monte_carlo_method=self.monte_carlo_method,
        )
        # 局末 GIF 直送本进程产物台账；与分布式同一 put_bytes 采集路径。
        sample_worker.render_gif_put_caller = self._put_render_gif
        # 学习：从 ReplayBuffer 取 batch，向当前阶段 TM 领取/回报学习令牌。
        learn_worker = RolloutLearnWorker(
            learners=self.learners,
            train_config=stage.train_config,
            replay_buffer=self.replay_buffer,
            task_manager=task_manager,
        )
        return sample_worker, learn_worker

    def _put_render_gif(
            self,
            task_id_value: str,
            stage: str,
            path: str,
            metadata: dict | None = None,
    ) -> None:
        """局末 GIF 直送本进程产物台账。"""
        put_role_render_gif(self.artifacts_manager, task_id_value, stage, path, metadata)

    def _clone_learner_snapshots(self, stage: StageConfig) -> None:
        """阶段完成后刷新默认快照，后续任务重启可回到最近一个已完成阶段的模型状态。"""
        for learner in self.learners:
            learner.clone_models_snapshot()
        LOG.info(f"SerialStageTrainRole cloned learner snapshots after stage: stage_name={stage.stage_name}")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.stage_task.close()
        finally:
            try:
                self.replay_buffer.close()
            finally:
                try:
                    self.env.close()
                finally:
                    if self.sim_manager is not None:
                        self.sim_manager.close()

    @staticmethod
    def _index_agents(agents: list[Agent]) -> dict[AgentName, Agent]:
        agents_by_name = {agent.name: agent for agent in agents}
        assert len(agents_by_name) == len(agents), "Agent.name must be unique in SerialStageTrainRole."
        return agents_by_name

    @staticmethod
    def _index_learners(learners: list[Learner]) -> dict[AgentName, Learner]:
        learners_by_name = {learner.name: learner for learner in learners}
        assert len(learners_by_name) == len(learners), "Learner.name must be unique in SerialStageTrainRole."
        return learners_by_name

    def _build_learners(self, learners_config) -> list[Learner]:
        learners = []
        model_version = str(self._bootstrap_train_config.model_version or "").strip()
        model_load_dir = str(self._bootstrap_train_config.model_load_dir or "").strip()
        for learner_config in learners_config:
            model_file_path = None
            net_source = "blanco"
            if model_version:
                model_file_path = resolve_learner_archive_model_files(
                    model_load_dir=model_load_dir,
                    model_version=model_version,
                    learner_name=learner_config.name,
                    model_names=required_model_names(learner_config),
                )
                net_source = "file"
                LOG.info(
                    f"SerialStageTrainRole bootstrap learner from AutoML archive: "
                    f"learner={learner_config.name}, version={model_version}, "
                    f"files={model_file_path}"
                )
            learners.append(
                create_learner(
                    learner_config=learner_config,
                    agent=self.agents_by_name[learner_config.name],
                    env=self.env,
                    device=learner_config.device,
                    net_source=net_source,
                    model_file_path=model_file_path,
                )
            )
        return learners

    def _sync_agents_from_learners(self) -> None:
        for name, learner in self.learners_by_name.items():
            self.agents_by_name[name].reset_net(learner.agent_net)

    @staticmethod
    def _log_episode_result(stage: StageConfig, episode_transitions) -> None:
        rewards = {
            name: sum(transition.episode_reward() for transition in transitions)
            for name, transitions in episode_transitions.items()
        }
        LOG.info(f"SerialStageTrainRole stage={stage.stage_name} episode finished, rewards={rewards}")

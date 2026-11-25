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
from envs.base import BaseEnvironment
from config import AgentConfig, EnvConfig, LearnerConfig, ReplayBufferConfig, TrainConfig, SampleType
from scheduler.rollout_learn import RolloutLearnWorker
from scheduler.rollout_sample import RolloutSampleWorker
from sample.replay_buffer import ReplayBuffer
from sample.monte_carlo import MonteCarloMethod
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
from roles.common.simulation_logging import log_simulation_middleware
from roles.common.artifact_wiring import (
    resolve_artifacts_manager,
    bind_local_artifact_callers,
    attach_artifact_producer,
    complete_artifact_producers,
    complete_role_gifs,
    complete_role_logs,
    declare_role_gif_producer,
    declare_role_log_producer,
    put_role_render_gif,
)
from scheduler.artifact import ArtifactsManager
from scheduler.train_task import TrainTaskManager
from utils.module_utils import (
    align_agent_configs,
    build_agent_compose,
    build_env,
    build_policy_agents,
    prepare_env_agent_names,
    updated_learners_from_result,
)
from utils.random_seed import apply_train_seed

LOG = init_logger("serial_train")


class SerialTrainRole:
    """
    生产用单进程串行训练角色，按“采样一局、学习一个批次、同步策略”循环执行。

    Agent和Learner通过唯一AgentName直接配对；本进程内同步Learner.agent_net，
    不经过RPC或ModelManager。Simulation可以使用local直连，也可从本进程SimManager资源池申请。
    """

    @close_sim_manager_on_init_error
    def __init__(
            self,
            config: TrainConfig,
            env_config: EnvConfig,
            sim_config: Optional[SimConfig],
            replay_buffer_config: ReplayBufferConfig,
            step_agents_config: list[AgentConfig],
            learners_config: list[LearnerConfig],
            sim_manager_config: Optional[SimManagerConfig] = None,
            artifacts_manager: Optional[ArtifactsManager] = None,
    ):
        # ---- 入口约束：串行单进程直接同步策略，不通过 ModelManager 拉取版本 ----
        assert config.model_version is None, (
            "SerialTrainRole does not use ModelManager; TrainConfig.model_version must be None."
        )
        self._assert_supported(env_config, sim_config, sim_manager_config)
        # 运行元数据：产物命名时间戳、配置引用、关闭标志。
        self.timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        self.config = config
        self.role_id = "serial_train"
        self._closed = False
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
        # 与SerialStageTrainRole保持一致：配置了SimManagerConfig就由当前串行角色持有资源池。
        self.sim_manager = (
            create_sim_manager(config=sim_manager_config, sample_type=config.sample_type)
            if sim_manager_config is not None
            else None
        )
        env_config.metadata = BaseEnvironment._deep_merge_metadata(
            env_config.metadata or {}, config.env_metadata,
        )

        # ---- 执行侧 Agent：名称对齐后一次性构造，全训练周期复用 ----
        prepare_env_agent_names(env_config, step_agents_config)
        align_agent_configs(step_agents_config, learners_config)
        assert_state_shape_matches_env(env_config, step_agents_config, learners_config)
        self.policy_agents = build_policy_agents(step_agents_config)
        self.agents_by_name = self._index_agents(self.policy_agents)
        self._assert_agent_learner_pairs(learners_config)

        # 多 Agent 并行采样编排（parallel/serial 等），供 RolloutSampleWorker 消费。
        self.agent_compose = build_agent_compose(config, self.policy_agents)
        # 仿真环境：local直接使用sim_config；manager从本进程资源池申请完整连接参数。
        self.env = build_env(
            env_config=env_config,
            sim_config=sim_config,
            policy_agents=self.policy_agents,
            connect_sim=config.sample_type != "offline",
            sim_manager=self.sim_manager,
            role_id="serial-train",
        )

        # ---- 任务总线：先确定种子（含 AutoML opening trial），再构造 Learner ----
        self.auto_ml = create_auto_ml_if_enabled(
            config.auto_ml_config,
            learners_config,
        )
        self.task_manager = TrainTaskManager(
            config=config,
            policy_agent_configs=step_agents_config,
            learner_configs=learners_config,
            env_name=env_config.sense_name,
            auto_ml=self.auto_ml,
            artifacts_manager=self.artifacts_manager,
        )
        apply_train_seed(
            config,
            role_id="serial-train",
            context="SerialTrainRole",
        )
        self.learners = self._build_learners(learners_config)
        self.learners_by_name = self._index_learners(self.learners)

        inherit_replay_buffer_seed(replay_buffer_config, config)
        self.replay_buffer = ReplayBuffer(config=replay_buffer_config)
        apply_learner_capacities(self.replay_buffer, learners_config)
        # 回放池攒够 batch 后回调 TM，由 TM 决定是否向学习侧释放对应 episode 的学习令牌。
        self.replay_buffer.learn_batches_update_caller = self.task_manager.learn_batches_update
        if config.sample_type == SampleType.OFFLINE.value and config.monte_carlo_method_config is None:
            raise ValueError("offline training requires monte_carlo_method_config")
        self.monte_carlo_method = None
        if config.monte_carlo_method_config is not None and (
                config.record_sample or config.sample_type == SampleType.OFFLINE.value
        ):
            mc_config = copy.deepcopy(config.monte_carlo_method_config)
            mc_config.write_enabled = bool(
                config.record_sample and config.sample_type == SampleType.ONLINE.value
            )
            self.monte_carlo_method = MonteCarloMethod(task_id=self.task_manager.task_id, config=mc_config)
            if config.sample_type == SampleType.OFFLINE.value:
                count = self.monte_carlo_method.prepare_offline_index(
                    stage_name=self.task_manager.stage_name,
                    expected_agents=[str(agent) for agent in self.env.possible_agents],
                )
                self.task_manager.prepare_offline_sample_plan(count)
            elif config.record_sample:
                self.env.record_caller = self.monte_carlo_method.receiver
        # TM 代次切换（AutoML restart 等）时清空回放池、恢复 Learner 快照并应用新超参。
        self.task_manager.add_restart_callback(self._on_task_restart)
        self.task_manager.register_learner_save_caller(self._learner_save_caller)
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
        if self.monte_carlo_method is not None:
            attach_artifact_producer(
                self.monte_carlo_method, "monte_carlo", register, put, complete_for,
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

        # ---- 滚动流程：采样一局 → 学习一批，均挂接同一 TM / ReplayBuffer ----
        self.rollout_sample_worker = RolloutSampleWorker(
            policy_agents=self.policy_agents,
            agent_compose=self.agent_compose,
            env=self.env,
            task_manager=self.task_manager,
            replay_buffer=self.replay_buffer,
            monte_carlo_method=self.monte_carlo_method,
        )
        # 局末 GIF 直送本进程产物台账；与分布式同一 put_bytes 采集路径。
        self.rollout_sample_worker.render_gif_put_caller = self._put_render_gif
        self.rollout_learn_worker = RolloutLearnWorker(
            learners=self.learners,
            train_config=config,
            replay_buffer=self.replay_buffer,
            task_manager=self.task_manager,
        ) if self.learners else None

        # 首局采样前：把 Learner 初始策略同步到执行 Agent，保证 rollout 有可用网络。
        self._sync_agents_from_learners()
        LOG.info("SerialTrainRole initialized modules:")
        LOG.info("  TrainTaskManager")
        LOG.info(f"  ReplayBuffer(agent_capacities={self.replay_buffer.config.agent_capacities})")
        LOG.info(f"  RolloutSampleWorker(agents={list(self.agents_by_name)})")
        if self.rollout_learn_worker is not None:
            LOG.info(f"  RolloutLearnWorker(learners={list(self.learners_by_name)})")
        LOG.info(f"  Environment({type(self.env).__name__})")
        if self.sim_manager is not None:
            LOG.info(f"  SimManager(file={self.sim_manager.config.config_file if self.sim_manager is not None else 'disabled'})")

    @staticmethod
    def _assert_supported(
            env_config: EnvConfig,
            sim_config: Optional[SimConfig],
            sim_manager_config: Optional[SimManagerConfig] = None,
    ) -> None:
        """校验SerialTrainRole的仿真配置组合，规则与SerialStageTrainRole保持一致。"""
        if env_config.use_sim:
            assert sim_config is not None, "SerialTrainRole requires sim_config when EnvConfig.use_sim=True."
            assert sim_config.config_source in {"local", "manager"}, (
                "SerialTrainRole SimConfig.config_source must be 'local' or 'manager'."
            )
            if sim_config.config_source == "manager":
                assert sim_manager_config is not None and sim_manager_config.use_sim, (
                    "SerialTrainRole with SimConfig.config_source='manager' requires an enabled SimManagerConfig."
                )

    def run(self) -> None:
        """串行执行采样和学习，任何时刻都只有一侧访问策略网络。"""
        declare_role_log_producer(
            self.artifacts_manager,
            lambda: self.task_manager.task_id,
            lambda: self.task_manager.stage_name,
            self.role_id,
        )
        declare_role_gif_producer(
            self.artifacts_manager,
            lambda: self.task_manager.task_id,
            lambda: self.task_manager.stage_name,
            self.role_id,
        )
        log_simulation_middleware(self.env, LOG)
        try:
            while True:
                if self.task_manager.is_sample_completed():
                    break
                try:
                    # 采样
                    episode_transitions = self.rollout_sample_worker.rollout_one_episode()
                    if episode_transitions is None:
                        break
                    # 学习一个批次
                    learn_results = (
                        self.rollout_learn_worker.rollout_after_episode_updated()
                        if self.rollout_learn_worker is not None else []
                    )
                    # 筛选本批次实际更新的 Learner
                    updated_learners = updated_learners_from_result(self.learners, learn_results)
                    # Agent 直接同步 Learner 策略网络
                    for learner in updated_learners:
                        self.agents_by_name[learner.name].reset_net(learner.agent_net)
                    # 打印本局结果
                    self._log_episode_result(episode_transitions)
                except SampleTaskReturned as exc:
                    LOG.warning(f"Sample task returned, skip this learning cycle: {exc}")
                    continue
        finally:
            self.close()
        complete_artifact_producers(self.replay_buffer, self.monte_carlo_method, *self.learners)
        complete_role_logs(
            self.artifacts_manager,
            self.task_manager.task_id,
            self.task_manager.stage_name,
            self.role_id,
        )
        complete_role_gifs(
            self.artifacts_manager,
            self.task_manager.task_id,
            self.task_manager.stage_name,
            self.role_id,
        )
        # 单阶段训练结束：保存 final 模型并通知 AutoML tell COMPLETE。
        self.task_manager.notify_stage_complete()
        LOG.info(
            f"SerialTrainRole finished: task_id={self.task_manager.task_id}, "
            f"stage_name={self.task_manager.stage_name}, "
            f"generation={self.task_manager.generation}, seed={self.task_manager.run_seed}"
        )

    def close(self) -> None:
        """关闭环境并停止可选的本进程SimManager后台刷新线程。"""
        if self._closed:
            return
        self._closed = True
        # 环境先关闭，使Simulation在SimManager停止前释放占用的sim_id。
        try:
            self.env.close()
        finally:
            if self.monte_carlo_method is not None:
                self.monte_carlo_method.close()
            if self.sim_manager is not None:
                self.sim_manager.close()

    @staticmethod
    def _index_agents(agents: list[Agent]) -> dict[AgentName, Agent]:
        agents_by_name = {agent.name: agent for agent in agents}
        assert len(agents_by_name) == len(agents), "Agent.name must be unique in SerialTrainRole."
        return agents_by_name

    @staticmethod
    def _index_learners(learners: list[Learner]) -> dict[AgentName, Learner]:
        learners_by_name = {learner.name: learner for learner in learners}
        assert len(learners_by_name) == len(learners), "Learner.name must be unique in SerialTrainRole."
        return learners_by_name

    def _assert_agent_learner_pairs(self, learners_config: list[LearnerConfig]) -> None:
        learner_names = [learner_config.name for learner_config in learners_config]
        assert len(set(learner_names)) == len(learner_names), "LearnerConfig.name 必须唯一。"
        learner_name_set = set(learner_names)
        network_agent_names = {
            name for name, agent in self.agents_by_name.items()
            if getattr(agent.config.algo_config, "use_net", True)
        }
        assert learner_name_set <= network_agent_names, (
            "SerialTrainRole 的 Learner 必须对应同名网络型 Agent；"
            f"网络型Agent={sorted(network_agent_names)}，Learner={sorted(learner_name_set)}"
        )
        missing_model_agents = [
            name for name in sorted(network_agent_names - learner_name_set)
            if not str(self.agents_by_name[name].config.model_file_path or "").strip()
        ]
        assert not missing_model_agents, (
            "SerialTrainRole 中未配置 Learner 的网络型 Agent 必须提供 model_file_path；"
            f"缺少权重的Agent={missing_model_agents}"
        )

    def _build_learners(self, learners_config: list[LearnerConfig]) -> list[Learner]:
        learners = []
        for learner_config in learners_config:
            learners.append(
                create_learner(
                    learner_config=learner_config,
                    agent=self.agents_by_name[learner_config.name],
                    env=self.env,
                    device=learner_config.device,
                    net_source="blanco",
                )
            )
        return learners

    def _sync_agents_from_learners(self) -> None:
        """按AgentName把全部Learner策略网络复制到对应执行Agent。"""
        for name, learner in self.learners_by_name.items():
            self.agents_by_name[name].reset_net(learner.agent_net)

    def _on_task_restart(self, generation: int, advice=None) -> None:
        """TaskManager 重启训练时，按代次恢复Learner快照并同步执行Agent。"""
        self.replay_buffer.clear(generation=generation)
        env_apply_restart_metadata(self.env, self.task_manager)
        for learner in self.learners:
            learner.restore_models_snapshot_for_generation(generation)
        learners_apply_restart_advice(self.learners_by_name, advice)
        apply_replay_capacity_advice(self.replay_buffer, advice)
        self._sync_agents_from_learners()
        LOG.warning(f"SerialTrainRole handled task restart: generation={generation}")

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
                learner.save_models(self.config.model_save_dir, filenames)
            )
        return paths

    def _put_render_gif(
            self,
            task_id_value: str,
            stage: str,
            path: str,
            metadata: dict | None = None,
    ) -> None:
        """局末 GIF 直送本进程产物台账。"""
        put_role_render_gif(self.artifacts_manager, task_id_value, stage, path, metadata)

    @staticmethod
    def _log_episode_result(episode_transitions) -> None:
        rewards = {
            name: sum(transition.episode_reward() for transition in transitions)
            for name, transitions in episode_transitions.items()
        }
        LOG.info(f"SerialTrainRole episode finished, rewards={rewards}")

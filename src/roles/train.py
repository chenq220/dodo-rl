from __future__ import annotations

from common.simulation import close_sim_manager_on_init_error, create_sim_manager

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.sim_manager import SimManager
    from dodo_sim.config import SimConfig, SimManagerConfig

import copy
from typing import Optional, TYPE_CHECKING
from scheduler.rollout_sample import RolloutSampleWorker
from scheduler.rollout_learn import RolloutLearnWorker
from scheduler.train_task import TrainTaskManager
from scheduler.sample_learn_task import SampleLearnManager
from common.creator import create_agent, create_env, create_learner
from common.exception import SampleTaskReturned
from algos.agents import resolve_agent_groups
from roles.common.config_alignment import (
    align_agent_config_with_learner_config,
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
)
from sample.replay_buffer import ReplayBuffer
from sample.monte_carlo import MonteCarloMethod
from scheduler.artifact import ArtifactsManager
from scheduler.model import AgentCompose
from models import ModelManager
from utils.checkpoint import extract_state_dict
from utils.model_weights import snapshot_model_weights
from config import (
    EnvConfig,
    LearnerConfig,
    MonteCarloMethodConfig,
    ReplayBufferConfig,
    SampleLearningTasksConfig,
    TrainConfig,
    AgentConfig,
    AgentType,
    ModelManagerConfig,
)
from common.logger import init_logger
from envs.base import BaseEnvironment
from utils.random_seed import apply_train_seed

if TYPE_CHECKING:
    from sample.sample_transfer import EpisodeTransition
    from common.types import AgentName

LOG = init_logger('train')


class TrainRole(object):
    """
    训练角色入口，负责把配置、Agent、Environment、Learner、ReplayBuffer 和任务调度器组装成完整训练流程。

    当前实现是单进程直连版本，但内部调度接口按分布式语义设计：采样进度、学习令牌、
    样本学习任务分别通过 TrainTaskManager 和 SampleLearnManager 完成，后续接 RPC 时应优先保持这些接口语义稳定。
    """

    @close_sim_manager_on_init_error
    def __init__(self, config: TrainConfig,
                 env_config: EnvConfig,
                 sim_config: SimConfig,
                 replay_buffer_config: ReplayBufferConfig,
                 step_agents_config: list[AgentConfig],
                 learners_config: list[LearnerConfig],
                 model_manager_config: ModelManagerConfig,
                 sim_manager_config: SimManagerConfig,
                 episode_agent_config: Optional[AgentConfig] = None,
                 sample_learning_config: Optional[SampleLearningTasksConfig] = None,
                 monte_carlo_method_config: Optional[MonteCarloMethodConfig] = None,
                 artifacts_manager: Optional[ArtifactsManager] = None):

        self.config = config
        self.sample_learning_config = sample_learning_config or SampleLearningTasksConfig()
        self.learners_config = learners_config
        self.step_agents_config = step_agents_config
        self.episode_agent_config = episode_agent_config
        self.agent_compose_type = config.agent_compose_type
        self._closed = False
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
        apply_train_seed(
            config,
            role_id="serial-train",
            context="TrainRole",
        )
        env_config.metadata = BaseEnvironment._deep_merge_metadata(
            env_config.metadata or {}, config.env_metadata,
        )

        # ModelManager 第一版只作为本地模型中转站。
        # model_name 使用 AgentName：AgentConfig.name、LearnerConfig.name 与模型池逻辑名保持一致。
        self.model_manager = ModelManager(config=model_manager_config)
        # SimManager 第一版在单进程训练中作为本地仿真资源管理器；是否消费由Simulation按SimConfig判断。
        self.sim_manager = create_sim_manager(config=sim_manager_config, sample_type=config.sample_type)

        # 初始化agent名称清单
        self._prepare_env_agent_names(env_config)
        # Learner 算法决定 Agent 输出语义和网络宽度，必须在创建 Agent 网络前完成对齐。
        self._align_agent_configs_with_learners()
        agent_configs = list(self.step_agents_config)
        if self.episode_agent_config is not None:
            agent_configs.append(self.episode_agent_config)
        assert_state_shape_matches_env(env_config, agent_configs, self.learners_config)

        # 初始化环境
        self.env = create_env(
            scene_name=env_config.sense_name,
            config=env_config,
            sim_config=sim_config,
            connect_sim=config.sample_type != "offline",
            sim_manager=self.sim_manager,
            role_id="serial-train",
        )
        # 初始化算法侧 Agent 实例（与环境实体名 env.agents 区分）
        self.agent_compose, self.policy_agents, self.policy_agent_names = self.build_agents()
        # 将 obs/action 维度注入给环境
        self.env.obs_dim, self.env.action_dim = self._inject_dimensional_info(self.policy_agents)
        # 初始化learner。 TIPS:learner 是 policy_agents 的子集
        self.learners, self.learner_names = self.build_learners()

        # 初始化训练管理
        self.task_manager = TrainTaskManager(
            config=config,
            policy_agent_configs=self._policy_agent_configs(),
            learner_configs=self.learners_config,
            artifacts_manager=self.artifacts_manager,
        )
        # 首次把Learner网络登记到模型池，采样Agent也从模型池拉取策略网络。
        self._publish_learners_to_model_manager(
            learners=self.learners,
            metrics={},
            promote=True,
            reason="init",
        )
        self._sync_agents_from_model_manager()
        # episode 级经验回放池，保存 RolloutSampleWorker 产出的 EpisodeTransition。
        inherit_replay_buffer_seed(replay_buffer_config, config)
        self.replay_buffer = ReplayBuffer(config=replay_buffer_config)
        apply_learner_capacities(self.replay_buffer, self.learners_config)
        # 回放池满足 batch 条件后通过该回调通知 TrainTaskManager 增加学习次数。
        self.replay_buffer.learn_batches_update_caller = self.task_manager.learn_batches_update
        self.task_manager.add_restart_callback(self._on_task_restart)

        # 样本学习调度器承载主训练学习结果收集。
        self.sample_learn_manager = SampleLearnManager(
            config=self.sample_learning_config,
            task_id=self.task_manager.task_id,
        )

        # 采样执行器，负责一局马尔科夫交互并将 transition 写入 ReplayBuffer。
        self.rollout_sample_worker = RolloutSampleWorker(
            policy_agents=self.policy_agents,
            agent_compose=self.agent_compose,
            env=self.env,
            task_manager=self.task_manager,
            replay_buffer=self.replay_buffer
        )
        # 学习执行器，负责从 ReplayBuffer 采样并驱动 Learner 更新。
        self.learn_rollout_worker = RolloutLearnWorker(
            learners=self.learners,
            train_config=self.config,
            replay_buffer=self.replay_buffer,
            task_manager=self.task_manager,
            sample_learn_manager=self.sample_learn_manager,
        )
        # 录制样本
        monte_carlo_method = None
        if self.config.record_sample:
            assert monte_carlo_method_config is not None
            # 可选蒙特卡洛方法，用于把在线交互数据落盘为离线样本。
            monte_carlo_method = MonteCarloMethod(
                task_id=self.task_manager.task_id,
                config=monte_carlo_method_config,
            )
            self.env.record_caller = monte_carlo_method.receiver
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
        if monte_carlo_method is not None:
            attach_artifact_producer(
                monte_carlo_method, "monte_carlo", register, put, complete_for,
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

        # 当前训练局序号，仅用于本地日志展示；真实任务进度以 TrainTaskManager 为准。
        self.train_episode = 1
        LOG.info("TrainRole initialized modules.")

    def _prepare_env_agent_names(self, env_config: EnvConfig) -> None:
        """ 如果环境未指定agent名称清单则将agents下面的名称注入；分组配置按组展开。 """
        if not env_config.agent_names:
            agent_names = []
            for agent_config in self.step_agents_config:
                agent_names += [
                    agent_id
                    for group in resolve_agent_groups(agent_config)
                    for agent_id in group
                ]
            if self.episode_agent_config is not None:
                agent_names += self.episode_agent_config.agent_names
            env_config.agent_names = agent_names

    def build_agents(self) -> tuple[AgentCompose, list[AgentType], list[AgentName]]:
        """按入口编排配置构造 AgentCompose 与扁平 policy_agents 列表。"""
        step_agents = self._build_agents_from_config(self.step_agents_config)
        assert step_agents, "step_agents_config must not be empty."
        episode_agent_config_list = (
            [self.episode_agent_config] if self.episode_agent_config is not None else []
        )
        episode_agents = self._build_agents_from_config(episode_agent_config_list)
        agent_compose = self._build_agent_compose(step_agents=step_agents, episode_agents=episode_agents)
        policy_agents = step_agents + episode_agents
        policy_agents_name = list(map(lambda agent: agent.name, policy_agents))
        return agent_compose, policy_agents, policy_agents_name

    @staticmethod
    def _inject_dimensional_info(policy_agents):
        """ 将维度信息注入给环境 """
        obs_dim = {}
        action_dim = {}
        for agent in policy_agents:
            obs_dim.update(dict(zip(agent.agent_list, [agent.obs_dim] * len(agent.agent_list))))
            action_dim.update(dict(zip(agent.agent_list, [agent.action_dim] * len(agent.agent_list))))
        return obs_dim, action_dim

    @staticmethod
    def _build_agents_from_config(agent_config_list):
        """根据入口组织好的AgentConfig列表创建Agent实例。"""
        assert agent_config_list is not None, "agent config list must not be None."
        agents = []
        for agent_config in agent_config_list:
            model_file_path = str(agent_config.model_file_path or "").strip() or None
            agents.append(
                create_agent(
                    agent_type=agent_config.agent_type,
                    agent_config=agent_config,
                    agent_net_source="file" if model_file_path else "blanco",
                    model_file_path=model_file_path,
                    device=agent_config.device,
                    agent_list=[
                        agent_id
                        for group in resolve_agent_groups(agent_config)
                        for agent_id in group
                    ],
                )
            )
        return agents

    def _build_agent_compose(self, step_agents: list, episode_agents: list) -> AgentCompose:
        """把入口的编排配置转换成RolloutSampleWorker使用的AgentCompose。"""
        compose_type = self.agent_compose_type
        assert compose_type is not None, "agent_compose_type must be defined."
        assert len(episode_agents) <= 1, (
            "AgentCompose currently supports at most one episode_agent; "
            "multi-layer episode decision needs a dedicated compose model."
        )
        episode_agent = episode_agents[0] if episode_agents else None
        return AgentCompose(
            compose_type=compose_type,
            episode_agent=episode_agent,
            step_agent=step_agents,
        )

    def _align_agent_configs_with_learners(self) -> None:
        """按AgentConfig.name和LearnerConfig.name配对，把算法侧网络要求同步给AgentConfig。"""
        learner_config_map = {learner_config.name: learner_config for learner_config in self.learners_config}
        assert len(learner_config_map) == len(self.learners_config), "LearnerConfig.name must be unique."
        agent_config_list = self.step_agents_config.copy()
        if self.episode_agent_config is not None:
            agent_config_list.append(self.episode_agent_config)
        for agent_config in agent_config_list:
            learner_config = learner_config_map.get(agent_config.name)
            if learner_config is None:
                continue
            align_agent_config_with_learner_config(agent_config, learner_config)

    def build_learners(self) -> tuple[list, list]:
        """按LearnerConfig列表构造Learner；每个Learner按name筛选对应Agent。"""
        learners = []
        learners_names = []
        for learner_config in self.learners_config:
            agent = self._find_agent_for_learner(learner_config)
            learner_net_source, learner_model_file_path, learner_caller = self._resolve_learner_source(learner_config)
            learners.append(
                create_learner(
                    learner_config=learner_config,
                    agent=agent,
                    env=self.env,
                    device=learner_config.device,
                    net_source=learner_net_source,
                    model_file_path=learner_model_file_path,
                    caller=learner_caller,
                )
            )
            learners_names.append(learner_config.name)
        assert learners, "learners_config must not be empty."
        return learners, learners_names

    def _find_agent_for_learner(self, learner_config: LearnerConfig):
        """从扁平Agent列表中按LearnerConfig.name筛选Agent。"""
        matched_agents = [
            agent
            for agent in self.policy_agents
            if agent.config.name == learner_config.name
        ]
        assert len(matched_agents) == 1, (
            f"LearnerConfig.name must match exactly one AgentConfig.name, "
            f"learner={learner_config.name}, matched={len(matched_agents)}"
        )
        return matched_agents[0]

    def _resolve_learner_source(self, learner_config: LearnerConfig):
        """决定Learner初始权重来源；继续训练时统一从ModelManager按AgentName取权重。"""
        if self.config.model_version is not None:
            return (
                "caller",
                None,
                lambda agent_only=False, name=learner_config.name: self.model_manager.get_model(
                    model_name=name,
                    version=self.config.model_version,
                    output_type="state_dict",
                ),
            )
        return "blanco", None, None

    def run(self):
        """ 串行任务执行入口 """
        LOG.info("TrainRole started.")
        log_simulation_middleware(self.env, LOG)
        try:
            while True:
                if self.task_manager.is_sample_completed():
                    break
                try:
                    # 打印开始日志
                    self.ep_start_log()

                    # 开始一局采样
                    ep_trans = self.rollout_sample_worker.rollout_one_episode(evaluate=False, random_sample=False)
                    if ep_trans is None:
                        break

                    # 学习一个批次
                    learn_results = self.learn_rollout_worker.rollout_after_episode_updated()

                    # 筛选本批次实际更新的 Learner
                    updated_learners = self._updated_learners_from_result(learn_results)
                    # Learner 更新后的模型提交到 ModelManager
                    self._publish_learners_to_model_manager(
                        learners=updated_learners,
                        metrics=self._latest_metrics_by_learner(learn_results),
                        promote=False,
                        is_learning_batch=True,
                        reason="online_train",
                    )

                    # Agent 拉取当前 champion 策略网络
                    self._sync_agents_from_model_manager()

                    # 打印本局结果
                    self.ep_end_log(ep_trans, learn_results)
                    self.train_episode += 1
                except SampleTaskReturned as exc:
                    # Rollout 已归还采样令牌并等待仿真恢复，本轮没有有效样本，不能触发学习。
                    LOG.warning(f"Sample task returned to task manager, skip learning for this train loop: {exc}")
                    continue

            self.post_episode()
            LOG.info(f"TrainRole finished: process run completed, task_id={self.task_manager.task_id}.")
        finally:
            self.close()

    def close(self) -> None:
        """关闭训练角色持有的本地资源，释放仿真占用并停止SimManager后台刷新。"""
        if self._closed:
            return
        self._closed = True
        try:
            if hasattr(self, "env") and self.env is not None:
                self.env.close()
        finally:
            if hasattr(self, "sim_manager") and self.sim_manager is not None:
                self.sim_manager.close()

    def _sync_agents_from_learners(self) -> None:
        """兼容旧流程：把每个Learner训练后的策略网络同步回它绑定的Agent。"""
        for learner in self.learners:
            learner.agent.reset_net(learner.agent_net)

    def _on_task_restart(self, generation: int, advice=None) -> None:
        """TaskManager 重启训练时，按代次恢复快照并将恢复后的策略重新发布。"""
        self.replay_buffer.clear(generation=generation)
        self.env.apply_env_metadata(self.task_manager.task_config.env_metadata)
        for learner in self.learners:
            learner.restore_models_snapshot_for_generation(generation)
        apply_replay_capacity_advice(self.replay_buffer, advice)
        self._publish_learners_to_model_manager(
            learners=self.learners,
            metrics={},
            promote=True,
            reason=f"task_restart:generation_{generation}",
        )
        self._sync_agents_from_model_manager()
        LOG.warning(f"TrainRole handled task restart: generation={generation}")

    def _publish_learners_to_model_manager(
            self,
            learners: list,
            metrics: dict[AgentName, dict],
            promote: bool,
            reason: str,
            is_learning_batch: bool = False,
    ) -> None:
        """把Learner网络快照提交给ModelManager；model_name统一使用AgentName。"""
        for learner in learners:
            agent_name: AgentName = learner.name
            parent = self.model_manager.get_champion(agent_name)
            # 模型池只需要权重；避免复制训练后的优化器状态和计算图。
            model_snapshots = snapshot_model_weights(learner.models)
            train_steps = self.task_manager.learner_state.get(agent_name, {}).get("learning_completed", 0)
            learn_info = getattr(self.learn_rollout_worker, "last_learn_info", None) if is_learning_batch else None
            metadata = {"task_id": self.task_manager.task_id, "reason": reason,
                        "episode": getattr(self, "train_episode", 0)}
            if learn_info is not None:
                metadata.update(sample_episode_num=learn_info.episode_num, learn_id=learn_info.learn_id,
                                stage_name=learn_info.stage_name, generation=learn_info.generation)
            submitted = self.model_manager.submit_model(
                model_name=agent_name,
                algo_type=learner.algo_type,
                models=model_snapshots,
                parent_version=parent.version if parent is not None else self.config.model_version,
                learner_id=agent_name,
                train_steps=train_steps,
                metrics=metrics.get(agent_name, {}),
                metadata=metadata,
                promote=promote,
                is_learning_batch=is_learning_batch,
            )
            if learn_info is not None:
                LOG.info("Learner model published: learner=%s, model_version=%s, sample_episode_num=%s, "
                         "learn_id=%s, stage=%s, generation=%s", agent_name, submitted.version,
                         learn_info.episode_num, learn_info.learn_id, learn_info.stage_name, learn_info.generation)

    def _sync_agents_from_model_manager(self) -> None:
        """Agent只从ModelManager按AgentName拉取champion策略网络权重，不直接读取Learner内部网络。"""
        for learner in self.learners:
            agent_name: AgentName = learner.name
            model_weights = self.model_manager.get_model(
                model_name=agent_name,
                output_type="state_dict",
            )
            agent_checkpoint = model_weights.get("agent_net")
            assert agent_checkpoint is not None, f"ModelManager[{agent_name}] must provide agent_net checkpoint."
            learner.agent.agent_net.load_state_dict(extract_state_dict(agent_checkpoint), strict=False)
            learner.agent.agent_net.to(learner.agent.device)
            learner.agent.agent_net.eval()

    def _updated_learners_from_result(self, learn_results) -> list:
        """根据本轮学习返回结果筛选真正发生过参数更新的Learner。"""
        updated_names = set(self._latest_metrics_by_learner(learn_results))
        return [learner for learner in self.learners if learner.name in updated_names]

    @staticmethod
    def _latest_metrics_by_learner(learn_results) -> dict[AgentName, dict]:
        """把RolloutLearnWorker返回结果压缩成每个AgentName对应的最新一次学习指标。"""
        metrics: dict[AgentName, dict] = {}
        result_items = learn_results if isinstance(learn_results, list) else [learn_results]
        for item in result_items:
            if not isinstance(item, dict):
                continue
            for agent_name, learner_result in item.items():
                if isinstance(learner_result, list):
                    if learner_result:
                        metrics[agent_name] = learner_result[-1]
                elif isinstance(learner_result, dict):
                    metrics[agent_name] = learner_result
        return metrics

    def ep_start_log(self):
        """ 一局开始打印日志 """
        LOG.info(f'{"=" * 20} Train {self.train_episode} START! {"=" * 20}')

    def ep_end_log(self, ep_trans: EpisodeTransition | dict, learn_results):
        """ 一局结束打印日志；值支持统一列表容器（隐式单组为长度1列表） """
        if isinstance(ep_trans, dict):
            total_reward = sum(
                transition.episode_reward()
                for transitions in ep_trans.values()
                for transition in (transitions if isinstance(transitions, list) else [transitions])
            )
        else:
            total_reward = ep_trans.episode_reward()
        LOG.info(f'{"=" * 10} Train {self.train_episode} END! >>> REWARD={total_reward} <<< {"=" * 10}')

    def post_episode(self):
        """ 训练完成后处理，项目定制化接口 """
        pass

    def _policy_agent_configs(self) -> list[AgentConfig]:
        """返回所有策略Agent配置，供任务总线记录任务上下文。"""
        agent_configs = self.step_agents_config.copy()
        if self.episode_agent_config is not None:
            agent_configs.append(self.episode_agent_config)
        return agent_configs

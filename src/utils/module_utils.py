from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimConfig


import copy
from typing import Any, Dict, List, Optional

from algos.agents import Agent, resolve_agent_groups
from algos.learners import Learner
from common.creator import create_agent, create_env, create_learner
from common.exception import AgentGroupCompatibilityException
from roles.common.archive_model_source import (
    required_model_names,
    resolve_learner_archive_model_files,
)
from common.logger import init_logger
from common.types import AgentName
from config import AgentConfig, AlgorithmType, EnvConfig, LearnerConfig, TrainConfig
from models.model import ModelInfo
from roles.common.config_alignment import align_agent_config_with_learner_config
from scheduler.model import AgentCompose, LearnInfo
from rpc.interfaces import ModelManagerPort, SimManagerPort
from utils.checkpoint import extract_state_dict
from utils.model_weights import snapshot_model_weights

LOG = init_logger("module_utils")


def prepare_env_agent_names(env_config: EnvConfig, step_agents_config: list[AgentConfig]) -> None:
    """把入口AgentConfig中的平台清单注入Environment配置；分组配置按组展开扁平清单。"""
    if env_config.agent_names:
        return
    agent_names = []
    for agent_config in step_agents_config:
        for group in resolve_agent_groups(agent_config):
            agent_names += list(group)
    env_config.agent_names = agent_names


def align_agent_configs(step_agents_config: list[AgentConfig], learners_config: list[LearnerConfig]) -> None:
    """按AgentConfig.name和LearnerConfig.name对齐Agent输出语义等算法字段。"""
    learner_config_map = {learner_config.name: learner_config for learner_config in learners_config}
    assert len(learner_config_map) == len(learners_config), "LearnerConfig.name must be unique."
    for agent_config in step_agents_config:
        learner_config = learner_config_map.get(agent_config.name)
        if learner_config is not None:
            if agent_config.agent_groups and not agent_config.fixed_mode:
                if learner_config.algo_type != AlgorithmType.QMIX.value:
                    raise AgentGroupCompatibilityException(
                        agent_name=agent_config.name,
                        detail=(
                            f"可训练显式分组第一版仅支持 qmix Learner，"
                            f"given algo_type={learner_config.algo_type}。"
                        ),
                    )
            align_agent_config_with_learner_config(agent_config, learner_config)


def build_policy_agents(step_agents_config: list[AgentConfig]) -> list[Agent]:
    """构造采样/学习侧共用的Agent实例；固定策略可直接从AgentConfig加载权重。"""
    agents = []
    for agent_config in step_agents_config:
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
    assert agents, "step_agents_config must not be empty."
    return agents


def build_agent_compose(train_config: TrainConfig, policy_agents: list[Agent]) -> AgentCompose:
    """构造当前第一版分布式训练使用的AgentCompose。"""
    return AgentCompose(
        compose_type=train_config.agent_compose_type,
        episode_agent=None,
        step_agent=policy_agents,
    )


def inject_env_dims(env, policy_agents: list[Agent]) -> None:
    """把Agent观测空间和动作空间维度注入环境。"""
    obs_dim = {}
    action_dim = {}
    for agent in policy_agents:
        obs_dim.update(dict(zip(agent.agent_list, [agent.obs_dim] * len(agent.agent_list))))
        action_dim.update(dict(zip(agent.agent_list, [agent.action_dim] * len(agent.agent_list))))
    env.obs_dim = obs_dim
    env.action_dim = action_dim
    env.action_space_models = {
        agent_id: agent.action_space for agent in policy_agents
        for agent_id in agent.agent_list if getattr(agent, "action_space", None) is not None
    }


def build_env(
        env_config: EnvConfig,
        sim_config: Optional[SimConfig],
        policy_agents: list[Agent],
        connect_sim: bool,
        sim_manager: SimManagerPort = None,
        role_id: str = None,
):
    """构造环境并完成维度注入。"""
    env = create_env(
        scene_name=env_config.sense_name,
        config=env_config,
        sim_config=sim_config,
        connect_sim=connect_sim,
        sim_manager=sim_manager,
        role_id=role_id,
    )
    inject_env_dims(env, policy_agents)
    return env


def find_agent_for_learner(learner_config: LearnerConfig, policy_agents: list[Agent]) -> Agent:
    """按LearnerConfig.name找到一一对应的Agent实例。"""
    matched_agents = [agent for agent in policy_agents if agent.config.name == learner_config.name]
    assert len(matched_agents) == 1, (
        f"LearnerConfig.name must match exactly one AgentConfig.name, "
        f"learner={learner_config.name}, matched={len(matched_agents)}"
    )
    return matched_agents[0]


def build_learners(
        train_config: TrainConfig,
        env,
        policy_agents: list[Agent],
        learners_config: list[LearnerConfig],
        model_manager: ModelManagerPort,
) -> list[Learner]:
    """构造学习进程里的Learner；初始权重从ModelManager拉取，拉不到时使用空网络。"""
    learners = []
    for learner_config in learners_config:
        agent = find_agent_for_learner(learner_config, policy_agents)
        model_file_path = None
        model_version = str(train_config.model_version or "").strip()
        model_load_dir = str(train_config.model_load_dir or "").strip()
        if model_version and model_load_dir:
            # A fresh LearnControl process has an empty in-memory ModelManager.
            # Bootstrap continuation models directly from the configured archive;
            # newly learned snapshots will still be published to ModelManager.
            model_file_path = resolve_learner_archive_model_files(
                model_load_dir=model_load_dir,
                model_version=model_version,
                learner_name=learner_config.name,
                model_names=required_model_names(learner_config),
            )
            net_source, caller = "file", None
            LOG.info(
                "Bootstrap learner from model archive: learner=%s, version=%s, files=%s",
                learner_config.name, model_version, model_file_path,
            )
        else:
            net_source, caller = resolve_learner_source(train_config, learner_config, model_manager)
        learners.append(
            create_learner(
                learner_config=learner_config,
                agent=agent,
                env=env,
                device=learner_config.device,
                net_source=net_source,
                model_file_path=model_file_path,
                caller=caller,
            )
        )
    assert learners, "learners_config must not be empty."
    return learners


def resolve_learner_source(
        train_config: TrainConfig,
        learner_config: LearnerConfig,
        model_manager: ModelManagerPort,
):
    """决定Learner初始权重来源。"""
    if train_config.model_version is None:
        return "blanco", None
    return (
        "caller",
        lambda agent_only=False, name=learner_config.name: model_manager.get_model(
            model_name=name,
            version=train_config.model_version,
            output_type="state_dict",
        ),
    )


def publish_learners_to_model_manager(
        learners: list[Learner],
        model_manager: ModelManagerPort,
        task_id: str,
        metrics: Optional[dict] = None,
        promote: Optional[bool] = None,
        is_learning_batch: bool = False,
        reason: str = "distributed_train",
        train_config: Optional[TrainConfig] = None,
        task_manager: Optional[Any] = None,
        replay_buffer: Optional[Any] = None,
        learn_info: Optional[LearnInfo] = None,
) -> list[AgentName]:
    """把 Learner 当前网络快照提交到模型池；返回本次提交的 learner 名称列表。

    is_learning_batch=True 时由 ModelManager 以 Agent 为粒度累计批次并决定是否自动晋升。
    """
    metrics = metrics or {}
    if promote is None:
        promote = False
    target = None
    if replay_buffer is not None:
        target = task_manager.model_publication_target(is_learning_batch)
        if target is not None:
            versions, requests = target
            parent_versions = task_manager.model_publication_target(False)[0] if is_learning_batch else {}
            publishing = {name for name, version in versions.items() if version != parent_versions.get(name)}
            if not publishing <= {learner.name for learner in learners}:
                raise RuntimeError(f"cannot publish versions: learner update missing for {publishing}")
            overlap_agents = task_manager.overlapping_sample_learners
            replay_buffer.set_sample_version(versions, requests, reset_episode_keys=not is_learning_batch,
                                             overlap_agents=overlap_agents)
        elif not is_learning_batch:
            replay_buffer.set_sample_version(None, {})
        elif task_manager.versioned_sampling_enabled:
            # The learning result belongs to a generation already replaced by a restart.
            return []

    submitted_names: list[AgentName] = []
    for learner in learners:
        versioned_learner = target is not None and learner.name in publishing
        if versioned_learner and learner.name in parent_versions:
            parent_version = parent_versions[learner.name]
        else:
            parent_version = _model_version(model_manager.get_champion(learner.name))
        if train_config is not None and parent_version is None and train_config.model_version is not None:
            parent_version = train_config.model_version
        train_steps = 0
        stage_name = None
        if task_manager is not None:
            train_steps = task_manager.learner_state.get(learner.name, {}).get("learning_completed", 0)
            stage_name = getattr(task_manager, "stage_name", None)
        metadata: dict = {"task_id": task_id, "reason": reason}
        if is_learning_batch and learn_info is not None:
            metadata.update(sample_episode_num=learn_info.episode_num, learn_id=learn_info.learn_id,
                            generation=learn_info.generation, stage_name=learn_info.stage_name)
        if stage_name is not None:
            metadata["stage_name"] = stage_name
        submitted = model_manager.submit_model(
            model_name=learner.name,
            algo_type=learner.algo_type,
            models=snapshot_model_weights(learner.models),
            parent_version=parent_version,
            learner_id=learner.name,
            train_steps=train_steps,
            metrics=metrics.get(learner.name, {}),
            metadata=metadata,
            **({"version": versions[learner.name]} if versioned_learner else {}),
            promote=promote,
            is_learning_batch=is_learning_batch,
        )
        if is_learning_batch and learn_info is not None:
            LOG.info("Learner model published: learner=%s, model_version=%s, sample_episode_num=%s, "
                     "learn_id=%s, stage=%s, generation=%s",
                     learner.name, _model_version(submitted), learn_info.episode_num, learn_info.learn_id,
                     learn_info.stage_name, learn_info.generation)
        submitted_names.append(learner.name)
    if target is not None:
        replay_state = replay_buffer.state() if overlap_agents else None
        task_manager.complete_model_publication(versions, replay_state=replay_state)
    return submitted_names


def sync_agents_from_model_manager(
        policy_agents: list[Agent],
        model_manager: ModelManagerPort,
        *,
        learner_names: Optional[list[str] | set[str] | dict[str, Any]] = None,
) -> None:
    """采样进程按AgentName从模型池拉取champion策略网络权重。

    固定策略、以及没有对应 Learner 的 Agent 不进模型池，跳过同步以免 get_model 找不到模型。
    """
    allowed = None if learner_names is None else set(learner_names)
    for agent in policy_agents:
        if bool(getattr(agent.config, "fixed_mode", False)):
            LOG.info(f"Skip ModelManager sync for fixed-policy agent: {agent.name}")
            continue
        if allowed is not None and agent.name not in allowed:
            LOG.info(f"Skip ModelManager sync for agent without learner: {agent.name}")
            continue
        model_weights = model_manager.get_model(model_name=agent.name, output_type="state_dict")
        agent_checkpoint = model_weights.get("agent_net")
        assert agent_checkpoint is not None, f"ModelManager[{agent.name}] must provide agent_net checkpoint."
        agent.agent_net.load_state_dict(extract_state_dict(agent_checkpoint), strict=False)
        agent.agent_net.to(agent.device)
        agent.agent_net.eval()


def sync_agent_from_model_manager(
        agent: Agent,
        model_manager: ModelManagerPort,
        version: Optional[str],
) -> None:
    """按明确版本只拉取一个Agent需要的agent_net，避免传输训练侧辅助网络。"""
    model_weights = model_manager.get_model(
        model_name=agent.name,
        version=version,
        output_type="state_dict",
        artifact_names=["agent_net"],
    )
    agent_checkpoint = model_weights.get("agent_net")
    assert agent_checkpoint is not None, f"ModelManager[{agent.name}] must provide agent_net checkpoint."
    agent.agent_net.load_state_dict(extract_state_dict(agent_checkpoint), strict=False)
    agent.agent_net.to(agent.device)
    agent.agent_net.eval()


def latest_metrics_by_learner(learn_results) -> dict:
    """把RolloutLearnWorker返回结果压缩成每个AgentName对应的最新一次学习指标。"""
    metrics = {}
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


def updated_learners_from_result(learners: list[Learner], learn_results) -> list[Learner]:
    """根据本轮学习结果筛选真正发生过参数更新的Learner。"""
    updated_names = set(latest_metrics_by_learner(learn_results))
    return [learner for learner in learners if learner.name in updated_names]


def _model_version(model_info) -> Optional[str]:
    """兼容本地ModelInfo和RPC返回dict。"""
    if model_info is None:
        return None
    if isinstance(model_info, ModelInfo):
        return model_info.version
    if isinstance(model_info, dict):
        return model_info.get("version")
    return getattr(model_info, "version", None)

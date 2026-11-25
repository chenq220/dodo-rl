from __future__ import annotations
import torch
from typing import List, Literal, Optional, TYPE_CHECKING, Union, TypeVar

from common.logger import init_logger

if TYPE_CHECKING:
    from algos.learners import Learner
    from algos.agents import Agent
    from envs.base import BaseEnvironment
    from dodo_sim.simulations.base import Simulation
    from config import EnvConfig, LearnerConfig, AgentConfig
    from dodo_sim.config import SimConfig
    from common.types import AgentID, FilePath
    from rpc.interfaces import SimManagerPort

    SimType = TypeVar('SimType', bound=Simulation)
    EnvType = TypeVar('EnvType', bound=BaseEnvironment)
    AgentType = TypeVar('AgentType', bound=Agent)
    LearnerType = TypeVar('LearnerType', bound=Learner)

LOG = init_logger('creator')


def create_env(
        scene_name: str,
        config: EnvConfig,
        sim_config: Optional[SimConfig] = None,
        connect_sim: bool = True,
        sim_manager: SimManagerPort = None,
        role_id: str = None,
) -> EnvType:
    """
    创建环境
    :param scene_name: 场景名称，清单见：envs/__init__.py
    :param config: 环境配置
    :param connect_sim: 是否连仿真
    :param sim_config: 仿真配置
    :param role_id: 调用它的角色ID
    :param sim_manager: 仿真管理模块输入
    :return: 环境实例
    """
    from common.plugins import RUNTIME_PLUGINS
    scene_env = RUNTIME_PLUGINS.resolve("env", scene_name)
    LOG.info(f'Created env {scene_env.__name__}.')
    effective_connect_sim = bool(connect_sim and config.use_sim)
    if effective_connect_sim:
        assert sim_config is not None, (
            f"EnvConfig.use_sim=True requires sim_config for scene={scene_name}."
        )
    return scene_env(
        config=config,
        connect_sim=effective_connect_sim,
        sim_config=sim_config,
        sim_manager=sim_manager,
        role_id=role_id,
    )


def create_sim(
        sim_type,
        sim_config: SimConfig,
        sim_manager: SimManagerPort = None,
        role_id: str = None,
) -> SimType:
    from common.plugins import RUNTIME_PLUGINS
    if RUNTIME_PLUGINS.has_resource("sim", sim_type):
        sim_class = RUNTIME_PLUGINS.resolve("sim", sim_type)
    else:
        from common.simulation import load_sim_module
        sim_class = load_sim_module("dodo_sim.simulations.registry").simulation_class(sim_type)
    LOG.info(f'Created sim {sim_class.__name__}.')
    return sim_class(
        sim_type=sim_config.sim_type,
        config=sim_config,
        sim_manager=sim_manager,
        role_id=role_id,
    )


def create_agent(
        agent_type: str,
        agent_config: AgentConfig,
        agent_net_source: Literal['leaner', 'file', 'blanco'],
        agent_list: List[AgentID],
        device: Optional[Union[torch.device, str]],
        model_file_path: FilePath = None,
        **kwargs) -> AgentType:
    """
    agent实例化
    :param agent_type: agent网络类型
    :param agent_config: 智能体配置
    :param agent_list: agent清单
    :param agent_net_source: 权重来源：learner ：从学习进程缓存出的网络权重，file：从文件加载网络权重，blanco：加载空网络权重
    :param device: 计算设备
    :param model_file_path: file源指定模型文件路径
    :return: agent类实例
    """
    from common.plugins import RUNTIME_PLUGINS
    agent_class = RUNTIME_PLUGINS.resolve("agent", agent_type)
    LOG.info(f'Created agent {agent_class.__name__}')
    return agent_class(agent_type=agent_type, config=agent_config, agent_list=agent_list,
                       agent_net_source=agent_net_source, device=device, model_file_path=model_file_path,
                       **kwargs)


def create_learner(learner_config: LearnerConfig, agent, env, device, net_source,
                   model_file_path=None, caller=None, uploader=None, **kwargs) -> LearnerType:
    """
    创建学习进程
    :param learner_config: Learner算法配置，必须包含algo_type。
    :param agent: 智能体实例
    :param env: 环境实例
    :param device: 计算设备
    :param net_source: 初始化权重来源：caller ：从外部进程的函数中获取网络权重，file：从文件加载网络权重，blanco：加载空网络权重
    :param model_file_path: 初始化模型file源指定模型文件路径
    :param caller: 初始化模型获取网络权重的外部函数
    :param uploader: 将模型上传到外部程序
    :param kwargs: 算法拓展参数
    :return: 学习进程类实例
    """
    from common.plugins import RUNTIME_PLUGINS
    algo_type = learner_config.algo_type
    learner_class = RUNTIME_PLUGINS.resolve("learner", algo_type)
    LOG.info(f'Created learner {learner_class.__name__}.')
    return learner_class(algo_type=algo_type, config=learner_config, agent=agent, env=env, device=device,
                         net_source=net_source, model_file_path=model_file_path, caller=caller,
                         uploader=uploader, **kwargs)

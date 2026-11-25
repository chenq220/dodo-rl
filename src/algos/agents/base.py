from __future__ import annotations
import os
import torch
import torch.nn as nn
import numpy as np
import copy
from gymnasium.spaces import Discrete, Box
from functools import wraps
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List, Callable, Optional, TYPE_CHECKING, Union, Generic, TypeVar
from common.logger import init_logger
from utils.print_model import print_model_info
from envs.action.base import ActionTable
from envs.action.spaces import resolve_action_space, require_action_space_support
from envs.action.model import Action, CompositeAction
from envs.action.spaces import ContinuousAction, MultiHeadAction
from common.exception import AgentGroupCompatibilityException, AgentGroupConfigException
from .input_shape import AgentInputShape

if TYPE_CHECKING:
    from config import AgentConfig
    from common.types import (
        ActionMask, AgentHiddenIn,
        AgentHiddenOut, ObsType, ModelSource,
        AgentActon, AgentLastAction, FilePath)

LOG = init_logger('agent')

AgentID = TypeVar("AgentID", str, int)
ActionType = TypeVar("ActionType", bound=Action | CompositeAction | ContinuousAction | MultiHeadAction)


def resolve_agent_groups(agent_config: "AgentConfig") -> tuple[tuple[AgentID, ...], ...]:
    """
    解析 AgentConfig 的隐式单组/显式无名分组入口，返回二维成员视图。

    校验规则（装配期执行，config.py/toml_config.py 不做）：
    1. agent_names 与 agent_groups 恰好一个非空；
    2. agent_groups 外层非空且每个内层组非空；
    3. 同一 Agent 内所有 AgentID 全局唯一，不能跨组重复；
    4. 显式分组的所有组等宽（第一版不做 padding）。

    旧 agent_names 入口解释为唯一隐式组，行为与历史版本一致。
    """
    names = list(agent_config.agent_names)
    groups_raw = agent_config.agent_groups
    if names and groups_raw:
        raise AgentGroupConfigException(
            agent_name=agent_config.name,
            detail="agent_names 与 agent_groups 不能同时非空，请只保留一个分组入口。",
        )
    if not names and not groups_raw:
        raise AgentGroupConfigException(
            agent_name=agent_config.name,
            detail="agent_names 与 agent_groups 至少需要一个非空。",
        )
    if not groups_raw:
        return (tuple(names),)

    groups: list[tuple[AgentID, ...]] = []
    seen: set[AgentID] = set()
    for group in groups_raw:
        if not isinstance(group, (list, tuple)) or not group:
            raise AgentGroupConfigException(
                agent_name=agent_config.name,
                detail=f"agent_groups 的每个内层组必须是非空列表，given={group!r}。",
            )
        for agent_id in group:
            if agent_id in seen:
                raise AgentGroupConfigException(
                    agent_name=agent_config.name,
                    detail=f"AgentID={agent_id!r} 在 agent_groups 内跨组重复。",
                )
            seen.add(agent_id)
        groups.append(tuple(group))
    widths = {len(group) for group in groups}
    if len(widths) > 1:
        raise AgentGroupConfigException(
            agent_name=agent_config.name,
            detail=f"显式分组的所有组必须等宽，当前组宽={sorted(widths)}。",
        )
    return tuple(groups)


def record_action_warp(func):
    """
    通用智能体动作记录装饰器。
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        actions, hidden_state = func(self, *args, **kwargs)
        space = getattr(self, "action_space", None)
        if space is not None:
            for action in actions.values():
                space.validate(action)
        self.last_actions = (actions, hidden_state)
        return actions, hidden_state

    return wrapper


class Agent(ABC, Generic[AgentID, ActionType]):
    """
    通用智能体框架，负责把业务观测转换为动作表中的 Action 实体。

    Agent 是业务侧直接调用的策略接口；具体实现可以是神经网络、规则策略或随机策略，
    但对外必须通过 get_actions 返回每个 agent 的业务动作和下一步隐藏状态。
    """

    # Custom rule/policy plugins opt in explicitly after implementing the corresponding codec.
    supported_action_spaces = ("discrete",)
    fixed_output_type: str | None = None

    def __init__(
            self,
            agent_type: str,
            config: AgentConfig,
            agent_net_source: ModelSource,
            device: Optional[Union[torch.device, str]],
            model_file_path: FilePath = None,
            caller: Callable[[], Dict[str, Dict]] = None,
            **kwargs
    ):
        """
        agent实例化
        :param agent_type: agent网络类型
        :param config: 算法配置
        :param agent_net_source: 权重来源：caller ：从外部进程的函数中获取网络权重，file：从文件加载网络权重，blanco：加载空网络权重
        :param device: 计算设备
        :param model_file_path: file源指定模型文件路径
        :param caller: 获取网络权重的外部函数
        :param dim_info: 网络维度配置
        """
        LOG.info(f'Agent name is [{config.name}]')
        LOG.info(f'Agent type is [{agent_type}]')
        LOG.info(f'Agent device is [{device}]')
        # Agent名称
        self.name = config.name
        # Agent 类型标识，对应注册表中的 agent 类型。
        self.agent_type = agent_type
        # torch 计算设备，策略网络推理和隐藏状态张量应放在该设备。
        self.device = torch.device(device) if isinstance(device, str) else device
        # 策略网络实例；规则 Agent 可构建空网络，但字段仍保留供统一接口访问。
        self.agent_net: Optional['nn.Module'] = None
        # Agent 配置，包含动作定义、观测维度、隐藏层维度和网络共享方式。
        self.config = config
        if self.fixed_output_type is not None:
            self.config.agent_output_type = self.fixed_output_type
        # 最近一次 get_actions 的返回值缓存，便于外部调试或录制动作。
        self.last_actions: Tuple[AgentActon, AgentHiddenOut]
        # 无名分组二维视图；旧 agent_names 入口解释为唯一隐式组。
        self.agent_groups: Tuple[Tuple[AgentID, ...], ...] = resolve_agent_groups(config)
        # 当前 Agent 管理的业务智能体名称列表；agent_groups 按外层声明顺序展开。
        self.agent_list: List[AgentID] = [
            agent_id for group in self.agent_groups for agent_id in group
        ]
        self._validate_grouped_agent_type()
        # 智能体数量；隐式单组等于扁平成员数，显式分组等于固定组宽。
        # 算法侧通常用它构造 [batch, time, n_agents, dim] 张量。
        self.n_agents = len(self.agent_groups[0])
        # 动作表，把离散动作索引映射为业务 Action 实体。
        self.action_space = resolve_action_space(config)
        require_action_space_support(self, self.action_space, label=f"Agent {self.agent_type} ({self.name})")
        self.action_layout = self.action_space.layout
        self.action_table = getattr(self.action_space, "table", None)
        # Agent输入与网络维度定义；n_agents 为固定组宽（隐式单组等于扁平成员数）。
        self.dim_info = AgentInputShape(
            obs_dim=self.config.obs_dim,
            action_dim=self.action_table.n_actions if self.action_table is not None else 0,
            action_layout=self.action_layout,
            n_agents=self.n_agents,
            obs_last_action=self.config.obs_last_action,
            obs_agent_id=self.config.obs_agent_id,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers
        )
        # 离散动作空间，Environment.action_space 会直接返回该对象。
        self.action_dim = self.action_space.to_gym_space()
        # 单智能体观测空间，Environment.observation_space 会直接返回该对象。
        self.obs_dim = Box(low=-1.0, high=1.0, shape=(config.obs_dim,), dtype=np.float32)
        # 初始化网络
        if self.config.algo_config.use_net:
            self._setup_network(agent_net_source, model_file_path, caller, self.dim_info)
        # 打印模型信息
        print_model_info(LOG, self.agent_net, self.agent_type)

    def _validate_grouped_agent_type(self) -> None:
        """显式分组首版仅支持共享 RNN Agent（可训练须配 QMIX，固定策略允许 fixed_mode）。"""
        if len(self.agent_groups) <= 1:
            return
        from config import AgentType
        if self.agent_type != AgentType.RNN.value:
            raise AgentGroupCompatibilityException(
                agent_name=self.name,
                detail=(
                    f"显式分组第一版仅支持 agent_type=rnn，given={self.agent_type}。"
                    f"SB3/RNN_NS/蒙特卡洛等类型请继续使用 agent_names 单组入口。"
                ),
            )

    def _setup_network(self, agent_net_source, model_file_path, caller, dim_info, **kwargs) -> None:
        """通用的网络构建与权重恢复控制流。"""
        if agent_net_source == 'caller':
            assert caller, "agent_net_source='caller' requires caller."
            LOG.info(f"Setup network caller is: {caller}")
            caller_checkpoint = caller()
            try:
                checkpoint_net = caller_checkpoint['agent_net']
            except KeyError:
                raise RuntimeError("Caller return don't has the key agent_net")

            self.agent_net, processed_state_dict = self.build_network_from_checkpoint(checkpoint_net, dim_info,
                                                                                      **kwargs)
            self.agent_net.load_state_dict(processed_state_dict)

        elif agent_net_source == 'file':
            assert model_file_path, "agent_net_source='file' requires model_file_path."
            LOG.info(f"Setup Agent network model file: {model_file_path}")
            if not os.path.exists(model_file_path):
                raise FileNotFoundError(f"Model file not found at: {model_file_path}")

            file_checkpoint = torch.load(model_file_path, map_location=self.device)
            self.agent_net, processed_state_dict = self.build_network_from_checkpoint(file_checkpoint, dim_info,
                                                                                      **kwargs)
            self.agent_net.load_state_dict(processed_state_dict, strict=False)

        elif agent_net_source == 'blanco':
            LOG.info("Build a blanco network.")
            self.agent_net = self.build_empty_network(dim_info, **kwargs)

        else:
            raise ValueError(f"Unsupported network source: {agent_net_source}")

        self.agent_net.to(self.device)
        self.agent_net.eval()

    def reset_net(self, model: 'nn.Module'):
        assert isinstance(model, type(self.agent_net)), (
            f"reset_net model type mismatch, expected={type(self.agent_net).__name__}, "
            f"given={type(model).__name__}"
        )
        self.agent_net = copy.deepcopy(model)
        self.agent_net.to(self.device)
        self.agent_net.eval()

    @abstractmethod
    def get_actions(
            self,
            obs: Dict[AgentID, ObsType],
            agent_list: List[AgentID],
            hidden_in: AgentHiddenIn,
            action_mask: ActionMask = None,
            epsilon: float = 0.0,
            evaluate: bool = False,
            last_actions: AgentLastAction = None,
            temperature: float = 1.0,
            deterministic: bool = False,
            **kwargs
    ) -> Tuple[AgentActon, AgentHiddenOut]:
        """
        根据业务观测直接返回动作实体。

        显式分组时 Rollout 在组循环内把 self.agent_list 临时切换为当前组成员，
        Agent 实现无需感知分组，仍按 self.agent_list 做槽位映射。

        :param obs: 各agent私有态势
        :param agent_list: 需要输出动作的agent清单
        :param hidden_in: agent隐藏信息
        :param action_mask: 动作掩码，None则所有动作可用
        :param epsilon: 随机探索系数，是否使用由具体Agent实现决定
        :param evaluate: 是否评估模式
        :param last_actions: 上步动作拼接，None则不拼接
        :param temperature: 温度系数，是否使用由具体Agent实现决定
        :param deterministic: 是否确定性动作选择
        :param kwargs: 下层实现需要的其他参数
        :return: 动作实体和下一步隐藏状态
        """
        raise NotImplementedError('子类必须实现 get_actions 方法')

    def build_empty_network(self, dim_info: 'AgentInputShape', **kwargs) -> 'nn.Module':
        """ 构建空网络权重 """
        raise NotImplementedError('子类必须实现 build_empty_network 方法')

    def build_network_from_checkpoint(self, checkpoint, dim_info: 'AgentInputShape', **kwargs) \
            -> Tuple['nn.Module', Dict[str, torch.Tensor]]:
        """
        从权重信息加载网络
        :param checkpoint: 权重信息结构
        :param dim_info: 检查checkpoint是否满足agent的维度信息要求
        :param kwargs: 拓展参数
        :return: 空网络结构和网络权重
        """
        raise NotImplementedError('子类必须实现 build_empty_network 方法')

    def get_init_hidden_state(self) -> Dict[AgentID, Any]:
        """ 获取初始隐藏态势 """
        raise NotImplementedError('子类必须实现 get_init_hidden_state 方法')

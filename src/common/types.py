from __future__ import annotations
import numpy as np
import torch
from torch import Tensor
from torch import nn
from typing import Dict, TypeVar, List, Optional,  TypedDict
# from typing import Any, Optional, Literal, TYPE_CHECKING, Union
from typing_extensions import LiteralString
from enum import Enum
from envs.action.model import Action, ActionDefined, CompositeAction, StepActionDefined, EpisodeActionDefined
from envs.action.spaces import ContinuousAction, MultiHeadAction
from algos.agents import Agent
from gymnasium.spaces import Discrete, Box

AgentType = TypeVar("AgentType", bound=Agent)

# 智能体名称 --> AgentConfig.name 和 LearnerConfig.name
AgentName = str
# Learner.models / ModelManager artifacts 的索引名 --> 如 agent_net, critic_net
ModelName = TypeVar("ModelName", bound=str)
# 多智能体下属实例
AgentID = TypeVar("AgentID", str, int)
# 空动作由Action.is_empty_action表达；None仍用于智能体未激活等无动作数据占位。
ActionType = TypeVar("ActionType", bound=Action | CompositeAction | ContinuousAction | MultiHeadAction | None)
StateType = TypeVar("StateType", Box, np.ndarray, torch.Tensor)
ObsType = TypeVar("ObsType", Box, np.ndarray, torch.Tensor)
AgentNet = TypeVar("AgentNet", bound=nn.Module)
QmixNet = TypeVar("QmixNet", bound=nn.Module)
FilePath = TypeVar("FilePath", LiteralString, str, bytes)
NetWeight = TypeVar("NetWeight", bound=dict)

ActionMask = Optional[Dict[AgentID, Optional[List[int] | Dict[str, List[int]]]]]
AgentHiddenIn = Dict[AgentID, Tensor]
AgentHiddenOut = Dict[AgentID, Tensor]
AgentActon = Dict[AgentID, ActionType]
# 上一步动作输入；episode第一步没有真实上一动作，允许用None作为占位。
AgentLastAction = Dict[AgentID, Optional[ActionType]]
ActionDefinition = TypeVar("ActionDefinition", ActionDefined, EpisodeActionDefined, StepActionDefined)
DefinedActionList = List[ActionDefinition]


class ParameterItem(TypedDict, total=False):
    # 动作参数按枚举项定义
    key: str
    values: List[int]
    # 动作参数按步进定义
    max: int
    min: int
    step: int
    unit: str


class ModelSource(Enum):
    """ 加载模型权重方式枚举 """
    # 从外部函数加载
    caller = "caller"
    # 从文件加载
    file = "file"
    # 创建空模型
    blanco = "blanco"

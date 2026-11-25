"""Environment 插件开发所需的稳定公共接口；Simulation 由 dodo-sim 提供。"""
from __future__ import annotations

from ._runtime import bootstrap_runtime

bootstrap_runtime()

# 源码工程中使用现有顶层导入；构建 wheel 时会统一改写为
# ``dodorl_sdk.dodorl.*``，保证公开接口与内部运行时共享同一类对象。
from common.exception import ValueErrorException
from config import EnvConfig
from envs.base import AECEnvironment, AgentID, ObsType, ParallelEnvironment, StateType
from envs.state.spatial import GeoPoint



__all__ = [
    "AgentID",
    "AECEnvironment",
    "EnvConfig",
    "GeoPoint",
    "ObsType",
    "ParallelEnvironment",
    "ValueErrorException",
    "StateType",
]

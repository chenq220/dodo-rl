"""Agent/Learner 高级插件开发公共接口。"""
from __future__ import annotations

from ._runtime import bootstrap_runtime

bootstrap_runtime()

from algos.agents import Agent, AgentInputShape
from algos.learners import Learner
from algos.map import AlogsMap
from common.plugins import (
    AlgorithmPluginSpec,
    agent_plugin,
    learner_plugin,
)
from config import AgentAlgoConfig, LearnerAlgoConfig
from algos.learners.batch_schema import (
    BatchSchema,
    DEFAULT_EPISODE_SCHEMA,
    DefaultEpisodeFields,
    DictField,
    EpisodeTensorFields,
    ListField,
    SINGLE_AGENT_SCHEMA,
    SingleAgentFields,
    TensorField,
)
from sample.model import EpisodeBatch

__all__ = [
    "Agent",
    "AgentAlgoConfig",
    "AgentInputShape",
    "AlgorithmPluginSpec",
    "AlogsMap",
    "BatchSchema",
    "DEFAULT_EPISODE_SCHEMA",
    "DefaultEpisodeFields",
    "DictField",
    "EpisodeBatch",
    "EpisodeTensorFields",
    "Learner",
    "LearnerAlgoConfig",
    "ListField",
    "SINGLE_AGENT_SCHEMA",
    "SingleAgentFields",
    "TensorField",
    "agent_plugin",
    "learner_plugin",
]

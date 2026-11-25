"""训练调度、样本、模型与 RPC 的稳定公共数据契约。"""
from __future__ import annotations

from ._runtime import bootstrap_runtime

bootstrap_runtime()

from models.model import ModelArtifactInfo, ModelInfo, ModelStatus, ModelUsage
from rpc.interfaces import (
    ModelManagerPort,
    MonteCarloMethodPort,
    ReplayBufferPort,
    SimManagerPort,
    TrainTaskManagerPort,
)
from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition
from scheduler.model import ActionParam, EpisodeInfo, LearnInfo, LearnTaskResult, StageLifecycleState

from .config import (
    AgentConfig,
    EnvConfig,
    EvalConfig,
    LearnerConfig,
    StageConfig,
    StageTaskConfig,
    TaskConfig,
    SampleType,
    TrainConfig,
)

__all__ = [
    "ActionParam", "AgentConfig", "EnvConfig", "EpisodeInfo", "EpisodeTransition",
    "EvalConfig", "LearnInfo", "LearnTaskResult", "LearnerConfig", "ModelArtifactInfo",
    "ModelInfo", "ModelManagerPort", "ModelStatus", "ModelUsage", "MonteCarloMethodPort",
    "ReplayBufferPort", "SAEpisodeTransition", "SimManagerPort", "StageConfig",
    "StageLifecycleState", "StageTaskConfig", "TaskConfig", "TrainConfig", "SampleType", "TrainTaskManagerPort",
]

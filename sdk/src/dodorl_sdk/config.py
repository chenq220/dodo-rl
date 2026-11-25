from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from common.plugins import PluginsConfig
from utils.toml_config import TomlTrainingConfig

from config import (
    A2CConfig,
    AgentAlgoConfig,
    AgentConfig,
    AgentOutputType,
    AgentType,
    AlgorithmType,
    AutoMLConfig,
    AutoMLOptunaConfig,
    ConfigEnum,
    DqnConfig,
    EntryConfig,
    EnvConfig,
    EvalConfig,
    LearnerAlgoConfig,
    LearnerConfig,
    LocalStorageConfig,
    LogConfig,
    LogHandlerKind,
    MaddpgAgentConfig,
    MaddpgConfig,
    MappoConfig,
    ModelManagerConfig,
    MonteCarloAgentConfig,
    MonteCarloMethodConfig,
    PpoConfig,
    QmixConfig,
    ReplayBufferConfig,
    ResourceDir,
    RnnAgentConfig,
    MappoRnnAgentConfig,
    PolicyRnnAgentConfig,
    QAgentConfig,
    RnnNsAgentConfig,
    GrpcConfig,
    S3StorageConfig,
    SB3AgentConfig,
    SampleLearningEvalConfig,
    SampleLearningTasksConfig,
    SampleType,
    StageConfig,
    StageTaskConfig,
    StorageBucket,
    StorageType,
    TaskConfig,
    TrainConfig,
    TorchRpcConfig,
)


PROJECT_ROOT_ENV = "DODORL_SDK_PROJECT_ROOT"


@dataclass
class DodoSDKConfig:
    project_root: Path = Path.cwd()


_SDK_CONFIG = DodoSDKConfig()


def configure_sdk(project_root: str | Path | None = None) -> None:
    """配置使用者工程根目录；仿真配置由 dodo-sim 单独管理。"""
    if project_root is not None:
        _SDK_CONFIG.project_root = Path(project_root).expanduser().resolve()


def get_project_root() -> Path:
    return Path(os.environ.get(PROJECT_ROOT_ENV) or _SDK_CONFIG.project_root).expanduser().resolve()


__all__ = [
    "PluginsConfig",
    "TomlTrainingConfig",
    "A2CConfig", "AgentAlgoConfig", "AgentConfig", "AgentOutputType", "AgentType",
    "AlgorithmType", "AutoMLConfig", "AutoMLOptunaConfig", "ConfigEnum", "DqnConfig",
    "EntryConfig", "EnvConfig", "EvalConfig", "LearnerAlgoConfig", "LearnerConfig",
    "LocalStorageConfig", "LogConfig", "LogHandlerKind", "MaddpgAgentConfig", "MaddpgConfig", "MappoConfig", "ModelManagerConfig",
    "MonteCarloAgentConfig", "MonteCarloMethodConfig", "SampleType", "PpoConfig", "QmixConfig",
    "ReplayBufferConfig", "ResourceDir", "RnnAgentConfig", "MappoRnnAgentConfig", "PolicyRnnAgentConfig", "QAgentConfig", "RnnNsAgentConfig", "GrpcConfig",
    "S3StorageConfig", "SB3AgentConfig", "SampleLearningEvalConfig",
    "SampleLearningTasksConfig", "StageConfig",
    "StageTaskConfig", "StorageBucket", "StorageType", "TaskConfig", "TrainConfig",
    "TorchRpcConfig",
    "DodoSDKConfig",
    "PROJECT_ROOT_ENV",
    "configure_sdk",
    "get_project_root",
]

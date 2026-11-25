from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from common.types import AgentName


class ModelStatus(str, Enum):
    """
    模型版本生命周期状态。

    ModelManager 第一版不负责判断模型好坏，只记录版本状态。评估进程或人工策略根据指标
    调用 promote/archive/quarantine/mark_specialist 等接口改变状态。
    """

    # 新提交版本，等待评估或人工处理；并发学习进程提交的模型默认进入该状态。
    CANDIDATE = "candidate"
    # 当前发布版本；采样进程或业务推理进程默认应拉取该版本。
    CHAMPION = "champion"
    # 专项能力版本；综合不一定最优，但可作为后续蒸馏、再训练或策略选择资源。
    SPECIALIST = "specialist"
    # 历史归档版本；不默认发布，但可回滚或作为新训练分支 parent。
    ARCHIVED = "archived"
    # 隔离版本；用于加载失败、结构错误、动作非法等硬错误模型。
    QUARANTINED = "quarantined"


class ModelUsage(str, Enum):
    """
    模型权重在算法中的用途。

    一个算法版本可能包含多个权重文件，例如 QMIX 有 agent_net 和 mixer_net，
    MAPPO 有 agent_net 和 critic_net。usage 用于后续按职责筛选和装配模型。
    """

    # 策略/智能体网络，通常是采样进程需要拉取的主要权重。
    AGENT = "agent"
    # QMIX/VDN/QTRAN 等 value decomposition 算法的混合网络。
    MIXER = "mixer"
    # Actor-Critic 类算法的价值网络或集中式 critic。
    CRITIC = "critic"
    # Learner 级别的其他权重，第一版暂不细分。
    LEARNER = "learner"
    # 无法从权重名称推断用途的其他文件。
    OTHER = "other"


@dataclass
class ModelArtifactInfo:
    """
    一个模型版本中的单个权重文件描述。

    它只描述“某个权重文件在哪里、是什么用途”，不持有 nn.Module 对象。这样 ModelInfo
    可以安全序列化到 JSON，模型权重本体通过 local_path/storage_key 另行保存。
    """

    # 权重名称，例如 agent_net、mixer_net、critic_net。
    name: str
    # 权重用途，用于后续按 agent/mixer/critic 等类型筛选。
    usage: str
    # 本地权重文件路径。
    local_path: str
    # 后端存储对象 key；第一版和本地文件名保持一致，使用扁平命名。
    storage_key: str
    # 权重文件格式，当前第一版统一使用 torch.save 生成的 pth。
    format: str = "pth"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelArtifactInfo":
        return cls(**data)


@dataclass
class ModelInfo:
    """
    一个可追踪、可晋升、可回滚的模型版本记录。

    ModelInfo 是模型池的核心索引数据：记录一个版本包含哪些权重、来自哪个学习进程、
    基于哪个父版本、当前处于什么状态。它不负责加载模型，也不负责评估模型。
    """

    # 逻辑模型名，训练链路中使用 AgentName，即 AgentConfig.name / LearnerConfig.name。
    model_name: AgentName
    # 版本号，由 ModelManager 生成或提交方指定。
    version: str
    # 算法类型，例如 qmix、mappo。
    algo_type: str
    # 当前版本状态。
    status: str = ModelStatus.CANDIDATE.value
    # 父版本；用于回滚和追踪并发学习分支来源。
    parent_version: Optional[str] = None
    # 当前版本包含的权重文件。
    artifacts: Dict[str, ModelArtifactInfo] = field(default_factory=dict)
    # 指标信息；第一版只保存，不参与自动评估。
    metrics: Dict[str, Any] = field(default_factory=dict)
    # 业务元数据，例如场景名、样本来源、训练配置摘要等。
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 已晋升过的历史版本链路快照。
    history_versions: list[str] = field(default_factory=list)
    # 提交模型的学习进程标识。
    learner_id: Optional[str] = None
    # 提交时学习步数。
    train_steps: int = 0
    # 创建时间，ISO-like 字符串。
    created_at: str = ""
    # 更新时间，ISO-like 字符串。
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为可写入单版本 metadata.json 的普通字典。"""
        data = asdict(self)
        data["artifacts"] = {
            name: artifact.to_dict()
            for name, artifact in self.artifacts.items()
        }
        return _json_safe(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """从 JSON 字典恢复模型版本记录。"""
        payload = dict(data)
        payload["artifacts"] = {
            name: ModelArtifactInfo.from_dict(artifact)
            for name, artifact in payload.get("artifacts", {}).items()
        }
        return cls(**payload)


def _json_safe(value: Any) -> Any:
    """把训练指标里的numpy/torch/path/enum对象递归转成JSON可写类型。"""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value

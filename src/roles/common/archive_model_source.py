"""从单个 AutoML 归档中解析课程启动所需的 Learner 模型文件。"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from config import AlgorithmType, LearnerConfig


_ALGO_MODEL_NAMES: dict[str, tuple[str, ...]] = {
    AlgorithmType.COMA.value: ("agent_net", "critic_net"),
    AlgorithmType.IA2C.value: ("agent_net", "critic_net"),
    AlgorithmType.IPPO.value: ("agent_net", "critic_net"),
    AlgorithmType.IQL.value: ("agent_net",),
    AlgorithmType.MAA2C.value: ("agent_net", "critic_net"),
    AlgorithmType.MADDPG.value: ("agent_net", "critic_net"),
    AlgorithmType.MAPPO.value: ("agent_net", "critic_net"),
    AlgorithmType.PAC.value: ("agent_net", "critic_net", "state_value_net"),
    AlgorithmType.PAC_DCG.value: ("agent_net", "critic_net", "state_value_net"),
    AlgorithmType.QMIX.value: ("agent_net", "mixer_net"),
    AlgorithmType.VDN.value: ("agent_net", "mixer_net"),
}


def required_model_names(learner_config: LearnerConfig) -> tuple[str, ...]:
    """返回 file 初始化必须提供的组件；SB3 类算法的 policy 已包含 actor/critic。"""
    return _ALGO_MODEL_NAMES.get(str(learner_config.algo_type), ("agent_net",))


def resolve_archive_models_dir(model_load_dir: str) -> Path:
    """允许配置 AutoML 归档根目录，也允许直接配置 artifacts/models 目录。"""
    root = Path(str(model_load_dir)).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Course bootstrap model_load_dir does not exist: {root}")
    archive_models = root / "artifacts" / "models"
    if archive_models.is_dir():
        return archive_models
    return root


def resolve_learner_archive_model_files(
        model_load_dir: str,
        model_version: str,
        learner_name: str,
        model_names: Iterable[str],
) -> dict[str, str]:
    """用完整模型版本补齐 Learner/组件名；缺失时拒绝启动。"""
    version = str(model_version).strip()
    if not version:
        raise ValueError("Course bootstrap model_version must be a non-empty filename prefix")
    models_dir = resolve_archive_models_dir(model_load_dir)
    resolved: dict[str, str] = {}
    for model_name in model_names:
        filename = f"{version}_{learner_name}_{model_name}.pth"
        path = models_dir / filename
        if not path.is_file():
            raise ValueError(
                f"Course bootstrap model file is missing: learner={learner_name}, "
                f"model={model_name}, filename={filename!r}, dir={models_dir}"
            )
        resolved[model_name] = str(path)
    return resolved

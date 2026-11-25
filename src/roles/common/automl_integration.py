"""Role 侧 AutoML：实例创建与 restart 建议落地。"""
from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

from algos.learners import Learner
from common.types import AgentName
from config import AutoMLConfig, LearnerConfig, StageConfig
from scheduler.model import AutoMLAdvice

if TYPE_CHECKING:
    from scheduler.automl import AutoML


def create_auto_ml_if_enabled(
        auto_ml_config: Optional[AutoMLConfig],
        learner_configs: List[LearnerConfig],
        *,
        stage_name: str = "train",
) -> Optional["AutoML"]:
    """auto_ml_config 缺失或监控关闭时返回 None，供 Role 注入 TrainTaskManager。"""
    if not auto_ml_config or not auto_ml_config.monitor_enabled:
        return None
    from scheduler.automl import AutoML
    from scheduler.utils.automl_config import finalize_auto_ml_config

    finalize_auto_ml_config(auto_ml_config)
    return AutoML(
        auto_ml_config,
        learner_configs,
        stage_name=stage_name,
    )


def build_auto_ml_by_stage(stages: List[StageConfig]) -> Dict[str, Optional["AutoML"]]:
    """按阶段预创建 AutoML 实例；StageTask 建 TM 时按 stage_name 取用。"""
    auto_ml_by_stage: Dict[str, Optional[AutoML]] = {}
    for stage in stages:
        auto_ml_by_stage[stage.stage_name] = create_auto_ml_if_enabled(
            stage.auto_ml_config,
            stage.learners_config,
            stage_name=stage.stage_name,
        )
    return auto_ml_by_stage


def learners_apply_restart_advice(
        learners_by_name: Dict[AgentName, Learner],
        advice: Optional[AutoMLAdvice],
) -> None:
    """restart 回调中把 AutoML 建议同步到 Learner 运行态。"""
    if advice is None:
        return
    for name, overrides in advice.algo_config_overrides.items():
        learner = learners_by_name.get(name)
        if learner is None or not overrides:
            continue
        learner.dynamic_modify_config(overrides)
    for name, overrides in advice.skeleton_overrides.items():
        learner = learners_by_name.get(name)
        if learner is None or not overrides:
            continue
        for key, value in overrides.items():
            if hasattr(learner.config, key):
                setattr(learner.config, key, value)


def env_apply_restart_metadata(env, task_manager) -> None:
    """AutoML restart 后让 Role 持有的本地/无头环境刷新当前任务 metadata。"""
    if env is None or task_manager is None:
        return
    metadata = getattr(task_manager.task_config, "env_metadata", None)
    if metadata:
        env.apply_env_metadata(metadata)

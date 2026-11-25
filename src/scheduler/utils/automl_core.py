"""AutoML 纯函数：指标聚合、坍塌检测、参数拆分、路径命名（不 import optuna）。"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import AutoMLConfig, LearnerConfig, ResourceDir
from scheduler.model import (
    AutoMLAdvice,
    AutoMLPrimaryMetric,
    TrainingRunArchive,
    TrainingSnapshot,
)


def _set_nested(root: Dict[str, Any], path: str, value: Any) -> None:
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _sanitize_archive_folder_component(value: str) -> str:
    sanitized = re.sub(r"[^\w.\-]+", "_", str(value).strip())
    return sanitized or "unknown"


def build_archive_folder_name(bundle: TrainingRunArchive) -> str:
    """归档目录名：{环境名}-{任务ID}-{已完成局数}-{阶段名}-gen{代次}。"""
    env_name = _sanitize_archive_folder_component(bundle.env_name)
    task_id = _sanitize_archive_folder_component(bundle.task_id)
    stage_name = _sanitize_archive_folder_component(bundle.stage_name)
    return (
        f"{env_name}-{task_id}-{bundle.completed_episodes}-"
        f"{stage_name}-gen{bundle.generation}"
    )


def resolve_optuna_sqlite_storage(subject_name: str, stage_name: str) -> tuple[str, str]:
    """Optuna sqlite 路径与 study 名：{subject_name}_{stage_name}.db。"""
    studies_dir = os.path.join(ResourceDir.AUTOML.value, "studies")
    os.makedirs(studies_dir, exist_ok=True)
    study_key = f"{subject_name}_{stage_name}".replace("/", "_")
    db_path = os.path.join(studies_dir, f"{study_key}.db")
    return study_key, f"sqlite:///{db_path}"


@dataclass
class TrialPeakTracker:
    peak: float = float("-inf")


def check_in_trial_collapse(
        metric_value: float,
        completed_episodes: int,
        config: AutoMLConfig,
        tracker: TrialPeakTracker,
) -> Optional[str]:
    """本 trial 内奖励坍塌检测，补充 MedianPruner 在单 trial 下永不剪枝的空洞。"""
    ratio = float(config.collapse_restart_ratio)
    if ratio <= 0:
        return None
    if not math.isfinite(metric_value):
        return None

    tracker.peak = max(tracker.peak, metric_value)
    if completed_episodes < int(config.collapse_min_episodes):
        return None

    peak = tracker.peak
    if peak <= 0 or not math.isfinite(peak):
        return None

    threshold = peak * ratio
    if metric_value <= threshold:
        return (
            f"reward_collapse: metric={metric_value:.4f} peak={peak:.4f} "
            f"threshold={threshold:.4f} ratio={ratio}"
        )
    return None


def check_target_metric_deadline(
        metric_value: float,
        completed_episodes: int,
        config: AutoMLConfig,
) -> Optional[str]:
    """在课程 deadline 检查绝对业务指标，未达标即剪枝当前 trial。"""
    business_metric = config.business_metric
    if business_metric is not None and business_metric.enabled:
        target = business_metric.target
        deadline = int(business_metric.deadline_episodes)
        reason_prefix = "business_metric_missed"
    else:
        target = config.target_metric
        deadline = int(config.target_metric_deadline_episodes)
        reason_prefix = "target_metric_missed"
    if target is None or deadline <= 0 or completed_episodes < deadline:
        return None
    if not math.isfinite(metric_value):
        return (
            f"{reason_prefix}: metric={metric_value} target={float(target):.4f} "
            f"deadline={deadline}"
        )

    direction = str(config.optuna.direction)
    achieved = metric_value >= float(target) if direction == "maximize" else metric_value <= float(target)
    if not achieved:
        comparator = ">=" if direction == "maximize" else "<="
        return (
            f"{reason_prefix}: metric={metric_value:.4f} target{comparator}{float(target):.4f} "
            f"deadline={deadline} direction={direction}"
        )
    return None


def aggregate_reward(snapshot: TrainingSnapshot, config: AutoMLConfig) -> float:
    """从奖励或episode业务指标的滑动窗口聚合 trial.report 主指标。"""
    business_metric = config.business_metric
    if business_metric is not None and business_metric.enabled:
        from scheduler.utils.business_metrics import (
            business_expression_names,
            evaluate_business_expression,
            reduce_values,
        )
        names = business_expression_names(business_metric.expression)
        values: List[float] = []
        for index in range(snapshot.completed_episodes):
            variables: Dict[str, float] = {}
            for name in names:
                series = snapshot.episode_metrics.get(name)
                if series is None or index >= len(series) or series[index] is None:
                    raise ValueError(
                        f"business metric output missing: name={name}, episode={index + 1}"
                    )
                variables[name] = float(series[index])
            values.append(evaluate_business_expression(business_metric.expression, variables))
        if not values:
            return float("-inf")
        return reduce_values(values[-business_metric.window:], business_metric.aggregation)

    window = max(1, int(config.reward_window))
    primary = config.metric_primary
    if primary == AutoMLPrimaryMetric.STAGE_SUCCESS_RATE.value:
        values = [
            float(value)
            for value in snapshot.episode_metrics.get("stage_success", [])
            if value is not None and math.isfinite(float(value))
        ]
        if not values:
            return float("-inf")
        tail = values[-window:]
        return sum(tail) / len(tail)

    curves = snapshot.reward_curve
    if not curves:
        return float("-inf")

    agent_windows: List[float] = []
    for rewards in curves.values():
        valid = [float(r) for r in rewards if r is not None and math.isfinite(float(r))]
        if not valid:
            continue
        tail = valid[-window:]
        agent_windows.append(sum(tail) / len(tail))

    if not agent_windows:
        return float("-inf")

    if primary == AutoMLPrimaryMetric.REWARD_SUM.value:
        return sum(agent_windows)
    if primary == AutoMLPrimaryMetric.REWARD_MIN_AGENT.value:
        return min(agent_windows)
    return sum(agent_windows) / len(agent_windows)


def build_tunable_catalog(
        learner_configs: List[LearnerConfig],
        config: AutoMLConfig,
) -> Dict[str, List[str]]:
    """{learner_name: [path, ...]}；仅含 TOML 显式声明的 Learner 搜索项。"""
    catalog: Dict[str, List[str]] = {}
    known = {item.name for item in learner_configs}
    for item in config.learner_search_space:
        if item.agent_name in known:
            catalog.setdefault(item.agent_name, []).append(item.path)
    return catalog


def split_trial_params(
        params: Dict[str, Any],
        learner_configs: List[LearnerConfig],
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Any], Optional[int]]:
    """把统一 trial.params 拆回 Learner、环境 metadata 与 seed 三类覆盖。"""
    algo_overrides: Dict[str, Dict] = {lc.name: {} for lc in learner_configs}
    skeleton_overrides: Dict[str, Dict] = {lc.name: {} for lc in learner_configs}
    seed: Optional[int] = None
    env_metadata_overrides: Dict[str, Any] = {}
    learners_by_name = {lc.name: lc for lc in learner_configs}

    for key, value in params.items():
        if key == "seed":
            seed = int(value)
            continue
        if key.startswith("env_metadata."):
            _set_nested(env_metadata_overrides, key[len("env_metadata."):], value)
            continue
        if not key.startswith("learner."):
            continue
        learner_path = key[len("learner."):]
        if "." not in learner_path:
            continue
        learner_name, config_path = learner_path.split(".", 1)
        learner_config = learners_by_name.get(learner_name)
        if learner_config is None:
            continue
        if config_path.startswith("algo_config."):
            algo_overrides[learner_name][config_path[len("algo_config."):]] = value
        else:
            skeleton_overrides[learner_name][config_path] = value

    algo_overrides = {k: v for k, v in algo_overrides.items() if v}
    skeleton_overrides = {k: v for k, v in skeleton_overrides.items() if v}
    return algo_overrides, skeleton_overrides, env_metadata_overrides, seed


def build_restart_advice(
        params: Dict[str, Any],
        learner_configs: List[LearnerConfig],
        reason: str,
) -> AutoMLAdvice:
    algo_overrides, skeleton_overrides, env_overrides, seed = split_trial_params(params, learner_configs)
    return AutoMLAdvice(
        decision="restart",
        algo_config_overrides=algo_overrides,
        skeleton_overrides=skeleton_overrides,
        env_metadata_overrides=env_overrides,
        seed=seed,
        reason=reason or "optuna_restart",
    )

"""归档 / 未来落库共用的行构建器。"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from scheduler.model import FRAMEWORK_UPDATE_KEYS, EpisodeInfo


def build_episode_rows(
        *,
        sample_completed: int,
        generation: int,
        sample_task_seq: Sequence[EpisodeInfo],
        learn_results: Mapping[str, Mapping[str, List[Any]]],
        learner_names: Iterable[str],
) -> List[Dict[str, Any]]:
    """按已完成采样局组装 algos_metrics.csv 行。"""
    enabled = set(learner_names)
    episode_rows: List[Dict[str, Any]] = []
    for ep_idx in range(sample_completed):
        episode_num = ep_idx + 1
        row: Dict[str, Any] = {
            "episode": episode_num,
            "generation": generation,
        }
        if ep_idx < len(sample_task_seq):
            row["epsilon"] = sample_task_seq[ep_idx].epsilon
        for learner_name, store in learn_results.items():
            if learner_name not in enabled:
                continue
            ep_nums = store.get("episode_num", [])
            for metric_key, series in store.items():
                if metric_key in ("episode_num",) or metric_key in FRAMEWORK_UPDATE_KEYS:
                    continue
                for i, ep in enumerate(ep_nums):
                    if ep == episode_num and i < len(series):
                        row[f"{learner_name}_{metric_key}"] = series[i]
                        break
        episode_rows.append(row)
    return episode_rows


def build_metric_output_rows(
        *,
        sample_completed: int,
        generation: int,
        sample_metrics: Mapping[str, Sequence[Optional[float]]],
        output_names: Iterable[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """按配置的业务输出名组装逐局指标行。"""
    metric_output_rows: Dict[str, List[Dict[str, Any]]] = {}
    for metric_name in output_names:
        values = sample_metrics.get(metric_name, [])
        rows: List[Dict[str, Any]] = []
        for ep_idx in range(sample_completed):
            value = values[ep_idx] if ep_idx < len(values) else None
            if value is not None:
                rows.append({
                    "episode": ep_idx + 1,
                    "generation": generation,
                    "value": value,
                })
        metric_output_rows[metric_name] = rows
    return metric_output_rows

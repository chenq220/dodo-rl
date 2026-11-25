"""训练任务状态存储协议。本次不接线、无默认实现调用。

自然主键：(task_id, stage_name, generation, episode_num)。
TrainTaskManager 的 `_record_*` 写入点是将来落库的唯一挂钩：

- `_record_episode_claimed`
- `_record_episode_completed`
- `_record_episode_failed`
- `_record_episode_reclaimed`
- `_record_learn_token_issued`
- `_record_learn_token_dispatched`
- `_record_learn_token_completed`
- `_record_metric_sample`
- `_record_progress`
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol


class TaskStateStore(Protocol):
    """任务进度 / 游标 / 指标的持久化后端。当前仅作类型声明。"""

    def record_episode_claimed(
            self,
            *,
            task_id: str,
            stage_name: str,
            generation: int,
            episode_num: int,
            lease_id: str,
    ) -> None: ...

    def record_episode_completed(
            self,
            *,
            task_id: str,
            stage_name: str,
            generation: int,
            episode_num: int,
            total_rewards: Mapping[str, float],
    ) -> None: ...

    def record_episode_failed(
            self,
            *,
            task_id: str,
            stage_name: str,
            generation: int,
            episode_num: int,
            failed_count: int,
    ) -> None: ...

    def record_episode_reclaimed(
            self,
            *,
            task_id: str,
            stage_name: str,
            generation: int,
            episode_num: int,
            lease_id: Optional[str],
    ) -> None: ...

    def record_learn_token_issued(
            self,
            *,
            task_id: str,
            stage_name: str,
            generation: int,
            episode_num: int,
            active_learners: list[str],
    ) -> None: ...

    def record_learn_token_dispatched(
            self,
            *,
            task_id: str,
            stage_name: str,
            generation: int,
            episode_num: int,
    ) -> None: ...

    def record_learn_token_completed(
            self,
            *,
            task_id: str,
            stage_name: str,
            generation: int,
            episode_num: int,
            learn_result: Mapping[str, Any],
    ) -> None: ...

    def record_metric_sample(
            self,
            *,
            task_id: str,
            stage_name: str,
            generation: int,
            episode_num: int,
            metrics: Mapping[str, float],
    ) -> None: ...

    def record_progress(
            self,
            *,
            task_id: str,
            stage_name: str,
            generation: int,
            sample_completed: int,
            max_episode: int,
    ) -> None: ...

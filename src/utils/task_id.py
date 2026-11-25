from __future__ import annotations

import os
import threading
import uuid


_TASK_ID_ENV = "DODO_TASK_ID"
_EMPTY_TASK_ID_VALUES = {"", "none", "null"}


class GetTaskId:
    """维护当前进程唯一且可在角色启动前冻结的任务 ID。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._task_id = self._read_environment_or_generate()
        self._frozen = False

    @staticmethod
    def _normalize(configured_task_id: str | None) -> str | None:
        if configured_task_id is None:
            return None
        if not isinstance(configured_task_id, str):
            raise TypeError(
                "task_id must be a string, null sentinel, or omitted; "
                f"got {type(configured_task_id).__name__}"
            )
        normalized = configured_task_id.strip()
        return None if normalized.lower() in _EMPTY_TASK_ID_VALUES else normalized

    def _read_environment_or_generate(self) -> str:
        environment_value = os.environ.get(_TASK_ID_ENV, "").strip()
        resolved = environment_value or uuid.uuid4().hex[:8]
        os.environ[_TASK_ID_ENV] = resolved
        return resolved

    def __str__(self) -> str:
        with self._lock:
            return self._task_id

    def configure(self, configured_task_id: str | None) -> str:
        """应用非空配置；空值保留当前 ID，冻结后禁止切换任务。"""
        normalized = self._normalize(configured_task_id)
        with self._lock:
            if normalized is None:
                return self._task_id
            if self._frozen and normalized != self._task_id:
                # 延迟导入，避免 common.logger -> utils.task_id -> common.exception
                # 的模块初始化环。
                from common.exception import TaskIdFrozenException

                raise TaskIdFrozenException(
                    current_task_id=self._task_id,
                    configured_task_id=normalized,
                )
            self._task_id = normalized
            os.environ[_TASK_ID_ENV] = normalized
            return self._task_id

    def freeze(self) -> str:
        """幂等冻结并返回当前进程最终任务 ID。"""
        with self._lock:
            self._frozen = True
            os.environ[_TASK_ID_ENV] = self._task_id
            return self._task_id


task_id = GetTaskId()

__all__ = ["task_id"]

from __future__ import annotations

import threading
from typing import Optional

from common.logger import init_logger
from scheduler.stage_task import StageTask
from scheduler.train_task import TrainTaskManager

LOG = init_logger("custom_stage_task")


class AlphaStageTask(StageTask):
    """
    项目定制 StageTask 示例。

    当前用途是验证任务重启闭环：在第二阶段采样完成数达到阈值后，等待学习侧空闲，再调用
    TrainTaskManager.restart_training_task。是否立刻重启由TaskManager按generation安全边界决定；
    重启后Role会按新generation清空ReplayBuffer、恢复Learner快照并同步Agent。
    每次重启后 sample_task_completed 归零，再次达到阈值会再次触发，不限制重启次数。
    """

    _RESTART_SAMPLE_COMPLETED_THRESHOLD = 50
    _RESTART_TRIGGER_STAGE_INDEX = 1
    _MONITOR_INTERVAL_SECONDS = 0.2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._restart_count = 0
        self._requested_generation: Optional[int] = None
        self._monitor_stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

    def start_next_stage(self) -> TrainTaskManager:
        task_manager = super().start_next_stage()
        if self.current_stage_index == self._RESTART_TRIGGER_STAGE_INDEX:
            self._start_restart_monitor(task_manager)
        return task_manager

    def _close_current_task_manager(self) -> None:
        self._stop_restart_monitor()
        super()._close_current_task_manager()

    def close(self) -> None:
        self._stop_restart_monitor()
        super().close()

    def _start_restart_monitor(self, task_manager: TrainTaskManager) -> None:
        self._stop_restart_monitor()
        self._requested_generation = None
        self._monitor_stop_event = threading.Event()
        stage_name = self.current_stage.stage_name if self.current_stage is not None else "unknown"
        self._monitor_thread = threading.Thread(
            target=self._restart_monitor_loop,
            args=(task_manager,),
            name=f"alpha_stage_task_restart_monitor_{stage_name}",
            daemon=True,
        )
        self._monitor_thread.start()
        LOG.info(
            f"AlphaStageTask restart monitor started: stage_name={stage_name}, "
            f"threshold={self._RESTART_SAMPLE_COMPLETED_THRESHOLD}, max_restart=unlimited"
        )

    def _stop_restart_monitor(self) -> None:
        thread = self._monitor_thread
        if thread is None:
            return
        self._monitor_stop_event.set()
        if thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._monitor_thread = None

    def _restart_monitor_loop(self, task_manager: TrainTaskManager) -> None:
        while not self._monitor_stop_event.wait(self._MONITOR_INTERVAL_SECONDS):
            if self.current_task_manager is not task_manager:
                return

            current_generation = task_manager.generation
            if self._requested_generation is not None:
                if current_generation > self._requested_generation:
                    self._restart_count += 1
                    self._requested_generation = None
                else:
                    continue

            completed = task_manager.sample_task_completed
            if completed < self._RESTART_SAMPLE_COMPLETED_THRESHOLD:
                continue

            LOG.warning(
                f"AlphaStageTask requests task restart: stage_name={task_manager.stage_name}, "
                f"completed_restart={self._restart_count}, "
                f"sample_task_completed={completed}, threshold={self._RESTART_SAMPLE_COMPLETED_THRESHOLD}"
            )
            result = task_manager.restart_training_task(
                reason=(
                    "AlphaStageTask monitor triggered: "
                    f"completed_restart={self._restart_count}, "
                    f"sample_task_completed={completed} >= {self._RESTART_SAMPLE_COMPLETED_THRESHOLD}"
                )
            )
            if result.get("pending_restart"):
                self._requested_generation = int(result["generation"])
            else:
                self._restart_count += 1

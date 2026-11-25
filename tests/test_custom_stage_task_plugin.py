from threading import RLock
from types import SimpleNamespace
from unittest.mock import MagicMock

from portal.plugins.custom_stage_task import AlphaStageTask


def test_alpha_stage_task_restart_threshold_and_pending_flag():
    task = AlphaStageTask.__new__(AlphaStageTask)
    task._restart_count = 0
    task._requested_generation = None
    task._lifecycle_lock = RLock()
    task._monitor_stop_event = MagicMock()
    task._monitor_stop_event.wait.side_effect = [False, True]
    manager = SimpleNamespace(
        generation=1,
        stage_name="evade",
        sample_task_completed=50,
        restart_training_task=MagicMock(return_value={"pending_restart": True, "generation": 1}),
    )
    task._current_task_manager = manager
    task._restart_monitor_loop(manager)
    manager.restart_training_task.assert_called_once()
    assert task._requested_generation == 1

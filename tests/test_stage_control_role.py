from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from config import StageConfig, StageTaskConfig, TrainConfig
from roles import stage_control
from roles.stage_control import StageControlRole


class _ReplayBuffer:
    def __init__(self) -> None:
        self.generations = []
        self.archive_calls = []
        self.sample_ratios = []
        self.complete_calls = []

    def clear(self, generation=None) -> None:
        self.generations.append(generation)

    def archive_from_active(self, stage_name, ratio, select, *, seed=None) -> int:
        self.archive_calls.append((stage_name, ratio, select))
        return 0

    def set_rehearsal_sample_ratio(self, ratio: float) -> None:
        self.sample_ratios.append(ratio)

    def notify_stage_complete(self, stage, *, task_id="") -> None:
        self.complete_calls.append((stage, task_id))

    def close(self) -> None:
        return None


class _StageTask:
    def __init__(self) -> None:
        self.unready = []
        self.prepared = []
        self.stage_task_config = StageTaskConfig(start_stage=0, skip_stage_replay_path=None)

    @contextmanager
    def preparing_current_stage(self, task_manager):
        self.prepared.append(task_manager)
        yield task_manager

    def mark_current_stage_unready(self, task_manager) -> None:
        self.unready.append(task_manager)


class _TaskManager:
    def __init__(self) -> None:
        self.task_id = "task-a"
        self.stage_name = "stage-a"
        self.restart_callbacks = []
        self.learning_checks = 0
        self.notified = False

    def add_restart_callback(self, callback) -> None:
        self.restart_callbacks.append(callback)

    def is_sample_completed(self) -> bool:
        return True

    def is_learning_completed(self) -> bool:
        self.learning_checks += 1
        return self.learning_checks >= 2

    def notify_stage_complete(self) -> None:
        self.notified = True


def test_stage_control_waits_for_learning_before_advancing(monkeypatch) -> None:
    role = object.__new__(StageControlRole)
    role.role_id = "stage-control-test"
    role.stage_task = _StageTask()
    role.replay_buffer = _ReplayBuffer()
    role.monte_carlo_method = SimpleNamespace(
        close=lambda: None,
        write_enabled=False,
    )
    role.artifacts_manager = SimpleNamespace(
        put_log=lambda *args, **kwargs: None,
        complete=lambda *args, **kwargs: None,
        expect_producer=lambda *args, **kwargs: None,
    )
    role._skip_stage_replay_loaded = False
    task_manager = _TaskManager()
    stage = StageConfig(
        stage_name="stage-a",
        train_config=TrainConfig(),
        clear_replay_buffer=True,
        rehearsal_archive_ratio=0.0,
        rehearsal_archive_select="even_span",
        rehearsal_sample_ratio=0.0,
    )
    sleep_calls = []

    monkeypatch.setattr(stage_control, "apply_train_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(stage_control, "load_skip_stage_replay_if_needed", lambda *args, **kwargs: False)
    monkeypatch.setattr(stage_control.time, "sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr(role, "_activate_stage_monte_carlo", lambda *args, **kwargs: None)

    role._run_stage(task_manager, stage)

    assert role.replay_buffer.generations == [0]
    assert role.replay_buffer.sample_ratios == [0.0]
    assert role.replay_buffer.complete_calls == [(stage, "task-a")]
    assert task_manager.restart_callbacks == [role._on_task_restart]
    assert task_manager.learning_checks == 2
    assert sleep_calls == [0.2]
    assert task_manager.notified is True
    assert role.stage_task.prepared == [task_manager]
    assert role.stage_task.unready == [task_manager, task_manager]


def test_stage_control_restart_updates_replay_generation() -> None:
    role = object.__new__(StageControlRole)
    role.replay_buffer = _ReplayBuffer()

    role._on_task_restart(3, advice={"reason": "retry"})

    assert role.replay_buffer.generations == [3]

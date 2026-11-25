from __future__ import annotations

from types import SimpleNamespace

from scheduler import rollout_learn
from scheduler.rollout_learn import RolloutLearnWorker
from scheduler.model import LearnInfo, LearnTaskResult


class _Learner:
    def __init__(self) -> None:
        self.env = SimpleNamespace(stage_name="train")
        self.reset_count = 0
        self.snapshot_count = 0

    def reset_generation(self) -> None:
        self.reset_count += 1

    def clone_models_snapshot(self) -> None:
        self.snapshot_count += 1


class _TaskManager:
    def __init__(self, stage_names: list[str]) -> None:
        self.stage_names = iter(stage_names)

    def get_learn_task_info(self) -> LearnTaskResult:
        return LearnTaskResult(learn_info=LearnInfo(stage_name=next(self.stage_names)))


def test_online_learner_resets_generation_and_snapshot_once_per_stage(monkeypatch) -> None:
    learner = _Learner()
    worker = object.__new__(RolloutLearnWorker)
    worker.learners = [learner]
    worker.task_manager = _TaskManager(["stage-a", "stage-a", "stage-b"])
    worker.sample_learn_manager = None
    worker._online_no_task_logged = False
    worker._online_stage_name = None
    monkeypatch.setattr(rollout_learn, "set_global_seed", lambda *args, **kwargs: None)

    for expected_stage in ("stage-a", "stage-a", "stage-b"):
        with worker.online_learn_task_context() as session:
            assert session.learn_info.stage_name == expected_stage

    assert learner.env.stage_name == "stage-b"
    assert learner.reset_count == 2
    assert learner.snapshot_count == 2

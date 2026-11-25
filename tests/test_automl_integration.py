from types import SimpleNamespace

from config import AutoMLConfig, StageConfig
from roles.common.automl_integration import (
    build_auto_ml_by_stage,
    create_auto_ml_if_enabled,
    env_apply_restart_metadata,
    learners_apply_restart_advice,
)
from scheduler.model import AutoMLAdvice


def test_create_auto_ml_if_enabled_returns_none_when_disabled():
    assert create_auto_ml_if_enabled(None, []) is None
    assert create_auto_ml_if_enabled(AutoMLConfig(subject_name="x", monitor_enabled=False), []) is None


def test_build_auto_ml_by_stage_maps_stage_names():
    stages = [
        StageConfig(stage_name="search", auto_ml_config=AutoMLConfig(subject_name="x", monitor_enabled=False)),
        StageConfig(stage_name="evade"),
    ]
    result = build_auto_ml_by_stage(stages)
    assert result["search"] is None
    assert result["evade"] is None


def test_learners_apply_restart_advice():
    learner = SimpleNamespace(config=SimpleNamespace(hidden_dim=8), calls=[])
    learner.dynamic_modify_config = lambda overrides: learner.calls.append(overrides)
    advice = AutoMLAdvice(
        decision="restart",
        algo_config_overrides={"uav": {"lr": 0.1}},
        skeleton_overrides={"uav": {"hidden_dim": 16}},
    )
    learners_apply_restart_advice({"uav": learner}, advice)
    assert learner.calls == [{"lr": 0.1}]
    assert learner.config.hidden_dim == 16
    learners_apply_restart_advice({"uav": learner}, None)


def test_env_apply_restart_metadata():
    env = SimpleNamespace(applied=None)
    env.apply_env_metadata = lambda meta: setattr(env, "applied", meta)
    task_manager = SimpleNamespace(task_config=SimpleNamespace(env_metadata={"k": 1}))
    env_apply_restart_metadata(env, task_manager)
    assert env.applied == {"k": 1}
    env_apply_restart_metadata(None, task_manager)

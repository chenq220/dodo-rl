"""Standalone evaluation must acquire and release the same resources as training."""
from unittest.mock import MagicMock

import pytest

from config import AgentConfig, EnvConfig, EvalConfig
from types import SimpleNamespace
from roles import evaluation


@pytest.fixture
def evaluation_components(monkeypatch):
    manager = MagicMock()
    pool = MagicMock()
    env = MagicMock()
    factories = {}
    for name, instance in [("TrainTaskManager", manager), ("create_sim_manager", pool),
                           ("ReplayBuffer", MagicMock()), ("create_agent", MagicMock()),
                           ("create_env", env), ("RolloutSampleWorker", MagicMock())]:
        factories[name] = MagicMock(return_value=instance)
        monkeypatch.setattr(evaluation, name, factories[name])
    monkeypatch.setattr(evaluation, "assert_state_shape_matches_env", lambda *args: None)
    monkeypatch.setattr(evaluation, "inject_env_dims", lambda *args: None)
    monkeypatch.setattr(evaluation, "resolve_artifacts_manager", lambda value: MagicMock())
    monkeypatch.setattr(evaluation, "bind_local_artifact_callers", lambda *args: (None, None, None))
    monkeypatch.setattr(evaluation, "attach_artifact_producer", lambda *args: None)
    monkeypatch.setattr(evaluation.EvaluationRole, "_resolve_model_file_paths", lambda *args: {})
    return factories, manager, pool, env


@pytest.mark.parametrize("source,use_sim", [("manager", True), ("local", True), ("manager", False)])
def test_evaluation_manager_wiring_and_close_order(evaluation_components, monkeypatch, source, use_sim):
    factories, manager, pool, env = evaluation_components
    config = SimpleNamespace(use_sim=True)
    role = evaluation.EvaluationRole(
        EvalConfig(), AgentConfig(name="a", agent_names=["a"]),
        EnvConfig(agent_names=["a"], use_sim=use_sim),
        SimpleNamespace(config_source=source), sim_manager_config=config,
    )
    args = factories["create_env"].call_args.kwargs
    expected = pool if source == "manager" and use_sim else None
    assert args["sim_manager"] is expected and args["role_id"] == "eval"
    assert factories["create_sim_manager"].call_count == int(expected is pool)
    if expected is pool:
        factories["create_sim_manager"].assert_called_once_with(config=config)
    order = []
    env.close.side_effect = lambda: order.append("env")
    pool.close.side_effect = lambda: order.append("pool")
    manager.is_sample_completed.return_value = True
    for method in ["_collect_eval_episode_results", "_log_summary", "finished_report"]:
        monkeypatch.setattr(role, method, lambda: None)
    monkeypatch.setattr(evaluation, "log_simulation_middleware", lambda *args: None)
    monkeypatch.setattr(evaluation, "complete_role_artifact_producers", lambda *args: None)
    role.run()
    assert order == (["env", "pool"] if expected is pool else ["env"])


def test_evaluation_disabled_manager_fails_without_acquiring_resource(evaluation_components):
    factories, _, _, _ = evaluation_components
    with pytest.raises(ValueError, match="use_sim=True"):
        evaluation.EvaluationRole(EvalConfig(), AgentConfig(), EnvConfig(),
            SimpleNamespace(config_source="manager"), sim_manager_config=SimpleNamespace(use_sim=False))
    factories["create_sim_manager"].assert_not_called()


def test_evaluation_initialization_failure_stops_manager(evaluation_components):
    factories, _, pool, _ = evaluation_components
    factories["create_env"].side_effect = RuntimeError("connection failed")
    with pytest.raises(RuntimeError, match="connection failed"):
        evaluation.EvaluationRole(EvalConfig(), AgentConfig(name="a", agent_names=["a"]),
            EnvConfig(agent_names=["a"]), SimpleNamespace(config_source="manager"),
            sim_manager_config=SimpleNamespace(use_sim=True))
    pool.close.assert_called_once_with()


def test_evaluation_manager_requires_external_config(evaluation_components):
    factories, _, _, _ = evaluation_components
    with pytest.raises(ValueError, match="injected SimManagerConfig"):
        evaluation.EvaluationRole(EvalConfig(), AgentConfig(), EnvConfig(),
                                  SimpleNamespace(config_source="manager"))
    factories['create_sim_manager'].assert_not_called()

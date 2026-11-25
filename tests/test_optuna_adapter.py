from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import scheduler.utils.optuna_adapter as optuna_adapter
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import AutoMLConfig, AutoMLOptunaConfig, LearnerConfig
from scheduler.model import AutoMLEnvMetadataSearchParamConfig, AutoMLLearnerSearchParamConfig, AutoMLSeedSearchConfig
from scheduler.utils.optuna_adapter import OptunaStudySession, build_study_distributions
from scheduler.utils.automl_core import split_trial_params


def test_optuna_logging_uses_project_logger(monkeypatch) -> None:
    """适配层关闭 Optuna 默认 stderr handler，并注册同名项目 logger。"""
    calls: list[str] = []
    monkeypatch.setattr(
        optuna_adapter.optuna.logging,
        "disable_default_handler",
        lambda: calls.append("disable_default_handler"),
    )
    monkeypatch.setattr(
        optuna_adapter,
        "init_logger",
        lambda name: calls.append(f"init_logger:{name}"),
    )

    optuna_adapter.configure_optuna_project_logging()

    assert calls == ["disable_default_handler", "init_logger:optuna"]


def test_build_distributions_mappo_lr_seed() -> None:
    config = AutoMLConfig(
        subject_name="distribution",
        learner_search_space=[AutoMLLearnerSearchParamConfig(
            agent_name="archer", path="algo_config.lr", low=1e-4, high=5e-4,
        )],
        env_metadata_search_space=[AutoMLEnvMetadataSearchParamConfig(
            path="target_policy", type="categorical", choices=["nearest", "threat"],
        )],
        seed_search=AutoMLSeedSearchConfig(low=10, high=20),
    )
    dists = build_study_distributions([LearnerConfig(name="archer", algo_type="mappo")], config)
    assert "learner.archer.algo_config.lr" in dists
    assert "seed" in dists
    assert "env_metadata.target_policy" in dists
    assert dists["learner.archer.algo_config.lr"].low == 1e-4
    assert dists["learner.archer.algo_config.lr"].high == 5e-4
    assert dists["seed"].low == 10


def test_optuna_ask_report_tell_loop() -> None:
    dists = build_study_distributions(
        [LearnerConfig(name="archer", algo_type="mappo")],
        AutoMLConfig(subject_name="loop", learner_search_space=[
            AutoMLLearnerSearchParamConfig(agent_name="archer", path="algo_config.lr")
        ]),
    )
    with tempfile.TemporaryDirectory() as tmp:
        storage = f"sqlite:///{tmp}/test.db"
        config = AutoMLOptunaConfig()
        session = OptunaStudySession.open(
            config,
            study_name="smoke_test",
            storage=storage,
        )
        handle, params = session.ask(dists)
        assert "learner.archer.algo_config.lr" in params
        decision = session.report(handle, 0.5, step=10)
        assert decision.should_prune is False
        session.tell_complete(handle, 0.5)
        summary = session.export_summary(handle)
        assert summary.trial_number == handle.trial_number
        assert summary.state in {"COMPLETE", "RUNNING", "PRUNED", "FAIL"}
        baseline_summary = session.export_summary(None)
        assert baseline_summary.trial_number == -1
        assert baseline_summary.state == "NONE"
        assert baseline_summary.params == {}


def test_split_trial_params_includes_nested_env_metadata() -> None:
    algo, skeleton, env, seed = split_trial_params({
        "learner.knight.algo_config.lr": 0.0003,
        "learner.knight.gamma": 0.98,
        "env_metadata.reward_weights.navigation": 0.25,
        "seed": 123,
    }, [LearnerConfig(name="knight", algo_type="mappo")])

    assert algo == {"knight": {"lr": 0.0003}}
    assert skeleton == {"knight": {"gamma": 0.98}}
    assert env == {"reward_weights": {"navigation": 0.25}}
    assert seed == 123


def test_cmaes_rejects_categorical_environment_search() -> None:
    config = AutoMLConfig(
        subject_name="categorical-cmaes",
        optuna=AutoMLOptunaConfig(sampler_name="cmaes"),
        env_metadata_search_space=[AutoMLEnvMetadataSearchParamConfig(
            path="target_policy", type="categorical", choices=["nearest", "threat"],
        )],
    )
    with pytest.raises(ValueError, match="CMA-ES"):
        build_study_distributions([], config)


def test_seed_search_validates_configurable_range() -> None:
    config = AutoMLConfig(
        subject_name="invalid-seed",
        seed_search=AutoMLSeedSearchConfig(low=-1, high=10),
    )
    with pytest.raises(ValueError, match="seed_search"):
        build_study_distributions([], config)

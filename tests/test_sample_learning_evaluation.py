from unittest.mock import MagicMock

import pytest

from common.exception import VarTypeErrorException
from config import (
    AgentConfig,
    EnvConfig,
    EvalConfig,
    LearnerConfig,
    MonteCarloMethodConfig,
    SampleLearningEvalConfig,
    SampleLearningTasksConfig,
    TrainConfig,
)
from roles.sample_learning_evaluation import SampleLearningEvaluationRole


def test_sample_learning_evaluation_requires_load_sample_name(monkeypatch):
    monkeypatch.setattr("roles.sample_learning_evaluation.create_agent", MagicMock())
    monkeypatch.setattr("roles.sample_learning_evaluation.create_env", MagicMock())
    monkeypatch.setattr("roles.sample_learning_evaluation.ReplayBuffer", MagicMock())
    monkeypatch.setattr("roles.sample_learning_evaluation.TrainTaskManager", MagicMock())
    monkeypatch.setattr(
        "roles.sample_learning_evaluation.align_agent_config_with_learner_config",
        lambda *args, **kwargs: None,
    )
    with pytest.raises((AssertionError, VarTypeErrorException)):
        SampleLearningEvaluationRole(
            config=SampleLearningEvalConfig(),
            agent_config=AgentConfig(name="solo", agent_names=["solo"]),
            env_config=EnvConfig(use_sim=False, agent_names=["solo"], state_dim=1),
            sim_config=None,
            monte_carlo_method_config=MonteCarloMethodConfig(),
            train_config=TrainConfig(),
            learner_config=LearnerConfig(name="solo"),
            eval_config=EvalConfig(),
            sample_learning_config=SampleLearningTasksConfig(),
        )

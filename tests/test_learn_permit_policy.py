from types import SimpleNamespace

import pytest

from roles.common.config_alignment import learner_partition_capacities, apply_learner_capacities
from roles.common.config_alignment import apply_replay_capacity_advice
from scheduler.model import AutoMLAdvice
from config import ReplayBufferConfig
from sample.model import ReplayBufferState, SamplingStrategy
from scheduler.tasks.learn_permit_policy import (
    LearnPermitRequest,
    ReplayReadyLearnPermitPolicy,
    replay_buffer_ready_for_learner,
    replay_warmup_remaining,
)


def test_replay_warmup_counts_active_capacity_units():
    cfg = SimpleNamespace(min_replay_size=3, capacity=5)
    state = ReplayBufferState(sample_packaging="independent", agent_trajectory_counts={"a": 2},
                              agent_episode_counts={"a": 1}, agent_rehearsal_step_counts={"a": 100})
    assert replay_warmup_remaining(state, cfg, "a") == 1
    from dataclasses import replace
    state = replace(state, sample_packaging="episode_groups")
    assert replay_warmup_remaining(state, cfg, "a") == 2
    state.agent_episode_counts["a"] = 3
    assert replay_warmup_remaining(state, cfg, "a") == 0


def test_invalid_replay_warmup_is_rejected():
    from config import MaddpgConfig
    for value in (-1, 11, True, 1.5):
        with pytest.raises(ValueError, match="min_replay_size"):
            MaddpgConfig(capacity=10, min_replay_size=value)


def test_task_manager_enforces_warmup_even_if_policy_grants():
    from config import AgentConfig, AgentType, AlgorithmType, LearnerConfig, TrainConfig
    from scheduler.train_task import TrainTaskManager
    from scheduler.tasks.learn_permit_policy import LearnPermitGrant

    learner = LearnerConfig(name="a", algo_type=AlgorithmType.PPO.value)
    learner.algo_config.min_replay_size = 2
    manager = TrainTaskManager(
        config=TrainConfig(versioned=False, max_episode=3, save_optimo_models=False),
        policy_agent_configs=[AgentConfig(name="a", agent_type=AgentType.SB3.value, agent_names=["a"])],
        learner_configs=[learner],
    )
    class GrantAlways:
        def decide(self, request):
            return LearnPermitGrant(active_learners=["a"])
    try:
        manager._learn_permit_policy = GrantAlways()
        manager.replay_buffer_state = ReplayBufferState(agent_trajectory_counts={"a": 1})
        assert manager._decide_learn_permit_locked() is None
        manager.replay_buffer_state = ReplayBufferState(agent_trajectory_counts={"a": 2})
        assert manager._decide_learn_permit_locked().active_learners == ["a"]
    finally:
        manager.close()


@pytest.mark.parametrize("versioned", [False, True])
def test_task_manager_warmup_is_private_to_each_learner(versioned):
    from config import AgentConfig, AgentType, AlgorithmType, LearnerConfig, TrainConfig
    from scheduler.train_task import TrainTaskManager
    from scheduler.tasks.learn_permit_policy import LearnPermitGrant

    learners = [LearnerConfig(name=name, algo_type=AlgorithmType.PPO.value) for name in ("a", "b")]
    learners[0].algo_config.min_replay_size = 3
    learners[1].algo_config.min_replay_size = 1
    for learner in learners:
        learner.algo_config.sample_strategy = "uniform_steps"
        learner.algo_config.learn_times_per_episode = 1
    agents = [AgentConfig(name=name, agent_type=AgentType.SB3.value, agent_names=[name]) for name in ("a", "b")]
    manager = TrainTaskManager(config=TrainConfig(versioned=versioned, max_episode=3, save_optimo_models=False),
                               policy_agent_configs=agents, learner_configs=learners,
                               supports_versioned_sampling=versioned)
    class GrantBoth:
        def decide(self, request):
            return LearnPermitGrant(active_learners=["a", "b"])
    try:
        manager._learn_permit_policy = GrantBoth()
        versions = manager._current_sample_versions_locked() if versioned else {}
        manager.replay_buffer_state = ReplayBufferState(
            agent_trajectory_counts={"a": 2, "b": 1}, sample_versions=versions,
            versioned_available={"a": 2, "b": 1}, versioned_fresh_agents={"a", "b"})
        assert manager._decide_learn_permit_locked().active_learners == ["b"]
    finally:
        manager.close()


def _req(size=0, learners=("a",), configs=None, agent_sizes=None):
    return LearnPermitRequest(
        generation=0,
        stage_name="train",
        replay_buffer_state=ReplayBufferState(
            size=size,
            agent_sizes={name: size for name in learners} if agent_sizes is None else agent_sizes,
        ),
        candidate_learners=learners,
        algo_configs=configs or {},
    )


def test_replay_buffer_ready_off_policy_needs_batch_size():
    cfg = SimpleNamespace(sample_strategy=SamplingStrategy.RECENT_BIASED.value, batch_size=4)
    assert replay_buffer_ready_for_learner(ReplayBufferState(size=3), cfg) is False
    assert replay_buffer_ready_for_learner(ReplayBufferState(size=4), cfg) is True


def test_replay_buffer_ready_step_strategies_need_any_data():
    cfg = SimpleNamespace(
        sample_strategy=SamplingStrategy.ON_POLICY_RECENT_STEPS.value,
        batch_size=100,
    )
    assert replay_buffer_ready_for_learner(ReplayBufferState(size=0), cfg) is False
    assert replay_buffer_ready_for_learner(ReplayBufferState(size=1), cfg) is True


def test_uniform_steps_readiness_is_per_agent_and_needs_only_one_step():
    cfg = SimpleNamespace(sample_strategy="uniform_steps", batch_size=32)
    state = ReplayBufferState(size=0, agent_sizes={"a": 0, "b": 0}, agent_step_counts={"a": 1, "b": 0})
    assert replay_buffer_ready_for_learner(state, cfg, "a")
    assert not replay_buffer_ready_for_learner(state, cfg, "b")


def test_replay_buffer_ready_time_slice_needs_any_data():
    cfg = SimpleNamespace(
        sample_strategy=SamplingStrategy.TIME_SLICE.value,
        batch_size=32,
    )
    assert replay_buffer_ready_for_learner(ReplayBufferState(size=0), cfg) is False
    assert replay_buffer_ready_for_learner(ReplayBufferState(size=1), cfg) is True


def test_decide_returns_none_when_no_learner_is_ready():
    cfg = SimpleNamespace(sample_strategy="recent_biased", batch_size=2)
    grant = ReplayReadyLearnPermitPolicy().decide(
        _req(size=0, configs={"a": cfg}),
    )
    assert grant is None


def test_decide_includes_ready_learners_and_skips_missing_config():
    cfg = SimpleNamespace(sample_strategy="recent_biased", batch_size=2)
    grant = ReplayReadyLearnPermitPolicy().decide(
        _req(size=2, learners=("a", "b"), configs={"a": cfg}),
    )
    assert grant is not None
    assert grant.active_learners == ["a"]


def test_decide_uses_each_learner_partition_only():
    cfg = SimpleNamespace(sample_strategy="recent_biased", batch_size=2)
    grant = ReplayReadyLearnPermitPolicy().decide(
        _req(size=3, learners=("a", "b"), configs={"a": cfg, "b": cfg},
             agent_sizes={"a": 1, "b": 3}),
    )
    assert grant is not None
    assert grant.active_learners == ["b"]


def test_role_applies_only_current_learner_capacities():
    def learner(name, capacity):
        return SimpleNamespace(name=name, algo_config=SimpleNamespace(capacity=capacity))

    class ReplayStub:
        capacities = {}

        def set_agent_capacity(self, name, capacity):
            self.capacities[name] = capacity

    replay = ReplayStub()
    apply_learner_capacities(replay, [learner("a", 2), learner("b", 7)])
    apply_learner_capacities(replay, [learner("a", 5)])
    assert replay.capacities == {"a": 5, "b": 7}
    with pytest.raises(ValueError, match="positive integer"):
        learner_partition_capacities([learner("a", 0)])


def test_automl_capacity_advice_reconfigures_replay_only_for_named_learner():
    class ReplayStub:
        config = ReplayBufferConfig(agent_capacities={"a": 2, "b": 4})

        def set_agent_capacity(self, name, capacity):
            self.config.agent_capacities[name] = capacity

    replay = ReplayStub()
    apply_replay_capacity_advice(
        replay, AutoMLAdvice(decision="restart", algo_config_overrides={"a": {"capacity": 3}}),
    )
    assert replay.config.agent_capacities == {"a": 3, "b": 4}

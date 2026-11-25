import importlib
import random

import numpy as np
import pytest

from config import ReplayBufferConfig
from sample.model import ROLLING_STRATEGIES, SamplingStrategy
from sample.replay_buffer import ReplayBuffer
from sample.sample_transfer import SAEpisodeTransition


AGENT_NAME = "sample_policy"


def _buffer(
        strategy: SamplingStrategy,
        batch_size: int,
        capacity: int = 20,
) -> ReplayBuffer:
    return ReplayBuffer(
        ReplayBufferConfig(
            agent_capacities={AGENT_NAME: capacity},
            batch_size=batch_size,
            strategy=strategy.value,
            recent_ratio=1 / 3,
        )
    )


def _episode(episode_num: int, seq_len: int = 4) -> SAEpisodeTransition:
    episode = SAEpisodeTransition()
    episode.episode_num = episode_num
    for step in range(seq_len):
        episode.add_step(
            obs=np.array([float(episode_num * 10 + step)], dtype=np.float32),
            action=step % 2,
            reward=float(step),
            next_obs=np.array([float(episode_num * 10 + step + 1)], dtype=np.float32),
            terminated=False,
            truncated=False,
            last_action=(step - 1) % 2,
        )
    return episode


def _fill(buffer: ReplayBuffer, count: int) -> None:
    for episode_num in range(1, count + 1):
        buffer.push(
            _episode(episode_num),
            notify_learning=False,
            agent_name=AGENT_NAME,
            episode_num=episode_num,
            sample_cost_secs=0.0,
        )


@pytest.mark.parametrize(
    "strategy,batch_size",
    [
        (SamplingStrategy.RECENT_BIASED, 4),
        (SamplingStrategy.ON_POLICY_RECENT, 4),
        (SamplingStrategy.ON_POLICY_RECENT_STEPS, 6),
        (SamplingStrategy.ON_POLICY_CURSOR, 4),
        (SamplingStrategy.ON_POLICY_CURSOR_STEPS, 6),
        (SamplingStrategy.SLIDING_WINDOW, 4),
        (SamplingStrategy.UNIFORM, 4),
        (SamplingStrategy.UNIFORM_STEPS, 4),
        (SamplingStrategy.TIME_SLICE, 4),
    ],
)
def test_sampling_strategies_return_requested_batch_after_warmup(strategy, batch_size):
    random.seed(7)
    np.random.seed(7)
    buffer = _buffer(strategy, batch_size)
    _fill(buffer, 12)
    batch = buffer.sample(agent_name=AGENT_NAME, batch_size=batch_size, strategy=strategy.value)
    assert batch
    if strategy in (
        SamplingStrategy.ON_POLICY_RECENT_STEPS,
        SamplingStrategy.ON_POLICY_CURSOR_STEPS,
    ):
        total_steps = sum(int(item.seq_len) for item in batch)
        assert total_steps >= 1
    elif strategy == SamplingStrategy.TIME_SLICE:
        assert len(batch) == 1
        assert int(batch[0].seq_len) == batch_size
    else:
        assert len(batch) == batch_size


def test_cursor_sampling_records_consumption_stats_for_pacing():
    # 按局游标：episodes 是精确消费局数，steps 为 0。
    buffer = _buffer(SamplingStrategy.ON_POLICY_CURSOR, 3)
    _fill(buffer, 8)
    batch = buffer.sample(
        agent_name=AGENT_NAME, batch_size=3, strategy=SamplingStrategy.ON_POLICY_CURSOR.value,
    )
    assert len(batch) == 3
    stats = buffer.last_sample_stats[AGENT_NAME]
    assert stats["episodes"] == 3
    assert stats["steps"] == 0
    assert stats["strategy"] == "on_policy_cursor"

    # 按步游标：steps 是实际消费步数，episodes 是跨到的局数（仅作观测）。
    buffer2 = _buffer(SamplingStrategy.ON_POLICY_CURSOR_STEPS, 6)
    _fill(buffer2, 8)
    batch2 = buffer2.sample(
        agent_name=AGENT_NAME, batch_size=6, strategy=SamplingStrategy.ON_POLICY_CURSOR_STEPS.value,
    )
    assert batch2
    stats2 = buffer2.last_sample_stats[AGENT_NAME]
    assert stats2["steps"] == 6
    assert stats2["episodes"] >= 1
    assert stats2["strategy"] == "on_policy_cursor_steps"


def test_uniform_samples_from_pushed_episodes():
    random.seed(11)
    np.random.seed(11)
    buffer = _buffer(SamplingStrategy.UNIFORM, 3)
    _fill(buffer, 8)
    batch = buffer.sample(
        agent_name=AGENT_NAME,
        batch_size=3,
        strategy=SamplingStrategy.UNIFORM.value,
        seed=11,
    )
    nums = {int(item.episode_num) for item in batch}
    assert nums <= set(range(1, 9))
    assert len(batch) == 3


def test_uniform_steps_weights_each_transition_and_samples_with_replacement():
    buffer = _buffer(SamplingStrategy.UNIFORM_STEPS, 32, capacity=2)
    try:
        for episode_num, length in [(1, 1), (2, 3)]:
            buffer.push(_episode(episode_num, length), agent_name=AGENT_NAME, episode_num=episode_num)
        seed = 19
        rng = random.Random(seed)
        expected_values = [10, 20, 21, 22]
        expected_values = [expected_values[rng.randrange(4)] for _ in range(32)]
        batch = buffer.sample(AGENT_NAME, 32, strategy="uniform_steps", seed=seed)
        assert [int(item.obs[0][0]) for item in batch] == expected_values
        assert all(item.seq_len == 1 for item in batch)
        assert all(float(item.next_obs[0][0]) == float(item.obs[0][0]) + 1 for item in batch)
        assert len({int(item.obs[0][0]) for item in batch}) < len(batch)
        assert [int(item.obs[0][0]) for item in buffer.sample_for_rpc(
            AGENT_NAME, 32, strategy="uniform_steps", seed=seed,
        )] == expected_values
    finally:
        buffer.close()


def test_uniform_steps_versioned_admits_one_episode_and_samples_history_without_bias():
    from scheduler.tasks.versioned_sampling import select_version_rule

    buffer = _buffer(SamplingStrategy.UNIFORM_STEPS, 32, capacity=2)
    try:
        _fill(buffer, 1)
        buffer.set_sample_version("policy-0", {AGENT_NAME: (SamplingStrategy.UNIFORM_STEPS, 32)})
        assert buffer.sample(AGENT_NAME, 32, strategy="uniform_steps", seed=0) == []
        buffer.push(_episode(2, 1), agent_name=AGENT_NAME, episode_num=2, sample_version="policy-0")
        state = buffer.state()
        rule = select_version_rule(SamplingStrategy.UNIFORM_STEPS)
        assert rule(32, state.versioned_available[AGENT_NAME], True) == 0
        assert rule(32, state.versioned_available[AGENT_NAME], False) == 1
        batch = buffer.sample(AGENT_NAME, 32, strategy="uniform_steps", seed=4)
        assert len(batch) == 32
        assert {item.episode_num for item in batch} == {1, 2}
        # Uniform draws need not contain the newest episode.
        assert buffer.sample(AGENT_NAME, 1, strategy="uniform_steps", seed=1)[0].episode_num == 1
    finally:
        buffer.close()


def test_uniform_steps_rehearsal_mixes_by_step_with_replacement():
    from types import SimpleNamespace
    from scheduler.tasks.learn_permit_policy import replay_buffer_ready_for_learner

    buffer = _buffer(SamplingStrategy.UNIFORM_STEPS, 8, capacity=4)
    try:
        _fill(buffer, 2)
        assert buffer.archive_from_active("old", 0.25, "newest") == 1
        buffer.set_rehearsal_sample_ratio(0.5)
        batch = buffer.sample(AGENT_NAME, 8, strategy="uniform_steps", seed=3)
        assert sum(item.episode_num == 1 for item in batch) == 4
        assert sum(item.episode_num == 2 for item in batch) == 4
        assert all(item.seq_len == 1 for item in batch)
        assert buffer.archive_from_active("remaining", 0.25, "newest") == 1
        state = buffer.state()
        cfg = SimpleNamespace(sample_strategy="uniform_steps", batch_size=8)
        assert replay_buffer_ready_for_learner(state, cfg, AGENT_NAME)
        assert len(buffer.sample(AGENT_NAME, 8, strategy="uniform_steps", seed=3)) == 8
    finally:
        buffer.close()


def test_uniform_steps_overlap_consumes_arrival_without_waiting_for_batch_size():
    buffer = _buffer(SamplingStrategy.UNIFORM_STEPS, 32, capacity=2)
    try:
        v0, v1 = "sample_test_g0_v000000", "sample_test_g0_v000001"
        buffer.set_sample_version(v1, {AGENT_NAME: (SamplingStrategy.UNIFORM_STEPS, 32)},
                                  overlap_agents=frozenset({AGENT_NAME}))
        buffer.push(_episode(1, 1), agent_name=AGENT_NAME, episode_num=1, sample_version=v0)
        assert AGENT_NAME in buffer.state().overlap_ready_agents
        assert len(buffer.sample(AGENT_NAME, seed=4)) == 32
        assert not buffer.sample(AGENT_NAME, seed=4)
    finally:
        buffer.close()


def test_uniform_steps_versioned_scheduler_learns_after_one_rollout():
    from config import AgentConfig, AgentType, AlgorithmType, LearnerConfig, TrainConfig
    from scheduler.train_task import TrainTaskManager

    learner = LearnerConfig(name=AGENT_NAME, algo_type=AlgorithmType.PPO.value)
    learner.algo_config.batch_size = 32
    learner.algo_config.sample_strategy = "uniform_steps"
    learner.algo_config.learn_times_per_episode = 1
    manager = TrainTaskManager(
        config=TrainConfig(versioned=True, max_episode=2, save_optimo_models=False),
        policy_agent_configs=[AgentConfig(name=AGENT_NAME, agent_type=AgentType.SB3.value,
                                          agent_names=[AGENT_NAME])],
        learner_configs=[learner], supports_versioned_sampling=True,
    )
    buffer = _buffer(SamplingStrategy.UNIFORM_STEPS, batch_size=32, capacity=2)
    buffer.learn_batches_update_caller = manager.learn_batches_update
    try:
        versions, requests = manager.model_publication_target(learning_batch=False)
        buffer.set_sample_version(versions, requests)
        ticket = manager.get_sample_task_info(sample_role_name="S1")
        assert ticket is not None
        assert manager.get_sample_task_info(sample_role_name="S2") is None
        buffer.push(_episode(ticket.episode_num, 1), agent_name=AGENT_NAME,
                    episode_num=ticket.episode_num, sample_version=versions[AGENT_NAME])
        manager.report_sample_result(ticket.episode_num, {}, lease_id=ticket.lease_id)
        assert manager.get_learn_task_info().learn_info is not None
        assert len(buffer.sample(AGENT_NAME, seed=3)) == 32
    finally:
        manager.close()
        buffer.close()


def test_versioned_scheduler_waits_for_replay_warmup():
    from config import AgentConfig, AgentType, AlgorithmType, LearnerConfig, TrainConfig
    from scheduler.train_task import TrainTaskManager

    learner = LearnerConfig(name=AGENT_NAME, algo_type=AlgorithmType.PPO.value)
    learner.algo_config.batch_size = 32
    learner.algo_config.sample_strategy = "uniform_steps"
    learner.algo_config.learn_times_per_episode = 1
    learner.algo_config.min_replay_size = 2
    manager = TrainTaskManager(
        config=TrainConfig(versioned=True, max_episode=3, save_optimo_models=False),
        policy_agent_configs=[AgentConfig(name=AGENT_NAME, agent_type=AgentType.SB3.value,
                                          agent_names=[AGENT_NAME])],
        learner_configs=[learner], supports_versioned_sampling=True,
    )
    buffer = _buffer(SamplingStrategy.UNIFORM_STEPS, batch_size=32, capacity=3)
    buffer.learn_batches_update_caller = manager.learn_batches_update
    try:
        versions, requests = manager.model_publication_target(learning_batch=False)
        buffer.set_sample_version(versions, requests)
        for index in range(2):
            ticket = manager.get_sample_task_info(sample_role_name="S1")
            assert ticket is not None
            buffer.push(_episode(ticket.episode_num, 1), agent_name=AGENT_NAME,
                        episode_num=ticket.episode_num, sample_version=versions[AGENT_NAME])
            manager.report_sample_result(ticket.episode_num, {}, lease_id=ticket.lease_id)
            info = manager.get_learn_task_info().learn_info
            assert (info is not None) == (index == 1)
    finally:
        manager.close()
        buffer.close()


def test_time_slice_returns_fixed_length_windows_inside_one_episode():
    buffer = _buffer(SamplingStrategy.TIME_SLICE, batch_size=3)
    _fill(buffer, 5)
    batch = buffer.sample(
        agent_name=AGENT_NAME,
        batch_size=3,
        strategy=SamplingStrategy.TIME_SLICE.value,
        seed=7,
    )
    assert len(batch) == 1
    item = batch[0]
    assert int(item.seq_len) == 3
    obs_values = [float(obs[0]) for obs in item.obs]
    assert obs_values == [obs_values[0] + offset for offset in range(3)]
    episode_base = int(item.episode_num) * 10
    assert all(episode_base <= value < episode_base + 4 for value in obs_values)
    starts_at_episode = obs_values[0] == float(episode_base)
    assert item.starts_with_episode is starts_at_episode


def test_time_slice_keeps_short_episodes_as_full_trajectory():
    buffer = _buffer(SamplingStrategy.TIME_SLICE, batch_size=3)
    buffer.push(
        _episode(1, seq_len=2),
        notify_learning=False,
        agent_name=AGENT_NAME,
        episode_num=1,
        sample_cost_secs=0.0,
    )
    short = buffer.sample(
        agent_name=AGENT_NAME,
        batch_size=3,
        strategy=SamplingStrategy.TIME_SLICE.value,
        seed=3,
    )
    assert len(short) == 1
    assert int(short[0].episode_num) == 1
    assert int(short[0].seq_len) == 2
    assert short[0].starts_with_episode is True

    buffer.push(
        _episode(2, seq_len=5),
        notify_learning=False,
        agent_name=AGENT_NAME,
        episode_num=2,
        sample_cost_secs=0.0,
    )
    sampled = {
        int(
            buffer.sample(
                agent_name=AGENT_NAME,
                batch_size=3,
                strategy=SamplingStrategy.TIME_SLICE.value,
                seed=seed,
            )[0].episode_num
        )
        for seed in range(40)
    }
    assert sampled == {1, 2}


def _time_slice_start_step(item: SAEpisodeTransition) -> int:
    episode_base = int(item.episode_num) * 10
    return int(float(item.obs[0][0]) - episode_base)


def _expected_time_slice_start(
        episode_lens: list[tuple[int, int]],
        slice_len: int,
        seed: int,
) -> tuple[int, int]:
    """重放 time_slice 的 choices + randint 序列，得到 (episode_num, start_step)。"""
    rng = random.Random(seed)
    weighted = []
    for episode_num, length in episode_lens:
        if length <= 0:
            continue
        if length < slice_len:
            weighted.append((episode_num, 1))
        else:
            weighted.append((episode_num, length - slice_len + 1))
    population = [episode_num for episode_num, _ in weighted]
    weights = [window_count for _, window_count in weighted]
    length_by_episode = dict(episode_lens)
    episode_num = rng.choices(population, weights=weights, k=1)[0]
    length = length_by_episode[episode_num]
    start_step = 0 if length < slice_len else rng.randint(0, length - slice_len)
    return episode_num, start_step


def test_time_slice_is_seed_deterministic():
    buffer = _buffer(SamplingStrategy.TIME_SLICE, batch_size=3)
    _fill(buffer, 8)
    first = buffer.sample(
        agent_name=AGENT_NAME,
        batch_size=3,
        strategy=SamplingStrategy.TIME_SLICE.value,
        seed=19,
    )
    second = buffer.sample(
        agent_name=AGENT_NAME,
        batch_size=3,
        strategy=SamplingStrategy.TIME_SLICE.value,
        seed=19,
    )
    third = buffer.sample(
        agent_name=AGENT_NAME,
        batch_size=3,
        strategy=SamplingStrategy.TIME_SLICE.value,
        seed=23,
    )
    first_obs = [[float(obs[0]) for obs in item.obs] for item in first]
    second_obs = [[float(obs[0]) for obs in item.obs] for item in second]
    third_obs = [[float(obs[0]) for obs in item.obs] for item in third]
    assert first_obs == second_obs
    assert first_obs != third_obs


def test_time_slice_seed_fixes_start_positions():
    slice_len = 3
    seq_len = 10
    seed = 19
    buffer = _buffer(SamplingStrategy.TIME_SLICE, batch_size=slice_len)
    buffer.push(
        _episode(1, seq_len=seq_len),
        notify_learning=False,
        agent_name=AGENT_NAME,
        episode_num=1,
        sample_cost_secs=0.0,
    )
    expected_episode, expected_start = _expected_time_slice_start(
        [(1, seq_len)],
        slice_len=slice_len,
        seed=seed,
    )
    other_episode, other_start = _expected_time_slice_start(
        [(1, seq_len)],
        slice_len=slice_len,
        seed=23,
    )
    assert expected_episode == 1
    assert other_episode == 1
    assert expected_start != other_start

    batch = buffer.sample(
        agent_name=AGENT_NAME,
        batch_size=slice_len,
        strategy=SamplingStrategy.TIME_SLICE.value,
        seed=seed,
    )
    assert len(batch) == 1
    assert int(batch[0].seq_len) == slice_len
    assert _time_slice_start_step(batch[0]) == expected_start
    assert batch[0].starts_with_episode is (expected_start == 0)

    again = buffer.sample(
        agent_name=AGENT_NAME,
        batch_size=slice_len,
        strategy=SamplingStrategy.TIME_SLICE.value,
        seed=seed,
    )
    assert _time_slice_start_step(again[0]) == expected_start

    other = buffer.sample(
        agent_name=AGENT_NAME,
        batch_size=slice_len,
        strategy=SamplingStrategy.TIME_SLICE.value,
        seed=23,
    )
    assert _time_slice_start_step(other[0]) == other_start


@pytest.mark.parametrize("strategy", list(SamplingStrategy))
def test_versioned_replay_scopes_history_by_strategy(strategy):
    buffer = _buffer(strategy, batch_size=3)
    rolling = strategy in ROLLING_STRATEGIES
    if strategy == SamplingStrategy.SLIDING_WINDOW:
        buffer.config.recent_ratio = 1.0
    try:
        _fill(buffer, 5)
        buffer.set_sample_version("policy-0", {AGENT_NAME: (strategy, 3)})
        state = buffer.state()
        assert (state.versioned_available[AGENT_NAME] >= 3) is rolling
        assert AGENT_NAME not in state.versioned_fresh_agents
        assert not buffer.sample(AGENT_NAME, batch_size=3, strategy=strategy)

        for episode in range(6, 9):
            buffer.push(_episode(episode, seq_len=2), agent_name=AGENT_NAME,
                        episode_num=episode, sample_version="policy-0")
        batch = buffer.sample(AGENT_NAME, batch_size=3, strategy=strategy, seed=19)
        assert batch
        if strategy != SamplingStrategy.UNIFORM_STEPS:
            assert "policy-0" in {getattr(item, "sample_version", None) for item in batch}
        if not rolling:
            assert {item.episode_num for item in batch} <= {6, 7, 8}
        previous_sliding_idx = dict(buffer.sliding_idx)

        buffer.set_sample_version("policy-1", {AGENT_NAME: (strategy, 3)})
        assert buffer.sliding_idx == previous_sliding_idx
        assert not buffer.sample(AGENT_NAME, batch_size=3, strategy=strategy)
        with pytest.raises(ValueError, match="does not match"):
            buffer.push(_episode(9), agent_name=AGENT_NAME, episode_num=9, sample_version="policy-0")
        assert len(buffer.ordered_buffer(AGENT_NAME)) == 8
        buffer.push(_episode(9, seq_len=1), agent_name=AGENT_NAME,
                    episode_num=9, sample_version="policy-1")
        batch = buffer.sample(AGENT_NAME, batch_size=3, strategy=strategy, seed=19)
        if strategy == SamplingStrategy.UNIFORM_STEPS:
            assert len(batch) == 3
        elif rolling:
            assert len({item.episode_num for item in batch}) >= 2
            assert any(item.episode_num == 9 for item in batch)
        elif strategy != SamplingStrategy.TIME_SLICE:
            assert not batch
        else:
            assert [item.episode_num for item in batch] == [9]
    finally:
        buffer.close()


@pytest.mark.parametrize(
    "strategy,workers,expected_batches",
    [
        (SamplingStrategy.ON_POLICY_RECENT, [("S1", "S2"), ("S3",), ("S1",)],
         [[0, 0], [0, 1], [1, 2]]),
        (SamplingStrategy.ON_POLICY_CURSOR, [("S1", "S2"), ("S3", "S1"), ("S2", "S3")],
         [[0, 0], [1, 1], [2, 2]]),
    ],
)
def test_versioned_sampling_follows_worker_requests_and_publications(strategy, workers, expected_batches):
    from config import AgentConfig, AgentType, AlgorithmType, LearnerConfig, TrainConfig
    from scheduler.train_task import TrainTaskManager

    learner = LearnerConfig(name=AGENT_NAME, algo_type=AlgorithmType.PPO.value)
    learner.algo_config.batch_size = 2
    learner.algo_config.sample_strategy = strategy.value
    learner.algo_config.learn_times_per_episode = 1
    manager = TrainTaskManager(
        config=TrainConfig(versioned=True, max_episode=20, save_optimo_models=False),
        policy_agent_configs=[AgentConfig(name=AGENT_NAME, agent_type=AgentType.SB3.value,
                                          agent_names=[AGENT_NAME])],
        learner_configs=[learner],
        supports_versioned_sampling=True,
    )
    buffer = _buffer(strategy, batch_size=2)
    buffer.learn_batches_update_caller = manager.learn_batches_update
    names_by_version = {}

    def claim(workers_in_round, version):
        tickets = [manager.get_sample_task_info(sample_role_name=worker) for worker in workers_in_round]
        assert all(ticket.required_model_versions[AGENT_NAME] == version for ticket in tickets)
        for worker in ("S1", "S2", "S3"):
            assert manager.get_sample_task_info(sample_role_name=worker) is None
        return tickets

    try:
        versions, requests = manager.model_publication_target(learning_batch=False)
        buffer.set_sample_version(versions, requests)
        tickets = claim(workers[0], versions[AGENT_NAME])
        for round_num, expected in enumerate(expected_batches):
            version = versions[AGENT_NAME]
            names_by_version[version] = round_num
            assert manager.get_learn_task_info().learn_info is None
            for ticket in tickets:
                buffer.push(_episode(ticket.episode_num), agent_name=AGENT_NAME,
                            episode_num=ticket.episode_num, sample_version=version)
                manager.report_sample_result(ticket.episode_num, {}, lease_id=ticket.lease_id)
            learn = manager.get_learn_task_info().learn_info
            assert learn is not None
            batch = buffer.sample(AGENT_NAME, batch_size=2, strategy=strategy)
            assert [names_by_version[item.sample_version] for item in batch] == expected

            versions, requests = manager.model_publication_target(learning_batch=True)
            if round_num + 1 < len(workers):
                tickets = claim(workers[round_num + 1], versions[AGENT_NAME])
            manager.report_learning_result({AGENT_NAME: {"loss": 1.0}},
                                           learn.episode_num, learn_id=learn.learn_id)
            buffer.set_sample_version(versions, requests)
            manager.complete_model_publication(versions)
    finally:
        manager.close()
        buffer.close()


def test_versioned_mixed_learners_publish_independently_and_keep_current_partitions(tmp_path):
    from types import SimpleNamespace

    import torch

    from common.exception import ModelVersionNotFound
    from config import AgentConfig, AgentType, AlgorithmType, LearnerConfig, ModelManagerConfig, TrainConfig
    from models.model_manager import ModelManager
    from scheduler.train_task import TrainTaskManager
    from utils.module_utils import publish_learners_to_model_manager

    rolling, cursor = "rolling", "cursor"
    configs = []
    for name, strategy, batch_size in [
        (rolling, SamplingStrategy.ON_POLICY_RECENT, 2),
        (cursor, SamplingStrategy.ON_POLICY_CURSOR, 4),
    ]:
        learner = LearnerConfig(name=name, algo_type=AlgorithmType.PPO.value)
        learner.algo_config.sample_strategy = strategy.value
        learner.algo_config.batch_size = batch_size
        learner.algo_config.learn_times_per_episode = 1
        configs.append(learner)

    config = TrainConfig(versioned=True, max_episode=20, save_optimo_models=False)
    manager = TrainTaskManager(
        config=config, supports_versioned_sampling=True, learner_configs=configs,
        policy_agent_configs=[AgentConfig(name=name, agent_type=AgentType.SB3.value, agent_names=[name])
                              for name in (rolling, cursor)],
    )
    buffer = ReplayBuffer(ReplayBufferConfig(capacity=20))
    buffer.learn_batches_update_caller = manager.learn_batches_update
    models = {name: SimpleNamespace(name=name, algo_type="ppo", models={"agent_net": torch.nn.Linear(1, 1)})
              for name in (rolling, cursor)}
    pool = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path)))

    def publish(names, permit=None):
        if permit is not None:
            manager.report_learning_result(
                {name: {"loss": 1.0} for name in names}, permit.episode_num, learn_id=permit.learn_id,
            )
        publish_learners_to_model_manager(
            [models[name] for name in names], pool, manager.task_id, task_manager=manager,
            train_config=config, is_learning_batch=permit is not None, replay_buffer=buffer,
            learn_info=permit,
        )

    def claim(worker):
        return manager.get_sample_task_info(sample_role_name=worker)

    def finish(ticket):
        buffer.push(
            {name: _episode(ticket.episode_num) for name in (rolling, cursor)},
            episode_num=ticket.episode_num, sample_version=ticket.required_model_versions,
        )
        manager.report_sample_result(ticket.episode_num, {}, lease_id=ticket.lease_id)

    try:
        publish([rolling, cursor])
        first, second, third = (claim(worker) for worker in ("S1", "S2", "S3"))
        initial = first.required_model_versions
        finish(first)
        finish(second)
        learn_rolling = manager.get_learn_task_info().learn_info
        assert learn_rolling.active_learners == [rolling]
        assert [item.episode_num for item in buffer.sample(rolling)] == [1, 2]

        # Cursor still needs more v0 data. A shared rollout can start while
        # rolling's next model is unpublished.
        fourth = claim("S1")
        assert fourth.required_model_versions == initial
        after_rolling, _ = manager.model_publication_target(True)
        assert after_rolling[rolling] != initial[rolling]
        assert after_rolling[cursor] == initial[cursor]
        with pytest.raises(ModelVersionNotFound):
            pool.get_model(rolling, version=after_rolling[rolling], output_type="info")
        publish([rolling], learn_rolling)
        rolling_model = pool.get_model(rolling, version=after_rolling[rolling], output_type="info")
        assert rolling_model.version == after_rolling[rolling]
        assert rolling_model.metadata["sample_episode_num"] == learn_rolling.episode_num
        assert rolling_model.metadata["learn_id"] == learn_rolling.learn_id
        fifth = claim("S2")
        assert fifth.required_model_versions == after_rolling

        finish(third)
        finish(fourth)
        assert [item.episode_num for item in buffer.ordered_buffer(rolling)] == [1, 2]
        learn_cursor = manager.get_learn_task_info().learn_info
        assert learn_cursor.active_learners == [cursor]
        assert [item.episode_num for item in buffer.sample(cursor)] == [1, 2, 3, 4]
        after_cursor, _ = manager.model_publication_target(True)
        assert after_cursor[rolling] == after_rolling[rolling]
        assert after_cursor[cursor] != initial[cursor]
        publish([cursor], learn_cursor)
        cursor_model = pool.get_model(cursor, version=after_cursor[cursor], output_type="info")
        assert cursor_model.metadata["sample_episode_num"] == learn_cursor.episode_num
        assert cursor_model.metadata["learn_id"] == learn_cursor.learn_id
        assert rolling_model.metadata["sample_episode_num"] == learn_rolling.episode_num

        finish(fifth)
        assert [item.episode_num for item in buffer.ordered_buffer(cursor)] == [1, 2, 3, 4]
        assert [item.episode_num for item in buffer.sample(rolling)] == [2, 5]
        assert manager.get_learn_task_info().learn_info.active_learners == [rolling]
    finally:
        manager.close()
        buffer.close()


def test_versioned_rolling_overlap_keeps_late_samples_and_consumes_each_arrival(tmp_path):
    from types import SimpleNamespace

    import torch

    from config import AgentConfig, AgentType, AlgorithmType, LearnerConfig, ModelManagerConfig, TrainConfig
    from models.model_manager import ModelManager
    from scheduler.train_task import TrainTaskManager
    from utils.module_utils import publish_learners_to_model_manager

    strategy = SamplingStrategy.ON_POLICY_RECENT
    learner = LearnerConfig(name=AGENT_NAME, algo_type=AlgorithmType.PPO.value)
    learner.algo_config.sample_strategy = strategy.value
    learner.algo_config.batch_size = 2
    learner.algo_config.learn_times_per_episode = 1
    config = TrainConfig(versioned=True, allow_sample_overlap=True, max_episode=5, save_optimo_models=False)
    manager = TrainTaskManager(
        config=config, supports_versioned_sampling=True, learner_configs=[learner],
        policy_agent_configs=[AgentConfig(name=AGENT_NAME, agent_type=AgentType.SB3.value,
                                          agent_names=[AGENT_NAME])],
    )
    buffer = _buffer(strategy, batch_size=2)
    buffer.learn_batches_update_caller = manager.learn_batches_update
    model = SimpleNamespace(name=AGENT_NAME, algo_type="ppo", models={"agent_net": torch.nn.Linear(1, 1)})
    pool = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path)))

    def publish(permit=None):
        if permit is not None:
            manager.report_learning_result({AGENT_NAME: {"loss": 1.0}},
                                           permit.episode_num, learn_id=permit.learn_id)
        publish_learners_to_model_manager(
            [model], pool, manager.task_id, task_manager=manager, train_config=config,
            is_learning_batch=permit is not None, replay_buffer=buffer,
        )

    def claim(worker):
        return manager.get_sample_task_info(sample_role_name=worker)

    def finish(ticket):
        buffer.push({AGENT_NAME: _episode(ticket.episode_num)}, episode_num=ticket.episode_num,
                    sample_version=ticket.required_model_versions)
        manager.report_sample_result(ticket.episode_num, {}, lease_id=ticket.lease_id)

    def learn(expected_episodes, expected_versions):
        permit = manager.get_learn_task_info().learn_info
        assert permit.active_learners == [AGENT_NAME]
        batch = buffer.sample(AGENT_NAME)
        assert [item.episode_num for item in batch] == expected_episodes
        assert [item.sample_version for item in batch] == expected_versions
        return permit

    try:
        publish()
        first, second, third = (claim(worker) for worker in ("S1", "S2", "S3"))
        v0 = first.required_model_versions[AGENT_NAME]
        assert all(ticket.required_model_versions[AGENT_NAME] == v0 for ticket in (second, third))
        assert claim("S1") is None
        finish(first)
        assert manager.get_learn_task_info().learn_info is None
        finish(second)
        first_update = learn([1, 2], [v0, v0])
        # A free worker keeps sampling during learning with the last published model.
        fourth = claim("S1")
        assert fourth.required_model_versions[AGENT_NAME] == v0
        publish(first_update)
        fifth = claim("S2")
        v1 = fifth.required_model_versions[AGENT_NAME]
        assert v1 != v0

        # Two late rollouts arrive while no learner reads, and one is retried.
        finish(fourth)
        finish(third)
        buffer.push({AGENT_NAME: _episode(third.episode_num)}, episode_num=third.episode_num,
                    sample_version=third.required_model_versions)
        finish(fifth)
        for expected, versions in [
            ([2, 4], [v0, v0]),
            ([4, 3], [v0, v0]),
            ([3, 5], [v0, v1]),
        ]:
            publish(learn(expected, versions))
        assert manager.get_learn_task_info().task_finished
        assert not buffer.sample(AGENT_NAME)
    finally:
        manager.close()
        buffer.close()


@pytest.mark.parametrize("strategy", [SamplingStrategy.ON_POLICY_RECENT_STEPS, SamplingStrategy.SLIDING_WINDOW])
def test_overlap_anchors_each_window_to_its_arrival(strategy):
    batch_size = 3
    buffer = _buffer(strategy, batch_size=batch_size)
    try:
        v0 = "sample_test_g0_v000000"
        v1 = "sample_test_g0_v000001"
        requests = {AGENT_NAME: (strategy, batch_size)}
        overlap_agents = frozenset({AGENT_NAME})
        buffer.set_sample_version(v1, requests, overlap_agents=overlap_agents)
        for episode, steps in enumerate((1, 2, 1, 2), 1):
            buffer.push(_episode(episode, seq_len=steps), agent_name=AGENT_NAME,
                        episode_num=episode, sample_version={AGENT_NAME: v0})
        anchors = (2, 3, 4) if strategy == SamplingStrategy.ON_POLICY_RECENT_STEPS else (3, 4)
        for episode in anchors:
            batch = buffer.sample(AGENT_NAME, seed=4)
            assert max(item.episode_num for item in batch) == episode
            assert all(item.sample_version == v0 for item in batch)
            if strategy == SamplingStrategy.ON_POLICY_RECENT_STEPS:
                assert sum(item.seq_len for item in batch) == batch_size
        assert not buffer.sample(AGENT_NAME)
    finally:
        buffer.close()


def test_overlap_accepts_older_partitions_only_for_rolling_learners():
    rolling, cursor = "rolling", "cursor"
    buffer = ReplayBuffer(ReplayBufferConfig(capacity=8))
    try:
        current = "sample_test_g0_v000001"
        old = "sample_test_g0_v000000"
        requests = {
            rolling: (SamplingStrategy.ON_POLICY_RECENT, 1),
            cursor: (SamplingStrategy.ON_POLICY_CURSOR, 1),
        }
        buffer.set_sample_version(current, requests, overlap_agents=frozenset({rolling}))
        data = {name: _episode(1) for name in requests}
        buffer.push(data, episode_num=1, sample_version={rolling: old, cursor: old})
        assert buffer.state().overlap_ready_agents == {rolling}
        assert buffer.state().versioned_available[cursor] == 0
        assert buffer.sample(rolling)[0].sample_version == old
        assert not buffer.sample(cursor)
        buffer.push(data, episode_num=1, sample_version={rolling: current, cursor: current})
        assert not buffer.sample(rolling)  # The retried partition is already recorded.
        assert buffer.sample(cursor)[0].sample_version == current
    finally:
        buffer.close()

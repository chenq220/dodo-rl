import pytest

from config import SampleLearningTasksConfig
from scheduler.sample_learn_task import SampleLearnManager


def test_should_learn_lifecycle():
    manager = SampleLearnManager(SampleLearningTasksConfig(learn_times=2))
    first = manager.should_learn()
    assert first.learn_info.episode_num == 1
    second = manager.should_learn()
    assert second.learn_info.episode_num == 2
    waiting = manager.should_learn()
    assert waiting.learn_info is None
    assert waiting.task_finished is False
    assert manager.push_learning_state({"loss": 1.0}) == 1
    assert manager.push_learning_state({"loss": 0.5}) == 2
    finished = manager.should_learn()
    assert finished.task_finished is True
    assert manager.is_learning_completed() is True


def test_push_learning_result_does_not_advance_completed():
    manager = SampleLearnManager(SampleLearningTasksConfig(learn_times=1))
    manager.should_learn()
    assert manager.push_learning_result({"x": 1}) == 1
    assert manager.learning_completed == 0
    assert manager.learning_results_count() == 1
    assert manager.get_learning_results() == [{"x": 1}]


def test_sample_cache_load_arbitration_and_wait():
    manager = SampleLearnManager(SampleLearningTasksConfig())
    assert manager.request_sample_cache_load("worker-a") is True
    assert manager.request_sample_cache_load("worker-b") is False
    manager.finish_sample_cache_load(42)
    assert manager.wait_sample_cache_loaded(timeout=0.1) == 42
    state = manager.get_sample_cache_load_state()
    assert state["completed"] is True
    assert state["loaded_episodes"] == 42


def test_sample_cache_load_failure_raises():
    manager = SampleLearnManager(SampleLearningTasksConfig())
    manager.request_sample_cache_load("worker-a")
    manager.fail_sample_cache_load(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        manager.wait_sample_cache_loaded(timeout=0.1)

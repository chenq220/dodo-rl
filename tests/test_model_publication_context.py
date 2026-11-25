from types import SimpleNamespace
from contextlib import contextmanager

import torch

from config import ModelManagerConfig
from models.model_manager import ModelManager
from scheduler.model import LearnInfo
from scheduler.rollout_learn import RolloutLearnWorker
from utils import module_utils


def test_model_publication_records_sample_progress_and_logs_actual_version(tmp_path, monkeypatch):
    pool = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path)))
    learner = SimpleNamespace(name="a", algo_type="maddpg", models={"actor": torch.nn.Linear(1, 1)})
    logged = []
    monkeypatch.setattr(module_utils.LOG, "info", lambda message, *args: logged.append(message % args))
    versions = []
    for learn_id in (1, 2):
        info = LearnInfo(episode_num=101, learn_id=learn_id, stage_name="stage1", generation=3)
        module_utils.publish_learners_to_model_manager(
            [learner], pool, "task", is_learning_batch=True, learn_info=info,
        )
        versions.append(logged[-1].split("model_version=")[1].split(",")[0])
        model = pool.get_model("a", version=versions[-1], output_type="info")
        assert model.metadata["sample_episode_num"] == 101
        assert model.metadata["learn_id"] == learn_id
        assert model.metadata["stage_name"] == "stage1"
        assert model.metadata["generation"] == 3
        assert f"sample_episode_num=101, learn_id={learn_id}" in logged[-1]
    assert versions[0] != versions[1]


def test_worker_clears_publication_context_when_no_learning_task():
    worker = object.__new__(RolloutLearnWorker)
    worker.last_learn_info = LearnInfo(episode_num=5, learn_id=1)
    worker._set_train_mode = lambda: None
    worker.post_episode = lambda **kwargs: None

    @contextmanager
    def no_task():
        yield SimpleNamespace(learn_info=None)

    worker.online_learn_task_context = no_task
    assert worker.rollout_after_episode_updated() == []
    assert worker.last_learn_info is None

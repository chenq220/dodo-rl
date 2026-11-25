import os

from scheduler.tasks.training_progress import (
    LEARN_TOKEN_ENV,
    SAMPLE_TOKEN_ENV,
    TRAINING_EPISODE_ENV,
    TRAINING_PROGRESS_FILE_ENV,
    TRAINING_STAGE_ENV,
    publish_training_progress,
    publish_worker_token,
)


def test_publish_sets_env_and_skips_rewrite_when_unchanged(monkeypatch):
    monkeypatch.delenv(TRAINING_PROGRESS_FILE_ENV, raising=False)
    first = publish_training_progress(stage="train", episode="5", last_published=None)
    assert first == ("train", "5")
    assert os.environ[TRAINING_STAGE_ENV] == "train"
    assert os.environ[TRAINING_EPISODE_ENV] == "5"
    second = publish_training_progress(stage="train", episode="5", last_published=first)
    assert second is first


def test_publish_writes_atomic_file_and_events(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.env"
    monkeypatch.setenv(TRAINING_PROGRESS_FILE_ENV, str(progress_file))
    publish_training_progress(stage="eval", episode="10", last_published=None)
    text = progress_file.read_text(encoding="utf-8")
    assert f"{TRAINING_STAGE_ENV}=eval" in text
    assert f"{TRAINING_EPISODE_ENV}=10" in text
    assert not (tmp_path / "progress.env.events").exists()


def test_publish_worker_token_sets_env_and_merges_progress_file(tmp_path, monkeypatch):
    progress_file = tmp_path / "progress.env"
    monkeypatch.setenv(TRAINING_PROGRESS_FILE_ENV, str(progress_file))
    monkeypatch.delenv(SAMPLE_TOKEN_ENV, raising=False)
    monkeypatch.delenv(LEARN_TOKEN_ENV, raising=False)
    publish_training_progress(stage="search", episode="3/5000", last_published=None)
    publish_worker_token(kind="learn", value="episode=3 stage=search learners=agent_a generation=0")
    text = progress_file.read_text(encoding="utf-8")
    assert f"{TRAINING_STAGE_ENV}=search" in text
    assert f"{TRAINING_EPISODE_ENV}=3/5000" in text
    assert "DODO_LEARN_TOKEN=episode=3 stage=search learners=agent_a generation=0" in text
    publish_training_progress(stage="search", episode="4/5000", last_published=("search", "3/5000"))
    text = progress_file.read_text(encoding="utf-8")
    assert f"{TRAINING_EPISODE_ENV}=4/5000" in text
    assert "DODO_LEARN_TOKEN=episode=3 stage=search learners=agent_a generation=0" in text
    publish_worker_token(kind="sample", value="episode=4 stage=search generation=0")
    text = progress_file.read_text(encoding="utf-8")
    assert "DODO_SAMPLE_TOKEN=episode=4 stage=search generation=0" in text
    publish_worker_token(kind="sample", value="")
    assert os.environ[SAMPLE_TOKEN_ENV] == ""
    assert "DODO_SAMPLE_TOKEN=" in progress_file.read_text(encoding="utf-8")


def test_rollout_sample_worker_exposes_current_token_env(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scheduler.rollout_sample import RolloutSampleWorker

    progress_file = tmp_path / "sample.progress.env"
    monkeypatch.setenv(TRAINING_PROGRESS_FILE_ENV, str(progress_file))
    monkeypatch.delenv(SAMPLE_TOKEN_ENV, raising=False)

    class _SampleTaskManager:
        def is_sample_completed(self):
            return False

        def sample_backpressure_state(self, **_kwargs):
            return {"enabled": False}

        def get_sample_task_info(self, **_kwargs):
            return SimpleNamespace(
                episode_num=7,
                epsilon=0.2,
                temperature=1.0,
                deterministic=False,
                seed=11,
                metadata={},
                stage_name="search",
                generation=1,
                lease_id="lease-7",
            )

        def report_sample_result(self, episode_num, **kwargs):
            return None

    worker = RolloutSampleWorker.__new__(RolloutSampleWorker)
    worker.task_manager = _SampleTaskManager()
    worker.env = SimpleNamespace(stage_name="search")
    monkeypatch.setattr(
        "scheduler.rollout_sample.set_global_seed",
        lambda *args, **kwargs: None,
    )

    with worker.episode_task_context(sample_role_name="sample-0") as episode_info:
        assert episode_info.episode_num == 7
        assert os.environ[SAMPLE_TOKEN_ENV] == "episode=7 stage=search generation=1"
        assert "DODO_SAMPLE_TOKEN=episode=7 stage=search generation=1" in (
            progress_file.read_text(encoding="utf-8")
        )
    assert os.environ[SAMPLE_TOKEN_ENV] == ""
    assert "DODO_SAMPLE_TOKEN=" in progress_file.read_text(encoding="utf-8")


def test_rollout_learn_worker_exposes_current_token_env(tmp_path, monkeypatch):
    from scheduler.model import LearnInfo, LearnTaskResult
    from scheduler.rollout_learn import RolloutLearnWorker
    from scheduler import rollout_learn

    progress_file = tmp_path / "learn.progress.env"
    monkeypatch.setenv(TRAINING_PROGRESS_FILE_ENV, str(progress_file))
    monkeypatch.delenv(LEARN_TOKEN_ENV, raising=False)

    class _LearnTaskManager:
        def get_learn_task_info(self):
            return LearnTaskResult(
                learn_info=LearnInfo(
                    learn_id=3,
                    episode_num=4,
                    stage_name="search",
                    active_learners=["uav_team"],
                    generation=2,
                )
            )

    worker = object.__new__(RolloutLearnWorker)
    worker.learners = []
    worker.task_manager = _LearnTaskManager()
    worker.sample_learn_manager = None
    worker._online_no_task_logged = False
    worker._online_stage_name = "search"
    monkeypatch.setattr(rollout_learn, "set_global_seed", lambda *args, **kwargs: None)

    with worker.online_learn_task_context() as session:
        assert session.learn_info.episode_num == 4
        assert os.environ[LEARN_TOKEN_ENV] == (
            "learn_id=3 sample_watermark=4 stage=search learners=uav_team generation=2"
        )
        assert "DODO_LEARN_TOKEN=learn_id=3 sample_watermark=4 stage=search learners=uav_team generation=2" in (
            progress_file.read_text(encoding="utf-8")
        )
    assert os.environ[LEARN_TOKEN_ENV] == ""

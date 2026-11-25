from pathlib import Path
from unittest.mock import MagicMock

import yaml

from config import AutoMLConfig, S3StorageConfig, StorageType
from scheduler.model import AutoMLAdvice, TrainingRunArchive, TrainingSnapshot
from scheduler.utils.automl_archive_storage import (
    move_artifacts_into_archive,
    persist_archive_bundle,
    write_archive_bundle_contents,
)


def _snapshot():
    return TrainingSnapshot(
        task_id="t1",
        stage_name="search",
        generation=0,
        completed_episodes=2,
        max_episodes=10,
        reward_curve={"uav": [1.0, 2.0]},
        learn_metrics={},
        current_algo_configs={},
        tunable_catalog={},
    )


def test_move_artifacts_into_archive(tmp_path: Path):
    src = tmp_path / "curve.png"
    src.write_bytes(b"png")
    dest_dir = tmp_path / "artifacts"
    moved = move_artifacts_into_archive({"reward_curve": [str(src)]}, str(dest_dir))
    assert len(moved) == 1
    assert Path(moved[0]).exists()
    assert not src.exists()


def test_write_archive_bundle_contents_schema(tmp_path: Path):
    archive_dir = tmp_path / "bundle"
    archive_dir.mkdir()
    bundle = TrainingRunArchive(
        env_name="cooperative_search",
        task_id="t1",
        stage_name="search",
        generation=0,
        archive_seq=1,
        completed_episodes=2,
        episode_rows=[{"episode": 1, "generation": 0, "uav_loss": 0.2}],
        artifact_paths={},
        snapshot=_snapshot(),
        advice=AutoMLAdvice(
            decision="restart",
            algo_config_overrides={"uav_team": {"lr": 0.0001}},
            seed=7,
            reason="optuna_suggest",
        ),
        metric_output_rows={"success": [{"episode": 1, "generation": 0, "value": 1.0}]},
        run_seed=7,
        initialization_seed=7,
    )
    summary = {"metric": {"reward_mean": 1.2}, "keep": 1}
    write_archive_bundle_contents(bundle, str(archive_dir), summary)
    assert (archive_dir / "algos_metrics.csv").exists()
    assert (archive_dir / "success_metrics.csv").exists()
    payload = yaml.safe_load((archive_dir / "advice.yaml").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["overview"]["run_seed"] == 7
    assert payload["advice"]["toml_overrides"]["train_config"]["seed"] == 7
    assert "uav_team" in payload["advice"]["toml_overrides"]["learner_config"]
    assert "metric" not in summary
    assert summary["keep"] == 1


def test_persist_local_returns_abspath(tmp_path: Path):
    archive_dir = tmp_path / "local_bundle"
    archive_dir.mkdir()
    (archive_dir / "note.txt").write_text("ok", encoding="utf-8")
    config = AutoMLConfig(subject_name="demo", storage_type=StorageType.LOCAL.value)
    result = persist_archive_bundle(config, str(archive_dir), "folder")
    assert Path(result) == archive_dir.resolve()
    assert archive_dir.exists()


def test_persist_s3_success_removes_local(tmp_path, monkeypatch):
    archive_dir = tmp_path / "s3_bundle"
    archive_dir.mkdir()
    (archive_dir / "note.txt").write_text("ok", encoding="utf-8")
    client = MagicMock()
    monkeypatch.setattr(
        "scheduler.utils.automl_archive_storage.build_s3_client",
        lambda storage: client,
    )
    monkeypatch.setattr(
        "scheduler.utils.automl_archive_storage.ensure_s3_bucket",
        lambda c, name: None,
    )
    config = AutoMLConfig(
        subject_name="demo",
        storage_type=StorageType.S3.value,
        storage=S3StorageConfig(bucket_name="automl"),
    )
    uri = persist_archive_bundle(config, str(archive_dir), "gen0")
    assert uri.startswith("s3://automl/")
    assert not archive_dir.exists()
    assert client.upload_file.called


def test_persist_s3_failure_keeps_local(tmp_path, monkeypatch):
    archive_dir = tmp_path / "s3_fail"
    archive_dir.mkdir()
    (archive_dir / "note.txt").write_text("ok", encoding="utf-8")
    client = MagicMock()
    client.upload_file.side_effect = RuntimeError("upload failed")
    monkeypatch.setattr(
        "scheduler.utils.automl_archive_storage.build_s3_client",
        lambda storage: client,
    )
    monkeypatch.setattr(
        "scheduler.utils.automl_archive_storage.ensure_s3_bucket",
        lambda c, name: None,
    )
    config = AutoMLConfig(
        subject_name="demo",
        storage_type=StorageType.S3.value,
        storage=S3StorageConfig(bucket_name="automl"),
    )
    result = persist_archive_bundle(config, str(archive_dir), "gen0")
    assert Path(result) == archive_dir.resolve()
    assert archive_dir.exists()

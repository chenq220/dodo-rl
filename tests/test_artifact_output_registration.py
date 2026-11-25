from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

import config as config_module
from scheduler.artifact import ArtifactPut, ArtifactType, ArtifactsManager
from common.exception import ArtifactsArchiveException
from config import (
    ArtifactsManagerConfig,
    EvalConfig,
    LocalStorageConfig,
    ModelManagerConfig,
    MonteCarloMethodConfig,
    ReplayBufferConfig,
    SampleLearningEvalConfig,
    SampleLearningTasksConfig,
    TaskConfig,
    TrainConfig,
)
from sample.monte_carlo import MonteCarloMethod
from scheduler.train_task import TrainTaskManager


def test_config_import_and_construction_do_not_create_directories(tmp_path):
    # 独立进程冷导入也不得通过默认配置实例创建 artifacts 目录。
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(config_module.__file__).parent)
    result = subprocess.run(
        [sys.executable, "-c", "import config; config.TrainConfig(); config.EvalConfig()"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "artifacts").exists()

    source = Path(config_module.__file__).read_text(encoding="utf-8")
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"mkdir", "makedirs"}
        for node in ast.walk(ast.parse(source))
    )

    output = tmp_path / "never-created"
    path = str(output)
    TaskConfig(result_save_dir=path, render_video_dir=path)
    TrainConfig(result_save_dir=path, render_video_dir=path, model_save_dir=path)
    EvalConfig(result_save_dir=path, render_video_dir=path, output_dir=path)
    ModelManagerConfig(tmp_local_dir=path)
    MonteCarloMethodConfig(tmp_local_dir=path)
    ReplayBufferConfig(storage=LocalStorageConfig(data_path=path))
    SampleLearningTasksConfig(
        load_sample_storage=LocalStorageConfig(data_path=path),
        load_model_storage=LocalStorageConfig(data_path=path),
    )
    SampleLearningEvalConfig(result_save_dir=path, model_local_dir=path)
    assert not output.exists()


def test_put_archives_only_the_file_not_sibling_junk(tmp_path):
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    output = tmp_path / "runtime" / "results"
    output.mkdir(parents=True)
    unrelated = output / "other-task.txt"
    unrelated.write_text("untouched")
    source = output / "reward.png"
    source.write_bytes(b"png")
    manager.put("task", "train", ArtifactPut(ArtifactType.RESULT, str(source)))
    archived = manager.archive_task("task", "train")

    assert unrelated.read_text() == "untouched"
    assert not source.exists()
    names = {path.name for path in Path(archived.archive_dir).rglob("*") if path.is_file()}
    assert "reward.png" in names
    assert "other-task.txt" not in names


def test_put_rejects_missing_file_and_does_not_mkdir(tmp_path):
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    missing = tmp_path / "late-output" / "data.csv"
    with pytest.raises(ArtifactsArchiveException, match="does not exist"):
        manager.put("task", "train", ArtifactPut(ArtifactType.RESULT, str(missing)))
    assert not missing.parent.exists()
    assert manager.list_entries("task") == []


@pytest.mark.parametrize("enabled", [True, False])
def test_task_manager_put_then_complete_archives_when_enabled(tmp_path, enabled):
    config = TrainConfig(
        max_episode=0,
        result_save_dir=str(tmp_path / "results"),
        render_video_dir=str(tmp_path / "recordings"),
        model_save_dir=str(tmp_path / "models"),
        artifacts_archive_enabled=enabled,
        artifacts_archive_save_dir=str(tmp_path / "archives"),
    )
    manager = TrainTaskManager(config, [], [], stage_name="train")
    try:
        assert not (tmp_path / "results").exists()
        source = tmp_path / "results" / "report.txt"
        source.parent.mkdir(parents=True)
        source.write_text("report")
        manager.put_runtime_file(ArtifactType.REPORT, str(source))
        manager.notify_task_complete()
        assert source.exists() is not enabled
        assert (tmp_path / "archives").exists() is enabled
    finally:
        manager.close()


def test_sample_writer_prepares_cold_output_and_stage_switch(tmp_path):
    config = MonteCarloMethodConfig(tmp_local_dir=str(tmp_path / "first"))
    assert not (tmp_path / "first").exists()
    writer = MonteCarloMethod("task", config)
    try:
        assert Path(writer.local_file).is_file()
        next_config = MonteCarloMethodConfig(
            load_file_path=str(tmp_path / "second" / "sample.bin"),
        )
        assert not (tmp_path / "second").exists()
        writer.configure(next_config)
        assert Path(writer.local_file).is_file()
    finally:
        writer.close()


def test_read_only_sample_initialization_does_not_register_or_create_input(tmp_path):
    config = MonteCarloMethodConfig(
        tmp_local_dir=str(tmp_path / "input"),
        load_file_path=str(tmp_path / "input" / "missing.bin"),
        write_enabled=False,
    )
    reader = MonteCarloMethod("task", config)
    try:
        assert reader.file is None
        assert not (tmp_path / "input").exists()
    finally:
        reader.close()

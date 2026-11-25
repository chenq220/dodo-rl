"""包一后：入口不再 TOML 预分析；模块 register/put 后可见台账。"""
from __future__ import annotations

import os
import subprocess
import sys
import runpy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scheduler.artifact import (
    ArtifactIntent,
    ArtifactLedgerStatus,
    ArtifactPut,
    ArtifactType,
    ArtifactsManager,
    get_active_artifacts_manager,
)
from config import ArtifactsManagerConfig
from scheduler.train_task import TrainTaskManager
from utils.toml_config import TomlTrainingConfig


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path, *, branch="train", extra=""):
    config_file = tmp_path / "experiment.toml"
    config_file.write_text(f'''
[env_config]
use_sim = false
[train_config]
max_episode = 0
record_sample = true
result_save_dir = "{tmp_path / 'results'}"
model_save_dir = "{tmp_path / 'models'}"
model_load_dir = "{tmp_path / 'input-models'}"
render_video_dir = "{tmp_path / 'recordings'}"
artifacts_archive_save_dir = "{tmp_path / 'archives'}"
[eval_config]
max_episode = 0
record_sample = false
render_gif_enabled = false
result_save_dir = "{tmp_path / 'eval-results'}"
render_video_dir = "{tmp_path / 'eval-recordings'}"
output_dir = "{tmp_path / 'eval-samples'}"
agent_model_path = "{tmp_path / 'input-models'}"
artifacts_archive_save_dir = "{tmp_path / 'archives'}"
[log_config]
log_dir = "{tmp_path / 'logs'}"
enabled_handlers = ["console"]
[monte_carlo_method_config]
tmp_local_dir = "{tmp_path / 'samples'}"
{extra}
''', encoding="utf-8")
    return TomlTrainingConfig(config_file, project_root=tmp_path, branch=branch)


def test_module_register_then_ledger_visible(tmp_path):
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    manager.register("task", "train", ArtifactIntent(ArtifactType.SAMPLE, str(tmp_path / "samples")))
    rows = manager.list_entries("task", "train")
    assert rows
    assert all(row.status is ArtifactLedgerStatus.REGISTERED and row.center_path is None for row in rows)
    assert {row.artifact_type for row in rows} == {ArtifactType.SAMPLE}


def test_put_then_task_manager_archives(tmp_path):
    archives = tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    tm = TrainTaskManager(
        _config(tmp_path).train_config, [], [], artifacts_manager=manager,
    )
    source = tmp_path / "samples" / "trajectory.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"sample")
    manager.put(
        tm.task_id, tm.stage_name,
        ArtifactPut(ArtifactType.SAMPLE, str(source), {"episodes": 1}),
    )
    manager.put(
        tm.task_id, tm.stage_name,
        ArtifactPut(ArtifactType.SAMPLE, str(source), {"complete": True}),
    )
    rows = [
        row for row in manager.list_entries(tm.task_id, tm.stage_name)
        if row.status is ArtifactLedgerStatus.AVAILABLE
    ]
    assert len(rows) == 1
    assert rows[0].metadata["episodes"] == 1
    assert rows[0].metadata["complete"] is True
    try:
        tm.notify_task_complete()
    finally:
        tm.close()
    rows = manager.list_entries(tm.task_id, tm.stage_name)
    sample = next(row for row in rows if row.artifact_type is ArtifactType.SAMPLE)
    assert sample.status is ArtifactLedgerStatus.ARCHIVED
    queried = manager.get_task_artifacts(tm.task_id)
    assert {item.artifact_type for item in queried} == {ArtifactType.SAMPLE}
    assert Path(queried[0].path).read_bytes() == b"sample"
    assert not source.exists()


def test_put_without_prior_register_is_allowed(tmp_path):
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    source = tmp_path / "result.txt"
    source.write_text("result")
    manager.put("ad_hoc", "train", ArtifactPut(ArtifactType.RESULT, str(source)))
    available = [
        row for row in manager.list_entries("ad_hoc", "train")
        if row.status is ArtifactLedgerStatus.AVAILABLE
    ]
    assert len(available) == 1


def test_concurrent_producers_complete_without_losing_records(tmp_path):
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    results = tmp_path / "results"
    results.mkdir()

    def complete(index):
        path = results / f"{index}.txt"
        path.write_text(str(index))
        manager.put("task", "train", ArtifactPut(ArtifactType.RESULT, str(path)))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(complete, range(20)))
    available = [
        row for row in manager.list_entries("task", "train")
        if row.status is ArtifactLedgerStatus.AVAILABLE
    ]
    assert len(available) == 20


def test_importing_proxy_does_not_create_files(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-c", "from scheduler.artifact import artifacts_archive as a; "
         "from scheduler.artifact import artifacts_archive as b; assert a is b"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_every_portal_and_sdk_entry_delegates_runtime_init():
    paths = list((ROOT / "src" / "portal").glob("*.py")) + [ROOT / "sdk/src/dodorl_sdk/portal.py"]
    entry = (ROOT / "src" / "common" / "portal_entry.py").read_text(encoding="utf-8")
    assert "build_artifacts_manager(" in entry
    assert "register_config_artifacts(" not in entry
    assert "log_startup_process(" in entry
    for path in paths:
        if path.name in {"_path.py", "__init__.py"}:
            continue
        source = path.read_text(encoding="utf-8")
        if "load_training_config(" not in source and "initialize_role_runtime(" not in source:
            continue
        assert "initialize_role_runtime(" in source, path
        assert "register_config_artifacts(" not in source, path
        assert "log_startup_process(" not in source, path
    assert not (ROOT / "src" / "roles" / "common" / "artifact_registration.py").exists()
    assert not (ROOT / "src" / "roles" / "common" / "artifact_updates.py").exists()


@pytest.mark.parametrize("entry", ["portal", "sdk"])
def test_entry_activates_manager_before_constructing_role(tmp_path, monkeypatch, entry):
    if os.environ.get("DODO_ARTIFACT_ENTRY_TEST_CHILD") != entry:
        env = dict(os.environ, DODO_ARTIFACT_ENTRY_TEST_CHILD=entry)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-c", str(ROOT / "pyproject.toml"),
             "--rootdir", str(ROOT), f"{__file__}::test_entry_activates_manager_before_constructing_role[{entry}]"],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return
    conf = _config(tmp_path)
    calls = []

    class Role:
        def __init__(self, **kwargs):
            assert get_active_artifacts_manager() is not None
            calls.append("role-created")

        def run(self):
            calls.append("role-run")

    if entry == "portal":
        monkeypatch.setattr("common.portal_entry.load_training_config", lambda *args, **kwargs: conf)
        monkeypatch.setattr("common.portal_entry.parse_config_args", lambda *args: None)
        monkeypatch.setattr("roles.serial_train.SerialTrainRole", Role)
        runpy.run_module("portal.serial_train", run_name="__main__")
    else:
        monkeypatch.syspath_prepend(str(ROOT / "sdk/src"))
        from dodorl_sdk import portal

        monkeypatch.setattr(portal, "get_project_root", lambda: tmp_path)
        monkeypatch.setattr(portal.SDKPluginRegistry, "activate", lambda *args: None)
        monkeypatch.setattr(portal.SDKPluginRegistry, "require_environment", lambda *args: None)
        monkeypatch.setattr(portal, "SerialTrainRole", Role)
        sdk = portal.SerialTrainSDK(config_file=conf.config_file)
        assert not calls
        sdk.run()
    assert calls == ["role-created", "role-run"]

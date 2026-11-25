from pathlib import Path
from argparse import Namespace

import pytest

from common.info_bridge import INFO_IS_MASTER, info_bridge
from common.plugins import RUNTIME_PLUGINS
from common.portal_entry import (
    initialize_role_runtime,
    load_training_config,
    parse_config_args,
    resolve_config_file,
    resolve_role_id,
)


ROOT = Path(__file__).resolve().parents[1]


def test_resolve_config_file_from_subjects_relative():
    path = resolve_config_file("cooperative_search/cooperative_search_qmix.toml", ROOT)
    assert path.is_file()
    assert path.name == "cooperative_search_qmix.toml"


def test_resolve_config_file_missing_lists_tried_paths(tmp_path: Path):
    with pytest.raises(FileNotFoundError) as exc:
        resolve_config_file("missing.toml", tmp_path)
    assert "Tried:" in str(exc.value)


def test_parse_config_args_requires_file(monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog"])
    with pytest.raises(SystemExit):
        parse_config_args("demo")


def test_load_training_config_is_pure_and_role_id(monkeypatch):
    monkeypatch.chdir(ROOT)
    args = Namespace(config_file=str(ROOT / "subjects/cooperative_search/cooperative_search_qmix.toml"))
    before_plugins = list(RUNTIME_PLUGINS.active_plugins)
    conf = load_training_config(args, ROOT)
    assert conf.stage_task_config.stages
    assert info_bridge.get("toml_config", None) is None
    assert RUNTIME_PLUGINS.active_plugins == before_plugins
    role_id = resolve_role_id(conf, "learn-control-0")
    assert role_id.endswith("learn-control-0") or "cooperative_search" in role_id


def test_initialize_role_runtime_activates_plugins_and_puts_is_master(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr("common.portal_entry.task_id.freeze", lambda: "frozen")
    monkeypatch.setattr("common.portal_entry.log_startup_process", lambda *args, **kwargs: None)
    args = Namespace(config_file=str(ROOT / "subjects/cooperative_search/cooperative_search_qmix.toml"))
    conf = load_training_config(args, ROOT)
    initialize_role_runtime(conf, role="serial_train", is_master=None)
    assert info_bridge.get(INFO_IS_MASTER) is None
    packed = info_bridge.get("toml_config")
    assert packed["env_config"] is conf.env_config
    assert packed["stage_task_config"] is conf.stage_task_config
    assert packed["train_config"] is conf.train_config
    assert RUNTIME_PLUGINS.active_plugins == [
        "environment.cooperative_search",
        "agent.rnn",
        "learner.qmix",
    ]


def test_initialize_role_runtime_skips_artifact_shells_for_worker(monkeypatch, tmp_path):
    from scheduler.artifact import artifacts_archive
    from utils.task_id import task_id
    from utils.toml_config import TomlTrainingConfig

    monkeypatch.setattr("common.portal_entry.task_id.freeze", lambda: "frozen")
    monkeypatch.setattr("common.portal_entry.log_startup_process", lambda *args, **kwargs: None)
    config_file = tmp_path / "worker.toml"
    config_file.write_text(f'''
[env_config]
use_sim = false
[design]
task_id = "worker-task"
[train_config]
max_episode = 0
result_save_dir = "{tmp_path / 'results'}"
model_save_dir = "{tmp_path / 'models'}"
artifacts_archive_save_dir = "{tmp_path / 'archives'}"
[log_config]
log_dir = "{tmp_path / 'logs'}"
enabled_handlers = ["console"]
''', encoding="utf-8")
    conf = TomlTrainingConfig(config_file, project_root=tmp_path)
    initialize_role_runtime(
        conf,
        role="sample",
        is_master=False,
        activate_portal_plugins=False,
    )
    assert info_bridge.get(INFO_IS_MASTER) is False
    assert str(task_id) not in artifacts_archive._tasks
    assert not (tmp_path / "logs").exists()

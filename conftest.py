"""pytest 运行时：测试只用 console 日志，并清理项目目录下的运行产物。"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

_ARTIFACT_ROOT_NAMES = ("logs", "artifacts")
_artifact_state: dict[str, dict] = {}


def _snapshot_tree(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path.relative_to(root) for path in root.rglob("*")}


def _cleanup_new_paths(root: Path, existed_before: bool, before: set[Path]) -> None:
    if not root.exists():
        return
    if not existed_before:
        shutil.rmtree(root)
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        relative = path.relative_to(root)
        if relative in before:
            continue
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def pytest_configure(config) -> None:
    cwd = Path.cwd()
    for name in _ARTIFACT_ROOT_NAMES:
        root = cwd / name
        _artifact_state[name] = {
            "existed": root.exists(),
            "paths": _snapshot_tree(root),
        }

    os.environ["DODORL_TEST_LOG_HANDLERS"] = "console"
    from common.logger import configure_logger, set_logger_role_id
    from config import LogConfig

    set_logger_role_id(None)
    configure_logger(LogConfig(enabled_handlers=["console"]))


@pytest.fixture(autouse=True)
def _isolate_artifacts_archive():
    from common.logger import set_logger_role_id
    from scheduler.artifact import _ArchiveStorage, artifacts_archive
    from scheduler.artifact import reset_active_artifacts_manager_for_tests

    # 每个用例激活新的 ArtifactsManager，并清空跨用例的路径所有权。
    # 顺带清掉兼容 proxy 上被 setattr 的方法，避免污染后续用例。
    reset_active_artifacts_manager_for_tests()
    set_logger_role_id(None)
    _ArchiveStorage._source_owners = {}
    artifacts_archive.__dict__.clear()
    yield
    reset_active_artifacts_manager_for_tests()
    set_logger_role_id(None)
    _ArchiveStorage._source_owners = {}
    artifacts_archive.__dict__.clear()


@pytest.fixture(autouse=True)
def _isolate_info_bridge():
    from common.info_bridge import info_bridge

    with info_bridge.scope():
        yield


@pytest.fixture(autouse=True)
def _isolate_progress_env(monkeypatch):
    # 进度/收口改走进程环境变量后，上一用例的 DODO_CENTER_FINISHED 会串到后面。
    for key in (
        "DODO_CENTER_FINISHED",
        "DODO_TRAINING_STAGE",
        "DODO_TRAINING_EPISODE",
        "DODO_SAMPLE_TOKEN",
        "DODO_LEARN_TOKEN",
        "DODO_PROGRESS_FILE",
        "DODO_TRAINING_MAX_EPISODE",
        "DODO_TRAINING_ETA_MIN",
    ):
        monkeypatch.delenv(key, raising=False)


def pytest_sessionfinish(session, exitstatus) -> None:
    cwd = Path.cwd()
    for name, state in _artifact_state.items():
        _cleanup_new_paths(cwd / name, state["existed"], state["paths"])

"""日志模块：enabled_handlers 控制 console/file，file 目录按需创建。"""
from __future__ import annotations

from pathlib import Path

import pytest

from common.logger import CONF, configure_logger, init_logger, log_startup_process, set_logger_role_id
from config import LogConfig, LogHandlerKind


@pytest.fixture(autouse=True)
def _restore_console_logger():
    yield
    set_logger_role_id(None)
    configure_logger(LogConfig(enabled_handlers=[LogHandlerKind.CONSOLE.value]))


def test_console_only_log_config_does_not_create_log_dir(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    LogConfig(log_dir=str(log_dir), enabled_handlers=[LogHandlerKind.CONSOLE.value])
    assert not log_dir.exists()


def test_file_handler_creates_log_dir_and_writes_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    configure_logger(LogConfig(
        log_dir=str(log_dir),
        log_file_name="unit.log",
        enabled_handlers=[LogHandlerKind.FILE.value],
    ))
    logger = init_logger("test_logger_file")
    logger.info("hello-file")
    from utils.task_id import task_id

    log_file = log_dir / f"unit_{task_id}.log"
    assert log_file.is_file()
    assert "hello-file" in log_file.read_text(encoding="utf-8")


def test_file_handler_prefixes_role_id_when_injected(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    set_logger_role_id("sample-0")
    configure_logger(LogConfig(
        log_dir=str(log_dir),
        log_file_name="unit.log",
        enabled_handlers=[LogHandlerKind.FILE.value],
    ))
    logger = init_logger("test_logger_role_prefix")
    logger.info("hello-role")
    from utils.task_id import task_id

    log_file = log_dir / f"sample-0_unit_{task_id}.log"
    assert log_file.is_file()
    assert "hello-role" in log_file.read_text(encoding="utf-8")
    assert not (log_dir / f"unit_{task_id}.log").exists()


def test_pytest_disables_file_handler_under_project_logs() -> None:
    assert LogHandlerKind.FILE.value not in CONF.enabled_handlers
    config = LogConfig(enabled_handlers=[LogHandlerKind.FILE.value])
    assert config.enabled_handlers == [LogHandlerKind.CONSOLE.value]
    log_dir = Path.cwd() / "logs"
    before = {path: path.stat().st_mtime_ns for path in log_dir.glob("*.log")} if log_dir.exists() else {}
    init_logger("pytest_no_file_log").info("should stay on console")
    after = {path: path.stat().st_mtime_ns for path in log_dir.glob("*.log")} if log_dir.exists() else {}
    assert after == before


def test_toml_file_log_config_stays_on_console_under_project_dir() -> None:
    from utils.toml_config import TomlTrainingConfig

    log_file = Path.cwd() / "logs" / "cooperative_search.log"
    before_mtime = log_file.stat().st_mtime_ns if log_file.exists() else None
    root = Path(__file__).resolve().parents[1]
    TomlTrainingConfig(
        root / "subjects/cooperative_search/cooperative_search_qmix.toml",
        project_root=root,
    )
    assert LogHandlerKind.FILE.value not in CONF.enabled_handlers
    if before_mtime is None:
        assert not log_file.exists()
    else:
        assert log_file.stat().st_mtime_ns == before_mtime


def test_startup_process_log_has_stable_field_order_and_optional_role_id(
        tmp_path: Path,
        monkeypatch,
) -> None:
    import common.logger as logger_module
    from utils.task_id import task_id

    messages = []

    class _Recorder:
        def info(self, message: str) -> None:
            messages.append(message)

    monkeypatch.setattr(logger_module, "_STARTUP_PROCESS_LOG", _Recorder())
    config_file = tmp_path / "role.toml"

    log_startup_process("serial_train", config_file)
    log_startup_process("sample", config_file, "sample-2")

    expected_prefix = (
        f"Startup process: role=serial_train config_file={config_file.resolve()} "
        f"task_id={task_id}"
    )
    assert messages == [
        expected_prefix,
        (
            f"Startup process: role=sample config_file={config_file.resolve()} "
            f"task_id={task_id} role_id=sample-2"
        ),
    ]

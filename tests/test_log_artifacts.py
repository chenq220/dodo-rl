"""日志在首次写入、轮转及阶段结束时遵守统一产物契约。"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from common import logger as logger_module
from scheduler.artifact import (
    Artifact,
    ArtifactLedgerStatus,
    ArtifactStatus,
    ArtifactType,
    _ArchiveStorage,
    artifacts_archive,
)
from common.exception import ArtifactsArchiveException
from config import EvalConfig, LogConfig
from utils.task_id import task_id


LOG_ROLE = "tester"


@pytest.fixture(autouse=True)
def restore_logger(_isolate_artifacts_archive):
    yield
    logger_module.set_logger_role_id(None)
    logger_module.configure_logger(LogConfig(enabled_handlers=["console"]))


def _logger(tmp_path):
    logger_module.set_logger_role_id(LOG_ROLE)
    logger_module.configure_logger(LogConfig(
        log_dir=str(tmp_path / "logs"), log_file_name="run.log", enabled_handlers=["file"],
    ))
    return logger_module.init_logger("test_log_artifacts")


def _plan(tmp_path, *stages):
    for stage in stages or ("train",):
        artifacts_archive.register_task(
            str(task_id), stage, archive_save_dir=str(tmp_path / "archives"),
            allowed_runtime_dirs=[str(tmp_path / "logs")], task_type="training",
        )
        artifacts_archive.register(str(task_id), stage, Artifact(ArtifactType.LOG, str(tmp_path / "logs")))


def _ingest(stage="train"):
    return artifacts_archive.put_log(str(task_id), LOG_ROLE, stage)


def _row(stage="train"):
    rows = artifacts_archive.get_registered_artifacts(str(task_id), stage)
    return next(row for row in rows if row.artifact_type is ArtifactType.LOG)


def _available(stage="train"):
    return [
        item for item in artifacts_archive.list_entries(str(task_id), stage)
        if item.artifact_type is ArtifactType.LOG
        and item.status is ArtifactLedgerStatus.AVAILABLE
    ]


def test_confirmed_runtime_cleanup_does_not_recreate_log_directory(tmp_path):
    from config import ArtifactsManagerConfig
    from scheduler.artifact import ArtifactsManager

    root = tmp_path / str(task_id)
    logger_module.set_logger_role_id(LOG_ROLE)
    logger_module.configure_logger(LogConfig(
        log_dir=str(root / "logs"), enabled_handlers=["file"],
    ))
    log = logger_module.init_logger("test_cleanup_logs")
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    manager.bind_task_runtime_directory(str(task_id), root)
    log.info("before archive")
    manager.put_log(str(task_id), LOG_ROLE, "train")
    manager.archive_task(str(task_id), "train")
    manager.close()
    log.info("resource shutdown after archive")
    assert not root.exists()
    logs = list((tmp_path / "archives" / str(task_id) / "logs").rglob("*.log"))
    assert logs and "before archive" in logs[0].read_text()


def test_first_file_before_entry_registration_is_archived_without_rotation(tmp_path):
    log = _logger(tmp_path)
    assert not (tmp_path / "logs").exists()  # delay=True，不生成空文件。
    log.info("startup-before-registration")
    source = Path(logger_module._FILE_HANDLER.baseFilename)
    assert source.name == f"{LOG_ROLE}_run_{task_id}.log"
    _plan(tmp_path)
    assert _row().status is ArtifactStatus.PLANNED
    assert not _row().artifacts
    entries = _ingest()
    assert entries
    assert all(item.status is ArtifactLedgerStatus.AVAILABLE for item in entries)
    assert source.exists()

    result = artifacts_archive.archive_task(str(task_id), "train")
    archived, = artifacts_archive.get_task_artifacts(str(task_id))
    snapshot = Path(archived.path).read_bytes()
    assert b"startup-before-registration" in snapshot
    assert source.exists()
    log.info("after-stage-archive")
    assert b"after-stage-archive" in source.read_bytes()
    assert Path(archived.path).read_bytes() == snapshot
    ledger_archived = [
        item for item in artifacts_archive.list_entries(str(task_id), "train")
        if item.status is ArtifactLedgerStatus.ARCHIVED
    ]
    assert ledger_archived
    assert artifacts_archive.archive_task(str(task_id), "train") == result


def test_rotations_keep_more_than_five_shards_and_archive_every_file(tmp_path):
    log = _logger(tmp_path)
    _plan(tmp_path)
    logger_module._FILE_HANDLER.maxBytes = 180
    for index in range(12):
        log.info(f"unique-message-{index:02d}-" + "x" * 80)
    assert not _available()
    _ingest()
    sources = logger_module.resolve_role_log_files(LOG_ROLE, str(task_id))
    assert len(sources) >= 12
    assert len(set(sources)) == len(sources)
    assert all(path.is_file() for path in sources)
    active = Path(logger_module._FILE_HANDLER.baseFilename)
    artifacts_archive.archive_task(str(task_id), "train")
    archived = artifacts_archive.get_task_artifacts(str(task_id))
    assert len(archived) == len(sources)
    text = "".join(Path(item.path).read_text() for item in archived)
    for index in range(12):
        assert text.count(f"unique-message-{index:02d}-") == 1
    assert active.exists()
    assert all(path.exists() for path in sources)
    log.info("logging-still-works")


def test_multistage_logs_snapshot_shared_file_without_removing_it(tmp_path):
    log = _logger(tmp_path)
    _plan(tmp_path, "stage-a", "stage-b")
    log.info("stage-a-message")
    source = Path(logger_module._FILE_HANDLER.baseFilename)
    _ingest("stage-a")
    artifacts_archive.archive_task(str(task_id), "stage-a")
    first, = artifacts_archive.get_task_artifacts(str(task_id))
    first_bytes = Path(first.path).read_bytes()
    log.info("stage-b-message")
    _ingest("stage-b")
    logger_module.configure_logger(LogConfig(enabled_handlers=["console"]))
    artifacts_archive.archive_task(str(task_id), "stage-b")
    assert source.exists()
    assert Path(first.path).read_bytes() == first_bytes
    entries = json.loads((tmp_path / "archives" / str(task_id) / "manifest.json").read_text())["artifacts"]
    assert {item["stage"] for item in entries} == {"stage-a", "stage-b"}
    for item in entries:
        contents = (tmp_path / "archives" / str(task_id) / item["relative_path"]).read_bytes()
        assert hashlib.sha256(contents).hexdigest() == item["sha256"]
        assert (b"stage-b-message" in contents) == (item["stage"] == "stage-b")


def test_disabled_archive_keeps_logs_and_console_only_has_no_file(tmp_path):
    from scheduler.train_task import TrainTaskManager

    _plan(tmp_path)
    assert _row().status is ArtifactStatus.PLANNED
    assert not _row().artifacts
    log = _logger(tmp_path)
    log.info("keep-runtime-log")
    manager = TrainTaskManager(EvalConfig(artifacts_archive_enabled=False), [], [])
    try:
        assert manager.notify_task_complete() is None
    finally:
        manager.close()
    assert not (tmp_path / "archives").exists()
    assert Path(logger_module._FILE_HANDLER.baseFilename).exists()


def test_task_manager_normal_completion_includes_log_snapshot(tmp_path):
    from scheduler.train_task import TrainTaskManager

    log = _logger(tmp_path)
    _plan(tmp_path)
    manager = TrainTaskManager(EvalConfig(artifacts_archive_save_dir=str(tmp_path / "archives")), [], [])
    try:
        log.info("before-task-complete")
        _ingest(manager.stage_name)
        result = manager.notify_task_complete()
        assert result is not None
        archived, = artifacts_archive.get_task_artifacts(str(task_id))
        assert archived.artifact_type is ArtifactType.LOG
        assert "before-task-complete" in Path(archived.path).read_text()
        current = Path(logger_module._FILE_HANDLER.baseFilename)
        assert "Task artifacts archived:" in current.read_text()
        assert manager.notify_task_complete() == result
    finally:
        manager.close()


def test_reconfiguring_logger_appends_to_same_task_file(tmp_path):
    log = _logger(tmp_path)
    _plan(tmp_path)
    log.info("before-reconfigure")
    first = Path(logger_module._FILE_HANDLER.baseFilename)
    _logger(tmp_path)
    log.info("after-reconfigure")
    assert Path(logger_module._FILE_HANDLER.baseFilename) == first
    _ingest()
    assert len(_available()) == 1
    artifacts_archive.archive_task(str(task_id), "train")
    assert first.exists()
    archived, = artifacts_archive.get_task_artifacts(str(task_id))
    content = Path(archived.path).read_text()
    assert "before-reconfigure" in content and "after-reconfigure" in content


def test_archive_failure_preserves_closed_and_active_logs_for_retry(tmp_path, monkeypatch):
    log = _logger(tmp_path)
    _plan(tmp_path)
    log.info("first-part")
    logger_module._FILE_HANDLER.doRollover()
    log.info("second-part")
    _ingest()
    sources = logger_module.resolve_role_log_files(LOG_ROLE, str(task_id))
    with monkeypatch.context() as patch:
        def fail(*args):
            raise OSError("manifest-write-failed")
        patch.setattr(_ArchiveStorage, "_write_manifest", fail)
        with pytest.raises(ArtifactsArchiveException, match="manifest-write-failed"):
            artifacts_archive.archive_task(str(task_id), "train")
    assert all(path.exists() for path in sources)
    artifacts_archive.archive_task(str(task_id), "train")
    assert len(artifacts_archive.get_task_artifacts(str(task_id))) == 2


def test_snapshot_does_not_chase_concurrent_appends(tmp_path, monkeypatch):
    source = tmp_path / "source.log"
    source.write_bytes(b"original")
    original_fstat = os.fstat
    def fstat_then_append(fd):
        stat = original_fstat(fd)
        with source.open("ab") as writer:
            writer.write(b"late-message")
        return stat
    monkeypatch.setattr(os, "fstat", fstat_then_append)
    destination = tmp_path / "snapshot.log"
    _ArchiveStorage._copy_log_snapshot(source, destination)
    assert destination.read_bytes() == b"original"
    assert source.read_bytes() == b"originallate-message"


def test_log_contract_rejects_cross_task_reuse_and_does_not_scan_directory(tmp_path):
    source = tmp_path / "logs" / "mine.log"
    _plan(tmp_path)
    source.write_text("mine")
    unrelated = source.with_name("unrelated.log")
    unrelated.write_text("other-task")
    artifact = Artifact(ArtifactType.LOG, str(source), {"closed": True})
    artifacts_archive.record_log(str(task_id), artifact)
    with pytest.raises(ArtifactsArchiveException, match="another task"):
        artifacts_archive.record_log("another-task", artifact)
    with pytest.raises(ArtifactsArchiveException, match="expected ArtifactType.LOG"):
        artifacts_archive.record_log(str(task_id), Artifact(ArtifactType.RESULT, str(source)))
    artifacts_archive.archive_task(str(task_id), "train")
    assert unrelated.read_text() == "other-task"
    assert len(artifacts_archive.get_task_artifacts(str(task_id))) == 1


def test_record_log_rejects_symlink_escape_after_entry_registration(tmp_path):
    _plan(tmp_path)
    outside = tmp_path / "outside.log"
    outside.write_text("do-not-archive")
    link = tmp_path / "logs" / "escaped.log"
    link.symlink_to(outside)
    with pytest.raises(ArtifactsArchiveException, match="outside (registered|allowed)"):
        artifacts_archive.record_log(str(task_id), Artifact(ArtifactType.LOG, str(link)))
    assert outside.read_text() == "do-not-archive"


def test_put_log_rejects_symlink_escape_outside_log_dir(tmp_path):
    _logger(tmp_path)
    _plan(tmp_path)
    outside = tmp_path / "outside.log"
    outside.write_text("do-not-archive")
    link = logger_module.log_file_path(LOG_ROLE, str(task_id))
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(outside)
    with pytest.raises(ArtifactsArchiveException, match="outside log_dir"):
        artifacts_archive.put_log(str(task_id), LOG_ROLE, "train")
    assert outside.read_text() == "do-not-archive"
    assert link.is_symlink()


def test_task_id_changes_isolate_tasks_but_processes_share_one_file(tmp_path):
    # 身份配置在独立子进程验证，不污染 pytest 的 task_id 单例。
    root = Path(__file__).resolve().parents[1]
    code = '''
from pathlib import Path
from common.logger import configure_logger, init_logger
from config import LogConfig
from utils.task_id import task_id
configure_logger(LogConfig(log_dir=str(Path.cwd() / "logs"), enabled_handlers=["file"]))
log = init_logger("identity-check")
task_id.configure("before-toml")
log.info("only-before")
task_id.configure("final-task")
log.info("only-final")
configure_logger(LogConfig(enabled_handlers=["console"]))
'''
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    env.pop("DODORL_TEST_LOG_HANDLERS", None)
    processes = [subprocess.Popen([sys.executable, "-c", code], cwd=tmp_path, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
    files = list((tmp_path / "logs").glob("*.log"))
    assert len(files) == 2
    for path in files:
        content = path.read_text()
        assert ("only-before" in content) == ("before-toml" in path.name)
        assert ("only-final" in content) == ("final-task" in path.name)
        assert content.count("only-before") + content.count("only-final") == 2


def _processes(tmp_path, code, count):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"), DODO_TASK_ID=str(task_id))
    env.pop("DODORL_TEST_LOG_HANDLERS", None)
    processes = [subprocess.Popen([sys.executable, "-c", code, str(index)], cwd=tmp_path, env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for index in range(count)]
    for process in processes:
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, stdout + stderr
        assert "Logging error" not in stderr, stderr


def test_concurrent_writers_rotate_and_each_archiver_gets_all_parts(tmp_path):
    log = _logger(tmp_path)
    _plan(tmp_path)
    log.info("parent-started")
    _processes(tmp_path, '''
import sys
from pathlib import Path
from common import logger
from config import LogConfig
logger.set_logger_role_id("tester")
logger.configure_logger(LogConfig(log_dir=str(Path.cwd() / "logs"), log_file_name="run.log",
                                  file_log_format="%(message)s", enabled_handlers=["file"]))
writer = logger.init_logger("writer")
logger._FILE_HANDLER.maxBytes = 1024
for index in range(40):
    writer.info(f"writer:{sys.argv[1]}:{index}:" + "中文内容" * 12)
logger.configure_logger(LogConfig(enabled_handlers=["console"]))
''', 3)
    sources = logger_module.resolve_role_log_files(LOG_ROLE, str(task_id))
    assert len(sources) > 5
    lines = [line for path in sources for line in path.read_text().splitlines() if line.startswith("writer:")]
    assert sorted(lines) == sorted(f"writer:{worker}:{index}:" + "中文内容" * 12
                                  for worker in range(3) for index in range(40))
    _ingest()
    artifacts_archive.archive_task(str(task_id), "train")
    archived = artifacts_archive.get_task_artifacts(str(task_id))
    assert len(archived) == len(sources)
    actual = [line for item in archived for line in Path(item.path).read_text().splitlines() if line.startswith("writer:")]
    assert sorted(actual) == sorted(lines)
    assert all(path.exists() for path in sources)
    _plan(tmp_path, "next-stage")
    log.info("after-shared-archive")
    _ingest("next-stage")
    artifacts_archive.archive_task(str(task_id), "next-stage")
    manifest = json.loads((tmp_path / "archives" / str(task_id) / "manifest.json").read_text())
    assert set(manifest["stages"]) == {"train", "next-stage"}


def test_emit_does_not_ingest_until_put_log(tmp_path):
    log = _logger(tmp_path)
    _plan(tmp_path)
    for index in range(20):
        log.info(f"line-{index}")
    assert _available() == []
    source = Path(logger_module._FILE_HANDLER.baseFilename)
    assert source.is_file()
    entries = _ingest()
    assert entries
    assert all(item.status is ArtifactLedgerStatus.AVAILABLE for item in entries)
    assert source.exists()
    cache = Path(entries[0].center_path)
    assert cache.is_file()
    assert cache.resolve() != source.resolve()
    assert "line-0" in cache.read_text()


def test_two_role_ids_keep_separate_cache_copies(tmp_path):
    _plan(tmp_path)
    logger_module.set_logger_role_id("alpha")
    logger_module.configure_logger(LogConfig(
        log_dir=str(tmp_path / "logs"), log_file_name="run.log", enabled_handlers=["file"],
    ))
    logger_module.init_logger("alpha-log").info("from-alpha")
    alpha_source = Path(logger_module._FILE_HANDLER.baseFilename)
    logger_module.set_logger_role_id("beta")
    logger_module.configure_logger(LogConfig(
        log_dir=str(tmp_path / "logs"), log_file_name="run.log", enabled_handlers=["file"],
    ))
    logger_module.init_logger("beta-log").info("from-beta")
    beta_source = Path(logger_module._FILE_HANDLER.baseFilename)
    assert alpha_source.name.startswith("alpha_")
    assert beta_source.name.startswith("beta_")
    assert alpha_source != beta_source
    artifacts_archive.put_log(str(task_id), "alpha", "train")
    artifacts_archive.put_log(str(task_id), "beta", "train")
    available = _available()
    by_role = {item.metadata["role_id"]: item for item in available}
    assert set(by_role) == {"alpha", "beta"}
    alpha_text = Path(by_role["alpha"].center_path).read_text()
    beta_text = Path(by_role["beta"].center_path).read_text()
    assert "from-alpha" in alpha_text and "from-beta" not in alpha_text
    assert "from-beta" in beta_text and "from-alpha" not in beta_text
    assert Path(by_role["alpha"].center_path).parent.name == "alpha"
    assert Path(by_role["beta"].center_path).parent.name == "beta"
    assert alpha_source.exists() and beta_source.exists()


def test_put_log_without_complete_blocks_archive(tmp_path):
    import threading
    import time

    log = _logger(tmp_path)
    _plan(tmp_path)
    log.info("waiting-for-complete")
    artifacts_archive.expect_producer(str(task_id), "train", f"log:{LOG_ROLE}")
    _ingest()
    started = threading.Event()
    finished: list[object] = []

    def _archive():
        started.set()
        finished.append(artifacts_archive.archive_task(str(task_id), "train"))

    worker = threading.Thread(target=_archive)
    worker.start()
    assert started.wait(1)
    time.sleep(0.05)
    assert worker.is_alive()
    artifacts_archive.complete(str(task_id), "train", f"log:{LOG_ROLE}")
    worker.join(2)
    assert not worker.is_alive()
    assert finished
    assert (tmp_path / "archives" / str(task_id) / "manifest.json").is_file()

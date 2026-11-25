"""ArtifactsManager 包一台账契约与本机 put/归档单测（§3.7 A–G）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scheduler.artifact import ArtifactType
from common.exception import ArtifactsArchiveException
from config import ArtifactsManagerConfig, TrainConfig
from scheduler.artifact import (
    REMOTE_CACHE_DIRNAME,
    ArtifactCallerSupport,
    ArtifactIntent,
    ArtifactLedgerStatus,
    ArtifactOrigin,
    ArtifactPut,
    ArtifactsManager,
    LedgerEntry,
    build_artifacts_manager,
    get_active_artifacts_manager,
    resolve_artifacts_manager_config,
)
from scheduler.train_task import TrainTaskManager


def test_register_intent_is_not_file_list(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    manager.register("t1", "train", ArtifactIntent(ArtifactType.RESULT, planned_hint=str(tmp_path / "out")))
    rows = manager.list_entries("t1", "train")
    assert len(rows) == 1
    assert rows[0].status is ArtifactLedgerStatus.REGISTERED
    assert rows[0].center_path is None
    assert [r for r in rows if r.status is ArtifactLedgerStatus.AVAILABLE] == []


def test_close_removes_confirmed_task_runtime_tree(tmp_path: Path) -> None:
    root = tmp_path / "artifacts" / "task"
    archives = root.parent / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    manager.bind_task_runtime_directory("task", root)
    for directory in ("models", "results", "samples", "logs", "processes/serial_train-main"):
        (root / directory).mkdir(parents=True)
    (root / "logs/.task.log.lock").touch()
    (root / "logs/.task.log.parts.json").write_text("[]")
    (root / "logs/task.log").write_text("log snapshot source")
    (root / "processes/serial_train-main/manifest.json").write_text("{}")
    source = root / "results/reward.png"
    source.write_bytes(b"reward")
    manager.put("task", "train", ArtifactPut(ArtifactType.RESULT, str(source)))
    neighbor = root.parent / "other-task"
    neighbor.mkdir()
    (neighbor / "keep.txt").write_text("keep")
    manager.archive_task("task", "train")
    # 文件日志与其他资源在 close 前仍可能被使用。
    assert root.exists()
    manager.close()
    manager.close()
    assert not root.exists()
    assert (archives / "task/results/train/reward.png").read_bytes() == b"reward"
    assert {path.name for path in (archives / "task").iterdir()} == {"results", "manifest.json"}
    assert (neighbor / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("case", ["unfinished", "failed", "nested_archive", "missing_file", "changed_file"])
def test_close_preserves_unconfirmed_runtime_tree(tmp_path: Path, monkeypatch, case: str) -> None:
    root = tmp_path / "task"
    archives = root / "archives" if case == "nested_archive" else tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    manager.bind_task_runtime_directory("task", root)
    root.mkdir()
    residue = root / "process.json"
    residue.write_text("keep until confirmed")
    source = root / "reward.png"
    source.write_bytes(b"reward")
    manager.put("task", "train", ArtifactPut(ArtifactType.RESULT, str(source)))
    if case == "unfinished":
        manager.ensure_task("task", "eval")
    if case == "failed":
        def fail(*args, **kwargs):
            raise OSError("manifest write failed")
        monkeypatch.setattr(manager.backend._store("task", "train"), "_write_manifest", fail)
        with pytest.raises(ArtifactsArchiveException, match="manifest write failed"):
            manager.archive_task("task", "train")
        assert source.exists()
    else:
        manager.archive_task("task", "train")
        archived_file = archives / "task/results/train/reward.png"
        if case == "missing_file":
            archived_file.unlink()
        elif case == "changed_file":
            archived_file.write_bytes(b"corrupted")
    manager.close()
    assert residue.read_text() == "keep until confirmed"


def test_close_removes_runtime_only_after_all_stages_archived(tmp_path: Path) -> None:
    root = tmp_path / "task"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    manager.bind_task_runtime_directory("task", root)
    root.mkdir()
    (root / "residue").touch()
    manager.ensure_task("task", "train")
    manager.ensure_task("task", "eval")
    manager.archive_task("task", "train")
    manager.archive_task("task", "eval")
    manager.close()
    assert not root.exists()


@pytest.mark.parametrize("finish_all", [False, True])
@pytest.mark.parametrize("start_stage", [0, 1])
def test_stage_task_preserves_sources_until_all_runnable_stages_finish(
        tmp_path: Path, finish_all: bool, start_stage: int,
) -> None:
    from config import AgentConfig, StageConfig, StageTaskConfig
    from scheduler.stage_task import StageTask
    from utils.task_id import task_id

    task = str(task_id)
    root = tmp_path / task
    archives = tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    manager.bind_task_runtime_directory(task, root)
    stages = [
        StageConfig(
            stage_name=name, train_config=TrainConfig(artifacts_archive_save_dir=str(archives)),
            agents_config=[AgentConfig(name="policy")],
        )
        for name in ("prepare", "learn", "refine")
    ]
    stage_task = StageTask(
        StageTaskConfig(stages=stages, start_stage=start_stage), artifacts_manager=manager,
    )
    runnable = stages[start_stage:]
    assert set(manager.backend._tasks) == {(task, stage.stage_name) for stage in runnable}
    source = root / "models" / "checkpoint.pth"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"stage-one-model")
    first = runnable[0].stage_name
    manager.put(task, first, ArtifactPut(ArtifactType.MODEL, str(source)))
    manager.archive_task(task, first)
    assert source.read_bytes() == b"stage-one-model"
    assert (archives / task / "models" / first / source.name).read_bytes() == b"stage-one-model"
    if finish_all:
        for stage in runnable[1:]:
            # 后续阶段可以复用原路径；前一阶段的归档快照不受覆盖影响。
            source.write_bytes(stage.stage_name.encode())
            manager.put(task, stage.stage_name, ArtifactPut(ArtifactType.MODEL, str(source)))
            manager.archive_task(task, stage.stage_name)
            assert source.exists()
        assert (archives / task / "models" / first / source.name).read_bytes() == b"stage-one-model"
    stage_task.close()
    manager.close()
    assert root.exists() is not finish_all
    if not finish_all:
        assert source.read_bytes() == b"stage-one-model"


def test_put_two_files_are_available_list(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    manager.put("t1", "train", ArtifactPut(ArtifactType.RESULT, str(a)))
    manager.put("t1", "train", ArtifactPut(ArtifactType.RESULT, str(b)))
    available = [
        r for r in manager.list_entries("t1", "train")
        if r.status is ArtifactLedgerStatus.AVAILABLE
    ]
    assert len(available) == 2
    assert {Path(r.center_path).name for r in available} == {"a.bin", "b.bin"}
    assert all(r.origin is ArtifactOrigin.LOCAL for r in available)
    assert all(Path(r.center_path).is_file() for r in available)


def test_put_same_file_is_idempotent(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    path = tmp_path / "once.bin"
    path.write_bytes(b"x")
    first = manager.put("t1", "train", ArtifactPut(ArtifactType.RESULT, str(path), {"k": 1}))
    second = manager.put("t1", "train", ArtifactPut(ArtifactType.RESULT, str(path), {"k": 2}))
    assert first.entry_id == second.entry_id
    assert second.metadata["k"] == 2
    assert len([
        r for r in manager.list_entries("t1") if r.status is ArtifactLedgerStatus.AVAILABLE
    ]) == 1


def test_put_rejects_directory_and_missing(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(ArtifactsArchiveException, match="directory"):
        manager.put("t1", "train", ArtifactPut(ArtifactType.RESULT, str(directory)))
    with pytest.raises(ArtifactsArchiveException, match="does not exist"):
        manager.put("t1", "train", ArtifactPut(ArtifactType.RESULT, str(tmp_path / "missing.bin")))
    assert manager.list_entries("t1") == []


def test_put_without_register_allowed(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    path = tmp_path / "solo.bin"
    path.write_bytes(b"1")
    entry = manager.put("t1", "train", ArtifactPut(ArtifactType.SAMPLE, str(path)))
    assert entry.status is ArtifactLedgerStatus.AVAILABLE
    assert Path(entry.center_path).is_file()


def test_put_model_after_sample_intent_allows_separate_runtime_dir(tmp_path: Path) -> None:
    """样本 planned_hint 先入白名单后，模型仍可 put 到另一运行时目录。"""
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    samples = tmp_path / "samples"
    models = tmp_path / "models" / "demo_ppo"
    samples.mkdir()
    models.mkdir(parents=True)
    manager.register("t1", "train", ArtifactIntent(ArtifactType.SAMPLE, planned_hint=str(samples)))
    model_file = models / "optimo.pth"
    model_file.write_bytes(b"net")
    entry = manager.put("t1", "train", ArtifactPut(ArtifactType.MODEL, str(model_file)))
    assert entry.status is ArtifactLedgerStatus.AVAILABLE
    assert Path(entry.center_path) == model_file.resolve()

    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"escaped")
    link = models / "escaped.pth"
    link.symlink_to(outside)
    with pytest.raises(ArtifactsArchiveException, match="outside allowed"):
        manager.put("t1", "train", ArtifactPut(ArtifactType.MODEL, str(link)))


def test_prepare_output_does_not_write_ledger(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    target = tmp_path / "nested" / "file.bin"
    from scheduler.artifact import Artifact
    manager.prepare_output(Artifact(ArtifactType.RESULT, str(target)))
    assert manager.list_entries("t1") == []
    assert target.parent.is_dir()


def test_archive_only_available_not_directory_scan(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    kept = tmp_path / "kept.bin"
    junk = tmp_path / "junk.bin"
    kept.write_bytes(b"keep")
    junk.write_bytes(b"junk")
    manager.put("taskA", "train", ArtifactPut(ArtifactType.RESULT, str(kept)))
    manager.archive_task("taskA", "train")
    archived_root = archives / "taskA"
    names = {path.name for path in archived_root.rglob("*") if path.is_file()}
    assert "kept.bin" in names
    assert "junk.bin" not in names
    assert (archived_root / "manifest.json").is_file()


def test_archive_with_only_registered_intent_has_no_files(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    manager.register("taskB", "train", ArtifactIntent(ArtifactType.RESULT, str(tmp_path / "hint")))
    archived = manager.archive_task("taskB", "train")
    assert archived.artifact_count == 0


def test_close_clears_ledger_and_is_idempotent(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    path = tmp_path / "x.bin"
    path.write_bytes(b"x")
    manager.put("t1", "train", ArtifactPut(ArtifactType.RESULT, str(path)))
    manager.close()
    assert manager.list_entries("t1") == []
    manager.close()


@pytest.mark.parametrize("has_lock_residue", [False, True])
def test_close_cleans_remote_cache_and_lock_residue(tmp_path: Path, has_lock_residue: bool) -> None:
    """任务收口：_remote_cache/<task>/ 与 .<save_dir>.locks 锁文件按 task 清理；他任务不受影响。"""
    archives = tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    gif = tmp_path / "g.gif"
    gif.write_bytes(b"GIF89a")
    manager.put_bytes(
        "taskA", "train",
        ArtifactPut(ArtifactType.RENDER_GIF, str(gif), {"role_id": "sample-0"}),
        b"GIF89a",
    )
    manager.put_config("taskA", "train")
    manager.archive_task("taskA", "train")
    cache_task = archives / "_remote_cache" / "taskA"
    lock_file = tmp_path / ".archives.locks" / ".taskA.archive.lock"
    assert cache_task.exists()
    # 锁释放后是否保留文件属于后端实现细节；显式构造两种清理前状态。
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    if has_lock_residue:
        lock_file.touch()
    else:
        lock_file.unlink(missing_ok=True)
    assert lock_file.exists() == has_lock_residue

    # 共存的其他任务残留不得被误删
    other_cache = archives / "_remote_cache" / "taskB"
    other_cache.mkdir(parents=True)
    other_lock = tmp_path / ".archives.locks" / ".taskB.archive.lock"
    other_lock.touch()

    manager.close()

    assert not cache_task.exists()
    assert not lock_file.exists()
    assert (tmp_path / ".archives.locks").exists(), "其他任务锁仍在，目录保留"
    assert other_cache.exists()
    assert other_lock.exists()
    assert manager.list_entries("taskA") == []
    manager.close()


@pytest.mark.parametrize("has_lock_residue", [False, True])
def test_close_removes_lock_dir_when_last_lock_gone(tmp_path: Path, has_lock_residue: bool) -> None:
    archives = tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    source = tmp_path / "r.bin"
    source.write_bytes(b"r")
    manager.put("taskA", "train", ArtifactPut(ArtifactType.RESULT, str(source)))
    manager.archive_task("taskA", "train")
    lock_dir = tmp_path / ".archives.locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / ".taskA.archive.lock"
    if has_lock_residue:
        lock_file.touch()
    else:
        lock_file.unlink(missing_ok=True)
    assert lock_file.exists() == has_lock_residue

    manager.close()

    assert not lock_dir.exists(), "最后一个锁清理后目录应一并移除"


# --- 包四：put_bytes 输送（GIF ≤100MB） ---


def test_put_bytes_lands_in_remote_cache_and_archives(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    source = tmp_path / "ep1.gif"
    source.write_bytes(b"GIF89a-ep1")
    entry = manager.put_bytes(
        "taskG", "train",
        ArtifactPut(ArtifactType.RENDER_GIF, str(source), {"role_id": "sample-0"}),
        b"GIF89a-ep1",
    )
    assert entry.status is ArtifactLedgerStatus.AVAILABLE
    assert entry.origin is ArtifactOrigin.REMOTE
    assert entry.producer_path == str(source)
    center = Path(entry.center_path)
    assert center.is_file()
    assert center.read_bytes() == b"GIF89a-ep1"
    assert REMOTE_CACHE_DIRNAME in center.parts
    assert center.parts[center.parts.index(REMOTE_CACHE_DIRNAME) + 1] == "taskG"
    assert "recordings" in center.parts
    assert source.exists(), "worker 原文件不删"

    manager.archive_task("taskG", "train")
    archived_files = {
        path.name for path in (archives / "taskG").rglob("*") if path.is_file()
    }
    assert "ep1.gif" in archived_files
    assert (archives / "taskG" / "manifest.json").is_file()


def test_remote_role_logs_archive_without_worker_files(tmp_path: Path, monkeypatch) -> None:
    import shutil
    from common import logger
    from roles.common.artifact_wiring import complete_role_logs
    from common.log_files import SharedLogFiles

    worker = tmp_path / "worker"
    worker.mkdir()
    source = worker / "sample-0_task.log"
    source.write_bytes(b"worker active log\n")
    with SharedLogFiles(source).locked() as files:
        part = files.reserve_part()
        part.write_bytes(b"worker rotated log\n")
    monkeypatch.setattr(logger, "log_file_path", lambda *args: source)
    center = tmp_path / "center"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(center)))
    manager.expect_producer("task", "train", "log:sample-0")

    complete_role_logs(manager, "task", "train", "sample-0", remote=True)
    entries = manager.list_entries("task", "train")
    assert len(entries) == 2
    assert all(entry.origin is ArtifactOrigin.REMOTE for entry in entries)
    assert manager.is_stage_archivable("task", "train")
    shutil.rmtree(worker)
    manager.archive_task("task", "train")
    assert (center / "task/logs/train" / source.name).read_bytes() == b"worker active log\n"
    assert (center / "task/logs/train" / part.name).read_bytes() == b"worker rotated log\n"


def test_put_bytes_same_producer_path_is_idempotent(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    source = tmp_path / "same.gif"
    source.write_bytes(b"gif")
    item = ArtifactPut(ArtifactType.RENDER_GIF, str(source))
    first = manager.put_bytes("t1", "train", item, b"gif")
    second = manager.put_bytes("t1", "train", item, b"gif")
    assert first.entry_id == second.entry_id
    available = [
        r for r in manager.list_entries("t1") if r.status is ArtifactLedgerStatus.AVAILABLE
    ]
    assert len(available) == 1


def test_put_bytes_basename_collision_keeps_distinct_cache_files(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    a = tmp_path / "worker_a" / "ep.gif"
    b = tmp_path / "worker_b" / "ep.gif"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"A")
    b.write_bytes(b"B")
    first = manager.put_bytes("t1", "train", ArtifactPut(ArtifactType.RENDER_GIF, str(a)), b"A")
    second = manager.put_bytes("t1", "train", ArtifactPut(ArtifactType.RENDER_GIF, str(b)), b"B")
    assert first.entry_id != second.entry_id
    assert first.center_path != second.center_path
    assert Path(first.center_path).read_bytes() == b"A"
    assert Path(second.center_path).read_bytes() == b"B"


def test_put_bytes_rejects_oversize_and_non_bytes(tmp_path: Path, monkeypatch) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    item = ArtifactPut(ArtifactType.RENDER_GIF, str(tmp_path / "big.gif"))
    monkeypatch.setattr("scheduler.artifact.PUT_BYTES_MAX_BYTES", 8)
    with pytest.raises(ArtifactsArchiveException, match="exceeds limit"):
        manager.put_bytes("t1", "train", item, b"0123456789")
    monkeypatch.setattr("scheduler.artifact.PUT_BYTES_MAX_BYTES", 1024)
    with pytest.raises(ArtifactsArchiveException, match="data must be bytes"):
        manager.put_bytes("t1", "train", item, "GIF89a")  # type: ignore[arg-type]
    assert manager.list_entries("t1") == []


def test_put_role_render_gif_skips_oversize_file(tmp_path: Path, monkeypatch) -> None:
    from roles.common.artifact_wiring import put_role_render_gif

    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    big = tmp_path / "big.gif"
    big.write_bytes(b"0123456789")
    monkeypatch.setattr("roles.common.artifact_wiring.PUT_BYTES_MAX_BYTES", 8)
    assert put_role_render_gif(manager, "t1", "train", str(big)) is None
    assert manager.list_entries("t1") == []
    monkeypatch.setattr("roles.common.artifact_wiring.PUT_BYTES_MAX_BYTES", 1024)
    entry = put_role_render_gif(manager, "t1", "train", str(big), {"role_id": "sample-0"})
    assert entry is not None
    assert entry.origin is ArtifactOrigin.REMOTE
    assert entry.metadata["role_id"] == "sample-0"


def test_task_manager_no_longer_collects_gif_runtime_files(tmp_path: Path) -> None:
    from scheduler.artifact import Artifact

    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    tm = TrainTaskManager(
        TrainConfig(
            max_episode=1,
            result_save_dir=str(tmp_path / "results"),
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ),
        [],
        [],
        artifacts_manager=manager,
    )
    gif = tmp_path / "legacy.gif"
    gif.write_bytes(b"gif")
    tm.episode_artifacts[0] = [Artifact(ArtifactType.RENDER_GIF, str(gif))]
    archived = tm.notify_task_complete()
    assert archived is not None
    names = {
        path.name for path in (tmp_path / "archives" / tm.task_id).rglob("*") if path.is_file()
    }
    assert "legacy.gif" not in names
    assert gif.exists()


# --- 收口补遗：put_config 原始 TOML 进运行期台账（§3.5） ---


def test_put_config_puts_raw_toml_copy_into_ledger(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    raw_toml = tmp_path / "subject.toml"
    raw_toml.write_text("[entry]\n", encoding="utf-8")
    manager.bind_raw_config(raw_toml.read_text(encoding="utf-8"), str(raw_toml))

    entry = manager.put_config("taskC", "train")

    assert entry is not None
    assert entry.status is ArtifactLedgerStatus.AVAILABLE
    assert entry.origin is ArtifactOrigin.REMOTE
    assert entry.producer_path == str(raw_toml)
    center = Path(entry.center_path)
    assert center.is_file()
    assert center.read_text(encoding="utf-8") == "[entry]\n"
    assert REMOTE_CACHE_DIRNAME in center.parts
    assert "config" in center.parts
    # 原始 TOML 不动
    assert raw_toml.exists()


def test_put_config_without_raw_bound_is_none(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    assert manager.put_config("taskC", "train") is None
    assert manager.list_entries("taskC") == []


def test_put_config_is_idempotent(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    manager.bind_raw_config("[entry]\n", str(tmp_path / "s.toml"))
    first = manager.put_config("taskC", "train")
    second = manager.put_config("taskC", "train")
    assert first.entry_id == second.entry_id
    config_rows = [
        row for row in manager.list_entries("taskC")
        if row.artifact_type is ArtifactType.CONFIG
        and row.status is ArtifactLedgerStatus.AVAILABLE
    ]
    assert len(config_rows) == 1


def test_put_config_survives_archive_without_duplicate_manifest(tmp_path: Path) -> None:
    import json as _json

    archives = tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    raw_toml = tmp_path / "subject.toml"
    raw_toml.write_text("[entry]\n", encoding="utf-8")
    manager.bind_raw_config(raw_toml.read_text(encoding="utf-8"), str(raw_toml))
    manager.put_config("taskC", "train")

    manager.archive_task("taskC", "train")

    manifest = _json.loads((archives / "taskC" / "manifest.json").read_text(encoding="utf-8"))
    config_entries = [
        entry for entry in manifest["artifacts"]
        if entry["artifact_type"] == "config"
    ]
    assert len(config_entries) == 1, "归档 config/ 仍由 raw 摘要唯一产出"
    config_entries[0]["relative_path"].startswith("config/")
    # 台账缓存副本不随归档清源，运行期仍可 open
    assert Path(manager.list_entries("taskC")[0].center_path).is_file()


def test_task_manager_plans_config_into_ledger(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    raw_toml = tmp_path / "subject.toml"
    raw_toml.write_text("[entry]\n", encoding="utf-8")
    manager.bind_raw_config(raw_toml.read_text(encoding="utf-8"), str(raw_toml))
    tm = TrainTaskManager(
        TrainConfig(
            max_episode=1,
            result_save_dir=str(tmp_path / "results"),
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ),
        [],
        [],
        artifacts_manager=manager,
    )
    config_rows = [
        row for row in manager.list_entries(tm.task_id, tm.stage_name)
        if row.artifact_type is ArtifactType.CONFIG
        and row.status is ArtifactLedgerStatus.AVAILABLE
    ]
    assert len(config_rows) == 1
    assert Path(config_rows[0].center_path).is_file()


# --- 收口补遗：目录产物逐文件 put，台账保持一文件一行（§1.4/§3.2 硬约束） ---


def test_tm_put_directory_registers_files_one_per_row(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    tm = TrainTaskManager(
        TrainConfig(
            max_episode=1,
            result_save_dir=str(tmp_path / "results"),
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ),
        [],
        [],
        artifacts_manager=manager,
    )
    generation = tmp_path / "generation-3"
    (generation / "nested").mkdir(parents=True)
    (generation / "metrics.json").write_text("{}", encoding="utf-8")
    (generation / "nested" / "weights.pt").write_bytes(b"w")

    tm.put_runtime_file(ArtifactType.AUTOML, str(generation))

    rows = [
        row for row in manager.list_entries(tm.task_id, tm.stage_name)
        if row.artifact_type is ArtifactType.AUTOML
        and row.status is ArtifactLedgerStatus.AVAILABLE
    ]
    assert len(rows) == 2, "一文件一行，目录本身不成为台账行"
    assert all(Path(row.center_path).is_file() for row in rows)
    assert all(row.origin is ArtifactOrigin.LOCAL for row in rows)
    relative_paths = {row.metadata["relative_path"] for row in rows}
    assert relative_paths == {"metrics.json", "nested/weights.pt"}


def test_tm_put_directory_excludes_files_added_after_put(tmp_path: Path) -> None:
    """缺口 2 回归：put 之后才出现的未登记文件不得进归档（不扫盘）。"""
    archives = tmp_path / "archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    tm = TrainTaskManager(
        TrainConfig(
            max_episode=1,
            result_save_dir=str(tmp_path / "results"),
            artifacts_archive_save_dir=str(archives),
        ),
        [],
        [],
        artifacts_manager=manager,
    )
    generation = tmp_path / "generation-3"
    generation.mkdir()
    (generation / "registered.json").write_text("{}", encoding="utf-8")

    tm.put_runtime_file(ArtifactType.AUTOML, str(generation))
    (generation / "late_junk.json").write_text("junk", encoding="utf-8")

    archived = tm.notify_task_complete()
    assert archived is not None
    archived_files = {
        path.name for path in (archives / tm.task_id / "automl").rglob("*") if path.is_file()
    }
    assert archived_files == {"registered.json"}
    assert "late_junk.json" not in archived_files


def test_manager_record_rejects_directory(tmp_path: Path) -> None:
    from scheduler.artifact import Artifact

    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    directory = tmp_path / "generation"
    directory.mkdir()
    (directory / "a.bin").write_bytes(b"a")
    with pytest.raises(ArtifactsArchiveException, match="per-file"):
        manager.record("t1", "train", Artifact(ArtifactType.AUTOML, str(directory)))
    assert manager.list_entries("t1") == []


def test_tm_put_empty_directory_registers_nothing(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    tm = TrainTaskManager(
        TrainConfig(
            max_episode=1,
            result_save_dir=str(tmp_path / "results"),
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ),
        [],
        [],
        artifacts_manager=manager,
    )
    empty = tmp_path / "generation-empty"
    empty.mkdir()
    tm.put_runtime_file(ArtifactType.AUTOML, str(empty))
    assert manager.list_entries(tm.task_id, tm.stage_name) == []


def test_task_manager_skips_archive_when_disabled(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class _Spy(ArtifactsManager):
        def archive_task(self, task_id: str, stage_name: str):
            calls.append((task_id, stage_name))
            return super().archive_task(task_id, stage_name)

    manager = _Spy(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    tm = TrainTaskManager(
        TrainConfig(artifacts_archive_enabled=False, max_episode=1),
        [],
        [],
        artifacts_manager=manager,
    )
    assert tm.notify_task_complete() is None
    assert calls == []


def test_config_resolution_optional_section(tmp_path: Path) -> None:
    default = resolve_artifacts_manager_config()
    assert default.save_dir.endswith("artifacts/archives") or "archives" in default.save_dir
    from_old = resolve_artifacts_manager_config(artifacts_archive_save_dir=str(tmp_path / "old"))
    assert from_old.save_dir == str(tmp_path / "old")
    from_new = resolve_artifacts_manager_config(
        artifacts_manager_config=ArtifactsManagerConfig(save_dir=str(tmp_path / "new")),
        artifacts_archive_save_dir=str(tmp_path / "old"),
    )
    assert from_new.save_dir == str(tmp_path / "new")


def test_build_artifacts_manager_activates(tmp_path: Path) -> None:
    manager = build_artifacts_manager(
        artifacts_archive_save_dir=str(tmp_path / "archives"),
        activate=True,
    )
    assert get_active_artifacts_manager() is manager


def test_intent_put_ledger_roundtrip_rejects_extra_keys() -> None:
    intent = ArtifactIntent(ArtifactType.RESULT, planned_hint="/tmp/out", metadata={"k": 1})
    put = ArtifactPut(ArtifactType.RESULT, "/tmp/out.bin", {"k": 2})
    entry = LedgerEntry(
        entry_id="e1",
        task_id="t1",
        stage_name="train",
        artifact_type=ArtifactType.RESULT,
        status=ArtifactLedgerStatus.AVAILABLE,
        origin=ArtifactOrigin.LOCAL,
        center_path="/tmp/out.bin",
        producer_path="/tmp/out.bin",
        metadata={"k": 2},
    )
    assert ArtifactIntent.from_dict(json.loads(json.dumps(intent.to_dict()))) == intent
    assert ArtifactPut.from_dict(json.loads(json.dumps(put.to_dict()))) == put
    restored = LedgerEntry.from_dict(json.loads(json.dumps(entry.to_dict())))
    assert restored.entry_id == entry.entry_id
    assert restored.status is ArtifactLedgerStatus.AVAILABLE
    with pytest.raises(ArtifactsArchiveException):
        ArtifactIntent.from_dict({"artifact_type": "result", "extra": 1})
    with pytest.raises(ArtifactsArchiveException):
        ArtifactPut.from_dict({"artifact_type": "result", "path": "/tmp/a", "extra": 1})
    with pytest.raises(ArtifactsArchiveException):
        ArtifactPut.from_dict({"artifact_type": "result", "path": ""})


def test_producer_callers_are_noop_until_role_wires(tmp_path: Path) -> None:
    class _Producer(ArtifactCallerSupport):
        def __init__(self) -> None:
            self._init_artifact_callers()

    producer = _Producer()
    missing = tmp_path / "missing.bin"
    producer.register_artifact_intent(ArtifactType.RESULT, planned_hint=str(tmp_path))
    producer.put_artifact_file(ArtifactType.RESULT, str(missing))

    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    from roles.common.artifact_wiring import bind_local_artifact_callers

    register, put, complete_for = bind_local_artifact_callers(manager, lambda: "t1", lambda: "train")
    path = tmp_path / "ready.bin"
    path.write_bytes(b"ok")
    producer.register_caller = register
    producer.put_caller = put
    producer.complete_caller = complete_for("producer")
    producer.register_artifact_intent(ArtifactType.RESULT, planned_hint=str(tmp_path))
    producer.put_artifact_file(ArtifactType.RESULT, str(path))
    producer.complete_artifact_stage()
    rows = manager.list_entries("t1", "train")
    statuses = {row.status for row in rows}
    assert ArtifactLedgerStatus.REGISTERED in statuses
    available = [row for row in rows if row.status is ArtifactLedgerStatus.AVAILABLE]
    assert len(available) == 1
    assert Path(available[0].center_path).is_file()


def test_attach_registers_intent_before_put_and_gates_archive(tmp_path: Path) -> None:
    from roles.common.artifact_wiring import (
        attach_artifact_producer,
        bind_local_artifact_callers,
    )

    class _Producer(ArtifactCallerSupport):
        def __init__(self) -> None:
            self._init_artifact_callers()

        def register_stage_intents(self) -> None:
            self.register_artifact_intent(ArtifactType.RESULT)

    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    register, put, complete_for = bind_local_artifact_callers(manager, lambda: "t1", lambda: "train")
    producer = _Producer()
    attach_artifact_producer(
        producer, "result_mod", register, put, complete_for,
        manager, lambda: "t1", lambda: "train",
    )
    rows = manager.list_entries("t1", "train")
    assert rows and rows[0].status is ArtifactLedgerStatus.REGISTERED
    assert manager.is_stage_archivable("t1", "train") is False
    path = tmp_path / "out.bin"
    path.write_bytes(b"ok")
    producer.put_artifact_file(ArtifactType.RESULT, str(path))
    producer.complete_artifact_stage()
    assert manager.is_stage_archivable("t1", "train") is True
    manager.archive_task("t1", "train")
    assert manager.is_stage_archivable("t1", "train") is True


def test_archive_waits_until_all_expected_producers_complete(tmp_path: Path) -> None:
    import threading
    import time

    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    path = tmp_path / "kept.bin"
    path.write_bytes(b"keep")
    manager.put("taskW", "train", ArtifactPut(ArtifactType.RESULT, str(path)))
    manager.expect_producer("taskW", "train", "mod_a")
    manager.expect_producer("taskW", "train", "mod_b")
    started = threading.Event()
    finished: list[object] = []

    def _archive() -> None:
        started.set()
        finished.append(manager.archive_task("taskW", "train"))

    worker = threading.Thread(target=_archive)
    worker.start()
    assert started.wait(1)
    time.sleep(0.05)
    assert worker.is_alive()
    manager.complete("taskW", "train", "mod_a")
    time.sleep(0.05)
    assert worker.is_alive()
    manager.complete("taskW", "train", "mod_b")
    worker.join(2)
    assert not worker.is_alive()
    assert finished
    assert (tmp_path / "archives" / "taskW" / "manifest.json").is_file()


def test_archive_resets_gate_for_next_stage(tmp_path: Path) -> None:
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(tmp_path / "archives")))
    first = tmp_path / "a.bin"
    first.write_bytes(b"a")
    manager.put("taskM", "stage-a", ArtifactPut(ArtifactType.RESULT, str(first)))
    manager.expect_producer("taskM", "stage-a", "mod")
    manager.complete("taskM", "stage-a", "mod")
    manager.archive_task("taskM", "stage-a")
    manager.expect_producer("taskM", "stage-b", "mod")
    assert manager.is_stage_archivable("taskM", "stage-b") is False
    second = tmp_path / "b.bin"
    second.write_bytes(b"b")
    manager.put("taskM", "stage-b", ArtifactPut(ArtifactType.RESULT, str(second)))
    manager.complete("taskM", "stage-b", "mod")
    archived = manager.archive_task("taskM", "stage-b")
    assert archived.task_id == "taskM"
    assert manager.is_stage_archivable("taskM", "stage-b") is True


def test_tm_plans_archive_dir_before_producer_register(tmp_path: Path) -> None:
    """TM 先按 artifacts_archive_save_dir 规划；装配期 register 不得改用 AM 默认目录。"""
    default_archives = tmp_path / "default-archives"
    planned_archives = tmp_path / "planned-archives"
    manager = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(default_archives)))
    tm = TrainTaskManager(
        TrainConfig(max_episode=1, artifacts_archive_save_dir=str(planned_archives)),
        [],
        [],
        artifacts_manager=manager,
    )
    manager.register(tm.task_id, tm.stage_name, ArtifactIntent(ArtifactType.REPLAY))
    path = tmp_path / "final.bin"
    path.write_bytes(b"m")
    tm.put_runtime_file(ArtifactType.MODEL, str(path))
    archived = tm.notify_task_complete()
    assert archived is not None
    assert Path(archived.archive_dir).parent == planned_archives.resolve()
    tm.close()

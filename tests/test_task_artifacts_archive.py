from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Event

import pytest

from scheduler.artifact import Artifact, _ArchiveStorage as ArtifactsArchive, ArtifactType, artifacts_archive
from common.exception import ArtifactsArchiveException
from common.info_bridge import (
    INFO_RAW_CONFIG_FILE,
    INFO_RAW_CONFIG_NAME,
    info_bridge,
    publish_training_info,
)
from config import EvalConfig, ResourceDir, TaskConfig, TrainConfig
from scheduler.train_task import LEASE_KIND_MID_EVAL, SampleLeaseEntry, TrainTaskManager
from rpc.dispatch import TRAIN_TASK_MANAGER, build_dispatcher
from utils.toml_config import TomlTrainingConfig


def _archive(tmp_path: Path, *, task_type: str = "training") -> tuple[ArtifactsArchive, Path]:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    archive = ArtifactsArchive(
        str(tmp_path / "archives"),
        allowed_runtime_dirs=[str(runtime)],
        task_type=task_type,
    )
    return archive, runtime


def test_task_config_declares_archive_defaults_once() -> None:
    assert "artifacts_archive_enabled" in TaskConfig.__annotations__
    assert "artifacts_archive_save_dir" in TaskConfig.__annotations__
    assert "artifacts_archive_enabled" not in TrainConfig.__annotations__
    assert "artifacts_archive_enabled" not in EvalConfig.__annotations__
    assert TaskConfig().artifacts_archive_enabled is True
    assert TaskConfig().artifacts_archive_save_dir == ResourceDir.ARCHIVES.value


def test_archive_moves_registered_artifact_and_writes_queryable_manifest(tmp_path: Path) -> None:
    archive, runtime = _archive(tmp_path)
    source = runtime / "reward.png"
    archive.register("task-1", "train", Artifact(ArtifactType.RESULT, str(runtime)))
    source.write_bytes(b"png")
    archive.record(
        "task-1",
        "train",
        Artifact(ArtifactType.RESULT, str(source), {"kind": "reward"}),
    )

    archived = archive.archive_task(
        "task-1",
        "train",
        raw_file="[design]\ntask_id = \"task-1\"\n",
        config_name="subject.toml",
    )

    target = Path(archived.archive_dir)
    assert not source.exists()
    assert (target / "results" / "train" / "reward.png").read_bytes() == b"png"
    assert {path.name for path in target.iterdir()} == {"manifest.json", "config", "results"}
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task_id"] == "task-1"
    assert manifest["task_type"] == "training"
    assert manifest["stages"]["train"]["artifact_count"] == 2
    assert {entry["artifact_type"] for entry in manifest["artifacts"]} == {
        ArtifactType.RESULT.value,
        ArtifactType.CONFIG.value,
    }
    assert len(archive.list_archived_tasks()) == 1
    queried = archive.get_task_artifacts("task-1")
    assert {item.artifact_type for item in queried} == {ArtifactType.RESULT, ArtifactType.CONFIG}
    assert all(isinstance(item.artifact_type, ArtifactType) for item in queried)
    assert all(Path(item.path).exists() for item in queried)
    assert not any((tmp_path / "archives").glob(".*.archive.lock"))


def test_archive_is_idempotent_and_merges_stages_without_duplicate_config(tmp_path: Path) -> None:
    archive, runtime = _archive(tmp_path)
    raw = "[design]\ntask_id = \"task-1\"\n"
    archive.register("task-1", "stage-a", Artifact(ArtifactType.SAMPLE, str(runtime)))
    archive.register("task-1", "stage-b", Artifact(ArtifactType.REPLAY, str(runtime)))
    first = runtime / "first.bin"
    first.write_bytes(b"first")
    archive.record("task-1", "stage-a", Artifact(ArtifactType.SAMPLE, str(first)))
    initial = archive.archive_task(
        "task-1", "stage-a", raw_file=raw, config_name="task.toml",
    )
    assert archive.archive_task(
        "task-1", "stage-a", raw_file=raw, config_name="task.toml",
    ) == initial

    second = runtime / "second.bin"
    second.write_bytes(b"second")
    archive.record("task-1", "stage-b", Artifact(ArtifactType.REPLAY, str(second)))
    merged = archive.archive_task(
        "task-1", "stage-b", raw_file=raw, config_name="task.toml",
    )
    manifest = json.loads(
        (Path(merged.archive_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert list(manifest["stages"]) == ["stage-a", "stage-b"]
    assert sum(item["artifact_type"] == "config" for item in manifest["artifacts"]) == 1


def test_archive_rejects_changed_config_and_new_artifact_after_stage_complete(tmp_path: Path) -> None:
    archive, runtime = _archive(tmp_path)
    archive.archive_task("task-1", "stage-a", raw_file="a=1\n", config_name="a.toml")
    with pytest.raises(ArtifactsArchiveException, match="raw TOML changed"):
        archive.archive_task("task-1", "stage-b", raw_file="a=2\n", config_name="a.toml")

    late = runtime / "late.txt"
    late.write_text("late", encoding="utf-8")
    with pytest.raises(ArtifactsArchiveException, match="stage already archived"):
        archive.record("task-1", "stage-a", Artifact(ArtifactType.CUSTOM, str(late)))


def test_archive_rejects_source_conflict_and_symlink_escape(tmp_path: Path) -> None:
    archive, runtime = _archive(tmp_path)
    source = runtime / "shared.txt"
    archive.register("task-a", "train", Artifact(ArtifactType.CUSTOM, str(runtime)))
    archive.register("task-b", "train", Artifact(ArtifactType.CUSTOM, str(runtime)))
    source.write_text("shared", encoding="utf-8")
    archive.record("task-a", "train", Artifact(ArtifactType.CUSTOM, str(source)))
    with pytest.raises(ArtifactsArchiveException, match="already belongs"):
        archive.record("task-b", "train", Artifact(ArtifactType.CUSTOM, str(source)))

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = runtime / "escape.txt"
    link.symlink_to(outside)
    with pytest.raises(ArtifactsArchiveException, match="outside allowed"):
        archive.record("task-a", "train", Artifact(ArtifactType.CUSTOM, str(link)))
    assert outside.read_text(encoding="utf-8") == "outside"


def test_archive_failure_keeps_runtime_source_and_no_half_package(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, runtime = _archive(tmp_path)
    source = runtime / "result.txt"
    archive.register("task-1", "train", Artifact(ArtifactType.RESULT, str(runtime)))
    source.write_text("result", encoding="utf-8")
    archive.record("task-1", "train", Artifact(ArtifactType.RESULT, str(source)))
    monkeypatch.setattr(
        archive,
        "_write_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("manifest failed")),
    )

    with pytest.raises(ArtifactsArchiveException, match="manifest failed"):
        archive.archive_task("task-1", "train", raw_file=None, config_name=None)

    assert source.read_text(encoding="utf-8") == "result"
    assert not (tmp_path / "archives" / "task-1").exists()


def test_archive_queries_reject_missing_or_broken_manifest(tmp_path: Path) -> None:
    archive, _ = _archive(tmp_path)
    broken = tmp_path / "archives" / "broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{}", encoding="utf-8")
    assert archive.list_archived_tasks() == []
    with pytest.raises(ArtifactsArchiveException):
        archive.get_task_artifacts("missing")
    with pytest.raises(ArtifactsArchiveException):
        archive.get_task_artifacts("broken")


def test_different_tasks_can_archive_concurrently_without_cross_move(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    archives = tmp_path / "archives"
    first_source = runtime / "first.txt"
    second_source = runtime / "second.txt"
    first_source.write_text("first", encoding="utf-8")
    second_source.write_text("second", encoding="utf-8")
    first = ArtifactsArchive(str(archives), allowed_runtime_dirs=[str(runtime)])
    second = ArtifactsArchive(str(archives), allowed_runtime_dirs=[str(runtime)])
    first.register("task-a", "train", Artifact(ArtifactType.CUSTOM, str(runtime)))
    second.register("task-b", "train", Artifact(ArtifactType.CUSTOM, str(runtime)))
    first.record("task-a", "train", Artifact(ArtifactType.CUSTOM, str(first_source)))
    second.record("task-b", "train", Artifact(ArtifactType.CUSTOM, str(second_source)))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda item: item[0].archive_task(
                item[1], "train", raw_file=None, config_name=None,
            ),
            [(first, "task-a"), (second, "task-b")],
        ))

    assert {result.task_id for result in results} == {"task-a", "task-b"}
    assert (archives / "task-a" / "custom" / "train" / "first.txt").exists()
    assert (archives / "task-b" / "custom" / "train" / "second.txt").exists()


def test_task_manager_completion_archives_registered_file_and_disabled_keeps_it(
        tmp_path: Path,
) -> None:
    runtime = tmp_path / "results"
    runtime.mkdir()
    enabled_config = EvalConfig(
        max_episode=0,
        result_save_dir=str(runtime),
        render_video_dir=str(tmp_path / "recordings"),
        output_dir=str(tmp_path / "samples"),
        artifacts_archive_save_dir=str(tmp_path / "archives"),
    )
    enabled = TrainTaskManager(enabled_config, [], [], stage_name="eval")
    source = runtime / "report.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    enabled.put_runtime_file(ArtifactType.REPORT, str(source))
    try:
        archived = enabled.notify_task_complete()
        assert archived is not None
        assert (Path(archived.archive_dir) / "reports" / "eval" / "report.csv").exists()
        assert enabled.notify_task_complete() == archived
    finally:
        enabled.close()

    disabled_source = runtime / "disabled.csv"
    disabled_source.write_text("keep", encoding="utf-8")
    disabled_config = EvalConfig(
        max_episode=0,
        result_save_dir=str(runtime),
        render_video_dir=str(tmp_path / "recordings-disabled"),
        output_dir=str(tmp_path / "samples-disabled"),
        artifacts_archive_enabled=False,
        artifacts_archive_save_dir=str(tmp_path / "disabled-archives"),
    )
    disabled = TrainTaskManager(disabled_config, [], [], stage_name="eval-disabled")
    disabled.put_runtime_file(ArtifactType.REPORT, str(disabled_source))
    try:
        assert disabled.notify_task_complete() is None
        assert disabled_source.exists()
        assert not (tmp_path / "disabled-archives").exists()
    finally:
        disabled.close()


def test_task_manager_concurrent_completion_runs_archive_once(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = tmp_path / "results"
    runtime.mkdir()
    manager = TrainTaskManager(
        EvalConfig(
            max_episode=0,
            result_save_dir=str(runtime),
            render_video_dir=str(tmp_path / "recordings"),
            output_dir=str(tmp_path / "samples"),
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ),
        [],
        [],
        stage_name="eval",
    )
    source = runtime / "report.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    manager.put_runtime_file(ArtifactType.REPORT, str(source))
    # TM 走注入的 ArtifactsManager，勿 patch 兼容 proxy（会污染后续用例）。
    original = manager.artifacts_manager.archive_task
    calls = 0

    def slow_archive(*args, **kwargs):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return original(*args, **kwargs)

    monkeypatch.setattr(manager.artifacts_manager, "archive_task", slow_archive)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first, second = executor.map(lambda _: manager.notify_task_complete(), range(2))
        assert first == second
        assert calls == 1
    finally:
        manager.close()


@pytest.mark.parametrize("plot_method", ["save_reward_curve", "save_reward_details"])
def test_completion_waits_for_reward_plot_submission(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        plot_method: str,
) -> None:
    monkeypatch.setattr(TrainTaskManager, "_start_monitors", lambda self: None)
    manager = TrainTaskManager(
        EvalConfig(
            max_episode=0,
            result_save_dir=str(tmp_path / "results"),
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ),
        [],
        [],
    )
    monkeypatch.setattr(manager, "_downstream_sample_rewards", lambda: {"agent": [1.0, 2.0]})
    manager._reward_details_capability = "supported"
    manager.episode_rewards_details = [{"bonus": 1.0}, {"bonus": 2.0}]

    def draw_plot(data, image_file, **kwargs):
        image_file.write(b"reward plot")

    monkeypatch.setattr(f"scheduler.train_task.{plot_method}", draw_plot)
    put_started = Event()
    release_put = Event()
    closing_started = Event()
    archive_started = Event()
    original_put = manager.put_runtime_file
    original_close = manager._close_result_artifact_handles
    original_archive = manager.artifacts_manager.archive_task

    def blocked_put(*args, **kwargs):
        put_started.set()
        assert release_put.wait(5)
        return original_put(*args, **kwargs)

    def close_handles():
        closing_started.set()
        return original_close()

    def archive(*args, **kwargs):
        archive_started.set()
        return original_archive(*args, **kwargs)

    monkeypatch.setattr(manager, "put_runtime_file", blocked_put)
    monkeypatch.setattr(manager, "_close_result_artifact_handles", close_handles)
    monkeypatch.setattr(manager.artifacts_manager, "archive_task", archive)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            plot = executor.submit(getattr(manager, plot_method))
            try:
                assert put_started.wait(5)
                completion = executor.submit(manager.notify_task_complete)
                assert closing_started.wait(5)
                assert not archive_started.wait(0.1)
            finally:
                release_put.set()
            image_path = plot.result(timeout=5)
            archived = completion.result(timeout=5)
        assert archived is not None
        archived_image = Path(archived.archive_dir) / "results" / "train" / Path(image_path).name
        assert archived_image.read_bytes() == b"reward plot"
        archived_csv = Path(archived.archive_dir) / "results" / "train" / "raw_data" / (Path(image_path).stem + ".csv")
        assert archived_csv.read_text(encoding="utf-8").startswith("episode,series,reward")
        assert getattr(manager, plot_method)() is None
        assert not Path(image_path).exists()
    finally:
        manager.close()


def test_raw_toml_is_cached_archived_once_and_redacted_from_rpc_info(tmp_path: Path) -> None:
    config_file = tmp_path / "eval.toml"
    raw = (
        "[env_config]\nuse_sim = false\n"
        "[eval_config]\n"
        "max_episode = 0\n"
        f'result_save_dir = "{tmp_path / "results"}"\n'
        f'render_video_dir = "{tmp_path / "recordings"}"\n'
        f'output_dir = "{tmp_path / "samples"}"\n'
        f'artifacts_archive_save_dir = "{tmp_path / "archives"}"\n'
    )
    config_file.write_text(raw, encoding="utf-8")
    with info_bridge.scope():
        conf = TomlTrainingConfig(config_file, project_root=tmp_path)
        assert conf.raw_file == raw
        publish_training_info(conf)
        manager = TrainTaskManager(conf.eval_config, [], [], stage_name="eval")
        manager.artifacts_manager.bind_raw_config(conf.raw_file, str(conf.config_file))
        try:
            assert manager.pull_info(INFO_RAW_CONFIG_FILE) is None
            assert INFO_RAW_CONFIG_FILE not in manager.pull_info()
            assert INFO_RAW_CONFIG_NAME not in manager.pull_info()
            archived = manager.notify_task_complete()
        finally:
            manager.close()
    assert archived is not None
    configs = list((Path(archived.archive_dir) / "config").glob("eval_*.toml"))
    assert len(configs) == 1
    assert configs[0].read_text(encoding="utf-8") == raw


def test_stage_task_rejects_different_archive_roots() -> None:
    from config import StageConfig, StageTaskConfig
    from scheduler.stage_task import StageTask

    first = StageConfig(stage_name="a")
    second = StageConfig(stage_name="b")
    first.train_config.artifacts_archive_save_dir = "/tmp/archive-a"
    second.train_config.artifacts_archive_save_dir = "/tmp/archive-b"
    with pytest.raises(ValueError, match="same artifacts_archive_save_dir"):
        StageTask(StageTaskConfig(stages=[first, second]))


def test_eval_main_route_ignores_legacy_mid_eval_artifact_flags(tmp_path: Path) -> None:
    manager = TrainTaskManager(
        EvalConfig(
            max_episode=1,
            artifacts_archive_enabled=False,
            result_save_dir=str(tmp_path / "results"),
            render_video_dir=str(tmp_path / "recordings"),
            output_dir=str(tmp_path / "samples"),
        ),
        [],
        [],
        stage_name="eval",
    )
    try:
        token = manager.get_sample_task_info()
        assert token is not None
        build_dispatcher(TRAIN_TASK_MANAGER, manager).invoke(
            "report_sample_result", (token.episode_num, {}), {
                "lease_id": token.lease_id,
                "artifacts": {"mid_eval": True, "mid_eval_gif": "/tmp/legacy.gif"},
            },
        )
        assert manager.sample_task_completed == 1
        assert manager._mid_eval_gif_paths == []
        assert manager.episode_artifacts[0] == [
            Artifact(ArtifactType.RENDER_GIF, "/tmp/legacy.gif", {"legacy_key": "mid_eval_gif"}),
        ]
    finally:
        manager.close()


def test_bypass_route_uses_only_lease_and_rejects_stale_report(tmp_path: Path) -> None:
    manager = TrainTaskManager(
        TrainConfig(
            max_episode=1,
            mid_eval_enabled=True,
            sample_learn_max_lag=None,
            artifacts_archive_enabled=False,
            result_save_dir=str(tmp_path / "results"),
            render_video_dir=str(tmp_path / "recordings"),
        ),
        [],
        [],
    )
    try:
        # 旁路路由只认租约表：手工登记一张 mid_eval 在途租约（正常由签发路径写入）。
        manager._lease_table["active-lease"] = SampleLeaseEntry(
            kind=LEASE_KIND_MID_EVAL,
            episode_num=1,
            generation=manager.generation,
            issued_at=datetime.now(),
            dispatch_count=1,
        )
        manager.report_sample_result(
            1,
            {},
            lease_id="active-lease",
            artifacts=[Artifact(ArtifactType.RENDER_GIF, "/tmp/mid.gif")],
        )
        assert manager._mid_eval_gif_paths == ["/tmp/mid.gif"]

        # 未知租约（含历史负号 episode 约定）一律按过期回报丢弃，不影响已登记产物。
        build_dispatcher(TRAIN_TASK_MANAGER, manager).invoke(
            "report_sample_result", (-1, {}), {
                "lease_id": "expired-lease",
                "artifacts": {"mid_eval_gif": "/tmp/stale.gif"},
            },
        )
        assert manager._mid_eval_gif_paths == ["/tmp/mid.gif"]
        assert manager._stale_report_count == 1
    finally:
        manager.close()

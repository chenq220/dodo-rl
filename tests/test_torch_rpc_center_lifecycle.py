from __future__ import annotations

import os

from rpc.center_lifecycle import (
    is_rpc_center_finished,
    mark_rpc_center_finished,
)
from scheduler.tasks.training_progress import CENTER_FINISHED_ENV
from config import GrpcConfig
from rpc.dispatch import MODEL_MANAGER, TRAIN_TASK_MANAGER
from rpc.factory import RpcServiceHost


def test_mark_rpc_center_finished_uses_env_without_progress_file(monkeypatch):
    monkeypatch.delenv("DODO_PROGRESS_FILE", raising=False)
    monkeypatch.delenv(CENTER_FINISHED_ENV, raising=False)
    assert is_rpc_center_finished() is False
    assert mark_rpc_center_finished() == CENTER_FINISHED_ENV
    assert os.environ[CENTER_FINISHED_ENV] == "1"
    assert is_rpc_center_finished() is True


def test_mark_rpc_center_finished_is_visible_to_other_checks(tmp_path, monkeypatch):
    progress_file = tmp_path / "training_progress.env"
    monkeypatch.setenv("DODO_PROGRESS_FILE", str(progress_file))
    monkeypatch.delenv(CENTER_FINISHED_ENV, raising=False)

    assert is_rpc_center_finished() is False
    marked = mark_rpc_center_finished()
    assert marked == CENTER_FINISHED_ENV
    assert f"{CENTER_FINISHED_ENV}=1" in progress_file.read_text(encoding="utf-8")
    assert is_rpc_center_finished() is True
    assert mark_rpc_center_finished() == marked


def test_grpc_host_marks_center_finished_only_when_stopping_task_manager(tmp_path, monkeypatch):
    monkeypatch.setenv("DODO_PROGRESS_FILE", str(tmp_path / "training_progress.env"))
    monkeypatch.delenv(CENTER_FINISHED_ENV, raising=False)

    model_host = RpcServiceHost(GrpcConfig(), "model-0", {MODEL_MANAGER: object()})
    model_host._started = True
    model_host.stop()
    assert is_rpc_center_finished() is False

    control_host = RpcServiceHost(GrpcConfig(), "control", {TRAIN_TASK_MANAGER: object()})
    control_host._started = True
    control_host.stop()
    assert is_rpc_center_finished() is True


def test_torch_center_notifies_isolated_workers_before_shutdown(monkeypatch):
    from rpc.torch_rpc import runtime

    events = []
    monkeypatch.delenv("DODO_PROGRESS_FILE", raising=False)
    monkeypatch.delenv(CENTER_FINISHED_ENV, raising=False)
    monkeypatch.setattr(runtime, "_INITIALIZED", True)
    monkeypatch.setattr(runtime, "_RANK", 0)
    monkeypatch.setattr(runtime, "_WORLD_SIZE", 3)
    monkeypatch.setattr(runtime.torch_rpc, "rpc_sync", lambda rank, fn, **kwargs: events.append((rank, fn)))
    monkeypatch.setattr(runtime.torch_rpc, "shutdown", lambda **kwargs: events.append("shutdown"))
    runtime.shutdown_runtime()
    assert events == [(1, mark_rpc_center_finished), (2, mark_rpc_center_finished), "shutdown"]
    assert is_rpc_center_finished()

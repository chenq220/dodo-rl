"""包五：ModelManager 收口导出 champion 权重文件进产物台账（§5.7）。"""
from __future__ import annotations

from pathlib import Path

import torch

from config import ModelManagerConfig
from models import ModelManager
from models.model import ModelStatus
from scheduler.artifact import (
    ArtifactLedgerStatus,
    ArtifactOrigin,
    ArtifactPut,
    ArtifactType,
    ArtifactsManager,
    ArtifactsManagerConfig,
)


class _FakeCaller:
    def __init__(self) -> None:
        self.puts: list[ArtifactPut] = []
        self.calls: list[None] = []

    def __call__(self, item: ArtifactPut) -> None:
        self.puts.append(item)


def _wire_callers(manager: ModelManager) -> tuple[_FakeCaller, _FakeCaller, _FakeCaller]:
    register, put, complete = _FakeCaller(), _FakeCaller(), _FakeCaller()

    def complete_caller() -> None:
        complete.calls.append(None)

    manager.register_caller = register
    manager.put_caller = put
    manager.complete_caller = complete_caller
    return register, put, complete


def _submit(manager: ModelManager, name: str, value: int, *, promote: bool = False):
    return manager.submit_model(
        model_name=name,
        algo_type="ppo",
        models={
            "agent_net": {"weight": torch.tensor([value])},
            "mixer_net": {"weight": torch.tensor([value, value])},
        },
        promote=promote,
    )


def test_complete_exports_only_champion_files(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path)))
    _, put, complete = _wire_callers(manager)
    first = _submit(manager, "agent_a", 0)
    candidate = _submit(manager, "agent_a", 1)
    champion = _submit(manager, "agent_a", 2)
    manager.promote_model("agent_a", champion.version)

    manager.complete_artifact_stage()

    assert len(complete.calls) == 1
    champion_paths = {info.local_path for info in champion.artifacts.values()}
    put_paths = {item.path for item in put.puts}
    assert put_paths == champion_paths
    assert all(item.artifact_type is ArtifactType.MODEL for item in put.puts)
    assert all(item.metadata["model_name"] == "agent_a" for item in put.puts)
    assert all(item.metadata["champion_version"] == champion.version for item in put.puts)
    assert {item.metadata["artifact"] for item in put.puts} == {"agent_net", "mixer_net"}
    candidate_paths = {info.local_path for info in candidate.artifacts.values()}
    first_paths = {info.local_path for info in first.artifacts.values()}
    assert put_paths.isdisjoint(candidate_paths | first_paths)


def test_complete_exports_champions_for_every_model_name(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path)))
    _, put, _ = _wire_callers(manager)
    champion_a = _submit(manager, "agent_a", 0)
    champion_b = _submit(manager, "agent_b", 5)

    manager.complete_artifact_stage()

    by_model = {
        item.metadata["model_name"]: item.metadata["champion_version"]
        for item in put.puts
    }
    assert by_model == {"agent_a": champion_a.version, "agent_b": champion_b.version}


def test_complete_without_wiring_is_noop(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path)))
    _submit(manager, "agent_a", 0)

    manager.complete_artifact_stage()


def test_complete_without_champion_reports_done(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path)))
    _, put, complete = _wire_callers(manager)

    manager.complete_artifact_stage()

    assert put.puts == []
    assert len(complete.calls) == 1


def test_complete_is_idempotent(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path)))
    _, put, _ = _wire_callers(manager)
    _submit(manager, "agent_a", 0)

    manager.complete_artifact_stage()
    manager.complete_artifact_stage()

    assert len(put.puts) == 4  # 2 artifacts x 2 次；AM 侧同 center_path upsert 幂等


def test_champion_export_reaches_archive(tmp_path: Path) -> None:
    archives = tmp_path / "archives"
    am = ArtifactsManager(ArtifactsManagerConfig(save_dir=str(archives)))
    manager = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path / "pool")))

    from roles.common.artifact_wiring import bind_local_artifact_callers

    register, put, complete_for = bind_local_artifact_callers(
        am, lambda: "taskM", lambda: "train",
    )
    manager.register_caller = register
    manager.put_caller = put
    manager.complete_caller = complete_for("model_manager")
    manager.register_stage_intents()

    champion = _submit(manager, "agent_a", 0)
    manager.complete_artifact_stage()

    archived = am.archive_task("taskM", "train")
    assert archived is not None
    model_files = {
        path.name
        for path in (archives / "taskM" / "models").rglob("*")
        if path.is_file()
    }
    assert model_files == {Path(info.local_path).name for info in champion.artifacts.values()}
    entries = [
        entry for entry in am.list_entries("taskM")
        if entry.artifact_type is ArtifactType.MODEL
        and entry.status is not ArtifactLedgerStatus.REGISTERED
    ]
    assert entries
    assert all(entry.origin is ArtifactOrigin.LOCAL for entry in entries)
    assert all(
        entry.status is ArtifactLedgerStatus.ARCHIVED for entry in entries
    )

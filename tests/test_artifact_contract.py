from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints

import pytest

from scheduler.artifact import Artifact, ArtifactPut, ArtifactType, artifacts_archive
from common.exception import ArtifactsArchiveException
from config import EvalConfig
from rpc.artifacts import decode_artifacts, encode_artifacts
from rpc.dispatch import TRAIN_TASK_MANAGER, build_dispatcher
from rpc.grpc.service.train_task_rpc import TrainTaskManagerRpcClient
from rpc.torch_rpc.client import TrainTaskManagerTorchRpcClient
from scheduler.train_task import TrainTaskManager


def test_artifact_type_is_enum_and_only_wire_format_is_string() -> None:
    assert get_type_hints(Artifact)["artifact_type"] is ArtifactType
    artifact = Artifact(ArtifactType.RENDER_GIF, "/tmp/episode.gif", {"episode": 1})
    assert artifact.artifact_type is ArtifactType.RENDER_GIF
    payload = artifact.to_dict()
    assert type(payload["artifact_type"]) is str
    assert Artifact.from_dict(json.loads(json.dumps(payload))) == artifact
    assert Artifact.from_dict(payload).artifact_type is ArtifactType.RENDER_GIF
    with pytest.raises(ArtifactsArchiveException, match="must be ArtifactType"):
        Artifact("render_gif", "/tmp/episode.gif")  # type: ignore[arg-type]


@pytest.mark.parametrize("payload", [
    [{"artifact_type": "unknown", "path": "/tmp/a"}],
    [{"artifact_type": "render_gif"}],
    [{"artifact_type": "render_gif", "path": "/tmp/a", "metadata": []}],
])
def test_rpc_rejects_invalid_artifact_records(payload) -> None:
    with pytest.raises(ArtifactsArchiveException):
        decode_artifacts(payload)


def test_old_dict_is_accepted_only_by_rpc_decoder() -> None:
    decoded = decode_artifacts({"mid_eval": True, "mid_eval_gif": "/tmp/a.gif"})
    assert decoded == [
        Artifact(ArtifactType.RENDER_GIF, "/tmp/a.gif", {"legacy_key": "mid_eval_gif"}),
    ]
    assert decoded[0].artifact_type is ArtifactType.RENDER_GIF
    with pytest.raises(ArtifactsArchiveException, match="list\\[Artifact\\]"):
        encode_artifacts({"render_gif": "/tmp/a.gif"})  # type: ignore[arg-type]


@pytest.mark.parametrize("client_cls", [None, TrainTaskManagerRpcClient, TrainTaskManagerTorchRpcClient])
def test_local_and_rpc_keep_contract_through_task_archive(tmp_path: Path, client_cls) -> None:
    runtime = tmp_path / "recordings"
    runtime.mkdir()
    source = runtime / "ordinary.gif"
    source.write_bytes(b"gif")
    manager = TrainTaskManager(
        EvalConfig(
            max_episode=1,
            render_gif_enabled=True,
            render_video_dir=str(runtime),
            result_save_dir=str(tmp_path / "results"),
            artifacts_archive_save_dir=str(tmp_path / "archives"),
        ), [], [],         stage_name="eval",
    )
    dispatcher = build_dispatcher(TRAIN_TASK_MANAGER, manager)
    wire_payloads = []

    def call(method, *args, **kwargs):
        payload = kwargs["artifacts"]
        assert isinstance(payload, list)
        assert all(type(item) is dict and type(item["artifact_type"]) is str for item in payload)
        wire_payloads.append(payload)
        # 模拟传输边界；不能靠 pickle 保留 Artifact 对象蒙混过关。
        kwargs["artifacts"] = json.loads(json.dumps(payload))
        return dispatcher.invoke(method, args, kwargs)

    client = manager
    if client_cls is not None:
        client = client_cls.__new__(client_cls)
        client.client = SimpleNamespace(call=call)
    artifact = Artifact(ArtifactType.RENDER_GIF, str(source), {"episode": 1})
    try:
        token = manager.get_sample_task_info()
        with pytest.raises(ArtifactsArchiveException, match="list\\[Artifact\\]"):
            manager.report_sample_result(token.episode_num, {}, artifacts={"gif": str(source)})
        client.report_sample_result(token.episode_num, {}, artifacts=[artifact], lease_id=token.lease_id)
        assert manager.episode_artifacts[0] == [artifact]
        assert manager.episode_artifacts[0][0].artifact_type is ArtifactType.RENDER_GIF
        assert manager.get_eval_episode_records()[0]["artifacts"] is None
        if client_cls is not None:
            assert wire_payloads == [[artifact.to_dict()]]
        # 包四：GIF 不再由 TM 代收；worker 侧 caller 经 put_bytes 直送中心台账。
        manager.artifacts_manager.put_bytes(
            manager.task_id,
            manager.stage_name,
            ArtifactPut(ArtifactType.RENDER_GIF, str(source), {"episode": 1}),
            b"gif",
        )
        archived = manager.notify_task_complete()
        assert archived is not None
        queried = artifacts_archive.get_task_artifacts(manager.task_id)
        gif = next(item for item in queried if item.artifact_type is ArtifactType.RENDER_GIF)
        assert gif.metadata.get("episode") == 1
        assert Path(gif.path).read_bytes() == b"gif"
        # 归档源自 cache 副本；worker 原文件保留。
        assert source.exists()
    finally:
        manager.close()

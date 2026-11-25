"""采样产物的 RPC 边界适配；核心模块只接收 Artifact。"""
from __future__ import annotations

from typing import Any

from scheduler.artifact import Artifact, ArtifactType
from common.exception import ArtifactsArchiveException


def encode_artifacts(artifacts: list[Artifact] | None) -> list[dict[str, Any]] | None:
    if artifacts is None:
        return None
    if not isinstance(artifacts, list) or any(not isinstance(item, Artifact) for item in artifacts):
        raise ArtifactsArchiveException(operation="encode", detail="expected list[Artifact]")
    return [artifact.to_dict() for artifact in artifacts]


def decode_artifacts(payload: list[dict[str, Any]] | dict | None) -> list[Artifact] | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        # 0.14 兼容旧 RPC artifacts 字典；0.15 删除。路由标记绝不进入契约。
        return _decode_legacy_artifacts(payload)
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise ArtifactsArchiveException(operation="decode", detail="expected artifact record list")
    return [Artifact.from_dict(item) for item in payload]


def _decode_legacy_artifacts(payload: dict) -> list[Artifact]:
    types = {
        "gif": ArtifactType.RENDER_GIF,
        "render_gif": ArtifactType.RENDER_GIF,
        "mid_eval_gif": ArtifactType.RENDER_GIF,
        "end_eval_gif": ArtifactType.RENDER_GIF,
        "model": ArtifactType.MODEL,
        "models": ArtifactType.MODEL,
        "sample": ArtifactType.SAMPLE,
        "replay": ArtifactType.REPLAY,
        "report": ArtifactType.REPORT,
        "result": ArtifactType.RESULT,
        "automl": ArtifactType.AUTOML,
    }
    artifacts = []
    for key, value in payload.items():
        if key == "mid_eval" or isinstance(value, bool) or value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for path in values:
            if isinstance(path, str) and path:
                artifacts.append(Artifact(
                    types.get(key, ArtifactType.CUSTOM),
                    path,
                    {"legacy_key": str(key)},
                ))
    return artifacts

"""Role 侧 ArtifactsManager 解析与本机 register/put/complete 闭包。"""
from __future__ import annotations

import os
from typing import Callable, Optional

from common.logger import init_logger
from scheduler.artifact import (
    PUT_BYTES_MAX_BYTES,
    ArtifactCallerSupport,
    ArtifactIntent,
    ArtifactPut,
    ArtifactType,
    ArtifactsManager,
    LedgerEntry,
    build_artifacts_manager,
    get_active_artifacts_manager,
)

LOG = init_logger("artifact_wiring")


def resolve_artifacts_manager(
        artifacts_manager: Optional[ArtifactsManager] = None,
) -> ArtifactsManager:
    """优先用 Role 注入；否则取入口激活的 Manager；再否则懒创建。"""
    if artifacts_manager is not None:
        return artifacts_manager
    active = get_active_artifacts_manager()
    if active is not None:
        return active
    return build_artifacts_manager(activate=True)


def bind_local_artifact_callers(
        manager: ArtifactsManager,
        task_id: Callable[[], str],
        stage_name: Callable[[], str],
) -> tuple[
    Callable[[ArtifactIntent], object],
    Callable[[ArtifactPut], object],
    Callable[[str], Callable[[], object]],
]:
    """把 Manager.register/put/complete 收成图 3.1 的 caller 闭包。"""

    def register(intent: ArtifactIntent):
        return manager.register(task_id(), stage_name(), intent)

    def put(item: ArtifactPut):
        return manager.put(task_id(), stage_name(), item)

    def complete_for(producer_id: str):
        def complete():
            return manager.complete(task_id(), stage_name(), producer_id)

        return complete

    return register, put, complete_for


def bind_artifact_producer_callers(
        producer: ArtifactCallerSupport,
        producer_id: str,
        register: Callable[[ArtifactIntent], object],
        put: Callable[[ArtifactPut], object],
        complete_for: Callable[[str], Callable[[], object]],
) -> None:
    """只接线 caller，不 expect、不 register。多阶段 Role 在阶段开始再 declare。"""
    producer.register_caller = register
    producer.put_caller = put
    producer.complete_caller = complete_for(producer_id)


def declare_stage_artifact_producer(
        producer: ArtifactCallerSupport,
        producer_id: str,
        manager: ArtifactsManager,
        task_id: Callable[[], str],
        stage_name: Callable[[], str],
) -> None:
    """本阶段 expect 该模块并立刻 register 意图。"""
    reset = getattr(producer, "reset_stage_artifact_intents", None)
    if not callable(reset):
        return
    manager.expect_producer(task_id(), stage_name(), producer_id)
    reset()


def attach_artifact_producer(
        producer: ArtifactCallerSupport,
        producer_id: str,
        register: Callable[[ArtifactIntent], object],
        put: Callable[[ArtifactPut], object],
        complete_for: Callable[[str], Callable[[], object]],
        manager: ArtifactsManager,
        task_id: Callable[[], str],
        stage_name: Callable[[], str],
) -> None:
    """接线 register/put/complete，向 AM expect 本模块，并立刻 register 意图。"""
    bind_artifact_producer_callers(producer, producer_id, register, put, complete_for)
    declare_stage_artifact_producer(producer, producer_id, manager, task_id, stage_name)


def complete_artifact_producers(*producers: ArtifactCallerSupport | None) -> None:
    """本阶段收口：对已接线产物模块再 complete 一次（幂等）。"""
    for producer in producers:
        if producer is None:
            continue
        complete = getattr(producer, "complete_artifact_stage", None)
        if callable(complete):
            complete()


def log_producer_id(role_id: str) -> str:
    return f"log:{role_id}"


def declare_role_log_producer(
        manager: ArtifactsManager,
        task_id: Callable[[], str] | str,
        stage_name: Callable[[], str] | str,
        role_id: str,
) -> None:
    """本阶段 expect 该 Role 的日志 producer。"""
    task = task_id() if callable(task_id) else task_id
    stage = stage_name() if callable(stage_name) else stage_name
    manager.expect_producer(task, stage, log_producer_id(role_id))


def complete_role_logs(
        manager: ArtifactsManager,
        task_id: Callable[[], str] | str,
        stage_name: Callable[[], str] | str,
        role_id: str,
        *,
        remote: bool = False,
) -> None:
    """先 put_log 再 complete；complete 不内嵌在 put_log 里。"""
    task = task_id() if callable(task_id) else task_id
    stage = stage_name() if callable(stage_name) else stage_name
    if remote:
        # 日志位于生产者文件系统，中心不能通过同名路径读取它。
        from common.logger import log_file_path
        from common.log_files import SharedLogFiles

        active = log_file_path(role_id, task)
        snapshots = []
        if active.parent.exists():
            with SharedLogFiles(active).locked() as files:
                snapshots = [(source, source.read_bytes()) for source in files.files()]
        for source, data in snapshots:
            item = ArtifactPut(ArtifactType.LOG, str(source), {
                "role_id": role_id,
                "closed": True,
                "producer_path": str(source),
            })
            manager.put_bytes(task, stage, item, data)
    else:
        manager.put_log(task, role_id, stage)
    manager.complete(task, stage, log_producer_id(role_id))


def gif_producer_id(role_id: str) -> str:
    return f"gif:{role_id}"


def declare_role_gif_producer(
        manager: ArtifactsManager,
        task_id: Callable[[], str] | str,
        stage_name: Callable[[], str] | str,
        role_id: str,
) -> None:
    """本阶段 expect 该 Role 的 GIF producer。"""
    task = task_id() if callable(task_id) else task_id
    stage = stage_name() if callable(stage_name) else stage_name
    manager.expect_producer(task, stage, gif_producer_id(role_id))


def put_role_render_gif(
        manager: ArtifactsManager,
        task_id: str,
        stage_name: str,
        path: str,
        metadata: dict | None = None,
) -> LedgerEntry | None:
    """读本机 GIF 文件，≤上限时经 put_bytes 输送到中心台账；超限跳过不送。"""
    size = os.path.getsize(path)
    if size > PUT_BYTES_MAX_BYTES:
        LOG.warning(
            f"Render GIF exceeds put limit, skip transport: path={path}, "
            f"size={size}, limit={PUT_BYTES_MAX_BYTES}"
        )
        return None
    with open(path, "rb") as handle:
        data = handle.read()
    item = ArtifactPut(ArtifactType.RENDER_GIF, str(path), dict(metadata or {}))
    return manager.put_bytes(task_id, stage_name, item, data)


def complete_role_gifs(
        manager: ArtifactsManager,
        task_id: Callable[[], str] | str,
        stage_name: Callable[[], str] | str,
        role_id: str,
) -> None:
    """本 Role 本阶段不会再产出 GIF。"""
    task = task_id() if callable(task_id) else task_id
    stage = stage_name() if callable(stage_name) else stage_name
    manager.complete(task, stage, gif_producer_id(role_id))


def complete_role_artifact_producers(role: object, *attr_names: str) -> None:
    """按 Role 属性名 complete；测试 stub 缺属性时跳过。"""
    complete_artifact_producers(*(getattr(role, name, None) for name in attr_names))

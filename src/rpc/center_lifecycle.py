"""中心关停 RPC 前的收口标记，供 sample 把关停断连收成 RpcServerGone。"""
from __future__ import annotations

from scheduler.tasks.training_progress import (
    CENTER_FINISHED_ENV,
    is_center_finished,
    publish_center_finished,
)


def mark_rpc_center_finished() -> str:
    """TrainTaskManager 宿主在拆除 RPC 之前写入 DODO_CENTER_FINISHED=1。

    Torch RPC 的关停 EOF 和 gRPC 的 UNAVAILABLE 都能据此收成 RpcServerGone，
    而不是 world 失败或 15 秒重试。
    """
    publish_center_finished()
    return CENTER_FINISHED_ENV


def is_rpc_center_finished() -> bool:
    return is_center_finished()

from __future__ import annotations

import os
import sys
import time
import traceback
from concurrent import futures
from typing import Any, Optional

import grpc

from common.logger import init_logger
from common.exception import (
    RpcCodecException,
    RpcRemoteApplicationException,
    RpcServerGone,
    RpcTransportException,
)
from rpc.grpc.codec import pack_call, pack_value, unpack_call, unpack_value
from rpc.grpc import dodorl_pb2, dodorl_pb2_grpc
from rpc.center_lifecycle import is_rpc_center_finished
from rpc.dispatch import RpcMethodDispatcher, RpcRemoteError

DEFAULT_GRPC_MAX_MESSAGE_LENGTH = 512 * 1024 * 1024
DEFAULT_GRPC_READY_TIMEOUT_SECONDS = 300.0
DEFAULT_GRPC_READY_LOG_INTERVAL_SECONDS = 5.0
DEFAULT_RPC_CALL_TIMEOUT_SECONDS = 60.0

LOG = init_logger("rpc")

_CENTER_FINISHED_STATUS_CODES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.CANCELLED,
    grpc.StatusCode.UNKNOWN,
    grpc.StatusCode.ABORTED,
    grpc.StatusCode.INTERNAL,
    grpc.StatusCode.DATA_LOSS,
}


class _BaseServicer:
    """各RPC服务共用的Call实现；不同服务只替换dispatcher。"""

    def __init__(self, dispatcher: RpcMethodDispatcher):
        self.dispatcher = dispatcher

    def Call(self, request, context):  # noqa: N802 - gRPC生成接口使用大写方法名
        cancel_var, token = None, None
        try:
            if request.method == "reacquire_simulation":
                module = sys.modules.get("dodo_sim.instances.base")
                if module is not None:
                    cancel_var = module.cancel_check
                    token = cancel_var.set(lambda: not context.is_active())
            args, kwargs = unpack_call(request.payload)
            payload = pack_value(self.dispatcher.invoke(request.method, args, kwargs))
            return dodorl_pb2.RpcResponse(ok=True, payload=payload)
        except Exception as exc:
            return dodorl_pb2.RpcResponse(
                ok=False,
                error_type=exc.__class__.__name__,
                error_message=str(exc),
                traceback=traceback.format_exc(),
            )
        finally:
            if token is not None:
                cancel_var.reset(token)


class ReplayBufferRpcServicer(_BaseServicer, dodorl_pb2_grpc.ReplayBufferRpcServicer):
    """ReplayBuffer gRPC服务端实现。"""


class TrainTaskManagerRpcServicer(_BaseServicer, dodorl_pb2_grpc.TrainTaskManagerRpcServicer):
    """TrainTaskManager gRPC服务端实现。"""


class ModelManagerRpcServicer(_BaseServicer, dodorl_pb2_grpc.ModelManagerRpcServicer):
    """ModelManager gRPC服务端实现。"""


class SimManagerRpcServicer(_BaseServicer, dodorl_pb2_grpc.SimManagerRpcServicer):
    """SimManager gRPC服务端实现。"""


class RpcClient:
    """通用RPC客户端，按服务端暴露的方法名发起远程调用。"""

    # 服务端断连后的重试次数与间隔；连不上即视为训练结束，进程据此自然退出。
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_INTERVAL_SECONDS = 5.0

    def __init__(
            self,
            stub: Any,
            max_retries: int = DEFAULT_MAX_RETRIES,
            retry_interval: float = DEFAULT_RETRY_INTERVAL_SECONDS,
            service_name: str = "RPC",
            call_timeout: Optional[float] = None,
            method_timeouts: dict[str, float | None] | None = None,
    ):
        self.stub = stub
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.service_name = service_name
        self.call_timeout = DEFAULT_RPC_CALL_TIMEOUT_SECONDS if call_timeout is None else (
            None if call_timeout <= 0 else float(call_timeout)
        )
        self.method_timeouts = dict(method_timeouts or {})

    def call(self, method: str, *args, **kwargs):
        try:
            request = dodorl_pb2.RpcRequest(method=method, payload=pack_call(*args, **kwargs))
        except RpcCodecException as exc:
            raise RpcCodecException(
                service=self.service_name,
                method=method,
                detail=exc.detail,
            ) from exc
        attempt = 0
        while True:
            attempt += 1
            try:
                timeout = self.method_timeouts.get(method, self.call_timeout)
                response = self.stub.Call(request, timeout=timeout)
            except grpc.RpcError as exc:
                status_code = exc.code()
                detail = str(exc.details() if hasattr(exc, "details") else exc)
                if is_rpc_center_finished() and status_code in _CENTER_FINISHED_STATUS_CODES:
                    LOG.info(
                        "RPC center finished; treating gRPC disconnect as server gone: "
                        f"service={self.service_name}, method={method}, "
                        f"status={status_code}, detail={detail}"
                    )
                    raise RpcServerGone(
                        service=self.service_name,
                        method=method,
                        attempts=attempt,
                        interval=0.0,
                        detail=detail,
                    ) from exc
                if status_code == grpc.StatusCode.UNAVAILABLE and attempt < self.max_retries:
                    LOG.warning(
                        f"RPC server unavailable, retry {attempt}/{self.max_retries} after "
                        f"{self.retry_interval}s: service={self.service_name}, method={method}"
                    )
                    time.sleep(self.retry_interval)
                    continue
                if status_code == grpc.StatusCode.UNAVAILABLE and self.service_name == "TrainTaskManager":
                    raise RpcServerGone(
                        service=self.service_name,
                        method=method,
                        attempts=attempt,
                        interval=self.retry_interval,
                        detail=detail,
                    ) from exc
                if status_code in {
                    grpc.StatusCode.CANCELLED,
                    grpc.StatusCode.UNKNOWN,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    grpc.StatusCode.ABORTED,
                    grpc.StatusCode.INTERNAL,
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DATA_LOSS,
                }:
                    raise RpcTransportException(
                        service=self.service_name,
                        method=method,
                        detail=detail,
                        status_code=status_code,
                    ) from exc
                raise RpcRemoteApplicationException(
                    service=self.service_name,
                    method=method,
                    detail=f"{status_code}: {detail}",
                    remote_error_type="grpc.RpcError",
                ) from exc
            if not response.ok:
                if response.error_type in {
                    "RpcCodecException", "PicklingError", "UnpicklingError", "EOFError",
                }:
                    raise RpcCodecException(
                        service=self.service_name,
                        method=method,
                        detail=f"{response.error_type}: {response.error_message}",
                    )
                raise RpcRemoteError(
                    response.error_type,
                    response.error_message,
                    response.traceback,
                    service=self.service_name,
                    method=method,
                )
            try:
                return unpack_value(response.payload)
            except RpcCodecException as exc:
                raise RpcCodecException(
                    service=self.service_name,
                    method=method,
                    detail=exc.detail,
                ) from exc


def _grpc_ready_timeout_from_env() -> Optional[float]:
    """读取RPC连接等待超时；小于等于0表示一直等待。"""
    raw_value = os.getenv("DODORL_GRPC_READY_TIMEOUT_SECONDS")
    if raw_value is None or raw_value == "":
        return DEFAULT_GRPC_READY_TIMEOUT_SECONDS
    timeout = float(raw_value)
    return None if timeout <= 0 else timeout


def wait_for_grpc_ready(
        channel,
        target: str,
        timeout: Optional[float] = None,
        service_name: str = "gRPC",
) -> None:
    """
    阻塞等待远程gRPC服务可用。

    分布式启动时，各进程没有严格先后顺序。消费方在构造RPC Client时等待服务端
    READY，可以避免第一通业务RPC因为端口尚未监听而直接失败。
    """
    if timeout is None:
        timeout = _grpc_ready_timeout_from_env()
    start_time = time.monotonic()
    next_log_time = start_time
    LOG.info(f"Waiting for {service_name} service ready: target={target}, timeout={timeout}s")
    while True:
        now = time.monotonic()
        cost_secs = now - start_time
        wait_timeout = 1.0
        if timeout is not None:
            remaining_timeout = timeout - cost_secs
            if remaining_timeout <= 0:
                raise TimeoutError(
                    f"gRPC service is not ready: service={service_name}, target={target}, timeout={timeout}s"
                )
            wait_timeout = min(wait_timeout, remaining_timeout)
        try:
            grpc.channel_ready_future(channel).result(timeout=wait_timeout)
            cost_secs = time.monotonic() - start_time
            LOG.info(f"{service_name} service is ready: target={target}, cost={cost_secs:.1f}s")
            return
        except grpc.FutureTimeoutError as exc:
            now = time.monotonic()
            cost_secs = now - start_time
            if timeout is not None and cost_secs >= timeout:
                raise TimeoutError(
                    f"gRPC service is not ready: service={service_name}, target={target}, timeout={timeout}s"
                ) from exc
            if now >= next_log_time:
                LOG.info(f"{service_name} service is not ready, keep waiting: target={target}, waited={cost_secs:.1f}s")
                next_log_time = now + DEFAULT_GRPC_READY_LOG_INTERVAL_SECONDS


def create_grpc_server(
        max_workers: int = 16,
        max_message_length: int = DEFAULT_GRPC_MAX_MESSAGE_LENGTH,
):
    """创建统一gRPC服务进程；调用方继续 add_*_service 后 start。"""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", max_message_length),
            ("grpc.max_receive_message_length", max_message_length),
            # 中心服务持有有状态的TaskManager、ReplayBuffer和ModelManager，同一端口绝不能
            # 被第二个进程共享，否则客户端请求会被内核分流到两套互不一致的训练状态。
            ("grpc.so_reuseport", 0),
        ],
    )
    add_insecure_port = server.add_insecure_port

    def checked_add_insecure_port(address: str) -> int:
        try:
            bound_port = add_insecure_port(address)
        except RuntimeError as exc:
            raise RuntimeError(f"gRPC服务端口绑定失败：地址={address}") from exc
        if bound_port == 0:
            raise RuntimeError(f"gRPC服务端口绑定失败：地址={address}")
        return bound_port

    server.add_insecure_port = checked_add_insecure_port
    return server


def serve_forever(server, address: str) -> None:
    """启动gRPC服务并阻塞当前线程。"""
    bound_port = server.add_insecure_port(address)
    if bound_port == 0:
        raise RuntimeError(f"gRPC服务端口绑定失败：地址={address}")
    server.start()
    server.wait_for_termination()

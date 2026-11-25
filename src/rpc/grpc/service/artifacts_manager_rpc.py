from __future__ import annotations

import grpc

from common.process_shutdown import process_close
from rpc.dispatch import ARTIFACTS_MANAGER, build_dispatcher
from rpc.grpc import dodorl_pb2
from rpc.grpc.service import (
    DEFAULT_GRPC_MAX_MESSAGE_LENGTH,
    RpcClient,
    _BaseServicer,
    wait_for_grpc_ready,
)
from scheduler.artifact import ArtifactsManager

ARTIFACTS_MANAGER_SERVICE_NAME = "dodorl.rpc.ArtifactsManagerRpc"


def add_artifacts_manager_service(server, artifacts_manager: ArtifactsManager) -> None:
    """把 ArtifactsManager 实例挂到 gRPC server。"""
    dispatcher = build_dispatcher(ARTIFACTS_MANAGER, artifacts_manager)
    servicer = _ArtifactsManagerRpcServicer(dispatcher)
    handler = grpc.method_handlers_generic_handler(
        ARTIFACTS_MANAGER_SERVICE_NAME,
        {
            "Call": grpc.unary_unary_rpc_method_handler(
                servicer.Call,
                request_deserializer=dodorl_pb2.RpcRequest.FromString,
                response_serializer=dodorl_pb2.RpcResponse.SerializeToString,
            )
        },
    )
    server.add_generic_rpc_handlers((handler,))


class _ArtifactsManagerRpcServicer(_BaseServicer):
    """ArtifactsManager gRPC 服务端：只转发白名单方法。"""


class _ArtifactsManagerRpcStub:
    """不依赖生成代码的轻量 stub，路径与 dodorl.proto 中的服务名保持一致。"""

    def __init__(self, channel):
        self.Call = channel.unary_unary(
            f"/{ARTIFACTS_MANAGER_SERVICE_NAME}/Call",
            request_serializer=dodorl_pb2.RpcRequest.SerializeToString,
            response_deserializer=dodorl_pb2.RpcResponse.FromString,
        )


class ArtifactsManagerRpcClient:
    """薄转发 stub：不含读盘或拼 content。"""

    def __init__(self, target: str, call_timeout: float | None = None):
        self.channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
            ],
        )
        wait_for_grpc_ready(self.channel, target, service_name="ArtifactsManager")
        self.client = RpcClient(
            _ArtifactsManagerRpcStub(self.channel),
            service_name="ArtifactsManager",
            call_timeout=call_timeout,
        )

    def register(self, task_id, stage_name, intent):
        return self.client.call("register", task_id, stage_name, intent)

    def put(self, task_id, stage_name, item):
        return self.client.call("put", task_id, stage_name, item)

    def put_bytes(self, task_id, stage_name, item, data):
        return self.client.call("put_bytes", task_id, stage_name, item, data)

    def put_log(self, task_id, role_id, stage_name=None):
        return self.client.call("put_log", task_id, role_id, stage_name)

    def expect_producer(self, task_id, stage_name, producer_id):
        return self.client.call("expect_producer", task_id, stage_name, producer_id)

    def complete(self, task_id, stage_name, producer_id):
        return self.client.call("complete", task_id, stage_name, producer_id)

    @process_close
    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.channel.close()

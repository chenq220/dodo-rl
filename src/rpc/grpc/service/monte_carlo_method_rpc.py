from __future__ import annotations

import grpc

from common.process_shutdown import process_close
from rpc.grpc import dodorl_pb2
from rpc.interfaces import MonteCarloMethodPort
from rpc.dispatch import build_dispatcher, MONTE_CARLO_METHOD
from rpc.grpc.service import (
    DEFAULT_GRPC_MAX_MESSAGE_LENGTH,
    _BaseServicer,
    RpcClient,
    wait_for_grpc_ready,
)
from sample.monte_carlo import MonteCarloMethod

MONTE_CARLO_METHOD_SERVICE_NAME = "dodorl.rpc.MonteCarloMethodRpc"


def add_monte_carlo_method_service(server, monte_carlo_method: MonteCarloMethod) -> None:
    """把MonteCarloMethod实例挂到gRPC server。"""
    dispatcher = build_dispatcher(MONTE_CARLO_METHOD, monte_carlo_method)
    servicer = _MonteCarloMethodRpcServicer(dispatcher)
    handler = grpc.method_handlers_generic_handler(
        MONTE_CARLO_METHOD_SERVICE_NAME,
        {
            "Call": grpc.unary_unary_rpc_method_handler(
                servicer.Call,
                request_deserializer=dodorl_pb2.RpcRequest.FromString,
                response_serializer=dodorl_pb2.RpcResponse.SerializeToString,
            )
        },
    )
    server.add_generic_rpc_handlers((handler,))


class _MonteCarloMethodRpcServicer(_BaseServicer):
    """MonteCarloMethod gRPC服务端实现。"""


class _MonteCarloMethodRpcStub:
    """不依赖生成代码的轻量stub，路径与dodorl.proto中的服务名保持一致。"""

    def __init__(self, channel):
        self.Call = channel.unary_unary(
            f"/{MONTE_CARLO_METHOD_SERVICE_NAME}/Call",
            request_serializer=dodorl_pb2.RpcRequest.SerializeToString,
            response_deserializer=dodorl_pb2.RpcResponse.FromString,
        )


class MonteCarloMethodRpcClient(MonteCarloMethodPort):
    """MonteCarloMethod远程客户端；采样进程通过它把样本记录推给中心录制器。"""

    def __init__(self, target: str, call_timeout: float | None = None):
        self.channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
            ],
        )
        wait_for_grpc_ready(self.channel, target, service_name="MonteCarloMethod")
        self.client = RpcClient(
            _MonteCarloMethodRpcStub(self.channel),
            service_name="MonteCarloMethod",
            call_timeout=call_timeout,
        )

    def receiver(self, payload):
        return self.client.call("receiver", payload)

    def state(self):
        return self.client.call("state")

    def get_recorded_episode(self, *, stage_name: str, episode_num: int):
        return self.client.call(
            "get_recorded_episode",
            stage_name=stage_name,
            episode_num=int(episode_num),
        )

    @process_close
    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.channel.close()

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.sim_manager import SimManager


import grpc

from common.exception import RpcServerGone
from common.process_shutdown import process_close
from rpc.grpc import dodorl_pb2_grpc
from rpc.interfaces import SimManagerPort
from rpc.dispatch import build_dispatcher, SIM_MANAGER
from rpc.grpc.service import (
    DEFAULT_GRPC_MAX_MESSAGE_LENGTH,
    RpcClient,
    SimManagerRpcServicer,
    wait_for_grpc_ready,
)


def add_sim_manager_service(server, sim_manager: SimManager) -> None:
    """把SimManager实例挂到gRPC server。"""
    dispatcher = build_dispatcher(SIM_MANAGER, sim_manager)
    dodorl_pb2_grpc.add_SimManagerRpcServicer_to_server(
        SimManagerRpcServicer(dispatcher),
        server,
    )


class SimManagerRpcClient(SimManagerPort):
    """SimManager远程客户端；采样进程通过它申请和释放仿真资源。"""

    def __init__(self, target: str, call_timeout: float | None = None):
        self.channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
            ],
        )
        wait_for_grpc_ready(self.channel, target, service_name="SimManager")
        self.client = RpcClient(
            dodorl_pb2_grpc.SimManagerRpcStub(self.channel),
            service_name="SimManager",
            call_timeout=call_timeout,
            method_timeouts={"reacquire_simulation": 600.0},
        )

    def get_sim_info(self, sample_id: str, sim_type: str | None = None, *, request_id=None):
        if request_id is not None:
            return self.client.call("get_sim_info", sample_id, sim_type=sim_type, request_id=request_id)
        if sim_type is not None:
            return self.client.call("get_sim_info", sample_id, sim_type=sim_type)
        return self.client.call("get_sim_info", sample_id)

    def release_sim(self, sim_id: str, **lease):
        return self.client.call("release_sim", sim_id, **lease)

    def reacquire_simulation(self, **request):
        return self.client.call("reacquire_simulation", **request)

    def validate_sim_lease(self, sim_id, **lease):
        return self.client.call("validate_sim_lease", sim_id, **lease)

    def get_sim_list(self):
        return self.client.call("get_sim_list")

    @process_close
    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.client.call("close")
        except RpcServerGone:
            # 中心已确认收口，远端管理器随中心释放；本地关闭仍须完成。
            pass
        finally:
            self.channel.close()

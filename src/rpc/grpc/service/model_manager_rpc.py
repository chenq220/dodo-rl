from __future__ import annotations

import grpc

from common.process_shutdown import process_close
from models import ModelManager
from rpc.grpc import dodorl_pb2_grpc
from rpc.interfaces import ModelManagerPort
from rpc.dispatch import build_dispatcher, normalize_models_payload, MODEL_MANAGER
from rpc.grpc.service import (
    DEFAULT_GRPC_MAX_MESSAGE_LENGTH,
    ModelManagerRpcServicer,
    RpcClient,
    wait_for_grpc_ready,
)


def add_model_manager_service(server, model_manager: ModelManager) -> None:
    """把ModelManager实例挂到gRPC server。"""
    dispatcher = build_dispatcher(MODEL_MANAGER, model_manager)
    dodorl_pb2_grpc.add_ModelManagerRpcServicer_to_server(
        ModelManagerRpcServicer(dispatcher),
        server,
    )


class ModelManagerRpcClient(ModelManagerPort):
    """ModelManager远程客户端；模型权重按state_dict传输，不跨进程传nn.Module。"""

    def __init__(self, target: str, call_timeout: float | None = None):
        self.channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
            ],
        )
        wait_for_grpc_ready(self.channel, target, service_name="ModelManager")
        self.client = RpcClient(
            dodorl_pb2_grpc.ModelManagerRpcStub(self.channel),
            service_name="ModelManager",
            call_timeout=call_timeout,
        )

    @property
    def model_list(self):
        return self.client.call("model_list")

    def submit_model(self, model_name, algo_type, models, **kwargs):
        return self.client.call(
            "submit_model",
            model_name,
            algo_type,
            normalize_models_payload(models),
            **kwargs,
        )

    def register_model(self, model_info):
        return self.client.call("register_model", model_info)

    def get_model(self, model_name, *, version=None, output_type="state_dict", artifact_names=None):
        if output_type == "nn":
            raise ValueError("RPC ModelManager client does not support output_type='nn'; use 'state_dict' or 'info'.")
        return self.client.call(
            "get_model",
            model_name,
            version=version,
            output_type=output_type,
            artifact_names=artifact_names,
        )

    def get_champion(self, model_name):
        return self.client.call("get_champion", model_name)

    def promote_model(self, model_name, version=None):
        return self.client.call("promote_model", model_name, version=version)

    def archive_model(self, model_name, version):
        return self.client.call("archive_model", model_name, version)

    def quarantine_model(self, model_name, version):
        return self.client.call("quarantine_model", model_name, version)

    def mark_specialist(self, model_name, version, tags=None):
        return self.client.call("mark_specialist", model_name, version, tags=tags)

    def rollback_model(self, model_name, version=None):
        return self.client.call("rollback_model", model_name, version=version)

    def list_versions(self, model_name=None, status=None):
        return self.client.call("list_versions", model_name=model_name, status=status)

    def get_agent_models(self):
        return self.client.call("get_agent_models")

    def save_models(self, save_model_path, save_model_name, *, agent_name, version=None):
        return self.client.call(
            "save_models",
            save_model_path,
            save_model_name,
            agent_name=agent_name,
            version=version,
        )

    @process_close
    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.channel.close()

from __future__ import annotations

import os
from typing import Any, Mapping

from common.process_shutdown import process_close
from config import GrpcConfig, TorchRpcConfig
from rpc.center_lifecycle import mark_rpc_center_finished
from rpc.dispatch import (
    ARTIFACTS_MANAGER,
    MODEL_MANAGER,
    MONTE_CARLO_METHOD,
    REPLAY_BUFFER,
    SIM_MANAGER,
    TRAIN_TASK_MANAGER,
)

RpcConfiguration = GrpcConfig | TorchRpcConfig

_LISTEN_HOST_ENV = "DODO_RPC_LISTEN_HOST"
_CONNECT_HOST_ENV = "DODO_RPC_CONNECT_HOST"


def rewrite_rpc_address_host(address: str, host: str | None) -> str:
    """只替换 host:port 的 host；host 为空则原样返回。"""
    if not host:
        return address
    stripped = address.removeprefix("tcp://")
    _prefix, separator, port = stripped.rpartition(":")
    if not separator or not port:
        return address
    return f"{host}:{port}"


def _env_host(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _grpc_target(config: GrpcConfig, service_name: str, *, listen: bool = False) -> str:
    address = {
        TRAIN_TASK_MANAGER: config.task_manager_address,
        REPLAY_BUFFER: config.replay_buffer_address,
        MODEL_MANAGER: config.model_manager_address,
        SIM_MANAGER: config.sim_manager_address,
        MONTE_CARLO_METHOD: config.monte_carlo_method_address,
        ARTIFACTS_MANAGER: config.artifacts_manager_address,
    }[service_name]
    env_name = _LISTEN_HOST_ENV if listen else _CONNECT_HOST_ENV
    return rewrite_rpc_address_host(address, _env_host(env_name))


def create_rpc_client(config: RpcConfiguration, service_name: str):
    if isinstance(config, GrpcConfig):
        from rpc.grpc.service.model_manager_rpc import ModelManagerRpcClient
        from rpc.grpc.service.monte_carlo_method_rpc import MonteCarloMethodRpcClient
        from rpc.grpc.service.replay_buffer_rpc import ReplayBufferRpcClient
        from rpc.grpc.service.sim_manager_rpc import SimManagerRpcClient
        from rpc.grpc.service.train_task_rpc import TrainTaskManagerRpcClient
        from rpc.grpc.service.artifacts_manager_rpc import ArtifactsManagerRpcClient
        client_class = {
            TRAIN_TASK_MANAGER: TrainTaskManagerRpcClient,
            REPLAY_BUFFER: ReplayBufferRpcClient,
            MODEL_MANAGER: ModelManagerRpcClient,
            SIM_MANAGER: SimManagerRpcClient,
            MONTE_CARLO_METHOD: MonteCarloMethodRpcClient,
            ARTIFACTS_MANAGER: ArtifactsManagerRpcClient,
        }[service_name]
        return client_class(
            _grpc_target(config, service_name), call_timeout=config.call_timeout_seconds,
        )
    if isinstance(config, TorchRpcConfig):
        from rpc.torch_rpc.client import (
            ArtifactsManagerTorchRpcClient,
            ModelManagerTorchRpcClient,
            MonteCarloMethodTorchRpcClient,
            ReplayBufferTorchRpcClient,
            SimManagerTorchRpcClient,
            TrainTaskManagerTorchRpcClient,
        )
        client_class = {
            TRAIN_TASK_MANAGER: TrainTaskManagerTorchRpcClient,
            REPLAY_BUFFER: ReplayBufferTorchRpcClient,
            MODEL_MANAGER: ModelManagerTorchRpcClient,
            SIM_MANAGER: SimManagerTorchRpcClient,
            MONTE_CARLO_METHOD: MonteCarloMethodTorchRpcClient,
            ARTIFACTS_MANAGER: ArtifactsManagerTorchRpcClient,
        }[service_name]
        if service_name == REPLAY_BUFFER:
            return client_class(config=config)
        return client_class(call_timeout=config.call_timeout_seconds)
    raise TypeError(f"Unsupported RPC configuration: {type(config).__name__}")


class RpcServiceHost:
    """协议无关的中心模块服务宿主。"""

    def __init__(
            self,
            config: RpcConfiguration,
            worker_name: str,
            services: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.worker_name = worker_name
        self.services = {name: target for name, target in services.items() if target is not None}
        self._servers: list[Any] = []
        self._started = False
        self._stopped = False

    def start(self) -> None:
        if self._started:
            return
        if isinstance(self.config, TorchRpcConfig):
            from rpc.torch_rpc.runtime import init_runtime
            init_runtime(
                self.config,
                self.worker_name,
                services=self.services,
            )
            self._started = True
            return
        if not isinstance(self.config, GrpcConfig):
            raise TypeError(f"Unsupported RPC configuration: {type(self.config).__name__}")
        from rpc.grpc.server import (
            build_artifacts_manager_server,
            build_model_server,
            build_monte_carlo_method_server,
            build_replay_buffer_server,
            build_sim_manager_server,
            build_train_task_manager_server,
        )
        builders = {
            TRAIN_TASK_MANAGER: build_train_task_manager_server,
            REPLAY_BUFFER: build_replay_buffer_server,
            MODEL_MANAGER: build_model_server,
            SIM_MANAGER: build_sim_manager_server,
            MONTE_CARLO_METHOD: build_monte_carlo_method_server,
            ARTIFACTS_MANAGER: build_artifacts_manager_server,
        }
        for service_name, target in self.services.items():
            server = builders[service_name](target)
            address = _grpc_target(self.config, service_name, listen=True)
            bound_port = server.add_insecure_port(address)
            if bound_port == 0:
                self.stop()
                raise RuntimeError(f"Failed to bind {service_name} RPC server: address={address}")
            server.start()
            self._servers.append(server)
        self._started = True

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        # TrainTaskManager 宿主先写收口标记再拆服务，避免 sample 把关停断连判成异常。
        if self._started and TRAIN_TASK_MANAGER in self.services:
            mark_rpc_center_finished()
        if isinstance(self.config, TorchRpcConfig):
            if self._started:
                from rpc.torch_rpc.runtime import shutdown_runtime
                shutdown_runtime()
        else:
            for server in reversed(self._servers):
                server.stop(0)
            self._servers.clear()
        self._started = False

    @process_close
    def close(self) -> None:
        self.stop()


def init_rpc_client_runtime(config: RpcConfiguration, worker_name: str) -> None:
    if isinstance(config, TorchRpcConfig):
        from rpc.torch_rpc.runtime import init_runtime
        init_runtime(config, worker_name)


def shutdown_rpc_client_runtime(config: RpcConfiguration) -> None:
    if isinstance(config, TorchRpcConfig):
        from rpc.torch_rpc.runtime import shutdown_runtime
        shutdown_runtime()

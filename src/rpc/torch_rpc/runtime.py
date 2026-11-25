from __future__ import annotations

import os
import re
import sys
import time
import threading
from typing import Any, Mapping

import torch.distributed.rpc as torch_rpc

from common.logger import init_logger
from config import TorchRpcConfig
from rpc.dispatch import (
    ARTIFACTS_MANAGER,
    MODEL_MANAGER,
    MONTE_CARLO_METHOD,
    REPLAY_BUFFER,
    SIM_MANAGER,
    TRAIN_TASK_MANAGER,
    RpcMethodDispatcher,
    RpcResponseEnvelope,
    build_dispatcher,
    prepare_torch_rpc_payload,
)
from rpc.center_lifecycle import mark_rpc_center_finished
from rpc.factory import rewrite_rpc_address_host
from rpc.torch_rpc.replay_push_permit import TorchReplayBufferService

LOG = init_logger("torch_rpc")

_DISPATCHERS: dict[str, RpcMethodDispatcher] = {}
_LOCK = threading.RLock()
_INITIALIZED = False
_WORKER_NAME = ""
_RANK: int | None = None
_WORLD_SIZE = 0
_SERVICE_RANKS: dict[str, int] = {}

_CONTROL_SERVICES = {
    TRAIN_TASK_MANAGER,
    REPLAY_BUFFER,
    SIM_MANAGER,
    MONTE_CARLO_METHOD,
    ARTIFACTS_MANAGER,
}
_ALL_SERVICES = _CONTROL_SERVICES | {MODEL_MANAGER}


def resolve_service_ranks(
        rank: int,
        worker_name: str,
        hosted_services: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """按启动器的连续 Rank 约定推导服务归属，不从 Role 或 TOML 读取路由。"""
    service_names = set(hosted_services or {})
    if MODEL_MANAGER in service_names and service_names & _CONTROL_SERVICES:
        # 1+N 合并中心：Rank 0 同时持有控制、学习和模型服务。
        return {name: 0 for name in _ALL_SERVICES}
    if service_names == {MODEL_MANAGER}:
        # 3+N 独立 ModelRole 固定为 Rank 2。
        return {name: (2 if name == MODEL_MANAGER else 0) for name in _ALL_SERVICES}
    if service_names & _CONTROL_SERVICES:
        # 3+N Control/StageControl 固定为 Rank 0，ModelManager 在 Rank 2。
        return {name: (2 if name == MODEL_MANAGER else 0) for name in _ALL_SERVICES}

    if re.search(r"(?:^|-)learn-\d+$", worker_name):
        if rank != 1:
            raise ValueError(f"3+N LearnRole must use Rank 1, worker={worker_name!r}, rank={rank}")
        return {name: (2 if name == MODEL_MANAGER else 0) for name in _ALL_SERVICES}

    sample_match = re.fullmatch(r"sample-(\d+)", worker_name)
    if sample_match is not None:
        sample_index = int(sample_match.group(1))
        rank_offset = rank - sample_index
        if rank_offset == 3:
            return {name: (2 if name == MODEL_MANAGER else 0) for name in _ALL_SERVICES}
        if rank_offset == 1:
            return {name: 0 for name in _ALL_SERVICES}
        raise ValueError(
            f"SampleRole Rank must equal sample_index + 1 (1+N) or sample_index + 3 (3+N), "
            f"worker={worker_name!r}, rank={rank}"
        )

    # 保持现有 1+N 及自定义测试 worker 名称兼容：未识别客户端仍访问 Rank 0。
    return {name: 0 for name in _ALL_SERVICES}


def service_rank(service_name: str) -> int:
    """返回当前进程视角下的服务 Rank；runtime 初始化前保持旧版 Rank 0 默认值。"""
    with _LOCK:
        return _SERVICE_RANKS.get(service_name, 0)


def _remote_call(
        service_name: str,
        method: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
) -> RpcResponseEnvelope:
    with _LOCK:
        dispatcher = _DISPATCHERS.get(service_name)
    if dispatcher is None:
        return RpcResponseEnvelope(
            ok=False,
            error_type="LookupError",
            error_message=f"RPC service is not registered: {service_name}",
        )
    # TensorPipe 默认只保证 CPU Tensor 可传输。请求侧已做同样处理；响应侧也统一
    # CPU 化，确保任意服务实现误返回 MPS/CUDA Tensor 时不会击穿整个 RPC world。
    module = sys.modules.get("dodo_sim.instances.base") if service_name == SIM_MANAGER and method == "reacquire_simulation" else None
    token = None
    if module is not None:
        deadline = time.monotonic() + 600.0
        token = module.cancel_check.set(lambda: time.monotonic() >= deadline)
    try:
        return prepare_torch_rpc_payload(dispatcher.invoke_enveloped(method, args, kwargs))
    finally:
        if token is not None:
            module.cancel_check.reset(token)


def register_services(services: Mapping[str, Any], config: TorchRpcConfig) -> None:
    with _LOCK:
        if _INITIALIZED:
            raise RuntimeError("Torch RPC services must be registered before init_rpc")
        _DISPATCHERS.clear()
        wrapped_services = {
            name: TorchReplayBufferService(target, config) if name == REPLAY_BUFFER else target
            for name, target in services.items()
        }
        _DISPATCHERS.update({
            name: build_dispatcher(name, target) for name, target in wrapped_services.items()
        })


def rank_from_env(config: TorchRpcConfig) -> int:
    raw_rank = os.getenv("RANK")
    if raw_rank is None or raw_rank == "":
        raise ValueError("Torch RPC requires the RANK environment variable")
    try:
        rank = int(raw_rank)
    except ValueError as exc:
        raise ValueError(f"Torch RPC RANK must be an integer, got {raw_rank!r}") from exc
    if not 0 <= rank < config.world_size:
        raise ValueError(f"Torch RPC RANK must be in [0, {config.world_size}), got {rank}")
    return rank


def init_runtime(
        config: TorchRpcConfig,
        worker_name: str,
        *,
        services: Mapping[str, Any] | None = None,
        require_center: bool | None = None,
) -> int:
    global _INITIALIZED, _WORKER_NAME, _RANK, _SERVICE_RANKS, _WORLD_SIZE
    rank = rank_from_env(config)
    if require_center is True and rank != 0:
        raise ValueError(f"central Torch RPC role must use RANK=0, got {rank}")
    if require_center is False and rank == 0:
        raise ValueError("sample Torch RPC role cannot use RANK=0")
    if services is not None:
        register_services(services, config)
    resolved_service_ranks = resolve_service_ranks(rank, worker_name, services)
    for service_name in services or {}:
        expected_rank = resolved_service_ranks[service_name]
        if expected_rank != rank:
            raise ValueError(
                f"Torch RPC service {service_name} must be hosted at Rank {expected_rank}, "
                f"worker={worker_name!r} uses Rank {rank}"
            )
    with _LOCK:
        if _INITIALIZED:
            return rank
        connect_host = os.environ.get("DODO_RPC_CONNECT_HOST", "").strip() or None
        init_address = rewrite_rpc_address_host(config.init_address, connect_host)
        options = torch_rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=int(config.num_worker_threads),
            rpc_timeout=0.0 if config.call_timeout_seconds <= 0 else float(config.call_timeout_seconds),
            init_method=f"tcp://{init_address.removeprefix('tcp://')}",
        )
        LOG.info(
            f"Initializing Torch RPC: worker={worker_name}, rank={rank}, "
            f"world_size={config.world_size}, init_address={init_address}"
        )
        torch_rpc.init_rpc(
            worker_name,
            rank=rank,
            world_size=int(config.world_size),
            backend=torch_rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        _INITIALIZED = True
        _WORKER_NAME = worker_name
        _RANK = rank
        _WORLD_SIZE = int(config.world_size)
        _SERVICE_RANKS = resolved_service_ranks
    LOG.info(f"Torch RPC ready: worker={worker_name}, rank={rank}, world_size={config.world_size}")
    return rank


def shutdown_runtime() -> None:
    global _INITIALIZED, _WORKER_NAME, _RANK, _SERVICE_RANKS, _WORLD_SIZE
    with _LOCK:
        if not _INITIALIZED:
            _DISPATCHERS.clear()
            _SERVICE_RANKS = {}
            return
        rank = _RANK
        world_size = _WORLD_SIZE
        _INITIALIZED = False
    # 先落盘再拆管：sample 随后看到的 EOF 才能被收成 RpcServerGone，而不是 world 失败。
    if rank == 0:
        mark_rpc_center_finished()
        # 独立容器没有共享进度文件，通过仍存活的 RPC 通知每个进程正常收口。
        for worker_rank in range(1, world_size):
            try:
                torch_rpc.rpc_sync(worker_rank, mark_rpc_center_finished, timeout=5.0)
            except Exception as exc:
                LOG.warning(f"Failed to notify RPC worker of center shutdown: rank={worker_rank}, error={exc}")
    try:
        torch_rpc.shutdown(graceful=False)
    finally:
        with _LOCK:
            _DISPATCHERS.clear()
            _WORKER_NAME = ""
            _RANK = None
            _WORLD_SIZE = 0
            _SERVICE_RANKS = {}
        LOG.info("Torch RPC runtime stopped")


def is_initialized() -> bool:
    with _LOCK:
        return _INITIALIZED


def worker_context() -> tuple[str, int | None]:
    with _LOCK:
        return _WORKER_NAME, _RANK

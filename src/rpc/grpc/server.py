from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.sim_manager import SimManager


from models import ModelManager
from rpc.grpc.service.artifacts_manager_rpc import add_artifacts_manager_service
from rpc.grpc.service.model_manager_rpc import add_model_manager_service
from rpc.grpc.service.monte_carlo_method_rpc import add_monte_carlo_method_service
from rpc.grpc.service.replay_buffer_rpc import add_replay_buffer_service
from rpc.grpc.service.sim_manager_rpc import add_sim_manager_service
from rpc.grpc.service import create_grpc_server, serve_forever
from rpc.grpc.service.train_task_rpc import add_train_task_manager_service
from sample.replay_buffer import ReplayBuffer
from sample.monte_carlo import MonteCarloMethod
from scheduler.artifact import ArtifactsManager
from scheduler.train_task import TrainTaskManager


def build_train_task_manager_server(
        task_manager: TrainTaskManager,
        *,
        max_workers: int = 16,
):
    """
    构造TrainTaskManager RPC服务。

    即使TrainTaskManager和ReplayBuffer部署在同一个控制容器里，也保持一模块一服务；
    ReplayBuffer满足学习条件后通过RPC或本地回调通知该服务释放学习令牌。
    """
    server = create_grpc_server(max_workers=max_workers)
    add_train_task_manager_service(server, task_manager)
    return server


def build_replay_buffer_server(
        replay_buffer: ReplayBuffer,
        *,
        max_workers: int = 32,
):
    """
    构造ReplayBuffer RPC服务。

    采样进程向该服务写入EpisodeTransition，学习进程从该服务采样训练批次。
    """
    server = create_grpc_server(max_workers=max_workers)
    add_replay_buffer_service(server, replay_buffer)
    return server


def build_model_server(model_manager: ModelManager, *, max_workers: int = 16):
    """
    构造模型进程RPC服务。

    模型进程只负责模型版本寄存和state_dict分发，不跨进程返回nn.Module对象。
    """
    server = create_grpc_server(max_workers=max_workers)
    add_model_manager_service(server, model_manager)
    return server


def build_sim_manager_server(sim_manager: SimManager, *, max_workers: int = 16):
    """
    构造仿真管理RPC服务。

    控制进程持有SimManager，采样进程通过该服务申请/释放仿真连接资源。
    """
    server = create_grpc_server(max_workers=max_workers)
    add_sim_manager_service(server, sim_manager)
    return server


def build_monte_carlo_method_server(monte_carlo_method: MonteCarloMethod, *, max_workers: int = 16):
    """
    构造中心蒙特卡洛方法RPC服务。

    分布式采样进程不再各自写样本文件，只通过该服务把record_sample payload推到控制进程。
    """
    server = create_grpc_server(max_workers=max_workers)
    add_monte_carlo_method_service(server, monte_carlo_method)
    return server


def build_artifacts_manager_server(artifacts_manager: ArtifactsManager, *, max_workers: int = 16):
    """构造 ArtifactsManager RPC 服务；只暴露台账白名单方法。"""
    server = create_grpc_server(max_workers=max_workers)
    add_artifacts_manager_service(server, artifacts_manager)
    return server


def serve_train_task_manager(
        task_manager: TrainTaskManager,
        address: str,
        *,
        max_workers: int = 16,
) -> None:
    """启动TrainTaskManager RPC服务并阻塞。"""
    serve_forever(
        build_train_task_manager_server(task_manager, max_workers=max_workers),
        address,
    )


def serve_replay_buffer(
        replay_buffer: ReplayBuffer,
        address: str,
        *,
        max_workers: int = 32,
) -> None:
    """启动ReplayBuffer RPC服务并阻塞。"""
    serve_forever(
        build_replay_buffer_server(replay_buffer, max_workers=max_workers),
        address,
    )


def serve_model(
        model_manager: ModelManager,
        address: str,
        *,
        max_workers: int = 16,
) -> None:
    """启动模型进程RPC服务并阻塞。"""
    serve_forever(
        build_model_server(model_manager, max_workers=max_workers),
        address,
    )


def serve_sim_manager(
        sim_manager: SimManager,
        address: str,
        *,
        max_workers: int = 16,
) -> None:
    """启动SimManager RPC服务并阻塞。"""
    serve_forever(
        build_sim_manager_server(sim_manager, max_workers=max_workers),
        address,
    )


def serve_monte_carlo_method(
        monte_carlo_method: MonteCarloMethod,
        address: str,
        *,
        max_workers: int = 16,
) -> None:
    """启动中心MonteCarloMethod RPC服务并阻塞。"""
    serve_forever(
        build_monte_carlo_method_server(monte_carlo_method, max_workers=max_workers),
        address,
    )

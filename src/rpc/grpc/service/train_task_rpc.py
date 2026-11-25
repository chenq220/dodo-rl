from __future__ import annotations

import grpc
from typing import Optional

from common.process_shutdown import process_close
from scheduler.artifact import Artifact
from rpc.artifacts import encode_artifacts
from rpc.grpc import dodorl_pb2_grpc
from rpc.interfaces import TrainTaskManagerPort
from rpc.dispatch import build_dispatcher, TRAIN_TASK_MANAGER
from rpc.grpc.service import (
    DEFAULT_GRPC_MAX_MESSAGE_LENGTH,
    RpcClient,
    TrainTaskManagerRpcServicer,
    wait_for_grpc_ready,
)
from scheduler.train_task import TrainTaskManager


def add_train_task_manager_service(server, task_manager: TrainTaskManager) -> None:
    """把TrainTaskManager实例挂到gRPC server。"""
    dispatcher = build_dispatcher(TRAIN_TASK_MANAGER, task_manager)
    dodorl_pb2_grpc.add_TrainTaskManagerRpcServicer_to_server(
        TrainTaskManagerRpcServicer(dispatcher),
        server,
    )


class TrainTaskManagerRpcClient(TrainTaskManagerPort):
    """TrainTaskManager远程客户端；对外保持本地任务管理器接口语义。"""

    def __init__(self, target: str, call_timeout: float | None = None):
        self.channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
            ],
        )
        wait_for_grpc_ready(self.channel, target, service_name="TrainTaskManager")
        self.client = RpcClient(
            dodorl_pb2_grpc.TrainTaskManagerRpcStub(self.channel),
            service_name="TrainTaskManager",
            call_timeout=call_timeout,
        )

    def get_sample_task_info(self, **kwargs):
        return self.client.call("get_sample_task_info", **kwargs)

    def sample_backpressure_state(self, sample_role_name: str = "unknown"):
        return self.client.call("sample_backpressure_state", sample_role_name=sample_role_name)

    def sample_gate_state(self, sample_role_name: str = "unknown"):
        return self.client.call("sample_gate_state", sample_role_name=sample_role_name)

    def report_sample_result(
            self,
            episode_num: int,
            total_rewards: dict[str, float],
            success: bool = True,
            artifacts: Optional[list[Artifact]] = None,
            episode_metrics: Optional[dict[str, float]] = None,
            episode_rewards_details: Optional[dict[str, float]] = None,
            lease_id: str = "",
            step_count: int = 0,
    ):
        return self.client.call(
            "report_sample_result",
            episode_num,
            total_rewards=total_rewards,
            success=success,
            artifacts=encode_artifacts(artifacts),
            episode_metrics=episode_metrics,
            episode_rewards_details=episode_rewards_details,
            lease_id=lease_id,
            step_count=step_count,
        )

    def accept_sample_task_result(self, episode_num: int, lease_id: str) -> bool:
        return self.client.call(
            "accept_sample_task_result", episode_num, lease_id=lease_id,
        )

    def save_reward_curve(self):
        return self.client.call("save_reward_curve")

    def is_sample_completed(self):
        return self.client.call("is_sample_completed")

    def get_learn_task_info(self):
        return self.client.call("get_learn_task_info")

    def report_learning_result(
            self,
            learn_result: dict,
            episode_num: int,
            learn_id: int = 0,
    ):
        return self.client.call(
            "report_learning_result",
            learn_result,
            episode_num=episode_num,
            learn_id=learn_id,
        )

    def is_learning_completed(self):
        return self.client.call("is_learning_completed")

    def get_task_manager_info(self):
        return self.client.call("get_task_manager_info")

    @property
    def sample_task_process(self):
        return self.client.call("sample_task_process")

    @property
    def learning_process(self):
        return self.client.call("learning_process")

    def custom_episode(self, episode_info, **kwargs):
        return self.client.call("custom_episode", episode_info, **kwargs)

    def learn_batches_update(self, replay_buffer_state, episode_num: int):
        return self.client.call("learn_batches_update", replay_buffer_state, episode_num=episode_num)

    def restart_training_task(self, reason: str = ""):
        return self.client.call("restart_training_task", reason=reason)

    def push_info(self, attrs: dict) -> None:
        return self.client.call("push_info", attrs)

    def pull_info(self, key: str | None = None):
        return self.client.call("pull_info", key=key)

    @process_close
    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.channel.close()

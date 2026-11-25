from __future__ import annotations

import grpc

from common.process_shutdown import process_close
from rpc.grpc import dodorl_pb2_grpc
from rpc.interfaces import ReplayBufferPort
from rpc.dispatch import build_dispatcher, REPLAY_BUFFER
from rpc.grpc.service import (
    DEFAULT_GRPC_MAX_MESSAGE_LENGTH,
    ReplayBufferRpcServicer,
    RpcClient,
    wait_for_grpc_ready,
)
from sample.replay_buffer import ReplayBuffer
from sample.replay.replay_payload import encode_replay_push_payload
from sample.replay.replay_perf import now as perf_now, record_seconds


def add_replay_buffer_service(server, replay_buffer: ReplayBuffer) -> None:
    """把ReplayBuffer实例挂到gRPC server。"""
    dispatcher = build_dispatcher(REPLAY_BUFFER, replay_buffer)
    dodorl_pb2_grpc.add_ReplayBufferRpcServicer_to_server(
        ReplayBufferRpcServicer(dispatcher),
        server,
    )


class ReplayBufferRpcClient(ReplayBufferPort):
    """ReplayBuffer远程客户端；方法名保持和本地ReplayBuffer公开接口一致。"""

    def __init__(self, target: str, call_timeout: float | None = None):
        self.channel = grpc.insecure_channel(
            target,
            options=[
                ("grpc.max_send_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", DEFAULT_GRPC_MAX_MESSAGE_LENGTH),
            ],
        )
        wait_for_grpc_ready(self.channel, target, service_name="ReplayBuffer")
        self.client = RpcClient(
            dodorl_pb2_grpc.ReplayBufferRpcStub(self.channel),
            service_name="ReplayBuffer",
            call_timeout=call_timeout,
        )

    def size(self):
        return self.client.call("size")

    def state(self):
        return self.client.call("state")

    def agent_size(self, agent_name):
        return self.client.call("agent_size", agent_name)

    def agent_names(self):
        return self.client.call("agent_names")

    def reset_sliding_idx(self, agent_name=None):
        return self.client.call("reset_sliding_idx", agent_name=agent_name)

    def ordered_buffer(self, agent_name=None):
        return self.client.call("ordered_buffer", agent_name=agent_name)

    def push(
            self,
            episode_data,
            save=False,
            notify_learning=True,
            agent_name=None,
            sample_role_name="Serial",
            episode_num=None,
            sample_cost_secs=None,
            generation=None,
            sample_version=None,
    ):
        encode_started = perf_now()
        rpc_episode_data = encode_replay_push_payload(
            episode_data,
            agent_name=agent_name,
        )
        record_seconds("rpc.encode_push", encode_started)
        push_started = perf_now()
        try:
            return self.client.call(
                "push",
                rpc_episode_data,
                save=save,
                notify_learning=notify_learning,
                agent_name=agent_name,
                sample_role_name=sample_role_name,
                episode_num=episode_num,
                sample_cost_secs=sample_cost_secs,
                generation=generation,
                **({"sample_version": sample_version} if sample_version is not None else {}),
            )
        finally:
            record_seconds("rpc.push_wait", push_started)

    def sample(self, agent_name, batch_size=0, strategy=None, seed=None,
               include_burn_in=True, sequence_length=1):
        return self.client.call(
            "sample",
            agent_name,
            batch_size=batch_size,
            strategy=strategy,
            seed=seed,
            include_burn_in=include_burn_in,
            sequence_length=sequence_length,
        )

    def save_buffer(self, episode_data):
        return self.client.call("save_buffer", episode_data)

    @process_close
    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.channel.close()

from __future__ import annotations

import multiprocessing as mp
import os
import socket

import pytest


def _mps_available() -> bool:
    import torch

    return hasattr(torch, "mps") and torch.mps.is_available()

def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _ReplayService:
    size = 3

    def state(self):
        import torch
        return {"size": self.size, "tensor": torch.arange(8)}

    def agent_buffer_size(self, agent_name):
        return self.size

    def agent_names(self):
        return ["agent"]

    def reset_sliding_idx(self, agent_name=None):
        return None

    def ordered_buffer(self, agent_name=None):
        return []

    def push(self, value, **_kwargs):
        from sample.replay.replay_payload import (
            ReplayPushEntryKey,
            ReplayPushEnvelope,
        )

        if isinstance(value, ReplayPushEnvelope):
            # v2 信封键为 ReplayPushEntryKey；对齐服务端 _normalize_push_entries
            # 语义，落池前只取 agent_name，且同一 AgentName 去重保持首次出现序。
            agent_names = list(dict.fromkeys(
                key.agent_name if isinstance(key, ReplayPushEntryKey) else key
                for key in value.episodes
            ))
        else:
            agent_names = []
        return {
            "is_envelope": isinstance(value, ReplayPushEnvelope),
            "agent_names": agent_names,
        }

    def sample(self, *args, **kwargs):
        return None

    def save_buffer(self, value):
        return value


class _StressReplayService(_ReplayService):
    def __init__(self):
        import threading

        self.records = []
        self.current = 0
        self.max_current = 0
        self.lock = threading.Lock()

    def push(self, value, **kwargs):
        from sample.replay.replay_payload import ReplayPushEnvelope

        assert isinstance(value, ReplayPushEnvelope)
        tensor = value.episodes["agent"]
        with self.lock:
            self.current += 1
            self.max_current = max(self.max_current, self.current)
        try:
            self.records.append((
                kwargs["sample_role_name"],
                kwargs["episode_num"],
                kwargs["generation"],
                tensor.numel(),
                float(tensor[0]),
                float(tensor[-1]),
            ))
        finally:
            with self.lock:
                self.current -= 1
        return True


class _ModelInfo:
    def to_dict(self):
        return {"name": "agent", "version": "v3n"}


class _ModelService:
    model_list = []

    def get_champion(self, name):
        assert name == "agent"
        return _ModelInfo()

    def __getattr__(self, _name):
        return lambda *args, **kwargs: _ModelInfo()


def _center(address: str, finished) -> None:
    from config import TorchRpcConfig
    from rpc.dispatch import REPLAY_BUFFER
    from rpc.torch_rpc.runtime import init_runtime, shutdown_runtime

    os.environ["RANK"] = "0"
    config = TorchRpcConfig(world_size=2, init_address=address, call_timeout_seconds=10)
    init_runtime(config, "center", services={REPLAY_BUFFER: _ReplayService()}, require_center=True)
    finished.wait(20)
    shutdown_runtime()


def _sample(address: str, finished, result) -> None:
    import numpy as np

    from config import TorchRpcConfig
    from rpc.torch_rpc.client import ReplayBufferTorchRpcClient
    from rpc.torch_rpc.runtime import init_runtime, shutdown_runtime
    from sample.sample_transfer import SAEpisodeTransition

    os.environ["RANK"] = "1"
    config = TorchRpcConfig(world_size=2, init_address=address, call_timeout_seconds=10)
    init_runtime(config, "sample-0", require_center=False)
    client = ReplayBufferTorchRpcClient(call_timeout=10)
    state = client.state()
    transition = SAEpisodeTransition()
    transition.add_step(
        obs=np.asarray([1.0, 2.0], dtype=np.float32),
        action=1,
        last_action=0,
        reward=0.5,
        next_obs=np.asarray([2.0, 3.0], dtype=np.float32),
        terminated=True,
        truncated=False,
        avail_actions=[1, 1],
    )
    transition.prepare_for_replay()
    push_result = client.push(
        transition,
        agent_name="agent",
        notify_learning=False,
    )
    result.put((
        state["size"],
        state["tensor"].tolist(),
        push_result["is_envelope"],
        push_result["agent_names"],
    ))
    finished.set()
    shutdown_runtime()


def test_torch_rpc_native_tensor_round_trip():
    address = f"127.0.0.1:{_free_port()}"
    context = mp.get_context("spawn")
    finished = context.Event()
    result = context.Queue()
    center = context.Process(target=_center, args=(address, finished))
    sample = context.Process(target=_sample, args=(address, finished, result))
    center.start()
    sample.start()
    sample.join(30)
    center.join(30)
    if sample.is_alive():
        sample.terminate()
    if center.is_alive():
        center.terminate()
    assert sample.exitcode == 0
    assert center.exitcode == 0
    assert result.get(timeout=2) == (3, list(range(8)), True, ["agent"])


class _MpsReplayService(_ReplayService):
    def state(self):
        import torch

        return {"tensor": torch.arange(8, device="mps")}


def _mps_center(address: str, finished) -> None:
    from config import TorchRpcConfig
    from rpc.dispatch import REPLAY_BUFFER
    from rpc.torch_rpc.runtime import init_runtime, shutdown_runtime

    os.environ["RANK"] = "0"
    config = TorchRpcConfig(world_size=2, init_address=address, call_timeout_seconds=10)
    init_runtime(config, "center", services={REPLAY_BUFFER: _MpsReplayService()}, require_center=True)
    finished.wait(20)
    shutdown_runtime()


def _mps_learner(address: str, finished, result) -> None:
    from config import TorchRpcConfig
    from rpc.torch_rpc.client import ReplayBufferTorchRpcClient
    from rpc.torch_rpc.runtime import init_runtime, shutdown_runtime

    os.environ["RANK"] = "1"
    config = TorchRpcConfig(world_size=2, init_address=address, call_timeout_seconds=10)
    init_runtime(config, "sample-0", require_center=False)
    tensor = ReplayBufferTorchRpcClient(call_timeout=10).state()["tensor"]
    result.put((tensor.device.type, tensor.tolist()))
    finished.set()
    shutdown_runtime()


@pytest.mark.skipif(not _mps_available(), reason="MPS is unavailable")
def test_torch_rpc_mps_response_is_normalized_to_cpu():
    address = f"127.0.0.1:{_free_port()}"
    context = mp.get_context("spawn")
    finished = context.Event()
    result = context.Queue()
    center = context.Process(target=_mps_center, args=(address, finished))
    learner = context.Process(target=_mps_learner, args=(address, finished, result))
    center.start()
    learner.start()
    learner.join(30)
    center.join(30)
    if learner.is_alive():
        learner.terminate()
    if center.is_alive():
        center.terminate()
    assert learner.exitcode == 0
    assert center.exitcode == 0
    assert result.get(timeout=2) == ("cpu", list(range(8)))


def _full_control(address: str, finished) -> None:
    from config import TorchRpcConfig
    from rpc.dispatch import REPLAY_BUFFER
    from rpc.torch_rpc.runtime import init_runtime, shutdown_runtime

    os.environ["RANK"] = "0"
    config = TorchRpcConfig(world_size=3, init_address=address, call_timeout_seconds=10)
    init_runtime(config, "control-0", services={REPLAY_BUFFER: _ReplayService()})
    finished.wait(20)
    shutdown_runtime()


def _full_model(address: str, finished) -> None:
    from config import TorchRpcConfig
    from rpc.dispatch import MODEL_MANAGER
    from rpc.torch_rpc.runtime import init_runtime, shutdown_runtime

    os.environ["RANK"] = "2"
    config = TorchRpcConfig(world_size=3, init_address=address, call_timeout_seconds=10)
    init_runtime(config, "model-0", services={MODEL_MANAGER: _ModelService()})
    finished.wait(20)
    shutdown_runtime()


def _full_learner(address: str, finished, result) -> None:
    from config import TorchRpcConfig
    from rpc.torch_rpc.client import ModelManagerTorchRpcClient, ReplayBufferTorchRpcClient
    from rpc.torch_rpc.runtime import init_runtime, shutdown_runtime

    os.environ["RANK"] = "1"
    config = TorchRpcConfig(world_size=3, init_address=address, call_timeout_seconds=10)
    init_runtime(config, "learn-0")
    replay_state = ReplayBufferTorchRpcClient(config=config).state()
    champion = ModelManagerTorchRpcClient(call_timeout=10).get_champion("agent")
    result.put((replay_state["size"], champion["version"]))
    finished.set()
    shutdown_runtime()


def test_torch_rpc_full_topology_routes_control_and_model_services():
    address = f"127.0.0.1:{_free_port()}"
    context = mp.get_context("spawn")
    finished = context.Event()
    result = context.Queue()
    processes = [
        context.Process(target=_full_control, args=(address, finished)),
        context.Process(target=_full_learner, args=(address, finished, result)),
        context.Process(target=_full_model, args=(address, finished)),
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
    for process in processes:
        if process.is_alive():
            process.terminate()
    assert [process.exitcode for process in processes] == [0, 0, 0]
    assert result.get(timeout=2) == (3, "v3n")


def _stress_center(address: str, finished, result) -> None:
    from config import TorchRpcConfig
    from rpc.dispatch import REPLAY_BUFFER
    from rpc.torch_rpc.runtime import init_runtime, shutdown_runtime

    os.environ["RANK"] = "0"
    service = _StressReplayService()
    config = TorchRpcConfig(
        world_size=5,
        init_address=address,
        call_timeout_seconds=30,
        max_concurrent_replay_pushes=1,
        replay_push_permit_poll_interval_seconds=0.01,
        replay_push_permit_lease_timeout_seconds=60,
    )
    init_runtime(config, "stress-center", services={REPLAY_BUFFER: service}, require_center=True)
    finished.wait(60)
    result.put((service.max_current, sorted(service.records)))
    shutdown_runtime()


def _stress_sample(address: str, rank: int, finished, ready) -> None:
    import torch

    from config import TorchRpcConfig
    from rpc.torch_rpc.client import ReplayBufferTorchRpcClient
    from rpc.torch_rpc.runtime import init_runtime, shutdown_runtime
    from sample.replay.replay_payload import ReplayPushEnvelope

    os.environ["RANK"] = str(rank)
    role = f"stress-sample-{rank}"
    config = TorchRpcConfig(
        world_size=5,
        init_address=address,
        call_timeout_seconds=30,
        max_concurrent_replay_pushes=1,
        replay_push_permit_poll_interval_seconds=0.01,
        replay_push_permit_lease_timeout_seconds=60,
    )
    init_runtime(config, role, require_center=False)
    client = ReplayBufferTorchRpcClient(config=config)
    for round_index in range(4):
        episode = rank * 100 + round_index
        value = float(episode)
        # 约 4 MiB CPU tensor；四个 sample 同轮竞争，但只有持 permit 者会发送大载荷。
        payload = ReplayPushEnvelope(
            version=ReplayPushEnvelope.CURRENT_VERSION,
            episodes={"agent": torch.full((1_000_000,), value, dtype=torch.float32)},
        )
        assert client.push(
            payload,
            notify_learning=False,
            sample_role_name=role,
            episode_num=episode,
            generation=7,
        ) is True
    ready.put(rank)
    finished.wait(60)
    shutdown_runtime()


def test_torch_rpc_four_sample_large_cpu_tensor_stress():
    address = f"127.0.0.1:{_free_port()}"
    context = mp.get_context("spawn")
    finished = context.Event()
    result = context.Queue()
    ready = context.Queue()
    center = context.Process(target=_stress_center, args=(address, finished, result))
    samples = [
        context.Process(target=_stress_sample, args=(address, rank, finished, ready))
        for rank in range(1, 5)
    ]
    processes = [center, *samples]
    for process in processes:
        process.start()
    try:
        completed = sorted(ready.get(timeout=60) for _ in samples)
        assert completed == [1, 2, 3, 4]
        finished.set()
        for process in processes:
            process.join(30)
        assert [process.exitcode for process in processes] == [0, 0, 0, 0, 0]
        max_current, records = result.get(timeout=5)
        assert max_current == 1
        assert len(records) == 16
        for role, episode, generation, numel, first, last in records:
            assert role == f"stress-sample-{episode // 100}"
            assert generation == 7
            assert numel == 1_000_000
            assert first == last == float(episode)
    finally:
        finished.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)

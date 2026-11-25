"""中心收口后，分布式 sample 应正常结束而不是判 world 失败。"""
from __future__ import annotations

import multiprocessing as mp
import os
import socket
import time
from types import SimpleNamespace

import pytest

from common.exception import RpcServerGone, RpcWorldFailure
from config import GrpcConfig, TorchRpcConfig
from rpc.center_lifecycle import is_rpc_center_finished
from rpc.dispatch import TRAIN_TASK_MANAGER
from rpc.factory import RpcServiceHost, create_rpc_client, init_rpc_client_runtime, shutdown_rpc_client_runtime
from roles.sample import SampleRole
from scheduler.rollout_sample import RolloutSampleWorker


class _TaskService:
    sample_task_process = 0
    learning_process = 0

    def is_sample_completed(self):
        return False

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sample_role(*, raise_from_sync) -> SampleRole:
    role = SampleRole.__new__(SampleRole)
    role.role_id = "sample-0"
    role.rpc_config = object()
    role.env = SimpleNamespace(close=lambda: None)
    role.policy_agents = [SimpleNamespace(name="uav_team", agent_list=["uav_0"])]
    role._trainable_agent_names = {"uav_team"}
    role._synced_model_versions = {}
    role.model_manager = SimpleNamespace(get_champion=lambda _name: (_ for _ in ()).throw(raise_from_sync))
    role.artifacts_manager = SimpleNamespace(
        put_log=lambda *args, **kwargs: None,
        put_bytes=lambda *args, **kwargs: None,
        complete=lambda *args, **kwargs: None,
        expect_producer=lambda *args, **kwargs: None,
    )
    role._artifact_stage = None
    role._completed_stages = set()
    episode = SimpleNamespace(episode_num=1, stage_name="train", generation=0, seed=None, lease_id="lease-1")
    role.task_manager = SimpleNamespace(
        get_task_manager_info=lambda: {},
        is_sample_completed=lambda: False,
        sample_gate_state=lambda **kwargs: {"blocked": False},
        get_sample_task_info=lambda **kwargs: episode,
        report_sample_result=lambda *args, **kwargs: None,
    )
    role.rollout_worker = RolloutSampleWorker(
        env=role.env,
        task_manager=role.task_manager,
        replay_buffer=None,
        agent_compose=SimpleNamespace(episode_agent=None, step_agent=role.policy_agents),
        policy_agents=role.policy_agents,
    )
    role.rollout_worker.pre_episode_sync_caller = role._sync_episode_models
    return role


def test_sample_role_run_returns_when_center_is_gone(monkeypatch):
    shutdowns = []
    monkeypatch.setattr("roles.sample.shutdown_rpc_client_runtime", lambda _config: shutdowns.append("rpc"))
    role = _sample_role(
        raise_from_sync=RpcServerGone(
            method="get_champion",
            attempts=1,
            interval=0.0,
            service="ModelManager",
            detail="center finished",
        ),
    )

    role.run()

    assert shutdowns == ["rpc"]


def test_sample_role_run_still_aborts_failed_world(monkeypatch):
    monkeypatch.setattr("roles.sample.shutdown_rpc_client_runtime", lambda _config: None)
    role = _sample_role(
        raise_from_sync=RpcWorldFailure(
            service="ModelManager",
            method="get_champion",
            detail="EOF: end of file",
        ),
    )

    with pytest.raises(RpcWorldFailure):
        role.run()


def test_sample_role_aborts_unexpected_rollout_error_and_closes_resources(monkeypatch):
    closed = []
    calls = []
    monkeypatch.setattr("roles.sample.shutdown_rpc_client_runtime", lambda _config: closed.append("rpc"))
    role = _sample_role(raise_from_sync=KeyError("unused"))
    role._simulation_middleware_logged = True
    role.env.close = lambda: closed.append("env")
    original = KeyError("drone_0")

    def rollout(**_kwargs):
        calls.append("rollout")
        raise original

    role.rollout_worker = SimpleNamespace(rollout_one_episode=rollout)
    with pytest.raises(KeyError) as caught:
        role.run()
    assert caught.value is original
    assert calls == ["rollout"]
    assert closed == ["env", "rpc"]


def test_grpc_distributed_client_finishes_cleanly_after_center_stop(tmp_path, monkeypatch):
    progress_file = tmp_path / "training_progress.env"
    monkeypatch.setenv("DODO_PROGRESS_FILE", str(progress_file))
    address = f"127.0.0.1:{_free_port()}"
    config = GrpcConfig(task_manager_address=address, call_timeout_seconds=2.0)
    host = RpcServiceHost(config, "control", {TRAIN_TASK_MANAGER: _TaskService()})
    host.start()
    client = create_rpc_client(config, TRAIN_TASK_MANAGER)

    assert client.is_sample_completed() is False
    started = time.monotonic()
    host.stop()

    assert is_rpc_center_finished() is True
    with pytest.raises(RpcServerGone):
        client.is_sample_completed()
    assert time.monotonic() - started < 2.0


def _torch_finish_center(address: str, progress_file: str, sample_ready) -> None:
    os.environ["RANK"] = "0"
    os.environ["DODO_PROGRESS_FILE"] = progress_file
    config = TorchRpcConfig(world_size=2, init_address=address, call_timeout_seconds=8)
    host = RpcServiceHost(config, "center", {TRAIN_TASK_MANAGER: _TaskService()})
    host.start()
    sample_ready.wait(20)
    host.stop()


def _torch_finish_sample(address: str, progress_file: str, sample_ready, result) -> None:
    os.environ["RANK"] = "1"
    os.environ["DODO_PROGRESS_FILE"] = progress_file
    config = TorchRpcConfig(world_size=2, init_address=address, call_timeout_seconds=8)
    init_rpc_client_runtime(config, "sample-0")
    client = create_rpc_client(config, TRAIN_TASK_MANAGER)
    assert client.is_sample_completed() is False
    sample_ready.set()
    deadline = time.time() + 15
    try:
        while time.time() < deadline:
            try:
                client.is_sample_completed()
                time.sleep(0.05)
            except RpcServerGone:
                result.put("server_gone")
                return
        result.put("timeout")
    finally:
        shutdown_rpc_client_runtime(config)


def test_torch_rpc_distributed_world_finishes_cleanly_after_center_stop(tmp_path):
    progress_file = str(tmp_path / "training_progress.env")
    address = f"127.0.0.1:{_free_port()}"
    context = mp.get_context("spawn")
    sample_ready = context.Event()
    result = context.Queue()
    center = context.Process(target=_torch_finish_center, args=(address, progress_file, sample_ready))
    sample = context.Process(target=_torch_finish_sample, args=(address, progress_file, sample_ready, result))
    center.start()
    sample.start()
    try:
        sample.join(30)
        center.join(30)
        assert sample.exitcode == 0, "sample should exit 0 after RpcServerGone"
        assert center.exitcode == 0, "center should exit 0 after host.stop"
        assert result.get(timeout=2) == "server_gone"
        assert "DODO_CENTER_FINISHED=1" in (tmp_path / "training_progress.env").read_text(encoding="utf-8")
    finally:
        for process in (sample, center):
            if process.is_alive():
                process.terminate()
            process.join(5)

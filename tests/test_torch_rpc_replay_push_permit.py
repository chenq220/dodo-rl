from __future__ import annotations

import threading
import time

import pytest

from common.exception import RpcServerGone, RpcWorldFailure
from config import TorchRpcConfig
from rpc.dispatch import RpcResponseEnvelope
from rpc.torch_rpc.client import ReplayBufferTorchRpcClient, TorchRpcClient
from rpc.torch_rpc.replay_push_permit import TorchReplayBufferService
from utils.task_id import task_id


class _ReplayTarget:
    def __init__(self, *, fail: bool = False, delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay
        self.calls = []
        self.current = 0
        self.max_current = 0
        self.lock = threading.Lock()

    def push(self, episode_data, **kwargs):
        with self.lock:
            self.calls.append((episode_data, kwargs))
            self.current += 1
            self.max_current = max(self.max_current, self.current)
        try:
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("push failed")
            return {"stored": kwargs.get("episode_num")}
        finally:
            with self.lock:
                self.current -= 1


def _service(target=None, *, maximum=1, lease=0.1):
    return TorchReplayBufferService(
        target or _ReplayTarget(),
        TorchRpcConfig(
            max_concurrent_replay_pushes=maximum,
            replay_push_permit_lease_timeout_seconds=lease,
        ),
    )


def _acquire(service, role, request, episode, generation=0):
    return service.acquire_replay_push_permit(role, request, episode, generation)


def test_four_samples_never_exceed_configured_large_push_concurrency():
    target = _ReplayTarget(delay=0.03)
    service = _service(target, maximum=1, lease=1.0)
    barrier = threading.Barrier(4)

    def upload(index):
        role = f"sample-{index}"
        barrier.wait()
        while True:
            permit = _acquire(service, role, f"request-{index}", 100 + index)
            if permit["acquired"]:
                break
            time.sleep(0.001)
        service.push_with_permit(
            permit["token"],
            {"tensor": index},
            sample_role_name=role,
            episode_num=100 + index,
            generation=0,
        )

    threads = [threading.Thread(target=upload, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert target.max_current == 1
    assert len(target.calls) == 4
    assert service.replay_push_permit_state()["active_count"] == 0


def test_denied_permit_does_not_call_large_push_and_success_releases_capacity():
    target = _ReplayTarget()
    service = _service(target)
    first = _acquire(service, "sample-1", "request-1", 1)
    denied = _acquire(service, "sample-2", "request-2", 2)

    assert first["acquired"] is True
    assert denied["acquired"] is False
    assert denied["reason"] == "capacity"
    assert target.calls == []

    service.push_with_permit(
        first["token"], {"large": True}, sample_role_name="sample-1", episode_num=1, generation=0,
    )
    assert service.replay_push_permit_state()["active_count"] == 0
    assert _acquire(service, "sample-2", "request-2", 2)["acquired"] is True


def test_push_exception_releases_capacity_and_duplicate_release_is_safe():
    service = _service(_ReplayTarget(fail=True))
    permit = _acquire(service, "sample-1", "request-1", 1)

    with pytest.raises(RuntimeError, match="push failed"):
        service.push_with_permit(
            permit["token"], {}, sample_role_name="sample-1", episode_num=1, generation=0,
        )

    assert service.replay_push_permit_state()["active_count"] == 0
    assert service.release_replay_push_permit(permit["token"], "sample-1") == {
        "released": False,
        "reason": "failed",
    }
    assert service.release_replay_push_permit(permit["token"], "wrong-role")["reason"] == "sample_role_mismatch"


def test_crashed_sample_permit_is_reclaimed_after_lease_timeout():
    service = _service(lease=0.03)
    first = _acquire(service, "sample-1", "request-1", 1)
    assert first["acquired"] is True
    assert _acquire(service, "sample-2", "request-2", 2)["acquired"] is False

    time.sleep(0.04)
    second = _acquire(service, "sample-2", "request-2", 2)
    assert second["acquired"] is True
    assert service.release_replay_push_permit(first["token"], "sample-1")["reason"] == "unknown_or_expired"


def test_token_identity_generation_and_duplicate_push_are_safe():
    target = _ReplayTarget()
    service = _service(target, lease=1.0)
    permit = _acquire(service, "sample-1", "request-1", 7, generation=3)

    with pytest.raises(ValueError, match="does not match"):
        service.push_with_permit(
            permit["token"], {}, sample_role_name="sample-2", episode_num=7, generation=3,
        )
    with pytest.raises(ValueError, match="does not match"):
        service.push_with_permit(
            permit["token"], {}, sample_role_name="sample-1", episode_num=7, generation=4,
        )

    expected = service.push_with_permit(
        permit["token"], {"episode": 7}, sample_role_name="sample-1", episode_num=7, generation=3,
    )
    duplicate = service.push_with_permit(
        permit["token"], {"episode": "duplicate"}, sample_role_name="sample-1", episode_num=7, generation=3,
    )
    assert duplicate == expected
    assert len(target.calls) == 1

    next_generation = _acquire(service, "sample-1", "request-2", 7, generation=4)
    assert next_generation["acquired"] is True


def test_in_flight_duplicate_push_does_not_release_or_store_twice():
    entered = threading.Event()
    unblock = threading.Event()

    class BlockingTarget(_ReplayTarget):
        def push(self, episode_data, **kwargs):
            with self.lock:
                self.calls.append((episode_data, kwargs))
                self.current += 1
                self.max_current = max(self.max_current, self.current)
            entered.set()
            try:
                assert unblock.wait(1)
                return "stored"
            finally:
                with self.lock:
                    self.current -= 1

    target = BlockingTarget()
    service = _service(target, lease=0.03)
    permit = _acquire(service, "sample-1", "request-1", 1)
    results = []
    uploader = threading.Thread(target=lambda: results.append(service.push_with_permit(
        permit["token"], {}, sample_role_name="sample-1", episode_num=1, generation=0,
    )))
    uploader.start()
    assert entered.wait(1)

    duplicate = service.push_with_permit(
        permit["token"], {}, sample_role_name="sample-1", episode_num=1, generation=0,
    )
    assert duplicate == {"__replay_push_permit__": True, "state": "pushing"}
    assert service.release_replay_push_permit(permit["token"], "sample-1")["reason"] == "pushing"
    time.sleep(0.04)
    assert service.replay_push_permit_state()["active_count"] == 1
    assert len(target.calls) == 1

    unblock.set()
    uploader.join(1)
    assert results == ["stored"]
    assert service.replay_push_permit_result(permit["token"], "sample-1") == {
        "__replay_push_permit__": True,
        "state": "completed",
        "result": "stored",
    }


def test_client_waits_for_permit_before_encoding_or_calling_large_rpc(monkeypatch):
    client = ReplayBufferTorchRpcClient(
        config=TorchRpcConfig(replay_push_permit_poll_interval_seconds=0.01)
    )
    events = []
    acquire_responses = iter([
        {"acquired": False, "reason": "capacity"},
        {"acquired": True, "token": "token-1"},
    ])

    def call(method, *args, **kwargs):
        events.append(method)
        if method == "acquire_replay_push_permit":
            return next(acquire_responses)
        if method == "push_with_permit":
            return "stored"
        return {"released": False, "reason": "completed"}

    client.client.call = call
    monkeypatch.setattr("rpc.torch_rpc.client.time.sleep", lambda _delay: events.append("poll_sleep"))
    monkeypatch.setattr(
        "rpc.torch_rpc.client.encode_replay_push_payload",
        lambda value, **_kwargs: events.append("encode") or value,
    )

    assert client.push({}, sample_role_name="sample-1", episode_num=9, generation=2) == "stored"
    assert events == [
        "acquire_replay_push_permit",
        "poll_sleep",
        "acquire_replay_push_permit",
        "encode",
        "push_with_permit",
        "release_replay_push_permit",
    ]


@pytest.mark.parametrize("detail", [
    "EPIPE: broken pipe",
    "EOF: end of file",
    "pipe closed",
    "connection closed",
    "EINVAL: invalid argument (tensorpipe/transport/uv/connection_impl.cc:132)",
])
def test_irrecoverable_tensorpipe_errors_fail_world_without_retry(monkeypatch, detail):
    calls = []

    def rpc_sync(*_args, **_kwargs):
        calls.append(detail)
        raise RuntimeError(detail)

    monkeypatch.setattr("rpc.torch_rpc.client.torch_rpc.rpc_sync", rpc_sync)
    monkeypatch.setattr("rpc.torch_rpc.client.worker_context", lambda: ("sample-4", 4))
    monkeypatch.setattr("rpc.torch_rpc.client.is_rpc_center_finished", lambda: False)
    client = TorchRpcClient("ReplayBuffer", call_timeout=1)

    with pytest.raises(RpcWorldFailure) as error:
        client.call(
            "push_with_permit",
            _rpc_context={"sample_role": "sample-4", "episode": 769, "generation": 3},
        )

    assert calls == [detail]
    message = str(error.value)
    assert all(value in message for value in (
        str(task_id), "sample-4", "Rank: 4", "ReplayBuffer", "push_with_permit",
        "Episode: 769", "Generation: 3", detail,
    ))


def test_center_finished_irrecoverable_error_is_server_gone(monkeypatch):
    calls = []

    def rpc_sync(*_args, **_kwargs):
        calls.append("eof")
        raise RuntimeError("EOF: end of file (tensorpipe/transport/uv/connection_impl.cc:132)")

    monkeypatch.setattr("rpc.torch_rpc.client.torch_rpc.rpc_sync", rpc_sync)
    monkeypatch.setattr("rpc.torch_rpc.client.is_rpc_center_finished", lambda: True)
    client = TorchRpcClient("ModelManager", call_timeout=1)

    with pytest.raises(RpcServerGone) as error:
        client.call("get_champion")

    assert calls == ["eof"]
    assert "ModelManager" in str(error.value)
    assert "get_champion" in str(error.value)


def test_temporary_torch_rpc_error_keeps_finite_retry(monkeypatch):
    responses = iter([
        RuntimeError("RPC request timed out"),
        RuntimeError("temporarily unavailable"),
        RpcResponseEnvelope(ok=True, value=42),
    ])
    sleeps = []

    def rpc_sync(*_args, **_kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("rpc.torch_rpc.client.torch_rpc.rpc_sync", rpc_sync)
    monkeypatch.setattr("rpc.torch_rpc.client.time.sleep", sleeps.append)

    assert TorchRpcClient("ReplayBuffer", call_timeout=1).call("state") == 42
    assert sleeps == [5.0, 10.0]

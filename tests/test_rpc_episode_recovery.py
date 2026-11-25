from types import SimpleNamespace

import grpc
import pytest

from common.exception import (
    EnvironmentException,
    EpisodeInterruptedException,
    RpcCodecException,
    RpcException,
    RpcServerGone,
    RpcTransportException,
    SampleTaskReturned,
)
from common.exception import SimulationException
from envs.base import BaseEnvironment
from scheduler.rollout_sample import RolloutSampleWorker
from rpc.grpc import dodorl_pb2
from rpc.grpc.codec import unpack_call
from rpc.grpc.service import RpcClient
from rpc.grpc.service.replay_buffer_rpc import ReplayBufferRpcClient
from rpc.grpc.service.train_task_rpc import TrainTaskManagerRpcClient
from rpc.torch_rpc.client import ReplayBufferTorchRpcClient
from sample.replay.replay_payload import ReplayPushEnvelope
from sample.sample_transfer import SAEpisodeTransition


class _TaskManager:
    def __init__(self):
        self.episode = SimpleNamespace(
            episode_num=63,
            epsilon=1.0,
            temperature=1.0,
            deterministic=False,
            seed=7,
            metadata={},
            stage_name="train",
            generation=0,
            lease_id="lease-63",
        )
        self.reports = []

    def is_sample_completed(self):
        return False

    def sample_backpressure_state(self, **_kwargs):
        return {"enabled": False}

    def get_sample_task_info(self, **_kwargs):
        return self.episode

    def report_sample_result(self, episode_num, **kwargs):
        self.reports.append((episode_num, kwargs))


class _SwitchingTaskManager(_TaskManager):
    """阶段切换中的任务总线：闸门阻塞，但不能让 sample role 永久等待。"""

    def get_sample_task_info(self, **_kwargs):
        return None

    def sample_gate_state(self, **_kwargs):
        return {
            "blocked": True,
            "reason": "stage_switching",
            "retry_after_seconds": 0.5,
        }


class _RpcError(grpc.RpcError):
    def __init__(self, status_code):
        self._status_code = status_code

    def code(self):
        return self._status_code

    def details(self):
        return "test transport failure"


class _FailingStub:
    def __init__(self, error):
        self.error = error

    def Call(self, _request, timeout=None):
        raise self.error


class _ResponseStub:
    def __init__(self, response):
        self.response = response

    def Call(self, _request, timeout=None):
        return self.response


class _TimeoutRecordingStub(_ResponseStub):
    def __init__(self, response):
        super().__init__(response)
        self.timeout = None

    def Call(self, _request, timeout=None):
        self.timeout = timeout
        return self.response


class _CallRecorder:
    def __init__(self):
        self.calls = []

    def call(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _prepared_sa_episode() -> SAEpisodeTransition:
    transition = SAEpisodeTransition()
    transition.add_step(
        obs=[1.0, 2.0],
        action=1,
        last_action=0,
        reward=0.5,
        next_obs=[2.0, 3.0],
        terminated=True,
        truncated=False,
        avail_actions=[1, 1],
    )
    transition.prepare_for_replay()
    return transition


@pytest.mark.parametrize(
    "client_type",
    [ReplayBufferRpcClient, ReplayBufferTorchRpcClient],
)
def test_replay_rpc_clients_use_the_internal_push_envelope(client_type):
    rpc_client = client_type.__new__(client_type)
    recorder = _CallRecorder()
    rpc_client.client = recorder
    rpc_client.permit_poll_interval_seconds = 0.001
    if client_type is ReplayBufferTorchRpcClient:
        responses = iter([
            {"acquired": True, "token": "permit-1"},
            None,
            {"released": False, "reason": "completed"},
        ])

        def call(*args, **kwargs):
            recorder.calls.append((args, kwargs))
            return next(responses)

        rpc_client.client.call = call
    transition = _prepared_sa_episode()

    rpc_client.push(
        transition,
        agent_name="agent",
        notify_learning=False,
    )

    push_index = 1 if client_type is ReplayBufferTorchRpcClient else 0
    args, kwargs = rpc_client.client.calls[push_index]
    assert args[0] == ("push_with_permit" if client_type is ReplayBufferTorchRpcClient else "push")
    payload_index = 2 if client_type is ReplayBufferTorchRpcClient else 1
    assert isinstance(args[payload_index], ReplayPushEnvelope)
    assert kwargs["agent_name"] == "agent"
    assert kwargs["notify_learning"] is False


def test_independent_simulation_and_rl_interruption_families():
    assert issubclass(SimulationException, EpisodeInterruptedException)
    assert issubclass(EnvironmentException, EpisodeInterruptedException)
    assert issubclass(RpcCodecException, EpisodeInterruptedException)
    assert issubclass(RpcTransportException, EpisodeInterruptedException)
    assert issubclass(RpcCodecException, RpcException)
    assert issubclass(RpcTransportException, RpcException)
    assert not issubclass(RpcServerGone, EpisodeInterruptedException)




def test_torch_replay_permit_precedes_lease_recheck_and_payload_prepare():
    events = []

    class PermitReplay:
        requires_replay_push_permit = True

        def acquire_replay_push_permit(self, **_kwargs):
            events.append("acquire")
            return {"token": "permit-1"}

        def push_with_permit(self, permit, payload, **_kwargs):
            events.append(("push", permit["token"], payload))

        def release_replay_push_permit(self, *_args, **_kwargs):
            events.append("release")

    worker = RolloutSampleWorker.__new__(RolloutSampleWorker)
    worker.sample_role_name = "sample-1"
    worker.replay_buffer = PermitReplay()
    worker.task_manager = SimpleNamespace(
        accept_sample_task_result=lambda episode, lease: events.append(("lease", episode, lease)) or True,
    )
    worker._prepare_replay_transitions = lambda payload: events.append(("prepare", payload))
    episode = SimpleNamespace(episode_num=12, generation=4, lease_id="lease-12")
    payload = {"agent": object()}

    worker._submit_replay_transitions(payload, episode, 0.0)

    assert events[0] == "acquire"
    assert events[1] == ("lease", 12, "lease-12")
    assert events[2] == ("prepare", payload)
    assert events[3] == ("push", "permit-1", payload)


def test_expired_episode_lease_releases_permit_without_large_push():
    events = []

    class PermitReplay:
        requires_replay_push_permit = True

        def acquire_replay_push_permit(self, **_kwargs):
            events.append("acquire")
            return {"token": "permit-1"}

        def push_with_permit(self, *_args, **_kwargs):
            events.append("push")

        def release_replay_push_permit(self, *_args, **_kwargs):
            events.append("release")

    worker = RolloutSampleWorker.__new__(RolloutSampleWorker)
    worker.sample_role_name = "sample-1"
    worker.replay_buffer = PermitReplay()
    worker.task_manager = SimpleNamespace(accept_sample_task_result=lambda *_args: False)
    episode = SimpleNamespace(episode_num=12, generation=4, lease_id="expired")

    with pytest.raises(EnvironmentException, match="lease expired"):
        worker._submit_replay_transitions({}, episode, 0.0)

    assert events == ["acquire", "release"]


def test_invalid_pickle_is_normalized_to_rpc_codec_exception():
    with pytest.raises(RpcCodecException) as error:
        unpack_call(b"\xf7")
    assert error.value.method == "unpack_call"


def test_remote_unpickling_error_is_recoverable_codec_exception():
    response = dodorl_pb2.RpcResponse(
        ok=False,
        error_type="UnpicklingError",
        error_message="invalid load key, '\\xf7'",
    )
    client = RpcClient(_ResponseStub(response), service_name="ReplayBuffer")
    with pytest.raises(RpcCodecException) as error:
        client.call("push", {"episode": 63})
    assert error.value.service == "ReplayBuffer"
    assert error.value.method == "push"


def test_unavailable_replay_buffer_is_recoverable_but_task_manager_is_terminal():
    transport = _RpcError(grpc.StatusCode.UNAVAILABLE)
    replay_client = RpcClient(
        _FailingStub(transport),
        max_retries=1,
        service_name="ReplayBuffer",
    )
    with pytest.raises(RpcTransportException):
        replay_client.call("push", {})

    task_client = RpcClient(
        _FailingStub(transport),
        max_retries=1,
        service_name="TrainTaskManager",
    )
    with pytest.raises(RpcServerGone):
        task_client.call("is_sample_completed")


def test_center_finished_unavailable_is_server_gone_for_any_grpc_service(monkeypatch):
    calls = []

    def is_finished():
        calls.append("check")
        return True

    monkeypatch.setattr("rpc.grpc.service.is_rpc_center_finished", is_finished)
    client = RpcClient(
        _FailingStub(_RpcError(grpc.StatusCode.UNAVAILABLE)),
        max_retries=3,
        service_name="ModelManager",
    )
    with pytest.raises(RpcServerGone) as error:
        client.call("get_champion")
    assert calls == ["check"]
    assert "ModelManager" in str(error.value)
    assert "get_champion" in str(error.value)


def test_rpc_call_has_default_and_configurable_deadline():
    response = dodorl_pb2.RpcResponse(ok=True)
    default_stub = _TimeoutRecordingStub(response)
    RpcClient(default_stub).call("state")
    assert default_stub.timeout == pytest.approx(60.0)

    configured_stub = _TimeoutRecordingStub(response)
    RpcClient(configured_stub, call_timeout=12.5).call("state")
    assert configured_stub.timeout == pytest.approx(12.5)

    disabled_stub = _TimeoutRecordingStub(response)
    RpcClient(disabled_stub, call_timeout=0).call("state")
    assert disabled_stub.timeout is None


def test_task_manager_rpc_client_forwards_episode_reward_details():
    rpc_client = TrainTaskManagerRpcClient.__new__(TrainTaskManagerRpcClient)
    rpc_client.client = _CallRecorder()

    rpc_client.report_sample_result(
        7,
        total_rewards={"agent": 1.0},
        episode_rewards_details={"bonus": 2.0, "cost": -1.0},
        lease_id="lease-7",
    )

    assert rpc_client.client.calls == [
        (
            ("report_sample_result", 7),
            {
                "total_rewards": {"agent": 1.0},
                "success": True,
                "artifacts": None,
                "episode_metrics": None,
                "episode_rewards_details": {"bonus": 2.0, "cost": -1.0},
                "lease_id": "lease-7",
                "step_count": 0,
            },
        )
    ]


def test_stage_switching_returns_none_instead_of_blocking_sample_role():
    worker = RolloutSampleWorker.__new__(RolloutSampleWorker)
    worker.task_manager = _SwitchingTaskManager()
    worker.env = SimpleNamespace(stage_name="train")

    # 必须立即返回 None，让 SampleRole 走到 _submit_current_stage_artifacts；
    # 若在闸门里睡眠，控制端会一直等 producer，形成阶段切换死锁。
    assert worker.rollout_one_episode(sample_role_name="sample-stage-switch") is None


def test_recoverable_rpc_failure_returns_episode_token():
    worker = RolloutSampleWorker.__new__(RolloutSampleWorker)
    worker.task_manager = _TaskManager()
    worker.env = SimpleNamespace(stage_name="train")

    with pytest.raises(SampleTaskReturned):
        with worker.episode_task_context(sample_role_name="sample-3") as episode_info:
            assert episode_info.episode_num == 63
            raise RpcCodecException(
                service="ReplayBuffer",
                method="push",
                detail="invalid load key",
            )

    assert worker.task_manager.reports == [
        (63, {"total_rewards": {}, "success": False, "artifacts": None, "lease_id": "lease-63", "step_count": 0})
    ]


def test_environment_failure_returns_episode_token():
    worker = RolloutSampleWorker.__new__(RolloutSampleWorker)
    worker.task_manager = _TaskManager()
    worker.env = SimpleNamespace(stage_name="train")

    with pytest.raises(SampleTaskReturned):
        with worker.episode_task_context(sample_role_name="sample-4") as episode_info:
            assert episode_info.episode_num == 63
            raise EnvironmentException(detail="action mask is empty")

    assert worker.task_manager.reports == [
        (63, {"total_rewards": {}, "success": False, "artifacts": None, "lease_id": "lease-63", "step_count": 0})
    ]


@pytest.mark.parametrize("report_fails", [False, True])
def test_unexpected_episode_error_returns_lease_and_preserves_error(monkeypatch, report_fails):
    worker = RolloutSampleWorker.__new__(RolloutSampleWorker)
    worker.task_manager = _TaskManager()
    worker.env = SimpleNamespace(stage_name="train")
    reports = []

    def report(episode_num, **kwargs):
        reports.append((episode_num, kwargs))
        if report_fails:
            raise RuntimeError("center unavailable during failure report")

    monkeypatch.setattr(worker.task_manager, "report_sample_result", report)
    original = KeyError("drone_0")
    with pytest.raises(KeyError) as caught:
        with worker.episode_task_context(sample_role_name="sample-4"):
            raise original

    assert caught.value is original
    assert reports == [
        (63, {"total_rewards": {}, "success": False, "artifacts": None, "lease_id": "lease-63", "step_count": 0})
    ]


def test_base_environment_reward_details_are_optional():
    assert BaseEnvironment.get_episode_rewards_details(object()) is None


def test_successful_episode_reports_validated_environment_reward_details(monkeypatch):
    from scheduler import rollout_sample
    messages = []
    monkeypatch.setattr(rollout_sample.LOG, "info", lambda message: messages.append(message))
    worker = RolloutSampleWorker.__new__(RolloutSampleWorker)
    worker.task_manager = _TaskManager()
    worker.env = SimpleNamespace(
        stage_name="train",
        get_episode_rewards_details=lambda: {"bonus": 2, "cost": -0.5},
    )

    with worker.episode_task_context(sample_role_name="sample-5") as episode_info:
        assert episode_info.episode_num == 63
        worker.total_rewards = {"agent": 1.5}

    assert worker.task_manager.reports == [
        (
            63,
            {
                "total_rewards": {"agent": 1.5},
                "episode_metrics": {},
                "episode_rewards_details": {"bonus": 2.0, "cost": -0.5},
                "artifacts": None,
                "lease_id": "lease-63",
                "step_count": 0,
            },
        )
    ]


    breakdown = [message for message in messages if "reward components:" in message]
    assert len(breakdown) == 1
    assert "bonus=+2.00000000" in breakdown[0]
    assert "cost=-0.50000000" in breakdown[0]
    assert "sum=+1.50000000" in breakdown[0]


def test_invalid_reward_details_disable_drawing_without_failing_episode():
    worker = RolloutSampleWorker.__new__(RolloutSampleWorker)
    worker.task_manager = _TaskManager()
    worker.env = SimpleNamespace(
        stage_name="train",
        get_episode_rewards_details=lambda: {"invalid": float("nan")},
    )

    with worker.episode_task_context(sample_role_name="sample-6"):
        worker.total_rewards = {"agent": 0.0}

    assert worker.task_manager.reports[0][1]["episode_rewards_details"] is None


def test_bypass_token_uses_unified_sample_report():
    """旁路令牌与主路走同一条回报；rollout 不因来源字段分叉。"""
    worker = RolloutSampleWorker.__new__(RolloutSampleWorker)
    worker.task_manager = _TaskManager()
    worker.task_manager.episode.metadata = {
        "replay_write": False,
        "record_render": True,
        "record_name": "probe.gif",
    }

    worker.env = SimpleNamespace(
        stage_name="train",
        get_episode_rewards_details=lambda: {"bonus": 1.0},
    )

    with worker.episode_task_context(sample_role_name="sample-7"):
        worker.total_rewards = {"agent": 1.0}

    report = worker.task_manager.reports[0][1]
    assert report["episode_rewards_details"] == {"bonus": 1.0}

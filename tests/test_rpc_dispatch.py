from dataclasses import dataclass

import torch

from rpc.dispatch import REPLAY_BUFFER, RpcMethodDispatcher, build_dispatcher, prepare_torch_rpc_payload


class _Replay:
    size = 1

    def state(self):
        return {"tensor": torch.arange(4)}

    def agent_buffer_size(self, agent_name):
        return 1

    def agent_names(self):
        return ["agent"]

    def reset_sliding_idx(self, agent_name=None):
        return None

    def ordered_buffer(self, agent_name=None):
        return []

    def push(self, value):
        return value

    def sample(self, *args, **kwargs):
        return torch.ones(2)

    def sample_for_rpc(self, *args, **kwargs):
        return "cpu-materialized"

    def save_buffer(self, value):
        return value


def test_dispatcher_preserves_native_tensors():
    response = build_dispatcher(REPLAY_BUFFER, _Replay()).invoke_enveloped("state", (), {})
    assert response.ok is True
    assert torch.equal(response.value["tensor"], torch.arange(4))


def test_replay_dispatcher_uses_rpc_specific_sample_entrypoint():
    response = build_dispatcher(REPLAY_BUFFER, _Replay()).invoke_enveloped(
        "sample", ("agent",), {},
    )
    assert response.ok is True
    assert response.value == "cpu-materialized"


def test_dispatcher_rejects_unexposed_method():
    response = build_dispatcher(REPLAY_BUFFER, _Replay()).invoke_enveloped("__dict__", (), {})
    assert response.ok is False
    assert response.error_type == "AttributeError"


def test_dispatcher_captures_remote_traceback():
    def fail():
        raise RuntimeError("boom")

    response = RpcMethodDispatcher(object(), {"fail": fail}).invoke_enveloped("fail", (), {})
    assert response.ok is False
    assert response.error_message == "boom"
    assert "RuntimeError: boom" in response.remote_traceback


def test_torch_payload_moves_nested_tensors_to_cpu_without_mutating_source():
    class Payload:
        pass

    source = Payload()
    source.values = {"tensor": [torch.arange(3)]}
    prepared = prepare_torch_rpc_payload(source)
    assert prepared is not source
    assert prepared.values is not source.values
    assert prepared.values["tensor"][0].device.type == "cpu"


def test_torch_payload_preserves_frozen_dataclass_replay_templates():
    @dataclass(frozen=True)
    class FrozenTemplate:
        kind: str
        tensor: torch.Tensor

    source = FrozenTemplate(kind="dict", tensor=torch.arange(3))
    prepared = prepare_torch_rpc_payload(source)

    assert prepared is not source
    assert prepared.kind == "dict"
    assert torch.equal(prepared.tensor, source.tensor)
    assert prepared.tensor.device.type == "cpu"


def test_torch_remote_call_normalizes_response_payload(monkeypatch):
    from rpc.torch_rpc import runtime

    dispatcher = RpcMethodDispatcher(object(), {"state": lambda: {"value": 1}})
    monkeypatch.setitem(runtime._DISPATCHERS, REPLAY_BUFFER, dispatcher)
    normalized = runtime.RpcResponseEnvelope(ok=True, value="normalized")
    captured = []

    def normalize(response):
        captured.append(response)
        return normalized

    monkeypatch.setattr(runtime, "prepare_torch_rpc_payload", normalize)

    assert runtime._remote_call(REPLAY_BUFFER, "state", (), {}) is normalized
    assert len(captured) == 1
    assert captured[0].ok is True
    assert captured[0].value == {"value": 1}

from __future__ import annotations

import pickle
import threading
import time
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from config import TorchRpcConfig
from rpc.dispatch import build_replay_buffer_methods, prepare_torch_rpc_payload
from rpc.grpc.service.replay_buffer_rpc import ReplayBufferRpcClient
from rpc.torch_rpc.client import ReplayBufferTorchRpcClient
from rpc.torch_rpc.replay_push_chunk import (
    ReplayPushChunkEnvelope,
    assemble_replay_push_envelope,
    envelope_tensor_nbytes,
    split_replay_push_envelope,
)
from rpc.torch_rpc.replay_push_permit import TorchReplayBufferService
from sample.replay.replay_payload import (
    PreparedReplayEpisode,
    ReplayPushEnvelope,
    ReplayPushEntryKey,
    ReplayTransitionEnvelope,
    TensorOwnership,
    encode_replay_push_payload,
)
from sample.sample_transfer import SAEpisodeTransition


class _ReplayTarget:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def push(self, episode_data, **kwargs):
        self.calls.append((episode_data, kwargs))
        return {"stored": kwargs.get("episode_num")}

    size = 0

    def state(self):
        return {}

    def agent_buffer_size(self, agent_name):
        return 0

    def agent_names(self):
        return []

    def reset_sliding_idx(self, agent_name=None):
        return None

    def ordered_buffer(self, agent_name=None):
        return []

    def sample(self, *args, **kwargs):
        return []

    def save_buffer(self, *args, **kwargs):
        return None


def _sa_episode(episode_num: int, step_count: int) -> SAEpisodeTransition:
    episode = SAEpisodeTransition()
    episode.episode_num = episode_num
    for step in range(step_count):
        episode.add_step(
            obs=np.asarray([episode_num, step, 0.25, step * 0.5], dtype=np.float32),
            action=step % 3,
            last_action=(step - 1) % 3,
            reward=float(step) + 0.125,
            next_obs=np.asarray([episode_num, step + 1, 0.25, step * 0.5], dtype=np.float32),
            terminated=step == step_count - 1,
            truncated=False,
            avail_actions=[1, 0, 1],
            log_prob=-0.25 * step,
            value=1.5 + step,
            metadata={
                "label": f"step-{episode_num}-{step}",
                "diagnostic": torch.tensor([episode_num, step], dtype=torch.int32),
            },
        )
    episode.prepare_for_replay()
    return episode


def _empty_prepared() -> PreparedReplayEpisode:
    return PreparedReplayEpisode(
        step_count=0,
        tensors={
            "obs": torch.zeros(0, 4, dtype=torch.float32),
            "flag": torch.zeros(0, dtype=torch.bool),
        },
        template=SimpleNamespace(kind="empty"),
        fields=(),
        inline_rows=[],
        step_field_layout={"obs": ("tensor", ("obs",))},
        static_fields={"role": "empty", "seed": 0},
    )


def _varied_columnar_envelope() -> ReplayPushEnvelope:
    long_episode = _sa_episode(episode_num=11, step_count=16)
    short_episode = _sa_episode(episode_num=7, step_count=9)
    encoded = encode_replay_push_payload({
        "long": long_episode,
        "short": short_episode,
    })
    extra = ReplayTransitionEnvelope(
        transition_type=SAEpisodeTransition,
        episode_num=0,
        starts_with_episode=True,
        has_starts_with_episode=True,
        prepared_episode=_empty_prepared(),
        tensor_ownership=TensorOwnership.TRANSFERRED,
    )
    extra_tensors = {
        "f32": torch.linspace(-3.5, 8.25, 16 * 6, dtype=torch.float32).reshape(16, 6),
        "f64": torch.arange(16, dtype=torch.float64).mul_(0.125),
        "i64": torch.arange(16, dtype=torch.int64),
        "i32": torch.arange(16, dtype=torch.int32).reshape(16, 1),
        "mask": torch.tensor([index % 2 == 0 for index in range(16)]),
    }
    extra_prepared = PreparedReplayEpisode(
        step_count=16,
        tensors=extra_tensors,
        template=("row", (16,)),
        fields=encoded.episodes[ReplayPushEntryKey("long", 0)].prepared_episode.fields,
        inline_rows=[(index, f"inline-{index}", {"k": index}) for index in range(16)],
        step_field_layout=dict(
            encoded.episodes[ReplayPushEntryKey("long", 0)].prepared_episode.step_field_layout
        ),
        static_fields={"name": "varied", "scale": 2.5, "flags": (True, False)},
    )
    return ReplayPushEnvelope(
        version=ReplayPushEnvelope.CURRENT_VERSION,
        episodes={
            **encoded.episodes,
            "empty": extra,
            "varied": ReplayTransitionEnvelope(
                transition_type=SAEpisodeTransition,
                episode_num=99,
                starts_with_episode=False,
                has_starts_with_episode=True,
                prepared_episode=extra_prepared,
                tensor_ownership=TensorOwnership.BORROWED,
            ),
        },
    )


def _assert_value_equal(actual: Any, expected: Any, *, path: str = "root") -> None:
    assert type(actual) is type(expected), f"{path}: type {type(actual)!r} != {type(expected)!r}"
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype, path
        assert tuple(actual.shape) == tuple(expected.shape), path
        assert actual.device == expected.device, path
        assert actual.layout == expected.layout, path
        assert actual.requires_grad == expected.requires_grad, path
        assert torch.equal(actual, expected), path
        actual_cpu = actual.detach().cpu().contiguous()
        expected_cpu = expected.detach().cpu().contiguous()
        try:
            assert actual_cpu.numpy().tobytes() == expected_cpu.numpy().tobytes(), path
        except (TypeError, RuntimeError):
            pass
        return
    if isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype, path
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert list(actual.keys()) == list(expected.keys()), path
        for key in expected:
            _assert_value_equal(actual[key], expected[key], path=f"{path}.{key!r}")
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_value_equal(actual_item, expected_item, path=f"{path}[{index}]")
        return
    if is_dataclass(expected):
        assert [field.name for field in fields(actual)] == [field.name for field in fields(expected)], path
        for field in fields(expected):
            _assert_value_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(expected, SimpleNamespace):
        _assert_value_equal(vars(actual), vars(expected), path=path)
        return
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _assert_replay_push_envelope_equal(actual: Any, expected: ReplayPushEnvelope) -> None:
    assert isinstance(actual, ReplayPushEnvelope)
    assert actual.version == expected.version
    assert list(actual.episodes) == list(expected.episodes)
    for name in expected.episodes:
        actual_item = actual.episodes[name]
        expected_item = expected.episodes[name]
        assert isinstance(actual_item, ReplayTransitionEnvelope)
        assert isinstance(expected_item, ReplayTransitionEnvelope)
        _assert_value_equal(actual_item, expected_item, path=f"episodes[{name!r}]")


def _clone_envelope(envelope: ReplayPushEnvelope) -> ReplayPushEnvelope:
    return pickle.loads(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))


def _chunk_bytes_for(envelope: ReplayPushEnvelope, *, min_chunks: int = 3) -> int:
    nbytes = envelope_tensor_nbytes(envelope)
    chunk_bytes = max(64, nbytes // max(min_chunks, 2))
    chunks = split_replay_push_envelope(envelope, chunk_bytes)
    assert len(chunks) >= min_chunks, (
        f"expected >= {min_chunks} chunks, got {len(chunks)} "
        f"(chunk_bytes={chunk_bytes}, nbytes={nbytes})"
    )
    return chunk_bytes


def _bind_inprocess_torch_rpc(client: ReplayBufferTorchRpcClient, service: TorchReplayBufferService):
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def call(method, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("_rpc_context", None)
        wire_args = pickle.loads(
            pickle.dumps(prepare_torch_rpc_payload(args), protocol=pickle.HIGHEST_PROTOCOL)
        )
        wire_kwargs = pickle.loads(
            pickle.dumps(prepare_torch_rpc_payload(kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        )
        calls.append((method, wire_args, wire_kwargs))
        return getattr(service, method)(*wire_args, **wire_kwargs)

    client.client.call = call
    return calls


def test_split_assemble_roundtrip_is_bit_identical_including_pickled_chunks():
    original = _varied_columnar_envelope()
    expected = _clone_envelope(original)
    chunk_bytes = _chunk_bytes_for(original, min_chunks=3)
    chunks = split_replay_push_envelope(original, chunk_bytes)
    assert all(isinstance(chunk, ReplayPushChunkEnvelope) for chunk in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[-1].is_last is True
    assert sum(chunk.step_count for chunk in chunks) == max(
        item.prepared_episode.step_count for item in original.episodes.values()
    )

    wired = [
        pickle.loads(pickle.dumps(prepare_torch_rpc_payload(chunk), protocol=pickle.HIGHEST_PROTOCOL))
        for chunk in chunks
    ]
    assembled = assemble_replay_push_envelope(wired)
    _assert_replay_push_envelope_equal(assembled, expected)


def test_split_returns_empty_when_payload_is_below_chunk_threshold():
    original = _varied_columnar_envelope()
    assert split_replay_push_envelope(original, envelope_tensor_nbytes(original)) == []
    assert split_replay_push_envelope(original, 0) == []
    assert split_replay_push_envelope({"not": "envelope"}, 64) == []


def test_chunked_torch_rpc_push_delivers_bit_identical_envelope():
    original = _varied_columnar_envelope()
    expected = _clone_envelope(original)
    chunk_bytes = _chunk_bytes_for(original, min_chunks=3)
    target = _ReplayTarget()
    service = TorchReplayBufferService(
        target,
        TorchRpcConfig(
            max_concurrent_replay_pushes=1,
            replay_push_permit_lease_timeout_seconds=5.0,
            replay_push_chunk_bytes=chunk_bytes,
        ),
    )
    client = ReplayBufferTorchRpcClient(
        config=TorchRpcConfig(
            replay_push_permit_poll_interval_seconds=0.01,
            replay_push_chunk_bytes=chunk_bytes,
        )
    )
    calls = _bind_inprocess_torch_rpc(client, service)

    result = client.push(
        original,
        sample_role_name="sample-1",
        episode_num=11,
        generation=3,
        notify_learning=False,
        save=False,
    )

    assert result == {"stored": 11}
    push_payloads = [args[1] for method, args, _kwargs in calls if method == "push_with_permit"]
    assert len(push_payloads) >= 3
    assert all(isinstance(payload, ReplayPushChunkEnvelope) for payload in push_payloads)
    assert push_payloads[0].chunk_index == 0
    assert push_payloads[-1].is_last is True
    assert len(target.calls) == 1
    received, kwargs = target.calls[0]
    assert kwargs["sample_role_name"] == "sample-1"
    assert kwargs["episode_num"] == 11
    assert kwargs["generation"] == 3
    _assert_replay_push_envelope_equal(received, expected)


def test_below_threshold_torch_rpc_push_stays_oneshot_envelope():
    original = _varied_columnar_envelope()
    expected = _clone_envelope(original)
    target = _ReplayTarget()
    service = TorchReplayBufferService(
        target,
        TorchRpcConfig(max_concurrent_replay_pushes=1, replay_push_permit_lease_timeout_seconds=5.0),
    )
    client = ReplayBufferTorchRpcClient(
        config=TorchRpcConfig(
            replay_push_permit_poll_interval_seconds=0.01,
            replay_push_chunk_bytes=0,
        )
    )
    calls = _bind_inprocess_torch_rpc(client, service)

    client.push(original, sample_role_name="sample-1", episode_num=11, generation=1)

    push_payloads = [args[1] for method, args, _kwargs in calls if method == "push_with_permit"]
    assert len(push_payloads) == 1
    assert isinstance(push_payloads[0], ReplayPushEnvelope)
    assert not isinstance(push_payloads[0], ReplayPushChunkEnvelope)
    _assert_replay_push_envelope_equal(target.calls[0][0], expected)


def test_out_of_order_and_incomplete_chunks_are_rejected_without_pushing():
    original = _varied_columnar_envelope()
    chunks = split_replay_push_envelope(original, _chunk_bytes_for(original))
    target = _ReplayTarget()
    service = TorchReplayBufferService(
        target,
        TorchRpcConfig(max_concurrent_replay_pushes=1, replay_push_permit_lease_timeout_seconds=1.0),
    )
    permit = service.acquire_replay_push_permit("sample-1", "request-1", 11, 0)
    first = service.push_with_permit(
        permit["token"], chunks[0], sample_role_name="sample-1", episode_num=11, generation=0,
    )
    assert first == {"__replay_push_permit__": True, "state": "chunking", "chunk_index": 0}

    with pytest.raises(ValueError, match="out of order"):
        service.push_with_permit(
            permit["token"], chunks[2], sample_role_name="sample-1", episode_num=11, generation=0,
        )
    with pytest.raises(ValueError, match="chunk count mismatch|out of order"):
        assemble_replay_push_envelope(chunks[:-1])

    assert target.calls == []
    assert service.replay_push_permit_state()["active_count"] == 1


def test_chunking_session_occupies_capacity_and_lease_reclaims_incomplete_chunks():
    original = _varied_columnar_envelope()
    chunks = split_replay_push_envelope(original, _chunk_bytes_for(original))
    target = _ReplayTarget()
    service = TorchReplayBufferService(
        target,
        TorchRpcConfig(
            max_concurrent_replay_pushes=1,
            replay_push_permit_lease_timeout_seconds=0.03,
        ),
    )
    first = service.acquire_replay_push_permit("sample-1", "request-1", 11, 0)
    service.push_with_permit(
        first["token"], chunks[0], sample_role_name="sample-1", episode_num=11, generation=0,
    )
    denied = service.acquire_replay_push_permit("sample-2", "request-2", 12, 0)
    assert denied["acquired"] is False
    assert denied["reason"] == "capacity"
    assert target.calls == []

    time.sleep(0.04)
    second = service.acquire_replay_push_permit("sample-2", "request-2", 12, 0)
    assert second["acquired"] is True
    assert service.replay_push_permit_result(first["token"], "sample-1")["state"] == "unknown_or_expired"
    assert target.calls == []


def test_pushing_state_is_not_reclaimed_while_assembled_push_runs():
    original = _varied_columnar_envelope()
    chunks = split_replay_push_envelope(original, _chunk_bytes_for(original))
    entered = threading.Event()
    unblock = threading.Event()

    class BlockingTarget(_ReplayTarget):
        def push(self, episode_data, **kwargs):
            self.calls.append((episode_data, kwargs))
            entered.set()
            assert unblock.wait(1)
            return {"stored": kwargs.get("episode_num")}

    target = BlockingTarget()
    service = TorchReplayBufferService(
        target,
        TorchRpcConfig(
            max_concurrent_replay_pushes=1,
            replay_push_permit_lease_timeout_seconds=0.03,
        ),
    )
    permit = service.acquire_replay_push_permit("sample-1", "request-1", 11, 0)
    for chunk in chunks[:-1]:
        service.push_with_permit(
            permit["token"], chunk, sample_role_name="sample-1", episode_num=11, generation=0,
        )

    results = []
    uploader = threading.Thread(target=lambda: results.append(service.push_with_permit(
        permit["token"], chunks[-1], sample_role_name="sample-1", episode_num=11, generation=0,
    )))
    uploader.start()
    assert entered.wait(1)
    time.sleep(0.04)
    assert service.replay_push_permit_state()["active_count"] == 1
    duplicate = service.push_with_permit(
        permit["token"], chunks[-1], sample_role_name="sample-1", episode_num=11, generation=0,
    )
    assert duplicate == {"__replay_push_permit__": True, "state": "pushing"}
    unblock.set()
    uploader.join(1)
    assert results == [{"stored": 11}]
    assert len(target.calls) == 1


def test_replay_buffer_dispatch_method_names_stay_unchanged():
    service = TorchReplayBufferService(_ReplayTarget(), TorchRpcConfig())
    assert set(build_replay_buffer_methods(service)) == {
        "size",
        "state",
        "agent_size",
        "agent_names",
        "reset_sliding_idx",
        "ordered_buffer",
        "push",
        "sample",
        "save_buffer",
        "acquire_replay_push_permit",
        "push_with_permit",
        "release_replay_push_permit",
        "replay_push_permit_result",
        "replay_push_permit_state",
    }


def test_grpc_replay_client_still_sends_oneshot_push():
    recorded = []

    class _Stub:
        def call(self, method, *args, **kwargs):
            recorded.append((method, args, kwargs))
            return "stored"

    client = ReplayBufferRpcClient.__new__(ReplayBufferRpcClient)
    client.client = _Stub()
    episode = _sa_episode(episode_num=3, step_count=4)
    assert client.push(episode, agent_name="agent", notify_learning=False) == "stored"
    assert recorded[0][0] == "push"
    assert isinstance(recorded[0][1][0], ReplayPushEnvelope)
    assert not isinstance(recorded[0][1][0], ReplayPushChunkEnvelope)

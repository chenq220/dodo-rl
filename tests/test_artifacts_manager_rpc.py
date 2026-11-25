"""ArtifactsManager 薄 RPC：白名单转发，禁止 worker 调归档/关闭。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rpc.dispatch import (
    ARTIFACTS_MANAGER,
    ARTIFACTS_MANAGER_ALLOWED_METHODS,
    build_dispatcher,
)
from rpc.grpc.service.artifacts_manager_rpc import ArtifactsManagerRpcClient
from rpc.torch_rpc.client import ArtifactsManagerTorchRpcClient


class _ArtifactsManagerStub:
    def register(self, *args, **kwargs):
        return "register"

    def put(self, *args, **kwargs):
        return "put"

    def put_bytes(self, *args, **kwargs):
        return "put_bytes"

    def put_log(self, *args, **kwargs):
        return "put_log"

    def expect_producer(self, *args, **kwargs):
        return "expect_producer"

    def complete(self, *args, **kwargs):
        return "complete"

    def archive_task(self, *args, **kwargs):
        raise AssertionError("archive_task must not be dispatched")

    def close(self):
        raise AssertionError("close must not be dispatched")

    def ensure_task(self, *args, **kwargs):
        raise AssertionError("ensure_task must not be dispatched")

    def prepare_output(self, *args, **kwargs):
        raise AssertionError("prepare_output must not be dispatched")


@pytest.mark.parametrize("method", ARTIFACTS_MANAGER_ALLOWED_METHODS)
def test_dispatcher_exposes_whitelist_methods(method):
    dispatcher = build_dispatcher(ARTIFACTS_MANAGER, _ArtifactsManagerStub())
    assert dispatcher.invoke(method, ("task", "train", "arg"), {}) == method


@pytest.mark.parametrize("method", ("archive_task", "close", "ensure_task", "prepare_output"))
def test_dispatcher_rejects_center_only_methods(method):
    dispatcher = build_dispatcher(ARTIFACTS_MANAGER, _ArtifactsManagerStub())
    with pytest.raises(AttributeError, match="not exposed"):
        dispatcher.invoke(method, (), {})


def test_torch_rpc_client_has_no_archive_or_ensure():
    assert not hasattr(ArtifactsManagerTorchRpcClient, "archive_task")
    assert not hasattr(ArtifactsManagerTorchRpcClient, "ensure_task")
    assert not hasattr(ArtifactsManagerTorchRpcClient, "prepare_output")
    client = ArtifactsManagerTorchRpcClient.__new__(ArtifactsManagerTorchRpcClient)
    client.client = MagicMock()
    client.put_log("task", "sample-0", "search")
    client.client.call.assert_called_with("put_log", "task", "sample-0", "search")
    client.complete("task", "search", "log:sample-0")
    client.client.call.assert_called_with("complete", "task", "search", "log:sample-0")
    client.put_bytes("task", "search", object(), b"GIF89a")
    call_args = client.client.call.call_args
    assert call_args.args[:3] == ("put_bytes", "task", "search")
    assert call_args.args[4] == b"GIF89a"


def test_grpc_rpc_client_forwards_whitelist_only():
    client = ArtifactsManagerRpcClient.__new__(ArtifactsManagerRpcClient)
    client.client = MagicMock()
    client.channel = MagicMock()
    client.register("task", "train", object())
    client.put("task", "train", object())
    client.put_bytes("task", "train", object(), b"GIF89a")
    client.put_log("task", "sample-0", "train")
    client.expect_producer("task", "train", "log:sample-0")
    client.complete("task", "train", "log:sample-0")
    names = [call.args[0] for call in client.client.call.call_args_list]
    assert names == [
        "register",
        "put",
        "put_bytes",
        "put_log",
        "expect_producer",
        "complete",
    ]
    assert not hasattr(ArtifactsManagerRpcClient, "archive_task")
    client.close()
    client.channel.close.assert_called_once()
    assert all(call.args[0] != "close" for call in client.client.call.call_args_list)

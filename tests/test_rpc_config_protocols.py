from pathlib import Path
from dataclasses import asdict

import pytest

from config import GrpcConfig, TorchRpcConfig
from rpc.factory import rewrite_rpc_address_host
from utils.toml_config import TomlTrainingConfig


def _load(tmp_path: Path, content: str) -> TomlTrainingConfig:
    path = tmp_path / "rpc.toml"
    path.write_text(content, encoding="utf-8")
    return TomlTrainingConfig(path, project_root=tmp_path)


def test_rpc_config_defaults_to_grpc(tmp_path: Path):
    assert isinstance(_load(tmp_path, "").rpc_config, GrpcConfig)


def test_grpc_config_is_selected(tmp_path: Path):
    config = _load(tmp_path, """
[grpc_config]
task_manager_address = "127.0.0.1:5001"
call_timeout_seconds = 12
""").rpc_config
    assert isinstance(config, GrpcConfig)
    assert config.task_manager_address == "127.0.0.1:5001"
    assert config.call_timeout_seconds == 12


def test_torch_rpc_config_is_selected_and_normalized(tmp_path: Path):
    config = _load(tmp_path, """
[torch_rpc_config]
world_size = 6
init_address = "tcp://127.0.0.1:53151"
num_worker_threads = 24
""").rpc_config
    assert isinstance(config, TorchRpcConfig)
    assert config.world_size == 6
    assert config.init_address == "127.0.0.1:53151"
    assert config.num_worker_threads == 24
    assert config.max_concurrent_replay_pushes == 1
    assert config.replay_push_permit_poll_interval_seconds == pytest.approx(0.1)
    assert config.replay_push_permit_lease_timeout_seconds == pytest.approx(120.0)
    assert config.replay_push_chunk_bytes == 31457280
    assert asdict(config)["max_concurrent_replay_pushes"] == 1


def test_rpc_protocol_sections_are_selected_by_dodo_rpc(tmp_path: Path, monkeypatch):
    content = """
[grpc_config]
task_manager_address = "127.0.0.1:5001"
[torch_rpc_config]
world_size = 2
init_address = "127.0.0.1:53151"
"""
    monkeypatch.delenv("DODO_RPC", raising=False)
    assert isinstance(_load(tmp_path, content).rpc_config, TorchRpcConfig)
    monkeypatch.setenv("DODO_RPC", "grpc")
    grpc = _load(tmp_path, content).rpc_config
    assert isinstance(grpc, GrpcConfig)
    assert grpc.task_manager_address == "127.0.0.1:5001"
    monkeypatch.setenv("DODO_RPC", "trpc")
    assert isinstance(_load(tmp_path, content).rpc_config, TorchRpcConfig)


def test_dodo_rpc_requires_matching_section(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DODO_RPC", "grpc")
    with pytest.raises(ValueError, match="DODO_RPC=grpc requires"):
        _load(tmp_path, """
[torch_rpc_config]
world_size = 2
init_address = "127.0.0.1:53151"
""")
    monkeypatch.setenv("DODO_RPC", "nope")
    with pytest.raises(ValueError, match="DODO_RPC must be grpc or trpc"):
        _load(tmp_path, "")


def test_legacy_rpc_config_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match=r"rpc.toml.*rpc_config.*grpc_config"):
        _load(tmp_path, "[rpc_config]\n")


@pytest.mark.parametrize("field", [
    "world_size = 1\ninit_address = '127.0.0.1:53151'",
    "world_size = 2\ninit_address = ''",
    "world_size = 2\ninit_address = '127.0.0.1:nope'",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nnum_worker_threads = 0",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nmax_concurrent_replay_pushes = 0",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nmax_concurrent_replay_pushes = true",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nmax_concurrent_replay_pushes = 1.5",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nreplay_push_permit_poll_interval_seconds = 0",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nreplay_push_permit_poll_interval_seconds = true",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nreplay_push_permit_lease_timeout_seconds = 0.5",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nreplay_push_permit_lease_timeout_seconds = true",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nreplay_push_chunk_bytes = 1",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nreplay_push_chunk_bytes = true",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nreplay_push_chunk_bytes = 1.5",
    "world_size = 2\ninit_address = '127.0.0.1:53151'\nreplay_push_chunk_bytes = 536870913",
])
def test_invalid_torch_rpc_config_is_rejected(tmp_path: Path, field: str):
    with pytest.raises(ValueError):
        _load(tmp_path, f"[torch_rpc_config]\n{field}\n")


def test_torch_rpc_replay_push_chunk_bytes_zero_disables_chunking(tmp_path: Path):
    config = _load(tmp_path, """
[torch_rpc_config]
world_size = 2
init_address = "127.0.0.1:53151"
replay_push_chunk_bytes = 0
""").rpc_config
    assert isinstance(config, TorchRpcConfig)
    assert config.replay_push_chunk_bytes == 0


def test_rewrite_rpc_address_host_keeps_port() -> None:
    assert rewrite_rpc_address_host("127.0.0.1:53151", "control") == "control:53151"
    assert rewrite_rpc_address_host("tcp://127.0.0.1:53151", "0.0.0.0") == "0.0.0.0:53151"
    assert rewrite_rpc_address_host("127.0.0.1:53151", None) == "127.0.0.1:53151"
    assert rewrite_rpc_address_host("127.0.0.1:53151", "") == "127.0.0.1:53151"
    assert rewrite_rpc_address_host("", "control") == ""


def test_grpc_listen_and_connect_hosts(monkeypatch):
    from rpc.dispatch import TRAIN_TASK_MANAGER
    from rpc.factory import _grpc_target

    config = GrpcConfig(task_manager_address="127.0.0.1:53151")
    monkeypatch.setenv("DODO_RPC_LISTEN_HOST", "0.0.0.0")
    monkeypatch.setenv("DODO_RPC_CONNECT_HOST", "control")
    assert _grpc_target(config, TRAIN_TASK_MANAGER, listen=True) == "0.0.0.0:53151"
    assert _grpc_target(config, TRAIN_TASK_MANAGER) == "control:53151"

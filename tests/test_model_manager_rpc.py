from types import SimpleNamespace
from unittest.mock import MagicMock

from rpc.grpc.service.model_manager_rpc import ModelManagerRpcClient
from rpc.grpc.service.monte_carlo_method_rpc import MonteCarloMethodRpcClient
from rpc.grpc.service.sim_manager_rpc import SimManagerRpcClient


def _client(cls):
    obj = cls.__new__(cls)
    obj.client = MagicMock()
    obj.channel = MagicMock()
    return obj


def test_model_manager_rpc_forwards_and_rejects_nn():
    client = _client(ModelManagerRpcClient)
    client.client.call.return_value = {"ok": True}
    assert client.get_champion("uav") == {"ok": True}
    client.client.call.assert_called_with("get_champion", "uav")
    try:
        client.get_model("uav", output_type="nn")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "nn" in str(exc)
    client.submit_model("uav", "qmix", {"agent_net": {}})
    assert client.client.call.call_args[0][0] == "submit_model"


def test_monte_carlo_method_rpc_forwards():
    client = _client(MonteCarloMethodRpcClient)
    client.receiver({"step": 1})
    client.client.call.assert_called_with("receiver", {"step": 1})
    client.state()
    client.client.call.assert_called_with("state")
    client.close()
    client.channel.close.assert_called_once()


def test_sim_manager_rpc_forwards():
    client = _client(SimManagerRpcClient)
    client.get_sim_info("s1")
    client.client.call.assert_called_with("get_sim_info", "s1")
    client.release_sim("sim-1")
    client.client.call.assert_called_with("release_sim", "sim-1")
    client.get_sim_list()
    client.client.call.assert_called_with("get_sim_list")

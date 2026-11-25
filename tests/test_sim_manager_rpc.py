from unittest.mock import MagicMock

from rpc.grpc.service.sim_manager_rpc import SimManagerRpcClient
from rpc.torch_rpc.client import SimManagerTorchRpcClient


def test_sim_manager_rpc_forwards():
    client = SimManagerRpcClient.__new__(SimManagerRpcClient)
    client.client = MagicMock()
    client.get_sim_info("s1")
    client.client.call.assert_called_with("get_sim_info", "s1")
    client.get_sim_info("s2", sim_type="afsim")
    client.client.call.assert_called_with("get_sim_info", "s2", sim_type="afsim")
    client.release_sim("sim-1")
    client.client.call.assert_called_with("release_sim", "sim-1")
    client.get_sim_list()
    client.client.call.assert_called_with("get_sim_list")
    client.channel = MagicMock()
    client.close()
    client.channel.close.assert_called_once()


def test_torch_sim_manager_forwards_resource_type():
    client = SimManagerTorchRpcClient.__new__(SimManagerTorchRpcClient)
    client.client = MagicMock()
    client.get_sim_info(sample_id="sample-0", sim_type="afsim")
    client.client.call.assert_called_once_with("get_sim_info", sample_id="sample-0", sim_type="afsim")


def test_sim_manager_close_forwards_once_and_closes_transport_on_error():
    import pytest

    for cls in (SimManagerRpcClient, SimManagerTorchRpcClient):
        client = cls.__new__(cls)
        client.client = MagicMock()
        client.channel = MagicMock()
        client.client.call.side_effect = RuntimeError("cleanup failed")
        with pytest.raises(RuntimeError, match="cleanup failed"):
            client.close()
        client.client.call.assert_called_once_with("close")
        transport = client.channel if cls is SimManagerRpcClient else client.client
        transport.close.assert_called_once_with()
        client.close()
        client.client.call.assert_called_once_with("close")


def test_sim_manager_dispatch_exposes_force_close():
    from rpc.dispatch import build_sim_manager_methods

    manager = MagicMock()
    build_sim_manager_methods(manager)["close"]()
    manager.close.assert_called_once_with()


def test_sim_manager_close_accepts_confirmed_center_shutdown():
    from common.exception import RpcServerGone

    for cls in (SimManagerRpcClient, SimManagerTorchRpcClient):
        client = cls.__new__(cls)
        client.client = MagicMock()
        client.channel = MagicMock()
        client.client.call.side_effect = RpcServerGone(service="SimManager", method="close", attempts=1, interval=0.0, detail="center finished")
        client.close()
        client.close()
        client.client.call.assert_called_once_with("close")
        transport = client.channel if cls is SimManagerRpcClient else client.client
        transport.close.assert_called_once_with()

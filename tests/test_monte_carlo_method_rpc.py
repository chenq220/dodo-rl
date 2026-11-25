from unittest.mock import MagicMock

from rpc.grpc.service.monte_carlo_method_rpc import MonteCarloMethodRpcClient


def test_monte_carlo_method_rpc_forwards():
    client = MonteCarloMethodRpcClient.__new__(MonteCarloMethodRpcClient)
    client.client = MagicMock()
    client.channel = MagicMock()
    client.receiver({"step": 1})
    client.client.call.assert_called_with("receiver", {"step": 1})
    client.state()
    client.client.call.assert_called_with("state")
    client.close()
    client.channel.close.assert_called_once()

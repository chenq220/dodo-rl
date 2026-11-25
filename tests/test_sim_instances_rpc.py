"""RL RPC adapter contracts and method-scoped transport deadlines."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rpc.grpc import dodorl_pb2
from rpc.grpc.codec import pack_value
from rpc.grpc.service import RpcClient
from rpc.grpc.service.sim_manager_rpc import SimManagerRpcClient
from rpc.dispatch import build_sim_manager_methods
from rpc.torch_rpc.client import SimManagerTorchRpcClient


def test_rpc_deadline_override_is_per_method_not_business_argument():
    stub = SimpleNamespace(Call=Mock(return_value=dodorl_pb2.RpcResponse(ok=True, payload=pack_value(True))))
    client = RpcClient(stub, method_timeouts={"reacquire_simulation": 600.0})
    assert client.call("reacquire_simulation", request_id="one") is True
    assert stub.Call.call_args.kwargs["timeout"] == 600.0
    assert client.call("get_sim_info", "sample") is True
    assert stub.Call.call_args.kwargs["timeout"] == 60.0
    assert client.call_timeout == 60.0


def test_clients_forward_recovery_and_lease_fields():
    for cls in (SimManagerRpcClient, SimManagerTorchRpcClient):
        client = cls.__new__(cls)
        client.client = Mock()
        client.reacquire_simulation(sim_id="sim.0", request_id="recover")
        client.client.call.assert_called_with("reacquire_simulation", sim_id="sim.0", request_id="recover")
        client.release_sim("sim.0", generation=2, lease_id="lease")
        client.client.call.assert_called_with("release_sim", "sim.0", generation=2, lease_id="lease")


def test_failure_reporting_is_not_rpc_exposed():
    target = SimpleNamespace(**{name: Mock() for name in (
        "get_sim_info", "release_sim", "get_sim_list", "reacquire_simulation", "validate_sim_lease", "report_sim_failure")})
    methods = build_sim_manager_methods(target)
    assert "reacquire_simulation" in methods
    assert "report_sim_failure" not in methods




def test_role_initialization_failure_immediately_reclaims_owned_manager():
    from common.simulation import close_sim_manager_on_init_error
    manager = Mock()
    class BrokenOwner:
        @close_sim_manager_on_init_error
        def __init__(self):
            self.sim_manager = manager
            raise ValueError("invalid environment")
    with pytest.raises(ValueError, match="invalid environment"):
        BrokenOwner()
    manager.close.assert_called_once_with()


def test_offline_factory_does_not_import_or_start_instance_driver():
    from common.simulation import create_sim_manager
    assert create_sim_manager(SimpleNamespace(use_sim=True), sample_type="offline") is None




def test_grpc_adapter_scopes_optional_cancellation_per_request(monkeypatch):
    import sys
    from contextvars import ContextVar
    from rpc.dispatch import RpcMethodDispatcher
    from rpc.grpc.service import SimManagerRpcServicer
    from rpc.grpc.codec import pack_call, unpack_value
    cancel_check = ContextVar('mock_optional_cancel_check', default=None)
    monkeypatch.setitem(sys.modules, 'dodo_sim.instances.base', SimpleNamespace(cancel_check=cancel_check))
    def recovery():
        return cancel_check.get()()
    service = SimManagerRpcServicer(RpcMethodDispatcher(None, {'reacquire_simulation': recovery}))
    context = SimpleNamespace(is_active=lambda: False)
    response = service.Call(dodorl_pb2.RpcRequest(method='reacquire_simulation', payload=pack_call()), context)
    assert response.ok and unpack_value(response.payload) is True
    assert cancel_check.get() is None

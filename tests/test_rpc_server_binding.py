import pytest

from rpc.grpc.service import create_grpc_server


def test_grpc_server_rejects_duplicate_port_binding():
    first = create_grpc_server()
    second = create_grpc_server()
    port = first.add_insecure_port("127.0.0.1:0")
    assert port > 0
    first.start()
    try:
        with pytest.raises(RuntimeError, match="gRPC服务端口绑定失败"):
            second.add_insecure_port(f"127.0.0.1:{port}")
    finally:
        second.stop(0)
        first.stop(0)

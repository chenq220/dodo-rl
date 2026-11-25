from types import SimpleNamespace

from roles.common.simulation_logging import log_simulation_middleware


class _ListLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_log_simulation_middleware_uses_effective_sim_config() -> None:
    logger = _ListLogger()
    middleware = SimpleNamespace(
        type="zmq", receive_buffer_len=200, receive_timeout=10.0, send_timeout=10.0,
        remote_ip="10.0.0.8", remote_port=9094, remote_mode="connect",
        local_ip="127.0.0.1", local_port=8001, local_mode="bind",
        send_socket_type="REQ", receive_socket_type="PULL", receive_socket_buffer_bytes=1048576,
    )
    env = SimpleNamespace(
        sim=SimpleNamespace(config=SimpleNamespace(middleware=middleware)),
    )

    assert log_simulation_middleware(env, logger)
    assert logger.messages == [
        "Simulation middleware: type=zmq, receive_buffer_len=200, "
        "receive_timeout=10.0, send_timeout=10.0, remote_ip=10.0.0.8, "
        "remote_port=9094, remote_mode=connect, local_ip=127.0.0.1, "
        "local_port=8001, local_mode=bind, send_socket_type=REQ, "
        "receive_socket_type=PULL, receive_socket_buffer_bytes=1048576"
    ]


def test_log_simulation_middleware_skips_environment_without_simulation() -> None:
    logger = _ListLogger()

    assert not log_simulation_middleware(SimpleNamespace(sim=None), logger)
    assert logger.messages == []

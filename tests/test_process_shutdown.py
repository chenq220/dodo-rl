from __future__ import annotations

import gc
import inspect
import signal
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
for _path in (SRC, PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common.process_shutdown import (
    handle_os_signal,
    is_close_registered,
    process_close,
    reset_for_tests,
    shutdown,
)
from scheduler.rollout_sample import RolloutSampleWorker


@pytest.fixture(autouse=True)
def _isolate_process_shutdown():
    reset_for_tests()
    yield
    reset_for_tests()


class _Hooked:
    def __init__(self):
        self.calls = 0
        self.closed = False

    @process_close
    def close(self):
        self.calls += 1
        self.closed = True


class _Plain:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_decorator_registers_close_without_role():
    hooked = _Hooked()
    assert is_close_registered(hooked) is True
    shutdown()
    assert hooked.closed is True
    assert hooked.calls == 1


def test_undecorated_close_is_not_called_by_shutdown():
    plain = _Plain()
    assert is_close_registered(plain) is False
    shutdown()
    assert plain.closed is False


def test_shutdown_calls_close_lifo():
    order: list[str] = []

    class First:
        @process_close
        def close(self):
            order.append("first")

    class Second:
        @process_close
        def close(self):
            order.append("second")

    first = First()
    second = Second()
    shutdown()
    assert order == ["second", "first"]
    assert first is not None and second is not None


def test_double_shutdown_calls_close_once():
    hooked = _Hooked()
    shutdown()
    shutdown()
    assert hooked.calls == 1


def test_one_close_failure_does_not_block_others():
    order: list[str] = []

    class Boom:
        @process_close
        def close(self):
            order.append("boom")
            raise RuntimeError("close failed")

    class Ok:
        @process_close
        def close(self):
            order.append("ok")

    boom = Boom()
    ok = Ok()
    shutdown()
    assert order == ["ok", "boom"]
    assert boom is not None and ok is not None


def test_collected_instance_is_dropped_from_registry():
    calls: list[int] = []

    class Transient:
        @process_close
        def close(self):
            calls.append(1)

    obj = Transient()
    assert is_close_registered(obj) is True
    del obj
    gc.collect()
    shutdown()
    assert calls == []


def test_undecorated_internal_pool_not_registered():
    class TensorPool:
        def close(self):
            self.closed = True

    pool = TensorPool()
    assert is_close_registered(pool) is False


def test_first_process_close_instance_installs_signal_handlers():
    assert signal.getsignal(signal.SIGTERM) is not handle_os_signal
    hooked = _Hooked()
    assert is_close_registered(hooked) is True
    assert signal.getsignal(signal.SIGTERM) is handle_os_signal
    assert signal.getsignal(signal.SIGINT) is handle_os_signal


def test_role_run_does_not_call_process_shutdown():
    from roles.sample import SampleRole
    from roles.train import TrainRole

    assert "process_shutdown" not in inspect.getsource(SampleRole.run)
    assert "process_shutdown" not in inspect.getsource(TrainRole.run)
    assert "process_shutdown" not in inspect.getsource(SampleRole)
    assert "process_shutdown" not in inspect.getsource(TrainRole)


def test_signal_handler_invokes_close_and_exit(monkeypatch):
    hooked = _Hooked()
    exits: list[int] = []

    def fake_exit(code=0):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr("common.process_shutdown.sys.exit", fake_exit)
    with pytest.raises(SystemExit):
        handle_os_signal(signal.SIGTERM, None)
    assert hooked.calls == 1
    assert exits == [0]


def test_handler_after_shutdown_is_idempotent(monkeypatch):
    hooked = _Hooked()
    shutdown()
    assert hooked.calls == 1

    exits: list[int] = []

    def fake_exit(code=0):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr("common.process_shutdown.sys.exit", fake_exit)
    with pytest.raises(SystemExit):
        handle_os_signal(signal.SIGTERM, None)
    assert hooked.calls == 1
    assert exits == [0]




def test_rollout_sample_worker_init_does_not_bind_signals():
    source = inspect.getsource(RolloutSampleWorker.__init__)
    assert "signal.signal" not in source
    assert not hasattr(RolloutSampleWorker, "_register_signal_handler")
    assert not hasattr(RolloutSampleWorker, "_signal_handler")




def test_shutdown_calls_already_loaded_optional_sdk_without_importing(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    callback = Mock()
    monkeypatch.setitem(sys.modules, 'dodo_sim', SimpleNamespace(close_resources=callback))
    shutdown()
    callback.assert_called_once_with()
    shutdown()
    callback.assert_called_once_with()


def test_shutdown_does_not_import_absent_optional_sdk(monkeypatch):
    import builtins
    original = builtins.__import__
    monkeypatch.delitem(sys.modules, 'dodo_sim', raising=False)
    def no_optional_import(name, *args, **kwargs):
        if name == 'dodo_sim' or name.startswith('dodo_sim.'):
            pytest.fail('Closing RL must not import an absent optional SDK')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_optional_import)
    shutdown()
    assert 'dodo_sim' not in sys.modules

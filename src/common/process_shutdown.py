"""进程级优雅关停：@process_close 注册表 + SIGINT/SIGTERM 共用 shutdown()。"""

from __future__ import annotations

import atexit
import functools
import signal
import sys
import threading
import types
import weakref
from typing import Any, Callable, Optional

from common.logger import init_logger

LOG = init_logger("process_shutdown")

_registry: list[weakref.WeakMethod] = []
_lock = threading.Lock()
_shutting_down = False
_shutdown_done = False
_handlers_installed = False
_atexit_registered = False
_saved_handlers: dict[int, Any] = {}


class process_close:
    """装饰实例 close()：类构造完成后把 bound close 以 WeakMethod 注册进进程表。"""

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        functools.update_wrapper(self, func)
        self.__isabstractmethod__ = bool(getattr(func, "__isabstractmethod__", False))

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        existing = owner.__dict__.get("__init__")
        if getattr(existing, "_process_close_wrapped_for", None) is owner:
            return
        orig_init = owner.__init__

        def wrapped_init(instance, *args, **kwargs):
            orig_init(instance, *args, **kwargs)
            register_close(getattr(instance, name))

        wrapped_init._process_close_wrapped_for = owner
        try:
            functools.update_wrapper(wrapped_init, orig_init, updated=())
        except AttributeError:
            wrapped_init.__name__ = "__init__"
        owner.__init__ = wrapped_init

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return types.MethodType(self.func, obj)


def register_close(bound_close: Callable[..., Any]) -> None:
    instance = getattr(bound_close, "__self__", None)
    if instance is None:
        return
    with _lock:
        _prune_unlocked()
        for weak_method in _registry:
            method = weak_method()
            if method is not None and method.__self__ is instance:
                return
        try:
            _registry.append(weakref.WeakMethod(bound_close))
        except TypeError:
            LOG.warning("process_close: bound close is not weak-referenceable, skip register")
            return
    install_process_shutdown()


def _prune_unlocked() -> None:
    _registry[:] = [weak_method for weak_method in _registry if weak_method() is not None]


def is_close_registered(instance: Any) -> bool:
    with _lock:
        _prune_unlocked()
        for weak_method in _registry:
            method = weak_method()
            if method is not None and method.__self__ is instance:
                return True
        return False


def registered_count() -> int:
    with _lock:
        _prune_unlocked()
        return len(_registry)


def shutdown() -> None:
    """LIFO 调用已注册 close；单次失败只记日志。信号与进程退出（atexit）都走这里。"""
    global _shutting_down, _shutdown_done
    with _lock:
        if _shutting_down or _shutdown_done:
            return
        _shutting_down = True
        _prune_unlocked()
        methods = []
        for weak_method in reversed(_registry):
            method = weak_method()
            if method is not None:
                methods.append(method)
        _registry.clear()
    try:
        for method in methods:
            try:
                method()
            except Exception:
                LOG.exception("process_shutdown: close() failed")
        # SIM owns an independent cleanup registry and never overrides our signals.
        # Close it through its public API after the RL environments/roles release resources.
        try:
            sim_sdk = sys.modules.get("dodo_sim")
            if sim_sdk is not None:
                sim_sdk.close_resources()
        except Exception:
            LOG.exception("process_shutdown: simulation resource cleanup failed")
        _shutdown_done = True
    finally:
        _shutting_down = False


def handle_os_signal(signum: int, frame: Optional[Any]) -> None:
    """SIGINT/SIGTERM 入口：shutdown 后 sys.exit(0)。测试应直接调用本函数，不要 kill pytest。"""
    LOG.info(f"process_shutdown captured signal: signum={signum}")
    shutdown()
    sys.exit(0)


def install_process_shutdown() -> None:
    """首次 @process_close 实例构造时自动调用；主线程幂等安装 SIGINT/SIGTERM 与 atexit。

    不是 Role API。Role 不感知关停。
    """
    global _handlers_installed, _atexit_registered
    if _handlers_installed:
        return
    if threading.current_thread() is not threading.main_thread():
        LOG.warning("install_process_shutdown skipped: not main thread")
        return
    for sig in (signal.SIGINT, signal.SIGTERM):
        _saved_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, handle_os_signal)
    if not _atexit_registered:
        atexit.register(shutdown)
        _atexit_registered = True
    _handlers_installed = True


def reset_for_tests() -> None:
    """清空注册表与关停标志；若本模块安装过信号则还原。供单测夹具使用。"""
    global _shutting_down, _shutdown_done, _handlers_installed, _atexit_registered
    with _lock:
        _registry.clear()
        _shutting_down = False
        _shutdown_done = False
    if _handlers_installed:
        for sig, handler in _saved_handlers.items():
            signal.signal(sig, handler)
        _saved_handlers.clear()
        _handlers_installed = False
    if _atexit_registered:
        try:
            atexit.unregister(shutdown)
        except Exception:
            pass
        _atexit_registered = False

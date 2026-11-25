from __future__ import annotations

import atexit
import functools
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

from common.logger import init_logger


LOG = init_logger("replay_perf")
_ENABLED = os.environ.get("DODO_REPLAY_PERF", "").lower() in {
    "1", "true", "yes", "on",
}
_LOCK = threading.Lock()
_VALUES = defaultdict(list)


def now() -> float:
    return time.perf_counter() if _ENABLED else 0.0


def record_seconds(metric: str, started_at: float) -> None:
    if _ENABLED:
        record(metric, time.perf_counter() - started_at)


def record(metric: str, value: float) -> None:
    if _ENABLED:
        with _LOCK:
            _VALUES[metric].append(float(value))


def timed(metric: str):
    """性能环境变量关闭时直接返回原函数，不在生产热路径增加包装调用。"""
    def decorate(function):
        if not _ENABLED:
            return function

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            started_at = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                record(metric, time.perf_counter() - started_at)

        return wrapper
    return decorate


def snapshot() -> dict:
    with _LOCK:
        values = {name: list(items) for name, items in _VALUES.items()}
    result = {}
    for name, items in values.items():
        ordered = sorted(items)
        count = len(ordered)
        result[name] = {
            "count": count,
            "total": sum(ordered),
            "median": ordered[count // 2] if count else 0.0,
            "p95": ordered[min(count - 1, int(count * 0.95))] if count else 0.0,
        }
    return result


def _stream_closed(stream) -> bool:
    return stream is None or bool(getattr(stream, "closed", False))


def _flush() -> None:
    if not _ENABLED:
        return
    payload = {"pid": os.getpid(), "metrics": snapshot()}
    # logging.Handler.handleError 默认把 emit 异常打到 stderr，进程退出时
    # stdout/stderr 可能已关，try/except LOG.info 接不到。关掉 raiseExceptions。
    if not (_stream_closed(getattr(sys, "stdout", None)) or _stream_closed(getattr(sys, "stderr", None))):
        previous = logging.raiseExceptions
        try:
            logging.raiseExceptions = False
            LOG.info("REPLAY_PERF %s", json.dumps(payload, sort_keys=True))
        except (ValueError, OSError):
            pass
        finally:
            logging.raiseExceptions = previous
    output_dir = os.environ.get("DODO_REPLAY_PERF_DIR")
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"replay_perf_{os.getpid()}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


atexit.register(_flush)

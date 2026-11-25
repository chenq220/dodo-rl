from __future__ import annotations

import pickle
from typing import Any, Dict

from common.exception import RpcCodecException


def pack_call(*args, **kwargs) -> bytes:
    """把一次方法调用参数打包为bytes；内部进程可信通信，优先保证Python对象无损传输。"""
    try:
        return pickle.dumps({"args": args, "kwargs": kwargs}, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise RpcCodecException(method="pack_call", detail=str(exc)) from exc


def unpack_call(payload: bytes) -> tuple[tuple, Dict[str, Any]]:
    """还原pack_call生成的调用参数。"""
    if not payload:
        return (), {}
    try:
        data = pickle.loads(payload)
    except Exception as exc:
        raise RpcCodecException(method="unpack_call", detail=str(exc)) from exc
    return tuple(data.get("args", ())), dict(data.get("kwargs", {}))


def pack_value(value: Any) -> bytes:
    """把返回值打包为bytes。"""
    try:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise RpcCodecException(method="pack_value", detail=str(exc)) from exc


def unpack_value(payload: bytes) -> Any:
    """还原pack_value生成的返回值。"""
    if not payload:
        return None
    try:
        return pickle.loads(payload)
    except Exception as exc:
        raise RpcCodecException(method="unpack_value", detail=str(exc)) from exc

from __future__ import annotations

import sys
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import torch


def estimate_object_memory_bytes(value: Any) -> int:
    """递归估算Python对象、numpy数组和torch Tensor占用的字节数。"""
    return _deep_sizeof(value, seen=set())


def estimate_object_memory_mb(value: Any) -> float:
    """递归估算对象内存占用，单位MB。"""
    return estimate_object_memory_bytes(value) / (1024 * 1024)


def _deep_sizeof(value: Any, seen: set[int]) -> int:
    """递归估算对象大小；用于趋势监控，不作为精确内存审计。"""
    obj_id = id(value)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(value, torch.Tensor):
        return sys.getsizeof(value) + value.element_size() * value.nelement()
    if isinstance(value, np.ndarray):
        return sys.getsizeof(value) + int(value.nbytes)

    size = sys.getsizeof(value)
    if isinstance(value, dict):
        for key, item in value.items():
            size += _deep_sizeof(key, seen)
            size += _deep_sizeof(item, seen)
        return size
    if isinstance(value, (list, tuple, set, frozenset)):
        return size + sum(_deep_sizeof(item, seen) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return size + sum(_deep_sizeof(getattr(value, field.name), seen) for field in fields(value))
    if hasattr(value, "__dict__"):
        return size + _deep_sizeof(vars(value), seen)
    if hasattr(value, "__slots__"):
        for slot in value.__slots__:
            if hasattr(value, slot):
                size += _deep_sizeof(getattr(value, slot), seen)
    return size

"""Admission rules for versioned sampling and learning."""
from __future__ import annotations

from collections.abc import Callable

from sample.model import ROLLING_STRATEGIES, SamplingStrategy


def cursor_tickets(target: int, available: int, has_fresh_data: bool) -> int:
    return max(0, target - available)


def rolling_tickets(target: int, available: int, has_fresh_data: bool) -> int:
    # Once the window is full, a single new rollout advances it.
    return max(target - available, int(not has_fresh_data))


def slice_tickets(target: int, available: int, has_fresh_data: bool) -> int:
    return int(available == 0)


def uniform_step_tickets(target: int, available: int, has_fresh_data: bool) -> int:
    # 有放回抽样只需要一个有效步，版本切换仍需等待新数据。
    return int(not available or not has_fresh_data)


def select_version_rule(strategy: SamplingStrategy) -> Callable[[int, int, bool], int]:
    if strategy == SamplingStrategy.UNIFORM_STEPS:
        return uniform_step_tickets
    if strategy in ROLLING_STRATEGIES:
        return rolling_tickets
    if strategy == SamplingStrategy.TIME_SLICE:
        return slice_tickets
    return cursor_tickets

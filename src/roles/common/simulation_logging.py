from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any


def log_simulation_middleware(env: Any, logger: Any) -> bool:
    """在首次采样前输出 Environment 实际连接的仿真中间件配置。"""
    sim = getattr(env, "sim", None)
    middleware = getattr(getattr(sim, "config", None), "middleware", None)
    if middleware is None:
        return False

    if is_dataclass(middleware):
        items = ((field.name, getattr(middleware, field.name)) for field in fields(middleware))
    elif isinstance(middleware, dict):
        items = middleware.items()
    else:
        items = (
            (name, value)
            for name, value in vars(middleware).items()
            if not name.startswith("_")
        )

    details = ", ".join(
        f"{key}={value.value if isinstance(value, Enum) else value}"
        for key, value in items
    )
    logger.info(f"Simulation middleware: {details}")
    return True


__all__ = ["log_simulation_middleware"]

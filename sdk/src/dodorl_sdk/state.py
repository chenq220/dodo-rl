"""SDK 对外状态基类。"""
from __future__ import annotations

from envs.state.base import CoordinateSystem, Entity, EnvState, State, StateDescribe
from envs.state.spatial import (
    CGCS2000Frame,
    GeoPoint,
    LocalFrame,
    WGS84Frame,
    XYZPosition,
)

__all__ = [
    "CGCS2000Frame",
    "CoordinateSystem",
    "Entity",
    "EnvState",
    "GeoPoint",
    "LocalFrame",
    "State",
    "StateDescribe",
    "WGS84Frame",
    "XYZPosition",
]

from __future__ import annotations

from dataclasses import dataclass, field

from envs.state.base import State


@dataclass
class RegionActivityState(State):
    """Region activity 单帧态势。"""

    observer_x: float = 0.0
    observer_y: float = 0.0
    hex_center_x: float = 0.0
    hex_center_y: float = 0.0
    hex_radius: float = 0.0
    hex_vertices: list[list[float]] = field(default_factory=list)
    hex_vx: float = 0.0
    hex_vy: float = 0.0
    frame_index: int = 0
    action_id: int = 0

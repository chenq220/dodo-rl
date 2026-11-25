from __future__ import annotations

from dataclasses import dataclass, field

from envs.state.base import State


@dataclass
class ThreatRegionState(State):
    """Threat region 单帧态势。"""

    observer_x: float = 0.0
    observer_y: float = 0.0
    goal_x: float = 0.0
    goal_y: float = 0.0
    threat_center_x: float = 0.0
    threat_center_y: float = 0.0
    threat_vertices: list[list[float]] = field(default_factory=list)
    threat_vx: float = 0.0
    threat_vy: float = 0.0
    frame_index: int = 0
    action_id: int = 0

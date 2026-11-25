from __future__ import annotations

from dataclasses import dataclass, field

from envs.state.base import State


@dataclass
class TrackSeekState(State):
    """Track seek 单帧态势。"""

    observer_x: float = 0.0
    observer_y: float = 0.0
    targets: list[list[float]] = field(default_factory=list)
    frame_index: int = 0
    action_id: int = 0

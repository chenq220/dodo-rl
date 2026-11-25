from __future__ import annotations

from dataclasses import dataclass, field

from envs.state.base import State


@dataclass
class TargetSelectionState(State):
    """蓝球目标选择小游戏的单帧态势。"""

    shaping_target_indices: list[int] = field(default_factory=list)
    primary_target_indices: list[int] = field(default_factory=list)
    selected_target: int = -1
    frame_index: int = 0

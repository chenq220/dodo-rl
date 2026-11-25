from __future__ import annotations

from dataclasses import dataclass, field

from envs.state.base import State


@dataclass
class CooperativeSearchThreatState(State):
    """Lightweight public frame for standalone dynamic-threat training."""

    threat_positions: dict[str, list[float]] = field(default_factory=dict)
    headings: dict[str, int | None] = field(default_factory=dict)
    speeds_mps: dict[str, float] = field(default_factory=dict)
    frame_index: int = 0
    total_reward: float = 0.0

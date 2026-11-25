from __future__ import annotations

from dataclasses import dataclass, field

from envs.state.base import State


@dataclass
class CooperativeSearchState(State):
    """Public cooperative-search frame without hidden target truth."""

    drone_positions: dict[str, list[float]] = field(default_factory=dict)
    dead_drone_positions: dict[str, list[float]] = field(default_factory=dict)
    threat_positions: dict[str, list[float]] = field(default_factory=dict)
    headings: dict[str, int | None] = field(default_factory=dict)
    speeds_mps: dict[str, float] = field(default_factory=dict)
    detected_flags: dict[str, bool] = field(default_factory=dict)
    threat_detected_flags: dict[str, bool] = field(default_factory=dict)
    drones_detected_by_threat: dict[str, bool] = field(default_factory=dict)
    visited_counts: list[list[int]] = field(default_factory=list)
    shared_belief: list[list[float]] = field(default_factory=list)
    positive_counts: list[list[int]] = field(default_factory=list)
    negative_counts: list[list[int]] = field(default_factory=list)
    frame_index: int = 0
    total_reward: float = 0.0

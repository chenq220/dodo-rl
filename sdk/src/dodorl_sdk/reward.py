"""SDK 对外奖励塑形函数。"""
from __future__ import annotations

from envs.reward.shaping import (
    region_activity_score,
    region_activity_shaping,
    region_activity_strip_score,
    region_activity_strip_shaping,
    region_threat_score,
    region_threat_shaping,
    region_threat_transition_score,
    region_visible_target_seek_score,
    region_visible_target_seek_shaping,
    target_selection_shaping,
)
from envs.reward.utils import ShapingOutputRange

__all__ = [
    "ShapingOutputRange",
    "target_selection_shaping",
    "region_visible_target_seek_score",
    "region_visible_target_seek_shaping",
    "region_activity_score",
    "region_activity_shaping",
    "region_activity_strip_score",
    "region_activity_strip_shaping",
    "region_threat_score",
    "region_threat_transition_score",
    "region_threat_shaping",
]

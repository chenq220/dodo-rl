"""塑形函数专项测试用 pygame 小游戏。"""

from .target_seek_game import MovingTarget, TargetSeekGame
from .region_activity_game import RegionActivityGame
from .threat_region_game import ThreatRegionGame
from .target_selection_game import TargetSelectionGame

__all__ = [
    "MovingTarget",
    "RegionActivityGame",
    "TargetSeekGame",
    "TargetSelectionGame",
    "ThreatRegionGame",
]

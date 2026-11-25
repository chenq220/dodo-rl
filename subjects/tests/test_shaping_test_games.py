import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from envs.games.shaping_test_games.region_activity_game import RegionActivityGame
from envs.games.shaping_test_games.target_seek_game import TargetSeekGame
from envs.games.shaping_test_games.threat_region_game import ThreatRegionGame


def test_region_activity_game_reset_seeded():
    game = RegionActivityGame(render_mode=None)
    try:
        game.reset(seed=3)
        assert game.NUM_ACTIONS == 4
        pos = game.observer
        game.reset(seed=3)
        assert game.observer == pos
    finally:
        close = getattr(game, "close", None)
        if callable(close):
            close()


def test_target_seek_and_threat_region_reset():
    seek = TargetSeekGame(render_mode=None, seed=5)
    threat = ThreatRegionGame(render_mode=None)
    try:
        seek.reset(seed=5)
        threat.reset(seed=7)
        assert seek.num_targets >= 1
    finally:
        for game in (seek, threat):
            close = getattr(game, "close", None)
            if callable(close):
                close()

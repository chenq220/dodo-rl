import os

from config import EnvConfig
from envs.sense.region_activity_env import RegionActivityEnv
from envs.sense.threat_region_env import ThreatRegionEnv
from envs.sense.track_seek_env import TrackSeekEnv


os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


def _single_agent_config(sense_name: str, agent: str, obs_dim: int) -> EnvConfig:
    return EnvConfig(
        sense_name=sense_name,
        use_sim=False,
        max_step_per_episode=4,
        state_dim=obs_dim,
        agent_names=[agent],
        metadata={"render_mode": "rgb_array", "render_every_step": False, "seed": 7},
    )


def test_track_seek_env_reset():
    env = TrackSeekEnv(_single_agent_config("track_seek", "seeker-0", TrackSeekEnv.OBS_DIM), connect_sim=False)
    try:
        obs, _ = env.reset(seed=5)
        assert obs["seeker-0"].shape == (TrackSeekEnv.OBS_DIM,)
    finally:
        env.close()


def test_threat_region_env_reset():
    env = ThreatRegionEnv(_single_agent_config("threat_region", "runner-0", ThreatRegionEnv.OBS_DIM), connect_sim=False)
    try:
        obs, _ = env.reset(seed=9)
        assert obs["runner-0"].shape == (ThreatRegionEnv.OBS_DIM,)
    finally:
        env.close()


def test_region_activity_env_reset():
    env = RegionActivityEnv(
        _single_agent_config("region_activity", "actor-0", RegionActivityEnv.OBS_DIM),
        connect_sim=False,
    )
    try:
        obs, _ = env.reset(seed=11)
        assert obs["actor-0"].shape == (RegionActivityEnv.OBS_DIM,)
    finally:
        env.close()

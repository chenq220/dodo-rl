"""Single-entity threat policies sharing one RNN on the partitioned map."""
from __future__ import annotations

from dataclasses import fields
import numpy as np
import gymnasium as gym

from config import EnvConfig, ThreatRoamingConfig
from common.exception import EnvironmentException
from envs.base import ParallelEnvironment, LOG
from envs.games.cooperative_search.partition_game import CooperativeSearchPartitionGame, PartitionedSearchConfig
from envs.sense.cooperative_search_threat_env import CooperativeSearchThreatEnv
from envs.sense.threat_roaming_observation import ThreatRoamingObservation


class CooperativeSearchPartitionThreatEnv(CooperativeSearchThreatEnv):
    """Independent roaming rewards; game physics shared with fixed-policy deployment."""

    _REWARD_KEYS = ("waypoint_progress", "waypoint_arrival", "circle_outside", "peer_clearance", "peer_danger")

    def __init__(self, config: EnvConfig, sim_config=None, connect_sim=True, sim_manager=None, role_id=None):
        ParallelEnvironment.__init__(self, config=config, sim_config=sim_config, connect_sim=False,
                                     sim_manager=sim_manager, role_id=role_id)
        metadata = config.metadata or {}
        raw = dict(metadata.get("game_config", {}))
        raw.setdefault("max_steps", config.max_step_per_episode)
        raw.setdefault("step_interval_secs", config.step_interval_secs)
        raw.setdefault("threat_roaming", {})
        self.game_config = self._build_game_config(raw)
        if self.game_config.num_drones != 0:
            raise EnvironmentException(detail="threat training requires num_drones=0")
        self.threat_agents = [f"dynamic_threat_{i}" for i in range(self.game_config.num_dynamic_threats)]
        if self.possible_agents != self.threat_agents:
            raise EnvironmentException(detail="agent_names must match consecutive dynamic threat slots")
        self.observation = ThreatRoamingObservation(self.game_config.threat_roaming)
        self._obs_dim = self.observation.observation_dim
        self._state_dim = len(self.threat_agents) * self._obs_dim
        if config.state_dim not in (0, self._state_dim):
            raise EnvironmentException(detail=f"state_dim must be {self._state_dim}; singleton group state is {self._obs_dim}")
        self.render_mode = metadata.get("render_mode")
        if self.render_mode not in (None, "human", "rgb_array"):
            raise EnvironmentException(detail=f"unsupported render_mode: {self.render_mode}")
        self.render_every_step = bool(metadata.get("render_every_step", False))
        self._seed = metadata.get("seed")
        self.game = CooperativeSearchPartitionGame(self.game_config, render_mode=self.render_mode)
        self.game.set_dynamic_threats_enabled(True)
        self._last_terminated = self._last_truncated = False
        self._last_team_reward = self._episode_team_reward = 0.0
        self.pre_state = self.post_state = self._snapshot()

    @staticmethod
    def _build_game_config(raw):
        unknown = set(raw) - {f.name for f in fields(PartitionedSearchConfig)}
        if unknown:
            raise EnvironmentException(detail=f"unsupported partitioned threat game config: {sorted(unknown)}")
        raw = dict(raw)
        if isinstance(raw.get("threat_roaming"), dict):
            raw["threat_roaming"] = ThreatRoamingConfig(**raw["threat_roaming"])
        return PartitionedSearchConfig(**raw)

    def state_space(self):
        return gym.spaces.Box(-1.0, 1.0, shape=(self._state_dim,), dtype=np.float32)

    def state(self):
        return np.concatenate([self.observation(self.game, a) for a in self.threat_agents])

    def state_for_agent_ids(self, *, agent_name, agent_ids):
        if len(agent_ids) != 1 or agent_ids[0] not in self.threat_agents:
            raise EnvironmentException(detail="roaming training requires singleton threat groups")
        return self.observation(self.game, agent_ids[0])

    def get_obs(self):
        return {a: self.observation(self.game, a) for a in self.threat_agents}

    def reset(self, seed=None, options=None):
        self._reset_common(options)
        self.game.reset(seed=seed if seed is not None else self._seed)
        self._last_terminated = self._last_truncated = False
        self._last_team_reward = self._episode_team_reward = 0.0
        self.pre_state = self.post_state = self._snapshot()
        self.rewards = dict.fromkeys(self.threat_agents, 0.0)
        self.record_sample(reset=True)
        return self.get_obs(), {a: self._info(a) for a in self.threat_agents}

    def step(self, actions):
        self.pre_state = self.post_state
        self.current_actions = self.send_actions(actions)
        self.rewards = self.get_reward()
        self._last_team_reward = sum(self.rewards.values())
        self._episode_team_reward += self._last_team_reward
        self.game.total_reward = self._episode_team_reward
        self.post_state = self._snapshot()
        terminated, truncated, resume = self.done()
        terminations = dict.fromkeys(self.threat_agents, terminated)
        truncations = dict.fromkeys(self.threat_agents, truncated)
        infos = {a: {**self._info(a), "need_resume": resume,
                     "episode_step_limit": self.max_step_per_episode,
                     "step_limit_reached": truncated and not terminated} for a in self.threat_agents}
        if self.render_mode == "human" or self.render_every_step:
            self.render()
        self.step_log()
        self.record_sample()
        self.step_num += 1
        observations = self._parallel_observations_after_step(terminations, truncations)
        return observations, self.rewards, terminations, truncations, infos

    def get_reward(self, **kwargs):
        return {a: sum(self.game.threat_roaming.breakdowns[a].values()) for a in self.threat_agents}

    def _info(self, agent):
        roaming = self.game.threat_roaming
        return {**roaming.metrics[agent], "steps": self.game.step_count,
                "reward_breakdown": roaming.breakdowns[agent].copy(),
                "episode_reward_breakdown": roaming.episode_breakdowns[agent].copy(),
                "episode_reward": sum(roaming.episode_breakdowns[agent].values()),
                **roaming.breakdowns[agent]}

    def step_log(self):
        if self.game.step_count % 50 == 0 or self._last_terminated or self._last_truncated:
            roaming = self.game.threat_roaming
            components = {k: sum(b[k] for b in roaming.episode_breakdowns.values()) for k in self._REWARD_KEYS}
            LOG.info(f"[Ep {self.episode_num} - Step {self.game.step_count}] roaming_reward={self._episode_team_reward:.3f}, episode_components={components}")

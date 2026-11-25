"""QMIX training environment for the cooperative-search background threats."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

import gymnasium as gym
import numpy as np

from common.exception import EnvironmentException
from config import EnvConfig
from envs.action.model import Action, CompositeAction
from envs.base import AgentID, EnvResetReturn, EnvStepReturn, LOG, ObsType, ParallelEnvironment
from envs.games.cooperative_search import (
    NUM_HEADINGS,
    CooperativeSearchConfig,
    CooperativeSearchGame,
    FlightCommand,
)
from envs.sense.cooperative_search_env import get_threat_obs, threat_observation_dim
from envs.state.tests.cooperative_search_threat_state import CooperativeSearchThreatState


class CooperativeSearchThreatEnv(ParallelEnvironment):
    """Train three shared-policy threats to keep clear of the ten search targets."""

    NUM_ACTIONS = NUM_HEADINGS
    _REWARD_KEYS = (
        "clearance_progress",
        "warning_zone",
        "target_intrusion",
        "threat_separation",
        "center_distance",
    )

    def __init__(
        self,
        config: EnvConfig,
        sim_config=None,
        connect_sim=True,
        sim_manager=None,
        role_id=None,
    ) -> None:
        super().__init__(
            config=config,
            sim_config=sim_config,
            connect_sim=False,
            sim_manager=sim_manager,
            role_id=role_id,
        )
        metadata = config.metadata or {}
        raw = dict(metadata.get("game_config", {}))
        raw.setdefault("max_steps", config.max_step_per_episode)
        raw.setdefault("step_interval_secs", config.step_interval_secs)
        self.game_config = self._build_game_config(raw)
        if self.game_config.num_drones != 0:
            raise EnvironmentException(
                detail="cooperative-search threat training requires game_config.num_drones=0"
            )
        self.threat_agents = [
            f"dynamic_threat_{index}"
            for index in range(self.game_config.num_dynamic_threats)
        ]
        if self.possible_agents != self.threat_agents:
            raise EnvironmentException(
                detail=(
                    "threat-training agent_names must match dynamic threat slots: "
                    f"actual={self.possible_agents}, expected={self.threat_agents}"
                )
            )
        self._obs_dim = threat_observation_dim(
            self.game_config.num_targets,
            self.game_config.num_dynamic_threats,
        )
        self._state_dim = (
            self.game_config.num_targets * 2
            + self.game_config.num_dynamic_threats * 4
        )
        if config.state_dim not in (0, self._state_dim):
            raise EnvironmentException(
                detail=f"state_dim mismatch: config={config.state_dim}, expected={self._state_dim}"
            )

        self.render_mode = metadata.get("render_mode")
        if self.render_mode not in (None, "human", "rgb_array"):
            raise EnvironmentException(detail=f"unsupported render_mode: {self.render_mode}")
        self.render_every_step = bool(metadata.get("render_every_step", False))
        self._seed = metadata.get("seed")
        self.clearance_progress_reward_weight = float(
            metadata.get("clearance_progress_reward_weight", 0.10)
        )
        self.warning_zone_penalty = float(metadata.get("warning_zone_penalty", 0.10))
        self.target_intrusion_penalty = float(
            metadata.get("target_intrusion_penalty", 1.0)
        )
        self.threat_separation_distance_km = float(
            metadata.get("threat_separation_distance_km", 60.0)
        )
        self.threat_separation_penalty = float(
            metadata.get("threat_separation_penalty", 0.15)
        )
        self.center_distance_penalty = float(
            metadata.get("center_distance_penalty", 0.15)
        )
        if min(
            self.clearance_progress_reward_weight,
            self.warning_zone_penalty,
            self.target_intrusion_penalty,
            self.threat_separation_penalty,
            self.center_distance_penalty,
        ) < 0:
            raise EnvironmentException(detail="threat-training reward weights must be non-negative")
        if self.threat_separation_distance_km <= 0:
            raise EnvironmentException(
                detail="threat_separation_distance_km must be positive"
            )
        auxiliary_penalty = (
            self.threat_separation_penalty + self.center_distance_penalty
        )
        if (
            self.target_intrusion_penalty <= 0
            or auxiliary_penalty > self.target_intrusion_penalty / 3.0 + 1e-9
        ):
            raise EnvironmentException(
                detail=(
                    "combined threat separation and center penalties must not exceed "
                    "one third of target_intrusion_penalty"
                )
            )

        self.game = CooperativeSearchGame(self.game_config, render_mode=self.render_mode)
        self.game.set_dynamic_threats_enabled(True)
        self._last_terminated = self._last_truncated = False
        self._last_team_reward = self._episode_team_reward = 0.0
        self._last_reward_breakdown = self._empty_reward_breakdown()
        self._episode_reward_breakdown = self._empty_reward_breakdown()
        self.pre_state = self.post_state = self._snapshot()

    @staticmethod
    def _build_game_config(raw: dict[str, Any]) -> CooperativeSearchConfig:
        valid = {item.name for item in fields(CooperativeSearchConfig)}
        unknown = set(raw) - valid
        if unknown:
            raise EnvironmentException(
                detail=f"unsupported cooperative-search threat game config: {sorted(unknown)}"
            )
        return CooperativeSearchConfig(**raw)

    def _empty_reward_breakdown(self) -> dict[str, float]:
        return {key: 0.0 for key in self._REWARD_KEYS}

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def centralized_state_dim(self) -> int:
        return self._state_dim

    def state_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self._state_dim,), dtype=np.float32)

    def state(self) -> np.ndarray:
        values: list[float] = []
        for target_x, target_y in self.game._target_positions:
            values.extend([
                target_x / self.game_config.map_width,
                target_y / self.game_config.map_height,
            ])
        for agent in self.threat_agents:
            x, y = self.game.threat_positions[agent]
            heading = self.game.headings[agent]
            angle = None if heading is None else math.radians(
                heading * 360.0 / NUM_HEADINGS
            )
            values.extend([
                x / self.game_config.map_width,
                y / self.game_config.map_height,
                0.0 if angle is None else math.sin(angle),
                0.0 if angle is None else math.cos(angle),
            ])
        state = np.asarray(values, dtype=np.float32)
        if state.size != self._state_dim:
            raise EnvironmentException(
                detail=f"threat state size mismatch: actual={state.size}, expected={self._state_dim}"
            )
        return state

    def get_obs(self) -> dict[AgentID, ObsType]:
        return {
            agent: get_threat_obs(self.game, agent)
            for agent in self.threat_agents
        }

    def get_avail_agent_actions(self, agent_name: AgentID, init: bool = False) -> list[int]:
        if agent_name not in self.threat_agents:
            raise EnvironmentException(detail=f"unknown threat in action mask: {agent_name}")
        available = set(self.game.available_headings(agent_name))
        mask = [int(heading in available) for heading in range(NUM_HEADINGS)]
        if not any(mask):
            raise EnvironmentException(
                detail=f"empty threat action mask for {agent_name}: {mask}"
            )
        return mask

    def reset(self, seed=None, options=None) -> EnvResetReturn:
        self._reset_common(options)
        self.game.reset(seed=seed if seed is not None else self._seed)
        self._last_terminated = self._last_truncated = False
        self._last_team_reward = self._episode_team_reward = 0.0
        self._last_reward_breakdown = self._empty_reward_breakdown()
        self._episode_reward_breakdown = self._empty_reward_breakdown()
        self.pre_state = self.post_state = self._snapshot()
        observations = self.get_obs()
        infos = {agent: self._info(agent) for agent in self.possible_agents}
        if self.render_mode == "human":
            self.render()
        self.record_sample(reset=True)
        return observations, infos

    def step(self, actions: dict[AgentID, Action | CompositeAction]) -> EnvStepReturn:
        self.pre_state = self.post_state
        previous_distances = self._nearest_target_distances()
        self.current_actions = self.send_actions(actions)
        current_distances = self._nearest_target_distances()
        self._last_reward_breakdown = self._reward_components(
            previous_distances,
            current_distances,
        )
        self._last_team_reward = float(sum(self._last_reward_breakdown.values()))
        self._episode_team_reward += self._last_team_reward
        for key, value in self._last_reward_breakdown.items():
            self._episode_reward_breakdown[key] += value
        self.game.total_reward = self._episode_team_reward
        self.post_state = self._snapshot()
        terminated, truncated, need_resume = self.done()
        self.rewards = self.get_reward()
        terminations = {agent: terminated for agent in self.possible_agents}
        truncations = {agent: truncated for agent in self.possible_agents}
        infos = {
            agent: {
                **self._info(agent),
                "need_resume": need_resume,
                "episode_step_limit": self.max_step_per_episode,
                "step_limit_reached": truncated and not terminated,
            }
            for agent in self.possible_agents
        }
        if self.render_mode == "human" or self.render_every_step:
            self.render()
        self.step_log()
        self.record_sample()
        self.step_num += 1
        observations = self._parallel_observations_after_step(terminations, truncations)
        if self.agents != self.game.agents:
            raise EnvironmentException(
                detail=(
                    "threat game/environment active-agent lists diverged: "
                    f"env={self.agents}, game={self.game.agents}"
                )
            )
        return observations, self.rewards, terminations, truncations, infos

    def send_actions(self, actions):
        missing = set(self.agents) - set(actions)
        if missing:
            raise EnvironmentException(
                detail=f"missing cooperative-search threat actions: {sorted(missing)}"
            )
        commands = {
            agent: self._convert_action(action)
            for agent, action in actions.items()
            if agent in self.agents
        }
        _, _, terminations, truncations, _ = self.game.step(commands)
        self._last_terminated = bool(terminations) and all(terminations.values())
        self._last_truncated = bool(truncations) and all(truncations.values())
        return actions.copy()

    @staticmethod
    def _convert_action(action: Action | CompositeAction) -> FlightCommand:
        heading = action["heading"] if isinstance(action, CompositeAction) else action
        if heading.action_key != "heading":
            raise EnvironmentException(
                detail="cooperative-search threat training requires heading actions"
            )
        return FlightCommand(heading=int(heading.parameters["value"]))

    def done(self) -> tuple[bool, bool, bool]:
        terminated = self._last_terminated
        truncated = self._last_truncated or (
            not terminated and self.game.step_count >= self.max_step_per_episode
        )
        return terminated, truncated, not (terminated or truncated)

    def get_reward(self, **kwargs) -> dict[AgentID, float]:
        per_threat = self._last_team_reward / len(self.threat_agents)
        return {agent: per_threat for agent in self.possible_agents}

    def _nearest_target_distances(self) -> dict[str, float]:
        return {
            agent: min(
                self.game._distance(position, target)
                for target in self.game._target_positions
            )
            for agent, position in self.game.threat_positions.items()
        }

    def _reward_components(
        self,
        previous_distances: dict[str, float],
        current_distances: dict[str, float],
    ) -> dict[str, float]:
        danger_radius = self.game_config.threat_sensor_radius
        warning_radius = self.game_config.threat_warning_radius
        movement = self.game_config.threat_movement_distance_km
        progress = 0.0
        warning_exposure = 0.0
        intrusions = 0
        for agent in self.threat_agents:
            previous = previous_distances[agent]
            current = current_distances[agent]
            if min(previous, current) < warning_radius:
                progress += float(np.clip((current - previous) / movement, -1.0, 1.0))
            warning_exposure += float(np.clip(
                (warning_radius - current) / (warning_radius - danger_radius),
                0.0,
                1.0,
            ))
            intrusions += int(current <= danger_radius)
        return {
            "clearance_progress": self.clearance_progress_reward_weight * progress,
            "warning_zone": -self.warning_zone_penalty * warning_exposure,
            "target_intrusion": -self.target_intrusion_penalty * intrusions,
            "threat_separation": self._threat_separation_reward(),
            "center_distance": self._center_distance_reward(),
        }

    def _threat_separation_reward(self) -> float:
        """过近惩罚按威胁对平均，二次增长且总量受配置权重限制。"""
        positions = [
            self.game.threat_positions[agent]
            for agent in self.threat_agents
        ]
        exposures = [
            max(
                0.0,
                1.0 - self.game._distance(first, second)
                / self.threat_separation_distance_km,
            ) ** 2
            for index, first in enumerate(positions)
            for second in positions[index + 1:]
        ]
        if not exposures:
            return 0.0
        return -self.threat_separation_penalty * float(np.mean(exposures))

    def _center_distance_reward(self) -> float:
        """离地图中心越远惩罚越大，在地图角点达到配置的最大惩罚。"""
        center = (
            self.game_config.map_width / 2.0,
            self.game_config.map_height / 2.0,
        )
        max_distance = math.hypot(center[0], center[1])
        exposures = [
            min(
                1.0,
                self.game._distance(position, center) / max_distance,
            ) ** 2
            for position in self.game.threat_positions.values()
        ]
        if not exposures:
            return 0.0
        return -self.center_distance_penalty * float(np.mean(exposures))

    def _minimum_threat_separation(self) -> float:
        positions = list(self.game.threat_positions.values())
        distances = [
            self.game._distance(first, second)
            for index, first in enumerate(positions)
            for second in positions[index + 1:]
        ]
        return min(distances, default=0.0)

    def _info(self, agent: str) -> dict[str, Any]:
        return {
            "nearest_target_distance_km": self._nearest_target_distances()[agent],
            "minimum_threat_separation_km": self._minimum_threat_separation(),
            "team_reward": self._last_team_reward,
            "reward_breakdown": self._last_reward_breakdown.copy(),
            "episode_reward_breakdown": self._episode_reward_breakdown.copy(),
            "episode_team_reward": self._episode_team_reward,
        }

    def _snapshot(self) -> CooperativeSearchThreatState:
        state = CooperativeSearchThreatState(
            entities=[],
            sim_time=float(self.game.step_count * self.game_config.step_interval_secs),
            threat_positions={
                agent: list(position)
                for agent, position in self.game.threat_positions.items()
            },
            headings=dict(self.game.headings),
            speeds_mps=dict(self.game.speeds_mps),
            frame_index=self.game.step_count,
            total_reward=self._episode_team_reward,
        )
        state._raw_data = state.to_dict()
        return state

    def render(self):
        return self.game.render()

    def close(self) -> None:
        self.game.close()
        super().close()

    def step_log(self) -> None:
        if self.game.step_count % 50 != 0 and not (
            self._last_terminated or self._last_truncated
        ):
            return
        distances = self._nearest_target_distances()
        LOG.info(
            f"[Ep {self.episode_num} - Step {self.game.step_count}] "
            f"nearest_target_km={min(distances.values()):.2f}, "
            f"min_threat_separation_km={self._minimum_threat_separation():.2f}, "
            f"reward={self._last_team_reward:.3f}, "
            f"step_components={self._last_reward_breakdown}, "
            f"episode_components={self._episode_reward_breakdown}, "
            f"episode_reward={self._episode_team_reward:.3f}"
        )

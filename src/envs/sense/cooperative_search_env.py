"""QMIX adapter for the shared-belief cooperative-search mock.

本环境被基准训练测试 ``tests/test_baseline_training.py`` 引用。该测试从
SerialStageTrainRole 起步，在用例内自行初始化配置类。修改本文件或其依赖的
``envs.games.cooperative_search`` 动力学/奖励后，需要重新录制基准实验数据，
并把新的每局奖励粘回测试中的 ``EXPECTED_REWARDS``：

    DODO_RECORD_BASELINE=1 python tests/test_baseline_training.py --dodo-collect
"""

from __future__ import annotations

import math
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from common.exception import EnvironmentException
from config import EnvConfig, SpatialRepresentationConfig, ThreatRoamingConfig
from envs.sense.threat_roaming_observation import ThreatRoamingObservation
from envs.action.model import Action, CompositeAction
from envs.base import AgentID, EnvResetReturn, EnvStepReturn, LOG, ObsType, ParallelEnvironment
from envs.games.cooperative_search import (
    NUM_HEADINGS,
    CooperativeSearchConfig,
    CooperativeSearchGame,
    FlightCommand,
)
from envs.state.tests.cooperative_search_state import CooperativeSearchState
from envs.state.representation import SpatialGridRepresentation
from utils.task_id import task_id


THREAT_SELF_FEATURES = 4
THREAT_TARGET_FEATURES = 3
THREAT_PEER_FEATURES = 3


def threat_observation_dim(num_targets: int, num_threats: int) -> int:
    """返回固定顺序的威胁策略原始观察维度。"""
    return (
        THREAT_SELF_FEATURES
        + int(num_targets) * THREAT_TARGET_FEATURES
        + max(0, int(num_threats) - 1) * THREAT_PEER_FEATURES
    )


def get_threat_obs(game: CooperativeSearchGame, agent: str) -> np.ndarray:
    """构造威胁策略观察；正式搜索和威胁专训环境共同调用此唯一实现。

    布局为：
    [自身x, 自身y, 航向sin, 航向cos,
     目标0距离, 相对方位sin, 相对方位cos, ...,
     友方威胁0距离, 相对方位sin, 相对方位cos, ...]

    十个目标和其他威胁分别按当前距离从近到远排序。威胁不读取任何
    无人机位置、方位或探测状态。
    """
    if agent not in game.threat_positions:
        raise EnvironmentException(detail=f"unknown dynamic threat in observation: {agent}")

    position = game.threat_positions[agent]
    heading = game.headings.get(agent)
    if heading is None:
        heading_angle = None
        heading_sin = heading_cos = 0.0
    else:
        heading_angle = math.radians(heading * 360.0 / NUM_HEADINGS)
        heading_sin = math.sin(heading_angle)
        heading_cos = math.cos(heading_angle)

    map_diagonal = math.hypot(game.config.map_width, game.config.map_height)
    target_features: list[tuple[float, float, float]] = []
    for target_x, target_y in game._target_positions:
        delta_x = target_x - position[0]
        delta_y = target_y - position[1]
        distance = math.hypot(delta_x, delta_y)
        absolute_bearing = math.atan2(delta_y, delta_x)
        relative_bearing = (
            absolute_bearing
            if heading_angle is None
            else absolute_bearing - heading_angle
        )
        target_features.append((
            distance / map_diagonal,
            math.sin(relative_bearing),
            math.cos(relative_bearing),
        ))
    target_features.sort(key=lambda item: item[0])

    peer_features: list[tuple[float, float, float]] = []
    for peer_agent, (peer_x, peer_y) in game.threat_positions.items():
        if peer_agent == agent:
            continue
        delta_x = peer_x - position[0]
        delta_y = peer_y - position[1]
        distance = math.hypot(delta_x, delta_y)
        absolute_bearing = math.atan2(delta_y, delta_x)
        relative_bearing = (
            absolute_bearing
            if heading_angle is None
            else absolute_bearing - heading_angle
        )
        peer_features.append((
            distance / map_diagonal,
            math.sin(relative_bearing),
            math.cos(relative_bearing),
        ))
    peer_features.sort(key=lambda item: item[0])

    observation = np.asarray([
        position[0] / game.config.map_width,
        position[1] / game.config.map_height,
        heading_sin,
        heading_cos,
        *(value for features in target_features for value in features),
        *(value for features in peer_features for value in features),
    ], dtype=np.float32)
    expected = threat_observation_dim(
        game.config.num_targets,
        game.config.num_dynamic_threats,
    )
    if observation.size != expected:
        raise EnvironmentException(
            detail=(
                f"threat observation size mismatch for {agent}: "
                f"actual={observation.size}, expected={expected}"
            )
        )
    return observation


class CooperativeSearchEnv(ParallelEnvironment):
    """Four-UAV Dec-POMDP backed by one shared probability database."""

    NUM_ACTIONS = NUM_HEADINGS
    _REWARD_KEYS = (
        "correct_detection",
        "false_detection",
        "exploration",
        "frontier_potential",
        "death",
        "revisited_empty",
    )

    def __init__(self, config: EnvConfig, sim_config=None, connect_sim=True,
                 sim_manager=None, role_id=None) -> None:
        super().__init__(config=config, sim_config=sim_config, connect_sim=False,
                         sim_manager=sim_manager, role_id=role_id)
        metadata = config.metadata or {}
        raw = dict(metadata.get("game_config", {}))
        raw.setdefault("max_steps", config.max_step_per_episode)
        raw.setdefault("step_interval_secs", config.step_interval_secs)
        self.game_config = self._build_game_config(raw)
        self.drone_agents = [f"drone_{index}" for index in range(self.game_config.num_drones)]
        self.threat_agents = [
            f"dynamic_threat_{index}"
            for index in range(self.game_config.num_dynamic_threats)
        ]
        expected = self.drone_agents + self.threat_agents
        if self.possible_agents != expected:
            raise EnvironmentException(
                detail=(
                    "agent_names must list consecutive drones followed by dynamic threats: "
                    f"actual={self.possible_agents}, expected={expected}"
                )
            )
        self.fov_size = int(metadata.get("fov_size", 11))
        if self.fov_size <= 0 or self.fov_size % 2 != 1:
            raise EnvironmentException(detail=f"fov_size must be a positive odd number: {self.fov_size}")
        self.observation_grid_scale = int(metadata.get("observation_grid_scale", 2))
        if (
            self.observation_grid_scale <= 0
            or self.game_config.belief_grid_width % self.observation_grid_scale != 0
            or self.game_config.belief_grid_height % self.observation_grid_scale != 0
        ):
            raise EnvironmentException(
                detail=(
                    "observation_grid_scale must be positive and evenly divide the belief grid: "
                    f"scale={self.observation_grid_scale}, "
                    f"belief=({self.game_config.belief_grid_width}, "
                    f"{self.game_config.belief_grid_height})"
                )
            )
        self.state_grid_width = int(metadata.get("state_grid_width", 50))
        self.state_grid_height = int(metadata.get("state_grid_height", 50))
        if not (
            0 < self.state_grid_width <= self.game_config.belief_grid_width
            and 0 < self.state_grid_height <= self.game_config.belief_grid_height
        ):
            raise EnvironmentException(
                detail=(
                    "state grid must be positive and no larger than belief grid: "
                    f"state=({self.state_grid_width}, {self.state_grid_height}), "
                    f"belief=({self.game_config.belief_grid_width}, "
                    f"{self.game_config.belief_grid_height})"
                )
            )
        self.render_mode = metadata.get("render_mode")
        if self.render_mode not in (None, "human", "rgb_array"):
            raise EnvironmentException(detail=f"unsupported render_mode: {self.render_mode}")
        self.render_every_step = bool(metadata.get("render_every_step", False))
        self._seed = metadata.get("seed")
        self.exploration_reward_per_cell = float(
            metadata.get("exploration_reward_per_cell", 0.00082)
        )
        self.revisited_empty_penalty_episode_limit = float(
            metadata.get("revisited_empty_penalty_episode_limit", 15.0)
        )
        self.frontier_potential_reward_weight = float(
            metadata.get("frontier_potential_reward_weight", 0.03)
        )
        self.correct_detection_reward_total = float(
            metadata.get("correct_detection_reward_total", 20.0)
        )
        self.correct_detection_reward_half_life_ratio = float(
            metadata.get("correct_detection_reward_half_life_ratio", 0.8)
        )
        self._validate_correct_detection_reward_curve()
        self._correct_detection_reward_growth_base = (
            self._solve_correct_detection_reward_growth_base()
        )
        self.false_detection_penalty = float(metadata.get("false_detection_penalty", 0.3))
        self.death_penalty_per_drone = float(metadata.get("death_penalty_per_drone", 6.0))
        self.dynamic_threats_enabled = bool(
            metadata.get("dynamic_threats_enabled", True)
        )
        self.save_probability_map_on_episode_end = bool(
            metadata.get("save_probability_map_on_episode_end", False)
        )
        self.probability_map_every_n_episodes = max(
            1,
            int(metadata.get("probability_map_every_n_episodes", 100)),
        )
        self.probability_map_output_dir = Path(
            metadata.get("probability_map_output_dir", "artifacts/results/cooperative_search")
        )
        if not (
            self.exploration_reward_per_cell >= 0
            and self.revisited_empty_penalty_episode_limit >= 0
            and self.frontier_potential_reward_weight >= 0
            and self.false_detection_penalty >= 0
            and self.death_penalty_per_drone >= 0
        ):
            raise EnvironmentException(detail="invalid cooperative-search reward metadata")
        self.game = CooperativeSearchGame(self.game_config, render_mode=self.render_mode)
        self.game.set_dynamic_threats_enabled(self.dynamic_threats_enabled)
        representation_dim = metadata.get("situation_representation_output_dim")
        self.situation_representation = None
        if representation_dim is not None:
            try:
                self.situation_representation = SpatialGridRepresentation(
                    SpatialRepresentationConfig(
                        input_shape=(
                            3,
                            self.game_config.belief_grid_height,
                            self.game_config.belief_grid_width,
                        ),
                        output_dim=representation_dim,
                    )
                )
            except ValueError as exc:
                raise EnvironmentException(detail=f"invalid situation representation: {exc}") from exc
        # 四个局部栅格通道依次为：共享概率、访问次数、友机位置、动态威胁位置。
        self._obs_dim = self.fov_size * self.fov_size * 4 + 3
        if self.situation_representation is not None:
            self._obs_dim += self.situation_representation.config.output_dim
        self.threat_roaming_observation = (
            ThreatRoamingObservation(self.game_config.threat_roaming)
            if self.game_config.threat_roaming is not None else None
        )
        self._threat_obs_dim = (
            self.threat_roaming_observation.observation_dim
            if self.threat_roaming_observation is not None else threat_observation_dim(
                self.game_config.num_targets, self.game_config.num_dynamic_threats,
            )
        )
        state_map_dim = (
            self.situation_representation.config.output_dim
            if self.situation_representation is not None
            else self.state_grid_width * self.state_grid_height * 2
        )
        self._state_dim = state_map_dim + (
            self.game_config.num_drones + self.game_config.num_dynamic_threats
        ) * 4
        if config.state_dim not in (0, self._state_dim):
            raise EnvironmentException(
                detail=f"state_dim mismatch: config={config.state_dim}, expected={self._state_dim}"
            )
        self._last_terminated = self._last_truncated = False
        self._episode_team_reward = self._last_team_reward = 0.0
        self._last_reward_breakdown = self._empty_reward_breakdown()
        self._episode_reward_breakdown = self._empty_reward_breakdown()
        self._last_revisited_empty_cell_count = 0
        self._last_revisited_empty_coverage_ratio = 0.0
        self._last_game_infos: dict[str, dict] = {}
        self._last_game_terminations: dict[str, bool] = {}
        self._last_game_truncations: dict[str, bool] = {}
        self._rewarded_target_indices: set[int] = set()
        self._penalized_false_detection_cells: set[tuple[int, int]] = set()
        self.pre_state = self.post_state = self._snapshot()

    def _on_env_metadata_changed(self) -> None:
        metadata = self.config.metadata or {}
        representation_dim = metadata.get("situation_representation_output_dim")
        current_dim = (
            self.situation_representation.config.output_dim
            if self.situation_representation is not None else None
        )
        if representation_dim != current_dim:
            raise EnvironmentException(
                detail="situation_representation_output_dim cannot change after environment construction"
            )
        render_mode = metadata.get("render_mode")
        if render_mode not in (None, "human", "rgb_array"):
            raise EnvironmentException(
                detail=f"unsupported cooperative-search render_mode: {render_mode}"
            )
        self.render_mode = render_mode
        self.render_every_step = bool(metadata.get("render_every_step", False))
        self.exploration_reward_per_cell = float(
            metadata.get("exploration_reward_per_cell", self.exploration_reward_per_cell)
        )
        self.revisited_empty_penalty_episode_limit = float(
            metadata.get(
                "revisited_empty_penalty_episode_limit",
                self.revisited_empty_penalty_episode_limit,
            )
        )
        self.frontier_potential_reward_weight = float(
            metadata.get(
                "frontier_potential_reward_weight",
                self.frontier_potential_reward_weight,
            )
        )
        self.correct_detection_reward_total = float(
            metadata.get(
                "correct_detection_reward_total",
                self.correct_detection_reward_total,
            )
        )
        self.correct_detection_reward_half_life_ratio = float(
            metadata.get(
                "correct_detection_reward_half_life_ratio",
                self.correct_detection_reward_half_life_ratio,
            )
        )
        self._validate_correct_detection_reward_curve()
        self._correct_detection_reward_growth_base = (
            self._solve_correct_detection_reward_growth_base()
        )
        self.false_detection_penalty = float(
            metadata.get("false_detection_penalty", self.false_detection_penalty)
        )
        self.death_penalty_per_drone = float(
            metadata.get("death_penalty_per_drone", self.death_penalty_per_drone)
        )
        self.dynamic_threats_enabled = bool(
            metadata.get("dynamic_threats_enabled", self.dynamic_threats_enabled)
        )
        self.save_probability_map_on_episode_end = bool(
            metadata.get(
                "save_probability_map_on_episode_end",
                self.save_probability_map_on_episode_end,
            )
        )
        self.probability_map_every_n_episodes = max(
            1,
            int(
                metadata.get(
                    "probability_map_every_n_episodes",
                    self.probability_map_every_n_episodes,
                )
            ),
        )
        self.probability_map_output_dir = Path(
            metadata.get(
                "probability_map_output_dir",
                self.probability_map_output_dir,
            )
        )
        self.game.render_mode = render_mode
        self.game.set_dynamic_threats_enabled(self.dynamic_threats_enabled)

    @staticmethod
    def _build_game_config(raw: dict[str, Any]) -> CooperativeSearchConfig:
        valid = {item.name for item in fields(CooperativeSearchConfig)}
        unknown = set(raw) - valid
        if unknown:
            raise EnvironmentException(
                detail=f"unsupported cooperative-search game config: {sorted(unknown)}"
            )
        raw = dict(raw)
        if isinstance(raw.get("threat_roaming"), dict):
            raw["threat_roaming"] = ThreatRoamingConfig(**raw["threat_roaming"])
        return CooperativeSearchConfig(**raw)

    def _empty_reward_breakdown(self) -> dict[str, float]:
        return {key: 0.0 for key in self._REWARD_KEYS}

    @property
    def observation_dim(self) -> int:
        return self._obs_dim

    @property
    def centralized_state_dim(self) -> int:
        return self._state_dim

    @property
    def sensor_footprint_cell_upper_bound(self) -> int:
        """Strict per-drone cell-count bound for one axis-aligned sensor footprint."""
        cell_width = (
            self.game_config.map_width / self.game_config.belief_grid_width
        )
        cell_height = (
            self.game_config.map_height / self.game_config.belief_grid_height
        )
        columns = math.floor(
            2.0 * self.game_config.sensor_radius / cell_width
        ) + 1
        rows = math.floor(
            2.0 * self.game_config.sensor_radius / cell_height
        ) + 1
        return min(
            self.game.shared_belief.size,
            max(1, columns) * max(1, rows),
        )

    def state_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(self._state_dim,), dtype=np.float32)

    def state_for_agent_ids(self, *, agent_name, agent_ids) -> np.ndarray:
        if (agent_ids and self.threat_roaming_observation is not None
                and all(agent in self.threat_agents for agent in agent_ids)):
            if len(agent_ids) != 1:
                raise EnvironmentException(detail="roaming threat inference requires singleton groups")
            return (self.threat_roaming_observation(self.game, agent_ids[0])
                    if self.dynamic_threats_enabled
                    else np.zeros(self._threat_obs_dim, dtype=np.float32))
        return self.state()

    def state(self) -> np.ndarray:
        if self.situation_representation is not None:
            values: list[np.ndarray] = [self._region_representation(self._normalized_visits())]
        else:
            values = [
                self._pool_grid(self.game.shared_belief, reducer="mean").ravel(),
                self._pool_grid(self._normalized_visits(), reducer="mean").ravel(),
            ]
        flight_state: list[float] = []
        for agent in self.possible_agents:
            if (
                agent in self.game.dead_drone_positions
                or (agent in self.threat_agents and not self.dynamic_threats_enabled)
            ):
                flight_state.extend([0.0, 0.0, 0.0, 0.0])
                continue
            x, y = self.game._agent_position(agent)
            heading = self.game.headings[agent]
            angle = 0.0 if heading is None else math.radians(heading * 360.0 / NUM_HEADINGS)
            flight_state.extend([
                x / self.game_config.map_width,
                y / self.game_config.map_height,
                math.sin(angle),
                math.cos(angle),
            ])
        values.append(np.asarray(flight_state, dtype=np.float32))
        state = np.concatenate(values).astype(np.float32)
        if state.size != self._state_dim:
            raise EnvironmentException(
                detail=f"centralized state size mismatch: actual={state.size}, expected={self._state_dim}"
            )
        return state

    def _region_representation(self, visits: np.ndarray) -> np.ndarray:
        """从当前可见数值网格编码态势；不读取渲染图或隐藏目标真值。"""
        source = np.stack((
            self._observed_mask().astype(np.float32),
            self.game.shared_belief,
            visits,
        ))
        try:
            return self.situation_representation(source)
        except ValueError as exc:
            raise EnvironmentException(detail=f"invalid situation representation: {exc}") from exc

    def get_obs(self) -> dict[AgentID, ObsType]:
        visits = self._normalized_visits()
        region_feature = None
        if self.situation_representation is not None:
            region_feature = self._region_representation(visits)
        observations = {}
        for agent in self.drone_agents:
            position = self.game.drone_positions.get(agent)
            if position is None:
                observations[agent] = np.zeros(self._obs_dim, dtype=np.float32)
                continue
            gx, gy = self.game._world_to_grid(position)
            features = [
                self._local_observation_crop(
                    self.game.shared_belief, gx, gy, outside=1.0, reducer="mean"
                ).ravel(),
                self._local_observation_crop(
                    visits, gx, gy, outside=1.0, reducer="mean"
                ).ravel(),
                self._local_friendly_map(agent, gx, gy).ravel(),
                self._local_threat_map(position, gx, gy).ravel(),
                np.asarray([
                    position[0] / self.game_config.map_width,
                    position[1] / self.game_config.map_height,
                    float(self.game.detected_flags[agent]),
                ], dtype=np.float32),
            ]
            if region_feature is not None:
                features.append(region_feature)
            observation = np.concatenate(features).astype(np.float32)
            if observation.size != self._obs_dim:
                raise EnvironmentException(
                    detail=(
                        f"observation size mismatch for {agent}: "
                        f"actual={observation.size}, expected={self._obs_dim}"
                    )
                )
            observations[agent] = observation
        for agent, position in self.game.threat_positions.items():
            observations[agent] = (
                (self.threat_roaming_observation(self.game, agent)
                 if self.threat_roaming_observation is not None else get_threat_obs(self.game, agent))
                if self.dynamic_threats_enabled
                else np.zeros(self._threat_obs_dim, dtype=np.float32)
            )
        return observations

    def get_avail_agent_actions(self, agent_name: AgentID, init: bool = False) -> list[int]:
        if agent_name not in self.possible_agents:
            raise EnvironmentException(detail=f"unknown agent in action mask: {agent_name}")
        if agent_name in self.game.dead_drone_positions:
            return [0] * self.NUM_ACTIONS
        available = set(self.game.available_headings(agent_name))
        mask = [int(heading in available) for heading in range(NUM_HEADINGS)]
        if len(mask) != self.NUM_ACTIONS or not any(mask):
            raise EnvironmentException(
                detail=(
                    f"invalid action mask for {agent_name}: "
                    f"length={len(mask)}, available={sum(mask)}, mask={mask}"
                )
            )
        return mask

    def reset(self, seed=None, options=None) -> EnvResetReturn:
        self._reset_common(options)
        self.game.reset(seed=seed if seed is not None else self._seed)
        self._last_terminated = self._last_truncated = False
        self._episode_team_reward = self._last_team_reward = 0.0
        self._last_reward_breakdown = self._empty_reward_breakdown()
        self._episode_reward_breakdown = self._empty_reward_breakdown()
        self._last_revisited_empty_cell_count = 0
        self._last_revisited_empty_coverage_ratio = 0.0
        self._last_game_infos = {}
        self._last_game_terminations = {}
        self._last_game_truncations = {}
        self._rewarded_target_indices.clear()
        self._penalized_false_detection_cells.clear()
        self.pre_state = self.post_state = self._snapshot()
        observations = self.get_obs()
        infos = {agent: self._info(agent) for agent in self.possible_agents}
        if self.render_mode == "human":
            self.render()
        self.record_sample(reset=True)
        return observations, infos

    def step(self, actions: dict[AgentID, Action | CompositeAction]) -> EnvStepReturn:
        self.pre_state = self.post_state
        previous_observed = self._observed_mask()
        previous_drone_positions = self.game.drone_positions.copy()
        previous_alive_drones = len(self.game.drone_positions)
        previous_correct_count = len(self._rewarded_target_indices)
        self.current_actions = self.send_actions(actions)
        newly_dead_drones = previous_alive_drones - len(self.game.drone_positions)
        correct_detections, false_detections = self._new_detection_events()
        current_correct_count = previous_correct_count + correct_detections
        frontier_progress = self._frontier_progress(
            previous_observed=previous_observed,
            previous_drone_positions=previous_drone_positions,
        )
        newly_explored_cells, revisited_ratio = self._sequential_search_settlement(
            previous_observed=previous_observed,
        )
        self._last_revisited_empty_coverage_ratio = revisited_ratio
        self._last_revisited_empty_cell_count = int(
            round(revisited_ratio * self.game_config.num_drones * self.sensor_footprint_cell_upper_bound)
        )
        self._last_reward_breakdown = self._reward_components(
            previous_correct_count=previous_correct_count,
            current_correct_count=current_correct_count,
            false_detections=false_detections,
            newly_explored_cells=newly_explored_cells,
            frontier_progress=frontier_progress,
            newly_dead_drones=newly_dead_drones,
            revisited_empty_coverage_ratio=self._last_revisited_empty_coverage_ratio,
        )
        self._last_team_reward = float(sum(self._last_reward_breakdown.values()))
        self._episode_team_reward += self._last_team_reward
        for key, value in self._last_reward_breakdown.items():
            self._episode_reward_breakdown[key] += value
        self.game.total_reward = self._episode_team_reward
        self.post_state = self._snapshot()
        terminated, truncated, need_resume = self.done()
        self.rewards = self.get_reward()
        terminations = {
            agent: bool(self._last_game_terminations.get(agent, terminated))
            for agent in self.possible_agents
        }
        truncations = {
            agent: bool(self._last_game_truncations.get(agent, truncated))
            for agent in self.possible_agents
        }
        infos = {
            agent: {
                **self._info(agent),
                "need_resume": need_resume,
                "episode_step_limit": self.max_step_per_episode,
                "step_limit_reached": truncated and not terminated,
            }
            for agent in self.possible_agents
        }
        if (
            (terminated or truncated)
            and self.save_probability_map_on_episode_end
            and self.episode_num > 0
            and self.episode_num % self.probability_map_every_n_episodes == 0
        ):
            self._save_probability_map()
        if self.render_mode == "human" or self.render_every_step:
            self.render()
        self.step_log()
        self.record_sample()
        self.step_num += 1
        observations = self._parallel_observations_after_step(terminations, truncations)
        if self.agents != self.game.agents:
            raise EnvironmentException(
                detail=(
                    "game/environment active-agent lists diverged: "
                    f"env={self.agents}, game={self.game.agents}"
                )
            )
        return observations, self.rewards, terminations, truncations, infos

    def send_actions(self, actions):
        missing = set(self.agents) - set(actions)
        if missing:
            raise EnvironmentException(
                detail=f"missing cooperative-search actions: {sorted(missing)}"
            )
        commands = {
            agent: self._convert_action(action)
            for agent, action in actions.items()
            if agent in self.agents
        }
        _, _, terminations, truncations, infos = self.game.step(commands)
        self._last_game_terminations = terminations
        self._last_game_truncations = truncations
        self._last_terminated = self.game.all_targets_fully_identified()
        self._last_truncated = bool(truncations) and all(truncations.values())
        self._last_game_infos = infos
        return actions.copy()

    @staticmethod
    def _convert_action(action: Action | CompositeAction) -> FlightCommand:
        heading = action["heading"] if isinstance(action, CompositeAction) else action
        if heading.action_key != "heading":
            raise EnvironmentException(detail="cooperative_search requires heading actions")
        return FlightCommand(heading=int(heading.parameters["value"]))

    def done(self) -> tuple[bool, bool, bool]:
        terminated = self._last_terminated
        truncated = self._last_truncated or (not terminated and self.game.step_count >= self.max_step_per_episode)
        return terminated, truncated, not (terminated or truncated)

    def get_reward(self, **kwargs) -> dict[AgentID, float]:
        per_drone = self._last_team_reward / len(self.drone_agents)
        return {
            agent: per_drone if agent in self.drone_agents else 0.0
            for agent in self.possible_agents
        }

    def get_episode_rewards_details(self) -> dict[str, float]:
        """Return the accumulated reward components for the current episode.

        The sampling worker calls this hook once after an episode ends. Returning
        a copy keeps task-level reward-details history independent from the
        mutable per-episode accumulator that reset() clears.
        """
        return {
            component_name: float(value)
            for component_name, value in self._episode_reward_breakdown.items()
        }

    def _reward_components(
        self, *, previous_correct_count: int = 0, current_correct_count: int = 0,
        false_detections: int = 0, newly_explored_cells: int = 0,
        frontier_progress: float = 0.0, newly_dead_drones: int = 0,
        revisited_empty_coverage_ratio: float = 0.0,
    ) -> dict[str, float]:
        return {
            "correct_detection": (
                self._correct_detection_cumulative_reward(current_correct_count)
                - self._correct_detection_cumulative_reward(previous_correct_count)
            ),
            "false_detection": -self.false_detection_penalty * false_detections,
            "exploration": self.exploration_reward_per_cell * newly_explored_cells,
            "frontier_potential": (
                self.frontier_potential_reward_weight
                * float(np.clip(frontier_progress, -1.0, 1.0))
            ),
            "death": -self.death_penalty_per_drone * newly_dead_drones,
            "revisited_empty": -(
                self.revisited_empty_penalty_episode_limit
                / self.max_step_per_episode
                * float(np.clip(revisited_empty_coverage_ratio, 0.0, 1.0))
            ),
        }

    def _sequential_search_settlement(
        self, *, previous_observed: np.ndarray
    ) -> tuple[int, float]:
        """按稳定无人机顺序结算同一步内的搜索足迹。

        先覆盖某格的无人机获得首次探索奖励；同一步内后续无人机再覆盖该格时，
        若该格已是确认空白，则计入重复空白搜索惩罚。不引入新的奖励分量。
        """
        if not self.game.drone_positions:
            return 0, 0.0
        observed = previous_observed.astype(bool, copy=True)
        confirmed_empty = (
            self.game.shared_belief
            <= self.game_config.min_belief_probability + 1e-6
        )
        newly_covered = 0
        repeated = 0
        for agent in self.drone_agents:
            position = self.game.drone_positions.get(agent)
            if position is None:
                continue
            footprint = self.game._sensor_grid_mask(position)
            newly_covered += int(np.count_nonzero(footprint & ~observed))
            repeated += int(
                np.count_nonzero(footprint & observed & confirmed_empty)
            )
            observed |= footprint
        upper = self.game_config.num_drones * self.sensor_footprint_cell_upper_bound
        revisited_ratio = float(np.clip(repeated / max(1, upper), 0.0, 1.0))
        return newly_covered, revisited_ratio

    def _revisited_empty_coverage(
        self, *, previous_observed: np.ndarray
    ) -> tuple[int, float]:
        """Return confirmed-empty cell-agent count and its normalized upper-bound ratio.

        Kept for unit tests and diagnostics. Training steps use
        ``_sequential_search_settlement`` so same-step overlaps settle in UAV order.
        """
        confirmed_empty = (
            previous_observed
            & (
                self.game.shared_belief
                <= self.game_config.min_belief_probability + 1e-6
            )
        )
        empty_cell_count = 0
        for position in self.game.drone_positions.values():
            footprint = self.game._sensor_grid_mask(position)
            empty_cell_count += int(
                np.count_nonzero(confirmed_empty & footprint)
            )
        theoretical_cell_count = (
            self.game_config.num_drones
            * self.sensor_footprint_cell_upper_bound
        )
        normalized_ratio = empty_cell_count / theoretical_cell_count
        return empty_cell_count, float(np.clip(normalized_ratio, 0.0, 1.0))

    def _frontier_progress(
        self, *, previous_observed: np.ndarray,
        previous_drone_positions: dict[str, tuple[float, float]],
    ) -> float:
        """Return mean normalized progress toward the nearest previously-unexplored cell."""
        unobserved = ~previous_observed
        if not np.any(unobserved) or not previous_drone_positions:
            return 0.0

        frontier_x = self.game._grid_world_x[unobserved]
        frontier_y = self.game._grid_world_y[unobserved]
        movement_distance = self.game_config.movement_distance_km
        normalized_progress: list[float] = []
        for agent, previous_position in previous_drone_positions.items():
            current_position = self.game.drone_positions.get(agent)
            if current_position is None:
                # Death already has its own penalty and must not alter the shaping scale.
                normalized_progress.append(0.0)
                continue
            previous_distance = float(np.min(np.hypot(
                frontier_x - previous_position[0],
                frontier_y - previous_position[1],
            )))
            current_distance = float(np.min(np.hypot(
                frontier_x - current_position[0],
                frontier_y - current_position[1],
            )))
            normalized_progress.append(float(np.clip(
                (previous_distance - current_distance) / movement_distance,
                -1.0,
                1.0,
            )))
        return float(np.mean(normalized_progress))

    def _new_detection_events(self) -> tuple[int, int]:
        """结算本局首次达到概率上限的正确目标和错误标记格。"""
        threshold = self.game_config.max_belief_probability - 1e-6
        target_cells = [self.game._world_to_grid(position) for position in self.game._target_positions]
        correct_detections = 0
        for target_index, (grid_x, grid_y) in enumerate(target_cells):
            if (
                target_index not in self._rewarded_target_indices
                and self.game.shared_belief[grid_y, grid_x] >= threshold
            ):
                self._rewarded_target_indices.add(target_index)
                correct_detections += 1

        true_cells = set(target_cells)
        marked_cells = {
            (int(grid_x), int(grid_y))
            for grid_y, grid_x in np.argwhere(self.game.shared_belief >= threshold)
        }
        new_false_cells = (
            marked_cells - true_cells - self._penalized_false_detection_cells
        )
        self._penalized_false_detection_cells.update(new_false_cells)
        return correct_detections, len(new_false_cells)

    def _validate_correct_detection_reward_curve(self) -> None:
        if self.correct_detection_reward_total <= 0:
            raise EnvironmentException(
                detail="correct_detection_reward_total must be positive"
            )
        if not (0.5 < self.correct_detection_reward_half_life_ratio < 1.0):
            raise EnvironmentException(
                detail=(
                    "correct_detection_reward_half_life_ratio must be in "
                    "the open interval (0.5, 1.0)"
                )
            )
        if self.game_config.num_targets < 1:
            raise EnvironmentException(
                detail="correct-detection reward curve requires at least one target"
            )

    def _solve_correct_detection_reward_growth_base(self) -> float:
        """Find the base that halves the score at the configured target ratio."""
        target_count = self.game_config.num_targets
        half_life_count = target_count * self.correct_detection_reward_half_life_ratio

        def cumulative_ratio(base: float) -> float:
            log_base = math.log(base)
            return math.expm1(half_life_count * log_base) / math.expm1(
                target_count * log_base
            )

        lower = 1.0 + 1e-9
        upper = 2.0
        while cumulative_ratio(upper) > 0.5:
            upper *= 2.0
        for _ in range(80):
            middle = (lower + upper) / 2.0
            if cumulative_ratio(middle) > 0.5:
                lower = middle
            else:
                upper = middle
        return (lower + upper) / 2.0

    def _correct_detection_cumulative_reward(self, count: int) -> float:
        """Return total target reward after confirming ``count`` targets."""
        target_count = self.game_config.num_targets
        count = int(np.clip(count, 0, target_count))
        if count == 0:
            return 0.0
        if count == target_count:
            return self.correct_detection_reward_total
        log_base = math.log(self._correct_detection_reward_growth_base)
        return self.correct_detection_reward_total * (
            math.expm1(count * log_base) / math.expm1(target_count * log_base)
        )

    def correct_detection_reward_schedule(self) -> tuple[float, ...]:
        """Return the marginal rewards for targets 1 through N."""
        cumulative = [
            self._correct_detection_cumulative_reward(count)
            for count in range(self.game_config.num_targets + 1)
        ]
        return tuple(
            cumulative[count] - cumulative[count - 1]
            for count in range(1, len(cumulative))
        )

    def _normalized_visits(self) -> np.ndarray:
        denominator = math.log1p(max(1, self.max_step_per_episode))
        return np.clip(np.log1p(self.game.visited_counts) / denominator, 0, 1).astype(np.float32)

    def _pool_grid(self, values: np.ndarray, *, reducer: str) -> np.ndarray:
        """Pool the shared database to the fixed-size network/public-state view."""
        source_height, source_width = values.shape
        if (
            source_height % self.state_grid_height != 0
            or source_width % self.state_grid_width != 0
        ):
            raise EnvironmentException(
                detail=(
                    "belief grid must be evenly divisible by state grid: "
                    f"belief=({source_width}, {source_height}), "
                    f"state=({self.state_grid_width}, {self.state_grid_height})"
                )
            )
        height_factor = source_height // self.state_grid_height
        width_factor = source_width // self.state_grid_width
        blocks = values.reshape(
            self.state_grid_height, height_factor,
            self.state_grid_width, width_factor,
        )
        if reducer == "mean":
            return blocks.mean(axis=(1, 3), dtype=np.float32)
        if reducer == "sum":
            return blocks.sum(axis=(1, 3))
        raise EnvironmentException(detail=f"unsupported grid pooling reducer: {reducer}")

    def _local_friendly_map(
        self, observing_agent: str, center_x: int, center_y: int
    ) -> np.ndarray:
        result = np.zeros(self.game.grid_shape, dtype=np.float32)
        for agent, position in self.game.drone_positions.items():
            if agent != observing_agent:
                x, y = self.game._world_to_grid(position)
                result[y, x] = 1.0
        return self._local_observation_crop(
            result, center_x, center_y, outside=0.0, reducer="max"
        )

    def _local_threat_map(
        self,
        observer_position: tuple[float, float],
        center_x: int,
        center_y: int,
    ) -> np.ndarray:
        """返回45公里预警范围内威胁形成的连续危险势场。"""
        result = np.zeros(self.game.grid_shape, dtype=np.float32)
        if not self.dynamic_threats_enabled:
            return self._local_observation_crop(
                result, center_x, center_y, outside=0.0, reducer="max"
            )
        warning_radius = self.game_config.threat_warning_radius
        danger_radius = self.game_config.threat_sensor_radius
        transition_width = warning_radius - danger_radius
        for position in self.game.threat_positions.values():
            if self.game._distance(observer_position, position) > warning_radius:
                continue
            distances = np.hypot(
                self.game._grid_world_x - position[0],
                self.game._grid_world_y - position[1],
            )
            threat_field = np.clip(
                (warning_radius - distances) / transition_width,
                0.0,
                1.0,
            ).astype(np.float32)
            np.maximum(result, threat_field, out=result)
        return self._local_observation_crop(
            result, center_x, center_y, outside=0.0, reducer="max"
        )

    def _local_observation_crop(
        self,
        source: np.ndarray,
        center_x: int,
        center_y: int,
        *,
        outside: float,
        reducer: str,
    ) -> np.ndarray:
        """将2×2概率格聚合为一个10公里观察格后裁取固定11×11窗口。"""
        scale = self.observation_grid_scale
        source_height, source_width = source.shape
        blocks = source.reshape(
            source_height // scale,
            scale,
            source_width // scale,
            scale,
        )
        if reducer == "mean":
            pooled = blocks.mean(axis=(1, 3), dtype=np.float32)
        elif reducer == "max":
            pooled = blocks.max(axis=(1, 3))
        else:
            raise EnvironmentException(detail=f"unsupported local-grid reducer: {reducer}")
        return self._fov_crop(
            pooled,
            center_x // scale,
            center_y // scale,
            outside=outside,
        )

    def _fov_crop(
        self, source: np.ndarray, center_x: int, center_y: int, *, outside: float
    ) -> np.ndarray:
        radius = self.fov_size // 2
        crop = np.full((self.fov_size, self.fov_size), outside, dtype=np.float32)
        for local_y, grid_y in enumerate(range(center_y - radius, center_y + radius + 1)):
            for local_x, grid_x in enumerate(range(center_x - radius, center_x + radius + 1)):
                if 0 <= grid_x < source.shape[1] and 0 <= grid_y < source.shape[0]:
                    crop[local_y, local_x] = source[grid_y, grid_x]
        return crop

    def _info(self, agent: str) -> dict[str, Any]:
        return {
            **self._last_game_infos.get(agent, {}),
            "detected_target": bool(self.game.detected_flags.get(agent, False)),
            "detected_drone": bool(self.game.threat_detected_flags.get(agent, False)),
            "detected_by_threat": bool(self.game.drones_detected_by_threat.get(agent, False)),
            "alive": agent not in self.game.dead_drone_positions,
            "eaten_by_threat": bool(
                self._last_game_infos.get(agent, {}).get("eaten_by_threat", False)
            ),
            "team_reward": self._last_team_reward,
            "reward_breakdown": self._last_reward_breakdown.copy(),
            "episode_reward_breakdown": self._episode_reward_breakdown.copy(),
            "episode_team_reward": self._episode_team_reward,
            "coverage": float(np.count_nonzero(self._observed_mask()) / self.game.shared_belief.size),
            "success": bool(
                self.game.all_targets_fully_identified()
                and len(self.game.dead_drone_positions) == 0
            ),
            "target_discovered_count": int(len(self._rewarded_target_indices)),
            "own_loss_count": int(len(self.game.dead_drone_positions)),
            "coverage_rate_percent": (
                float(np.count_nonzero(self._observed_mask()) / self.game.shared_belief.size) * 100.0
                if self.game.shared_belief.size
                else 0.0
            ),
            "false_target_count": int(len(self._penalized_false_detection_cells)),
            "episode_steps": int(max(self.step_num - 1, 0)),
            "revisited_empty_cell_count": self._last_revisited_empty_cell_count,
            "revisited_empty_coverage_ratio": self._last_revisited_empty_coverage_ratio,
            "revisited_empty_penalty_step_limit": (
                self.revisited_empty_penalty_episode_limit
                / self.max_step_per_episode
            ),
            "revisited_empty_cell_step_upper_bound": (
                self.game_config.num_drones
                * self.sensor_footprint_cell_upper_bound
            ),
        }

    def _observed_mask(self) -> np.ndarray:
        return (self.game.positive_counts + self.game.negative_counts) > 0

    def _save_probability_map(self) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise EnvironmentException(
                detail="saving cooperative-search probability maps requires Pillow"
            ) from exc
        safe_task_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(task_id)).strip("_") or "task"
        # 各训练阶段的episode都会从1重新编号，阶段名必须进入文件名，避免第二阶段
        # 的0001/0200/...静默覆盖第一阶段图片。
        safe_stage_name = (
            re.sub(r"[^0-9A-Za-z_.-]+", "_", self.stage_name).strip("_") or "train"
        )
        output_path = self.probability_map_output_dir / (
            f"probability_{safe_task_id}_{safe_stage_name}_episode_{self.episode_num:04d}.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        probability = np.clip(self.game.global_belief(), 0.0, 1.0)[::-1]
        heatmap = Image.fromarray(self.game._belief_rgb(probability))
        if heatmap.size != (500, 500):
            # 概率数据库是离散网格，放大时必须保留每个网格的清晰边界；
            # 双线性插值会制造数据库中不存在的过渡概率并让整张图发糊。
            heatmap = heatmap.resize((500, 500), resample=Image.Resampling.NEAREST)

        image = Image.new("RGB", (500, 540), "white")
        image.paste(heatmap, (0, 0))
        draw = ImageDraw.Draw(image)
        highlighted_cells = np.argwhere(
            probability >= self.game_config.detection_threshold
        )
        cell_width = 500.0 / probability.shape[1]
        cell_height = 500.0 / probability.shape[0]
        marker_radius = max(3.0, min(cell_width, cell_height) * 0.7)
        target_cells = self.game._target_grid_cells()
        for grid_y, grid_x in highlighted_cells:
            center_x = (float(grid_x) + 0.5) * cell_width
            center_y = (float(grid_y) + 0.5) * cell_height
            # probability在绘图前做了上下翻转，分类时还原为游戏原始网格Y坐标。
            original_grid_y = probability.shape[0] - 1 - int(grid_y)
            if (int(grid_x), original_grid_y) in target_cells:
                draw.ellipse(
                    (
                        center_x - marker_radius,
                        center_y - marker_radius,
                        center_x + marker_radius,
                        center_y + marker_radius,
                    ),
                    outline=(60, 211, 62),
                    width=2,
                )
            else:
                draw.line(
                    (
                        center_x - marker_radius, center_y - marker_radius,
                        center_x + marker_radius, center_y + marker_radius,
                    ),
                    fill=(255, 48, 48), width=2,
                )
                draw.line(
                    (
                        center_x - marker_radius, center_y + marker_radius,
                        center_x + marker_radius, center_y - marker_radius,
                    ),
                    fill=(255, 48, 48), width=2,
                )
        bar_left, bar_top, bar_width, bar_height = 30, 508, 440, 14
        gradient = np.linspace(0.0, 1.0, bar_width, dtype=np.float32)
        gradient_rgb = self.game._belief_rgb(gradient)
        gradient_image = Image.fromarray(
            np.repeat(gradient_rgb[None, :, :], bar_height + 1, axis=0)
        )
        image.paste(gradient_image, (bar_left, bar_top))
        draw.text((bar_left, 524), "0%", fill="black")
        draw.text((bar_left + bar_width // 2 - 10, 524), "50%", fill="black")
        draw.text((bar_left + bar_width - 24, 524), "100%", fill="black")
        temporary_path = output_path.with_suffix(".tmp.png")
        image.save(temporary_path)
        temporary_path.replace(output_path)

    def render(self):
        return self.game.render()

    def close(self) -> None:
        self.game.close()
        super().close()

    def step_log(self) -> None:
        if self.game.step_count % 50 != 0 and not (self._last_terminated or self._last_truncated):
            return
        LOG.info(
            f"[Ep {self.episode_num} - Step {self.game.step_count}] "
            f"coverage={np.count_nonzero(self._observed_mask())/self.game.shared_belief.size:.1%}, "
            f"reward={self._last_team_reward:.3f}, step_components={self._last_reward_breakdown}, "
            f"episode_components={self._episode_reward_breakdown}, "
            f"episode_reward={self._episode_team_reward:.3f}"
        )

    def _snapshot(self) -> CooperativeSearchState:
        state = CooperativeSearchState(
            entities=[],
            sim_time=float(self.game.step_count * self.game_config.step_interval_secs),
            drone_positions={agent: list(position) for agent, position in self.game.drone_positions.items()},
            dead_drone_positions={
                agent: list(position)
                for agent, position in self.game.dead_drone_positions.items()
            },
            threat_positions=(
                {
                    agent: list(position)
                    for agent, position in self.game.threat_positions.items()
                }
                if self.dynamic_threats_enabled else {}
            ),
            headings=dict(self.game.headings),
            speeds_mps=dict(self.game.speeds_mps),
            detected_flags=dict(self.game.detected_flags),
            threat_detected_flags=dict(self.game.threat_detected_flags),
            drones_detected_by_threat=dict(self.game.drones_detected_by_threat),
            visited_counts=self._pool_grid(self.game.visited_counts, reducer="sum").tolist(),
            shared_belief=self._pool_grid(self.game.shared_belief, reducer="mean").tolist(),
            positive_counts=self._pool_grid(self.game.positive_counts, reducer="sum").tolist(),
            negative_counts=self._pool_grid(self.game.negative_counts, reducer="sum").tolist(),
            frame_index=self.game.step_count,
            total_reward=self._episode_team_reward,
        )
        state._raw_data = state.to_dict()
        return state

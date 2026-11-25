"""cooperative_search 四分区 QMIX 适配环境（v0.15.2 联合验证科目）。

一张环境实例管理 600×600 大地图（2×2 责任区），一个 ``uav_team`` Agent 的
四个无名组分别负责一个 300×300 责任区。观测、组状态与奖励全部按责任区
局部语义计算：本区 ``get_obs``/``state_for_agent_ids`` 使用统一的区内坐标、
网格方向和分辨率，与原四机 300×300 科目保持同构；每区奖励由本区扫描、
发现等事件产生并只分配给本组四机。必选链路为
**当前责任区探索网格 → SpatialGridRepresentation → 本组各机策略观测**。

通用装配、动作转换、终止/截断契约、概率图保存等复用
:class:`envs.sense.cooperative_search_env.CooperativeSearchEnv`；本文件重写
观测/状态/奖励/信息的分区口径，不改变原四机环境。
"""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from common.exception import EnvironmentException
from config import EnvConfig, SpatialRepresentationConfig, ThreatRoamingConfig
from envs.base import AgentID, EnvStepReturn, ParallelEnvironment
from envs.games.cooperative_search.game import NUM_HEADINGS
from envs.games.cooperative_search.partition_game import (
    CooperativeSearchPartitionGame,
    PartitionedSearchConfig,
)
from envs.sense.cooperative_search_env import (
    LOG,
    CooperativeSearchEnv,
    get_threat_obs,
    threat_observation_dim,
)
from envs.state.representation import SpatialGridRepresentation
from envs.sense.threat_roaming_observation import ThreatRoamingObservation


class CooperativeSearchPartitionEnv(CooperativeSearchEnv):
    """16 机四分区责任区隔离的 Dec-POMDP；组宽固定为本区无人机数。"""

    def __init__(self, config: EnvConfig, sim_config=None, connect_sim=True,
                 sim_manager=None, role_id=None) -> None:
        # 分区环境自己完成装配：父类装配按“单区游戏 + 全局奖励”设计，
        # 直接复用会在游戏构建与维度计算处与分区语义冲突。
        ParallelEnvironment.__init__(
            self, config=config, sim_config=sim_config, connect_sim=False,
            sim_manager=sim_manager, role_id=role_id,
        )
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
        # 分区观测/组状态使用区局部网格；50×50 状态池化图仍只作为旧科目对照，
        # 本环境要求启用态势表征，不再消费该池化图。
        self.state_grid_width = int(metadata.get("state_grid_width", 50))
        self.state_grid_height = int(metadata.get("state_grid_height", 50))
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

        self.game = CooperativeSearchPartitionGame(self.game_config, render_mode=self.render_mode)
        self.game.set_dynamic_threats_enabled(self.dynamic_threats_enabled)
        self.num_regions = self.game.num_regions
        self.drones_per_region = self.game.drones_per_region
        self.targets_per_region = self.game.targets_per_region
        self._validate_correct_detection_reward_curve()
        self._correct_detection_reward_growth_base = (
            self._solve_correct_detection_reward_growth_base()
        )

        # 必选链路：责任区探索网格 → 态势表征 → 策略观测。缺失输出维度
        # 属于配置错误，明确失败，不自动选值。
        representation_dim = metadata.get("situation_representation_output_dim")
        if representation_dim is None:
            raise EnvironmentException(
                detail=(
                    "partitioned cooperative-search requires "
                    "situation_representation_output_dim in env metadata"
                )
            )
        try:
            self.situation_representation = SpatialGridRepresentation(
                SpatialRepresentationConfig(
                    input_shape=(
                        3,
                        self.game_config.belief_grid_height // self.game.region_rows,
                        self.game_config.belief_grid_width // self.game.region_cols,
                    ),
                    output_dim=representation_dim,
                )
            )
        except ValueError as exc:
            raise EnvironmentException(detail=f"invalid situation representation: {exc}") from exc

        # 四个局部栅格通道依次为：共享概率、访问次数、友机位置、动态威胁位置。
        self._obs_dim = self.fov_size * self.fov_size * 4 + 3
        self._obs_dim += self.situation_representation.config.output_dim
        self.threat_roaming_observation = (
            ThreatRoamingObservation(self.game_config.threat_roaming)
            if self.game_config.threat_roaming is not None else None
        )
        self._threat_obs_dim = self._threat_observation_dim()
        # 全局集中状态 = 各区表征拼接 + 全部实体区局部坐标/航向； Mixer 消费
        # 的组状态见 state_for_agent_ids（区表征 + 本区实体），两者维度不同。
        state_map_dim = self.num_regions * self.situation_representation.config.output_dim
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
        self._init_region_reward_state()
        # 父类同名全局跟踪结构保留初始化，维持与父类路径的属性兼容。
        self._rewarded_target_indices: set[int] = set()
        self._penalized_false_detection_cells: set[tuple[int, int]] = set()
        self._last_revisited_empty_cell_count = 0
        self._last_revisited_empty_coverage_ratio = 0.0
        self._last_game_infos: dict[str, dict] = {}
        self._last_game_terminations: dict[str, bool] = {}
        self._last_game_truncations: dict[str, bool] = {}
        self.pre_state = self.post_state = self._snapshot()

    def _threat_observation_dim(self) -> int:
        if not self.threat_agents:
            return 0
        if self.threat_roaming_observation is not None:
            return self.threat_roaming_observation.observation_dim
        return threat_observation_dim(
            self.game_config.num_targets,
            self.game_config.num_dynamic_threats,
        )

    def _init_region_reward_state(self) -> None:
        count = self.num_regions
        self._rewarded_region_target_indices: list[set[int]] = [set() for _ in range(count)]
        self._penalized_region_false_cells: list[set[tuple[int, int]]] = [
            set() for _ in range(count)
        ]
        self._last_region_reward_breakdown: list[dict[str, float]] = [
            self._empty_reward_breakdown() for _ in range(count)
        ]
        self._episode_region_reward_breakdown: list[dict[str, float]] = [
            self._empty_reward_breakdown() for _ in range(count)
        ]
        self._last_region_team_rewards: list[float] = [0.0] * count
        self._episode_region_team_rewards: list[float] = [0.0] * count
        self._last_region_revisited_ratios: list[float] = [0.0] * count
        self._last_region_revisited_cell_counts: list[int] = [0] * count

    @staticmethod
    def _build_game_config(raw: dict[str, Any]) -> PartitionedSearchConfig:
        valid = {item.name for item in fields(PartitionedSearchConfig)}
        unknown = set(raw) - valid
        if unknown:
            raise EnvironmentException(
                detail=f"unsupported cooperative-search game config: {sorted(unknown)}"
            )
        raw = dict(raw)
        if isinstance(raw.get("threat_roaming"), dict):
            raw["threat_roaming"] = ThreatRoamingConfig(**raw["threat_roaming"])
        return PartitionedSearchConfig(**raw)

    # ---------- 每区正确检测奖励曲线（按 targets_per_region） ----------

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
        if self.targets_per_region < 1:
            raise EnvironmentException(
                detail="correct-detection reward curve requires at least one target per region"
            )

    def _solve_correct_detection_reward_growth_base(self) -> float:
        target_count = self.targets_per_region
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
        target_count = self.targets_per_region
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
        cumulative = [
            self._correct_detection_cumulative_reward(count)
            for count in range(self.targets_per_region + 1)
        ]
        return tuple(
            cumulative[count] - cumulative[count - 1]
            for count in range(1, len(cumulative))
        )

    # ---------- 责任区视图与表征 ----------

    @property
    def group_state_dim(self) -> int:
        """无名组的集中状态维度：区表征 + 本区无人机 4 维飞行状态。"""
        return (
            self.situation_representation.config.output_dim
            + self.drones_per_region * 4
        )

    def _agent_region(self, agent: AgentID) -> int:
        """无人机当前所在责任区（动态）：观测裁剪、区事件结算与奖励分配口径。

        正常推进中无人机受固定职责区边界约束；已死亡实体回退到任务归属区。
        已收工实体保留停机位置用于展示。
        """
        position = self.game.drone_positions.get(agent)
        if position is None:
            return self._home_region(agent)
        return self.game.region_of_position(position)

    def _home_region(self, agent: AgentID) -> int:
        """固定任务归属区：出生簇、组状态锚定与区完成关闭本组的口径。"""
        return self.game.drone_region[agent]

    def _agents_in_region(
        self,
        region: int,
        positions: dict[str, tuple[float, float]],
    ) -> list[str]:
        """当前位置位于指定责任区的存活无人机，按 drone_agents 稳定顺序。"""
        return [
            agent for agent in self.drone_agents
            if agent in positions
            and self.game.region_of_position(positions[agent]) == region
        ]

    def _region_view(self, source: np.ndarray, region: int) -> np.ndarray:
        rows, cols = self.game.region_grid_slice(region)
        return source[rows, cols]

    def _region_observed_mask(self, region: int) -> np.ndarray:
        return self._region_view(self._observed_mask(), region)

    def _region_normalized_visits(self, region: int) -> np.ndarray:
        return self._region_view(self._normalized_visits(), region)

    def _region_representation(self, region: int) -> np.ndarray:
        """从本区当前数值网格编码态势；不读取渲染图、目标真值或他区数据。"""
        source = np.stack((
            self._region_observed_mask(region).astype(np.float32),
            self._region_view(self.game.shared_belief, region),
            self._region_normalized_visits(region),
        ))
        try:
            return self.situation_representation(source)
        except ValueError as exc:
            raise EnvironmentException(detail=f"invalid situation representation: {exc}") from exc

    def _region_flight_state(self, region: int) -> list[float]:
        """责任区固定分组的飞行状态；坐标以本责任区原点归一化。

        正常推进时成员被限制在固定职责区。保留外部注入/历史恢复状态的原始坐标，
        不对归一化结果截断；组状态始终锚定任务责任区。
        """
        values: list[float] = []
        region_origin_x, _, region_origin_y, _ = self.game.region_bounds(region)
        region_width = self.game_config.map_width / self.game.region_cols
        region_height = self.game_config.map_height / self.game.region_rows
        for agent in self.game.region_agents(region):
            if agent in self.game.dead_drone_positions or (
                agent in self.threat_agents and not self.dynamic_threats_enabled
            ):
                values.extend([0.0, 0.0, 0.0, 0.0])
                continue
            x, y = self.game._agent_position(agent)
            heading = self.game.headings[agent]
            angle = 0.0 if heading is None else math.radians(heading * 360.0 / NUM_HEADINGS)
            values.extend([
                (x - region_origin_x) / region_width,
                (y - region_origin_y) / region_height,
                math.sin(angle),
                math.cos(angle),
            ])
        return values

    def state(self) -> np.ndarray:
        if self.situation_representation is None:
            raise EnvironmentException(detail="partitioned env requires situation representation")
        values: list[np.ndarray] = [
            np.concatenate([
                self._region_representation(region) for region in range(self.num_regions)
            ])
        ]
        flight_state: list[float] = []
        for region in range(self.num_regions):
            flight_state.extend(self._region_flight_state(region))
        for agent in self.threat_agents:
            if not self.dynamic_threats_enabled:
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

    def state_for_agent_ids(
            self,
            *,
            agent_name: str,
            agent_ids: tuple[AgentID, ...],
    ) -> np.ndarray:
        """返回该无名组的集中状态：任务责任区表征 + 本组固定成员飞行状态。

        组状态锚定固定分组（home 区）；无人机飞行边界也使用该固定职责区。
        """
        if agent_ids and all(agent in self.threat_agents for agent in agent_ids) and self.threat_roaming_observation is not None:
            if len(agent_ids) != 1:
                raise EnvironmentException(detail="roaming threat inference requires singleton groups")
            return (self.threat_roaming_observation(self.game, agent_ids[0])
                    if self.dynamic_threats_enabled
                    else np.zeros(self._threat_obs_dim, dtype=np.float32))
        regions = {self._home_region(agent) for agent in agent_ids}
        if len(regions) != 1:
            raise EnvironmentException(
                detail=(
                    "partitioned group state requires members from one responsibility "
                    f"region, got regions={sorted(regions)} for members={list(agent_ids)}"
                )
            )
        region = regions.pop()
        state = np.concatenate([
            self._region_representation(region),
            np.asarray(self._region_flight_state(region), dtype=np.float32),
        ]).astype(np.float32)
        if state.size != self.group_state_dim:
            raise EnvironmentException(
                detail=(
                    "group state size mismatch: actual="
                    f"{state.size}, expected={self.group_state_dim}"
                )
            )
        return state

    # ---------- 观测（区局部坐标与槽位语义） ----------

    def get_obs(self) -> dict[AgentID, Any]:
        region_features = {
            region: self._region_representation(region)
            for region in range(self.num_regions)
        }
        region_visits = {
            region: self._region_normalized_visits(region)
            for region in range(self.num_regions)
        }
        observations = {}
        for agent in self.drone_agents:
            position = self.game.drone_positions.get(agent)
            if position is None:
                observations[agent] = np.zeros(self._obs_dim, dtype=np.float32)
                continue
            region = self._agent_region(agent)
            rows, cols = self.game.region_grid_slice(region)
            gx, gy = self.game._world_to_grid(position)
            local_gx = gx - cols.start
            local_gy = gy - rows.start
            region_origin_x, _, region_origin_y, _ = self.game.region_bounds(region)
            region_width = self.game_config.map_width / self.game.region_cols
            region_height = self.game_config.map_height / self.game.region_rows
            belief_view = self._region_view(self.game.shared_belief, region)
            features = [
                # 区视图 100×100 与原 300×300 科目同分辨率；窗外(outside=1.0)
                # 表示责任区边界外，观测不含其他责任区数据。
                self._local_observation_crop(
                    belief_view, local_gx, local_gy, outside=1.0, reducer="mean"
                ).ravel(),
                self._local_observation_crop(
                    region_visits[region], local_gx, local_gy, outside=1.0, reducer="mean"
                ).ravel(),
                self._region_friendly_map(agent, region, local_gx, local_gy).ravel(),
                self._region_threat_map(position, region, local_gx, local_gy).ravel(),
                np.asarray([
                    (position[0] - region_origin_x) / region_width,
                    (position[1] - region_origin_y) / region_height,
                    float(self.game.detected_flags[agent]),
                ], dtype=np.float32),
                region_features[region],
            ]
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

    def _region_friendly_map(
        self, observing_agent: str, region: int, center_x: int, center_y: int,
    ) -> np.ndarray:
        result = np.zeros_like(self._region_view(self.game.shared_belief, region), dtype=np.float32)
        rows, cols = self.game.region_grid_slice(region)
        for agent in self._agents_in_region(region, self.game.drone_positions):
            if agent == observing_agent:
                continue
            position = self.game.drone_positions[agent]
            gx, gy = self.game._world_to_grid(position)
            result[gy - rows.start, gx - cols.start] = 1.0
        return self._local_observation_crop(
            result, center_x, center_y, outside=0.0, reducer="max"
        )

    def _region_threat_map(
        self,
        observer_position: tuple[float, float],
        region: int,
        center_x: int,
        center_y: int,
    ) -> np.ndarray:
        """本区视图内的威胁危险势场；威胁源本身不在本区时不进入观测。"""
        rows, cols = self.game.region_grid_slice(region)
        result = np.zeros_like(self._region_view(self.game.shared_belief, region), dtype=np.float32)
        if not self.dynamic_threats_enabled:
            return self._local_observation_crop(
                result, center_x, center_y, outside=0.0, reducer="max"
            )
        region_mask = self.game.region_grid_masks[region]
        warning_radius = self.game_config.threat_warning_radius
        danger_radius = self.game_config.threat_sensor_radius
        transition_width = warning_radius - danger_radius
        for position in self.game.threat_positions.values():
            if not region_mask[tuple(reversed(self.game._world_to_grid(position)))]:
                continue
            if self.game._distance(observer_position, position) > warning_radius:
                continue
            distances = np.hypot(
                self.game._grid_world_x[rows, cols] - position[0],
                self.game._grid_world_y[rows, cols] - position[1],
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

    # ---------- 每区结算 ----------

    @property
    def sensor_footprint_cell_upper_bound(self) -> int:
        """单机一次轴向对齐传感器足迹在本区网格内的严格格数上界。"""
        cell_width = (
            self.game_config.map_width / self.game_config.belief_grid_width
        )
        cell_height = (
            self.game_config.map_height / self.game_config.belief_grid_height
        )
        columns = math.floor(
            2.0 * self.game_config.sensor_radius / cell_width
        ) + 1
        footprint_rows = math.floor(
            2.0 * self.game_config.sensor_radius / cell_height
        ) + 1
        region_cells = (
            self.game_config.belief_grid_height // self.game.region_rows
        ) * (
            self.game_config.belief_grid_width // self.game.region_cols
        )
        return min(region_cells, max(1, columns) * max(1, footprint_rows))

    def _region_new_detection_events(self, region: int) -> tuple[int, int]:
        """结算本区本局首次达到概率上限的正确目标和错误标记格。"""
        threshold = self.game_config.max_belief_probability - 1e-6
        rows, cols = self.game.region_grid_slice(region)
        belief_view = self._region_view(self.game.shared_belief, region)
        region_targets = self.game.region_target_positions(region)
        target_cells = [
            self.game._world_to_grid(position) for position in region_targets
        ]
        correct_detections = 0
        for target_index, (grid_x, grid_y) in enumerate(target_cells):
            if (
                target_index not in self._rewarded_region_target_indices[region]
                and self.game.shared_belief[grid_y, grid_x] >= threshold
            ):
                self._rewarded_region_target_indices[region].add(target_index)
                correct_detections += 1

        true_cells = set(target_cells)
        marked_cells = {
            (int(grid_x), int(grid_y))
            for grid_y, grid_x in np.argwhere(belief_view >= threshold)
        }
        new_false_cells = (
            marked_cells - true_cells - self._penalized_region_false_cells[region]
        )
        self._penalized_region_false_cells[region].update(new_false_cells)
        return correct_detections, len(new_false_cells)

    def _region_sequential_settlement(
        self, region: int, *, previous_observed: np.ndarray,
    ) -> tuple[int, float]:
        """按稳定无人机顺序结算本区同一步内的搜索足迹。

        结算对象是当前位于本区的无人机；正常推进下与固定职责区一致。
        重复空白上界按区内当前飞机数计算。
        """
        active = self._agents_in_region(region, self.game.drone_positions)
        if not active:
            return 0, 0.0
        rows, cols = self.game.region_grid_slice(region)
        observed = previous_observed[rows, cols].astype(bool, copy=True)
        confirmed_empty = (
            self._region_view(self.game.shared_belief, region)
            <= self.game_config.min_belief_probability + 1e-6
        )
        region_mask = self.game.region_grid_masks[region]
        newly_covered = 0
        repeated = 0
        for agent in active:
            position = self.game.drone_positions[agent]
            footprint = self.game._sensor_grid_mask(position) & region_mask
            footprint_view = footprint[rows, cols]
            newly_covered += int(np.count_nonzero(footprint_view & ~observed))
            repeated += int(
                np.count_nonzero(footprint_view & observed & confirmed_empty)
            )
            observed |= footprint_view
        upper = len(active) * self.sensor_footprint_cell_upper_bound
        revisited_ratio = float(np.clip(repeated / max(1, upper), 0.0, 1.0))
        return newly_covered, revisited_ratio

    def _region_frontier_progress(
        self,
        region: int,
        *,
        previous_observed: np.ndarray,
        previous_drone_positions: dict[str, tuple[float, float]],
    ) -> float:
        """当前位于本区的无人机向本区最近未探索网格的平均归一化推进。"""
        rows, cols = self.game.region_grid_slice(region)
        unobserved = ~previous_observed[rows, cols]
        region_previous = {
            agent: position
            for agent, position in previous_drone_positions.items()
            if self.game.region_of_position(position) == region
        }
        if not np.any(unobserved) or not region_previous:
            return 0.0
        frontier_x = self.game._grid_world_x[rows, cols][unobserved]
        frontier_y = self.game._grid_world_y[rows, cols][unobserved]
        movement_distance = self.game_config.movement_distance_km
        normalized_progress: list[float] = []
        for agent, previous_position in region_previous.items():
            current_position = self.game.drone_positions.get(agent)
            if current_position is None:
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

    def step(self, actions: dict[AgentID, Any]) -> EnvStepReturn:
        self.pre_state = self.post_state
        previous_observed = self._observed_mask()
        previous_drone_positions = self.game.drone_positions.copy()
        previous_alive_drones = len(self.game.drone_positions)
        previous_region_counts = [
            len(self._rewarded_region_target_indices[region])
            for region in range(self.num_regions)
        ]
        self.current_actions = self.send_actions(actions)
        newly_dead_drones = previous_alive_drones - len(self.game.drone_positions)
        region_newly_dead = [0] * self.num_regions
        if newly_dead_drones:
            for agent, previous_position in previous_drone_positions.items():
                if agent not in self.game.drone_positions:
                    # 死亡惩罚计入无人机当时所在的区（动态归属）。
                    region_newly_dead[
                        self.game.region_of_position(previous_position)
                    ] += 1

        for region in range(self.num_regions):
            correct_detections, false_detections = self._region_new_detection_events(region)
            current_correct_count = previous_region_counts[region] + correct_detections
            frontier_progress = self._region_frontier_progress(
                region,
                previous_observed=previous_observed,
                previous_drone_positions=previous_drone_positions,
            )
            newly_explored_cells, revisited_ratio = self._region_sequential_settlement(
                region, previous_observed=previous_observed,
            )
            self._last_region_revisited_ratios[region] = revisited_ratio
            self._last_region_revisited_cell_counts[region] = int(
                round(
                    revisited_ratio
                    * self.drones_per_region
                    * self.sensor_footprint_cell_upper_bound
                )
            )
            breakdown = self._reward_components(
                previous_correct_count=previous_region_counts[region],
                current_correct_count=current_correct_count,
                false_detections=false_detections,
                newly_explored_cells=newly_explored_cells,
                frontier_progress=frontier_progress,
                newly_dead_drones=region_newly_dead[region],
                revisited_empty_coverage_ratio=revisited_ratio,
            )
            self._last_region_reward_breakdown[region] = breakdown
            region_team = float(sum(breakdown.values()))
            self._last_region_team_rewards[region] = region_team
            self._episode_region_team_rewards[region] += region_team
            for key, value in breakdown.items():
                self._episode_region_reward_breakdown[region][key] += value

        # 全局口径 = 各区求和；不得把四区总奖励再次均分给所有无人机。
        self._last_reward_breakdown = {
            key: sum(
                self._last_region_reward_breakdown[region][key]
                for region in range(self.num_regions)
            )
            for key in self._REWARD_KEYS
        }
        for key in self._REWARD_KEYS:
            merged = sum(
                self._episode_region_reward_breakdown[region][key]
                for region in range(self.num_regions)
            )
            self._episode_reward_breakdown[key] = merged
        self._last_team_reward = float(sum(self._last_reward_breakdown.values()))
        self._episode_team_reward += self._last_team_reward
        self.game.total_reward = self._episode_team_reward
        self._last_revisited_empty_coverage_ratio = float(
            sum(self._last_region_revisited_ratios)
        )
        self._last_revisited_empty_cell_count = sum(
            self._last_region_revisited_cell_counts
        )
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

    def get_reward(self, **kwargs) -> dict[AgentID, float]:
        """区团队奖励均分给"当前位于本区"的无人机（动态归属）。

        区内没有无人机时该区本步奖励不发放（无人扫描通常也没有区事件）。
        """
        rewards: dict[AgentID, float] = {agent: 0.0 for agent in self.possible_agents}
        for region in range(self.num_regions):
            members = self._agents_in_region(region, self.game.drone_positions)
            if not members:
                continue
            share = self._last_region_team_rewards[region] / len(members)
            for agent in members:
                rewards[agent] = share
        return rewards

    def get_episode_rewards_details(self) -> dict[str, float]:
        return {
            component_name: float(value)
            for component_name, value in self._episode_reward_breakdown.items()
        }

    def reset(self, seed=None, options=None) -> Any:
        self._init_region_reward_state()
        self._rewarded_target_indices = set()
        self._penalized_false_detection_cells = set()
        return super().reset(seed=seed, options=options)

    # ---------- 信息与日志 ----------

    def _region_completed_count(self) -> int:
        return sum(
            1
            for region in range(self.num_regions)
            if self.game.region_targets_fully_identified(region)
        )

    def _info(self, agent: str) -> dict[str, Any]:
        info = {
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
            "coverage": float(
                np.count_nonzero(self._observed_mask()) / self.game.shared_belief.size
            ),
            # 完整成功要求四区全部完成（每区 10 个、共 40 个信号源）。
            "success": bool(
                self.game.all_targets_fully_identified()
                and len(self.game.dead_drone_positions) == 0
            ),
            "target_discovered_count": int(sum(
                len(self._rewarded_region_target_indices[region])
                for region in range(self.num_regions)
            )),
            "own_loss_count": int(len(self.game.dead_drone_positions)),
            "coverage_rate_percent": (
                float(
                    np.count_nonzero(self._observed_mask())
                    / self.game.shared_belief.size
                ) * 100.0
                if self.game.shared_belief.size
                else 0.0
            ),
            "false_target_count": int(sum(
                len(self._penalized_region_false_cells[region])
                for region in range(self.num_regions)
            )),
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
            "regions_completed": self._region_completed_count(),
            "regions_total": self.num_regions,
        }
        if agent in self.drone_agents:
            region = self._agent_region(agent)
            region_observed = self._region_observed_mask(region)
            info.update({
                "region_index": region,
                "region_targets_total": self.targets_per_region,
                "region_target_discovered_count": int(
                    len(self._rewarded_region_target_indices[region])
                ),
                "region_success": bool(
                    self.game.region_targets_fully_identified(region)
                ),
                "region_team_reward": self._last_region_team_rewards[region],
                "region_episode_team_reward": self._episode_region_team_rewards[region],
                "region_reward_breakdown": self._last_region_reward_breakdown[region].copy(),
                "region_episode_reward_breakdown": (
                    self._episode_region_reward_breakdown[region].copy()
                ),
                "region_coverage_rate_percent": (
                    float(np.count_nonzero(region_observed) / region_observed.size) * 100.0
                    if region_observed.size
                    else 0.0
                ),
                "region_revisited_empty_cell_count": (
                    self._last_region_revisited_cell_counts[region]
                ),
            })
        return info

    def step_log(self) -> None:
        if self.game.step_count % 50 != 0 and not (self._last_terminated or self._last_truncated):
            return
        region_summary = " | ".join(
            f"R{region}: found={len(self._rewarded_region_target_indices[region])}"
            f"/{self.targets_per_region}"
            for region in range(self.num_regions)
        )
        LOG.info(
            f"[Ep {self.episode_num} - Step {self.game.step_count}] "
            f"regions_completed={self._region_completed_count()}/{self.num_regions}, "
            f"{region_summary}, "
            f"reward={self._last_team_reward:.3f}, "
            f"episode_reward={self._episode_team_reward:.3f}"
        )

    # ---------- 元数据更新 ----------

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

"""600×600 四分区协同搜索游戏：责任区隔离的 mock 连续仿真。

v0.15.2 四分区科目的游戏层。地图按 ``region_rows × region_cols`` 平铺成若干
300×300 责任区（默认 2×2），无人机按 ``drone_*`` 序号连续分组固定归属某区，
只能在本区内飞行与扫描；每局各区独立随机生成恰好 ``num_targets / 分区数``
个信号源。信念更新、威胁机制、渲染等通用动力学全部复用
:class:`envs.games.cooperative_search.game.CooperativeSearchGame`，本文件只
重写落点采样、边界约束、传感器裁剪和分区完成判定。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from envs.games.cooperative_search.threat_roaming import ThreatRoaming

from common.exception import EnvironmentException
from envs.games.cooperative_search.game import (
    HEADING_STEP_DEGREES,
    NUM_HEADINGS,
    CooperativeSearchConfig,
    CooperativeSearchGame,
    FlightCommand,
)


@dataclass(frozen=True)
class PartitionedSearchConfig(CooperativeSearchConfig):
    """分区协同搜索配置；在原科目字段上增加责任区布局。

    ``region_rows``/``region_cols`` 是责任区行/列数；无人机数量必须能被分区
    数整除且每区 1..4 架，信号源数量必须能被分区数整除。地图与信念网格的
    宽高也必须被分区布局整除，保证每个责任区的网格与几何边界一致。
    """

    region_rows: int = 2
    region_cols: int = 2

    def __post_init__(self) -> None:
        # 父类校验按“单区至多四机”设计（num_drones>4 直接拒绝），与分区科目
        # 冲突，因此这里完整重写校验：先做与父类等价的通用检查，再把无人机
        # 数约束落到“每区”口径上。
        if min(self.map_width, self.map_height) < 2:
            raise EnvironmentException(detail="map dimensions must both be at least 2")
        if min(self.belief_grid_width, self.belief_grid_height) < 2:
            raise EnvironmentException(detail="belief grid dimensions must both be at least 2")
        if min(self.fixed_speed_mps, self.threat_speed_mps, self.step_interval_secs) <= 0:
            raise EnvironmentException(detail="fixed speed and step interval must be positive")
        if self.threat_sensor_radius <= 0:
            raise EnvironmentException(detail="threat sensor radius must be positive")
        if self.threat_warning_radius <= self.threat_sensor_radius:
            raise EnvironmentException(
                detail="threat warning radius must exceed threat sensor radius"
            )
        if self.initial_threat_separation < self.threat_warning_radius:
            raise EnvironmentException(
                detail="initial threat separation must cover the threat warning radius"
            )
        if not 0.5 < self.detection_probability <= 1.0:
            raise EnvironmentException(detail="detection_probability must be in (0.5, 1]")
        if not 0.0 <= self.false_alarm_probability < 0.5:
            raise EnvironmentException(detail="false_alarm_probability must be in [0, 0.5)")
        if self.detection_probability <= self.false_alarm_probability:
            raise EnvironmentException(
                detail="detection_probability must exceed false_alarm_probability"
            )
        if not 0.5 < self.detection_threshold <= 1.0:
            raise EnvironmentException(detail="detection_threshold must be in (0.5, 1]")
        if not 0.0 < self.min_belief_probability < 0.5:
            raise EnvironmentException(detail="min_belief_probability must be in (0, 0.5)")
        if not 0.5 < self.max_belief_probability < 1.0:
            raise EnvironmentException(detail="max_belief_probability must be in (0.5, 1)")
        if self.detection_threshold > self.max_belief_probability:
            raise EnvironmentException(
                detail="detection_threshold cannot exceed max_belief_probability"
            )
        if not 0 <= self.max_turn_degrees <= 180:
            raise EnvironmentException(detail="max_turn_degrees must be in [0, 180]")

        if self.region_rows < 1 or self.region_cols < 1:
            raise EnvironmentException(
                detail="region_rows and region_cols must both be at least 1"
            )
        region_count = self.region_rows * self.region_cols
        if self.map_width % self.region_cols or self.map_height % self.region_rows:
            raise EnvironmentException(
                detail="map dimensions must be evenly divisible by the region layout"
            )
        if (
            self.belief_grid_width % self.region_cols
            or self.belief_grid_height % self.region_rows
        ):
            raise EnvironmentException(
                detail="belief grid must be evenly divisible by the region layout"
            )
        if self.num_dynamic_threats < 0 or self.num_drones + self.num_dynamic_threats < 1:
            raise EnvironmentException(detail="invalid drone, threat, or target count")
        if self.num_dynamic_threats and self.num_dynamic_threats % region_count:
            # 威胁出生按区均分（每区恰好 num_dynamic_threats/分区数 个），
            # 保证每个责任区开局就有威胁存在。
            raise EnvironmentException(
                detail=(
                    "num_dynamic_threats must be evenly divisible by the region "
                    "layout so every region spawns threats, got "
                    f"{self.num_dynamic_threats} threats over {region_count} regions"
                )
            )
        if self.num_targets < region_count or self.num_targets % region_count:
            raise EnvironmentException(
                detail=(
                    "num_targets must be evenly divisible by the region layout, got "
                    f"{self.num_targets} targets over {region_count} regions"
                )
            )
        if self.num_drones > 0:
            drones_per_region, remainder = divmod(self.num_drones, region_count)
            if remainder or drones_per_region < 1 or drones_per_region > 4:
                raise EnvironmentException(
                    detail=(
                        "num_drones must split into 1..4 drones per region, got "
                        f"{self.num_drones} drones over {region_count} regions"
                    )
                )
        region_width = self.map_width // self.region_cols
        region_height = self.map_height // self.region_rows
        if not 0.0 < self.initial_formation_side_km <= min(region_width, region_height):
            raise EnvironmentException(
                detail="initial formation side must fit inside one responsibility region"
            )
        if self.num_drones > 1 and self.initial_formation_side_km < self.initial_separation:
            raise EnvironmentException(
                detail="initial formation side must cover initial separation"
            )

    @property
    def region_count(self) -> int:
        return self.region_rows * self.region_cols

    @property
    def drones_per_region(self) -> int:
        return self.num_drones // self.region_count

    @property
    def targets_per_region(self) -> int:
        return self.num_targets // self.region_count


class CooperativeSearchPartitionGame(CooperativeSearchGame):
    """一张大地图管理多个责任区的协同搜索游戏。

    分区只影响：无人机出生簇与边界约束按本责任区、传感器观测裁剪到本区、
    每区目标独立采样与独立完成判定。信念数据库仍是一张覆盖全图的大网格，
    每区对应其中一个连续子块；动态威胁（若配置）仍按整图活动。
    """

    def __init__(
        self,
        config: PartitionedSearchConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        self.partition_config = config or PartitionedSearchConfig()
        self.region_rows = self.partition_config.region_rows
        self.region_cols = self.partition_config.region_cols
        self.num_regions = self.partition_config.region_count
        self.drones_per_region = self.partition_config.drones_per_region
        self.targets_per_region = self.partition_config.targets_per_region
        # 父类构造末尾会调用 reset()，分区视图必须先就位；config、drone_agents
        # 等基础属性先按配置生成，父类构造会以相同内容覆盖。
        self.config = self.partition_config
        self.drone_agents = [
            f"drone_{index}" for index in range(self.partition_config.num_drones)
        ]
        # drone_i 的责任区 = i // drones_per_region；成员及责任区在本局固定。
        self.drone_region = {
            agent: index // self.drones_per_region
            for index, agent in enumerate(self.drone_agents)
        }
        self.region_grid_masks = [self._build_region_grid_mask(r) for r in range(self.num_regions)]
        super().__init__(config=self.partition_config, render_mode=render_mode)

    def _build_threat_roaming(self):
        if self.partition_config.threat_roaming is None:
            return None
        if self.partition_config.num_dynamic_threats < 2 * self.num_regions:
            raise EnvironmentException(detail="roaming requires at least two threats per region")
        return ThreatRoaming(
            self.partition_config.threat_roaming,
            self.partition_config.map_width, self.partition_config.map_height,
            region_rows=self.region_rows, region_cols=self.region_cols,
        )

    # ---------- 责任区几何 ----------

    def flight_bounds(self, agent: str) -> tuple[float, float, float, float]:
        if agent not in self.drone_region:
            return super().flight_bounds(agent)
        x_min, x_max, y_min, y_max = self.region_bounds(self.drone_region[agent])
        # 内部上界不归本区，避免落在分界线上后被观察/奖励判为邻区。
        if x_max < self.config.map_width:
            x_max = math.nextafter(x_max, x_min)
        if y_max < self.config.map_height:
            y_max = math.nextafter(y_max, y_min)
        return x_min, x_max, y_min, y_max

    def region_bounds(self, region: int) -> tuple[float, float, float, float]:
        """返回责任区 (x_min, x_max, y_min, y_max)；边界点归左侧/下侧分区。"""
        row, col = divmod(int(region), self.region_cols)
        return (
            col * self.config.map_width / self.region_cols,
            (col + 1) * self.config.map_width / self.region_cols,
            row * self.config.map_height / self.region_rows,
            (row + 1) * self.config.map_height / self.region_rows,
        )

    def region_grid_slice(self, region: int) -> tuple[slice, slice]:
        """返回责任区在信念网格上的 (行切片, 列切片)；半开区间保证格归属唯一。"""
        row, col = divmod(int(region), self.region_cols)
        grid_row = slice(
            row * self.config.belief_grid_height // self.region_rows,
            (row + 1) * self.config.belief_grid_height // self.region_rows,
        )
        grid_col = slice(
            col * self.config.belief_grid_width // self.region_cols,
            (col + 1) * self.config.belief_grid_width // self.region_cols,
        )
        return grid_row, grid_col

    def _build_region_grid_mask(self, region: int) -> np.ndarray:
        mask = np.zeros(self.grid_shape, dtype=bool)
        rows, cols = self.region_grid_slice(region)
        mask[rows, cols] = True
        return mask

    def region_target_positions(self, region: int) -> list[tuple[float, float]]:
        """返回指定责任区的信号源真实坐标；仅供环境内部奖励/完成判定使用。"""
        start = int(region) * self.targets_per_region
        return self._target_positions[start:start + self.targets_per_region]

    def region_agents(self, region: int) -> list[str]:
        """责任区固定分组的成员，同时决定无人机的飞行边界。"""
        return [
            agent for agent, value in self.drone_region.items() if value == int(region)
        ]

    def region_of_position(self, position: tuple[float, float]) -> int:
        """按当前位置判定所在责任区；半开区间保证边界点归属唯一。"""
        col = min(
            self.region_cols - 1,
            max(0, int(position[0] / (self.config.map_width / self.region_cols))),
        )
        row = min(
            self.region_rows - 1,
            max(0, int(position[1] / (self.config.map_height / self.region_rows))),
        )
        return row * self.region_cols + col

    # ---------- 落点采样 ----------

    def _sample_drone_cluster(self) -> list[tuple[float, float]]:
        """每区在责任区左上角锚定的固定正方形顶点放置本区无人机。"""
        side = float(self.config.initial_formation_side_km)
        points: list[tuple[float, float]] = []
        for region in range(self.num_regions):
            x_min, _, y_min, _ = self.region_bounds(region)
            vertices = [
                (x_min, y_min),
                (x_min + side, y_min),
                (x_min, y_min + side),
                (x_min + side, y_min + side),
            ]
            points.extend(vertices[:self.drones_per_region])
        return points

    def _sample_region_threats(
        self,
        region: int,
        existing: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """在本责任区内随机出生 threats_per_region 个威胁。

        出生按区均分保证每个责任区开局就有威胁；区内均匀采样并与全部
        已放置实体（无人机簇、其他区已放的威胁）保持 initial_threat_separation。
        """
        per_region = self.config.num_dynamic_threats // self.num_regions
        if per_region <= 0:
            return []
        x_min, x_max, y_min, y_max = self.region_bounds(region)
        points = list(existing)
        separation = self.config.initial_threat_separation
        first_new = len(points)
        for _ in range(per_region):
            for _attempt in range(20_000):
                candidate = (
                    float(self._rng.uniform(x_min, x_max)),
                    float(self._rng.uniform(y_min, y_max)),
                )
                if all(self._distance(candidate, point) >= separation for point in points):
                    points.append(candidate)
                    break
            else:
                raise EnvironmentException(
                    detail=(
                        "unable to place threats with configured separation inside "
                        f"region {region}"
                    )
                )
        return points[first_new:]

    def _sample_region_targets(
        self,
        region: int,
        existing: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """在本责任区内独立随机生成恰好 targets_per_region 个信号源。

        不是全图撒点后按区接受数量不均，也不是同一组随机坐标平移复制；
        与已放置实体保持 initial_separation，共用任务 rng 保证 seed 可复现。
        """
        x_min, x_max, y_min, y_max = self.region_bounds(region)
        points = list(existing)
        separation = self.config.initial_separation
        first_new = len(points)
        for _ in range(self.targets_per_region):
            for _attempt in range(20_000):
                candidate = (
                    float(self._rng.uniform(x_min, x_max)),
                    float(self._rng.uniform(y_min, y_max)),
                )
                if all(self._distance(candidate, point) >= separation for point in points):
                    points.append(candidate)
                    break
            else:
                raise EnvironmentException(
                    detail=(
                        "unable to place targets with configured separation inside "
                        f"region {region}"
                    )
                )
        return points[first_new:]

    # ---------- reset / step ----------

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.step_count = 0
        self.running = True
        self.total_reward = 0.0
        drone_points = self._sample_drone_cluster()
        self.drone_positions = dict(zip(self.drone_agents, drone_points, strict=True))
        # 威胁出生按区均分：每区恰好 num_dynamic_threats/分区数 个，与无人机
        # 簇、已放置威胁保持 initial_threat_separation（跨区点也计入距离约束）。
        threat_points: list[tuple[float, float]] = []
        for region in range(self.num_regions):
            threat_points.extend(self._sample_region_threats(
                region, existing=drone_points + threat_points,
            ))
        self.threat_positions = dict(zip(self.threat_agents, threat_points, strict=True))
        occupied_points = drone_points + (
            threat_points if self.dynamic_threats_enabled else []
        )
        self._target_positions: list[tuple[float, float]] = []
        for region in range(self.num_regions):
            self._target_positions.extend(self._sample_region_targets(
                region, existing=occupied_points + list(self._target_positions),
            ))
        self.headings = {agent: None for agent in self.possible_agents}
        self.speeds_mps = {agent: 0.0 for agent in self.possible_agents}
        self.positive_counts = np.zeros(self.grid_shape, dtype=np.float32)
        self.negative_counts = np.zeros(self.grid_shape, dtype=np.float32)
        self.shared_belief = np.full(self.grid_shape, 0.5, dtype=np.float32)
        self.visited_counts = np.zeros(self.grid_shape, dtype=np.int32)
        self.detected_flags = {agent: False for agent in self.possible_agents}
        self.threat_detected_flags = {agent: False for agent in self.threat_agents}
        self.drones_detected_by_threat = {agent: False for agent in self.drone_agents}
        self.dead_drone_positions: dict[str, tuple[float, float]] = {}
        self._mark_visits(self.drone_positions.values())
        self._observe_shared_belief()
        self._update_threat_detections()
        if self.threat_roaming is not None:
            self.threat_roaming.reset(self.threat_positions, self._rng)
        infos = {agent: self._agent_info(agent, False, None) for agent in self.agents}
        return self._observations(), infos

    def region_targets_fully_identified(self, region: int) -> bool:
        """本责任区全部信号源达到概率上限时该区完成。"""
        threshold = self.config.max_belief_probability - 1e-6
        probabilities = self._region_target_probabilities(region)
        if not probabilities:
            return False
        return bool(np.all(np.asarray(probabilities) >= threshold))

    def completed_region_count(self) -> int:
        """已完成（找齐本区全部信号源）的责任区数量。"""
        return sum(
            1
            for region in range(self.num_regions)
            if self.region_targets_fully_identified(region)
        )

    def region_identified_target_count(self, region: int) -> int:
        """本区达到概率上限（完成口径 max_belief-1e-6）的信号源数量。

        与 ``region_targets_fully_identified`` 同口径，供 demo/日志显示；
        父类 ``confirmed_target_count`` 的 detection_threshold 口径受 float32
        精度影响（0.9 的 f32 表示略小于阈值），不适合作进度显示。
        """
        threshold = self.config.max_belief_probability - 1e-6
        return int(np.count_nonzero(
            np.asarray(self._region_target_probabilities(region)) >= threshold
        ))

    def region_confirmed_target_count(self, region: int) -> int:
        threshold = self.config.detection_threshold
        return int(np.count_nonzero(
            np.asarray(self._region_target_probabilities(region)) >= threshold
        ))

    def _region_target_probabilities(self, region: int) -> list[float]:
        return [
            float(self.shared_belief[gy, gx])
            for gx, gy in map(self._world_to_grid, self.region_target_positions(region))
        ]

    def step(self, actions: dict[str, FlightCommand]):
        if not self.agents:
            raise EnvironmentException(
                detail="step called after the episode ended; call reset()"
            )
        missing = set(self.agents) - set(actions)
        if missing:
            raise EnvironmentException(detail=f"missing actions for: {sorted(missing)}")
        previous_threat_positions = dict(self.threat_positions)
        self.step_count += 1
        proposed: dict[str, tuple[float, float]] = {}
        predicted: dict[str, tuple[float, float] | None] = {agent: None for agent in self.agents}
        invalid = {agent: False for agent in self.agents}
        boundary_agents: list[str] = []

        for agent in self.agents:
            command = actions[agent]
            if command.heading not in range(NUM_HEADINGS):
                raise EnvironmentException(
                    detail=f"invalid flight command for {agent}: {command}"
                )
            if not self._turn_allowed(self.headings[agent], command.heading):
                proposed[agent] = self._agent_position(agent)
                invalid[agent] = True
                continue
            position = self._agent_position(agent)
            candidate = self._predict_position(
                position, command, self._movement_distance_km(agent)
            )
            predicted[agent] = candidate
            # 无人机不得越出固定职责区；威胁仍使用整张地图边界。
            if not self._inside_bounds(candidate, self.flight_bounds(agent)):
                boundary_agents.append(agent)
                candidate = position
            proposed[agent] = candidate

        for agent in self.agents:
            self._set_agent_position(agent, proposed[agent])
            if not invalid[agent]:
                self.headings[agent] = actions[agent].heading
                self.speeds_mps[agent] = (
                    0.0 if agent in boundary_agents else self._speed_mps(agent)
                )

        if self.threat_roaming is not None and self.dynamic_threats_enabled:
            self.threat_roaming.step(previous_threat_positions, self.threat_positions,
                                     self.config.threat_movement_distance_km, self._rng)
        self._mark_visits(self.drone_positions.values())
        self._observe_shared_belief()
        self._update_threat_detections()
        eaten_drones = self._eat_detected_drones()
        success = self.all_targets_fully_identified()
        region_completions = {
            region: self.region_targets_fully_identified(region)
            for region in range(self.num_regions)
        }
        region_done_agents = {
            agent
            for agent in self.drone_agents
            if agent in self.drone_positions and region_completions[self.drone_region[agent]]
        }
        all_drones_dead = bool(self.drone_agents) and not self.drone_positions
        truncated = (
            all_drones_dead or self.step_count >= self.config.max_steps
        ) and not success
        acting_agents = list(self.agents)
        terminations = {
            agent: (
                success
                or (agent in eaten_drones and not all_drones_dead)
                or agent in region_done_agents
            )
            for agent in acting_agents
        }
        truncations = {agent: truncated for agent in acting_agents}
        rewards = {agent: 0.0 for agent in acting_agents}
        infos = {
            agent: self._agent_info(
                agent, invalid[agent], predicted[agent], agent in boundary_agents
            )
            for agent in acting_agents
        }
        for agent in eaten_drones:
            infos[agent]["eaten_by_threat"] = True
        observations = self._observations()
        if success or truncated:
            self.agents = []
        else:
            # 已完成责任区的无人机停止行动；其余无人机继续，包括被吃掉的除外。
            self.agents = [
                agent for agent in acting_agents
                if agent not in eaten_drones and agent not in region_done_agents
            ]
        return observations, rewards, terminations, truncations, infos

    def _agent_info(
        self, agent: str, invalid: bool, predicted_position, boundary_violation: bool = False,
    ) -> dict:
        info = super()._agent_info(agent, invalid, predicted_position, boundary_violation)
        if self.threat_roaming is not None and agent in self.threat_positions:
            info.update(self.threat_roaming.metrics[agent])
        if agent in self.drone_region:
            # home_region 是固定任务归属（出生簇/完成判定）；region 是当前
            # 位置所在区（观测裁剪与奖励结算口径）。
            info["home_region"] = self.drone_region[agent]
            if agent in self.drone_positions:
                info["region"] = self.region_of_position(self.drone_positions[agent])
            else:
                info["region"] = self.drone_region[agent]
            info["region_completed"] = self.region_targets_fully_identified(
                self.drone_region[agent]
            )
        return info

    def _draw(self) -> None:
        """完整接管渲染：无人机三角形缩小一半，叠加灰色虚线责任区分界。

        其余画面元素（信念图、范围圈、目标标记、威胁、状态栏）与父类一致；
        因为三角形尺寸是父类 ``_draw`` 内的字面常量，无法通过参数注入，
        只能在子类整体重写。
        """
        import pygame

        active_drones = list(self.drone_positions)
        self._surface.fill(self._UNEXPLORED_COLOR)
        observed = (self.positive_counts + self.negative_counts) > 0
        belief_rgb = np.empty((*self.grid_shape, 3), dtype=np.uint8)
        belief_rgb[:] = self._UNEXPLORED_COLOR
        belief_rgb[observed] = self._belief_rgb(self.shared_belief[observed])
        belief_surface = pygame.surfarray.make_surface(np.transpose(belief_rgb, (1, 0, 2)))
        render_size = (self._RENDER_MAP_PIXELS, self._RENDER_MAP_PIXELS)
        if belief_surface.get_size() != render_size:
            belief_surface = pygame.transform.scale(
                belief_surface, render_size
            )
        self._surface.blit(belief_surface, (0, 0))
        if self.show_ranges:
            ranges = pygame.Surface(render_size, pygame.SRCALPHA)
            for agent in active_drones:
                center = self._world_to_render(self.drone_positions[agent])
                radius = self._distance_to_render(self.config.sensor_radius)
                pygame.draw.circle(ranges, (60, 211, 62, 20), center, radius)
                pygame.draw.circle(ranges, (60, 211, 62, 100), center, radius, 2)
            for agent in self.threat_agents if self.dynamic_threats_enabled else ():
                center = self._world_to_render(self.threat_positions[agent])
                radius = self._distance_to_render(self.config.threat_sensor_radius)
                pygame.draw.circle(
                    ranges, (230, 45, 45, 25), center,
                    radius,
                )
                pygame.draw.circle(
                    ranges, (230, 45, 45, 130), center,
                    radius, 2,
                )
            self._surface.blit(ranges, (0, 0))
        self._draw_threat_roaming()
        cell_width = self._RENDER_MAP_PIXELS / self.config.belief_grid_width
        cell_height = self._RENDER_MAP_PIXELS / self.config.belief_grid_height
        marker_radius = max(3, round(min(cell_width, cell_height) * 0.7))
        target_cells = self._target_grid_cells()
        if self.config.show_target_truth:
            for target in self._target_positions:
                center = self._world_to_render(target)
                pygame.draw.circle(self._surface, (255, 215, 0), center, 5)
                pygame.draw.circle(self._surface, (255, 255, 255), center, 5, 1)
        for grid_y, grid_x in np.argwhere(
            self.shared_belief >= self.config.detection_threshold
        ):
            center = self._world_to_render(self._grid_cell_center(grid_x, grid_y))
            if (int(grid_x), int(grid_y)) in target_cells:
                pygame.draw.circle(
                    self._surface, (60, 211, 62), center, marker_radius, 2,
                )
            else:
                pygame.draw.line(
                    self._surface, (255, 48, 48),
                    (center[0] - marker_radius, center[1] - marker_radius),
                    (center[0] + marker_radius, center[1] + marker_radius), 2,
                )
                pygame.draw.line(
                    self._surface, (255, 48, 48),
                    (center[0] - marker_radius, center[1] + marker_radius),
                    (center[0] + marker_radius, center[1] - marker_radius), 2,
                )
        # 无人机三角形：机头 6px、机尾 4px、半宽 3.5px（父类的一半），线宽 1。
        for agent in active_drones:
            center = self._world_to_render(self.drone_positions[agent])
            heading = self.headings[agent] or 0
            angle = math.radians(heading * HEADING_STEP_DEGREES)
            direction = (math.cos(angle), math.sin(angle))
            side = (-direction[1], direction[0])
            points = [
                (center[0] + direction[0] * 6, center[1] + direction[1] * 6),
                (center[0] - direction[0] * 4 + side[0] * 3.5,
                 center[1] - direction[1] * 4 + side[1] * 3.5),
                (center[0] - direction[0] * 4 - side[0] * 3.5,
                 center[1] - direction[1] * 4 - side[1] * 3.5),
            ]
            pygame.draw.polygon(self._surface, (60, 211, 62), points)
            pygame.draw.polygon(self._surface, (245, 249, 250), points, 1)
        for agent in self.threat_agents if self.dynamic_threats_enabled else ():
            center = self._world_to_render(self.threat_positions[agent])
            pygame.draw.circle(self._surface, (220, 38, 38), center, 8)
            pygame.draw.circle(self._surface, (255, 225, 225), center, 8, 2)
        # 责任区分界线：灰色虚线（12px 线段 + 8px 间隙）。
        dashes = pygame.Surface(render_size, pygame.SRCALPHA)
        segment, gap = 12, 8
        for col in range(1, self.region_cols):
            x = round(col * self._RENDER_MAP_PIXELS / self.region_cols)
            offset = 0
            while offset < self._RENDER_MAP_PIXELS:
                end = min(offset + segment, self._RENDER_MAP_PIXELS)
                pygame.draw.line(
                    dashes, (160, 160, 160, 200), (x, offset), (x, end), 1,
                )
                offset = end + gap
        for row in range(1, self.region_rows):
            y = round(row * self._RENDER_MAP_PIXELS / self.region_rows)
            offset = 0
            while offset < self._RENDER_MAP_PIXELS:
                end = min(offset + segment, self._RENDER_MAP_PIXELS)
                pygame.draw.line(
                    dashes, (160, 160, 160, 200), (offset, y), (end, y), 1,
                )
                offset = end + gap
        self._surface.blit(dashes, (0, 0))
        x = self._RENDER_MAP_PIXELS
        pygame.draw.rect(self._surface, (24, 35, 48), (x, 0, self.config.sidebar_width, self.height))
        lines = [
            "PARTITIONED SEARCH",
            f"{self.config.map_width} x {self.config.map_height} km",
            f"{self.num_regions} regions, free flight",
            f"Step       {self.step_count} / {self.config.max_steps}",
            f"Coverage   {np.count_nonzero(observed)/observed.size:.1%}",
            f"Detected   {sum(self.region_identified_target_count(r) for r in range(self.num_regions))} / {self.config.num_targets}",
            f"Regions    {self.completed_region_count()} / {self.num_regions}",
            f"Speed      {self.config.fixed_speed_mps:.0f} m/s",
            (
                f"Threat V   {self.config.threat_speed_mps:.0f} m/s"
                if self.dynamic_threats_enabled else "Threats    OFF"
            ),
            f"Sensor R   {self.config.sensor_radius:.0f} km",
            (
                f"Threat R   {self.config.threat_sensor_radius:.0f} km"
                if self.dynamic_threats_enabled else "Threat R   --"
            ),
            "", "V  ranges", "Esc  exit",
        ]
        top = 28
        for line in lines:
            font = self._font if line == "PARTITIONED SEARCH" else self._small_font
            self._surface.blit(font.render(line, True, (225, 235, 245)), (x + 20, top))
            top += 34 if font is self._font else 27

    # ---------- 传感器观测（裁剪到本责任区） ----------

    def _observe_shared_belief(self) -> None:
        """与父类相同的贝叶斯证据更新，但无人机只观测本责任区网格。

        邻区信号源既不进入本区可观测集合，也不触发本区无人机的探测报告，
        避免一次跨界传感器覆盖把其他区信号源算作本区成果。
        """
        target_cells: dict[int, list[tuple[float, float]]] = {}
        for region in range(self.num_regions):
            for target in self.region_target_positions(region):
                gx, gy = self._world_to_grid(target)
                target_cells.setdefault(
                    gy * self.config.belief_grid_width + gx, []
                ).append(target)
        region_target_cells: list[dict[int, list[tuple[float, float]]]] = [
            {} for _ in range(self.num_regions)
        ]
        for flat_index, targets in target_cells.items():
            gx = flat_index % self.config.belief_grid_width
            gy = flat_index // self.config.belief_grid_width
            for region in range(self.num_regions):
                rows, cols = self.region_grid_slice(region)
                if rows.start <= gy < rows.stop and cols.start <= gx < cols.stop:
                    region_target_cells[region][flat_index] = targets
                    break

        self.detected_flags = {agent: False for agent in self.possible_agents}
        for agent, position in self.drone_positions.items():
            # 传感器观测跟随当前位置所在责任区：跨区飞行的无人机可以
            # 帮扫所在区的网格与信号源，但不能同时观测多个区。
            region = self.region_of_position(position)
            distance_grid = np.hypot(
                self._grid_world_x - position[0], self._grid_world_y - position[1]
            )
            observable = distance_grid <= self.config.sensor_radius
            observable &= self.region_grid_masks[region]
            flat_observable = observable.ravel()
            flat_distance = distance_grid.ravel()
            observable_target_distances: dict[int, float] = {}
            for flat_index, targets in region_target_cells[region].items():
                target_distances = [self._distance(position, target) for target in targets]
                in_range = [
                    distance for distance in target_distances
                    if distance <= self.config.sensor_radius
                ]
                if in_range:
                    flat_observable[flat_index] = True
                    observable_target_distances[flat_index] = min(in_range)
                else:
                    # Target truth is defined by its physical point, not by whether the
                    # centre of its grid cell happens to enter the footprint.
                    flat_observable[flat_index] = False

            selected = np.flatnonzero(flat_observable)
            if selected.size == 0:
                continue
            evidence_distance = flat_distance[selected].copy()
            has_target = np.zeros(selected.size, dtype=bool)
            selected_lookup = {
                int(flat_index): index for index, flat_index in enumerate(selected)
            }
            for flat_index, target_distance in observable_target_distances.items():
                local_index = selected_lookup.get(flat_index)
                if local_index is not None:
                    has_target[local_index] = True
                    evidence_distance[local_index] = target_distance

            weights = np.square(
                1.0 - np.clip(evidence_distance / self.config.sensor_radius, 0.0, 1.0)
            ).astype(np.float32)
            observation_probability = np.where(
                has_target,
                self.config.detection_probability,
                self.config.false_alarm_probability,
            )
            positive_evidence = self._rng.random(selected.size) < observation_probability
            self.detected_flags[agent] = self._sample_detection_report(
                has_target_in_range=bool(observable_target_distances)
            )

            positive_flat = self.positive_counts.ravel()
            negative_flat = self.negative_counts.ravel()
            belief_flat = self.shared_belief.ravel()
            positive_flat[selected] += weights * positive_evidence
            negative_flat[selected] += weights * ~positive_evidence

            p = float(np.clip(self.config.detection_probability, 1e-6, 1.0 - 1e-6))
            q = float(np.clip(self.config.false_alarm_probability, 1e-6, 1.0 - 1e-6))
            inverse_log_odds = (
                positive_flat[selected] * math.log(q / p)
                + negative_flat[selected] * math.log((1.0 - q) / (1.0 - p))
            )
            probabilities = 1.0 / (1.0 + np.exp(np.clip(inverse_log_odds, -60.0, 60.0)))
            belief_flat[selected] = np.clip(
                probabilities,
                self.config.min_belief_probability,
                self.config.max_belief_probability,
            )

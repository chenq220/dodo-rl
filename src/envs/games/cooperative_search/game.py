"""Mock continuous multi-UAV cooperative-search simulation.

World coordinates are kilometres.  Policy actions choose heading only; the
environment converts the configured physical speed and decision interval into
the distance travelled by the mock.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pygame

from common.exception import EnvironmentException
from config import ThreatRoamingConfig
from envs.games.cooperative_search.threat_roaming import ThreatRoaming

NUM_HEADINGS = 12
HEADING_STEP_DEGREES = 360.0 / NUM_HEADINGS


@dataclass(frozen=True)
class FlightCommand:
    heading: int

    @property
    def degrees(self) -> float:
        return (self.heading % NUM_HEADINGS) * HEADING_STEP_DEGREES


@dataclass(frozen=True)
class CooperativeSearchConfig:
    map_width: int = 300
    map_height: int = 300
    belief_grid_width: int = 100
    belief_grid_height: int = 100
    num_drones: int = 4
    num_dynamic_threats: int = 3
    num_targets: int = 10
    sensor_radius: float = 40.0
    threat_sensor_radius: float = 30.0
    threat_warning_radius: float = 45.0
    communication_radius: float = 100.0
    initial_separation: float = 20.0
    initial_formation_side_km: float = 30.0
    initial_threat_separation: float = 60.0
    detection_probability: float = 0.8
    false_alarm_probability: float = 0.2
    detection_threshold: float = 0.90
    min_belief_probability: float = 0.10
    max_belief_probability: float = 0.90
    max_turn_degrees: float = 30.0
    fixed_speed_mps: float = 100.0
    threat_speed_mps: float = 80.0
    step_interval_secs: float = 30.0
    max_steps: int = 500
    fps: int = 20
    sidebar_width: int = 300
    # 仅用于威胁专训/调试渲染；正式搜索场景必须保持关闭，避免泄露目标真值。
    show_target_truth: bool = False
    # 活动圈、航点和引导线只用于威胁专训/调试渲染。
    show_threat_roaming: bool = False
    threat_roaming: ThreatRoamingConfig | None = None

    def __post_init__(self) -> None:
        if min(self.map_width, self.map_height) < 2:
            raise EnvironmentException(detail="map dimensions must both be at least 2")
        if min(self.belief_grid_width, self.belief_grid_height) < 2:
            raise EnvironmentException(detail="belief grid dimensions must both be at least 2")
        if (
            self.num_drones < 0
            or self.num_targets < 1
            or self.num_dynamic_threats < 0
            or self.num_drones + self.num_dynamic_threats < 1
        ):
            raise EnvironmentException(detail="invalid drone, threat, or target count")
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
        if not 0.0 < self.initial_formation_side_km <= min(
            self.map_width, self.map_height
        ):
            raise EnvironmentException(
                detail="initial formation side must fit inside the map"
            )
        if self.num_drones > 4:
            raise EnvironmentException(
                detail="fixed corner formation supports at most four drones"
            )
        if self.num_drones > 1 and self.initial_formation_side_km < self.initial_separation:
            raise EnvironmentException(
                detail="initial formation side must cover initial separation"
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

    @property
    def movement_distance_km(self) -> float:
        return self.fixed_speed_mps * self.step_interval_secs / 1000.0

    @property
    def threat_movement_distance_km(self) -> float:
        return self.threat_speed_mps * self.step_interval_secs / 1000.0


class CooperativeSearchGame:
    metadata = {"name": "cooperative_search_shared_belief_v2", "render_modes": ["human", "rgb_array"]}
    _RENDER_MAP_PIXELS = 500
    _COLORS = ((52, 152, 219), (155, 89, 182), (26, 188, 156), (241, 196, 15))
    _UNEXPLORED_COLOR = (68, 72, 76)
    _BELIEF_COLOR_STOPS = np.asarray((0.0, 0.2, 0.8, 1.0), dtype=np.float32)
    _BELIEF_COLORS = np.asarray(
        (
            (19.0, 13.0, 104.0),  # #130d68 at 0% and 20%
            (19.0, 13.0, 104.0),
            (59.0, 148.0, 160.0),  # #3b94a0 at 80%
            (60.0, 211.0, 62.0),   # #3cd33e at 100%
        ),
        dtype=np.float32,
    )

    def __init__(self, config: CooperativeSearchConfig | None = None,
                 render_mode: Literal["human", "rgb_array"] | None = None) -> None:
        if render_mode not in (None, "human", "rgb_array"):
            raise EnvironmentException(detail=f"unsupported render_mode: {render_mode}")
        self.config = config or CooperativeSearchConfig()
        self.render_mode = render_mode
        self.drone_agents = [f"drone_{index}" for index in range(self.config.num_drones)]
        self.threat_agents = [
            f"dynamic_threat_{index}" for index in range(self.config.num_dynamic_threats)
        ]
        self.possible_agents = self.drone_agents + self.threat_agents
        # 课程一仍保留威胁Agent槽位，只关闭其运动、探测和渲染，
        # 从而保持两个阶段的观察、集中状态和Mixer维度一致。
        self.dynamic_threats_enabled = True
        self._rng = np.random.default_rng()
        self.threat_roaming = self._build_threat_roaming()
        grid_y, grid_x = np.indices(self.grid_shape)
        self._grid_world_x = (
            (grid_x + 0.5) * self.config.map_width / self.config.belief_grid_width
        )
        self._grid_world_y = (
            (grid_y + 0.5) * self.config.map_height / self.config.belief_grid_height
        )
        self._screen = self._surface = self._clock = self._font = self._small_font = None
        self.show_ranges = True
        self.reset()

    def _build_threat_roaming(self):
        if self.config.threat_roaming is None:
            return None
        return ThreatRoaming(self.config.threat_roaming, self.config.map_width, self.config.map_height)

    def set_dynamic_threats_enabled(self, enabled: bool) -> None:
        self.dynamic_threats_enabled = bool(enabled)
        if not self.dynamic_threats_enabled:
            self.threat_detected_flags = {agent: False for agent in self.threat_agents}
            self.drones_detected_by_threat = {agent: False for agent in self.drone_agents}

    @property
    def width(self) -> int:
        return self._RENDER_MAP_PIXELS + self.config.sidebar_width

    @property
    def height(self) -> int:
        return self._RENDER_MAP_PIXELS

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.config.belief_grid_height, self.config.belief_grid_width

    def action_space(self, agent: str) -> Any:
        if agent not in self.possible_agents:
            raise EnvironmentException(detail=f"unknown agent in action_space: {agent}")
        try:
            from gymnasium.spaces import Discrete
            return Discrete(NUM_HEADINGS)
        except ImportError:
            return range(NUM_HEADINGS)

    def is_command_available(self, agent: str, command: FlightCommand) -> bool:
        if agent not in self.possible_agents:
            raise EnvironmentException(detail=f"unknown agent in command mask: {agent}")
        return command.heading in self.available_headings(agent)

    def available_headings(self, agent: str) -> list[int]:
        """Return a non-empty mask whose normal actions retain a boundary escape path."""
        if agent not in self.possible_agents:
            raise EnvironmentException(detail=f"unknown agent in available_headings: {agent}")
        previous = self.headings[agent]
        position = self._agent_position(agent)
        movement_distance = self._movement_distance_km(agent)
        bounds = self.flight_bounds(agent)
        turn_legal = [
            heading for heading in range(NUM_HEADINGS)
            if self._turn_allowed(previous, heading)
        ]
        recoverable = []
        immediate = []
        current_clearance = self._boundary_clearance(position, bounds)
        for heading in turn_legal:
            candidate = self._predict_position(
                position, FlightCommand(heading), movement_distance
            )
            if not self._inside_bounds(candidate, bounds):
                continue
            if (
                current_clearance < movement_distance
                and self._boundary_clearance(candidate, bounds) <= current_clearance + 1e-9
            ):
                continue
            immediate.append(heading)
            if self._has_boundary_recovery(candidate, heading, movement_distance, bounds):
                recoverable.append(heading)
        if recoverable:
            return recoverable

        if immediate:
            return [max(immediate, key=lambda heading: self._recovery_score(
                self._predict_position(
                    position, FlightCommand(heading), movement_distance
                ), heading, bounds
            ))]

        # A legacy/restored state may already be too close to a wall for
        # fixed-speed motion. Expose one emergency pivot instead of an empty mask.
        return [max(turn_legal, key=lambda heading: self._recovery_score(position, heading, bounds))]

    def available_commands(self, agent: str) -> list[FlightCommand]:
        return [FlightCommand(heading) for heading in self.available_headings(agent)]

    def _has_boundary_recovery(
        self, position: tuple[float, float], heading: int, movement_distance: float,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> bool:
        """Check that one continuous max-rate turn can point back into the map."""
        bounds = bounds if bounds is not None else self._map_bounds()
        if self._heading_points_to_center(position, heading, bounds):
            return True
        for turn_delta in (-1, 1):
            candidate = position
            candidate_heading = heading
            for _ in range(NUM_HEADINGS // 2):
                candidate_heading = (candidate_heading + turn_delta) % NUM_HEADINGS
                candidate = self._predict_position(
                    candidate, FlightCommand(candidate_heading), movement_distance
                )
                if not self._inside_bounds(candidate, bounds):
                    break
                if self._heading_points_to_center(candidate, candidate_heading, bounds):
                    return True
        return False

    def _heading_points_to_center(
        self, position: tuple[float, float], heading: int,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> bool:
        angle = math.radians(heading * HEADING_STEP_DEGREES)
        x_min, x_max, y_min, y_max = bounds if bounds is not None else self._map_bounds()
        toward_center = (
            (x_min + x_max) / 2.0 - position[0],
            (y_min + y_max) / 2.0 - position[1],
        )
        return (
            math.cos(angle) * toward_center[0]
            + math.sin(angle) * toward_center[1]
        ) > 1e-9

    def _recovery_score(
        self, position: tuple[float, float], heading: int,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> tuple[float, float]:
        angle = math.radians(heading * HEADING_STEP_DEGREES)
        bounds = bounds if bounds is not None else self._map_bounds()
        x_min, x_max, y_min, y_max = bounds
        toward_center = (
            (x_min + x_max) / 2.0 - position[0],
            (y_min + y_max) / 2.0 - position[1],
        )
        inward = math.cos(angle) * toward_center[0] + math.sin(angle) * toward_center[1]
        clearance = self._boundary_clearance(position, bounds)
        return inward, clearance

    def _boundary_clearance(
        self, position: tuple[float, float],
        bounds: tuple[float, float, float, float] | None = None,
    ) -> float:
        x_min, x_max, y_min, y_max = bounds if bounds is not None else self._map_bounds()
        return min(
            position[0] - x_min, x_max - position[0],
            position[1] - y_min, y_max - position[1],
        )

    def _predict_position(
        self, position: tuple[float, float], command: FlightCommand,
        movement_distance: float | None = None,
    ) -> tuple[float, float]:
        angle = math.radians(command.degrees)
        distance = (
            self.config.movement_distance_km
            if movement_distance is None else movement_distance
        )
        return (
            position[0] + distance * math.cos(angle),
            position[1] + distance * math.sin(angle),
        )

    def _agent_position(self, agent: str) -> tuple[float, float]:
        if agent in self.drone_positions:
            return self.drone_positions[agent]
        if agent in self.threat_positions:
            return self.threat_positions[agent]
        raise EnvironmentException(detail=f"unknown cooperative-search agent: {agent}")

    def _set_agent_position(self, agent: str, position: tuple[float, float]) -> None:
        if agent in self.drone_positions:
            self.drone_positions[agent] = position
            return
        if agent in self.threat_positions:
            self.threat_positions[agent] = position
            return
        raise EnvironmentException(detail=f"unknown cooperative-search agent: {agent}")

    def _speed_mps(self, agent: str) -> float:
        if agent in self.threat_agents and not self.dynamic_threats_enabled:
            return 0.0
        return (
            self.config.threat_speed_mps
            if agent in self.threat_agents else self.config.fixed_speed_mps
        )

    def _movement_distance_km(self, agent: str) -> float:
        return self._speed_mps(agent) * self.config.step_interval_secs / 1000.0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.step_count = 0
        self.running = True
        self.total_reward = 0.0
        drone_points = self._sample_drone_cluster()
        self.drone_positions = dict(zip(self.drone_agents, drone_points, strict=True))
        threat_points = self._sample_separated_points(
            self.config.num_dynamic_threats,
            existing=drone_points,
            min_separation=self.config.initial_threat_separation,
        )
        self.threat_positions = dict(zip(self.threat_agents, threat_points, strict=True))
        occupied_points = drone_points + (
            threat_points if self.dynamic_threats_enabled else []
        )
        self._target_positions = self._sample_separated_points(
            self.config.num_targets, existing=occupied_points
        )
        self.headings: dict[str, int | None] = {agent: None for agent in self.possible_agents}
        self.speeds_mps: dict[str, float] = {agent: 0.0 for agent in self.possible_agents}
        # These arrays hold distance-weighted evidence rather than raw scan counts.
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
        # 威胁专训环境复用本Game但不创建无人机；只有场景原本配置了无人机时，
        # “无人机清空”才代表任务失败。
        all_drones_dead = bool(self.drone_agents) and not self.drone_positions
        truncated = (
            all_drones_dead or self.step_count >= self.config.max_steps
        ) and not success
        acting_agents = list(self.agents)
        terminations = {
            agent: success or (agent in eaten_drones and not all_drones_dead)
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
            self.agents = [agent for agent in acting_agents if agent not in eaten_drones]
        return observations, rewards, terminations, truncations, infos

    def _sample_drone_cluster(self) -> list[tuple[float, float]]:
        """Place UAVs at fixed vertices of a square anchored at the map's top-left."""
        side = float(self.config.initial_formation_side_km)
        vertices = [
            (0.0, 0.0),
            (side, 0.0),
            (0.0, side),
            (side, side),
        ]
        return vertices[:self.config.num_drones]

    def _sample_separated_points(
        self,
        count: int,
        existing: list[tuple[float, float]] | None = None,
        min_separation: float | None = None,
    ) -> list[tuple[float, float]]:
        points = list(existing or [])
        separation = (
            self.config.initial_separation
            if min_separation is None
            else float(min_separation)
        )
        first_new = len(points)
        for _ in range(count):
            for _attempt in range(20_000):
                candidate = (
                    float(self._rng.uniform(0, self.config.map_width)),
                    float(self._rng.uniform(0, self.config.map_height)),
                )
                if all(
                    self._distance(candidate, point) >= separation
                    for point in points
                ):
                    points.append(candidate)
                    break
            else:
                raise EnvironmentException(
                    detail="unable to place entities with configured separation"
                )
        return points[first_new:]

    def _mark_visits(self, positions) -> None:
        for position in positions:
            gx, gy = self._world_to_grid(position)
            self.visited_counts[gy, gx] += 1

    def _observe_shared_belief(self) -> None:
        target_cells: dict[int, list[tuple[float, float]]] = {}
        for target in self._target_positions:
            gx, gy = self._world_to_grid(target)
            target_cells.setdefault(gy * self.config.belief_grid_width + gx, []).append(target)
        self.detected_flags = {agent: False for agent in self.possible_agents}
        for agent, position in self.drone_positions.items():
            distance_grid = np.hypot(
                self._grid_world_x - position[0], self._grid_world_y - position[1]
            )
            observable = distance_grid <= self.config.sensor_radius
            flat_observable = observable.ravel()
            flat_distance = distance_grid.ravel()
            observable_target_distances: dict[int, float] = {}
            for flat_index, targets in target_cells.items():
                target_distances = [self._distance(position, target) for target in targets]
                in_range = [distance for distance in target_distances if distance <= self.config.sensor_radius]
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
            selected_lookup = {int(flat_index): index for index, flat_index in enumerate(selected)}
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
            # The public boolean is one sensor-level report per drone and step.
            # It must not be the OR of all per-cell evidence: with roughly 200
            # observed cells and q=0.2 that would be true almost every step even
            # over an empty footprint.  Per-cell samples below still drive the
            # shared Bayesian belief map independently.
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

    def _sample_detection_report(self, *, has_target_in_range: bool) -> bool:
        """Sample one public detection/false-alarm report for a sensor footprint."""
        probability = (
            self.config.detection_probability
            if has_target_in_range
            else self.config.false_alarm_probability
        )
        return bool(self._rng.random() < probability)

    def _update_threat_detections(self) -> None:
        self.threat_detected_flags = {agent: False for agent in self.threat_agents}
        self.drones_detected_by_threat = {agent: False for agent in self.drone_agents}
        if not self.dynamic_threats_enabled:
            return
        active_agents = set(self.agents)
        for threat, threat_position in self.threat_positions.items():
            for drone, drone_position in self.drone_positions.items():
                # 完成区域任务的无人机保留位置供渲染，但已经退出本局交互。
                if drone not in active_agents:
                    continue
                if self._distance(threat_position, drone_position) <= self.config.threat_sensor_radius:
                    self.threat_detected_flags[threat] = True
                    self.drones_detected_by_threat[drone] = True

    def _eat_detected_drones(self) -> set[str]:
        active_agents = set(self.agents)
        eaten = {
            drone for drone, detected in self.drones_detected_by_threat.items()
            if detected and drone in self.drone_positions and drone in active_agents
        }
        for drone in eaten:
            self.dead_drone_positions[drone] = self.drone_positions.pop(drone)
        return eaten

    def _probability_from_counts(self, positive: float, negative: float) -> float:
        """Fuse weighted evidence while avoiding visually meaningless certainty."""
        p = float(np.clip(self.config.detection_probability, 1e-6, 1.0 - 1e-6))
        q = float(np.clip(self.config.false_alarm_probability, 1e-6, 1.0 - 1e-6))
        inverse_log_odds = (
            positive * math.log(q / p)
            + negative * math.log((1.0 - q) / (1.0 - p))
        )
        if inverse_log_odds >= 0.0:
            exp_value = math.exp(-inverse_log_odds)
            probability = exp_value / (1.0 + exp_value)
        else:
            probability = 1.0 / (1.0 + math.exp(inverse_log_odds))
        return float(np.clip(
            probability,
            self.config.min_belief_probability,
            self.config.max_belief_probability,
        ))

    def _sensor_evidence_weight(self, distance: float) -> float:
        """Smoothly reduce evidence to zero at the sensor footprint boundary."""
        normalized = float(np.clip(distance / self.config.sensor_radius, 0.0, 1.0))
        return (1.0 - normalized) ** 2

    def _grid_cell_center(self, gx: int, gy: int) -> tuple[float, float]:
        return (
            (gx + 0.5) * self.config.map_width / self.config.belief_grid_width,
            (gy + 0.5) * self.config.map_height / self.config.belief_grid_height,
        )

    def _sensor_grid_mask(self, position: tuple[float, float]) -> np.ndarray:
        return np.hypot(
            self._grid_world_x - position[0], self._grid_world_y - position[1]
        ) <= self.config.sensor_radius

    def _observations(self):
        visited = (self.visited_counts > 0).astype(np.float32)
        belief = self.shared_belief.copy()
        observations = {
            agent: {
                "position": np.array(
                    [position[0] / self.config.map_width, position[1] / self.config.map_height],
                    dtype=np.float32,
                ),
                "belief": belief,
                "visited": visited,
                "detected": self.detected_flags[agent],
            }
            for agent, position in self.drone_positions.items()
        }
        observations.update({
            agent: {
                "position": (
                    np.asarray(
                        [position[0] / self.config.map_width,
                         position[1] / self.config.map_height],
                        dtype=np.float32,
                    )
                    if self.dynamic_threats_enabled
                    else np.zeros(2, dtype=np.float32)
                ),
                "detected_drone": (
                    self.dynamic_threats_enabled
                    and self.threat_detected_flags[agent]
                ),
            }
            for agent, position in self.threat_positions.items()
        })
        return observations

    def _agent_info(
        self, agent: str, invalid: bool, predicted_position,
        boundary_violation: bool = False,
    ) -> dict[str, Any]:
        is_threat = agent in self.threat_agents
        position = (
            self.dead_drone_positions[agent]
            if agent in self.dead_drone_positions else self._agent_position(agent)
        )
        return {
            "position": position,
            "heading": self.headings[agent],
            "speed_mps": self.speeds_mps[agent],
            "detected_target": bool(self.detected_flags.get(agent, False)),
            "detected_drone": bool(self.threat_detected_flags.get(agent, False)),
            "detected_by_threat": bool(self.drones_detected_by_threat.get(agent, False)),
            "is_dynamic_threat": is_threat,
            "alive": agent not in self.dead_drone_positions,
            "invalid_action": invalid,
            "boundary_action_rejected": boundary_violation,
            "predicted_position": predicted_position,
            "coverage": float(np.count_nonzero(self.visited_counts) / self.visited_counts.size),
            "step": self.step_count,
        }

    def global_belief(self) -> np.ndarray:
        return self.shared_belief.copy()

    def _target_probabilities(self) -> np.ndarray:
        return np.asarray([
            self.shared_belief[gy, gx]
            for gx, gy in map(self._world_to_grid, self._target_positions)
        ], dtype=np.float32)

    def _target_grid_cells(self) -> set[tuple[int, int]]:
        """仅供环境内部奖励与可视化判定使用，不进入公开态势。"""
        return set(map(self._world_to_grid, self._target_positions))

    def all_targets_confirmed(self) -> bool:
        return bool(np.all(self._target_probabilities() >= self.config.detection_threshold))

    def all_targets_fully_identified(self) -> bool:
        """全部目标达到概率上限时完成。"""
        threshold = self.config.max_belief_probability - 1e-6
        return bool(np.all(self._target_probabilities() >= threshold))

    def confirmed_target_count(self) -> int:
        """Return only the aggregate number of targets at or above the confidence threshold."""
        return int(np.count_nonzero(
            self._target_probabilities() >= self.config.detection_threshold
        ))

    def _world_to_grid(self, position: tuple[float, float]) -> tuple[int, int]:
        x = min(
            self.config.belief_grid_width - 1,
            max(0, int(position[0] / self.config.map_width * self.config.belief_grid_width)),
        )
        y = min(
            self.config.belief_grid_height - 1,
            max(0, int(position[1] / self.config.map_height * self.config.belief_grid_height)),
        )
        return x, y

    def _turn_allowed(self, previous: int | None, current: int) -> bool:
        if previous is None:
            return True
        bins = abs(current - previous) % NUM_HEADINGS
        bins = min(bins, NUM_HEADINGS - bins)
        return bins * HEADING_STEP_DEGREES <= self.config.max_turn_degrees + 1e-9

    def _in_bounds(self, position: tuple[float, float]) -> bool:
        return self._inside_bounds(position, self._map_bounds())

    def _map_bounds(self) -> tuple[float, float, float, float]:
        return 0.0, float(self.config.map_width), 0.0, float(self.config.map_height)

    def flight_bounds(self, agent: str) -> tuple[float, float, float, float]:
        return self._map_bounds()

    @staticmethod
    def _inside_bounds(position, bounds) -> bool:
        x_min, x_max, y_min, y_max = bounds
        return x_min <= position[0] <= x_max and y_min <= position[1] <= y_max

    def _world_to_render(self, position: tuple[float, float]) -> tuple[int, int]:
        return (
            round(position[0] / self.config.map_width * self._RENDER_MAP_PIXELS),
            round(position[1] / self.config.map_height * self._RENDER_MAP_PIXELS),
        )

    def _distance_to_render(self, distance_km: float) -> int:
        return round(distance_km / self.config.map_width * self._RENDER_MAP_PIXELS)

    @staticmethod
    def _distance(first, second) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    def render(self):
        if self.render_mode is None:
            return None
        self._ensure_pygame()
        self._draw()
        if self.render_mode == "human":
            # macOS Cocoa窗口必须持续泵送事件，否则评估进程虽然继续运行，
            # 窗口可能停留在后台或显示为无响应。
            self.handle_events()
            self._screen.blit(self._surface, (0, 0))
            pygame.display.flip()
            self._clock.tick(self.config.fps)
            return None
        return np.transpose(pygame.surfarray.array3d(self._surface), (1, 0, 2))

    def handle_events(self) -> None:
        if self.render_mode != "human":
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                self.show_ranges = not self.show_ranges

    def _ensure_pygame(self) -> None:
        if not pygame.get_init():
            pygame.init()
        if self._surface is None:
            self._surface = pygame.Surface((self.width, self.height))
            self._font = pygame.font.Font(None, 28)
            self._small_font = pygame.font.Font(None, 21)
        if self.render_mode == "human" and self._screen is None:
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Shared-Belief Cooperative Search")
            self._clock = pygame.time.Clock()

    def _draw_threat_roaming(self) -> None:
        if (self.config.show_threat_roaming and self.threat_roaming is not None
                and self.dynamic_threats_enabled):
            roaming = self.threat_roaming
            for index, agent in enumerate(self.threat_agents):
                color = self._COLORS[index % len(self._COLORS)]
                pygame.draw.circle(
                    self._surface, color, self._world_to_render(roaming.centers[agent]),
                    self._distance_to_render(roaming.config.circle_radius_km), 1,
                )
                start = self._world_to_render(self.threat_positions[agent])
                goal = self._world_to_render(roaming.waypoints[agent])
                pygame.draw.line(self._surface, color, start, goal, 1)
                pygame.draw.circle(self._surface, color, goal, 3, 1)

    def _draw(self) -> None:
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
        for agent in active_drones:
            center = self._world_to_render(self.drone_positions[agent])
            heading = self.headings[agent] or 0
            angle = math.radians(heading * HEADING_STEP_DEGREES)
            direction = (math.cos(angle), math.sin(angle))
            side = (-direction[1], direction[0])
            points = [
                (center[0] + direction[0] * 12, center[1] + direction[1] * 12),
                (center[0] - direction[0] * 8 + side[0] * 7, center[1] - direction[1] * 8 + side[1] * 7),
                (center[0] - direction[0] * 8 - side[0] * 7, center[1] - direction[1] * 8 - side[1] * 7),
            ]
            pygame.draw.polygon(self._surface, (60, 211, 62), points)
            pygame.draw.polygon(self._surface, (245, 249, 250), points, 2)
        for agent in self.threat_agents if self.dynamic_threats_enabled else ():
            center = self._world_to_render(self.threat_positions[agent])
            pygame.draw.circle(self._surface, (220, 38, 38), center, 8)
            pygame.draw.circle(self._surface, (255, 225, 225), center, 8, 2)
        x = self._RENDER_MAP_PIXELS
        pygame.draw.rect(self._surface, (24, 35, 48), (x, 0, self.config.sidebar_width, self.height))
        lines = [
            "COOPERATIVE SEARCH",
            f"{self.config.map_width} x {self.config.map_height} km",
            f"Step       {self.step_count} / {self.config.max_steps}",
            f"Coverage   {np.count_nonzero(observed)/observed.size:.1%}",
            f"Detected   {self.confirmed_target_count()} / {self.config.num_targets}",
            f"Reward     {self.total_reward:.2f}",
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
            font = self._font if line == "COOPERATIVE SEARCH" else self._small_font
            self._surface.blit(font.render(line, True, (225, 235, 245)), (x + 20, top))
            top += 34 if font is self._font else 27

    @staticmethod
    def _belief_color(probability: float) -> tuple[int, int, int]:
        """Map belief through the configured blue-cyan-green CSS gradient."""
        return tuple(CooperativeSearchGame._belief_rgb(np.asarray(probability)).tolist())

    @staticmethod
    def _belief_rgb(probability: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
        channels = [
            np.interp(
                values,
                CooperativeSearchGame._BELIEF_COLOR_STOPS,
                CooperativeSearchGame._BELIEF_COLORS[:, channel],
            )
            for channel in range(3)
        ]
        return np.rint(np.stack(channels, axis=-1)).astype(np.uint8)

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
        self._screen = self._surface = self._clock = None

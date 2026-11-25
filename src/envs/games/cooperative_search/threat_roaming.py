"""Shared game component for per-entity roaming goals and rewards."""
from __future__ import annotations

import math
import numpy as np

from common.exception import EnvironmentException
from config import ThreatRoamingConfig


class ThreatRoaming:
    REWARD_KEYS = ("waypoint_progress", "waypoint_arrival", "circle_outside", "peer_clearance", "peer_danger")

    def __init__(self, config: ThreatRoamingConfig, map_width: float, map_height: float,
                 *, region_rows: int = 1, region_cols: int = 1):
        self.config = config
        self.map_width = map_width
        self.map_height = map_height
        self.region_rows = region_rows
        self.region_cols = region_cols
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (region_rows, region_cols)):
            raise EnvironmentException(detail="circle region layout must contain positive integers")
        self.centers: dict[str, tuple[float, float]] = {}
        self.center_slots: dict[str, tuple[float, float, float, float]] = {}
        finite_positive = (config.circle_radius_km, config.circle_center_max_shift_km, config.waypoint_reached_km,
                           config.waypoint_min_distance_km, config.waypoint_max_distance_km, config.perception_radius_km,
                           config.separation_distance_km, config.danger_distance_km)
        if any(not math.isfinite(v) or v <= 0 for v in finite_positive):
            raise EnvironmentException(detail="roaming distances must be finite and positive")
        if (not math.isfinite(config.circle_center_min_shift_km)
                or not 0 <= config.circle_center_min_shift_km < config.circle_center_max_shift_km):
            raise EnvironmentException(detail="require 0 <= minimum circle shift < maximum circle shift")
        if config.circle_radius_km > min(map_width, map_height) / 2:
            raise EnvironmentException(detail="roaming radius must not exceed half the shorter map dimension")
        if not config.waypoint_reached_km < config.waypoint_min_distance_km < config.waypoint_max_distance_km <= config.circle_radius_km:
            raise EnvironmentException(detail="require reached distance < waypoint minimum < waypoint maximum <= circle radius")
        if not config.danger_distance_km < config.separation_distance_km <= config.perception_radius_km:
            raise EnvironmentException(detail="require danger distance < separation distance <= perception radius")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in
               (config.radial_bins, config.angular_bins, config.waypoint_timeout_steps,
                config.waypoint_refresh_min_steps, config.waypoint_refresh_max_steps)):
            raise EnvironmentException(detail="roaming bins and timeout must be positive integers")
        if config.waypoint_refresh_min_steps > config.waypoint_refresh_max_steps:
            raise EnvironmentException(detail="minimum waypoint refresh interval exceeds maximum")
        if config.representation_output_dim != config.radial_bins * config.angular_bins:
            raise EnvironmentException(detail="polar representation must retain every grid cell")
        if any(not math.isfinite(v) or v < 0 for v in (config.progress_reward_weight,
               config.arrival_reward, config.circle_penalty, config.separation_penalty, config.danger_penalty)):
            raise EnvironmentException(detail="roaming reward weights must be finite and nonnegative")
        self.waypoints: dict[str, tuple[float, float]] = {}
        self.ages: dict[str, int] = {}
        self.layout_age = 0
        self.layout_refresh_after = 0
        self.completed: dict[str, int] = {}
        self.breakdowns: dict[str, dict[str, float]] = {}
        self.episode_breakdowns: dict[str, dict[str, float]] = {}
        self.metrics: dict[str, dict[str, float]] = {}

    def _circle_slots(self, count):
        """Equal-area strata; fitting each entire disk guarantees pairwise clearance."""
        regions = self.region_rows * self.region_cols
        if count < regions or count % regions:
            raise EnvironmentException(detail="circles must split evenly across the region layout")
        per_region = count // regions
        width = self.map_width / self.region_cols
        height = self.map_height / self.region_rows
        diameter = 2 * self.config.circle_radius_km
        layouts = [(abs(math.log((width / (per_region // rows)) / (height / rows))), rows, per_region // rows)
                   for rows in range(1, per_region + 1) if per_region % rows == 0
                   and width / (per_region // rows) >= diameter and height / rows >= diameter]
        if not layouts:
            raise EnvironmentException(detail="cannot fit non-overlapping roaming circles; reduce circle radius or threat count")
        _, rows, cols = min(layouts)
        slots = []
        for region_row in range(self.region_rows):
            for region_col in range(self.region_cols):
                for row in range(rows):
                    for col in range(cols):
                        slots.append((region_col * width + col * width / cols,
                                      region_col * width + (col + 1) * width / cols,
                                      region_row * height + row * height / rows,
                                      region_row * height + (row + 1) * height / rows))
        return slots

    def _new_center(self, slot, rng):
        radius = self.config.circle_radius_km
        x0, x1, y0, y1 = slot
        return (float(rng.uniform(x0 + radius, x1 - radius)),
                float(rng.uniform(y0 + radius, y1 - radius)))

    def _refresh_centers(self, rng):
        """Atomically reassign a complete stratified layout across different slots."""
        agents = list(self.centers)
        slots = self._circle_slots(len(agents))
        c = self.config
        for _ in range(1_000):
            sampled = [self._new_center(slot, rng) for slot in slots]
            edges = {}
            for aid in agents:
                options = [index for index, center in enumerate(sampled)
                           if slots[index] != self.center_slots[aid]
                           and c.circle_center_min_shift_km <= math.dist(center, self.centers[aid]) <= c.circle_center_max_shift_km]
                rng.shuffle(options)
                edges[aid] = options
            assigned = {}

            def augment(aid, seen):
                for index in edges[aid]:
                    if index in seen:
                        continue
                    seen.add(index)
                    if index not in assigned or augment(assigned[index], seen):
                        assigned[index] = aid
                        return True
                return False

            order = sorted(agents, key=lambda aid: len(edges[aid]))
            if all(augment(aid, set()) for aid in order):
                # No partial assignment may escape: spatial coverage is a joint invariant.
                self.centers = {aid: sampled[index] for index, aid in assigned.items()}
                self.center_slots = {aid: slots[index] for index, aid in assigned.items()}
                return
        raise EnvironmentException(detail="cannot refresh stratified circles within the configured shift range")

    def _reset_centers(self, positions, rng):
        slots = self._circle_slots(len(positions))
        sampled = [self._new_center(slot, rng) for slot in slots]
        self.centers = {}
        self.center_slots = {}
        per_region = len(positions) // (self.region_rows * self.region_cols)
        agents = list(positions)
        for start in range(0, len(agents), per_region):
            members = agents[start:start + per_region]
            indexes = range(start, start + per_region)
            candidates = sorted((math.dist(positions[aid], sampled[index]), aid, index)
                                for aid in members for index in indexes)
            used = set()
            for _, aid, index in candidates:
                if aid in self.centers or index in used:
                    continue
                self.centers[aid] = sampled[index]
                self.center_slots[aid] = slots[index]
                used.add(index)

    def _new_waypoint(self, center, rng):
        """Uniform annulus about the assigned circle center, independent of entity position."""
        for _ in range(10_000):
            radius = self.config.waypoint_max_distance_km * math.sqrt(float(rng.random()))
            angle = float(rng.uniform(0, 2 * math.pi))
            point = (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
            if (0 <= point[0] <= self.map_width and 0 <= point[1] <= self.map_height
                    and math.dist(center, point) >= self.config.waypoint_min_distance_km):
                return point
        raise EnvironmentException(detail="unable to sample roaming waypoint")

    def _new_refresh_interval(self, rng):
        # Preserve timeout as the hard cap, including short diagnostic episodes.
        low = min(self.config.waypoint_refresh_min_steps, self.config.waypoint_timeout_steps)
        high = min(self.config.waypoint_refresh_max_steps, self.config.waypoint_timeout_steps)
        return int(rng.integers(low, high + 1))

    def reset(self, positions, rng):
        self._reset_centers(positions, rng)
        self.waypoints = {a: self._new_waypoint(self.centers[a], rng) for a in positions}
        self.ages = dict.fromkeys(positions, 0)
        self.layout_age = 0
        self.layout_refresh_after = self._new_refresh_interval(rng)
        self.completed = dict.fromkeys(positions, 0)
        self.breakdowns = {a: dict.fromkeys(self.REWARD_KEYS, 0.0) for a in positions}
        self.episode_breakdowns = {a: dict.fromkeys(self.REWARD_KEYS, 0.0) for a in positions}
        self.metrics = {a: self.position_metrics(a, positions) for a in positions}

    def position_metrics(self, agent, positions):
        near = min((math.dist(positions[agent], p) for a, p in positions.items() if a != agent), default=math.inf)
        return {"inside_roaming_circle": float(math.dist(positions[agent], self.centers[agent]) <= self.config.circle_radius_km),
                "nearest_peer_distance_km": near,
                "dangerous_peer_proximity": float(near < self.config.danger_distance_km),
                "waypoints_completed": float(self.completed.get(agent, 0))}

    def step(self, previous, current, movement, rng):
        c = self.config
        if movement <= 0:
            raise EnvironmentException(detail="roaming requires positive movement distance")
        # Settle all rewards against the old goals before generating any new goal.
        renew = []
        for agent, position in current.items():
            goal = self.waypoints[agent]
            old_distance = math.dist(previous[agent], goal)
            distance = math.dist(position, goal)
            reached = distance <= c.waypoint_reached_km
            self.ages[agent] += 1
            self.completed[agent] += int(reached)
            metrics = self.position_metrics(agent, current)
            near = metrics["nearest_peer_distance_km"]
            components = {
                "waypoint_progress": c.progress_reward_weight * float(np.clip((old_distance-distance)/movement, -1, 1)),
                "waypoint_arrival": c.arrival_reward * int(reached),
                "circle_outside": -c.circle_penalty * float(np.clip((math.dist(position, self.centers[agent])-c.circle_radius_km)/c.circle_radius_km, 0, 1))**2,
                "peer_clearance": -c.separation_penalty * max(0.0, 1-near/c.separation_distance_km)**2,
                "peer_danger": -c.danger_penalty * max(0.0, 1-near/c.danger_distance_km)**2,
            }
            self.breakdowns[agent] = components
            for key, value in components.items():
                self.episode_breakdowns[agent][key] += value
            self.metrics[agent] = metrics
            if reached or self.ages[agent] >= c.waypoint_timeout_steps:
                renew.append(agent)
        self.layout_age += 1
        if self.layout_age >= self.layout_refresh_after:
            self._refresh_centers(rng)
            renew = list(current)
            self.layout_age = 0
            self.layout_refresh_after = self._new_refresh_interval(rng)
        for agent in renew:
            self.waypoints[agent] = self._new_waypoint(self.centers[agent], rng)
            self.ages[agent] = 0

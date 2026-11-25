"""Heading-relative polar peer grid shared by threat training and fixed inference."""
from __future__ import annotations

import math
import numpy as np

from config import SpatialRepresentationConfig
from envs.state.representation import SpatialGridRepresentation
from envs.games.cooperative_search.game import NUM_HEADINGS


class ThreatRoamingObservation:
    def __init__(self, config):
        self.config = config
        self.representation = SpatialGridRepresentation(SpatialRepresentationConfig(
            input_shape=(1, config.radial_bins, config.angular_bins),
            output_dim=config.representation_output_dim,
        ))
        self.observation_dim = 13 + config.representation_output_dim
        self._cache_key = None
        self._observations = {}

    @staticmethod
    def _relative(game, agent, position):
        own = game.threat_positions[agent]
        dx, dy = position[0]-own[0], position[1]-own[1]
        heading = game.headings[agent]
        angle = 0.0 if heading is None else heading * 2*math.pi/NUM_HEADINGS
        return (dx*math.cos(angle)+dy*math.sin(angle), -dx*math.sin(angle)+dy*math.cos(angle))

    def peer_grid(self, game, agent):
        c = self.config
        grid = np.zeros((1, c.radial_bins, c.angular_bins), dtype=np.float32)
        for peer, position in game.threat_positions.items():
            if peer == agent:
                continue
            x, y = self._relative(game, agent, position)
            distance = math.hypot(x, y)
            if distance >= c.perception_radius_km:
                continue
            radial = float(np.clip(distance/c.perception_radius_km*c.radial_bins-.5, 0, c.radial_bins-1))
            angular = ((math.atan2(y, x) if distance else 0.0) % (2*math.pi)) / (2*math.pi) * c.angular_bins
            r0, a0 = math.floor(radial), math.floor(angular)
            fr, fa = radial-r0, angular-a0
            strength = 1-distance/c.perception_radius_km
            for row, rw in ((r0, 1-fr), (r0+1, fr)):
                if row >= c.radial_bins or rw == 0:
                    continue
                for col, aw in ((a0 % c.angular_bins, 1-fa), ((a0+1) % c.angular_bins, fa)):
                    grid[0, row, col] = max(grid[0, row, col], strength*rw*aw)
        return grid

    def __call__(self, game, agent):
        # Repeated group inference reads the same physical world many times.
        # These observations do not consume the action-context buffer. Include
        # every input in the key so reset and externally restored states are safe.
        roaming = game.threat_roaming
        key = (id(game), game.config.map_width, game.config.map_height,
               tuple(game.threat_positions.items()),
               tuple((a, game.headings[a]) for a in game.threat_positions),
               tuple(roaming.centers.items()), tuple(roaming.waypoints.items()), tuple(roaming.ages.items()))
        if key != self._cache_key:
            self._cache_key = key
            self._observations.clear()
        if agent not in self._observations:
            self._observations[agent] = self._encode(game, agent)
        # Callers may retain or edit an observation; do not expose the cache.
        return self._observations[agent].copy()

    def _encode(self, game, agent):
        c = self.config
        x, y = game.threat_positions[agent]
        heading = game.headings[agent]
        angle = None if heading is None else heading * 2*math.pi/NUM_HEADINGS
        roaming = game.threat_roaming
        center, goal = roaming.centers[agent], roaming.waypoints[agent]
        peers = [(math.dist((x,y),p), peer, p) for peer,p in game.threat_positions.items() if peer != agent]
        diagonal = math.hypot(game.config.map_width, game.config.map_height)
        if peers:
            distance, _, peer_position = min(peers)
            px, py = self._relative(game, agent, peer_position)
            nearest = [px/diagonal, py/diagonal, distance/diagonal]
        else:
            nearest = [0.0, 0.0, 1.0]
        features = np.array([
            x/game.config.map_width, y/game.config.map_height,
            0.0 if angle is None else math.sin(angle), 0.0 if angle is None else math.cos(angle),
            (center[0]-x)/game.config.map_width, (center[1]-y)/game.config.map_height,
            c.circle_radius_km/min(game.config.map_width, game.config.map_height),
            (goal[0]-x)/game.config.map_width, (goal[1]-y)/game.config.map_height,
            roaming.ages[agent]/c.waypoint_timeout_steps, *nearest,
        ], dtype=np.float32)
        return np.concatenate((features, self.representation(self.peer_grid(game, agent))))

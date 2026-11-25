from __future__ import annotations

import gymnasium as gym
import numpy as np

from config import EnvConfig
from envs.action.model import Action
from envs.base import AgentID, ParallelEnvironment, LOG, ObsType
from envs.games.shaping_test_games.threat_region_game import ThreatRegionGame
from envs.reward.shaping import (
    region_threat_score,
    region_threat_shaping,
    region_threat_transition_score,
    region_visible_target_seek_shaping,
)
from envs.reward.utils import (
    Point,
    ShapingOutputRange,
    pbrs_shaping,
    point_in_region,
    point_to_region_boundary_distance,
    segment_intersects_region,
)
from envs.state.tests.threat_region_state import ThreatRegionState


class ThreatRegionEnv(ParallelEnvironment):
    """单智能体威胁区域环境：绕开威胁区并受南侧红点 seek 引导。"""

    OBS_DIM = 14
    TARGET_DISTANCE = 0.0
    THREAT_VERTEX_COUNT = 4

    def __init__(self, config: EnvConfig, sim_config=None, connect_sim=True, sim_manager=None, role_id=None):
        super().__init__(
            config=config,
            sim_config=sim_config,
            connect_sim=connect_sim,
            sim_manager=sim_manager,
            role_id=role_id,
        )
        assert len(self.possible_agents) == 1, (
            f"ThreatRegionEnv is single-agent, agents={self.possible_agents}"
        )

        metadata = self.config.metadata or {}
        self.render_mode = metadata.get("render_mode", "rgb_array")
        self.render_every_step = bool(metadata.get("render_every_step", False))
        self._seed = metadata.get("seed")
        self._shaping_gamma = float(metadata.get("shaping_gamma", 1.0))
        self._approach_decay_coef = float(
            metadata.get("approach_decay_coef", 1.0),
        )
        self._threat_speed = float(metadata.get("threat_speed", 0.0))
        self._output_range = ShapingOutputRange(metadata.get("output_range", "signed"))
        self._seek_output_range = ShapingOutputRange(metadata.get("seek_output_range", "unit"))
        self._max_deviation = float(metadata.get("max_deviation", float(ThreatRegionGame.WIDTH)))
        reward_weights = metadata.get("reward_weights") or {}
        self._threat_weight = float(reward_weights.get("threat", 1.0))
        self._seek_weight = float(reward_weights.get("seek", self._threat_weight / 3.0))
        self._last_action_id = 0

        self.game: ThreatRegionGame | None = None
        if not self.connect_sim:
            self.game = ThreatRegionGame(
                render_mode=None,
                threat_speed=self._threat_speed,
                approach_decay_coef=self._approach_decay_coef,
                gamma=self._shaping_gamma,
                output_range=self._output_range,
                seed=self._seed,
            )

    def state_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32)

    def state(self) -> np.ndarray:
        if self.post_state is None:
            return np.zeros(self.OBS_DIM, dtype=np.float32)
        assert isinstance(self.post_state, ThreatRegionState)
        return self._obs_from_state(self.post_state)

    def get_obs(self) -> dict[AgentID, ObsType]:
        observation = self.state()
        return {agent: observation.copy() for agent in self.possible_agents}

    def get_avail_agent_actions(self, agent_name: AgentID, init: bool = False) -> list[int]:
        return [1] * ThreatRegionGame.NUM_ACTIONS

    def reset(self, seed=None, options=None):
        if self.connect_sim:
            return super().reset(seed=seed, options=options)
        assert self.game is not None
        if options is not None:
            step_limit = options.get("step_limit", options.get("episode_step_limit", None))
            if step_limit is not None:
                self.set_episode_step_limit(step_limit)
        reset_seed = seed if seed is not None else self._seed
        self.game.reset(seed=reset_seed)
        self._last_action_id = 0
        self.pre_state = self.post_state = self._snapshot(action_id=0)
        self.step_num = 1
        self.current_actions = {}
        self.agents = self.possible_agents[:]
        obs = self.get_obs()
        infos = {agent: {} for agent in self.possible_agents}
        self.record_sample(reset=True)
        return obs, infos

    def step(self, actions):
        if self.connect_sim:
            return super().step(actions)
        assert self.game is not None
        self.pre_state = self.post_state
        self.current_actions = self.send_actions(actions)
        self.post_state = self._snapshot(action_id=self._last_action_id)
        if self.render_every_step:
            self.render()
        terminated, truncated, need_resume = self.done()
        self.rewards = self.get_reward()
        terminations = {agent: terminated for agent in self.possible_agents}
        truncations = {agent: truncated for agent in self.possible_agents}
        infos = {
            agent: {
                "need_resume": need_resume,
                "episode_step_limit": self.max_step_per_episode,
                "step_limit_reached": truncated and not terminated,
            }
            for agent in self.possible_agents
        }
        self.step_log()
        self.record_sample()
        self.step_num += 1
        observations = self._parallel_observations_after_step(terminations, truncations)
        return observations, self.rewards, terminations, truncations, infos

    def send_actions(self, actions):
        if self.connect_sim:
            return super().send_actions(actions)
        assert self.game is not None
        live_actions = {agent: action for agent, action in actions.items() if agent in self.agents}
        self._last_action_id = self._convert_action_to_id(live_actions)
        self.game.step_action(self._last_action_id)
        return actions.copy()

    def done(self) -> tuple[bool, bool, bool]:
        terminated = False
        truncated = self.step_num >= self.max_step_per_episode
        need_resume = not truncated
        return terminated, truncated, need_resume

    def get_reward(self, **kwargs) -> dict[AgentID, float]:
        assert isinstance(self.pre_state, ThreatRegionState)
        assert isinstance(self.post_state, ThreatRegionState)
        pre_xy = (self.pre_state.observer_x, self.pre_state.observer_y)
        post_xy = (self.post_state.observer_x, self.post_state.observer_y)
        pre_region = self._threat_points(self.pre_state)
        post_region = self._threat_points(self.post_state)
        threat = self._threat_shaping(pre_xy, post_xy, pre_region, post_region)
        pre_goal = (self.pre_state.goal_x, self.pre_state.goal_y)
        post_goal = (self.post_state.goal_x, self.post_state.goal_y)
        seek = region_visible_target_seek_shaping(
            pre_xy,
            post_xy,
            [pre_goal],
            [post_goal],
            self.TARGET_DISTANCE,
            self._max_deviation,
            self._shaping_gamma,
            self._seek_output_range,
            region=None,
        )
        reward = self._threat_weight * threat + self._seek_weight * seek
        return {agent: reward for agent in self.possible_agents}

    def render(self):
        if self.game is None:
            return None
        if self.game.render_mode != self.render_mode:
            self.game.render_mode = self.render_mode
            self.game._init_pygame()
        return self.game.draw()

    def close(self):
        if self.game is not None and self.game.render_mode is not None:
            import pygame
            pygame.quit()
        super().close()

    def step_log(self) -> None:
        assert isinstance(self.pre_state, ThreatRegionState)
        assert isinstance(self.post_state, ThreatRegionState)
        if self.step_num % 100 != 0 and self.step_num < self.max_step_per_episode:
            return
        pre_xy = (self.pre_state.observer_x, self.pre_state.observer_y)
        post_xy = (self.post_state.observer_x, self.post_state.observer_y)
        post_region = self._threat_points(self.post_state)
        crossed = segment_intersects_region(pre_xy, post_xy, post_region)
        inside = point_in_region(post_xy[0], post_xy[1], post_region)
        threat = self._threat_shaping(
            pre_xy,
            post_xy,
            self._threat_points(self.pre_state),
            post_region,
        )
        seek = region_visible_target_seek_shaping(
            pre_xy,
            post_xy,
            [(self.pre_state.goal_x, self.pre_state.goal_y)],
            [(self.post_state.goal_x, self.post_state.goal_y)],
            self.TARGET_DISTANCE,
            self._max_deviation,
            self._shaping_gamma,
            self._seek_output_range,
            region=None,
        )
        LOG.info(
            f"[Ep {self.episode_num} - Step {self.step_num}] "
            f"inside={inside} crossed={crossed} threat={threat:.3f} seek={seek:.3f}"
        )

    @staticmethod
    def _convert_action_to_id(agent_action: dict[AgentID, Action]) -> int:
        assert len(agent_action) == 1
        action = next(iter(agent_action.values()))
        assert action.action_key == "threat_region_move", (
            f"unsupported threat_region action_key={action.action_key}"
        )
        action_id = int(action.parameters["action_id"])
        assert 0 <= action_id < ThreatRegionGame.NUM_ACTIONS, f"invalid action_id={action_id}"
        return action_id

    @staticmethod
    def _threat_points(state: ThreatRegionState) -> list[Point]:
        return [(float(v[0]), float(v[1])) for v in state.threat_vertices]

    def _threat_shaping(
        self,
        pre_xy: Point,
        post_xy: Point,
        pre_region: list[Point],
        post_region: list[Point],
    ) -> float:
        """
        region_threat_shaping 只接受单个 region，步前/步后共用同一威胁框。

        静态威胁区时 pre_region == post_region，直接委托即可。
        横向移动时须分别用 pre_region / post_region 计 Φ 与轨迹穿越判定。
        """
        targets_pre = [pre_xy]
        targets_post = [post_xy]
        if pre_region == post_region:
            return region_threat_shaping(
                targets_pre,
                targets_post,
                post_region,
                self._approach_decay_coef,
                self._shaping_gamma,
                self._output_range,
            )
        phi_pre = region_threat_score(
            targets_pre, pre_region, self._approach_decay_coef, self._output_range,
        )
        phi_post = region_threat_transition_score(
            targets_pre, targets_post, post_region, self._approach_decay_coef, self._output_range,
        )
        return pbrs_shaping(phi_pre, phi_post, self._shaping_gamma, self._output_range)

    @staticmethod
    def _rel_offset_norm(dx: float, dy: float) -> tuple[float, float]:
        rel_x = np.clip(dx / ThreatRegionGame.WIDTH, -1.0, 1.0) * 0.5 + 0.5
        rel_y = np.clip(dy / ThreatRegionGame.HEIGHT, -1.0, 1.0) * 0.5 + 0.5
        return float(rel_x), float(rel_y)

    def _obs_from_state(self, threat_state: ThreatRegionState) -> np.ndarray:
        ox = threat_state.observer_x
        oy = threat_state.observer_y
        rel_gx, rel_gy = self._rel_offset_norm(
            threat_state.goal_x - ox,
            threat_state.goal_y - oy,
        )
        vertex_rels: list[float] = []
        for vertex in self._threat_points(threat_state):
            vx, vy = self._rel_offset_norm(vertex[0] - ox, vertex[1] - oy)
            vertex_rels.extend([vx, vy])
        while len(vertex_rels) < self.THREAT_VERTEX_COUNT * 2:
            vertex_rels.extend([0.5, 0.5])

        region = self._threat_points(threat_state)
        inside = point_in_region(ox, oy, region)
        d_out = 0.0 if inside else point_to_region_boundary_distance(ox, oy, region)
        d_out_norm = min(1.0, d_out / max(ThreatRegionGame.WIDTH, 1.0))
        speed = float(np.hypot(threat_state.threat_vx, threat_state.threat_vy))
        speed_norm = min(1.0, speed / max(ThreatRegionGame.THREAT_SPEED, 1e-6)) * 0.5 + 0.5
        return np.array(
            [
                ox / ThreatRegionGame.WIDTH,
                oy / ThreatRegionGame.HEIGHT,
                rel_gx,
                rel_gy,
                *vertex_rels[: self.THREAT_VERTEX_COUNT * 2],
                d_out_norm,
                speed_norm,
            ],
            dtype=np.float32,
        )

    def _snapshot(self, action_id: int) -> ThreatRegionState:
        assert self.game is not None
        state = ThreatRegionState(
            entities=[],
            sim_time=self.game._frame / 60.0,
            observer_x=float(self.game.observer[0]),
            observer_y=float(self.game.observer[1]),
            goal_x=float(self.game.goal[0]),
            goal_y=float(self.game.goal[1]),
            threat_center_x=float(self.game.threat_center[0]),
            threat_center_y=float(self.game.threat_center[1]),
            threat_vertices=[[float(x), float(y)] for x, y in self.game.threat_vertices],
            threat_vx=float(self.game.threat_velocity[0]),
            threat_vy=float(self.game.threat_velocity[1]),
            frame_index=int(self.game._frame),
            action_id=int(action_id),
        )
        state._raw_data = state.to_dict()
        return state

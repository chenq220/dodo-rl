from __future__ import annotations

import gymnasium as gym
import numpy as np

from config import EnvConfig
from envs.action.model import Action
from envs.base import AgentID, ParallelEnvironment, LOG, ObsType
from envs.games.shaping_test_games.region_activity_game import RegionActivityGame
from envs.reward.shaping import region_activity_score, region_activity_shaping
from envs.reward.utils import (
    Point,
    ShapingOutputRange,
    pbrs_shaping,
    point_in_region,
    point_to_region_boundary_distance,
)
from envs.state.tests.region_activity_state import RegionActivityState


class RegionActivityEnv(ParallelEnvironment):
    """单智能体 region activity 环境：蓝点在随机六边形目标区域内活动。"""

    OBS_DIM = 8

    def __init__(self, config: EnvConfig, sim_config=None, connect_sim=True, sim_manager=None, role_id=None):
        super().__init__(
            config=config,
            sim_config=sim_config,
            connect_sim=connect_sim,
            sim_manager=sim_manager,
            role_id=role_id,
        )
        assert len(self.possible_agents) == 1, (
            f"RegionActivityEnv is single-agent, agents={self.possible_agents}"
        )

        metadata = self.config.metadata or {}
        self.render_mode = metadata.get("render_mode", "rgb_array")
        self.render_every_step = bool(metadata.get("render_every_step", False))
        self._seed = metadata.get("seed")
        self._shaping_gamma = float(metadata.get("shaping_gamma", 1.0))
        self._leave_decay_coef = float(metadata.get("leave_decay_coef", 0.01))
        self._region_moving = bool(metadata.get("region_moving", False))
        self._output_range = ShapingOutputRange(metadata.get("output_range", "unit"))
        reward_weights = metadata.get("reward_weights") or {}
        self._activity_weight = float(reward_weights.get("activity", 1.0))
        self._last_action_id = 0

        self.game: RegionActivityGame | None = None
        if not self.connect_sim:
            self.game = RegionActivityGame(
                render_mode=None,
                region_moving=self._region_moving,
                leave_decay_coef=self._leave_decay_coef,
                gamma=self._shaping_gamma,
                output_range=self._output_range,
                seed=self._seed,
            )

    def state_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32)

    def state(self) -> np.ndarray:
        if self.post_state is None:
            return np.zeros(self.OBS_DIM, dtype=np.float32)
        assert isinstance(self.post_state, RegionActivityState)
        return self._obs_from_state(self.post_state)

    def get_obs(self) -> dict[AgentID, ObsType]:
        observation = self.state()
        return {agent: observation.copy() for agent in self.possible_agents}

    def get_avail_agent_actions(self, agent_name: AgentID, init: bool = False) -> list[int]:
        return [1] * RegionActivityGame.NUM_ACTIONS

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
        assert isinstance(self.pre_state, RegionActivityState)
        assert isinstance(self.post_state, RegionActivityState)
        pre_xy = (self.pre_state.observer_x, self.pre_state.observer_y)
        post_xy = (self.post_state.observer_x, self.post_state.observer_y)
        pre_region = self._region_points(self.pre_state)
        post_region = self._region_points(self.post_state)
        # 移动框时 pre/post 顶点不同，见 _activity_shaping
        activity = self._activity_shaping(pre_xy, post_xy, pre_region, post_region)
        reward = self._activity_weight * activity
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
        assert isinstance(self.pre_state, RegionActivityState)
        assert isinstance(self.post_state, RegionActivityState)
        if self.step_num % 100 != 0 and self.step_num < self.max_step_per_episode:
            return
        post_xy = (self.post_state.observer_x, self.post_state.observer_y)
        post_region = self._region_points(self.post_state)
        inside = point_in_region(post_xy[0], post_xy[1], post_region)
        d_out = 0.0 if inside else point_to_region_boundary_distance(
            post_xy[0], post_xy[1], post_region,
        )
        activity = self._activity_shaping(
            (self.pre_state.observer_x, self.pre_state.observer_y),
            post_xy,
            self._region_points(self.pre_state),
            post_region,
        )
        LOG.info(
            f"[Ep {self.episode_num} - Step {self.step_num}] "
            f"inside={inside} d_out={d_out:.1f} activity={activity:.3f}"
        )

    @staticmethod
    def _convert_action_to_id(agent_action: dict[AgentID, Action]) -> int:
        assert len(agent_action) == 1
        action = next(iter(agent_action.values()))
        assert action.action_key == "region_move", (
            f"unsupported region action_key={action.action_key}"
        )
        action_id = int(action.parameters["action_id"])
        assert 0 <= action_id < RegionActivityGame.NUM_ACTIONS, f"invalid action_id={action_id}"
        return action_id

    @staticmethod
    def _region_points(state: RegionActivityState) -> list[Point]:
        return [(float(v[0]), float(v[1])) for v in state.hex_vertices]

    def _activity_shaping(
        self,
        pre_xy: Point,
        post_xy: Point,
        pre_region: list[Point],
        post_region: list[Point],
    ) -> float:
        """
        region_activity_shaping 只接受单个 region，步前/步后共用同一框。

        静态框（region_moving=false）时 pre_region == post_region，直接委托即可。
        移动框时 PBRS 应对 Φ(pre_xy, pre_region) 与 Φ(post_xy, post_region) 分别计分，
        不能只用 post_region 或 pre_region，否则蓝点会被错框评判。
        """
        if pre_region == post_region:
            return region_activity_shaping(
                pre_xy,
                post_xy,
                post_region,
                self._leave_decay_coef,
                self._shaping_gamma,
                self._output_range,
            )
        phi_pre = region_activity_score(
            pre_xy[0], pre_xy[1], pre_region, self._leave_decay_coef, self._output_range,
        )
        phi_post = region_activity_score(
            post_xy[0], post_xy[1], post_region, self._leave_decay_coef, self._output_range,
        )
        return pbrs_shaping(phi_pre, phi_post, self._shaping_gamma, self._output_range)

    def _obs_from_state(self, activity_state: RegionActivityState) -> np.ndarray:
        ox = activity_state.observer_x / RegionActivityGame.WIDTH
        oy = activity_state.observer_y / RegionActivityGame.HEIGHT
        rel_x = np.clip(
            (activity_state.hex_center_x - activity_state.observer_x) / RegionActivityGame.WIDTH,
            -1.0,
            1.0,
        ) * 0.5 + 0.5
        rel_y = np.clip(
            (activity_state.hex_center_y - activity_state.observer_y) / RegionActivityGame.HEIGHT,
            -1.0,
            1.0,
        ) * 0.5 + 0.5
        radius_norm = min(
            1.0,
            activity_state.hex_radius / max(RegionActivityGame.MAX_HEX_RADIUS, 1.0),
        )
        region = self._region_points(activity_state)
        inside = point_in_region(activity_state.observer_x, activity_state.observer_y, region)
        inside_flag = 1.0 if inside else 0.0
        d_out = 0.0 if inside else point_to_region_boundary_distance(
            activity_state.observer_x, activity_state.observer_y, region,
        )
        d_out_norm = min(1.0, d_out / max(RegionActivityGame.WIDTH, 1.0))
        hex_speed = float(np.hypot(activity_state.hex_vx, activity_state.hex_vy))
        speed_norm = min(1.0, hex_speed / max(RegionActivityGame.REGION_SPEED, 1e-6)) * 0.5 + 0.5
        return np.array(
            [ox, oy, rel_x, rel_y, radius_norm, inside_flag, d_out_norm, speed_norm],
            dtype=np.float32,
        )

    def _snapshot(self, action_id: int) -> RegionActivityState:
        assert self.game is not None
        state = RegionActivityState(
            entities=[],
            sim_time=self.game._frame / 60.0,
            observer_x=float(self.game.observer[0]),
            observer_y=float(self.game.observer[1]),
            hex_center_x=float(self.game.hex_center[0]),
            hex_center_y=float(self.game.hex_center[1]),
            hex_radius=float(self.game.hex_radius),
            hex_vertices=[[float(x), float(y)] for x, y in self.game.hex_vertices],
            hex_vx=float(self.game.hex_velocity[0]),
            hex_vy=float(self.game.hex_velocity[1]),
            frame_index=int(self.game._frame),
            action_id=int(action_id),
        )
        state._raw_data = state.to_dict()
        return state

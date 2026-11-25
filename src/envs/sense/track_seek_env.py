from __future__ import annotations

import gymnasium as gym
import numpy as np

from config import EnvConfig
from envs.action.model import Action
from envs.base import AgentID, ParallelEnvironment, LOG, ObsType
from envs.games.shaping_test_games.target_seek_game import TargetSeekGame
from envs.reward.shaping import region_visible_target_seek_shaping
from envs.reward.utils import ShapingOutputRange, nearest_target, point_distance
from envs.state.tests.track_seek_state import TrackSeekState


class TrackSeekEnv(ParallelEnvironment):
    """单智能体 target seek（d=0）环境；蓝点追踪红点，奖励由 get_reward 计算 seek 塑形。"""

    OBS_DIM = 8
    TARGET_DISTANCE = 0.0

    def __init__(self, config: EnvConfig, sim_config=None, connect_sim=True, sim_manager=None, role_id=None):
        super().__init__(
            config=config,
            sim_config=sim_config,
            connect_sim=connect_sim,
            sim_manager=sim_manager,
            role_id=role_id,
        )
        assert len(self.possible_agents) == 1, (
            f"TrackSeekEnv is single-agent, agents={self.possible_agents}"
        )

        metadata = self.config.metadata or {}
        self.render_mode = metadata.get("render_mode", "rgb_array")
        self.render_every_step = bool(metadata.get("render_every_step", False))
        self._seed = metadata.get("seed")
        self._shaping_gamma = float(metadata.get("shaping_gamma", 1.0))
        self._max_deviation = float(metadata.get("max_deviation", float(TargetSeekGame.WIDTH)))
        self._num_targets = int(metadata.get("num_targets", 1))
        self._targets_moving = bool(metadata.get("targets_moving", True))
        self._output_range = ShapingOutputRange(metadata.get("output_range", "unit"))
        reward_weights = metadata.get("reward_weights") or {}
        self._seek_weight = float(reward_weights.get("seek", 1.0))
        self._game_region = TargetSeekGame.game_region()
        self._last_action_id = 0

        self.game: TargetSeekGame | None = None
        if not self.connect_sim:
            self.game = TargetSeekGame(
                render_mode=None,
                num_targets=self._num_targets,
                targets_moving=self._targets_moving,
                target_distance=self.TARGET_DISTANCE,
                max_deviation=self._max_deviation,
                gamma=self._shaping_gamma,
                output_range=self._output_range,
                region=self._game_region,
                seed=self._seed,
            )

    def state_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32)

    def state(self) -> np.ndarray:
        if self.post_state is None:
            return np.zeros(self.OBS_DIM, dtype=np.float32)
        assert isinstance(self.post_state, TrackSeekState)
        return self._obs_from_state(self.post_state)

    def get_obs(self) -> dict[AgentID, ObsType]:
        observation = self.state()
        return {agent: observation.copy() for agent in self.possible_agents}

    def get_avail_agent_actions(self, agent_name: AgentID, init: bool = False) -> list[int]:
        return [1] * TargetSeekGame.NUM_ACTIONS

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
        assert isinstance(self.pre_state, TrackSeekState)
        assert isinstance(self.post_state, TrackSeekState)
        pre_observer = (self.pre_state.observer_x, self.pre_state.observer_y)
        post_observer = (self.post_state.observer_x, self.post_state.observer_y)
        targets_pre = self._target_points(self.pre_state)
        targets_post = self._target_points(self.post_state)
        seek = region_visible_target_seek_shaping(
            pre_observer,
            post_observer,
            targets_pre,
            targets_post,
            self.TARGET_DISTANCE,
            self._max_deviation,
            self._shaping_gamma,
            self._output_range,
            self._game_region,
        )
        reward = self._seek_weight * seek
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
        assert isinstance(self.pre_state, TrackSeekState)
        assert isinstance(self.post_state, TrackSeekState)
        if self.step_num % 100 != 0 and self.step_num < self.max_step_per_episode:
            return
        pre_xy = (self.pre_state.observer_x, self.pre_state.observer_y)
        post_xy = (self.post_state.observer_x, self.post_state.observer_y)
        targets_post = self._target_points(self.post_state)
        nearest = nearest_target(post_xy, targets_post) if targets_post else None
        dist = (
            point_distance(post_xy[0], post_xy[1], nearest[0], nearest[1])
            if nearest is not None
            else float("inf")
        )
        seek = region_visible_target_seek_shaping(
            pre_xy,
            post_xy,
            self._target_points(self.pre_state),
            targets_post,
            self.TARGET_DISTANCE,
            self._max_deviation,
            self._shaping_gamma,
            self._output_range,
            self._game_region,
        )
        LOG.info(
            f"[Ep {self.episode_num} - Step {self.step_num}] "
            f"dist={dist:.1f} seek={seek:.3f}"
        )

    @staticmethod
    def _convert_action_to_id(agent_action: dict[AgentID, Action]) -> int:
        assert len(agent_action) == 1
        action = next(iter(agent_action.values()))
        assert action.action_key == "track_move", (
            f"unsupported track action_key={action.action_key}"
        )
        action_id = int(action.parameters["action_id"])
        assert 0 <= action_id < TargetSeekGame.NUM_ACTIONS, f"invalid action_id={action_id}"
        return action_id

    @staticmethod
    def _target_points(state: TrackSeekState) -> list[tuple[float, float]]:
        return [(float(t[0]), float(t[1])) for t in state.targets]

    def _obs_from_state(self, track_state: TrackSeekState) -> np.ndarray:
        ox = track_state.observer_x / TargetSeekGame.WIDTH
        oy = track_state.observer_y / TargetSeekGame.HEIGHT
        rel_x = rel_y = 0.5
        dist_norm = 0.0
        tvx = tvy = 0.5
        has_target = 0.0
        observer = (track_state.observer_x, track_state.observer_y)
        target_points = self._target_points(track_state)
        nearest = nearest_target(observer, target_points) if target_points else None
        if nearest is not None:
            for target in track_state.targets:
                if float(target[0]) == nearest[0] and float(target[1]) == nearest[1]:
                    rel_x = np.clip((nearest[0] - track_state.observer_x) / TargetSeekGame.WIDTH, -1.0, 1.0) * 0.5 + 0.5
                    rel_y = np.clip((nearest[1] - track_state.observer_y) / TargetSeekGame.HEIGHT, -1.0, 1.0) * 0.5 + 0.5
                    dist = point_distance(observer[0], observer[1], nearest[0], nearest[1])
                    dist_norm = min(1.0, dist / max(self._max_deviation, 1.0))
                    if len(target) >= 4:
                        tvx = np.clip(float(target[2]) / TargetSeekGame.TARGET_SPEED, -1.0, 1.0) * 0.5 + 0.5
                        tvy = np.clip(float(target[3]) / TargetSeekGame.TARGET_SPEED, -1.0, 1.0) * 0.5 + 0.5
                    has_target = 1.0
                    break
        return np.array(
            [ox, oy, rel_x, rel_y, dist_norm, tvx, tvy, has_target],
            dtype=np.float32,
        )

    def _snapshot(self, action_id: int) -> TrackSeekState:
        assert self.game is not None
        state = TrackSeekState(
            entities=[],
            sim_time=self.game._frame / 60.0,
            observer_x=float(self.game.observer[0]),
            observer_y=float(self.game.observer[1]),
            targets=[
                [float(t.x), float(t.y), float(t.vx), float(t.vy)]
                for t in self.game.targets
            ],
            frame_index=int(self.game._frame),
            action_id=int(action_id),
        )
        state._raw_data = state.to_dict()
        return state

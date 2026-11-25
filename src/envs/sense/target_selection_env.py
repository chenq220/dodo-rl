from __future__ import annotations

import gymnasium as gym
import numpy as np

from config import EnvConfig
from envs.action.model import Action
from envs.base import AgentID, ParallelEnvironment, LOG, ObsType
from envs.games.shaping_test_games.target_selection_game import TargetSelectionGame
from envs.reward.shaping import target_selection_shaping
from envs.reward.utils import ShapingOutputRange
from envs.state.tests.target_selection_state import TargetSelectionState


class TargetSelectionEnv(ParallelEnvironment):
    """单智能体目标选择环境：蓝球连接下方十个目标球。"""

    FEATURES_PER_TARGET = 2
    OBS_DIM = TargetSelectionGame.NUM_TARGETS * FEATURES_PER_TARGET
    ALL_TARGETS = tuple(range(TargetSelectionGame.NUM_TARGETS))

    def __init__(
        self,
        config: EnvConfig,
        sim_config=None,
        connect_sim=True,
        sim_manager=None,
        role_id=None,
    ):
        super().__init__(
            config=config,
            sim_config=sim_config,
            connect_sim=connect_sim,
            sim_manager=sim_manager,
            role_id=role_id,
        )
        assert len(self.possible_agents) == 1, (
            f"TargetSelectionEnv is single-agent, agents={self.possible_agents}"
        )

        metadata = self.config.metadata or {}
        self.render_mode = metadata.get("render_mode", "rgb_array")
        self.render_every_step = bool(metadata.get("render_every_step", False))
        self._step_delay_seconds = float(metadata.get("step_delay_seconds", 0.0))
        if self._step_delay_seconds < 0.0:
            raise ValueError("step_delay_seconds must be >= 0")
        self._seed = metadata.get("seed")
        self._primary_mode = metadata.get("primary_mode", "random")
        self._primary_probability = float(metadata.get("primary_probability", 0.5))
        self._output_range = ShapingOutputRange(metadata.get("output_range", "signed"))
        reward_weights = metadata.get("reward_weights") or {}
        self._primary_weight = float(reward_weights.get("primary", 1.0))
        self._selection_weight = float(reward_weights.get("selection", 1.0))
        self._last_target_index = -1
        self._last_reward_breakdown = {"primary": 0.0, "selection": 0.0}

        self.game: TargetSelectionGame | None = None
        if not self.connect_sim:
            self.game = TargetSelectionGame(
                render_mode=None,
                primary_mode=self._primary_mode,
                primary_probability=self._primary_probability,
                output_range=self._output_range,
                seed=self._seed,
            )

    def state_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.OBS_DIM,), dtype=np.float32)

    def state(self) -> np.ndarray:
        if self.post_state is None:
            return np.zeros(self.OBS_DIM, dtype=np.float32)
        assert isinstance(self.post_state, TargetSelectionState)
        return self._obs_from_state(self.post_state)

    def get_obs(self) -> dict[AgentID, ObsType]:
        observation = self.state()
        return {agent: observation.copy() for agent in self.possible_agents}

    def get_avail_agent_actions(self, agent_name: AgentID, init: bool = False) -> list[int]:
        return [1] * TargetSelectionGame.NUM_TARGETS

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
        self._last_target_index = -1
        self._last_reward_breakdown = {"primary": 0.0, "selection": 0.0}
        self.pre_state = self.post_state = self._snapshot(selected_target=-1)
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
        self.post_state = self._snapshot(selected_target=self._last_target_index)
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
                "reward_breakdown": self._last_reward_breakdown.copy(),
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
        self._last_target_index = self._convert_action_to_target(live_actions)
        self.game.step_action(self._last_target_index)
        return actions.copy()

    def done(self) -> tuple[bool, bool, bool]:
        terminated = False
        truncated = self.step_num >= self.max_step_per_episode
        need_resume = not truncated
        return terminated, truncated, need_resume

    def get_reward(self, **kwargs) -> dict[AgentID, float]:
        assert isinstance(self.pre_state, TargetSelectionState)
        assert isinstance(self.post_state, TargetSelectionState)
        selected_target = self.post_state.selected_target
        primary = float(selected_target in self.pre_state.primary_target_indices)
        selection = target_selection_shaping(
            selected_target,
            self.pre_state.shaping_target_indices,
            self.ALL_TARGETS,
            primary_target_indices=self.pre_state.primary_target_indices,
            available_target_indices=self.ALL_TARGETS,
            output_range=self._output_range,
        )
        weighted_primary = self._primary_weight * primary
        weighted_selection = self._selection_weight * selection
        self._last_reward_breakdown = {
            "primary": weighted_primary,
            "selection": weighted_selection,
        }
        reward = weighted_primary + weighted_selection
        return {agent: reward for agent in self.possible_agents}

    def render(self):
        if self.game is None:
            return None
        if self.game.render_mode != self.render_mode:
            self.game.render_mode = self.render_mode
            self.game._init_pygame()
        if self.render_mode == "human":
            import pygame

            pygame.event.pump()
        # 评估画面展示策略决策时看到的颜色，避免与已采样的下一状态颜色错位。
        surface = self.game.draw(show_last_decision=True)
        if self.render_mode == "human":
            import pygame

            if self._step_delay_seconds > 0.0:
                pygame.time.wait(round(self._step_delay_seconds * 1000.0))
            elif self.game.clock is not None:
                self.game.clock.tick(self.game.FPS)
        return surface

    def close(self):
        if self.game is not None and self.game.render_mode is not None:
            import pygame

            pygame.quit()
        super().close()

    def step_log(self) -> None:
        assert isinstance(self.pre_state, TargetSelectionState)
        if self.step_num % 100 != 0 and self.step_num < self.max_step_per_episode:
            return
        LOG.info(
            f"[Ep {self.episode_num} - Step {self.step_num}] "
            f"selected={self._last_target_index} "
            f"primary={self.pre_state.primary_target_indices} "
            f"shaping={self.pre_state.shaping_target_indices} "
            f"reward={self._last_reward_breakdown}"
        )

    @staticmethod
    def _convert_action_to_target(agent_action: dict[AgentID, Action]) -> int:
        assert len(agent_action) == 1
        action = next(iter(agent_action.values()))
        assert action.action_key == "target_select", (
            f"unsupported target selection action_key={action.action_key}"
        )
        target_index = int(action.target)
        assert 0 <= target_index < TargetSelectionGame.NUM_TARGETS, (
            f"invalid target_index={target_index}"
        )
        return target_index

    @classmethod
    def _obs_from_state(cls, selection_state: TargetSelectionState) -> np.ndarray:
        shaping_targets = set(selection_state.shaping_target_indices)
        primary_targets = set(selection_state.primary_target_indices)
        features: list[float] = []
        for target_index in cls.ALL_TARGETS:
            features.extend(
                [
                    float(target_index in shaping_targets),
                    float(target_index in primary_targets),
                ]
            )
        return np.asarray(features, dtype=np.float32)

    def _snapshot(self, selected_target: int) -> TargetSelectionState:
        assert self.game is not None
        state = TargetSelectionState(
            entities=[],
            sim_time=self.game.frame_index / 60.0,
            shaping_target_indices=list(self.game.shaping_target_indices),
            primary_target_indices=list(self.game.primary_target_indices),
            selected_target=int(selected_target),
            frame_index=int(self.game.frame_index),
        )
        state._raw_data = state.to_dict()
        return state

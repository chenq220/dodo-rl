from pathlib import Path

import numpy as np
import pytest

from envs.action.base import ActionTable
from envs.games.cooperative_search import (
    CooperativeSearchConfig,
    CooperativeSearchGame,
    FlightCommand,
)
from envs.sense.cooperative_search_env import get_threat_obs
from envs.sense.cooperative_search_threat_env import CooperativeSearchThreatEnv
from utils.toml_config import TomlTrainingConfig


ROOT = Path(__file__).resolve().parents[2]
MAIN_CONFIG_PATH = ROOT / "subjects/cooperative_search/cooperative_search_qmix.toml"
THREAT_CONFIG_PATH = (
    ROOT / "subjects/cooperative_search/cooperative_search_threat_qmix.toml"
)


def _configured_threat_env():
    config = TomlTrainingConfig(THREAT_CONFIG_PATH, project_root=ROOT)
    action_table = ActionTable(
        config.agents_config[0].define_action,
        config.agents_config[0].agent_target_names,
        config.agents_config[0].multi_action_mode,
    )
    return config, action_table, CooperativeSearchThreatEnv(config.env_config)


def test_threat_training_and_formal_fixed_agent_have_identical_policy_contract():
    main = TomlTrainingConfig(MAIN_CONFIG_PATH, project_root=ROOT)
    threat = TomlTrainingConfig(THREAT_CONFIG_PATH, project_root=ROOT)

    formal_agent = main.agents_config[1]
    training_agent = threat.agents_config[0]
    assert threat.env_config.sense_name == "cooperative_search_threat_training"
    assert threat.env_config.state_dim == 32
    assert training_agent.name == formal_agent.name == "dynamic_threats"
    assert training_agent.agent_type == formal_agent.agent_type == "rnn"
    assert training_agent.agent_names == formal_agent.agent_names
    assert training_agent.obs_dim == formal_agent.obs_dim == 40
    assert training_agent.define_action == formal_agent.define_action
    assert ActionTable(
        training_agent.define_action,
        multi_action_mode=training_agent.multi_action_mode,
    ).n_actions == 12
    assert training_agent.obs_last_action == formal_agent.obs_last_action is True
    assert training_agent.obs_agent_id == formal_agent.obs_agent_id is True
    assert training_agent.hidden_dim == formal_agent.hidden_dim == 128
    assert formal_agent.fixed_mode is True
    assert training_agent.fixed_mode is False
    assert threat.learners_config[0].name == training_agent.name
    assert all(learner.name != formal_agent.name for learner in main.learners_config)


def test_threat_training_reuses_exact_formal_observation_function():
    _, _, training_env = _configured_threat_env()
    formal_game = CooperativeSearchGame(CooperativeSearchConfig())
    try:
        training_env.reset(seed=41)
        formal_game.reset(seed=43)
        agent = "dynamic_threat_0"
        formal_game.threat_positions = training_env.game.threat_positions.copy()
        formal_game.headings.update(training_env.game.headings)
        formal_game._target_positions = list(training_env.game._target_positions)

        training_obs = training_env.get_obs()[agent]
        formal_obs = get_threat_obs(formal_game, agent)
        assert training_obs.shape == (40,)
        assert np.array_equal(training_obs, formal_obs)
    finally:
        training_env.close()
        formal_game.close()


def test_threat_only_game_does_not_treat_zero_configured_drones_as_all_dead():
    game = CooperativeSearchGame(CooperativeSearchConfig(
        num_drones=0,
        num_dynamic_threats=3,
        num_targets=10,
        max_steps=2,
    ))
    try:
        game.reset(seed=47)
        _, _, terminations, truncations, _ = game.step({
            agent: FlightCommand(game.available_headings(agent)[0])
            for agent in game.agents
        })
        assert not any(terminations.values())
        assert not any(truncations.values())
    finally:
        game.close()


def test_threat_reward_prefers_moving_away_and_strongly_penalizes_intrusion():
    _, _, env = _configured_threat_env()
    try:
        env.reset(seed=53)
        env.game.threat_positions = {
            "dynamic_threat_0": (90.0, 150.0),
            "dynamic_threat_1": (150.0, 90.0),
            "dynamic_threat_2": (210.0, 150.0),
        }
        previous = {agent: 40.0 for agent in env.threat_agents}
        away = {agent: 42.4 for agent in env.threat_agents}
        toward_intrusion = {agent: 28.0 for agent in env.threat_agents}

        away_parts = env._reward_components(previous, away)
        intrusion_parts = env._reward_components(previous, toward_intrusion)
        assert away_parts["clearance_progress"] == pytest.approx(0.3)
        assert away_parts["target_intrusion"] == 0.0
        assert sum(away_parts.values()) > 0.0
        assert intrusion_parts["clearance_progress"] == pytest.approx(-0.3)
        assert intrusion_parts["target_intrusion"] == pytest.approx(-3.0)
        assert sum(intrusion_parts.values()) < -3.0
    finally:
        env.close()


def test_threat_auxiliary_penalties_are_bounded_below_one_third_of_core():
    _, _, env = _configured_threat_env()
    try:
        env.reset(seed=57)
        assert (
            env.threat_separation_penalty + env.center_distance_penalty
            <= env.target_intrusion_penalty / 3.0
        )

        env.game.threat_positions = {
            agent: (150.0, 150.0)
            for agent in env.threat_agents
        }
        assert env._threat_separation_reward() == pytest.approx(-0.15)
        assert env._center_distance_reward() == pytest.approx(0.0)

        env.game.threat_positions = {
            "dynamic_threat_0": (0.0, 0.0),
            "dynamic_threat_1": (300.0, 0.0),
            "dynamic_threat_2": (0.0, 300.0),
        }
        assert env._threat_separation_reward() == pytest.approx(0.0)
        assert env._center_distance_reward() == pytest.approx(-0.15)
    finally:
        env.close()


def test_threat_training_environment_dimensions_mask_step_and_team_reward():
    config, table, env = _configured_threat_env()
    try:
        observations, _ = env.reset(seed=59)
        assert config.train_config.max_episode == 1601
        assert env.observation_dim == 40
        assert env.centralized_state_dim == 32
        assert env.state().shape == (32,)
        assert all(observation.shape == (40,) for observation in observations.values())
        assert all(any(env.get_avail_agent_actions(agent)) for agent in env.agents)

        actions = {
            agent: table.action_map[env.game.available_headings(agent)[0]]
            for agent in env.agents
        }
        _, rewards, terminations, truncations, infos = env.step(actions)
        assert not any(terminations.values())
        assert not any(truncations.values())
        assert sum(rewards.values()) == pytest.approx(env._last_team_reward)
        assert all("reward_breakdown" in info for info in infos.values())
    finally:
        env.close()

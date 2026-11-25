from dataclasses import fields
from pathlib import Path

import numpy as np
import pygame
import pytest

from common.exception import EnvironmentException
from envs.action.base import ActionTable
from envs.action.model import ActionParameterDefined, CompositeAction, StepActionDefined
from envs.games.cooperative_search import (
    NUM_HEADINGS,
    CooperativeSearchConfig,
    CooperativeSearchGame,
    FlightCommand,
)
from envs.sense.cooperative_search_env import CooperativeSearchEnv, get_threat_obs
from scheduler.stage_task import StageTask
from utils.task_id import task_id
from utils.toml_config import TomlTrainingConfig


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "subjects/cooperative_search/cooperative_search_qmix.toml"
MAPPO_CONFIG_PATH = ROOT / "subjects/cooperative_search/cooperative_search_mappo.toml"


def _definitions():
    return [
        StepActionDefined(
            "heading", [ActionParameterDefined("value", values=list(range(NUM_HEADINGS)))], False
        )
    ]


def _configured_env(tmp_path: Path | None = None):
    conf = TomlTrainingConfig(CONFIG_PATH, project_root=ROOT)
    if tmp_path is not None:
        conf.env_config.metadata["probability_map_output_dir"] = str(tmp_path)
    table = ActionTable(
        conf.agents_config[0].define_action,
        conf.agents_config[0].agent_target_names,
        conf.agents_config[0].multi_action_mode,
    )
    return conf, table, CooperativeSearchEnv(conf.env_config)


def test_cooperative_search_uses_sixty_second_sample_task_timeout():
    conf = TomlTrainingConfig(CONFIG_PATH, project_root=ROOT)
    assert conf.train_config.sample_task_timeout == 60


def test_cooperative_search_mappo_matches_qmix_environment_and_uses_on_policy_steps():
    if not MAPPO_CONFIG_PATH.exists():
        pytest.skip("cooperative_search_mappo.toml is not present in this checkout")
    qmix = TomlTrainingConfig(CONFIG_PATH, project_root=ROOT)
    mappo = TomlTrainingConfig(MAPPO_CONFIG_PATH, project_root=ROOT)

    assert mappo.learners_config[0].algo_type == "mappo"
    algo = mappo.learners_config[0].algo_config
    assert algo.hidden_dim == 128
    assert algo.sample_strategy == "on_policy_cursor_steps"
    assert algo.batch_size == 3600
    assert algo.learn_times_per_episode == 1
    assert algo.eps_clip == pytest.approx(0.1)
    assert algo.epochs == 4
    assert algo.obs_individual_obs is False
    assert mappo.train_config.sample_learn_max_lag == 2
    assert mappo.eval_config.max_episode == 20
    assert mappo.env_config.metadata["game_config"] == qmix.env_config.metadata["game_config"]
    reward_keys = {
        "revisited_empty_penalty_episode_limit",
        "exploration_reward_per_cell",
        "frontier_potential_reward_weight",
        "correct_detection_reward_total",
        "correct_detection_reward_half_life_ratio",
        "false_detection_penalty",
        "death_penalty_per_drone",
    }
    assert all(
        mappo.env_config.metadata[key] == qmix.env_config.metadata[key]
        for key in reward_keys
    )
    search, evade = mappo.stage_task_config.stages
    assert search.learners_config[0].algo_config.lr == pytest.approx(0.00005)
    assert search.learners_config[0].algo_config.entropy_coef == pytest.approx(0.005)
    assert search.learners_config[0].algo_config.epochs == 8
    assert evade.learners_config[0].algo_config.lr == pytest.approx(0.00005)
    assert evade.learners_config[0].algo_config.entropy_coef == pytest.approx(0.005)
    assert evade.learners_config[0].algo_config.epochs == 4


def test_cooperative_search_reward_metadata_uses_detection_curve():
    conf = TomlTrainingConfig(CONFIG_PATH, project_root=ROOT)
    metadata = conf.env_config.metadata
    assert "correct_detection_reward_total" in metadata
    assert metadata["correct_detection_reward_total"] > 0
    assert 0 < metadata["correct_detection_reward_half_life_ratio"] < 1
    assert "correct_detection_reward" not in metadata
    assert metadata["death_penalty_per_drone"] >= 0
    assert conf.entry_config.evaluation_class == (
        "subjects.cooperative_search.evaluation:CooperativeSearchEvaluation"
    )
    assert conf.entry_config.distributed_evaluation_class == (
        "subjects.cooperative_search.evaluation:CooperativeSearchEvalControl"
    )


def test_cooperative_search_retrains_evade_from_refine_optimo():
    conf = TomlTrainingConfig(CONFIG_PATH, project_root=ROOT)

    assert Path(conf.train_config.model_load_dir) == ROOT / "artifacts/models/cooperative_search"
    if conf.stage_task_config.start_stage != 2:
        pytest.skip("current qmix config is a full curriculum, not evade-only refine")
    assert (
        conf.stage_task_config.skip_stage_model_version == "optimo_e00f36eb_refine_0"
        or conf.stage_task_config.stage_model_versions == ["optimo_e00f36eb_refine_0"]
    )

    stage_task = StageTask(conf.stage_task_config)
    try:
        assert stage_task.starting_stage.stage_name == "evade"
        assert (
            stage_task.starting_stage.train_config.model_version
            == "optimo_e00f36eb_refine_0"
        )
    finally:
        stage_task.close()



def test_heading_only_action_table_has_twelve_composite_actions():
    table = ActionTable(_definitions(), multi_action_mode="combination")
    assert table.n_actions == 12
    assert isinstance(table.action_map[0], CompositeAction)
    assert table.action_map[11]["heading"].parameters["value"] == 11


def test_cooperative_search_active_validation_uses_environment_exception():
    with pytest.raises(EnvironmentException, match="fixed speed"):
        CooperativeSearchConfig(fixed_speed_mps=0.0)


def test_physical_speed_moves_3_km_per_thirty_second_step():
    config = CooperativeSearchConfig(num_drones=1, num_dynamic_threats=0, num_targets=1)
    game = CooperativeSearchGame(config)
    game.reset(seed=11)
    start = game.drone_positions["drone_0"]
    game.step({"drone_0": FlightCommand(heading=0)})
    assert config.fixed_speed_mps == 100.0
    assert config.step_interval_secs == 30.0
    assert config.movement_distance_km == pytest.approx(3.0)
    assert game.drone_positions["drone_0"] == pytest.approx((start[0] + 3.0, start[1]))


def test_scenario_has_three_dynamic_threats_and_drones_start_connected():
    config_fields = {item.name for item in fields(CooperativeSearchConfig)}
    assert {
        "num_dynamic_threats",
        "threat_sensor_radius",
        "threat_warning_radius",
        "threat_speed_mps",
        "initial_formation_side_km",
        "initial_threat_separation",
    } <= config_fields
    game = CooperativeSearchGame(CooperativeSearchConfig())
    assert game.threat_agents == [
        "dynamic_threat_0", "dynamic_threat_1", "dynamic_threat_2"
    ]
    expected_drone_positions = {
        "drone_0": (0.0, 0.0),
        "drone_1": (30.0, 0.0),
        "drone_2": (0.0, 30.0),
        "drone_3": (30.0, 30.0),
    }
    sampled_threat_positions = set()
    for seed in range(20):
        game.reset(seed=seed)
        assert game.drone_positions == expected_drone_positions
        sampled_threat_positions.add(tuple(game.threat_positions.values()))
        positions = list(game.drone_positions.values())
        distances = [
            game._distance(first, second)
            for index, first in enumerate(positions)
            for second in positions[index + 1:]
        ]
        assert max(distances) <= game.config.communication_radius + 1e-9
        assert min(
            game._distance(drone, threat)
            for drone in game.drone_positions.values()
            for threat in game.threat_positions.values()
        ) >= game.config.initial_threat_separation
    assert len(sampled_threat_positions) > 1


def test_dynamic_threat_moves_at_80_mps_detects_drone_and_uses_uav_turn_mask():
    game = CooperativeSearchGame(CooperativeSearchConfig(
        num_drones=1, num_dynamic_threats=1, num_targets=1
    ))
    game.reset(seed=17)
    game.drone_positions["drone_0"] = (100.0, 100.0)
    game.threat_positions["dynamic_threat_0"] = (119.0, 100.0)
    game._target_positions = [(280.0, 280.0)]
    game.shared_belief.fill(0.5)
    game._update_threat_detections()

    assert game.config.threat_speed_mps == 80.0
    assert game.config.threat_sensor_radius == 30.0
    assert game.config.threat_warning_radius == 45.0
    assert game.config.threat_movement_distance_km == pytest.approx(2.4)
    assert game.threat_detected_flags["dynamic_threat_0"] is True
    assert game.drones_detected_by_threat["drone_0"] is True
    assert len(game.available_headings("dynamic_threat_0")) == 12

    game.threat_positions["dynamic_threat_0"] = (140.0, 100.0)
    game.step({
        "drone_0": FlightCommand(0),
        "dynamic_threat_0": FlightCommand(0),
    })
    assert game.drone_positions["drone_0"] == pytest.approx((103.0, 100.0))
    assert game.threat_positions["dynamic_threat_0"] == pytest.approx((142.4, 100.0))
    assert len(game.available_headings("dynamic_threat_0")) == 3


def test_dynamic_threat_observes_targets_and_peer_threats_but_not_drones(tmp_path):
    conf, _, env = _configured_env(tmp_path)
    threat_config = conf.agents_config[1]
    assert threat_config.name == "dynamic_threats"
    assert threat_config.agent_type == "rnn"
    assert threat_config.algo_config.use_net is True
    assert threat_config.fixed_mode is True
    assert threat_config.obs_dim == 40
    assert threat_config.obs_last_action is True
    assert threat_config.obs_agent_id is True
    assert threat_config.hidden_dim == 128
    assert threat_config.model_file_path.endswith(
        "subjects/cooperative_search/final_b1813237_avoid_targets_0_dynamic_threats_agent_net.pth"
    )
    assert all(learner.name != threat_config.name for learner in conf.learners_config)

    try:
        env.reset(seed=23)
        threat = env.threat_agents[0]
        env.game.threat_positions[threat] = (150.0, 150.0)
        env.game.threat_positions["dynamic_threat_1"] = (180.0, 150.0)
        env.game.threat_positions["dynamic_threat_2"] = (150.0, 210.0)
        env.game.headings[threat] = 0
        env.game._target_positions = [
            (180.0, 150.0),
            (150.0, 210.0),
            *[(280.0, 280.0)] * 8,
        ]
        before = get_threat_obs(env.game, threat)
        env.game.drone_positions = {
            agent: (float(index), float(index))
            for index, agent in enumerate(env.drone_agents)
        }
        after = get_threat_obs(env.game, threat)

        assert before.shape == after.shape == (40,)
        assert np.array_equal(before, after)
        assert before[:4] == pytest.approx([0.5, 0.5, 0.0, 1.0])
        assert before[4] == pytest.approx(30.0 / np.hypot(300.0, 300.0))
        assert before[5:7] == pytest.approx([0.0, 1.0])
        assert before[34] == pytest.approx(30.0 / np.hypot(300.0, 300.0))
        assert before[35:37] == pytest.approx([0.0, 1.0])

        env.game.threat_positions["dynamic_threat_1"] = (210.0, 150.0)
        peer_moved = get_threat_obs(env.game, threat)
        assert not np.array_equal(after[34:], peer_moved[34:])
    finally:
        env.close()


def test_dynamic_threat_boundary_mask_turns_it_back_into_map():
    game = CooperativeSearchGame(CooperativeSearchConfig(
        num_drones=1, num_dynamic_threats=1, num_targets=1
    ))
    game.reset(seed=27)
    game.drone_positions["drone_0"] = (150.0, 150.0)
    game.threat_positions["dynamic_threat_0"] = (299.9, 100.0)
    game.headings["dynamic_threat_0"] = 0
    game._target_positions = [(0.0, 0.0)]
    game.shared_belief.fill(0.5)

    rejected_steps = 0
    for _ in range(NUM_HEADINGS // 2 + 2):
        threat_heading = game.available_headings("dynamic_threat_0")[0]
        drone_heading = game.available_headings("drone_0")[0]
        _, _, _, _, infos = game.step({
            "drone_0": FlightCommand(drone_heading),
            "dynamic_threat_0": FlightCommand(threat_heading),
        })
        rejected_steps += int(
            infos["dynamic_threat_0"]["boundary_action_rejected"]
        )
        assert game.available_headings("dynamic_threat_0")

    assert rejected_steps > 0
    assert game.threat_positions["dynamic_threat_0"][0] < 299.9


def test_render_distinguishes_correct_and_false_high_probability_marks():
    game = CooperativeSearchGame(
        CooperativeSearchConfig(num_drones=1, num_dynamic_threats=1, num_targets=1),
        render_mode="rgb_array",
    )
    try:
        game.reset(seed=29)
        game.show_ranges = False
        game.drone_positions["drone_0"] = (100.0, 100.0)
        game.threat_positions["dynamic_threat_0"] = (200.0, 200.0)
        game.shared_belief.fill(0.5)
        game.shared_belief[60, 60] = 0.9
        game.shared_belief[70, 70] = 0.9
        game._target_positions = [game._grid_cell_center(60, 60)]
        pygame.font.init()
        game._surface = pygame.Surface((game.width, game.height))
        game._font = pygame.font.Font(None, 28)
        game._small_font = pygame.font.Font(None, 21)
        game._draw()
        frame = np.transpose(pygame.surfarray.array3d(game._surface), (1, 0, 2))
        drone_x, drone_y = game._world_to_render((100.0, 100.0))
        threat_x, threat_y = game._world_to_render((200.0, 200.0))
        assert np.array_equal(frame[drone_y, drone_x], [60, 211, 62])
        assert np.array_equal(frame[threat_y, threat_x], [220, 38, 38])
        assert game.confirmed_target_count() == 1
        correct_marker_area = frame[295:311, 295:311]
        false_marker_area = frame[345:361, 345:361]
        assert np.any(np.all(correct_marker_area == [60, 211, 62], axis=-1))
        assert np.any(np.all(false_marker_area == [255, 48, 48], axis=-1))
    finally:
        game.close()


def test_drone_entering_threat_radius_is_eaten_and_penalized(tmp_path):
    _, table, env = _configured_env(tmp_path)
    try:
        env.reset(seed=43)
        eaten = "drone_0"
        threat = "dynamic_threat_0"
        env.game.drone_positions[eaten] = (100.0, 100.0)
        env.game.threat_positions[threat] = (110.0, 100.0)
        for index, drone in enumerate(env.drone_agents[1:], start=1):
            env.game.drone_positions[drone] = (220.0 + index * 20.0, 250.0)
        env.game._target_positions = [(280.0, 280.0)] * env.game_config.num_targets
        env.game.shared_belief.fill(0.5)

        actions = {
            agent: table.action_map[env.game.available_headings(agent)[0]]
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)

        assert eaten not in env.agents
        assert eaten not in env.game.agents
        assert eaten not in env.game.drone_positions
        assert eaten in env.game.dead_drone_positions
        assert terminations[eaten] is True
        assert not any(truncations.values())
        assert infos[eaten]["eaten_by_threat"] is True
        assert infos[eaten]["alive"] is False
        assert env._last_reward_breakdown["death"] == pytest.approx(-env.death_penalty_per_drone)
        assert np.count_nonzero(observations[eaten]) == 0
        assert sum(rewards.values()) == pytest.approx(env._last_team_reward)
        pygame.font.init()
        env.game._surface = pygame.Surface((env.game.width, env.game.height))
        env.game._font = pygame.font.Font(None, 28)
        env.game._small_font = pygame.font.Font(None, 21)
        env.game._draw()
        frame = np.transpose(
            pygame.surfarray.array3d(env.game._surface), (1, 0, 2)
        )
        assert frame.shape == (500, 800, 3)

        next_actions = {
            agent: table.action_map[env.game.available_headings(agent)[0]]
            for agent in env.agents
        }
        env.step(next_actions)
    finally:
        env.close()


def test_all_drones_eaten_truncates_episode(tmp_path):
    _, table, env = _configured_env(tmp_path)
    try:
        env.reset(seed=47)
        threat_position = (250.0, 250.0)
        env.game.threat_positions["dynamic_threat_0"] = threat_position
        for index, drone in enumerate(env.drone_agents):
            env.game.drone_positions[drone] = (250.0 + index, 250.0)
        env.game._target_positions = [(0.0, 0.0)] * env.game_config.num_targets
        env.game.shared_belief.fill(0.5)
        actions = {
            agent: table.action_map[env.game.available_headings(agent)[0]]
            for agent in env.agents
        }

        _, _, terminations, truncations, infos = env.step(actions)

        assert not env.agents
        assert not env.game.agents
        assert not env.game.drone_positions
        assert not any(terminations.values())
        assert all(truncations.values())
        assert env._last_reward_breakdown["death"] == pytest.approx(
            -env.death_penalty_per_drone * len(env.drone_agents)
        )
        assert all(infos[drone]["eaten_by_threat"] for drone in env.drone_agents)
    finally:
        env.close()


def test_bayesian_sensor_updates_one_shared_probability_database():
    game = CooperativeSearchGame(CooperativeSearchConfig(
        num_drones=2, num_dynamic_threats=0, num_targets=1
    ))
    assert game.config.detection_probability == 0.8
    assert game.config.false_alarm_probability == 0.2
    assert game._probability_from_counts(1, 0) == pytest.approx(0.8)
    assert game._probability_from_counts(0, 1) == pytest.approx(0.2)
    assert game._probability_from_counts(2, 0) == pytest.approx(0.9)
    assert game._probability_from_counts(0, 2) == pytest.approx(0.1)
    assert game._sensor_evidence_weight(0.0) == pytest.approx(1.0)
    assert game._sensor_evidence_weight(20.0) == pytest.approx(0.25)
    assert game._sensor_evidence_weight(40.0) == pytest.approx(0.0)
    game.reset(seed=2)
    game.drone_positions = {"drone_0": (150.0, 150.0), "drone_1": (280.0, 280.0)}
    game._target_positions = [(152.0, 150.0)]
    game.positive_counts.fill(0)
    game.negative_counts.fill(0)
    game.shared_belief.fill(0.5)
    game._observe_shared_belief()

    target_x, target_y = game._world_to_grid(game._target_positions[0])
    empty_x, empty_y = game._world_to_grid((120.0, 150.0))
    target_positive = float(game.positive_counts[target_y, target_x])
    target_negative = float(game.negative_counts[target_y, target_x])
    empty_positive = float(game.positive_counts[empty_y, empty_x])
    empty_negative = float(game.negative_counts[empty_y, empty_x])
    assert 0 < target_positive + target_negative <= 1
    assert 0 < empty_positive + empty_negative <= 1
    assert game.shared_belief[target_y, target_x] == pytest.approx(
        game._probability_from_counts(target_positive, target_negative)
    )
    assert game.shared_belief[empty_y, empty_x] == pytest.approx(
        game._probability_from_counts(empty_positive, empty_negative)
    )
    assert game.shared_belief[0, 0] == 0.5


def test_sensor_boolean_is_one_footprint_level_detection_report():
    game = CooperativeSearchGame(CooperativeSearchConfig(
        num_drones=1, num_dynamic_threats=0, num_targets=1
    ))
    game.reset(seed=20260726)
    game.drone_positions["drone_0"] = (150.0, 150.0)
    game._target_positions = [(280.0, 280.0)]
    game.positive_counts.fill(0)
    game.negative_counts.fill(0)
    game.shared_belief.fill(0.5)

    class PositiveCellsWithoutSensorReport:
        @staticmethod
        def random(size=None):
            return 0.99 if size is None else np.zeros(size)

    game._rng = PositiveCellsWithoutSensorReport()
    game._observe_shared_belief()
    assert np.count_nonzero(game.positive_counts) > 0
    assert game.detected_flags["drone_0"] is False

    game._rng = np.random.default_rng(20260726)
    samples = 10_000

    false_alarm_rate = np.mean([
        game._sample_detection_report(has_target_in_range=False)
        for _ in range(samples)
    ])
    detection_rate = np.mean([
        game._sample_detection_report(has_target_in_range=True)
        for _ in range(samples)
    ])

    assert false_alarm_rate == pytest.approx(
        game.config.false_alarm_probability, abs=0.02
    )
    assert detection_rate == pytest.approx(
        game.config.detection_probability, abs=0.02
    )


def test_render_palette_matches_blue_cyan_green_css_stops():
    probabilities = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    colors = CooperativeSearchGame._belief_rgb(probabilities)

    assert np.array_equal(colors[0], [19, 13, 104])
    assert np.array_equal(colors[20], [19, 13, 104])
    assert np.array_equal(colors[80], [59, 148, 160])
    assert np.array_equal(colors[-1], [60, 211, 62])
    assert len(np.unique(colors, axis=0)) > 75


def test_training_config_dimensions_turn_mask_and_hidden_size(tmp_path):
    conf, table, env = _configured_env(tmp_path)
    try:
        env.reset(seed=5, options={"episode_num": 3})
        assert table.n_actions == env.NUM_ACTIONS == 12
        assert conf.agents_config[0].obs_dim == env.observation_dim == 487
        assert conf.agents_config[0].hidden_dim == conf.learners_config[0].algo_config.hidden_dim
        search, evade = conf.stage_task_config.stages
        assert search.stage_name == "search"
        assert search.train_config.env_metadata["dynamic_threats_enabled"] is False
        assert search.clear_replay_buffer is True
        assert evade.stage_name == "evade"
        assert evade.train_config.env_metadata["dynamic_threats_enabled"] is True
        assert evade.clear_replay_buffer is True
        assert conf.env_config.metadata["save_probability_map_on_episode_end"] is False
        assert conf.env_config.metadata["probability_map_every_n_episodes"] == 100
        assert conf.eval_config.env_metadata["render_mode"] == "rgb_array"
        assert conf.eval_config.env_metadata["render_every_step"] is False
        assert conf.eval_config.env_metadata["save_probability_map_on_episode_end"] is True
        assert env.game.grid_shape == (100, 100)
        assert env.game_config.initial_formation_side_km == pytest.approx(30.0)
        assert env.game.drone_positions == {
            "drone_0": (0.0, 0.0),
            "drone_1": (30.0, 0.0),
            "drone_2": (0.0, 30.0),
            "drone_3": (30.0, 30.0),
        }
        assert env.game_config.map_width == env.game_config.map_height == 300
        assert env.game_config.map_width / env.game_config.belief_grid_width == 3
        assert env.game_config.map_height / env.game_config.belief_grid_height == 3
        assert env.observation_grid_scale == 2
        assert (env.state_grid_height, env.state_grid_width) == (50, 50)
        assert env.centralized_state_dim == 5028
        assert env.state().shape == (5028,)
        assert env.get_obs()["drone_0"].shape == (487,)
        assert env.get_obs()["dynamic_threat_0"].shape == (40,)
        assert np.flatnonzero(env.get_avail_agent_actions("drone_0")).tolist() == [1, 2]
        env.game.headings["drone_0"] = 0
        assert np.flatnonzero(env.get_avail_agent_actions("drone_0")).tolist() == [1]
    finally:
        env.close()


def test_stage_threat_switch_keeps_dimensions_and_disables_all_threat_effects(tmp_path):
    conf, table, env = _configured_env(tmp_path)
    try:
        state_dim = env.centralized_state_dim
        observation_dim = env.observation_dim
        env.apply_env_metadata({"dynamic_threats_enabled": False})
        env.reset(seed=53)
        env.game.drone_positions["drone_0"] = (100.0, 100.0)
        env.game.threat_positions["dynamic_threat_0"] = (110.0, 100.0)
        env.game._target_positions = [(280.0, 280.0)] * env.game_config.num_targets
        env.game.shared_belief.fill(0.5)

        observations = env.get_obs()
        channel_size = env.fov_size * env.fov_size
        threat_channel = observations["drone_0"][channel_size * 3:channel_size * 4]
        assert env.centralized_state_dim == state_dim == 5028
        assert env.observation_dim == observation_dim == 487
        assert np.count_nonzero(threat_channel) == 0
        assert np.count_nonzero(observations["dynamic_threat_0"]) == 0
        assert np.count_nonzero(env.state()[-len(env.threat_agents) * 4:]) == 0
        assert env._snapshot().threat_positions == {}

        actions = {
            agent: table.action_map[env.game.available_headings(agent)[0]]
            for agent in env.agents
        }
        _, _, terminations, truncations, infos = env.step(actions)
        assert "drone_0" in env.agents
        assert not terminations["drone_0"]
        assert not any(truncations.values())
        assert infos["drone_0"]["eaten_by_threat"] is False
        assert env._last_reward_breakdown["death"] == 0.0

        env.apply_env_metadata({"dynamic_threats_enabled": True})
        env.reset(seed=59)
        env.game.drone_positions["drone_0"] = (100.0, 100.0)
        env.game.threat_positions["dynamic_threat_0"] = (110.0, 100.0)
        for index, drone in enumerate(env.drone_agents[1:], start=1):
            env.game.drone_positions[drone] = (220.0 + index * 20.0, 250.0)
        env.game._target_positions = [(280.0, 280.0)] * env.game_config.num_targets
        env.game.shared_belief.fill(0.5)
        actions = {
            agent: table.action_map[env.game.available_headings(agent)[0]]
            for agent in env.agents
        }
        _, _, terminations, _, infos = env.step(actions)
        assert terminations["drone_0"]
        assert infos["drone_0"]["eaten_by_threat"] is True
        assert env._last_reward_breakdown["death"] == pytest.approx(-env.death_penalty_per_drone)
    finally:
        env.close()


def test_drone_observation_has_six_km_friendly_grid_and_60_km_threat_warning_field(tmp_path):
    _, _, env = _configured_env(tmp_path)
    try:
        env.reset(seed=17)
        observer = "drone_0"
        env.game.drone_positions[observer] = (150.0, 150.0)
        env.game.drone_positions["drone_1"] = (157.0, 150.0)
        env.game.threat_positions["dynamic_threat_0"] = (194.0, 150.0)
        for threat in env.threat_agents[1:]:
            env.game.threat_positions[threat] = (280.0, 280.0)

        observation = env.get_obs()[observer]
        channel_size = env.fov_size * env.fov_size
        friendly = observation[channel_size * 2:channel_size * 3].reshape(
            env.fov_size, env.fov_size
        )
        threats = observation[channel_size * 3:channel_size * 4].reshape(
            env.fov_size, env.fov_size
        )

        center = env.fov_size // 2
        assert friendly[center, center + 1] == 1.0
        assert friendly[center, center] == 0.0
        assert np.max(threats[:, center + 1:]) > np.max(threats[:, :center]) > 0.0

        env.game.threat_positions["dynamic_threat_0"] = (250.0, 150.0)
        observation = env.get_obs()[observer]
        threats = observation[channel_size * 3:channel_size * 4]
        assert np.count_nonzero(threats) == 0
    finally:
        env.close()


def test_boundary_mask_rejects_outward_heading_without_ending_episode():
    game = CooperativeSearchGame(CooperativeSearchConfig(
        num_drones=1, num_dynamic_threats=0, num_targets=1
    ))
    game.reset(seed=3)
    game.drone_positions["drone_0"] = (299.0, 150.0)
    game.headings["drone_0"] = 0
    assert not game.is_command_available("drone_0", FlightCommand(0))
    _, _, terminations, truncations, infos = game.step({"drone_0": FlightCommand(0)})
    assert game.drone_positions["drone_0"] == (299.0, 150.0)
    assert infos["drone_0"]["boundary_action_rejected"] is True
    assert not any(terminations.values())
    assert not any(truncations.values())


def test_boundary_mask_never_becomes_empty_and_emergency_turn_recovers():
    game = CooperativeSearchGame(CooperativeSearchConfig(
        num_drones=1, num_dynamic_threats=0, num_targets=1
    ))
    game.reset(seed=31)
    game.drone_positions["drone_0"] = (299.9, 150.0)
    game.headings["drone_0"] = 0
    game._target_positions = [(0.0, 0.0)]
    game.shared_belief.fill(0.5)

    rejected_steps = 0
    for _ in range(NUM_HEADINGS // 2 + 2):
        headings = game.available_headings("drone_0")
        assert headings
        _, _, terminations, truncations, infos = game.step(
            {"drone_0": FlightCommand(headings[0])}
        )
        rejected_steps += int(infos["drone_0"]["boundary_action_rejected"])
        assert not any(terminations.values())
        assert not any(truncations.values())

    assert rejected_steps > 0
    assert game.drone_positions["drone_0"][0] < 299.9


def test_recoverable_mask_survives_adversarial_boundary_seeking_actions():
    game = CooperativeSearchGame(CooperativeSearchConfig(
        num_drones=1, num_dynamic_threats=0, num_targets=1
    ))
    game.reset(seed=37)
    agent = "drone_0"

    for _ in range(2_000):
        headings = game.available_headings(agent)
        assert headings
        position = game.drone_positions[agent]

        def clearance(heading):
            x, y = game._predict_position(position, FlightCommand(heading))
            return min(x, game.config.map_width - x, y, game.config.map_height - y)

        # Always choose the legal action that most aggressively approaches a wall.
        heading = min(headings, key=clearance)
        candidate = game._predict_position(position, FlightCommand(heading))
        assert game._in_bounds(candidate)
        game.drone_positions[agent] = candidate
        game.headings[agent] = heading


def test_game_truncates_at_five_hundred_steps():
    game = CooperativeSearchGame(CooperativeSearchConfig(
        num_drones=1, num_dynamic_threats=0, num_targets=1
    ))
    game.reset(seed=4)
    game.drone_positions["drone_0"] = (250.0, 250.0)
    game._target_positions = [(0.0, 0.0)]
    game.shared_belief.fill(0.5)
    game.step_count = 499
    _, _, terminations, truncations, _ = game.step({"drone_0": FlightCommand(0)})
    assert not any(terminations.values())
    assert all(truncations.values())


def test_reward_uses_detection_quality_exploration_and_revisited_empty_cost(tmp_path):
    _, _, env = _configured_env(tmp_path)
    try:
        env.reset(seed=7)
        env.game.drone_positions = {
            "drone_0": (0.0, 100.0),
            "drone_1": (90.0, 100.0),
            "drone_2": (180.0, 100.0),
            "drone_3": (270.0, 100.0),
        }
        parts = env._reward_components(
            previous_correct_count=0,
            current_correct_count=1,
            false_detections=2,
            newly_explored_cells=3,
            frontier_progress=0.5,
            revisited_empty_coverage_ratio=0.75,
        )
        assert parts["correct_detection"] == pytest.approx(
            env._correct_detection_cumulative_reward(1)
        )
        assert parts["false_detection"] == pytest.approx(-2 * env.false_detection_penalty)
        assert parts["exploration"] == pytest.approx(
            3 * env.exploration_reward_per_cell
        )
        assert env.exploration_reward_per_cell * env.game.shared_belief.size == pytest.approx(8.2)
        assert parts["frontier_potential"] == pytest.approx(
            0.5 * env.frontier_potential_reward_weight
        )
        assert parts["revisited_empty"] == pytest.approx(
            -(
                env.revisited_empty_penalty_episode_limit
                / env.max_step_per_episode
                * 0.75
            )
        )
        assert "step" not in parts
        assert "communication" not in parts
        assert "probability" not in parts
        assert "success" not in parts

        schedule = env.correct_detection_reward_schedule()
        assert len(schedule) == env.game_config.num_targets
        assert sum(schedule) == pytest.approx(env.correct_detection_reward_total)
        assert env._correct_detection_cumulative_reward(8) == pytest.approx(
            0.5 * env.correct_detection_reward_total
        )
        assert schedule[-1] > schedule[0]

        connected_parts = parts.copy()
        env.game.drone_positions = {
            "drone_0": (0.0, 100.0),
            "drone_1": (100.0, 100.0),
            "drone_2": (200.0, 100.0),
            "drone_3": (300.0, 100.0),
        }
        disconnected_parts = env._reward_components(
            previous_correct_count=0,
            current_correct_count=1,
            false_detections=2,
            newly_explored_cells=3,
            frontier_progress=0.5,
            revisited_empty_coverage_ratio=0.75,
        )
        assert disconnected_parts == connected_parts

        env._last_team_reward = 4.0
        rewards = env.get_reward()
        assert sum(rewards[agent] for agent in env.drone_agents) == pytest.approx(4.0)
        assert all(rewards[agent] == 0.0 for agent in env.threat_agents)
    finally:
        env.close()


def test_cooperative_search_reports_completed_episode_reward_details(tmp_path):
    _, _, env = _configured_env(tmp_path)
    try:
        env._episode_reward_breakdown["correct_detection"] = 3.0

        details = env.get_episode_rewards_details()
        assert details is not None
        assert details["correct_detection"] == pytest.approx(3.0)
        assert set(details) == set(env._REWARD_KEYS)
        details["correct_detection"] = 99.0
        assert env._episode_reward_breakdown["correct_detection"] == pytest.approx(3.0)
    finally:
        env.close()


def test_revisited_empty_penalty_has_balanced_episode_limit(tmp_path):
    _, _, env = _configured_env(tmp_path)
    try:
        env.reset(seed=7)
        env.game.positive_counts.fill(0.0)
        env.game.negative_counts.fill(1.0)
        env.game.shared_belief.fill(env.game_config.min_belief_probability)
        env.game.drone_positions = {
            agent: (150.0, 150.0)
            for agent in env.drone_agents
        }

        empty_cell_count, ratio = env._revisited_empty_coverage(
            previous_observed=np.ones(env.game.grid_shape, dtype=bool)
        )
        assert empty_cell_count > 0
        assert env.sensor_footprint_cell_upper_bound == 729
        assert ratio == pytest.approx(
            empty_cell_count
            / (env.game_config.num_drones * env.sensor_footprint_cell_upper_bound)
        )
        assert 0.0 < ratio <= 1.0
        parts = env._reward_components(revisited_empty_coverage_ratio=ratio)
        assert parts["revisited_empty"] == pytest.approx(
            -(
                env.revisited_empty_penalty_episode_limit
                / env.max_step_per_episode
                * ratio
            )
        )
        assert env.revisited_empty_penalty_episode_limit == pytest.approx(15.0)

        # Cells first opened by the current action must not be penalized yet
        # when only previous-observed cells are considered by the diagnostic helper.
        empty_cell_count, ratio = env._revisited_empty_coverage(
            previous_observed=np.zeros(env.game.grid_shape, dtype=bool)
        )
        assert empty_cell_count == 0
        assert ratio == pytest.approx(0.0)

        newly_covered, sequential_ratio = env._sequential_search_settlement(
            previous_observed=np.zeros(env.game.grid_shape, dtype=bool)
        )
        assert newly_covered > 0
        # Same-step overlaps on newly opened confirmed-empty cells can still
        # contribute revisit after the first drone marks the cell observed.
        assert sequential_ratio >= 0.0

        env.game.positive_counts.fill(0.0)
        env.game.negative_counts.fill(0.0)
        env.game.shared_belief.fill(0.5)
        empty_cell_count, ratio = env._revisited_empty_coverage(
            previous_observed=np.zeros(env.game.grid_shape, dtype=bool)
        )
        assert empty_cell_count == 0
        assert ratio == pytest.approx(0.0)
        newly_covered, sequential_ratio = env._sequential_search_settlement(
            previous_observed=np.zeros(env.game.grid_shape, dtype=bool)
        )
        assert newly_covered > 0
        assert sequential_ratio == pytest.approx(0.0)
    finally:
        env.close()


def test_frontier_potential_rewards_motion_toward_unexplored_cells(tmp_path):
    _, _, env = _configured_env(tmp_path)
    try:
        env.reset(seed=7)
        previous_observed = np.ones(env.game.grid_shape, dtype=bool)
        previous_observed[20, 80] = False
        previous_position = (100.0, 61.5)

        env.game.drone_positions = {"drone_0": (103.0, 61.5)}
        toward = env._frontier_progress(
            previous_observed=previous_observed,
            previous_drone_positions={"drone_0": previous_position},
        )
        assert toward == pytest.approx(1.0)

        env.game.drone_positions = {"drone_0": (97.0, 61.5)}
        away = env._frontier_progress(
            previous_observed=previous_observed,
            previous_drone_positions={"drone_0": previous_position},
        )
        assert away == pytest.approx(-1.0)

        no_frontier = env._frontier_progress(
            previous_observed=np.ones(env.game.grid_shape, dtype=bool),
            previous_drone_positions={"drone_0": previous_position},
        )
        assert no_frontier == 0.0
    finally:
        env.close()


def test_detection_quality_reward_is_settled_once_per_target_or_false_cell(tmp_path):
    _, _, env = _configured_env(tmp_path)
    try:
        env.reset(seed=8)
        env.game.shared_belief.fill(0.5)
        env.game._target_positions = [
            (2.5 + 5.0 * index, 2.5)
            for index in range(env.game_config.num_targets)
        ]
        target_x, target_y = env.game._world_to_grid(env.game._target_positions[0])
        false_cell = (50, 50)
        env.game.shared_belief[target_y, target_x] = env.game_config.max_belief_probability
        env.game.shared_belief[false_cell[1], false_cell[0]] = env.game_config.max_belief_probability

        assert env._new_detection_events() == (1, 1)
        assert env._new_detection_events() == (0, 0)

        # 概率回落后再次升到上限也不能重复刷奖励或惩罚。
        env.game.shared_belief[target_y, target_x] = 0.5
        env.game.shared_belief[false_cell[1], false_cell[0]] = 0.5
        assert env._new_detection_events() == (0, 0)
        env.game.shared_belief[target_y, target_x] = env.game_config.max_belief_probability
        env.game.shared_belief[false_cell[1], false_cell[0]] = env.game_config.max_belief_probability
        assert env._new_detection_events() == (0, 0)
    finally:
        env.close()


def test_success_terminates_and_public_state_hides_target_truth(tmp_path):
    _, table, env = _configured_env(tmp_path)
    try:
        _, reset_infos = env.reset(seed=9, options={"episode_num": 8})
        position = env.game.drone_positions["drone_0"]
        env.game._target_positions = [position] * env.game_config.num_targets
        env.game.shared_belief.fill(0.5)
        env.game.positive_counts.fill(0)
        env.game.negative_counts.fill(0)
        target_x, target_y = env.game._world_to_grid(position)
        env.game.shared_belief[target_y, target_x] = env.game_config.max_belief_probability
        env.game.positive_counts[target_y, target_x] = 100.0
        actions = {agent: table.action_map[0] for agent in env.possible_agents}
        _, rewards, terminations, truncations, infos = env.step(actions)
        assert all(terminations.values())
        assert not any(truncations.values())
        assert sum(rewards.values()) > 0
        public_state = env.post_state.to_dict()
        public_text = repr(public_state).lower()
        assert "target_positions" not in public_text
        assert "target_id" not in public_text
        assert "discovered" not in public_text
        for info in [*reset_infos.values(), *infos.values()]:
            assert "target_positions" not in info
            assert "targets_found" not in info
    finally:
        env.close()


def test_probability_image_is_saved_every_100_evaluation_episodes(tmp_path):
    _, table, env = _configured_env(tmp_path)
    try:
        env.apply_env_metadata({
            "save_probability_map_on_episode_end": True,
            "probability_map_every_n_episodes": 100,
        })
        actions = {agent: table.action_map[0] for agent in env.possible_agents}
        env.stage_name = "search"

        env.reset(seed=13, options={"episode_num": 50})
        env.game.step_count = env.max_step_per_episode - 1
        env.step(actions)
        safe_task_id = str(task_id)
        assert not list(tmp_path.glob("probability_*.png"))

        env.reset(seed=13, options={"episode_num": 100})
        env.game.step_count = env.max_step_per_episode - 1
        env.step(actions)
        expected = tmp_path / (
            f"probability_{safe_task_id}_search_episode_0100.png"
        )
        assert expected.is_file()
        assert set(tmp_path.glob("probability_*.png")) == {expected}

        env.stage_name = "evade"
        env.reset(seed=13, options={"episode_num": 1})
        env.game.step_count = env.max_step_per_episode - 1
        env.step(actions)
        assert set(tmp_path.glob("probability_*.png")) == {expected}

        env.game.shared_belief[:] = np.linspace(0.0, 1.0, 100, dtype=np.float32)[None, :]
        env.game._target_positions = [(271.5, 151.5)] * env.game_config.num_targets
        env.stage_name = "search"
        env.episode_num = 100
        env._save_probability_map()
        from PIL import Image
        pixels = np.asarray(Image.open(expected).crop((0, 0, 500, 500)))
        colors = np.unique(pixels.reshape(-1, 3), axis=0)
        assert 70 < len(colors) <= 103
        assert np.array_equal(pixels[0, 0], [19, 13, 104])
        assert np.all(pixels[:, 250:255] == pixels[:, 250:251])
        assert not np.array_equal(pixels[0, 254], pixels[0, 255])
        assert np.any(np.all(pixels == [60, 211, 62], axis=-1))
        assert np.any(np.all(pixels == [255, 48, 48], axis=-1))
    finally:
        env.close()

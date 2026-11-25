"""Polar observations, independent rewards and real singleton-policy integration."""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from config import ThreatRoamingConfig
from common.exception import EnvironmentException, ModelCheckpointCompatibilityException
from envs.games.cooperative_search.threat_roaming import ThreatRoaming
from envs.sense.threat_roaming_observation import ThreatRoamingObservation
from envs.sense.cooperative_search_partition_threat_env import CooperativeSearchPartitionThreatEnv
from envs.sense.cooperative_search_partition_env import CooperativeSearchPartitionEnv
from algos.modules.agents.rnn_agent import RNNAgent
from algos.training.qmix_learner import QMixerLearner
from utils.toml_config import TomlTrainingConfig
from utils.module_utils import inject_env_dims
from scheduler.rollout_sample import RolloutSampleWorker
from scheduler.model import AgentCompose, ActionParam
from test_agent_grouping import _group_worker, _group_episode_info, _RecordingReplayBuffer

ROOT = Path(__file__).resolve().parents[1]


def subject(name="partition_threat_qmix"):
    return TomlTrainingConfig(ROOT/f"subjects/cooperative_search/cooperative_search_{name}.toml", project_root=ROOT)


def toy(peers, heading=0):
    return SimpleNamespace(threat_positions={"self": (0., 0.), **peers}, headings={"self": heading})


@pytest.mark.parametrize("distance,angle,row,col,expected", [
    (5, 0, 0, 0, .95), (15, math.pi/2, 1, 4, .85),
    (95, math.pi, 9, 8, .05), (0, 0, 0, 0, 1),
    (100, 0, 9, 0, 0), (101, 0, 9, 0, 0),
])
def test_polar_cells(distance, angle, row, col, expected):
    encoder = ThreatRoamingObservation(ThreatRoamingConfig())
    game = toy({"other": (distance*math.cos(angle), distance*math.sin(angle))})
    grid = encoder.peer_grid(game, "self")
    assert grid.shape == (1, 10, 16)
    assert grid[0, row, col] == pytest.approx(expected)
    assert grid.sum() == pytest.approx(max(0, 1-distance/100))
    np.testing.assert_array_equal(encoder.representation(grid), grid.ravel())


def test_polar_interpolation_wrap_max_and_rotation():
    encoder = ThreatRoamingObservation(ThreatRoamingConfig())
    angle = -math.pi/16
    p = (10*math.cos(angle), 10*math.sin(angle))
    grid = encoder.peer_grid(toy({"a": p}), "self")
    assert np.count_nonzero(grid > 1e-6) == 4
    np.testing.assert_allclose(grid[0, :2, [0, 15]], .225)
    np.testing.assert_array_equal(grid, encoder.peer_grid(toy({"a": p, "b": p}), "self"))
    rotation = math.pi/6
    rotated = (p[0]*math.cos(rotation)-p[1]*math.sin(rotation), p[0]*math.sin(rotation)+p[1]*math.cos(rotation))
    np.testing.assert_allclose(grid, encoder.peer_grid(toy({"a": rotated}, heading=1), "self"), atol=1e-6)
    np.testing.assert_array_equal(encoder.peer_grid(toy({}, heading=None), "self"), 0)
    near_left = encoder.peer_grid(toy({"a": (15, -1e-8)}), "self")
    near_right = encoder.peer_grid(toy({"a": (15, 1e-8)}), "self")
    np.testing.assert_allclose(near_left, near_right, atol=1e-7)


def test_waypoint_rewards_old_goal_timeout_and_spacing():
    config = ThreatRoamingConfig(waypoint_timeout_steps=2, circle_center_min_shift_km=0, circle_center_max_shift_km=1000)
    roaming = ThreatRoaming(config, 600, 600)
    rng = np.random.default_rng(2)
    old = {"a": (300., 300.), "b": (400., 400.)}
    roaming.reset(old, rng)
    roaming.waypoints["a"] = (302.4, 300.)
    new = {**old, "a": (302.4, 300.)}
    roaming.step(old, new, 2.4, rng)
    assert roaming.breakdowns["a"]["waypoint_progress"] == pytest.approx(.2)
    assert roaming.breakdowns["a"]["waypoint_arrival"] == .5
    assert roaming.completed["a"] == 1 and roaming.ages["a"] == 0
    assert 10 <= math.dist(roaming.centers["a"], roaming.waypoints["a"]) <= 25
    assert math.dist(roaming.centers["a"], roaming.waypoints["a"]) <= 50
    roaming.waypoints["b"] = (300., 300.)
    roaming.step(new, new, 2.4, rng)
    assert roaming.ages["b"] == 0  # timed out, no arrival bonus
    assert roaming.completed["b"] == 0
    for distance in (40, 30, 20, 15, 5):
        positions = {"a": (300., 300.), "b": (300.+distance, 300.)}
        roaming.step(positions, positions, 2.4, rng)
        assert roaming.breakdowns["a"]["peer_clearance"] == pytest.approx(-.5*max(0,1-distance/30)**2)
        assert roaming.breakdowns["a"]["peer_danger"] == pytest.approx(-2*max(0,1-distance/15)**2)
    positions = {"a": (400., 300.), "b": (200., 300.)}
    old_center = roaming.centers["a"]
    expected_penalty = -float(np.clip((math.dist(positions["a"], old_center)-50)/50, 0, 1))**2
    roaming.step(positions, positions, 2.4, rng)
    assert roaming.breakdowns["a"]["circle_outside"] == pytest.approx(expected_penalty)
    expected = roaming.breakdowns["a"].copy()
    # Adding a remote peer changes neither the nearest distance nor this entity's penalties.
    bigger = ThreatRoaming(config, 600, 600)
    extended = {**positions, "far": (0.,0.)}
    bigger.reset(extended, np.random.default_rng(9))
    bigger.waypoints["a"] = roaming.waypoints["a"]
    bigger.centers["a"] = roaming.centers["a"]
    bigger.step(extended, extended, 2.4, np.random.default_rng(10))
    for key in ("circle_outside", "peer_clearance", "peer_danger"):
        assert bigger.breakdowns["a"][key] == expected[key]


@pytest.mark.parametrize("kwargs", [{"radial_bins": 0}, {"circle_radius_km": 301},
    {"waypoint_min_distance_km": 50}, {"danger_distance_km": 30}, {"arrival_reward": -1}])
def test_invalid_roaming_config(kwargs):
    with pytest.raises(EnvironmentException):
        ThreatRoaming(replace(ThreatRoamingConfig(), **kwargs), 600, 600)


def build_agent(config, model=None):
    return RNNAgent(agent_type="rnn", config=deepcopy(config), agent_net_source="file" if model else "blanco",
                    model_file_path=str(model) if model else None, device="cpu")


def test_real_rollout_checkpoint_fixed_scene_and_qmix_update(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    conf = subject()
    env_config = deepcopy(conf.env_config)
    env_config.metadata["render_mode"] = None
    env_config.metadata["render_every_step"] = False
    env_config.max_step_per_episode = 4
    env_config.metadata["game_config"]["max_steps"] = 4
    env = CooperativeSearchPartitionThreatEnv(env_config)
    formal = None
    try:
        agent = build_agent(conf.agents_config[0])
        inject_env_dims(env, [agent])
        calls, worlds = [], []
        hidden_inputs = {}
        original_actions, original_step = agent.get_actions, env.step
        def counted_actions(**kwargs):
            calls.append((tuple(kwargs["agent_list"]), id(agent.agent_net)))
            aid = kwargs["agent_list"][0]
            hidden_inputs.setdefault(aid, []).append(kwargs["hidden_in"][aid].clone())
            return original_actions(**kwargs)
        def counted_step(actions):
            worlds.append(tuple(actions))
            return original_step(actions)
        monkeypatch.setattr(agent, "get_actions", counted_actions)
        monkeypatch.setattr(env, "step", counted_step)
        worker = _group_worker(env, agent)
        episodes = worker.rolling_parallel_episode(_group_episode_info("dynamic_threats"), step_limit=4)["dynamic_threats"]
        assert len(calls) == 32 and len(worlds) == 4
        assert len({net for _,net in calls}) == 1
        assert all(len(ids)==1 for ids,_ in calls)
        assert all(len(ids)==8 for ids in worlds)
        assert len(episodes) == 8 and all(ep.seq_len==4 for ep in episodes)
        assert len({hidden_inputs[a][0].data_ptr() for a in hidden_inputs}) == 8
        assert any(not torch.equal(hidden_inputs[a][1], hidden_inputs[b][1]) for a,b in zip(env.threat_agents,env.threat_agents[1:]))
        for ep in episodes:
            aid = ep.agent_ids[0]
            assert ep.states[0].shape == (173,)
            assert ep.terminations[aid][-1] is False
            assert ep.next_active_agent_mask[aid][-1] == 1
        learner = QMixerLearner(algo_type="qmix", config=deepcopy(conf.learners_config[0]), agent=agent, env=env, device="cpu")
        before = {k:v.clone() for k,v in learner.agent_net.state_dict().items()}
        learner.update(episodes)
        assert learner.update_cnt == 1
        assert any(not torch.equal(before[k],v) for k,v in learner.agent_net.state_dict().items())
        checkpoint = tmp_path/"roaming_agent.pth"
        torch.save(learner.agent_net.state_dict(), checkpoint)

        deployed = subject("qmix_partitions")
        deployed.env_config.metadata["render_mode"] = None
        deployed.env_config.metadata["render_every_step"] = False
        deployed.env_config.max_step_per_episode = 3
        deployed.env_config.metadata["game_config"]["max_steps"] = 3
        formal = CooperativeSearchPartitionEnv(deployed.env_config)
        threat_agent = build_agent(deployed.agents_config[1], checkpoint)
        uav_agent = build_agent(deployed.agents_config[0])
        inject_env_dims(formal, [uav_agent, threat_agent])
        formal.reset(seed=11)
        # Same game state produces exactly the same observation in both deployment paths.
        for aid in env.threat_agents:
            np.testing.assert_array_equal(env.observation(formal.game, aid), formal.get_obs()[aid])
            np.testing.assert_array_equal(formal.state_for_agent_ids(agent_name="dynamic_threats",agent_ids=(aid,)), formal.get_obs()[aid])
        fixed_calls, actions_per_step = [], []
        original_fixed, original_formal_step = threat_agent.get_actions, formal.step
        def fixed_actions(**kwargs):
            assert kwargs["epsilon"] == 0 and kwargs["deterministic"]
            fixed_calls.append(tuple(kwargs["agent_list"]))
            return original_fixed(**kwargs)
        def formal_step(actions):
            actions_per_step.append(set(actions))
            return original_formal_step(actions)
        monkeypatch.setattr(threat_agent, "get_actions", fixed_actions)
        monkeypatch.setattr(formal, "step", formal_step)
        replay = _RecordingReplayBuffer()
        fixed_before = {k:v.clone() for k,v in threat_agent.agent_net.state_dict().items()}
        mixed = RolloutSampleWorker(env=formal, task_manager=SimpleNamespace(accept_sample_task_result=lambda *args,**kwargs:True),
            replay_buffer=replay, policy_agents=[uav_agent,threat_agent], agent_compose=AgentCompose(step_agent=[uav_agent,threat_agent]))
        fixed_info = _group_episode_info("uav_team")
        fixed_info.actions_param["dynamic_threats"] = ActionParam(epsilon=0.0, deterministic=True, temperature=1.0)
        result = mixed.rolling_parallel_episode(fixed_info, step_limit=3)
        assert "dynamic_threats" not in result
        assert len(fixed_calls)==24 and all(len(ids)==1 for ids in fixed_calls)
        assert len(actions_per_step)==3 and all(len(ids)==24 for ids in actions_per_step)
        assert all(torch.equal(fixed_before[k],v) for k,v in threat_agent.agent_net.state_dict().items())
    finally:
        env.close()
        if formal is not None:
            formal.close()


def test_spawn_seed_observation_purity_and_no_target_leak():
    conf = subject()
    env = CooperativeSearchPartitionThreatEnv(conf.env_config)
    try:
        first,_ = env.reset(seed=17)
        points = dict(env.game.threat_positions)
        goals = dict(env.game.threat_roaming.waypoints)
        centers = dict(env.game.threat_roaming.centers)
        assert centers != points
        assert all(10 <= math.dist(centers[a], goal) <= 25 for a, goal in goals.items())
        for region in range(4):
            assert sum(env.game.region_of_position(p) == region for p in centers.values()) == 2
        for region in range(4):
            assert sum(env.game.region_of_position(p)==region for p in points.values()) == 2
        assert any(math.dist(p,(300,300))>50 for p in points.values())
        assert all(math.dist(a,b)>=60 for i,a in enumerate(points.values()) for b in list(points.values())[i+1:])
        assert first[env.threat_agents[0]].shape == (173,)
        rng = deepcopy(env.game._rng.bit_generator.state)
        env.get_obs(); env.get_obs()
        assert env.game._rng.bit_generator.state == rng
        assert goals == env.game.threat_roaming.waypoints
        env.game._target_positions = [(0.,0.)]*40
        for aid,obs in first.items():
            np.testing.assert_array_equal(obs, env.get_obs()[aid])
        again,_ = env.reset(seed=17)
        assert points == env.game.threat_positions and goals == env.game.threat_roaming.waypoints
        assert centers == env.game.threat_roaming.centers
        for aid,obs in first.items():
            np.testing.assert_array_equal(obs, again[aid])
        image = env.game.render()
        assert image is not None and image.ndim==3
    finally:
        env.close()


def test_old_joint_checkpoint_rejected(tmp_path):
    cfg = subject().agents_config[0]
    old_config = deepcopy(cfg)
    old_config.agent_names = [g[0] for g in old_config.agent_groups]
    old_config.agent_groups = []
    old_config.obs_dim = 145
    old_config.obs_agent_id = True
    old = build_agent(old_config)
    path = tmp_path / "old_joint.pth"
    torch.save(old.agent_net.state_dict(), path)
    with pytest.raises(ModelCheckpointCompatibilityException):
        build_agent(cfg, path)


def test_observation_cache_invalidation_and_copy(monkeypatch):
    env = CooperativeSearchPartitionThreatEnv(subject().env_config)
    try:
        env.reset(seed=1)
        encoder = env.observation
        aid = env.threat_agents[0]
        calls = []
        original = encoder._encode
        def counted(game, agent):
            calls.append(agent)
            return original(game, agent)
        monkeypatch.setattr(encoder, "_encode", counted)
        # Clear data populated by reset before counting repeated reads.
        encoder._cache_key = None
        first = encoder(env.game, aid)
        first[:] = 100
        second = encoder(env.game, aid)
        assert len(calls) == 1 and np.max(second) <= 1
        env.game.threat_roaming.ages[aid] += 1
        third = encoder(env.game, aid)
        assert len(calls) == 2 and third[9] > second[9]
        env.game.threat_positions[aid] = (300., 300.)
        assert encoder(env.game, aid)[0] == .5
        assert len(calls) == 3
    finally:
        env.close()


@pytest.mark.parametrize("class_name", ["ThreatRoamingEvaluation", "ThreatRoamingEvalControl"])
def test_roaming_report_exports_metrics(tmp_path, monkeypatch, class_name):
    import csv
    from subjects.cooperative_search import evaluation
    monkeypatch.chdir(tmp_path)
    report = getattr(evaluation, class_name).__new__(getattr(evaluation, class_name))
    metrics = {key: .25 for key in report.metric_fields if key not in ("episode", "steps", "total_reward")}
    metrics["steps"] = 12
    record = report.episode_report(1, total_rewards={"dynamic_threats": -3.}, episode_metrics=metrics)
    report.episode_results = [record]
    output = report.finished_report()
    with Path(output["episode_metrics_csv"]).open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["inside_roaming_circle"] == "0.25"
    assert rows[0]["total_reward"] == "-3.0"
    assert rows[0]["steps"] == "12"
    assert set(rows[0]) == set(report.metric_fields)


def test_subject_budget_and_metric_totals():
    from envs.action.model import Action
    from scheduler.utils.business_metrics import EpisodeMetricCollector
    conf = subject()
    assert conf.train_config.max_episode == 500
    assert conf.train_config.epsilon_decay_episodes == 375
    assert conf.train_config.mid_eval_episode_interval == 100
    assert conf.agents_config[0].obs_agent_id is False
    env = CooperativeSearchPartitionThreatEnv(conf.env_config)
    try:
        env.reset(seed=11)
        collector = EpisodeMetricCollector(conf.eval_config.metric_outputs)
        total = 0.0
        for _ in range(3):
            actions = {}
            for aid in env.threat_agents:
                heading = env.game.available_headings(aid)[0]
                actions[aid] = Action(action_key="heading", parameters={"value": heading}, target=-2, action_idx=heading)
            _, rewards, _, _, infos = env.step(actions)
            collector.add_infos(infos)
            total += sum(rewards.values())
        metrics = collector.finish({})
        assert metrics["steps"] == 3
        assert 0 <= metrics["inside_roaming_circle"] <= 1
        assert 0 <= metrics["dangerous_peer_proximity"] <= 1
        assert metrics["waypoints_completed"] == sum(env.game.threat_roaming.completed.values())
        assert sum(metrics[key] for key in env._REWARD_KEYS) == pytest.approx(total)
    finally:
        env.close()


@pytest.mark.parametrize("position", [(0.,0.), (600.,0.), (0.,600.), (600.,600.), (1.,300.), (300.,599.)])
def test_local_waypoints_stay_inside_map_at_boundaries(position):
    roaming = ThreatRoaming(ThreatRoamingConfig(), 600, 600)
    rng = np.random.default_rng(15)
    for _ in range(200):
        goal = roaming._new_waypoint(position, rng)
        assert 0 <= goal[0] <= 600 and 0 <= goal[1] <= 600
        assert 10 <= math.dist(position, goal) <= 25


def test_waypoints_uniform_about_assigned_circle():
    roaming = ThreatRoaming(ThreatRoamingConfig(), 600, 600)
    for own in ((100.,100.), (500.,100.), (100.,500.), (500.,500.)):
        rng = np.random.default_rng(17)
        displacements = np.array([np.subtract(roaming._new_waypoint(own, rng), own) for _ in range(2000)])
        assert np.all(np.abs(displacements.mean(axis=0)) < 2.)
    # Translating an interior entity translates its goals exactly, for the same random stream.
    first_rng, second_rng = np.random.default_rng(18), np.random.default_rng(18)
    for _ in range(50):
        first = roaming._new_waypoint((100.,100.), first_rng)
        second = roaming._new_waypoint((500.,500.), second_rng)
        np.testing.assert_allclose(np.subtract(second, first), (400.,400.), atol=1e-12)
    # A temporary circle can cross a responsibility boundary.
    rng = np.random.default_rng(19)
    assert any(roaming._new_waypoint((295.,150.), rng)[0] > 300 for _ in range(100))


def test_joint_refresh_settles_old_circle_before_reassignment():
    config = ThreatRoamingConfig(circle_center_min_shift_km=0, circle_center_max_shift_km=1000)
    roaming = ThreatRoaming(config, 600, 600)
    rng = np.random.default_rng(20)
    previous = {"a": (100.,100.), "b": (500.,400.)}
    roaming.reset(previous, rng)
    old_slots = dict(roaming.center_slots)
    roaming.centers["a"] = (100.,100.)
    roaming.waypoints["a"] = (125.,100.)
    roaming.layout_age = roaming.layout_refresh_after - 1
    current = {**previous, "a": (160.,100.)}
    roaming.step(previous, current, 2.4, rng)
    assert roaming.breakdowns["a"]["circle_outside"] == pytest.approx(-.04)
    assert roaming.metrics["a"]["inside_roaming_circle"] == 0
    assert all(roaming.ages[a] == 0 for a in current)
    assert all(roaming.center_slots[a] != old_slots[a] for a in current)
    assert 10 <= math.dist(roaming.centers["a"], roaming.waypoints["a"]) <= 25


def test_stratified_circle_observation_and_cache():
    env = CooperativeSearchPartitionThreatEnv(subject().env_config)
    try:
        observations,_ = env.reset(seed=20)
        centers = dict(env.game.threat_roaming.centers)
        for aid, obs in observations.items():
            expected = np.subtract(centers[aid], env.game.threat_positions[aid]) / 600
            np.testing.assert_allclose(obs[4:6], expected, atol=1e-7)
        aid = env.threat_agents[0]
        original = env.game.threat_positions[aid]
        env.game.threat_roaming.centers[aid] = (original[0]+6., original[1]-12.)
        obs = env.get_obs()[aid]
        np.testing.assert_allclose(obs[4:6], [.01,-.02])
        assert env.game.threat_roaming.centers[env.threat_agents[1]] == centers[env.threat_agents[1]]
    finally:
        env.close()


def test_joint_large_scale_refresh_intervals_and_cross_slot_assignment():
    roaming = ThreatRoaming(ThreatRoamingConfig(), 600, 600, region_rows=2, region_cols=2)
    rng = np.random.default_rng(23)
    positions = {str(i): (0.,0.) for i in range(8)}
    roaming.reset(positions, rng)
    intervals = []
    last = 0
    for step in range(1, 1001):
        old = dict(roaming.centers)
        old_slots = dict(roaming.center_slots)
        roaming.step(positions, positions, 2.4, rng)
        if roaming.centers != old:
            intervals.append(step-last)
            last = step
            assert all(old_slots[a] != roaming.center_slots[a] for a in positions)
            assert all(100-1e-8 <= math.dist(old[a], roaming.centers[a]) <= 200+1e-8 for a in positions)
    assert all(40 <= value <= 80 for value in intervals)
    assert len(intervals) >= 12 and len(set(intervals)) > 1
    assert roaming.config.waypoint_timeout_steps == 200
    assert ThreatRoamingObservation(roaming.config).observation_dim == 173


@pytest.mark.parametrize("count", [8, 12, 16])
def test_stratified_circles_stay_distributed_even_when_agents_cluster(count):
    roaming = ThreatRoaming(ThreatRoamingConfig(), 600, 600, region_rows=2, region_cols=2)
    rng = np.random.default_rng(25)
    positions = {f"dynamic_threat_{i}": (300.,300.) for i in range(count)}
    roaming.reset(positions, rng)
    assert len(set(roaming.center_slots.values())) == count
    for _ in range(100):
        centers = list(roaming.centers.values())
        assert all(math.dist(a,b) >= 100-1e-8 for i,a in enumerate(centers) for b in centers[i+1:])
        goals = list(roaming.waypoints.values())
        assert all(math.dist(a,b) >= 50-1e-8 for i,a in enumerate(goals) for b in goals[i+1:])
        for row in range(2):
            for col in range(2):
                assert sum(int(p[0]//300)==col and int(p[1]//300)==row for p in centers) == count//4
        for aid,center in roaming.centers.items():
            x0,x1,y0,y1 = roaming.center_slots[aid]
            assert x0+50-1e-8 <= center[0] <= x1-50+1e-8
            assert y0+50-1e-8 <= center[1] <= y1-50+1e-8
        before = dict(roaming.centers)
        roaming.step(positions, positions, 2.4, rng)
        for aid,center in roaming.centers.items():
            distance = math.dist(before[aid],center)
            assert distance == 0 or 100-1e-8 <= distance <= 200+1e-8


def test_infeasible_stratified_circles_fail_explicitly():
    roaming = ThreatRoaming(ThreatRoamingConfig(circle_radius_km=100),600,600,region_rows=2,region_cols=2)
    with pytest.raises(EnvironmentException, match="cannot fit"):
        roaming.reset({str(i):(300.,300.) for i in range(16)}, np.random.default_rng(1))


def test_arrival_only_changes_waypoint_without_early_circle_refresh():
    roaming = ThreatRoaming(ThreatRoamingConfig(), 600, 600, region_rows=2, region_cols=2)
    positions = {str(i): (0.,0.) for i in range(8)}
    rng = np.random.default_rng(29)
    roaming.reset(positions, rng)
    before = dict(roaming.centers)
    roaming.waypoints["0"] = positions["0"]
    roaming.step(positions, positions, 2.4, rng)
    assert roaming.completed["0"] == 1
    assert roaming.waypoints["0"] != positions["0"]
    assert roaming.centers == before
    assert roaming.layout_age == 1


def test_impossible_cross_slot_refresh_is_atomic_and_explicit():
    config = ThreatRoamingConfig(circle_center_min_shift_km=800, circle_center_max_shift_km=801)
    roaming = ThreatRoaming(config,600,600,region_rows=2,region_cols=2)
    rng = np.random.default_rng(30)
    roaming.reset({str(i):(0.,0.) for i in range(8)},rng)
    before, slots = dict(roaming.centers),dict(roaming.center_slots)
    with pytest.raises(EnvironmentException,match="cannot refresh"):
        roaming._refresh_centers(rng)
    assert roaming.centers == before and roaming.center_slots == slots


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("name", ["qmix_partitions", "qmix_partitions_bundled"])
def test_partition_fixed_roaming_rollout(tmp_path, monkeypatch, enabled, name):
    """Eight singleton policies load the training architecture in both course stages."""

    torch.set_num_threads(1)
    conf = subject(name)
    conf.env_config.metadata.update(render_mode=None, render_every_step=False,
                                    dynamic_threats_enabled=enabled)
    conf.env_config.max_step_per_episode = 4
    conf.env_config.metadata["game_config"]["max_steps"] = 4
    conf.env_config.metadata["game_config"]["threat_roaming"].update(
        waypoint_refresh_min_steps=2, waypoint_refresh_max_steps=2)
    checkpoint = tmp_path / "roaming_agent.pth"
    trained = build_agent(subject().agents_config[0])
    torch.save(trained.agent_net.state_dict(), checkpoint)
    env = CooperativeSearchPartitionEnv(conf.env_config)
    try:
        threat = build_agent(conf.agents_config[1], checkpoint)
        uav = build_agent(conf.agents_config[0])
        inject_env_dims(env, [uav, threat])
        env.reset(seed=11)
        encoder = ThreatRoamingObservation(env.game_config.threat_roaming)
        initial_centers = dict(env.game.threat_roaming.centers)
        for aid in env.threat_agents:
            obs = env.get_obs()[aid]
            assert obs.shape == (173,)
            expected = encoder(env.game, aid) if enabled else np.zeros(173)
            np.testing.assert_array_equal(obs, expected)
            np.testing.assert_array_equal(env.state_for_agent_ids(
                agent_name="dynamic_threats", agent_ids=(aid,)), obs)
        calls = []
        original = threat.get_actions
        def actions(**kwargs):
            assert kwargs["epsilon"] == 0 and kwargs["deterministic"]
            calls.append(tuple(kwargs["agent_list"]))
            return original(**kwargs)
        monkeypatch.setattr(threat, "get_actions", actions)
        weights = {k: v.clone() for k, v in threat.agent_net.state_dict().items()}
        worker = RolloutSampleWorker(env=env,
            task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
            replay_buffer=_RecordingReplayBuffer(), policy_agents=[uav, threat],
            agent_compose=AgentCompose(step_agent=[uav, threat]))
        info = _group_episode_info("uav_team")
        info.actions_param["dynamic_threats"] = ActionParam(epsilon=0, deterministic=True, temperature=1)
        result = worker.rolling_parallel_episode(info, step_limit=4)
        assert "dynamic_threats" not in result
        assert len(calls) == 32 and all(len(ids) == 1 for ids in calls)
        assert all(torch.equal(weights[k], v) for k, v in threat.agent_net.state_dict().items())
        if enabled:
            assert env.game.threat_roaming.centers != initial_centers
        else:
            assert env.game.threat_roaming.layout_age == 0
        assert env.state().shape == (608,)
    finally:
        env.close()

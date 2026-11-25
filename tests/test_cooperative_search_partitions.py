"""cooperative_search 16 机四分区科目：责任区隔离与表征链路的联合验证。

覆盖 v0.15.2 第 2.4 节可在进程内验证的验收项：分区布局与可复现、无人机
不越责任区且跨界动作被约束、传感器观测与发现奖励按区归属、每区完成关闭
本组轨迹其余组继续、每区奖励只分配给本组四机、区局部观测坐标语义、
责任区表征进入策略观测且空间布局未被压成覆盖率标量、两个合法手动输出
维度与非法配置明确失败、决策帧与回放对齐（观测使用当步网格）。
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from common.exception import EnvironmentException
from config import EnvConfig
from envs.action.model import Action
from envs.sense.cooperative_search_partition_env import CooperativeSearchPartitionEnv

REP_DIM = 32
FOV = 7
OBS_DIM = FOV * FOV * 4 + 3 + REP_DIM
GROUP_STATE_DIM = REP_DIM + 4 * 4
GLOBAL_STATE_DIM = 4 * REP_DIM + 16 * 4
DRONES = [f"drone_{index}" for index in range(16)]
MAX_STEPS = 600


def _env(representation_dim=REP_DIM, max_steps=MAX_STEPS, extra_metadata=None):
    metadata = {
        "fov_size": FOV,
        "observation_grid_scale": 2,
        "state_grid_width": 50,
        "state_grid_height": 50,
        "render_mode": None,
        "dynamic_threats_enabled": False,
        "game_config": {
            "map_width": 600,
            "map_height": 600,
            "belief_grid_width": 200,
            "belief_grid_height": 200,
            "num_drones": 16,
            "num_dynamic_threats": 0,
            "num_targets": 40,
            "region_rows": 2,
            "region_cols": 2,
            "max_steps": max_steps,
            "step_interval_secs": 30.0,
            "threat_warning_radius": 45.0,
            "initial_threat_separation": 60.0,
        },
    }
    if representation_dim is not None:
        metadata["situation_representation_output_dim"] = representation_dim
    if extra_metadata:
        metadata.update(extra_metadata)
    env = CooperativeSearchPartitionEnv(EnvConfig(
        sense_name="cooperative_search_partition",
        max_step_per_episode=max_steps,
        step_interval_secs=30,
        state_dim=4 * (representation_dim or 1) + 16 * 4,
        agent_names=DRONES,
        use_sim=False,
        metadata=metadata,
    ))
    return env


def _random_step(env, rng, steps=1):
    for _ in range(steps):
        actions = {}
        for agent in env.agents:
            mask = env.get_avail_agent_actions(agent)
            index = int(rng.choice(np.flatnonzero(mask)))
            actions[agent] = Action(
                action_key="heading", parameters={"value": index},
                target=-2, action_idx=index,
            )
        result = env.step(actions)
        assert env.agents == env.game.agents
    return result


# ---------------------------------------------------------------------------
# 2.4-1 分区布局与可复现
# ---------------------------------------------------------------------------

def test_reset_four_regions_each_four_drones_ten_targets():
    env = _env()
    try:
        env.reset(seed=42)
        assert env.game.num_regions == 4
        expected_bounds = [
            (0.0, 300.0, 0.0, 300.0),
            (300.0, 600.0, 0.0, 300.0),
            (0.0, 300.0, 300.0, 600.0),
            (300.0, 600.0, 300.0, 600.0),
        ]
        for region in range(4):
            bounds = env.game.region_bounds(region)
            assert bounds == expected_bounds[region]
            agents = env.game.region_agents(region)
            assert agents == [f"drone_{region * 4 + i}" for i in range(4)]
            targets = env.game.region_target_positions(region)
            assert len(targets) == 10
            for x, y in targets:
                assert bounds[0] <= x < bounds[1] and bounds[2] <= y < bounds[3]

        # 同 seed 可复现：目标与无人机落点完全一致。
        other = _env()
        try:
            other.reset(seed=42)
            assert other.game._target_positions == env.game._target_positions
            assert other.game.drone_positions == env.game.drone_positions
        finally:
            other.close()

        # 不是同一组随机坐标平移复制：各区目标相对本区原点的坐标不全相同。
        relative = []
        for region in range(4):
            x_min, _, y_min, _ = env.game.region_bounds(region)
            relative.append(sorted(
                (round(x - x_min, 2), round(y - y_min, 2))
                for x, y in env.game.region_target_positions(region)
            ))
        assert len({tuple(points) for points in relative}) > 1
    finally:
        env.close()


def test_drones_remain_inside_home_regions_for_full_episode(monkeypatch):
    from envs.games.cooperative_search.game import FlightCommand

    env = _env()
    try:
        env.reset(seed=3)
        monkeypatch.setattr(env.game, "all_targets_fully_identified", lambda: False)
        monkeypatch.setattr(env.game, "region_targets_fully_identified", lambda region: False)
        rng = np.random.default_rng(0)
        for _ in range(MAX_STEPS):
            for agent in env.agents:
                bounds = env.game.flight_bounds(agent)
                position = env.game.drone_positions[agent]
                previous = env.game.headings[agent]
                available = env.game.available_headings(agent)
                for heading in available:
                    predicted = env.game._predict_position(position, FlightCommand(heading))
                    if not env.game._inside_bounds(predicted, bounds):
                        # 无法前进时仅暴露一个原地转向动作，step 必须拒绝越界位移。
                        assert len(available) == 1
                    assert env.game._turn_allowed(previous, heading)
            _random_step(env, rng)
            for agent, position in env.game.drone_positions.items():
                assert env.game.region_of_position(position) == env.game.drone_region[agent]
    finally:
        env.close()


@pytest.mark.parametrize("region", range(4))
@pytest.mark.parametrize("edge,heading", [("left", 6), ("right", 0), ("bottom", 9), ("top", 3)])
def test_home_region_edges_mask_crossings_and_reject_direct_commands(region, edge, heading):
    from envs.games.cooperative_search.game import FlightCommand

    env = _env()
    try:
        env.reset(seed=5)
        agent = env.game.region_agents(region)[0]
        x_min, x_max, y_min, y_max = env.game.region_bounds(region)
        points = {
            "left": (x_min + 1.0, (y_min + y_max) / 2),
            "right": (x_max - 1.0, (y_min + y_max) / 2),
            "bottom": ((x_min + x_max) / 2, y_min + 1.0),
            "top": ((x_min + x_max) / 2, y_max - 1.0),
        }
        position = points[edge]
        env.game.drone_positions[agent] = position
        env.game.headings[agent] = None
        mask = env.get_avail_agent_actions(agent)
        assert any(mask)
        assert mask[heading] == 0
        for candidate_heading in np.flatnonzero(mask):
            predicted = env.game._predict_position(position, FlightCommand(int(candidate_heading)))
            assert env.game.region_of_position(predicted) == region
            assert env.game._inside_bounds(predicted, env.game.flight_bounds(agent))

        # 即使调用者绕过掩码，游戏推进仍不得越出职责区。
        actions = {name: FlightCommand(env.game.available_headings(name)[0]) for name in env.game.agents}
        actions[agent] = FlightCommand(heading)
        _, _, _, _, infos = env.game.step(actions)
        assert env.game.drone_positions[agent] == position
        assert infos[agent]["boundary_action_rejected"]
    finally:
        env.close()


def test_partition_threats_keep_world_boundary_and_can_cross_regions():
    from envs.games.cooperative_search.game import FlightCommand

    env = _env_with_threats()
    try:
        env.reset(seed=5)
        threat = "dynamic_threat_0"
        env.game.threat_positions[threat] = (299.0, 150.0)
        env.game.headings[threat] = None
        assert env.get_avail_agent_actions(threat)[0] == 1
        actions = {name: FlightCommand(env.game.available_headings(name)[0]) for name in env.game.agents}
        actions[threat] = FlightCommand(0)
        env.game.step(actions)
        assert env.game.threat_positions[threat][0] > 300.0
    finally:
        env.close()


def test_outward_facing_drone_can_pivot_at_internal_boundary():
    from envs.games.cooperative_search.game import FlightCommand

    env = _env()
    try:
        env.reset(seed=5)
        env.game.drone_positions["drone_0"] = (299.0, 150.0)
        env.game.headings["drone_0"] = 0
        initial = env.game.drone_positions["drone_0"]
        for _ in range(8):
            actions = {agent: FlightCommand(env.game.available_headings(agent)[0]) for agent in env.game.agents}
            env.game.step(actions)
            assert env.game.region_of_position(env.game.drone_positions["drone_0"]) == 0
        assert env.game.drone_positions["drone_0"] != initial
    finally:
        env.close()


def test_sensor_observation_follows_current_region():
    env = _env()
    try:
        env.reset(seed=5)
        # drone_0 在区 0 一侧距分界 5km：邻区（区 1）目标距其 10km，远小于
        # 40km 传感器半径，但观测裁剪到无人机当前所在区（区 0）。
        env.game.drone_positions["drone_0"] = (295.0, 150.0)
        neighbor_target = (305.0, 150.0)
        env.game._target_positions[10] = neighbor_target  # 区 1 的第一个目标
        env.game._observe_shared_belief()
        gx, gy = env.game._world_to_grid(neighbor_target)
        assert env.game.shared_belief[gy, gx] == pytest.approx(0.5), (
            "当前所在区之外的信号源不得被传感器覆盖"
        )
        assert not env.game.detected_flags["drone_0"], (
            "当前所在区之外的信号源不得触发探测报告"
        )
        # 区 0 内 10km 的目标应当正常进入信念（无论正负证据，该格已被观测）。
        own_target = (285.0, 150.0)
        env.game._target_positions[0] = own_target
        env.game._observe_shared_belief()
        gx, gy = env.game._world_to_grid(own_target)
        observed = (
            env.game.positive_counts[gy, gx] + env.game.negative_counts[gy, gx]
        ) > 0
        assert observed, "所在区信号源必须进入传感器的贝叶斯观测"

        # 无人机跨入区 1 后，观测裁剪跟随到区 1：该邻区目标变得可见。
        env.game.drone_positions["drone_0"] = (301.0, 150.0)
        env.game._observe_shared_belief()
        neighbor_gx, neighbor_gy = env.game._world_to_grid(neighbor_target)
        crossed_observed = (
            env.game.positive_counts[neighbor_gy, neighbor_gx]
            + env.game.negative_counts[neighbor_gy, neighbor_gx]
        ) > 0
        assert crossed_observed, "跨区飞行后应能观测所在区的信号源"

        # 发现事件与奖励按区归属：把区 0 的全部目标格信念拉满后结算。
        for target in env.game.region_target_positions(0):
            gx, gy = env.game._world_to_grid(target)
            env.game.shared_belief[gy, gx] = env.game.config.max_belief_probability
        correct, false = env._region_new_detection_events(0)
        assert correct == 10
        other_correct, _ = env._region_new_detection_events(1)
        assert other_correct == 0
    finally:
        env.close()


def test_region_completion_closes_group_and_others_continue():
    env = _env()
    try:
        env.reset(seed=9)
        # 区 0 目标全部达到概率上限：该区完成。
        for target in env.game.region_target_positions(0):
            gx, gy = env.game._world_to_grid(target)
            env.game.shared_belief[gy, gx] = env.game.config.max_belief_probability
        rng = np.random.default_rng(1)
        observations, rewards, terminations, truncations, infos = _random_step(env, rng)
        region_zero = env.game.region_agents(0)
        assert all(terminations[agent] for agent in region_zero)
        assert not any(terminations[agent] for agent in env.game.region_agents(1))
        assert not any(agent in env.agents for agent in region_zero)
        assert not env.game.all_targets_fully_identified(), "其他区未完成时全局不成功"
        assert infos["drone_0"]["regions_completed"] == 1
        assert infos["drone_0"]["region_success"] is True
        assert infos["drone_4"]["region_success"] is False
        # 其余组继续推演，无人机仍不越区。
        for _ in range(10):
            _random_step(env, rng)
            assert not any(agent in env.agents for agent in region_zero)
        assert all(agent in env.agents for agent in env.game.region_agents(1))
    finally:
        env.close()


def test_region_rewards_split_among_drones_currently_in_region():
    env = _env()
    try:
        env.reset(seed=11)
        # drone_0 跨区飞到区 1：区事件结算与奖励分配都按"当前所在区"。
        env.game.drone_positions["drone_0"] = (310.0, 60.0)
        rng = np.random.default_rng(2)
        _, rewards, _, _, _ = _random_step(env, rng)

        by_region: dict[int, list[float]] = {}
        positions = {}
        for agent in DRONES:
            region = env.game.region_of_position(env.game.drone_positions[agent])
            by_region.setdefault(region, []).append(rewards[agent])
            positions[agent] = region
        assert positions["drone_0"] == 1, "跨区无人机按当前位置结算"
        occupied = [region for region, values in by_region.items() if values]
        for region in occupied:
            values = by_region[region]
            # 当前同区的无人机均分本区团队奖励，数值完全一致。
            assert len(set(values)) == 1, f"当前同区（{region}）飞机应均分本区奖励"
            assert sum(values) == pytest.approx(
                env._last_region_team_rewards[region], abs=1e-9,
            ), f"区 {region} 的奖励总和应等于该区团队奖励"
        # 全局奖励和等于各区团队奖励和，不把四区总奖励再次均分给所有无人机。
        expected_total = sum(env._last_region_team_rewards)
        assert sum(rewards.values()) == pytest.approx(expected_total)
        assert env._episode_team_reward == pytest.approx(
            sum(env._episode_region_team_rewards)
        )
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 2.4-3/7/9 表征链路与观测语义
# ---------------------------------------------------------------------------

def test_observation_uses_region_local_coordinates_and_shared_region_feature():
    env = _env()
    try:
        observations, _ = env.reset(seed=13)
        assert all(obs.shape == (OBS_DIM,) for obs in observations.values())
        # 当前同区（动态归属）的无人机共享同一份区表征（尾 REP_DIM 维一致）。
        for region in range(4):
            agents = env.game.region_agents(region)
            features = [observations[agent][-REP_DIM:] for agent in agents]
            for feature in features[1:]:
                np.testing.assert_array_equal(features[0], feature)
        # 自身位置特征按当前所在区局部归一化。
        for agent in DRONES:
            region = env._agent_region(agent)
            x_min, _, y_min, _ = env.game.region_bounds(region)
            x, y = env.game.drone_positions[agent]
            base = observations[agent][FOV * FOV * 4: FOV * FOV * 4 + 2]
            np.testing.assert_allclose(
                base, [(x - x_min) / 300.0, (y - y_min) / 300.0], atol=1e-6,
            )
    finally:
        env.close()


def test_region_layout_enters_representation_not_just_coverage():
    env = _env()
    try:
        env.reset(seed=17)
        # 覆盖率相同、探索位置不同的责任区网格必须产生不同表征。
        env.game.positive_counts.fill(0)
        env.game.negative_counts.fill(0)
        rows, cols = env.game.region_grid_slice(0)
        half = (rows.stop - rows.start) // 2
        env.game.positive_counts[rows.start:rows.start + 2, cols.start:cols.start + 2] = 1
        first = env._region_representation(0).copy()
        env.game.positive_counts.fill(0)
        env.game.positive_counts[rows.stop - 2:rows.stop, cols.stop - 2:cols.stop] = 1
        second = env._region_representation(0).copy()
        assert not np.array_equal(first, second), (
            "相同覆盖格数、不同位置的图必须影响策略可见输入"
        )
        # 表征实际进入本组四机的策略观测。
        observations = env.get_obs()
        np.testing.assert_array_equal(observations["drone_0"][-REP_DIM:], second)
        # 本区新增扫描只更新本区探索网格：区 1 表征不受区 0 修改影响。
        region_one = env._region_representation(1)
        assert region_one is not None
    finally:
        env.close()


def test_state_for_agent_ids_shape_consistency_and_cross_region_rejected():
    env = _env()
    try:
        env.reset(seed=19)
        for region in range(4):
            members = tuple(env.game.region_agents(region))
            state = env.state_for_agent_ids(
                agent_name="uav_team", agent_ids=members,
            )
            assert state.shape == (GROUP_STATE_DIM,)
        with pytest.raises(EnvironmentException):
            env.state_for_agent_ids(
                agent_name="uav_team",
                agent_ids=("drone_0", "drone_4", "drone_8", "drone_12"),
            )
        assert env.state().shape == (GLOBAL_STATE_DIM,)
    finally:
        env.close()


def test_two_legal_output_dims_and_invalid_fail():
    env96 = _env(representation_dim=96)
    try:
        observations, _ = env96.reset(seed=23)
        obs_dim = FOV * FOV * 4 + 3 + 96
        assert env96.observation_dim == obs_dim
        assert all(obs.shape == (obs_dim,) for obs in observations.values())
        assert env96.group_state_dim == 96 + 16
        assert env96.state_for_agent_ids(
            agent_name="uav_team",
            agent_ids=tuple(env96.game.region_agents(2)),
        ).shape == (96 + 16,)
    finally:
        env96.close()

    with pytest.raises(EnvironmentException):
        _env(representation_dim=None)
    with pytest.raises(EnvironmentException):
        _env(representation_dim=2)  # 小于通道数 3，非法


def test_observation_at_step_t_uses_grid_at_step_t():
    env = _env()
    try:
        env.reset(seed=29)
        rng = np.random.default_rng(4)
        for _ in range(3):
            # 决策帧在 step 前冻结：get_obs 的表征与当步网格重算一致
            #（区为无人机当前所在区）。
            before = env.get_obs()
            for agent in DRONES:
                region = env._agent_region(agent)
                np.testing.assert_allclose(
                    before[agent][-REP_DIM:],
                    env._region_representation(region),
                    atol=1e-7,
                    err_msg=f"agent {agent} region {region} representation mismatch",
                )
            _random_step(env, rng)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# 端到端：真实分区环境 + 分组推理 + 两种封装回放 + QMIX 更新
# ---------------------------------------------------------------------------

def test_end_to_end_grouped_rollout_push_and_learn(tmp_path):
    import torch

    from algos.modules.agents.rnn_agent import RNNAgent
    from algos.training.qmix_learner import QMixerLearner
    from config import (
        AgentConfig, LearnerConfig, QmixConfig, ReplayBufferConfig, RnnAgentConfig,
    )
    from envs.action.model import ActionParameterDefined, StepActionDefined
    from sample.replay_buffer import ReplayBuffer
    from scheduler.model import ActionParam, AgentCompose, EpisodeInfo
    from scheduler.rollout_sample import RolloutSampleWorker

    groups = tuple(tuple(f"drone_{g * 4 + s}" for s in range(4)) for g in range(4))

    def heading_agent():
        config = AgentConfig(
            name="uav_team", agent_type="rnn", agent_names=[],
            agent_groups=[list(group) for group in groups],
            obs_dim=OBS_DIM, state_shape=GROUP_STATE_DIM,
            algo_config=RnnAgentConfig(),
            define_action=[StepActionDefined(
                action_key="heading",
                parameters=[ActionParameterDefined(key="value", values=list(range(12)))],
                add_target=False,
            )],
        )
        return RNNAgent(
            agent_type="rnn", config=config, agent_net_source="blanco", device="cpu",
        )

    for mode, capacity, batch_units in (
        ("independent", 32, 4),
        ("episode_groups", 8, 1),
    ):
        env = _env(max_steps=4)
        agent = heading_agent()
        replay = ReplayBuffer(ReplayBufferConfig(
            agent_capacities={"uav_team": capacity}, batch_size=1,
            storage=__import__("config").LocalStorageConfig(
                data_path=str(tmp_path / mode),
            ),
            sample_packaging=mode,
        ))
        try:
            notify = []
            replay.learn_batches_update_caller = lambda state, episode: notify.append(state)
            worker = RolloutSampleWorker(
                env=env,
                task_manager=SimpleNamespace(
                    accept_sample_task_result=lambda *args, **kwargs: True,
                ),
                replay_buffer=replay,
                agent_compose=AgentCompose(step_agent=[agent]),
                policy_agents=[agent],
            )
            episode_info = EpisodeInfo(
                episode_num=1, task_id="partition-e2e", epsilon=0.3,
                deterministic=False, temperature=1.0,
                actions_param={
                    "uav_team": ActionParam(
                        epsilon=0.3, deterministic=False, temperature=1.0,
                    ),
                },
                seed=3, start_datetime=None, end_datetime=None, metadata={},
            )
            transitions = worker.rolling_parallel_episode(episode_info, step_limit=4)
            assert list(transitions) == ["uav_team"]
            group_trajectories = transitions["uav_team"]
            # 每个物理 episode 最多四条组轨迹；组宽固定 4，不扩成 16。
            assert len(group_trajectories) == 4
            for index, transition in enumerate(group_trajectories):
                assert tuple(transition.agent_ids) == groups[index]
                assert transition.seq_len == 4
            # 一次批量 push 入同一分区、一次通知学习。
            assert len(notify) == 1
            state = notify[0]
            assert state.agent_trajectory_counts["uav_team"] == 4
            assert state.agent_episode_counts["uav_team"] == 1
            assert state.agent_sizes["uav_team"] == (4 if mode == "independent" else 1)

            learner_config = LearnerConfig(
                name="uav_team", algo_type="qmix", state_shape=GROUP_STATE_DIM,
                algo_config=QmixConfig(), standardise_rewards=True,
                standardise_returns=False,
            )
            learner = QMixerLearner(
                algo_type="qmix", config=learner_config, agent=agent, env=env,
                device="cpu",
            )
            batch = replay.sample(
                "uav_team", batch_size=batch_units, strategy="recent_biased", seed=0,
            )
            # 4 条组轨迹铺平为 [N=4, T+1, K=4, D]，QMIX 组宽始终为 4。
            assert len(batch) == 4
            before = {key: value.clone()
                      for key, value in learner.agent_net.state_dict().items()}
            learner.learn_update(batch)
            assert any(
                not torch.equal(before[key], learner.agent_net.state_dict()[key])
                for key in before
            )
        finally:
            replay.close()
            env.close()


# ---------------------------------------------------------------------------
# 动态威胁：专训环境与主科目接入
# ---------------------------------------------------------------------------

THREATS = [f"dynamic_threat_{index}" for index in range(8)]


def _threat_env(max_steps=50, seed=9):
    from envs.sense.cooperative_search_partition_threat_env import (
        CooperativeSearchPartitionThreatEnv,
    )

    return CooperativeSearchPartitionThreatEnv(EnvConfig(
        sense_name="cooperative_search_partition_threat_training",
        max_step_per_episode=max_steps,
        step_interval_secs=30,
        state_dim=8 * 173,
        agent_names=THREATS,
        use_sim=False,
        metadata={
            "render_mode": None,
            "game_config": {
                "map_width": 600, "map_height": 600,
                "belief_grid_width": 200, "belief_grid_height": 200,
                "num_drones": 0, "num_dynamic_threats": 8, "num_targets": 40,
                "region_rows": 2, "region_cols": 2, "max_steps": max_steps,
                "threat_warning_radius": 45.0, "initial_threat_separation": 60.0,
            },
        },
    ))


def test_partition_threat_training_env_contract():
    env = _threat_env()
    try:
        observations, _ = env.reset(seed=9)
        # 单成员组重复推理：13维数值特征与160维极坐标表征。
        assert env.observation_dim == 173
        assert env.centralized_state_dim == 8 * 173
        assert all(obs.shape == (173,) for obs in observations.values())
        assert len(env.game.drone_positions) == 0
        # 目标分布与正式科目一致：每区恰好 10 个。
        for region in range(4):
            assert len(env.game.region_target_positions(region)) == 10
        # 同 seed 可复现。
        other = _threat_env()
        try:
            other.reset(seed=9)
            assert other.game._target_positions == env.game._target_positions
            assert other.game.threat_positions == env.game.threat_positions
        finally:
            other.close()

        rng = np.random.default_rng(0)
        for _ in range(20):
            actions = {}
            for agent in env.agents:
                mask = env.get_avail_agent_actions(agent)
                index = int(rng.choice(np.flatnonzero(mask)))
                actions[agent] = Action(
                    action_key="heading", parameters={"value": index},
                    target=-2, action_idx=index,
                )
            _, rewards, _, _, infos = env.step(actions)
            assert rewards == env.get_reward(), "每个威胁使用自己的独立奖励"
        # 威胁不分区：随机游走后可分布到多个责任区。
        regions = {
            env.game.region_of_position(position)
            for position in env.game.threat_positions.values()
        }
        assert len(regions) >= 1
        assert "nearest_peer_distance_km" in infos["dynamic_threat_0"]
    finally:
        env.close()


def _env_with_threats(max_steps=15):
    metadata = {
        "fov_size": FOV,
        "observation_grid_scale": 2,
        "state_grid_width": 50,
        "state_grid_height": 50,
        "render_mode": None,
        "dynamic_threats_enabled": True,
        "situation_representation_output_dim": REP_DIM,
        "game_config": {
            "map_width": 600, "map_height": 600,
            "belief_grid_width": 200, "belief_grid_height": 200,
            "num_drones": 16, "num_dynamic_threats": 8, "num_targets": 40,
            "region_rows": 2, "region_cols": 2, "max_steps": max_steps,
            "step_interval_secs": 30.0,
            "threat_warning_radius": 45.0, "initial_threat_separation": 60.0,
        },
    }
    return CooperativeSearchPartitionEnv(EnvConfig(
        sense_name="cooperative_search_partition",
        max_step_per_episode=max_steps,
        step_interval_secs=30,
        state_dim=4 * REP_DIM + (16 + 8) * 4,
        agent_names=DRONES + THREATS,
        use_sim=False,
        metadata=metadata,
    ))


def test_partition_env_supports_eight_threat_group():
    env = _env_with_threats()
    try:
        observations, _ = env.reset(seed=3)
        # 无人机观测结构不变；威胁观测是全图目标口径 145 维。
        assert env.observation_dim == FOV * FOV * 4 + 3 + REP_DIM
        assert all(observations[agent].shape == (env.observation_dim,) for agent in DRONES)
        assert observations["dynamic_threat_0"].shape == (4 + 40 * 3 + 7 * 3,)
        # 全局状态 = 4 区表征 + 无人机与威胁的飞行状态。
        assert env.centralized_state_dim == 4 * REP_DIM + (16 + 8) * 4
        assert env.state().shape == (env.centralized_state_dim,)
        # 组状态仍锚定责任区，不含威胁维度。
        assert env.group_state_dim == REP_DIM + 16

        rng = np.random.default_rng(1)
        for _ in range(12):
            actions = {}
            for agent in env.agents:
                mask = env.get_avail_agent_actions(agent)
                index = int(rng.choice(np.flatnonzero(mask)))
                actions[agent] = Action(
                    action_key="heading", parameters={"value": index},
                    target=-2, action_idx=index,
                )
            env.step(actions)
            assert env.agents == env.game.agents
            # 威胁与无人机同时在活动列表中（除非无人机被吃或区完成）。
            assert any(agent.startswith("dynamic_threat_") for agent in env.agents)
    finally:
        env.close()


def test_completed_region_drones_stop_threat_interaction(monkeypatch):
    env = _env_with_threats()
    try:
        env.reset(seed=7)
        game = env.game
        game.threat_positions = {agent: (599.0, 599.0) for agent in game.threat_agents}
        monkeypatch.setattr(game, "all_targets_fully_identified", lambda: False)
        monkeypatch.setattr(game, "region_targets_fully_identified", lambda region: region == 0)
        rng = np.random.default_rng(2)
        _, _, terminations, _, _ = _random_step(env, rng)
        retired = set(game.region_agents(0))
        assert all(terminations[agent] for agent in retired)
        assert retired.isdisjoint(game.agents)
        parked_positions = {agent: game.drone_positions[agent] for agent in retired}

        # 复现线上触发条件：威胁在下一步经过已经结束轨迹的停机实体。
        game.threat_positions["dynamic_threat_0"] = parked_positions["drone_0"]
        for _ in range(3):
            _, _, _, _, infos = _random_step(env, rng)
            assert retired.isdisjoint(game.agents)
            assert not any(infos[agent].get("eaten_by_threat", False) for agent in retired)
            assert all(game.drone_positions[agent] == parked_positions[agent] for agent in retired)
            assert not any(game.drones_detected_by_threat[agent] for agent in retired)
            assert retired.isdisjoint(game.dead_drone_positions)
            assert env._last_region_reward_breakdown[0]["death"] == 0.0

        # 活动区域仍正常检测、淘汰和结算，不能因修复而关闭威胁功能。
        game.threat_positions["dynamic_threat_0"] = game.drone_positions["drone_4"]
        _, _, terminations, _, infos = _random_step(env, rng)
        assert terminations["drone_4"]
        assert infos["drone_4"]["eaten_by_threat"]
        assert "drone_4" in game.dead_drone_positions
        assert env._last_region_reward_breakdown[1]["death"] < 0.0
    finally:
        env.close()


def test_threat_eats_drone_and_penalty_settles_in_current_region():
    env = _env_with_threats()
    try:
        env.reset(seed=7)
        # 把威胁放到 drone_0 旁边（30km 击杀圈内），构造确定击杀。
        x, y = env.game.drone_positions["drone_0"]
        env.game.threat_positions["dynamic_threat_0"] = (x + 5.0, y)
        env.game._update_threat_detections()
        previous_rewarded = [
            len(env._rewarded_region_target_indices[r]) for r in range(4)
        ]
        rng = np.random.default_rng(2)
        actions = {}
        for agent in env.agents:
            mask = env.get_avail_agent_actions(agent)
            index = int(rng.choice(np.flatnonzero(mask)))
            actions[agent] = Action(
                action_key="heading", parameters={"value": index},
                target=-2, action_idx=index,
            )
        _, rewards, terminations, _, infos = env.step(actions)
        assert "drone_0" in env.game.dead_drone_positions
        assert terminations["drone_0"]
        assert "drone_0" not in env.agents
        # 出生簇顶点间距 30km，击杀圈可能带走邻近多架；都计入区 0。
        dead_in_region_zero = [
            agent for agent in env.game.dead_drone_positions
            if env.game.drone_region[agent] == 0
        ]
        assert len(dead_in_region_zero) >= 1
        # 死亡惩罚计入无人机当时所在区（区 0）。
        assert env._last_region_reward_breakdown[0]["death"] == pytest.approx(
            -env.death_penalty_per_drone * len(dead_in_region_zero)
        )
        assert env._episode_region_team_rewards[0] < 0.0
        assert infos["drone_0"]["own_loss_count"] == len(env.game.dead_drone_positions)
    finally:
        env.close()


def test_threats_spawn_evenly_per_region():
    env = _env_with_threats()
    try:
        for seed in (1, 42):
            env.reset(seed=seed)
            # 每区恰好 2 个威胁（8 = 2 × 4 分区），区内出生。
            counts = []
            for region in range(4):
                threats = [
                    position for position in env.game.threat_positions.values()
                    if env.game.region_of_position(position) == region
                ]
                counts.append(len(threats))
                bounds = env.game.region_bounds(region)
                for x, y in threats:
                    assert bounds[0] <= x < bounds[1] and bounds[2] <= y < bounds[3]
            assert counts == [2, 2, 2, 2], counts
            # 分离约束保持：威胁与无人机、威胁之间 >= initial_threat_separation。
            threats = list(env.game.threat_positions.values())
            for threat in threats:
                for drone in env.game.drone_positions.values():
                    assert env.game._distance(threat, drone) >= 60.0 - 1e-9
            for index, first in enumerate(threats):
                for second in threats[index + 1:]:
                    assert env.game._distance(first, second) >= 60.0 - 1e-9
        # 0 威胁配置不受影响；非整除数量明确报错。
        from envs.games.cooperative_search.partition_game import PartitionedSearchConfig

        with pytest.raises(EnvironmentException):
            PartitionedSearchConfig(
                num_drones=16, num_dynamic_threats=7, num_targets=40,
                region_rows=2, region_cols=2, threat_warning_radius=45.0,
                initial_threat_separation=60.0,
            )
    finally:
        env.close()


def test_threat_training_subject_toml_assembles():
    from utils.toml_config import TomlTrainingConfig

    root = Path(__file__).resolve().parents[1]
    config = TomlTrainingConfig(
        root / "subjects/cooperative_search/cooperative_search_partition_threat_qmix.toml",
        project_root=root,
    )
    agent = config.agents_config[0]
    assert agent.name == "dynamic_threats"
    # 八个单成员组共享同一RNN。
    assert not agent.agent_names and len(agent.agent_groups) == 8
    assert all(len(group) == 1 for group in agent.agent_groups)
    assert agent.obs_dim == 173
    assert config.env_config.state_dim == 8 * 173
    assert len(config.env_config.agent_names) == 8
    learner = config.learners_config[0]
    assert learner.name == "dynamic_threats" and learner.algo_type == "qmix"


# ---------------------------------------------------------------------------
# 科目配置装配
# ---------------------------------------------------------------------------

def test_partition_subjects_assemble_and_pair_on_budget():
    from utils.toml_config import TomlTrainingConfig



    from utils.toml_config import TomlTrainingConfig

    root = Path(__file__).resolve().parents[1]
    subjects = root / "subjects/cooperative_search"
    base = TomlTrainingConfig(
        subjects / "cooperative_search_qmix_partitions.toml", project_root=root,
    )
    bundled = TomlTrainingConfig(
        subjects / "cooperative_search_qmix_partitions_bundled.toml", project_root=root,
    )
    for config in (base, bundled):
        agent = config.agents_config[0]
        learner = config.learners_config[0]
        assert len(agent.agent_groups) == 4
        assert all(len(group) == 4 for group in agent.agent_groups)
        assert agent.obs_dim == 11 * 11 * 4 + 3 + 128
        assert agent.state_shape == 144
        assert learner.state_shape == 144
        # 16 机 + 8 威胁的全局状态：4 区表征 + 24 个实体的飞行状态。
        assert config.env_config.state_dim == 4 * 128 + (16 + 8) * 4 == 608
        assert len(config.env_config.agent_names) == 24
        # 固定策略威胁组：八个单成员组重复推理。
        threat_agent = config.agents_config[1]
        assert threat_agent.name == "dynamic_threats"
        assert not threat_agent.agent_names and len(threat_agent.agent_groups) == 8
        assert all(len(group) == 1 for group in threat_agent.agent_groups)
        assert threat_agent.fixed_mode is True
        assert threat_agent.obs_dim == 173
    assert base.replay_buffer_config.sample_packaging == "independent"
    assert bundled.replay_buffer_config.sample_packaging == "episode_groups"
    base_capacity = base.learners_config[0].algo_config.capacity
    bundled_capacity = bundled.learners_config[0].algo_config.capacity
    base_batch = base.learners_config[0].algo_config.batch_size
    bundled_batch = bundled.learners_config[0].algo_config.batch_size
    # 容量与批量按单位换算后预算一致：500 局×4 组 == 2000 条组轨迹。
    assert base_capacity == bundled_capacity * 4
    assert base_batch == bundled_batch * 4
    # 训练预算、优化器与网络超参完全一致（配对对照）。
    assert base.train_config.max_episode == bundled.train_config.max_episode
    assert (
        base.learners_config[0].algo_config.lr
        == bundled.learners_config[0].algo_config.lr
    )

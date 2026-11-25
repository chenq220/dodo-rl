"""四区域训练过程基准：独立CPU进程，固定配置，2局search + 2局evade。

保留原 test_baseline_training.py；本文件不读取科目TOML或本机artifacts。
开发期显式录制（仅输出JSON，不覆盖基准）：
    DODO_RECORD_BASELINE=1 .venv/bin/python tests/test_baseline_training_partitions.py --dodo-collect
正常 pytest 只读取 fixtures/baseline_partitions/expected_trace.json 比对过程数据。
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "baseline_partitions_4"
if __name__ == "__main__":
    os.environ.setdefault("DODO_TASK_ID", TASK_ID)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _path in (ROOT, ROOT / "src", ROOT / "sdk"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
import pytest
import torch

from common.info_bridge import info_bridge
from common.plugins import PluginsConfig, activate_plugins
from config import (
    AgentConfig, EnvConfig, LearnerConfig, LocalStorageConfig, QmixConfig,
    ReplayBufferConfig, RnnAgentConfig, StageConfig, StageTaskConfig,
    ThreatRoamingConfig, TrainConfig,
)
from envs.action.model import ActionParameterDefined, StepActionDefined
from portal.plugins.registry import project_plugin_registry
from sample.model import SamplingStrategy

FIXTURES = Path(__file__).resolve().parent / "fixtures/baseline_partitions"
THREAT_MODEL = FIXTURES / "threat_agent_6f548e77.pth"
THREAT_SHA256 = "c88b839cd24ea6913e4040b20673bc9b528126696cdb6218937e4a68646cfc5a"
EXPECTED_TRACE = FIXTURES / "expected_trace.json"
EXPECTED_TRACE_LINUX = FIXTURES / "expected_trace_linux.json"
SEED = 42
CHILD_FLAG = "--dodo-collect"
BEGIN = "BEGIN_PARTITION_BASELINE_TRACE"
END = "END_PARTITION_BASELINE_TRACE"
MAX_STEPS = 600
EPISODES_PER_STAGE = 2
CAPACITY = 8  # 每阶段2个物理局，每局4条组轨迹。
BATCH_SIZE = 4  # 第一物理局即有4条组轨迹，无需等待第二局。
FLOAT_DECIMALS = 10
REP_DIM = 128
OBS_DIM = 11 * 11 * 4 + 3 + REP_DIM
GROUP_STATE_DIM = REP_DIM + 4 * 4
STATE_DIM = 4 * REP_DIM + (16 + 8) * 4
UAV = "uav_team"
THREATS = "dynamic_threats"
DRONES = [f"drone_{i}" for i in range(16)]
THREAT_IDS = [f"dynamic_threat_{i}" for i in range(8)]


def _action():
    return StepActionDefined(
        action_key="heading", add_target=False, allow_null_targets=False,
        parameters=[ActionParameterDefined(key="value", values=list(range(12)))],
    )


def build_partition_baseline_role_inputs(work: Path) -> dict:
    """冻结蓝本参数；修改生产TOML不会改变本基准。"""
    assert hashlib.sha256(THREAT_MODEL.read_bytes()).hexdigest() == THREAT_SHA256
    agents = [
        AgentConfig(
            name=UAV, device=torch.device("cpu"), agent_type="rnn",
            multi_action_mode="combination",
            agent_groups=[DRONES[i:i + 4] for i in range(0, 16, 4)],
            obs_dim=OBS_DIM, state_shape=GROUP_STATE_DIM,
            obs_last_action=True, obs_agent_id=True, hidden_dim=128,
            define_action=[_action()],
            algo_config=RnnAgentConfig(
                use_net=True, agent="rnn", mac="basic_mac", use_rnn=True,
                sharing_scope="all", mask_before_softmax=True, action_selector=None,
            ),
        ),
        AgentConfig(
            name=THREATS, device=torch.device("cpu"), agent_type="rnn",
            multi_action_mode="combination",
            agent_groups=[[agent] for agent in THREAT_IDS],
            obs_dim=173, state_shape=173, obs_last_action=True, obs_agent_id=False,
            hidden_dim=128, define_action=[_action()],
            model_file_path=str(THREAT_MODEL), fixed_mode=True,
            algo_config=RnnAgentConfig(
                use_net=True, agent="rnn", mac="basic_mac", use_rnn=True,
                sharing_scope="all", mask_before_softmax=True, action_selector=None,
            ),
        ),
    ]
    # AgentConfig构造期回填RNN默认值；同TOML翻译入口一样在装配期显式覆盖。
    agents[1].obs_agent_id = False
    roaming = ThreatRoamingConfig(
        circle_radius_km=50.0, circle_center_min_shift_km=100.0,
        circle_center_max_shift_km=200.0, waypoint_reached_km=5.0,
        waypoint_min_distance_km=10.0, waypoint_max_distance_km=25.0,
        waypoint_refresh_min_steps=40, waypoint_refresh_max_steps=80,
        waypoint_timeout_steps=200, perception_radius_km=100.0,
        radial_bins=10, angular_bins=16, representation_output_dim=160,
        separation_distance_km=30.0, danger_distance_km=15.0,
        progress_reward_weight=0.2, arrival_reward=0.5, circle_penalty=1.0,
        separation_penalty=0.5, danger_penalty=2.0,
    )
    env = EnvConfig(
        sense_name="cooperative_search_partition", use_sim=False,
        max_step_per_episode=MAX_STEPS, step_interval_secs=30,
        state_dim=STATE_DIM, agent_names=DRONES + THREAT_IDS,
        metadata={
            "fov_size": 11, "observation_grid_scale": 2,
            "state_grid_width": 50, "state_grid_height": 50,
            "situation_representation_output_dim": REP_DIM,
            "render_mode": "rgb_array", "render_every_step": False,
            "dynamic_threats_enabled": True,
            "exploration_reward_per_cell": 0.00082,
            "revisited_empty_penalty_episode_limit": 15,
            "frontier_potential_reward_weight": 0.03,
            "correct_detection_reward_total": 20.0,
            "correct_detection_reward_half_life_ratio": 0.8,
            "false_detection_penalty": 0.3, "death_penalty_per_drone": 10.0,
            "save_probability_map_on_episode_end": False,
            "probability_map_every_n_episodes": 100,
            "probability_map_output_dir": str(work / "results"),
            "game_config": {
                "map_width": 600, "map_height": 600,
                "belief_grid_width": 200, "belief_grid_height": 200,
                "region_rows": 2, "region_cols": 2,
                "num_drones": 16, "num_dynamic_threats": 8, "num_targets": 40,
                "sensor_radius": 40.0, "threat_sensor_radius": 30.0,
                "threat_warning_radius": 45.0, "communication_radius": 100.0,
                "initial_separation": 20.0, "initial_formation_side_km": 30.0,
                "initial_threat_separation": 60.0,
                "detection_probability": 0.8, "false_alarm_probability": 0.2,
                "detection_threshold": 0.9, "min_belief_probability": 0.1,
                "max_belief_probability": 0.9, "max_turn_degrees": 30.0,
                "fixed_speed_mps": 100.0, "threat_speed_mps": 80.0,
                "step_interval_secs": 30.0, "max_steps": MAX_STEPS,
                "fps": 20, "sidebar_width": 300, "show_threat_roaming": False,
                "threat_roaming": asdict(roaming),
            },
        },
    )
    stages = []
    for name, epsilon, decay, lr, updates in (
        ("search", 1.0, 2250, 0.0002, 2),
        ("evade", 0.3, 750, 0.0001, 1),
    ):
        learner = LearnerConfig(
            name=UAV, device=torch.device("cpu"), algo_type="qmix",
            state_shape=GROUP_STATE_DIM, common_reward=True,
            standardise_rewards=True, standardise_returns=False,
            grad_norm_clip=5.0, gamma=0.995,
            algo_config=QmixConfig(
                capacity=CAPACITY, batch_size=BATCH_SIZE, lr=lr, hidden_dim=128,
                learner="q_learner", mixer="qmix", td_lambda=0.6,
                double_q=True, target_update_interval_or_tau=0.005,
                mixing_embed_dim=32, hypernet_layers=2, hypernet_embed=64,
                sample_strategy=SamplingStrategy.RECENT_BIASED.value,
                learn_times_per_episode=updates,
            ),
        )
        train = TrainConfig(
            max_episode=EPISODES_PER_STAGE, sample_task_timeout=5400,
            seed=SEED, initialization_seed=SEED,
            start_epsilon=epsilon, end_epsilon=0.05, epsilon_decay_episodes=decay,
            agent_compose_type="order", record_sample=False,
            artifacts_archive_enabled=False,
            artifacts_archive_save_dir=str(work / "archives"),
            model_save_dir=str(work / name / "models"),
            model_load_dir=str(work / name / "models"),
            result_save_dir=str(work / name / "results"),
            render_video_dir=str(work / name / "recordings"),
            mid_eval_enabled=False, end_eval_episodes=0,
            env_metadata={"dynamic_threats_enabled": name == "evade"},
        )
        stages.append(StageConfig(
            stage_name=name, train_config=train, agents_config=copy.deepcopy(agents),
            learners_config=[learner], clear_replay_buffer=True,
            rehearsal_archive_ratio=0.25 if name == "search" else 0.0,
            rehearsal_archive_select="even_span",
            rehearsal_sample_ratio=0.25 if name == "evade" else 0.0,
        ))
    return {
        "stage_task_config": StageTaskConfig(
            stages=stages, bootstrap_learners_config=[copy.deepcopy(stages[0].learners_config[0])],
        ),
        "env_config": env,
        # This benchmark runs a local environment; it must not depend on dodo-sim.
        "sim_config": None,
        "sim_manager_config": None,
        "model_manager_config": None,
        "replay_buffer_config": ReplayBufferConfig(
            capacity=CAPACITY, batch_size=BATCH_SIZE, recent_ratio=0.35, seed=SEED,
            tensor_pool="lazy_memmap", sample_packaging="independent",
            storage=LocalStorageConfig(data_path=str(work / "samples"), scratch_subdir="scratch"),
        ),
    }


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return round(value, FLOAT_DECIMALS)
    return value


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _continuous_trace(value) -> dict:
    """逐步均值保留时间顺序，连续值以十位小数数值比较，不哈希浮点字节。"""
    array = np.asarray(value, dtype=np.float64)
    steps = array.reshape(array.shape[0], -1)
    return _json_value({
        "shape": list(array.shape), "mean": float(array.mean()),
        "std": float(array.std()), "min": float(array.min()), "max": float(array.max()),
        "step_mean": steps.mean(axis=1),
    })


def _group_trace(transition) -> dict:
    return {
        "agent_ids": list(transition.agent_ids),
        "steps": transition.seq_len,
        "reward": float(transition.episode_reward()),
        "observations": {
            member: {name: _continuous_trace(field[member]) for name, field in (
                ("current", transition.obs), ("next", transition.next_obs),
            )} for member in transition.agent_ids
        },
        "states": {name: _continuous_trace(field) for name, field in (
            ("current", transition.states), ("next", transition.next_states),
        )},
        "actions_masks_dones_sha256": _digest({
            name: getattr(transition, name) for name in (
                "actions", "last_actions", "terminations", "truncations",
                "avail_actions", "avail_actions_next",
                "active_agent_mask", "next_active_agent_mask",
            )
        }),
    }


def collect_partition_trace(work: Path) -> dict:
    """从真实 SerialStageTrainRole 采集4局过程，不替代环境/网络/更新。"""
    from roles.serial_stage_train import SerialStageTrainRole
    from scheduler.rollout_learn import RolloutLearnWorker
    from utils.random_seed import set_global_seed

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    set_global_seed(SEED)
    activate_plugins(PluginsConfig(), default_plugins=project_plugin_registry())
    inputs = build_partition_baseline_role_inputs(work)
    # mmap预算只需这两份配置；不加载TOML，也不通过桥传递运行模块。
    info_bridge.put(toml_config={
        "env_config": inputs["env_config"],
        "stage_task_config": inputs["stage_task_config"],
    })
    role = SerialStageTrainRole(**inputs)
    fixed_agent = role.agents_by_name[THREATS]
    fixed_weights = {k: v.detach().clone() for k, v in fixed_agent.agent_net.state_dict().items()}
    records = []
    pending_learning = []
    pushed_names = []
    original_learn = RolloutLearnWorker.rollout_after_episode_updated
    original_log = SerialStageTrainRole._log_episode_result
    original_push = role.replay_buffer.push
    original_step = role.env.step

    def tracking_learn(worker):
        result = original_learn(worker)
        pending_learning.append(_json_value(result))
        return result

    def tracking_push(data, *args, **kwargs):
        payload = getattr(data, "episodes", data)
        names = tuple(payload) if isinstance(payload, dict) else (kwargs.get("agent_name"),)
        assert names == (UAV,), f"fixed policy entered replay: {names}"
        pushed_names.append(names)
        return original_push(data, *args, **kwargs)

    def tracking_step(actions):
        result = original_step(actions)
        game = role.env.game
        for agent, position in game.drone_positions.items():
            assert game.region_of_position(position) == game.drone_region[agent]
        return result

    def tracking_episode(stage, transitions):
        assert set(transitions) == {UAV}
        groups = transitions[UAV]
        assert len(groups) == 4
        assert [list(t.agent_ids) for t in groups] == [DRONES[i:i + 4] for i in range(0, 16, 4)]
        assert role.replay_buffer.agent_buffer_size(THREATS) == 0
        info = role.env._info("drone_0")
        records.append({
            "stage": stage.stage_name,
            "episode": len([r for r in records if r["stage"] == stage.stage_name]) + 1,
            "steps": role.env.game.step_count,
            "reward": float(sum(t.episode_reward() for t in groups)),
            "groups": [_group_trace(t) for t in groups],
            "metrics": {name: _json_value(info[name]) for name in (
                "success", "target_discovered_count", "own_loss_count",
                "coverage_rate_percent", "false_target_count", "regions_completed",
            )},
            "region_reward_breakdown": _json_value(role.env._episode_region_reward_breakdown),
            "replay_size": role.replay_buffer.agent_buffer_size(UAV),
            "learning": pending_learning[-1],
        })
        print(f"BASELINE_PROGRESS {stage.stage_name} {records[-1]['episode']}/{EPISODES_PER_STAGE}", flush=True)
        return original_log(stage, transitions)

    role.replay_buffer.push = tracking_push
    role.env.step = tracking_step
    with patch.object(RolloutLearnWorker, "rollout_after_episode_updated", tracking_learn), \
            patch.object(SerialStageTrainRole, "_log_episode_result", staticmethod(tracking_episode)):
        role.run()
    assert len(records) == len(pushed_names) == 2 * EPISODES_PER_STAGE
    for stage in ("search", "evade"):
        rows = [r for r in records if r["stage"] == stage]
        assert len(rows) == EPISODES_PER_STAGE
        for row in rows:
            updates = row["learning"]
            assert len(updates) == 1, f"missing learning: {stage} episode={row['episode']}"
            stats = updates[0][UAV]
            assert len(stats) == (2 if stage == "search" else 1)
            assert all("loss" in result and not result.get("skip_update") for result in stats)
    assert role.replay_buffer.agent_names() == [UAV]
    for key, tensor in fixed_agent.agent_net.state_dict().items():
        assert torch.equal(tensor, fixed_weights[key]), f"fixed policy weight changed: {key}"
    return _json_value({"seed": SEED, "threat_checkpoint_sha256": THREAT_SHA256, "episodes": records})


def _assert_process_matches(actual, expected, path="trace"):
    """报告首个偏离字段；浮点统计允许CPU舍入误差，离散轨迹摘要必须相同。"""
    if isinstance(expected, dict):
        if "stage" in expected and "episode" in expected:
            path += f"({expected['stage']} episode={expected['episode']})"
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            _assert_process_matches(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (got, want) in enumerate(zip(actual, expected)):
            _assert_process_matches(got, want, f"{path}[{i}]")
    elif isinstance(expected, float):
        assert round(actual, FLOAT_DECIMALS) == pytest.approx(
            round(expected, FLOAT_DECIMALS), rel=1e-7, abs=1e-10,
        ), f"{path}: {actual} != {expected}"
    else:
        assert actual == expected, f"{path}: {actual} != {expected}"


def test_partition_baseline_continuous_precision():
    value = {"float": 0.12345678901234, "numpy": np.float64(0.12345678901234),
             "tensor": torch.tensor([0.12345678901234], dtype=torch.float64),
             "integer": 3, "flag": True}
    rounded = _json_value(value)
    assert rounded == {"float": 0.123456789, "numpy": 0.123456789,
                       "tensor": [0.123456789], "integer": 3, "flag": True}
    assert type(rounded["integer"]) is int and type(rounded["flag"]) is bool
    trace = _continuous_trace([[0.12345678901234], [0.23456789012345]])
    assert trace["step_mean"] == [0.123456789, 0.2345678901]
    _assert_process_matches(0.12345678901234, 0.12345678904999)
    with pytest.raises(AssertionError):
        _assert_process_matches(0.123456, 0.123457)


def test_partition_baseline_fixture_has_four_rounded_episodes():
    trace = json.loads(EXPECTED_TRACE.read_text())
    assert trace["seed"] == SEED
    assert trace["threat_checkpoint_sha256"] == THREAT_SHA256
    assert trace == _json_value(trace)
    assert [(row["stage"], row["episode"]) for row in trace["episodes"]] == [
        (stage, episode) for stage in ("search", "evade") for episode in range(1, EPISODES_PER_STAGE + 1)
    ]
    for row in trace["episodes"]:
        assert len(row["learning"]) == 1
        updates = row["learning"][0][UAV]
        assert len(updates) == (2 if row["stage"] == "search" else 1)
        assert all("loss" in result and not result.get("skip_update") for result in updates)


def test_partition_baseline_inputs_are_frozen_and_self_contained(tmp_path):
    inputs = build_partition_baseline_role_inputs(tmp_path)
    assert inputs["env_config"].state_dim == STATE_DIM
    assert inputs["env_config"].metadata["game_config"]["map_width"] == 600
    assert inputs["env_config"].metadata["death_penalty_per_drone"] == 10.0
    assert inputs["replay_buffer_config"].tensor_pool == "lazy_memmap"
    assert inputs["replay_buffer_config"].sample_packaging == "independent"
    stages = inputs["stage_task_config"].stages
    assert [s.stage_name for s in stages] == ["search", "evade"]
    assert [s.train_config.epsilon_decay_episodes for s in stages] == [2250, 750]
    for stage in stages:
        assert stage.train_config.seed == stage.train_config.initialization_seed == SEED
        assert stage.train_config.max_episode == EPISODES_PER_STAGE == 2
        assert not stage.train_config.mid_eval_enabled and stage.train_config.end_eval_episodes == 0
        uav, threats = stage.agents_config
        assert len(uav.agent_groups) == 4 and all(len(g) == 4 for g in uav.agent_groups)
        assert uav.obs_dim == 615 and uav.state_shape == 144
        assert len(threats.agent_groups) == 8 and all(len(g) == 1 for g in threats.agent_groups)
        assert threats.fixed_mode and not threats.obs_agent_id and threats.obs_dim == 173
        assert stage.learners_config[0].algo_config.capacity == CAPACITY
        assert stage.learners_config[0].algo_config.batch_size == BATCH_SIZE


@pytest.mark.parametrize("field,replacement", [
    ("reward", 1.5), ("actions_masks_dones_sha256", "different"), ("loss", 0.2),
])
def test_partition_baseline_detects_process_drift(field, replacement):
    expected = {"episodes": [{"reward": 1.0, "actions_masks_dones_sha256": "original", "loss": 0.1}]}
    actual = copy.deepcopy(expected)
    actual["episodes"][0][field] = replacement
    with pytest.raises(AssertionError, match=field):
        _assert_process_matches(actual, expected)


@pytest.mark.slow
def test_baseline_training_partitions(tmp_path):
    recording = os.environ.get("DODO_RECORD_BASELINE") == "1"
    if not EXPECTED_TRACE.is_file() and not recording:
        pytest.fail("Missing expected_trace.json; record explicitly with DODO_RECORD_BASELINE=1 --dodo-collect")
    env = os.environ.copy()
    for key in ("DODO_PROGRESS_FILE", "DODO_CENTER_FINISHED", "DODO_RECORD_BASELINE"):
        env.pop(key, None)
    env.update({
        "DODO_TASK_ID": TASK_ID, "DODO_BASELINE_TMP": str(tmp_path),
        "CUDA_VISIBLE_DEVICES": "", "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "MPLCONFIGDIR": str(tmp_path / "mpl"),
        "DODORL_TEST_LOG_HANDLERS": "console",
    })
    child_log = tmp_path / "training.log"
    with child_log.open("w") as stream:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), CHILD_FLAG],
            cwd=tmp_path, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=10800,
        )
    stdout = child_log.read_text()
    assert proc.returncode == 0, f"baseline child failed; log={child_log}\n{stdout[-6000:]}"
    assert BEGIN in stdout and END in stdout, stdout[-4000:]
    actual = json.loads(stdout.split(BEGIN, 1)[1].split(END, 1)[0])
    if recording:
        print(BEGIN, json.dumps(actual, sort_keys=True, allow_nan=False), END, sep="\n")
    else:
        mismatches = []
        for baseline in (EXPECTED_TRACE, EXPECTED_TRACE_LINUX):
            try:
                _assert_process_matches(actual, json.loads(baseline.read_text()))
            except AssertionError as exc:
                mismatches.append(f"{baseline.name}: {exc}")
            else:
                break
        else:
            raise AssertionError("process trace matched neither platform baseline:\n" + "\n".join(mismatches))


if __name__ == "__main__":
    if CHILD_FLAG not in sys.argv:
        raise SystemExit(f"usage: {Path(__file__).name} {CHILD_FLAG}")
    os.environ.setdefault("DODO_TASK_ID", TASK_ID)
    os.environ.setdefault("DODORL_TEST_LOG_HANDLERS", "console")
    with tempfile.TemporaryDirectory(prefix="partition-baseline-") as temporary:
        work = Path(os.environ.get("DODO_BASELINE_TMP", temporary))
        work.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(work / "mpl"))
        result = collect_partition_trace(work)
        print(BEGIN, json.dumps(result, sort_keys=True, allow_nan=False), END, sep="\n", flush=True)

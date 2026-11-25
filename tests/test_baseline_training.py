"""基准训练测试：cooperative_search 分阶段串行 QMIX 同种子每局奖励复现。

测试起点是 SerialStageTrainRole：用例自己初始化 Env/Agent/Learner/Stage 等配置类，
再交给 Role。不经过 toml_config，也不读科目 cooperative_search_qmix.toml，
科目调参不会使本基准种子失效。

开发期分别在 macOS/Linux 录制，把打印的常量粘回对应平台的基准：

    DODO_RECORD_BASELINE=1 python tests/test_baseline_training.py --dodo-collect

默认 pytest 逐局精确匹配 macOS 或 Linux 的硬编码奖励，不改写基准。
"""
from __future__ import annotations

import copy
import json
import math
import os
import pprint
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from common.plugins import PluginsConfig, activate_plugins
from config import (
    AgentConfig,
    EnvConfig,
    LearnerConfig,
    QmixConfig,
    ReplayBufferConfig,
    StageConfig,
    StageTaskConfig,
    TrainConfig,
)
from envs.action.model import ActionParameterDefined, StepActionDefined
from portal.plugins.registry import project_plugin_registry
from sample.model import SamplingStrategy


ROOT = Path(__file__).resolve().parents[1]
THREAT_MODEL_FILE = (
    ROOT / "subjects/cooperative_search/final_b1813237_avoid_targets_0_dynamic_threats_agent_net.pth"
)
CHILD_FLAG = "--dodo-collect"
TASK_ID = "idempo01"
SEED = 42
SEARCH_EPISODES = 10
EVADE_EPISODES = 10
# 学习 batch 不超过容量；满池后 FIFO 丢掉最旧一局。
REPLAY_CAPACITY = 10
LEARN_BATCH_SIZE = 4
# epsilon 衰减按原 1300/700 局冻结，避免短课程公式改变探索日程。
SEARCH_EPSILON_DECAY_EPISODES = 975
EVADE_EPSILON_DECAY_EPISODES = 525
FOV_SIZE = 11
STATE_GRID = 50
NUM_DRONES = 4
NUM_THREATS = 3
NUM_TARGETS = 10
MAX_STEP_PER_EPISODE = 600
DRONE_AGENT_NAMES = ["drone_0", "drone_1", "drone_2", "drone_3"]
THREAT_AGENT_NAMES = ["dynamic_threat_0", "dynamic_threat_1", "dynamic_threat_2"]
OBS_DIM = FOV_SIZE * FOV_SIZE * 4 + 3
THREAT_OBS_DIM = 4 + NUM_TARGETS * 3 + (NUM_THREATS - 1) * 3
STATE_DIM = STATE_GRID * STATE_GRID * 2 + (NUM_DRONES + NUM_THREATS) * 4
REWARDS_MARK_BEGIN = "BEGIN_COOP_SEARCH_REWARDS"
REWARDS_MARK_END = "END_COOP_SEARCH_REWARDS"

LEARNING_AGENT_NAME = "uav_team"
FIXED_AGENT_NAME = "dynamic_threats"

# macOS 基准，原始记录保留。
# 录制后粘贴：{"search": [ {uav_team: float}, ×10 ], "evade": [ ... ×10 ]}
# 固定策略 Agent 不入 ReplayBuffer，奖励快照只有学习中的 uav_team。
EXPECTED_REWARDS_MACOS: dict[str, list[dict[str, float]]] = {
    "search": [
        {LEARNING_AGENT_NAME: 11.109654008721844},
        {LEARNING_AGENT_NAME: 3.128304376674285},
        {LEARNING_AGENT_NAME: 15.550927806319823},
        {LEARNING_AGENT_NAME: 9.731856398187908},
        {LEARNING_AGENT_NAME: 8.174444101659276},
        {LEARNING_AGENT_NAME: 9.309642041852918},
        {LEARNING_AGENT_NAME: 17.537132587467767},
        {LEARNING_AGENT_NAME: 21.102979477450376},
        {LEARNING_AGENT_NAME: 8.38944693213962},
        {LEARNING_AGENT_NAME: 15.15630475212654},
    ],
    "evade": [
        {LEARNING_AGENT_NAME: -3.546600685432753},
        {LEARNING_AGENT_NAME: -5.541657667376353},
        {LEARNING_AGENT_NAME: -8.505216141041497},
        {LEARNING_AGENT_NAME: -18.789467990841064},
        {LEARNING_AGENT_NAME: -6.206643620778379},
        {LEARNING_AGENT_NAME: -8.890618256079502},
        {LEARNING_AGENT_NAME: -21.329378910082447},
        {LEARNING_AGENT_NAME: -14.585569183075574},
        {LEARNING_AGENT_NAME: -3.43148305878299},
        {LEARNING_AGENT_NAME: -6.284374419322094},
    ],
}


# Linux, recorded independently with the same seed and configuration.
EXPECTED_REWARDS_LINUX: dict[str, list[dict[str, float]]] = {
    "search": [
        {LEARNING_AGENT_NAME: 11.109654008721845},
        {LEARNING_AGENT_NAME: 3.128304376674285},
        {LEARNING_AGENT_NAME: 15.550927806319825},
        {LEARNING_AGENT_NAME: 9.731856398187908},
        {LEARNING_AGENT_NAME: 8.174444101659276},
        {LEARNING_AGENT_NAME: 9.144731631147536},
        {LEARNING_AGENT_NAME: 17.53713258746777},
        {LEARNING_AGENT_NAME: 23.067030333825794},
        {LEARNING_AGENT_NAME: 7.80433324907485},
        {LEARNING_AGENT_NAME: 14.990940438002522},
    ],
    "evade": [
        {LEARNING_AGENT_NAME: -14.735178171043385},
        {LEARNING_AGENT_NAME: 25.949463339025513},
        {LEARNING_AGENT_NAME: -19.808430201967223},
        {LEARNING_AGENT_NAME: -21.710682927946536},
        {LEARNING_AGENT_NAME: -18.76732091598097},
        {LEARNING_AGENT_NAME: -12.59828693373686},
        {LEARNING_AGENT_NAME: -11.285516191393947},
        {LEARNING_AGENT_NAME: -17.495381699086924},
        {LEARNING_AGENT_NAME: -9.565232376422772},
        {LEARNING_AGENT_NAME: -17.88554416617279},
    ],
}


def _heading_action() -> StepActionDefined:
    return StepActionDefined(
        action_key="heading",
        parameters=[ActionParameterDefined(key="value", values=list(range(12)))],
        add_target=False,
        allow_null_targets=False,
    )


def _make_train_config(
        work_dir: Path,
        *,
        max_episode: int,
        start_epsilon: float,
        epsilon_decay_episodes: int,
        env_metadata: dict,
) -> TrainConfig:
    models = work_dir / "models"
    results = work_dir / "results"
    recordings = work_dir / "recordings"
    config = TrainConfig(
        max_episode=max_episode,
        sample_task_timeout=5400,
        start_epsilon=start_epsilon,
        end_epsilon=0.05,
        epsilon_decay_episodes=epsilon_decay_episodes,
        agent_compose_type="order",
        model_save_dir=str(models),
        model_load_dir=str(models),
        result_save_dir=str(results),
        render_video_dir=str(recordings),
        # 两个阶段共用本轮测试的归档根目录，不复用仓库里固定 TASK_ID 的旧归档。
        artifacts_archive_save_dir=str(work_dir.parent / "archives"),
        mid_eval_enabled=False,
        end_eval_episodes=0,
        seed=SEED,
        initialization_seed=SEED,
        env_metadata=env_metadata,
    )
    config.record_sample = False
    return config


def _make_uav_agent() -> AgentConfig:
    return AgentConfig(
        name=LEARNING_AGENT_NAME,
        device=torch.device("cpu"),
        agent_type="rnn",
        multi_action_mode="combination",
        agent_names=list(DRONE_AGENT_NAMES),
        obs_dim=OBS_DIM,
        obs_last_action=True,
        hidden_dim=128,
        define_action=[_heading_action()],
        state_shape=STATE_DIM,
    )


def _make_threat_agent() -> AgentConfig:
    if not THREAT_MODEL_FILE.is_file():
        raise FileNotFoundError(f"threat_model_file missing: {THREAT_MODEL_FILE}")
    return AgentConfig(
        name=FIXED_AGENT_NAME,
        device=torch.device("cpu"),
        agent_type="rnn",
        multi_action_mode="combination",
        agent_names=list(THREAT_AGENT_NAMES),
        obs_dim=THREAT_OBS_DIM,
        obs_last_action=True,
        hidden_dim=128,
        define_action=[_heading_action()],
        state_shape=STATE_DIM,
        model_file_path=str(THREAT_MODEL_FILE),
        fixed_mode=True,
    )


def _make_uav_learner(*, lr: float, learn_times_per_episode: int) -> LearnerConfig:
    return LearnerConfig(
        name=LEARNING_AGENT_NAME,
        device=torch.device("cpu"),
        algo_type="qmix",
        common_reward=True,
        standardise_rewards=True,
        standardise_returns=False,
        grad_norm_clip=5.0,
        gamma=0.995,
        state_shape=STATE_DIM,
        algo_config=QmixConfig(
            capacity=REPLAY_CAPACITY,
            lr=lr,
            hidden_dim=128,
            double_q=True,
            target_update_interval_or_tau=0.005,
            mixing_embed_dim=32,
            hypernet_layers=2,
            hypernet_embed=64,
            sample_strategy=SamplingStrategy.RECENT_BIASED.value,
            batch_size=LEARN_BATCH_SIZE,
            learn_times_per_episode=learn_times_per_episode,
        ),
    )


def _make_env_config(work_dir: Path) -> EnvConfig:
    return EnvConfig(
        sense_name="cooperative_search",
        use_sim=False,
        max_step_per_episode=MAX_STEP_PER_EPISODE,
        step_interval_secs=30,
        state_dim=STATE_DIM,
        agent_names=list(DRONE_AGENT_NAMES) + list(THREAT_AGENT_NAMES),
        metadata={
            "fov_size": FOV_SIZE,
            "observation_grid_scale": 2,
            "state_grid_width": STATE_GRID,
            "state_grid_height": STATE_GRID,
            "render_mode": "rgb_array",
            "render_every_step": False,
            "dynamic_threats_enabled": True,
            "exploration_reward_per_cell": 0.00082,
            "revisited_empty_penalty_episode_limit": 15,
            "frontier_potential_reward_weight": 0.03,
            "correct_detection_reward_total": 20.0,
            "correct_detection_reward_half_life_ratio": 0.8,
            "false_detection_penalty": 0.3,
            "death_penalty_per_drone": 6.0,
            "save_probability_map_on_episode_end": False,
            "probability_map_every_n_episodes": 100,
            "probability_map_output_dir": str(work_dir / "results"),
            "game_config": {
                "map_width": 300,
                "map_height": 300,
                "belief_grid_width": 100,
                "belief_grid_height": 100,
                "num_drones": NUM_DRONES,
                "num_dynamic_threats": NUM_THREATS,
                "num_targets": NUM_TARGETS,
                "sensor_radius": 40.0,
                "threat_sensor_radius": 30.0,
                "threat_warning_radius": 60.0,
                "communication_radius": 100.0,
                "initial_separation": 20.0,
                "initial_formation_side_km": 30.0,
                "initial_threat_separation": 60.0,
                "detection_probability": 0.8,
                "false_alarm_probability": 0.2,
                "detection_threshold": 0.90,
                "min_belief_probability": 0.10,
                "max_belief_probability": 0.90,
                "max_turn_degrees": 30.0,
                "fixed_speed_mps": 100.0,
                "threat_speed_mps": 80.0,
                "step_interval_secs": 30.0,
                "max_steps": MAX_STEP_PER_EPISODE,
                "fps": 20,
                "sidebar_width": 300,
            },
        },
    )


def build_baseline_role_inputs(work_dir: Path) -> dict:
    """构造 SerialStageTrainRole 入参；实验超参只写在本函数，与科目 TOML 无关。"""
    agents = [_make_uav_agent(), _make_threat_agent()]
    search_learner = _make_uav_learner(lr=0.0002, learn_times_per_episode=2)
    evade_learner = _make_uav_learner(lr=0.0001, learn_times_per_episode=1)
    search = StageConfig(
        stage_name="search",
        train_config=_make_train_config(
            work_dir / "search",
            max_episode=SEARCH_EPISODES,
            start_epsilon=1.0,
            epsilon_decay_episodes=SEARCH_EPSILON_DECAY_EPISODES,
            env_metadata={"dynamic_threats_enabled": False},
        ),
        agents_config=copy.deepcopy(agents),
        learners_config=[search_learner],
        clear_replay_buffer=True,
        rehearsal_archive_ratio=0.25,
        rehearsal_archive_select="even_span",
    )
    evade = StageConfig(
        stage_name="evade",
        train_config=_make_train_config(
            work_dir / "evade",
            max_episode=EVADE_EPISODES,
            start_epsilon=0.3,
            epsilon_decay_episodes=EVADE_EPSILON_DECAY_EPISODES,
            env_metadata={"dynamic_threats_enabled": True},
        ),
        agents_config=copy.deepcopy(agents),
        learners_config=[evade_learner],
        clear_replay_buffer=True,
        rehearsal_sample_ratio=0.25,
    )
    return {
        "stage_task_config": StageTaskConfig(
            stages=[search, evade],
            bootstrap_learners_config=[copy.deepcopy(search_learner)],
        ),
        "env_config": _make_env_config(work_dir),
        "sim_config": None,
        "replay_buffer_config": ReplayBufferConfig(
            batch_size=LEARN_BATCH_SIZE,
            recent_ratio=0.35,
            seed=SEED,
        ),
        "model_manager_config": None,
        "sim_manager_config": None,
    }


def _pushed_replay_agent_names(episode_data, agent_name: str | None = None) -> tuple[str, ...]:
    episodes = getattr(episode_data, "episodes", None)
    if episodes is not None:
        episode_data = episodes
    if isinstance(episode_data, dict):
        return tuple(episode_data)
    if agent_name is not None:
        return (agent_name,)
    return ()


def _assert_fixed_agent_not_in_replay(replay_buffer, pushed_batches: list[tuple[str, ...]]) -> None:
    pushed_names = [name for batch in pushed_batches for name in batch]
    assert LEARNING_AGENT_NAME in pushed_names, (
        f"learning agent {LEARNING_AGENT_NAME} must be pushed to ReplayBuffer, "
        f"pushed_batches={pushed_batches}"
    )
    assert FIXED_AGENT_NAME not in pushed_names, (
        f"fixed agent {FIXED_AGENT_NAME} must not be pushed to ReplayBuffer, "
        f"pushed_batches={pushed_batches}"
    )
    buffer_names = replay_buffer.agent_names()
    assert LEARNING_AGENT_NAME in buffer_names, (
        f"learning agent {LEARNING_AGENT_NAME} missing from ReplayBuffer, "
        f"agent_names={buffer_names}"
    )
    assert FIXED_AGENT_NAME not in buffer_names, (
        f"fixed agent {FIXED_AGENT_NAME} must not occupy ReplayBuffer, "
        f"agent_names={buffer_names}"
    )
    assert replay_buffer.agent_buffer_size(FIXED_AGENT_NAME) == 0


def collect_stage_rewards(work_dir: Path | None = None) -> dict[str, list[dict[str, float]]]:
    """初始化基准配置类，构造 SerialStageTrainRole 并采集每局 episode_reward。"""
    os.environ["DODO_TASK_ID"] = TASK_ID
    os.chdir(ROOT)
    work = Path(work_dir or os.environ.get("DODO_BASELINE_TMP") or tempfile.mkdtemp(prefix="coop-baseline-"))
    work.mkdir(parents=True, exist_ok=True)

    activate_plugins(PluginsConfig(), default_plugins=project_plugin_registry())
    role_inputs = build_baseline_role_inputs(work)

    from roles.serial_stage_train import SerialStageTrainRole

    stages = role_inputs["stage_task_config"].stages
    collected: dict[str, list[dict[str, float]]] = {stage.stage_name: [] for stage in stages}
    original = SerialStageTrainRole._log_episode_result
    role = SerialStageTrainRole(**role_inputs)
    original_push = role.replay_buffer.push
    pushed_batches: list[tuple[str, ...]] = []

    def _tracking_push(episode_data, *args, **kwargs):
        pushed_batches.append(
            _pushed_replay_agent_names(episode_data, kwargs.get("agent_name"))
        )
        return original_push(episode_data, *args, **kwargs)

    def _hook(stage, episode_transitions):
        rewards = {
            name: float(sum(t.episode_reward() for t in transitions))
            for name, transitions in episode_transitions.items()
        }
        assert FIXED_AGENT_NAME not in rewards, (
            f"fixed agent {FIXED_AGENT_NAME} must not appear in episode_transitions "
            f"stage={stage.stage_name} rewards={rewards}"
        )
        collected.setdefault(stage.stage_name, []).append(rewards)
        return original(stage, episode_transitions)

    role.replay_buffer.push = _tracking_push
    SerialStageTrainRole._log_episode_result = staticmethod(_hook)
    try:
        role.run()
        _assert_fixed_agent_not_in_replay(role.replay_buffer, pushed_batches)
    finally:
        role.replay_buffer.push = original_push
        SerialStageTrainRole._log_episode_result = original
    return collected


def _emit_rewards(rewards: dict[str, list[dict[str, float]]]) -> None:
    print(REWARDS_MARK_BEGIN)
    print(json.dumps(rewards, sort_keys=True))
    print(REWARDS_MARK_END)
    if os.environ.get("DODO_RECORD_BASELINE") == "1":
        print("SEED =", SEED)
        platform_name = "MACOS" if sys.platform == "darwin" else "LINUX"
        print(f"EXPECTED_REWARDS_{platform_name} =", pprint.pformat(rewards, width=120, sort_dicts=True))


def _parse_child_rewards(stdout: str) -> dict[str, list[dict[str, float]]]:
    if REWARDS_MARK_BEGIN not in stdout or REWARDS_MARK_END not in stdout:
        raise AssertionError(f"child process did not emit reward payload\n{stdout[-4000:]}")
    payload = stdout.split(REWARDS_MARK_BEGIN, 1)[1].split(REWARDS_MARK_END, 1)[0].strip()
    return json.loads(payload)


def _assert_rewards_match(
    actual: dict[str, list[dict[str, float]]],
    expected_macos: dict[str, list[dict[str, float]]],
    expected_linux: dict[str, list[dict[str, float]]],
) -> None:
    for stage_name, expected_episodes in expected_macos.items():
        linux_episodes = expected_linux.get(stage_name)
        assert linux_episodes is not None, f"missing Linux baseline stage {stage_name}"
        assert len(linux_episodes) == len(expected_episodes), f"baseline episode counts differ for {stage_name}"
        actual_episodes = actual.get(stage_name)
        assert actual_episodes is not None, f"missing stage {stage_name}"
        assert len(actual_episodes) == len(expected_episodes), (
            f"{stage_name} episode count actual={len(actual_episodes)} expected={len(expected_episodes)}"
        )
        for index, (got, want_macos, want_linux) in enumerate(
                zip(actual_episodes, expected_episodes, linux_episodes)):
            if got != want_macos and got != want_linux:
                raise AssertionError(
                    f"reward mismatch stage={stage_name} episode={index} actual={got} "
                    f"expected_macos={want_macos} expected_linux={want_linux}"
                )


def test_baseline_role_inputs_are_self_contained(tmp_path):
    role_inputs = build_baseline_role_inputs(tmp_path)
    assert role_inputs["model_manager_config"] is None
    env_config = role_inputs["env_config"]
    assert env_config.use_sim is False
    assert env_config.sense_name == "cooperative_search"
    assert env_config.metadata["render_mode"] == "rgb_array"
    assert env_config.metadata["game_config"]["threat_warning_radius"] == 60.0

    stages = role_inputs["stage_task_config"].stages
    assert [stage.stage_name for stage in stages] == ["search", "evade"]
    assert stages[0].train_config.max_episode == SEARCH_EPISODES
    assert stages[1].train_config.max_episode == EVADE_EPISODES
    replay_config = role_inputs["replay_buffer_config"]
    assert stages[0].learners_config[0].algo_config.capacity == REPLAY_CAPACITY
    assert replay_config.seed == SEED
    for stage in stages:
        assert Path(stage.train_config.artifacts_archive_save_dir) == tmp_path / "archives"
        assert stage.train_config.seed == SEED
        assert stage.train_config.initialization_seed == SEED
        assert stage.train_config.mid_eval_enabled is False
        assert stage.train_config.end_eval_episodes == 0
        for agent in stage.agents_config:
            assert agent.device == torch.device("cpu")
        for learner in stage.learners_config:
            assert learner.device == torch.device("cpu")
            assert learner.algo_config.batch_size == LEARN_BATCH_SIZE

    threat_agent = next(
        agent for agent in stages[0].agents_config if agent.name == FIXED_AGENT_NAME
    )
    assert threat_agent.fixed_mode is True
    assert Path(threat_agent.model_file_path).is_file()


@pytest.mark.parametrize("value", [11.109654008721844, 11.109654008721845])
def test_reward_assertion_accepts_either_exact_platform_value(value):
    macos = {"search": [{"uav_team": 11.109654008721844}]}
    linux = {"search": [{"uav_team": 11.109654008721845}]}
    _assert_rewards_match({"search": [{"uav_team": value}]}, macos, linux)


def test_reward_assertion_rejects_values_outside_both_platform_baselines():
    macos = {"search": [{"uav_team": 11.109654008721844}]}
    linux = {"search": [{"uav_team": 11.109654008721845}]}
    different_value = math.nextafter(linux["search"][0]["uav_team"], math.inf)
    with pytest.raises(AssertionError, match="reward mismatch"):
        _assert_rewards_match({"search": [{"uav_team": different_value}]}, macos, linux)


@pytest.mark.slow
def test_baseline_training(tmp_path):
    if not EXPECTED_REWARDS_MACOS.get("search") and os.environ.get("DODO_RECORD_BASELINE") != "1":
        pytest.fail(
            "EXPECTED_REWARDS_MACOS is empty; record once with "
            "DODO_RECORD_BASELINE=1 python tests/test_baseline_training.py --dodo-collect"
        )

    env = os.environ.copy()
    env["DODO_TASK_ID"] = TASK_ID
    env["DODO_BASELINE_TMP"] = str(tmp_path)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    (tmp_path / "mpl").mkdir(parents=True, exist_ok=True)
    pythonpath = os.pathsep.join([str(ROOT / "src"), str(ROOT), str(ROOT / "sdk"), env.get("PYTHONPATH", "")])
    env["PYTHONPATH"] = pythonpath
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), CHILD_FLAG],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=10800,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"baseline training child failed rc={proc.returncode}\nstdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
        )
    actual = _parse_child_rewards(proc.stdout)
    if os.environ.get("DODO_RECORD_BASELINE") == "1":
        _emit_rewards(actual)
        return
    _assert_rewards_match(actual, EXPECTED_REWARDS_MACOS, EXPECTED_REWARDS_LINUX)


if __name__ == "__main__":
    if CHILD_FLAG not in sys.argv:
        raise SystemExit(
            f"usage: DODO_RECORD_BASELINE=1 python {Path(__file__).name} {CHILD_FLAG}"
        )
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "sdk"))
    os.environ.setdefault("DODO_TASK_ID", TASK_ID)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    work = Path(os.environ.get("DODO_BASELINE_TMP") or tempfile.mkdtemp(prefix="coop-baseline-"))
    os.environ.setdefault("MPLCONFIGDIR", str(work / "mpl"))
    (work / "mpl").mkdir(parents=True, exist_ok=True)
    rewards = collect_stage_rewards(work)
    _emit_rewards(rewards)

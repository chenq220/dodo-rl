"""功能包三：单 Agent 无名分组重复推理与组内 QMIX 的定向测试。"""
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch

from algos.agents import resolve_agent_groups
from common.exception import (
    AgentGroupCompatibilityException,
    AgentGroupConfigException,
)
from config import AgentConfig, EnvConfig, LearnerConfig, QmixConfig, ReplayBufferConfig, RnnAgentConfig
from envs.action.base import ActionTable
from envs.action.model import StepActionDefined
from envs.sense.paralle_examples import CounterParallelEnvironment
from roles.common.config_alignment import assert_state_shape_matches_env
from sample.replay.replay_payload import (
    ReplayPushEntryKey,
    ReplayPushEnvelope,
    encode_replay_push_payload,
)
from sample.replay_buffer import ReplayBuffer
from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition
from scheduler.model import AgentCompose, ActionParam, EpisodeInfo
from scheduler.rollout_sample import RolloutSampleWorker
from utils.module_utils import align_agent_configs

AGENT_NAME = "uav_team"
GROUPS = (
    ("uav_0", "uav_1", "uav_2", "uav_3"),
    ("uav_4", "uav_5", "uav_6", "uav_7"),
    ("uav_8", "uav_9", "uav_10", "uav_11"),
)
FLAT_IDS = [agent_id for group in GROUPS for agent_id in group]


# ---------------------------------------------------------------------------
# 配置与装配
# ---------------------------------------------------------------------------

def _agent_config(**overrides) -> AgentConfig:
    values = dict(
        name=AGENT_NAME,
        agent_type="rnn",
        agent_names=[],
        agent_groups=[list(group) for group in GROUPS],
        obs_dim=8,
    )
    values.update(overrides)
    return AgentConfig(**values)


def test_legacy_agent_names_resolves_to_single_implicit_group():
    config = _agent_config(agent_names=["a0", "a1", "a2"], agent_groups=[])
    groups = resolve_agent_groups(config)
    assert groups == (("a0", "a1", "a2"),)


def test_agent_groups_keep_declaration_order_and_flat_view():
    groups = resolve_agent_groups(_agent_config())
    assert groups == GROUPS


def test_agent_groups_conflict_empty_unequal_and_duplicate_rejected():
    with pytest.raises(AgentGroupConfigException, match="不能同时非空"):
        resolve_agent_groups(_agent_config(agent_names=["a0"], agent_groups=[["b0"]]))
    with pytest.raises(AgentGroupConfigException, match="至少需要一个非空"):
        resolve_agent_groups(_agent_config(agent_names=[], agent_groups=[]))
    with pytest.raises(AgentGroupConfigException, match="内层组必须是非空列表"):
        resolve_agent_groups(_agent_config(agent_groups=[["a0"], []]))
    with pytest.raises(AgentGroupConfigException, match="跨组重复"):
        resolve_agent_groups(_agent_config(agent_groups=[["a0"], ["a0"]]))
    with pytest.raises(AgentGroupConfigException, match="必须等宽"):
        resolve_agent_groups(_agent_config(agent_groups=[["a0", "a1"], ["b0"]]))


def test_grouped_agent_rejects_non_rnn_agent_type():
    config = _agent_config(agent_type="sb3")
    with pytest.raises(AgentGroupCompatibilityException, match="仅支持 agent_type=rnn"):
        from algos.modules.agents.sb3_agent import SB3Agent
        SB3Agent(
            agent_type="sb3",
            config=config,
            agent_net_source="blanco",
            device="cpu",
        )


def test_grouped_trainable_agent_requires_qmix_learner():
    agent_config = _agent_config()
    learner_config = LearnerConfig(name=AGENT_NAME, algo_type="mappo")
    with pytest.raises(AgentGroupCompatibilityException, match="仅支持 qmix Learner"):
        align_agent_configs([agent_config], [learner_config])


def test_grouped_state_shape_skips_env_dim_and_requires_learner_match():
    env_config = EnvConfig(state_dim=12, use_sim=False)
    agent_config = _agent_config(state_shape=(6,))
    learner_config = LearnerConfig(
        name=AGENT_NAME, algo_type="qmix", algo_config=QmixConfig(), state_shape=(6,),
    )
    assert_state_shape_matches_env(env_config, [agent_config], [learner_config])

    learner_config.state_shape = 1
    with pytest.raises(Exception, match="dimensions do not match"):
        assert_state_shape_matches_env(env_config, [agent_config], [learner_config])


def test_toml_translates_agent_groups_without_normalization(tmp_path):
    from utils.toml_config import TomlTrainingConfig

    config_file = tmp_path / "grouped.toml"
    config_file.write_text(
        '''
[agents.0]
name = "uav_team"
agent_type = "rnn"
agent_groups = [
    ["uav_0", "uav_1", "uav_2", "uav_3"],
    ["uav_4", "uav_5", "uav_6", "uav_7"],
]

[learners.0]
name = "uav_team"
algo_type = "qmix"
''',
        encoding="utf-8",
    )

    config = TomlTrainingConfig(config_file, project_root=tmp_path)
    agent_config = config.agents_config[0]
    # TOML 只做直通翻译：二维数组原样落入字段，不推导 agent_names。
    assert agent_config.agent_groups == [
        ["uav_0", "uav_1", "uav_2", "uav_3"],
        ["uav_4", "uav_5", "uav_6", "uav_7"],
    ]
    assert agent_config.agent_names == []
    assert resolve_agent_groups(agent_config) == (
        ("uav_0", "uav_1", "uav_2", "uav_3"),
        ("uav_4", "uav_5", "uav_6", "uav_7"),
    )


# ---------------------------------------------------------------------------
# RNNAgent 组内槽位推理
# ---------------------------------------------------------------------------

def _build_rnn_agent(groups=GROUPS, obs_dim=8):
    from algos.modules.agents.rnn_agent import RNNAgent

    config = AgentConfig(
        name=AGENT_NAME,
        agent_type="rnn",
        agent_names=[],
        agent_groups=[list(group) for group in groups],
        obs_dim=obs_dim,
        algo_config=RnnAgentConfig(),
        define_action=[
            StepActionDefined(action_key="down", parameters=[], add_target=False),
            StepActionDefined(action_key="up", parameters=[], add_target=False),
        ],
    )
    return RNNAgent(
        agent_type="rnn",
        config=config,
        agent_net_source="blanco",
        device="cpu",
    )


def test_rnn_agent_builds_dual_view_and_group_width():
    agent = _build_rnn_agent()
    assert agent.agent_groups == GROUPS
    assert agent.agent_list == FLAT_IDS
    assert agent.n_agents == 4
    assert agent.dim_info.n_agents == 4


def test_rnn_agent_groups_share_single_net_and_isolate_hidden():
    from scheduler.rollout_sample import _agent_group_view

    agent = _build_rnn_agent()
    hidden = agent.get_init_hidden_state()
    obs0 = {agent_id: np.ones(8, dtype=np.float32) for agent_id in GROUPS[0]}
    obs1 = {agent_id: np.ones(8, dtype=np.float32) for agent_id in GROUPS[1]}

    # 组视图下推理：agent_list 切成当前组成员，网络按组宽槽位执行。
    with _agent_group_view(agent, GROUPS[0]):
        actions0, hidden_out0 = agent.get_actions(
            obs=obs0, agent_list=list(GROUPS[0]), hidden_in=hidden,
            action_mask=None, epsilon=0.0, evaluate=True,
        )
        net_identity = id(agent.agent_net)
        init_hidden0 = agent.get_init_hidden_state()
    with _agent_group_view(agent, GROUPS[1]):
        actions1, hidden_out1 = agent.get_actions(
            obs=obs1, agent_list=list(GROUPS[1]), hidden_in=hidden,
            action_mask=None, epsilon=0.0, evaluate=True,
        )

    assert id(agent.agent_net) == net_identity
    # 视图退出后恢复全量成员表
    assert agent.agent_list == FLAT_IDS
    # 相同槽位、相同输入、相同 hidden 在确定性模式下得到相同动作
    assert [a.action_idx for a in actions0.values()] == [a.action_idx for a in actions1.values()]
    # 一个组的 hidden 输出只包含本组成员，不串组
    assert set(hidden_out0) == set(GROUPS[0])
    assert set(hidden_out1) == set(GROUPS[1])
    # 组视图下的初始 hidden 只为本组成员建立槽位
    assert set(init_hidden0) == set(GROUPS[0])
    for agent_id in GROUPS[1]:
        assert agent_id not in hidden_out0


def test_rnn_agent_partial_death_keeps_slot_layout():
    from scheduler.rollout_sample import _agent_group_view

    agent = _build_rnn_agent()
    hidden = agent.get_init_hidden_state()
    obs = {agent_id: np.ones(8, dtype=np.float32) for agent_id in GROUPS[0]}
    alive = ["uav_1", "uav_3"]
    with _agent_group_view(agent, GROUPS[0]):
        actions, hidden_out = agent.get_actions(
            obs=obs, agent_list=alive, hidden_in=hidden,
            action_mask=None, epsilon=0.0, evaluate=True,
        )
    assert set(actions) == set(alive)
    assert set(hidden_out) == set(alive)
    # 死亡成员 uav_0/uav_2 的历史 hidden 保持不变
    for dead in ("uav_0", "uav_2"):
        assert dead not in hidden_out


# ---------------------------------------------------------------------------
# Parallel Rollout 分组
# ---------------------------------------------------------------------------

class _GroupedCounterEnvironment(CounterParallelEnvironment):
    """12 机三组的计数器环境：观测拼接动作缓存上下文，支持组级真实终止。"""

    def __init__(self, max_steps: int = 8, group_done_at: Optional[dict[int, int]] = None):
        super().__init__(list(FLAT_IDS), max_steps=max_steps)
        # group_index -> 达到该物理步后该组全员真实终止
        self._group_done_at = dict(group_done_at or {})
        # 已真实终止的固定槽位；动作掩码返回全零，供终帧预览识别死亡成员。
        self._dead_agent_ids: set[str] = set()
        self.obs_calls = 0
        self.step_calls = 0
        self.step_action_counts: list[int] = []

    def get_avail_agent_actions(self, agent_name: str, init: bool = False) -> list[int]:
        if agent_name in self._dead_agent_ids:
            return [0, 0]
        return [1, 1]

    def get_obs(self) -> dict[str, np.ndarray]:
        self.obs_calls += 1
        base = super().get_obs()
        # 顺序协同上下文：观测拼接当前决策步已发布的动作索引之和。
        buffer_ctx = float(sum(action.action_idx for action in self.multi_agent_action_buffer.values()))
        return {
            agent_id: np.asarray([obs[0], buffer_ctx], dtype=np.float32)
            for agent_id, obs in base.items()
        }

    def state_for_agent_ids(self, *, agent_name, agent_ids) -> np.ndarray:
        full = self.state()
        index = {agent_id: idx for idx, agent_id in enumerate(self.possible_agents)}
        return np.asarray([full[index[agent_id]] for agent_id in agent_ids], dtype=np.float32)

    def send_actions(self, actions):
        # 测试环境不过滤非活动成员：已终止成员由 step() 用占位动作补齐固定槽位。
        self.sim.send_actions(actions)
        self.current_actions = actions.copy()
        return self.current_actions

    def step(self, actions):
        self.step_calls += 1
        self.step_action_counts.append(len(actions))
        # 已真实终止的成员不再提交动作；用占位动作补齐固定槽位供内存仿真消费。
        full_actions = dict(actions)
        placeholder = self.action_table_placeholder()
        for group_index in self._group_done_at:
            for agent_id in GROUPS[group_index]:
                if agent_id not in full_actions:
                    full_actions[agent_id] = placeholder
        obs, rewards, terminations, truncations, infos = super().step(full_actions)
        # step_num 从 1 起计且在步内自增：第 done_step 次物理步结束后 step_num == done_step+1。
        for group_index, done_step in self._group_done_at.items():
            if self.step_num > done_step:
                for agent_id in GROUPS[group_index]:
                    terminations[agent_id] = True
                self._dead_agent_ids.update(GROUPS[group_index])
                self.agents = [a for a in self.agents if a not in GROUPS[group_index]]
        return obs, rewards, terminations, truncations, infos

    @staticmethod
    def action_table_placeholder():
        # 带零增量的占位动作：补齐固定槽位且不影响存活成员的计数器结算。
        return SimpleNamespace(action_idx=0, parameters={"delta": 0.0})


class _RecordingAgent:
    """记录逐组推理调用的策略替身；组循环内经 agent_list 组视图调用。"""

    def __init__(self, name, groups, fixed_mode=False):
        self.name = name
        self.agent_groups = tuple(tuple(group) for group in groups)
        self.agent_list = [aid for group in self.agent_groups for aid in group]
        self.agent_type = "rnn"
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(agent_output_type="q", fixed_mode=fixed_mode)
        self.dim_info = SimpleNamespace(hidden_dim=4)
        self.action_table = ActionTable([
            StepActionDefined(action_key="down", parameters=[], add_target=False),
            StepActionDefined(action_key="up", parameters=[], add_target=False),
        ])
        self.get_actions_calls: list[tuple] = []
        self.n_agents = len(self.agent_groups[0])

    def get_init_hidden_state(self):
        # Rollout 在组视图下逐组调用；此时 agent_list 已是当前组成员。
        return {
            agent_id: torch.zeros(1, self.dim_info.hidden_dim)
            for agent_id in self.agent_list
        }

    def get_actions(self, obs, agent_list, hidden_in, **kwargs):
        self.get_actions_calls.append(tuple(agent_list))
        # 固定选 action_idx=1，使观测中的动作缓存上下文等于已发布动作数。
        action = self.action_table.action_map[1]
        return (
            {agent_id: action for agent_id in agent_list},
            {agent_id: hidden_in[agent_id].clone() for agent_id in agent_list},
        )


class _RecordingReplayBuffer:
    def __init__(self):
        self.items = []
        self.notify_calls = 0

    def push(self, episode_data, **kwargs):
        self.items.append(episode_data)

    def acquire_replay_push_permit(self, *args, **kwargs):
        return None

    def release_replay_push_permit(self, *args, **kwargs):
        return None


def _group_episode_info(agent_name=AGENT_NAME):
    return EpisodeInfo(
        episode_num=1,
        task_id="group-test",
        epsilon=0.0,
        deterministic=True,
        temperature=1.0,
        actions_param={
            agent_name: ActionParam(epsilon=0.0, deterministic=True, temperature=1.0)
        },
        seed=11,
        start_datetime=None,
        end_datetime=None,
        metadata={},
    )


def _group_worker(env, agent, replay=None):
    return RolloutSampleWorker(
        env=env,
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=replay or _RecordingReplayBuffer(),
        agent_compose=AgentCompose(step_agent=[agent]),
        policy_agents=[agent],
    )


def test_grouped_parallel_rollout_runs_three_inferences_and_one_env_step():
    env = _GroupedCounterEnvironment(max_steps=3)
    agent = _RecordingAgent(AGENT_NAME, GROUPS)
    worker = _group_worker(env, agent)

    transitions = worker.rolling_parallel_episode(_group_episode_info(), step_limit=3)

    # 每个物理步同一 Agent 重复推理 3 次（3 组 × 3 步），无额外推理。
    assert len(agent.get_actions_calls) == 9
    # env.step 每物理步只执行一次且收到 12 个动作
    assert env.step_calls == 3
    assert env.step_action_counts == [12, 12, 12]
    # 返回统一列表容器：3 条组轨迹，每条只含本组成员
    assert list(transitions) == [AGENT_NAME]
    assert len(transitions[AGENT_NAME]) == 3
    for group_index, transition in enumerate(transitions[AGENT_NAME]):
        assert tuple(transition.agent_ids) == GROUPS[group_index]
        assert transition.group_index == group_index
        assert transition.seq_len == 3
    # AgentName 汇总回报 = 各组轨迹回报之和
    assert worker.total_rewards[AGENT_NAME] == sum(
        t.episode_reward() for t in transitions[AGENT_NAME]
    )


def test_grouped_rollout_publishes_actions_and_freezes_decision_frames():
    env = _GroupedCounterEnvironment(max_steps=2)
    agent = _RecordingAgent(AGENT_NAME, GROUPS)
    worker = _group_worker(env, agent)

    transitions = worker.rolling_parallel_episode(_group_episode_info(), step_limit=2)[AGENT_NAME]

    # 决策顺序可见性：组0观测不含本步任何动作上下文，组2可读前两组。
    group0, group1, group2 = transitions
    for t in range(2):
        assert group0.obs["uav_0"][t][1] == 0.0
        assert group1.obs["uav_4"][t][1] == 4.0
        assert group2.obs["uav_8"][t][1] == 8.0
        # 决策帧的当前动作掩码来自冻结帧
        assert group0.avail_actions[t]["uav_0"] == [1, 1]

    # 铺平迭代天然对齐 next 帧：中间步的 next_* 逐值等于下一时间步本组真实输入
    # （组1 的 t+1 帧含组0 在 t+1 发布的新动作，而非 t 步残留）。
    for transition in transitions:
        for t in range(transition.seq_len - 2):
            for agent_id in transition.agent_ids:
                np.testing.assert_array_equal(
                    transition.next_obs[agent_id][t],
                    transition.obs[agent_id][t + 1],
                )
                np.testing.assert_array_equal(
                    transition.next_states[t],
                    transition.states[t + 1],
                )
    # 末帧为 env.step 后立即采集（旧语义）：动作缓存仍含本步 12 个动作上下文。
    for transition in transitions:
        assert transition.next_obs[transition.agent_ids[0]][-1][1] == 12.0

    # 组状态为按成员裁剪的向量，且 next_state 对齐下一决策帧
    assert group0.states[0].shape == (4,)
    np.testing.assert_array_equal(group0.next_states[0], group0.states[1])

    # 活动掩码：全部存活时组内全1
    assert group0.active_agent_mask["uav_0"] == [1.0, 1.0]


def test_grouped_rollout_terminal_group_closes_while_others_continue():
    env = _GroupedCounterEnvironment(max_steps=4, group_done_at={0: 2})
    agent = _RecordingAgent(AGENT_NAME, GROUPS)
    worker = _group_worker(env, agent)

    transitions = worker.rolling_parallel_episode(_group_episode_info(), step_limit=4)[AGENT_NAME]
    group0, group1, group2 = transitions

    # 组0 在第2步真实终止：轨迹长度2且最后一帧不再 bootstrap
    assert group0.seq_len == 2
    assert group0.next_active_agent_mask["uav_0"][-1] == 0.0
    assert group0.terminations["uav_0"][-1] is True
    # 其他组继续到局末
    assert group1.seq_len == 4
    assert group2.seq_len == 4
    # 后续物理步只推理存活两组；组0 终止后不再产生推理
    assert len(agent.get_actions_calls) == 3 * 2 + 2 * 2


def test_grouped_rollout_time_limit_truncation_keeps_bootstrap():
    env = _GroupedCounterEnvironment(max_steps=8)
    agent = _RecordingAgent(AGENT_NAME, GROUPS)
    worker = _group_worker(env, agent)

    transitions = worker.rolling_parallel_episode(_group_episode_info(), step_limit=2)[AGENT_NAME]

    for transition in transitions:
        first_member = transition.agent_ids[0]
        assert transition.seq_len == 2
        # 单纯时限截断不是真实终止：终帧保留可继续槽位供 bootstrap
        assert transition.terminations[first_member][-1] is False
        assert transition.next_active_agent_mask[first_member][-1] == 1.0
        # 终帧 next_obs 来自截断时刻的只读预览
        assert transition.next_obs[first_member][-1].shape == (2,)


def test_grouped_fixed_mode_agent_acts_but_records_nothing():
    env = _GroupedCounterEnvironment(max_steps=2)
    agent = _RecordingAgent(AGENT_NAME, GROUPS, fixed_mode=True)
    replay = _RecordingReplayBuffer()
    worker = _group_worker(env, agent, replay)

    transitions = worker.rolling_parallel_episode(_group_episode_info(), step_limit=2)

    # 固定策略分组正常出动作（3组×2步），但无轨迹、无入池
    assert len(agent.get_actions_calls) == 6
    assert env.step_action_counts == [12, 12]
    assert transitions == {}
    assert replay.items == []


def test_grouped_aec_rollout_rejected_at_assembly():
    from envs.sense.aec_examples import MultiAgentCounterAECEnvironment

    env = MultiAgentCounterAECEnvironment(max_steps=1)
    agent = _RecordingAgent(AGENT_NAME, GROUPS)
    with pytest.raises(AgentGroupCompatibilityException, match="不支持 AEC 编排"):
        _group_worker(env, agent)


def test_grouped_add_step_rejects_foreign_members():
    transition = EpisodeTransition(agent_ids=GROUPS[0], group_index=0)
    obs = {"uav_0": np.zeros(2, dtype=np.float32), "intruder": np.zeros(2, dtype=np.float32)}
    with pytest.raises(Exception, match="不属于本组成员"):
        transition.add_step(
            obs=obs,
            action={"uav_0": None, "intruder": None},
            last_action={},
            reward={"uav_0": 0.0, "intruder": 0.0},
            next_obs=obs,
            termination={"uav_0": False, "intruder": False},
            truncation={"uav_0": False, "intruder": False},
            state=np.zeros(4),
            next_state=np.zeros(4),
            avail_actions_next=None,
        )


# ---------------------------------------------------------------------------
# ReplayBuffer 多轨迹原子入池
# ---------------------------------------------------------------------------

def _ma_group_episode(group, episode_num, steps=3):
    transition = EpisodeTransition(agent_ids=tuple(group), group_index=0)
    transition.episode_num = episode_num
    action = SimpleNamespace(action_idx=0)
    for step in range(steps):
        obs = {agent_id: np.full(4, float(step), dtype=np.float32) for agent_id in group}
        transition.add_step(
            obs=dict(obs),
            action={agent_id: action for agent_id in group},
            last_action={agent_id: None for agent_id in group},
            reward={agent_id: 0.5 for agent_id in group},
            next_obs={agent_id: np.full(4, float(step + 1), dtype=np.float32) for agent_id in group},
            termination={agent_id: False for agent_id in group},
            truncation={agent_id: step == steps - 1 for agent_id in group},
            state=np.full(4, float(step), dtype=np.float32),
            next_state=np.full(4, float(step + 1), dtype=np.float32),
            avail_actions_next={agent_id: [1, 1] for agent_id in group},
            avail_actions={agent_id: [1, 1] for agent_id in group},
            metadata={"episode_num": episode_num, "step": step + 1, "group_index": 0},
            active_agent_ids=set(group),
            next_active_agent_ids=set(group) if step < steps - 1 else set(),
        )
    transition.set_hidden(
        {agent_id: torch.zeros(1, 4) for agent_id in group},
        {agent_id: torch.zeros(1, 4) for agent_id in group},
    )
    return transition


def test_replay_push_accepts_same_name_list_with_consecutive_write_seq():
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 8}, batch_size=1))
    notify_calls = []
    replay.learn_batches_update_caller = lambda state, episode: notify_calls.append(state)

    transitions = [_ma_group_episode(GROUPS[i], episode_num=5) for i in range(3)]
    replay.push({AGENT_NAME: transitions}, notify_learning=True, episode_num=5)

    assert replay.agent_buffer_size(AGENT_NAME) == 3
    assert replay.write_count[AGENT_NAME] == 3
    # 一次物理 push 只通知一次学习
    assert len(notify_calls) == 1
    assert notify_calls[0].agent_sizes[AGENT_NAME] == 3
    # write_seq 连续且按列表顺序
    metas = replay.buffer[AGENT_NAME]
    assert [meta.write_seq for meta in metas] == [1, 2, 3]
    assert [meta.episode_num for meta in metas] == [5, 5, 5]


def test_replay_push_envelope_v2_encodes_entry_keys_and_round_trips():
    import pickle

    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 8}, batch_size=1))
    transitions = [_ma_group_episode(GROUPS[i], episode_num=7) for i in range(3)]
    for transition in transitions:
        transition.prepare_for_replay()

    envelope = encode_replay_push_payload({AGENT_NAME: transitions})
    assert isinstance(envelope, ReplayPushEnvelope)
    assert envelope.version == ReplayPushEnvelope.CURRENT_VERSION
    assert list(envelope.episodes) == [
        ReplayPushEntryKey(AGENT_NAME, 0),
        ReplayPushEntryKey(AGENT_NAME, 1),
        ReplayPushEntryKey(AGENT_NAME, 2),
    ]

    wire = pickle.loads(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))
    replay.push(wire, notify_learning=False)
    restored = replay.ordered_buffer(AGENT_NAME)
    assert len(restored) == 3
    for original, item in zip(transitions, restored):
        assert tuple(item.agent_ids) == tuple(original.agent_ids)
        assert item.group_index == original.group_index


def test_replay_push_mid_entry_failure_rolls_back_whole_batch():
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 8}, batch_size=1))
    good = [_ma_group_episode(GROUPS[0], episode_num=9), _ma_group_episode(GROUPS[1], episode_num=9)]
    for transition in good:
        transition.prepare_for_replay()
    encoded = encode_replay_push_payload({AGENT_NAME: good})
    valid_entry = encoded.episodes[ReplayPushEntryKey(AGENT_NAME, 0)]

    poison_prepared = valid_entry.prepared_episode
    poison_prepared.inline_rows = poison_prepared.inline_rows[:-1]
    from sample.replay.replay_payload import ReplayTransitionEnvelope
    poison = ReplayTransitionEnvelope(
        transition_type=valid_entry.transition_type,
        episode_num=10,
        starts_with_episode=True,
        has_starts_with_episode=True,
        prepared_episode=poison_prepared,
    )
    envelope = ReplayPushEnvelope(
        version=ReplayPushEnvelope.CURRENT_VERSION,
        episodes={
            ReplayPushEntryKey(AGENT_NAME, 0): valid_entry,
            ReplayPushEntryKey(AGENT_NAME, 1): poison,
        },
    )
    with pytest.raises(ValueError, match="inline_rows length"):
        replay.push(envelope, notify_learning=False)

    assert replay.agent_buffer_size(AGENT_NAME) == 0
    assert replay.tensor_pool.memory_bytes == 0
    assert not replay.step_meta.get(AGENT_NAME)
    assert replay.write_count.get(AGENT_NAME, 0) == 0


def test_replay_push_capacity_keeps_only_newest_trajectories():
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 2}, batch_size=1))
    batch_one = [_ma_group_episode(GROUPS[i], episode_num=1) for i in range(3)]
    batch_two = [_ma_group_episode(GROUPS[i], episode_num=2, steps=2) for i in range(3)]

    replay.push({AGENT_NAME: batch_one}, notify_learning=False)
    replay.push({AGENT_NAME: batch_two}, notify_learning=False)

    metas = replay.buffer[AGENT_NAME]
    assert len(metas) == 2
    # 只保留最新的轨迹（episode 2 的后两条）
    assert [meta.episode_num for meta in metas] == [2, 2]
    assert replay.write_count[AGENT_NAME] == 6


def test_replay_push_chunk_splits_and_reassembles_multi_trajectories():
    from rpc.torch_rpc.replay_push_chunk import (
        assemble_replay_push_envelope,
        split_replay_push_envelope,
    )

    long_episode = _ma_group_episode(GROUPS[0], episode_num=21, steps=24)
    short_episode = _ma_group_episode(GROUPS[1], episode_num=21, steps=5)
    for transition in (long_episode, short_episode):
        transition.prepare_for_replay()
    envelope = encode_replay_push_payload({AGENT_NAME: [long_episode, short_episode]})

    chunks = split_replay_push_envelope(envelope, chunk_bytes=1)
    assert len(chunks) >= 2
    assembled = assemble_replay_push_envelope(chunks)
    assert assembled.version == envelope.version
    assert list(assembled.episodes) == list(envelope.episodes)

    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 8}, batch_size=1))
    replay.push(assembled, notify_learning=False)
    restored = replay.ordered_buffer(AGENT_NAME)
    assert [item.seq_len for item in restored] == [24, 5]
    assert [tuple(item.agent_ids) for item in restored] == [GROUPS[0], GROUPS[1]]


# ---------------------------------------------------------------------------
# to_batch 分组混采
# ---------------------------------------------------------------------------

def test_to_batch_mixes_groups_with_own_member_slots():
    episodes = [_ma_group_episode(GROUPS[i], episode_num=i) for i in range(2)]
    batch = EpisodeTransition.to_batch(
        agent_ids=list(FLAT_IDS),
        episode_trans=episodes,
        expected_n_agents=4,
        action_dim=2,
    )

    assert tuple(batch["obs"].shape) == (2, 4, 4, 4)
    assert tuple(batch["state"].shape) == (2, 4, 4)
    # 每条轨迹按自己的成员映射槽位：batch 行0是组0，行1是组1
    np.testing.assert_allclose(
        batch["obs"][0, 0, 0].cpu().numpy(), np.full(4, 0.0, dtype=np.float32)
    )
    np.testing.assert_allclose(
        batch["obs"][1, 0, 0].cpu().numpy(), np.full(4, 0.0, dtype=np.float32)
    )
    # 帧掩码来自记录的当前帧 avail_actions，首帧不再默认全1
    np.testing.assert_allclose(batch["avail_actions"][0, 0, 0].cpu().numpy(), np.ones(2))
    # terminated 由组语义决定（最后一帧截断且无存活槽位）
    assert batch["terminated"][0, 2, 0].item() == 1.0


def test_to_batch_rejects_width_mismatch():
    episodes = [_ma_group_episode(GROUPS[i], episode_num=i) for i in range(2)]
    with pytest.raises(ValueError, match="does not match"):
        EpisodeTransition.to_batch(
            agent_ids=list(FLAT_IDS),
            episode_trans=episodes,
            expected_n_agents=3,
            action_dim=2,
        )

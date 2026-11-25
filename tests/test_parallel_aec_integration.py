from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pettingzoo.test import api_test, parallel_api_test

from common.exception import AgentComposeException, AgentOwnerNotFoundException
from scheduler.artifact import Artifact, ArtifactType
from envs.action.base import ActionTable
from envs.action.model import Action, EpisodeActionDefined, StepActionDefined
from envs.base import AECEnvironment, ParallelAgentOrderContext, ParallelEnvironment
from envs.sense.aec_examples import (
    MultiAgentCounterAECEnvironment,
    SingleAgentCounterAECEnvironment,
)
from envs.sense.paralle_examples import (
    CounterParallelEnvironment,
    MultiAgentCounterParallelEnvironment,
    SingleAgentCounterParallelEnvironment,
)
from scheduler.model import AgentCompose
from scheduler.rollout_sample import RolloutSampleWorker
from sample.sample_transfer import EpisodeTransition
from scheduler.model import ActionParam, EpisodeInfo


class _StubAgent:
    def __init__(self, name, agent_ids):
        self.name = name
        self.agent_list = list(agent_ids)
        self.agent_type = "rnn"
        self.device = torch.device("cpu")
        self.config = SimpleNamespace(agent_output_type="q")
        self.dim_info = SimpleNamespace(hidden_dim=1)
        self.action_table = ActionTable([
            StepActionDefined(
                action_key="counter",
                parameters=[],
                add_target=False,
            )
        ])

    def get_init_hidden_state(self):
        return {agent_id: torch.zeros(1, 1) for agent_id in self.agent_list}

    def get_actions(self, agent_list, hidden_in, **kwargs):
        action = self.action_table.action_map[0]
        return (
            {agent_id: action for agent_id in agent_list},
            {agent_id: hidden_in[agent_id] for agent_id in agent_list},
        )


class _ReplayBuffer:
    def __init__(self):
        self.items = []

    def push(self, item, **kwargs):
        self.items.append(item)


@pytest.mark.parametrize("mode", ["parallel", "aec"])
def test_rollout_gif_producer_uses_artifact_contract(tmp_path, mode):
    env = (
        MultiAgentCounterParallelEnvironment(max_steps=1)
        if mode == "parallel" else MultiAgentCounterAECEnvironment(max_steps=1)
    )
    agent = _StubAgent("policy", env.possible_agents)
    worker = RolloutSampleWorker(
        env=env,
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=_ReplayBuffer(),
        agent_compose=AgentCompose(step_agent=[agent]),
        policy_agents=[agent],
    )
    path = tmp_path / f"{mode}.gif"
    path.write_bytes(b"gif")
    recorder = SimpleNamespace(capture=lambda *args: None, finish=lambda name: path)
    gif_calls: list[tuple] = []
    worker.render_gif_put_caller = (
        lambda task, stage, gif_path, metadata: gif_calls.append((task, stage, gif_path, metadata))
    )
    try:
        rollout = worker.rolling_parallel_episode if mode == "parallel" else worker.rolling_aec_episode
        info = _episode_info()
        rollout(info, step_limit=1, recorder=recorder)
        assert worker._pending_episode_artifacts == [Artifact(ArtifactType.RENDER_GIF, str(path))]
        assert gif_calls == [(info.task_id, info.stage_name, str(path), {"role_id": worker.sample_role_name})]
        artifacts = worker._consume_episode_artifacts()
        assert artifacts[0].artifact_type is ArtifactType.RENDER_GIF
        assert isinstance(artifacts[0].path, str)
        assert worker._pending_episode_artifacts == []
        assert worker._consume_episode_artifacts() == []
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["parallel", "aec"])
def test_rollout_gif_put_caller_failure_does_not_break_sampling(tmp_path, mode):
    env = (
        MultiAgentCounterParallelEnvironment(max_steps=1)
        if mode == "parallel" else MultiAgentCounterAECEnvironment(max_steps=1)
    )
    agent = _StubAgent("policy", env.possible_agents)
    worker = RolloutSampleWorker(
        env=env,
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=_ReplayBuffer(),
        agent_compose=AgentCompose(step_agent=[agent]),
        policy_agents=[agent],
    )
    path = tmp_path / f"{mode}.gif"
    path.write_bytes(b"gif")
    recorder = SimpleNamespace(capture=lambda *args: None, finish=lambda name: path)

    def broken_caller(*args, **kwargs):
        raise RuntimeError("rpc down")

    worker.render_gif_put_caller = broken_caller
    try:
        rollout = worker.rolling_parallel_episode if mode == "parallel" else worker.rolling_aec_episode
        transitions = rollout(_episode_info(), step_limit=1, recorder=recorder)
        assert transitions, "GIF 输送失败不阻断采样"
        assert worker._pending_episode_artifacts == [Artifact(ArtifactType.RENDER_GIF, str(path))]
    finally:
        env.close()


def test_qmix_batch_treats_partial_death_as_active_team_transition():
    agent_ids = ["drone_0", "drone_1"]
    episode = EpisodeTransition()
    action = SimpleNamespace(action_idx=0)
    obs = {agent_id: np.asarray([1.0], dtype=np.float32) for agent_id in agent_ids}
    state = np.asarray([0.0], dtype=np.float32)

    episode.add_step(
        obs=obs,
        action={agent_id: action for agent_id in agent_ids},
        last_action={agent_id: None for agent_id in agent_ids},
        reward={agent_id: -0.5 for agent_id in agent_ids},
        next_obs=obs,
        termination={"drone_0": True, "drone_1": False},
        truncation={agent_id: False for agent_id in agent_ids},
        state=state,
        next_state=state,
        avail_actions_next={"drone_0": [0], "drone_1": [1]},
        active_agent_ids=set(agent_ids),
        next_active_agent_ids={"drone_1"},
    )
    episode.add_step(
        obs=obs,
        action={"drone_0": None, "drone_1": action},
        last_action={agent_id: action for agent_id in agent_ids},
        reward={agent_id: -0.5 for agent_id in agent_ids},
        next_obs=obs,
        termination={agent_id: False for agent_id in agent_ids},
        truncation={agent_id: True for agent_id in agent_ids},
        state=state,
        next_state=state,
        avail_actions_next={agent_id: [0] for agent_id in agent_ids},
        active_agent_ids={"drone_1"},
        next_active_agent_ids=set(),
    )

    batch = EpisodeTransition.to_batch(agent_ids, [episode], action_dim=1)

    assert batch["terminated"][0, :2, 0].tolist() == [0.0, 1.0]
    assert batch["active_agents"][0, :, :, 0].tolist() == [
        [1.0, 1.0],
        [0.0, 1.0],
        [0.0, 0.0],
    ]


def _episode_info(agent_name="policy"):
    return EpisodeInfo(
        episode_num=1,
        task_id="aec-test",
        epsilon=0.0,
        deterministic=True,
        temperature=1.0,
        actions_param={
            agent_name: ActionParam(epsilon=0.0, deterministic=True, temperature=1.0)
        },
        seed=7,
        start_datetime=None,
        end_datetime=None,
        metadata={},
    )


def test_aec_examples_pass_pettingzoo_api():
    api_test(SingleAgentCounterAECEnvironment(max_steps=3), num_cycles=10, verbose_progress=False)
    api_test(MultiAgentCounterAECEnvironment(max_steps=4), num_cycles=20, verbose_progress=False)


def test_direct_environment_bases_expose_required_business_contract():
    required_hooks = {
        "done",
        "state",
        "get_obs",
        "get_avail_agent_actions",
        "get_reward",
    }
    assert required_hooks <= ParallelEnvironment.__dict__.keys()
    assert required_hooks <= AECEnvironment.__dict__.keys()
    assert required_hooks == set(ParallelEnvironment.__abstractmethods__)
    assert required_hooks == set(AECEnvironment.__abstractmethods__)


def test_parallel_examples_pass_pettingzoo_api():
    parallel_api_test(SingleAgentCounterParallelEnvironment(max_steps=3), num_cycles=10)
    parallel_api_test(MultiAgentCounterParallelEnvironment(max_steps=4), num_cycles=20)


@pytest.mark.parametrize(
    "env",
    [
        SingleAgentCounterParallelEnvironment(max_steps=2),
        MultiAgentCounterParallelEnvironment(max_steps=2),
    ],
)
def test_parallel_examples_apply_joint_actions_in_one_physical_step(env):
    obs, infos = env.reset(options={"step_limit": 2})
    assert set(obs) == set(infos) == set(env.possible_agents)
    actions = {
        agent_id: index % 2
        for index, agent_id in enumerate(env.possible_agents)
    }
    next_obs, rewards, terminations, truncations, _ = env.step(actions)
    state = env.state()
    expected = np.asarray(
        [-1.0 if action == 0 else 1.0 for action in actions.values()],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(state, expected)
    assert set(next_obs) == set(env.possible_agents)
    assert rewards == dict(zip(env.possible_agents, expected.tolist()))
    assert not any(terminations.values())
    assert not any(truncations.values())
    assert env.post_state.sim_time == 1.0
    assert env.step_num == 2


def test_agent_order_rule_expands_one_action_per_enum():
    table = ActionTable([
        EpisodeActionDefined(
            action_key="select_order",
            parameters=[],
            agent_order_rules=["env_native", "reverse_round_robin", "first_live"],
        )
    ])
    assert table.n_actions == 3
    assert [table.action_map[i].agent_order_rule for i in range(3)] == [
        "env_native",
        "reverse_round_robin",
        "first_live",
    ]


def _episode_order_action(agent_order_rule, **metadata):
    return Action(
        action_key="select_order",
        parameters={},
        target=-2,
        action_idx=0,
        agent_order_rule=agent_order_rule,
        metadata=metadata,
    )


def test_agent_compose_defaults_to_transparent_environment_order():
    agent = _StubAgent("policy", ["agent_0"])
    compose = AgentCompose(step_agent=[agent])
    assert compose.compose_type == "order"
    assert compose.episode_agent is None
    with pytest.raises(AgentComposeException, match="environment base class"):
        AgentCompose(compose_type="parallel", step_agent=[agent])


def test_parallel_rollout_passes_complete_episode_action_and_selection_history():
    env = CounterParallelEnvironment(
        ["agent_0", "agent_1"],
        max_steps=2,
        policy_agent_names=["first", "second"],
    )
    resolved_groups = []
    resolved_histories = []
    received_actions = []
    order_resolve = env.episode_agent_order_resolve

    def recording_resolver(episode_agent_action, context):
        received_actions.append(episode_agent_action)
        resolved_histories.append(context.resolved_agent_names)
        selected = order_resolve(episode_agent_action, context)
        resolved_groups.append(selected)
        return selected

    env.episode_agent_order_resolve = recording_resolver
    first = _StubAgent("first", ["agent_0"])
    second = _StubAgent("second", ["agent_1"])
    episode_agent = _StubAgent("episode", ["commander"])
    episode_action = _episode_order_action("reverse_policy_groups")
    replay = _ReplayBuffer()
    worker = RolloutSampleWorker(
        env=env,
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=replay,
        agent_compose=AgentCompose(
            compose_type="decision",
            episode_agent=episode_agent,
            step_agent=[first, second],
        ),
        policy_agents=[episode_agent, first, second],
    )
    transitions = worker.rolling_parallel_episode(
        _episode_info("first"),
        random_sample=True,
        step_limit=1,
        episode_agent_action=episode_action,
    )
    assert resolved_groups == ["second", "first"]
    assert resolved_histories == [(), ("second",)]
    assert received_actions == [episode_action, episode_action]
    assert env.step_num == 2
    assert set(env.current_actions) == set(env.possible_agents)
    assert [t.seq_len for t in transitions["first"]] == [1]
    assert [t.seq_len for t in transitions["second"]] == [1]
    assert len(replay.items) == 1


def test_parallel_rollout_without_episode_agent_does_not_call_order_resolver():
    env = CounterParallelEnvironment(["agent_0", "agent_1"], max_steps=2)
    env.episode_agent_order_resolve = lambda *args: pytest.fail(
        "transparent order must not call episode_agent_order_resolve"
    )
    first = _StubAgent("first", ["agent_0"])
    second = _StubAgent("second", ["agent_1"])
    worker = RolloutSampleWorker(
        env=env,
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=_ReplayBuffer(),
        agent_compose=AgentCompose(step_agent=[first, second]),
        policy_agents=[first, second],
    )

    transitions = worker.rolling_parallel_episode(
        _episode_info("first"),
        random_sample=True,
        step_limit=1,
    )

    assert [t.seq_len for t in transitions["first"]] == [1]
    assert [t.seq_len for t in transitions["second"]] == [1]


def test_aec_rollout_uses_episode_action_for_atomic_agent_order():
    env = MultiAgentCounterAECEnvironment(max_steps=3)
    resolved_histories = []
    order_resolve = env.episode_agent_order_resolve

    def recording_order_resolve(episode_agent_action, context):
        resolved_histories.append(context.resolved_agent_ids)
        return order_resolve(episode_agent_action, context)

    env.episode_agent_order_resolve = recording_order_resolve
    agent = _StubAgent("policy", env.possible_agents)
    episode_agent = _StubAgent("episode", ["commander"])
    episode_action = _episode_order_action("reverse_round_robin")
    replay = _ReplayBuffer()
    worker = RolloutSampleWorker(
        env=env,
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=replay,
        agent_compose=AgentCompose(
            compose_type="decision",
            episode_agent=episode_agent,
            step_agent=[agent],
        ),
        policy_agents=[episode_agent, agent],
    )
    transitions = worker.rolling_aec_episode(
        _episode_info(),
        step_limit=3,
        episode_agent_action=episode_action,
    )
    # AEC 统一列表返回；隐式单组包装为长度1列表。
    assert len(transitions["policy"]) == 1
    episode = transitions["policy"][0]
    assert episode.seq_len == 3
    assert [sum(episode.active_agent_mask[agent_id][t] for agent_id in env.possible_agents) for t in range(3)] == [1, 1, 1]
    assert [
        next(agent_id for agent_id in env.possible_agents if episode.active_agent_mask[agent_id][t])
        for t in range(3)
    ] == ["agent_2", "agent_1", "agent_0"]
    assert resolved_histories == [(), ("agent_2",), ("agent_2", "agent_1")]
    assert [metadata["next_active_agent_id"] for metadata in episode.metadata[:2]] == [
        "agent_1",
        "agent_0",
    ]
    batch = EpisodeTransition.to_batch(env.possible_agents, [episode], action_dim=1)
    assert tuple(batch["active_agents"].shape) == (1, 4, 3, 1)
    assert batch["active_agents"][:, :3].sum().item() == 3
    # 时间截断保留下一游标槽位，供算法按time-limit语义bootstrap。
    assert batch["active_agents"][0, 3, env.possible_agents.index("agent_0"), 0].item() == 1
    assert len(replay.items) == 1


def test_aec_rollout_without_episode_agent_uses_native_agent_selection_directly():
    env = MultiAgentCounterAECEnvironment(max_steps=3)
    agent_iter_calls = []
    original_agent_iter = env.agent_iter

    def recording_agent_iter(*args, **kwargs):
        agent_iter_calls.append((args, kwargs))
        return original_agent_iter(*args, **kwargs)

    env.agent_iter = recording_agent_iter
    env.episode_agent_order_resolve = lambda *args: pytest.fail(
        "native AEC order must not call episode_agent_order_resolve"
    )
    agent = _StubAgent("policy", env.possible_agents)
    worker = RolloutSampleWorker(
        env=env,
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=_ReplayBuffer(),
        agent_compose=AgentCompose(step_agent=[agent]),
        policy_agents=[agent],
    )

    episode = worker.rolling_aec_episode(
        _episode_info(),
        step_limit=3,
    )["policy"][0]

    assert [
        next(agent_id for agent_id in env.possible_agents if episode.active_agent_mask[agent_id][t])
        for t in range(3)
    ] == ["agent_0", "agent_1", "agent_2"]
    assert agent_iter_calls == [((), {})]


def test_aec_rollout_uses_get_obs_as_the_only_observation_snapshot_source():
    env = MultiAgentCounterAECEnvironment(max_steps=1)
    original_get_obs = env.get_obs
    get_obs_calls = []

    def recording_get_obs():
        get_obs_calls.append(env.step_num)
        return original_get_obs()

    env.get_obs = recording_get_obs
    env.observe = lambda agent_id: pytest.fail(
        "rolling_aec_episode must call last(observe=False), not last() observation"
    )
    agent = _StubAgent("policy", env.possible_agents)
    worker = RolloutSampleWorker(
        env=env,
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=_ReplayBuffer(),
        agent_compose=AgentCompose(step_agent=[agent]),
        policy_agents=[agent],
    )

    worker.rolling_aec_episode(
        _episode_info(),
        step_limit=1,
    )

    # 每个原子步只获取一次O_k和一次步进后的O_{k+1}。
    assert len(get_obs_calls) == 2


def test_aec_rollout_reports_unowned_resolver_output_at_policy_routing_boundary():
    env = MultiAgentCounterAECEnvironment(max_steps=1)
    env.episode_agent_order_resolve = lambda episode_agent_action, context: "unowned_agent"
    agent = _StubAgent("policy", env.possible_agents)
    episode_agent = _StubAgent("episode", ["commander"])
    worker = RolloutSampleWorker(
        env=env,
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=_ReplayBuffer(),
        agent_compose=AgentCompose(
            compose_type="decision",
            episode_agent=episode_agent,
            step_agent=[agent],
        ),
        policy_agents=[episode_agent, agent],
    )

    with pytest.raises(AgentOwnerNotFoundException, match="unowned_agent"):
        worker.rolling_aec_episode(
            _episode_info(),
            step_limit=1,
            episode_agent_action=_episode_order_action("free_rule"),
        )


def test_rollout_init_does_not_require_step_agents_to_own_every_possible_agent_slot():
    env = MultiAgentCounterAECEnvironment(max_steps=1)
    step_agent = _StubAgent("policy", ["agent_0"])

    worker = RolloutSampleWorker(
        env=env,
        task_manager=SimpleNamespace(accept_sample_task_result=lambda *args, **kwargs: True),
        replay_buffer=_ReplayBuffer(),
        agent_compose=AgentCompose(step_agent=[step_agent]),
        policy_agents=[step_agent],
    )

    assert worker.step_agent == [step_agent]


@pytest.mark.parametrize("mode", ["parallel", "aec"])
def test_rollout_reports_steps_without_business_metric_configuration(mode):
    from contextlib import contextmanager
    env = (SingleAgentCounterParallelEnvironment(max_steps=3) if mode == "parallel"
           else SingleAgentCounterAECEnvironment(max_steps=3))
    agent = _StubAgent("policy", env.possible_agents)
    worker = RolloutSampleWorker(
        env=env, task_manager=SimpleNamespace(accept_sample_task_result=lambda *a, **k: True),
        replay_buffer=_ReplayBuffer(), agent_compose=AgentCompose(step_agent=[agent]),
        policy_agents=[agent],
    )
    @contextmanager
    def episode(**kwargs):
        yield _episode_info()
    worker.episode_task_context = episode
    try:
        for _ in range(2):
            worker.rollout_one_episode(step_limit=3)
            assert worker.episode_metrics["steps"] == 3
    finally:
        env.close()

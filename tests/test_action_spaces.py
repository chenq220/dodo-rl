from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from algos.agents import Agent, record_action_warp
from algos.modules.agents.random_agent import RandomAgent
from algos.modules.agents.rnn_agent import RNNAgent
from algos.learners import Learner
from config import AgentConfig, LearnerConfig
from envs.action.base import ActionTable
from envs.action.model import Action, ActionParameterDefined, CompositeAction, StepActionDefined
from envs.action.spaces import (
    ACTION_SPACE_KINDS, ActionSpace, ContinuousAction, ContinuousActionSpace,
    ContinuousParameter, DiscreteActionSpace, MultiContinuousActionSpace,
    MultiDiscreteActionSpace, MultiHeadAction, resolve_action_space,
)
from envs.sense.action_space_examples import ActionSpaceCounterAECEnvironment, ActionSpaceCounterParallelEnvironment
from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition
from scheduler.model import AgentCompose, ActionParam, EpisodeInfo
from scheduler.rollout_sample import RolloutSampleWorker
from utils.module_utils import inject_env_dims


def discrete(count=3):
    return DiscreteActionSpace(ActionTable([StepActionDefined("move", [
        ActionParameterDefined("delta", values=list(range(count)))], add_target=False)]))


def continuous(dim=3):
    return ContinuousActionSpace("aim", [ContinuousParameter(f"v{i}", -10, 20, "m") for i in range(dim)])


def spaces():
    return [discrete(), MultiDiscreteActionSpace([("move", discrete()), ("mode", discrete(2))]),
            continuous(), MultiContinuousActionSpace([("move", continuous(2)), ("aim", continuous())])]


@pytest.mark.parametrize("space", spaces(), ids=lambda space: space.kind)
def test_codec_json_sampling_layout_and_previous_action(space):
    restored = ActionSpace.from_dict(json.loads(json.dumps(space.to_dict())))
    assert restored.to_dict() == space.to_dict()
    assert restored.signature == space.signature
    np.testing.assert_array_equal(space.encode_previous_action(None), np.zeros(space.layout["previous_action_dim"]))
    for seed in range(5):
        action = space.sample(np.random.default_rng(seed))
        space.validate(action)
        value = space.encode(action)
        assert space.to_gym_space().contains(value)
        again = restored.decode(value)
        np.testing.assert_allclose(restored.encode(again), value, atol=1e-7)
        record = json.loads(json.dumps(space.record(action)))
        assert ActionSpace.restore(record, space).to_dict() == action.to_dict()
        assert space.encode_previous_action(action).shape == (space.layout["previous_action_dim"],)
    if space.kind in {"continuous", "multi_continuous"}:
        assert space.encode(action).dtype == np.float32


@pytest.mark.parametrize("value", [[0, 0], [True, 0, 0], [True, False, True], ["0", "0", "0"], [0, np.nan, 0],
                                    [0, np.inf, 0], [0, 1.00001, 0], [[0, 0, 0]]])
def test_continuous_rejects_invalid_policy_vectors(value):
    with pytest.raises(ValueError):
        continuous().decode(value)


@pytest.mark.parametrize("value", [True, "2", np.inf, np.nan, 21, -11])
def test_continuous_rejects_bad_physical_values(value):
    with pytest.raises(ValueError):
        continuous(1).encode(ContinuousAction("aim", {"v0": value}))


def test_continuous_bounds_and_explicit_physical_conversion():
    space = continuous()
    assert space.layout["head_count"] == 1  # xyz is one vector head
    for normalized, physical in [(-1, -10), (1, 20), (0, 5)]:
        action = space.decode([normalized] * 3)
        assert list(action.parameters.values()) == [physical] * 3
        np.testing.assert_array_equal(space.encode(action), [normalized] * 3)
    with pytest.raises(ValueError):
        space.sample(mask=[1, 1, 1])
    with pytest.raises(ValueError):
        space.encode(ContinuousAction("other", {"v0": 0}))
    for low, high in [(1, 1), (2, 1), (np.inf, 5), (True, 5)]:
        with pytest.raises(ValueError):
            ContinuousParameter("x", low, high)
    with pytest.raises(ValueError):
        ContinuousActionSpace("aim", [ContinuousParameter("x", 0, 1)] * 2)


def test_discrete_masks_indices_and_business_validation():
    space = discrete()
    assert space.sample(mask=[0, 1, 0]).action_idx == 1
    for mask in [[0, 0, 0], [1, 0], [1, 0, 2], [np.nan, 0, 1]]:
        with pytest.raises(ValueError):
            space.sample(mask=mask)
    for value in [True, 1.0, "1", -1, 3]:
        with pytest.raises(ValueError):
            space.decode(value)
    action = space.decode(1)
    action.parameters["delta"] = 100
    with pytest.raises(ValueError):
        space.validate(action)
    assert space.decode(1).parameters["delta"] == 1  # no shared mutable entries


def test_multi_heads_preserve_order_without_cartesian_enumeration():
    multi = MultiDiscreteActionSpace([("b", discrete(100)), ("a", discrete(200))])
    assert multi.layout["category_counts"] == [100, 200]
    assert multi.layout["previous_action_dim"] == 300
    assert multi.to_gym_space().nvec.tolist() == [100, 200]
    action = multi.sample(mask={"a": [0] * 199 + [1], "b": [1] + [0] * 99})
    np.testing.assert_array_equal(multi.encode(action), [0, 199])
    assert list(action.heads) == ["b", "a"]
    with pytest.raises(ValueError):
        multi.sample(mask={"b": [1] * 100})
    with pytest.raises(ValueError):
        multi.sample(mask={"a": [0] * 200, "b": [1] * 100})
    with pytest.raises(ValueError):
        multi.decode([0.0, 1.0])
    with pytest.raises(ValueError):
        MultiDiscreteActionSpace([("x", discrete()), ("x", discrete())])
    with pytest.raises(ValueError):
        MultiDiscreteActionSpace([("x", discrete()), ("y", continuous())])
    continuous_multi = spaces()[3]
    assert [head["slice"] for head in continuous_multi.layout["heads"]] == [[0, 2], [2, 5]]
    with pytest.raises(ValueError):
        continuous_multi.sample(mask={})
    raw = multi.to_dict()
    raw["heads"].reverse()
    with pytest.raises(ValueError, match="signature"):
        ActionSpace.restore(multi.record(action), ActionSpace.from_dict(raw))


def test_records_reject_version_layout_and_value_tampering():
    space = continuous(1)
    action = ContinuousAction("aim", {"v0": 3.123456789}, {"note": "preserve"})
    record = space.record(action)
    assert ActionSpace.restore(json.loads(json.dumps(record))) == action
    for field, bad in [("format_version", 2), ("value", [0.9]), ("space_signature", "bad")]:
        raw = deepcopy(record)
        raw[field] = bad
        with pytest.raises(ValueError):
            ActionSpace.restore(raw)


def test_configuration_legacy_adapter_free_table_and_composite():
    config = AgentConfig(define_action=[StepActionDefined("move", [], False)])
    legacy = resolve_action_space(config)
    assert legacy.encode(legacy.decode(0)) == 0
    assert legacy.sample(mask=[1, 1]).action_idx == 0
    table = ActionTable([StepActionDefined("x", [], False), StepActionDefined("y", [], False)],
                        multi_action_mode="combination")
    adapter = DiscreteActionSpace(table)
    assert isinstance(adapter.decode(0), CompositeAction)
    assert ActionSpace.restore(adapter.record(adapter.decode(0))).to_dict() == table.action_map[0].to_dict()
    free = resolve_action_space(AgentConfig(multi_action_mode="free"))
    free.table.action_map[0] = Action("new", {}, -2, 0)
    assert free.to_gym_space().n == 1
    config.action_space_spec = continuous().to_dict()
    with pytest.raises(ValueError, match="conflict"):
        resolve_action_space(config)
    config.define_action = []
    assert resolve_action_space(AgentConfig.from_json(config.to_json())).kind == "continuous"


class ZeroRuleAgent(RandomAgent):
    @record_action_warp
    def get_actions(self, obs, agent_list, hidden_in, action_mask, **kwargs):
        space = self.action_space
        value = 0 if space.kind == "discrete" else np.zeros(
            space.layout["value_dim"], dtype=np.int64 if space.kind == "multi_discrete" else np.float32)
        return {key: space.decode(value) for key in agent_list}, {key: None for key in agent_list}


def make_agent(space, cls=RandomAgent):
    config = AgentConfig(name="policy", agent_type="monte_carlo", agent_names=["a", "b"],
                         obs_dim=2, action_space_spec=space.to_dict())
    return cls(agent_type="monte_carlo", config=config, agent_net_source="blanco", device="cpu")


@pytest.mark.parametrize("agent_type", [
    "rnn", "rnn_ns", "policy_rnn", "mappo_rnn", "q_rnn", "maddpg_mlp", "monte_carlo", "sb3",
])
def test_builtin_agents_accept_common_action_call(agent_type):
    from common.creator import create_agent
    from common.plugins import activate_plugins
    from portal.plugins.registry import project_plugin_registry

    activate_plugins(default_plugins=project_plugin_registry())
    space = spaces()[3] if agent_type == "maddpg_mlp" else discrete()
    config = AgentConfig(name="policy", agent_type=agent_type, agent_names=["a"],
                         obs_dim=2, action_space_spec=space.to_dict())
    agent = create_agent(agent_type, config, "blanco", config.agent_names, "cpu")
    actions, hidden = agent.get_actions(
        obs={"a": np.zeros(2)}, agent_list=["a"], hidden_in=agent.get_init_hidden_state(),
    )
    assert set(actions) == {"a"}
    assert isinstance(hidden, dict)
    space.validate(actions["a"])


@pytest.mark.parametrize("env_cls", [ActionSpaceCounterParallelEnvironment, ActionSpaceCounterAECEnvironment])
@pytest.mark.parametrize("random_sample", [False, True])
@pytest.mark.parametrize("policy_cls", [RandomAgent, ZeroRuleAgent])
@pytest.mark.parametrize("space", spaces(), ids=lambda space: space.kind)
def test_random_policy_worker_environment_record_and_transition(env_cls, random_sample, policy_cls, space):
    agent = make_agent(space, policy_cls)
    env = env_cls({"a": space, "b": space})
    inject_env_dims(env, [agent])
    records = []
    env.record_caller = records.append
    replay = SimpleNamespace(push=lambda *a, **kw: None)
    worker = RolloutSampleWorker(env, SimpleNamespace(accept_sample_task_result=lambda *a, **kw: True),
                                 replay, AgentCompose(step_agent=[agent]), [agent])
    random_actions, _ = worker.get_random_actions({key: env.get_avail_agent_actions(key) for key in ["a", "b"]})
    for action in random_actions.values():
        space.validate(action)
    info = EpisodeInfo(episode_num=1, task_id="actions", epsilon=0, deterministic=True, temperature=1,
                       actions_param={"policy": ActionParam(epsilon=0, deterministic=True, temperature=1)},
                       seed=3, start_datetime=None, end_datetime=None, metadata={})
    run = worker.rolling_aec_episode if env_cls is ActionSpaceCounterAECEnvironment else worker.rolling_parallel_episode
    transitions = run(info, random_sample=random_sample, step_limit=2)
    assert records and transitions["policy"]
    for action in env.current_actions.values():
        space.validate(action)
    episode = transitions["policy"][0]
    assert episode.seq_len > 0
    for action in episode.actions["a"]:
        if space.kind == "discrete":
            assert isinstance(action, int)
        elif action is not None:
            space.validate(ActionSpace.restore(json.loads(json.dumps(action)), space))
    for record in records:
        for value in record["actions"].values():
            if space.kind != "discrete":
                space.validate(ActionSpace.restore(json.loads(json.dumps(value)), space))
    assert agent.dim_info.value == 2 + space.layout["previous_action_dim"]
    env.close()


def test_rule_plugin_capability_and_rejections():
    class RuleAgent(RandomAgent):
        def get_actions(self, obs, agent_list, hidden_in, action_mask, **kwargs):
            action = self.action_space.decode([0, 0, 0])
            return {key: action for key in agent_list}, {key: None for key in agent_list}
    rule = make_agent(continuous(), RuleAgent)
    actions, _ = rule.get_actions({}, ["a"], {}, None)
    assert actions["a"].parameters["v0"] == 5
    for cls in [RNNAgent]:
        with pytest.raises(ValueError, match="supported=.*discrete"):
            make_agent(continuous(), cls)
    # Execute Learner's actual early guard before any network or config coupling.
    with pytest.raises(ValueError, match="Learner qmix.*continuous.*supported"):
        ConcreteLearner = type("ConcreteLearner", (Learner,),
                               {method: lambda *a, **k: None for method in Learner.__abstractmethods__})
        Learner.__init__(object.__new__(ConcreteLearner), "qmix", LearnerConfig(name="test"), rule,
                         None, "cpu", "blanco")


def test_sa_recording_and_legacy_tensor_adapter_guard():
    space = continuous()
    transition = SAEpisodeTransition(action_space=space)
    transition.add_step(np.zeros(1), space.decode([0, 0, 0]), 0, np.ones(1), False, False)
    assert ActionSpace.restore(transition.actions[0]).parameters["v0"] == 5
    assert transition.last_actions[0] is None
    with pytest.raises(ValueError, match="only discrete"):
        SAEpisodeTransition.to_batch_tensor([transition], "cpu")
    multi = EpisodeTransition(action_spaces={"a": space})
    with pytest.raises(ValueError, match="only discrete"):
        EpisodeTransition._infer_action_dim(["a"], [multi])


def test_toml_spec_and_public_sdk(tmp_path, monkeypatch):
    import toml
    from pathlib import Path
    from utils.toml_config import TomlTrainingConfig
    path = tmp_path / "actions.toml"
    spec = spaces()[3].to_dict()
    path.write_text(toml.dumps({"agents": {"0": {"name": "policy", "agent_type": "monte_carlo",
                    "agent_names": ["a"], "action_space_spec": spec}}}))
    config = TomlTrainingConfig(path, project_root=tmp_path).agents_config[0]
    assert resolve_action_space(config).to_dict() == spec
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "sdk" / "src"))
    import dodorl_sdk
    from dodorl_sdk import action as sdk_actions
    for name in ["ActionSpace", "ContinuousParameter", "ContinuousAction", "MultiHeadAction",
                 "DiscreteActionSpace", "ContinuousActionSpace", "MultiDiscreteActionSpace",
                 "MultiContinuousActionSpace", "CompositeAction"]:
        assert getattr(dodorl_sdk, name) is getattr(sdk_actions, name)
        assert name in dodorl_sdk.__all__


def test_environment_restores_new_and_legacy_records():
    from envs.base import BaseEnvironment
    for space in spaces():
        action = space.sample(np.random.default_rng(4))
        encoded = space.record(action) if space.kind != "discrete" else action.to_dict()
        result = BaseEnvironment._actions_from_record({"a": json.loads(json.dumps(encoded))})
        assert result["a"].to_dict() == action.to_dict()


@pytest.mark.parametrize("space", spaces(), ids=lambda space: space.kind)
def test_generic_replay_preserves_action_records(space):
    from config import ReplayBufferConfig
    from sample.replay_buffer import ReplayBuffer
    episode = SAEpisodeTransition(action_space=space)
    action = space.sample(np.random.default_rng(5))
    episode.add_step(np.zeros(1), action, 0.25, np.ones(1), False, False)
    episode.add_step(np.ones(1), action, 0.5, np.ones(1) * 2, True, False, last_action=action)
    episode.prepare_for_replay()
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={"policy": 4}, batch_size=1))
    try:
        replay.push({"policy": [episode]}, notify_learning=False)
        recovered = replay.sample("policy", batch_size=1)[0]
        assert recovered.action_space_spec == episode.action_space_spec
        if space.kind == "discrete":
            assert recovered.actions == episode.actions
        else:
            for record in recovered.actions:
                assert ActionSpace.restore(record, space).to_dict() == action.to_dict()
    finally:
        replay.close()


@pytest.mark.parametrize("space", spaces()[1:], ids=lambda space: space.kind)
def test_builtin_agents_and_learners_reject_unsupported_spaces(space):
    from algos.modules.agents.sb3_agent import SB3Agent
    from algos.modules.agents.rnn_ns_agent import RNNNSAgent
    from algos.training.qmix_learner import QMixerLearner
    from algos.training.sb3_ppo_learner import SB3PPOLearner
    for cls in [RNNAgent, RNNNSAgent, SB3Agent]:
        with pytest.raises(ValueError, match="requested action space.*supported"):
            make_agent(space, cls)
    agent = make_agent(space)
    for agent_type, learner_cls, algo in [("rnn", QMixerLearner, "qmix"), ("sb3", SB3PPOLearner, "ppo")]:
        agent.agent_type = agent_type
        with pytest.raises(ValueError, match="requested action space.*supported"):
            learner_cls(algo_type=algo, config=LearnerConfig(name="test", algo_type=algo), agent=agent,
                        env=None, device="cpu", net_source="blanco")

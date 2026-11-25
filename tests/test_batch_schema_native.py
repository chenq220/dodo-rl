from typing import Annotated, TypedDict
from types import SimpleNamespace

import pytest
import torch

from envs.sense.action_space_examples import ActionSpaceCounterAECEnvironment, ActionSpaceCounterParallelEnvironment
from algos.learners.batch_schema import BatchSchema, DEFAULT_EPISODE_SCHEMA, ListField, TensorField, assemble_episodes
from sample.model import EpisodeBatch, SampledEpisodes
from test_action_spaces import spaces
from test_mappo_action_spaces import rollout, setup


class CustomFields(TypedDict):
    obs: Annotated[torch.Tensor, TensorField("obs_with_bootstrap", ("T+1", "B", "N", "O"), torch.float32)]
    score: Annotated[torch.Tensor, TensorField("policy_score", ("T", "B", "N"), torch.float32)]
    lengths: Annotated[list[int], ListField("seq_len", "B")]


@pytest.mark.parametrize('field', ['active_agent_mask', 'next_active_agent_mask'])
@pytest.mark.parametrize('damage', ['missing_agent', 'short_sequence'])
def test_assembly_rejects_incomplete_activity_masks(field, damage):
    agent, env, _ = setup(spaces()[0])
    try:
        episode = rollout(agent, env, 2)
        masks = getattr(episode, field)
        if damage == 'missing_agent':
            del masks['b']
        else:
            masks['b'].pop()
        with pytest.raises(ValueError, match=field):
            assemble_episodes([episode], DEFAULT_EPISODE_SCHEMA, agent=agent, device='cpu')
    finally:
        env.close()


def test_assembly_accepts_explicitly_empty_optional_fields_but_not_missing_attributes():
    agent, env, _ = setup(spaces()[0])
    try:
        episode = rollout(agent, env, 2)
        episode.active_agent_mask = {}
        episode.next_active_agent_mask = {}
        episode.action_space_specs = {}
        episode.hidden_in = {}
        batch = assemble_episodes([episode], DEFAULT_EPISODE_SCHEMA, agent=agent, device='cpu')
        assert batch.data['active'][0, :2].all()
        assert batch.data['hidden_in'] is None
        assert batch.data['burn_in'] == [None]
        del episode.burn_in
        with pytest.raises(AttributeError, match='burn_in'):
            assemble_episodes([episode], DEFAULT_EPISODE_SCHEMA, agent=agent, device='cpu')
    finally:
        env.close()


@pytest.mark.parametrize("env_class", [ActionSpaceCounterParallelEnvironment, ActionSpaceCounterAECEnvironment])
def test_custom_schema_uses_replay_fields_and_declared_layout(env_class):
    agent, env, _ = setup(spaces()[0], env_class)
    try:
        episodes = [rollout(agent, env, 2), rollout(agent, env, 3)]
        for episode in episodes:
            for time, metadata in enumerate(episode.metadata):
                metadata["batch_fields"] = {
                    "policy_score": {
                        aid: float(time + 1)
                        for aid in episode.agent_ids if episode.active_agent_mask[aid][time]
                    }
                }
        schema = BatchSchema.from_type(CustomFields)
        assert schema.sample_sources() == ["policy_score"]
        batch = assemble_episodes(episodes, schema, agent=agent, device=torch.device("cpu"),
                                  sample_strategy="uniform")
        assert isinstance(batch, EpisodeBatch)
        assert batch.sample_strategy == "uniform"
        assert batch.data["obs"].shape == (4, 2, 2, 2)
        assert batch.data["score"].shape == (3, 2, 2)
        assert batch.data["lengths"] == [2, 3]
        default = assemble_episodes(episodes, DEFAULT_EPISODE_SCHEMA, agent=agent, device=torch.device("cpu"))
        assert default.sample_strategy is None
        sampled = SampledEpisodes(episodes, sample_strategy="uniform_steps")
        forwarded = type(episodes[0])._assemble_batch(
            sampled, learner_name=agent.name, schema=schema,
            context=SimpleNamespace(agent=agent, device="cpu"),
        )
        assert forwarded.sample_strategy == "uniform_steps"
        torch.testing.assert_close(batch.data["obs"], default.data["obs"].transpose(0, 1))
        for row, episode in enumerate(episodes):
            for time in range(episode.seq_len):
                for slot, aid in enumerate(episode.agent_ids):
                    if episode.active_agent_mask[aid][time]:
                        assert batch.data["score"][time, row, slot] == time + 1
    finally:
        env.close()


def test_custom_required_source_reports_missing_step():
    agent, env, _ = setup(spaces()[0])
    try:
        episode = rollout(agent, env, 2)
        episode.metadata[0]["batch_fields"] = {"policy_score": {"a": 1.0, "b": 2.0}}
        episode.metadata[1]["batch_fields"] = {"policy_score": {"a": 3.0}}
        with pytest.raises(ValueError, match="episode=1 step=1 agent=b"):
            assemble_episodes([episode], BatchSchema.from_type(CustomFields),
                              agent=agent, device=torch.device("cpu"))
    finally:
        env.close()


class PaddedFields(TypedDict):
    obs: Annotated[torch.Tensor, TensorField('obs_with_bootstrap', ('T+1', 'B', 'N', 'O'),
                                             torch.float32, padding_value=-7)]
    score: Annotated[torch.Tensor | None, TensorField('optional_score', ('B', 'T', 'N'), torch.float32)]


def test_schema_padding_and_optional_sources():
    agent, env, _ = setup(spaces()[0])
    try:
        episodes = [rollout(agent, env, 1), rollout(agent, env, 3)]
        schema = BatchSchema.from_type(PaddedFields)
        assert schema.sample_sources() == ['optional_score']
        assert schema.sample_sources(required_only=True) == []
        fields = assemble_episodes(episodes, schema, agent=agent, device='cpu').data
        assert fields['score'] is None
        assert torch.all(fields['obs'][2:, 0] == -7)
        assert not torch.all(fields['obs'][1, 0] == -7)  # the short episode's bootstrap frame
    finally:
        env.close()


@pytest.mark.parametrize('env_class', [ActionSpaceCounterParallelEnvironment, ActionSpaceCounterAECEnvironment])
def test_custom_source_survives_rollout_and_replay_until_learner_update(env_class):
    from types import SimpleNamespace
    from algos.training.mappo_learner import MAPPOActorCriticLearner
    from config import LearnerConfig, ReplayBufferConfig
    from sample.replay_buffer import ReplayBuffer
    from scheduler.model import ActionParam, AgentCompose, EpisodeInfo
    from scheduler.rollout_sample import RolloutSampleWorker

    class CustomLearner(MAPPOActorCriticLearner):
        batch_schema = BatchSchema.from_type(CustomFields)

        def update(self, batch: EpisodeBatch[CustomFields]):
            return batch.data

    agent, env, _ = setup(spaces()[0], env_class)
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={'policy': 8}, batch_size=1))
    try:
        active_ids = []
        original = agent.get_actions

        def get_actions(obs, agent_list, *args, **kwargs):
            active_ids[:] = agent_list
            return original(obs, agent_list, *args, **kwargs)

        agent.get_actions = get_actions
        agent.consume_rollout_fields = lambda names: {
            name: {aid: float(i + 1) for i, aid in enumerate(active_ids)} for name in names
        }
        info = EpisodeInfo(
            episode_num=5, task_id='custom-schema', epsilon=0, deterministic=False, temperature=1,
            actions_param={'policy': ActionParam(epsilon=0, deterministic=False, temperature=1)},
            seed=3, start_datetime=None, end_datetime=None,
            metadata={'batch_sample_sources': {'policy': ['policy_score']},
                      'batch_required_sources': {'policy': ['policy_score']}},
        )
        worker = RolloutSampleWorker(env, SimpleNamespace(accept_sample_task_result=lambda *a, **kw: True),
                                    replay, AgentCompose(step_agent=[agent]), [agent])
        run = worker.rolling_aec_episode if isinstance(env, ActionSpaceCounterAECEnvironment) else worker.rolling_parallel_episode
        episode = run(info, step_limit=3)['policy'][0]
        replay.push({'policy': [episode]}, notify_learning=False)
        recovered = replay.sample('policy', batch_size=1)
        learner = CustomLearner('mappo', LearnerConfig(name='policy', algo_type='mappo'),
                                agent, env, 'cpu')
        fields = learner.learn_update(recovered)
        assert fields['obs'].shape[:3] == (4, 1, 2)
        assert fields['lengths'] == [3]
        for t in range(3):
            for slot, aid in enumerate(episode.agent_ids):
                if episode.active_agent_mask[aid][t]:
                    assert fields['score'][t, 0, slot] > 0
    finally:
        replay.close()
        env.close()


def test_task_manager_routes_schema_sources_to_the_correct_learner(monkeypatch):
    from types import SimpleNamespace
    from common.plugins import RUNTIME_PLUGINS
    from config import AgentConfig, LearnerConfig, TrainConfig
    from scheduler.train_task import TrainTaskManager

    classes = {
        'qmix': SimpleNamespace(batch_schema=DEFAULT_EPISODE_SCHEMA),
        'mappo': SimpleNamespace(batch_schema=BatchSchema.from_type(CustomFields)),
    }
    monkeypatch.setattr(RUNTIME_PLUGINS, 'has_resource', lambda kind, name: name in classes)
    monkeypatch.setattr(RUNTIME_PLUGINS, 'resolve', lambda kind, name: classes[name])
    manager = TrainTaskManager(
        TrainConfig(max_episode=1, sample_learn_max_lag=None),
        policy_agent_configs=[AgentConfig(name='first'), AgentConfig(name='second')],
        learner_configs=[LearnerConfig(name='first', algo_type='qmix'),
                         LearnerConfig(name='second', algo_type='mappo')],
    )
    try:
        task = manager.get_sample_task_info()
        assert task.metadata['batch_sample_sources'] == {'second': ['policy_score']}
        assert task.metadata['batch_required_sources'] == {'second': ['policy_score']}
    finally:
        manager.close()


def test_episode_transition_subclass_survives_replay_and_assembles_batch():
    from algos.training.mappo_learner import MAPPOActorCriticLearner
    from config import LearnerConfig, ReplayBufferConfig
    from sample.replay_buffer import ReplayBuffer
    from sample.sample_transfer import EpisodeTransition

    class CustomTransition(EpisodeTransition):
        assembled_for = None

        @staticmethod
        def _assemble_batch(episodes, *, learner_name, schema, context):
            CustomTransition.assembled_for = learner_name
            return EpisodeTransition._assemble_batch(episodes, learner_name=learner_name,
                                                      schema=schema, context=context)

    class CustomLearner(MAPPOActorCriticLearner):
        batch_schema = BatchSchema.from_type(CustomFields)

        def update(self, batch):
            return batch.data

    agent, env, _ = setup(spaces()[0])
    replay = ReplayBuffer(ReplayBufferConfig(agent_capacities={'policy': 8}, batch_size=1))
    try:
        episode = rollout(agent, env, 2)
        episode.__class__ = CustomTransition
        for t, metadata in enumerate(episode.metadata):
            metadata['batch_fields'] = {'policy_score': {aid: 1.0 for aid in episode.agent_ids}}
        episode._prepared_replay_episode = None
        replay.push({'policy': [episode]}, notify_learning=False)
        recovered = replay.sample('policy', batch_size=1)
        assert isinstance(recovered[0], CustomTransition)
        learner = CustomLearner('mappo', LearnerConfig(name='policy', algo_type='mappo'),
                                agent, env, 'cpu')
        assert learner.learn_update(recovered)['lengths'] == [2]
        assert CustomTransition.assembled_for == 'policy'
    finally:
        replay.close()
        env.close()


def test_schema_rejects_mismatched_field_types_and_invalid_axes():
    class WrongType(TypedDict):
        value: Annotated[str, TensorField('actions', ('B', 'T', 'N', 'D'), None)]

    class RepeatedAxis(TypedDict):
        value: Annotated[torch.Tensor, TensorField('actions', ('B', 'T', 'T', 'D'), torch.float32)]

    class WrongDtype(TypedDict):
        value: Annotated[torch.Tensor, TensorField('actions', ('B', 'T', 'N', 'D'), 'float32')]

    with pytest.raises(TypeError, match='must have Tensor type'):
        BatchSchema.from_type(WrongType)
    with pytest.raises(ValueError, match='repeated axis'):
        BatchSchema.from_type(RepeatedAxis)
    with pytest.raises(TypeError, match='dtype'):
        BatchSchema.from_type(WrongDtype)


def test_list_field_checks_declared_element_type():
    class WrongLengths(TypedDict):
        lengths: Annotated[list[str], ListField('seq_len', 'B')]

    agent, env, _ = setup(spaces()[0])
    try:
        with pytest.raises(TypeError, match='inconsistent with its list type'):
            assemble_episodes([rollout(agent, env, 2)], BatchSchema.from_type(WrongLengths),
                              agent=agent, device='cpu')
    finally:
        env.close()


def test_single_agent_custom_fields_follow_flattened_transition_order():
    from algos.learners.batch_schema import assemble_single_agent
    from test_a2c import _build_a2c_pair, _build_episode

    class SingleFields(TypedDict):
        obs: Annotated[torch.Tensor, TensorField('single_obs', ('M', 'O'), torch.float32)]
        score: Annotated[torch.Tensor, TensorField('custom_score', ('M', 2), torch.float32)]
        lengths: Annotated[list[int], ListField('seq_len', 'B')]
        scores_by_episode: Annotated[list[list[dict[str, list[int]]]], ListField('custom_score', 'B')]

    agent, _ = _build_a2c_pair()
    episodes = [_build_episode(1, terminated_at_end=True), _build_episode(2, terminated_at_end=False)]
    for episode in episodes:
        for step in range(episode.seq_len):
            episode.metadata[step] = {'batch_fields': {'custom_score': {'solo-0': [episode.episode_num, step]}}}
    schema = BatchSchema.from_type(SingleFields)
    fields = assemble_single_agent(episodes, schema, agent=agent, device='cpu').data
    assert fields['obs'].shape == (8, 3)
    assert fields['lengths'] == [4, 4]
    assert fields['scores_by_episode'][1][2] == {'solo-0': [2, 2]}
    torch.testing.assert_close(fields['score'][:, 0], torch.tensor([1., 1., 1., 1., 2., 2., 2., 2.]))
    torch.testing.assert_close(fields['score'][:, 1], torch.tensor([0., 1., 2., 3., 0., 1., 2., 3.]))
    episodes[1].metadata[2]['batch_fields'] = {}
    with pytest.raises(ValueError, match='episode=2 step=2 agent=solo-0'):
        assemble_single_agent(episodes, schema, agent=agent, device='cpu')


def test_two_schemas_assemble_nested_masks_from_the_same_trajectories():
    class Masks(TypedDict):
        valid: Annotated[torch.Tensor, TensorField('valid', ('T', 'B'), torch.bool)]
        active: Annotated[torch.Tensor, TensorField('active_steps', ('T', 'B', 'N'), torch.bool)]

    class NestedFields(TypedDict):
        obs: Annotated[torch.Tensor, TensorField('obs_with_bootstrap', ('T+1', 'B', 'N', 'O'), torch.float32)]
        masks: Masks
        lengths: Annotated[list[int], ListField('seq_len', 'B')]

    agent, env, _ = setup(spaces()[0])
    try:
        episodes = [rollout(agent, env, 2), rollout(agent, env, 3)]
        standard = assemble_episodes(episodes, DEFAULT_EPISODE_SCHEMA, agent=agent, device='cpu').data
        custom = assemble_episodes(episodes, BatchSchema.from_type(NestedFields), agent=agent, device='cpu').data
        torch.testing.assert_close(custom['obs'], standard['obs'].transpose(0, 1))
        torch.testing.assert_close(custom['masks']['valid'], standard['valid'].transpose(0, 1))
        torch.testing.assert_close(custom['masks']['active'], standard['active'][:, :-1].transpose(0, 1))
        assert custom['lengths'] == [2, 3]
    finally:
        env.close()


def test_required_custom_field_missing_everywhere_identifies_first_decision():
    agent, env, _ = setup(spaces()[0])
    try:
        episode = rollout(agent, env, 2)
        with pytest.raises(ValueError, match='Learner policy episode=1 step=0 agent=a missing field'):
            assemble_episodes([episode], BatchSchema.from_type(CustomFields), agent=agent, device='cpu')
    finally:
        env.close()


def test_list_field_can_preserve_custom_per_step_agent_values():
    class RawFields(TypedDict):
        scores: Annotated[list[list[dict[str, float]]], ListField('raw_score', 'B')]

    agent, env, _ = setup(spaces()[0])
    try:
        episodes = [rollout(agent, env, 2), rollout(agent, env, 3)]
        for episode in episodes:
            for step, metadata in enumerate(episode.metadata):
                metadata['batch_fields'] = {'raw_score': {aid: float(step) for aid in episode.agent_ids}}
        result = assemble_episodes(episodes, BatchSchema.from_type(RawFields), agent=agent, device='cpu')
        assert result.data['scores'] == [
            [{'a': 0., 'b': 0.}, {'a': 1., 'b': 1.}],
            [{'a': 0., 'b': 0.}, {'a': 1., 'b': 1.}, {'a': 2., 'b': 2.}],
        ]
    finally:
        env.close()


@pytest.mark.parametrize('transport', ['grpc', 'torch_rpc'])
def test_custom_fields_survive_rpc_payload_snapshot_and_step_slice(transport, tmp_path):
    import pickle
    from config import ReplayBufferConfig
    from rpc.dispatch import prepare_torch_rpc_payload
    from rpc.grpc.codec import pack_value, unpack_value
    from sample.replay.replay_payload import encode_replay_push_payload
    from sample.replay_buffer import ReplayBuffer

    agent, env, _ = setup(spaces()[0])
    config = ReplayBufferConfig(agent_capacities={'policy': 8}, batch_size=1)
    source, restored = ReplayBuffer(config), ReplayBuffer(config)
    try:
        episode = rollout(agent, env, 3)
        for step, metadata in enumerate(episode.metadata):
            metadata['batch_fields'] = {'policy_score': {aid: float(10 + step) for aid in episode.agent_ids}}
        episode._prepared_replay_episode = None
        episode.prepare_for_replay()
        payload = encode_replay_push_payload({'policy': [episode]})
        if transport == 'grpc':
            payload = unpack_value(pack_value(payload))
        else:
            payload = pickle.loads(pickle.dumps(prepare_torch_rpc_payload(payload)))
        source.push(payload, notify_learning=False)
        path = source.save_snapshot(tmp_path / f'{transport}.drbf')
        restored.load_snapshot(path)
        window = restored._materialize_episode(restored.buffer['policy'][0], 1, 3, sliced=True)
        assert window.burn_in.seq_len == 1
        fields = assemble_episodes([window], BatchSchema.from_type(CustomFields), agent=agent, device='cpu').data
        assert fields['lengths'] == [2]
        torch.testing.assert_close(fields['score'][:, 0], torch.tensor([[11., 11.], [12., 12.]]))
    finally:
        source.close()
        restored.close()
        env.close()

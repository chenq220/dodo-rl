"""历史轨迹在读取边界补齐契约，当前组批只消费明确的属性。"""
import pickle

import pytest

from config import ReplayBufferConfig
from sample.replay.replay_persist import deserialize_persist_value, serialize_persist_value
from sample.replay_buffer import ReplayBuffer
from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition


@pytest.mark.parametrize('transition_type', [EpisodeTransition, SAEpisodeTransition])
@pytest.mark.parametrize('codec', ['pickle', 'snapshot', 'legacy_json'])
def test_legacy_trajectory_read_restores_optional_contract(transition_type, codec):
    episode = transition_type()
    episode.episode_num = 37
    episode.obs = {'a': [[1.0]]} if transition_type is EpisodeTransition else [[1.0]]
    missing = ['starts_with_episode', '_prepared_replay_episode']
    if transition_type is EpisodeTransition:
        missing += ['agent_ids', 'burn_in', 'action_space_specs', 'active_agent_mask',
                    'next_active_agent_mask', 'hidden_in', 'hidden_out', 'avail_actions']
    else:
        missing += ['action_space_spec', 'old_log_probs', 'old_values', 'avail_actions']
    for name in missing:
        delattr(episode, name)
    if codec == 'pickle':
        restored = pickle.loads(pickle.dumps(episode))
    elif codec == 'snapshot':
        restored = deserialize_persist_value(serialize_persist_value(episode))
    else:
        replay = ReplayBuffer(ReplayBufferConfig())
        try:
            restored = replay._deserialize_value(replay._serialize_value(episode))
        finally:
            replay.close()
    assert restored.episode_num == 37
    assert restored.starts_with_episode is True
    assert restored._prepared_replay_episode is None
    assert restored.avail_actions == []
    if transition_type is EpisodeTransition:
        assert restored.agent_ids == ('a',)
        assert restored.burn_in is None
        assert restored.action_space_specs == {}
        assert restored.active_agent_mask == restored.next_active_agent_mask == {}
        assert restored.hidden_in == restored.hidden_out == {}
    else:
        assert restored.action_space_spec is None
        assert restored.old_log_probs == restored.old_values == []


def test_roundtrip_preserves_recorded_window_boundaries():
    episode = EpisodeTransition(agent_ids=('a',))
    prefix = EpisodeTransition(agent_ids=('a',))
    episode.starts_with_episode = False
    episode.burn_in = prefix
    restored = pickle.loads(pickle.dumps(episode))
    assert restored.starts_with_episode is False
    assert restored.burn_in.agent_ids == ('a',)
    assert restored.burn_in.burn_in is None

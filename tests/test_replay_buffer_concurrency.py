from concurrent.futures import ThreadPoolExecutor

from config import ReplayBufferConfig
from sample.replay_buffer import ReplayBuffer
from sample.sample_transfer import SAEpisodeTransition


AGENT_NAME = "concurrent_agent"


def _episode(episode_num: int) -> SAEpisodeTransition:
    transition = SAEpisodeTransition()
    transition.episode_num = episode_num
    return transition


def test_concurrent_push_fifo_keeps_newest_capacity():
    capacity = 7
    push_count = 2_000
    replay_buffer = ReplayBuffer(ReplayBufferConfig(
        agent_capacities={AGENT_NAME: capacity},
        batch_size=1,
    ))

    def push(episode_num: int) -> None:
        replay_buffer.push(
            _episode(episode_num),
            agent_name=AGENT_NAME,
            episode_num=episode_num,
            notify_learning=False,
        )

    with ThreadPoolExecutor(max_workers=32) as executor:
        list(executor.map(push, range(push_count)))

    assert len(replay_buffer.buffer[AGENT_NAME]) == capacity
    assert len(replay_buffer.sample_seq[AGENT_NAME]) == capacity
    assert replay_buffer.write_count[AGENT_NAME] == push_count
    seqs = [seq for seq, _ in replay_buffer._ordered_buffer_with_seq(AGENT_NAME)]
    assert seqs == list(range(push_count - capacity + 1, push_count + 1))


def test_state_returns_detached_cursor_snapshots():
    replay_buffer = ReplayBuffer(ReplayBufferConfig(agent_capacities={AGENT_NAME: 2}, batch_size=1))
    replay_buffer.push(
        _episode(1),
        agent_name=AGENT_NAME,
        episode_num=1,
        notify_learning=False,
    )

    state = replay_buffer.state()
    state.position[AGENT_NAME] = 99
    state.sliding_idx[AGENT_NAME] = 99

    assert replay_buffer.position[AGENT_NAME] == 0
    assert AGENT_NAME not in replay_buffer.sliding_idx

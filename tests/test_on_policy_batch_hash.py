from types import SimpleNamespace

import numpy as np

from utils.transition_hash import transition_batch_hash


def test_transition_batch_hash_stable_and_sensitive():
    first = SimpleNamespace(
        episode_num=1,
        seq_len=2,
        obs=np.array([[1.0, 2.0]], dtype=np.float32),
        rewards=np.array([1.0], dtype=np.float32),
    )
    second = SimpleNamespace(
        episode_num=1,
        seq_len=2,
        obs=np.array([[1.0, 2.0]], dtype=np.float32),
        rewards=np.array([1.0], dtype=np.float32),
    )
    third = SimpleNamespace(
        episode_num=1,
        seq_len=2,
        obs=np.array([[1.0, 3.0]], dtype=np.float32),
        rewards=np.array([1.0], dtype=np.float32),
    )
    assert transition_batch_hash([first]) == transition_batch_hash([second])
    assert transition_batch_hash([first]) != transition_batch_hash([third])
    assert transition_batch_hash([]) == transition_batch_hash([])

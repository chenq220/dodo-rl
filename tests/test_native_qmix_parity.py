"""Fixed legacy-QMIX update and recurrent-window regression checks."""

import pytest
import torch

from algos.learners.batch_schema import DEFAULT_EPISODE_SCHEMA, assemble_episodes
from test_grouped_replay_qmix_contracts import _episodes, _learner, _replay, AGENT_NAME


def test_native_qmix_matches_recorded_legacy_update():
    # Recorded on CPU from 0.16.2 EPyMARL with this seed and fixture, before
    # removing the vendor implementation. This goes through the public schema.
    torch.manual_seed(91)
    learner = _learner(standardise=True)
    metrics = learner.learn_update(_episodes())
    assert metrics == pytest.approx({
        "loss": 357877.5625,
        "grad_norm": 12796798.0,
        "td_error_abs": 317.7453884548611,
        "q_taken_mean": 79.36723497178819,
        "target_mean": 1.1564956771002876e-06,
    }, rel=1e-6, abs=1e-7)
    expected = {
        "agent_net": {
            "fc1.weight": [-0.106109381, -0.055437330, -0.139364868],
            "fc2.bias": [0.041396890, 0.061113629],
        },
        "mixer_net": {
            "hyper_w_1.0.weight": [0.312836945, 0.060756613, -0.100144103],
            "V.2.bias": [0.122221537],
        },
    }
    for name, tensors in expected.items():
        state = learner.models[name].state_dict()
        for key, values in tensors.items():
            torch.testing.assert_close(state[key].flatten()[:len(values)], torch.tensor(values),
                                       rtol=1e-6, atol=1e-7)


def test_schema_burn_in_matches_direct_recurrent_reconstruction(tmp_path):
    torch.manual_seed(37)
    learner = _learner()
    replay = _replay(tmp_path, capacities={AGENT_NAME: 8})
    try:
        replay.push({AGENT_NAME: _episodes()}, notify_learning=False)
        starts = [1, 3, 0]
        windows = [replay._materialize_episode(meta, start, start + 2, sliced=True)
                   for meta, start in zip(replay.buffer[AGENT_NAME], starts)]
        expected = learner._slice_initial_states(windows)
        fields = assemble_episodes(windows, DEFAULT_EPISODE_SCHEMA,
                                   agent=learner.agent, device='cpu').data
        actual = learner._schema_initial_states(fields)
        for expected_state, actual_state in zip(expected, actual):
            torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    finally:
        replay.close()

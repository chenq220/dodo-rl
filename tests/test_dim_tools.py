import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.dim_tools import extend_obs


def test_extend_obs_empty_special_values_returns_unchanged():
    base = np.array([1.0, 2.0], dtype=np.float32)
    assert extend_obs(base, []) is base


def test_extend_obs_dim_numpy():
    base = np.array([1.0, 2.0], dtype=np.float32)
    result = extend_obs(base, [3, 5], method="dim")
    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32))


def test_extend_obs_onehot_numpy():
    base = np.array([0.5], dtype=np.float32)
    result = extend_obs(base, [1, 0], method="onehot")
    np.testing.assert_array_equal(
        result,
        np.array([0.5, 0.0, 1.0, 1.0, 0.0], dtype=np.float32),
    )


def test_extend_obs_onehot_torch():
    base = torch.tensor([0.5, 0.25], dtype=torch.float32)
    result = extend_obs(base, [2], method="onehot")
    expected = torch.tensor([0.5, 0.25, 0.0, 0.0, 1.0], dtype=torch.float32)
    torch.testing.assert_close(result, expected)


def test_extend_obs_dim_torch_preserves_device():
    base = torch.tensor([1.0], dtype=torch.float32, device="cpu")
    result = extend_obs(base, [7], method="dim")
    assert result.device == base.device
    torch.testing.assert_close(result, torch.tensor([1.0, 7.0], dtype=torch.float32))


def test_extend_obs_box_space():
    space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    extended = extend_obs(space, [1], method="onehot")
    assert extended.shape == (4,)
    assert isinstance(extended, Box)


def test_extend_obs_invalid_onehot_index():
    base = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="out of range"):
        extend_obs(base, [-1], method="onehot")

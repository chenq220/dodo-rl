from typing import Literal, List, Union

import numpy as np
import torch
from gymnasium.spaces import Box

from common.types import ObsType


def state_shape(env):
    shape = env.state_space().shape
    return shape[0] if len(shape) == 1 else shape


def _validate_onehot_values(special_values: List[int]) -> int:
    if not special_values:
        return 0
    max_val = max(special_values)
    for value in special_values:
        if value < 0 or value > max_val:
            raise ValueError(
                f"onehot index {value} out of range [0, {max_val}]"
            )
    return max_val + 1


def _onehot_numpy(special_values: List[int], width: int, dtype) -> np.ndarray:
    encoded = np.zeros((len(special_values), width), dtype=dtype)
    for row, value in enumerate(special_values):
        encoded[row, value] = 1.0
    return encoded.reshape(-1)


def _onehot_torch(
    special_values: List[int],
    width: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.tensor(special_values, dtype=torch.long, device=device)
    return torch.nn.functional.one_hot(indices, num_classes=width).to(dtype).reshape(-1)


def _extend_box(
    space: Box,
    special_values: List[int],
    method: Literal["onehot", "dim"],
) -> Box:
    base_dim = int(np.prod(space.shape))
    dtype = space.dtype

    if method == "onehot":
        width = _validate_onehot_values(special_values)
        extra_dim = len(special_values) * width
        extra_low = np.zeros(extra_dim, dtype=dtype)
        extra_high = np.ones(extra_dim, dtype=dtype)
    else:
        extra_dim = len(special_values)
        extra_low = np.full(extra_dim, -np.inf, dtype=dtype)
        extra_high = np.full(extra_dim, np.inf, dtype=dtype)

    base_low = np.full((base_dim,), space.low, dtype=dtype)
    base_high = np.full((base_dim,), space.high, dtype=dtype)
    return Box(
        low=np.concatenate([base_low, extra_low]),
        high=np.concatenate([base_high, extra_high]),
        shape=(base_dim + extra_dim,),
        dtype=dtype,
    )


def extend_obs(
    base_obs: ObsType,
    special_values: List[int],
    method: Literal["onehot", "dim"] = "onehot",
) -> ObsType:
    """在观察态势中拼入其他特征值"""
    if not special_values:
        return base_obs

    if isinstance(base_obs, Box):
        return _extend_box(base_obs, special_values, method)

    if isinstance(base_obs, np.ndarray):
        if method == "onehot":
            width = _validate_onehot_values(special_values)
            extra = _onehot_numpy(special_values, width, base_obs.dtype)
        else:
            extra = np.asarray(special_values, dtype=base_obs.dtype)
        return np.concatenate([base_obs, extra])

    if isinstance(base_obs, torch.Tensor):
        if method == "onehot":
            width = _validate_onehot_values(special_values)
            extra = _onehot_torch(
                special_values,
                width,
                dtype=base_obs.dtype,
                device=base_obs.device,
            )
        else:
            extra = torch.tensor(
                special_values,
                dtype=base_obs.dtype,
                device=base_obs.device,
            )
        return torch.cat([base_obs, extra])

    raise TypeError(f"unsupported observation type: {type(base_obs)!r}")

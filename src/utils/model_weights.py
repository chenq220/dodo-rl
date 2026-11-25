from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def snapshot_model_weights(
        models: Mapping[str, nn.Module | Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """生成可安全传递给 ModelManager/RPC 的 CPU 权重快照。

    只复制 state_dict，不复制优化器、计算图或 SB3 算法对象。
    """
    snapshots: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        state_dict = model.state_dict() if isinstance(model, nn.Module) else model
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Model {model_name} must be nn.Module or state_dict, given={type(model).__name__}"
            )
        snapshots[model_name] = {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in state_dict.items()
        }
    return snapshots

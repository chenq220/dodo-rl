"""Algorithm-independent network checkpoint validation."""
from typing import Dict
import torch
from torch import nn
from common.exception import ModelCheckpointCompatibilityException


def extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("agent_net", "model_state_dict", "state_dict"):
            state_dict = checkpoint.get(key)
            if isinstance(state_dict, dict):
                return state_dict
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError("unsupported checkpoint format; expected module or state_dict")


def validated_checkpoint_state_dict(
        network: nn.Module,
        checkpoint: object,
        *,
        agent_name: str,
        model_name: str,
        expected_dimensions: dict,
) -> Dict[str, torch.Tensor]:
    """加载前检查完整键集合和每个张量 shape；旧裸权重不伪造组宽元数据。"""
    expected = network.state_dict()
    source_args = getattr(checkpoint, "args", None)
    given_dimensions = {
        key: getattr(source_args, key)
        for key in expected_dimensions
        if source_args is not None and hasattr(source_args, key)
    }
    context = (
        f"expected dimensions={expected_dimensions}; "
        f"given dimensions={given_dimensions or 'unknown (checkpoint has no architecture metadata)'}"
    )
    try:
        given = extract_state_dict(checkpoint)
    except ValueError as exc:
        raise ModelCheckpointCompatibilityException(
            agent_name, model_name, f"{context}; {exc}",
        ) from exc
    missing = sorted(set(expected) - set(given))
    unexpected = sorted(set(given) - set(expected))
    mismatches = []
    for key in sorted(set(expected) & set(given)):
        value = given[key]
        actual_shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        if actual_shape != tuple(expected[key].shape):
            mismatches.append(f"{key}: expected={tuple(expected[key].shape)}, given={actual_shape}")
    dimension_mismatches = [
        f"{key}: expected={expected_dimensions[key]}, given={value}"
        for key, value in given_dimensions.items()
        if value != expected_dimensions[key]
    ]
    if missing or unexpected or mismatches or dimension_mismatches:
        raise ModelCheckpointCompatibilityException(
            agent_name, model_name,
            f"{context}; missing keys={missing}; unexpected keys={unexpected}; "
            f"shape mismatches={mismatches}; dimension mismatches={dimension_mismatches}",
        )
    return given

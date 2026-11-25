from __future__ import annotations

import hashlib
from typing import Iterable, Any

import numpy as np


def transition_batch_hash(batch_data: Iterable[Any]) -> str:
    """生成on-policy样本批次指纹，用于算法侧拒绝重复batch更新。"""
    digest = hashlib.blake2b(digest_size=16)
    for transition in batch_data:
        _update_transition_digest(digest, transition)
    return digest.hexdigest()


def single_transition_key(episode: Any, step: int) -> str:
    """Stable fingerprint of a single transition, including its original episode."""
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(getattr(episode, "episode_num", 0)).encode("utf-8"))
    for value in (
        episode.obs[step], int(episode.actions[step]), float(episode.rewards[step]),
        episode.next_obs[step], bool(episode.terminations[step]), bool(episode.truncations[step]),
    ):
        _update_value_digest(digest, value)
    return digest.hexdigest()


def _update_transition_digest(digest: "hashlib._Hash", value: Any) -> None:
    """递归写入transition中的核心轨迹字段，避免依赖对象id。"""
    digest.update(type(value).__name__.encode("utf-8"))
    for field_name in (
            "episode_num",
            "starts_with_episode",
            "seq_len",
            "obs",
            "actions",
            "last_actions",
            "rewards",
            "next_obs",
            "terminations",
            "truncations",
            "states",
            "next_states",
            "step_rewards",
    ):
        if hasattr(value, field_name):
            digest.update(field_name.encode("utf-8"))
            _update_value_digest(digest, getattr(value, field_name))


def _update_value_digest(digest: "hashlib._Hash", value: Any) -> None:
    if value is None:
        digest.update(b"<none>")
        return
    if isinstance(value, (str, int, float, bool)):
        digest.update(repr(value).encode("utf-8"))
        return
    if isinstance(value, np.ndarray):
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(value).tobytes())
        return
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        arr = value.detach().cpu().numpy()
        _update_value_digest(digest, arr)
        return
    if isinstance(value, dict):
        for key in sorted(value.keys(), key=lambda item: repr(item)):
            digest.update(repr(key).encode("utf-8"))
            _update_value_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(str(len(value)).encode("utf-8"))
        for item in value:
            _update_value_digest(digest, item)
        return
    digest.update(repr(value).encode("utf-8"))

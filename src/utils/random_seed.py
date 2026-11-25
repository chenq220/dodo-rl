from __future__ import annotations

import hashlib
import random
from typing import Optional

import numpy as np
import torch

from common.logger import init_logger

LOG = init_logger("random_seed")

_MAX_NUMPY_SEED = 2 ** 32 - 1


def normalize_seed(seed: Optional[int]) -> Optional[int]:
    """把用户配置的随机种子规范到 Python/NumPy/PyTorch 都能接受的整数范围。"""
    if seed is None:
        return None
    if isinstance(seed, str):
        stripped = seed.strip()
        if stripped == "" or stripped.lower() == "none":
            return None
    seed_value = int(seed) % _MAX_NUMPY_SEED
    return seed_value if seed_value > 0 else 1


def derive_generation_run_seed(task_id: str, stage_name: str, generation: int) -> int:
    """TOML/AutoML 均未指定 task seed 时，按 task+stage+generation 派生稳定可复现种子。"""
    payload = f"{task_id}|{stage_name}|{generation}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return normalize_seed(int.from_bytes(digest[:8], "big")) or 1


def derive_seed(seed: Optional[int], *parts: object) -> Optional[int]:
    """
    从训练任务级 seed 派生稳定子 seed。

    分布式训练中多个采样进程如果使用完全相同的 seed，会得到相同动作随机序列；
    因此每个角色用 role_id/stage_name 等稳定字符串派生自己的进程级 seed。
    """
    base_seed = normalize_seed(seed)
    if base_seed is None:
        return None
    if not parts:
        return base_seed
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return normalize_seed(int.from_bytes(digest[:8], "big"))


def set_global_seed(seed: Optional[int], *, context: str = "") -> Optional[int]:
    """
    固定当前进程内主要随机源。

    覆盖范围：
    - Python random：RandomAgent、RNNAgent epsilon 分支、ReplayBuffer random.sample。
    - NumPy random：np.random.choice 动作采样、epsilon-greedy 随机动作。
    - PyTorch random：网络初始化、torch.randperm、策略分布内部采样。
    - stable_baselines3：若已安装，额外调用其 set_random_seed。
    """
    seed_value = normalize_seed(seed)
    if seed_value is None:
        return None

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    try:
        from stable_baselines3.common.utils import set_random_seed as sb3_set_random_seed

        sb3_set_random_seed(seed_value)
    except Exception as exc:
        LOG.debug(f"stable_baselines3 seed hook skipped: {exc}")

    LOG.debug(f"Global random seed applied: seed={seed_value}, context={context}")
    return seed_value


def apply_train_seed(
        train_config,
        *,
        role_id: Optional[str] = None,
        context: str,
) -> Optional[int]:
    """
    以 TrainConfig.seed 为权威，设置当前进程初始随机源。

    运行期采样/学习随机性由 TrainTaskManager 下发的 EpisodeInfo.seed / LearnInfo.seed 控制；
    这里仅负责角色初始化阶段的网络初始化、对象构造等随机性。
    """
    # AutoML 的后续 generation 会恢复首轮模型快照，但使用新的运行 seed。
    # 归档复现时 initialization_seed 固定初始权重，seed 继续控制 episode/learn 令牌。
    base_seed = getattr(train_config, "initialization_seed", None)
    if base_seed is None:
        base_seed = getattr(train_config, "seed", None)
    process_seed = derive_seed(base_seed, role_id) if role_id else normalize_seed(base_seed)
    if process_seed is None:
        return None

    set_global_seed(process_seed, context=context)
    return process_seed

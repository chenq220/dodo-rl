"""学习许可策略：依据 ReplayBuffer 状态决定当前哪些 Learner 可以更新。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from sample.model import ReplayBufferState, SamplingStrategy


@dataclass(frozen=True)
class LearnPermitRequest:
    """许可决策的只读输入；不表达按 episode 累积的学习债务。"""

    generation: int
    stage_name: str
    replay_buffer_state: ReplayBufferState
    candidate_learners: tuple[str, ...]
    algo_configs: Mapping[str, Any]


@dataclass(frozen=True)
class LearnPermitGrant:
    """一张可执行许可；没有 Learner 就不生成许可。"""

    active_learners: list[str]


class LearnPermitPolicy(Protocol):
    """根据当前缓存状态决定是否允许一次学习。"""

    def decide(self, request: LearnPermitRequest) -> LearnPermitGrant | None:
        """返回当前可学习的 Learner；返回 None 表示继续等待样本。"""


def replay_warmup_remaining(state: ReplayBufferState, algo_config: Any, learner_name: str) -> int:
    """按回放容量单位检查 active 分区，不把 rehearsal 存档计入。"""
    threshold = getattr(algo_config, "min_replay_size", None)
    if threshold is None:
        return 0
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= algo_config.capacity:
        raise ValueError("min_replay_size must be an integer between 0 and capacity")
    counts = state.agent_episode_counts if state.sample_packaging == "episode_groups" else state.agent_trajectory_counts
    return max(0, threshold - counts.get(learner_name, 0))


def replay_buffer_ready_for_learner(
    replay_buffer_state: ReplayBufferState,
    algo_config: Any,
    learner_name: str | None = None,
) -> bool:
    """判断 ReplayBuffer 是否具备一次 Learner.update 的最低条件。"""
    # 旧的调用方可能只携带标量 size；真实 ReplayBuffer.state 会给出分区明细。
    size = (
        replay_buffer_state.agent_sizes.get(learner_name, 0)
        if learner_name is not None and replay_buffer_state.agent_sizes
        else replay_buffer_state.size
    )
    sample_strategy = getattr(algo_config, "sample_strategy")
    if sample_strategy == SamplingStrategy.UNIFORM_STEPS.value:
        steps = replay_buffer_state.agent_step_counts
        archive_steps = replay_buffer_state.agent_rehearsal_step_counts
        if learner_name is not None and (steps or archive_steps):
            return steps.get(learner_name, 0) > 0 or archive_steps.get(learner_name, 0) > 0
        return any(steps.values()) or any(archive_steps.values()) or size > 0
    if sample_strategy in (
            SamplingStrategy.ON_POLICY_RECENT_STEPS.value,
            SamplingStrategy.ON_POLICY_CURSOR_STEPS.value,
            SamplingStrategy.TIME_SLICE.value,
    ):
        return size > 0
    batch_size = int(getattr(algo_config, "batch_size"))
    return size >= batch_size


class ReplayReadyLearnPermitPolicy:
    """默认策略：只要缓存满足条件，就允许当前可用 Learner 执行一次更新。"""

    def decide(self, request: LearnPermitRequest) -> LearnPermitGrant | None:
        ready: list[str] = []
        for learner_name in request.candidate_learners:
            algo_config = request.algo_configs.get(learner_name)
            if algo_config is None:
                continue
            if replay_buffer_ready_for_learner(
                request.replay_buffer_state, algo_config, learner_name
            ):
                ready.append(learner_name)
        if not ready:
            return None
        return LearnPermitGrant(active_learners=ready)

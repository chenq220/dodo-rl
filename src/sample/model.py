from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TYPE_CHECKING, TypeVar
from enum import Enum

if TYPE_CHECKING:
    from common.types import AgentName


BatchFields = TypeVar("BatchFields")


@dataclass(frozen=True)
class EpisodeBatch(Generic[BatchFields]):
    """Training fields assembled according to a Learner's batch schema."""

    data: BatchFields
    # 本批实际使用的回放策略；直接组装轨迹时可以为空。
    sample_strategy: Optional[str] = None

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "data")
        try:
            return data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


# Compatibility name for callers using the former single-agent batch type.
SABatchTensorData = EpisodeBatch


class SampledEpisodes(list):
    """携带实际抽样策略的轨迹列表，供 Learner 组批时读取。"""

    def __init__(self, episodes, *, sample_strategy: str):
        super().__init__(episodes)
        self.sample_strategy = sample_strategy

class SamplingStrategy(str, Enum):
    """ReplayBuffer 抽样策略枚举。"""
    # 近期偏好抽样：样本的1/3从最近样本中取一部分，2/3从早历史样本中随机抽样
    # 适合QMIX/VDN/DQN这类off-policy价值学习算法。
    RECENT_BIASED = "recent_biased"
    # 近期滚动抽样：取当前样本池最后batch_size个样本
    # 使用于PPO/MAPPO/A2C策略优化等算法，激进、偏最新、可能赌出更好策略，也可能崩。
    # 局内时间维连续切片：随机抽取一个起点，只在同一局内取 [t:t+L]；短于 L 的局跳过。
    # 适用于QMIX/RNN、n-step、GAE等需要时序片段的算法；batch_size语义是连续步数L，不是episode数量。
    TIME_SLICE = "time_slice"
    ON_POLICY_RECENT = "on_policy"
    # 近期滚动按步抽样：从当前ReplayBuffer尾部向前累计batch_size个transition。
    # batch_size语义是transition步数，不是episode数量；无游标，多次调用会重复返回当前最新窗口。
    ON_POLICY_RECENT_STEPS = "on_policy_recent_steps"
    # 近期游动抽样：取当前ReplayBuffer中最近batch_size个后，往后等待batch_size个轮次再抽样，等待期间样本返回空列表，即所有样本只会被抽样到一次。
    # 适用于等PPO/MAPPO/A2C策略优化等算法，稳定、可解释。
    ON_POLICY_CURSOR = "on_policy_cursor"
    # 近期游动按步抽样：按step游标切出batch_size步，成功后游标前进batch_size步。
    # batch_size语义是transition步数；样本不足时返回空列表，已返回step不会被重复抽样。
    ON_POLICY_CURSOR_STEPS = "on_policy_cursor_steps"
    # 滑动窗口抽样：从第一个样本开始设定游标，每次抽样后游标向后移动一格，从游标位置开始抽样，游标滑动末尾则从头开始。
    # 用于离线学习样本
    SLIDING_WINDOW = "sliding_window"
    # 完全随机抽样：从指定Agent的全部缓存episode中均匀随机抽取batch_size个。
    # 实验算法
    UNIFORM = "uniform"
    # 从所有有效单步转移中均匀、有放回地抽取，batch_size 表示抽取的步数。
    UNIFORM_STEPS = "uniform_steps"


ROLLING_STRATEGIES = frozenset({
    SamplingStrategy.ON_POLICY_RECENT,
    SamplingStrategy.ON_POLICY_RECENT_STEPS,
    SamplingStrategy.RECENT_BIASED,
    SamplingStrategy.SLIDING_WINDOW,
    SamplingStrategy.UNIFORM,
    SamplingStrategy.UNIFORM_STEPS,
})


def is_older_sample_version(version: str, current_version: Optional[str]) -> bool:
    """Compare rounds only within the same TaskManager namespace and generation."""
    if current_version is None:
        return False
    scope, sep, number = version.rpartition("_v")
    current_scope, _, current_number = current_version.rpartition("_v")
    return bool(sep and scope == current_scope and number.isdecimal()
                and current_number.isdecimal() and int(number) < int(current_number))


@dataclass(frozen=True)
class ReplayBufferState:
    """ReplayBuffer 状态快照，用于任务调度模块判断是否可以发放学习令牌。"""

    # 当前回放池所属训练代次；TaskManager重启训练后Role会同步该值。
    generation: int = 0
    # 当前缓存的 episode 数量。
    size: int = 0
    # 回放池最大 episode 容量。
    capacity: int = 0
    # 每个 Learner/AgentName 分区的当前可采样轨迹数和 active 容量。
    agent_sizes: dict[AgentName, int] = field(default_factory=dict)
    agent_capacities: dict[AgentName, int] = field(default_factory=dict)
    # 各 Agent 回放池写入游标；active 按时间顺序追加后恒为 0。
    position: dict[AgentName, int] = 0
    # 当前默认采样策略。
    strategy: SamplingStrategy | str = 0
    # 滑动窗口采样游标。
    sliding_idx: dict[AgentName, int] = 0
    # 样本封装/抽样单位：independent（独立组轨迹）或 episode_groups（同局捆绑）。
    # size/agent_sizes 的单位随模式不同：独立组为组轨迹数，捆绑为物理局数。
    sample_packaging: str = "independent"
    # 各分区当前 active 的物理局数（同局多组封成一个样本时的抽样单位数）。
    agent_episode_counts: dict[AgentName, int] = field(default_factory=dict)
    # 各分区当前 active 的有效组轨迹数（无论封装模式都按组轨迹计数）。
    agent_trajectory_counts: dict[AgentName, int] = field(default_factory=dict)
    # 各分区当前 active 的有效环境步数（全部组轨迹 step_count 之和）。
    agent_step_counts: dict[AgentName, int] = field(default_factory=dict)
    # 启用 rehearsal 混采时，各分区归档中的有效组轨迹步数。
    agent_rehearsal_step_counts: dict[AgentName, int] = field(default_factory=dict)
    sample_version: Optional[str] = None
    sample_versions: dict[AgentName, str] = field(default_factory=dict)
    versioned_writes: int = 0
    overlap_reads: int = 0
    overlap_ready_agents: set[AgentName] = field(default_factory=set)
    versioned_available: dict[AgentName, int] = field(default_factory=dict)
    versioned_fresh_agents: set[AgentName] = field(default_factory=set)

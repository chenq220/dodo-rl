from __future__ import annotations
import copy
import os
import random
import importlib
import sys
import threading
from bisect import bisect_right
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict, List, Optional, Iterator, Any, Callable, TYPE_CHECKING, Union, Tuple, Mapping
)
import numpy as np
import torch
from common.exception import ReplayPackagingException, S3ProtocolException
from scheduler.artifact import ArtifactCallerSupport, ArtifactType, ensure_output_parent
from common.process_shutdown import process_close
from common.registry import extensible
from sample.model import ROLLING_STRATEGIES, ReplayBufferState, SamplingStrategy, SampledEpisodes, is_older_sample_version
from sample.replay.replay_payload import (
    BatchEncodingUnsupported as _BatchEncodingUnsupported,
    ContainerTemplate as _ContainerTemplate,
    InlineRef as _InlineRef,
    OptionalTemplate as _OptionalTemplate,
    PreparedReplayEpisode,
    ReplayPushEnvelope,
    ReplayPushEntryKey,
    ReplayTransitionEnvelope,
    TensorFieldSpec,
    TensorOwnership,
    TensorRef as _TensorRef,
    contains_tensor_data,
)
from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition
from sample.replay.replay_perf import record, timed
from sample.replay.step_tensor_pool import (
    KIND_MMAP_DIR,
    KIND_TORCH_SAVE,
    LazyMemmapStepTensorPool,
    StepTensorPool,
    build_step_tensor_pool,
    lazy_memmap_max_steps_of,
    resolve_mmap_scratch_dir,
)
from common.logger import init_logger
from utils.memory import estimate_object_memory_mb
from utils.task_id import task_id

if TYPE_CHECKING:
    from config import ReplayBufferConfig, StageConfig
    from common.types import AgentName

LOG = init_logger('replay_buffer')


def _process_rss_bytes() -> int:
    """当前进程 RSS，供 mmap 日志与内存池区分；失败时返回 0。"""
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        pass
    try:
        import resource
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss
        return rss * 1024
    except Exception:
        return 0


def _path_has_files(path: Path) -> bool:
    """目录树里是否有普通文件；空壳（空目录或只有空子目录）视为没有产物。"""
    if not path.exists():
        return False
    return any(child.is_file() for child in path.rglob("*"))


# 本地调用按样本原始设备恢复；RPC 服务入口临时覆盖为 CPU，避免控制进程
# 先把回放数据搬到 MPS/CUDA，再由传输层搬回 CPU。
_RESTORE_TENSOR_DEVICE: ContextVar[Optional[torch.device]] = ContextVar(
    "replay_restore_tensor_device",
    default=None,
)
# 单步窗口采样是否附带同局完整 burn-in 前缀。非 RNN Learner 不需要该前缀，
# 关闭后可避免每个样本重复物化 1~3MB 历史数据。
_INCLUDE_BURN_IN: ContextVar[bool] = ContextVar(
    "replay_include_burn_in",
    default=True,
)

# IDEA: 后续可能算法可能会要求开发自己的样本数据格式，要考虑给样本或者样本转化器做一个模型
TransitionType = Union[EpisodeTransition, SAEpisodeTransition]


@dataclass(frozen=True)
class TensorSchema:
    """
    可复用的Step张量布局说明。

    Schema只描述字段结构、shape和dtype；实际张量在StepTensorPool中，逐步变化的非张量值
    在StepMeta.inline_values中。
    """

    schema_id: int
    template: Any
    fields: Tuple[TensorFieldSpec, ...]


@dataclass(frozen=True)
class StepMeta:
    """一个逻辑Step的元数据；通过tensor_index关联独立张量池。"""

    tensor_index: int
    schema_id: int
    episode_seq: int
    step_offset: int
    inline_values: Tuple[Any, ...]
    is_episode_start: bool
    is_episode_end: bool


@dataclass
class EpisodeMeta:
    """一个Episode的元数据，只索引StepMeta范围，不持有任何张量。"""

    agent_name: Any
    transition_type: type
    episode_num: int
    write_seq: int
    step_meta_start: int
    step_meta_end: int
    step_count: int
    step_field_layout: Dict[str, Tuple[str, Tuple[Any, ...]]]
    static_fields: Dict[str, Any]
    starts_with_episode: bool
    has_starts_with_episode: bool
    sample_version: Optional[str] = None


@extensible
class ReplayBuffer(ArtifactCallerSupport):
    """
    经验回放池，负责以 episode 为单位存储、覆盖、抽样和可选持久化轨迹数据。

    ReplayBuffer按AgentName保存完整episode，兼容多智能体EpisodeTransition和单智能体SAEpisodeTransition；
    不同SamplingStrategy只影响样本选择方式，不改变池内对象。满池超出capacity时按FIFO丢掉最旧一局。
    """

    # 对齐 torchrl SliceSampler.strict_length。False：T<L 的局整段留下。
    STRICT_LENGTH = False

    # 样本封装/抽样单位合法值；容量与 batch 的单位随模式变化。
    SAMPLE_PACKAGING_INDEPENDENT = "independent"
    SAMPLE_PACKAGING_EPISODE_GROUPS = "episode_groups"

    def __init__(self, config: ReplayBufferConfig):
        if isinstance(config.capacity, bool) or not isinstance(config.capacity, int) or config.capacity <= 0:
            raise ValueError("ReplayBufferConfig.capacity must be a positive integer")
        if config.sample_packaging not in (
            self.SAMPLE_PACKAGING_INDEPENDENT,
            self.SAMPLE_PACKAGING_EPISODE_GROUPS,
        ):
            raise ReplayPackagingException(
                detail=(
                    "sample_packaging must be 'independent' or 'episode_groups', "
                    f"got {config.sample_packaging!r}"
                )
            )
        # 产物 register/put caller；由 Role 接线，未绑定时跳过。
        self._init_artifact_callers()
        # 回放池配置，包含容量、batch_size、采样策略和近期/滑窗比例。
        self.config = config
        # gRPC线程池会并发执行push/sample/clear；相关容器必须作为一个整体更新。
        self._buffer_lock = threading.RLock()
        # 当前缓存池所属训练代次；TaskManager重启任务后由Role同步。
        # push按代次判断样本归属，采样端不需要额外感知“重启/回滚”等控制细节。
        self.generation = 0
        # 张量和元数据物理解耦：张量池只存按Step拆分的CPU张量，buffer只保存EpisodeMeta。
        self._partition_capacities = self._validate_partition_capacities(config.agent_capacities)
        self.tensor_pool: StepTensorPool = build_step_tensor_pool(config)
        self.buffer: Dict[AgentName, list[EpisodeMeta]] = {}
        self.episode_meta = self.buffer
        self.step_meta: Dict[AgentName, Dict[int, StepMeta]] = {}
        self._next_step_meta_id: Dict[AgentName, int] = {}
        # TensorSchema是共享元数据，StepMeta只引用schema_id，避免每一步重复shape/dtype描述。
        self.tensor_schemas: Dict[int, TensorSchema] = {}
        self._schema_ids_by_key: Dict[Tuple[Any, ...], int] = {}
        self._next_schema_id = 1
        # 兼容快照字段；active 按时间顺序追加，写入后恒为 0。
        self.position: Dict[AgentName, int] = {}
        # 滑动窗口采样游标，只在 SamplingStrategy.SLIDING_WINDOW 下使用。
        self.sliding_idx: Dict[AgentName, int] = {}
        # 每个Agent累计写入episode次数，单调递增，用于ON_POLICY_CURSOR逻辑消费游标。
        self.write_count: Dict[AgentName, int] = {}
        # 与buffer平行保存每个槽位对应的单调写入序号。
        self.sample_seq: Dict[AgentName, list[int]] = {}
        # ON_POLICY_CURSOR已消费到的写入序号；样本不物理删除，只通过该游标避免重复消费。
        self.on_policy_cursor: Dict[AgentName, int] = {}
        self._sample_versions: dict[AgentName, str] = {}
        self._overlap_agents: frozenset[AgentName] = frozenset()
        self._overlap_cursor: dict[AgentName, int] = {}
        self._overlap_reads = 0
        self._sample_end_seq: Optional[int] = None
        self._version_requests: dict[AgentName, tuple[SamplingStrategy, int]] = {}
        self._versioned_episodes: set[tuple[int, AgentName]] = set()
        # ON_POLICY_CURSOR_STEPS已消费到的位置：(episode写入序号, episode内step偏移)。
        self.on_policy_step_cursor: Dict[AgentName, Tuple[int, int]] = {}
        # 最近一次抽样实际消费的局数或步数。
        self.last_sample_stats: Dict[AgentName, Dict[str, Any]] = {}
        # 回放池写入后的状态回调，通常接 TrainTaskManager.learn_batches_update。
        self.learn_batches_update_caller: Optional[Callable[[ReplayBufferState, int], None]] = None
        # 按累计写入局数每 100 局上报一次内存占用；满池后 size 不再涨也继续打。
        self.memory_report_size_interval = 100
        self._last_memory_report_size = 0
        self._monitor_running = True
        self._size_changed_cv = threading.Condition()
        # 容量只约束 active；rehearsal 是额外存档，不挤占配额。
        # stage_name -> agent_name -> 冻结的 EpisodeMeta 列表（只读，不被 active 淘汰）。
        self._rehearsal: Dict[str, Dict[AgentName, list[EpisodeMeta]]] = {}
        # 当前阶段 off-policy 混采比例；由 Stage 切换写入。
        self._rehearsal_sample_ratio = 0.0
        # 阶段收口后的 S3 上传线程；close() 会 join，避免进程退出截断大文件。
        self._storage_upload_threads: List[threading.Thread] = []
        self._storage_upload_lock = threading.Lock()
        self._storage_upload_error: Optional[S3ProtocolException] = None
        self._closed = False
        self.thread = threading.Thread(
            target=self.monitor_thread,
            args=(),
            name="replay_buffer_monitor",
            daemon=True
        )
        self.thread.start()

    @staticmethod
    def _validate_partition_capacities(
        capacities: Mapping[AgentName, int],
    ) -> dict[AgentName, int]:
        resolved: dict[AgentName, int] = {}
        for name, capacity in capacities.items():
            if not isinstance(name, str) or not name:
                raise ValueError("ReplayBuffer AgentName must be a nonempty string")
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
                raise ValueError(f"ReplayBuffer partition {name!r} capacity must be positive")
            resolved[name] = capacity
        return resolved

    def _capacity_for_agent(self, agent_name: AgentName) -> int:
        return self._partition_capacities.get(agent_name, self.config.capacity)

    # ---------- 样本封装/抽样单位 ----------

    @property
    def sample_packaging(self) -> str:
        return self.config.sample_packaging

    def _bundled(self) -> bool:
        """episode_groups 模式：同局多组封成一个张量样本并按该样本抽取。"""
        return self.config.sample_packaging == self.SAMPLE_PACKAGING_EPISODE_GROUPS

    @classmethod
    def _split_bundles(cls, entries: List[EpisodeMeta]) -> List[List[EpisodeMeta]]:
        """把按写入顺序排列的 entries 划分为同局捆绑样本。

        一次原子 push 的同名多条组轨迹连续写入且共享物理 episode_num，
        因此“连续相同 episode_num 的游程”即一个捆绑样本。该规则可从
        buffer/快照原位派生，不需要维护第二套并行索引；缺组槽位就是
        游程中缺失的成员，不参与训练。
        """
        bundles: List[List[EpisodeMeta]] = []
        for meta in entries:
            if bundles and (
                    bundles[-1][0].episode_num, bundles[-1][0].sample_version,
            ) == (meta.episode_num, meta.sample_version):
                bundles[-1].append(meta)
            else:
                bundles.append([meta])
        return bundles

    def _evict_bundle_front_unlocked(
        self,
        agent_name: AgentName,
        agent_buffer: List[EpisodeMeta],
        seq_buffer: List[int],
    ) -> List[EpisodeMeta]:
        """捆绑模式满池时丢掉最旧一个捆绑样本（整局的所有组轨迹）。

        与独立模式的逐条淘汰不同，捆绑淘汰的粒度是物理局；返回被移除的
        EpisodeMeta 列表，由调用方释放底层张量与 StepMeta。
        """
        if not agent_buffer:
            return []
        first_episode = agent_buffer[0].episode_num
        count = 0
        while count < len(agent_buffer) and agent_buffer[count].episode_num == first_episode:
            count += 1
        evicted = agent_buffer[:count]
        del agent_buffer[:count]
        del seq_buffer[:count]
        sliding_idx = self.sliding_idx.get(agent_name)
        if sliding_idx is not None and sliding_idx > 0:
            sliding_idx = max(0, sliding_idx - count)
            self.sliding_idx[agent_name] = (
                (sliding_idx % len(agent_buffer)) if agent_buffer else 0
            )
        return evicted

    def _evict_bundles_to_capacity_unlocked(self, agent_name: AgentName) -> None:
        agent_buffer = self.buffer.get(agent_name)
        if not agent_buffer:
            return
        seq_buffer = self.sample_seq.get(agent_name)
        if seq_buffer is None or len(seq_buffer) != len(agent_buffer):
            seq_buffer = [meta.write_seq for meta in agent_buffer]
            self.sample_seq[agent_name] = seq_buffer
        capacity = self._capacity_for_agent(agent_name)
        while len(self._split_bundles(agent_buffer)) > capacity and agent_buffer:
            for evicted in self._evict_bundle_front_unlocked(agent_name, agent_buffer, seq_buffer):
                self._release_episode(agent_name, evicted)

    def register_stage_intents(self) -> None:
        self.register_artifact_intent(ArtifactType.REPLAY)

    @property
    def rehearsal_enabled(self) -> bool:
        return any(
            entries
            for partitions in self._rehearsal.values()
            for entries in partitions.values()
        )

    def _rehearsal_occupied(self) -> int:
        """各 agent 上 rehearsal 占用的最大 episode 数（不计入 capacity）。"""
        by_agent: Dict[AgentName, int] = {}
        for partitions in self._rehearsal.values():
            for agent_name, entries in partitions.items():
                by_agent[agent_name] = by_agent.get(agent_name, 0) + len(entries)
        return max(by_agent.values(), default=0)

    def _rehearsal_count_for_agent(self, agent_name: AgentName) -> int:
        return sum(
            len(partitions.get(agent_name, []))
            for partitions in self._rehearsal.values()
        )

    def _rehearsal_entries_for_agent(self, agent_name: AgentName) -> List[EpisodeMeta]:
        entries: List[EpisodeMeta] = []
        for partitions in self._rehearsal.values():
            entries.extend(partitions.get(agent_name, []))
        return entries

    def set_rehearsal_sample_ratio(self, ratio: float) -> None:
        """更新当前阶段的 rehearsal 混采比例。"""
        ratio = float(ratio)
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"rehearsal_sample_ratio must be in [0, 1], got {ratio}")
        with self._buffer_lock:
            self._rehearsal_sample_ratio = ratio
            if ratio > 0.0:
                strategy = str(self.config.strategy)
                on_policy_prefixes = (
                    SamplingStrategy.ON_POLICY_RECENT.value,
                    SamplingStrategy.ON_POLICY_CURSOR.value,
                    SamplingStrategy.ON_POLICY_RECENT_STEPS.value,
                    SamplingStrategy.ON_POLICY_CURSOR_STEPS.value,
                )
                if strategy in on_policy_prefixes or strategy.startswith("on_policy"):
                    LOG.warning(
                        "rehearsal_sample_ratio>0 but default strategy is on-policy "
                        f"({strategy}); mixed rehearsal sampling is ignored for on-policy."
                    )

    def reconfigure(self, config: ReplayBufferConfig) -> None:
        """在回放池锁内更新配置：先为 mmap 扩容，再按新配额淘汰 active。"""
        capacities = self._validate_partition_capacities(config.agent_capacities)
        if isinstance(config.capacity, bool) or not isinstance(config.capacity, int) or config.capacity <= 0:
            raise ValueError("ReplayBufferConfig.capacity must be a positive integer")
        with self._buffer_lock:
            if isinstance(self.tensor_pool, LazyMemmapStepTensorPool):
                budget = lazy_memmap_max_steps_of(config)
                live = len(self.tensor_pool._schema_of_index)
                if not self.tensor_pool._columns:
                    self.tensor_pool.configure_empty_budget(budget)
                else:
                    self.tensor_pool.reserve(live + budget)
            self.config = config
            self._partition_capacities = capacities
            for name, entries in self.buffer.items():
                if self._bundled():
                    self._evict_bundles_to_capacity_unlocked(name)
                    continue
                seqs = self.sample_seq[name]
                while len(entries) > self._capacity_for_agent(name):
                    replaced = self._evict_one_unlocked(name, entries, seqs)
                    self._release_episode(name, replaced)

    def set_agent_capacity(self, agent_name: AgentName, capacity: int) -> None:
        """按 AgentName 更新 active 配额；Role 用当前 Learner 算法属性驱动。"""
        if not isinstance(agent_name, str) or not agent_name:
            raise ValueError("ReplayBuffer AgentName must be a nonempty string")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"ReplayBuffer partition {agent_name!r} capacity must be positive")
        with self._buffer_lock:
            if self._partition_capacities.get(agent_name) == capacity:
                return
            config = copy.deepcopy(self.config)
            config.agent_capacities[agent_name] = capacity
            self.reconfigure(config)

    def _evict_one_unlocked(
            self,
            agent_name: AgentName,
            agent_buffer: List[EpisodeMeta],
            seq_buffer: List[int],
    ) -> EpisodeMeta:
        """满池时丢掉最旧一局；列表按写入时间从旧到新排列。"""
        replaced = agent_buffer.pop(0)
        seq_buffer.pop(0)
        sliding_idx = self.sliding_idx.get(agent_name)
        if sliding_idx is not None:
            if sliding_idx > 0:
                sliding_idx -= 1
            buffer_len = len(agent_buffer)
            self.sliding_idx[agent_name] = (sliding_idx % buffer_len) if buffer_len else 0
        return replaced

    @property
    def size(self) -> int:
        """返回可组成合法学习 batch 的等价 episode 容量。"""
        with self._buffer_lock:
            return self._sampleable_size_unlocked()

    def _sampleable_size_unlocked(self) -> int:
        return max(
            (self._sampleable_size_for_agent_unlocked(name) for name in self.agent_names()),
            default=0,
        )

    def _sampleable_size_for_agent_unlocked(self, agent_name: AgentName) -> int:
        ratio = self._rehearsal_sample_ratio
        if self._bundled():
            # 捆绑模式的准入单位是物理局（捆绑样本数）；batch_size 语义同单位。
            active_entries = self.buffer.get(agent_name, [])
            if ratio <= 0.0 or not self.rehearsal_enabled:
                return len(self._split_bundles(active_entries))
            active_size = len(self._split_bundles(active_entries))
            rehearsal_size = len(self._split_bundles(
                self._rehearsal_entries_for_agent(agent_name)
            ))
            capacity = self._capacity_for_agent(agent_name)
            if ratio >= 1.0:
                return min(capacity, rehearsal_size)
            active_equivalent = int(active_size / (1.0 - ratio))
            rehearsal_equivalent = int(rehearsal_size / ratio)
            return min(capacity, active_equivalent, rehearsal_equivalent)
        ratio = self._rehearsal_sample_ratio
        if ratio <= 0.0 or not self.rehearsal_enabled:
            return len(self.buffer.get(agent_name, []))
        active_size = len(self.buffer.get(agent_name, []))
        rehearsal_size = self._rehearsal_count_for_agent(agent_name)
        capacity = self._capacity_for_agent(agent_name)
        if ratio >= 1.0:
            return min(capacity, rehearsal_size)
        active_equivalent = int(active_size / (1.0 - ratio))
        rehearsal_equivalent = int(rehearsal_size / ratio)
        return min(capacity, active_equivalent, rehearsal_equivalent)

    def set_sample_version(
            self, version: str | Mapping[AgentName, str] | None,
            requests: dict[AgentName, tuple[SamplingStrategy, int]], *, reset_episode_keys: bool = False,
            overlap_agents: frozenset[AgentName] = frozenset(),
    ) -> None:
        """Scope online sampling to the policy snapshot being collected."""
        with self._buffer_lock:
            versions = dict.fromkeys(requests, version) if isinstance(version, str) else dict(version or {})
            if not overlap_agents <= {name for name, (strategy, _) in requests.items()
                                      if strategy in ROLLING_STRATEGIES and name in versions}:
                raise ValueError("overlap requires versioned rolling sampling")
            if (not reset_episode_keys and versions == self._sample_versions
                    and requests == self._version_requests and overlap_agents == self._overlap_agents):
                return
            if self._bundled() and any(strategy not in {
                SamplingStrategy.UNIFORM, SamplingStrategy.UNIFORM_STEPS,
                SamplingStrategy.RECENT_BIASED, SamplingStrategy.TIME_SLICE,
            } for strategy, _ in requests.values()):
                raise ReplayPackagingException(detail="unsupported versioned sampling strategy for episode_groups")
            for name, (strategy, batch_size) in requests.items():
                if strategy in {
                    SamplingStrategy.TIME_SLICE, SamplingStrategy.ON_POLICY_CURSOR_STEPS,
                    SamplingStrategy.ON_POLICY_RECENT_STEPS, SamplingStrategy.UNIFORM_STEPS,
                }:
                    continue
                if self._capacity_for_agent(name) < batch_size:
                    raise ValueError(f"replay capacity for {name} is smaller than versioned batch size {batch_size}")
            self._sample_versions = versions
            self._version_requests = dict(requests)
            if reset_episode_keys or not versions:
                self._versioned_episodes.clear()
                self._overlap_cursor.clear()
                self._overlap_reads = 0
            self._overlap_cursor = {
                name: self._overlap_cursor.get(name, self.write_count.get(name, 0))
                for name in overlap_agents
            }
            self._overlap_agents = overlap_agents

    def _sampling_entries(self, agent_name: AgentName) -> List[EpisodeMeta]:
        entries = self.buffer.get(agent_name, [])
        version = self._sample_versions.get(agent_name)
        if version is None:
            return entries
        request = self._version_requests.get(agent_name)
        if request is not None and request[0] in ROLLING_STRATEGIES:
            end = self._sample_end_seq
            return [meta for meta in entries if meta.step_count > 0 and (end is None or meta.write_seq <= end)]
        return [meta for meta in entries if meta.sample_version == version and meta.step_count > 0]

    def _has_current_sample(self, agent_name: AgentName) -> bool:
        version = self._sample_versions.get(agent_name)
        return any(meta.sample_version == version and meta.step_count > 0
                   for meta in self.buffer.get(agent_name, []))

    def _versioned_available(self, agent_name: AgentName, strategy: SamplingStrategy, batch_size: int) -> int:
        entries = self._sampling_entries(agent_name)
        if strategy == SamplingStrategy.UNIFORM_STEPS:
            if self._bundled():
                return sum(max(meta.step_count for meta in bundle) for bundle in self._split_bundles(entries))
            return sum(meta.step_count for meta in entries)
        if strategy == SamplingStrategy.TIME_SLICE:
            if self._bundled():
                return len(self._bundled_time_slice_windows(self._split_bundles(entries), batch_size))
            return len(self._time_slice_windows(entries, batch_size))
        if strategy == SamplingStrategy.ON_POLICY_CURSOR_STEPS:
            return self._available_steps_from_cursor(
                [(meta.write_seq, meta) for meta in entries], self.on_policy_step_cursor.get(agent_name, (0, 0)),
            )
        if strategy == SamplingStrategy.ON_POLICY_RECENT_STEPS:
            return sum(meta.step_count for meta in entries)
        if strategy == SamplingStrategy.ON_POLICY_CURSOR:
            return sum(meta.write_seq > self.on_policy_cursor.get(agent_name, 0) for meta in entries)
        count = len(self._split_bundles(entries)) if self._bundled() else len(entries)
        return count + self._versioned_rehearsal_quota(agent_name, strategy, batch_size)

    def _versioned_rehearsal_quota(self, agent_name: AgentName, strategy: SamplingStrategy, batch_size: int) -> int:
        if (not self.rehearsal_enabled or self._rehearsal_sample_ratio <= 0.0
                or strategy not in {SamplingStrategy.RECENT_BIASED, SamplingStrategy.UNIFORM}):
            return 0
        archive = self._rehearsal_entries_for_agent(agent_name)
        count = len(self._split_bundles(archive)) if self._bundled() else len(archive)
        return self._rehearsal_quota(batch_size, count)

    def _next_overlap_sample_end(self, agent_name: AgentName, batch_size: int) -> Optional[int]:
        """Find the earliest unconsumed rollout that can end a full rolling batch."""
        strategy, _ = self._version_requests[agent_name]
        available = self._versioned_rehearsal_quota(agent_name, strategy, batch_size)
        cursor = self._overlap_cursor[agent_name]
        bundled = self._bundled()
        for rollout in self._split_bundles(self._sampling_entries(agent_name)):
            if strategy == SamplingStrategy.UNIFORM_STEPS:
                if rollout[-1].write_seq > cursor and any(meta.step_count for meta in rollout):
                    return rollout[-1].write_seq
                continue
            if strategy == SamplingStrategy.ON_POLICY_RECENT_STEPS:
                available += sum(meta.step_count for meta in rollout)
            else:
                available += 1 if bundled else len(rollout)
            end = rollout[-1].write_seq
            if end > cursor and available >= batch_size:
                return end
        return None

    def _accept_sample_version(self, agent_name: AgentName, version: Optional[str]) -> bool:
        current = self._sample_versions.get(agent_name)
        if version == current:
            return True
        return (agent_name in self._overlap_agents and version is not None
                and is_older_sample_version(version, current))

    def _rehearsal_quota(self, batch_size: int, available: int) -> int:
        count = max(1, round(batch_size * self._rehearsal_sample_ratio))
        if self._sample_versions:
            # Reserve at least one active slot for the current version.
            return min(count, batch_size - 1, available)
        return min(count, batch_size)

    def _include_current_sample(self, agent_name, selected, population):
        """Reserve a fresh slot when a rolling strategy selected only historical units."""
        version = self._sample_versions.get(agent_name)
        if version is None or not selected:
            return selected
        bundled = self._bundled()
        end = self._sample_end_seq

        def includes_anchor(unit):
            meta = unit[-1] if bundled else unit
            return meta.write_seq == end if end is not None else meta.sample_version == version

        if any(includes_anchor(unit) for unit in selected):
            return selected
        fresh = next(unit for unit in reversed(population) if includes_anchor(unit))
        return [*selected[:-1], fresh]

    def state(self) -> ReplayBufferState:
        """返回回放池当前状态快照，供任务调度或监控模块读取。"""
        with self._buffer_lock:
            return ReplayBufferState(
                generation=self.generation,
                size=self._sampleable_size_unlocked(),
                capacity=sum(
                    self._capacity_for_agent(name) for name in self.agent_names()
                ) or self.config.capacity,
                agent_sizes={
                    name: self._sampleable_size_for_agent_unlocked(name)
                    for name in self.agent_names()
                },
                agent_capacities={
                    name: self._capacity_for_agent(name) for name in self.agent_names()
                },
                position=dict(self.position),
                strategy=self.config.strategy,
                sliding_idx=dict(self.sliding_idx),
                # 容量必须同时报告物理局数、有效组轨迹数和有效步数；
                # 不用同一个裸 size 数值暗示不同模式下的相同数据预算。
                sample_packaging=self.sample_packaging,
                sample_versions=dict(self._sample_versions),
                sample_version=(next(iter(self._sample_versions.values()))
                                if len(self._sample_versions) == 1 else None),
                versioned_writes=len(self._versioned_episodes),
                overlap_reads=self._overlap_reads,
                overlap_ready_agents={
                    name for name in self._overlap_agents
                    if self._next_overlap_sample_end(name, self._version_requests[name][1]) is not None
                },
                versioned_fresh_agents={
                    name for name in self._version_requests
                    if self._has_current_sample(name)
                },
                versioned_available={
                    name: self._versioned_available(name, strategy, batch_size)
                    for name, (strategy, batch_size) in self._version_requests.items()
                },
                agent_episode_counts={
                    name: self._episode_count_for_agent_unlocked(name)
                    for name in self.agent_names()
                },
                agent_trajectory_counts={
                    name: len(self.buffer.get(name, []))
                    for name in self.agent_names()
                },
                agent_step_counts={
                    name: sum(meta.step_count for meta in self.buffer.get(name, []))
                    for name in self.agent_names()
                },
                agent_rehearsal_step_counts={
                    name: sum(meta.step_count for meta in self._rehearsal_entries_for_agent(name))
                    if self._rehearsal_sample_ratio > 0 else 0
                    for name in self.agent_names()
                },
            )

    def _episode_count_for_agent_unlocked(self, agent_name: AgentName) -> int:
        """active 中的物理局数；独立组模式等于组轨迹数（每条轨迹一局）。"""
        entries = self.buffer.get(agent_name, [])
        if not entries:
            return 0
        return len(self._split_bundles(entries))

    def agent_buffer_size(self, agent_name: AgentName) -> int:
        """返回指定Agent回放池中的episode数量。"""
        with self._buffer_lock:
            return len(self.buffer.get(agent_name, []))

    def agent_names(self) -> List[AgentName]:
        """返回当前已经建立回放池的Agent名称（含 rehearsal）。"""
        with self._buffer_lock:
            names = list(dict.fromkeys([
                *self.buffer.keys(),
                *(
                    agent_name
                    for partitions in self._rehearsal.values()
                    for agent_name in partitions
                ),
            ]))
            return names

    def clear(self, generation: Optional[int] = None) -> None:
        """清空 active 环缓存与采样游标；rehearsal 存档默认保留。"""
        with self._buffer_lock:
            if generation is not None:
                self.generation = int(generation)
            cleared = max((len(buffer) for buffer in self.buffer.values()), default=0)
            self._clear_active_unlocked()
        LOG.info(
            f"ReplayBuffer cleared active episodes: removed_episodes≈{cleared}, "
            f"generation={self.generation}, "
            f"rehearsal_occupied={self._rehearsal_occupied()}."
        )
        with self._size_changed_cv:
            self._size_changed_cv.notify()

    def _clear_active_unlocked(self) -> None:
        """只清空 active 环；保留 rehearsal 及其张量/StepMeta。"""
        for agent_name, entries in list(self.buffer.items()):
            for episode_meta in list(entries):
                self._release_episode(agent_name, episode_meta)
        self.buffer.clear()
        self.position.clear()
        self.sliding_idx.clear()
        self.write_count.clear()
        self.sample_seq.clear()
        self.on_policy_cursor.clear()
        self._versioned_episodes.clear()
        self._overlap_cursor = dict.fromkeys(self._overlap_agents, 0)
        self._overlap_reads = 0
        self.on_policy_step_cursor.clear()
        self.last_sample_stats.clear()
        if not self.rehearsal_enabled:
            # 无存档时彻底回收 schema / 张量池，保持历史行为。
            self.step_meta.clear()
            self._next_step_meta_id.clear()
            self.tensor_schemas.clear()
            self._schema_ids_by_key.clear()
            self._next_schema_id = 1
            self.tensor_pool.clear()
        self._last_memory_report_size = 0

    def _clear_all_unlocked(self) -> None:
        """清空 active + rehearsal 及全部底层资源（快照加载前使用）。"""
        for partitions in self._rehearsal.values():
            for agent_name, entries in partitions.items():
                for episode_meta in list(entries):
                    self._release_episode(agent_name, episode_meta)
        self._rehearsal.clear()
        self._clear_active_unlocked()
        self.step_meta.clear()
        self._next_step_meta_id.clear()
        self.tensor_schemas.clear()
        self._schema_ids_by_key.clear()
        self._next_schema_id = 1
        self.tensor_pool.clear()

    def archive_from_active(
            self,
            stage_name: str,
            ratio: float,
            select: str = "even_span",
            *,
            seed: Optional[int] = None,
    ) -> int:
        """
        离开阶段时从 active 环冻结一批 episode 到 rehearsal[stage_name]。

        同名 stage 再次归档会先释放旧 partition 再替换。返回实际冻结的 episode 总数
        （各 agent 之和）。capacity<=1 时跳过（评估池）。
        """
        ratio = float(ratio)
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"rehearsal_archive_ratio must be in [0, 1], got {ratio}")
        select = str(select or "even_span").strip()
        if select not in {"even_span", "newest", "uniform"}:
            raise ValueError(
                f"rehearsal_archive_select must be even_span|newest|uniform, got {select!r}"
            )
        with self._buffer_lock:
            if ratio <= 0.0 or all(
                self._capacity_for_agent(name) <= 1 for name in self.buffer
            ):
                return 0
            # 同名替换：先腾出旧分区占用。
            old_partition = self._rehearsal.pop(stage_name, None)
            if old_partition:
                for agent_name, entries in old_partition.items():
                    for episode_meta in list(entries):
                        self._release_episode(agent_name, episode_meta)

            other_occupied = self._rehearsal_occupied()
            if not any(
                round(self._capacity_for_agent(name) * ratio) > 0
                for name in self.buffer
            ):
                LOG.warning(
                    f"ReplayBuffer archive skipped: stage_name={stage_name}, "
                    f"ratio={ratio}, other_occupied={other_occupied}, "
                    f"agent_capacities={self._partition_capacities}."
                )
                return 0

            rng = random.Random(seed) if seed is not None else random
            archived_total = 0
            new_partition: Dict[AgentName, list[EpisodeMeta]] = {}
            for agent_name in list(self.buffer.keys()):
                ordered = self._ordered_episode_meta_with_seq(agent_name)
                if not ordered:
                    continue
                if self._bundled():
                    # 捆绑模式按物理局整局冻结/保留，归档单位与淘汰单位一致。
                    units: List[List[Tuple[int, EpisodeMeta]]] = []
                    for pair in ordered:
                        if units and units[-1][0][1].episode_num == pair[1].episode_num:
                            units[-1].append(pair)
                        else:
                            units.append([pair])
                else:
                    units = [[pair] for pair in ordered]
                target_keep = max(0, round(self._capacity_for_agent(agent_name) * ratio))
                keep_count = min(target_keep, len(units))
                if keep_count <= 0:
                    continue
                selected_units = self._select_archive_metas(
                    units, keep_count, select=select, rng=rng,
                )
                selected_ids = {id(pair[1]) for unit in selected_units for pair in unit}
                kept_pairs = [
                    (seq, meta) for seq, meta in ordered if id(meta) in selected_ids
                ]
                remaining_pairs = [
                    (seq, meta) for seq, meta in ordered if id(meta) not in selected_ids
                ]
                new_partition[agent_name] = [meta for _, meta in kept_pairs]
                self.buffer[agent_name] = [meta for _, meta in remaining_pairs]
                self.sample_seq[agent_name] = [seq for seq, _ in remaining_pairs]
                self.position[agent_name] = 0
                if not remaining_pairs:
                    self.sliding_idx.pop(agent_name, None)
                    self.on_policy_cursor[agent_name] = 0
                    self.on_policy_step_cursor.pop(agent_name, None)
                archived_total += len(kept_pairs)

            if new_partition:
                self._rehearsal[stage_name] = new_partition
            for agent_name, entries in self.buffer.items():
                self.position[agent_name] = 0
            LOG.info(
                f"ReplayBuffer archived active episodes: stage_name={stage_name}, "
                f"select={select}, ratio={ratio}, archived={archived_total}, "
                f"agent_capacities={self._partition_capacities}, "
                f"rehearsal_occupied={self._rehearsal_occupied()}."
            )
            return archived_total

    def notify_stage_complete(
            self,
            stage: "StageConfig",
            *,
            task_id: str = "",
    ) -> Optional[Path]:
        """阶段正常结束：本地落盘、归档 rehearsal，S3 在本方法内完成推送。

        Role 在本阶段采样/学习序列消费完后先调本方法完成回放池落盘，再把返回路径
        登记给 TrainTaskManager 并进入统一结束点。不清 active、不 close 张量池；
        进入下一阶段的 clear / rehearsal_sample_ratio 仍走 apply_stage_replay_policy。
        进程异常退出只调 close() 清本地 scratch，不走本方法，因此不会上传 S3。
        """
        from config import StorageType
        from sample.replay.replay_persist import (
            default_replay_snapshot_name,
            samples_dir_from_config,
            sync_snapshot_to_storage,
        )

        if stage.rehearsal_archive_ratio > 0.0:
            self.archive_from_active(
                stage.stage_name,
                stage.rehearsal_archive_ratio,
                stage.rehearsal_archive_select,
            )

        # 先归档再保存，确保跳过阶段入口加载的 DRBF 同时包含本阶段 rehearsal。
        snapshot_path: Optional[Path] = None
        storage_is_s3 = StorageType(self.config.storage_type) == StorageType.S3
        if stage.save_replay_buffer:
            samples_dir = samples_dir_from_config(self.config)
            filename = default_replay_snapshot_name(stage.stage_name, task_id)
            snapshot_path = self.save_snapshot(Path(samples_dir) / filename)
            if storage_is_s3:
                self._submit_storage_upload(
                    "snapshot",
                    sync_snapshot_to_storage,
                    snapshot_path,
                    self.config,
                )
        LOG.info(
            f"ReplayBuffer stage complete: stage_name={stage.stage_name}, "
            f"snapshot={snapshot_path}, archive_ratio={stage.rehearsal_archive_ratio}, "
            f"storage_type={self.config.storage_type}"
        )
        self.wait_storage_uploads()
        self.complete_artifact_stage()
        return snapshot_path

    def _submit_storage_upload(self, label: str, func: Callable[..., Any], *args: Any) -> None:
        """把对象存储推送放到后台线程，立刻返回给训练循环。"""
        thread = threading.Thread(
            target=self._run_storage_upload,
            args=(label, func, args),
            name=f"replay-s3-{label}",
            daemon=False,
        )
        thread.start()
        with self._storage_upload_lock:
            alive = [item for item in self._storage_upload_threads if item.is_alive()]
            alive.append(thread)
            self._storage_upload_threads = alive
        LOG.info(f"ReplayBuffer S3 upload scheduled: label={label}, thread={thread.name}")

    def _run_storage_upload(self, label: str, func: Callable[..., Any], args: tuple) -> None:
        try:
            func(*args)
        except S3ProtocolException as exc:
            LOG.exception(f"ReplayBuffer S3 upload failed: label={label}")
            self._set_storage_upload_error(exc)
        except Exception as exc:
            wrapped = S3ProtocolException(operation=label, detail=exc)
            LOG.exception(f"ReplayBuffer S3 upload failed: label={label}")
            self._set_storage_upload_error(wrapped)

    def _set_storage_upload_error(self, error: S3ProtocolException) -> None:
        with self._storage_upload_lock:
            if self._storage_upload_error is None:
                self._storage_upload_error = error

    def wait_storage_uploads(self, timeout: Optional[float] = None) -> None:
        """等待已提交的 S3 上传结束；timeout 为 None 时等到全部完成。"""
        with self._storage_upload_lock:
            threads = list(self._storage_upload_threads)
        for thread in threads:
            thread.join(timeout)
        with self._storage_upload_lock:
            self._storage_upload_threads = [
                thread for thread in self._storage_upload_threads if thread.is_alive()
            ]
            error = self._storage_upload_error
            self._storage_upload_error = None
        if error is not None:
            raise error

    @staticmethod
    def _select_archive_metas(
            ordered: List[EpisodeMeta],
            keep_count: int,
            *,
            select: str,
            rng,
    ) -> List[EpisodeMeta]:
        if keep_count <= 0 or not ordered:
            return []
        keep_count = min(keep_count, len(ordered))
        if select == "newest":
            return list(ordered[-keep_count:])
        if select == "uniform":
            return rng.sample(ordered, keep_count)
        # even_span：等间距铺满整段历史。
        if keep_count == 1:
            return [ordered[0]]
        selected_indices = {
            index * len(ordered) // keep_count
            for index in range(keep_count)
        }
        return [
            episode_meta
            for index, episode_meta in enumerate(ordered)
            if index in selected_indices
        ]

    def set_generation(self, generation: int) -> None:
        """同步当前训练代次；不清空缓存，通常只用于外部明确完成缓存生命周期管理后的补偿设置。"""
        with self._buffer_lock:
            self.generation = int(generation)

    def reset_sliding_idx(self, agent_name: Optional[AgentName] = None) -> None:
        """重置滑动窗口游标；agent_name为空时重置全部Agent。"""
        with self._buffer_lock:
            if agent_name is None:
                self.sliding_idx.clear()
            else:
                self.sliding_idx[agent_name] = 0

    def ordered_buffer(self, agent_name: Optional[AgentName] = None) -> List[TransitionType]:
        """返回按写入时间排序的buffer视图；agent_name为空时取当前最长的Agent池。"""
        with self._buffer_lock:
            if agent_name is None:
                if not self.buffer:
                    return []
                agent_name = max(self.buffer, key=lambda name: len(self.buffer[name]))
            return self._materialize_episode_metas(self._ordered_episode_meta(agent_name))

    @timed("replay.push_total")
    def push(
            self,
            episode_data: Union[
                Dict[AgentName, Union[TransitionType, List[TransitionType]]],
                TransitionType,
                ReplayPushEnvelope,
            ],
            save=False,
            notify_learning: bool = True,
            agent_name: Optional[AgentName] = None,
            sample_role_name: str = "Serial",
            episode_num: Optional[int] = None,
            sample_cost_secs: Optional[float] = None,
            generation: Optional[int] = None,
            sample_version: str | Mapping[AgentName, str] | None = None,
    ):
        """按时间顺序追加；满池后丢掉最旧一局再写入。

        同一 AgentName 的值支持轨迹列表（无名多组一次物理 episode 的原子批量）：
        加锁前规范化为有序条目并预分配连续 write_seq；任一条目失败整批回滚。
        """
        entries = self._normalize_push_entries(episode_data, agent_name)
        entry_agent_names = {name for name, _ in entries}

        saved_episode_data: Optional[Dict[AgentName, Union[TransitionType, List[TransitionType]]]] = None
        with self._buffer_lock:
            if generation is None:
                generation = self.generation
            assert int(generation) == self.generation, (
                f"回放池拒绝过期样本：样本代次={generation}，"
                f"回放池代次={self.generation}，局数={episode_num}，"
                f"采样角色={sample_role_name}。"
            )
            if isinstance(sample_version, str):
                if not self._sample_versions or any(
                    not self._accept_sample_version(name, sample_version) for name in self._sample_versions
                ):
                    raise ValueError(f"replay sample version {sample_version!r} does not match current versions")
                versions = dict.fromkeys(self._sample_versions, sample_version)
            else:
                versions = dict(sample_version or {})
                if versions.keys() != self._sample_versions.keys():
                    raise ValueError("replay sample version keys do not match current learners")
            if versions:
                if episode_num is None:
                    raise ValueError("versioned replay push requires an episode number")
                # Store each partition once, even when retrying the same rollout
                # with another learner's updated model.
                entries = [(name, transition) for name, transition in entries
                           if self._accept_sample_version(name, versions.get(name))
                           and (episode_num, name) not in self._versioned_episodes]
                entry_agent_names = {name for name, _ in entries}
            for name in entry_agent_names:
                if name not in self._partition_capacities:
                    self.set_agent_capacity(name, self.config.capacity)

            if isinstance(self.tensor_pool, LazyMemmapStepTensorPool):
                # 配置预算覆盖常驻数据；暂存峰值取决于本次实际条目数和长度，
                # 必须在 stage 前扩容，不能先淘汰旧轨迹来换取空间。
                incoming_steps = 0
                for _, transition in entries:
                    prepared = (
                        transition.prepared_episode
                        if isinstance(transition, ReplayTransitionEnvelope)
                        else getattr(transition, "prepared_replay_episode", None)
                    )
                    if prepared is not None:
                        self._validate_prepared_episode(prepared)
                        incoming_steps += prepared.step_count
                    else:
                        incoming_steps += self._transition_steps(transition)
                self.tensor_pool.reserve_for_append(incoming_steps)

            # 同一 AgentName 的多条轨迹预分配连续 write_seq：stage 阶段不得
            # 反复读取尚未 commit 的 write_count，否则多条轨迹会得到相同序号。
            next_seq_by_name: Dict[AgentName, int] = {
                name: self.write_count.get(name, 0)
                for name in entry_agent_names
            }
            staged: List[Tuple[AgentName, int, EpisodeMeta]] = []
            next_step_meta_before = dict(self._next_step_meta_id)
            next_schema_id_before = self._next_schema_id
            try:
                for name, transition in entries:
                    next_seq_by_name[name] += 1
                    seq = next_seq_by_name[name]
                    if isinstance(transition, ReplayTransitionEnvelope):
                        episode_meta = self._store_transition_envelope(
                            agent_name=name,
                            envelope=transition,
                            write_seq=seq,
                        )
                    else:
                        episode_meta = self._store_transition(
                            agent_name=name,
                            transition=transition,
                            write_seq=seq,
                        )
                    episode_meta.sample_version = versions.get(name)
                    staged.append((name, seq, episode_meta))
                if save:
                    # 暂存条目可能在本次 commit 就被容量淘汰；在淘汰前复制完整
                    # 明细，按同名列表保存所有组，避免字典覆盖或读取已释放行。
                    saved_groups: Dict[AgentName, List[TransitionType]] = {}
                    for name, _, episode_meta in staged:
                        saved_groups.setdefault(name, []).append(
                            self._materialize_episode(episode_meta)
                        )
                    saved_episode_data = {
                        name: items[0] if len(items) == 1 else items
                        for name, items in saved_groups.items()
                    }
            except Exception:
                for staged_name, _, staged_meta in staged:
                    self._release_episode(staged_name, staged_meta)
                self._next_step_meta_id = next_step_meta_before
                for schema_id in list(self.tensor_schemas):
                    if schema_id >= next_schema_id_before:
                        self.tensor_schemas.pop(schema_id)
                self._schema_ids_by_key = {
                    (schema.template, schema.fields): schema_id
                    for schema_id, schema in self.tensor_schemas.items()
                }
                self._next_schema_id = next_schema_id_before
                raise

            if self._bundled():
                # 捆绑模式：同一次 push 的同名条目是一个新捆绑样本，淘汰和
                # 写入都按物理局粒度进行；容量单位是捆绑样本数。
                staged_by_name: Dict[AgentName, List[Tuple[AgentName, int, EpisodeMeta]]] = {}
                for item in staged:
                    staged_by_name.setdefault(item[0], []).append(item)
                for name, items in staged_by_name.items():
                    agent_buffer = self.buffer.setdefault(name, [])
                    seq_buffer = self.sample_seq.setdefault(name, [])
                    while (
                        len(self._split_bundles(agent_buffer))
                        >= self._capacity_for_agent(name)
                        and agent_buffer
                    ):
                        for evicted in self._evict_bundle_front_unlocked(
                            name, agent_buffer, seq_buffer,
                        ):
                            self._release_episode(name, evicted)
                    for _, seq, episode_meta in items:
                        agent_buffer.append(episode_meta)
                        seq_buffer.append(seq)
                        self.write_count[name] = seq
                    self.position[name] = 0
            else:
                for name, seq, episode_meta in staged:
                    agent_buffer = self.buffer.setdefault(name, [])
                    seq_buffer = self.sample_seq.setdefault(name, [])
                    replaced_meta: Optional[EpisodeMeta] = None
                    if len(agent_buffer) >= self._capacity_for_agent(name):
                        replaced_meta = self._evict_one_unlocked(
                            name, agent_buffer, seq_buffer
                        )
                    agent_buffer.append(episode_meta)
                    seq_buffer.append(seq)
                    self.write_count[name] = seq
                    self.position[name] = 0
                    if replaced_meta is not None:
                        self._release_episode(name, replaced_meta)
            if versions:
                self._versioned_episodes.update((episode_num, name) for name in entry_agent_names)
            replay_state = self.state()
        # 持久化保存缓存数据
        if save:
            self.save_buffer(saved_episode_data)
        # 通知任务模块读取ReplayBuffer状态，由任务模块决定是否发放学习次数。
        if notify_learning and self.learn_batches_update_caller is not None:
            assert episode_num is not None, (
                "episode_num is required when ReplayBuffer.push notifies online learning."
            )
            self.learn_batches_update_caller(replay_state, episode_num)
        LOG.info(
            f"ReplayBuffer pushed sample: sample_role={sample_role_name}, "
            f"episode_num={episode_num}, generation={generation}, sample_cost_secs={sample_cost_secs}, "
            f"agent_names={sorted(entry_agent_names)}, entries={len(staged)}, size={replay_state.size}"
        )
        with self._size_changed_cv:
            self._size_changed_cv.notify()

    def _normalize_push_entries(
            self,
            episode_data: Any,
            agent_name: Optional[AgentName],
    ) -> List[Tuple[AgentName, Any]]:
        """把全部 push 入口规范化为有序 (AgentName, Transition) 条目。

        顺序为外层 AgentName 插入顺序、内层轨迹列表顺序；ReplayPushEnvelope
        支持 v1（AgentName 键）与 v2（ReplayPushEntryKey 多轨迹键）。
        """
        if isinstance(episode_data, ReplayPushEnvelope):
            if episode_data.version not in ReplayPushEnvelope.SUPPORTED_VERSIONS:
                raise ValueError(
                    f"unsupported ReplayPushEnvelope version: {episode_data.version}"
                )
            entries: List[Tuple[AgentName, Any]] = []
            for key, value in episode_data.episodes.items():
                if isinstance(key, ReplayPushEntryKey):
                    entries.append((key.agent_name, value))
                else:
                    entries.append((key, value))
            return entries
        if isinstance(episode_data, dict):
            entries = []
            for name, value in episode_data.items():
                if isinstance(value, list):
                    entries.extend((name, item) for item in value)
                else:
                    entries.append((name, value))
            return entries
        if agent_name is not None:
            return [(agent_name, episode_data)]
        raise TypeError(
            "episode_data must be a Transition, agent Transition dict, "
            "or ReplayPushEnvelope."
        )

    def _store_transition(
            self,
            agent_name: AgentName,
            transition: TransitionType,
            write_seq: int,
    ) -> EpisodeMeta:
        """接收Transition预制列式载荷，写Step张量池并建立两级元数据索引。"""
        reserved_tensor_indices: List[int] = []
        prepared: Optional[PreparedReplayEpisode] = getattr(
            transition, "prepared_replay_episode", None
        )

        if prepared is not None:
            schema_id = self._register_tensor_schema(
                prepared.template,
                prepared.fields,
            )
            tensor_indices = self.tensor_pool.append_batch(
                schema_id,
                prepared.tensors,
                ownership=TensorOwnership.BORROWED,
            )
            reserved_tensor_indices.extend(tensor_indices)
            encoded_steps = [
                (schema_id, {}, inline_values)
                for inline_values in prepared.inline_rows
            ]
            step_count = prepared.step_count
            step_field_layout = prepared.step_field_layout
            static_fields = prepared.static_fields
        else:
            step_payloads, step_field_layout, static_fields = self._split_transition(transition)
            step_count = len(step_payloads)
            try:
                schema_id, batch_tensors, inline_rows = self._encode_step_payloads_batch(
                    step_payloads
                )
                tensor_indices = self.tensor_pool.append_batch(schema_id, batch_tensors)
                reserved_tensor_indices.extend(tensor_indices)
                encoded_steps = [
                    (schema_id, {}, inline_values)
                    for inline_values in inline_rows
                ]
            except _BatchEncodingUnsupported:
                encoded_steps = [self._encode_step_payload(payload) for payload in step_payloads]
                tensor_indices = [-1] * len(encoded_steps)
                encoded_groups: Dict[int, List[Tuple[int, Dict[str, torch.Tensor]]]] = {}
                for step_offset, (schema_id, tensors, _) in enumerate(encoded_steps):
                    encoded_groups.setdefault(schema_id, []).append((step_offset, tensors))
                try:
                    for schema_id, group in encoded_groups.items():
                        keys = tuple(group[0][1])
                        batch_tensors = {
                            key: torch.stack([tensors[key] for _, tensors in group], dim=0)
                            for key in keys
                        }
                        group_indices = self.tensor_pool.append_batch(schema_id, batch_tensors)
                        reserved_tensor_indices.extend(group_indices)
                        for (step_offset, _), tensor_index in zip(group, group_indices):
                            tensor_indices[step_offset] = tensor_index
                except Exception:
                    self.tensor_pool.release(reserved_tensor_indices)
                    raise
            except Exception:
                self.tensor_pool.release(reserved_tensor_indices)
                raise

        step_meta_store = self.step_meta.setdefault(agent_name, {})
        step_meta_start = self._next_step_meta_id.get(agent_name, 0)
        try:
            for step_offset, ((schema_id, _, inline_values), tensor_index) in enumerate(
                    zip(encoded_steps, tensor_indices)
            ):
                step_meta_id = step_meta_start + step_offset
                step_meta_store[step_meta_id] = StepMeta(
                    tensor_index=tensor_index,
                    schema_id=schema_id,
                    episode_seq=write_seq,
                    step_offset=step_offset,
                    inline_values=inline_values,
                    is_episode_start=step_offset == 0,
                    is_episode_end=step_offset == step_count - 1,
                )
            self._next_step_meta_id[agent_name] = step_meta_start + step_count
        except Exception:
            for step_meta_id in range(step_meta_start, step_meta_start + step_count):
                step_meta_store.pop(step_meta_id, None)
            self.tensor_pool.release(reserved_tensor_indices)
            raise

        return EpisodeMeta(
            agent_name=agent_name,
            transition_type=type(transition),
            episode_num=int(transition.episode_num),
            write_seq=write_seq,
            step_meta_start=step_meta_start,
            step_meta_end=step_meta_start + step_count,
            step_count=step_count,
            step_field_layout=step_field_layout,
            static_fields=static_fields,
            starts_with_episode=bool(transition.starts_with_episode),
            has_starts_with_episode=True,
        )

    @timed("replay.envelope_store")
    def _store_transition_envelope(
            self,
            agent_name: AgentName,
            envelope: ReplayTransitionEnvelope,
            write_seq: int,
    ) -> EpisodeMeta:
        """校验并写入RPC精简信封，合法CPU连续张量可由内存池直接接管。"""
        prepared = envelope.prepared_episode
        self._validate_prepared_episode(prepared)
        schema_id = self._register_tensor_schema(prepared.template, prepared.fields)
        tensor_indices: List[int] = []
        step_meta_store = self.step_meta.setdefault(agent_name, {})
        step_meta_start = self._next_step_meta_id.get(agent_name, 0)
        try:
            tensor_indices = self.tensor_pool.append_batch(
                schema_id,
                prepared.tensors,
                ownership=envelope.tensor_ownership,
            )
            if len(tensor_indices) != prepared.step_count:
                raise ValueError(
                    "prepared replay tensor batch length does not match step_count"
                )
            for step_offset, (tensor_index, inline_values) in enumerate(
                    zip(tensor_indices, prepared.inline_rows)
            ):
                step_meta_id = step_meta_start + step_offset
                step_meta_store[step_meta_id] = StepMeta(
                    tensor_index=tensor_index,
                    schema_id=schema_id,
                    episode_seq=write_seq,
                    step_offset=step_offset,
                    inline_values=inline_values,
                    is_episode_start=step_offset == 0,
                    is_episode_end=step_offset == prepared.step_count - 1,
                )
            self._next_step_meta_id[agent_name] = (
                step_meta_start + prepared.step_count
            )
        except Exception:
            for step_meta_id in range(
                    step_meta_start,
                    step_meta_start + prepared.step_count,
            ):
                step_meta_store.pop(step_meta_id, None)
            self.tensor_pool.release(tensor_indices)
            raise

        return EpisodeMeta(
            agent_name=agent_name,
            transition_type=envelope.transition_type,
            episode_num=envelope.episode_num,
            write_seq=write_seq,
            step_meta_start=step_meta_start,
            step_meta_end=step_meta_start + prepared.step_count,
            step_count=prepared.step_count,
            step_field_layout=prepared.step_field_layout,
            static_fields=prepared.static_fields,
            starts_with_episode=envelope.starts_with_episode,
            has_starts_with_episode=envelope.has_starts_with_episode,
        )

    @staticmethod
    def _validate_prepared_episode(prepared: PreparedReplayEpisode) -> None:
        if not isinstance(prepared, PreparedReplayEpisode):
            raise TypeError("prepared_episode must be PreparedReplayEpisode")
        if prepared.step_count < 0:
            raise ValueError("prepared replay step_count must not be negative")
        if len(prepared.inline_rows) != prepared.step_count:
            raise ValueError(
                "prepared replay inline_rows length does not match step_count"
            )
        for key, tensor in prepared.tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"prepared replay field {key!r} is not a Tensor")
            if tensor.ndim == 0 or tensor.shape[0] != prepared.step_count:
                raise ValueError(
                    f"prepared replay field {key!r} length does not match step_count"
                )

    def _split_transition(
            self,
            transition: TransitionType,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[str, Tuple[Any, ...]]], Dict[str, Any]]:
        """把Transition拆成逐Step业务值；此阶段尚未把值编码成纯张量。"""
        batch_exporter = getattr(transition, "to_replay_step_payloads", None)
        if callable(batch_exporter):
            step_payloads, step_field_layout, static_fields = batch_exporter()
            if step_payloads:
                tensor_static_fields = {
                    field_name: value
                    for field_name, value in static_fields.items()
                    if self._contains_tensor_data(value)
                }
                if tensor_static_fields:
                    step_payloads[0]["episode_tensor_fields"] = tensor_static_fields
                    for field_name in tensor_static_fields:
                        del static_fields[field_name]
            return step_payloads, step_field_layout, static_fields

        seq_len = self._transition_steps(transition)
        step_payloads: List[Dict[str, Any]] = [{"fields": {}} for _ in range(seq_len)]
        step_field_layout: Dict[str, Tuple[str, Tuple[Any, ...]]] = {}

        if isinstance(transition, SAEpisodeTransition):
            known_list_fields = {
                "obs", "actions", "last_actions", "rewards", "next_obs",
                "terminations", "truncations", "avail_actions", "old_log_probs",
                "old_values", "metadata",
            }
            known_dict_fields: set[str] = set()
        elif isinstance(transition, EpisodeTransition):
            known_list_fields = {
                "states", "next_states", "metadata", "step_rewards", "avail_actions_next",
            }
            known_dict_fields = {
                "hidden_states", "obs", "actions", "last_actions", "rewards", "next_obs",
                "terminations", "truncations", "active_agent_mask", "next_active_agent_mask",
            }
        else:
            raise TypeError(f"unsupported transition type: {type(transition).__name__}")

        static_fields: Dict[str, Any] = {}
        attrs = vars(transition)
        for field_name, value in attrs.items():
            if field_name in {"seq_len", "episode_num", "starts_with_episode", "hidden_in", "hidden_out"}:
                continue

            is_list_field = field_name in known_list_fields or (
                isinstance(value, list) and len(value) == seq_len
            )
            is_dict_field = field_name in known_dict_fields or (
                isinstance(value, dict)
                and bool(value)
                and all(isinstance(items, list) and len(items) == seq_len for items in value.values())
            )

            if is_list_field and isinstance(value, list):
                step_field_layout[field_name] = ("list", ())
                for step_offset in range(min(seq_len, len(value))):
                    step_payloads[step_offset]["fields"][field_name] = value[step_offset]
                continue

            if is_dict_field and isinstance(value, dict):
                field_keys = tuple(value.keys())
                step_field_layout[field_name] = ("dict_list", field_keys)
                for step_offset in range(seq_len):
                    step_payloads[step_offset]["fields"][field_name] = {
                        key: items[step_offset]
                        for key, items in value.items()
                        if step_offset < len(items)
                    }
                continue

            static_fields[field_name] = copy.deepcopy(value)

        if seq_len > 0:
            if hasattr(transition, "hidden_in"):
                step_payloads[0]["hidden_in"] = getattr(transition, "hidden_in")
            if hasattr(transition, "hidden_out"):
                step_payloads[-1]["hidden_out"] = getattr(transition, "hidden_out")
            tensor_static_fields = {
                field_name: value
                for field_name, value in static_fields.items()
                if self._contains_tensor_data(value)
            }
            if tensor_static_fields:
                step_payloads[0]["episode_tensor_fields"] = tensor_static_fields
                for field_name in tensor_static_fields:
                    del static_fields[field_name]
        else:
            # 空Episode没有可绑定的边界Step；正常业务不会携带非空隐藏状态，仍保持兼容。
            if hasattr(transition, "hidden_in"):
                static_fields["hidden_in"] = copy.deepcopy(getattr(transition, "hidden_in"))
            if hasattr(transition, "hidden_out"):
                static_fields["hidden_out"] = copy.deepcopy(getattr(transition, "hidden_out"))

        return step_payloads, step_field_layout, static_fields

    @staticmethod
    def _contains_tensor_data(value: Any) -> bool:
        return contains_tensor_data(value)

    @timed("replay.schema_register")
    def _register_tensor_schema(
            self,
            template: Any,
            fields: Tuple[TensorFieldSpec, ...],
    ) -> int:
        """登记Transition预制的共享恢复模板，StepMeta只保存schema_id。"""
        schema_key = (template, fields)
        schema_id = self._schema_ids_by_key.get(schema_key)
        if schema_id is None:
            schema_id = self._next_schema_id
            self._next_schema_id += 1
            self.tensor_schemas[schema_id] = TensorSchema(
                schema_id=schema_id,
                template=template,
                fields=fields,
            )
            self._schema_ids_by_key[schema_key] = schema_id
        return schema_id

    def _encode_step_payloads_batch(
            self,
            payloads: List[Dict[str, Any]],
    ) -> Tuple[int, Dict[str, torch.Tensor], List[Tuple[Any, ...]]]:
        """把一个Episode的同构Step一次性编码为列式张量。"""
        if not payloads:
            return 0, {}, []

        tensors: Dict[str, torch.Tensor] = {}
        field_specs: List[TensorFieldSpec] = []
        inline_rows: List[List[Any]] = [[] for _ in payloads]

        def encode(values: List[Any]) -> Any:
            field_key = f"v{len(field_specs)}"
            tensor_value = self._values_to_storage_tensor(values, field_key)
            if tensor_value is not None:
                tensor, spec = tensor_value
                tensors[spec.key] = tensor
                field_specs.append(spec)
                return _TensorRef(spec.key)

            present = [value is not None for value in values]
            if any(present) and not all(present):
                exemplar = next(value for value in values if value is not None)
                mask_key = f"v{len(field_specs)}"
                tensors[mask_key] = torch.tensor(present, dtype=torch.bool)
                field_specs.append(TensorFieldSpec(
                    key=mask_key,
                    source_kind="python_bool",
                    source_dtype="bool",
                    shape=(),
                ))
                filled = [
                    value if value is not None else self._empty_value_like(exemplar)
                    for value in values
                ]
                return _OptionalTemplate(
                    present=_TensorRef(mask_key),
                    value=encode(filled),
                )

            first = values[0]
            if all(isinstance(value, dict) for value in values):
                keys = tuple(first)
                if any(tuple(value) != keys for value in values):
                    if any(self._contains_tensor_data(value) for value in values):
                        raise _BatchEncodingUnsupported("Step dict layouts are inconsistent.")
                else:
                    return _ContainerTemplate(
                        kind="dict",
                        items=tuple(
                            (copy.deepcopy(key), encode([value[key] for value in values]))
                            for key in keys
                        ),
                    )
            if (
                    isinstance(first, tuple)
                    and hasattr(first, "_asdict")
                    and all(type(value) is type(first) for value in values)
            ):
                keys = tuple(first._asdict())
                return _ContainerTemplate(
                    kind="namedtuple",
                    items=tuple(
                        (key, encode([value._asdict()[key] for value in values]))
                        for key in keys
                    ),
                    module=first.__class__.__module__,
                    class_name=first.__class__.__name__,
                )
            if isinstance(first, list) and all(isinstance(value, list) for value in values):
                item_count = len(first)
                if all(len(value) == item_count for value in values):
                    return _ContainerTemplate(
                        kind="list",
                        items=tuple(
                            encode([value[index] for value in values])
                            for index in range(item_count)
                        ),
                    )
            if (
                    isinstance(first, tuple)
                    and all(type(value) is tuple and len(value) == len(first) for value in values)
            ):
                return _ContainerTemplate(
                    kind="tuple",
                    items=tuple(
                        encode([value[index] for value in values])
                        for index in range(len(first))
                    ),
                )
            if hasattr(first, "__dict__") and all(type(value) is type(first) for value in values):
                keys = tuple(vars(first))
                if all(tuple(vars(value)) == keys for value in values):
                    return _ContainerTemplate(
                        kind="custom_object",
                        items=tuple(
                            (key, encode([vars(value)[key] for value in values]))
                            for key in keys
                        ),
                        module=first.__class__.__module__,
                        class_name=first.__class__.__name__,
                    )
            if hasattr(first, "__slots__") and all(type(value) is type(first) for value in values):
                slots = tuple(
                    slot for slot in first.__slots__
                    if all(hasattr(value, slot) for value in values)
                )
                return _ContainerTemplate(
                    kind="slots_object",
                    items=tuple(
                        (slot, encode([getattr(value, slot) for value in values]))
                        for slot in slots
                    ),
                    module=first.__class__.__module__,
                    class_name=first.__class__.__name__,
                )

            if any(self._contains_tensor_data(value) for value in values):
                raise _BatchEncodingUnsupported(
                    "heterogeneous Step values containing tensors require the compatibility encoder."
                )
            inline_index = len(inline_rows[0])
            for row, value in zip(inline_rows, values):
                row.append(copy.deepcopy(value))
            return _InlineRef(inline_index)

        template = encode(payloads)
        fields = tuple(field_specs)
        schema_key = (template, fields)
        schema_id = self._schema_ids_by_key.get(schema_key)
        if schema_id is None:
            schema_id = self._next_schema_id
            self._next_schema_id += 1
            self.tensor_schemas[schema_id] = TensorSchema(
                schema_id=schema_id,
                template=template,
                fields=fields,
            )
            self._schema_ids_by_key[schema_key] = schema_id
        return schema_id, tensors, [tuple(row) for row in inline_rows]

    @staticmethod
    def _empty_value_like(value: Any) -> Any:
        """为可选字段构造仅用于列式占位的同构零值。"""
        if isinstance(value, torch.Tensor):
            return torch.zeros_like(value)
        if isinstance(value, np.ndarray):
            return np.zeros_like(value)
        if isinstance(value, np.generic):
            return type(value)(0)
        if type(value) is bool:
            return False
        if type(value) is int:
            return 0
        if type(value) is float:
            return 0.0
        if type(value) is complex:
            return 0j
        if isinstance(value, dict):
            return {
                copy.deepcopy(key): ReplayBuffer._empty_value_like(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [ReplayBuffer._empty_value_like(item) for item in value]
        if isinstance(value, tuple):
            return tuple(ReplayBuffer._empty_value_like(item) for item in value)
        return copy.deepcopy(value)

    @staticmethod
    def _values_to_storage_tensor(
            values: List[Any],
            field_key: str,
    ) -> Optional[Tuple[torch.Tensor, TensorFieldSpec]]:
        """把同一字段的一整局Step值堆叠成一个张量列。"""
        first = values[0]
        if isinstance(first, torch.Tensor) and all(isinstance(value, torch.Tensor) for value in values):
            if any(
                    value.layout != torch.strided
                    or value.dtype != first.dtype
                    or tuple(value.shape) != tuple(first.shape)
                    or value.device != first.device
                    for value in values
            ):
                return None
            tensor = torch.stack(
                [value.detach().to("cpu").contiguous() for value in values],
                dim=0,
            )
            return tensor, TensorFieldSpec(
                key=field_key,
                source_kind="torch",
                source_dtype=str(first.dtype),
                shape=tuple(first.shape),
                source_device=str(first.device),
                requires_grad=bool(first.requires_grad),
            )
        if isinstance(first, np.ndarray) and all(isinstance(value, np.ndarray) for value in values):
            if any(
                    value.dtype != first.dtype
                    or tuple(value.shape) != tuple(first.shape)
                    or value.dtype.hasobject
                    for value in values
            ):
                return None
            array = np.stack(values)
            try:
                tensor = torch.from_numpy(np.ascontiguousarray(array))
                source_kind = "ndarray"
            except (TypeError, ValueError):
                return None
            return tensor, TensorFieldSpec(
                key=field_key,
                source_kind=source_kind,
                source_dtype=str(first.dtype),
                shape=tuple(first.shape),
            )
        if isinstance(first, np.generic) and all(
                isinstance(value, np.generic) and value.dtype == first.dtype
                for value in values
        ):
            array = np.asarray(values)
            try:
                tensor = torch.from_numpy(np.ascontiguousarray(array))
            except (TypeError, ValueError):
                return None
            return tensor, TensorFieldSpec(
                key=field_key,
                source_kind="numpy_scalar",
                source_dtype=str(first.dtype),
                shape=(),
            )
        if type(first) is bool and all(type(value) is bool for value in values):
            return torch.tensor(values, dtype=torch.bool), TensorFieldSpec(
                key=field_key, source_kind="python_bool", source_dtype="bool", shape=()
            )
        if type(first) is int and all(type(value) is int for value in values):
            try:
                tensor = torch.tensor(values, dtype=torch.int64)
            except (RuntimeError, OverflowError):
                return None
            return tensor, TensorFieldSpec(
                key=field_key, source_kind="python_int", source_dtype="int", shape=()
            )
        if type(first) is float and all(type(value) is float for value in values):
            return torch.tensor(values, dtype=torch.float64), TensorFieldSpec(
                key=field_key, source_kind="python_float", source_dtype="float", shape=()
            )
        if type(first) is complex and all(type(value) is complex for value in values):
            return torch.tensor(values, dtype=torch.complex128), TensorFieldSpec(
                key=field_key, source_kind="python_complex", source_dtype="complex", shape=()
            )
        if (
                isinstance(first, list)
                and all(
                    isinstance(value, list) and ReplayBuffer._is_numeric_sequence(value)
                    for value in values
                )
        ):
            array = np.asarray(values)
            if array.dtype.hasobject or array.shape[0] != len(values):
                return None
            try:
                tensor = torch.from_numpy(np.ascontiguousarray(array))
            except (TypeError, ValueError):
                return None
            return tensor, TensorFieldSpec(
                key=field_key,
                source_kind="numeric_list",
                source_dtype=str(array.dtype),
                shape=tuple(array.shape[1:]),
            )
        return None

    def _encode_step_payload(
            self,
            payload: Dict[str, Any],
    ) -> Tuple[int, Dict[str, torch.Tensor], Tuple[Any, ...]]:
        """把一个Step递归拆成纯张量记录和可复用Schema。"""
        tensors: Dict[str, torch.Tensor] = {}
        field_specs: List[TensorFieldSpec] = []
        inline_values: List[Any] = []

        def encode(value: Any) -> Any:
            tensor_value = self._value_to_storage_tensor(value, f"v{len(field_specs)}")
            if tensor_value is not None:
                tensor, spec = tensor_value
                tensors[spec.key] = tensor
                field_specs.append(spec)
                return _TensorRef(spec.key)

            if isinstance(value, dict):
                return _ContainerTemplate(
                    kind="dict",
                    items=tuple((copy.deepcopy(key), encode(item)) for key, item in value.items()),
                )
            if hasattr(value, "_asdict") and isinstance(value, tuple):
                return _ContainerTemplate(
                    kind="namedtuple",
                    items=tuple((key, encode(item)) for key, item in value._asdict().items()),
                    module=value.__class__.__module__,
                    class_name=value.__class__.__name__,
                )
            if isinstance(value, list):
                return _ContainerTemplate(kind="list", items=tuple(encode(item) for item in value))
            if isinstance(value, tuple):
                return _ContainerTemplate(kind="tuple", items=tuple(encode(item) for item in value))
            if hasattr(value, "__dict__"):
                return _ContainerTemplate(
                    kind="custom_object",
                    items=tuple((key, encode(item)) for key, item in vars(value).items()),
                    module=value.__class__.__module__,
                    class_name=value.__class__.__name__,
                )
            if hasattr(value, "__slots__"):
                return _ContainerTemplate(
                    kind="slots_object",
                    items=tuple(
                        (slot, encode(getattr(value, slot)))
                        for slot in value.__slots__
                        if hasattr(value, slot)
                    ),
                    module=value.__class__.__module__,
                    class_name=value.__class__.__name__,
                )

            inline_index = len(inline_values)
            inline_values.append(copy.deepcopy(value))
            return _InlineRef(inline_index)

        template = encode(payload)
        fields = tuple(field_specs)
        schema_key = (template, fields)
        schema_id = self._schema_ids_by_key.get(schema_key)
        if schema_id is None:
            schema_id = self._next_schema_id
            self._next_schema_id += 1
            schema = TensorSchema(schema_id=schema_id, template=template, fields=fields)
            self.tensor_schemas[schema_id] = schema
            self._schema_ids_by_key[schema_key] = schema_id
        return schema_id, tensors, tuple(inline_values)

    @staticmethod
    def _is_numeric_sequence(value: list | tuple) -> bool:
        if not value:
            return True
        for item in value:
            if isinstance(item, (list, tuple)):
                if not ReplayBuffer._is_numeric_sequence(item):
                    return False
            elif not isinstance(item, (bool, int, float, complex, np.generic)):
                return False
        return True

    @staticmethod
    def _value_to_storage_tensor(
            value: Any,
            field_key: str,
    ) -> Optional[Tuple[torch.Tensor, TensorFieldSpec]]:
        """把可无损恢复的数值叶子编码成CPU张量；其他值留在StepMeta。"""
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                return None
            tensor = value.detach().to("cpu").contiguous().clone()
            return tensor, TensorFieldSpec(
                key=field_key,
                source_kind="torch",
                source_dtype=str(value.dtype),
                shape=tuple(value.shape),
                source_device=str(value.device),
                requires_grad=bool(value.requires_grad),
            )

        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            if array.dtype.hasobject:
                return None
            try:
                tensor = torch.from_numpy(array).clone()
                source_kind = "ndarray"
            except (TypeError, ValueError):
                tensor = torch.frombuffer(bytearray(array.tobytes()), dtype=torch.uint8).clone()
                source_kind = "ndarray_bytes"
            return tensor, TensorFieldSpec(
                key=field_key,
                source_kind=source_kind,
                source_dtype=str(array.dtype),
                shape=tuple(array.shape),
            )

        if isinstance(value, np.generic):
            array = np.asarray(value)
            try:
                tensor = torch.from_numpy(array.copy())
                source_kind = "numpy_scalar"
            except (TypeError, ValueError):
                tensor = torch.frombuffer(bytearray(array.tobytes()), dtype=torch.uint8).clone()
                source_kind = "numpy_scalar_bytes"
            return tensor, TensorFieldSpec(
                key=field_key,
                source_kind=source_kind,
                source_dtype=str(array.dtype),
                shape=tuple(array.shape),
            )

        if type(value) is bool:
            return torch.tensor(value, dtype=torch.bool), TensorFieldSpec(
                key=field_key, source_kind="python_bool", source_dtype="bool", shape=()
            )
        if type(value) is int:
            try:
                tensor = torch.tensor(value, dtype=torch.int64)
            except (RuntimeError, OverflowError):
                return None
            return tensor, TensorFieldSpec(
                key=field_key, source_kind="python_int", source_dtype="int", shape=()
            )
        if type(value) is float:
            return torch.tensor(value, dtype=torch.float64), TensorFieldSpec(
                key=field_key, source_kind="python_float", source_dtype="float", shape=()
            )
        if type(value) is complex:
            return torch.tensor(value, dtype=torch.complex128), TensorFieldSpec(
                key=field_key, source_kind="python_complex", source_dtype="complex", shape=()
            )

        if isinstance(value, list) and ReplayBuffer._is_numeric_sequence(value):
            array = np.asarray(value)
            if array.dtype.hasobject:
                return None
            try:
                tensor = torch.from_numpy(np.ascontiguousarray(array)).clone()
                source_kind = "numeric_list"
            except (TypeError, ValueError):
                return None
            return tensor, TensorFieldSpec(
                key=field_key,
                source_kind=source_kind,
                source_dtype=str(array.dtype),
                shape=tuple(array.shape),
            )
        return None

    def _decode_step_payload(
            self,
            step_meta: StepMeta,
            tensor_record: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        schema = self.tensor_schemas[step_meta.schema_id]
        if tensor_record is None:
            tensor_record = self.tensor_pool.get([step_meta.tensor_index])[0]
        field_specs = {field.key: field for field in schema.fields}

        def decode(template: Any) -> Any:
            if isinstance(template, _TensorRef):
                return self._restore_tensor_value(tensor_record[template.key], field_specs[template.key])
            if isinstance(template, _InlineRef):
                return copy.deepcopy(step_meta.inline_values[template.index])
            if isinstance(template, _OptionalTemplate):
                present = bool(
                    self._restore_tensor_value(
                        tensor_record[template.present.key],
                        field_specs[template.present.key],
                    )
                )
                return decode(template.value) if present else None
            if isinstance(template, _ContainerTemplate):
                if template.kind == "dict":
                    return {copy.deepcopy(key): decode(item) for key, item in template.items}
                if template.kind == "list":
                    return [decode(item) for item in template.items]
                if template.kind == "tuple":
                    return tuple(decode(item) for item in template.items)
                if template.kind == "namedtuple":
                    module = importlib.import_module(template.module)
                    cls = getattr(module, template.class_name)
                    return cls(**{key: decode(item) for key, item in template.items})
                if template.kind in {"custom_object", "slots_object"}:
                    module = importlib.import_module(template.module)
                    cls = getattr(module, template.class_name)
                    obj = cls.__new__(cls)
                    values = {key: decode(item) for key, item in template.items}
                    if template.kind == "custom_object":
                        obj.__dict__.update(values)
                    else:
                        for key, value in values.items():
                            setattr(obj, key, value)
                    return obj
            raise TypeError(f"unsupported tensor schema template: {type(template).__name__}")

        return decode(schema.template)

    def _decode_step_payloads_batch(
            self,
            step_metas: List[StepMeta],
            tensor_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> List[Dict[str, Any]]:
        """同一Schema的多个Step一次读取、一次按字段恢复，避免逐标量Tensor.item。"""
        if not step_metas:
            return []
        schema_id = step_metas[0].schema_id
        if any(step_meta.schema_id != schema_id for step_meta in step_metas):
            tensor_records = self.tensor_pool.get([
                step_meta.tensor_index for step_meta in step_metas
            ])
            return [
                self._decode_step_payload(step_meta, tensor_record)
                for step_meta, tensor_record in zip(step_metas, tensor_records)
            ]

        schema = self.tensor_schemas[schema_id]
        if tensor_batch is None:
            tensor_batch = self.tensor_pool.get_batch([
                step_meta.tensor_index for step_meta in step_metas
            ])
        field_specs = {field.key: field for field in schema.fields}
        step_count = len(step_metas)

        def decode(template: Any) -> List[Any]:
            if isinstance(template, _TensorRef):
                return self._restore_tensor_batch(
                    tensor_batch[template.key],
                    field_specs[template.key],
                )
            if isinstance(template, _InlineRef):
                return [
                    copy.deepcopy(step_meta.inline_values[template.index])
                    for step_meta in step_metas
                ]
            if isinstance(template, _OptionalTemplate):
                present = self._restore_tensor_batch(
                    tensor_batch[template.present.key],
                    field_specs[template.present.key],
                )
                values = decode(template.value)
                return [
                    value if is_present else None
                    for is_present, value in zip(present, values)
                ]
            if isinstance(template, _ContainerTemplate):
                if template.kind == "dict":
                    decoded_items = [
                        (copy.deepcopy(key), decode(item))
                        for key, item in template.items
                    ]
                    return [
                        {key: values[step] for key, values in decoded_items}
                        for step in range(step_count)
                    ]
                if template.kind in {"list", "tuple"}:
                    decoded_items = [decode(item) for item in template.items]
                    rows = [
                        [values[step] for values in decoded_items]
                        for step in range(step_count)
                    ]
                    return rows if template.kind == "list" else [tuple(row) for row in rows]
                if template.kind == "namedtuple":
                    module = importlib.import_module(template.module)
                    cls = getattr(module, template.class_name)
                    decoded_items = [(key, decode(item)) for key, item in template.items]
                    return [
                        cls(**{key: values[step] for key, values in decoded_items})
                        for step in range(step_count)
                    ]
                if template.kind in {"custom_object", "slots_object"}:
                    module = importlib.import_module(template.module)
                    cls = getattr(module, template.class_name)
                    decoded_items = [(key, decode(item)) for key, item in template.items]
                    result = []
                    for step in range(step_count):
                        obj = cls.__new__(cls)
                        values = {key: items[step] for key, items in decoded_items}
                        if template.kind == "custom_object":
                            obj.__dict__.update(values)
                        else:
                            for key, value in values.items():
                                setattr(obj, key, value)
                        result.append(obj)
                    return result
            raise TypeError(f"unsupported tensor schema template: {type(template).__name__}")

        return decode(schema.template)

    @staticmethod
    def _restore_tensor_value(tensor: torch.Tensor, spec: TensorFieldSpec) -> Any:
        if spec.source_kind == "torch":
            target_device = _RESTORE_TENSOR_DEVICE.get() or spec.source_device or "cpu"
            result = tensor.clone().to(target_device)
            if spec.requires_grad and (result.is_floating_point() or result.is_complex()):
                result.requires_grad_(True)
            return result
        if spec.source_kind in {"ndarray", "numpy_scalar"}:
            array = tensor.detach().cpu().numpy().copy().astype(spec.source_dtype, copy=False)
            return array[()] if spec.source_kind == "numpy_scalar" else array.reshape(spec.shape)
        if spec.source_kind in {"ndarray_bytes", "numpy_scalar_bytes"}:
            raw = bytes(tensor.detach().cpu().tolist())
            array = np.frombuffer(raw, dtype=np.dtype(spec.source_dtype)).copy().reshape(spec.shape)
            return array[()] if spec.source_kind == "numpy_scalar_bytes" else array
        if spec.source_kind == "python_bool":
            return bool(tensor.item())
        if spec.source_kind == "python_int":
            return int(tensor.item())
        if spec.source_kind == "python_float":
            return float(tensor.item())
        if spec.source_kind == "python_complex":
            return complex(tensor.item())
        if spec.source_kind == "numeric_list":
            return tensor.detach().cpu().numpy().astype(spec.source_dtype, copy=False).tolist()
        raise TypeError(f"unsupported tensor source kind: {spec.source_kind}")

    @staticmethod
    def _restore_tensor_batch(
            tensor: torch.Tensor,
            spec: TensorFieldSpec,
    ) -> List[Any]:
        """把一个字段的Step列批量恢复为原始Python/NumPy/Tensor值。"""
        if spec.source_kind == "torch":
            target_device = _RESTORE_TENSOR_DEVICE.get() or spec.source_device or "cpu"
            result = tensor.detach().to(target_device)
            rows = [row.clone() for row in result.unbind(0)]
            if spec.requires_grad:
                for row in rows:
                    if row.is_floating_point() or row.is_complex():
                        row.requires_grad_(True)
            return rows
        if spec.source_kind in {"ndarray", "numpy_scalar"}:
            array = tensor.detach().cpu().numpy().astype(spec.source_dtype, copy=False)
            if spec.source_kind == "numpy_scalar":
                return [row[()] for row in array]
            return [row.copy().reshape(spec.shape) for row in array]
        if spec.source_kind in {"ndarray_bytes", "numpy_scalar_bytes"}:
            return [
                ReplayBuffer._restore_tensor_value(row, spec)
                for row in tensor.unbind(0)
            ]
        if spec.source_kind == "python_bool":
            return [bool(value) for value in tensor.tolist()]
        if spec.source_kind == "python_int":
            return [int(value) for value in tensor.tolist()]
        if spec.source_kind == "python_float":
            return [float(value) for value in tensor.tolist()]
        if spec.source_kind == "python_complex":
            return [complex(value) for value in tensor.tolist()]
        if spec.source_kind == "numeric_list":
            values = tensor.detach().cpu().numpy().astype(spec.source_dtype, copy=False).tolist()
            return values
        raise TypeError(f"unsupported tensor source kind: {spec.source_kind}")

    def _materialize_episode(
            self,
            episode_meta: EpisodeMeta,
            start_step: int = 0,
            end_step: Optional[int] = None,
            sliced: bool = False,
            tensor_batch: Optional[Dict[str, torch.Tensor]] = None,
    ) -> TransitionType:
        """根据EpisodeMeta -> StepMeta -> tensor_index重建原Transition或按步片段。"""
        end_step = episode_meta.step_count if end_step is None else end_step
        assert 0 <= start_step <= end_step <= episode_meta.step_count
        transition = episode_meta.transition_type()
        for field_name, value in episode_meta.static_fields.items():
            setattr(transition, field_name, copy.deepcopy(value))

        for field_name, (mode, keys) in episode_meta.step_field_layout.items():
            if mode == "list":
                setattr(transition, field_name, [])
            else:
                setattr(transition, field_name, {key: [] for key in keys})

        step_meta_store = self.step_meta[episode_meta.agent_name]
        selected_step_metas = [
            step_meta_store[episode_meta.step_meta_start + step_offset]
            for step_offset in range(start_step, end_step)
        ]
        decoded_steps = self._decode_step_payloads_batch(selected_step_metas, tensor_batch)
        for decoded in decoded_steps:
            fields = decoded["fields"]
            for field_name, (mode, _) in episode_meta.step_field_layout.items():
                if field_name not in fields:
                    continue
                if mode == "list":
                    getattr(transition, field_name).append(fields[field_name])
                else:
                    target = getattr(transition, field_name)
                    for key, value in fields[field_name].items():
                        target[key].append(value)

        transition.episode_num = episode_meta.episode_num
        if episode_meta.sample_version is not None:
            transition.sample_version = episode_meta.sample_version
        transition.seq_len = end_step - start_step
        if sliced or episode_meta.has_starts_with_episode:
            transition.starts_with_episode = (
                start_step == 0 and episode_meta.starts_with_episode
            )

        if isinstance(transition, EpisodeTransition):
            transition.hidden_in = (
                decoded_steps[0].get("hidden_in") or {}
                if start_step == 0 and decoded_steps
                else {}
            )
            transition.hidden_out = (
                decoded_steps[-1].get("hidden_out") or {}
                if end_step == episode_meta.step_count and decoded_steps
                else {}
            )
        if start_step == 0 and decoded_steps:
            for field_name, value in decoded_steps[0].get("episode_tensor_fields", {}).items():
                setattr(transition, field_name, value)
        if (start_step > 0 and isinstance(transition, EpisodeTransition)
                and _INCLUDE_BURN_IN.get()):
            # 附带同一组、同一 episode 的完整历史；保持窗口 seq_len 和采样
            # 游标不变。学习端用当前参数无梯度重建 hidden，不使用旧采样 hidden。
            transition.burn_in = self._materialize_episode(
                episode_meta, start_step=0, end_step=start_step,
            )
        return transition

    def _materialize_episode_windows(
            self,
            windows: List[Tuple[EpisodeMeta, int, int]],
    ) -> List[TransitionType]:
        """调用方持锁；同一 schema 的窗口合并取数，再按原窗口顺序解码。"""
        indices_by_schema: Dict[int, List[int]] = {}
        window_rows = []
        for meta, start, end in windows:
            store = self.step_meta[meta.agent_name]
            steps = [store[meta.step_meta_start + offset] for offset in range(start, end)]
            if not steps or any(step.schema_id != steps[0].schema_id for step in steps):
                window_rows.append(None)
                continue
            schema_id = steps[0].schema_id
            indices = indices_by_schema.setdefault(schema_id, [])
            first = len(indices)
            indices.extend(step.tensor_index for step in steps)
            window_rows.append((schema_id, first, len(indices)))

        batches = {
            schema_id: self.tensor_pool.get_batch(indices)
            for schema_id, indices in indices_by_schema.items()
        }
        result = []
        for (meta, start, end), rows in zip(windows, window_rows):
            tensor_batch = None
            if rows is not None:
                schema_id, first, last = rows
                tensor_batch = {
                    key: column[first:last]
                    for key, column in batches[schema_id].items()
                }
            result.append(self._materialize_episode(
                meta, start, end, sliced=True, tensor_batch=tensor_batch,
            ))
        return result

    def _materialize_episode_metas(self, episode_metas: List[EpisodeMeta]) -> List[TransitionType]:
        return [self._materialize_episode(episode_meta) for episode_meta in episode_metas]

    def _release_episode(self, agent_name: AgentName, episode_meta: EpisodeMeta) -> None:
        step_meta_store = self.step_meta.get(agent_name, {})
        step_meta_ids = range(episode_meta.step_meta_start, episode_meta.step_meta_end)
        tensor_indices = [
            step_meta_store[step_meta_id].tensor_index
            for step_meta_id in step_meta_ids
            if step_meta_id in step_meta_store
        ]
        self.tensor_pool.release(tensor_indices)
        for step_meta_id in range(episode_meta.step_meta_start, episode_meta.step_meta_end):
            step_meta_store.pop(step_meta_id, None)

    @timed("replay.sample")
    def sample(self, agent_name: AgentName,
               batch_size: Optional[int] = 0,
               strategy: Optional[SamplingStrategy | str] = None,
               seed: Optional[int] = None,
               include_burn_in: bool = True,
               sequence_length: int = 1) \
            -> List[TransitionType]:
        """获取样本数据；strategy为空时使用ReplayBufferConfig.strategy。

        include_burn_in=False 时不为步内窗口附带同局完整历史前缀。非 RNN Learner
        （如 MADDPG）不需要该前缀，关闭后显著降低物化和 RPC 传输开销。
        """
        if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer")
        burn_in_token = _INCLUDE_BURN_IN.set(bool(include_burn_in))
        try:
            with self._buffer_lock:
                request = self._version_requests.get(agent_name)
                actual_strategy = request[0] if request is not None else strategy or self.config.strategy
                if agent_name in self._overlap_agents:
                    end = self._next_overlap_sample_end(agent_name, batch_size or self._version_requests[agent_name][1])
                    if end is None:
                        record("replay.sample_empty", 1.0)
                        return []
                    self._sample_end_seq = end
                    try:
                        result = self._sample_unlocked(
                            agent_name, batch_size, strategy, seed, sequence_length,
                        )
                    finally:
                        self._sample_end_seq = None
                    if result:
                        self._overlap_cursor[agent_name] = end
                        self._overlap_reads += 1
                else:
                    result = self._sample_unlocked(
                        agent_name, batch_size, strategy, seed, sequence_length,
                    )
                if result and agent_name in self._sample_versions:
                    trajectories_by_version = Counter()
                    steps_by_version = Counter()
                    for item in result:
                        version = getattr(item, "sample_version", None) or "unknown"
                        trajectories_by_version[version] += 1
                        steps_by_version[version] += item.seq_len
                    LOG.info(f"ReplayBuffer versioned batch: agent_name={agent_name}, "
                             f"sample_version={self._sample_versions[agent_name]}, "
                             f"trajectories_by_version={dict(trajectories_by_version)}, "
                             f"steps_by_version={dict(steps_by_version)}")
                result = SampledEpisodes(result, sample_strategy=actual_strategy)
            record("replay.sample_empty", 1.0 if not result else 0.0)
            return result
        finally:
            _INCLUDE_BURN_IN.reset(burn_in_token)

    def sample_for_rpc(self, agent_name: AgentName,
                       batch_size: Optional[int] = 0,
                       strategy: Optional[SamplingStrategy | str] = None,
                       seed: Optional[int] = None,
                       include_burn_in: bool = True,
                       sequence_length: int = 1) -> List[TransitionType]:
        """为跨进程传输直接物化 CPU Tensor；不改变本地 sample 的设备语义。"""
        token = _RESTORE_TENSOR_DEVICE.set(torch.device("cpu"))
        try:
            return self.sample(
                agent_name, batch_size, strategy, seed,
                include_burn_in=include_burn_in,
                sequence_length=sequence_length,
            )
        finally:
            _RESTORE_TENSOR_DEVICE.reset(token)

    def _sample_unlocked(self, agent_name: AgentName,
                         batch_size: Optional[int] = 0,
                         strategy: Optional[SamplingStrategy | str] = None,
                         seed: Optional[int] = None,
                         sequence_length: int = 1) \
            -> List[TransitionType]:
        """在持有_buffer_lock时执行实际采样。"""
        version_request = self._version_requests.get(agent_name)
        if version_request is None:
            sample_strategy = strategy or self.config.strategy
            batch_size = batch_size or self.config.batch_size
        else:
            sample_strategy, configured_batch_size = version_request
            if strategy is not None and strategy != sample_strategy:
                raise ValueError(f"versioned replay strategy for {agent_name} must be {sample_strategy.value}")
            batch_size = batch_size or configured_batch_size
        if (agent_name in self._sample_versions and sample_strategy in ROLLING_STRATEGIES
                and agent_name not in self._overlap_agents and not self._has_current_sample(agent_name)):
            return []
        rng = random.Random(seed) if seed is not None else random
        LOG.debug(
            f'Learner {agent_name} sample strategy is: {sample_strategy}, '
            f'batch size is: {batch_size}, seed={seed}'
        )
        if sample_strategy == SamplingStrategy.UNIFORM_STEPS:
            return self._sample_uniform_steps(
                agent_name, batch_size, rng, sequence_length=sequence_length,
            )
        if self._bundled():
            return self._sample_unlocked_bundled(agent_name, batch_size, sample_strategy, rng)
        if sample_strategy == SamplingStrategy.ON_POLICY_CURSOR:
            return self._sample_on_policy_cursor(agent_name, batch_size)
        if sample_strategy == SamplingStrategy.ON_POLICY_RECENT_STEPS:
            return self._sample_on_policy_recent_steps(agent_name, batch_size)
        if sample_strategy == SamplingStrategy.ON_POLICY_CURSOR_STEPS:
            return self._sample_on_policy_cursor_steps(agent_name, batch_size)
        if sample_strategy == SamplingStrategy.TIME_SLICE:
            return self._sample_time_slice(agent_name, batch_size, rng)

        # rehearsal 混采仅作用于 off-policy 的 recent_biased / uniform。
        if (
                self._rehearsal_sample_ratio > 0.0
                and self.rehearsal_enabled
                and sample_strategy in (
                    SamplingStrategy.UNIFORM,
                    SamplingStrategy.RECENT_BIASED,
                    SamplingStrategy.UNIFORM.value,
                    SamplingStrategy.RECENT_BIASED.value,
                )
        ):
            return self._sample_with_rehearsal(
                agent_name, batch_size, sample_strategy, rng
            )

        if len(self._sampling_entries(agent_name)) < batch_size:
            return []
        if sample_strategy == SamplingStrategy.UNIFORM:
            return self._sample_uniform(agent_name, batch_size, rng)
        elif sample_strategy == SamplingStrategy.RECENT_BIASED:
            return self._sample_recent_biased(agent_name, batch_size, rng)
        elif sample_strategy == SamplingStrategy.SLIDING_WINDOW:
            return self._sample_sliding_window(agent_name, batch_size, rng)
        elif sample_strategy == SamplingStrategy.ON_POLICY_RECENT:
            return self._sample_on_policy(agent_name, batch_size)
        else:
            raise ValueError(f"[ReplayBuffer 错误] 不支持的抽样策略: {sample_strategy}")

    def _sample_uniform_steps(
            self, agent_name: AgentName, batch_size: int, rng, *, sequence_length: int = 1,
    ) -> List[TransitionType]:
        """按有效决策步均匀有放回抽样，并从起点物化同局连续窗口。"""
        def population(entries):
            groups = self._split_bundles(entries) if self._bundled() else [[meta] for meta in entries]
            candidates, ends, total = [], [], 0
            for group in groups:
                length = max(meta.step_count for meta in group)
                if length:
                    total += length
                    candidates.append(group)
                    ends.append(total)
            return candidates, ends, total

        active = population(self._sampling_entries(agent_name))
        archive = population(self._rehearsal_entries_for_agent(agent_name)) if (
            self.rehearsal_enabled and self._rehearsal_sample_ratio > 0
        ) else ([], [], 0)
        if not active[2] and not archive[2]:
            return []
        if not active[2]:
            archive_count = batch_size
        elif not archive[2]:
            archive_count = 0
        else:
            archive_count = max(1, round(batch_size * self._rehearsal_sample_ratio))
            if agent_name in self._sample_versions:
                archive_count = min(archive_count, batch_size - 1)
            archive_count = min(archive_count, batch_size)
        sources = [active] * (batch_size - archive_count) + [archive] * archive_count
        if archive_count and archive_count < batch_size:
            rng.shuffle(sources)
        windows = []
        for groups, ends, total in sources:
            index = rng.randrange(total)
            group_index = bisect_right(ends, index)
            start = index - (ends[group_index - 1] if group_index else 0)
            windows.extend(
                (meta, start, min(start + sequence_length, meta.step_count))
                for meta in groups[group_index] if start < meta.step_count
            )
        return self._materialize_episode_windows(windows)

    # ---------- 同局多组捆绑抽样（episode_groups 模式） ----------

    def _sample_unlocked_bundled(
            self,
            agent_name: AgentName,
            batch_size: int,
            sample_strategy: SamplingStrategy | str,
            rng,
    ) -> List[TransitionType]:
        """episode_groups 模式的统一抽样入口：以物理局（捆绑样本）为单位抽取。

        返回值仍是铺平的有效组轨迹列表，网络消费形状为 [N,T+1,K,D]；
        缺组槽位（提前完成/未写入的组）不产生样本，组间不拼接时间序列。
        """
        if sample_strategy in (
                SamplingStrategy.TIME_SLICE, SamplingStrategy.TIME_SLICE.value):
            return self._sample_time_slice_bundled(agent_name, batch_size, rng)
        off_policy = sample_strategy in (
            SamplingStrategy.UNIFORM,
            SamplingStrategy.RECENT_BIASED,
            SamplingStrategy.UNIFORM.value,
            SamplingStrategy.RECENT_BIASED.value,
        )
        if not off_policy:
            raise ReplayPackagingException(
                detail=(
                    "sample_packaging='episode_groups' does not support sampling "
                    f"strategy {sample_strategy!r}; only uniform / recent_biased / "
                    "time_slice are bundle-aware."
                )
            )
        if (
                self._rehearsal_sample_ratio > 0.0
                and self.rehearsal_enabled
        ):
            return self._sample_with_rehearsal_bundles(
                agent_name, batch_size, sample_strategy, rng
            )
        bundles = self._split_bundles(self._sampling_entries(agent_name))
        if len(bundles) < batch_size:
            return []
        if sample_strategy in (
                SamplingStrategy.RECENT_BIASED, SamplingStrategy.RECENT_BIASED.value):
            selected = self._select_recent_biased_metas(bundles, batch_size, rng)
        else:
            selected = rng.sample(bundles, batch_size)
        selected = self._include_current_sample(agent_name, selected, bundles)
        flattened = [meta for bundle in selected for meta in bundle]
        return self._materialize_episode_metas(flattened)

    def _sample_with_rehearsal_bundles(
            self,
            agent_name: AgentName,
            batch_size: int,
            sample_strategy: SamplingStrategy | str,
            rng,
    ) -> List[TransitionType]:
        """按 rehearsal_sample_ratio 从存档与 active 以捆绑样本定额混采。"""
        rehearsal_bundles = self._split_bundles(self._rehearsal_entries_for_agent(agent_name))
        active_bundles = self._split_bundles(self._sampling_entries(agent_name))
        rehearsal_count = self._rehearsal_quota(batch_size, len(rehearsal_bundles))
        active_count = batch_size - rehearsal_count
        if (
                len(rehearsal_bundles) < rehearsal_count
                or len(active_bundles) < active_count
        ):
            return []
        selected_rehearsal = rng.sample(rehearsal_bundles, rehearsal_count)
        if sample_strategy in (
                SamplingStrategy.RECENT_BIASED,
                SamplingStrategy.RECENT_BIASED.value,
        ):
            selected_active = self._select_recent_biased_metas(
                active_bundles, active_count, rng
            )
        else:
            selected_active = rng.sample(active_bundles, active_count)
        selected_active = self._include_current_sample(agent_name, selected_active, active_bundles)
        selected = selected_rehearsal + selected_active
        rng.shuffle(selected)
        return self._materialize_episode_metas([
            meta for bundle in selected for meta in bundle
        ])

    @classmethod
    def _bundled_time_slice_windows(
            cls,
            bundles: List[List[EpisodeMeta]],
            slice_len: int,
    ) -> List[Tuple[List[EpisodeMeta], int]]:
        """各捆绑样本的按共同物理时间窗口数；窗口数按最长组的可行起点计。"""
        windows = []
        for bundle in bundles:
            max_len = max(
                (meta.step_count for meta in bundle if meta.step_count > 0),
                default=0,
            )
            if max_len <= 0:
                continue
            if max_len >= slice_len:
                windows.append((bundle, max_len - slice_len + 1))
            elif not cls.STRICT_LENGTH:
                windows.append((bundle, 1))
        return windows

    def _sample_time_slice_bundled(
            self,
            agent_name: AgentName,
            slice_len: int,
            rng,
    ) -> List[TransitionType]:
        """捆绑模式局内连续切片：slice_len 为共同物理时间窗长度。"""
        if slice_len <= 0:
            return []
        if agent_name not in self._sample_versions and self._rehearsal_sample_ratio > 0.0 and self.rehearsal_enabled:
            rehearsal_bundles = self._split_bundles(
                self._rehearsal_entries_for_agent(agent_name)
            )
            active_bundles = self._split_bundles(self._sampling_entries(agent_name))
            if (
                    not self._bundled_time_slice_windows(rehearsal_bundles, slice_len)
                    or not self._bundled_time_slice_windows(active_bundles, slice_len)
            ):
                return []
            bundles = (
                rehearsal_bundles
                if rng.random() < self._rehearsal_sample_ratio
                else active_bundles
            )
            return self._sample_time_slice_bundled_from(bundles, slice_len, rng)
        return self._sample_time_slice_bundled_from(
            self._split_bundles(self._sampling_entries(agent_name)),
            slice_len,
            rng,
        )

    def _sample_time_slice_bundled_from(
            self,
            bundles: List[List[EpisodeMeta]],
            slice_len: int,
            rng,
    ) -> List[TransitionType]:
        """选一个捆绑样本并按共同物理时间 [t0, t0+L) 切窗。

        每个组只保留自己在窗口内的实际决策帧 [max(0,t0), min(len_g, t0+L))；
        窗口内没有实际帧的组不产生样本，不补造观测、动作或奖励。中途开始
        的组沿用同组同局历史 burn-in，hidden 不跨组。
        """
        weighted = self._bundled_time_slice_windows(bundles, slice_len)
        if not weighted:
            return []
        population = [bundle for bundle, _ in weighted]
        weights = [window_count for _, window_count in weighted]
        bundle = rng.choices(population, weights=weights, k=1)[0]
        max_len = max(
            (meta.step_count for meta in bundle if meta.step_count > 0),
            default=0,
        )
        if max_len < slice_len:
            start_step = 0
        else:
            start_step = rng.randint(0, max_len - slice_len)
        result: List[TransitionType] = []
        for meta in bundle:
            if meta.step_count <= 0:
                continue
            end_step = min(meta.step_count, start_step + slice_len)
            if start_step >= end_step:
                continue
            result.append(self._materialize_episode(
                episode_meta=meta,
                start_step=start_step,
                end_step=end_step,
                sliced=True,
            ))
        return result

    def _sample_with_rehearsal(
            self,
            agent_name: AgentName,
            batch_size: int,
            sample_strategy: SamplingStrategy | str,
            rng,
    ) -> List[TransitionType]:
        """按 rehearsal_sample_ratio 从存档与 active 定额混采。"""
        rehearsal_entries = self._rehearsal_entries_for_agent(agent_name)
        active_entries = self._sampling_entries(agent_name)
        rehearsal_count = self._rehearsal_quota(batch_size, len(rehearsal_entries))
        active_count = batch_size - rehearsal_count
        if (
                len(rehearsal_entries) < rehearsal_count
                or len(active_entries) < active_count
        ):
            return []

        selected_rehearsal = rng.sample(rehearsal_entries, rehearsal_count)
        if sample_strategy in (
                SamplingStrategy.RECENT_BIASED,
                SamplingStrategy.RECENT_BIASED.value,
        ):
            selected_active = self._select_recent_biased_metas(
                active_entries, active_count, rng
            )
        else:
            selected_active = rng.sample(active_entries, active_count)
        selected_active = self._include_current_sample(agent_name, selected_active, active_entries)
        selected = selected_rehearsal + selected_active
        rng.shuffle(selected)
        return self._materialize_episode_metas(selected)

    def _select_recent_biased_metas(
            self, entries: List[EpisodeMeta], batch_size: int, rng
    ) -> List[EpisodeMeta]:
        if batch_size <= 0:
            return []
        sample_last_n = int(batch_size * self.config.recent_ratio)
        if self.config.recent_ratio > 0:
            sample_last_n = max(1, sample_last_n)
        sample_last_n = min(sample_last_n, len(entries), batch_size)
        recent_part = entries[-sample_last_n:]
        old_pool = entries[:-sample_last_n]
        needed_from_old = batch_size - sample_last_n
        if needed_from_old <= 0:
            return list(recent_part)
        if len(old_pool) < needed_from_old:
            return rng.sample(entries, batch_size)
        return recent_part + rng.sample(old_pool, needed_from_old)

    def _agent_buffer(self, agent_name: AgentName) -> List[EpisodeMeta]:
        assert agent_name in self.buffer, f"[ReplayBuffer 错误] 未找到Agent回放池: {agent_name}"
        return self.buffer[agent_name]

    def _sample_uniform(self, agent_name: AgentName, batch_size: int, rng=random) -> List[TransitionType]:
        """ 完全随机抽样 """
        entries = self._sampling_entries(agent_name)
        selected = rng.sample(entries, batch_size)
        return self._materialize_episode_metas(self._include_current_sample(agent_name, selected, entries))

    def _sample_recent_biased(self, agent_name: AgentName, batch_size: int, rng=random) -> List[TransitionType]:
        entries = self._sampling_entries(agent_name)
        selected = self._select_recent_biased_metas(entries, batch_size, rng)
        return self._materialize_episode_metas(self._include_current_sample(agent_name, selected, entries))

    def _sample_time_slice(
            self,
            agent_name: AgentName,
            batch_size: int,
            rng,
    ) -> List[TransitionType]:
        """局内连续切片：batch_size 为窗口长度 L。"""
        if batch_size <= 0:
            return []

        if agent_name not in self._sample_versions and self._rehearsal_sample_ratio > 0.0 and self.rehearsal_enabled:
            rehearsal_entries = self._rehearsal_entries_for_agent(agent_name)
            active_entries = self.buffer.get(agent_name, [])
            if (
                    not self._time_slice_windows(rehearsal_entries, batch_size)
                    or not self._time_slice_windows(active_entries, batch_size)
            ):
                return []
            entries = (
                rehearsal_entries
                if rng.random() < self._rehearsal_sample_ratio
                else active_entries
            )
            return self._sample_time_slice_from_entries(entries, batch_size, rng)

        return self._sample_time_slice_from_entries(
            self._sampling_entries(agent_name),
            batch_size,
            rng,
        )

    @classmethod
    def _time_slice_windows(
            cls,
            entries: List[EpisodeMeta],
            slice_len: int,
    ) -> List[Tuple[EpisodeMeta, int]]:
        windows = []
        for episode_meta in entries:
            step_count = episode_meta.step_count
            if step_count <= 0:
                continue
            if step_count >= slice_len:
                windows.append((episode_meta, step_count - slice_len + 1))
            elif not cls.STRICT_LENGTH:
                windows.append((episode_meta, 1))
        return windows

    def _sample_time_slice_from_entries(
            self,
            entries: List[EpisodeMeta],
            slice_len: int,
            rng,
    ) -> List[TransitionType]:
        """按窗口数加权抽一个切片。"""
        weighted = self._time_slice_windows(entries, slice_len)
        if not weighted:
            return []
        population = [episode_meta for episode_meta, _ in weighted]
        weights = [window_count for _, window_count in weighted]
        episode_meta = rng.choices(population, weights=weights, k=1)[0]
        step_count = episode_meta.step_count
        if step_count < slice_len:
            start_step = 0
            end_step = step_count
        else:
            start_step = rng.randint(0, step_count - slice_len)
            end_step = start_step + slice_len
        return [
            self._materialize_episode(
                episode_meta=episode_meta,
                start_step=start_step,
                end_step=end_step,
                sliced=True,
            )
        ]

    def _sample_on_policy(self, agent_name: AgentName, batch_size: int) -> List[TransitionType]:
        """on-policy原始抽样：严格返回最近batch_size个episode，不移动任何消费游标。"""
        return self._materialize_episode_metas(self._sampling_entries(agent_name)[-batch_size:])

    def _sample_on_policy_recent_steps(self, agent_name: AgentName, batch_size: int) -> List[TransitionType]:
        """
        on-policy近期滚动步抽样：从当前回放池尾部向前累计batch_size个transition。

        该策略不维护消费游标；只要回放池没有新增样本，多次调用会返回完全相同的尾部窗口。
        PPO/A2C等算法如果配置learn_times_per_episode>1，应由算法侧的重复batch哈希保护跳过重复更新。
        """
        ordered_pairs = self._sampling_pairs(agent_name)
        available_steps = sum(
            episode_meta.step_count
            for _, episode_meta in ordered_pairs
            if episode_meta.step_count > 0
        )
        if available_steps < batch_size:
            LOG.debug(
                f"ON_POLICY_RECENT_STEPS samples not enough: agent_name={agent_name}, "
                f"steps={available_steps}, target_steps={batch_size}"
            )
            return []

        selected_reversed: List[TransitionType] = []
        remaining_steps = batch_size
        first_cursor: Optional[Tuple[int, int]] = None
        last_cursor: Optional[Tuple[int, int]] = None

        for seq, episode_meta in reversed(ordered_pairs):
            episode_steps = episode_meta.step_count
            if episode_steps <= 0:
                continue

            end_step = episode_steps
            start_step = max(0, end_step - remaining_steps)
            chunk_steps = end_step - start_step
            if chunk_steps <= 0:
                continue

            chunk = self._materialize_episode(
                episode_meta=episode_meta,
                start_step=start_step,
                end_step=end_step,
                sliced=True,
            )
            selected_reversed.append(chunk)
            first_cursor = (seq, start_step)
            if last_cursor is None:
                last_cursor = (seq, end_step)
            remaining_steps -= chunk_steps
            if remaining_steps <= 0:
                break

        assert remaining_steps == 0, "metadata preflight guaranteed enough recent steps"

        selected = list(reversed(selected_reversed))
        LOG.info(
            f"ReplayBuffer on_policy_recent_steps sampled: agent_name={agent_name}, "
            f"target_steps={batch_size}, actual_steps={batch_size}, chunks={len(selected)}, "
            f"start_seq={first_cursor[0] if first_cursor else None}, "
            f"start_step={first_cursor[1] if first_cursor else None}, "
            f"end_seq={last_cursor[0] if last_cursor else None}, "
            f"end_step={last_cursor[1] if last_cursor else None}"
        )
        return selected

    @staticmethod
    def _transition_steps(transition: TransitionType | EpisodeMeta) -> int:
        """返回一个episode transition包含的有效环境步数。"""
        if isinstance(transition, EpisodeMeta):
            return transition.step_count
        seq_len = getattr(transition, "seq_len", None)
        if seq_len is not None:
            return int(seq_len)
        rewards = getattr(transition, "rewards", None)
        if isinstance(rewards, dict):
            return max((len(v) for v in rewards.values()), default=0)
        if rewards is not None:
            return len(rewards)
        return 0

    def _sample_on_policy_cursor(self, agent_name: AgentName, batch_size: int) -> List[TransitionType]:
        """
        on-policy游标抽样：从上次消费游标之后取连续batch_size个新episode。

        样本不物理删除；ReplayBuffer继续保存历史episode，ON_POLICY_CURSOR只推进逻辑消费游标。
        新样本不足batch_size时返回空列表，表示这是采样策略的正常等待状态，不是异常。
        """
        ordered_pairs = self._sampling_pairs(agent_name)
        cursor = self.on_policy_cursor.setdefault(agent_name, 0)
        new_pairs = [(seq, transition) for seq, transition in ordered_pairs if seq > cursor]
        if len(new_pairs) < batch_size:
            LOG.debug(
                f"ON_POLICY_CURSOR samples not enough: agent_name={agent_name}, "
                f"new_samples={len(new_pairs)}, batch_size={batch_size}, cursor={cursor}"
            )
            return []
        consumed_pairs = new_pairs[:batch_size]
        self.on_policy_cursor[agent_name] = consumed_pairs[-1][0]
        self.last_sample_stats[agent_name] = {
            "episodes": len(consumed_pairs),
            "steps": 0,
            "strategy": "on_policy_cursor",
        }
        LOG.info(
            f"ReplayBuffer on-policy cursor samples consumed: agent_name={agent_name}, "
            f"batch_size={batch_size}, cursor={self.on_policy_cursor[agent_name]}"
        )
        return self._materialize_episode_metas([episode_meta for _, episode_meta in consumed_pairs])

    def _sample_on_policy_cursor_steps(self, agent_name: AgentName, batch_size: int) -> List[TransitionType]:
        """
        on-policy游标步抽样：按step游标切片，累计到batch_size个transition就返回。

        这个策略不物理删除样本，且允许从一个长episode中间切出片段；
        返回成功后推进到(episode序号, episode内step偏移)。
        样本不足batch_size步时返回空列表，游标不移动，等待后续采样补足。
        """
        return self._sample_on_policy_steps_by_cursor(
            agent_name=agent_name,
            batch_size=batch_size,
            cursor_store=self.on_policy_step_cursor,
            advance_steps=batch_size,
            strategy_name="ON_POLICY_CURSOR_STEPS",
        )

    def _sample_on_policy_steps_by_cursor(
            self,
            agent_name: AgentName,
            batch_size: int,
            cursor_store: Dict[AgentName, Tuple[int, int]],
            advance_steps: int,
            strategy_name: str,
    ) -> List[TransitionType]:
        """
        按step游标切出固定步数窗口。

        :param cursor_store: 不同采样策略各自持有的step游标。
        :param advance_steps: 成功返回后窗口起点推进多少step；滚动=1，游动=batch_size。
        """
        ordered_pairs = self._sampling_pairs(agent_name)
        cursor = cursor_store.get(agent_name, (0, 0))
        available_steps = self._available_steps_from_cursor(ordered_pairs, cursor)
        if available_steps < batch_size:
            LOG.debug(
                f"{strategy_name} samples not enough: agent_name={agent_name}, "
                f"steps={available_steps}, target_steps={batch_size}, "
                f"cursor_seq={cursor[0]}, cursor_step={cursor[1]}"
            )
            return []

        sample_result = self._collect_step_window(
            ordered_pairs=ordered_pairs,
            cursor=cursor,
            batch_size=batch_size,
        )
        assert sample_result is not None, "metadata preflight guaranteed enough cursor steps"

        selected, start_cursor, actual_steps = sample_result
        next_cursor = self._advance_step_cursor(
            ordered_pairs=ordered_pairs,
            cursor=start_cursor,
            advance_steps=advance_steps,
        )
        cursor_store[agent_name] = next_cursor
        self.last_sample_stats[agent_name] = {
            "episodes": len(selected),
            "steps": int(actual_steps),
            "start_cursor": tuple(start_cursor),
            "next_cursor": tuple(next_cursor),
            "strategy": strategy_name.lower(),
        }
        LOG.info(
            f"ReplayBuffer {strategy_name.lower()} sampled: agent_name={agent_name}, "
            f"target_steps={batch_size}, actual_steps={actual_steps}, chunks={len(selected)}, "
            f"cursor_seq={next_cursor[0]}, cursor_step={next_cursor[1]}, advance_steps={advance_steps}"
        )
        return selected

    def _collect_step_window(
            self,
            ordered_pairs: List[tuple[int, EpisodeMeta]],
            cursor: Tuple[int, int],
            batch_size: int,
    ) -> Optional[Tuple[List[TransitionType], Tuple[int, int], int]]:
        """从step游标处尝试切出batch_size步；不足时返回None且不产生副作用。"""
        cursor_seq, cursor_step = cursor
        selected: List[TransitionType] = []
        total_steps = 0
        start_cursor: Optional[Tuple[int, int]] = None
        for seq, episode_meta in ordered_pairs:
            if seq < cursor_seq:
                continue
            episode_steps = episode_meta.step_count
            if episode_steps <= 0:
                continue
            start_step = cursor_step if seq == cursor_seq else 0
            if start_step >= episode_steps:
                continue
            if start_cursor is None:
                start_cursor = (seq, start_step)

            remaining_steps = batch_size - total_steps
            end_step = min(episode_steps, start_step + remaining_steps)
            chunk = self._materialize_episode(
                episode_meta=episode_meta,
                start_step=start_step,
                end_step=end_step,
                sliced=True,
            )
            selected.append(chunk)
            total_steps += end_step - start_step
            if total_steps >= batch_size:
                assert start_cursor is not None, "step window start cursor should be initialized."
                return selected, start_cursor, total_steps

        return None

    def _advance_step_cursor(
            self,
            ordered_pairs: List[tuple[int, EpisodeMeta]],
            cursor: Tuple[int, int],
            advance_steps: int,
    ) -> Tuple[int, int]:
        """从给定step游标向前推进advance_steps步，返回新的(step cursor)。"""
        cursor_seq, cursor_step = cursor
        remaining = max(0, int(advance_steps))
        for seq, episode_meta in ordered_pairs:
            if seq < cursor_seq:
                continue
            episode_steps = episode_meta.step_count
            if episode_steps <= 0:
                continue
            start_step = cursor_step if seq == cursor_seq else 0
            if start_step >= episode_steps:
                continue
            available = episode_steps - start_step
            if remaining <= available:
                return seq, start_step + remaining
            remaining -= available
        return cursor

    def _available_steps_from_cursor(
            self,
            ordered_pairs: List[tuple[int, EpisodeMeta]],
            cursor: Tuple[int, int],
    ) -> int:
        """统计从step游标开始到当前buffer末尾可用的transition数量。"""
        cursor_seq, cursor_step = cursor
        total_steps = 0
        for seq, episode_meta in ordered_pairs:
            if seq < cursor_seq:
                continue
            episode_steps = episode_meta.step_count
            if episode_steps <= 0:
                continue
            start_step = cursor_step if seq == cursor_seq else 0
            if start_step < episode_steps:
                total_steps += episode_steps - start_step
        return total_steps

    def _slice_transition_steps(
            self,
            transition: TransitionType,
            start_step: int,
            end_step: int,
    ) -> TransitionType:
        """从episode transition中切出[start_step, end_step)片段，供按步游标采样使用。"""
        if isinstance(transition, SAEpisodeTransition):
            return self._slice_sa_transition_steps(transition, start_step, end_step)
        if isinstance(transition, EpisodeTransition):
            return self._slice_ma_transition_steps(transition, start_step, end_step)
        raise TypeError(f"unsupported transition type: {type(transition).__name__}")

    @staticmethod
    def _slice_sa_transition_steps(
            transition: SAEpisodeTransition,
            start_step: int,
            end_step: int,
    ) -> SAEpisodeTransition:
        """切分单智能体episode片段；starts_with_episode保留真实episode边界语义。"""
        chunk = SAEpisodeTransition()
        chunk.episode_num = transition.episode_num
        chunk.starts_with_episode = start_step == 0 and transition.starts_with_episode
        for field_name in (
                "obs",
                "actions",
                "last_actions",
                "rewards",
                "next_obs",
                "terminations",
                "truncations",
                "avail_actions",
                "old_log_probs",
                "old_values",
                "metadata",
        ):
            setattr(chunk, field_name, list(getattr(transition, field_name)[start_step:end_step]))
        chunk.seq_len = end_step - start_step
        return chunk

    @staticmethod
    def _slice_ma_transition_steps(
            transition: EpisodeTransition,
            start_step: int,
            end_step: int,
    ) -> EpisodeTransition:
        """切分多智能体episode片段；RNN类算法使用中间片段时需注意隐藏态连续性。"""
        chunk = EpisodeTransition(
            agent_ids=tuple(transition.agent_ids), group_index=transition.group_index,
        )
        chunk.episode_num = transition.episode_num
        chunk.starts_with_episode = start_step == 0 and transition.starts_with_episode
        for field_name in (
                "hidden_states",
                "obs",
                "actions",
                "last_actions",
                "rewards",
                "next_obs",
                "terminations",
                "truncations",
                "active_agent_mask",
                "next_active_agent_mask",
        ):
            # 兼容修复前已序列化、尚无下一时刻存活掩码的样本。
            source = getattr(transition, field_name, {})
            setattr(chunk, field_name, {
                agent_id: list(values[start_step:end_step])
                for agent_id, values in source.items()
            })
        chunk.states = list(transition.states[start_step:end_step])
        chunk.next_states = list(transition.next_states[start_step:end_step])
        chunk.metadata = list(transition.metadata[start_step:end_step])
        chunk.step_rewards = list(transition.step_rewards[start_step:end_step])
        chunk.avail_actions_next = list(transition.avail_actions_next[start_step:end_step])
        chunk.avail_actions = list(getattr(transition, "avail_actions", [])[start_step:end_step])
        chunk.hidden_in = transition.hidden_in if start_step == 0 else {}
        chunk.hidden_out = transition.hidden_out if end_step == transition.seq_len else {}
        chunk.seq_len = end_step - start_step
        if start_step > 0:
            chunk.burn_in = ReplayBuffer._slice_ma_transition_steps(transition, 0, start_step)
        return chunk

    def _ordered_episode_meta(self, agent_name: AgentName) -> List[EpisodeMeta]:
        """返回按写入时间排序的EpisodeMeta。"""
        return list(self._agent_buffer(agent_name))

    def _ordered_episode_meta_with_seq(self, agent_name: AgentName) -> List[tuple[int, EpisodeMeta]]:
        """返回按写入时间排序的(seq, EpisodeMeta)，供游标策略消费。"""
        agent_buffer = self._agent_buffer(agent_name)
        seq_buffer = self.sample_seq.get(agent_name)
        if seq_buffer is None or len(seq_buffer) != len(agent_buffer):
            seq_buffer = [episode_meta.write_seq for episode_meta in agent_buffer]
            self.sample_seq[agent_name] = seq_buffer
            self.write_count[agent_name] = max(seq_buffer, default=0)
        return list(zip(seq_buffer, agent_buffer))

    def _ordered_buffer(self, agent_name: AgentName) -> List[TransitionType]:
        """兼容原私有方法：返回按写入时间排序并重建后的Transition。"""
        return self._materialize_episode_metas(self._ordered_episode_meta(agent_name))

    def _ordered_buffer_with_seq(self, agent_name: AgentName) -> List[tuple[int, TransitionType]]:
        """兼容原私有方法：返回按写入时间排序的(seq, Transition)。"""
        return [
            (seq, self._materialize_episode(episode_meta))
            for seq, episode_meta in self._ordered_episode_meta_with_seq(agent_name)
        ]

    def _sampling_pairs(self, agent_name: AgentName) -> List[tuple[int, EpisodeMeta]]:
        if agent_name not in self._sample_versions:
            return self._ordered_episode_meta_with_seq(agent_name)
        return [(meta.write_seq, meta) for meta in self._sampling_entries(agent_name)]

    def _sample_sliding_window(self, agent_name: AgentName, batch_size: int, rng=random) -> List[TransitionType]:
        """ 滑动窗口抽样：取当前游标开始的连续片段，剩余部分在池外随机抽取，游标每次后移一格 """
        agent_buffer = self._sampling_entries(agent_name)
        buffer_size = len(agent_buffer)
        sliding_idx = self.sliding_idx.setdefault(agent_name, 0)

        # 1. 确定窗口大小 (这里复用 recent_ratio 参数，即窗口占比)
        window_size = int(batch_size * self.config.recent_ratio)
        window_size = max(1, min(window_size, buffer_size, batch_size))
        needed_random = batch_size - window_size

        # 2. 提取连续的窗口数据 (利用 % buffer_size 完美处理环形越界，滑到末尾自动从头接续)
        window_indices = [(sliding_idx + i) % buffer_size for i in range(window_size)]
        window_part = [agent_buffer[i] for i in window_indices]

        # 3. 游标向后滑动一格 (同样利用模运算保证安全)
        self.sliding_idx[agent_name] = (sliding_idx + 1) % buffer_size

        # 4. 剩余数据完全随机抽取 (必须排除掉刚才已经抽过的窗口数据)
        if needed_random > 0:
            # 用 O(1) 的 Set 差集运算，优雅剔除重叠索引
            all_indices = set(range(buffer_size))
            window_indices_set = set(window_indices)
            available_indices = list(all_indices - window_indices_set)

            # Fail-Safe 容错：如果在训练极早期，池子太小导致池外没数据了，退化为允许重复的全局随机
            if len(available_indices) >= needed_random:
                random_indices = rng.sample(available_indices, needed_random)
            else:
                # 当池外剩余量不足时，放宽限制，从整个池子盲抽补齐批次
                random_indices = rng.choices(list(all_indices), k=needed_random)

            random_part = [agent_buffer[i] for i in random_indices]
        else:
            random_part = []

        selected = self._include_current_sample(agent_name, window_part + random_part, agent_buffer)
        return self._materialize_episode_metas(selected)

    def _report_memory_usage_by_size(self, current_size: int) -> None:
        """按累计写入局数每 100 局上报一次张量池与元数据占用。"""
        written_episodes = max(self.write_count.values(), default=0)
        if written_episodes <= 0:
            return
        report_size = (written_episodes // self.memory_report_size_interval) * self.memory_report_size_interval
        if report_size <= 0 or report_size <= self._last_memory_report_size:
            return
        self._last_memory_report_size = report_size
        with self._buffer_lock:
            tensor_memory_mb = self.tensor_pool.memory_bytes / (1024 ** 2)
            metadata_memory_mb = estimate_object_memory_mb({
                "episode_meta": self.buffer,
                "step_meta": self.step_meta,
                "tensor_schemas": self.tensor_schemas,
            })
            agent_names = list(self.buffer.keys())
            pool_kind = getattr(self.tensor_pool, "kind", "torch_save")
        rss_mb = _process_rss_bytes() / (1024 ** 2)
        if pool_kind == KIND_MMAP_DIR:
            LOG.info(
                f"ReplayBuffer memory usage: tensors_on_disk={tensor_memory_mb:,.3f} MB, "
                f"process_rss={rss_mb:,.3f} MB, metadata={metadata_memory_mb:,.3f} MB, "
                f"size={current_size}, writes={written_episodes}, agents={agent_names}"
            )
        else:
            LOG.info(
                f"ReplayBuffer memory usage: tensors={tensor_memory_mb:,.3f} MB, "
                f"metadata={metadata_memory_mb:,.3f} MB, "
                f"size={current_size}, writes={written_episodes}, agents={agent_names}"
            )

    def monitor_thread(self):
        """阻塞观察写入局数变化；后续缓存、吞吐、内存等指标都挂在这里。"""
        last_observed_writes = max(self.write_count.values(), default=0)
        while self._monitor_running:
            with self._size_changed_cv:
                self._size_changed_cv.wait_for(
                    lambda: max(self.write_count.values(), default=0) != last_observed_writes
                )
                previous_size = last_observed_writes
                last_observed_writes = max(self.write_count.values(), default=0)
                current_size = self.size
            self._on_size_changed(previous_size=previous_size, current_size=current_size)

    def _on_size_changed(self, previous_size: int, current_size: int) -> None:
        """ReplayBuffer 写入后的统一监控入口。"""
        self._report_memory_usage_by_size(current_size=current_size)

    def _unwrap_ring_snapshot_unlocked(self) -> None:
        """把旧环形快照展开成时间顺序，后续淘汰不再依赖 position。"""
        for agent_name, entries in list(self.buffer.items()):
            seqs = self.sample_seq.get(agent_name)
            if seqs is None or len(seqs) != len(entries):
                seqs = [episode_meta.write_seq for episode_meta in entries]
            position = int(self.position.get(agent_name, 0) or 0)
            if entries and position and len(entries) >= self._capacity_for_agent(agent_name):
                entries = entries[position:] + entries[:position]
                seqs = seqs[position:] + seqs[:position]
            self.buffer[agent_name] = list(entries)
            self.sample_seq[agent_name] = list(seqs)
            self.position[agent_name] = 0
        self.episode_meta = self.buffer

    def save_buffer(self, episode_data: Union[
        Dict[AgentName, Union[TransitionType, List[TransitionType]]], TransitionType,
    ]):
        """ 持久化采样数据 """
        raw_data = self._cover_data_to_dict(episode_data)
        self._save_to_backend_storage(raw_data)

    def save_snapshot(self, path: str | Path) -> Path:
        """
        阶段结束把当前回放池写成 DRBF 单文件快照。

        保存张量池 + EpisodeMeta/StepMeta/schema + 环缓冲结构字段；
        不保存 on-policy/sliding 游标与 generation（加载后由模块重置）。
        张量段走池的 checkpoint 后流式写入，禁止整池进 BytesIO。
        """
        import shutil
        import tempfile

        from sample.replay.replay_persist import (
            pack_tensor_checkpoint,
            serialize_persist_value,
            write_drbf,
        )

        target = Path(path)
        ensure_output_parent(target)
        work = Path(tempfile.mkdtemp(prefix=f".{target.name}.ckpt_", dir=str(target.parent)))
        blob = target.parent / f".{target.name}.tensors"
        try:
            with self._buffer_lock:
                info = self.tensor_pool.checkpoint(work)
                meta = {
                    "format_version": 1,
                    "sample_packaging": self.sample_packaging,
                    "agent_capacities": {
                        name: self._capacity_for_agent(name) for name in self.agent_names()
                    },
                    "strategy": str(self.config.strategy),
                    "batch_size": int(self.config.batch_size),
                    "recent_ratio": float(self.config.recent_ratio),
                    "rehearsal_sample_ratio": float(self._rehearsal_sample_ratio),
                    "tensor_kind": info.kind,
                    "tensor_logical_bytes": int(info.logical_bytes),
                    "tensor_files": list(info.relative_files),
                    "buffer": serialize_persist_value(self.buffer),
                    "rehearsal": serialize_persist_value(self._rehearsal),
                    "step_meta": serialize_persist_value(self.step_meta),
                    "tensor_schemas": serialize_persist_value(self.tensor_schemas),
                    "position": serialize_persist_value(dict(self.position)),
                    "write_count": serialize_persist_value(dict(self.write_count)),
                    "sample_seq": serialize_persist_value(dict(self.sample_seq)),
                    "next_step_meta_id": serialize_persist_value(dict(self._next_step_meta_id)),
                    "next_schema_id": int(self._next_schema_id),
                }
            pack_tensor_checkpoint(work, info, blob)
            written = write_drbf(target, meta, blob)
        finally:
            shutil.rmtree(work, ignore_errors=True)
            blob.unlink(missing_ok=True)
        LOG.info(
            f"ReplayBuffer snapshot saved: path={written}, "
            f"agents={list(self.buffer.keys())}, size={self.size}, "
            f"rehearsal_stages={list(self._rehearsal.keys())}, tensor_kind={info.kind}"
        )
        self.put_artifact_file(ArtifactType.REPLAY, str(written))
        return written

    def load_snapshot(self, path: str | Path) -> None:
        """
        从 DRBF 快照恢复元数据与张量池；重置 generation 与采样游标。
        """
        import shutil
        import tempfile

        from sample.replay.replay_persist import (
            deserialize_persist_value,
            ensure_local_snapshot,
            extract_drbf_tensors,
            unpack_tensor_checkpoint,
        )

        source = ensure_local_snapshot(Path(path), self.config)
        work = Path(tempfile.mkdtemp(prefix=f".{source.name}.restore_", dir=str(source.parent)))
        try:
            blob = work / "tensor_blob"
            meta, blob = extract_drbf_tensors(source, blob)
            format_version = int(meta.get("format_version", 0))
            if format_version != 1:
                raise ValueError(f"unsupported replay snapshot format_version={format_version}")
            # DRBF 存储布局与封装模式无关（buffer 是按写入顺序的 EpisodeMeta
            # 列表）；跨模式恢复时捆绑样本由“连续相同 episode_num”重新派生：
            # independent 快照 → episode_groups 运行得到同局捆绑；
            # episode_groups 快照 → independent 运行退化为逐条组轨迹。
            snapshot_packaging = str(meta.get("sample_packaging", "independent"))
            if snapshot_packaging != self.sample_packaging:
                LOG.warning(
                    "ReplayBuffer snapshot packaging differs from current config: "
                    f"snapshot={snapshot_packaging}, current={self.sample_packaging}; "
                    "bundles are re-derived from consecutive same-episode runs."
                )
            snapshot_kind = str(meta.get("tensor_kind", KIND_TORCH_SAVE))
            pool_kind = getattr(self.tensor_pool, "kind", KIND_TORCH_SAVE)
            if snapshot_kind != pool_kind:
                raise ValueError(
                    f"ReplayBuffer snapshot tensor_kind={snapshot_kind!r} does not match "
                    f"current tensor_pool.kind={pool_kind!r}; refuse loading mmap tensors into memory."
                )
            snapshot_capacities = meta.get("agent_capacities") or {}
            if not snapshot_capacities and "capacity" in meta:
                snapshot_capacities = {
                    name: int(meta["capacity"]) for name in meta.get("buffer", {})
                }
            for name, saved_capacity in snapshot_capacities.items():
                if int(saved_capacity) > self._capacity_for_agent(name):
                    raise ValueError(
                        f"ReplayBuffer snapshot capacity for {name!r} exceeds current capacity"
                    )
            restore_src = unpack_tensor_checkpoint(blob, work / "checkpoint", snapshot_kind)
            with self._buffer_lock:
                self._clear_all_unlocked()
                self.generation = 0
                self.buffer = deserialize_persist_value(meta.get("buffer", {}))
                self.episode_meta = self.buffer
                self._rehearsal = deserialize_persist_value(meta.get("rehearsal", {})) or {}
                self.step_meta = deserialize_persist_value(meta.get("step_meta", {}))
                self.tensor_schemas = deserialize_persist_value(meta.get("tensor_schemas", {}))
                self.position = deserialize_persist_value(meta.get("position", {}))
                self.write_count = deserialize_persist_value(meta.get("write_count", {}))
                self.sample_seq = deserialize_persist_value(meta.get("sample_seq", {}))
                self._next_step_meta_id = deserialize_persist_value(meta.get("next_step_meta_id", {}))
                self._next_schema_id = int(meta.get("next_schema_id", 1))
                self._schema_ids_by_key = {
                    (schema.template, schema.fields): schema_id
                    for schema_id, schema in self.tensor_schemas.items()
                }
                self._rehearsal_sample_ratio = float(meta.get("rehearsal_sample_ratio", 0.0))
                self.sliding_idx.clear()
                self.on_policy_cursor.clear()
                self.on_policy_step_cursor.clear()
                self.tensor_pool.restore(restore_src)
                if isinstance(self.tensor_pool, LazyMemmapStepTensorPool):
                    self.tensor_pool.reserve(
                        len(self.tensor_pool._schema_of_index)
                        + lazy_memmap_max_steps_of(self.config)
                    )
                self._unwrap_ring_snapshot_unlocked()
        finally:
            shutil.rmtree(work, ignore_errors=True)
        with self._size_changed_cv:
            self._size_changed_cv.notify()
        LOG.info(
            f"ReplayBuffer snapshot loaded: path={source}, "
            f"agents={list(self.buffer.keys())}, size={self.size}, generation=0, "
            f"rehearsal_stages={list(self._rehearsal.keys())}"
        )

    @process_close
    def close(self) -> None:
        """关闭张量池；默认删除本任务 mmap scratch，keep_scratch 时仅保留本地目录。

        给进程异常退出用；正常结束也会再调一次。S3 上传不在这里：业务快照由
        notify_stage_complete 在 save_replay_buffer 打开时完成。
        """
        with self._buffer_lock:
            if self._closed:
                return
            self._closed = True
            pool_scratch = getattr(self.tensor_pool, "created_scratch_dir", None)
            self.tensor_pool.close()
        self.complete_artifact_stage()
        if self.tensor_pool.kind != KIND_MMAP_DIR:
            return
        if (
            self.config.keep_scratch
            and pool_scratch is not None
            and _path_has_files(pool_scratch)
        ):
            LOG.info(f"ReplayBuffer mmap scratch kept: path={pool_scratch}")
            return
        if pool_scratch is not None:
            self._remove_mmap_scratch(pool_scratch)
        else:
            self._remove_empty_task_scratch()

    def _remove_empty_task_scratch(self) -> None:
        """从未写入 mmap 时删掉空的任务 scratch 目录（例如只 mkdir 就被杀掉留下的空壳）。"""
        import shutil

        task_scratch = resolve_mmap_scratch_dir(self.config) / str(task_id)
        if task_scratch.is_dir() and not _path_has_files(task_scratch):
            shutil.rmtree(task_scratch, ignore_errors=True)
            LOG.info(f"ReplayBuffer empty mmap scratch removed: path={task_scratch}")

    def _remove_mmap_scratch(self, scratch_dir: Path) -> None:
        """unmap 之后删除本任务的 mmap scratch 目录。"""
        import shutil

        task_scratch = scratch_dir.parent
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
        if task_scratch.exists():
            shutil.rmtree(task_scratch, ignore_errors=True)
        LOG.info(f"ReplayBuffer mmap scratch removed: path={task_scratch}")

    def load_replay_buffer(
            self,
            data_inter: Iterator[EpisodeTransition],
            cover_func: Optional[Callable[[Any], EpisodeTransition]] = None,
            notify_learning: bool = False,
            agent_name: Optional[AgentName] = None,
    ):
        """ 加载采样数据 """
        for raw_data in data_inter:
            if cover_func is not None:
                data = cover_func(raw_data)
            elif isinstance(raw_data, (EpisodeTransition, SAEpisodeTransition)):
                data = raw_data
            elif isinstance(raw_data, dict) and all(
                isinstance(value, (EpisodeTransition, SAEpisodeTransition))
                or (isinstance(value, list) and all(
                    isinstance(item, (EpisodeTransition, SAEpisodeTransition)) for item in value
                ))
                for value in raw_data.values()
            ):
                self.push(raw_data, save=False, notify_learning=notify_learning)
                continue
            else:
                data = self._cover_data_from_dict(raw_data)
            self.push(data, save=False, notify_learning=notify_learning, agent_name=agent_name)

    def _cover_data_to_dict(self, episode_data: EpisodeTransition) -> Dict:
        """
        【实现组件接口】将 EpisodeTransition 转换为完全可逆互换的明细 Dict。
        极致兼容：无损处理 NDArray、Tensor、NamedTuple、Slots 以及各种自定义 Action 包装对象。
        """
        return self._serialize_value(episode_data)

    def _cover_data_from_dict(self, raw_data: Dict) -> EpisodeTransition:
        """
        【实现组件接口】将明细 Dict 结构完美无损还原为原始的 EpisodeTransition 实例。
        与 _cover_data_to_dict 强对齐，确保空间状态、动作序列完全一致。
        """
        return self._deserialize_value(raw_data)

    def _serialize_value(self, val: Any) -> Any:
        """递归无损序列化核心引擎"""
        # 1. 处理 Numpy 数组
        if isinstance(val, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": val.tolist(),
                "dtype": str(val.dtype)
            }

        # 2. 处理 PyTorch 张量
        if isinstance(val, torch.Tensor):
            return {
                "__type__": "tensor",
                "data": val.detach().cpu().numpy().tolist(),
                "dtype": str(val.dtype)
            }

        # 3. 处理 NamedTuple (RL中常用于组织状态轨迹，继承自 tuple 但拥有 _asdict 接口)
        if hasattr(val, "_asdict") and isinstance(val, tuple):
            return {
                "__type__": "namedtuple",
                "__module__": val.__class__.__module__,
                "__class__": val.__class__.__name__,
                "data": {k: self._serialize_value(v) for k, v in val._asdict().items()}
            }

        # 4. 处理带常规 __dict__ 的自定义实体对象或 Dataclass (例如你五花八门的 Action 对象)
        if hasattr(val, "__dict__"):
            return {
                "__type__": "custom_object",
                "__module__": val.__class__.__module__,
                "__class__": val.__class__.__name__,
                "data": {k: self._serialize_value(v) for k, v in val.__dict__.items()}
            }

        # 5. 极端防御：处理采用了 __slots__ 内存优化的紧凑型对象
        if hasattr(val, "__slots__"):
            slots_data = {}
            for slot in val.__slots__:
                if hasattr(val, slot):
                    slots_data[slot] = self._serialize_value(getattr(val, slot))
            return {
                "__type__": "slots_object",
                "__module__": val.__class__.__module__,
                "__class__": val.__class__.__name__,
                "data": slots_data
            }

        # 6. 处理标准容器
        if isinstance(val, dict):
            return {k: self._serialize_value(v) for k, v in val.items()}

        if isinstance(val, list):
            return [self._serialize_value(v) for v in val]

        if isinstance(val, tuple):
            return {
                "__type__": "pure_tuple",
                "data": [self._serialize_value(v) for v in val]
            }

        # 7. 基础原生类型 (int, float, str, bool, None) 直接返回
        return val

    def _deserialize_value(self, val: Any) -> Any:
        """递归反序列化还原引擎"""
        if isinstance(val, dict):
            # 命中元数据控制标签，开启高级类型恢复
            if "__type__" in val:
                t = val["__type__"]

                if t == "ndarray":
                    return np.array(val["data"], dtype=val["dtype"])

                if t == "tensor":
                    # 从字符串平滑提取 PyTorch 数据类型，如 "torch.float32" -> torch.float32
                    dtype_str = val["dtype"].split(".")[-1]
                    torch_dtype = getattr(torch, dtype_str, torch.float32)
                    return torch.tensor(val["data"], dtype=torch_dtype)

                if t == "pure_tuple":
                    return tuple(self._deserialize_value(x) for x in val["data"])

                # 针对 NamedTuple、__dict__ 对象以及 __slots__ 对象的元编程恢复
                if t in ("custom_object", "namedtuple", "slots_object"):
                    module = importlib.import_module(val["__module__"])
                    cls = getattr(module, val["__class__"])

                    if t == "namedtuple":
                        # NamedTuple 是不可变的，必须通过解包构造
                        return cls(**{k: self._deserialize_value(v) for k, v in val["data"].items()})

                    elif t == "custom_object":
                        # 🌟 核心技巧：使用 __new__ 绕过复杂的 __init__ 校验或必填参数限制，实现纯净的状态灌入
                        obj = cls.__new__(cls)
                        data = {k: self._deserialize_value(v) for k, v in val["data"].items()}
                        if isinstance(obj, (EpisodeTransition, SAEpisodeTransition)):
                            obj.__setstate__(data)
                        else:
                            obj.__dict__.update(data)
                        return obj

                    elif t == "slots_object":
                        obj = cls.__new__(cls)
                        for k, v in val["data"].items():
                            setattr(obj, k, self._deserialize_value(v))
                        return obj

            # 普通字典递归处理
            return {k: self._deserialize_value(v) for k, v in val.items()}

        if isinstance(val, list):
            return [self._deserialize_value(v) for v in val]

        return val

    def _save_to_backend_storage(self, raw_data: Dict):
        pass

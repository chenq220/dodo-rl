from __future__ import annotations
import copy
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from config import (TaskConfig, TrainConfig, EvalConfig, LearnerConfig, AgentConfig,
                    AgentOutputType, AgentType, ResourceDir, SampleType)
from typing import BinaryIO, Deque, Dict, TYPE_CHECKING, Callable, Sequence, Literal, List, Tuple, Optional, Any
from common.registry import extensible
from scheduler.artifact import (
    Artifact,
    ArtifactPut,
    ArtifactType,
    ArchivedTask,
    ArtifactsManager,
    ensure_output_parent,
    get_active_artifacts_manager,
    build_artifacts_manager,
)
from common.exception import ArtifactsArchiveException, SampleLeaseException
from common.info_bridge import (
    info_bridge,
    public_training_info,
)
from common.logger import init_logger
from common.process_shutdown import process_close
from datetime import datetime
from scheduler.tasks.versioned_sampling import select_version_rule
from scheduler.tasks.learn_permit_policy import (
    LearnPermitGrant,
    LearnPermitPolicy,
    LearnPermitRequest,
    ReplayReadyLearnPermitPolicy,
    replay_buffer_ready_for_learner,
    replay_warmup_remaining,
)
from scheduler.model import (
    ActionParam,
    EpisodeInfo,
    LearnInfo,
    LearnTaskResult,
    AutoMLAdvice,
    StageLifecycleState,
    TrainingSnapshot,
    TrainingRunArchive,
    TrainingResourceCallback,
    TrainingResourceContext,
    TrainingResourceContribution,
    TrainingResourceEvent,
    FRAMEWORK_UPDATE_KEYS,
)
from scheduler.tasks.task_records import build_episode_rows, build_metric_output_rows
from scheduler.tasks.training_progress import (
    publish_dispatch_progress,
    publish_training_progress,
)
from sample.model import ROLLING_STRATEGIES, ReplayBufferState, SamplingStrategy
from utils.grahp_drawer import (save_end_eval_reward_bars, save_reward_curve, save_reward_details,
                               save_reward_curve_csv, save_reward_details_csv)
from utils.random_seed import derive_seed, derive_generation_run_seed, normalize_seed
from utils.task_id import task_id

LOG = init_logger('train_task')
MIN_REWARD_CURVE_EPISODES = 2

# 统一租约 kind：回报路由与超时回收策略的唯一依据。
LEASE_KIND_SAMPLE = "sample"        # 训练/评估主路局
LEASE_KIND_MID_EVAL = "mid_eval"    # 盘中评估旁路（仅 TrainConfig）
LEASE_KIND_END_EVAL = "end_eval"    # 终局评估旁路（仅 TrainConfig）

LEASE_KINDS = (LEASE_KIND_SAMPLE, LEASE_KIND_MID_EVAL, LEASE_KIND_END_EVAL)


@dataclass
class SampleLeaseEntry:
    """一张在途采样租约。签发时登记，回报/回收时移除；租约表是唯一在途视图。"""
    kind: str             # LEASE_KINDS 之一
    episode_num: int      # 签发时令牌序号：主路=局号；mid_eval=里程碑；end_eval=max_episode
    generation: int       # 签发时 TM.generation；跨代回报按迟到丢弃
    issued_at: datetime   # 签发时刻；超时回收判据；accept 时刷新为活动心跳
    dispatch_count: int   # 旁路第几次签发（重试/补发轮次）；主路恒 0
    # 领取版本令牌的 worker，同一 worker 同时只能占用一张。
    worker_id: str = ""


if TYPE_CHECKING:
    from common.types import AgentName, ModelName
    from scheduler.automl import AutoML

ModelNamePrefix = Literal["optimo", "final", "pruned"]


def format_model_version(
        name_prefix: ModelNamePrefix,
        task_id: str,
        stage_name: str,
        generation: int,
) -> str:
    """生成一次模型快照的公共版本：prefix_task_stage_generation。"""
    safe_task_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(task_id)).strip("_") or "task"
    safe_stage_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(stage_name)).strip("_") or "stage"
    return f"{name_prefix}_{safe_task_id}_{safe_stage_name}_{int(generation)}"


def format_model_filename(
        name_prefix: ModelNamePrefix,
        agent_name: AgentName,
        stage_name: str,
        model_name: ModelName,
        generation: int,
        task_id: str,
) -> str:
    """生成标准模型文件名；TrainTaskManager.resolve_model_filename 与离线 eval 共用。"""
    version = format_model_version(name_prefix, task_id, stage_name, generation)
    safe_agent_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(agent_name)).strip("_") or "agent"
    safe_model_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(model_name)).strip("_") or "model"
    return f"{version}_{safe_agent_name}_{safe_model_name}.pth"


@extensible
class TrainTaskManager(object):
    """
    训练任务调度器，集中协调在线训练采样 episode、主训练学习令牌和任务进度。

    该类当前是本地内存实现，但方法语义按未来 RPC 服务设计：采样进程申请 episode，
    学习进程申请 LearnInfo 学习令牌。历史样本学习由 SampleLearnManager 独立负责。
    """

    def __init__(self, config: TaskConfig,
                 policy_agent_configs: List[AgentConfig], learner_configs: List[LearnerConfig],
                 stage_name: str = "train",
                 env_name: str = "",
                 auto_ml: Optional["AutoML"] = None,
                 artifacts_manager: Optional[ArtifactsManager] = None,
                 *, supports_versioned_sampling: bool = False):
        """

        :param config: 任务配置
        :param policy_agent_configs:
        :param learner_configs:
        :param stage_name: 当前训练阶段名称。普通训练（无StageTask）默认就是第一阶段 train；
            分阶段训练时由 StageTask 按 StageConfig.stage_name 为每个阶段实例化独立的 TrainTaskManager。
        :param env_name: 环境名称（EnvConfig.sense_name），用于 AutoML 归档目录命名。
        :param auto_ml: 可选 AutoML 实例；为 None 时不启监控线程，所有 AutoML 流程跳过。
        :param artifacts_manager: 产物台账；默认取进程内 active Manager。
        """
        # 任务总线主锁；以 _locked 结尾的方法要求调用方已持有它。
        self._lock = threading.RLock()
        self.artifacts_manager = artifacts_manager or get_active_artifacts_manager()
        if self.artifacts_manager is None:
            self.artifacts_manager = build_artifacts_manager(
                artifacts_archive_save_dir=config.artifacts_archive_save_dir,
                activate=True,
            )
        self._init_identity(
            config=config,
            policy_agent_configs=policy_agent_configs,
            learner_configs=learner_configs,
            stage_name=stage_name,
            env_name=env_name,
        )
        self._versioned_sampling = (
            supports_versioned_sampling and isinstance(config, TrainConfig)
            and config.versioned and config.sample_type == SampleType.ONLINE.value
        )
        self._version_namespace = uuid.uuid4().hex if self._versioned_sampling else ""
        self._version_requests = {}
        self._overlap_learners: frozenset[str] = frozenset()
        self._version_rules = {}
        self._version_round: dict[str, int] = {}
        self._version_sealed: set[str] = set()
        self._version_outstanding: set[int] = set()
        self._load_version_rules()
        self._plan_artifact_task()
        self._init_artifact_completion()
        self._init_automl(auto_ml)
        # 学习许可策略只判断当前缓存能否学习，不建立逐 episode 的学习债务。
        self._learn_permit_policy: LearnPermitPolicy = self.build_learn_permit_policy()
        if isinstance(config, TrainConfig):
            if config.sample_type not in {SampleType.ONLINE.value, SampleType.OFFLINE.value}:
                raise ValueError(
                    f"TrainConfig.sample_type must be online/offline, got={config.sample_type!r}"
                )
            if config.sample_type == SampleType.OFFLINE.value and (
                    config.mid_eval_enabled or int(config.end_eval_episodes) != 0
            ):
                raise ValueError("offline training requires mid_eval_enabled=false and end_eval_episodes=0")
            self._ensure_run_seed_locked()
            self._reset_sample_state()
            self._reset_learn_state()
            interval = int(config.mid_eval_episode_interval)
            if interval > 0 and not bool(config.mid_eval_enabled):
                LOG.warning(
                    f"mid_eval_episode_interval={interval} but mid_eval_enabled=false: "
                    f"mid-eval GIF will not run and will not appear in AutoML archives "
                    f"(stage={stage_name}, task_id={self.task_id})"
                )
        elif isinstance(config, EvalConfig):
            if config.sample_type != SampleType.ONLINE.value:
                raise ValueError(
                    "EvalConfig.sample_type must be online, "
                    f"got={config.sample_type!r}"
                )
            self._reset_sample_state()
            self._reset_learn_state()
            interval = int(config.render_gif_episode_interval)
            if interval > 0 and not bool(config.render_gif_enabled):
                LOG.warning(
                    f"render_gif_episode_interval={interval} but render_gif_enabled=false: "
                    f"eval GIF will not run "
                    f"(stage={stage_name}, task_id={self.task_id})"
                )
        else:
            raise ValueError(f'Config {config} not supported')
        self._init_metrics_state()
        self._publish_training_progress_locked()
        self._start_monitors()

    def _init_identity(
            self,
            *,
            config: TaskConfig,
            policy_agent_configs: List[AgentConfig],
            learner_configs: List[LearnerConfig],
            stage_name: str,
            env_name: str,
    ) -> None:
        # 当前阶段任务配置（TrainConfig 或 EvalConfig）。
        self.task_config: TaskConfig = config
        assert self.task_config.sample_task_timeout > 0, (
            "TaskConfig.sample_task_timeout must be greater than 0."
        )
        # 当前训练阶段名；普通训练默认 train，分阶段时由 StageTask 注入。
        self.stage_name = stage_name
        # 环境名（EnvConfig.sense_name），供 AutoML 归档目录命名。
        self.env_name = env_name
        # 本阶段参与采样的 Agent 配置全集。
        self.policy_agent_configs = policy_agent_configs
        # 本阶段上场的 Learner 配置；决定学习令牌候选与算法快照。
        self.learner_configs = learner_configs
        # True 时奖励曲线 / AutoML 使用全部采样 Agent 的回报，而不仅是 Learner。
        self.monitor_all_agent_rewards = False
        # 本任务实例 ID，令牌、模型文件名和归档共用。
        self.task_id = str(task_id)
        # 任务启动时间戳，奖励曲线等产物文件名使用。
        self.start_timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
        # 本阶段计划采样局数，由 _gen_*_sample_task_seq 写入。
        self.max_episode: int
        self._offline_plan_prepared = False
        self._offline_valid_episode_count = 0
        # 当前代次；每次 restart_training_task 加一，令牌与 ReplayBuffer 按它对齐。
        self.generation = 0
        # close() 置位后后台监控线程在下一轮检查时退出。
        self._closed = False
        # Role 注册的代次同步回调；restart 在释放锁后逐个调用。
        self._restart_callbacks = []
        # 已请求但尚未兑现的 restart 原因；非 None 表示挂起中。
        self._pending_restart_reason: Optional[str] = None
        # 与挂起 restart 绑定的 AutoML 建议，兑现时写入超参与 seed。
        self._pending_restart_advice: Optional[AutoMLAdvice] = None
        # 当前代次已生效的环境 metadata 覆盖，写入训练快照供 AutoML 观测。
        self._current_env_metadata_overrides: Dict[str, Any] = {}
        # Role 注册的模型落盘 caller；save_models(prefix) 经此触发。
        self._learner_save_caller: Optional[Callable[[str], List[str]]] = None
        # 训练资源生命周期回调（归档前贡献产物 / 重启后扩展）。
        self._training_resource_callbacks: List[TrainingResourceCallback] = []
        # 本 generation 任务级随机种子，供 episode / learn 令牌派生。
        self._generation_run_seed: Optional[int] = None
        # 网络初始化种子；用户未指定 initialization_seed 时回退到 config.seed。
        self._initialization_seed: Optional[int] = (
            normalize_seed(config.initialization_seed)
            if isinstance(config, TrainConfig) and config.initialization_seed is not None
            else normalize_seed(config.seed) if isinstance(config, TrainConfig) else None
        )
        # True 表示用户固定了 config.seed，restart 时不再派生新的 run seed。
        self._user_fixed_seed = (
                isinstance(config, TrainConfig) and normalize_seed(config.seed) is not None
        )
        # 上次已发布的 (stage, episode) 进度，避免重复写进度文件。
        self._last_published_progress: Optional[Tuple[str, str]] = None
        # 最近一次采样准入闸门的阻塞原因；非 None 时 sample_backpressure_state.enabled=True。
        self._admission_block_reason: Optional[str] = None
        # 奖励曲线 GIF 的打开文件句柄；跨局原地覆盖同一路径。
        self._reward_curve_file: Optional[BinaryIO] = None
        # 上述句柄对应的落盘路径。
        self._reward_curve_file_path: Optional[str] = None
        # 奖励曲线文件读写锁，绘图线程与主路径共用。
        self._reward_curve_file_lock = threading.Lock()
        # 奖励构成图 GIF 的打开文件句柄；环境不支持构成时关闭并删除。
        self._reward_details_file: Optional[BinaryIO] = None
        # 上述句柄对应的落盘路径。
        self._reward_details_file_path: Optional[str] = None
        # 奖励构成图文件读写锁。
        self._reward_details_file_lock = threading.Lock()
        # 采样令牌超时扫描的下一轮 Timer；close() 时取消。
        self._sample_task_monitor_timer: Optional[threading.Timer] = None
        # 奖励曲线后台刷新线程。
        self._reward_curve_thread: Optional[threading.Thread] = None
        # 子类覆盖 custom_monitor 后才启动的定制轮询线程。
        self._custom_monitor_thread: Optional[threading.Thread] = None
        # AutoML.advice() 监控线程；未注入 auto_ml 时不启动。
        self._auto_ml_thread: Optional[threading.Thread] = None

    def _init_artifact_completion(self) -> None:
        """只初始化结束状态；产物计划和原始 TOML 已由入口写入共享台账。"""
        self._automl_archive_paths: list[str] = []
        self._task_completion_done = False
        self._task_completion_in_progress = False
        self._task_completion_result: Optional[ArchivedTask] = None
        self._task_completion_condition = threading.Condition(self._lock)

    def _init_automl(self, auto_ml: Optional["AutoML"]) -> None:
        # 可选 AutoML 实例；为 None 时跳过监控、判决和归档。
        self._auto_ml = auto_ml
        # AutoML 监控线程捕获的致命异常；下一次外部 tick 再抛出。
        self._auto_ml_fatal_error: Optional[BaseException] = None
        # 本阶段已落盘归档包序号，写入 TrainingRunArchive.archive_seq。
        self._archive_seq = 0
        # 已经归档过的 generation 集合，保证每代次只落盘一次。
        self._archived_generations: set[int] = set()
        if auto_ml is not None:
            assert isinstance(self.task_config, TrainConfig), (
                f"auto_ml requires TrainConfig, given={type(self.task_config).__name__}"
            )
            self.add_training_resource_callback(auto_ml.training_resource_callback)

    def _reset_sample_state(self) -> None:
        # ReplayBuffer 最新状态快照；TM 不持有 Buffer 本身，只消费这张状态。
        self.replay_buffer_state = ReplayBufferState(generation=self.generation)
        # 主路采样环形扫描游标，指向下一次优先检查的下标。
        self.current_sample_task = 0
        # 已成功完成的主路采样局数，是进度、背压和阶段结束的分子。
        self.sample_task_completed = 0
        self._dispatched_sample_episodes: set[int] = set()
        # 上次已触发的盘中评估里程碑局号；用于跨过整点时不漏触发。
        self._last_mid_eval_train_episode = 0
        # 本代次已签发的盘中评估次数，只用于生成 record_name，不写入令牌来源字段。
        self._mid_eval_dispatch_count = 0
        # 本代次已签发的结束评估次数。
        self._end_eval_dispatch_count = 0
        # 本代次已成功完成的结束评估局数；达到 end_eval_episodes 后才允许采样收口。
        self._end_eval_completed_count = 0
        # 结束评估逐局奖励构成，按下标 0..N-1 对齐柱状图 X 轴。
        self._end_eval_reward_details: List[Dict[str, float]] = []
        # 本代次采样收口是否已结算（AutoML 判决 / 归档只做一次）。
        self._sample_completion_settled = False
        # 旁路采样 FIFO；优先级高于主路，且不受准入闸门与背压约束。
        self._bypass_sample_queue: Deque[Tuple[EpisodeInfo, str, int]] = deque()
        # 统一租约表：lease_id → entry。主路与旁路共用，是唯一在途令牌视图。
        self._lease_table: Dict[str, SampleLeaseEntry] = {}
        # 迟到/未知租约回报的丢弃计数（超时回收、换代清场后到达）。
        self._stale_report_count = 0
        self._admission_block_reason = None
        if isinstance(self.task_config, TrainConfig):
            # 离线样本数量须由中心 MC 索引决定；阶段 ready 前保持不可领取。
            if self.task_config.sample_type == SampleType.OFFLINE.value:
                if self._offline_valid_episode_count > 0:
                    self.sample_task_seq = self._gen_train_sample_task_seq(self._offline_valid_episode_count)
                    self._offline_plan_prepared = True
                else:
                    self.sample_task_seq = []
                self.max_episode = len(self.sample_task_seq)
            else:
                self.sample_task_seq = self._gen_train_sample_task_seq()
        elif isinstance(self.task_config, EvalConfig):
            self.sample_task_seq = self._gen_eval_sample_task_seq()
        else:
            self.sample_task_seq = []
            self.max_episode = 0

    def _reset_learn_state(self) -> None:
        # 已完成的 Learner.update 次数，只作实际进度，不再与采样计划构成固定配额。
        self.learning_completed = 0
        # 当前是否存在一张可领取许可；整个任务总线最多保留一张，避免形成学习债务。
        self._learn_ready = False
        self._learn_ready_active_learners: list[AgentName] = []
        self.learn_count = 0
        # 当前仅允许一张 in-flight；learn_id 独立于采样 episode。
        self._next_learn_id = 1
        self._inflight_learn_tokens: Dict[int, LearnInfo] = {}
        self._completed_learn_ids: set[int] = set()
        # 同一采样水位若因学习期间又 push 而再发令牌，首次保留原 episode seed，后续追加子序号。
        self._learn_seed_watermark: Optional[int] = None
        self._learn_seed_iteration = 0
        # 主路采样关闭后 learn_batches_update 不再置 ready；最后一次 push 已置 ready 时仍允许消费一次尾部许可。
        self._sample_closed = False
        # 最近一次 ReplayBuffer 更新可覆盖的采样水位，以及已完成学习覆盖到的水位。
        self._latest_replay_sample_watermark = 0
        self._learned_sample_watermark = 0
        # 按 Learner 拆开的已完成 / 已派发更新次数，供观测。
        self.learner_state: dict[AgentName, dict] = {
            learner_config.name: {
                "learning_completed": 0,  # 该 Learner 已完成的 update 次数
                "dispatched_learning_count": 0,  # 已派发令牌累计的 learn_times
            }
            for learner_config in self.learner_configs
        }
        # 各 Learner 上报的列式学习指标，归档 algos_metrics 时按 episode 对齐。
        self.learn_results: Dict[AgentName, Dict[str, List[Any]]] = {
            learner_config.name: {"episode_num": []}
            for learner_config in self.learner_configs
        }

    def _init_metrics_state(self) -> None:
        # 本代次已保存的 checkpoint / optimo / pruned 模型路径，随归档包带走。
        self._checkpoint_model_paths: List[str] = []
        # 本代次盘中评估 GIF 路径，写入 snapshot / archive 的 mid_eval_gifs。
        self._mid_eval_gif_paths: List[str] = []
        # 本代次结束评估 GIF 路径，写入 snapshot / archive 的 end_eval_gifs。
        self._end_eval_gif_paths: List[str] = []
        self._bypass_episode_artifacts: list[Artifact] = []
        # 结束评估奖励堆叠柱状图路径；end_eval_episodes=0 时保持为空。
        self._last_end_eval_reward_path: Optional[str] = None
        # 最近一次写出的奖励曲线路径，供归档引用。
        self._last_reward_curve_path: Optional[str] = None
        # 最近一次写出的奖励构成图路径，供归档引用。
        self._last_reward_details_path: Optional[str] = None
        # 本 generation 历史最佳团队回报；用于决定是否落盘 optimo。
        self._best_team_reward = float("-inf")
        # _best_team_reward 所属 generation；换代后必须重新起算。
        self._best_team_reward_generation = -1
        # AutoML 存量资源回调在边界转换后，内部只保存产物契约。
        self._archive_contributed_artifacts: list[Artifact] = []
        # 上次因 AutoML 触发 restart 时的完成局数；restart 后必须归零。
        self._last_automl_restart_episode = 0
        # 上次调用 AutoML.advice() 时的完成局数，控制 poll_interval_episodes。
        self._last_automl_check_episode = 0
        # 全部采样 Agent 的逐局回报表，下标 = episode_num - 1。
        self.sample_rewards: Dict[AgentName, List[Optional[float]]] = (
            self._new_sample_rewards_store()
        )
        # 仅 Learner 对应 Agent 的逐局回报表，默认给奖励曲线 / AutoML 使用。
        self.learn_sample_rewards: Dict[AgentName, List[Optional[float]]] = (
            self._new_learn_sample_rewards_store()
        )
        # 环境上报的逐局奖励构成；None 表示该局未采集或能力已关闭。
        self.episode_rewards_details: List[Optional[Dict[str, float]]] = (
            self._new_episode_rewards_details_store()
        )
        # 采样进程随 report_sample_result 上报的主路产物。
        self.episode_artifacts: list[list[Artifact]] = (
            self._new_episode_artifacts_store()
        )
        # 奖励构成观测能力：未知 / 支持 / 不支持。不支持后不再向环境取值。
        self._reward_details_capability: Literal["unknown", "supported", "unsupported"] = "unknown"
        # 业务指标逐局序列，键为 metric 名，供 AutoML metric_outputs 归档。
        self.sample_metrics: Dict[str, List[Optional[float]]] = {}
        # 奖励曲线线程上次刷新时对应的 sample_task_completed，避免无进展空转绘图。
        self._reward_curve_last_completed = 0

    def _reset_metrics_state(self) -> None:
        self._init_metrics_state()

    def _start_monitors(self) -> None:
        self._schedule_sample_task_timeout_monitor()
        if self._reward_curve_thread is None or not self._reward_curve_thread.is_alive():
            self._reward_curve_thread = threading.Thread(
                target=self._reward_curve_monitor_thread,
                name="train_task_reward_curve_monitor",
                daemon=True,
            )
            self._reward_curve_thread.start()
        if self._auto_ml is not None and (
                self._auto_ml_thread is None or not self._auto_ml_thread.is_alive()
        ):
            self._auto_ml_thread = threading.Thread(
                target=self._auto_ml_monitor_loop,
                name=f"automl_monitor_{self.stage_name}",
                daemon=True,
            )
            self._auto_ml_thread.start()
        if self._custom_monitor_enabled():
            if self._custom_monitor_thread is None or not self._custom_monitor_thread.is_alive():
                self._custom_monitor_thread = threading.Thread(
                    target=self._custom_monitor_loop,
                    name=f"custom_monitor_{self.stage_name}",
                    daemon=True,
                )
                self._custom_monitor_thread.start()

    def _custom_monitor_enabled(self) -> bool:
        return (
                type(self).custom_monitor is not TrainTaskManager.custom_monitor
                and float(self.custom_monitor_interval_seconds()) > 0
        )

    def _custom_monitor_loop(self) -> None:
        """锁外周期调用 custom_monitor；异常记堆栈后继续，不拖垮训练。"""
        while True:
            with self._lock:
                if self._closed:
                    return
                interval = float(self.custom_monitor_interval_seconds())
            if interval <= 0:
                return
            try:
                self.custom_monitor()
            except Exception:
                LOG.exception(
                    f"custom_monitor failed: stage_name={self.stage_name}, task_id={self.task_id}"
                )
            time.sleep(interval)

    def _new_sample_rewards_store(self) -> Dict[AgentName, List[Optional[float]]]:
        return {
            agent_config.name: [None] * len(self.sample_task_seq)
            for agent_config in self.policy_agent_configs
        }

    def _new_learn_sample_rewards_store(self) -> Dict[AgentName, List[Optional[float]]]:
        return {
            learner_config.name: [None] * len(self.sample_task_seq)
            for learner_config in self.learner_configs
        }

    def _new_episode_rewards_details_store(self) -> List[Optional[Dict[str, float]]]:
        return [None] * len(self.sample_task_seq)

    def _new_episode_artifacts_store(self) -> list[list[Artifact]]:
        return [[] for _ in self.sample_task_seq]

    def _publish_training_progress_locked(self) -> None:
        """发布当前阶段和局数。调用方必须持有 ``self._lock``，或处于构造阶段。"""
        stage = str(self.stage_name).replace("\r", " ").replace("\n", " ")
        episode = f"{self.sample_task_completed}/{self.max_episode}"
        self._last_published_progress = publish_training_progress(
            stage=stage,
            episode=episode,
            last_published=self._last_published_progress,
        )
        self._record_progress()
        self._publish_dispatch_progress_locked()

    def _publish_dispatch_progress_locked(self) -> None:
        if isinstance(self.task_config, TrainConfig):
            publish_dispatch_progress(
                stage=str(self.stage_name),
                generation=int(self.generation),
                sample_dispatched=len(self._dispatched_sample_episodes),
                sample_completed=self.sample_task_completed,
                learn_dispatched=self._next_learn_id - 1,
                sample_total=self.max_episode,
            )

    def _downstream_sample_rewards(self) -> Dict[AgentName, List[Optional[float]]]:
        """奖励曲线 / AutoML / 归档 metrics 使用的奖励表。"""
        if self.monitor_all_agent_rewards:
            return self.sample_rewards
        if self.learn_sample_rewards:
            return self.learn_sample_rewards
        return self.sample_rewards

    def _metric_outputs_payload(self) -> Dict[str, Dict[str, Any]]:
        """把配置对象转换成可经本地/RPC sample令牌传输的普通字典。"""
        eval_outputs = getattr(self.task_config, "metric_outputs", None)
        if isinstance(self.task_config, EvalConfig) and eval_outputs:
            payload: Dict[str, Dict[str, Any]] = {}
            for name, output in eval_outputs.items():
                payload[name] = output.to_dict() if hasattr(output, "to_dict") else dict(output)
            return payload
        if self._auto_ml is None:
            return {}
        payload = {
            name: output.to_dict()
            for name, output in self._auto_ml.config.metric_outputs.items()
        }
        # 内置成功率也从 PettingZoo infos 采集；不再调用环境私有 get_episode_metrics。
        if self._auto_ml.config.metric_primary == "stage_success_rate":
            payload.setdefault("stage_success", {
                "info_path": "stage_success",
                "metadata_path": None,
                "agent_reduce": "max",
                "episode_reduce": "latest",
            })
        return payload

    def _gen_eval_sample_task_seq(self) -> List[EpisodeInfo]:
        """生成评估采样任务序列；评估不产生学习令牌，只按固定策略跑 episode。"""
        assert isinstance(self.task_config, EvalConfig), (
            f"_gen_eval_sample_task_seq requires EvalConfig, given={type(self.task_config).__name__}"
        )
        seq: List[EpisodeInfo] = []
        # IDEA 现在的的动作参数衰减是全局同步衰减的
        #   如果是顺序学习，每个智能体组是否要做自己独立的衰减序列？或者将改learning_compose_rotation_time改小，缩小轮流频度？
        #   chatgpt建议：每个智能体组维护独立的衰减序列。
        for ep in range(self.task_config.max_episode):
            episode_num = ep + 1
            agent_action_params: Dict[AgentName, ActionParam] = self._build_episode_action_params(
                epsilon=0,
                deterministic=self.task_config.deterministic,
                temperature=1.0,
            )
            ep_info = EpisodeInfo(
                episode_num=episode_num,
                task_id=self.task_id,
                epsilon=0,
                deterministic=self.task_config.deterministic,
                temperature=1.0,
                actions_param=agent_action_params,
                seed=self._episode_seed(episode_num),
                start_datetime=None,
                end_datetime=None,
                metadata=self._eval_record_metadata(episode_num),
                env_metadata=copy.deepcopy(self.task_config.env_metadata),
                metric_outputs=self._metric_outputs_payload(),
                stage_name=self.stage_name,
                generation=self.generation,
                sample_type=self.task_config.sample_type,
            )
            self.custom_episode(episode_info=ep_info)
            seq.append(ep_info)
        self.max_episode = self.task_config.max_episode
        return seq

    def _eval_record_metadata(self, episode_num: int) -> Optional[Dict[str, Any]]:
        """里程碑评估局写入 GIF 录制执行参数；非里程碑返回 None，rollout 默认不录。"""
        if not self._should_record_eval_gif(episode_num):
            return None
        return {
            "record_render": True,
            "record_output_dir": self.task_config.render_video_dir,
            "record_fps": self.task_config.render_gif_fps,
            "record_frame_stride": self.task_config.render_gif_frame_stride,
            "record_max_frames": self.task_config.render_gif_max_frames,
            "record_name": self._eval_record_name(episode_num=episode_num),
        }

    def _should_record_eval_gif(self, episode_num: int) -> bool:
        if not isinstance(self.task_config, EvalConfig):
            return False
        if not bool(self.task_config.render_gif_enabled):
            return False
        interval = int(self.task_config.render_gif_episode_interval)
        if interval <= 0:
            return False
        return episode_num % interval == 0

    def _eval_record_name(self, *, episode_num: int) -> str:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S")
        safe_task_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(self.task_id)).strip("_") or "task"
        safe_stage_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(self.stage_name)).strip("_") or "eval"
        return (
            f"eval_{safe_task_id}_{timestamp}_{safe_stage_name}_"
            f"episode_{int(episode_num):04d}.gif"
        )

    def _gen_train_sample_task_seq(self, episode_count: Optional[int] = None) -> List[EpisodeInfo]:
        """生成在线训练采样任务序列；每个EpisodeInfo就是采样进程领取的一局任务。"""
        assert isinstance(self.task_config, TrainConfig), (
            f"_gen_train_sample_task_seq requires TrainConfig, given={type(self.task_config).__name__}"
        )
        seq: List[EpisodeInfo] = []

        epsilon_warmup_episodes = self._epsilon_warmup_episodes()
        episode_count = int(self.task_config.max_episode if episode_count is None else episode_count)
        for ep in range(1, episode_count + 1):
            # epsilon 在单个阶段内只做一次线性衰减；分阶段训练中每个阶段是独立的 TrainTaskManager，
            # 因此各阶段都从本阶段第 1 局重新衰减，不需要额外的全局重置周期。
            epsilon_ep = ep
            # 1. 计算 epsilon 线性衰减
            if epsilon_ep <= epsilon_warmup_episodes:
                ep_epsilon = 1
            elif epsilon_ep < self.task_config.epsilon_decay_episodes:
                ep_epsilon = self.task_config.start_epsilon - (
                        epsilon_ep / self.task_config.epsilon_decay_episodes
                ) * (
                                     self.task_config.start_epsilon - self.task_config.end_epsilon)
            else:
                ep_epsilon = self.task_config.end_epsilon

            ep_epsilon = max(self.task_config.end_epsilon, min(self.task_config.start_epsilon, ep_epsilon))

            # 2. 计算 temperature 阶跃变化
            if ep <= self.task_config.temperature_episodes:
                ep_temperature = self.task_config.temperature
            else:
                ep_temperature = 1.0

            ep_info = EpisodeInfo(
                episode_num=ep,
                task_id=self.task_id,
                epsilon=ep_epsilon,
                deterministic=False,
                temperature=ep_temperature,
                actions_param=self._build_episode_action_params(
                    epsilon=ep_epsilon,
                    deterministic=False,
                    temperature=ep_temperature,
                ),
                seed=self._episode_seed(ep),
                start_datetime=None,
                end_datetime=None,
                metadata=None,
                env_metadata=copy.deepcopy(self.task_config.env_metadata),
                metric_outputs=self._metric_outputs_payload(),
                stage_name=self.stage_name,
                generation=self.generation,
                sample_type=SampleType.ONLINE.value if isinstance(self.task_config, EvalConfig)
                else self.task_config.sample_type,
            )
            self.custom_episode(episode_info=ep_info)
            seq.append(ep_info)
        self.max_episode = episode_count
        return seq

    def prepare_offline_sample_plan(self, valid_episode_count: int) -> None:
        """在阶段 ready 前按 MC 索引的有效轨迹数生成主路令牌。"""
        if not isinstance(self.task_config, TrainConfig) or self.task_config.sample_type != SampleType.OFFLINE.value:
            raise ValueError("prepare_offline_sample_plan requires an offline TrainConfig")
        if int(valid_episode_count) <= 0:
            raise ValueError("valid_episode_count must be positive")
        with self._lock:
            if self._offline_plan_prepared:
                if self.max_episode != int(valid_episode_count):
                    raise ValueError("offline sample plan cannot be replaced with a different count")
                return
            self.sample_task_seq = self._gen_train_sample_task_seq(int(valid_episode_count))
            self._offline_plan_prepared = True
            self._offline_valid_episode_count = int(valid_episode_count)
            self._reset_metrics_state()
            self._reset_learn_state()
            self._publish_training_progress_locked()

    def get_sample_task_info(self, **kwargs) -> Optional[EpisodeInfo]:
        """优先发放旁路令牌，主路令牌经过准入和背压检查。"""
        self._raise_auto_ml_fatal_error()
        worker_id = str(kwargs.get("sample_role_name") or kwargs.get("worker_id") or "")
        with self._lock:
            self._refill_bypass_producers_locked()
            bypass_task = self._pop_bypass_task_locked()
            if bypass_task is not None:
                return bypass_task
            sample_completed = int(self.sample_task_completed)
            generation = int(self.generation)

        # 准入闸门是唯一锁外判断，可能发起网络调用。
        admission_reason = self.check_sample_admission(sample_completed, generation)
        with self._lock:
            self._admission_block_reason = admission_reason
            if admission_reason or self._sample_backpressure_state_locked()["enabled"]:
                return None
            return self._claim_main_sample_task_locked(worker_id)

    def _claim_main_sample_task_locked(self, worker_id: str) -> Optional[EpisodeInfo]:
        """持锁挑选主路任务、绑定模型版本并签发租约。"""
        required_versions = None
        if self._versioned_sampling:
            if worker_id and any(entry.worker_id == worker_id for entry in self._lease_table.values()):
                return None
            required_versions = self._versions_for_sample_ticket_locked()
            if required_versions is None:
                return None

        task = self._next_available_sample_task_locked()
        if task is None:
            return None
        from common.plugins import RUNTIME_PLUGINS

        requests = {}
        required_requests = {}
        for learner in self.learner_configs:
            if not RUNTIME_PLUGINS.has_resource("learner", learner.algo_type):
                continue
            learner_class = RUNTIME_PLUGINS.resolve("learner", learner.algo_type)
            schema = getattr(learner_class, "batch_schema", None)
            if schema is not None:
                names = set(schema.sample_sources()) - {"behavior_log_probs", "behavior_temperature"}
                if names:
                    requests[learner.name] = sorted(names)
                    required_requests[learner.name] = sorted(
                        set(schema.sample_sources(required_only=True)) - {"behavior_log_probs", "behavior_temperature"}
                    )
        if requests:
            task.metadata = {
                **(task.metadata or {}),
                "batch_sample_sources": requests,
                "batch_required_sources": required_requests,
            }
        if required_versions is not None:
            if task.required_model_versions:
                # During a retry, keep still-current and preissued versions.
                # An already superseded snapshot may have been pruned.
                for name, current in self._current_sample_versions_locked().items():
                    future = self._sampling_version(self._version_round[name] + 1)
                    if task.required_model_versions[name] not in (current, future):
                        task.required_model_versions[name] = current
            else:
                task.required_model_versions = required_versions
                self._version_outstanding.add(task.episode_num)
        self._issue_lease_locked(task, kind=LEASE_KIND_SAMPLE, worker_id=worker_id)
        self._record_episode_claimed(task)
        token = copy.deepcopy(task)
        self._dispatched_sample_episodes.add(task.episode_num)
        self._publish_dispatch_progress_locked()
        return token

    def check_sample_admission(self, completed_episodes: int, generation: int) -> Optional[str]:
        """采样准入闸门。返回 None 放行，返回字符串作为阻塞原因。默认恒定放行。"""
        return None

    def enqueue_bypass_sample_task(self, episode_info: EpisodeInfo, *, kind: str) -> None:
        """把一张旁路采样令牌插入优先队列。本次只支持 replay_write=False。"""
        metadata = episode_info.metadata or {}
        assert metadata.get("replay_write") is False, (
            "bypass sample tokens must set metadata['replay_write']=False"
        )
        with self._lock:
            self._enqueue_bypass_locked(episode_info, kind=kind)

    def build_learn_permit_policy(self) -> LearnPermitPolicy:
        """学习许可策略注入点；策略只判断当前 ReplayBuffer 能否执行一次更新。"""
        return ReplayReadyLearnPermitPolicy()

    def _load_version_rules(self) -> None:
        """按 Learner 当前抽样契约，选择版本令牌的发放规则。"""
        if not self._versioned_sampling:
            return
        requests = {}
        rules = {}
        for config in self.learner_configs:
            algo = config.algo_config
            if not algo.learn_times_per_episode:
                continue
            if algo.learn_times_per_episode != 1 or algo.batch_size < 1:
                raise ValueError("versioned sampling requires one learning batch per version")
            strategy = SamplingStrategy(algo.sample_strategy)
            requests[config.name] = (strategy, int(algo.batch_size))
            rules[config.name] = select_version_rule(strategy)
        if not requests:
            raise ValueError("versioned sampling requires at least one enabled learner")
        self._version_requests = requests
        self._version_rules = rules
        self._version_round = dict.fromkeys(requests, 0)
        self._overlap_learners = frozenset(
            name for name, (strategy, _) in requests.items()
            if self.task_config.allow_sample_overlap and strategy in ROLLING_STRATEGIES
        )

    def _sampling_version(self, round_num: int) -> str:
        return f"sample_{self._version_namespace}_g{self.generation}_v{round_num:06d}"

    def _current_sample_versions_locked(self) -> dict[str, str]:
        return {name: self._sampling_version(round_num) for name, round_num in self._version_round.items()}

    def _versions_for_sample_ticket_locked(self) -> Optional[dict[str, str]]:
        if self._pending_restart_reason is not None:
            return None
        versions = self._current_sample_versions_locked()
        if self._overlap_learners:
            # Available workers keep collecting with the published version vector.
            return versions
        active = [entry.episode_num for entry in self._lease_table.values()
                  if entry.kind == LEASE_KIND_SAMPLE]
        if len(active) < len(self._version_outstanding):
            # A returned ticket retains its original version vector.
            return versions

        # Preissue only when every policy in the rollout is already being updated.
        # Otherwise a sampler uses published weights while collecting for a learner
        # whose update may depend on this very rollout.
        if self._version_sealed == self._version_requests.keys():
            versions = {name: self._sampling_version(self._version_round[name] + 1) for name in versions}
            candidates = versions
        else:
            candidates = {name: version for name, version in versions.items()
                          if name not in self._version_sealed}
        for name, version in candidates.items():
            reserved = sum(self.sample_task_seq[episode - 1].required_model_versions.get(name) == version
                           for episode in active)
            if reserved < self._version_ticket_demand_locked(version, name):
                return versions
        return None

    def _version_ticket_demand_locked(self, version: str, learner_name: str) -> int:
        strategy, target = self._version_requests[learner_name]
        state = self.replay_buffer_state
        observed = state.sample_versions.get(learner_name, state.sample_version)
        same_version = observed == version
        available = (state.versioned_available.get(learner_name, 0)
                     if same_version or strategy in ROLLING_STRATEGIES else 0)
        has_fresh_data = same_version and learner_name in state.versioned_fresh_agents
        algo = next(config.algo_config for config in self.learner_configs if config.name == learner_name)
        return max(self._version_rules[learner_name](target, available, has_fresh_data),
                   replay_warmup_remaining(state, algo, learner_name))

    @property
    def versioned_sampling_enabled(self) -> bool:
        return self._versioned_sampling

    @property
    def overlapping_sample_learners(self) -> frozenset[str]:
        return self._overlap_learners

    def model_publication_target(self, learning_batch: bool) -> tuple[dict[str, str], dict] | None:
        """Return the version vector after publishing this learning permit."""
        with self._lock:
            if not self._versioned_sampling or (learning_batch and not self._version_sealed):
                return None
            versions = {name: self._sampling_version(round_num + int(learning_batch and name in self._version_sealed))
                        for name, round_num in self._version_round.items()}
            return versions, dict(self._version_requests)

    def complete_model_publication(
            self, versions: dict[str, str], *, replay_state: Optional[ReplayBufferState] = None,
    ) -> None:
        with self._lock:
            if not self._version_sealed:
                return
            target, _ = self.model_publication_target(True)
            if versions != target:
                return
            for name in self._version_sealed:
                self._version_round[name] += 1
            self._version_sealed.clear()
            if replay_state is not None and self._is_latest_versioned_replay_state_locked(replay_state):
                self.replay_buffer_state = replay_state
            if self._pending_restart_reason is None:
                self._refresh_learn_ready_locked()
        self._maybe_apply_pending_restart()

    def custom_monitor(self) -> None:
        """独立线程周期调用的定制入口。默认 pass，未覆盖时不启动线程。"""
        return None

    def custom_monitor_interval_seconds(self) -> float:
        return 1.0

    def _issue_lease_locked(
            self,
            task: EpisodeInfo,
            *,
            kind: str,
            dispatch_count: int = 0,
            worker_id: str = "",
    ) -> str:
        """签发一张采样租约并登记租约表。调用方必须持有 self._lock。

        主路 claim 与旁路出队共用：lease_id 写回令牌、start_datetime 置位、
        表登记 kind/generation，回报与回收只认这张表。
        """
        lease_id = uuid.uuid4().hex
        dispatched_at = datetime.now()
        task.lease_id = lease_id
        task.start_datetime = dispatched_at
        self._lease_table[lease_id] = SampleLeaseEntry(
            kind=kind,
            episode_num=int(task.episode_num),
            generation=int(self.generation),
            issued_at=dispatched_at,
            dispatch_count=int(dispatch_count),
            worker_id=str(worker_id or ""),
        )
        return lease_id

    def accept_sample_task_result(self, episode_num: int, lease_id: str) -> bool:
        """在写ReplayBuffer前原子确认租约仍有效，并把确认时刻刷新为最新活动心跳。"""
        self._reclaim_timed_out_leases()
        with self._lock:
            entry = self._lease_table.get(lease_id)
            if entry is None or entry.kind != LEASE_KIND_SAMPLE:
                return False
            if entry.episode_num != int(episode_num):
                return False
            idx = entry.episode_num - 1
            if not (0 <= idx < len(self.sample_task_seq)):
                return False
            task = self.sample_task_seq[idx]
            if task.lease_id != lease_id or task.end_datetime is not None:
                return False
            entry.issued_at = datetime.now()
            task.start_datetime = entry.issued_at
            return True

    def report_sample_result(
            self,
            episode_num: int,
            total_rewards: Dict[AgentName, float],
            success: bool = True,
            artifacts: Optional[list[Artifact]] = None,
            episode_metrics: Optional[Dict[str, float]] = None,
            episode_rewards_details: Optional[Dict[str, float]] = None,
            lease_id: str = "",
            step_count: int = 0,
    ) -> None:
        """采样进程在一局推演完结并将数据放入Buffer后调用，更新采样进度。

        路由是全函数：lease 查租约表 → entry.kind 分派。不看载荷内容，
        artifact 只做归档不参与路由。
        """
        if artifacts is not None and (
                not isinstance(artifacts, list)
                or any(not isinstance(artifact, Artifact) for artifact in artifacts)
        ):
            raise ArtifactsArchiveException(
                operation="report_sample_result", detail="expected list[Artifact]",
            )
        artifacts = copy.deepcopy(artifacts or [])
        if not lease_id:
            raise SampleLeaseException(
                operation="report_sample_result", detail="lease_id is required",
            )
        should_save_optimo = False
        should_remove_reward_details = False
        bypass_entry: Optional[SampleLeaseEntry] = None
        with self._lock:
            entry = self._lease_table.pop(lease_id, None)
            if entry is None or entry.generation != int(self.generation):
                # 预期内迟到：超时回收、换代清场或重复回报；丢弃计数，不静默无痕。
                self._stale_report_count += 1
                LOG.warning(
                    f"Discard stale sample report: episode={episode_num}, "
                    f"lease_id={lease_id}, stale_total={self._stale_report_count}"
                )
                return
            if entry.episode_num != int(episode_num):
                raise SampleLeaseException(
                    operation="report_sample_result",
                    detail=(
                        f"lease {lease_id} registered episode={entry.episode_num}, "
                        f"reported={episode_num}"
                    ),
                )
            if entry.kind == LEASE_KIND_SAMPLE:
                should_remove_reward_details, should_save_optimo = (
                    self._settle_sample_lease_locked(
                        entry,
                        success=success,
                        total_rewards=total_rewards,
                        episode_metrics=episode_metrics,
                        episode_rewards_details=episode_rewards_details,
                        artifacts=artifacts,
                    )
                )
            else:
                # 旁路令牌不进主路序列、不推进 sample_task_completed；
                # 记账内部自取锁，必须在锁外调用。
                bypass_entry = entry

        if bypass_entry is not None:
            self._handle_bypass_sample_report(
                bypass_entry.kind,
                episode_num,
                success=success,
                artifacts=artifacts,
                total_rewards=total_rewards,
                episode_rewards_details=episode_rewards_details,
            )

        if should_remove_reward_details:
            self._remove_reward_details_image()
        if should_save_optimo:
            optimo_paths = self.save_models("optimo")
            if optimo_paths:
                self.record_checkpoint_paths(optimo_paths)
        self._maybe_apply_pending_restart()

    def _settle_sample_lease_locked(
            self,
            entry: SampleLeaseEntry,
            *,
            success: bool,
            total_rewards: Dict[AgentName, float],
            episode_metrics: Optional[Dict[str, float]],
            episode_rewards_details: Optional[Dict[str, float]],
            artifacts: list[Artifact],
    ) -> Tuple[bool, bool]:
        """主路租约结算：失败归还重试，成功推进进度并登记产物。

        调用方必须持有 self._lock；返回 (should_remove_reward_details, should_save_optimo)。
        """
        idx = entry.episode_num - 1
        if not (0 <= idx < len(self.sample_task_seq)):
            raise SampleLeaseException(
                operation="report_sample_result",
                detail=(
                    f"lease episode {entry.episode_num} out of range: "
                    f"seq_len={len(self.sample_task_seq)}"
                ),
            )
        task = self.sample_task_seq[idx]
        if not success:
            task.start_datetime = None
            task.end_datetime = None
            task.failed_count += 1
            task.last_failed_datetime = datetime.now()
            self.current_sample_task = min(self.current_sample_task, idx)
            self._record_episode_failed(task)
            LOG.warning(
                f"Sample task returned for retry: episode={entry.episode_num}, "
                f"failed_count={task.failed_count}"
            )
            return False, False
        self._version_outstanding.discard(entry.episode_num)
        should_remove_reward_details = self._on_sample_episode_completed(
            idx,
            total_rewards,
            episode_metrics or {},
            episode_rewards_details,
            artifacts,
        )
        should_save_optimo = self._maybe_save_optimo(total_rewards)
        return should_remove_reward_details, should_save_optimo

    def _handle_bypass_sample_report(
            self,
            kind: str,
            episode_num: int,
            *,
            success: bool,
            artifacts: Optional[list[Artifact]],
            total_rewards: Optional[Dict[AgentName, float]] = None,
            episode_rewards_details: Optional[Dict[str, float]] = None,
    ) -> None:
        episode_artifacts = list(artifacts or [])
        gif_path = next((
            artifact.path for artifact in episode_artifacts
            if artifact.artifact_type is ArtifactType.RENDER_GIF
        ), None)
        with self._lock:
            self._bypass_episode_artifacts.extend(episode_artifacts)
        if kind == LEASE_KIND_MID_EVAL:
            with self._lock:
                if gif_path:
                    self._mid_eval_gif_paths.append(gif_path)
            LOG.info(
                f"Mid-eval task reported: episode={episode_num}, success={success}, "
                f"mid_eval_gif={gif_path}"
            )
            return
        if kind == LEASE_KIND_END_EVAL:
            should_save_chart = self._on_end_eval_reported(
                episode_num,
                success=success,
                gif_path=gif_path,
                total_rewards=total_rewards or {},
                episode_rewards_details=episode_rewards_details,
            )
            if should_save_chart:
                try:
                    self.save_end_eval_reward_chart()
                except Exception:
                    LOG.exception("Failed to save end-eval reward chart")
            return
        LOG.info(
            f"Bypass sample reported: kind={kind}, episode={episode_num}, success={success}"
        )

    def _on_end_eval_reported(
            self,
            episode_num: int,
            *,
            success: bool,
            gif_path: Optional[str],
            total_rewards: Dict[AgentName, float],
            episode_rewards_details: Optional[Dict[str, float]],
    ) -> bool:
        """处理结束评估回报。失败则补发一张令牌；全部成功后返回 True 以画图。"""
        with self._lock:
            if not success:
                self._end_eval_dispatch_count += 1
                retry = self._build_end_eval_task(self._end_eval_dispatch_count)
                if retry is not None:
                    self._enqueue_bypass_locked(
                        retry,
                        kind=LEASE_KIND_END_EVAL,
                        dispatch_count=self._end_eval_dispatch_count,
                    )
                LOG.warning(
                    f"End-eval task returned for retry: episode={episode_num}, "
                    f"dispatch={self._end_eval_dispatch_count}"
                )
                return False
            if gif_path:
                self._end_eval_gif_paths.append(gif_path)
            self._end_eval_reward_details.append(
                self._end_eval_reward_components(total_rewards, episode_rewards_details)
            )
            self._end_eval_completed_count += 1
            target = self._end_eval_target()
            LOG.info(
                f"End-eval task reported: episode={episode_num}, "
                f"completed={self._end_eval_completed_count}/{target}, "
                f"end_eval_gif={gif_path}"
            )
            return bool(target > 0 and self._end_eval_completed_count >= target)

    @staticmethod
    def _end_eval_reward_components(
            total_rewards: Dict[AgentName, float],
            episode_rewards_details: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        if episode_rewards_details:
            return {
                str(component_name): float(value)
                for component_name, value in episode_rewards_details.items()
            }
        return {"total": float(sum(float(value) for value in total_rewards.values()))}

    def _on_sample_episode_completed(
            self,
            idx: int,
            total_rewards: Dict[AgentName, float],
            episode_metrics: Dict[str, float],
            episode_rewards_details: Optional[Dict[str, float]],
            artifacts: Optional[list[Artifact]] = None,
    ) -> bool:
        """持锁：写采样奖励并推进 sample_task_completed。"""
        task = self.sample_task_seq[idx]
        self._record_episode_completed(task, total_rewards)
        self._record_metric_sample(idx, episode_metrics)
        if artifacts:
            self.episode_artifacts[idx] = list(artifacts)
        should_remove_reward_details = self._record_reward_details_capability(
            idx,
            episode_rewards_details,
        )
        self.sample_task_completed += 1
        if self.sample_task_completed >= int(self.max_episode):
            # 业务阶段在主路采样耗尽时关闭；已由最后一次 push 置 ready 的许可仍可消费一次。
            self._sample_closed = True
        self.current_sample_task = self._first_unfinished_sample_index_locked()
        self._publish_training_progress_locked()
        return should_remove_reward_details

    def _record_episode_claimed(self, task: EpisodeInfo) -> None:
        return None

    def _record_episode_completed(
            self,
            task: EpisodeInfo,
            total_rewards: Dict[AgentName, float],
    ) -> None:
        idx = task.episode_num - 1
        task.end_datetime = datetime.now()
        for agent_name, reward in total_rewards.items():
            if agent_name in self.sample_rewards:
                self.sample_rewards[agent_name][idx] = float(reward)
            if agent_name in self.learn_sample_rewards:
                self.learn_sample_rewards[agent_name][idx] = float(reward)

    def _record_episode_failed(self, task: EpisodeInfo) -> None:
        return None

    def _record_episode_reclaimed(self, task: EpisodeInfo) -> None:
        return None

    def _record_metric_sample(self, idx: int, episode_metrics: Dict[str, float]) -> None:
        for metric_name, value in episode_metrics.items():
            series = self.sample_metrics.setdefault(
                metric_name, [None] * len(self.sample_task_seq),
            )
            series[idx] = float(value)

    def _record_reward_details_capability(
            self,
            idx: int,
            episode_rewards_details: Optional[Dict[str, float]],
    ) -> bool:
        if episode_rewards_details is None:
            self._reward_details_capability = "unsupported"
            return True
        if self._reward_details_capability != "unsupported":
            self._reward_details_capability = "supported"
            self.episode_rewards_details[idx] = {
                str(component_name): float(value)
                for component_name, value in episode_rewards_details.items()
            }
        return False

    def _record_progress(self) -> None:
        return None

    def _record_learn_permit_dispatched(self, learn_info: LearnInfo) -> None:
        self._inflight_learn_tokens[int(learn_info.learn_id)] = learn_info
        for learner_name in learn_info.active_learners:
            if learner_name in self.learner_state:
                algo_config = learn_info.algo_configs[learner_name]
                self.learner_state[learner_name]["dispatched_learning_count"] += int(
                    getattr(algo_config, "learn_times_per_episode")
                )

    def _record_learn_permit_completed(self, learn_info: LearnInfo) -> None:
        learn_id = int(learn_info.learn_id)
        self._inflight_learn_tokens.pop(learn_id, None)
        self._completed_learn_ids.add(learn_id)

    def _maybe_save_optimo(self, total_rewards: Dict[AgentName, float]) -> bool:
        """持锁：判断本局是否达到或刷新本 generation 历史最佳团队回报；应在锁外落盘 optimo。"""
        if not isinstance(self.task_config, TrainConfig) or not self.task_config.save_optimo_models:
            return False
        team_reward = sum(float(v) for v in total_rewards.values())
        if self._best_team_reward_generation != self.generation:
            self._best_team_reward = float("-inf")
            self._best_team_reward_generation = self.generation
        if team_reward < self._best_team_reward:
            return False
        self._best_team_reward = team_reward
        return True

    def _maybe_apply_pending_restart(self) -> None:
        """检查 restart 安全边界并在锁外执行回调与资源事件。"""
        with self._lock:
            restart_result, callbacks = self._apply_pending_restart_if_ready_locked()
        self._run_restart_callbacks(restart_result, callbacks)
        if restart_result is not None:
            self._notify_training_resource_event(
                TrainingResourceEvent.POST_RESTART,
                int(restart_result["generation"]),
                int(restart_result.get("prior_completed_episodes", 0)),
                advice=restart_result.get("advice"),
                reason=str(restart_result.get("reason") or ""),
            )

    def _refill_bypass_producers_locked(self) -> None:
        # 盘中评估必须跟采样进度同步判定，不能挪到 custom_monitor 线程。
        if self.should_trigger_mid_eval():
            task = self._build_mid_eval_task()
            if task is not None:
                self._enqueue_bypass_locked(
                    task,
                    kind=LEASE_KIND_MID_EVAL,
                    dispatch_count=self._mid_eval_dispatch_count,
                )
        self._refill_end_eval_locked()

    def _end_eval_target(self) -> int:
        if not isinstance(self.task_config, TrainConfig):
            return 0
        return max(0, int(self.task_config.end_eval_episodes))

    def _end_eval_outstanding_locked(self) -> bool:
        target = self._end_eval_target()
        if target <= 0:
            return False
        if self.sample_task_completed < int(self.max_episode):
            return False
        return self._end_eval_completed_count < target

    def _refill_end_eval_locked(self) -> None:
        """主路采样消费完后一次性补齐结束评估令牌；已签发过的不重复发。"""
        target = self._end_eval_target()
        if target <= 0:
            return
        if not isinstance(self.task_config, TrainConfig):
            return
        if self.sample_task_completed < int(self.max_episode):
            return
        while self._end_eval_dispatch_count < target:
            self._end_eval_dispatch_count += 1
            task = self._build_end_eval_task(self._end_eval_dispatch_count)
            if task is None:
                self._end_eval_dispatch_count -= 1
                return
            self._enqueue_bypass_locked(
                task,
                kind=LEASE_KIND_END_EVAL,
                dispatch_count=self._end_eval_dispatch_count,
            )

    def _enqueue_bypass_locked(
            self,
            episode_info: EpisodeInfo,
            *,
            kind: str,
            dispatch_count: int = 0,
    ) -> None:
        self._bypass_sample_queue.append((episode_info, kind, int(dispatch_count)))

    def _pop_bypass_task_locked(self) -> Optional[EpisodeInfo]:
        if not self._bypass_sample_queue:
            return None
        task, kind, dispatch_count = self._bypass_sample_queue.popleft()
        # 旁路令牌必须签发租约，回报时才能与主路同号 episode 区分。
        self._issue_lease_locked(task, kind=kind, dispatch_count=dispatch_count)
        return copy.deepcopy(task)

    def _build_mid_eval_task(self) -> Optional[EpisodeInfo]:
        """按训练进度间隔构造盘中评估令牌。调用方必须已确认 should_trigger_mid_eval。"""
        if not isinstance(self.task_config, TrainConfig):
            return None
        milestone = self._pending_mid_eval_milestone()
        if milestone is None:
            return None
        completed = int(self.sample_task_completed)
        self._last_mid_eval_train_episode = milestone
        self._mid_eval_dispatch_count += 1
        train_episode = milestone
        record_name = self._bypass_record_name(
            train_episode=train_episode,
            dispatch_count=self._mid_eval_dispatch_count,
        )
        task = EpisodeInfo(
            episode_num=train_episode,
            task_id=self.task_id,
            epsilon=0.0,
            deterministic=True,
            temperature=1.0,
            actions_param=self._build_episode_action_params(
                epsilon=0.0,
                deterministic=True,
                temperature=1.0,
            ),
            seed=self._episode_seed(train_episode, "mid_eval", self._mid_eval_dispatch_count),
            start_datetime=None,
            end_datetime=None,
            metadata=self._bypass_eval_record_metadata(record_name),
            env_metadata=copy.deepcopy(self.task_config.env_metadata),
            stage_name=self.stage_name,
            generation=self.generation,
        )
        self.custom_episode(episode_info=task)
        LOG.info(
            f"Dispatch mid-eval task: train_episode={train_episode}, "
            f"completed_episodes={completed}, mid_eval_index={self._mid_eval_dispatch_count}"
        )
        return task

    def _build_end_eval_task(self, dispatch_count: int) -> Optional[EpisodeInfo]:
        """构造一张结束评估旁路令牌；执行参数与盘中评估相同，含 GIF 录制。"""
        if not isinstance(self.task_config, TrainConfig):
            return None
        train_episode = int(self.max_episode)
        record_name = self._end_eval_record_name(
            train_episode=train_episode,
            dispatch_count=dispatch_count,
        )
        task = EpisodeInfo(
            episode_num=train_episode,
            task_id=self.task_id,
            epsilon=0.0,
            deterministic=True,
            temperature=1.0,
            actions_param=self._build_episode_action_params(
                epsilon=0.0,
                deterministic=True,
                temperature=1.0,
            ),
            seed=self._episode_seed(train_episode, "end_eval", dispatch_count),
            start_datetime=None,
            end_datetime=None,
            metadata=self._bypass_eval_record_metadata(record_name),
            env_metadata=copy.deepcopy(self.task_config.env_metadata),
            stage_name=self.stage_name,
            generation=self.generation,
        )
        self.custom_episode(episode_info=task)
        LOG.info(
            f"Dispatch end-eval task: train_episode={train_episode}, "
            f"end_eval_index={dispatch_count}/{self._end_eval_target()}"
        )
        return task

    def _bypass_eval_record_metadata(self, record_name: str) -> Dict[str, Any]:
        return {
            # 令牌只带执行参数。来源（mid_eval / end_eval）留在 TM 的 lease_id 表里。
            "replay_write": False,  # False：本局 transitions 不写入 ReplayBuffer，也不走租约确认。
            "record_render": True,  # True：rollout 启动 GIF 录制；False 则整局不录。
            "record_output_dir": self.task_config.render_video_dir,  # GIF 落盘目录，环境只负责 render 单帧。
            "record_fps": self.task_config.render_gif_fps,  # 回放帧率。
            "record_frame_stride": self.task_config.render_gif_frame_stride,  # 隔多少环境步抽一帧。
            "record_max_frames": self.task_config.render_gif_max_frames,  # 最多保留帧数，防止超长局撑爆内存。
            "record_name": record_name,  # 产物文件名，由 TM 签发，rollout 只负责用。
            "random_sample": False,  # True 时 rollout 走随机动作；结束/盘中评估默认跟策略。
        }

    def _bypass_record_name(self, *, train_episode: int, dispatch_count: int) -> str:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S")
        safe_task_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(self.task_id)).strip("_") or "task"
        safe_stage_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(self.stage_name)).strip("_") or "train"
        return (
            f"mid_eval_{safe_task_id}_{timestamp}_{safe_stage_name}_"
            f"episode_{int(train_episode):04d}_eval_{dispatch_count}.gif"
        )

    def _end_eval_record_name(self, *, train_episode: int, dispatch_count: int) -> str:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S")
        safe_task_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(self.task_id)).strip("_") or "task"
        safe_stage_name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(self.stage_name)).strip("_") or "train"
        return (
            f"end_eval_{safe_task_id}_{timestamp}_{safe_stage_name}_"
            f"episode_{int(train_episode):04d}_eval_{dispatch_count}.gif"
        )

    def should_trigger_mid_eval(self) -> bool:
        """已完成局数跨过新的固定间隔里程碑时触发盘中评估。"""
        return self._pending_mid_eval_milestone() is not None

    def _pending_mid_eval_milestone(self) -> Optional[int]:
        """返回尚未执行的下一个评估里程碑，避免并行上报跨过整点时漏触发。"""
        if not isinstance(self.task_config, TrainConfig):
            return None
        if not bool(self.task_config.mid_eval_enabled):
            return None
        interval = int(self.task_config.mid_eval_episode_interval)
        if interval <= 0:
            return None
        completed = int(self.sample_task_completed)
        if completed <= 0 or completed >= int(self.max_episode):
            return None
        last_milestone = int(self._last_mid_eval_train_episode)
        next_milestone = (last_milestone // interval + 1) * interval
        if next_milestone >= int(self.max_episode):
            return None
        return next_milestone if completed >= next_milestone else None

    def _maybe_build_mid_eval_task(self) -> Optional[EpisodeInfo]:
        """兼容既有测试：确认门控后构造盘中评估令牌。"""
        if not self.should_trigger_mid_eval():
            return None
        return self._build_mid_eval_task()

    def save_reward_curve(
            self,
            *,
            y_axis_tick_interval: Optional[float] = None,
            color_sequence: Optional[Sequence[Any]] = None,
    ) -> Optional[str]:
        """保存各 Agent 的 episode reward 曲线。

        这是插件可覆盖的绘图接口。子类可直接重写整套绘图逻辑，或调用 ``super()`` 并通过
        ``y_axis_tick_interval``、``color_sequence`` 定制 Y 轴刻度和曲线配色。
        """
        with self._lock:
            if self._closed or self._task_completion_in_progress or self._task_completion_done:
                return None
            agent_episode_rewards = {
                agent_name: [reward for reward in rewards if reward is not None]
                for agent_name, rewards in self._downstream_sample_rewards().items()
                if any(reward is not None for reward in rewards)
            }
            generation = self.generation
        if not agent_episode_rewards:
            return None
        max_episode_count = max(len(rewards) for rewards in agent_episode_rewards.values())
        if max_episode_count < MIN_REWARD_CURVE_EPISODES:
            return None
        # 奖励曲线是结果产物，只允许写入 result_save_dir；output_dir 专用于样本录制。
        save_dir = getattr(self.task_config, "result_save_dir", ResourceDir.RESULTS.value)
        # 分阶段训练每个阶段单独出一张奖励曲线，用 stage_name 尾缀区分；普通训练尾缀为 train。
        run_image_name = f"reward_{self.start_timestamp}_{self.task_id}_{self.stage_name}_{generation}.png"
        run_image_path = os.path.join(save_dir, run_image_name)
        with self._reward_curve_file_lock:
            with self._lock:
                if (
                        self._closed
                        or self._task_completion_in_progress
                        or self._task_completion_done
                        or self.generation != generation
                ):
                    return None
            if self._reward_curve_file_path != run_image_path:
                if self._reward_curve_file is not None:
                    self._reward_curve_file.close()
                ensure_output_parent(run_image_path)
                self._reward_curve_file = open(run_image_path, "w+b")
                self._reward_curve_file_path = run_image_path
            assert self._reward_curve_file is not None
            save_reward_curve(
                agent_episode_rewards,
                self._reward_curve_file,
                image_name=run_image_name,
                y_axis_tick_interval=y_axis_tick_interval,
                color_sequence=color_sequence,
            )
            csv_path = os.path.join(save_dir, "raw_data", os.path.splitext(run_image_name)[0] + ".csv")
            save_reward_curve_csv(agent_episode_rewards, csv_path)
            LOG.debug(f"Save reward curve: {run_image_path}")
            with self._lock:
                self._last_reward_curve_path = run_image_path
            # 写图和登记同属一次生产；收口等待此锁，防止归档后才登记。
            self.put_runtime_file(ArtifactType.RESULT, run_image_path)
            self.put_runtime_file(ArtifactType.RESULT, csv_path, {"category": "plot_csv"})
        return run_image_path

    def save_reward_details(
            self,
            *,
            y_axis_tick_interval: Optional[float] = None,
            color_sequence: Optional[Sequence[Any]] = None,
    ) -> Optional[str]:
        """保存正负分离的奖励构成堆叠图。

        这是插件可覆盖的绘图接口。子类可直接重写整套绘图逻辑，或调用 ``super()`` 并通过
        ``y_axis_tick_interval``、``color_sequence`` 定制 Y 轴刻度和堆叠色块。
        """
        with self._lock:
            if (
                    self._closed
                    or self._task_completion_in_progress
                    or self._task_completion_done
                    or self._reward_details_capability != "supported"
            ):
                return None
            episode_details = {
                idx + 1: dict(details)
                for idx, details in enumerate(self.episode_rewards_details)
                if details is not None
            }
            generation = self.generation
        if (
                len(episode_details) < MIN_REWARD_CURVE_EPISODES
                or not any(details for details in episode_details.values())
        ):
            return None

        run_image_path = self._reward_details_image_path(generation)
        run_image_name = os.path.basename(run_image_path)
        with self._reward_details_file_lock:
            with self._lock:
                if (
                        self._closed
                        or self._task_completion_in_progress
                        or self._task_completion_done
                        or self.generation != generation
                        or self._reward_details_capability != "supported"
                ):
                    return None
            if self._reward_details_file_path != run_image_path:
                if self._reward_details_file is not None:
                    self._reward_details_file.close()
                ensure_output_parent(run_image_path)
                self._reward_details_file = open(run_image_path, "w+b")
                self._reward_details_file_path = run_image_path
            assert self._reward_details_file is not None
            save_reward_details(
                episode_details,
                self._reward_details_file,
                image_name=run_image_name,
                y_axis_tick_interval=y_axis_tick_interval,
                color_sequence=color_sequence,
            )
            csv_path = os.path.join(os.path.dirname(run_image_path), "raw_data",
                                    os.path.splitext(run_image_name)[0] + ".csv")
            save_reward_details_csv(episode_details, csv_path)
            LOG.debug(f"Save reward details: {run_image_path}")
            with self._lock:
                self._last_reward_details_path = run_image_path
            self.put_runtime_file(ArtifactType.RESULT, run_image_path)
            self.put_runtime_file(ArtifactType.RESULT, csv_path, {"category": "plot_csv"})
        return run_image_path

    def save_end_eval_reward_chart(self) -> Optional[str]:
        """主路结束后的评估奖励堆叠柱状图；end_eval_episodes=0 时不生成。"""
        with self._lock:
            if self._closed or self._task_completion_in_progress or self._end_eval_target() <= 0:
                return None
            episode_details = {
                idx + 1: dict(details)
                for idx, details in enumerate(self._end_eval_reward_details)
            }
            generation = self.generation
        if not episode_details:
            return None
        run_image_path = self._end_eval_reward_image_path(generation)
        run_image_name = os.path.basename(run_image_path)
        ensure_output_parent(run_image_path)
        with open(run_image_path, "w+b") as image_file:
            save_end_eval_reward_bars(
                episode_details,
                image_file,
                image_name=run_image_name,
            )
        LOG.info(f"Save end-eval reward chart: {run_image_path}")
        with self._lock:
            self._last_end_eval_reward_path = run_image_path
        self.put_runtime_file(ArtifactType.RESULT, run_image_path)
        return run_image_path

    def _end_eval_reward_image_path(self, generation: int) -> str:
        save_dir = getattr(self.task_config, "result_save_dir", ResourceDir.RESULTS.value)
        run_image_name = (
            f"end_eval_reward_{self.start_timestamp}_{self.task_id}_"
            f"{self.stage_name}_{generation}.png"
        )
        return os.path.join(save_dir, run_image_name)

    def _reward_details_image_path(self, generation: int) -> str:
        save_dir = getattr(self.task_config, "result_save_dir", ResourceDir.RESULTS.value)
        run_image_name = (
            f"reward_details_{self.start_timestamp}_{self.task_id}_"
            f"{self.stage_name}_{generation}.png"
        )
        return os.path.join(save_dir, run_image_name)

    def _remove_reward_details_image(self) -> None:
        """能力冲突时移除本generation已经生成的构成图，保证unsupported最终无产物。"""
        with self._lock:
            expected_image_path = self._reward_details_image_path(self.generation)
        with self._reward_details_file_lock:
            if self._reward_details_file is not None:
                self._reward_details_file.close()
            self._reward_details_file = None
            self._reward_details_file_path = None
            with self._lock:
                if self._last_reward_details_path == expected_image_path:
                    self._last_reward_details_path = None
            # restart后句柄可能仍指向上一generation的有效产物，只关闭、不删除；仅清理当前代次。
            if os.path.exists(expected_image_path):
                try:
                    os.remove(expected_image_path)
                except OSError:
                    LOG.exception(
                        f"Failed to remove unsupported reward details image: {expected_image_path}"
                    )
                else:
                    LOG.warning(
                        f"Removed unsupported reward details image: {expected_image_path}"
                    )
            csv_path = os.path.join(os.path.dirname(expected_image_path), "raw_data",
                                    os.path.splitext(os.path.basename(expected_image_path))[0] + ".csv")
            if os.path.exists(csv_path):
                try:
                    os.remove(csv_path)
                except OSError:
                    LOG.exception(f"Failed to remove unsupported reward details CSV: {csv_path}")

    def _save_reward_plots(self) -> None:
        """逐局更新当前TaskManager拥有的全部奖励图。"""
        self.save_reward_curve()
        self.save_reward_details()

    def _log_reward_plot_paths(self) -> None:
        with self._lock:
            generation = self.generation
        save_dir = getattr(self.task_config, "result_save_dir", ResourceDir.RESULTS.value)
        reward_curve_path = os.path.join(
            save_dir,
            f"reward_{self.start_timestamp}_{self.task_id}_{self.stage_name}_{generation}.png",
        )
        LOG.info(f"Save reward curve: {reward_curve_path}")
        LOG.info(f"Save reward details: {self._reward_details_image_path(generation)}")

    def _reward_curve_monitor_thread(self):
        """后台观察采样完成进度，有新样本奖励时由任务管理器自己更新奖励曲线。"""
        self._log_reward_plot_paths()
        while True:
            with self._lock:
                completed_count = self.sample_task_completed
                task_finished = completed_count >= len(self.sample_task_seq)
                closed = (
                    self._closed
                    or self._task_completion_in_progress
                    or self._task_completion_done
                )
            if closed:
                return
            if completed_count > self._reward_curve_last_completed:
                self._save_reward_plots()
                self._reward_curve_last_completed = completed_count
            # 阶段被切换关闭时退出线程，避免被丢弃的实例残留空转守护线程。
            if task_finished:
                return
            time.sleep(1.0)

    def _reclaim_timed_out_leases(self, now: Optional[datetime] = None) -> int:
        """回收超过配置时限仍未返回结果的采样租约；主路与旁路统一表驱动。

        失败语义与回报失败对齐：主路归还可重领；end_eval 移除后补发一张；
        mid_eval 仅移除（盘中评估失败本就不补发）。拆成独立方法便于确定性测试。
        """
        current_time = now or datetime.now()
        now_mono = time.monotonic()
        timeout_seconds = float(self.task_config.sample_task_timeout)
        sample_reclaimed: List[Tuple[int, str, float]] = []
        bypass_reclaimed: List[Tuple[str, str, float]] = []
        with self._lock:
            for lease_id, entry in list(self._lease_table.items()):
                occupied_seconds = (current_time - entry.issued_at).total_seconds()
                if occupied_seconds <= timeout_seconds:
                    continue
                self._lease_table.pop(lease_id, None)
                if entry.kind == LEASE_KIND_SAMPLE:
                    idx = entry.episode_num - 1
                    if 0 <= idx < len(self.sample_task_seq):
                        task = self.sample_task_seq[idx]
                        if task.end_datetime is None:
                            task.start_datetime = None
                            task.failed_count += 1
                            task.last_failed_datetime = current_time
                            self.current_sample_task = min(self.current_sample_task, idx)
                            self._record_episode_reclaimed(task)
                    sample_reclaimed.append((entry.episode_num, lease_id, occupied_seconds))
                else:
                    if entry.kind == LEASE_KIND_END_EVAL:
                        self._end_eval_dispatch_count += 1
                        retry = self._build_end_eval_task(self._end_eval_dispatch_count)
                        if retry is not None:
                            self._enqueue_bypass_locked(
                                retry,
                                kind=LEASE_KIND_END_EVAL,
                                dispatch_count=self._end_eval_dispatch_count,
                            )
                    bypass_reclaimed.append((entry.kind, lease_id, occupied_seconds))
        for episode_num, lease_id, occupied_seconds in sample_reclaimed:
            LOG.warning(
                f"Sample task lease timed out and reclaimed: episode={episode_num}, "
                f"lease_id={lease_id}, occupied_secs={occupied_seconds:.1f}, "
                f"timeout_seconds={self.task_config.sample_task_timeout}"
            )
        for kind, lease_id, occupied_seconds in bypass_reclaimed:
            LOG.warning(
                f"Bypass lease timed out: kind={kind}, lease_id={lease_id}, "
                f"occupied_secs={occupied_seconds:.1f}, "
                f"timeout_seconds={self.task_config.sample_task_timeout}, "
                f"action={'re-dispatched' if kind == LEASE_KIND_END_EVAL else 'dropped'}"
            )
        return len(sample_reclaimed) + len(bypass_reclaimed)

    def _has_bypass_leases_locked(self) -> bool:
        """租约表中是否存在旁路（非主路）在途条目。调用方必须持有 self._lock。"""
        return any(entry.kind != LEASE_KIND_SAMPLE for entry in self._lease_table.values())

    def _schedule_sample_task_timeout_monitor(self) -> None:
        """登记下一次一分钟后的采样令牌超时扫描。"""
        timer = threading.Timer(60.0, self._run_sample_task_timeout_monitor)
        timer.name = f"sample_task_timeout_monitor_{self.stage_name}"
        timer.daemon = True
        self._sample_task_monitor_timer = timer
        timer.start()

    def _run_sample_task_timeout_monitor(self) -> None:
        """执行一轮扫描，并在任务仍运行时登记下一轮定时线程。"""
        with self._lock:
            if self._closed:
                return
        self._reclaim_timed_out_leases()
        with self._lock:
            if self._closed:
                return
            self._schedule_sample_task_timeout_monitor()

    def poll_stage_lifecycle(self) -> StageLifecycleState:
        """唯一的外部驱动 tick：自检采样结算、AutoML 判决与阶段结束。"""
        self._raise_auto_ml_fatal_error()
        # 采样收口可能触发 AutoML 归档 / restart，必须在读 finished 之前执行。
        sample_completed = self._settle_sample_completion()
        with self._lock:
            learning_completed = self._is_learning_completed_locked()
            pending_restart = self._pending_restart_reason is not None
            reason = self._pending_restart_reason or ""
            finished = self._evaluate_stage_finished_locked(sample_completed)
        return StageLifecycleState(
            sample_completed=sample_completed,
            learning_completed=learning_completed,
            finished=finished,
            pending_restart=pending_restart,
            reason=reason,
        )

    def _evaluate_stage_finished_locked(self, sample_completed: bool) -> bool:
        """采样已结算且没有挂起的 restart 时，本阶段结束。调用方必须持有 self._lock。"""
        return bool(sample_completed) and self._pending_restart_reason is None

    def is_sample_completed(self) -> bool:
        """采样是否跑满；若 extra_complete_rounds 未用尽则走既有 restart 流程再开一轮。"""
        return self.poll_stage_lifecycle().sample_completed

    def _raise_sample_progress_deadlock_locked(self) -> None:
        """簿记死角报警：无在途租约、无待签发令牌、无主路可领令牌，却未完成。

        completed + 在途 + 可领 三者之和恒等于计划总量；三者缺口并存不可能由
        正常时序产生，只能是记账 bug——响亮失败，避免 Role 侧无限死等。
        调用方必须持有 self._lock。
        """
        if self._lease_table or self._bypass_sample_queue:
            return
        if any(
                task.start_datetime is None and task.end_datetime is None
                for task in self.sample_task_seq
        ):
            return
        raise SampleLeaseException(
            operation="poll_stage_lifecycle",
            detail=(
                f"sample progress deadlocked: completed={self.sample_task_completed}, "
                f"max_episode={self.max_episode}, in_flight_leases=0, bypass_queue=0"
            ),
        )

    def _clear_inflight_leases_locked(self, reason: str) -> int:
        """换代/阶段切换时作废全部在途租约；迟到回报按未知租约丢弃计数。

        调用方必须持有 self._lock；返回作废条数。
        """
        dropped = len(self._lease_table)
        self._lease_table.clear()
        if dropped:
            LOG.warning(
                f"In-flight sample leases cleared: count={dropped}, reason={reason}"
            )
        return dropped

    def _settle_sample_completion(self) -> bool:
        with self._lock:
            if isinstance(self.task_config, TrainConfig) and (
                    self.task_config.sample_type == SampleType.OFFLINE.value
                    and not self._offline_plan_prepared
            ):
                return False
            if self.sample_task_completed < self.max_episode:
                self._sample_completion_settled = False
                self._raise_sample_progress_deadlock_locked()
                return False
            # 主路消费完后先签发结束评估令牌；未跑完结束评估不能宣告训练结束。
            self._refill_end_eval_locked()
            if self._end_eval_outstanding_locked():
                return False
            if self._sample_completion_settled:
                return True
            # 末里程碑盘中评估可能与阶段收口抢跑；旁路未回报前不能结算，
            # 否则 Role 会换掉本 TM，迟到的 bypass report 落到新实例上丢 lease。
            if self._bypass_sample_queue or self._has_bypass_leases_locked():
                return False
            auto_ml = self._auto_ml
            if auto_ml is None:
                return True
            # 在途采样租约或学习令牌未排空时不能判决，等下一轮 tick。
            if not self._can_restart_now_locked():
                return False
            snapshot = self._collect_training_snapshot_locked()

        advice = auto_ml.on_sample_completed(snapshot)
        if advice is None:
            # advice 为空：本 trial 收口，归档后标记 settled。
            self._archive_completed_generation(
                AutoMLAdvice(decision="complete", reason="sample_completed"),
            )
            with self._lock:
                self._sample_completion_settled = True
            return True

        result = self._execute_automl_restart_with_archive(advice)
        if result.get("pending_restart"):
            return False
        with self._lock:
            self._sample_completion_settled = False
        return False

    def _archive_completed_generation(self, advice: AutoMLAdvice) -> None:
        """归档正常跑满且不再 restart 的 generation；可重复调用但只落盘一次。"""
        with self._lock:
            if self._closed or self._auto_ml is None:
                return
            generation = self.generation
            if generation in self._archived_generations:
                return
            episode_num = self.sample_task_completed
            auto_ml_ref = self._auto_ml
            if not auto_ml_ref.config.archive_enabled:
                self._archived_generations.add(generation)
                return

        self._prepare_archive_resources(advice, episode_num, generation)
        with self._lock:
            if self._closed or generation in self._archived_generations:
                return
            snapshot = self._collect_training_snapshot_locked()
            archive = self._collect_training_run_archive_locked(advice, snapshot)
            self._archived_generations.add(generation)
            self._archive_seq += 1

        archive_dir = auto_ml_ref.archive_train(archive)
        if archive_dir:
            if os.path.exists(archive_dir):
                with self._lock:
                    self._automl_archive_paths.append(str(archive_dir))
            LOG.info(f"AutoML completed generation archive saved: {archive_dir}")

    def _next_available_sample_task_locked(self) -> Optional[EpisodeInfo]:
        """从当前扫描游标开始查找最近可领取采样令牌，兼容失败归还后的重跑。"""
        total = len(self.sample_task_seq)
        retry_pending = any(self.sample_task_seq[episode - 1].start_datetime is None
                            for episode in self._version_outstanding)
        for offset in range(total):
            idx = (self.current_sample_task + offset) % total
            task = self.sample_task_seq[idx]
            if task.end_datetime is not None:
                continue
            if task.start_datetime is not None:
                continue
            if retry_pending and not task.required_model_versions:
                continue
            self.current_sample_task = (idx + 1) % total
            return task
        return None

    def _first_unfinished_sample_index_locked(self) -> int:
        """返回第一个未完成采样任务下标；全部完成时返回序列长度。"""
        for idx, task in enumerate(self.sample_task_seq):
            if task.end_datetime is None:
                return idx
        return len(self.sample_task_seq)

    def get_learn_task_info(self) -> LearnTaskResult:
        """
        领取一张学习许可。

        许可只由 ReplayBuffer.push 置位，不按采样 episode 累积，任意时刻最多一张 ready 和一张 in-flight。
        采样关闭后不再置 ready，但最后一次 push 留下的 ready 仍可消费一次。
        """
        assert isinstance(self.task_config, TrainConfig), (
            f"get_learn_task_info requires TrainConfig, given={type(self.task_config).__name__}"
        )
        with self._lock:
            if self._is_learning_completed_locked():
                return LearnTaskResult(task_finished=True)
            if self._inflight_learn_tokens or self._version_sealed or not self._learn_ready:
                return LearnTaskResult()
            if len(self._version_requests) == 1 and not self._overlap_learners and self._version_outstanding:
                return LearnTaskResult()
            active_learners = list(self._learn_ready_active_learners)
            if not active_learners:
                self._clear_learn_ready_locked()
                return LearnTaskResult(task_finished=self._is_learning_completed_locked())
            learn_id = int(self._next_learn_id)
            self._next_learn_id += 1
            sample_watermark = max(
                1,
                int(self.sample_task_completed),
                int(self._latest_replay_sample_watermark),
            )
            learn_info = self._build_learn_token(
                learn_id=learn_id,
                sample_watermark=sample_watermark,
                active_learners=active_learners,
                seed_iteration=self._next_learn_seed_iteration_locked(sample_watermark),
            )
            self._clear_learn_ready_locked()
            self._record_learn_permit_dispatched(learn_info)
            if self._versioned_sampling:
                self._version_sealed.update(active_learners)
            self._publish_dispatch_progress_locked()
            return LearnTaskResult(learn_info=learn_info)

    def report_learning_result(
            self,
            learn_result: dict,
            episode_num: int,
            learn_id: int = 0,
    ):
        """
        完成一张 in-flight 学习许可。学习完成不自动续发。

        学习期间若已有新的 ReplayBuffer.push 置位 ready，该 ready 保留待领。
        learn_id=0 仅兼容旧调用；生产 LearnInfo 始终携带独立 learn_id。
        """
        assert isinstance(self.task_config, TrainConfig), (
            f"report_learning_result requires TrainConfig, given={type(self.task_config).__name__}"
        )
        callbacks = []
        restart_result = None
        with self._lock:
            assert 1 <= episode_num <= int(self.max_episode), (
                f"learning result episode_num must be in [1, {self.max_episode}], "
                f"given={episode_num}"
            )
            learn_info = self._resolve_inflight_learn_info_locked(
                learn_id=int(learn_id),
                episode_num=int(episode_num),
            )
            if learn_info is None:
                return self.learning_completed
            self._record_learn_permit_completed(learn_info)
            had_update = False
            for learner_name, result in learn_result.items():
                learner_result_store = self.learn_results.setdefault(learner_name, {"episode_num": []})
                result_items = result if isinstance(result, list) else [result]
                for result_item in result_items:
                    if not result_item:
                        continue
                    had_update = True
                    learner_result_store.setdefault("episode_num", []).append(learn_info.episode_num)
                    for key, value in result_item.items():
                        learner_result_store.setdefault(key, []).append(value)
                    self.learning_completed += 1
                    if learner_name in self.learner_state:
                        self.learner_state[learner_name]["learning_completed"] += 1
            if had_update:
                self._learned_sample_watermark = max(
                    int(self._learned_sample_watermark),
                    int(learn_info.episode_num),
                )
            if self._pending_restart_reason is not None:
                self._clear_learn_ready_locked()
            self._publish_training_progress_locked()
            learning_completed = self.learning_completed
            restart_result, callbacks = self._apply_pending_restart_if_ready_locked()
        self._run_restart_callbacks(restart_result, callbacks)
        if restart_result is not None:
            self._notify_training_resource_event(
                TrainingResourceEvent.POST_RESTART,
                int(restart_result["generation"]),
                int(restart_result.get("prior_completed_episodes", 0)),
                advice=restart_result.get("advice"),
                reason=str(restart_result.get("reason") or ""),
            )
        return learning_completed

    def is_task_finished(self) -> bool:
        """
        【生命周期判定】阶段任务是否结束。

        以采样结束为准：采样序列消费完毕即认为本阶段训练结束（req设计），学习序列是否消费完不影响判定。
        """
        return self.poll_stage_lifecycle().finished

    @process_close
    def close(self) -> None:
        """
        停止任务管理器后台线程；阶段切换或训练结束时由StageTask/Role调用。

        阶段切换会丢弃旧 TrainTaskManager 实例，这里置位关闭标志让奖励曲线线程及时退出，
        避免被丢弃的实例残留一个空转的守护线程。
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            timer = self._sample_task_monitor_timer
            self._sample_task_monitor_timer = None
        if timer is not None:
            timer.cancel()
        with self._reward_curve_file_lock:
            if self._reward_curve_file is not None:
                self._reward_curve_file.close()
                self._reward_curve_file = None
                self._reward_curve_file_path = None
        with self._reward_details_file_lock:
            if self._reward_details_file is not None:
                self._reward_details_file.close()
                self._reward_details_file = None
                self._reward_details_file_path = None
        self._log_reward_plot_paths()

    @property
    def automl_enabled(self) -> bool:
        return self._auto_ml is not None

    def register_learner_save_caller(self, caller: Callable[[str], List[str]]) -> None:
        """Role 注册 Learner 落盘 caller；TrainTaskManager.save_models 按前缀触发。"""
        with self._lock:
            self._learner_save_caller = caller

    def resolve_model_filename(
            self,
            name_prefix: ModelNamePrefix,
            agent_name: AgentName,
            model_name: ModelName,
    ) -> str:
        """生成标准模型文件名：{prefix}_{task_id}_{stage}_{gen}_{AgentName}_{ModelName}.pth"""
        return format_model_filename(
            name_prefix,
            agent_name,
            self.stage_name,
            model_name,
            self.generation,
            self.task_id,
        )

    def save_models(self, name_prefix: ModelNamePrefix) -> List[str]:
        """按 name_prefix 触发已注册的 Learner 落盘 caller。"""
        with self._lock:
            caller = self._learner_save_caller
        if caller is None:
            LOG.warning(
                f"save_models skipped: no learner save caller registered, prefix={name_prefix}"
            )
            return []
        paths = caller(name_prefix)
        return list(paths or [])

    def add_training_resource_callback(self, callback: TrainingResourceCallback) -> None:
        """注册训练资源生命周期回调（归档前贡献产物 / 资源就绪通知 / 重启后扩展）。"""
        with self._lock:
            if callback not in self._training_resource_callbacks:
                self._training_resource_callbacks.append(callback)

    def _merge_resource_contribution_locked(self, contribution: TrainingResourceContribution) -> None:
        """持锁合并回调贡献的产物路径。"""
        if contribution.model_paths:
            self._checkpoint_model_paths.extend(contribution.model_paths)
        for category, paths in contribution.artifact_paths.items():
            if not paths:
                continue
            # 保留 AutoML 存量回调协议；产物分类只用显式类型，未知类别作为 CUSTOM。
            artifact_type = (
                ArtifactType(category) if category in {item.value for item in ArtifactType}
                else ArtifactType.CUSTOM
            )
            self._archive_contributed_artifacts.extend(
                Artifact(artifact_type, path, {"category": category}) for path in paths
            )

    def _build_training_resource_context(
            self,
            event: TrainingResourceEvent,
            *,
            generation: int,
            completed_episodes: int,
            advice: Optional[AutoMLAdvice] = None,
            reason: str = "",
    ) -> TrainingResourceContext:
        return TrainingResourceContext(
            event=event,
            task_id=self.task_id,
            stage_name=self.stage_name,
            generation=generation,
            completed_episodes=completed_episodes,
            advice=advice,
            reason=reason,
        )

    def _invoke_training_resource_callbacks(
            self,
            context: TrainingResourceContext,
            *,
            accept_contribution: bool,
    ) -> None:
        """调用资源回调；PRE_ARCHIVE_REQUEST 可返回路径，其余事件通常只监听。"""
        with self._lock:
            callbacks = list(self._training_resource_callbacks)
        for callback in callbacks:
            try:
                contribution = callback(context)
            except Exception as exc:
                LOG.exception(
                    f"training_resource_callback failed: event={context.event.value}, "
                    f"stage={context.stage_name}, error={exc}"
                )
                continue
            if not accept_contribution or contribution is None:
                continue
            with self._lock:
                self._merge_resource_contribution_locked(contribution)

    def _notify_training_resource_event(
            self,
            event: TrainingResourceEvent,
            generation: int,
            completed_episodes: int,
            *,
            advice: Optional[AutoMLAdvice] = None,
            reason: str = "",
    ) -> None:
        ctx = self._build_training_resource_context(
            event,
            generation=generation,
            completed_episodes=completed_episodes,
            advice=advice,
            reason=reason,
        )
        self._invoke_training_resource_callbacks(ctx, accept_contribution=False)

    def _prepare_archive_resources(
            self,
            advice: AutoMLAdvice,
            episode_num: int,
            generation: int,
    ) -> None:
        """restart 归档前收集模型等产物；大模型 save 在锁外执行，避免阻塞采样/学习令牌。"""
        with self._lock:
            has_auto_ml = self._auto_ml is not None

        if has_auto_ml:
            model_prefix = "pruned" if advice.reason != "extra_complete_round" and advice.decision == "restart" else "final"
            model_paths = self.save_models(model_prefix)
            if model_paths:
                self.record_checkpoint_paths(model_paths)
                LOG.info(
                    f"Checkpoint before archive: prefix={model_prefix}, episode={episode_num}, "
                    f"generation={generation}, model_paths={len(model_paths)}"
                )

        request_ctx = self._build_training_resource_context(
            TrainingResourceEvent.PRE_ARCHIVE_REQUEST,
            generation=generation,
            completed_episodes=episode_num,
            advice=advice,
            reason=advice.reason or "automl_restart",
        )
        self._invoke_training_resource_callbacks(request_ctx, accept_contribution=True)

        self._notify_training_resource_event(
            TrainingResourceEvent.ARCHIVE_RESOURCES_READY,
            generation,
            episode_num,
            advice=advice,
            reason=advice.reason or "automl_restart",
        )

    def _execute_automl_restart_with_archive(self, advice: AutoMLAdvice) -> Dict[str, Any]:
        """归档当前代次后请求 restart；坍塌/prune 与 extra_complete_round 共用。"""
        with self._lock:
            if self._closed:
                return {"pending_restart": False, "closed": True}
            auto_ml_ref = self._auto_ml
            if auto_ml_ref is None:
                episode_num = None
                generation = None
            else:
                episode_num = self.sample_task_completed
                generation = self.generation

        if auto_ml_ref is None:
            return self.restart_training_task(
                reason=advice.reason or "automl",
                advice=advice,
            )

        if auto_ml_ref.config.archive_enabled:
            self._prepare_archive_resources(advice, episode_num, generation)

            with self._lock:
                if self._closed:
                    return {"pending_restart": False, "closed": True}
                snapshot = self._collect_training_snapshot_locked()
                archive = self._collect_training_run_archive_locked(advice, snapshot)
                auto_ml_ref = self._auto_ml
                self._archived_generations.add(generation)

            archive_dir = auto_ml_ref.archive_train(archive)
            if archive_dir:
                if os.path.exists(archive_dir):
                    with self._lock:
                        self._automl_archive_paths.append(str(archive_dir))
                LOG.info(f"AutoML archive saved: {archive_dir}")

        result = None
        callbacks: list = []
        with self._lock:
            if self._closed:
                return {"pending_restart": False, "closed": True}
            self._archive_seq += 1
            self._last_automl_restart_episode = episode_num
            self._pending_restart_reason = advice.reason
            self._pending_restart_advice = advice
            if self._can_restart_now_locked():
                result, callbacks = self._restart_training_task_locked(
                    advice.reason or "automl",
                    advice,
                )
            else:
                return {
                    "stage_name": self.stage_name,
                    "generation": self.generation,
                    "pending_restart": True,
                    "reason": advice.reason or "automl",
                }

        self._run_restart_callbacks(result, callbacks, advice)
        if result is not None:
            self._notify_training_resource_event(
                TrainingResourceEvent.POST_RESTART,
                int(result["generation"]),
                int(result.get("prior_completed_episodes", 0)),
                advice=advice,
                reason=str(result.get("reason") or ""),
            )
        return result

    def _auto_ml_monitor_loop(self) -> None:
        """捕获后台线程全部异常，交回Role主生命周期显式失败。"""
        try:
            self._auto_ml_monitor_loop_impl()
        except BaseException as exc:
            with self._lock:
                self._auto_ml_fatal_error = exc
            LOG.exception("AutoML monitor failed; task will fail on the next lifecycle call")

    def _auto_ml_monitor_loop_impl(self) -> None:
        """按已完成采样局数触发 advice()，不依赖墙钟时间。

        盘中路径：advice 返回 restart → 归档当前代次（含 mid_eval GIF）→ 请求 restart。
        阶段正常结束由 Role 调用 notify_stage_complete()，与本循环的 prune 路径互斥。
        """
        # 距下一局采样完成前的空转等待；触发条件本身按 episode 计。
        idle_sleep_secs = 0.2
        while True:
            with self._lock:
                closed = self._closed
                auto_ml = self._auto_ml
            if closed or auto_ml is None:
                return

            monitor = auto_ml.config
            poll_every = max(1, int(monitor.poll_interval_episodes))

            with self._lock:
                if self._closed or self._auto_ml is None:
                    return
                completed = self.sample_task_completed
                max_episode = self.max_episode
                last_check = self._last_automl_check_episode
                last_restart = self._last_automl_restart_episode

            # 采样跑满后由 is_sample_completed() 收口（含 extra_complete_rounds），避免与 monitor 竞态双重启。
            if completed >= max_episode:
                time.sleep(idle_sleep_secs)
                continue
            if completed < monitor.min_episodes_before_check:
                time.sleep(idle_sleep_secs)
                continue
            if (
                    last_restart > 0
                    and completed - last_restart < monitor.min_episodes_between_restart
            ):
                time.sleep(idle_sleep_secs)
                continue
            if completed - last_check < poll_every:
                time.sleep(idle_sleep_secs)
                continue

            with self._lock:
                if self._closed or self._auto_ml is None:
                    return
                snapshot = self._collect_training_snapshot_locked()
                auto_ml_ref = self._auto_ml
                self._last_automl_check_episode = completed

            advice = auto_ml_ref.advice(snapshot)
            if advice.decision != "restart":
                continue

            self._execute_automl_restart_with_archive(advice)

    def _raise_auto_ml_fatal_error(self) -> None:
        with self._lock:
            error = self._auto_ml_fatal_error
        if error is not None:
            raise RuntimeError("AutoML monitor failed") from error

    def _collect_training_snapshot_locked(self) -> TrainingSnapshot:
        """持锁组装训练快照。"""
        reward_curve: Dict[str, List[float]] = {}
        for agent_name, rewards in self._downstream_sample_rewards().items():
            valid = [
                float(r) for r in rewards[: self.sample_task_completed]
                if r is not None
            ]
            if valid:
                reward_curve[agent_name] = valid

        learn_metrics: Dict[str, Dict[str, List[float]]] = {}
        learners_by_name = {lc.name: lc for lc in self.learner_configs}
        for learner_name, store in self.learn_results.items():
            learner_config = learners_by_name.get(learner_name)
            if learner_config is None:
                continue
            metrics: Dict[str, List[float]] = {}
            for key, values in store.items():
                if key == "episode_num" or key in FRAMEWORK_UPDATE_KEYS:
                    continue
                floats = []
                for v in values:
                    try:
                        floats.append(float(v))
                    except (TypeError, ValueError):
                        continue
                if floats:
                    metrics[key] = floats
            if metrics:
                learn_metrics[learner_name] = metrics

        current_algo_configs = {
            lc.name: lc.algo_config.to_dict() if hasattr(lc.algo_config, "to_dict") else {}
            for lc in self.learner_configs
        }
        from scheduler.utils.automl_core import build_tunable_catalog
        tunable_catalog = build_tunable_catalog(
            self.learner_configs,
            self._auto_ml.config,
        ) if self._auto_ml is not None else {}

        artifact_paths: Dict[str, List[str]] = {}
        if self._last_reward_curve_path:
            artifact_paths["reward_curve"] = [self._last_reward_curve_path]
        if self._last_reward_details_path:
            artifact_paths["reward_details"] = [self._last_reward_details_path]
        if self._checkpoint_model_paths:
            artifact_paths["models"] = list(self._checkpoint_model_paths)
        # monitor / on_stage_complete 读 snapshot 时附带已收集的盘中评估 GIF 路径。
        if self._mid_eval_gif_paths:
            artifact_paths["mid_eval_gifs"] = list(self._mid_eval_gif_paths)
        if self._end_eval_gif_paths:
            artifact_paths["end_eval_gifs"] = list(self._end_eval_gif_paths)
        if self._last_end_eval_reward_path:
            artifact_paths["end_eval_reward"] = [self._last_end_eval_reward_path]

        return TrainingSnapshot(
            task_id=self.task_id,
            stage_name=self.stage_name,
            generation=self.generation,
            completed_episodes=self.sample_task_completed,
            max_episodes=self.max_episode,
            reward_curve=reward_curve,
            episode_metrics={
                name: [float(value) if value is not None else None
                       for value in values[: self.sample_task_completed]]
                for name, values in self.sample_metrics.items()
            },
            learn_metrics=learn_metrics,
            current_algo_configs=current_algo_configs,
            tunable_catalog=tunable_catalog,
            artifact_paths=artifact_paths,
            current_env_metadata_overrides=copy.deepcopy(self._current_env_metadata_overrides),
        )

    def _collect_training_run_archive_locked(
            self,
            advice: AutoMLAdvice,
            snapshot: TrainingSnapshot,
    ) -> TrainingRunArchive:
        """持锁组装 restart 归档包（算法指标与逐项业务输出分开）。"""
        episode_rows = build_episode_rows(
            sample_completed=self.sample_task_completed,
            generation=self.generation,
            sample_task_seq=self.sample_task_seq,
            learn_results=self.learn_results,
            learner_names=[lc.name for lc in self.learner_configs],
        )
        output_names = set(self._auto_ml.config.metric_outputs) if self._auto_ml is not None else set()
        metric_output_rows = build_metric_output_rows(
            sample_completed=self.sample_task_completed,
            generation=self.generation,
            sample_metrics=self.sample_metrics,
            output_names=output_names,
        )

        artifact_paths: Dict[str, List[str]] = {}
        if self._last_reward_curve_path:
            artifact_paths["reward_curve"] = [self._last_reward_curve_path]
        if self._last_reward_details_path:
            artifact_paths["reward_details"] = [self._last_reward_details_path]
        if self._checkpoint_model_paths:
            artifact_paths["models"] = list(self._checkpoint_model_paths)
        # 与 snapshot 同源：盘中评估 GIF 随 restart 归档包一并落盘。
        if self._mid_eval_gif_paths:
            artifact_paths["mid_eval_gifs"] = list(self._mid_eval_gif_paths)
        if self._end_eval_gif_paths:
            artifact_paths["end_eval_gifs"] = list(self._end_eval_gif_paths)
        if self._last_end_eval_reward_path:
            artifact_paths["end_eval_reward"] = [self._last_end_eval_reward_path]
        # 转换回 AutoML 的既有代次归档协议；它不是任务级产物登记格式。
        for artifact in self._archive_contributed_artifacts:
            category = artifact.metadata["category"]
            artifact_paths.setdefault(category, []).append(artifact.path)

        return TrainingRunArchive(
            env_name=self.env_name,
            task_id=self.task_id,
            stage_name=self.stage_name,
            generation=self.generation,
            archive_seq=self._archive_seq,
            completed_episodes=self.sample_task_completed,
            episode_rows=episode_rows,
            metric_output_rows=metric_output_rows,
            artifact_paths=artifact_paths,
            snapshot=snapshot,
            advice=advice,
            run_seed=self._generation_run_seed,
            initialization_seed=self._initialization_seed,
        )

    def _ensure_run_seed_locked(
            self,
            *,
            advice: Optional[AutoMLAdvice] = None,
    ) -> Optional[int]:
        """确定本 generation 的 task 级 seed，并写回 TrainConfig 供令牌派生。"""
        if not isinstance(self.task_config, TrainConfig):
            self._generation_run_seed = None
            return None
        if advice is not None and advice.seed is not None:
            self._generation_run_seed = normalize_seed(self.task_config.seed)
            return self._generation_run_seed
        if self._user_fixed_seed:
            self._generation_run_seed = normalize_seed(self.task_config.seed)
            return self._generation_run_seed
        derived = derive_generation_run_seed(self.task_id, self.stage_name, self.generation)
        self.task_config.seed = derived
        self._generation_run_seed = derived
        LOG.info(
            f"TrainTaskManager derived run seed: seed={derived}, "
            f"stage_name={self.stage_name}, generation={self.generation}"
        )
        return derived

    def _apply_automl_advice_locked(self, advice: AutoMLAdvice) -> None:
        """restart 前把 AutoML 建议写入 Learner、seed 与任务级环境 metadata。"""
        if advice.seed is not None and isinstance(self.task_config, TrainConfig):
            self.task_config.seed = advice.seed
        if advice.env_metadata_overrides:
            from envs.base import BaseEnvironment
            self.task_config.env_metadata = BaseEnvironment._deep_merge_metadata(
                self.task_config.env_metadata or {}, advice.env_metadata_overrides,
            )
        self._current_env_metadata_overrides = copy.deepcopy(advice.env_metadata_overrides)
        for learner_name, overrides in advice.algo_config_overrides.items():
            learner_config = next(
                (lc for lc in self.learner_configs if lc.name == learner_name),
                None,
            )
            if learner_config is None or learner_config.algo_config is None:
                continue
            for key, value in overrides.items():
                if hasattr(learner_config.algo_config, key):
                    setattr(learner_config.algo_config, key, value)
        for learner_name, overrides in advice.skeleton_overrides.items():
            learner_config = next(
                (lc for lc in self.learner_configs if lc.name == learner_name),
                None,
            )
            if learner_config is None:
                continue
            for key, value in overrides.items():
                if hasattr(learner_config, key):
                    setattr(learner_config, key, value)

    def record_checkpoint_paths(self, paths: List[str]) -> None:
        """Learner 落盘路径写入 TM，供 AutoML 归档使用。"""
        with self._lock:
            self._checkpoint_model_paths.extend(paths)

    def _plan_artifact_task(self) -> None:
        """用本任务 archive_save_dir 规划 (task, stage)，避免模块 register 抢先写入 AM 默认目录。"""
        if not self.task_config.artifacts_archive_enabled:
            return
        self.artifacts_manager.ensure_task(
            self.task_id,
            self.stage_name,
            save_dir=self.task_config.artifacts_archive_save_dir,
            task_type="evaluation" if isinstance(self.task_config, EvalConfig) else "training",
        )
        # 原始 TOML 副本进运行期台账（Harness 查询）；归档 config/ 仍由 raw 摘要唯一产出。
        self.artifacts_manager.put_config(self.task_id, self.stage_name)

    def put_runtime_file(
            self,
            artifact_type: ArtifactType,
            path: str,
            metadata: dict | None = None,
    ) -> None:
        """生产者或 TM 自己 put 已落盘文件；目录逐文件 put（一文件一行）；未启用归档时跳过。"""
        if not self.task_config.artifacts_archive_enabled:
            return
        if not path or not os.path.exists(path):
            return
        self._plan_artifact_task()
        if os.path.isdir(path):
            self._put_directory_files(artifact_type, path, metadata)
            return
        self.artifacts_manager.put(
            self.task_id,
            self.stage_name,
            ArtifactPut(artifact_type, str(path), dict(metadata or {})),
        )

    def _put_directory_files(
            self,
            artifact_type: ArtifactType,
            directory: str,
            metadata: dict | None = None,
    ) -> None:
        """目录产物（如 AutoML 代次归档）逐文件登记；put 之后新增的文件不进本次归档。"""
        root = Path(directory)
        files = sorted(p for p in root.rglob("*") if p.is_file())
        seen_names: dict[str, Path] = {}
        for file_path in files:
            relative = file_path.relative_to(root)
            if relative.name in seen_names:
                raise ArtifactsArchiveException(
                    operation="put",
                    detail=(
                        f"directory contains duplicate basenames, cannot archive per-file: "
                        f"{seen_names[relative.name]} vs {file_path}"
                    ),
                )
            seen_names[relative.name] = file_path
            payload = dict(metadata or {})
            payload["relative_path"] = relative.as_posix()
            self.artifacts_manager.put(
                self.task_id,
                self.stage_name,
                ArtifactPut(artifact_type, str(file_path), payload),
            )

    def put_report_outputs(self, outputs) -> None:
        """把 finished_report 返回字典中的实际文件登记为报告产物。"""
        if not isinstance(outputs, dict):
            return
        for name, raw_value in outputs.items():
            values = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
            for value in values:
                if not isinstance(value, (str, os.PathLike)):
                    continue
                path = os.path.expanduser(str(value))
                if os.path.isfile(path):
                    self.put_runtime_file(
                        ArtifactType.REPORT, path, {"output_name": str(name)},
                    )

    def _put_owned_runtime_files(self) -> None:
        """收口时 put TM 已知路径：checkpoint / AutoML 等；GIF 由 sample 侧 put_bytes 直送。"""
        with self._lock:
            checkpoint_paths = list(self._checkpoint_model_paths)
            automl_paths = list(self._automl_archive_paths)
            artifacts = list(self._archive_contributed_artifacts)
            artifacts.extend(self._bypass_episode_artifacts)
            for episode_artifacts in self.episode_artifacts:
                artifacts.extend(episode_artifacts)
        for path in checkpoint_paths:
            self.put_runtime_file(ArtifactType.MODEL, path)
        for path in automl_paths:
            self.put_runtime_file(ArtifactType.AUTOML, path)
        for artifact in artifacts:
            if artifact.artifact_type is ArtifactType.RENDER_GIF:
                # 包四起 GIF 经 RolloutSampleWorker.render_gif_put_caller 直送中心台账，
                # 含 worker 跨机场景；TM 侧业务回报路径(_mid/_end_eval_gif_paths)不受影响。
                continue
            self.put_runtime_file(artifact.artifact_type, artifact.path, artifact.metadata)

    def _close_result_artifact_handles(self) -> None:
        """正常任务收口时刷盘并关闭将被归档的结果文件。"""
        with self._reward_curve_file_lock:
            if self._reward_curve_file is not None:
                self._reward_curve_file.flush()
                self._reward_curve_file.close()
                self._reward_curve_file = None
                self._reward_curve_file_path = None
        with self._reward_details_file_lock:
            if self._reward_details_file is not None:
                self._reward_details_file.flush()
                self._reward_details_file.close()
                self._reward_details_file = None
                self._reward_details_file_path = None

    def record_mid_eval_gif(self, path: str) -> None:
        """直接登记 mid_eval GIF；常规路径是 rollout 经 report_sample_result(artifacts=...) 上报。"""
        with self._lock:
            if path:
                self._mid_eval_gif_paths.append(path)
                self._bypass_episode_artifacts.append(Artifact(ArtifactType.RENDER_GIF, path))

    def notify_task_complete(self) -> Optional[ArchivedTask]:
        """任务正常结束的唯一收口：业务产物完成后自动执行阶段级归档。"""
        with self._task_completion_condition:
            while self._task_completion_in_progress and not self._task_completion_done:
                self._task_completion_condition.wait()
            if self._task_completion_done:
                return self._task_completion_result
            if self._closed:
                return None
            self._task_completion_in_progress = True

        try:
            if isinstance(self.task_config, TrainConfig):
                final_paths = self.save_models("final")
                if final_paths:
                    self.record_checkpoint_paths(final_paths)
                    LOG.info(
                        f"Stage final models saved: stage={self.stage_name}, files={len(final_paths)}"
                    )
                with self._lock:
                    auto_ml = self._auto_ml
                    snapshot = (
                        self._collect_training_snapshot_locked()
                        if auto_ml is not None
                        else None
                    )
                if auto_ml is not None:
                    auto_ml.on_stage_complete(snapshot)
                    self._archive_completed_generation(
                        AutoMLAdvice(decision="complete", reason="stage_complete"),
                    )

            self._close_result_artifact_handles()
            self._put_owned_runtime_files()
            archived = None
            if self.task_config.artifacts_archive_enabled:
                self._plan_artifact_task()
                archived = self.artifacts_manager.archive_task(
                    self.task_id,
                    self.stage_name,
                )
                LOG.info(
                    f"Task artifacts archived: task_id={self.task_id}, "
                    f"stage={self.stage_name}, archive_dir={archived.archive_dir}"
                )
        except BaseException:
            with self._task_completion_condition:
                self._task_completion_in_progress = False
                self._task_completion_condition.notify_all()
            raise
        with self._task_completion_condition:
            self._task_completion_result = archived
            self._task_completion_done = True
            self._task_completion_in_progress = False
            self._task_completion_condition.notify_all()
        return archived

    def notify_stage_complete(self) -> Optional[ArchivedTask]:
        """兼容训练/分阶段调用；统一转发到任务结束入口。"""
        return self.notify_task_complete()

    def add_restart_callback(self, callback) -> None:
        """注册任务重启后的本地资源清理回调；RPC场景下由服务端Role注册。"""
        with self._lock:
            if callback not in self._restart_callbacks:
                self._restart_callbacks.append(callback)

    def restart_training_task(
            self,
            reason: str = "",
            advice: Optional[AutoMLAdvice] = None,
    ) -> Dict[str, Any]:
        """
        请求重启当前训练任务的采样/学习进度。

        该接口面向未来AutoML监控进程：调用方可以在任意时刻请求重启，但TaskManager只会在安全边界
        真正执行。安全边界要求没有正在执行的采样局、没有已释放未领取的学习令牌、没有已分发未回报的
        学习令牌。generation 是重启后的唯一版本信号，Role/ReplayBuffer/Learner按该值自行同步状态。
        """
        assert isinstance(self.task_config, TrainConfig), (
            f"restart_training_task requires TrainConfig, given={type(self.task_config).__name__}"
        )
        callbacks = []
        with self._lock:
            if not self._can_restart_now_locked():
                if self._pending_restart_reason is None:
                    self._pending_restart_reason = reason
                if advice is not None:
                    self._pending_restart_advice = advice
                    LOG.warning(
                        f"TrainTaskManager restart request pending: stage_name={self.stage_name}, "
                        f"generation={self.generation}, reason={reason}"
                    )
                return {
                    "stage_name": self.stage_name,
                    "generation": self.generation,
                    "pending_restart": True,
                    "reason": self._pending_restart_reason,
                }
            result, callbacks = self._restart_training_task_locked(reason, advice)
        self._run_restart_callbacks(result, callbacks, advice)
        if result is not None and not result.get("pending_restart"):
            self._notify_training_resource_event(
                TrainingResourceEvent.POST_RESTART,
                int(result["generation"]),
                int(result.get("prior_completed_episodes", 0)),
                advice=result.get("advice"),
                reason=str(result.get("reason") or ""),
            )
        return result

    def _apply_pending_restart_if_ready_locked(self) -> Tuple[Optional[Dict[str, Any]], list]:
        """在采样/学习回报后的安全边界兑现挂起的重启请求。调用方必须持有self._lock。"""
        if self._pending_restart_reason is None:
            return None, []
        if not self._can_restart_now_locked():
            return None, []
        reason = self._pending_restart_reason
        advice = self._pending_restart_advice
        self._pending_restart_reason = None
        self._pending_restart_advice = None
        return self._restart_training_task_locked(reason, advice)

    def _can_restart_now_locked(self) -> bool:
        """判断当前是否处于可重启安全边界。调用方必须持有self._lock。"""
        # 租约表是唯一在途视图：任一 kind 在途都不可换代。
        if self._lease_table or self._version_sealed:
            return False
        if self._bypass_sample_queue:
            return False
        # ready 或 in-flight 未排空时不能换代，否则许可或迟到回报会跨 generation。
        if self._versioned_sampling and self._version_outstanding:
            # A returned ticket may block a ready batch. Restart discards that batch.
            return not self._inflight_learn_tokens
        return not self._learn_ready and not self._inflight_learn_tokens

    def _restart_training_task_locked(
            self,
            reason: str,
            advice: Optional[AutoMLAdvice] = None,
    ) -> Tuple[Dict[str, Any], list]:
        """执行真实重启动作。调用方必须持有self._lock，回调由外层释放锁后执行。"""
        effective_advice = advice or self._pending_restart_advice
        if effective_advice is not None:
            self._apply_automl_advice_locked(effective_advice)
        prior_completed_episodes = self.sample_task_completed
        self._pending_restart_reason = None
        self._pending_restart_advice = None
        self.generation += 1
        self._clear_inflight_leases_locked(reason=f"generation_restart:{reason}")
        self._ensure_run_seed_locked(advice=effective_advice)
        # 与 __init__ 共用同一套 _reset_*，避免漏重置（例如 _last_automl_restart_episode）。
        self._version_sealed.clear()
        self._version_outstanding.clear()
        self._load_version_rules()
        self._reset_sample_state()
        self._reset_learn_state()
        self._reset_metrics_state()
        self._publish_training_progress_locked()
        self._start_monitors()
        LOG.warning(
            f"TrainTaskManager restarted training task: stage_name={self.stage_name}, "
            f"generation={self.generation}, reason={reason}"
        )
        result = {
            "stage_name": self.stage_name,
            "generation": self.generation,
            "pending_restart": False,
            "reason": reason,
            "advice": effective_advice,
            "prior_completed_episodes": prior_completed_episodes,
        }
        return result, list(self._restart_callbacks)

    @staticmethod
    def _run_restart_callbacks(
            restart_result: Optional[Dict[str, Any]],
            callbacks: list,
            advice: Optional[AutoMLAdvice] = None,
    ) -> None:
        """
        释放TaskManager锁后通知本地Role同步代次。

        generation是任务重启的唯一状态信号；Role、ReplayBuffer、Learner各自按代次完成本地同步，
        TaskManager不向外传播“清缓存/恢复网络”等动作意图。
        """
        if restart_result is None:
            return
        generation = int(restart_result["generation"])
        effective_advice = advice if advice is not None else restart_result.get("advice")
        for callback in callbacks:
            try:
                callback(generation, effective_advice)
            except TypeError:
                callback(generation)

    def is_learning_completed(self) -> bool:
        """采样已关闭，且没有尾部 ready 或 in-flight 许可时进入学习静默态。"""
        assert isinstance(self.task_config, TrainConfig), (
            f"is_learning_completed requires TrainConfig, given={type(self.task_config).__name__}"
        )
        with self._lock:
            return self._is_learning_completed_locked()

    def _sample_quota_consumed_locked(self) -> bool:
        """主路采样是否已经耗尽；offline 索引未准备时不能提前关闭。"""
        if isinstance(self.task_config, TrainConfig) and (
                self.task_config.sample_type == SampleType.OFFLINE.value
                and not self._offline_plan_prepared
        ):
            return False
        return self._sample_closed or self.sample_task_completed >= int(self.max_episode)

    def _is_learning_completed_locked(self) -> bool:
        if not isinstance(self.task_config, TrainConfig):
            return False
        return (
            self._sample_quota_consumed_locked()
            and not self._learn_ready
            and not self._inflight_learn_tokens
            and not self._version_sealed
        )

    @property
    def run_seed(self) -> Optional[int]:
        """本 generation 任务级随机种子，供 episode/learn 令牌派生。"""
        with self._lock:
            return self._generation_run_seed

    def get_task_manager_info(self) -> Dict:
        """ 返回任务基本信息汇总 """
        with self._lock:
            learned_episode_count = int(self._learned_sample_watermark)
            sample_versions = self._current_sample_versions_locked()
        return {
            'task_id': self.task_id,
            'stage_name': self.stage_name,
            'generation': self.generation,
            'total_sample_episodes': self.max_episode,
            'completed_sample_episode': self.sample_task_completed,
            'current_sample_episode': self.current_sample_task,
            'completed_learning_steps': self.learning_completed,
            'completed_learning_episode': learned_episode_count,
            'sample_learn_lag': self.sample_task_completed - learned_episode_count,
            'learn_ready': self._learn_ready,
            'inflight_learning_count': len(self._inflight_learn_tokens),
            'sample_model_version': next(iter(sample_versions.values())) if len(sample_versions) == 1 else None,
            'sample_model_versions': sample_versions,
        }

    def get_eval_episode_records(self) -> List[Dict[str, Any]]:
        """评估中心抽取每局采样上报结果，以及当时的令牌快照。"""
        with self._lock:
            records: List[Dict[str, Any]] = []
            for idx, task in enumerate(self.sample_task_seq):
                if task.end_datetime is None:
                    continue
                total_rewards = {
                    agent_name: float(rewards[idx])
                    for agent_name, rewards in self.sample_rewards.items()
                    if idx < len(rewards) and rewards[idx] is not None
                }
                episode_metrics = {
                    metric_name: float(series[idx])
                    for metric_name, series in self.sample_metrics.items()
                    if idx < len(series) and series[idx] is not None
                }
                reward_details = None
                if idx < len(self.episode_rewards_details) and self.episode_rewards_details[idx]:
                    reward_details = dict(self.episode_rewards_details[idx])
                records.append({
                    "episode": int(task.episode_num),
                    "total_rewards": total_rewards,
                    "episode_metrics": episode_metrics,
                    "episode_rewards_details": reward_details,
                    # 保留报告旧字段，但不向业务钩子泄露内部产物清单。
                    "artifacts": None,
                    "episode_info": self._eval_episode_info_snapshot(task),
                })
            return records

    @staticmethod
    def _eval_episode_info_snapshot(task: EpisodeInfo) -> Dict[str, Any]:
        snapshot = dict(task.to_dict())
        snapshot["task_id"] = task.task_id
        snapshot["start_datetime"] = task.start_datetime
        snapshot["end_datetime"] = task.end_datetime
        return snapshot

    @property
    def sample_task_process(self) -> Tuple[int, int]:
        """ 返回采样任务进度，平台上报进度 """
        return self.sample_task_completed, self.max_episode

    @property
    def learning_process(self) -> Tuple[int, int]:
        """返回实际更新次数；学习由样本更新触发，没有预先固定的总轮次，分母为 0。"""
        return self.learning_completed, 0

    def custom_episode(self, episode_info: EpisodeInfo, **kwargs):
        """ 修改训练局元数信息，项目定制化接口 """
        episode_info.metadata = episode_info.metadata or {}

    def push_info(self, attrs: dict[str, Any]) -> None:
        """合并进本进程 info_bridge；后写覆盖同名键。"""
        info_bridge.put(**dict(attrs or {}))

    def pull_info(self, key: str | None = None) -> Any:
        """key 为 None 返回 snapshot()；否则 get(key)，缺键返回 None。"""
        return public_training_info(key)

    def _build_episode_action_params(
            self,
            epsilon: float,
            deterministic: bool,
            temperature: float,
    ) -> Dict[AgentName, ActionParam]:
        """按AgentName生成本局动作选择参数，作为EpisodeInfo令牌的权威字段。"""
        action_params: Dict[AgentName, ActionParam] = {}
        for agent_config in self.policy_agent_configs:
            fixed_mode = bool(agent_config.fixed_mode)
            action_params[agent_config.name] = ActionParam(
                epsilon=(
                    0.0
                    if fixed_mode
                    else self._resolve_agent_epsilon(agent_config, epsilon)
                ),
                deterministic=bool(deterministic or fixed_mode),
                temperature=temperature,
            )
        return action_params

    @staticmethod
    def _resolve_agent_epsilon(agent_config: AgentConfig, global_epsilon: float) -> float:
        """下发策略显式声明会消费的探索退火系数。"""
        if (agent_config.agent_output_type == AgentOutputType.Q.value
                or agent_config.agent_type == AgentType.MADDPG_MLP.value):
            return float(global_epsilon)
        return 0.0

    def _epsilon_warmup_episodes(self) -> int:
        """用Learner算法配置决定初始全随机探索局数，避免采样策略依赖ReplayBuffer配置。"""
        warmup_episodes = 0
        for learner_config in self.learner_configs:
            algo_config = learner_config.algo_config
            assert algo_config is not None, (
                f"LearnerConfig[{learner_config.name}].algo_config must be initialized before warmup planning."
            )
            if getattr(algo_config, "sample_strategy") in (
                    SamplingStrategy.UNIFORM_STEPS.value,
                    SamplingStrategy.ON_POLICY_RECENT_STEPS.value,
                    SamplingStrategy.ON_POLICY_CURSOR_STEPS.value,
                    SamplingStrategy.TIME_SLICE.value,
            ):
                warmup_episodes = max(warmup_episodes, 1)
                continue
            warmup_episodes = max(warmup_episodes, int(getattr(algo_config, "batch_size")))
        return warmup_episodes

    def sample_gate_state(self, sample_role_name: str = "unknown") -> Dict[str, Any]:
        """采样进程等待/领取前的统一闸门状态。"""
        with self._lock:
            admission_reason = self._admission_block_reason
            backpressure = self._sample_backpressure_state_locked()
            version_blocked = (
                self._versioned_sampling and not self._bypass_sample_queue
                and self.sample_task_completed < self.max_episode
                and self._versions_for_sample_ticket_locked() is None
            )
        reason = admission_reason
        if reason is None and backpressure["enabled"]:
            reason = "backpressure"
        if reason is None and version_blocked:
            reason = "waiting_for_version_batch"
        return {"blocked": bool(reason), "reason": reason,
                "retry_after_seconds": 0.5 if reason else 0.0, "backpressure": backpressure}

    def sample_backpressure_state(self, sample_role_name: str = "unknown") -> Dict[str, Any]:
        """采样进程领取令牌前查询采样背压状态；日志由采样滚动流程负责打印。"""
        with self._lock:
            admission_reason = self._admission_block_reason
            state = self._sample_backpressure_state_locked()
        # 准入拒绝必须 enabled=True，否则 rollout 会把 None 当成「无任务」直接退出。
        if admission_reason:
            state["enabled"] = True
            state["reason"] = admission_reason
            return state
        if state["enabled"]:
            state["reason"] = (
                "Sample backpressure enabled: "
                f"sample_role={sample_role_name}, "
                f"sample_completed={state['sample_completed']}, "
                f"learn_completed_episode={state['learn_completed_episode']}, "
                f"lag={state['lag']}, max_lag={state['max_lag']}. "
                "Sample worker is waiting for learner/model update."
            )
        else:
            state["reason"] = None
        return state

    def _sample_backpressure_state_locked(self) -> Dict[str, Any]:
        """
        判断采样是否需要等待学习追进度。调用方必须持有 self._lock。

        使用“完成采样局数 - 最近完成学习所覆盖的采样水位”。ReplayBuffer 尚未达到
        任一 Learner 的 batch 条件时不启背压，避免 warmup 阶段形成等待环。
        """
        learned_episode_count = int(self._learned_sample_watermark)
        retry_episode = next((
            task.episode_num
            for task in self.sample_task_seq
            if task.failed_count > 0
            and task.start_datetime is None
            and task.end_datetime is None
        ), None)
        default_state = {
            "enabled": False,
            "sample_completed": self.sample_task_completed,
            "learn_completed_episode": learned_episode_count,
            "lag": self.sample_task_completed - learned_episode_count,
            "max_lag": None,
            "retry_episode": retry_episode,
        }
        if not isinstance(self.task_config, TrainConfig):
            return default_state
        max_lag = self.task_config.sample_learn_max_lag
        if max_lag is None:
            return default_state
        assert max_lag >= 0, f"sample_learn_max_lag must be non-negative when set, got={max_lag}"
        if self._decide_learn_permit_locked() is None and not self._inflight_learn_tokens:
            return {
                **default_state,
                "max_lag": max_lag,
            }
        lag = self.sample_task_completed - learned_episode_count
        return {
            # A returned token must be claimable even when later parallel
            # episodes have pushed the global lag over the limit. Once one
            # worker claims it, start_datetime is set and backpressure resumes.
            "enabled": lag > max_lag and retry_episode is None,
            "sample_completed": self.sample_task_completed,
            "learn_completed_episode": learned_episode_count,
            "lag": lag,
            "max_lag": max_lag,
            "retry_episode": retry_episode,
        }

    def learn_batches_update(self, replay_buffer_state: ReplayBufferState, episode_num: int):
        """
        缓存池完成一次样本更新后刷新学习许可。

        TrainTaskManager 只消费 ReplayBufferState，不直接持有 ReplayBuffer；
        多次 push 只合并成一张 ready，不累积逐 episode 学习债务。
        """
        assert isinstance(self.task_config, TrainConfig), (
            f"learn_batches_update requires TrainConfig, given={type(self.task_config).__name__}"
        )
        assert isinstance(replay_buffer_state, ReplayBufferState), (
            f"learn_batches_update requires ReplayBufferState, given={type(replay_buffer_state).__name__}"
        )
        with self._lock:
            if replay_buffer_state.generation != self.generation:
                LOG.warning(
                    f"Ignore stale ReplayBufferState: state_generation={replay_buffer_state.generation}, "
                    f"task_generation={self.generation}, episode_num={episode_num}"
                )
                return
            assert 1 <= episode_num <= len(self.sample_task_seq), (
                f"sample episode_num must be in [1, {len(self.sample_task_seq)}], given={episode_num}"
            )
            if self._versioned_sampling and not self._is_latest_versioned_replay_state_locked(replay_buffer_state):
                return
            self.replay_buffer_state = replay_buffer_state
            self._latest_replay_sample_watermark = max(
                int(self._latest_replay_sample_watermark),
                min(int(self.max_episode), int(self.sample_task_completed) + 1),
            )
            if self._sample_closed or self._pending_restart_reason is not None:
                return
            self._refresh_learn_ready_locked()

    def _is_latest_versioned_replay_state_locked(self, state: ReplayBufferState) -> bool:
        previous = self.replay_buffer_state
        return (state.generation == self.generation
                and state.versioned_writes >= previous.versioned_writes
                and state.overlap_reads >= previous.overlap_reads)

    def _decide_learn_permit_locked(self) -> Optional[LearnPermitGrant]:
        """按当前 ReplayBuffer 快照计算一张许可；不修改状态。"""
        if self._versioned_sampling:
            ready = [name for name, version in self._current_sample_versions_locked().items()
                     if name not in self._version_sealed
                     and not replay_warmup_remaining(
                         self.replay_buffer_state,
                         next(config.algo_config for config in self.learner_configs if config.name == name), name)
                     and (name in self.replay_buffer_state.overlap_ready_agents if name in self._overlap_learners
                          else not self._version_ticket_demand_locked(version, name))]
            return LearnPermitGrant(active_learners=ready) if ready else None
        algo_configs = {
            learner_config.name: learner_config.algo_config
            for learner_config in self.learner_configs
            if learner_config.algo_config is not None
        }
        grant = self._learn_permit_policy.decide(LearnPermitRequest(
            generation=self.generation,
            stage_name=self.stage_name,
            replay_buffer_state=self.replay_buffer_state,
            candidate_learners=tuple(self._enabled_learner_names()),
            algo_configs=algo_configs,
        ))
        if grant is None:
            return None
        ready = [name for name in grant.active_learners if name in algo_configs
                 and not replay_warmup_remaining(self.replay_buffer_state, algo_configs[name], name)]
        return LearnPermitGrant(active_learners=ready) if ready else None

    def _refresh_learn_ready_locked(self) -> None:
        """把当前可学习状态合并成至多一张 ready 许可。"""
        grant = self._decide_learn_permit_locked()
        if grant is None:
            self._clear_learn_ready_locked()
            return
        self._learn_ready = True
        self._learn_ready_active_learners = list(grant.active_learners)
        self.learn_count = 1

    def _clear_learn_ready_locked(self) -> None:
        self._learn_ready = False
        self._learn_ready_active_learners = []
        self.learn_count = 0

    def _resolve_inflight_learn_info_locked(
            self,
            *,
            learn_id: int,
            episode_num: int,
    ) -> Optional[LearnInfo]:
        """解析学习回报；learn_id=0 时兼容唯一 in-flight 的旧调用。"""
        resolved_id = int(learn_id)
        if resolved_id <= 0:
            if len(self._inflight_learn_tokens) != 1:
                LOG.warning(
                    f"Cannot infer learn_id: episode={episode_num}, "
                    f"inflight={sorted(self._inflight_learn_tokens)}"
                )
                return None
            resolved_id = next(iter(self._inflight_learn_tokens))
        learn_info = self._inflight_learn_tokens.get(resolved_id)
        if learn_info is None:
            if resolved_id not in self._completed_learn_ids:
                LOG.warning(
                    f"Discard unknown learning result: learn_id={resolved_id}, "
                    f"episode={episode_num}, generation={self.generation}"
                )
            return None
        if int(learn_info.episode_num) != int(episode_num):
            LOG.warning(
                f"Discard mismatched learning result: learn_id={resolved_id}, "
                f"episode={episode_num}, expected_episode={learn_info.episode_num}"
            )
            return None
        return learn_info

    def _enabled_learner_names(self) -> List[AgentName]:
        """筛选 learn_times_per_episode 大于 0 的 Learner，作为学习编排候选。"""
        learner_names: List[AgentName] = []
        for learner_config in self.learner_configs:
            algo_config = learner_config.algo_config
            assert algo_config is not None, (
                f"LearnerConfig[{learner_config.name}].algo_config must be initialized before learning planning."
            )
            if int(getattr(algo_config, "learn_times_per_episode")) > 0:
                learner_names.append(learner_config.name)
        return learner_names

    def _build_learn_token(
            self,
            *,
            learn_id: int,
            sample_watermark: int,
            active_learners: List[AgentName],
            seed_iteration: int,
    ) -> LearnInfo:
        """把一张许可封装为 LearnInfo；算法配置在领取时做快照。"""
        learner_configs = {
            learner_config.name: learner_config
            for learner_config in self.learner_configs
        }
        learn_seed = self._learn_seed(sample_watermark, seed_iteration)
        algo_configs = {
            learner_name: copy.deepcopy(learner_configs[learner_name].algo_config)
            for learner_name in active_learners
        }
        return LearnInfo(
            learn_id=learn_id,
            episode_num=sample_watermark,
            active_learners=active_learners,
            algo_configs=algo_configs,
            seed=learn_seed,
            stage_name=self.stage_name,
            generation=self.generation,
        )

    def _episode_seed(self, episode_num: int, *parts: object) -> Optional[int]:
        """按任务seed、阶段和episode派生采样令牌seed；seed=None时保持完全随机。"""
        return derive_seed(getattr(self.task_config, "seed", None), self.stage_name, "sample", episode_num, *parts)

    def _next_learn_seed_iteration_locked(self, sample_watermark: int) -> int:
        """同一采样水位再次发令牌时分配从 0 开始的 seed 子序号。"""
        watermark = int(sample_watermark)
        if self._learn_seed_watermark != watermark:
            self._learn_seed_watermark = watermark
            self._learn_seed_iteration = 0
            return 0
        self._learn_seed_iteration += 1
        return int(self._learn_seed_iteration)

    def _learn_seed(self, sample_watermark: int, seed_iteration: int) -> Optional[int]:
        """
        派生学习 seed。

        每个新采样水位的首次更新保留旧版 episode seed，使串行基线不变；
        学习期间又 push 导致同一水位再发令牌时再用子序号避免重复抽取同一批数据。
        """
        parts: tuple[object, ...] = (
            (sample_watermark,)
            if int(seed_iteration) == 0
            else (sample_watermark, int(seed_iteration))
        )
        return derive_seed(
            getattr(self.task_config, "seed", None),
            self.stage_name,
            "learn",
            *parts,
        )

    @staticmethod
    def _replay_buffer_ready_for_learner(replay_buffer_state: ReplayBufferState, algo_config) -> bool:
        """判断ReplayBuffer是否具备发放学习令牌的最低条件。"""
        return replay_buffer_ready_for_learner(replay_buffer_state, algo_config)

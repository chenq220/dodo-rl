from __future__ import annotations
from datetime import datetime
from dataclasses import dataclass, field
from dataclasses_json import DataClassJsonMixin
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from common.types import AgentName, AgentType
    from config import MappoConfig, QmixConfig

    AlogsConfigType = Union[MappoConfig, QmixConfig]


@dataclass
class ActionParam:
    """单个Agent在一局采样中的动作选择参数，由EpisodeInfo令牌直接下发。"""
    # Q 策略的 epsilon-greedy 探索概率。
    epsilon: float
    # 是否使用确定性动作选择；评估模式可打开，训练模式通常关闭。
    deterministic: bool
    # 动作分布温度系数；具体Agent可选择是否使用。
    temperature: float


@dataclass
class EpisodeInfo:
    """一局采样任务令牌，任务总线生成令牌，滚动流程获取令牌后执行。"""
    # 当前 episode 编号，从 1 开始，通常也用于日志和样本录制。
    episode_num: int
    # 当前训练/评估任务 ID，来自 TrainTaskManager.task_id。
    task_id: str
    # 本局全局探索系数，用于日志和快速查看。
    epsilon: float
    # 全局默认确定性动作标记，只用于日志和快速查看。
    deterministic: bool
    # 全局默认动作分布温度，只用于日志和快速查看。
    temperature: float
    # 各Agent本局动作选择参数；滚动流程必须严格读取此字段，不得自行推断或改写。
    actions_param: Dict[AgentName, ActionParam]
    # 本局采样随机种子；由TrainTaskManager从TaskConfig.seed派生，采样流程必须按令牌应用。
    seed: Optional[int]
    # episode 分发给采样进程的时间；非空表示该令牌正在执行，不能再次分发。
    start_datetime: Optional[datetime]
    # episode 采样完成并回写状态的时间；非空表示该令牌已经完成，不能再次分发。
    end_datetime: Optional[datetime]
    # 项目定制元数据，TrainTaskManager.custom_episode 可在此注入额外信息。
    metadata: Optional[Dict]
    # 每次领取生成的唯一租约ID；同一episode超时重发后，旧进程的迟到结果据此被丢弃。
    lease_id: str = ""
    # 当前任务对环境 metadata 的增量覆盖；采样进程必须在本局 reset 前应用。
    env_metadata: Dict = field(default_factory=dict)
    # AutoML 逐局指标提取定义，随 sample 令牌下发给采样进程。
    metric_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 本局采样失败次数；失败归还时累加，不计入采样进度，用于调度观察和后续熔断策略。
    failed_count: int = 0
    # 最近一次采样失败时间。
    last_failed_datetime: Optional[datetime] = None
    # 当前所属训练阶段名称；普通训练为默认阶段 train，分阶段训练由 StageConfig.stage_name 决定。
    stage_name: str = "train"
    # 当前任务代次；TaskManager重启任务后递增，用于后续过滤重启前遗留采样结果。
    generation: int = 0
    # 本局由在线环境还是历史记录驱动；必须随令牌下发到采样进程。
    sample_type: str = "online"
    # Empty means load the champion. All entries belong to the same sampling round.
    required_model_versions: Dict[AgentName, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "episode_num": self.episode_num,
            "epsilon": self.epsilon,
            "deterministic": self.deterministic,
            "temperature": self.temperature,
            "actions_param": {
                agent_name: {
                    "epsilon": action_param.epsilon,
                    "deterministic": action_param.deterministic,
                    "temperature": action_param.temperature,
                }
                for agent_name, action_param in self.actions_param.items()
            },
            "seed": self.seed,
            "lease_id": self.lease_id,
            "metadata": self.metadata,
            "env_metadata": self.env_metadata,
            "metric_outputs": self.metric_outputs,
            "failed_count": self.failed_count,
            "last_failed_datetime": self.last_failed_datetime,
            "stage_name": self.stage_name,
            "generation": self.generation,
            "sample_type": self.sample_type,
            "required_model_versions": self.required_model_versions,
        }


@dataclass
class LearnInfo:
    """学习任务令牌，任务总线生成令牌，滚动流程按令牌执行 Learner.update。"""
    # 领取许可时已被 ReplayBuffer 覆盖的采样完成水位；离线样本学习旧入口可为0。
    # 一张令牌对应一次样本更新触发的学习，不是「第几次梯度更新」。
    episode_num: int = 0
    # 本 generation 内单调递增的学习许可编号；TrainTaskManager 用它结算 in-flight。
    learn_id: int = 0
    # 哪些Learner需要学习；为空表示当前没有具体Learner约束。
    active_learners: list[AgentName] = field(default_factory=list)
    # 各Learner本次学习使用的算法配置，由任务总线下发给滚动流程。
    algo_configs: Dict[AgentName, AlogsConfigType] = field(default_factory=dict)
    # 本学习批次随机种子；由TrainTaskManager从TaskConfig.seed派生，学习流程必须按令牌应用。
    seed: Optional[int] = None
    # 当前所属训练阶段名称；分阶段训练时由 StageConfig.stage_name 决定，普通训练为 train。
    stage_name: str = "train"
    # 当前任务代次；TaskManager重启任务后递增，Learner据此决定是否恢复自己的快照。
    generation: int = 0


@dataclass
class LearnTaskResult:
    """学习任务申请结果，表达任务总线当前是否下发了学习令牌。"""

    # 可执行的学习令牌；None表示当前没有可领取令牌或样本量暂不足。
    learn_info: Optional[LearnInfo] = None
    # 学习任务生命周期是否结束。
    task_finished: bool = False


@dataclass(frozen=True)
class StageLifecycleState:
    """阶段生命周期一次 tick 的只读结果。"""
    sample_completed: bool
    learning_completed: bool
    finished: bool
    pending_restart: bool = False
    reason: str = ""


# =============================================================================
# AutoML 契约（框架侧，非 Learner 接口）
#
# - 可调超参：ALGO_TUNABLE_PARAM[algo_type] + TOML search_space；Learner 不实现任何枚举方法。
# - learn 指标：update() 自由 dict 全量进 learn_results / algos_metrics.csv；推荐键仅作参考。
# =============================================================================


class AutoMLPrimaryMetric(str, Enum):
    """Optuna trial.report 主指标。"""
    REWARD_MEAN = "reward_mean"
    REWARD_MIN_AGENT = "reward_min_agent"
    REWARD_SUM = "reward_sum"
    STAGE_SUCCESS_RATE = "stage_success_rate"


class AutoMLSamplerType(str, Enum):
    TPE = "tpe"
    RANDOM = "random"
    CMAES = "cmaes"


class AutoMLPrunerType(str, Enum):
    MEDIAN = "median"
    HYPERBAND = "hyperband"
    NONE = "none"


@dataclass
class AutoMLParamRangeConfig(DataClassJsonMixin):
    """单个搜索参数的分布范围；空值表示沿用框架注册的默认值。"""
    low: Optional[Union[int, float]] = None
    high: Optional[Union[int, float]] = None
    log: Optional[bool] = None
    step: Optional[Union[int, float]] = None
    choices: Optional[list] = None


@dataclass
class AutoMLLearnerSearchParamConfig(AutoMLParamRangeConfig):
    """Learner 搜索项；path 是相对 LearnerConfig 的字段路径。"""
    agent_name: str = ""
    path: str = ""


@dataclass
class AutoMLEnvMetadataSearchParamConfig(AutoMLParamRangeConfig):
    """自由环境 metadata 搜索项，必须显式给出分布类型与范围。"""
    path: str = ""
    type: str = ""


@dataclass
class AutoMLSeedSearchConfig(DataClassJsonMixin):
    """全局运行种子的整数搜索范围。"""
    low: int = 0
    high: int = 2147483647


@dataclass
class AutoMLMetricOutputConfig(DataClassJsonMixin):
    """逐局业务指标输出；从标准 infos 或本局生效的环境 metadata 中取值。"""
    info_path: Optional[str] = None
    metadata_path: Optional[str] = None
    agent_reduce: str = "mean"
    episode_reduce: str = "mean"


@dataclass
class AutoMLBusinessMetricConfig(DataClassJsonMixin):
    """将多个逐局指标组合成供 AutoML 决策的单一业务指标。"""
    enabled: bool = False
    name: str = ""
    expression: str = ""
    aggregation: str = "mean"
    window: int = 50
    target: Optional[float] = None
    deadline_episodes: int = 0


class ParamLayer(Enum):
    """TunableParam 所属层；决定 restart 时建议值写回何处。"""
    ALGO_CONFIG = "algo_config"  # 写入 learner.algo_config（如 lr / batch_size）
    SKELETON = "skeleton"  # 写入 learner.config skeleton（如 gamma）


@dataclass(frozen=True)
class TunableParamSpec:
    """单个可调超参的 Optuna 搜索空间描述。"""
    # algo_config / skeleton 字段名。
    key: str
    layer: ParamLayer
    # "float" | "int" | "categorical"
    dist_type: str
    low: Union[float, int, None] = None
    high: Union[float, int, None] = None
    log: bool = False
    step: Optional[int] = None
    choices: Optional[tuple] = None


class QmixTunableParam(Enum):
    LR = TunableParamSpec("lr", ParamLayer.ALGO_CONFIG, "float", 1e-5, 1e-2, log=True)
    BATCH_SIZE = TunableParamSpec("batch_size", ParamLayer.ALGO_CONFIG, "int", 8, 32, step=4)
    LEARN_TIMES = TunableParamSpec("learn_times_per_episode", ParamLayer.ALGO_CONFIG, "int", 1, 8)
    TARGET_UPDATE = TunableParamSpec("target_update_interval_or_tau", ParamLayer.ALGO_CONFIG, "float", 50, 500)
    TD_LAMBDA = TunableParamSpec("td_lambda", ParamLayer.ALGO_CONFIG, "float", 0.3, 1.0)
    GAMMA = TunableParamSpec("gamma", ParamLayer.SKELETON, "float", 0.95, 0.999)


class MappoTunableParam(Enum):
    LR = TunableParamSpec("lr", ParamLayer.ALGO_CONFIG, "float", 1e-5, 1e-2, log=True)
    BATCH_SIZE = TunableParamSpec("batch_size", ParamLayer.ALGO_CONFIG, "int", 8, 32, step=4)
    LEARN_TIMES = TunableParamSpec("learn_times_per_episode", ParamLayer.ALGO_CONFIG, "int", 1, 8)
    EPS_CLIP = TunableParamSpec("eps_clip", ParamLayer.ALGO_CONFIG, "float", 0.1, 0.3)
    EPOCHS = TunableParamSpec("epochs", ParamLayer.ALGO_CONFIG, "int", 2, 10)
    ENTROPY_COEF = TunableParamSpec("entropy_coef", ParamLayer.ALGO_CONFIG, "float", 1e-4, 5e-2, log=True)
    GAMMA = TunableParamSpec("gamma", ParamLayer.SKELETON, "float", 0.95, 0.999)


class EpymarlActorCriticTunableParam(Enum):
    LR = TunableParamSpec("lr", ParamLayer.ALGO_CONFIG, "float", 1e-5, 1e-2, log=True)
    BATCH_SIZE = TunableParamSpec("batch_size", ParamLayer.ALGO_CONFIG, "int", 8, 64, step=2)
    LEARN_TIMES = TunableParamSpec("learn_times_per_episode", ParamLayer.ALGO_CONFIG, "int", 1, 8)
    ENTROPY_COEF = TunableParamSpec("entropy_coef", ParamLayer.ALGO_CONFIG, "float", 1e-4, 5e-2, log=True)
    GAMMA = TunableParamSpec("gamma", ParamLayer.SKELETON, "float", 0.95, 0.999)


class ComaTunableParam(Enum):
    LR = TunableParamSpec("lr", ParamLayer.ALGO_CONFIG, "float", 1e-5, 1e-2, log=True)
    BATCH_SIZE = TunableParamSpec("batch_size", ParamLayer.ALGO_CONFIG, "int", 8, 64, step=2)
    ENTROPY_COEF = TunableParamSpec("entropy_coef", ParamLayer.ALGO_CONFIG, "float", 1e-4, 5e-2, log=True)
    GAMMA = TunableParamSpec("gamma", ParamLayer.SKELETON, "float", 0.95, 0.999)


class MaddpgTunableParam(Enum):
    LR = TunableParamSpec("lr", ParamLayer.ALGO_CONFIG, "float", 1e-5, 1e-2, log=True)
    BATCH_SIZE = TunableParamSpec("batch_size", ParamLayer.ALGO_CONFIG, "int", 8, 64, step=4)
    LEARN_TIMES = TunableParamSpec("learn_times_per_episode", ParamLayer.ALGO_CONFIG, "int", 1, 8)
    TARGET_UPDATE = TunableParamSpec(
        "target_update_interval_or_tau", ParamLayer.ALGO_CONFIG, "float", 0.001, 1.0, log=True
    )
    GAMMA = TunableParamSpec("gamma", ParamLayer.SKELETON, "float", 0.95, 0.999)


class PacTunableParam(Enum):
    LR = TunableParamSpec("lr", ParamLayer.ALGO_CONFIG, "float", 1e-5, 1e-2, log=True)
    BATCH_SIZE = TunableParamSpec("batch_size", ParamLayer.ALGO_CONFIG, "int", 8, 64, step=2)
    LEARN_TIMES = TunableParamSpec("learn_times_per_episode", ParamLayer.ALGO_CONFIG, "int", 1, 8)
    INITIAL_ENTROPY = TunableParamSpec(
        "initial_entropy_coef", ParamLayer.ALGO_CONFIG, "float", 0.1, 50.0, log=True
    )
    FINAL_ENTROPY = TunableParamSpec(
        "final_entropy_coef", ParamLayer.ALGO_CONFIG, "float", 1e-4, 1e-1, log=True
    )
    GAMMA = TunableParamSpec("gamma", ParamLayer.SKELETON, "float", 0.95, 0.999)


class PpoTunableParam(Enum):
    LR = TunableParamSpec("lr", ParamLayer.ALGO_CONFIG, "float", 1e-5, 1e-2, log=True)
    BATCH_SIZE = TunableParamSpec("batch_size", ParamLayer.ALGO_CONFIG, "int", 8, 64, step=4)
    N_EPOCHS = TunableParamSpec("n_epochs", ParamLayer.ALGO_CONFIG, "int", 2, 10)
    CLIP_RANGE = TunableParamSpec("clip_range", ParamLayer.ALGO_CONFIG, "float", 0.1, 0.3)
    ENT_COEF = TunableParamSpec("ent_coef", ParamLayer.ALGO_CONFIG, "float", 0.0, 5e-2)
    GAE_LAMBDA = TunableParamSpec("gae_lambda", ParamLayer.ALGO_CONFIG, "float", 0.8, 1.0)
    GAMMA = TunableParamSpec("gamma", ParamLayer.SKELETON, "float", 0.95, 0.999)


class A2cTunableParam(Enum):
    LR = TunableParamSpec("lr", ParamLayer.ALGO_CONFIG, "float", 1e-5, 1e-2, log=True)
    BATCH_SIZE = TunableParamSpec("batch_size", ParamLayer.ALGO_CONFIG, "int", 8, 32, step=4)
    ENT_COEF = TunableParamSpec("ent_coef", ParamLayer.ALGO_CONFIG, "float", 0.0, 5e-2)
    GAE_LAMBDA = TunableParamSpec("gae_lambda", ParamLayer.ALGO_CONFIG, "float", 0.8, 1.0)
    GAMMA = TunableParamSpec("gamma", ParamLayer.SKELETON, "float", 0.95, 0.999)


class DqnTunableParam(Enum):
    LR = TunableParamSpec("lr", ParamLayer.ALGO_CONFIG, "float", 1e-6, 1e-2, log=True)
    BATCH_SIZE = TunableParamSpec("batch_size", ParamLayer.ALGO_CONFIG, "int", 8, 64, step=4)
    LEARN_TIMES = TunableParamSpec("learn_times_per_episode", ParamLayer.ALGO_CONFIG, "int", 1, 8)
    TAU = TunableParamSpec("tau", ParamLayer.ALGO_CONFIG, "float", 0.1, 1.0)
    GAMMA = TunableParamSpec("gamma", ParamLayer.SKELETON, "float", 0.95, 0.999)


# 算法 → 可调超参枚举注册表（框架单一事实来源）。
# AutoML 按 LearnerConfig.algo_type 查表生成 Optuna 搜索空间；**不要求** Learner 实现 tunable_param_enum()。
ALGO_TUNABLE_PARAM: Dict[str, type[Enum]] = {
    "coma": ComaTunableParam,
    "ia2c": EpymarlActorCriticTunableParam,
    "ippo": MappoTunableParam,
    "iql": QmixTunableParam,
    "maa2c": EpymarlActorCriticTunableParam,
    "maddpg": MaddpgTunableParam,
    "qmix": QmixTunableParam,
    "mappo": MappoTunableParam,
    "pac": PacTunableParam,
    "pac_dcg": PacTunableParam,
    "vdn": QmixTunableParam,
    "ppo": PpoTunableParam,
    "a2c": A2cTunableParam,
    "dqn": DqnTunableParam,
}


def resolve_tunable_param_enum(algo_type: str) -> type[Enum]:
    """按 algo_type 返回框架侧 TunableParam 枚举；Learner 无需实现任何类方法。"""
    enum_cls = ALGO_TUNABLE_PARAM.get(algo_type)
    if enum_cls is None:
        raise KeyError(
            f"No TunableParam enum registered for algo_type={algo_type!r}, "
            f"available={sorted(ALGO_TUNABLE_PARAM)}"
        )
    return enum_cls


def tunable_param_keys(algo_type: str) -> tuple[str, ...]:
    """include/exclude 可写的 TunableParamSpec.key；与 ALGO_TUNABLE_PARAM[algo_type] 枚举成员一一对应。"""
    return tuple(member.value.key for member in resolve_tunable_param_enum(algo_type))


# 各算法推荐写入 update() dict 的 canonical 键名；algos_metrics.csv 存全量，不按此表过滤。
AUTOML_LEARN_METRIC_KEYS: Dict[str, tuple[str, ...]] = {
    "coma": (
        "policy_loss", "critic_loss", "entropy", "agent_grad_norm",
        "critic_grad_norm", "counterfactual_advantage",
    ),
    "ia2c": (
        "critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean",
        "target_mean", "advantage_mean", "pg_loss", "agent_grad_norm", "pi_max",
    ),
    "ippo": (
        "critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean",
        "target_mean", "advantage_mean", "pg_loss", "agent_grad_norm", "pi_max",
    ),
    "iql": ("loss", "grad_norm", "td_error_abs", "q_taken_mean", "target_mean"),
    "maa2c": (
        "critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean",
        "target_mean", "advantage_mean", "pg_loss", "agent_grad_norm", "pi_max",
    ),
    "maddpg": (
        "critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean",
        "target_mean", "pg_loss", "agent_grad_norm",
    ),
    "qmix": ("loss", "grad_norm", "td_error_abs", "q_taken_mean", "target_mean"),
    "mappo": (
        "critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean",
        "target_mean", "advantage_mean", "pg_loss", "agent_grad_norm", "pi_max",
    ),
    "pac": (
        "critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean",
        "target_mean", "advantage_mean", "pg_loss", "agent_grad_norm", "pi_max",
        "entropy_coef",
    ),
    "pac_dcg": (
        "critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean",
        "target_mean", "advantage_mean", "pg_loss", "agent_grad_norm", "pi_max",
        "entropy_coef",
    ),
    "vdn": ("loss", "grad_norm", "td_error_abs", "q_taken_mean", "target_mean"),
    "ppo": (
        "pg_loss", "value_loss", "entropy_loss", "approx_kl",
        "clip_fraction", "explained_variance", "update_cnt",
    ),
    "a2c": ("pg_loss", "value_loss", "explained_variance", "update_cnt"),
    "dqn": (
        "loss", "update_cnt", "inserted", "duplicate_skipped",
        "replay_size", "seen_transitions",
    ),
}

# 永远忽略的框架白名单键（不进 learn_results 归档列 / learn_metrics）。
FRAMEWORK_UPDATE_KEYS = frozenset({"skip_update", "skip_reason"})

# SB3 logger.name_to_value 的原始键带 "train/" 前缀，与 csv 列名/Optuna auxiliary 配置不兼容；
# 这里把常见键映射为 canonical 名（与 EPyMARL snake_case 对齐），未列出的键把 "/" 替换为 "_"。
SB3_LEARN_METRIC_KEY_MAP: Dict[str, str] = {
    "train/policy_loss": "pg_loss",
    "train/value_loss": "value_loss",
    "train/entropy_loss": "entropy_loss",
    "train/approx_kl": "approx_kl",
    "train/clip_fraction": "clip_fraction",
    "train/loss": "loss",
    "train/explained_variance": "explained_variance",
}


def normalize_sb3_learn_metrics(raw: Dict[str, Any]) -> Dict[str, Any]:
    """把 SB3 logger 的带前缀键规范化为 canonical 键；非 SB3 来源不应调用。"""
    normalized: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in SB3_LEARN_METRIC_KEY_MAP:
            normalized[SB3_LEARN_METRIC_KEY_MAP[key]] = value
        elif "/" in key:
            normalized[key.replace("/", "_")] = value
        else:
            normalized[key] = value
    return normalized


# =============================================================================
# AutoML 入参 / 出参 dataclass
# =============================================================================

@dataclass
class TrainingSnapshot:
    """TrainTaskManager 收集后传给 AutoML.advice() 的训练快照。"""
    task_id: str
    stage_name: str
    generation: int
    completed_episodes: int
    max_episodes: int
    # per-agent 奖励序列；已过滤 None/NaN/Inf。
    reward_curve: Dict[str, List[float]]
    # {learner_name: {metric_key: series}}；metric_key 已为 canonical 键。
    learn_metrics: Dict[str, Dict[str, List[float]]]
    # {learner_name: algo_config.to_dict()}；当前生效算法配置。
    current_algo_configs: Dict[str, Dict]
    # {learner_name: [TunableParam.key, ...]}；由 ALGO_TUNABLE_PARAM[algo_type] + TOML search_space 收窄得出。
    tunable_catalog: Dict[str, List[str]]
    # episode级业务指标，如 stage_success=0/1；与奖励分开用于业务毕业判断。
    episode_metrics: Dict[str, List[Optional[float]]] = field(default_factory=dict)
    # 本轮已知产物路径（可选，monitor 用）。
    artifact_paths: Dict[str, List[str]] = field(default_factory=dict)
    # 当前 generation 由 AutoML 采样并应用的环境 metadata 增量覆盖。
    current_env_metadata_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutoMLAdvice:
    """AutoML.advice() 返回的决策建议。"""
    decision: str  # "continue" | "restart" | "complete"（complete 仅用于结束归档）
    algo_config_overrides: Dict[str, Dict] = field(default_factory=dict)  # {learner_name: {field: value}}
    skeleton_overrides: Dict[str, Dict] = field(default_factory=dict)
    env_metadata_overrides: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    reason: str = ""


class TrainingResourceEvent(Enum):
    """TrainTaskManager 训练资源生命周期事件（归档 / 重启）。"""
    # restart 建议已下达：请贡献方可保存模型等大产物（锁外执行）。
    PRE_ARCHIVE_REQUEST = "pre_archive_request"
    # 归档前资源已齐（含 force checkpoint）；监听方可开始读路径 / 触发 archive。
    ARCHIVE_RESOURCES_READY = "archive_resources_ready"
    # generation 已切换且 Role restart 回调已跑完；供后续扩展。
    POST_RESTART = "post_restart"


@dataclass
class TrainingResourceContext:
    """传给 training_resource_callback 的上下文。"""
    event: TrainingResourceEvent
    task_id: str
    stage_name: str
    generation: int
    completed_episodes: int
    advice: Optional[AutoMLAdvice] = None
    reason: str = ""


@dataclass
class TrainingResourceContribution:
    """资源贡献方在 PRE_ARCHIVE_REQUEST 时返回的路径（可选）。"""
    model_paths: List[str] = field(default_factory=list)
    artifact_paths: Dict[str, List[str]] = field(default_factory=dict)


# 返回 None 表示本回调不贡献路径；ARCHIVE_RESOURCES_READY / POST_RESTART 通常只监听。
TrainingResourceCallback = Callable[
    [TrainingResourceContext],
    Optional[TrainingResourceContribution],
]


@dataclass
class TrainingRunArchive:
    """restart 归档包；TM 持锁组装后交 AutoML.archive_train 落盘。"""
    env_name: str
    task_id: str
    stage_name: str
    generation: int
    archive_seq: int
    completed_episodes: int
    # algos_metrics.csv 行数据（TM 预组装）。
    episode_rows: List[Dict]
    # {category: [path]}，category 如 reward_curve / mid_eval_gifs / models。
    artifact_paths: Dict[str, List[str]]
    snapshot: TrainingSnapshot
    advice: AutoMLAdvice
    # 每个 metric_output 独立的逐局行数据。
    metric_output_rows: Dict[str, List[Dict]] = field(default_factory=dict)
    # 本归档代次实际使用的 task 级 seed（与 EpisodeInfo/LearnInfo 派生链一致）。
    run_seed: Optional[int] = None
    # 创建本阶段首轮模型快照时使用的 seed；generation>0 时通常与 run_seed 不同。
    initialization_seed: Optional[int] = None
    # trial/study 摘要，写入 advice.yaml 的 optuna 段。
    optuna: Dict = field(default_factory=dict)


@dataclass
class AgentCompose:
    """
    Agent采样编排定义。

    order：没有Episode Agent，使用环境模型的透明默认顺序。
    decision：由Episode Agent在局开始时输出完整动作，再交给环境解释顺序。

    Parallel/AEC是环境的执行模型，不再属于AgentCompose的编排类型。
    """
    # agent编排方式。
    compose_type: Literal['order', 'decision'] = 'order'
    # 在一局开始前执行的agent；可为空，表示没有episode级流程决策。
    episode_agent: Optional[AgentType] = None
    # 在一步内执行的agent列表。
    step_agent: List[AgentType] = field(default_factory=list)
    def __post_init__(self):
        from common.exception import AgentComposeException
        if self.compose_type not in {"order", "decision"}:
            raise AgentComposeException(
                compose_type=self.compose_type,
                detail="Parallel/AEC is selected by the environment base class.",
            )
        if not self.step_agent:
            raise AgentComposeException(
                compose_type=self.compose_type,
                detail="At least one Step Agent is required to interact with the environment.",
            )
        if self.compose_type == "order" and self.episode_agent is not None:
            raise AgentComposeException(
                compose_type=self.compose_type,
                detail="Order compose uses the transparent environment order and no Episode Agent.",
            )
        if self.compose_type == "decision" and self.episode_agent is None:
            raise AgentComposeException(
                compose_type=self.compose_type,
                detail="Decision compose requires an Episode Agent.",
            )

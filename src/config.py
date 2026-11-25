from __future__ import annotations

"""
config.py 只定义配置的项目的结构和默认值，不对数据有效性做判断，没有业务逻辑和动态变更，这些操作方模块中做。
"""
import os
import torch
from enum import Enum
from dataclasses_json import DataClassJsonMixin, Undefined
from dataclasses import dataclass, field
# from typing import Any, Optional, LiteralString, Literal, TYPE_CHECKING, Union
from typing import Any, ClassVar, Optional, Literal, TYPE_CHECKING, Union
from typing_extensions import LiteralString
from pathlib import Path
from functools import wraps
from sample.model import SamplingStrategy
from scheduler.model import (
    AutoMLEnvMetadataSearchParamConfig,
    AutoMLBusinessMetricConfig,
    AutoMLLearnerSearchParamConfig,
    AutoMLMetricOutputConfig,
    AutoMLPrimaryMetric,
    AutoMLPrunerType,
    AutoMLSamplerType,
    AutoMLSeedSearchConfig,
)
from envs.action.model import ActionDefined

if TYPE_CHECKING:
    from common.types import DefinedActionList, AgentName
else:
    DefinedActionList = list
    AgentName = str

# 运行期配置、日志和资源目录以进程启动时的 working directory 为准。
# 不使用源码文件位置，避免 src 布局把运行产物错误写入 src/。
root_dir_position = str(Path.cwd().resolve())


class ConfigEnum(str, Enum):
    """配置枚举基类，保持枚举值和普通字符串兼容。"""

    def __str__(self) -> str:
        return self.value


class AlgorithmType(ConfigEnum):
    """Learner算法类型。"""
    # 反事实多智能体 Actor-Critic。
    COMA = "coma"
    # 历史独立 A2C 配置标识。
    IA2C = "ia2c"
    # 历史独立 PPO 配置标识。
    IPPO = "ippo"
    # 历史独立 Q-learning 配置标识。
    IQL = "iql"
    # 历史多智能体 A2C 配置标识。
    MAA2C = "maa2c"
    # 基于集中式 Critic 的多智能体连续动作算法。
    MADDPG = "maddpg"
    # 用单调混合网络分解团队动作价值的多智能体算法。
    QMIX = "qmix"
    # 基于 PPO 更新的多智能体 Actor-Critic。
    MAPPO = "mappo"
    # 历史 PAC 配置标识。
    PAC = "pac"
    # 历史 PAC 协调图变体配置标识。
    PAC_DCG = "pac_dcg"
    # 历史加和式价值分解配置标识。
    VDN = "vdn"
    # SB3 单智能体 PPO。
    PPO = "ppo"
    # SB3 单智能体 A2C。
    A2C = "a2c"
    # SB3 单智能体 DQN。
    DQN = "dqn"


class AgentType(ConfigEnum):
    """Agent网络类型。"""
    # 共享参数的循环 Agent。
    RNN = "rnn"
    # 每个槽位使用独立参数的循环 Agent。
    RNN_NS = "rnn_ns"
    # 从循环策略分布选动作的 Agent。
    POLICY_RNN = "policy_rnn"
    # 按合法动作掩码选择动作的独立循环 Q 值实现。
    Q_RNN = "q_rnn"
    # 每个槽位拥有独立高斯 Actor 的连续动作 Agent。
    MADDPG_MLP = "maddpg_mlp"
    MAPPO_RNN = "mappo_rnn"  # Legacy policy alias.
    # 蒙特卡洛采样使用的随机动作 Agent。
    MONTE_CARLO = "monte_carlo"
    # SB3 的策略执行适配，供 PPO、A2C 和 DQN Learner 使用。
    SB3 = "sb3"


class AgentOutputType(ConfigEnum):
    """Agent网络输出语义。"""
    # 按候选动作输出 Q 值。
    Q = "q"
    # 输出构造策略分布所需的参数。
    PI_LOGITS = "pi_logits"
    # 连续 Actor 的输出类型，用于配对 Agent 和 Learner。
    CONTINUOUS_ACTION = "continuous_action"


class StorageType(ConfigEnum):
    """后端存储类型。"""
    # 本地文件系统存储。
    LOCAL = "local"
    # 兼容 S3 协议的对象存储。
    S3 = "s3"


class SampleType(ConfigEnum):
    """训练主路的样本驱动方式。"""

    # 与环境交互并生成训练样本。
    ONLINE = "online"
    # 读取已录制的轨迹作为训练样本。
    OFFLINE = "offline"


class ResourceDir(ConfigEnum):
    """框架本地资源目录。"""
    # 模型权重的默认输出目录。
    MODELS = os.path.join(root_dir_position, "artifacts", "models")
    # 训练期间由 ModelManager 高频写入的候选快照；与可交付的 final/optimo 模型分开。
    MODEL_POOL = os.path.join(MODELS, "_snapshot")
    # 采样数据和回放快照的默认目录。
    SAMPLES = os.path.join(root_dir_position, "artifacts", "samples")
    # 训练和评估结果的默认目录。
    RESULTS = os.path.join(root_dir_position, "artifacts", "results")
    # 录屏和动画的默认目录。
    RECORDINGS = os.path.join(root_dir_position, "artifacts", "recordings")
    # AutoML 工作文件的默认目录。
    AUTOML = os.path.join(root_dir_position, "artifacts", "automl")
    # 任务归档的默认目录。
    ARCHIVES = os.path.join(root_dir_position, "artifacts", "archives")


@dataclass
class EntryConfig:
    """入口元数据：角色进程名、评估实现与评估配置构建器。"""

    # 当前角色进程的标识，空值时由入口生成。
    role_id: str = ""
    # 单进程评估角色类的导入路径。
    evaluation_class: str = ""
    # 分布式评估控制角色类的导入路径。
    distributed_evaluation_class: str = ""
    # 从训练配置构造 EvalConfig 的函数导入路径。
    eval_config_builder: str = ""


@dataclass
class GrpcConfig:
    """训练角色之间的 gRPC 地址；串行角色可以忽略。"""

    # 任务总线 gRPC 服务的主机和端口。
    task_manager_address: str = ""
    # 回放池 gRPC 服务的主机和端口。
    replay_buffer_address: str = ""
    # 模型管理 gRPC 服务的主机和端口。
    model_manager_address: str = ""
    # 仿真管理 gRPC 服务的主机和端口。
    sim_manager_address: str = ""
    # 蒙特卡洛样本服务的 gRPC 地址。
    monte_carlo_method_address: str = ""
    # 训练产物管理 gRPC 服务的主机和端口。
    artifacts_manager_address: str = "127.0.0.1:53156"
    # 单次RPC业务调用deadline；小于等于0表示不设deadline。
    call_timeout_seconds: float = 60.0


@dataclass
class TorchRpcConfig:
    """静态 TensorPipe RPC world 配置；rank 由启动器通过 RANK 环境变量注入。"""

    # TensorPipe RPC world 中的进程总数。
    world_size: int = 0
    # TensorPipe RPC 进程建立连接时使用的初始化地址。
    init_address: str = ""
    # 单次 TensorPipe RPC 调用的超时秒数。
    call_timeout_seconds: float = 60.0
    # 每个 TensorPipe RPC 进程的处理线程数。
    num_worker_threads: int = 16
    # ReplayBuffer 整局大载荷进入 TensorPipe 之前的中心准入上限；不是服务端 handler 线程数。
    max_concurrent_replay_pushes: int = 1
    # 未获得准入时由 sample 本地短暂等待后重新发起轻量 RPC，不占用 Rank 0 worker thread。
    replay_push_permit_poll_interval_seconds: float = 0.1
    # sample 在编码/发送前崩溃时，Rank 0 自动回收准入令牌的最长时间。
    replay_push_permit_lease_timeout_seconds: float = 120.0
    # 单次 TensorPipe replay push 的步窗口字节上限；0 表示关闭切分，整包发送。
    replay_push_chunk_bytes: int = 31457280


class LogHandlerKind(ConfigEnum):
    """Logger 可启用的输出 Handler。"""
    # 在控制台输出日志。
    CONSOLE = "console"
    # 将日志写入文件。
    FILE = "file"


@dataclass
class LocalStorageConfig(DataClassJsonMixin):
    # 本地数据根目录；ReplayBuffer 阶段快照默认写这里。
    data_path: str = ResourceDir.SAMPLES.value
    # mmap 等工作目录相对 data_path 的子目录。
    scratch_subdir: str = "replay_scratch"


class StorageBucket(ConfigEnum):
    """S3协议存储桶名称。"""
    # 蒙特卡洛采样数据的存储桶。
    MONTE_CARLO_SAMPLES = "monte-carlo-samples"
    # 评估样本的存储桶。
    EVAL_SAMPLES = "sample-samples"
    # 模型快照的存储桶。
    MODELS_POOL = "models-pool"
    # AutoML 训练归档的存储桶。
    AUTOML = "automl-archives"
    # 回放池快照的存储桶。
    REPLAY_BUFFER = "replay-buffer"


@dataclass
class S3StorageConfig(DataClassJsonMixin):
    endpoint_url: str = "http://localhost:9000"  # MinIO 服务地址
    access_key: str = "minioadmin"  # MinIO Access Key
    secret_key: str = "minioadmin"  # MinIO Secret Key
    bucket_name: str = StorageBucket.MONTE_CARLO_SAMPLES.value  # 桶名称


@dataclass
class AgentAlgoConfig(DataClassJsonMixin):
    """
    Agent网络/控制器算法特有配置基类。

    这里保留 Agent/controller 需要的网络结构相关字段；
    AgentConfig在__post_init__阶段按agent_type索引到具体子类并赋给algo_config，
    原生循环网络和 SB3 Agent 直接读取标准配置。
    """
    # agent是否为神经网络agent
    use_net: bool = True
    # EPyMARL agent网络注册名。
    agent: str = AgentType.RNN.value
    # EPyMARL controller(mac)注册名。
    mac: str = "basic_mac"
    # agent网络是否使用RNN。
    use_rnn: bool = True
    # 参数共享范围：all=全共享，none=每个槽位独立。
    sharing_scope: str = "all"
    # 策略softmax前是否屏蔽不可用动作。
    mask_before_softmax: bool = True
    # EPyMARL动作选择器；项目侧Agent自行选动作，通常为None。
    action_selector: Optional[str] = None


@dataclass
class RnnAgentConfig(AgentAlgoConfig):
    """共享参数RNN agent网络配置。"""
    # 执行网络或策略在配置中的 Agent 标识。
    agent: str = AgentType.RNN.value
    # 共享循环网络使用的控制器名称。
    mac: str = "basic_mac"
    # 是否在执行网络中使用循环层。
    use_rnn: bool = True
    # 所有 Agent 槽位共用一套网络参数。
    sharing_scope: str = "all"


@dataclass
class PolicyRnnAgentConfig(AgentAlgoConfig):
    """共享循环策略：输出动作分布，不由 Learner 切换为 Q 值。"""
    # 执行网络或策略在配置中的 Agent 标识。
    agent: str = AgentType.POLICY_RNN.value
    # 控制器名称，循环策略 Agent 直接执行策略网络时使用空值。
    mac: str = ""
    # 是否在执行网络中使用循环层。
    use_rnn: bool = True
    # 所有 Agent 槽位共用一套策略网络参数。
    sharing_scope: str = "all"


# Compatibility for SDK clients that imported the original config name.
MappoRnnAgentConfig = PolicyRnnAgentConfig


@dataclass
class QAgentConfig(AgentAlgoConfig):
    """独立循环 Q 值执行策略；不绑定 EPyMARL Controller。"""
    # 执行网络或策略在配置中的 Agent 标识。
    agent: str = AgentType.Q_RNN.value
    # 控制器名称，Q Agent 直接执行 Q 网络时使用空值。
    mac: str = ""
    # 是否在执行网络中使用循环层。
    use_rnn: bool = True
    # 所有 Agent 槽位共用一套 Q 网络参数。
    sharing_scope: str = "all"


@dataclass
class MaddpgAgentConfig(AgentAlgoConfig):
    """MADDPG 执行端的独立确定性 Actor，rollout 通过高斯噪声探索。"""
    # MADDPG Actor 对应的 Agent 标识。
    agent: str = AgentType.MADDPG_MLP.value
    # 控制器名称，MADDPG 前馈 Actor 使用空值。
    mac: str = ""
    # 继承的循环网络配置字段，当前 MADDPG Actor 固定为前馈网络。
    use_rnn: bool = False
    # 每个 Agent 槽位使用独立的 Actor 参数。
    sharing_scope: str = "none"
    # tanh 之前的 rollout 高斯探索标准差；独立于 checkpoint 中的兼容 scale head，
    # 且不参与确定性 policy-gradient 目标。
    exploration_std: float = 0.2
    # 配置后，rollout 探索噪声按训练任务下发的 epsilon 在
    # exploration_std_final（epsilon=0）和 exploration_std（epsilon=1）之间线性退火。
    # None 保持历史的固定标准差语义。
    exploration_std_final: Optional[float] = None


@dataclass
class RnnNsAgentConfig(AgentAlgoConfig):
    """非共享参数RNN agent网络配置。"""
    # 执行网络或策略在配置中的 Agent 标识。
    agent: str = AgentType.RNN_NS.value
    # 非共享循环网络使用的控制器名称。
    mac: str = "non_shared_mac"
    # 是否在执行网络中使用循环层。
    use_rnn: bool = True
    # 每个 Agent 槽位使用独立的网络参数。
    sharing_scope: str = "none"


@dataclass
class MonteCarloAgentConfig(AgentAlgoConfig):
    """蒙特卡洛/随机策略agent配置；不构建EPyMARL网络。"""
    # 是否初始化执行网络，蒙特卡洛 Agent 直接采样动作。
    use_net: bool = False
    # 执行网络或策略在配置中的 Agent 标识。
    agent: str = AgentType.MONTE_CARLO.value
    # 控制器名称，直接采样动作时使用空值。
    mac: str = ""
    # 继承的循环网络开关，直接采样动作时默认关闭。
    use_rnn: bool = False
    # 继承的参数共享选项，蒙特卡洛 Agent 使用 none。
    sharing_scope: str = "none"


@dataclass
class SB3AgentConfig(AgentAlgoConfig):
    """
    单智能体 SB3 ActorCriticPolicy 网络配置。

    PPO/A2C 共用 stable_baselines3 的 ActorCriticPolicy(MLP)，不构建 EPyMARL controller/网络，
    因此 mac 置空、不使用 RNN，也不做参数共享或 agent_id 拼接(单智能体下无意义)。
    """
    # 执行网络或策略在配置中的 Agent 标识。
    agent: str = AgentType.SB3.value
    # 控制器名称，SB3 Agent 直接调用策略时使用空值。
    mac: str = ""
    # 继承的循环网络开关，SB3 前馈策略默认关闭。
    use_rnn: bool = False
    # SB3 单智能体策略使用 none。
    sharing_scope: str = "none"


AGENT_ALGO_CONFIG: dict[AgentType, type[AgentAlgoConfig]] = {
    AgentType.RNN: RnnAgentConfig,
    AgentType.POLICY_RNN: PolicyRnnAgentConfig,
    AgentType.Q_RNN: QAgentConfig,
    AgentType.MADDPG_MLP: MaddpgAgentConfig,
    AgentType.MAPPO_RNN: PolicyRnnAgentConfig,
    AgentType.RNN_NS: RnnNsAgentConfig,
    AgentType.MONTE_CARLO: MonteCarloAgentConfig,
    AgentType.SB3: SB3AgentConfig,
}

# 各agent类型对obs_agent_id(是否拼接agent id one-hot)的推荐默认值。
AGENT_DEFAULT_OBS_AGENT_ID: dict[AgentType, bool] = {
    AgentType.RNN: True,
    AgentType.POLICY_RNN: True,
    AgentType.Q_RNN: True,
    AgentType.MADDPG_MLP: False,
    AgentType.MAPPO_RNN: True,
    AgentType.RNN_NS: False,
    AgentType.MONTE_CARLO: False,
    AgentType.SB3: False,
}


@dataclass
class AgentConfig(DataClassJsonMixin):
    """
    Agent骨架配置：只保留Agent基类和执行期需要的业务/维度字段；
    网络结构相关的算法特有配置统一收敛到algo_config。
    """
    # Agent配置名称，用于和LearnerConfig一一对应，也表达业务角色。
    name: AgentName = "default"
    # 默认使用CPU
    device: Optional[Union[torch.device, str]] = "cpu"
    # agent网络类型；Learner算法类型由LearnerConfig.algo_type控制。
    agent_type: str = AgentType.RNN.value
    # 旧版单头离散动作列表；新业务配置使用 action_defined。
    define_action: DefinedActionList = field(default_factory=list)
    # 一个智能体对应一个统一业务动作定义，可表达四种动作空间。
    action_defined: Optional[ActionDefined] = None
    # 旧版底层动作空间规格；新业务配置应使用 action_defined。
    action_space_spec: Optional[dict] = None
    # 多动作展开模式：single保持传统动作表；combination组合多个动作；free由调用方填表。
    multi_action_mode: str = "single"
    # agent实体名称列表；旧版单组入口，与agent_groups互斥。
    agent_names: list[str] = field(default_factory=list)
    # agent实体无名二维分组入口；外层为组声明顺序，内层为组内固定网络槽位顺序。
    agent_groups: list[list[str]] = field(default_factory=list)
    # agent实体动作可选目标名称列表
    agent_target_names: list[str] = field(default_factory=list)

    # 单agent私有观测维度。
    obs_dim: int = field(default_factory=int)
    # 网络隐藏层维度。
    hidden_dim: int = 64
    # 循环网络层数。
    num_layers: int = 1
    # 是否把上一动作拼接到观测。
    obs_last_action: bool = True
    # 是否把agent id拼接到观测；默认值由algo_config按agent_type回填。
    obs_agent_id: bool = True
    # agent网络输出语义；和配对的LearnerConfig.agent_output_type保持一致。
    agent_output_type: str = AgentOutputType.Q.value
    # 固定策略 Agent 的网络权重文件。未配置 Learner 的网络型 Agent 必须提供该文件。
    model_file_path: Optional[str] = None
    # 是否作为固定策略运行；启用后采样令牌中该 Agent 的 epsilon=0、deterministic=True，
    # 且局末不把该 Agent 的轨迹推入 ReplayBuffer。
    fixed_mode: bool = False

    # 运行期智能体数量。
    n_agents: int = 0
    # 运行期动作数量。
    n_actions: int = 0
    # 运行期全局状态维度。
    state_shape: Any = 1

    # 算法特有网络配置；None时由__post_init__按agent_type索引填充。
    algo_config: Optional[AgentAlgoConfig] = None

    class Meta:
        undefined = Undefined.EXCLUDE

    def __post_init__(self):
        try:
            agent_type = AgentType(self.agent_type)
        except ValueError:
            if self.algo_config is None:
                raise ValueError(
                    f"Custom agent_type={self.agent_type!r} requires an AgentAlgoConfig "
                    "provided by the active Agent plugin declaration"
                ) from None
            self.agent_type = str(self.agent_type)
            self.agent_output_type = AgentOutputType(self.agent_output_type).value
            return
        self.agent_type = agent_type.value
        if self.algo_config is None:
            self.algo_config = AGENT_ALGO_CONFIG[agent_type]()
        self.obs_agent_id = AGENT_DEFAULT_OBS_AGENT_ID[agent_type]
        fixed_output = {
            AgentType.POLICY_RNN: AgentOutputType.PI_LOGITS,
            AgentType.MAPPO_RNN: AgentOutputType.PI_LOGITS,
            AgentType.Q_RNN: AgentOutputType.Q,
            AgentType.MADDPG_MLP: AgentOutputType.CONTINUOUS_ACTION,
        }
        self.agent_output_type = AgentOutputType(fixed_output.get(agent_type, self.agent_output_type)).value

    def bind_runtime(self, n_agents: int, n_actions: int, state_shape: Any = 1, device=None):
        """把运行期维度写回配置本身，供原生网络使用。"""
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_shape = state_shape
        if device is not None:
            self.device = device
        return self


@dataclass
class LearnerAlgoConfig(DataClassJsonMixin):
    """Learner算法配置基类，约束所有算法必须声明通用学习超参。"""
    # 配对 Agent 的默认插件标识，供训练 TOML 装配时使用。
    default_agent_type: ClassVar[str | None] = None

    # 学习率；具体算法必须给出自己的默认值或由入口显式传入。
    lr: float
    # 一次Learner.update采样的episode数量。
    batch_size: int
    # 每个学习令牌触发的 Learner.update 次数；令牌由一次 ReplayBuffer 样本更新发放。
    learn_times_per_episode: int
    # 配对Agent的网络输出语义，例如q或pi_logits。
    agent_output_type: AgentOutputType
    # 此 Learner 的 ReplayBuffer 分区最多保留的 active episode 数。
    capacity: int = field(default=1000, kw_only=True)
    # active 回放池达到此容量单位数后才允许学习；None 表示不额外等待。
    min_replay_size: Optional[int] = field(default=None, kw_only=True)
    # ReplayBuffer 抽样策略，算法可覆盖默认值。
    sample_strategy: str = field(default=SamplingStrategy.RECENT_BIASED.value, kw_only=True)

    def __post_init__(self):
        if self.min_replay_size is not None and (
            isinstance(self.min_replay_size, bool) or not isinstance(self.min_replay_size, int)
            or self.min_replay_size < 0 or self.min_replay_size > self.capacity
        ):
            raise ValueError("min_replay_size must be an integer between 0 and capacity")


@dataclass
class QmixConfig(LearnerAlgoConfig):
    """原生 QMIX 学习器算法配置。"""
    # 同名 Agent 省略类型时使用的默认插件标识。
    default_agent_type: ClassVar[str] = AgentType.RNN.value

    # EPyMARL learner 注册标识。
    learner: str = "q_learner"
    # 价值混合网络类型。
    mixer: Optional[str] = "qmix"
    # 学习率。
    lr: float = 1e-3
    # 目标网络更新参数；>1 按硬更新间隔，<=1 按软更新 tau。
    target_update_interval_or_tau: float = 200
    # 网络隐藏层维度；运行期 bind_runtime 会同步为 Agent 的 hidden_dim。
    hidden_dim: int = 64
    # Agent 网络输出语义。
    agent_output_type: AgentOutputType = AgentOutputType.Q
    # Q-learning是否使用double q。
    double_q: bool = True
    # 保留旧配置字段；原生 QMIX 使用一步 TD。
    td_lambda: float = 0.6
    # QMIX mixer内部嵌入维度，对应EPyMARL mixing_embed_dim。
    mixing_embed_dim: int = 32
    # QMIX超网络层数，对应EPyMARL hypernet_layers。
    hypernet_layers: int = 2
    # QMIX超网络隐藏层维度，对应EPyMARL hypernet_embed。
    hypernet_embed: int = 64
    # 采样策略
    sample_strategy: str = SamplingStrategy.RECENT_BIASED.value
    # 一次学习样本大小
    batch_size: int = 16
    # 样本更新后学习次数
    learn_times_per_episode: int = 4


@dataclass
class MappoConfig(LearnerAlgoConfig):
    """Recurrent MAPPO with typed distributions and a centralized value critic."""
    # 同名 Agent 省略类型时使用的默认插件标识。
    default_agent_type: ClassVar[str] = AgentType.POLICY_RNN.value

    # GAE 优势估计的衰减系数。
    gae_lambda: float = 0.95
    # 每轮优化将轨迹批次拆分成的 minibatch 数量。
    num_mini_batch: int = 1
    # 训练损失中 Critic 价值误差的权重。
    value_loss_coef: float = 1.0
    # Huber 损失从平方误差切换为线性误差的阈值。
    huber_delta: float = 10.0
    # 是否限制价值预测相对于旧预测的变化幅度。
    use_clipped_value_loss: bool = True
    # actor-critic 无价值混合网络。
    mixer: Optional[str] = None
    # 学习率。
    lr: float = 1e-4
    # 分别覆盖 Actor/Critic 学习率；None 沿用 lr，保持旧配置行为。
    actor_lr: Optional[float] = None
    # Critic 学习率覆盖值，None 时沿用 lr。
    critic_lr: Optional[float] = None
    # Critic MLP 隐藏层维度；运行期 bind_runtime 会同步为 Agent 的 hidden_dim。
    hidden_dim: int = 128
    # Agent 网络输出语义。
    agent_output_type: AgentOutputType = AgentOutputType.PI_LOGITS
    # critic类型：cv_critic=集中式V。
    critic_type: str = "cv_critic"
    # PPO裁剪范围。
    eps_clip: float = 0.1
    # 每批on-policy样本重复优化轮数。
    epochs: int = 4
    # 分别覆盖优化轮数；None 沿用 epochs。轮数必须为正整数。
    actor_epochs: Optional[int] = None
    # 每批样本的 Critic 优化轮数，None 时沿用 epochs。
    critic_epochs: Optional[int] = None
    # 前 N 次成功 update 只训练 Critic；Actor 权重和优化器状态保持不变。
    critic_warmup_updates: int = 0
    # 仅在 Critic 预热期间使用完整折扣回报；时间截断/窗口尾仍 bootstrap。
    warmup_use_mc_returns: bool = False
    # 策略熵奖励系数。
    entropy_coef: float = 0.005
    # critic是否额外拼接所有agent的局部观测；按配置选择。
    obs_individual_obs: bool = True
    # critic是否拼接上一时刻联合动作one-hot；按配置选择。
    obs_last_action: bool = False
    # critic是否拼接agent id。
    obs_agent_id: bool = True
    # 策略softmax前是否屏蔽不可用动作。
    mask_before_softmax: bool = True
    # 采样策略
    sample_strategy: str = SamplingStrategy.ON_POLICY_RECENT.value
    # 一次学习样本大小
    batch_size: int = 64
    # 样本更新后学习次数
    learn_times_per_episode: int = 1


@dataclass
class IqlConfig(LearnerAlgoConfig):
    """IQL 学习器算法配置（EPyMARL q_learner，无 mixer）。"""
    # 历史 IQL 配置中的 Q Learner 标识。
    learner: str = "q_learner"
    # 历史配置的价值混合网络标识，None 表示直接使用个体价值。
    mixer: Optional[str] = None
    # 优化器学习率。
    lr: float = 3e-4
    # 目标网络更新参数，大于 1 表示硬更新间隔，至多为 1 时表示软更新系数。
    target_update_interval_or_tau: float = 200
    # 算法网络的隐藏层宽度。
    hidden_dim: int = 128
    # 配对 Agent 按候选动作输出 Q 值。
    agent_output_type: AgentOutputType = AgentOutputType.Q
    # 是否用在线网络选动作、目标网络估值。
    double_q: bool = True
    # 历史配置中用于 TD(λ) 回报的衰减系数。
    td_lambda: float = 0.6
    # 从框架 ReplayBuffer 抽取训练样本的策略。
    sample_strategy: str = SamplingStrategy.RECENT_BIASED.value
    # 每次 Learner 更新从框架 ReplayBuffer 抽取的样本量。
    batch_size: int = 16
    # 每个学习令牌触发的 Learner 更新次数。
    learn_times_per_episode: int = 4


@dataclass
class VdnConfig(IqlConfig):
    """VDN 学习器算法配置（EPyMARL q_learner + vdn mixer）。"""
    # 历史 VDN 配置中将个体 Q 值相加的 Mixer 标识。
    mixer: Optional[str] = "vdn"


@dataclass
class EpymarlActorCriticConfig(LearnerAlgoConfig):
    """EPyMARL 离散动作 actor-critic 算法的公共配置。"""
    # 历史 Actor-Critic 配置的 Learner 标识。
    learner: str = "actor_critic_learner"
    # 历史配置的价值混合网络标识，None 表示直接使用个体价值。
    mixer: Optional[str] = None
    # 优化器学习率。
    lr: float = 5e-4
    # 目标网络更新参数，大于 1 表示硬更新间隔，至多为 1 时表示软更新系数。
    target_update_interval_or_tau: float = 0.01
    # 算法网络的隐藏层宽度。
    hidden_dim: int = 128
    # 配对 Agent 输出构造动作分布所需的参数。
    agent_output_type: AgentOutputType = AgentOutputType.PI_LOGITS
    # 历史配置中的集中式 Critic 标识。
    critic_type: str = "cv_critic"
    # 策略熵奖励的权重。
    entropy_coef: float = 0.001
    # 计算多步 TD 目标时使用的步数。
    q_nstep: int = 5
    # 是否给 Critic 拼接 Agent 的局部观测。
    obs_individual_obs: bool = False
    # 是否给 Critic 拼接上一步动作。
    obs_last_action: bool = False
    # 是否给 Critic 拼接 Agent 标识。
    obs_agent_id: bool = True
    # 是否在历史多步回报中加入末步的价值估计。
    add_value_last_step: bool = True
    # 是否在 softmax 前屏蔽不可用动作。
    mask_before_softmax: bool = True
    # 从框架 ReplayBuffer 抽取训练样本的策略。
    sample_strategy: str = SamplingStrategy.ON_POLICY_RECENT.value
    # 每次 Learner 更新从框架 ReplayBuffer 抽取的样本量。
    batch_size: int = 10
    # 每个学习令牌触发的 Learner 更新次数。
    learn_times_per_episode: int = 1


@dataclass
class Ia2cConfig(EpymarlActorCriticConfig):
    """IA2C 配置（独立局部 V critic）。"""
    # 历史 IA2C 配置的独立局部价值 Critic 标识。
    critic_type: str = "ac_critic"


@dataclass
class Maa2cConfig(EpymarlActorCriticConfig):
    """MAA2C 配置（集中式 V critic）。"""
    # 历史 MAA2C 配置的集中式价值 Critic 标识。
    critic_type: str = "cv_critic"
    # 策略熵奖励的权重。
    entropy_coef: float = 0.01


@dataclass
class IppoConfig(EpymarlActorCriticConfig):
    """IPPO 配置（独立局部 V critic + PPO clip objective）。"""
    # 历史 IPPO 配置的独立 PPO Learner 标识。
    learner: str = "ppo_learner"
    # 优化器学习率。
    lr: float = 3e-4
    # 历史 IPPO 配置的独立局部价值 Critic 标识。
    critic_type: str = "ac_critic"
    # PPO 策略概率比的裁剪幅度。
    eps_clip: float = 0.2
    # 同一批样本的重复优化轮数。
    epochs: int = 4


@dataclass
class ComaConfig(EpymarlActorCriticConfig):
    """COMA 配置（反事实优势 + 集中式动作价值 critic）。"""
    # 同名 Agent 省略类型时使用的默认插件标识。
    default_agent_type: ClassVar[str] = AgentType.POLICY_RNN.value

    # COMA Learner 的配置标识。
    learner: str = "coma_learner"
    # 优化器学习率。
    lr: float = 3e-4
    # 目标网络更新参数，大于 1 表示硬更新间隔，至多为 1 时表示软更新系数。
    target_update_interval_or_tau: float = 200
    # 算法网络的隐藏层宽度。
    hidden_dim: int = 64
    # COMA 集中式动作价值 Critic 的配置标识。
    critic_type: str = "coma_critic"
    # COMA 动作价值函数的配置标识。
    critic_q_fn: str = "coma"
    # COMA 反事实基线函数的配置标识。
    critic_baseline_fn: str = "coma"
    # Critic 训练模式配置，seq 表示序列。
    critic_train_mode: str = "seq"
    # Critic 重复训练次数的配置值。
    critic_train_reps: int = 1
    # 策略熵奖励的权重。
    entropy_coef: float = 0.001
    # 计算多步 TD 目标时使用的步数。
    q_nstep: int = 10
    # 连续动作反事实基线的蒙特卡洛样本数。
    counterfactual_samples: int = 16
    # 从框架 ReplayBuffer 抽取训练样本的策略。
    sample_strategy: str = SamplingStrategy.ON_POLICY_CURSOR.value


@dataclass
class MaddpgConfig(LearnerAlgoConfig):
    """原生连续动作 MADDPG 配置。"""
    # 同名 Agent 省略类型时使用的默认插件标识。
    default_agent_type: ClassVar[str] = AgentType.MADDPG_MLP.value

    # Actor 与 Critic 共用的 Adam 学习率；分离学习率未设置时回退到该值。
    lr: float = 5e-4
    # 可选的 Actor 独立学习率。BC 初始化时应显著小于 Critic，避免 Critic 未收敛时破坏策略。
    actor_lr: Optional[float] = None
    # 可选的 Critic 独立学习率。未设置时回退到 lr。
    critic_lr: Optional[float] = None
    # 前 N 次 Learner 更新只训练 Critic，冻结 Actor，等待价值函数建立。
    actor_warmup_updates: int = 0
    # TD 目标最多累积的连续转移数。uniform_steps 会为每个随机起点
    # 物化最多 n_step 步的同局窗口，不跨 episode 边界。
    n_step: int = 1
    # Actor 对初始（含文件加载）策略的行为克隆锚定系数。为 0 时关闭。
    bc_anchor_coef: float = 0.0
    # 锚定系数线性退火的终值。
    bc_anchor_final_coef: float = 0.0
    # 从 Actor warmup 结束后起计的退火 Learner 更新数；0 表示固定使用起始系数。
    bc_anchor_decay_updates: int = 0
    # 目标网络更新参数，(0, 1] 为软更新系数，大于 1 的整数为硬更新间隔。
    target_update_interval_or_tau: float = 0.01
    # Actor 和 Critic 的隐藏层宽度。
    hidden_dim: int = 128
    # 将配对 Agent 的输出类型设为连续 Actor。
    agent_output_type: AgentOutputType = AgentOutputType.CONTINUOUS_ACTION
    # 确定性 Actor 均值的 L2 正则系数。
    reg: float = 0.001
    # 从 ReplayBuffer 中按转移起点均匀、有放回地抽样。
    sample_strategy: str = SamplingStrategy.UNIFORM_STEPS.value
    # 一次 Learner 更新的回放抽样量，默认按步计数。
    batch_size: int = 32
    # 每个学习令牌触发的 Learner 更新次数。
    learn_times_per_episode: int = 1


@dataclass
class PacConfig(EpymarlActorCriticConfig):
    """PAC 配置；EPyMARL 仅提供非参数共享版本。"""
    # 历史 PAC 配置的 Learner 标识。
    learner: str = "pac_learner"
    # 优化器学习率。
    lr: float = 3e-4
    # 算法网络的隐藏层宽度。
    hidden_dim: int = 64
    # 历史 PAC 配置的非共享动作价值 Critic 标识。
    critic_type: str = "pac_critic_ns"
    # 历史 PAC 配置的非共享状态价值网络标识。
    state_value_type: str = "cv_critic_ns"
    # 熵正则日程的起始系数。
    initial_entropy_coef: float = 20.0
    # 熵正则日程的结束系数。
    final_entropy_coef: float = 0.01
    # 熵系数衰减结束时所处的训练进度比例。
    entropy_end_ratio: float = 0.8
    # 计算多步 TD 目标时使用的步数。
    q_nstep: int = 10
    # 历史 PAC 配置的训练环境步数上限。
    t_max: int = 500_500
    # 是否标准化训练奖励。
    standardise_rewards: bool = False


@dataclass
class PacDcgConfig(PacConfig):
    """PAC-DCG 配置；用深度协调图 critic 进行联合动作推理。"""
    # 历史 PAC 协调图变体的 Learner 标识。
    learner: str = "pac_dcg_learner"
    # 优化器学习率。
    lr: float = 5e-4
    # 历史 PAC 配置的非共享协调图 Critic 标识。
    critic_type: str = "pac_dcg_critic_ns"
    # 协调图变体的初始熵正则系数。
    initial_entropy_coef: float = 30.0
    # 协调图连接方式、边数或显式边列表。
    cg_edges: Union[str, int, list] = "full"
    # 协调图个体价值网络的隐藏层宽度。
    cg_utilities_hidden_dim: Optional[Union[int, list[int]]] = None
    # 协调图成对价值网络的隐藏层宽度。
    cg_payoffs_hidden_dim: Optional[Union[int, list[int]]] = None
    # 协调图成对价值矩阵分解的秩。
    cg_payoff_rank: Optional[int] = None
    # 是否采用分离状态价值和动作优势的结构。
    duelling: bool = False
    # 是否保留消息传递过程中评分最高的动作。
    msg_anytime: bool = True
    # 协调图消息传递的迭代次数。
    msg_iterations: int = 8
    # 是否对协调图传递的消息做归一化。
    msg_normalized: bool = True


@dataclass
class PpoConfig(LearnerAlgoConfig):
    """
    单智能体 PPO 学习器算法配置（stable_baselines3 PPO）。

    字段直接对应 SB3 PPO 的同名超参；框架通用超参(gamma/grad_norm_clip 等)仍走
    LearnerConfig skeleton。Learner 在边界处把这些字段映射到 stable_baselines3.PPO。
    """
    # 同名 Agent 省略类型时使用的默认插件标识。
    default_agent_type: ClassVar[str] = AgentType.SB3.value

    # 学习率。
    lr: float = 3e-4
    # Actor/Critic MLP 隐藏层宽度。
    hidden_dim: int = 64
    # 配对 Agent 的输出语义（Learner 初始化时同步到 AgentConfig）。
    agent_output_type: AgentOutputType = AgentOutputType.PI_LOGITS
    # SB3 策略网络类型；离散/连续向量观测用 MlpPolicy。
    policy: str = "MlpPolicy"
    # 占位 rollout 长度；离线训练时按样本量重建 RolloutBuffer，不依赖该值。
    n_steps: int = 2048
    # 每批 on-policy 样本重复优化轮数。
    n_epochs: int = 10
    # GAE 的 lambda。
    gae_lambda: float = 0.95
    # PPO 策略裁剪范围。
    clip_range: float = 0.2
    # 价值函数裁剪范围；None 表示不裁剪。
    clip_range_vf: Optional[float] = None
    # 策略熵奖励系数。
    ent_coef: float = 0.0
    # 价值损失系数。
    vf_coef: float = 0.5
    # 梯度裁剪阈值（SB3 内部使用，与 skeleton.grad_norm_clip 区分）。
    max_grad_norm: float = 0.5
    # 是否对优势做标准化。
    normalize_advantage: bool = True
    # KL 早停阈值；None 表示不早停。
    target_kl: Optional[float] = None
    # 采样策略
    sample_strategy: str = SamplingStrategy.ON_POLICY_RECENT.value
    # PPO minibatch 大小。
    batch_size: int = 10
    # SB3每次梯度更新使用的transition数量；与框架采样的episode数量分开配置。
    minibatch_size: int = 256
    # 样本更新后学习次数。
    learn_times_per_episode: int = 1


@dataclass
class A2CConfig(LearnerAlgoConfig):
    """单智能体 A2C 学习器算法配置（stable_baselines3 A2C）。"""
    # 同名 Agent 省略类型时使用的默认插件标识。
    default_agent_type: ClassVar[str] = AgentType.SB3.value

    # SB3 A2C 默认学习率。
    lr: float = 7e-4
    # Actor/Critic MLP 隐藏层宽度。
    hidden_dim: int = 64
    # 配对 Agent 输出策略 logits。
    agent_output_type: AgentOutputType = AgentOutputType.PI_LOGITS
    # SB3 策略网络类型。
    policy: str = "MlpPolicy"
    # SB3 原始 rollout 长度；框架内会按实际样本数重建 RolloutBuffer。
    n_steps: int = 5
    # GAE lambda；1.0 对齐 SB3 A2C 默认值。
    gae_lambda: float = 1.0
    # 策略熵奖励系数。
    ent_coef: float = 0.0
    # 价值损失系数。
    vf_coef: float = 0.5
    # SB3 内部梯度裁剪阈值。
    max_grad_norm: float = 0.5
    # RMSprop epsilon。
    rms_prop_eps: float = 1e-5
    # 使用 A2C 原论文风格的 RMSprop；False 时使用 Adam。
    use_rms_prop: bool = True
    # 是否对优势做标准化。
    normalize_advantage: bool = False
    # A2C 消费最近的 on-policy episode，消费后清空旧行为策略样本。
    sample_strategy: str = SamplingStrategy.ON_POLICY_RECENT.value
    # 每次学习令牌采样的 episode 数量。
    batch_size: int = 10
    # 样本更新后执行的 A2C 整批梯度更新次数。
    learn_times_per_episode: int = 1


@dataclass
class DqnConfig(LearnerAlgoConfig):
    """
    单智能体 DQN 学习器算法配置（stable_baselines3 DQN）。

    SB3 DQN 自带内部 transition replay buffer；框架 ReplayBuffer 仍按 episode 存储。
    Learner.update 会把框架样本中尚未写入过的 transition 同步到 SB3 replay buffer，
    再调用 DQN.train，避免同一条历史样本被外层采样策略重复放大。
    如果样本携带动态动作掩码，Learner 会绕过 SB3 replay，使用自定义 masked TD target 更新。
    当前直接复用 SB3 基础 DQN，不包含 Double/Dueling/Rainbow 扩展。
    """
    # 同名 Agent 省略类型时使用的默认插件标识。
    default_agent_type: ClassVar[str] = AgentType.SB3.value

    # 学习率。
    lr: float = 1e-4
    # Q 网络 MLP 隐藏层宽度。
    hidden_dim: int = 64
    # 配对 Agent 输出 Q 值。
    agent_output_type: AgentOutputType = AgentOutputType.Q
    # SB3 策略网络类型；离散向量观测用 MlpPolicy。
    policy: str = "MlpPolicy"
    # SB3 内部 replay buffer transition 容量。
    buffer_size: int = 100000
    # SB3 内部 replay buffer 至少积累多少 transition 后开始训练。
    learning_starts: int = 1000
    # 目标网络软更新系数；1.0 表示硬同步。
    tau: float = 1.0
    # SB3 DQN 训练频率占位；本框架手动调用 train，不依赖 learn() 调度。
    train_freq: int = 1
    # 每次 Learner.update 调用 DQN.train 的梯度步数。
    gradient_steps: int = 1
    # SB3 DQN 从内部 replay buffer 抽取的 transition 数量。
    train_batch_size: int = 64
    # 目标网络同步间隔。
    target_update_interval: int = 1000
    # DQN 执行侧 epsilon 由框架采样令牌控制，SB3 内部探索参数保留默认占位。
    exploration_fraction: float = 0.1
    # 传给 SB3 DQN 的初始探索概率，框架采样使用任务令牌。
    exploration_initial_eps: float = 1.0
    # 传给 SB3 DQN 的最终探索概率，框架采样使用任务令牌。
    exploration_final_eps: float = 0.05
    # SB3 内部梯度裁剪阈值。
    max_grad_norm: float = 10.0
    # 框架 ReplayBuffer 采样策略；DQN 是 off-policy，默认近期偏好。
    sample_strategy: str = SamplingStrategy.RECENT_BIASED.value
    # 每次 Learner.update 从框架 ReplayBuffer 抽取的 episode 数量。
    batch_size: int = 16
    # 每个学习令牌执行的 Learner.update 次数。
    learn_times_per_episode: int = 1


LEARNER_ALGO_CONFIG: dict[AlgorithmType, type[LearnerAlgoConfig]] = {
    AlgorithmType.COMA: ComaConfig,
    AlgorithmType.IA2C: Ia2cConfig,
    AlgorithmType.IPPO: IppoConfig,
    AlgorithmType.IQL: IqlConfig,
    AlgorithmType.MAA2C: Maa2cConfig,
    AlgorithmType.MADDPG: MaddpgConfig,
    AlgorithmType.QMIX: QmixConfig,
    AlgorithmType.MAPPO: MappoConfig,
    AlgorithmType.PAC: PacConfig,
    AlgorithmType.PAC_DCG: PacDcgConfig,
    AlgorithmType.VDN: VdnConfig,
    AlgorithmType.PPO: PpoConfig,
    AlgorithmType.A2C: A2CConfig,
    AlgorithmType.DQN: DqnConfig,
}


@dataclass
class LearnerConfig(DataClassJsonMixin):
    """
    Learner骨架配置。

    一个LearnerConfig实例对应一个Learner实例，通过name和AgentConfig.name绑定；
    algo_type决定具体算法实现并在__post_init__索引到对应的algo_config；
    skeleton只保留各算法通用的训练超参和运行期维度，算法特有字段收敛到algo_config。
    """

    # Learner配置名称，必须和它训练的AgentConfig.name一致。
    name: AgentName = "default"
    # Learner算法类型，用于注册表选择具体Learner实现。
    algo_type: str = AlgorithmType.QMIX.value
    # 是否使用公共团队奖励。
    common_reward: bool = True
    # 是否标准化回报。
    standardise_returns: bool = False
    # 是否标准化奖励。
    standardise_rewards: bool = True
    # 梯度裁剪阈值。
    grad_norm_clip: float = 5.0
    # Learner日志打印间隔。
    learner_log_interval: int = 1
    # 折扣因子。
    gamma: float = 0.99
    # torch cuda标记，运行时会根据device回填。
    use_cuda: bool = False
    # 计算设备。
    device: Optional[Union[torch.device, str]] = "cpu"
    # 运行期智能体数量。
    n_agents: int = 0
    # 运行期动作数量。
    n_actions: int = 0
    # 运行期全局状态维度。
    state_shape: Any = 1
    # EPyMARL batch scheme。
    scheme: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 算法特有配置；None 时由 __post_init__ 按 algo_type 索引填充对应 *Config。
    algo_config: Optional[LearnerAlgoConfig] = None

    def __post_init__(self):
        try:
            algo_type = AlgorithmType(self.algo_type)
        except ValueError:
            if self.algo_config is None:
                raise ValueError(
                    f"Custom algo_type={self.algo_type!r} requires a LearnerAlgoConfig "
                    "provided by the active Learner plugin declaration"
                ) from None
            self.algo_type = str(self.algo_type)
            if not isinstance(self.algo_config, LearnerAlgoConfig):
                raise TypeError(
                    "LearnerConfig.algo_config must inherit LearnerAlgoConfig, "
                    f"given={type(self.algo_config).__name__}"
                )
            return
        self.algo_type = algo_type.value
        if self.algo_config is None:
            self.algo_config = LEARNER_ALGO_CONFIG[algo_type]()
        assert isinstance(self.algo_config, LearnerAlgoConfig), (
            f"LearnerConfig.algo_config must inherit LearnerAlgoConfig, "
            f"given={type(self.algo_config).__name__}"
        )

    def bind_runtime(
            self,
            *,
            n_agents: int,
            n_actions: int,
            state_shape: Any,
            obs_dim: int,
            hidden_dim: Optional[int] = None,
            obs_agent_id: Optional[bool] = None,
            device: Optional[Union[torch.device, str]] = None,
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_shape = state_shape
        # 让Learner网络隐藏维度与Agent网络保持一致。
        if hidden_dim is not None:
            self.algo_config.hidden_dim = hidden_dim
        # obs_agent_id是critic特有字段，回填到algo_config(仅actor-critic类算法持有)。
        if obs_agent_id is not None and hasattr(self.algo_config, "obs_agent_id"):
            self.algo_config.obs_agent_id = obs_agent_id
        if device is not None:
            self.device = device
        self.use_cuda = str(self.device).startswith("cuda")
        self.scheme = {
            "obs": {"vshape": obs_dim},
            "actions_onehot": {"vshape": (n_actions,)},
            "state": {"vshape": state_shape},
        }
        return self


@dataclass
class ModelManagerConfig(DataClassJsonMixin):
    # 模型池本地工作目录，保存本次运行产生的临时模型权重和单版本 metadata。
    # final/optimo 等可交付模型仍由 TrainConfig.model_save_dir 保存到 artifacts/models。
    tmp_local_dir: str = ResourceDir.MODEL_POOL.value
    # 第一个提交的模型是否自动晋升为 champion。
    auto_promote_first_model: bool = True
    # 每个 Agent 独立累计多少个完成的学习批次后自动晋升最新模型，注意一个批次会学习多次：learn_times_per_episode，-1 表示关闭。
    auto_promote_interval: int = 1
    # 开启后，在一个自动晋升周期内选择 metrics 中 reward 最高的候选模型；关闭时选择最后一次更新。
    auto_promote_optimo_reward: bool = False
    # 每个 Agent 在本地模型池中最多保留的最近版本数；-1 表示关闭自动回收。
    # champion、specialist、quarantined 版本，以及尚在自动晋升窗口中的 candidate 始终额外保留。
    max_versions_per_model: int = 8
    # 后端存储类型
    storage_type: str = StorageType.LOCAL.value
    # 后端存储信息
    storage = None

    def __post_init__(self):
        assert self.auto_promote_interval == -1 or self.auto_promote_interval > 0, (
            "ModelManagerConfig.auto_promote_interval must be -1 or a positive integer."
        )
        assert self.max_versions_per_model == -1 or self.max_versions_per_model > 0, (
            "ModelManagerConfig.max_versions_per_model must be -1 or a positive integer."
        )
        storage_type = StorageType(self.storage_type)
        self.storage_type = storage_type.value
        if storage_type == StorageType.S3:
            self.storage = S3StorageConfig(bucket_name=StorageBucket.MODELS_POOL.value)
        elif storage_type == StorageType.LOCAL:
            self.storage = LocalStorageConfig()
            self.storage.data_path = ResourceDir.MODELS.value


@dataclass
class TaskConfig(DataClassJsonMixin):
    """TaskManager任务配置基类，收敛训练和评估任务共享的调度与产物字段。"""
    # 任务总局数；训练和评估子类按各自语义覆盖默认值。
    max_episode: int = 1
    # 采样驱动方式；训练支持online/offline，评估仅支持online。
    sample_type: str = SampleType.ONLINE.value
    # 是否录制每局结构化态势与动作样本。
    record_sample: bool = False
    # 任务正常结束后是否自动归档已登记产物。
    artifacts_archive_enabled: bool = True
    # 任务归档根目录；归档器会在其下创建TASK_ID目录。
    artifacts_archive_save_dir: str = ResourceDir.ARCHIVES.value
    # 结果输出目录
    result_save_dir: str = ResourceDir.RESULTS.value
    # render录屏GIF保存目录；环境只提供单帧render数据，GIF落盘由RolloutSampleWorker负责。
    render_video_dir: LiteralString = ResourceDir.RECORDINGS.value
    # render录屏GIF帧率；帧率越高动作越顺滑，但文件体积也越大。
    render_gif_fps: int = 15
    # render录屏GIF采帧间隔；1表示每个环境step都记录一帧，长任务可调大以压缩文件体积。
    render_gif_frame_stride: int = 1
    # render录屏GIF最大帧数；None表示完整记录，长任务可设置上限避免GIF过大。
    render_gif_max_frames: Optional[int] = None
    # Agent顺序来源；order使用环境默认顺序，decision由Episode Agent动作交给环境解释。
    agent_compose_type: Literal['order', 'decision'] = 'order'
    # 当前任务对 EnvConfig.metadata 的递归增量覆盖；随采样令牌下发，并供 Learner 无头环境使用。
    env_metadata: dict = field(default_factory=dict)
    # 当前任务对托管 SimInstances metadata 的覆盖。
    sim_metadata: dict = field(default_factory=dict)
    # 采样令牌领取后允许无结果占用的最长时间，单位秒；超时后由任务管理器回收并重新分发。
    sample_task_timeout: float = 300.0
    # 样本文件读写、校验及离线索引配置；Train/Eval共享同一配置源。
    monte_carlo_method_config: Optional["MonteCarloMethodConfig"] = None

    class Meta:
        undefined = Undefined.EXCLUDE


@dataclass
class TrainConfig(TaskConfig):
    # 最大训练局数
    max_episode: int = 2000
    # 结果输出目录
    result_save_dir: str = ResourceDir.RESULTS.value
    # render录屏GIF保存目录；环境只提供单帧render数据，GIF落盘由RolloutSampleWorker负责。
    render_video_dir: LiteralString = ResourceDir.RECORDINGS.value
    # Agent顺序规则来源。
    agent_compose_type: Literal['order', 'decision'] = 'order'
    # 训练任务级随机种子；None表示不固定随机性。分布式角色会按role_id派生子seed，避免多采样进程同序列。
    seed: Optional[int] = None
    # 模型初始化专用种子。None 时沿用 seed；从 AutoML 归档复现实验时，可与该代运行 seed 分开填写。
    initialization_seed: Optional[int] = None
    # 为 True 时，达到或刷新本 generation 历史最大 episode 回报则由 TrainTaskManager 覆盖保存 optimo 前缀模型。
    save_optimo_models: bool = True
    # 多少轮agent更新使用learner的模型。
    agent_reload_model_cycle: int = 1
    # 分布式学习多少轮更新总线网络
    center_model_cycle: int = 10
    # 模型保存路径
    model_save_dir: str = ResourceDir.MODELS.value
    # 从哪个模型版本开始训练
    model_version: Optional[str] = None
    # 模型加载路径，如果为None，则使用model_save_dir
    model_load_dir: Optional[str] = ResourceDir.MODELS.value
    # 初始随机探索值
    start_epsilon: float = 1
    # 结束时随机探索值
    end_epsilon: float = 0.05
    # epsilon衰减局数（在多少局内从start_epsilon衰减到end_epsilon）
    epsilon_decay_episodes: int = 200
    # 使用Categorical随机采样时的温度
    temperature: float = 1.0
    # 在多少局内使用temperature系数
    temperature_episodes: int = 0
    # 是否开启盘中评估。
    mid_eval_enabled: bool = False
    # 每完成多少局训练采样触发一次盘中评估。
    mid_eval_episode_interval: int = 100
    # 主路采样序列消费完后自动评估的局数；0 表示阶段结束不评估、也不画结束评估奖励图。
    # 分阶段训练时每个阶段读自己的 TrainConfig，阶段结束后独立执行。
    end_eval_episodes: int = 0
    # 加载已有缓存数据的后端存储配置
    storage = None
    # AutoML配置；None 或 monitor_enabled=False 时不启动监控线程（单阶段Role使用）。
    auto_ml_config: Optional["AutoMLConfig"] = None
    # Maximum episode lag allowed between sampling and learning. None disables backpressure.
    sample_learn_max_lag: Optional[int] = None
    # 在线分布式训练按令牌版本采样，每次更新只消费该版本的数据。
    versioned: bool = True
    # 滚动采样允许多个 episode 同时在途，并接收旧版本回传。
    allow_sample_overlap: bool = False

    class Meta:
        undefined = Undefined.EXCLUDE


@dataclass
class ArtifactsManagerConfig(DataClassJsonMixin):
    """产物台账与归档管理器配置；TOML 节可选，缺省用类默认值。"""
    # 任务归档根目录；归档器在其下创建 TASK_ID 目录。远端 put 缓存固定为 <save_dir>/_remote_cache/。
    save_dir: str = ResourceDir.ARCHIVES.value

    class Meta:
        undefined = Undefined.EXCLUDE


def _env_interval_alias(config_cls):
    """保留 dataclass 构造器，仅将有序输入中的旧名称翻译成同一字段。"""
    original_init = config_cls.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        values = {("step_interval" if key == "step_interval_secs" else key): value
                  for key, value in kwargs.items()}
        original_init(self, *args, **values)

    config_cls.__init__ = init
    return config_cls


@_env_interval_alias
@dataclass
class EnvConfig(DataClassJsonMixin):
    # 运行场景名称（环境插件名），须由科目 TOML 显式配置。
    sense_name: str = ""
    # 一局最大多少步
    max_step_per_episode: int = 50
    # 智能体清单
    agent_names: list[str] = field(default_factory=list)
    # 每次交互推进的原生时间跨度；时间流单位或逻辑步数由仿真定义。
    step_interval: float = 1
    # 全局态势维度
    state_dim: int = field(default_factory=int)
    # 其他场景定制信息
    metadata: dict = field(default_factory=dict)
    # 环境是否依赖Simulation层；本地游戏类环境可设为False，由环境自己管理游戏实例。
    use_sim: bool = True

    @property
    def step_interval_secs(self) -> float:
        """旧配置名称的读写别名；本地游戏仍可用它表示秒数。"""
        return self.step_interval

    @step_interval_secs.setter
    def step_interval_secs(self, value: float) -> None:
        self.step_interval = value

    @classmethod
    def from_dict(cls, kvs: dict, *, infer_missing: bool = False):
        values = {("step_interval" if key == "step_interval_secs" else key): value
                  for key, value in kvs.items()}
        return super().from_dict(values, infer_missing=infer_missing)

    class Meta:
        undefined = Undefined.EXCLUDE


@dataclass(frozen=True)
class SpatialRepresentationConfig:
    """固定网格态势表征配置；输出长度由使用者显式指定。"""

    # 固定网格态势输入的三维形状。
    input_shape: tuple[int, int, int]
    # 态势表征输出的特征向量长度。
    output_dim: int


@dataclass
class EvalConfig(TaskConfig):
    # 评估局数
    max_episode: int = 50
    # 是否使用确定性动作（argmax）
    deterministic: bool = True
    # 模型加载路径
    agent_model_path: str = ResourceDir.MODELS.value
    # 模型加载文件名（单组评估）。多组Agent编排评估请改用agent_model_files。
    agent_model_file: Optional[str] = None
    # 多组Agent编排评估的模型清单：key为AgentConfig.name，value为相对agent_model_path的模型文件名。
    # 例如KAZ：{"commander": "stage_defense_commander_agent_net.pth", "archer": "...", "knight": "..."}。
    agent_model_files: dict[AgentName, str] = field(default_factory=dict)
    # Agent顺序规则来源；环境的Parallel/AEC基类独立决定步进语义。
    agent_compose_type: Literal['order', 'decision'] = 'order'
    # 评估奖励曲线输出目录；样本录制目录由 output_dir 单独控制。
    result_save_dir: str = ResourceDir.RESULTS.value
    # 默认评估采样数据保存目录，未显式传monte_carlo_method_config时使用。
    output_dir: LiteralString = ResourceDir.SAMPLES.value
    # render录屏GIF保存目录；环境只提供单帧render数据，GIF落盘由RolloutSampleWorker负责。
    render_video_dir: LiteralString = ResourceDir.RECORDINGS.value
    # 是否按间隔录制评估 GIF。须显式开启；仅设 render_gif_episode_interval 不会触发录制。
    render_gif_enabled: bool = False
    # 每过多少局评估录一次 GIF；局号整除该间隔才录（10/20/30…）。
    render_gif_episode_interval: int = 1
    # 采样数据文件前缀
    output_prefix: str = "eval"
    # 评估局业务指标：随 sample 令牌下发，RolloutWorker 从 infos 提取后上报 TM。
    metric_outputs: dict[str, dict] = field(default_factory=dict)

    class Meta:
        undefined = Undefined.EXCLUDE


@dataclass
class StageConfig(DataClassJsonMixin):
    """单个训练阶段配置；StageTask按顺序把这些阶段转换成TrainTaskManager实例。"""
    # 阶段名称，会写入EpisodeInfo/LearnInfo，并同步到Environment.stage_name。
    stage_name: str = "stage0"
    # 本阶段训练任务配置，包含episode数量、epsilon衰减、背压等阶段级计划。
    train_config: TrainConfig = field(default_factory=TrainConfig)
    # 本阶段参与采样的全部Agent配置；Agent是训练对象，各阶段agent列表必须相同。
    agents_config: list[AgentConfig] = field(default_factory=list)
    # 本阶段参与采样的全部Agent配置；Agent是训练对象，各阶段agent列表必须相同。
    learners_config: list[LearnerConfig] = field(default_factory=list)
    # 进入本阶段时是否清空 active 池；不影响 rehearsal 存档。
    clear_replay_buffer: bool = True
    # 离开本阶段时，把当时 active 中按比例冻结进 rehearsal[stage_name]；0 表示不归档。
    # 归档条数为 round(capacity * ratio)，不超过当时 active 条数；rehearsal 不占用 capacity。
    rehearsal_archive_ratio: float = 0.0
    # 归档抽样策略：even_span（等间距铺满整段，默认）/ newest（留末期）/ uniform（随机）。
    rehearsal_archive_select: str = "even_span"
    # 进入并训练本阶段时，off-policy batch 从 rehearsal store 抽取的比例；0 表示只用 active。
    # on-policy 策略忽略该比例，只采 active。
    rehearsal_sample_ratio: float = 0.0
    # 本阶段结束时是否把当前回放池快照落到 artifacts/samples（默认不保存）。
    save_replay_buffer: bool = False
    # AutoML配置；None 或 monitor_enabled=False 时该阶段不启动监控线程。
    auto_ml_config: Optional["AutoMLConfig"] = None


@dataclass
class StageTaskConfig(DataClassJsonMixin):
    """分阶段训练总计划，Role持有资源对象，StageTask只负责阶段任务生命周期。"""
    # 阶段列表，按顺序执行；普通训练可视为只有一个stage0的StageTask。
    stages: list[StageConfig] = field(default_factory=list)
    # 课程启动时需要构造/恢复的Learner全集；阶段 learners_config 只决定当阶段谁参与更新。
    bootstrap_learners_config: list[LearnerConfig] = field(default_factory=list)
    # 首个执行阶段的下标；调试时可跳过已经完成的前置阶段。
    start_stage: int = 0
    # 从start_stage起，本次运行各阶段依次加载的模型版本（兼容旧配置；跳过入口优先用 skip_stage_model_*）。
    stage_model_versions: list[str] = field(default_factory=list)
    # --- 跳过前置阶段（start_stage>0）时的入口加载 ---
    # 模型目录；None 表示沿用 starting_stage.train_config.model_load_dir。
    skip_stage_model_load_dir: Optional[str] = None
    # 模型版本前缀（archive `{version}_{learner}_{model}.pth`）；None/空表示不覆盖。
    skip_stage_model_version: Optional[str] = None
    # 回放池快照路径；绝对路径直用，否则相对 ResourceDir.SAMPLES；None/空表示不灌池。
    skip_stage_replay_path: Optional[str] = None


@dataclass
class AutoMLOptunaConfig(DataClassJsonMixin):
    """Optuna 采样与剪枝；sqlite 路径与 study 名由 AutoML 按 subject_name+stage_name 运行时生成。"""
    # trial.report / tell 的优化方向；主指标越大越好用 maximize，越小越好用 minimize。
    direction: str = "maximize"
    # 本地 sqlite study 已存在时是否加载续跑；true 时同一 subject+stage 复用历史 trial。
    load_if_exists: bool = True
    # 下一组超参的采样器：tpe（默认，贝叶斯 TPE）/ random / cmaes。
    sampler_name: str = AutoMLSamplerType.TPE.value
    # TPE、CMA-ES 在攒满这么多个 complete trial 之前用随机采样做冷启动探索。
    sampler_n_startup_trials: int = 10
    # 盘中早停剪枝器：median（默认，按 step 中位数比较）/ hyperband / none（不剪，仅靠坍塌检测）。
    pruner_name: str = AutoMLPrunerType.MEDIAN.value
    # MedianPruner：study 内前 N 个 trial 不参与中位数比较，不会被 median 剪枝。
    pruner_n_startup_trials: int = 5
    # MedianPruner：单个 trial 的 step（已完成局数）低于此值时不剪枝，给慢热 RL trial 热身时间。
    pruner_n_warmup_steps: int = 30
    # HyperbandPruner：最低 rung 资源（step 单位=已完成局数）；低于此 step 不触发剪枝。
    pruner_min_resource: int = 1
    # HyperbandPruner：最高 rung 资源上限（step 单位=已完成局数）。
    pruner_max_resource: int = 200
    # HyperbandPruner：相邻 rung 的资源按该因子递增（如 3 表示 1→3→9→…）。
    pruner_reduction_factor: int = 3

    class Meta:
        undefined = Undefined.EXCLUDE


@dataclass
class AutoMLConfig(DataClassJsonMixin):
    """AutoML 总配置；monitor_enabled=False 时 TM 不启监控线程。除 optuna 外字段均扁平在本层。"""
    # 用于区分 Optuna study 和 AutoML 归档的科目名称。
    subject_name: str
    # 运行期监控开关；TOML 的 enabled 由 TomlTrainingConfig 消费，用于决定是否创建本对象。
    monitor_enabled: bool = True
    # TrainTaskManager 监控线程
    poll_interval_episodes: int = 10
    # 启动 AutoML 指标检查所需的最少完成局数。
    min_episodes_before_check: int = 30
    # 相邻两次重启之间至少完成的局数。
    min_episodes_between_restart: int = 20
    # Optuna trial.report 主指标与坍塌检测
    metric_primary: str = AutoMLPrimaryMetric.REWARD_MEAN.value
    # 计算滑动平均奖励时纳入的最近局数。
    reward_window: int = 50
    # 绝对业务毕业线：到 deadline 时主指标仍未达到该值，则当前 Optuna trial 记为 PRUNED。
    # None / deadline<=0 表示关闭；方向遵循 optuna.direction。
    target_metric: Optional[float] = None
    # 检查主指标是否达标的完成局数，非正值关闭截止检查。
    target_metric_deadline_episodes: int = 0
    # 标准 infos / env_metadata 的逐局输出，以及可选的表达式业务主指标。
    metric_outputs: dict[str, AutoMLMetricOutputConfig] = field(default_factory=dict)
    # 可作为 Optuna 主指标的业务指标表达式及达标条件。
    business_metric: Optional[AutoMLBusinessMetricConfig] = None
    # 当前指标降至正峰值的该比例时触发坍塌检测。
    collapse_restart_ratio: float = 0.35
    # 启用 trial 内指标坍塌检测所需的最少完成局数。
    collapse_min_episodes: int = 80
    # restart 是否落盘归档；开启时固定全量（algos_metrics.csv / 指标 CSV / advice.yaml / artifacts）
    archive_enabled: bool = True
    # restart 归档：本地 archives 根目录（s3 模式上传成功后删除该目录下组装包）
    archive_root_dir: str = os.path.join(ResourceDir.AUTOML.value, "archives")
    # 与 ModelManagerConfig / MonteCarloMethodConfig 一致：storage_type + storage
    storage_type: str = StorageType.LOCAL.value
    # AutoML 归档后端的本地或 S3 存储配置。
    storage: Optional[Union[LocalStorageConfig, S3StorageConfig]] = None
    # Optuna 的优化方向、采样器和剪枝器配置。
    optuna: AutoMLOptunaConfig = field(default_factory=AutoMLOptunaConfig)
    # Learner / 环境 metadata / 全局种子三类业务搜索入口，进入 Optuna 后合并为同一 trial。
    learner_search_space: list[AutoMLLearnerSearchParamConfig] = field(default_factory=list)
    # 参与 Optuna 搜索的环境 metadata 字段及取值范围。
    env_metadata_search_space: list[AutoMLEnvMetadataSearchParamConfig] = field(default_factory=list)
    # 可选的训练随机种子搜索范围。
    seed_search: Optional[AutoMLSeedSearchConfig] = None
    # 采样跑满后额外完整探索轮数（0=只跑一轮）；由 TrainTaskManager.is_sample_completed 触发 restart。
    extra_complete_rounds: int = 0

    class Meta:
        undefined = Undefined.EXCLUDE

    def __post_init__(self):
        storage_type = StorageType(self.storage_type)
        self.storage_type = storage_type.value
        if storage_type == StorageType.S3:
            if self.storage is None:
                self.storage = S3StorageConfig(bucket_name=StorageBucket.AUTOML.value)
            elif isinstance(self.storage, S3StorageConfig) and not self.storage.bucket_name:
                self.storage.bucket_name = StorageBucket.AUTOML.value
            elif not isinstance(self.storage, S3StorageConfig):
                self.storage = S3StorageConfig(bucket_name=StorageBucket.AUTOML.value)
        else:
            local_storage = (
                self.storage
                if isinstance(self.storage, LocalStorageConfig)
                else LocalStorageConfig()
            )
            local_storage.data_path = self.archive_root_dir
            self.storage = local_storage


@dataclass
class ReplayBufferConfig(DataClassJsonMixin):
    # 没有显式分区容量时的初始容量（每个 AgentName 独立计数）。
    capacity: int = 1000
    # ReplayBuffer 运行期通过接口维护；直接构造分区回放池时也可显式配置。
    agent_capacities: dict[str, int] = field(default_factory=dict)
    # 默认从回放池中采样数据批大小
    batch_size: int = 64
    # 默认的经验池抽样策略
    strategy: str = SamplingStrategy.RECENT_BIASED.value
    # 近期偏好抽样时的最新数据比例
    recent_ratio: float = 0.3
    # 归档与抽样用的随机种子；None 表示不固定。未指定时由 Role 沿用 train_config.seed。
    seed: Optional[int] = None
    # 阶段快照落盘/加载的后端存储类型（local / s3）。
    storage_type: str = StorageType.LOCAL.value
    # 后端存储；local 默认 LocalStorageConfig(data_path=ResourceDir.SAMPLES)。
    storage: Union[LocalStorageConfig, S3StorageConfig] = field(default_factory=LocalStorageConfig)
    # 张量池后端：memory（进程内 CPU 张量）或 lazy_memmap（磁盘 mmap）。
    tensor_pool: str = "memory"
    # True 时关闭任务后保留 mmap scratch 的磁盘文件。
    keep_scratch: bool = False
    # 样本封装/抽样单位：independent 保存与抽取独立组轨迹（默认）；
    # episode_groups 把同一物理 episode、同一 AgentName 的多组轨迹封成一个
    # 张量样本并按该样本抽取（[B,G,T+1,K,D] 的有效组铺平为 [N,T+1,K,D]）。
    # episode_groups 下 capacity 的单位是物理局；batch_size 的单位由抽样策略决定。
    # on-policy 与 sliding_window 策略不支持捆绑抽取。
    sample_packaging: str = "independent"

    def __post_init__(self):
        storage_type = StorageType(self.storage_type)
        self.storage_type = storage_type.value
        if storage_type == StorageType.S3:
            if not isinstance(self.storage, S3StorageConfig):
                self.storage = S3StorageConfig(bucket_name=StorageBucket.REPLAY_BUFFER.value)
            elif not self.storage.bucket_name:
                self.storage.bucket_name = StorageBucket.REPLAY_BUFFER.value
        else:
            local_storage = (
                self.storage
                if isinstance(self.storage, LocalStorageConfig)
                else LocalStorageConfig()
            )
            if not local_storage.data_path:
                local_storage.data_path = ResourceDir.SAMPLES.value
            self.storage = local_storage


@dataclass
class MonteCarloMethodConfig(DataClassJsonMixin):
    """ 蒙特卡洛方法配置，用于样本写入、读取、校验和回放池加载。 """
    # 最大样本容量
    sample_size: int = 50000
    # 每采样多少次上传一次存储
    upload_interval: int = 100
    # 本地存文件临时位置
    tmp_local_dir: str = LocalStorageConfig.data_path
    # 后端存储类型
    storage_type: str = StorageType.LOCAL.value
    # 后端存储配置
    storage = None
    # 是否开启写入；False时只作为历史样本读取器使用。
    write_enabled: bool = True
    # 读取已有样本时指定样本文件名或S3对象key。
    load_sample_name: Optional[str] = None
    # 读取本地已有样本时指定完整文件路径，优先级高于tmp_local_dir + load_sample_name。
    load_file_path: Optional[str] = None
    # 是否接受最后一步未达到 episode_step_limit 的连续片段。
    accept_truncated_episodes: bool = False
    # 发现步号缺口时是否拆成多个连续候选轨迹。
    split_on_step_gap: bool = False

    def __post_init__(self):
        storage_type = StorageType(self.storage_type)
        self.storage_type = storage_type.value
        if storage_type == StorageType.S3:
            self.storage = S3StorageConfig()
        elif storage_type == StorageType.LOCAL:
            self.storage = LocalStorageConfig()


@dataclass
class SampleLearningTasksConfig(DataClassJsonMixin):
    # 学习次数
    learn_times: int = 2000
    # 采用什么采样策略
    sample_policy: str = "sliding_window"
    # 加载已有缓存数据文件名；本地存储时相对 load_sample_storage.data_path 查找
    load_sample_name: Optional[str] = None
    # 加载已有缓存数据的后端存储类型
    load_storage_type: str = StorageType.LOCAL.value
    # 样本文件所在存储
    load_sample_storage: Optional[Union[S3StorageConfig, LocalStorageConfig]] = None
    # 加载样本局数
    max_sample_episodes: int = 1000
    # 加载模型字典，模型名称：模型文件名
    load_model_names: dict[str, str] = field(default_factory=dict)
    # 加载已有缓存数据的后端存储类型
    load_model_storage_type: str = StorageType.LOCAL.value
    # 样本存储池
    load_model_storage: Optional[Union[S3StorageConfig, LocalStorageConfig]] = None

    def __post_init__(self):
        if self.max_sample_episodes <= 0:
            raise ValueError(f"max_sample_episodes must be positive, got {self.max_sample_episodes}")
        load_storage_type = StorageType(self.load_storage_type)
        load_model_storage_type = StorageType(self.load_model_storage_type)
        self.load_storage_type = load_storage_type.value
        self.load_model_storage_type = load_model_storage_type.value

        if load_storage_type == StorageType.S3 and self.load_sample_storage is None:
            self.load_sample_storage = S3StorageConfig()
        elif load_storage_type == StorageType.LOCAL and self.load_sample_storage is None:
            self.load_sample_storage = LocalStorageConfig()

        if load_model_storage_type == StorageType.S3 and self.load_model_storage is None:
            self.load_model_storage = S3StorageConfig(bucket_name=StorageBucket.MODELS_POOL.value)
        elif load_model_storage_type == StorageType.LOCAL and self.load_model_storage is None:
            self.load_model_storage = LocalStorageConfig()


@dataclass
class SampleLearningEvalConfig(DataClassJsonMixin):
    """样本学习评估角色配置；训练、评估、样本学习任务配置由入口单独注入。"""
    # 评估结果输出目录。
    result_save_dir: str = ResourceDir.RESULTS.value
    # 是否在评估结束后保存学习到的模型。
    save_model: bool = True
    # 模型保存后端类型。
    model_storage_type: str = StorageType.LOCAL.value
    # 模型保存后端配置；local默认artifacts/models，s3默认models-pool桶。
    model_storage: Optional[Union[S3StorageConfig, LocalStorageConfig]] = None
    # 模型本地保存目录；S3上传前也会先落到该目录。
    model_local_dir: str = ResourceDir.MODELS.value
    # 模型文件名前缀；None时使用sample_eval_{algo}_{timestamp}_{task_id}。
    model_save_name: Optional[str] = None

    class Meta:
        undefined = Undefined.EXCLUDE

    def __post_init__(self):
        model_storage_type = StorageType(self.model_storage_type)
        self.model_storage_type = model_storage_type.value
        if model_storage_type == StorageType.S3 and self.model_storage is None:
            self.model_storage = S3StorageConfig(bucket_name=StorageBucket.MODELS_POOL.value)
        elif model_storage_type == StorageType.LOCAL and self.model_storage is None:
            self.model_storage = LocalStorageConfig(data_path=self.model_local_dir)


@dataclass
class LogConfig(DataClassJsonMixin):
    # 日志保存目录
    log_dir: str = os.path.join(root_dir_position, 'logs')
    # 日志基础名称；logger 追加任务 ID，同任务各进程共同写入，轮转分片追加序号。
    log_file_name: str = 'rl.log'
    # console日志格式
    console_log_format: str = '%(asctime)s | %(task_id)s | %(name)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s'
    # file日志格式
    file_log_format: str = '%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s'
    # 打印日志级别
    log_level: str = 'INFO'
    # 启用的日志 Handler：console / file，默认两者都开。
    # 配置构造不创建目录；入口预注册日志产物时准备目录，独立 file handler 按需准备。
    enabled_handlers: list[str] = field(
        default_factory=lambda: [LogHandlerKind.CONSOLE.value, LogHandlerKind.FILE.value],
    )
    # 允许打印的debug列表
    log_debug_enable: list[str] = field(default_factory=list)

    class Meta:
        undefined = Undefined.EXCLUDE

    def __post_init__(self):
        if "show" not in self.log_debug_enable:
            self.log_debug_enable.append("show")
        # pytest 设置 DODORL_TEST_LOG_HANDLERS=console，避免测试在项目 logs/ 下落盘。
        # log_dir 在项目目录外（例如 tmp_path）时仍尊重配置，便于单测 file handler。
        test_handlers = os.environ.get("DODORL_TEST_LOG_HANDLERS", "").strip()
        if test_handlers:
            try:
                Path(self.log_dir).resolve().relative_to(Path.cwd().resolve())
            except ValueError:
                pass
            else:
                self.enabled_handlers = [
                    part.strip() for part in test_handlers.split(",") if part.strip()
                ]
        valid_handlers = {kind.value for kind in LogHandlerKind}
        normalized_handlers: list[str] = []
        for handler_name in self.enabled_handlers:
            normalized = str(handler_name).strip().lower()
            if normalized not in valid_handlers:
                raise ValueError(
                    f"Unknown log handler '{handler_name}', valid={sorted(valid_handlers)}"
                )
            if normalized not in normalized_handlers:
                normalized_handlers.append(normalized)
        if not normalized_handlers:
            raise ValueError("LogConfig.enabled_handlers must not be empty")
        self.enabled_handlers = normalized_handlers


@dataclass(frozen=True)
class ThreatRoamingConfig:
    """四分区威胁游走、极坐标观察与独立奖励参数。"""

    # 各责任区分层生成互不重叠的随机圈；圈心独立于威胁实际位置。
    circle_radius_km: float = 50.0
    # 更换责任区时新旧圈心之间的最小距离，单位公里。
    circle_center_min_shift_km: float = 100.0
    # 更换责任区时新旧圈心之间的最大距离，单位公里。
    circle_center_max_shift_km: float = 200.0
    # 判定到达航点的距离阈值，单位公里。
    waypoint_reached_km: float = 5.0
    # 生成航点时距圈心的最小距离，单位公里。
    waypoint_min_distance_km: float = 10.0
    # 生成航点时距圈心的最大距离，单位公里。
    waypoint_max_distance_km: float = 25.0
    # 责任区布局随机刷新间隔的最小步数。
    waypoint_refresh_min_steps: int = 40
    # 责任区布局随机刷新间隔的最大步数。
    waypoint_refresh_max_steps: int = 80
    # 单个航点的最长追踪步数，也限制布局刷新间隔。
    waypoint_timeout_steps: int = 200
    # 极坐标态势表征的感知半径，单位公里。
    perception_radius_km: float = 100.0
    # 极坐标态势表征的径向分箱数量。
    radial_bins: int = 10
    # 极坐标态势表征的角度分箱数量。
    angular_bins: int = 16
    # 极坐标态势表征的输出特征数。
    representation_output_dim: int = 160
    # 开始施加同伴间距惩罚的距离阈值，单位公里。
    separation_distance_km: float = 30.0
    # 开始施加额外危险接近惩罚的距离阈值，单位公里。
    danger_distance_km: float = 15.0
    # 按航点距离改善程度计算的进度奖励权重。
    progress_reward_weight: float = 0.2
    # 到达一个航点时获得的奖励。
    arrival_reward: float = 0.5
    # 超出负责圈时的距离惩罚权重。
    circle_penalty: float = 1.0
    # 同伴间距小于间距阈值时的惩罚权重。
    separation_penalty: float = 0.5
    # 同伴间距小于危险阈值时的额外惩罚权重。
    danger_penalty: float = 2.0

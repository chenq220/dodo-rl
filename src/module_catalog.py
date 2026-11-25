"""
核心模块目录。

该文件只用于人工查阅和展示框架地基类，不参与运行期注册和依赖注入；
入口、注册表、配置对象仍以各模块自己的实现为准。
"""
# 核心模块
# 角色
from roles.train import TrainRole
from roles.serial_train import SerialTrainRole
from roles.control import ControlRole
from roles.stage_control import StageControlRole
from roles.model import ModelRole
from roles.sample import SampleRole
from roles.learn import LearnRole
from roles.learn_control import LearnControlRole
from roles.evaluation import EvaluationRole
from roles.eval_control import EvalControlRole
from roles.random_sampling import RandomSamplingRole
from roles.sample_learning_evaluation import SampleLearningEvaluationRole
from roles.serial_stage_train import SerialStageTrainRole

# 调度与滚动流程
from scheduler.rollout_sample import RolloutSampleWorker
from scheduler.rollout_learn import RolloutLearnWorker

# 环境：
from envs.base import ParallelEnvironment
from envs.base import AECEnvironment
from envs.state.base import Entity, EnvState, State
from envs.action.base import ActionTable

# 仿真：

# 中间件：

# 算法：
from algos.learners import Learner
from algos.agents import Agent

# 样本：
from sample.replay_buffer import ReplayBuffer
from sample.sample_transfer import EpisodeTransition
from sample.monte_carlo import MonteCarloMethod

# 调度
from scheduler.train_task import TrainTaskManager
from scheduler.stage_task import StageTask
from scheduler.automl import AutoML
from scheduler.sample_learn_task import SampleLearnManager

# 模型
from models.model_manager import ModelManager

# 核心数据类
# 算法地图
from algos.map import AlogsMap
# agent模型维度定义
from algos.agents.input_shape import AgentInputShape

# 样本数据结构
from sample.model import EpisodeBatch
from sample.model import ReplayBufferState
# 学习和采样令牌
from scheduler.model import EpisodeInfo
from scheduler.model import LearnInfo
# 动作定义
from envs.action.model import Action
# 模型类型定义
from models.model import ModelArtifactInfo
from models.model import ModelInfo

__all__ = [
    Learner, Agent, AgentInputShape, TrainRole, SerialTrainRole, EvaluationRole, EvalControlRole, RandomSamplingRole, SampleLearningEvaluationRole,
    ControlRole, StageControlRole, SampleRole, ModelRole, LearnRole, LearnControlRole, SerialStageTrainRole,
    RolloutSampleWorker, RolloutLearnWorker, AECEnvironment, AlogsMap, AutoML,
    ParallelEnvironment, State, Entity, EnvState, ActionTable, Action,
    EpisodeBatch, ReplayBufferState, EpisodeTransition,
    ReplayBuffer, MonteCarloMethod, EpisodeInfo, LearnInfo, TrainTaskManager, StageTask, SampleLearnManager,
    ModelArtifactInfo, ModelInfo, ModelManager
]

for name in __all__:
    print(name.__name__, name.__doc__)

"""SDK 可扩展模块基线。

用户的 ``[plugins_config.module]`` 插件应继承本模块导出的某个基线类；
框架会在插件激活时自动识别最近的 ``@extensible`` 父类并建立覆盖关系。
"""
from models.model_manager import ModelManager
from scheduler.rollout_learn import RolloutLearnWorker
from scheduler.rollout_sample import RolloutSampleWorker
from sample.monte_carlo import MonteCarloMethod
from sample.replay_buffer import ReplayBuffer
from scheduler.automl import AutoML
from scheduler.sample_learn_task import SampleLearnManager

from scheduler.stage_task import StageTask
from scheduler.train_task import TrainTaskManager
from scheduler.tasks.learn_permit_policy import (
    LearnPermitGrant,
    LearnPermitPolicy,
    LearnPermitRequest,
    ReplayReadyLearnPermitPolicy,
)

__all__ = [
    "AutoML",
    "LearnPermitGrant",
    "LearnPermitPolicy",
    "LearnPermitRequest",
    "ModelManager",
    "MonteCarloMethod",
    "ReplayReadyLearnPermitPolicy",
    "ReplayBuffer",
    "RolloutLearnWorker",
    "RolloutSampleWorker",
    "SampleLearnManager",
    "StageTask",
    "TrainTaskManager",
]

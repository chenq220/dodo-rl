from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimConfig


import math
import os
from typing import Optional

from algos.modules.agents.random_agent import RandomAgent
from common.creator import create_agent, create_env
from algos.agents import resolve_agent_groups
from common.logger import init_logger
from config import (
    AgentConfig,
    AgentType,
    EnvConfig,
    MonteCarloAgentConfig,
    MonteCarloMethodConfig,
    ReplayBufferConfig,
    TrainConfig,
)
from sample.monte_carlo import MonteCarloMethod
from sample.replay_buffer import ReplayBuffer
from scheduler.artifact import ArtifactsManager
from scheduler.model import AgentCompose
from scheduler.rollout_sample import RolloutSampleWorker
from roles.common.config_alignment import assert_state_shape_matches_env
from roles.common.simulation_logging import log_simulation_middleware
from roles.common.artifact_wiring import (
    resolve_artifacts_manager,
    bind_local_artifact_callers,
    attach_artifact_producer,
)
from scheduler.train_task import TrainTaskManager

LOG = init_logger("random_sampling")


class RandomSamplingRole:
    """
    随机采样角色，使用固定随机策略与环境交互并把轨迹写入本地或 S3 协议存储。

    该角色不训练网络，核心用途是为行为克隆类算法准备离线样本；
    运行时强制使用 RandomAgent，避免误把训练 Agent 的探索策略混入随机采样数据。
    """

    def __init__(
            self,
            monte_carlo_method_config: MonteCarloMethodConfig,
            agent_config: AgentConfig,
            env_config: EnvConfig,
            sim_config: SimConfig,
            replay_buffer_config: Optional[ReplayBufferConfig] = None,
            artifacts_manager: Optional[ArtifactsManager] = None,
    ):
        # 蒙特卡洛方法配置，控制目标样本数、临时目录、存储协议和上传间隔。
        self.config = monte_carlo_method_config
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
        # Agent 构造配置；初始化时会被改写为 monte_carlo 随机策略专用配置。
        self.agent_config = agent_config
        # 环境配置，决定场景、智能体清单、动作表、最大步数和奖励公式。
        self.env_config = env_config
        # 仿真连接配置，决定使用的 Simulation 和 MessageMiddleWare。
        self.sim_config = sim_config

        # 随机采样必须使用不共享神经网络、不拼接 agent id/last action 的空策略 Agent。
        self.agent_config.agent_type = AgentType.MONTE_CARLO.value
        self.agent_config.algo_config = MonteCarloAgentConfig()
        self.agent_config.obs_agent_id = False
        assert_state_shape_matches_env(env_config, [self.agent_config])
        max_episode = self._estimate_max_episode()
        # RandomSamplingRole 复用 TrainConfig 驱动 RolloutSampleWorker；不配置Learner，因此不会生成学习令牌。
        self.rollout_config = TrainConfig()
        self.rollout_config.max_episode = max_episode

        # 采样过程只需要临时保存 episode，容量按预计局数裁剪，避免随机采样占用过大内存。
        self.replay_buffer_config = replay_buffer_config or ReplayBufferConfig()
        self.replay_buffer_config.batch_size = 1

        # 任务调度器为随机采样分发 episode；不会产生学习令牌。
        self.task_manager = TrainTaskManager(
            config=self.rollout_config,
            policy_agent_configs=[self.agent_config],
            learner_configs=[],
            artifacts_manager=self.artifacts_manager,
        )
        # RandomSamplingRole 的回放池只保存最近采样 episode，真实持久化由 MonteCarloMethod 完成。
        self.replay_buffer = ReplayBuffer(config=self.replay_buffer_config)
        self.replay_buffer.set_agent_capacity(self.agent_config.name, 1)
        # 固定随机 Agent，输出动作表中的合法 Action 实体。
        self.agent = create_agent(
            agent_type=RandomAgent.sig,
            agent_config=self.agent_config,
            agent_net_source="blanco",
            agent_list=[
                agent_id
                for group in resolve_agent_groups(self.agent_config)
                for agent_id in group
            ],
            device=self.agent_config.device,
        )
        # 随机采样环境，默认连接真实仿真。
        self.env = create_env(
            scene_name=self.env_config.sense_name,
            config=self.env_config,
            sim_config=self.sim_config,
            connect_sim=True,
        )
        # 蒙特卡洛方法，通过 Environment.record_caller 接收每步记录。
        self.monte_carlo_method = MonteCarloMethod(task_id=self.task_manager.task_id, config=self.config)
        self.env.record_caller = self.monte_carlo_method.receiver
        register, put, complete_for = bind_local_artifact_callers(
            self.artifacts_manager,
            lambda: self.task_manager.task_id,
            lambda: self.task_manager.stage_name,
        )
        attach_artifact_producer(
            self.replay_buffer, "replay_buffer", register, put, complete_for,
            self.artifacts_manager,
            lambda: self.task_manager.task_id,
            lambda: self.task_manager.stage_name,
        )
        attach_artifact_producer(
            self.monte_carlo_method, "monte_carlo", register, put, complete_for,
            self.artifacts_manager,
            lambda: self.task_manager.task_id,
            lambda: self.task_manager.stage_name,
        )
        # 复用通用 rollout 流程，把随机动作和环境状态组织为完整 episode。
        self.rollout_worker = RolloutSampleWorker(
            agents=[self.agent],
            agent_compose=AgentCompose(compose_type='order', step_agent=[self.agent]),
            env=self.env,
            task_manager=self.task_manager,
            replay_buffer=self.replay_buffer,
        )
        LOG.info("RandomSamplingRole initialized modules.")

    def run(self):
        LOG.info(
            f"Random sampling start, sample_size={self.config.sample_size}, "
            f"storage_type={self.config.storage_type}"
        )
        log_simulation_middleware(self.env, LOG)
        try:
            while self.monte_carlo_method.current_sample_count < self.config.sample_size:
                if self.task_manager.is_sample_completed():
                    break
                ep_trans = self.rollout_worker.rollout_one_episode(evaluate=False, random_sample=False)
                if ep_trans is None:
                    break
        finally:
            if self.config.storage_type == "s3":
                self.monte_carlo_method.upload_to_oss()
            self.monte_carlo_method.close()
            self._log_sample_cache_file()
        LOG.info(
            f"Random sampling finished, samples={self.monte_carlo_method.current_sample_count}, "
            f"task_id={self.task_manager.task_id}"
        )

    def _estimate_max_episode(self) -> int:
        max_step = max(1, self.env_config.max_step_per_episode)
        return max(1, math.ceil(self.config.sample_size / max_step))

    def _log_sample_cache_file(self):
        local_dir = os.path.dirname(self.monte_carlo_method.local_file)
        file_name = os.path.basename(self.monte_carlo_method.local_file)
        if self.config.storage_type == "s3":
            LOG.info(
                f"Random sampling sample cache saved, storage_type=s3, "
                f"local_dir={local_dir}, file_name={file_name}, "
                f"s3_bucket={self.config.storage.bucket_name}, s3_key={self.monte_carlo_method.s3_key}"
            )
            return
        LOG.info(
            f"Random sampling sample cache saved, storage_type=local, "
            f"local_dir={local_dir}, file_name={file_name}, path={self.monte_carlo_method.local_file}"
        )

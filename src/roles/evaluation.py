from __future__ import annotations

from common.simulation import close_sim_manager_on_init_error, create_sim_manager

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimConfig, SimManagerConfig
    from dodo_sim.sim_manager import SimManager

import os.path
from typing import List, Optional, Union

from config import EvalConfig, EnvConfig, AgentConfig, ReplayBufferConfig, MonteCarloMethodConfig
from scheduler.train_task import TrainTaskManager
from common.creator import create_agent, create_env
from common.exception import SampleTaskReturned
from algos.agents import resolve_agent_groups
from common.logger import init_logger
from common.types import AgentName
from envs.base import BaseEnvironment
from sample.monte_carlo import MonteCarloMethod
from sample.replay_buffer import ReplayBuffer
from scheduler.artifact import ArtifactsManager
from scheduler.model import AgentCompose
from scheduler.rollout_sample import RolloutSampleWorker
from roles.common.config_alignment import assert_state_shape_matches_env
from roles.common.simulation_logging import log_simulation_middleware
from roles.common.eval_reports import EvalEpisodeReportMixin
from roles.common.artifact_wiring import (
    resolve_artifacts_manager,
    bind_local_artifact_callers,
    attach_artifact_producer,
    complete_role_artifact_producers,
    put_role_render_gif,
)
from utils.module_utils import inject_env_dims

LOG = init_logger("evaluation")


class EvaluationRole(EvalEpisodeReportMixin):
    """
    评估角色入口，负责加载已训练 Agent 网络，按评估配置驱动环境交互。

    支持单组与多组（异构）Agent 编排：传入单个 AgentConfig 时按单组评估；传入 AgentConfig 列表时，
    每组按 name 各自从文件加载模型，并以并行编排驱动同一个环境（例如 KAZ 的 commander/archer/knight）。

    评估过程不创建 Learner、不执行参数更新；ReplayBuffer 只用于保存评估 episode 结果，
    可选接入 MonteCarloMethod 把评估轨迹写入离线样本。
    单局报告从本进程 TaskManager 抽取，入参与 EvalControlRole 相同。
    """

    @close_sim_manager_on_init_error
    def __init__(self, eval_config: EvalConfig,
                 agent_config: Union[AgentConfig, List[AgentConfig]], env_config: EnvConfig,
                 sim_config: Optional[SimConfig] = None, replay_buffer_config: Optional[ReplayBufferConfig] = None,
                 artifacts_manager: Optional[ArtifactsManager] = None,
                 sim_manager_config: Optional[SimManagerConfig] = None):
        # 统一成 AgentConfig 列表，单组评估等价于只有一组的多组评估。
        agent_configs = list(agent_config) if isinstance(agent_config, (list, tuple)) else [agent_config]
        assert agent_configs, "EvaluationRole requires at least one AgentConfig."
        self._assert_unique_agent_names(agent_configs)
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
        # 评估总配置，包含模型路径、评估局数、确定性动作和样本录制开关。
        self.eval_config = eval_config
        # 各组 Agent 构造配置，必须与训练时的动作表、观测维度和网络结构保持一致。
        self.agent_configs = agent_configs
        # 环境构造配置，定义场景名称、智能体列表、最大步数和奖励公式等业务参数。
        self.env_config = env_config
        # 仿真连接配置；EnvConfig.use_sim=False 的本地环境不需要该配置。
        self.sim_config = sim_config
        if (env_config.use_sim and sim_config is not None and sim_config.config_source == "manager"
                and (sim_manager_config is None or not sim_manager_config.use_sim)):
            raise ValueError("manager simulation requires an injected SimManagerConfig with use_sim=True")
        # Evaluation metadata must be present both on the environment and on
        # the episode task.  The rollout worker reapplies task metadata before
        # every reset; leaving it only on EnvConfig makes it easy for a later
        # empty task override to hide render settings.
        merged_env_metadata = BaseEnvironment._deep_merge_metadata(
            env_config.metadata or {}, eval_config.env_metadata or {},
        )
        env_config.metadata = merged_env_metadata
        self.eval_config.env_metadata = BaseEnvironment._deep_merge_metadata(
            eval_config.env_metadata or {},
            {key: value for key, value in merged_env_metadata.items()
             if key in ("render_mode", "render_every_step")},
        )
        assert_state_shape_matches_env(env_config, agent_configs)
        # 每组 Agent 的模型文件绝对路径，按 AgentConfig.name 解析。
        model_file_paths = self._resolve_model_file_paths(eval_config, agent_configs)
        # 评估回放池配置；评估模式强制 batch_size=1，容量按评估局数裁剪。
        self.replay_buffer_config = replay_buffer_config or ReplayBufferConfig()
        self.replay_buffer_config.batch_size = 1
        # 评估任务调度器，只分发评估 episode，不生成学习次数；按所有组下发动作参数。
        self.task_manager = TrainTaskManager(
            config=eval_config,
            policy_agent_configs=agent_configs,
            learner_configs=[],
            artifacts_manager=self.artifacts_manager,
        )
        # 保存评估 episode 的回放池，主要服务统计、检查和可选导出。
        self.replay_buffer = ReplayBuffer(config=self.replay_buffer_config)
        for agent in agent_configs:
            self.replay_buffer.set_agent_capacity(agent.name, max(1, eval_config.max_episode))
        # 可选蒙特卡洛方法；只有 EvalConfig.record_sample=True 时才初始化。
        self.monte_carlo_method: Optional[MonteCarloMethod] = None
        # 从文件加载的各组策略 Agent，评估时不再由 Learner 同步权重。
        self.agents = [
            create_agent(
                agent_type=cfg.agent_type,
                agent_config=cfg,
                agent_list=[
                    agent_id
                    for group in resolve_agent_groups(cfg)
                    for agent_id in group
                ],
                agent_net_source=("file" if cfg.name in model_file_paths else "blanco"),
                model_file_path=model_file_paths.get(cfg.name),
                device=cfg.device,
            )
            for cfg in agent_configs
        ]
        # 兼容旧接口：单组评估时保留 self.agent 引用。
        self.agent = self.agents[0]
        # 与串行训练一致：manager 仿真从本进程资源池申请连接参数。
        self.sim_manager = (
            create_sim_manager(config=sim_manager_config)
            if env_config.use_sim and sim_config is not None and sim_config.config_source == "manager"
            else None
        )
        # 评估环境；本地游戏类环境(use_sim=False)不连接仿真。
        self.env = create_env(
            scene_name=env_config.sense_name,
            config=env_config,
            sim_config=sim_config,
            connect_sim=env_config.use_sim,
            sim_manager=self.sim_manager,
            role_id="eval",
        )
        LOG.info(
            "EvaluationRole environment render settings: "
            f"render_mode={getattr(self.env, 'render_mode', None)!r}, "
            f"render_every_step={getattr(self.env, 'render_every_step', False)}"
        )
        inject_env_dims(self.env, self.agents)
        self._setup_monte_carlo_method()
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
        if self.monte_carlo_method is not None:
            attach_artifact_producer(
                self.monte_carlo_method, "monte_carlo", register, put, complete_for,
                self.artifacts_manager,
                lambda: self.task_manager.task_id,
                lambda: self.task_manager.stage_name,
            )
        # 复用采样执行器完成评估交互；evaluate=True 时策略内部可关闭探索。
        # 多组异构 Agent 全部作为 step_agent，固定编排的默认顺序按传入的 agent_configs 顺序，
        # 上游组（如 commander）应排在前面，其动作经 multi_agent_action_buffer 影响下游组观测。
        self.rollout_worker = RolloutSampleWorker(
            policy_agents=self.agents,
            agent_compose=AgentCompose(
                compose_type=eval_config.agent_compose_type,
                step_agent=self.agents,
            ),
            env=self.env,
            task_manager=self.task_manager,
            replay_buffer=self.replay_buffer,
        )
        # 局末评估 GIF 直送本进程产物台账（包四后 TM 不再代收 GIF）。
        self.rollout_worker.render_gif_put_caller = self._put_render_gif
        LOG.info("EvaluationRole initialized modules.")

    def _put_render_gif(
            self,
            task_id_value: str,
            stage: str,
            path: str,
            metadata: dict | None = None,
    ) -> None:
        """局末 GIF 直送本进程产物台账。"""
        put_role_render_gif(self.artifacts_manager, task_id_value, stage, path, metadata)

    @staticmethod
    def _assert_unique_agent_names(agent_configs: List[AgentConfig]) -> None:
        names = [cfg.name for cfg in agent_configs]
        assert len(set(names)) == len(names), f"EvaluationRole AgentConfig.name must be unique, given={names}"

    @staticmethod
    def _resolve_model_file_paths(eval_config: EvalConfig,
                                  agent_configs: List[AgentConfig]) -> dict[AgentName, str]:
        """
        解析每组 Agent 的模型文件绝对路径。

        优先使用 EvalConfig.agent_model_files（按 name 指定，多组评估必用）；
        单组评估且未提供该映射时，回退到 EvalConfig.agent_model_file。
        """
        model_configs = [
            cfg for cfg in agent_configs
            if getattr(cfg.algo_config, "use_net", True)
        ]
        model_files = dict(eval_config.agent_model_files or {})
        if not model_files:
            assert len(model_configs) == 1, (
                "多组评估需要通过 EvalConfig.agent_model_files "
                "为每个网络型 AgentConfig.name 指定模型文件。"
            )
            assert eval_config.agent_model_file, (
                "单组评估必须提供 EvalConfig.agent_model_file。"
            )
            model_files = {model_configs[0].name: eval_config.agent_model_file}

        resolved: dict[AgentName, str] = {}
        for cfg in model_configs:
            assert cfg.name in model_files, (
                f"EvalConfig.agent_model_files 缺少智能体 '{cfg.name}' 的模型文件；"
                f"当前可用配置={sorted(model_files)}"
            )
            file_path = os.path.join(eval_config.agent_model_path, model_files[cfg.name])
            assert os.path.exists(file_path), f"未找到智能体 '{cfg.name}' 的评估模型文件：{file_path}"
            resolved[cfg.name] = file_path
        return resolved

    def _setup_monte_carlo_method(self):
        """按需把评估过程录制到MonteCarloMethod支持的后端。"""
        if not self.eval_config.record_sample:
            return

        monte_carlo_method_config = self.eval_config.monte_carlo_method_config
        if monte_carlo_method_config is None:
            monte_carlo_method_config = MonteCarloMethodConfig(
                tmp_local_dir=self.eval_config.output_dir,
                sample_size=max(1, self.eval_config.max_episode * self.env_config.max_step_per_episode),
                storage_type='local',
            )
        self.monte_carlo_method = MonteCarloMethod(
            task_id=self.task_manager.task_id,
            config=monte_carlo_method_config,
        )
        self.env.record_caller = self.monte_carlo_method.receiver

    def run(self):
        """ 开始执行；逐局评估，结束后从本进程 TaskManager 抽记录写报告。 """
        LOG.info("EvaluationRole started.")
        self.episode_results: list[dict] = []
        log_simulation_middleware(self.env, LOG)
        try:
            while True:
                if self.task_manager.is_sample_completed():
                    break
                try:
                    ep_trans = self.rollout_worker.rollout_one_episode(
                        evaluate=True,
                        random_sample=False,
                    )
                except SampleTaskReturned as exc:
                    # Rollout has already returned the invalid token and, for
                    # simulation failures, blocked until the simulator passed
                    # its reset health check. Retry the same pending evaluation
                    # episode without discarding completed results or closing
                    # the recovered environment.
                    LOG.warning(
                        f"EvaluationRole sample task returned, retry current "
                        f"episode: {exc}"
                    )
                    continue
                if ep_trans is None:
                    break
        finally:
            try:
                if self.monte_carlo_method is not None:
                    if self.monte_carlo_method.config.storage_type == "s3":
                        self.monte_carlo_method.upload_to_oss()
                    self.monte_carlo_method.close()
            finally:
                try:
                    self.env.close()
                finally:
                    if self.sim_manager is not None:
                        self.sim_manager.close()
        self._collect_eval_episode_results()
        self._log_summary()
        report_outputs = self.finished_report()
        self.task_manager.put_report_outputs(report_outputs)
        complete_role_artifact_producers(self, "replay_buffer", "monte_carlo_method")
        self.task_manager.notify_task_complete()
        LOG.info(f"EvaluationRole finished: process run completed, task_id={self.task_manager.task_id}.")

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimConfig


import time
from typing import Optional, TYPE_CHECKING

from common.exception import (
    EpisodeInterruptedException,
    ModelVersionNotFound,
    RpcRemoteApplicationException,
    RpcServerGone,
    RpcWorldFailure,
    SampleTaskReturned,
)
from common.logger import init_logger
from envs.base import BaseEnvironment
from config import AgentConfig, EnvConfig, EvalConfig, GrpcConfig, SampleType, TorchRpcConfig, TrainConfig
from sample.model import is_older_sample_version
from scheduler.rollout_sample import RolloutSampleWorker
from rpc.interfaces import MonteCarloMethodPort, SimManagerPort
from rpc.dispatch import (
    ARTIFACTS_MANAGER,
    MODEL_MANAGER,
    MONTE_CARLO_METHOD,
    REPLAY_BUFFER,
    SIM_MANAGER,
    TRAIN_TASK_MANAGER,
)
from rpc.factory import (
    RpcConfiguration,
    create_rpc_client,
    init_rpc_client_runtime,
    shutdown_rpc_client_runtime,
)
from roles.common.config_alignment import assert_state_shape_matches_env
from roles.common.simulation_logging import log_simulation_middleware
from roles.common.artifact_wiring import (
    complete_role_gifs,
    complete_role_logs,
    declare_role_gif_producer,
    declare_role_log_producer,
    put_role_render_gif,
)
from utils.task_id import task_id
from utils.random_seed import apply_train_seed
from utils.module_utils import (
    align_agent_configs,
    build_agent_compose,
    build_env,
    build_policy_agents,
    prepare_env_agent_names,
    sync_agent_from_model_manager,
)

if TYPE_CHECKING:
    from models import ModelManager
    from sample.replay_buffer import ReplayBuffer
    from dodo_sim.sim_manager import SimManager
    from scheduler.train_task import TrainTaskManager

LOG = init_logger("sample")


class SampleRole:
    """采样进程角色：持有Agent/Environment/RolloutSampleWorker，通过RPC消费任务和写入ReplayBuffer。"""

    def __init__(
            self,
            role_id: str,
            train_config: TrainConfig | EvalConfig,
            env_config: EnvConfig,
            sim_config: Optional[SimConfig],
            step_agents_config: list[AgentConfig],
            learners_config,
            task_manager_address: str,
            replay_buffer_address: str,
            model_manager_address: str,
            sim_manager_address: str | None = None,
            monte_carlo_method_address: str | None = None,
            sim_manager: SimManagerPort | None = None,
            rpc_config: RpcConfiguration | None = None,
    ):
        if (
                isinstance(train_config, EvalConfig)
                and train_config.sample_type != SampleType.ONLINE.value
        ):
            raise ValueError(
                "EvalConfig.sample_type must be online before SampleRole initialization, "
                f"got={train_config.sample_type!r}"
            )
        self.role_id = role_id
        self.train_config = train_config
        apply_train_seed(
            train_config,
            role_id=role_id,
            context=f"SampleRole:{role_id}",
        )
        env_config.metadata = BaseEnvironment._deep_merge_metadata(
            env_config.metadata or {}, train_config.env_metadata,
        )
        rpc_config = rpc_config or GrpcConfig(
            task_manager_address=task_manager_address,
            replay_buffer_address=replay_buffer_address,
            model_manager_address=model_manager_address,
            sim_manager_address=sim_manager_address or "",
            monte_carlo_method_address=monte_carlo_method_address or "",
        )
        self.rpc_config = rpc_config
        init_rpc_client_runtime(rpc_config, role_id)
        self.task_manager: TrainTaskManager = create_rpc_client(rpc_config, TRAIN_TASK_MANAGER)
        self.replay_buffer: ReplayBuffer = create_rpc_client(rpc_config, REPLAY_BUFFER)
        self.model_manager: ModelManager = create_rpc_client(rpc_config, MODEL_MANAGER)
        self.artifacts_manager = create_rpc_client(rpc_config, ARTIFACTS_MANAGER)
        self._artifact_stage: str | None = None
        self._completed_stages: set[str] = set()
        self.monte_carlo_method: MonteCarloMethodPort | None = None
        if isinstance(rpc_config, TorchRpcConfig) or monte_carlo_method_address:
            self.monte_carlo_method = create_rpc_client(rpc_config, MONTE_CARLO_METHOD)
        self._synced_model_versions: dict[str, str] = {}
        # 没有 Learner 的固定 Agent 使用 AgentConfig.model_file_path，不参与模型池同步。
        self._trainable_agent_names = {learner_config.name for learner_config in learners_config}
        self.sim_manager: SimManager | None = sim_manager or (
            create_rpc_client(rpc_config, SIM_MANAGER)
            if env_config.use_sim and (isinstance(rpc_config, TorchRpcConfig) or sim_manager_address) else None
        )

        prepare_env_agent_names(env_config, step_agents_config)
        align_agent_configs(step_agents_config, learners_config)
        assert_state_shape_matches_env(env_config, step_agents_config, learners_config)
        self.policy_agents = build_policy_agents(step_agents_config)
        self.agent_compose = build_agent_compose(train_config, self.policy_agents)
        self.env = build_env(
            env_config,
            sim_config,
            self.policy_agents,
            connect_sim=train_config.sample_type != SampleType.OFFLINE.value,
            sim_manager=self.sim_manager,
            role_id=self.role_id,
        )
        if train_config.record_sample and train_config.sample_type == SampleType.ONLINE.value:
            assert self.monte_carlo_method is not None, "record_sample=True requires monte_carlo_method_address"
            self.env.record_caller = self.monte_carlo_method.receiver
        self.rollout_worker = RolloutSampleWorker(
            policy_agents=self.policy_agents,
            agent_compose=self.agent_compose,
            env=self.env,
            task_manager=self.task_manager,
            replay_buffer=self.replay_buffer,
            monte_carlo_method=self.monte_carlo_method,
            render_output_dir=train_config.render_video_dir,
        )
        # 局末 GIF 直送中心产物台账（本地或 RPC 同名 put_bytes），不再由 TM 代收。
        self.rollout_worker.render_gif_put_caller = self._put_render_gif
        # 采样令牌决定开局时加载指定版本还是当前 champion。
        self.rollout_worker.pre_episode_sync_caller = self._sync_episode_models
        sim = getattr(self.env, "sim", None)
        middleware = getattr(getattr(sim, "config", sim_config), "middleware", None)
        LOG.info(f"SampleRole initialized modules: role_id={self.role_id}")
        LOG.info("  RolloutSampleWorker")
        LOG.info(f"  AgentCompose(type={self.agent_compose.compose_type})")
        LOG.info(f"  Agents({[agent.name for agent in self.policy_agents]})")
        LOG.info(f"  Environment({type(self.env).__name__})")
        LOG.info(f"  Simulation({getattr(sim_config, 'sim_type', None) if env_config.use_sim else 'disabled'})")
        LOG.info(
            f"  MessageMiddleWare(remote={getattr(middleware, 'remote_ip', None)}:"
            f"{getattr(middleware, 'remote_port', None)}, "
            f"local={getattr(middleware, 'local_ip', None)}:{getattr(middleware, 'local_port', None)})"
        )
        LOG.info(f"  MonteCarloMethod({'enabled' if self.monte_carlo_method is not None else 'disabled'})")
        LOG.info(f"  TrainTaskManagerRpcClient({task_manager_address})")
        LOG.info(f"  ReplayBufferRpcClient({replay_buffer_address})")
        LOG.info(f"  ModelManagerRpcClient({model_manager_address})")
        if self.monte_carlo_method is not None:
            LOG.info(f"  MonteCarloMethodRpcClient({monte_carlo_method_address})")
        if self.sim_manager is not None:
            LOG.info(f"  SimManager({'rpc' if sim_manager_address else 'local'})")

    def run(self) -> None:
        LOG.info(f"SampleRole {self.role_id} started.")
        try:
            while True:
                try:
                    self._sync_stage_artifacts()
                    if not getattr(self, "_simulation_middleware_logged", False):
                        self._simulation_middleware_logged = log_simulation_middleware(self.env, LOG)
                    episode_transitions = self.rollout_worker.rollout_one_episode(
                        evaluate=False,
                        random_sample=False,
                        sample_role_name=self.role_id,
                    )
                    if episode_transitions is None:
                        # 当前阶段采样完成或阶段切换中暂无令牌；先上报本阶段产物再等待。
                        if self.task_manager.is_sample_completed():
                            self._submit_current_stage_artifacts()
                        LOG.info(
                            f"SampleRole {self.role_id} no sample task now (stage may be switching), "
                            f"wait and retry."
                        )
                        time.sleep(1.0)
                        continue
                except RpcServerGone as exc:
                    self._submit_current_stage_artifacts()
                    LOG.info(f"SampleRole {self.role_id} finished: central RPC server gone, task_id={task_id}. {exc}")
                    return
                except RpcWorldFailure as exc:
                    LOG.critical(
                        f"SampleRole {self.role_id} aborting failed Torch RPC world: "
                        f"task_id={task_id}, detail={exc}"
                    )
                    raise
                except SampleTaskReturned as exc:
                    LOG.warning(f"SampleRole {self.role_id} sample task returned, continue next loop: {exc}")
                    continue
                except Exception as exc:
                    LOG.exception(
                        f"SampleRole {self.role_id} aborting unexpected sample error: {exc}"
                    )
                    raise
        finally:
            try:
                self._submit_current_stage_artifacts()
            except Exception:
                LOG.exception("SampleRole artifact ingest failed during close")
            try:
                self.env.close()
            finally:
                shutdown_rpc_client_runtime(self.rpc_config)

    def _task_stage(self) -> tuple[str, str] | None:
        try:
            info = self.task_manager.get_task_manager_info()
        except Exception:
            return None
        task = str(info.get("task_id") or "")
        stage = str(info.get("stage_name") or "")
        if not task or not stage:
            return None
        return task, stage

    def _put_render_gif(
            self,
            task_id_value: str,
            stage: str,
            path: str,
            metadata: dict | None = None,
    ) -> None:
        """局末 GIF 直送中心台账；超限或输送失败在 helper / worker 内兜底。"""
        put_role_render_gif(self.artifacts_manager, task_id_value, stage, path, metadata)

    def _declare_stage_producers(self, task_id_value: str, stage: str) -> None:
        declare_role_log_producer(self.artifacts_manager, task_id_value, stage, self.role_id)
        declare_role_gif_producer(self.artifacts_manager, task_id_value, stage, self.role_id)

    def _sync_stage_artifacts(self) -> None:
        current = self._task_stage()
        if current is None:
            return
        task_id_value, stage = current
        if self._artifact_stage == stage:
            return
        if self._artifact_stage is not None:
            self._submit_stage_artifacts(task_id_value, self._artifact_stage)
        self._declare_stage_producers(task_id_value, stage)
        self._artifact_stage = stage

    def _submit_stage_artifacts(self, task_id_value: str, stage: str) -> None:
        if stage in self._completed_stages:
            return
        complete_role_logs(self.artifacts_manager, task_id_value, stage, self.role_id, remote=True)
        complete_role_gifs(self.artifacts_manager, task_id_value, stage, self.role_id)
        self._completed_stages.add(stage)

    def _submit_current_stage_artifacts(self) -> None:
        current = self._task_stage()
        if current is None:
            if self._artifact_stage is None:
                return
            task_id_value = str(task_id)
            stage = self._artifact_stage
        else:
            task_id_value, stage = current
            if self._artifact_stage is None:
                self._declare_stage_producers(task_id_value, stage)
                self._artifact_stage = stage
            elif self._artifact_stage != stage:
                self._submit_stage_artifacts(task_id_value, self._artifact_stage)
                self._declare_stage_producers(task_id_value, stage)
                self._artifact_stage = stage
        self._submit_stage_artifacts(task_id_value, self._artifact_stage or stage)

    def _sync_episode_models(self, episode_info) -> None:
        required = getattr(episode_info, "required_model_versions", {})
        if not required:
            self._sync_agents_if_available()
            return

        if not required.keys() <= {agent.name for agent in self.policy_agents}:
            raise ValueError(f"unknown policy in sample version request: {required}")
        self._sync_agents_if_available(exclude=required)
        for agent in self.policy_agents:
            version = required.get(agent.name)
            if version is None:
                continue
            waiting_since = None
            while self.task_manager.accept_sample_task_result(episode_info.episode_num, episode_info.lease_id):
                if self._synced_model_versions.get(agent.name) == version:
                    break
                try:
                    sync_agent_from_model_manager(agent, self.model_manager, version=version)
                except (ModelVersionNotFound, RpcRemoteApplicationException) as exc:
                    if isinstance(exc, RpcRemoteApplicationException) and (
                            exc.remote_error_type != ModelVersionNotFound.__name__):
                        raise
                    versions = self.task_manager.get_task_manager_info().get("sample_model_versions", {})
                    if is_older_sample_version(version, versions.get(agent.name)):
                        raise EpisodeInterruptedException(detail=f"sample model version was pruned: {version}") from exc
                    if waiting_since is None:
                        waiting_since = time.monotonic()
                        LOG.info(f"Waiting for sample model: episode={episode_info.episode_num}, "
                                 f"agent={agent.name}, version={version}")
                    time.sleep(0.2)
                    continue
                self._synced_model_versions[agent.name] = version
                wait_seconds = 0.0 if waiting_since is None else time.monotonic() - waiting_since
                LOG.info(f"SampleRole {self.role_id} synced required model: episode={episode_info.episode_num}, "
                         f"agent={agent.name}, version={version}, wait_seconds={wait_seconds:.3f}")
                break
            else:
                raise EpisodeInterruptedException(detail=f"sample model lease expired: {episode_info.lease_id}")

    def _sync_agents_if_available(self, exclude=()) -> None:
        """模型池首版模型可能稍晚于采样进程启动，拉不到时等待下一轮。"""
        try:
            for agent in self.policy_agents:
                if agent.name in exclude or agent.name not in self._trainable_agent_names:
                    continue
                champion = self.model_manager.get_champion(agent.name)
                version = champion.get("version") if isinstance(champion, dict) else getattr(champion, "version", None)
                if version is None or self._synced_model_versions.get(agent.name) == version:
                    continue
                sync_agent_from_model_manager(agent, self.model_manager, version=version)
                self._synced_model_versions[agent.name] = version
                LOG.info(f"SampleRole {self.role_id} synced model: agent={agent.name}, version={version}")
        except (RpcServerGone, RpcWorldFailure):
            raise
        except Exception as exc:
            LOG.warning(f"SampleRole {self.role_id} sync model skipped: {exc}")

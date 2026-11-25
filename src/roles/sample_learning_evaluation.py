from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimConfig


import os
import copy
import boto3
from typing import Any, Dict, Optional

from common.creator import create_agent, create_env, create_learner
from algos.agents import resolve_agent_groups
from common.exception import VarTypeErrorException
from scheduler.artifact import ArtifactType, ArtifactsManager, ensure_output_parent
from common.logger import init_logger
from roles.common.config_alignment import (
    align_agent_config_with_learner_config,
    assert_state_shape_matches_env,
    inherit_replay_buffer_seed,
)
from roles.common.artifact_wiring import (
    resolve_artifacts_manager,
    bind_local_artifact_callers,
    attach_artifact_producer,
    complete_artifact_producers,
    put_role_render_gif,
)
from config import (
    AgentConfig,
    EnvConfig,
    EvalConfig,
    LearnerConfig,
    LocalStorageConfig,
    MonteCarloMethodConfig,
    ReplayBufferConfig,
    S3StorageConfig,
    SampleLearningEvalConfig,
    SampleLearningTasksConfig,
    TrainConfig,
)
from sample.monte_carlo import MonteCarloMethod
from sample.replay_buffer import ReplayBuffer
from scheduler.model import AgentCompose
from scheduler.rollout_learn import RolloutLearnWorker
from scheduler.rollout_sample import RolloutSampleWorker
from roles.common.simulation_logging import log_simulation_middleware
from scheduler.sample_learn_task import SampleLearnManager
from scheduler.train_task import TrainTaskManager
from utils.grahp_drawer import save_reward_curve

LOG = init_logger("sample_learning_evaluation")


class SampleLearningEvaluationRole:
    """
    样本学习评估角色，先用离线样本训练空模型，再切到评估模式跑仿真。

    该角色不加载已有模型文件；Learner 和 Agent 都从 blanco 网络开始，样本学习完成后
    Agent 克隆 Learner.agent_net，再复用 RolloutSampleWorker 以 evaluate=True 执行若干局评估。
    """

    def __init__(
            self,
            config: SampleLearningEvalConfig,
            agent_config: AgentConfig,
            env_config: EnvConfig,
            sim_config: SimConfig,
            monte_carlo_method_config: MonteCarloMethodConfig,
            train_config: TrainConfig,
            learner_config: LearnerConfig,
            eval_config: EvalConfig,
            sample_learning_config: SampleLearningTasksConfig,
            replay_buffer_config: Optional[ReplayBufferConfig] = None,
            artifacts_manager: Optional[ArtifactsManager] = None,
    ):
        # 角色级配置，描述样本学习评估角色自身的输出、模型保存等策略。
        self.config = config
        self.artifacts_manager = resolve_artifacts_manager(artifacts_manager)
        # 入口已组织好的训练流程配置；本角色只消费任务调度相关字段，不承载算法参数。
        self.train_config = train_config
        # 入口已组织好的Learner算法配置；一个Learner实例只绑定一个LearnerConfig。
        self.learner_config = learner_config
        # 入口已组织好的评估配置；用于创建评估episode任务序列。
        self.eval_config = eval_config
        # 入口已组织好的离线样本学习任务配置。
        self.sample_learning_config = sample_learning_config
        # Agent/Env/Sim配置必须与采样样本对应的场景保持一致。
        self.agent_config = agent_config
        self.env_config = env_config
        self.sim_config = sim_config
        self.monte_carlo_method_config = monte_carlo_method_config
        align_agent_config_with_learner_config(self.agent_config, self.learner_config)
        assert_state_shape_matches_env(env_config, [self.agent_config], [self.learner_config])

        # 样本学习ReplayBuffer，保存从历史文件恢复出的EpisodeTransition。
        self.sample_replay_buffer_config = replay_buffer_config or ReplayBufferConfig()
        inherit_replay_buffer_seed(self.sample_replay_buffer_config, train_config)
        self.sample_replay_buffer = ReplayBuffer(config=self.sample_replay_buffer_config)
        self.sample_replay_buffer.set_agent_capacity(
            self.learner_config.name, self.learner_config.algo_config.capacity,
        )
        # 评估ReplayBuffer只保存评估局结果，不参与学习令牌生成。
        self.eval_replay_buffer_config = ReplayBufferConfig(
            batch_size=1,
            strategy=self.sample_replay_buffer_config.strategy,
            recent_ratio=self.sample_replay_buffer_config.recent_ratio,
            seed=self.sample_replay_buffer_config.seed,
        )
        self.eval_replay_buffer = ReplayBuffer(config=self.eval_replay_buffer_config)
        self.eval_replay_buffer.set_agent_capacity(
            self.agent_config.name, max(1, self.eval_config.max_episode),
        )

        # 空模型Agent，样本学习完成后会通过reset_net克隆Learner策略网络。
        self.agent = create_agent(
            agent_type=agent_config.agent_type,
            agent_config=agent_config,
            agent_net_source="blanco",
            device=agent_config.device,
            agent_list=[
                agent_id
                for group in resolve_agent_groups(agent_config)
                for agent_id in group
            ],
        )
        # 连接真实仿真的环境；离线样本恢复也复用该环境的fake_reset/fake_step向量化逻辑。
        self.env = create_env(
            scene_name=env_config.sense_name,
            config=env_config,
            sim_config=sim_config,
            connect_sim=True,
        )

        # 样本学习调度器和学习任务执行器。
        self.learn_task_manager = TrainTaskManager(
            config=self.train_config,
            policy_agent_configs=[self.agent_config],
            learner_configs=[self.learner_config],
            artifacts_manager=self.artifacts_manager,
        )
        # 历史样本读取方法。
        assert self.monte_carlo_method_config.load_sample_name is not None, \
            VarTypeErrorException(
                name="monte_carlo_method_config.load_sample_name",
                given_var=self.monte_carlo_method_config.load_sample_name,
            )
        read_config = copy.deepcopy(self.monte_carlo_method_config)
        read_config.write_enabled = False
        self.sample_monte_carlo_method = MonteCarloMethod(
            task_id=self.learn_task_manager.task_id,
            config=read_config,
        )
        self.sample_learn_manager = SampleLearnManager(
            config=self.sample_learning_config,
            task_id=self.learn_task_manager.task_id,
        )
        self.learner = create_learner(
            learner_config=self.learner_config,
            agent=self.agent,
            env=self.env,
            device=self.learner_config.device,
            net_source="blanco",
        )
        self.sample_rollout_worker = RolloutSampleWorker(
            policy_agents=[self.agent],
            agent_compose=AgentCompose(compose_type='order', step_agent=[self.agent]),
            env=self.env,
            task_manager=self.learn_task_manager,
            replay_buffer=self.sample_replay_buffer,
        )
        self.learn_rollout_worker = RolloutLearnWorker(
            learners=[self.learner],
            train_config=self.train_config,
            replay_buffer=self.sample_replay_buffer,
            task_manager=self.learn_task_manager,
            sample_learn_manager=self.sample_learn_manager,
        )
        self.learn_task_manager.register_learner_save_caller(self._learner_save_caller)

        # 评估调度器和评估采样执行器，与样本学习进度完全隔离。
        self.eval_task_manager = TrainTaskManager(
            config=self.eval_config,
            policy_agent_configs=[self.agent_config],
            learner_configs=[],
            artifacts_manager=self.artifacts_manager,
        )
        # 评估写入器独立于历史样本读取器，不能复用其输入文件路径。
        self.eval_monte_carlo_method: Optional[MonteCarloMethod] = None
        if self.eval_config.record_sample:
            record_config = copy.deepcopy(self.eval_config.monte_carlo_method_config)
            if record_config is None:
                record_config = MonteCarloMethodConfig(
                    tmp_local_dir=self.eval_config.output_dir,
                    sample_size=max(1, self.eval_config.max_episode * (self.env_config.max_step_per_episode + 1)),
                    storage_type="local",
                )
            record_config.write_enabled = True
            record_config.load_sample_name = None
            record_config.load_file_path = None
            self.eval_monte_carlo_method = MonteCarloMethod(
                task_id=self.eval_task_manager.task_id,
                config=record_config,
            )
            self.env.record_caller = self.eval_monte_carlo_method.receiver
        register, put, complete_for = bind_local_artifact_callers(
            self.artifacts_manager,
            lambda: self.learn_task_manager.task_id,
            lambda: self.learn_task_manager.stage_name,
        )
        attach_artifact_producer(
            self.sample_replay_buffer, "sample_replay_buffer", register, put, complete_for,
            self.artifacts_manager,
            lambda: self.learn_task_manager.task_id,
            lambda: self.learn_task_manager.stage_name,
        )
        attach_artifact_producer(
            self.eval_replay_buffer, "eval_replay_buffer", register, put, complete_for,
            self.artifacts_manager,
            lambda: self.learn_task_manager.task_id,
            lambda: self.learn_task_manager.stage_name,
        )
        attach_artifact_producer(
            self.sample_monte_carlo_method, "monte_carlo", register, put, complete_for,
            self.artifacts_manager,
            lambda: self.learn_task_manager.task_id,
            lambda: self.learn_task_manager.stage_name,
        )
        if self.eval_monte_carlo_method is not None:
            attach_artifact_producer(
                self.eval_monte_carlo_method, "eval_monte_carlo", register, put, complete_for,
                self.artifacts_manager,
                lambda: self.learn_task_manager.task_id,
                lambda: self.learn_task_manager.stage_name,
            )
        attach_artifact_producer(
            self.learner, f"learner:{self.learner.name}", register, put, complete_for,
            self.artifacts_manager,
            lambda: self.learn_task_manager.task_id,
            lambda: self.learn_task_manager.stage_name,
        )
        self.eval_rollout_worker = RolloutSampleWorker(
            policy_agents=[self.agent],
            agent_compose=AgentCompose(compose_type='order', step_agent=[self.agent]),
            env=self.env,
            task_manager=self.eval_task_manager,
            replay_buffer=self.eval_replay_buffer,
        )
        # 局末 GIF 直送本进程产物台账（包四后 TM 不再代收 GIF）；两个 worker 按各自 TM 归属。
        self.sample_rollout_worker.render_gif_put_caller = self._put_render_gif
        self.eval_rollout_worker.render_gif_put_caller = self._put_render_gif
        LOG.info("SampleLearningEvaluationRole initialized modules.")

    def _put_render_gif(
            self,
            task_id_value: str,
            stage: str,
            path: str,
            metadata: dict | None = None,
    ) -> None:
        """局末 GIF 直送本进程产物台账；caller 由 worker 携带所属 task/stage。"""
        put_role_render_gif(self.artifacts_manager, task_id_value, stage, path, metadata)

    def sample_learning(self) -> list[Dict[str, Any]]:
        """加载离线样本并执行样本学习。"""
        requester = f"sample_learning_eval:{id(self)}"
        should_load_cache = self.sample_learn_manager.request_sample_cache_load(requester=requester)
        loaded_episodes = 0
        if not should_load_cache:
            loaded_episodes = self.sample_learn_manager.wait_sample_cache_loaded(requester=requester)

        LOG.info(
            f"Sample learning evaluation start, algo={self.learner_config.algo_type}, "
            f"storage_type={self.sample_learning_config.load_storage_type}, "
            f"load_sample_name={self.sample_learning_config.load_sample_name}, "
            f"max_sample_episodes={self.sample_learning_config.max_sample_episodes}, "
            f"learn_times={self.sample_learning_config.learn_times}"
        )
        if should_load_cache:
            try:
                loaded_episodes = self.sample_monte_carlo_method.load_replay_buffer(
                    self.sample_rollout_worker,
                    max_episodes=self.sample_learning_config.max_sample_episodes,
                )
                self.sample_learn_manager.finish_sample_cache_load(loaded_episodes)
            except Exception as e:
                self.sample_learn_manager.fail_sample_cache_load(e)
                raise

        if loaded_episodes == 0:
            LOG.warning("Sample learning evaluation skipped: no episode loaded from recorded samples.")
            return []
        if self.sample_replay_buffer.size < self.sample_replay_buffer.config.batch_size:
            LOG.warning(
                f"Sample learning evaluation skipped: replay buffer size={self.sample_replay_buffer.size}, "
                f"batch_size={self.sample_replay_buffer.config.batch_size}"
            )
            return []

        self.sample_replay_buffer.reset_sliding_idx()
        learn_results = self.learn_rollout_worker.rollout_after_offline_sample_updated()
        self.agent.reset_net(self.learner.agent_net)
        LOG.info(
            f"Sample learning evaluation learned, loaded_episodes={loaded_episodes}, "
            f"learn_times={self.sample_learn_manager.learning_completed}"
        )
        return learn_results

    def evaluate(self) -> None:
        """使用样本学习后的Agent执行评估。"""
        log_simulation_middleware(self.env, LOG)
        try:
            while True:
                if self.eval_task_manager.is_sample_completed():
                    break
                ep_trans = self.eval_rollout_worker.rollout_one_episode(evaluate=True, random_sample=False)
                if ep_trans is None:
                    break
                self.eval_ep_end_log(ep_trans)
        finally:
            if self.eval_monte_carlo_method is not None:
                try:
                    if self.eval_monte_carlo_method.config.storage_type == "s3":
                        self.eval_monte_carlo_method.upload_to_oss()
                finally:
                    self.env.record_caller = None
                    self.eval_monte_carlo_method.close()
        self._draw_rewards_png()

    def eval_ep_end_log(self, ep_trans):
        """打印单局评估奖励总结；ep_trans 为 dict[AgentName, list[Transition]]。"""
        total_reward = sum(
            transition.episode_reward()
            for transitions in ep_trans.values()
            for transition in transitions
        )
        episode_num = self.eval_task_manager.sample_task_completed
        LOG.info(f'{"=" * 10} Sample Eval {episode_num} END! REWARD={total_reward} {"=" * 10}')

    def run(self) -> list[Dict[str, Any]]:
        """执行样本学习，然后进入评估模式。"""
        LOG.info("SampleLearningEvaluationRole started.")
        try:
            learn_results = self.sample_learning()
            self.evaluate()
        finally:
            self.sample_monte_carlo_method.close()
            if self.eval_monte_carlo_method is not None:
                self.env.record_caller = None
                self.eval_monte_carlo_method.close()
        complete_artifact_producers(
            self.sample_replay_buffer,
            self.eval_replay_buffer,
            self.sample_monte_carlo_method,
            self.eval_monte_carlo_method,
            self.learner,
        )
        if self.config.save_model:
            self.learn_task_manager.notify_stage_complete()
        LOG.info(f"SampleLearningEvaluationRole finished: process run completed, task_id={self.learn_task_manager.task_id}.")
        return learn_results

    def _draw_rewards_png(self):
        """保存评估奖励曲线。"""
        eval_episodes = self.eval_replay_buffer.ordered_buffer()
        if not eval_episodes:
            return
        episode_rewards = [ep_trans.episode_reward() for ep_trans in eval_episodes]
        image_name = (
            f"sample_eval_reward_{self.eval_task_manager.start_timestamp}_"
            f"{self.eval_task_manager.task_id}.png"
        )
        image_path = os.path.join(self.config.result_save_dir, image_name)
        ensure_output_parent(image_path)
        with open(image_path, "w+b") as image_file:
            save_reward_curve(episode_rewards, image_file, image_name=image_name)
        LOG.info(f"Save sample evaluation reward curve: {image_path}")
        self.eval_task_manager.put_runtime_file(ArtifactType.RESULT, image_path)

    def _learner_save_caller(self, name_prefix: str) -> list[str]:
        if not self.config.save_model:
            return []
        model_dir = self._model_local_dir()
        filenames = {
            model_name: self.learn_task_manager.resolve_model_filename(
                name_prefix, self.learner.name, model_name,
            )
            for model_name in self.learner.models
        }
        saved_files = self.learner.save_models(model_dir, filenames)
        LOG.info(f"Sample learning evaluation models saved locally: files={saved_files}")
        if self.config.model_storage_type == "s3":
            self._upload_models_to_s3(saved_files)
        return saved_files

    def _model_local_dir(self) -> str:
        """返回模型本地保存目录。"""
        if isinstance(self.config.model_storage, LocalStorageConfig):
            model_dir = self.config.model_storage.data_path
        else:
            model_dir = self.config.model_local_dir
        return model_dir

    def _upload_models_to_s3(self, saved_files: list[str]) -> None:
        """把本地模型文件上传到S3协议模型池。"""
        assert self.config.model_storage is not None, \
            VarTypeErrorException(name="self.config.model_storage", given_var=self.config.model_storage)
        storage = self.config.model_storage
        assert isinstance(storage, S3StorageConfig), \
            VarTypeErrorException(name="self.config.model_storage", given_var=storage)
        client = boto3.client(
            "s3",
            endpoint_url=storage.endpoint_url,
            aws_access_key_id=storage.access_key,
            aws_secret_access_key=storage.secret_key,
        )
        try:
            client.head_bucket(Bucket=storage.bucket_name)
            LOG.info(f"s3 model bucket exists: bucket={storage.bucket_name}")
        except client.exceptions.ClientError:
            client.create_bucket(Bucket=storage.bucket_name)
            LOG.info(f"s3 model bucket created: bucket={storage.bucket_name}")

        for file_path in saved_files:
            object_key = os.path.basename(file_path)
            client.upload_file(
                Filename=file_path,
                Bucket=storage.bucket_name,
                Key=object_key,
            )
            LOG.info(
                f"s3 model uploaded: bucket={storage.bucket_name}, "
                f"key={object_key}, local_file={file_path}"
            )

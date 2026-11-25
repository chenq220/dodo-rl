from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING, TypeVar
from prettytable import PrettyTable
from algos.learners import algo_config_to_updates
from common.logger import init_logger
from common.registry import extensible
from rpc.interfaces import ReplayBufferPort, TrainTaskManagerPort
from scheduler.model import LearnInfo
from sample.model import SampledEpisodes
from scheduler.tasks.training_progress import publish_worker_token
from utils.random_seed import derive_seed, set_global_seed

if TYPE_CHECKING:
    from algos.learners import Learner
    from config import TrainConfig
    from scheduler.sample_learn_task import SampleLearnManager
    from scheduler.train_task import TrainTaskManager
    from sample.replay_buffer import ReplayBuffer
    LearnerType = TypeVar("LearnerType", bound=Learner)

LOG = init_logger("learn_worker")


class _OnlineLearnSession:
    """在线学习上下文：上文领取令牌，下文回写学习结果。"""

    def __init__(self):
        self.learn_info: Optional[LearnInfo] = None
        self.learn_result: Optional[Dict[str, Any]] = None


@extensible
class RolloutLearnWorker(object):
    """
    学习滚动流程执行器。

    在线主训练只消费 TrainTaskManager 下发的 LearnInfo 学习令牌；
    离线样本学习只消费 SampleLearnManager 的样本学习任务。两条路径共享 Learner.update
    执行逻辑，但任务来源、完成计数和结果回写目标保持隔离。
    """

    def __init__(
            self,
            learners: List[LearnerType],
            train_config: TrainConfig,
            replay_buffer: ReplayBuffer,
            task_manager: TrainTaskManager,
            sample_learn_manager: Optional[SampleLearnManager] = None,
    ):
        self.learners = learners
        assert self.learners, "RolloutLearnWorker requires at least one learner."
        self.learners_by_name = {learner.name: learner for learner in self.learners}
        assert len(self.learners_by_name) == len(self.learners), "Learner.name must be unique."

        self.train_config = train_config
        # 构造参数使用 Port 兼容本地/RPC；实例属性标成本地实现，方便 IDE 跳转业务逻辑。
        self.replay_buffer: ReplayBuffer = replay_buffer
        self.task_manager: TrainTaskManager = task_manager
        # 在线串行训练不需要样本学习管理器；离线学习入口会在使用前强校验。
        self.sample_learn_manager: Optional[SampleLearnManager] = sample_learn_manager
        self._online_no_task_logged = False
        self._online_stage_name: Optional[str] = None
        # 最近一次在线学习令牌，供模型发布关联采样进度；无任务时清空。
        self.last_learn_info: Optional[LearnInfo] = None

    def rollout_after_episode_updated(self) -> List[Dict[str, Any]]:
        """在线训练：消费一个在线学习批次令牌，并返回本批次学习结果。"""
        # 在线入口一次最多消费一个LearnInfo；学习进程外层循环由进程自己控制。
        self._set_train_mode()
        self.last_learn_info = None
        learning_results: List[Dict[str, Any]] = []

        with self.online_learn_task_context() as session:
            if session.learn_info is None:
                self.post_episode(learning_results=learning_results)
                return learning_results
            self.last_learn_info = session.learn_info
            session.learn_result = self._update_by_learn_info(session.learn_info)
            learning_results.append(session.learn_result)

        self.post_episode(learning_results=learning_results)
        return learning_results

    def rollout_after_offline_sample_updated(self) -> List[Dict[str, Any]]:
        """离线样本学习：在历史样本加载到ReplayBuffer后执行学习流程，并返回本轮新增结果。"""
        assert self.sample_learn_manager is not None, (
            "Offline learning requires SampleLearnManager."
        )
        # 离线入口只消费SampleLearnManager任务，不触碰TrainTaskManager学习进度。
        self._set_train_mode()
        learning_results: List[Dict[str, Any]] = []
        while True:
            with self.offline_learn_task_context() as learn_info:
                if learn_info is None:
                    if self.sample_learn_manager.is_learning_completed():
                        break
                    time.sleep(0.2)
                    continue
                # 离线样本已在ReplayBuffer中，所有Learner按自身算法配置从缓存池抽样更新。
                learn_result = self._update_all_learners(
                    sample_strategy=self.sample_learn_manager.config.sample_policy,
                    episode_num=learn_info.episode_num,
                )
                self.sample_learn_manager.push_learning_state(learn_result)
                learning_results.append(learn_result)
        self.post_episode(learning_results=learning_results)
        return learning_results

    @contextmanager
    def online_learn_task_context(self) -> Generator[_OnlineLearnSession, None, None]:
        """
        在线学习上下文。

        上文从 TrainTaskManager 领取 LearnInfo；没有可执行令牌时 session.learn_info 为 None。
        下文在 finally 中回写学习结果，保证 report_learning_result 与上下文边界一致。
        """
        session = _OnlineLearnSession()
        learn_task_result = self.task_manager.get_learn_task_info()
        if learn_task_result.task_finished:
            yield session
            return
        if learn_task_result.learn_info is None:
            if not self._online_no_task_logged:
                LOG.info("Waiting for next learn task from TrainTaskManager.")
                self._online_no_task_logged = True
            yield session
            return
        session.learn_info = learn_task_result.learn_info
        # 进度信息推送给环境变量，让外部shell获取
        publish_worker_token(
            kind="learn",
            value=(
                f"learn_id={session.learn_info.learn_id} "
                f"sample_watermark={session.learn_info.episode_num} "
                f"stage={session.learn_info.stage_name} "
                f"learners={','.join(session.learn_info.active_learners)} "
                f"generation={session.learn_info.generation}"
            ),
        )
        # 分阶段完整拓扑复用同一个远端 Learner；令牌是阶段切换的权威信号。
        if session.learn_info.stage_name != self._online_stage_name:
            for learner in self.learners:
                if getattr(learner, "env", None) is not None:
                    learner.env.stage_name = session.learn_info.stage_name
                # 新阶段的 generation 从 0 重新计数，并以跨阶段继承后的当前网络作为
                # 本阶段重启基线，避免前一阶段较大的 generation 抑制恢复逻辑。
                learner.reset_generation()
                learner.clone_models_snapshot()
            self._online_stage_name = session.learn_info.stage_name
        set_global_seed(
            session.learn_info.seed,
            context=f"RolloutLearnWorker:episode:{session.learn_info.episode_num}",
        )
        if not session.learn_info.active_learners:
            LOG.info(
                f"Sample episode {session.learn_info.episode_num} completed, "
                f"no learner is ready for this episode."
            )
        else:
            self._online_no_task_logged = False
        try:
            yield session
        finally:
            if (
                    session.learn_info is not None
                    and session.learn_result is not None
                    and session.learn_info.episode_num > 0
            ):
                self.task_manager.report_learning_result(
                    session.learn_result,
                    episode_num=session.learn_info.episode_num,
                    learn_id=session.learn_info.learn_id,
                )
                if self.sample_learn_manager is not None:
                    self.sample_learn_manager.push_learning_result(session.learn_result)
            publish_worker_token(kind="learn", value="")

    @contextmanager
    def offline_learn_task_context(self) -> Generator[Optional[LearnInfo], None, None]:
        """
        离线样本学习上下文。

        上文从 SampleLearnManager 领取样本学习令牌；没有可执行令牌时yield None。
        """
        assert self.sample_learn_manager is not None, (
            "Offline learning requires SampleLearnManager."
        )
        # 离线令牌来自样本学习管理器；它不代表在线训练episode，也不更新TrainTaskManager。
        learn_task_result = self.sample_learn_manager.should_learn()
        if learn_task_result.task_finished:
            yield None
            return
        if learn_task_result.learn_info is None:
            yield None
            return
        learn_info = learn_task_result.learn_info
        publish_worker_token(
            kind="learn",
            value=(
                f"episode={learn_info.episode_num} "
                f"stage={learn_info.stage_name} "
                f"learners={','.join(learn_info.active_learners)} "
                f"generation={learn_info.generation}"
            ),
        )
        try:
            yield learn_info
        finally:
            publish_worker_token(kind="learn", value="")

    def _update_by_learn_info(self, learn_info: LearnInfo) -> Dict[str, Any]:
        """按 TrainTaskManager 下发的 LearnInfo 更新指定 Learner。"""
        learn_results: Dict[str, Any] = {}
        for learner_name in learn_info.active_learners:
            # 在线训练必须严格执行令牌里的算法配置，不从Learner本地配置读取兜底超参。
            learner = self.learners_by_name[learner_name]
            algo_config = learn_info.algo_configs[learner_name]
            learn_results[learner_name] = self._update_one_learner_by_token(
                learner=learner,
                token_algo_config=algo_config,
                episode_num=learn_info.episode_num,
                learn_seed=learn_info.seed,
                generation=learn_info.generation,
            )
        return learn_results

    def _update_all_learners(
            self,
            sample_strategy: Optional[str] = None,
            episode_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        """离线样本学习使用：所有 Learner 各自按算法配置更新一次。"""
        learn_results: Dict[str, Any] = {}
        for learner in self.learners:
            learn_results[learner.name] = self._update_one_learner_by_config(
                learner=learner,
                algo_config=learner.config.algo_config,
                sample_strategy=sample_strategy,
                episode_num=episode_num,
            )
        return learn_results

    def _update_one_learner_by_token(
            self,
            learner: LearnerType,
            token_algo_config,
            episode_num: Optional[int] = None,
            learn_seed: Optional[int] = None,
            generation: int = 0,
    ) -> List[Dict[str, Any]]:
        """在线主训练：严格按LearnInfo令牌中的算法超参执行一个完整学习批次。"""
        # 动态超参修改集中放在Learner接口中，避免RolloutLearnWorker了解各算法内部字段。
        learner.dynamic_modify_config(algo_config_to_updates(token_algo_config))
        learn_times = token_algo_config.learn_times_per_episode
        learn_results = []
        learning_rows = []
        for i in range(learn_times):
            time_s = time.time()
            learn_result = self._update_one_learner(
                learner=learner,
                sample_strategy=token_algo_config.sample_strategy,
                batch_size=token_algo_config.batch_size,
                episode_num=episode_num,
                sample_seed=derive_seed(learn_seed, learner.name, i),
                generation=generation,
            )
            if learn_result.get("skip_update"):
                LOG.info(
                    f"Learner {learner.name} skipped update: "
                    f"reason={learn_result.get('skip_reason')}, "
                    f"sample_strategy={token_algo_config.sample_strategy}, "
                    f"batch_size={token_algo_config.batch_size}, "
                    f"batch={i + 1}/{learn_times}, episode_num={episode_num}"
                )
                break
            if not learn_result:
                LOG.info(
                    f"Learner {learner.name} skipped update: "
                    f"sample_strategy={token_algo_config.sample_strategy}, "
                    f"batch_size={token_algo_config.batch_size}, episode_num={episode_num}"
                )
                break
            learning_rows.append({
                "batch": f"{i + 1}/{learn_times}",
                "time_cost": time.time() - time_s,
                "result": learn_result,
            })
            learn_results.append(learn_result)
        self._log_learning_batch_table(
            learner=learner,
            episode_num=episode_num,
            sample_strategy=token_algo_config.sample_strategy,
            batch_size=token_algo_config.batch_size,
            learning_rows=learning_rows,
        )
        return learn_results

    def _update_one_learner_by_config(
            self,
            learner: LearnerType,
            algo_config,
            sample_strategy: Optional[str] = None,
            episode_num: Optional[int] = None,
    ) -> Dict[str, Any]:
        """离线样本学习：按Learner算法配置执行，可由SampleLearningTasksConfig覆盖采样策略。"""
        # 离线样本学习允许SampleLearningTasksConfig指定采样策略，例如滑动窗口读取历史样本。
        strategy = sample_strategy or algo_config.sample_strategy
        batch_size = getattr(algo_config, "batch_size")
        return self._update_one_learner(
            learner=learner,
            sample_strategy=strategy,
            batch_size=batch_size,
            episode_num=episode_num,
        )

    def _update_one_learner(
            self,
            learner: LearnerType,
            sample_strategy,
            batch_size,
            episode_num: Optional[int] = None,
            sample_seed: Optional[int] = None,
            generation: int = 0,
    ) -> Dict[str, Any]:
        """执行一次 Learner.update。调用方必须显式给出采样策略和批大小。"""
        # 这里是学习执行的最小公共单元：采样一个batch，然后调用算法自己的update。
        # 非 RNN Learner 的 schema 不声明 burn_in，采样侧直接跳过历史前缀物化。
        include_burn_in = (
            learner.batch_schema is None
            or learner.batch_schema.uses_source("burn_in")
        )
        sample_kwargs = {}
        sequence_length = getattr(learner.config.algo_config, "n_step", 1)
        if sequence_length > 1:
            sample_kwargs["sequence_length"] = sequence_length
        batch_trans = self.replay_buffer.sample(
            agent_name=learner.name,
            strategy=sample_strategy,
            batch_size=batch_size,
            seed=sample_seed,
            include_burn_in=include_burn_in,
            **sample_kwargs,
        )
        if not batch_trans:
            LOG.info("ReplayBuffer return is empty sample data, skip update.")
            return {}
        set_global_seed(
            sample_seed,
            context=(
                f"RolloutLearnWorker:update:{learner.name}:"
                f"episode:{episode_num}:generation:{generation}"
            ),
        )
        sampled_episodes = (batch_trans if isinstance(batch_trans, SampledEpisodes)
                            else SampledEpisodes(batch_trans, sample_strategy=sample_strategy))
        learn_result = learner.learn_update(sampled_episodes, generation=generation)
        if learn_result == {}:
            # 有些算法的底层logger按间隔吐指标，网络已经更新但本批次没有统计项。
            # 空dict不能向上传递成“跳过学习”，否则TaskManager学习完成游标会落后。
            learn_result = {"update_cnt": getattr(learner, "update_cnt", 0)}
        if not learn_result.get("skip_update"):
            reward = self._mean_episode_reward(batch_trans)
            if reward is not None:
                # ModelManager 的 optimo 自动晋升以该字段作为默认奖励指标；算法已有同名指标时不覆盖。
                learn_result.setdefault("reward", reward)
        return learn_result

    @staticmethod
    def _mean_episode_reward(batch_trans) -> Optional[float]:
        """提取一个学习 batch 内各 episode 的平均回报，作为模型提交的通用 reward 指标。"""
        rewards = []
        for transition in batch_trans:
            episode_reward = getattr(transition, "episode_reward", None)
            if not callable(episode_reward):
                continue
            try:
                rewards.append(float(episode_reward()))
            except (TypeError, ValueError):
                continue
        return sum(rewards) / len(rewards) if rewards else None

    def _set_train_mode(self) -> None:
        """把所有 Learner 网络切到 train 模式。"""
        for learner in self.learners:
            for model in learner.net_list:
                model.train()

    @classmethod
    def _log_learning_batch_table(
            cls,
            learner: LearnerType,
            episode_num: Optional[int],
            sample_strategy,
            batch_size: int,
            learning_rows: List[Dict[str, Any]],
    ) -> None:
        """把一个学习令牌内的全部网络更新汇总为表格，每次Learner.update占一行。"""
        metric_names = list(dict.fromkeys(
            metric_name
            for row in learning_rows
            for metric_name in row["result"]
            if metric_name not in {"skip_update", "skip_reason"}
        ))
        if not learning_rows:
            return
        # 诊断指标保留完整浮点精度；表格只负责便于阅读的摘要。
        for row in learning_rows:
            LOG.info("Learner metrics: " + json.dumps({
                "learner": learner.name,
                "algo": learner.algo_type,
                "episode_num": episode_num,
                "generation": learner.generation,
                "batch": row["batch"],
                "sample_strategy": str(sample_strategy),
                "batch_size": batch_size,
                "time_cost": row["time_cost"],
                "metrics": row["result"],
            }, ensure_ascii=False, default=str))
        metric_names = metric_names[:12]
        table = PrettyTable()
        table.field_names = [
            "Batch",
            "Strategy",
            "Batch Size",
            "Time(s)",
            *metric_names,
        ]
        table.align = "r"
        table.align["Batch"] = "c"
        table.align["Strategy"] = "l"

        for row in learning_rows:
            learn_result = row["result"]
            table.add_row([
                row["batch"],
                str(sample_strategy),
                batch_size,
                f"{row['time_cost']:.2f}",
                *[
                    cls._format_metric_value(learn_result.get(metric_name))
                    for metric_name in metric_names
                ],
            ])

        LOG.info(
            f"\nLearner {learner.name} update net: algo={learner.algo_type}, episode_num={episode_num}, "
            f"generation={learner.generation}"
            f"\n{table}"
        )

    @staticmethod
    def _format_metric_value(value: Any) -> Any:
        """统一表格指标精度，非数值结果保持原值。"""
        if isinstance(value, float):
            return f"{value:.4f}"
        if value is None:
            return "-"
        return value

    def post_episode(self, *args, **kwargs):
        """一局完成后的动作，项目定制化接口。"""
        pass

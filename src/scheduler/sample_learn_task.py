from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from common.registry import extensible
from config import SampleLearningTasksConfig
from scheduler.model import LearnInfo, LearnTaskResult
from utils.task_id import task_id as process_task_id


@extensible
class SampleLearnManager(object):
    """
    样本学习任务控制总线。

    它独立于 TrainTaskManager，只负责历史样本或指定样本集的加载仲裁、学习任务派发、
    学习结果回收和可选模型加载信息描述。当前是本地内存实现，方法语义按后续 RPC 控制进程设计。
    """

    def __init__(self, config: SampleLearningTasksConfig, task_id: Optional[str] = None):
        # 样本学习配置，描述学习哪个样本、学习多少次、使用什么采样策略、是否预加载模型。
        self.config = config
        # 样本学习任务ID；默认跟随 DODO_TASK_ID，显式传入时保留调用方指定值。
        self.task_id = task_id or str(process_task_id)
        # 服务端状态锁；后续 RPC 化时应由中心进程保证这些状态更新的原子性。
        self._lock = threading.RLock()

        # 样本学习计划总更新次数。
        self.total_learn_count = int(self.config.learn_times)
        # 已派发给学习进程的更新任务数。
        self.dispatched_learn_count = 0
        # 已完成并回写结果的更新任务数。
        self.learning_completed = 0
        # 每次 Learner.update 返回值的原始列表，用于调试或后续指标汇总。
        self.learning_results: list[Dict[Any, Any]] = []

        # 样本缓存加载仲裁：分布式学习进程组中只允许一个进程执行样本加载。
        self.sample_cache_load_requested = False
        self.sample_cache_load_requester: Optional[str] = None
        self.sample_cache_load_completed = False
        self.sample_cache_load_failed = False
        self.sample_cache_loaded_episodes = 0
        self.sample_cache_load_error: Optional[str] = None

    def should_learn(self) -> LearnTaskResult:
        """
        申请一次样本学习任务。

        分布式实现中该方法应作为中心调度进程的原子 RPC：返回 learn_info 表示调用方拿到一次
        Learner.update 机会；返回 task_finished=True 表示样本学习生命周期已经结束。
        """
        with self._lock:
            if self.learning_completed >= self.total_learn_count:
                return LearnTaskResult(task_finished=True)
            if self.dispatched_learn_count >= self.total_learn_count:
                return LearnTaskResult()
            self.dispatched_learn_count += 1
            return LearnTaskResult(
                learn_info=LearnInfo(episode_num=self.dispatched_learn_count),
            )

    def push_learning_state(self, result: Dict[Any, Any]) -> int:
        """
        学习进程完成一次 Learner.update 后回写结果。

        :param result: Learner.update 返回的指标字典。
        :return: 样本学习已完成次数。
        """
        with self._lock:
            self._push_learning_result_unlocked(result)
            self.learning_completed += 1
            return self.learning_completed

    def push_learning_result(self, result: Dict[Any, Any]) -> int:
        """
        只记录 Learner.update 返回值，不改变样本学习任务完成计数。

        主训练也可以把学习指标统一写到 SampleLearnManager，但不应污染样本学习任务进度。
        """
        with self._lock:
            self._push_learning_result_unlocked(result)
            return len(self.learning_results)

    def _push_learning_result_unlocked(self, result: Dict[Any, Any]) -> None:
        self.learning_results.append(result)

    def learning_results_count(self) -> int:
        """返回当前已经保存的学习结果数量。"""
        with self._lock:
            return len(self.learning_results)

    def get_learning_results(self, start: int = 0) -> list[Dict[Any, Any]]:
        """返回已经回写的学习结果列表。"""
        with self._lock:
            return list(self.learning_results[start:])

    def is_learning_completed(self) -> bool:
        """检查样本学习是否已经完成全部计划更新次数。"""
        with self._lock:
            return self.learning_completed >= self.total_learn_count

    def request_sample_cache_load(self, requester: Optional[str] = None) -> bool:
        """
        申请执行样本缓存加载。

        多个学习进程同时启动时，只有第一个申请者负责把历史样本加载进共享 ReplayBuffer；
        其他进程通过 wait_sample_cache_loaded 轮询等待加载完成。
        """
        with self._lock:
            if self.sample_cache_load_requested:
                return False
            self.sample_cache_load_requested = True
            self.sample_cache_load_requester = requester
            return True

    def finish_sample_cache_load(self, loaded_episodes: int) -> None:
        """通知样本缓存加载成功。"""
        assert loaded_episodes >= 0, f"loaded episodes must be non-negative, given={loaded_episodes}"
        with self._lock:
            self.sample_cache_loaded_episodes = loaded_episodes
            self.sample_cache_load_completed = True
            self.sample_cache_load_failed = False
            self.sample_cache_load_error = None

    def fail_sample_cache_load(self, error: Exception) -> None:
        """通知样本缓存加载失败。"""
        with self._lock:
            self.sample_cache_load_completed = True
            self.sample_cache_load_failed = True
            self.sample_cache_load_error = repr(error)

    def get_sample_cache_load_state(self) -> Dict[str, Any]:
        """查询样本缓存加载状态；RPC版本应直接暴露为远程状态接口。"""
        with self._lock:
            return {
                "requested": self.sample_cache_load_requested,
                "requester": self.sample_cache_load_requester,
                "completed": self.sample_cache_load_completed,
                "failed": self.sample_cache_load_failed,
                "loaded_episodes": self.sample_cache_loaded_episodes,
                "error": self.sample_cache_load_error,
            }

    def wait_sample_cache_loaded(
            self,
            requester: Optional[str] = None,
            timeout: Optional[float] = None,
            poll_interval: float = 0.2,
    ) -> int:
        """
        轮询等待样本缓存加载完成。

        这里不使用 Condition 等本地阻断器，方便后续替换为跨进程 RPC 轮询。
        """
        assert poll_interval > 0, f"poll_interval must be positive, given={poll_interval}"
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            state = self.get_sample_cache_load_state()
            if not state["requested"]:
                return 0
            if state["completed"]:
                if state["failed"]:
                    raise RuntimeError(f"sample cache load failed: {state['error']}")
                return int(state["loaded_episodes"])
            if deadline is not None:
                wait_time = deadline - time.monotonic()
                if wait_time <= 0:
                    raise TimeoutError(
                        f"wait sample cache loaded timeout, requester={requester}, timeout={timeout}"
                    )
                time.sleep(min(poll_interval, wait_time))
            else:
                time.sleep(poll_interval)

    def init_model_info(self) -> Dict[str, Any]:
        """
        返回样本学习前需要加载的模型描述。

        当前只生成结构化描述，不实际加载模型；模型池进程接入后可直接消费该返回结构。
        """
        storage = self.config.load_model_storage.to_dict() if self.config.load_model_storage is not None else None
        return {
            "models": dict(self.config.load_model_names),
            "storage_type": self.config.load_model_storage_type,
            "storage": storage,
        }

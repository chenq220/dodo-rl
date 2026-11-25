from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol

from common.types import AgentName
from scheduler.artifact import Artifact


class TrainTaskManagerPort(Protocol):
    """训练任务总线端口；本地 TrainTaskManager 和 TrainTaskManagerRpcClient 都应满足该接口。"""

    def get_sample_task_info(self, **kwargs) -> Any:
        """领取一局采样任务令牌。"""
        ...

    def sample_backpressure_state(self, sample_role_name: str = "unknown") -> dict:
        """查询采样背压状态。"""
        ...

    def sample_gate_state(self, sample_role_name: str = "unknown") -> dict:
        """查询采样统一闸门状态（背压和版本批次）。"""
        ...

    def report_sample_result(
            self,
            episode_num: int,
            total_rewards: Dict[AgentName, float],
            success: bool = True,
            artifacts: Optional[list[Artifact]] = None,
            episode_metrics: Optional[Dict[str, float]] = None,
            episode_rewards_details: Optional[Dict[str, float]] = None,
            lease_id: str = "",
            step_count: int = 0,
    ) -> Any:
        """上报一局采样结果。"""
        ...

    def accept_sample_task_result(self, episode_num: int, lease_id: str) -> bool:
        """在写入回放池前确认采样租约仍有效。"""
        ...

    def is_sample_completed(self) -> bool:
        """采样任务是否全部完成。"""
        ...

    def get_learn_task_info(self) -> Any:
        """领取一个学习批次令牌。"""
        ...

    def report_learning_result(
            self,
            learn_result: dict,
            episode_num: int,
            learn_id: int = 0,
    ) -> Any:
        """上报学习结果。"""
        ...

    def is_learning_completed(self) -> bool:
        """学习任务是否全部完成。"""
        ...

    def get_task_manager_info(self) -> dict:
        """返回任务总线基本状态。"""
        ...

    def learn_batches_update(self, replay_buffer_state: Any, episode_num: int) -> Any:
        """ReplayBuffer 样本更新后通知任务总线释放学习令牌。"""
        ...

    def restart_training_task(self, reason: str = "") -> dict:
        """重启当前训练任务序列；资源清理由持有资源的 Role 执行。"""
        ...

    def push_info(self, attrs: dict[str, Any]) -> None:
        """合并零散信息进中心进程 info_bridge。"""
        ...

    def pull_info(self, key: str | None = None) -> Any:
        """从中心进程 info_bridge 拉取；key 为 None 返回整袋，缺键返回 None。"""
        ...


class ReplayBufferPort(Protocol):
    """回放池端口；本地 ReplayBuffer 和 ReplayBufferRpcClient 都应满足该接口。"""

    def push(
            self,
            episode_data,
            save=False,
            notify_learning: bool = True,
            agent_name=None,
            sample_role_name: str = "Serial",
            episode_num: Optional[int] = None,
            sample_cost_secs: Optional[float] = None,
            generation: Optional[int] = None,
            sample_version: str | Mapping[str, str] | None = None,
    ) -> Any:
        """写入一局或一组 episode transition。"""
        ...

    def sample(self, agent_name, batch_size: int = 0, strategy=None, seed: Optional[int] = None,
               include_burn_in: bool = True, sequence_length: int = 1) -> Any:
        """按 AgentName 抽取训练 batch；include_burn_in=False 时跳过 RNN 历史前缀。"""
        ...

    def reset_sliding_idx(self, agent_name=None) -> Any:
        """重置滑动窗口采样游标。"""
        ...

    def ordered_buffer(self, agent_name=None) -> Any:
        """返回按写入顺序排列的回放池视图。"""
        ...

    def state(self) -> Any:
        """返回回放池状态快照。"""
        ...


class ModelManagerPort(Protocol):
    """模型池端口；本地 ModelManager 和 ModelManagerRpcClient 都应满足该接口。"""

    def submit_model(
            self,
            model_name,
            algo_type: str,
            models: Mapping[str, Any],
            **kwargs,
    ) -> Any:
        """提交一个模型版本。"""
        ...

    def get_model(
            self,
            model_name,
            *,
            version=None,
            output_type: str = "state_dict",
            artifact_names=None,
    ) -> Any:
        """获取模型权重或元数据。"""
        ...

    def get_champion(self, model_name) -> Any:
        """获取当前 champion 模型信息。"""
        ...

    def save_models(
            self,
            save_model_path,
            save_model_name: Mapping[str, str],
            *,
            agent_name,
            version=None,
    ) -> list[str]:
        """按调用方文件名将模型池权重落盘。"""
        ...


class SimManagerPort(Protocol):
    """仿真资源管理端口；本地 SimManager 和 SimManagerRpcClient 都应满足该接口。"""

    def get_sim_info(self, sample_id: str, sim_type: Optional[str] = None, *, request_id=None) -> Optional[dict[str, Any]]:
        """申请指定类型的空闲仿真资源，返回包含完整 sim_config 的资源信息。"""
        ...

    def release_sim(self, sim_id: str, *, lease_id=None, generation=None) -> Any:
        """释放一个仿真资源。"""
        ...

    def reacquire_simulation(self, **request) -> dict[str, Any]: ...

    def validate_sim_lease(self, sim_id, *, lease_id, generation) -> bool: ...

    def get_sim_list(self) -> list[dict[str, Any]]:
        """查询仿真资源表。"""
        ...

    def close(self) -> None:
        """主动终止管理器并强制回收全部托管仿真资源。"""
        ...


class MonteCarloMethodPort(Protocol):
    """蒙特卡洛方法端口；本地 MonteCarloMethod 和 MonteCarloMethodRpcClient 都应满足该接口。"""

    def receiver(self, payload: dict[str, Any]) -> Any:
        """接收一条Environment.record_sample产生的原始样本记录。"""
        ...

    def state(self) -> dict[str, Any]:
        """返回中心录制器状态，便于分布式调试和收尾检查。"""
        ...

    def get_recorded_episode(self, *, stage_name: str, episode_num: int) -> dict[str, Any]:
        """按阶段和1-based编号读取中心离线索引中的轨迹。"""
        ...

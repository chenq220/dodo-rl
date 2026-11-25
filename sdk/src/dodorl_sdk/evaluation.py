"""评估角色基类和项目扩展协议。"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dodo_sim.config import SimConfig


from typing import Any, Callable, Protocol, Sequence, TypeAlias

from ._runtime import bootstrap_runtime

bootstrap_runtime()

from config import AgentConfig, EnvConfig, EvalConfig, ReplayBufferConfig
from roles.eval_control import EvalControlRole
from roles.evaluation import EvaluationRole as Evaluation
from utils.toml_config import TomlTrainingConfig

EvaluationConfigBuilder: TypeAlias = Callable[[TomlTrainingConfig], EvalConfig]


class EvaluationProtocol(Protocol):
    """自定义评估实现应保持的运行与报告接口。"""

    def __init__(
            self,
            eval_config: EvalConfig,
            agent_config: AgentConfig | Sequence[AgentConfig],
            env_config: EnvConfig,
            sim_config: SimConfig | None = None,
            replay_buffer_config: ReplayBufferConfig | None = None,
    ) -> None:
        ...

    def run(self) -> Any:
        ...

    def episode_report(
            self,
            episode_index: int,
            *,
            total_rewards: dict | None = None,
            episode_metrics: dict | None = None,
            episode_rewards_details: dict | None = None,
            artifacts: dict | None = None,
            episode_info: dict | None = None,
            **payload,
    ) -> dict:
        ...

    def finished_report(self) -> Any:
        ...


__all__ = [
    "Evaluation",
    "EvalControlRole",
    "EvaluationConfigBuilder",
    "EvaluationProtocol",
]

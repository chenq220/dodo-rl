from common.exception import EnvAlgoDimException
from config import AgentConfig, AgentOutputType, EnvConfig, LearnerConfig, ReplayBufferConfig, TrainConfig
from scheduler.model import AutoMLAdvice


def align_agent_config_with_learner_config(agent_config: AgentConfig, learner_config: LearnerConfig) -> None:
    """
    按同名 Learner 的算法配置回填 Agent 配置。

    Agent 网络在创建时就需要知道输出语义和隐藏层维度；这些字段本质上由 Learner 算法决定，
    所以入口只负责声明 Agent/Learner 的配对关系，具体对齐在角色装配阶段完成。
    """
    algo_config = learner_config.algo_config
    assert algo_config is not None, (
        f"LearnerConfig[{learner_config.name}].algo_config must be initialized before aligning AgentConfig."
    )
    agent_config.agent_output_type = AgentOutputType(algo_config.agent_output_type).value
    agent_config.hidden_dim = algo_config.hidden_dim


def assert_state_shape_matches_env(
        env_config: EnvConfig,
        agent_configs: list[AgentConfig],
        learner_configs: list[LearnerConfig] | None = None,
) -> None:
    """
    校验 state_shape 与环境/同名 Learner 的维度契约。

    旧隐式单组：Agent/Learner 的 state_shape 必须等于 EnvConfig.state_dim；
    显式分组：组状态维度不再要求等于全局 state_dim，但同名 Agent/Learner 配置
    必须一致（该值是组状态维度的配置真值，由 QMIX 构网与 Rollout 运行期校验使用）。
    """
    expected = env_config.state_dim
    learners_by_name = {
        learner_config.name: learner_config
        for learner_config in (learner_configs or [])
    }
    for agent_config in agent_configs:
        if agent_config.agent_groups:
            paired_learner = learners_by_name.get(agent_config.name)
            if paired_learner is not None and paired_learner.state_shape != agent_config.state_shape:
                raise EnvAlgoDimException(
                    config_name=(
                        f"AgentConfig[{agent_config.name}] vs "
                        f"LearnerConfig[{paired_learner.name}]"
                    ),
                    expected=agent_config.state_shape,
                    given=paired_learner.state_shape,
                )
            continue
        if agent_config.state_shape != expected:
            raise EnvAlgoDimException(
                config_name=f"AgentConfig[{agent_config.name}]",
                expected=expected,
                given=agent_config.state_shape,
            )
    for learner_config in learner_configs or []:
        paired_agent = next(
            (
                agent_config
                for agent_config in agent_configs
                if agent_config.name == learner_config.name and agent_config.agent_groups
            ),
            None,
        )
        if paired_agent is not None:
            continue
        if learner_config.state_shape != expected:
            raise EnvAlgoDimException(
                config_name=f"LearnerConfig[{learner_config.name}]",
                expected=expected,
                given=learner_config.state_shape,
            )


def inherit_replay_buffer_seed(
        replay_buffer_config: ReplayBufferConfig,
        train_config: TrainConfig,
) -> None:
    """
    ReplayBufferConfig 自身不绑定训练种子。

    Role 装配时若 seed 仍为 None，沿用当前任务 train_config.seed；显式配置的 seed 保持不变。
    """
    if replay_buffer_config.seed is None:
        replay_buffer_config.seed = train_config.seed


def learner_partition_capacities(learner_configs: list[LearnerConfig]) -> dict[str, int]:
    """由 Learner 私有算法配置形成 Replay 的运行时分区配额。"""
    capacities: dict[str, int] = {}
    for learner in learner_configs:
        if learner.name in capacities:
            raise ValueError(f"duplicate Learner capacity for {learner.name!r}")
        capacity = learner.algo_config.capacity
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"Learner {learner.name!r} capacity must be a positive integer")
        capacities[learner.name] = capacity
    return capacities


def apply_learner_capacities(replay_buffer, learners: list[LearnerConfig]) -> None:
    """Role 仅从本轮 Learner 属性更新同名 Replay 分区。"""
    for name, capacity in learner_partition_capacities(learners).items():
        replay_buffer.set_agent_capacity(name, capacity)


def apply_replay_capacity_advice(replay_buffer, advice) -> None:
    """AutoML restart 的容量建议由持有 Replay 的 Role 同步落地。"""
    if not isinstance(advice, AutoMLAdvice):
        return
    for name, overrides in advice.algo_config_overrides.items():
        if "capacity" in overrides:
            replay_buffer.set_agent_capacity(name, overrides["capacity"])

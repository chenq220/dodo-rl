"""
算法配置地图（IDE 自动联想友好）。

入口处配置 Agent/Learner 时，不用手敲 "qmix"/"rnn" 这类字符串字面量，改用本模块的
层级命名空间，靠 IDE 自动补全选出合法组合，避免拼错或张冠李戴。

层级结构（learner 与 agent 并列，各自带 config_type）：
    AlogsMap.[coma | mappo | maddpg | qmix | ppo | a2c | dqn]
        .learner                 -> 传给 LearnerConfig.algo_type 的算法类型字符串
        .learner.config_type     -> 该算法 LearnerConfig.algo_config 的具体类型
        .agent.[...]             -> 传给 AgentConfig.agent_type 的 agent 网络类型字符串
        .agent.[...].config_type -> 该 agent AgentConfig.algo_config 的具体类型

每个 ref 同时是「字符串」(可直接传给配置)和「带 config_type 的对象」，且通过泛型让
config_type 携带具体子类型，因此::

    assert isinstance(learner_config.algo_config, AlogsMap.qmix.learner.config_type)

之后 IDE 能把 learner_config.algo_config 收窄成 QmixConfig，联想出 td_lambda 等子类专有字段。

所有叶子值都直接取自 config 中的枚举(AlgorithmType / AgentType)，保持单一事实来源。

用法示例（``python -m portal.train``）::

    from algos.map import AlogsMap

    learner_config = LearnerConfig(name=agent_name, algo_type=AlogsMap.mappo.learner)
    assert isinstance(learner_config.algo_config, AlogsMap.mappo.learner.config_type)
    agent_config = AgentConfig(name=agent_name, agent_type=AlogsMap.mappo.agent.rnn)
    assert isinstance(agent_config.algo_config, AlogsMap.mappo.agent.rnn.config_type)
"""
from __future__ import annotations

from typing import Final, Generic, TypeVar
from enum import Enum
from config import (
    A2CConfig,
    AgentType,
    AlgorithmType,
    ComaConfig,
    DqnConfig,
    Ia2cConfig,
    IppoConfig,
    IqlConfig,
    Maa2cConfig,
    MaddpgAgentConfig,
    MaddpgConfig,
    MappoConfig,
    PolicyRnnAgentConfig,
    PacConfig,
    PacDcgConfig,
    PpoConfig,
    QmixConfig,
    QAgentConfig,
    RnnAgentConfig,
    RnnNsAgentConfig,
    SB3AgentConfig,
    VdnConfig,
)

ConfigT = TypeVar("ConfigT")


class _TypedRef(str, Generic[ConfigT]):
    """
    带类型信息的配置 ref。

    本质是一个普通字符串(可直接传给 LearnerConfig.algo_type / AgentConfig.agent_type)，
    额外挂上 config_type 指向对应的 *Config 具体类型；通过泛型参数 ConfigT 让 config_type
    携带具体子类型(如 type[QmixConfig])，从而 isinstance 断言后 IDE 能收窄到子类并联想专有字段。
    """
    config_type: type[ConfigT]

    def __new__(cls, value: str, config_type: type[ConfigT]) -> "_TypedRef[ConfigT]":
        obj = super().__new__(cls, value)
        obj.config_type = config_type
        return obj


class AlgosType(str, Enum):
    QLearning = "q_learning"
    OnPolicy = "on_policy"


class _EpymarlAgents:
    """EPyMARL 系算法共用的 agent 网络。"""
    # 共享参数 RNN agent：所有实体共用一套网络，通过 agent_id 区分身份。
    rnn: Final[_TypedRef[RnnAgentConfig]] = _TypedRef(AgentType.RNN.value, RnnAgentConfig)
    # 非共享参数 RNN agent：每个实体槽位各自持有一套网络。
    rnn_ns: Final[_TypedRef[RnnNsAgentConfig]] = _TypedRef(AgentType.RNN_NS.value, RnnNsAgentConfig)


class _EpymarlNonSharedAgents:
    """EPyMARL 中仅实现了非共享参数版本的算法。"""
    rnn_ns: Final[_TypedRef[RnnNsAgentConfig]] = _TypedRef(AgentType.RNN_NS.value, RnnNsAgentConfig)


class _SB3Agents:
    """SB3 单智能体 actor-critic agent 网络，PPO/A2C 共用。"""
    # SB3 ActorCriticPolicy(MLP)，单智能体执行。
    sb3: Final[_TypedRef[SB3AgentConfig]] = _TypedRef(AgentType.SB3.value, SB3AgentConfig)


class _QmixAgents(_EpymarlAgents):
    rnn: Final[_TypedRef[RnnAgentConfig]] = _EpymarlAgents.rnn
    rnn_ns: Final[_TypedRef[RnnNsAgentConfig]] = _EpymarlAgents.rnn_ns
    q_rnn: Final[_TypedRef[QAgentConfig]] = _TypedRef(AgentType.Q_RNN.value, QAgentConfig)


class _Qmix:
    """QMIX 算法：learner(带 config_type) + 兼容的 agent 网络。"""
    learner: Final[_TypedRef[QmixConfig]] = _TypedRef(AlgorithmType.QMIX.value, QmixConfig)
    agent = _QmixAgents
    algo_type = AlgosType.QLearning


class _MappoAgents:
    rnn: Final[_TypedRef[PolicyRnnAgentConfig]] = _TypedRef(AgentType.POLICY_RNN.value, PolicyRnnAgentConfig)


class _Mappo:
    """MAPPO 算法：learner(带 config_type) + 兼容的 agent 网络。"""
    learner: Final[_TypedRef[MappoConfig]] = _TypedRef(AlgorithmType.MAPPO.value, MappoConfig)
    agent = _MappoAgents
    algo_type = AlgosType.OnPolicy


class _Coma:
    learner: Final[_TypedRef[ComaConfig]] = _TypedRef(AlgorithmType.COMA.value, ComaConfig)
    agent = _MappoAgents
    algo_type = AlgosType.OnPolicy


class _Ia2c:
    learner: Final[_TypedRef[Ia2cConfig]] = _TypedRef(AlgorithmType.IA2C.value, Ia2cConfig)
    agent = _EpymarlAgents
    algo_type = AlgosType.OnPolicy


class _Ippo:
    learner: Final[_TypedRef[IppoConfig]] = _TypedRef(AlgorithmType.IPPO.value, IppoConfig)
    agent = _EpymarlAgents
    algo_type = AlgosType.OnPolicy


class _Iql:
    learner: Final[_TypedRef[IqlConfig]] = _TypedRef(AlgorithmType.IQL.value, IqlConfig)
    agent = _EpymarlAgents
    algo_type = AlgosType.QLearning


class _Maa2c:
    learner: Final[_TypedRef[Maa2cConfig]] = _TypedRef(AlgorithmType.MAA2C.value, Maa2cConfig)
    agent = _EpymarlAgents
    algo_type = AlgosType.OnPolicy


class _MaddpgAgents:
    maddpg_mlp: Final[_TypedRef[MaddpgAgentConfig]] = _TypedRef(AgentType.MADDPG_MLP.value, MaddpgAgentConfig)


class _Maddpg:
    learner: Final[_TypedRef[MaddpgConfig]] = _TypedRef(AlgorithmType.MADDPG.value, MaddpgConfig)
    agent = _MaddpgAgents
    algo_type = AlgosType.QLearning


class _Pac:
    learner: Final[_TypedRef[PacConfig]] = _TypedRef(AlgorithmType.PAC.value, PacConfig)
    agent = _EpymarlNonSharedAgents
    algo_type = AlgosType.OnPolicy


class _PacDcg:
    learner: Final[_TypedRef[PacDcgConfig]] = _TypedRef(AlgorithmType.PAC_DCG.value, PacDcgConfig)
    agent = _EpymarlNonSharedAgents
    algo_type = AlgosType.OnPolicy


class _Vdn:
    learner: Final[_TypedRef[VdnConfig]] = _TypedRef(AlgorithmType.VDN.value, VdnConfig)
    agent = _EpymarlAgents
    algo_type = AlgosType.QLearning


class _Ppo:
    """单智能体 PPO 算法：learner(带 config_type) + 兼容的 agent 网络。"""
    learner: Final[_TypedRef[PpoConfig]] = _TypedRef(AlgorithmType.PPO.value, PpoConfig)
    agent = _SB3Agents
    algo_type = AlgosType.OnPolicy


class _A2C:
    """单智能体 A2C 算法：learner(带 config_type) + SB3 actor-critic agent。"""
    learner: Final[_TypedRef[A2CConfig]] = _TypedRef(AlgorithmType.A2C.value, A2CConfig)
    agent = _SB3Agents
    algo_type = AlgosType.OnPolicy


class _Dqn:
    """单智能体 DQN 算法：learner(带 config_type) + SB3 Q-network agent。"""
    learner: Final[_TypedRef[DqnConfig]] = _TypedRef(AlgorithmType.DQN.value, DqnConfig)
    agent = _SB3Agents
    algo_type = AlgosType.QLearning


class AlogsMap:
    """ 算法知识地图 """
    coma = _Coma
    qmix = _Qmix
    mappo = _Mappo
    maddpg = _Maddpg
    ppo = _Ppo
    a2c = _A2C
    dqn = _Dqn

    @staticmethod
    def compatible_agent_types(algo_node: type) -> tuple[str, ...]:
        """返回某算法节点在知识地图中声明的全部 AgentType。"""
        return tuple(
            str(agent_ref)
            for agent_ref in vars(algo_node.agent).values()
            if isinstance(agent_ref, _TypedRef)
        )

    @classmethod
    def to_dict(cls) -> dict:
        """
        把整张算法知识地图导出为嵌套字典。

        结构::

            {
              "<algo>": {
                "learner": {
                  "algo_type": "<algo_type 字符串>",
                  "config_type": "<QmixConfig | MappoConfig | PpoConfig | A2CConfig | DqnConfig>",
                },
                "agent": {
                  "<agent 别名>": {
                    "agent_type": "<agent_type 字符串>",
                    "config_type": "<AgentAlgoConfig 子类名>",
                  },
                  ...
                },
              },
              ...
            }
        """
        result: dict = {}
        for algo_name, algo_node in vars(cls).items():
            # 只挑选算法节点(具备 learner 字段的类)，跳过 dunder 和方法。
            if algo_name.startswith("_") or not isinstance(algo_node, type):
                continue
            if not hasattr(algo_node, "learner"):
                continue

            agents: dict = {}
            for agent_alias, agent_ref in vars(algo_node.agent).items():
                if not isinstance(agent_ref, _TypedRef):
                    continue
                agents[agent_alias] = {
                    "agent_type": str(agent_ref),
                    "config_type": agent_ref.config_type.__name__,
                }
            algo_type = algo_node.algo_type.value
            result[algo_name] = {
                "learner": {
                    "algo_type": str(algo_node.learner),
                    "config_type": algo_node.learner.config_type.__name__,
                },
                "agent": agents,
                "algo_type": algo_type
            }
        return result


if __name__ == '__main__':
    print(AlogsMap.to_dict())

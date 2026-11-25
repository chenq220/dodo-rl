from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pettingzoo.butterfly import knights_archers_zombies_v10


@dataclass
class KazGameConfig:
    """PettingZoo KAZ v10 游戏参数，对应 knights_archers_zombies_v10.env() 的全部配置项。"""

    # 僵尸生成间隔（环境周期数）。数值越小生成越快；12 对 5骑士+3弓箭手是中等压力。
    spawn_rate: int = 12
    # 开局生成的弓箭手（archer）智能体数量。
    num_archers: int = 3
    # 开局生成的骑士（knight）智能体数量。
    num_knights: int = 5
    # 场上同时存在的僵尸数量上限；控制观测维度和防守压力。
    max_zombies: int = 20
    # 场上同时存在的箭矢数量上限（所有弓箭手共享）；3个弓箭手给30基本够用。
    max_arrows: int = 30
    # 骑士是否会被僵尸碰撞击杀；False 时骑士无敌。
    killable_knights: bool = True
    # 弓箭手是否会被僵尸碰撞击杀；False 时弓箭手无敌。
    killable_archers: bool = True
    # 智能体靠近屏幕边缘、局部观测填不满时，是否用黑色像素补齐观测区域。
    pad_observation: bool = True
    # 智能体触碰到屏幕上/下边界是否立即死亡；False 时越界不会因此死亡。
    line_death: bool = False
    # 单局最大环境周期数，达到后触发 truncation 结束对局；测试算法接入不需要900步长局。
    max_cycles: int = 300
    # 是否使用向量观测；True 为 (N+1)x5 向量，False 为 512x512x3 图像观测。
    vector_state: bool = True
    # 仅在 vector_state=True 时生效；为每行实体向量前追加 6 维类型掩码（zombie/archer/knight 等）。
    use_typemasks: bool = True
    # 实验性选项，仅在 vector_state=True 时生效；移除向量中不存在的实体，使观测维度不再固定。
    sequence_space: bool = False

    def to_env_kwargs(self) -> dict[str, Any]:
        return {
            "spawn_rate": self.spawn_rate,
            "num_archers": self.num_archers,
            "num_knights": self.num_knights,
            "max_zombies": self.max_zombies,
            "max_arrows": self.max_arrows,
            "killable_knights": self.killable_knights,
            "killable_archers": self.killable_archers,
            "pad_observation": self.pad_observation,
            "line_death": self.line_death,
            "max_cycles": self.max_cycles,
            "vector_state": self.vector_state,
            "use_typemasks": self.use_typemasks,
            "sequence_space": self.sequence_space,
        }


class KazGame:
    """PettingZoo KAZ v10 并行环境封装，供 Simulation 层同步步进调用。"""

    FPS = 15

    def __init__(
            self,
            config: KazGameConfig | None = None,
            render_mode: Literal["human", "rgb_array"] | None = None,
    ):
        self.render_mode = render_mode
        self.env_config = config or KazGameConfig()
        self.env = knights_archers_zombies_v10.parallel_env(
            render_mode=render_mode,
            **self.env_config.to_env_kwargs(),
        )
        self.observations: dict[str, Any] = {}
        self.rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict] = {}
        self.frame_index = 0
        self.reset()

    @property
    def agents(self) -> list[str]:
        if not hasattr(self.env, "agents"):
            return list(self.possible_agents)
        return list(self.env.agents)

    @property
    def possible_agents(self) -> list[str]:
        return list(self.env.possible_agents)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            observations, infos = self.env.reset(seed=seed)
        else:
            observations, infos = self.env.reset()
        self.observations = observations
        self.rewards = {agent: 0.0 for agent in self.possible_agents}
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}
        self.infos = infos
        self.frame_index = 0
        return self.observations

    def step(self, actions: dict[str, int]) -> None:
        if not self.agents:
            return
        sim_actions = {
            agent: int(actions[agent])
            for agent in self.agents
        }
        (
            self.observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        ) = self.env.step(sim_actions)
        self.frame_index += 1

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

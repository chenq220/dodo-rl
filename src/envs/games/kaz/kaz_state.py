from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from envs.state.base import Entity, EnvState, State


@dataclass(frozen=True)
class KazEntity(Entity):
    """
    KAZ游戏实体。

    KAZ是二维屏幕游戏，使用基类默认 local 坐标系；
    Entity.position 为 LocalFrame，x/y 对应屏幕坐标，z 固定为 0。
    """

    # KAZ 场景专属的阵营和基础类别。
    side: str
    base_type: str
    # KAZ 二维游戏只实际使用 heading；pitch/roll 保留为场景专属姿态字段。
    heading: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    # KAZ实体唯一编号，玩家使用agent_name，僵尸和武器按当前帧序号生成。
    entity_id: str = ""
    # 当前实体是否仍然有效；死亡玩家、越界武器会标记为False。
    alive: bool = True
    # 屏幕左上角坐标。
    screen_x: float = 0.0
    screen_y: float = 0.0
    # 屏幕中心点坐标。
    center_x: float = 0.0
    center_y: float = 0.0
    # 按屏幕宽高归一化后的位置。
    normalized_x: float = 0.0
    normalized_y: float = 0.0
    # 碰撞盒尺寸。
    width: float = 0.0
    height: float = 0.0
    # KAZ内部方向向量。
    direction_x: float = 0.0
    direction_y: float = 0.0
    # 最近一次动作编号；玩家有效。
    action: int = -1
    # 当前局累计击杀分；玩家有效。
    score: float = 0.0
    # 是否处于攻击动作；玩家或近战武器有效。
    attacking: bool = False
    # 武器冷却计数；玩家有效。
    weapon_timeout: int = 0
    # 当前附属武器数量；玩家有效。
    weapon_count: int = 0
    # 武器拥有者；箭矢和剑/锤有效。
    owner_name: str | None = None
    # 武器是否处于有效攻击阶段；剑/锤有效。
    active: bool = False
    # 武器动画阶段；剑/锤有效。
    phase: int | None = None
    # 僵尸距离底线的像素距离；僵尸有效。
    distance_to_line: float | None = None
    # 僵尸推进比例，0表示刚生成附近，1表示到达底线。
    line_progress: float | None = None


@dataclass(frozen=True)
class KazEnvState(EnvState):
    """KAZ局级环境态势，承载非单实体的全局游戏状态。"""

    screen_width: int = 0
    screen_height: int = 0
    fps: int = 0
    frame_index: int = 0
    max_cycles: int = 0
    spawn_rate: int = 0
    zombie_spawn_rate: int = 0
    max_zombies: int = 0
    max_arrows: int = 0
    line_y: float = 0.0
    run: bool = True
    line_breached: bool = False
    all_players_dead: bool = False
    alive_archers: int = 0
    alive_knights: int = 0
    alive_players: int = 0
    zombie_count: int = 0
    arrow_count: int = 0
    sword_count: int = 0
    total_score: float = 0.0


@dataclass
class KazState(State):
    """KAZ完整态势，包含玩家、僵尸、武器、观测、奖励和终止信息。"""

    entities: list[KazEntity] = field(default_factory=list)
    env: KazEnvState = field(default_factory=KazEnvState)
    sim_time: float = 0.0
    possible_agents: list[str] = field(default_factory=list)
    live_agents: list[str] = field(default_factory=list)
    dead_agents: list[str] = field(default_factory=list)
    raw_rewards: dict[str, float] = field(default_factory=dict)
    terminations: dict[str, bool] = field(default_factory=dict)
    truncations: dict[str, bool] = field(default_factory=dict)
    infos: dict[str, dict[str, Any]] = field(default_factory=dict)
    observations: dict[str, list[float]] = field(default_factory=dict)

    def agent_observation(self, agent_name: str, obs_dim: int) -> np.ndarray:
        """按agent名读取固定维度观测；死亡或缺失agent返回零向量。"""
        obs = self.observations.get(agent_name)
        if obs is None:
            return np.zeros(obs_dim, dtype=np.float32)
        return np.asarray(obs, dtype=np.float32)

    def global_observation_vector(self, agent_order: list[str], obs_dim: int) -> np.ndarray:
        """按固定agent顺序拼接全局观测向量。"""
        return np.concatenate(
            [self.agent_observation(agent, obs_dim) for agent in agent_order],
            dtype=np.float32,
        )

    @property
    def players(self) -> list[KazEntity]:
        return [entity for entity in self.entities if entity.base_type == "player"]

    @property
    def zombies(self) -> list[KazEntity]:
        return [entity for entity in self.entities if entity.base_type == "enemy"]

    @property
    def weapons(self) -> list[KazEntity]:
        return [entity for entity in self.entities if entity.base_type == "weapon"]

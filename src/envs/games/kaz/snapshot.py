from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from pettingzoo.butterfly.knights_archers_zombies.src import constants as kaz_const

from envs.games.kaz.kaz import KazGame, KazGameConfig
from envs.games.kaz.kaz_state import KazEntity, KazEnvState, KazState
from envs.state.spatial import LocalFrame


def flatten_observation(obs) -> np.ndarray:
    """把 PettingZoo 观测压平成固定长度向量。"""
    return np.asarray(obs, dtype=np.float32).flatten()


def alive_player_names(game: KazGame) -> list[str]:
    """
    返回业务意义上的存活玩家。

    PettingZoo 在 max_cycles 截断时会清空 env.agents，但原始玩家对象仍 alive=True；
    判断“是否全灭”必须按玩家真实 alive 字段，不能按 env.agents。
    """
    raw_env = game.env.unwrapped
    return [
        str(player.agent_name)
        for player in getattr(raw_env, "agent_list", [])
        if bool(getattr(player, "alive", False))
    ]


def line_y_value() -> float:
    """僵尸触线判定的纵坐标阈值（越过即视为突破底线）。"""
    return float(kaz_const.SCREEN_HEIGHT - kaz_const.ZOMBIE_Y_SPEED)


def build_kaz_state(
    game: KazGame,
    game_config: KazGameConfig,
    possible_agents: list[str],
    *,
    zombie_name_resolver: Callable[[Any], str] | None = None,
) -> KazState:
    """
    把 PettingZoo KAZ 游戏态势转换为框架 KazState。

    这是纯粹的“游戏事实”提取（玩家/僵尸/武器实体、防线信息、原始奖励/观测），
    不含任何解法相关的奖励或战术逻辑，供环境层在其上构造观测与奖励。
    """
    raw_env = game.env.unwrapped
    screen_width = int(kaz_const.SCREEN_WIDTH)
    screen_height = int(kaz_const.SCREEN_HEIGHT)
    line_y = line_y_value()

    player_entities = [
        _player_to_entity(player, screen_width, screen_height)
        for player in getattr(raw_env, "agent_list", [])
    ]
    zombie_entities = []
    for index, zombie in enumerate(getattr(raw_env, "zombie_list", [])):
        stable_name = zombie_name_resolver(zombie) if zombie_name_resolver is not None else None
        zombie_entities.append(
            _zombie_to_entity(
                index,
                zombie,
                line_y,
                screen_width,
                screen_height,
                entity_name=stable_name,
            )
        )
    weapon_entities: list[KazEntity] = []
    for player in getattr(raw_env, "agent_list", []):
        for index, weapon in enumerate(getattr(player, "weapons", [])):
            weapon_entities.append(_weapon_to_entity(index, weapon, screen_width, screen_height))

    live_agents = alive_player_names(game)
    agents = list(game.agents)
    dead_agents = [agent for agent in agents if agent not in live_agents]
    observations = {
        agent: flatten_observation(obs).tolist()
        for agent, obs in game.observations.items()
    }
    raw_rewards = {agent: float(game.rewards.get(agent, 0.0)) for agent in agents}

    env_state = KazEnvState(
        screen_width=screen_width,
        screen_height=screen_height,
        fps=int(getattr(KazGame, "FPS", 0)),
        frame_index=int(getattr(game, "frame_index", 0)),
        max_cycles=int(game_config.max_cycles),
        spawn_rate=int(game_config.spawn_rate),
        zombie_spawn_rate=int(getattr(raw_env, "zombie_spawn_rate", 0)),
        max_zombies=int(game_config.max_zombies),
        max_arrows=int(game_config.max_arrows),
        line_y=line_y,
        run=bool(getattr(raw_env, "run", True)),
        line_breached=any(
            entity.distance_to_line is not None and entity.distance_to_line < 0
            for entity in zombie_entities
        ),
        all_players_dead=len(live_agents) == 0,
        alive_archers=len(getattr(raw_env, "archer_list", [])),
        alive_knights=len(getattr(raw_env, "knight_list", [])),
        alive_players=len(live_agents),
        zombie_count=len(zombie_entities),
        arrow_count=sum(1 for entity in weapon_entities if entity.type == "arrow"),
        sword_count=sum(1 for entity in weapon_entities if entity.type == "sword"),
        total_score=float(sum(entity.score for entity in player_entities)),
    )
    state = KazState(
        entities=player_entities + zombie_entities + weapon_entities,
        env=env_state,
        sim_time=float(getattr(game, "frame_index", 0)) / max(float(KazGame.FPS), 1.0),
        possible_agents=list(possible_agents),
        live_agents=live_agents,
        dead_agents=dead_agents,
        raw_rewards=raw_rewards,
        terminations={agent: bool(game.terminations.get(agent, False)) for agent in agents},
        truncations={agent: bool(game.truncations.get(agent, False)) for agent in agents},
        infos={agent: dict(game.infos.get(agent, {})) for agent in agents},
        observations=observations,
    )
    state._raw_data = {
        "possible_agents": list(possible_agents),
        "live_agents": live_agents,
        "dead_agents": dead_agents,
        "raw_rewards": raw_rewards,
        "entity_count": len(state.entities),
    }
    return state


def _player_to_entity(player, screen_width: int, screen_height: int) -> KazEntity:
    entity_type = "archer" if getattr(player, "is_archer", False) else "knight"
    return KazEntity(
        entity_id=str(player.agent_name),
        name=str(player.agent_name),
        side="defender",
        type=entity_type,
        base_type="player",
        position=_sprite_position(player),
        heading=float(getattr(player, "angle", 0.0)),
        pitch=0.0,
        roll=0.0,
        speed=float(getattr(player, "speed", 0.0)),
        alive=bool(getattr(player, "alive", False)),
        **_sprite_common_fields(player, screen_width, screen_height),
        direction_x=float(getattr(getattr(player, "direction", None), "x", 0.0)),
        direction_y=float(getattr(getattr(player, "direction", None), "y", 0.0)),
        action=int(getattr(player, "action", -1)),
        score=float(getattr(player, "score", 0.0)),
        attacking=bool(getattr(player, "attacking", False)),
        weapon_timeout=int(getattr(player, "weapon_timeout", 0)),
        weapon_count=len(getattr(player, "weapons", [])),
    )


def _zombie_to_entity(
    index: int,
    zombie,
    line_y: float,
    screen_width: int,
    screen_height: int,
    *,
    entity_name: str | None = None,
) -> KazEntity:
    distance_to_line = float(line_y - float(zombie.rect.y))
    line_progress = float(np.clip(float(zombie.rect.y) / max(line_y, 1.0), 0.0, 1.0))
    name = entity_name or f"zombie_{index}"
    return KazEntity(
        entity_id=name,
        name=name,
        side="attacker",
        type="zombie",
        base_type="enemy",
        position=_sprite_position(zombie),
        heading=180.0,
        pitch=0.0,
        roll=0.0,
        speed=float(kaz_const.ZOMBIE_Y_SPEED),
        alive=True,
        **_sprite_common_fields(zombie, screen_width, screen_height),
        direction_x=0.0,
        direction_y=1.0,
        distance_to_line=distance_to_line,
        line_progress=line_progress,
    )


def _weapon_to_entity(index: int, weapon, screen_width: int, screen_height: int) -> KazEntity:
    owner = getattr(weapon, "archer", None) or getattr(weapon, "knight", None)
    owner_name = getattr(owner, "agent_name", None)
    weapon_type = "arrow" if hasattr(weapon, "archer") else "sword"
    active = bool(getattr(weapon, "active", True))
    return KazEntity(
        entity_id=f"{weapon_type}_{owner_name}_{index}",
        name=f"{weapon_type}_{owner_name}_{index}",
        side="defender",
        type=weapon_type,
        base_type="weapon",
        position=_sprite_position(weapon),
        heading=float(getattr(owner, "angle", 0.0)),
        pitch=0.0,
        roll=0.0,
        speed=float(getattr(kaz_const, "ARROW_SPEED", 0.0)) if weapon_type == "arrow" else 0.0,
        alive=active,
        **_sprite_common_fields(weapon, screen_width, screen_height),
        direction_x=float(getattr(getattr(weapon, "direction", None), "x", 0.0)),
        direction_y=float(getattr(getattr(weapon, "direction", None), "y", 0.0)),
        owner_name=owner_name,
        active=active,
        phase=getattr(weapon, "phase", None),
    )


def _sprite_position(sprite) -> LocalFrame:
    return LocalFrame(
        x=float(getattr(sprite.rect, "x", 0.0)),
        y=float(getattr(sprite.rect, "y", 0.0)),
        z=0.0,
    )


def _sprite_common_fields(sprite, screen_width: int, screen_height: int) -> dict[str, float]:
    rect = sprite.rect
    center_x = float(getattr(rect, "centerx", getattr(rect, "x", 0.0)))
    center_y = float(getattr(rect, "centery", getattr(rect, "y", 0.0)))
    return {
        "screen_x": float(getattr(rect, "x", 0.0)),
        "screen_y": float(getattr(rect, "y", 0.0)),
        "center_x": center_x,
        "center_y": center_y,
        "normalized_x": center_x / max(float(screen_width), 1.0),
        "normalized_y": center_y / max(float(screen_height), 1.0),
        "width": float(getattr(rect, "width", 0.0)),
        "height": float(getattr(rect, "height", 0.0)),
    }

"""KAZ 向量观测 canonical 布局：全课程共用最大槽位表，缺失实体用空槽行填充。"""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from envs.games.kaz.kaz import KazGameConfig

# 默认 canonical 槽位（3 弓 + 5 骑 + 30 僵尸 + 30 箭）；仅影响观测矩阵行数，不改变本局实际出兵。
CANONICAL_NUM_ARCHERS = 3
CANONICAL_NUM_KNIGHTS = 5
CANONICAL_MAX_ZOMBIES = 30
CANONICAL_MAX_ARROWS = 30

# PettingZoo 空槽在 observe() 后的典型行（无实体：typemask 全 0，无 zombie/archer 标记）。
# 与「未刷新的 zombie 槽位」相同；不是 (0,0) 处的真实僵尸（真实僵尸 typemask[0]=1）。
EMPTY_OBSERVED_ROW_11: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def canonical_game_config(base: KazGameConfig | None = None) -> KazGameConfig:
    """构造仅用于观测 padding 的 canonical game_config（保留 vector_state 等开关）。"""
    kwargs: dict = {}
    if base is not None:
        kwargs = {field.name: getattr(base, field.name) for field in fields(KazGameConfig)}
    kwargs.update(
        num_archers=CANONICAL_NUM_ARCHERS,
        num_knights=CANONICAL_NUM_KNIGHTS,
        max_zombies=CANONICAL_MAX_ZOMBIES,
        max_arrows=CANONICAL_MAX_ARROWS,
    )
    return KazGameConfig(**kwargs)


def kaz_obs_row_width(game_config: KazGameConfig) -> int:
    assert game_config.vector_state, "KAZ obs_layout requires vector_state=true"
    vector_width = 4 + (6 if game_config.use_typemasks else 0)
    return vector_width + 1


def kaz_raw_obs_row_count(game_config: KazGameConfig) -> int:
    tracked = (
        game_config.num_archers
        + game_config.num_knights
        + game_config.max_zombies
        + game_config.num_knights
        + game_config.max_arrows
    )
    return tracked + 1


def canonical_raw_obs_dim(game_config: KazGameConfig) -> int:
    canon = canonical_game_config(game_config)
    return kaz_raw_obs_row_count(canon) * kaz_obs_row_width(canon)


def empty_obs_row(row_width: int, *, use_typemasks: bool) -> np.ndarray:
    """与 PettingZoo 空实体槽一致：整行 0（含 typemask），不会被读成 zombie。"""
    del use_typemasks  # 列宽已含 typemask；空槽语义与 use_typemasks 无关
    if row_width == len(EMPTY_OBSERVED_ROW_11):
        return np.array(EMPTY_OBSERVED_ROW_11, dtype=np.float32)
    return np.zeros(row_width, dtype=np.float32)


def _split_obs_rows(obs_2d: np.ndarray, game_config: KazGameConfig) -> tuple[np.ndarray, ...]:
    row = 0
    self_row = obs_2d[row : row + 1]
    row += 1
    arch = obs_2d[row : row + game_config.num_archers]
    row += game_config.num_archers
    knight = obs_2d[row : row + game_config.num_knights]
    row += game_config.num_knights
    sword = obs_2d[row : row + game_config.num_knights]
    row += game_config.num_knights
    arrow = obs_2d[row : row + game_config.max_arrows]
    row += game_config.max_arrows
    zombie = obs_2d[row : row + game_config.max_zombies]
    return self_row, arch, knight, sword, arrow, zombie


def _pad_segment(
    segment: np.ndarray,
    target_rows: int,
    row_width: int,
    *,
    use_typemasks: bool,
) -> np.ndarray:
    if target_rows <= 0:
        return np.zeros((0, row_width), dtype=np.float32)
    if segment.shape[0] >= target_rows:
        return segment[:target_rows]
    pad_count = target_rows - segment.shape[0]
    empty = np.stack(
        [empty_obs_row(row_width, use_typemasks=use_typemasks) for _ in range(pad_count)],
        axis=0,
    )
    if segment.shape[0] == 0:
        return empty
    return np.concatenate([segment, empty], axis=0)


def pad_kaz_raw_obs_to_canonical(
    native_flat: np.ndarray,
    game_config: KazGameConfig,
    canonical_config: KazGameConfig | None = None,
) -> np.ndarray:
    """
    将 PettingZoo 原生观测 pad 到 canonical 槽位表长度。

    缺失的 archer/knight 等槽位插入与 PettingZoo 未占用槽位相同的全零行
    （typemask 全 0，无 zombie bit），使同一套网络输入维可在各课程间复用。
    """
    canon = canonical_config or canonical_game_config(game_config)
    row_width = kaz_obs_row_width(game_config)
    native_rows = kaz_raw_obs_row_count(game_config)
    native = np.asarray(native_flat, dtype=np.float32).reshape(native_rows, row_width)

    self_row, arch, knight, sword, arrow, zombie = _split_obs_rows(native, game_config)
    use_typemasks = bool(game_config.use_typemasks)
    padded = np.concatenate(
        [
            self_row,
            _pad_segment(arch, canon.num_archers, row_width, use_typemasks=use_typemasks),
            _pad_segment(knight, canon.num_knights, row_width, use_typemasks=use_typemasks),
            _pad_segment(sword, canon.num_knights, row_width, use_typemasks=use_typemasks),
            _pad_segment(arrow, canon.max_arrows, row_width, use_typemasks=use_typemasks),
            _pad_segment(zombie, canon.max_zombies, row_width, use_typemasks=use_typemasks),
        ],
        axis=0,
    )
    expected_rows = kaz_raw_obs_row_count(canon)
    assert padded.shape == (expected_rows, row_width), (
        f"pad shape mismatch: got={padded.shape}, expected=({expected_rows}, {row_width})"
    )
    return padded.reshape(-1)

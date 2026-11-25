"""固定网格态势表征。

单份输入 x 的形状为 [C, H, W]，元素数 N=C*H*W；批量输入
[B, C, H, W] 的 B 不计入单份长度。输出 z 的长度由配置的
output_dim=d 决定，批量输出为 [B, d]。输入/输出元素数之比 N/d
只是长度预算；如果使用者先选比例 r>=1，可以手算 d=ceil(N/r)
作为候选值，再明确填入配置，本模块不会自动选维度。

实际字节数分别为 N*s_x 和 d*s_z（s 为每元素字节数）。只有
s_x==s_z 时，字节数之比才是 N/d。例如 [3,100,100] 输入且
d=128 时，N=30000，元素数之比为 234.375:1；两端均为
float32 时，单份数据分别占 120000 和 512 字节。该估算不含
网络参数、中间张量、回放容器开销，也不保证表征保留决策信息。

若输入仅有 M 种可能状态，要求所有状态可无损恢复的固定长度
二进制编码至少需 ceil(log2(M)) bit。取值范围本身不足以推导
有损决策表征所需的 d；不同区域布局可能需要不同的空间细节。
"""

from __future__ import annotations

import math

import numpy as np

from config import SpatialRepresentationConfig


class SpatialGridRepresentation:
    """按固定空间分块求每通道均值，输出手动指定长度的区域特征。

切分仅依赖 shape 和 output_dim，没有可训练权重。同一数值网格在
采样、回放和评估中得到相同结果；下游策略网络学习如何使用特征。
同一覆盖率但位置不同的地图通常产生不同向量，但分块均值是有损的，
同一块内的细节可能不可区分。环境应先裁出本组的责任区。
"""

    def __init__(self, config: SpatialRepresentationConfig) -> None:
        shape = config.input_shape
        if (
            not isinstance(shape, tuple) or len(shape) != 3
            or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
        ):
            raise ValueError("representation input_shape must be a positive [C,H,W] tuple")
        if isinstance(config.output_dim, bool) or not isinstance(config.output_dim, int):
            raise ValueError("representation output_dim must be an integer")
        channels, height, width = shape
        if not channels <= config.output_dim <= channels * height * width:
            raise ValueError(
                f"representation output_dim must be in [{channels}, "
                f"{channels * height * width}]"
            )
        self.config = config
        # 输入 shape 和输出维度固定，空间分块只需在装配时计算一次。
        self._tiles: list[tuple[int, int, int, int, int]] = []
        base, extra = divmod(config.output_dim, channels)
        for channel in range(channels):
            count = base + int(channel < extra)
            rows = max(math.ceil(count / width), round(math.sqrt(count * height / width)))
            rows = min(height, count, rows)
            row_edges = np.linspace(0, height, rows + 1, dtype=np.int64)
            col_base, col_extra = divmod(count, rows)
            for row in range(rows):
                cols = col_base + int(row < col_extra)
                col_edges = np.linspace(0, width, cols + 1, dtype=np.int64)
                for col in range(cols):
                    self._tiles.append((
                        channel,
                        int(row_edges[row]), int(row_edges[row + 1]),
                        int(col_edges[col]), int(col_edges[col + 1]),
                    ))
        if len(self._tiles) != config.output_dim:
            raise ValueError("representation tile count does not match output_dim")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x)
        if values.ndim not in (3, 4):
            raise ValueError("representation input must be [C,H,W] or [B,C,H,W]")
        real_numeric = (
            values.dtype == np.dtype("bool")
            or (
                np.issubdtype(values.dtype, np.number)
                and not np.issubdtype(values.dtype, np.complexfloating)
            )
        )
        if not real_numeric or not np.all(np.isfinite(values)):
            raise ValueError("representation input must contain finite real numeric values")
        single = values.ndim == 3
        if single:
            values = values[None, ...]
        batch = values.shape[0]
        if values.shape[1:] != self.config.input_shape:
            raise ValueError(
                f"representation input shape mismatch: got {values.shape[1:]}, "
                f"expected {self.config.input_shape}"
            )
        encoded: list[np.ndarray] = []
        for channel, row_start, row_end, col_start, col_end in self._tiles:
            tile = values[:, channel, row_start:row_end, col_start:col_end]
            encoded.append(tile.mean(axis=(1, 2), dtype=np.float32))
        result = np.stack(encoded, axis=-1).astype(np.float32, copy=False)
        if result.shape != (batch, self.config.output_dim):
            raise ValueError(f"representation shape mismatch: {result.shape}")
        return result[0] if single else result

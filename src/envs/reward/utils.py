"""奖励塑形配套工具：区域几何、PBRS、分值 clip。"""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal, Sequence

XAxis = float
YAxis = float
Point = tuple[XAxis, YAxis]
Region = list[Point]

SHAPING_MIN = -1.0
SHAPING_MAX = 1.0
UNIT_MIN = 0.0
UNIT_MAX = 1.0
_POINT_EPS = 1e-6
_SEGMENT_EPS = 1e-6

RegionKind = Literal["point", "line", "polygon"]


class ShapingOutputRange(Enum):
    """塑形势函数 Φ 与 PBRS 步奖励 F 的输出区间。

    - UNIT：Φ∈[0, 1]，F=clip(γΦ'−Φ)∈[-1, 1]
    - DELTA_UNIT：Φ∈[0, 1]（同 UNIT），F=(clip(γΦ'−Φ)+1)/2∈[0, 1]（0.5=不变）
    - SIGNED：Φ∈[-1, 1]，F=clip(γΦ'−Φ)∈[-1, 1]
    """

    UNIT = "unit"
    DELTA_UNIT = "delta_unit"
    SIGNED = "signed"


def phi_output_range(output_range: ShapingOutputRange) -> ShapingOutputRange:
    """score 层位势区间：DELTA_UNIT 与 UNIT 共用 Φ∈[0,1]。"""
    if output_range == ShapingOutputRange.DELTA_UNIT:
        return ShapingOutputRange.UNIT
    return output_range


def clip_shaping(value: float) -> float:
    return float(max(SHAPING_MIN, min(SHAPING_MAX, value)))


def clip_to_output_range(value: float, output_range: ShapingOutputRange) -> float:
    phi_range = phi_output_range(output_range)
    if phi_range == ShapingOutputRange.UNIT:
        return float(max(UNIT_MIN, min(UNIT_MAX, value)))
    return clip_shaping(value)


def empty_shaping_score(output_range: ShapingOutputRange) -> float:
    if phi_output_range(output_range) == ShapingOutputRange.UNIT:
        return UNIT_MIN
    return SHAPING_MIN


def pbrs_raw_delta(phi_pre: float, phi_post: float, gamma: float) -> float:
    return gamma * phi_post - phi_pre


def pbrs_shaping(
    phi_pre: float,
    phi_post: float,
    gamma: float,
    output_range: ShapingOutputRange,
) -> float:
    """PBRS 步奖励 F = clip(γΦ'−Φ) 或 DELTA_UNIT 线性映射到 [0, 1]。"""
    f_raw = pbrs_raw_delta(phi_pre, phi_post, gamma)
    if output_range == ShapingOutputRange.DELTA_UNIT:
        return (clip_shaping(f_raw) + SHAPING_MAX) / 2.0
    return clip_shaping(f_raw)


def pbrs_delta(phi_pre: float, phi_post: float, gamma: float) -> float:
    """兼容别名：等价于 ``pbrs_shaping(..., ShapingOutputRange.SIGNED)``。"""
    return pbrs_shaping(phi_pre, phi_post, gamma, ShapingOutputRange.SIGNED)


def classify_region(region: Sequence[Point]) -> RegionKind:
    n = len(region)
    if n == 1:
        return "point"
    if n == 2:
        return "line"
    if n >= 3:
        return "polygon"
    raise ValueError(f"region must have at least 1 point, got {n}")


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return float(math.hypot(ax - bx, ay - by))


def point_distance(ax: float, ay: float, bx: float, by: float) -> float:
    return _dist(ax, ay, bx, by)


def _cross(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * by - ay * bx


def _on_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float, eps: float) -> bool:
    if abs(_cross(bx - ax, by - ay, px - ax, py - ay)) > eps:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    if dot < -eps:
        return False
    seg_len_sq = (bx - ax) ** 2 + (by - ay) ** 2
    return dot <= seg_len_sq + eps


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """点是否在多边形内或边界上。"""
    if len(polygon) < 3:
        raise ValueError("point_in_polygon requires >= 3 vertices")

    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]
        if _on_segment(x, y, x0, y0, x1, y1, _SEGMENT_EPS):
            return True

    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-18) + xi
        ):
            inside = not inside
        j = i
    return inside


def point_in_region(x: float, y: float, region: Sequence[Point]) -> bool:
    kind = classify_region(region)
    if kind == "point":
        px, py = region[0]
        return _dist(x, y, px, py) <= _POINT_EPS
    if kind == "line":
        ax, ay = region[0]
        bx, by = region[1]
        return _on_segment(x, y, ax, ay, bx, by, _SEGMENT_EPS)
    return point_in_polygon(x, y, region)


def _point_to_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx, vy = bx - ax, by - ay
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= _SEGMENT_EPS:
        return _dist(px, py, ax, ay)
    t = ((px - ax) * vx + (py - ay) * vy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * vx
    proj_y = ay + t * vy
    return _dist(px, py, proj_x, proj_y)


def point_to_region_boundary_distance(x: float, y: float, region: Sequence[Point]) -> float:
    """点到区域边界的最短距离（像素）。

    在区域内或压在边界/顶点上返回 0（不算区外，不触发 leave_decay_coef 衰减）。
    """
    if point_in_region(x, y, region):
        return 0.0

    kind = classify_region(region)
    if kind == "point":
        px, py = region[0]
        return _dist(x, y, px, py)
    if kind == "line":
        ax, ay = region[0]
        bx, by = region[1]
        return _point_to_segment_distance(x, y, ax, ay, bx, by)

    min_dist = float("inf")
    for i in range(len(region)):
        x0, y0 = region[i]
        x1, y1 = region[(i + 1) % len(region)]
        min_dist = min(min_dist, _point_to_segment_distance(x, y, x0, y0, x1, y1))
    return min_dist


def _segments_intersect(
    ax: float, ay: float, bx: float, by: float,
    cx: float, cy: float, dx: float, dy: float,
) -> bool:
    def orient(px: float, py: float, qx: float, qy: float, rx: float, ry: float) -> float:
        return _cross(qx - px, qy - py, rx - px, ry - py)

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)

    if (o1 > _SEGMENT_EPS and o2 < -_SEGMENT_EPS or o1 < -_SEGMENT_EPS and o2 > _SEGMENT_EPS) and (
        o3 > _SEGMENT_EPS and o4 < -_SEGMENT_EPS or o3 < -_SEGMENT_EPS and o4 > _SEGMENT_EPS
    ):
        return True

    if _on_segment(cx, cy, ax, ay, bx, by, _SEGMENT_EPS):
        return True
    if _on_segment(dx, dy, ax, ay, bx, by, _SEGMENT_EPS):
        return True
    if _on_segment(ax, ay, cx, cy, dx, dy, _SEGMENT_EPS):
        return True
    if _on_segment(bx, by, cx, cy, dx, dy, _SEGMENT_EPS):
        return True
    return False


def segment_intersects_region(start: Point, end: Point, region: Sequence[Point]) -> bool:
    sx, sy = start
    ex, ey = end
    if point_distance(sx, sy, ex, ey) <= _POINT_EPS:
        return False
    kind = classify_region(region)

    if kind == "point":
        px, py = region[0]
        return _point_to_segment_distance(px, py, sx, sy, ex, ey) <= _POINT_EPS

    if kind == "line":
        ax, ay = region[0]
        bx, by = region[1]
        return _segments_intersect(sx, sy, ex, ey, ax, ay, bx, by)

    if point_in_polygon(sx, sy, region) or point_in_polygon(ex, ey, region):
        return True
    for i in range(len(region)):
        x0, y0 = region[i]
        x1, y1 = region[(i + 1) % len(region)]
        if _segments_intersect(sx, sy, ex, ey, x0, y0, x1, y1):
            return True
    return False


def distance_seek_score(
    dist: float,
    target_distance: float,
    max_deviation: float,
    output_range: ShapingOutputRange = ShapingOutputRange.SIGNED,
) -> float:
    phi_range = phi_output_range(output_range)
    deviation = abs(dist - target_distance)
    if max_deviation <= _POINT_EPS:
        perfect = deviation <= _POINT_EPS
        if phi_range == ShapingOutputRange.UNIT:
            return UNIT_MAX if perfect else UNIT_MIN
        return SHAPING_MAX if perfect else SHAPING_MIN
    if deviation >= max_deviation:
        return empty_shaping_score(output_range)
    if phi_range == ShapingOutputRange.UNIT:
        return float(UNIT_MAX - deviation / max_deviation)
    return clip_shaping(SHAPING_MAX - 2.0 * deviation / max_deviation)


def x_strip_index(x: float, screen_width: float, num_strips: int) -> int:
    if screen_width <= 0.0 or num_strips <= 0:
        return 0
    strip = int(x / screen_width * num_strips)
    return int(max(0, min(num_strips - 1, strip)))


def strip_duty_score(
    x: float,
    screen_width: float,
    duty_strip: int,
    num_strips: int,
    strip_decay_coef: float,
    output_range: ShapingOutputRange = ShapingOutputRange.UNIT,
) -> float:
    """x 方向条带职责势：本条约 1，跨条按条距软衰减；strip_decay_coef 越大越严格。"""
    current_strip = x_strip_index(x, screen_width, num_strips)
    strip_dist = abs(current_strip - duty_strip)
    raw = 1.0 - strip_decay_coef * strip_dist
    return clip_to_output_range(raw, output_range)


def region_activity_score(
    x: float,
    y: float,
    region: Sequence[Point],
    leave_decay_coef: float,
    output_range: ShapingOutputRange = ShapingOutputRange.SIGNED,
) -> float:
    """几何区职责势：区内满分，区外按到边界距离衰减。"""
    phi_range = phi_output_range(output_range)
    if point_in_region(x, y, region):
        return clip_to_output_range(SHAPING_MAX, output_range)
    d_out = point_to_region_boundary_distance(x, y, region)
    if leave_decay_coef <= 0.0:
        return empty_shaping_score(output_range)
    if phi_range == ShapingOutputRange.UNIT:
        return float(max(UNIT_MIN, UNIT_MAX - min(d_out * leave_decay_coef, UNIT_MAX)))
    return clip_shaping(SHAPING_MAX - 2.0 * min(d_out * leave_decay_coef, 1.0))


def position_threat_score(
    x: float,
    y: float,
    region: Sequence[Point],
    approach_decay_coef: float,
    output_range: ShapingOutputRange,
) -> float:
    """单僵尸威胁势：禁区内最低；区外离边界越远越安全。"""
    phi_range = phi_output_range(output_range)
    if point_in_region(x, y, region):
        return empty_shaping_score(output_range)
    d_out = point_to_region_boundary_distance(x, y, region)
    if approach_decay_coef <= 0.0:
        return empty_shaping_score(output_range)
    proximity = min(d_out * approach_decay_coef, 1.0)
    if phi_range == ShapingOutputRange.UNIT:
        return float(proximity)
    return clip_shaping(2.0 * proximity - 1.0)


def pair_threat_score(
    pre: Point,
    post: Point,
    region: Sequence[Point],
    approach_decay_coef: float,
    output_range: ShapingOutputRange,
) -> float:
    """单僵尸步后有效势：步内穿越禁区或 post 落点违规则取最低分。"""
    if point_in_region(post[0], post[1], region):
        return empty_shaping_score(output_range)
    if segment_intersects_region(pre, post, region):
        return empty_shaping_score(output_range)
    return position_threat_score(post[0], post[1], region, approach_decay_coef, output_range)


def aggregate_threat_score(
    positions: Sequence[Point],
    region: Sequence[Point],
    approach_decay_coef: float,
    output_range: ShapingOutputRange,
) -> float:
    """多僵尸聚合势：取最差（min）；无候选时视为安全满分。"""
    if not positions:
        return clip_to_output_range(SHAPING_MAX, output_range)
    return min(
        position_threat_score(x, y, region, approach_decay_coef, output_range)
        for x, y in positions
    )


def threat_transition_score(
    targets_pre: Sequence[Point],
    targets_post: Sequence[Point],
    region: Sequence[Point],
    approach_decay_coef: float,
    output_range: ShapingOutputRange,
) -> float:
    """步后有效威胁势：逐步检测 post 与 pre→post 轨迹。"""
    if len(targets_pre) != len(targets_post):
        raise ValueError(
            f"targets_pre/post length mismatch: {len(targets_pre)} vs {len(targets_post)}"
        )
    if not targets_pre:
        return clip_to_output_range(SHAPING_MAX, output_range)
    return min(
        pair_threat_score(pre, post, region, approach_decay_coef, output_range)
        for pre, post in zip(targets_pre, targets_post)
    )


def nearest_target(observer: Point, targets: Sequence[Point]) -> Point | None:
    if not targets:
        return None
    ox, oy = observer
    return min(targets, key=lambda t: _dist(ox, oy, t[0], t[1]))


def nearest_target_in_region(
    observer: Point,
    targets: Sequence[Point],
    region: Sequence[Point],
) -> Point | None:
    ox, oy = observer
    in_region = [t for t in targets if point_in_region(t[0], t[1], region)]
    if not in_region:
        return None
    return min(in_region, key=lambda t: _dist(ox, oy, t[0], t[1]))

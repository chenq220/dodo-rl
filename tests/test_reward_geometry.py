"""envs/reward/utils.py 几何判定工具单元测试 + matplotlib 示意图。"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import Polygon as MplPolygon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from envs.reward.utils import (
    classify_region,
    nearest_target,
    nearest_target_in_region,
    point_in_polygon,
    point_in_region,
    point_to_region_boundary_distance,
    segment_intersects_region,
)

MIN_POLYGON_VERTICES = 6
NUM_POLYGON_SAMPLE_POINTS = 6
CANVAS_BBOX = (0.0, 0.0, 1280.0, 720.0)
THREAT_BBOX = (0.0, 480.0, 1280.0, 720.0)

# pytest 固定种子；直接运行脚本时可在 show_geometry_overview 里换随机种子
LARGE_ZONE_SEED = 42
THREAT_ZONE_SEED = 1042
POLYGON_SAMPLE_SEED = 2042

MIDLINE = [(200.0, 100.0), (600.0, 500.0)]


def random_star_polygon(
    n_vertices: int,
    bbox: tuple[float, float, float, float],
    seed: int,
    *,
    min_vertices: int = MIN_POLYGON_VERTICES,
) -> list[tuple[float, float]]:
    """在 bbox 内生成星形简单多边形（顶点数 ≥ min_vertices，半径随机 → 不规则形状）。"""
    if n_vertices < min_vertices:
        raise ValueError(f"n_vertices must be >= {min_vertices}, got {n_vertices}")

    rng = random.Random(seed)
    xmin, ymin, xmax, ymax = bbox
    w, h = xmax - xmin, ymax - ymin
    cx = rng.uniform(xmin + 0.15 * w, xmax - 0.15 * w)
    cy = rng.uniform(ymin + 0.15 * h, ymax - 0.15 * h)
    max_r = min(xmax - cx, cx - xmin, ymax - cy, cy - ymin) * rng.uniform(0.55, 0.92)

    angles = sorted(rng.uniform(0.0, 2.0 * math.pi) for _ in range(n_vertices))
    points: list[tuple[float, float]] = []
    for a in angles:
        # 内外半径交替 + 抖动 → 凹多边形、锯齿边
        if rng.random() < 0.45:
            r = max_r * rng.uniform(0.22, 0.58)
        else:
            r = max_r * rng.uniform(0.72, 1.05)
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        points.append((float(x), float(y)))
    return points


def _region_bbox(region: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in region]
    ys = [p[1] for p in region]
    return min(xs), min(ys), max(xs), max(ys)


def random_points_in_bbox(
    region: list[tuple[float, float]],
    n: int,
    *,
    seed: int,
    margin: float = 80.0,
) -> list[tuple[float, float]]:
    """在 region 包围盒（可外扩 margin）内均匀采 n 个点，不筛内外。"""
    rng = random.Random(seed)
    xmin, ymin, xmax, ymax = _region_bbox(region)
    return [
        (
            rng.uniform(xmin - margin, xmax + margin),
            rng.uniform(ymin - margin, ymax + margin),
        )
        for _ in range(n)
    ]


def random_segments_in_bbox(
    region: list[tuple[float, float]],
    n: int,
    *,
    seed: int,
    margin: float = 80.0,
) -> list[tuple[tuple[float, float], tuple[float, float], bool]]:
    """在包围盒内随机取 n 条线段，返回 (start, end, intersects)。"""
    rng = random.Random(seed)
    xmin, ymin, xmax, ymax = _region_bbox(region)
    lo_x, hi_x = xmin - margin, xmax + margin
    lo_y, hi_y = ymin - margin, ymax + margin
    segments: list[tuple[tuple[float, float], tuple[float, float], bool]] = []
    for _ in range(n):
        start = (rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y))
        end = (rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y))
        hit = segment_intersects_region(start, end, region)
        segments.append((start, end, hit))
    return segments


def random_points_in_region(
    region: list[tuple[float, float]],
    n: int,
    *,
    seed: int,
) -> list[tuple[float, float]]:
    """在 region 包围盒内 rejection 采样，返回 n 个落在 region 内的随机点。"""
    rng = random.Random(seed)
    xmin, ymin, xmax, ymax = _region_bbox(region)
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < n:
        if attempts > 50_000:
            raise RuntimeError(f"failed to sample {n} inside points after {attempts} attempts")
        x = rng.uniform(xmin, xmax)
        y = rng.uniform(ymin, ymax)
        attempts += 1
        if point_in_region(x, y, region):
            points.append((x, y))
    return points


def random_points_outside_region(
    region: list[tuple[float, float]],
    n: int,
    *,
    seed: int,
    margin: float = 80.0,
) -> list[tuple[float, float]]:
    """在略大于 region 的包围盒内采样 n 个落在 region 外的点。"""
    rng = random.Random(seed)
    xmin, ymin, xmax, ymax = _region_bbox(region)
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < n:
        if attempts > 50_000:
            raise RuntimeError(f"failed to sample {n} outside points after {attempts} attempts")
        x = rng.uniform(xmin - margin, xmax + margin)
        y = rng.uniform(ymin - margin, ymax + margin)
        attempts += 1
        if not point_in_region(x, y, region):
            points.append((x, y))
    return points


def find_outside_segment_missing_region(
    region: list[tuple[float, float]],
    *,
    seed: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """找一条完全在 region 外的线段（两端点均在多边形外且不相交）。"""
    candidates = random_points_outside_region(region, 24, seed=seed)
    for i, start in enumerate(candidates):
        for end in candidates[i + 1 :]:
            if not segment_intersects_region(start, end, region):
                return start, end
    raise RuntimeError("failed to find outside miss segment")


def make_test_zones(
    *,
    large_seed: int = LARGE_ZONE_SEED,
    threat_seed: int = THREAT_ZONE_SEED,
    n_vertices: int = 8,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    large = random_star_polygon(n_vertices, CANVAS_BBOX, large_seed)
    threat = random_star_polygon(n_vertices, THREAT_BBOX, threat_seed)
    return large, threat


LARGE_ZONE, THREAT_ZONE = make_test_zones()
POLYGON_MIXED_SAMPLES = random_points_in_bbox(
    LARGE_ZONE, NUM_POLYGON_SAMPLE_POINTS, seed=POLYGON_SAMPLE_SEED
)
POLYGON_INSIDE_SAMPLES = [p for p in POLYGON_MIXED_SAMPLES if point_in_region(p[0], p[1], LARGE_ZONE)]
POLYGON_OUTSIDE_SAMPLES = [p for p in POLYGON_MIXED_SAMPLES if not point_in_region(p[0], p[1], LARGE_ZONE)]
OUTSIDE_SAMPLES = random_points_outside_region(LARGE_ZONE, 2, seed=POLYGON_SAMPLE_SEED + 1)
MISS_SEGMENT = find_outside_segment_missing_region(LARGE_ZONE, seed=POLYGON_SAMPLE_SEED + 2)


class TestClassifyRegion:
    def test_point(self):
        assert classify_region([(1.0, 2.0)]) == "point"

    def test_line(self):
        assert classify_region([(0.0, 0.0), (1.0, 1.0)]) == "line"

    def test_polygon_at_least_six_edges(self):
        assert len(LARGE_ZONE) >= MIN_POLYGON_VERTICES
        assert classify_region(LARGE_ZONE) == "polygon"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 1 point"):
            classify_region([])


class TestPointInRegion:
    def test_point_region(self):
        assert point_in_region(1.0, 1.0, [(1.0, 1.0)])
        assert not point_in_region(2.0, 2.0, [(1.0, 1.0)])

    def test_line_region_on_segment(self):
        assert point_in_region(400.0, 300.0, MIDLINE)
        assert not point_in_region(50.0, 50.0, MIDLINE)

    def test_polygon_mixed_bbox_samples(self):
        assert len(POLYGON_MIXED_SAMPLES) == NUM_POLYGON_SAMPLE_POINTS
        for x, y in POLYGON_MIXED_SAMPLES:
            inside = point_in_region(x, y, LARGE_ZONE)
            d = point_to_region_boundary_distance(x, y, LARGE_ZONE)
            if inside:
                assert d == 0.0
            else:
                assert d > 0.0

    def test_polygon_mixed_samples_not_all_same_side(self):
        assert POLYGON_INSIDE_SAMPLES and POLYGON_OUTSIDE_SAMPLES

    def test_polygon_random_outside_samples(self):
        for x, y in OUTSIDE_SAMPLES:
            assert not point_in_region(x, y, LARGE_ZONE)

    def test_point_in_polygon_requires_three_vertices(self):
        with pytest.raises(ValueError):
            point_in_polygon(0.0, 0.0, [(0.0, 0.0), (1.0, 1.0)])


@pytest.mark.parametrize("zone_seed", [11, 42, 137, 999, 2026])
class TestRandomPolygonShapes:
    """同一套断言在多种随机多边形上重复验证。"""

    def test_inside_samples(self, zone_seed: int):
        zone = random_star_polygon(8, CANVAS_BBOX, zone_seed)
        samples = random_points_in_region(zone, NUM_POLYGON_SAMPLE_POINTS, seed=zone_seed + 10_000)
        for x, y in samples:
            assert point_in_region(x, y, zone)
            assert point_to_region_boundary_distance(x, y, zone) == 0.0

    def test_outside_to_inside_segment_hits(self, zone_seed: int):
        zone = random_star_polygon(8, CANVAS_BBOX, zone_seed)
        inside = random_points_in_region(zone, 1, seed=zone_seed + 20_000)[0]
        outside = random_points_outside_region(zone, 1, seed=zone_seed + 30_000)[0]
        assert segment_intersects_region(outside, inside, zone)

    def test_threat_random_shape(self, zone_seed: int):
        threat = random_star_polygon(8, THREAT_BBOX, zone_seed + 50_000)
        assert len(threat) >= MIN_POLYGON_VERTICES
        outside = random_points_outside_region(threat, 1, seed=zone_seed + 60_000)[0]
        inside = random_points_in_region(threat, 1, seed=zone_seed + 70_000)[0]
        assert segment_intersects_region(outside, inside, threat)


class TestBoundaryDistance:
    def test_inside_mixed_samples_return_zero(self):
        for x, y in POLYGON_INSIDE_SAMPLES:
            assert point_to_region_boundary_distance(x, y, LARGE_ZONE) == 0.0

    def test_outside_mixed_samples_positive(self):
        for x, y in POLYGON_OUTSIDE_SAMPLES:
            assert point_to_region_boundary_distance(x, y, LARGE_ZONE) > 0.0

    def test_outside_positive(self):
        x, y = OUTSIDE_SAMPLES[0]
        assert point_to_region_boundary_distance(x, y, LARGE_ZONE) > 0.0

    def test_line_region(self):
        d = point_to_region_boundary_distance(50.0, 50.0, MIDLINE)
        # 线区域按有限线段计距，最近点是端点 (200, 100)。
        assert d == pytest.approx(math.hypot(150.0, 50.0), rel=1e-3)

    def test_threat_outside_positive(self):
        outside = random_points_outside_region(THREAT_ZONE, 1, seed=THREAT_ZONE_SEED + 3)[0]
        assert point_to_region_boundary_distance(outside[0], outside[1], THREAT_ZONE) > 0.0


class TestSegmentIntersectsRegion:
    def test_zero_length_segment_false(self):
        x, y = POLYGON_MIXED_SAMPLES[0]
        assert not segment_intersects_region((x, y), (x, y), LARGE_ZONE)

    def test_random_segments_mixed_hit_miss(self):
        segments = random_segments_in_bbox(LARGE_ZONE, 12, seed=POLYGON_SAMPLE_SEED + 9)
        hits = [hit for _, _, hit in segments]
        assert any(hits) and not all(hits)

    def test_crosses_polygon(self):
        if not POLYGON_INSIDE_SAMPLES or not POLYGON_OUTSIDE_SAMPLES:
            pytest.skip("mixed samples lack both sides for this seed")
        x_in, y_in = POLYGON_INSIDE_SAMPLES[0]
        x_out, y_out = POLYGON_OUTSIDE_SAMPLES[0]
        assert segment_intersects_region((x_out, y_out), (x_in, y_in), LARGE_ZONE)

    def test_outside_miss(self):
        start, end = MISS_SEGMENT
        assert not segment_intersects_region(start, end, LARGE_ZONE)

    def test_starts_inside(self):
        if len(POLYGON_INSIDE_SAMPLES) < 2:
            pytest.skip("need >=2 inside samples")
        x0, y0 = POLYGON_INSIDE_SAMPLES[0]
        x1, y1 = POLYGON_INSIDE_SAMPLES[1]
        assert segment_intersects_region((x0, y0), (x1, y1), LARGE_ZONE)

    def test_threat_cross_outside_to_inside(self):
        outside = random_points_outside_region(THREAT_ZONE, 1, seed=THREAT_ZONE_SEED + 4)[0]
        inside = random_points_in_region(THREAT_ZONE, 1, seed=THREAT_ZONE_SEED + 5)[0]
        assert segment_intersects_region(outside, inside, THREAT_ZONE)


class TestNearestTarget:
    def test_empty(self):
        assert nearest_target((0.0, 0.0), []) is None

    def test_picks_closest(self):
        targets = [(10.0, 0.0), (3.0, 4.0), (100.0, 100.0)]
        assert nearest_target((0.0, 0.0), targets) == (3.0, 4.0)

    def test_in_region_filter(self):
        if len(POLYGON_INSIDE_SAMPLES) < 3:
            pytest.skip("need >=3 inside samples")
        targets = list(POLYGON_INSIDE_SAMPLES[:3]) + [OUTSIDE_SAMPLES[0]]
        origin = POLYGON_INSIDE_SAMPLES[0]
        hit = nearest_target_in_region(origin, targets, LARGE_ZONE)
        assert hit in POLYGON_INSIDE_SAMPLES[:3]


def _draw_region(ax, region, *, facecolor, edgecolor, label):
    kind = classify_region(region)
    if kind == "polygon":
        patch = MplPolygon(
            region,
            closed=True,
            fill=True,
            alpha=0.25,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2,
            label=label,
        )
        ax.add_patch(patch)
        xs, ys = zip(*region)
        ax.plot(xs, ys, "o", color=edgecolor, markersize=4)
    elif kind == "line":
        xs, ys = zip(*region)
        ax.plot(xs, ys, color=edgecolor, linewidth=3, label=label)
    else:
        ax.plot(region[0][0], region[0][1], "o", color=edgecolor, markersize=10, label=label)


def _pad_limits(region: list[tuple[float, float]], margin: float = 40.0):
    xmin, ymin, xmax, ymax = _region_bbox(region)
    return (xmin - margin, xmax + margin, ymin - margin, ymax + margin)


def show_geometry_overview(
    *,
    large_seed: int | None = None,
    threat_seed: int | None = None,
    sample_seed: int | None = None,
) -> None:
    """弹窗展示；默认每次随机新多边形，种子会打在标题里便于复现。"""
    if large_seed is None:
        large_seed = random.randrange(1_000_000)
    if threat_seed is None:
        threat_seed = random.randrange(1_000_000)
    if sample_seed is None:
        sample_seed = random.randrange(1_000_000)

    large_zone, threat_zone = make_test_zones(large_seed=large_seed, threat_seed=threat_seed)
    mixed_points = random_points_in_bbox(large_zone, NUM_POLYGON_SAMPLE_POINTS, seed=sample_seed)
    large_segments = random_segments_in_bbox(large_zone, NUM_POLYGON_SAMPLE_POINTS, seed=sample_seed + 1)
    threat_segments = random_segments_in_bbox(threat_zone, NUM_POLYGON_SAMPLE_POINTS, seed=sample_seed + 2)

    xmin, xmax, ymin, ymax = _pad_limits(large_zone, 60.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"geometry helpers | large_seed={large_seed} threat_seed={threat_seed} sample_seed={sample_seed}",
        fontsize=12,
    )

    # --- 1. 随机点：可能在区域内也可能在区域外 ---
    ax = axes[0, 0]
    ax.set_title(f"point_in_region / boundary ({len(large_zone)}-gon)")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    _draw_region(ax, large_zone, facecolor="steelblue", edgecolor="navy", label="large zone")
    for i, (x, y) in enumerate(mixed_points):
        inside = point_in_region(x, y, large_zone)
        d = point_to_region_boundary_distance(x, y, large_zone)
        color = "green" if inside else "red"
        ax.plot(x, y, "o", color=color, markersize=8)
        ax.annotate(
            f"p{i + 1} {'in' if inside else 'out'}\nd={d:.0f}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color=color,
        )
    ax.plot([], [], "o", color="green", label="inside")
    ax.plot([], [], "o", color="red", label="outside")
    ax.legend(loc="upper right")

    # --- 2. line region ---
    ax = axes[0, 1]
    ax.set_title("point_in_region (line segment)")
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 600)
    ax.set_aspect("equal")
    _draw_region(ax, MIDLINE, facecolor="none", edgecolor="purple", label="duty line")
    for x, y in [(400.0, 300.0), (100.0, 100.0), (650.0, 500.0)]:
        on = point_in_region(x, y, MIDLINE)
        ax.plot(x, y, "o", color="green" if on else "red", markersize=8)
    ax.legend()

    # --- 3. 随机线段：可能穿越也可能不穿越 ---
    ax = axes[1, 0]
    ax.set_title(f"segment_intersects_region ({len(large_zone)}-gon)")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    _draw_region(ax, large_zone, facecolor="wheat", edgecolor="saddlebrown", label="large zone")
    hit_labeled = miss_labeled = False
    for i, (start, end, hit) in enumerate(large_segments):
        color = "crimson" if hit else "gray"
        label = None
        if hit and not hit_labeled:
            label = "cross"
            hit_labeled = True
        elif not hit and not miss_labeled:
            label = "no cross"
            miss_labeled = True
        ax.plot([start[0], end[0]], [start[1], end[1]], "-", color=color, linewidth=2, label=label)
        ax.plot(start[0], start[1], "s", color=color, markersize=5)
        ax.plot(end[0], end[1], "^", color=color, markersize=5)
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, f"s{i + 1}", fontsize=7, color=color)
    ax.legend(loc="upper right")

    # --- 4. 威胁带：同样随机线段 ---
    ax = axes[1, 1]
    tx0, tx1, ty0, ty1 = _pad_limits(threat_zone, 40.0)
    ax.set_title(f"threat zone ({len(threat_zone)}-gon)")
    ax.set_xlim(tx0, tx1)
    ax.set_ylim(ty0, ty1)
    ax.set_aspect("equal")
    _draw_region(ax, threat_zone, facecolor="salmon", edgecolor="darkred", label="threat zone")
    hit_labeled = miss_labeled = False
    for i, (start, end, hit) in enumerate(threat_segments):
        color = "crimson" if hit else "forestgreen"
        label = None
        if hit and not hit_labeled:
            label = "cross"
            hit_labeled = True
        elif not hit and not miss_labeled:
            label = "no cross"
            miss_labeled = True
        if start != end:
            ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        ax.plot(start[0], start[1], "s", color=color, markersize=5)
        ax.plot(end[0], end[1], "x", color=color, markersize=6)
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, f"t{i + 1} cross={hit}", fontsize=7, color=color)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    plt.show()

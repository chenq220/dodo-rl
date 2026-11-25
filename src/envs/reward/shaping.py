"""奖励塑形算子：离散目标选择，以及可见目标搜寻、区域活动、区域威胁判定。"""

from __future__ import annotations

from typing import Hashable, Sequence, TypeVar

from envs.reward.utils import (
    Point,
    ShapingOutputRange,
    aggregate_threat_score,
    classify_region,
    distance_seek_score,
    empty_shaping_score,
    nearest_target,
    nearest_target_in_region,
    pbrs_shaping,
    point_distance,
    region_activity_score as _geometry_activity_score,
    strip_duty_score,
    threat_transition_score,
)

TargetId = TypeVar("TargetId", bound=Hashable)


def target_selection_shaping(
    selected_target: TargetId | None,
    shaping_target_indices: Sequence[TargetId],
    all_target_indices: Sequence[TargetId],
    *,
    primary_target_indices: Sequence[TargetId] = (),
    available_target_indices: Sequence[TargetId] | None = None,
    output_range: ShapingOutputRange = ShapingOutputRange.SIGNED,
) -> float:
    """在主要目标缺席时，鼓励选择指定的塑形目标。

    这是离散目标选择动作的即时塑形奖励，不是 PBRS 势函数。只要任一主要目标
    当前可用，本函数就返回中性信号，把决策交还给环境的主要任务奖励；主要
    目标全部缺席时，选择当前可用的塑形目标获得高分，选择其他目标获得低分。
    如果当前没有可选的塑形目标，也返回中性分，避免惩罚不可能完成的选择。

    :param selected_target: 本步实际选择的目标序号；未选择时传 ``None``
    :param shaping_target_indices: 希望在主要目标缺席时优先选择的目标序号
    :param all_target_indices: 合法目标序号全集
    :param primary_target_indices: 由主要任务奖励负责的目标序号
    :param available_target_indices: 本步出现/可选择的目标序号；None 表示全集均出现
    :param output_range: 遵循本库 shaping 层规范：SIGNED / UNIT 输出命中 ``+1``、
        未命中 ``-1``、中性 ``0``；DELTA_UNIT 映射为命中 ``1``、未命中 ``0``、
        中性 ``0.5``
    :return: 即时目标选择塑形奖励
    :raises ValueError: 序号重复、集合越界、主目标与塑形目标重叠，或选择了不可用目标
    """
    all_targets = _validated_unique_targets("all_target_indices", all_target_indices)
    if not all_targets:
        raise ValueError("all_target_indices must not be empty")

    shaping_targets = _validated_unique_targets(
        "shaping_target_indices", shaping_target_indices,
    )
    primary_targets = _validated_unique_targets(
        "primary_target_indices", primary_target_indices,
    )
    available_targets = (
        all_targets
        if available_target_indices is None
        else _validated_unique_targets("available_target_indices", available_target_indices)
    )

    for name, targets in (
        ("shaping_target_indices", shaping_targets),
        ("primary_target_indices", primary_targets),
        ("available_target_indices", available_targets),
    ):
        unknown = targets - all_targets
        if unknown:
            raise ValueError(f"{name} contains targets outside all_target_indices: {unknown!r}")

    overlap = shaping_targets & primary_targets
    if overlap:
        raise ValueError(f"primary and shaping targets must be disjoint: {overlap!r}")
    if selected_target is not None and selected_target not in available_targets:
        raise ValueError(f"selected_target {selected_target!r} is not currently available")

    # 主要目标出现时不改变原任务的最优策略。
    if primary_targets & available_targets:
        return _target_selection_output(0.0, output_range)

    available_shaping_targets = shaping_targets & available_targets
    if not available_shaping_targets:
        return _target_selection_output(0.0, output_range)

    if selected_target in available_shaping_targets:
        return _target_selection_output(1.0, output_range)
    return _target_selection_output(-1.0, output_range)


def _target_selection_output(
    signed_signal: float,
    output_range: ShapingOutputRange,
) -> float:
    """复用 shaping 层区间约定；这里的 signed_signal 已是一步即时信号。"""
    return pbrs_shaping(0.0, signed_signal, 1.0, output_range)


def _validated_unique_targets(
    name: str,
    targets: Sequence[TargetId],
) -> set[TargetId]:
    try:
        target_set = set(targets)
    except TypeError as exc:
        raise ValueError(f"{name} must contain hashable target indices") from exc
    if len(target_set) != len(targets):
        raise ValueError(f"{name} must not contain duplicate target indices")
    return target_set


def region_visible_target_seek_score(
    observer: Point,
    targets: Sequence[Point],
    target_distance: float,
    max_deviation: float,
    output_range: ShapingOutputRange,
    region: Sequence[Point] | None = None,
) -> float:
    """
    范围内最近可见目标搜寻分值
    :param observer: 观察者坐标 (x, y)
    :param targets: 候选目标坐标列表
    :param target_distance: 理想距离；与最近目标距离等于该值时得满分
    :param max_deviation: 最大允许偏差；偏差达该值时得最低分
    :param output_range: SIGNED：Φ∈[-1, 1]；UNIT / DELTA_UNIT：Φ∈[0, 1]（DELTA_UNIT 的 score 与 UNIT 相同）
    :param region: 搜寻区域；None 表示全场最近目标，否则仅面（≥3 点）内取最近
    :return: 位势 Φ（非 PBRS 步奖励 F）
    """
    if not targets:
        return empty_shaping_score(output_range)

    if region is None:
        nearest = nearest_target(observer, targets)
    else:
        if classify_region(region) != "polygon":
            raise ValueError("region_visible_target_seek requires a polygon region (>= 3 points)")
        nearest = nearest_target_in_region(observer, targets, region)

    if nearest is None:
        return empty_shaping_score(output_range)

    ox, oy = observer
    dist = point_distance(ox, oy, nearest[0], nearest[1])
    return distance_seek_score(dist, target_distance, max_deviation, output_range)


def region_visible_target_seek_shaping(
    pre_observer: Point,
    post_observer: Point,
    targets_pre: Sequence[Point],
    targets_post: Sequence[Point],
    target_distance: float,
    max_deviation: float,
    gamma: float,
    output_range: ShapingOutputRange,
    region: Sequence[Point] | None = None,
) -> float:
    """
    范围内最近可见目标搜寻塑形
    :param pre_observer: 步前观察者坐标 (x, y)
    :param post_observer: 步后观察者坐标 (x, y)
    :param targets_pre: 步前候选目标坐标列表
    :param targets_post: 步后候选目标坐标列表
    :param target_distance: 理想距离
    :param max_deviation: 最大允许偏差；应覆盖场上典型寻敌距离
    :param gamma: PBRS 折扣 γ；F=γΦ'−Φ，γ<1 且 Φ>0 时每步有 (γ−1)Φ 负漂移
    :param output_range: SIGNED / UNIT：F=clip(γΦ'−Φ)∈[-1, 1]；
        DELTA_UNIT：F=(clip(γΦ'−Φ)+1)/2∈[0, 1]（0.5=Φ 不变）。逼近目标时常用 UNIT
    :param region: 搜寻区域；None 为全场寻敌
    :return: PBRS 步奖励 F
    """
    phi_pre = region_visible_target_seek_score(
        pre_observer, targets_pre, target_distance, max_deviation, output_range, region,
    )
    phi_post = region_visible_target_seek_score(
        post_observer, targets_post, target_distance, max_deviation, output_range, region,
    )
    return pbrs_shaping(phi_pre, phi_post, gamma, output_range)


def region_activity_score(
    x: float,
    y: float,
    region: Sequence[Point],
    leave_decay_coef: float,
    output_range: ShapingOutputRange = ShapingOutputRange.SIGNED,
) -> float:
    """
    ② 目标区域活动位势 Φ（点/线/面几何区）。

    :param x: 智能体 x 坐标
    :param y: 智能体 y 坐标
    :param region: 职责区域，点 / 线 / 面均可
    :param leave_decay_coef: 区外衰减系数（须 > 0，仅几何区 ``region_activity_*`` 使用，与
        条带 ``strip_decay_coef`` 无关）。压线/贴边/区内：d_out=0，满分，不触发衰减。完全出界后
        按到边界最短距离 d_out（像素）线性扣分：UNIT 为 ``1−min(d_out·coef,1)``，SIGNED 为
        ``1−2·min(d_out·coef,1)``。**coef 越大越严格**（同样距离扣更多、更快到最低分）；**coef
        越小越宽松**（需走更远才扣到底）。饱和距离 ``d_sat=1/coef``：d_out≥d_sat 已达最低分
        （UNIT→0，SIGNED→−1）。coef≤0 非法，直接返回最低分。
    :param output_range: SIGNED：Φ∈[-1, 1]；UNIT / DELTA_UNIT：Φ∈[0, 1]（DELTA_UNIT 的 score 与 UNIT 相同）
    :return: 位势 Φ
    """
    return _geometry_activity_score(x, y, region, leave_decay_coef, output_range)


def region_activity_shaping(
    pre_xy: Point,
    post_xy: Point,
    region: Sequence[Point],
    leave_decay_coef: float,
    gamma: float,
    output_range: ShapingOutputRange,
) -> float:
    """
    指定目标区域活动塑形
    :param pre_xy: 步前智能体坐标 (x, y)
    :param post_xy: 步后智能体坐标 (x, y)
    :param region: 职责区域，在区域内给满分
    :param leave_decay_coef: 区外衰减系数（须 > 0）；**越大越严格，越小越宽松**；饱和距离
        d_sat=1/coef；压线满分。详见 region_activity_score
    :param gamma: PBRS 折扣系数 γ
    :param output_range: SIGNED / UNIT：F=clip(γΦ'−Φ)∈[-1, 1]；
        DELTA_UNIT：F=(clip(γΦ'−Φ)+1)/2∈[0, 1]（0.5=Φ 不变）
    :return: PBRS 步奖励 F
    """
    phi_pre = region_activity_score(pre_xy[0], pre_xy[1], region, leave_decay_coef, output_range)
    phi_post = region_activity_score(post_xy[0], post_xy[1], region, leave_decay_coef, output_range)
    return pbrs_shaping(phi_pre, phi_post, gamma, output_range)


def region_activity_strip_score(
    x: float,
    screen_width: float,
    duty_strip: int,
    num_strips: int,
    strip_decay_coef: float,
    output_range: ShapingOutputRange = ShapingOutputRange.UNIT,
) -> float:
    """
    指定目标区域活动位势 Φ

    :param x: 智能体 x 坐标
    :param screen_width: 战场宽度
    :param duty_strip: 负责条带编号（0 .. num_strips-1）
    :param num_strips: 横向条带总数
    :param strip_decay_coef: 跨条衰减系数（例如 0.28：本条约 1.0，邻条约 0.72）。
        势函数 ``clip(1−coef×|current−duty|, 0, 1)``。**coef 越大越严格**（跨一条扣更多）；
        **越小越宽松**（允许更远跨条接敌）。
    :param output_range: SIGNED：Φ∈[-1, 1]；UNIT / DELTA_UNIT：Φ∈[0, 1]（DELTA_UNIT 的 score 与 UNIT 相同）。活动塑形常用 UNIT
    :return: 位势 Φ
    """
    return strip_duty_score(x, screen_width, duty_strip, num_strips, strip_decay_coef, output_range)


def region_activity_strip_shaping(
    pre_x: float,
    post_x: float,
    screen_width: float,
    duty_strip: int,
    num_strips: int,
    strip_decay_coef: float,
    gamma: float,
    output_range: ShapingOutputRange,
) -> float:
    """
    目标区域活动塑形
    :param pre_x: 步前 x 坐标
    :param post_x: 步后 x 坐标
    :param screen_width: 战场宽度
    :param duty_strip: 负责条带编号
    :param num_strips: 横向条带总数
    :param strip_decay_coef: 跨条衰减系数；**越大越严格，越小越宽松**（见 region_activity_strip_score）
    :param gamma: PBRS 折扣系数 γ
    :param output_range: SIGNED / UNIT：F=clip(γΦ'−Φ)∈[-1, 1]；
        DELTA_UNIT：F=(clip(γΦ'−Φ)+1)/2∈[0, 1]（0.5=Φ 不变）。逼近目标时常用 UNIT
    :return: PBRS 步奖励 F
    """
    phi_pre = region_activity_strip_score(
        pre_x, screen_width, duty_strip, num_strips, strip_decay_coef, output_range,
    )
    phi_post = region_activity_strip_score(
        post_x, screen_width, duty_strip, num_strips, strip_decay_coef, output_range,
    )
    return pbrs_shaping(phi_pre, phi_post, gamma, output_range)


def region_threat_score(
    positions: Sequence[Point],
    region: Sequence[Point],
    approach_decay_coef: float,
    output_range: ShapingOutputRange = ShapingOutputRange.SIGNED,
) -> float:
    """
    区域威胁位势 Φ(s)
    :param positions: 当前步目标坐标列表（env 侧可先过滤观察带候选）
    :param region: 威胁禁区，点 / 线 / 面均可
    :param approach_decay_coef: 逼近衰减系数（须 > 0）；区外离边界越远 Φ 越高（越安全），
        饱和距离 d_sat=1/coef。UNIT：禁区内 Φ=0；SIGNED：禁区/low 映射至 −1。
    :param output_range: SIGNED：Φ∈[-1, 1]；UNIT / DELTA_UNIT：Φ∈[0, 1]（DELTA_UNIT 的 score 与 UNIT 相同）。威胁塑形常用 SIGNED
    :return: 聚合位势 Φ；无候选时视为安全满分
    """
    return aggregate_threat_score(positions, region, approach_decay_coef, output_range)


def region_threat_transition_score(
    targets_pre: Sequence[Point],
    targets_post: Sequence[Point],
    region: Sequence[Point],
    approach_decay_coef: float,
    output_range: ShapingOutputRange = ShapingOutputRange.SIGNED,
) -> float:
    """
    区域威胁判定分值
    :param targets_pre: 步前目标坐标列表，与 targets_post 等长、按 index 对齐
    :param targets_post: 步后目标坐标列表
    :param region: 威胁禁区
    :param approach_decay_coef: 逼近衰减系数（见 region_threat_score）
    :param output_range: SIGNED：Φ∈[-1, 1]；UNIT / DELTA_UNIT：Φ∈[0, 1]（DELTA_UNIT 的 score 与 UNIT 相同）
    :return: 聚合位势 Φ；无候选时视为安全满分
    """
    return threat_transition_score(
        targets_pre, targets_post, region, approach_decay_coef, output_range,
    )


def region_threat_shaping(
    targets_pre: Sequence[Point],
    targets_post: Sequence[Point],
    region: Sequence[Point],
    approach_decay_coef: float,
    gamma: float,
    output_range: ShapingOutputRange = ShapingOutputRange.SIGNED,
) -> float:
    """
    区域威胁判定塑形
    :param targets_pre: 步前目标坐标列表，与 targets_post 等长、按 index 对齐
    :param targets_post: 步后目标坐标列表
    :param region: 威胁禁区，点 / 线 / 面均可
    :param approach_decay_coef: 逼近衰减系数（观察带宽度 H 时可用 coef=5/H）
    :param gamma: PBRS 折扣系数 γ
    :param output_range: SIGNED / UNIT：F=clip(γΦ'−Φ)∈[-1, 1]；
        DELTA_UNIT：F=(clip(γΦ'−Φ)+1)/2∈[0, 1]（0.5=Φ 不变）。威胁塑形常用 SIGNED
    :return: PBRS 步奖励 F
    """
    phi_pre = region_threat_score(targets_pre, region, approach_decay_coef, output_range)
    phi_post = region_threat_transition_score(
        targets_pre, targets_post, region, approach_decay_coef, output_range,
    )
    return pbrs_shaping(phi_pre, phi_post, gamma, output_range)

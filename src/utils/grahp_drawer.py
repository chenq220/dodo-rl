import math
import os
import csv
import tempfile
from typing import Any, BinaryIO, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

_CJK_FONTS = (
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
)


def _configure_cjk_font() -> None:
    """优先使用系统中文字体，避免中文图例显示为方块。"""
    try:
        available = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:
        return
    for name in _CJK_FONTS:
        if name not in available:
            continue
        sans_serif = list(matplotlib.rcParams.get("font.sans-serif", []))
        if name not in sans_serif:
            matplotlib.rcParams["font.sans-serif"] = [name, *sans_serif]
        matplotlib.rcParams["axes.unicode_minus"] = False
        return


_configure_cjk_font()

EpisodeRewards = Union[List[float], Mapping[str, List[float]]]
EpisodeRewardDetails = Mapping[int, Mapping[str, float]]
TREND_WINDOW = 30
MIN_REWARD_CURVE_EPISODES = 2


def _apply_y_axis_tick_interval(ax, y_axis_tick_interval: Optional[float]) -> None:
    """按绝对数值间隔设置 Y 轴主刻度；None 时沿用 matplotlib 自动刻度。"""
    if y_axis_tick_interval is None:
        return
    if (
            isinstance(y_axis_tick_interval, bool)
            or not isinstance(y_axis_tick_interval, (int, float))
            or not math.isfinite(y_axis_tick_interval)
            or y_axis_tick_interval <= 0
    ):
        raise ValueError("y_axis_tick_interval must be a positive finite number.")
    ax.yaxis.set_major_locator(MultipleLocator(float(y_axis_tick_interval)))


def _resolve_colors(
        color_sequence: Optional[Sequence[Any]],
        item_count: int,
) -> Optional[List[Any]]:
    """将用户色块序列循环扩展至绘图对象数量；未指定时交给默认配色。"""
    if color_sequence is None:
        return None
    if isinstance(color_sequence, (str, bytes)):
        raise TypeError("color_sequence must be a sequence of color values, not a single color string.")
    colors = list(color_sequence)
    if not colors:
        raise ValueError("color_sequence must contain at least one color.")
    return [colors[idx % len(colors)] for idx in range(item_count)]


def _normalize_rewards(episode_rewards: EpisodeRewards) -> Dict[str, List[float]]:
    if isinstance(episode_rewards, Mapping):
        return {
            str(agent_name): list(rewards)
            for agent_name, rewards in episode_rewards.items()
            if rewards
        }
    return {"Episode Reward": list(episode_rewards)}


def _rolling_mean(rewards: List[float], window: int = TREND_WINDOW) -> List[float]:
    """仅从收集满 window 局后，计算包含当前局的最近 window 局均值。"""
    if window <= 0:
        raise ValueError("滚动均值窗口必须为正数。")
    trend: List[float] = []
    running_sum = 0.0
    for idx, reward in enumerate(rewards):
        running_sum += reward
        if idx >= window:
            running_sum -= rewards[idx - window]
        trend.append(running_sum / window if idx >= window - 1 else float("nan"))
    return trend


def _normalize_reward_details(
        episode_reward_details: EpisodeRewardDetails,
) -> Tuple[List[int], List[str], Dict[str, List[float]]]:
    """按真实episode编号对齐；未回报episode留空，已回报局的缺失分量补0。"""
    reported_episodes = sorted(int(episode) for episode in episode_reward_details)
    episodes = list(range(1, max(reported_episodes) + 1)) if reported_episodes else []
    component_names: List[str] = []
    known_components = set()
    for episode in reported_episodes:
        for component_name in episode_reward_details[episode]:
            if component_name not in known_components:
                component_names.append(component_name)
                known_components.add(component_name)
    values_by_component = {
        component_name: [
            (
                float(episode_reward_details[episode].get(component_name, 0.0))
                if episode in episode_reward_details
                else float("nan")
            )
            for episode in episodes
        ]
        for component_name in component_names
    }
    return episodes, component_names, values_by_component


def _rolling_mean_with_gaps(
        rewards: List[float],
        window: int = TREND_WINDOW,
) -> List[float]:
    """仅在最近window个真实episode均有数据时计算均值，缺局不压缩窗口。"""
    if window <= 0:
        raise ValueError("滚动均值窗口必须为正数。")
    trend: List[float] = []
    for idx in range(len(rewards)):
        if idx < window - 1:
            trend.append(float("nan"))
            continue
        values = rewards[idx - window + 1: idx + 1]
        if any(not math.isfinite(value) for value in values):
            trend.append(float("nan"))
            continue
        trend.append(sum(values) / window)
    return trend


def _write_plot_csv(path: str, rows) -> None:
    """原子替换 CSV，避免读取方看到正在写入的半个文件。"""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".plot-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("episode", "series", "reward"))
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def save_reward_curve_csv(episode_rewards: EpisodeRewards, path: str) -> None:
    rows = []
    for name, rewards in _normalize_rewards(episode_rewards).items():
        for episode, value in enumerate(rewards, 1):
            rows.append((episode, name, value))
    _write_plot_csv(path, rows)


def save_reward_details_csv(episode_reward_details: EpisodeRewardDetails, path: str) -> None:
    rows = [(episode, name, value)
            for episode, details in sorted(episode_reward_details.items())
            for name, value in details.items()]
    _write_plot_csv(path, rows)


def save_reward_curve(
        episode_rewards: EpisodeRewards,
        image_file: BinaryIO,
        image_name: Optional[str] = None,
        trend_window: int = TREND_WINDOW,
        y_axis_tick_interval: Optional[float] = None,
        color_sequence: Optional[Sequence[Any]] = None,
) -> None:
    """保存训练 reward 曲线。

    ``y_axis_tick_interval`` 指定 Y 轴主刻度的绝对数值间隔；
    ``color_sequence`` 按 Agent 顺序循环应用到原始曲线及其趋势线。
    """
    rewards_by_agent = _normalize_rewards(episode_rewards)
    if not rewards_by_agent:
        raise ValueError("episode_rewards has no non-empty reward series.")

    max_episodes = max(len(rewards) for rewards in rewards_by_agent.values())
    if max_episodes < MIN_REWARD_CURVE_EPISODES:
        raise ValueError("episode_rewards must contain at least two episodes to draw a curve.")

    resolved_image_name = image_name or os.path.basename(str(getattr(image_file, "name", "reward")))
    title_name = os.path.splitext(resolved_image_name)[0]
    fig, (ax_raw, ax_trend) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    colors = _resolve_colors(color_sequence, len(rewards_by_agent))

    try:
        for idx, (agent_name, rewards) in enumerate(rewards_by_agent.items()):
            # 使用从第1局开始的业务编号；因此第30局对应严格的前30局均值。
            x = list(range(1, len(rewards) + 1))
            line_kwargs = {"color": colors[idx]} if colors is not None else {}
            ax_raw.plot(x, rewards, label=agent_name, linewidth=2, **line_kwargs)
            ax_trend.plot(
                x,
                _rolling_mean(rewards, window=trend_window),
                label=agent_name,
                linewidth=2,
                **line_kwargs,
            )

        # 仅统一数字显示格式；各图按自身数据自动缩放纵轴和划分刻度。
        for axis in (ax_raw, ax_trend):
            axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax_raw.set_title(title_name)
        ax_trend.set_title(f"Reward Trend ({trend_window}-episode moving average)")
        ax_trend.set_xlabel("Episode")
        ax_raw.set_ylabel("Reward")
        ax_trend.set_ylabel("Reward")
        ax_raw.set_xlim(1, max(max_episodes, 1))
        ax_raw.legend()
        ax_trend.legend()
        ax_raw.grid(True)
        ax_trend.grid(True)
        _apply_y_axis_tick_interval(ax_raw, y_axis_tick_interval)
        _apply_y_axis_tick_interval(ax_trend, y_axis_tick_interval)
        fig.tight_layout()

        image_file.seek(0)
        image_file.truncate(0)
        fig.savefig(image_file, format="png")
        image_file.flush()
    finally:
        plt.close(fig)


def save_reward_details(
        episode_reward_details: EpisodeRewardDetails,
        image_file: BinaryIO,
        image_name: Optional[str] = None,
        trend_window: int = TREND_WINDOW,
        y_axis_tick_interval: Optional[float] = None,
        color_sequence: Optional[Sequence[Any]] = None,
) -> None:
    """保存正负分离的奖励构成堆叠图。

    ``y_axis_tick_interval`` 指定 Y 轴主刻度的绝对数值间隔；
    ``color_sequence`` 按奖励分量顺序循环应用到堆叠色块。
    """
    episodes, component_names, values_by_component = _normalize_reward_details(
        episode_reward_details
    )
    if len(episodes) < MIN_REWARD_CURVE_EPISODES:
        raise ValueError("episode_reward_details must contain at least two episodes to draw a curve.")
    if not component_names:
        raise ValueError("episode_reward_details has no reward components.")

    resolved_image_name = image_name or os.path.basename(
        str(getattr(image_file, "name", "reward_details"))
    )
    title_name = os.path.splitext(resolved_image_name)[0]
    colors = _resolve_colors(color_sequence, len(component_names))
    if colors is None:
        colors = [plt.get_cmap("tab20")(idx % 20) for idx in range(len(component_names))]
    trend_values_by_component = {
        component_name: _rolling_mean_with_gaps(
            values_by_component[component_name],
            window=trend_window,
        )
        for component_name in component_names
    }
    positive_values = [
        [
            value if not math.isfinite(value) else max(value, 0.0)
            for value in trend_values_by_component[component_name]
        ]
        for component_name in component_names
    ]
    negative_values = [
        [
            value if not math.isfinite(value) else min(value, 0.0)
            for value in trend_values_by_component[component_name]
        ]
        for component_name in component_names
    ]
    fig, ax = plt.subplots(figsize=(12, 8))

    try:
        ax.stackplot(
            episodes,
            *positive_values,
            labels=component_names,
            colors=colors,
            alpha=0.75,
        )
        ax.stackplot(
            episodes,
            *negative_values,
            labels=["_nolegend_"] * len(component_names),
            colors=colors,
            alpha=0.75,
        )
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(f"{title_name} ({trend_window}-episode moving average)")
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward Moving Average")
        ax.set_xlim(min(episodes), max(episodes))
        ax.legend()
        ax.grid(True)
        _apply_y_axis_tick_interval(ax, y_axis_tick_interval)
        fig.tight_layout()

        image_file.seek(0)
        image_file.truncate(0)
        fig.savefig(image_file, format="png")
        image_file.flush()
    finally:
        plt.close(fig)


def save_end_eval_reward_bars(
        episode_reward_details: EpisodeRewardDetails,
        image_file: BinaryIO,
        image_name: Optional[str] = None,
        y_axis_tick_interval: Optional[float] = None,
        color_sequence: Optional[Sequence[Any]] = None,
) -> None:
    """保存结束评估的奖励构成堆叠柱状图；允许只有一局。"""
    episodes, component_names, values_by_component = _normalize_reward_details(
        episode_reward_details
    )
    if not episodes:
        raise ValueError("episode_reward_details must contain at least one episode.")
    if not component_names:
        raise ValueError("episode_reward_details has no reward components.")

    resolved_image_name = image_name or os.path.basename(
        str(getattr(image_file, "name", "end_eval_reward"))
    )
    title_name = os.path.splitext(resolved_image_name)[0]
    colors = _resolve_colors(color_sequence, len(component_names))
    if colors is None:
        colors = [plt.get_cmap("tab20")(idx % 20) for idx in range(len(component_names))]

    xs = list(range(len(episodes)))
    bottoms_pos = [0.0] * len(episodes)
    bottoms_neg = [0.0] * len(episodes)
    fig, ax = plt.subplots(figsize=(max(8, len(episodes) * 0.6 + 4), 6))
    try:
        for idx, component_name in enumerate(component_names):
            values = values_by_component[component_name]
            positives = [
                value if not math.isfinite(value) else max(value, 0.0)
                for value in values
            ]
            negatives = [
                value if not math.isfinite(value) else min(value, 0.0)
                for value in values
            ]
            ax.bar(
                xs,
                positives,
                bottom=bottoms_pos,
                width=0.6,
                label=component_name,
                color=colors[idx],
                alpha=0.85,
            )
            ax.bar(
                xs,
                negatives,
                bottom=bottoms_neg,
                width=0.6,
                color=colors[idx],
                alpha=0.85,
            )
            bottoms_pos = [base + value for base, value in zip(bottoms_pos, positives)]
            bottoms_neg = [base + value for base, value in zip(bottoms_neg, negatives)]

        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title(title_name)
        ax.set_xlabel("End-eval episode")
        ax.set_ylabel("Reward")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(episode) for episode in episodes])
        ax.legend()
        ax.grid(True, axis="y")
        _apply_y_axis_tick_interval(ax, y_axis_tick_interval)
        fig.tight_layout()

        image_file.seek(0)
        image_file.truncate(0)
        fig.savefig(image_file, format="png")
        image_file.flush()
    finally:
        plt.close(fig)

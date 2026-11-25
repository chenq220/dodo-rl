import io
import math
from types import SimpleNamespace

import matplotlib
import pytest

from utils import grahp_drawer
from utils.grahp_drawer import (
    _normalize_reward_details,
    _rolling_mean,
    _rolling_mean_with_gaps,
    save_end_eval_reward_bars,
    save_reward_curve,
    save_reward_details,
)


def test_cjk_font_configuration_prefers_available_font(monkeypatch):
    monkeypatch.setattr(grahp_drawer.font_manager, "fontManager", SimpleNamespace(ttflist=[
        SimpleNamespace(name="DejaVu Sans"),
        SimpleNamespace(name="Noto Sans CJK SC"),
    ]))
    monkeypatch.setitem(matplotlib.rcParams, "font.sans-serif", ["DejaVu Sans"])
    monkeypatch.setitem(matplotlib.rcParams, "axes.unicode_minus", True)

    grahp_drawer._configure_cjk_font()

    assert matplotlib.rcParams["font.sans-serif"][0] == "Noto Sans CJK SC"
    assert matplotlib.rcParams["axes.unicode_minus"] is False


def test_rolling_mean_starts_at_window_then_uses_recent_window_episodes():
    rewards = [float(value) for value in range(1, 36)]

    trend = _rolling_mean(rewards, window=30)

    assert all(math.isnan(value) for value in trend[:29])
    assert trend[29] == pytest.approx(sum(range(1, 31)) / 30)
    assert trend[30] == pytest.approx(sum(range(2, 32)) / 30)
    assert trend[34] == pytest.approx(sum(range(6, 36)) / 30)


def test_rolling_mean_rejects_non_positive_window():
    with pytest.raises(ValueError, match="滚动均值窗口必须为正数"):
        _rolling_mean([1.0], window=0)


def test_save_reward_curve_overwrites_and_keeps_file_handle_open():
    image_file = io.BytesIO()

    save_reward_curve([1.0, 2.0], image_file, image_name="reward.png")
    first_png = image_file.getvalue()
    save_reward_curve([3.0, 4.0, 5.0], image_file, image_name="reward.png")
    second_png = image_file.getvalue()

    assert not image_file.closed
    assert first_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert second_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert second_png != first_png


def test_save_reward_curve_accepts_custom_y_ticks_and_colors():
    image_file = io.BytesIO()

    save_reward_curve(
        {"agent_a": [1.0, 3.0], "agent_b": [-2.0, 4.0]},
        image_file,
        y_axis_tick_interval=2,
        color_sequence=["#123456", "#abcdef"],
    )

    assert image_file.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_save_reward_curve_rejects_invalid_y_tick_interval():
    with pytest.raises(ValueError, match="y_axis_tick_interval"):
        save_reward_curve([1.0, 2.0], io.BytesIO(), y_axis_tick_interval=0)


def test_normalize_reward_details_keeps_real_episode_axis_and_fills_missing_components():
    episodes, component_names, values = _normalize_reward_details({
        3: {"bonus": 2.0, "swing": -1.0},
        2: {},
        1: {"cost": -3.0, "swing": 1.5},
    })

    assert episodes == [1, 2, 3]
    assert component_names == ["cost", "swing", "bonus"]
    assert values == {
        "cost": [-3.0, 0.0, 0.0],
        "swing": [1.5, 0.0, -1.0],
        "bonus": [0.0, 0.0, 2.0],
    }


def test_reward_details_moving_average_starts_at_episode_thirty_and_preserves_gaps():
    rewards = [float(value) for value in range(1, 36)]

    trend = _rolling_mean_with_gaps(rewards, window=30)

    assert all(math.isnan(value) for value in trend[:29])
    assert trend[29] == pytest.approx(sum(range(1, 31)) / 30)
    assert trend[34] == pytest.approx(sum(range(6, 36)) / 30)

    rewards_with_gap = [1.0] * 41
    rewards_with_gap[10] = float("nan")
    trend_with_gap = _rolling_mean_with_gaps(rewards_with_gap, window=30)
    assert math.isnan(trend_with_gap[29])
    assert math.isnan(trend_with_gap[39])
    assert trend_with_gap[40] == pytest.approx(1.0)


def test_save_reward_details_overwrites_positive_negative_stack_and_keeps_handle_open():
    image_file = io.BytesIO()

    save_reward_details(
        {
            1: {"explore": 2.0, "cost": -1.0, "swing": 0.5},
            2: {"explore": 3.0, "cost": -2.0, "swing": -0.25},
        },
        image_file,
        image_name="reward_details.png",
        trend_window=2,
    )
    first_png = image_file.getvalue()
    save_reward_details(
        {
            1: {"explore": 2.0, "cost": -1.0},
            3: {"explore": 4.0, "cost": -3.0, "new_bonus": 1.0},
        },
        image_file,
        image_name="reward_details.png",
        trend_window=2,
    )
    second_png = image_file.getvalue()

    assert not image_file.closed
    assert first_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert second_png.startswith(b"\x89PNG\r\n\x1a\n")
    assert second_png != first_png


def test_save_reward_details_accepts_custom_y_ticks_and_colors():
    image_file = io.BytesIO()

    save_reward_details(
        {1: {"bonus": 2.0, "cost": -1.0}, 2: {"bonus": 4.0, "cost": -3.0}},
        image_file,
        trend_window=2,
        y_axis_tick_interval=0.5,
        color_sequence=["#2ca02c", "#d62728"],
    )

    assert image_file.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_save_end_eval_reward_bars_allows_single_episode_and_total_stack():
    image_file = io.BytesIO()

    save_end_eval_reward_bars({1: {"total": 3.5}}, image_file, image_name="end_eval_reward.png")

    assert not image_file.closed
    assert image_file.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_reward_details_csv_retains_raw_first_29_episodes(tmp_path):
    import csv
    from utils.grahp_drawer import save_reward_details_csv

    path = tmp_path / "raw_data" / "details.csv"
    details = {episode: {"奖励": float(episode)} for episode in range(1, 32)}
    save_reward_details_csv(details, str(path))
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 31
    assert rows[0] == {"episode": "1", "series": "奖励", "reward": "1.0"}
    assert rows[28]["episode"] == "29"
    assert rows[29]["reward"] == "30.0"


def test_save_end_eval_reward_bars_stacks_positive_and_negative_components():
    image_file = io.BytesIO()

    save_end_eval_reward_bars(
        {
            1: {"bonus": 2.0, "cost": -1.0},
            2: {"bonus": 4.0, "cost": -3.0},
        },
        image_file,
        image_name="end_eval_reward.png",
    )

    assert image_file.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_reward_curve_formats_ticks_consistently_with_independent_y_scales(monkeypatch):
    from matplotlib.figure import Figure
    captured = []
    monkeypatch.setattr(Figure, 'savefig', lambda fig, *a, **kw: captured.append(fig))
    # 偶数窗口抵消交替的大幅波动，均值图应保留较小的上升趋势。
    rewards = [-1.0047 + (-1) ** i * 0.1 + i * 1e-6 for i in range(60)]
    save_reward_curve({'jammer': rewards}, io.BytesIO())
    raw, trend = captured[0].axes
    raw_low, raw_high = raw.get_ylim()
    trend_low, trend_high = trend.get_ylim()
    assert raw_low < min(rewards) < max(rewards) < raw_high
    trend_values = [value for value in _rolling_mean(rewards) if math.isfinite(value)]
    assert trend_low < min(trend_values) < max(trend_values) < trend_high
    assert trend_high - trend_low < (raw_high - raw_low) / 100
    assert trend.get_yticks()[1] - trend.get_yticks()[0] < raw.get_yticks()[1] - raw.get_yticks()[0]
    for axis in (raw, trend):
        assert not axis.yaxis.get_major_formatter().get_useOffset()
        assert not axis.yaxis.get_major_formatter()._scientific


def test_reward_stack_boundary_matches_total_trend_without_extra_panels(monkeypatch):
    from matplotlib.figure import Figure
    captured = []
    monkeypatch.setattr(Figure, 'savefig', lambda fig, *a, **kw: captured.append(fig))
    details = {i: {'failure': -1., 'rotation': -.0047 + i * 1e-6}
               for i in range(1, 46)}
    save_reward_details(details, io.BytesIO())
    fig = captured[0]
    assert len(fig.axes) == 1
    axis = fig.axes[0]
    boundary = axis.collections[-1].get_paths()[0].vertices
    trend = _rolling_mean_with_gaps([sum(d.values()) for d in details.values()])
    for episode in range(30, 46):
        values = boundary[boundary[:, 0] == episode, 1]
        assert min(values) == pytest.approx(trend[episode - 1], abs=1e-12)
    assert trend[-1] > trend[29]
    assert not axis.yaxis.get_major_formatter().get_useOffset()
    assert not axis.yaxis.get_major_formatter()._scientific

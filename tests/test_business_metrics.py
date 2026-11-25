from __future__ import annotations

import pytest

from config import AutoMLConfig
from scheduler.model import (
    AutoMLBusinessMetricConfig,
    AutoMLMetricOutputConfig,
    TrainingSnapshot,
)
from scheduler.utils.automl_config import finalize_auto_ml_config
from scheduler.utils.automl_core import aggregate_reward
from scheduler.utils.business_metrics import EpisodeMetricCollector, evaluate_business_expression


def _snapshot(metrics):
    return TrainingSnapshot(
        task_id="business", stage_name="combat", generation=0,
        completed_episodes=len(next(iter(metrics.values()))), max_episodes=10,
        reward_curve={}, learn_metrics={}, current_algo_configs={}, tunable_catalog={},
        episode_metrics=metrics,
    )


def test_metric_output_collects_infos_and_metadata() -> None:
    collector = EpisodeMetricCollector({
        "kills": AutoMLMetricOutputConfig(
            info_path="combat.kills", agent_reduce="sum", episode_reduce="sum",
        ),
        "pressure": AutoMLMetricOutputConfig(metadata_path="game.pressure"),
    })
    collector.add_infos({"a": {"combat": {"kills": 1}}, "b": {"combat": {"kills": 2}}})
    collector.add_infos({"a": {"combat": {"kills": 3}}, "b": {"combat": {"kills": 4}}})
    assert collector.finish({"game": {"pressure": 5}}) == {"kills": 10.0, "pressure": 5.0}


def test_metric_output_accepts_boolean_business_events() -> None:
    collector = EpisodeMetricCollector({
        "stage_success": AutoMLMetricOutputConfig(
            info_path="stage_success", agent_reduce="max", episode_reduce="latest",
        ),
    })
    collector.add_infos({"a": {"stage_success": False}, "b": {"stage_success": True}})
    assert collector.finish({}) == {"stage_success": 1.0}


def test_business_expression_and_window_aggregation() -> None:
    config = AutoMLConfig(
        subject_name="business",
        metric_outputs={
            "kills": AutoMLMetricOutputConfig(info_path="kills"),
            "deaths": AutoMLMetricOutputConfig(info_path="deaths"),
        },
        business_metric=AutoMLBusinessMetricConfig(
            enabled=True, name="score", expression="kills - 2 * deaths",
            aggregation="mean", window=2,
        ),
    )
    assert aggregate_reward(_snapshot({"kills": [1.0, 4.0, 6.0], "deaths": [0.0, 1.0, 1.0]}), config) == 3.0
    assert evaluate_business_expression("max(kills, 2) + abs(-1)", {"kills": 1.0}) == 3.0


def test_business_expression_rejects_code_and_missing_episode_value(tmp_path) -> None:
    config = AutoMLConfig(
        subject_name="business",
        archive_root_dir=str(tmp_path),
        metric_outputs={"kills": AutoMLMetricOutputConfig(info_path="kills")},
        business_metric=AutoMLBusinessMetricConfig(
            enabled=True, name="score", expression="kills", window=2,
        ),
    )
    finalize_auto_ml_config(config)
    with pytest.raises(ValueError, match="episode=2"):
        aggregate_reward(_snapshot({"kills": [1.0, None]}), config)
    config.business_metric.expression = "__import__('os')"
    with pytest.raises(ValueError, match="unsupported|unknown"):
        finalize_auto_ml_config(config)

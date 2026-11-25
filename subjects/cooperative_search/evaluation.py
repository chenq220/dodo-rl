"""Cooperative-search evaluation report integration."""

from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.logger import init_logger
from roles.eval_control import EvalControlRole
from roles.evaluation import EvaluationRole

LOG = init_logger("cooperative_search_evaluation")

EPISODE_METRIC_FIELDS = (
    "episode",
    "steps",
    "success",
    "target_discovered_count",
    "own_loss_count",
    "coverage_rate_percent",
    "false_target_count",
)


def write_episode_metrics_csv(
    rows: Sequence[Mapping[str, Any]],
    file_path: str | os.PathLike[str],
    *,
    fields: Sequence[str] = EPISODE_METRIC_FIELDS,
) -> Path:
    """Write one cooperative-search evaluation record per episode."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in fields}
            )
    LOG.info(f"Episode evaluation metrics written: {path}")
    return path


def _csv_episode_metrics(episode_index: int, metrics: Mapping[str, Any], *, steps: int) -> dict:
    return {
        "episode": int(episode_index),
        "steps": int(metrics.get("steps", steps) or 0),
        "success": bool(metrics.get("success")),
        "target_discovered_count": int(metrics.get("target_discovered_count") or 0),
        "own_loss_count": int(metrics.get("own_loss_count") or 0),
        "coverage_rate_percent": round(float(metrics.get("coverage_rate_percent") or 0.0), 6),
        "false_target_count": int(metrics.get("false_target_count") or 0),
    }


class CooperativeSearchEvalReports:
    """CSV export shared by serial evaluation and distributed EvalControl."""

    eval_report_prefix = "cooperative_search_qmix_eval"
    metric_fields = EPISODE_METRIC_FIELDS

    def episode_report(
            self,
            episode_index: int,
            *,
            episode_metrics: dict | None = None,
            **payload,
    ) -> dict:
        record = super().episode_report(
            episode_index,
            episode_metrics=episode_metrics,
            **payload,
        )
        record["episode_metrics"] = _csv_episode_metrics(
            episode_index,
            dict(episode_metrics or {}),
            steps=int(record.get("steps") or 0),
        )
        return record

    def finished_report(self):
        rows = [
            record["episode_metrics"]
            for record in self.episode_results
            if "episode_metrics" in record
        ]
        if not rows:
            return
        timestamp = time.strftime("%Y%m%d%H%M%S")
        output_path = (
            Path.cwd()
            / "artifacts"
            / "report"
            / f"{self.eval_report_prefix}_{timestamp}_pid{os.getpid()}.csv"
        )
        write_episode_metrics_csv(rows, output_path, fields=self.metric_fields)
        return {"episode_metrics_csv": str(output_path)}


class CooperativeSearchEvaluation(CooperativeSearchEvalReports, EvaluationRole):
    """Serial evaluation: write the same CSV from local TaskManager records."""


class CooperativeSearchEvalControl(CooperativeSearchEvalReports, EvalControlRole):
    """Distributed eval center: write the same CSV from TaskManager records."""

    eval_report_prefix = "cooperative_search_qmix_dist_eval"


class ThreatRoamingEvalReports(CooperativeSearchEvalReports):
    """Export the independent roaming metrics without search-task columns."""

    eval_report_prefix = "cooperative_search_threat_roaming_eval"
    metric_fields = (
        "episode", "steps", "total_reward", "inside_roaming_circle",
        "nearest_peer_distance_km", "dangerous_peer_proximity", "waypoints_completed",
        "waypoint_progress", "waypoint_arrival", "circle_outside", "peer_clearance", "peer_danger",
    )

    def episode_report(self, episode_index, *, episode_metrics=None, **payload):
        record = super().episode_report(episode_index, episode_metrics=episode_metrics, **payload)
        record["episode_metrics"] = {
            **dict(episode_metrics or {}), "episode": int(episode_index),
            "steps": record["steps"], "total_reward": record["total_reward"],
        }
        LOG.info(f"Roaming episode {episode_index} metrics: {record['episode_metrics']}")
        return record


class ThreatRoamingEvaluation(ThreatRoamingEvalReports, EvaluationRole):
    """Serial roaming evaluation with per-episode CSV and GIF artifacts."""


class ThreatRoamingEvalControl(ThreatRoamingEvalReports, EvalControlRole):
    """The same report for distributed roaming evaluation."""

    eval_report_prefix = "cooperative_search_threat_roaming_dist_eval"

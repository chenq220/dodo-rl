"""AutoMLConfig 运行时初始化：路径归一化、归档 storage、Optuna sqlite 默认路径。"""
from __future__ import annotations

import os

from config import (
    AutoMLConfig,
    LocalStorageConfig,
    ResourceDir,
    S3StorageConfig,
    StorageType,
    root_dir_position,
)


def finalize_auto_ml_config(config: AutoMLConfig) -> AutoMLConfig:
    reducers = {"sum", "mean", "min", "max", "latest"}
    for name, output in config.metric_outputs.items():
        if not name.isidentifier():
            raise ValueError(f"metric_outputs name must be a valid identifier: {name!r}")
        source_count = int(bool(output.info_path)) + int(bool(output.metadata_path))
        if source_count != 1:
            raise ValueError(
                f"metric_outputs.{name} must define exactly one of info_path/metadata_path"
            )
        if output.info_path:
            if output.agent_reduce not in reducers or output.episode_reduce not in reducers:
                raise ValueError(f"metric_outputs.{name} has unsupported reducer")

    metric = config.business_metric
    if metric is not None and metric.enabled:
        if not metric.name or not metric.name.isidentifier():
            raise ValueError("business_metric.name must be a valid identifier")
        if metric.aggregation not in {"mean", "min", "max", "latest"}:
            raise ValueError(f"unsupported business_metric.aggregation={metric.aggregation!r}")
        if metric.window <= 0:
            raise ValueError("business_metric.window must be positive")
        from scheduler.utils.business_metrics import validate_business_expression
        validate_business_expression(metric.expression, set(config.metric_outputs))
    if config.archive_root_dir and not os.path.isabs(config.archive_root_dir):
        config.archive_root_dir = os.path.normpath(
            os.path.join(root_dir_position, config.archive_root_dir)
        )
    os.makedirs(config.archive_root_dir, exist_ok=True)

    storage_type = StorageType(config.storage_type)
    if storage_type == StorageType.LOCAL:
        assert isinstance(config.storage, LocalStorageConfig), (
            f"AutoML local archive requires LocalStorageConfig, given={type(config.storage).__name__}"
        )
        config.storage.data_path = config.archive_root_dir
    else:
        assert isinstance(config.storage, S3StorageConfig), (
            f"AutoML S3 archive requires S3StorageConfig, given={type(config.storage).__name__}"
        )

    studies_dir = os.path.join(ResourceDir.AUTOML.value, "studies")
    os.makedirs(studies_dir, exist_ok=True)
    return config

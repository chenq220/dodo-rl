"""AutoML restart 归档存储：local 移动产物，s3 组装后整树上传。"""
from __future__ import annotations

import csv
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List

import yaml

from common.logger import init_logger
from config import AutoMLConfig, S3StorageConfig, StorageType
from scheduler.model import TrainingRunArchive

LOG = init_logger("automl_archive_storage")


def build_s3_client(storage: S3StorageConfig):
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=storage.endpoint_url,
        aws_access_key_id=storage.access_key,
        aws_secret_access_key=storage.secret_key,
    )


def ensure_s3_bucket(client, bucket_name: str) -> None:
    try:
        client.head_bucket(Bucket=bucket_name)
    except client.exceptions.ClientError:
        client.create_bucket(Bucket=bucket_name)
        LOG.info(f"S3 archive bucket created: bucket={bucket_name}")


def move_artifacts_into_archive(
        artifact_paths: Dict[str, List[str]],
        artifacts_dir: str,
) -> List[str]:
    """将外部产物移动进归档 artifacts/ 子目录；返回归档内最终路径列表。"""
    moved: List[str] = []
    for category, paths in artifact_paths.items():
        dest_sub = os.path.join(artifacts_dir, category)
        os.makedirs(dest_sub, exist_ok=True)
        for src in paths:
            if not src or not os.path.isfile(src):
                continue
            dest = os.path.join(dest_sub, os.path.basename(src))
            if os.path.abspath(src) == os.path.abspath(dest):
                moved.append(dest)
                continue
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(src, dest)
            moved.append(dest)
            LOG.info(f"AutoML archive moved artifact: {src} -> {dest}")
    return moved


def write_archive_bundle_contents(
        bundle: TrainingRunArchive,
        archive_dir: str,
        optuna_summary: Dict[str, Any],
) -> None:
    """写算法指标、各业务输出、精简建议和训练产物。"""
    if bundle.episode_rows:
        csv_path = os.path.join(archive_dir, "algos_metrics.csv")
        fieldnames: List[str] = []
        for row in bundle.episode_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(bundle.episode_rows)

    for metric_name, rows in bundle.metric_output_rows.items():
        if not rows:
            continue
        csv_path = os.path.join(archive_dir, f"{metric_name}_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=["episode", "generation", "value"])
            writer.writeheader()
            writer.writerows(rows)

    # snapshot不持有配置；归档调用方把指标摘要放在optuna_summary旁路字段中。
    metric_summary = optuna_summary.pop("metric", {})
    semantic_learner: Dict[str, Dict[str, Any]] = {}
    for learner_name, overrides in bundle.advice.algo_config_overrides.items():
        semantic_learner.setdefault(learner_name, {})["algo_config"] = dict(overrides)
    for learner_name, overrides in bundle.advice.skeleton_overrides.items():
        semantic_learner.setdefault(learner_name, {}).update(overrides)
    toml_overrides: Dict[str, Any] = {}
    if bundle.advice.seed is not None:
        toml_overrides["train_config"] = {"seed": bundle.advice.seed}
    if semantic_learner:
        toml_overrides["learner_config"] = semantic_learner
    if bundle.advice.env_metadata_overrides:
        toml_overrides["env_metadata"] = bundle.advice.env_metadata_overrides
    payload = {
        "schema_version": 2,
        "overview": {
            "env_name": bundle.env_name,
            "task_id": bundle.task_id,
            "stage_name": bundle.stage_name,
            "generation": bundle.generation,
            "completed_episodes": bundle.completed_episodes,
            "archived_at": datetime.now().isoformat(),
            "run_seed": bundle.run_seed,
            "initialization_seed": bundle.initialization_seed,
            "decision": bundle.advice.decision,
            "decision_source": (
                "business_target" if bundle.advice.reason.startswith(("target_metric_missed:", "business_metric_missed:"))
                else "optuna" if "optuna" in bundle.advice.reason
                else "framework"
            ),
            "reason": bundle.advice.reason,
            "metric": metric_summary,
        },
        "optuna": optuna_summary,
        "advice": {
            "type": (
                "next_trial" if bundle.advice.decision == "restart"
                else "graduated" if bundle.advice.decision == "complete"
                else "none"
            ),
            "source": (
                "optuna" if bundle.advice.decision == "restart"
                else "baseline" if optuna_summary.get("current_trial", {}).get("number") is None
                else "current_trial" if bundle.advice.decision == "complete"
                else "framework"
            ),
            "apply_to_stage": bundle.stage_name,
            "toml_overrides": toml_overrides,
        },
    }
    with open(os.path.join(archive_dir, "advice.yaml"), "w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, allow_unicode=True, sort_keys=False)

    if bundle.artifact_paths:
        artifacts_dir = os.path.join(archive_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        move_artifacts_into_archive(bundle.artifact_paths, artifacts_dir)


def upload_directory_to_s3(
        client,
        local_dir: str,
        bucket_name: str,
        key_prefix: str,
) -> str:
    """上传本地归档目录到 S3，返回 s3://bucket/prefix。任一脚本失败则抛错，不清理本地。"""
    prefix = key_prefix.strip("/")
    uploaded = 0
    for root, _, files in os.walk(local_dir):
        for name in files:
            local_path = os.path.join(root, name)
            rel = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
            key = f"{prefix}/{rel}" if prefix else rel
            client.upload_file(Filename=local_path, Bucket=bucket_name, Key=key)
            uploaded += 1
    uri = f"s3://{bucket_name}/{prefix}" if prefix else f"s3://{bucket_name}"
    LOG.info(f"AutoML archive uploaded to {uri}, files={uploaded}")
    return uri


def remove_local_archive_bundle(local_archive_dir: str) -> None:
    """S3 上传成功后删除本地组装目录（含已移入的 artifacts）。"""
    local_archive_abs = os.path.abspath(local_archive_dir)
    if not os.path.isdir(local_archive_abs):
        return
    shutil.rmtree(local_archive_abs)
    LOG.info(f"AutoML archive local archives removed: {local_archive_abs}")


def persist_archive_bundle(
        config: AutoMLConfig,
        local_archive_dir: str,
        folder_name: str,
) -> str:
    """按 storage_type 落盘归档包：local 保留目录；s3 上传成功后删本地，失败则记 error 并保留本地路径。"""
    local_archive_abs = os.path.abspath(local_archive_dir)
    storage_type = StorageType(config.storage_type)
    if storage_type == StorageType.LOCAL:
        return local_archive_abs

    storage = config.storage
    assert isinstance(storage, S3StorageConfig), (
        f"AutoML S3 archive requires S3StorageConfig, given={type(storage).__name__}"
    )
    key_prefix = f"archives/{folder_name}"
    try:
        client = build_s3_client(storage)
        ensure_s3_bucket(client, storage.bucket_name)
        uri = upload_directory_to_s3(
            client,
            local_archive_abs,
            storage.bucket_name,
            key_prefix,
        )
    except Exception as exc:
        LOG.error(
            f"AutoML S3 upload failed, local bundle retained: {local_archive_abs}, "
            f"error={exc}",
            exc_info=True,
        )
        return local_archive_abs
    remove_local_archive_bundle(local_archive_abs)
    return uri

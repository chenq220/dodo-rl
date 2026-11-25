"""训练进度发布：写本进程环境变量；跨进程时同步到 DODO_PROGRESS_FILE 快照。"""
from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

from common.logger import init_logger

LOG = init_logger("train_task")
TRAINING_STAGE_ENV = "DODO_TRAINING_STAGE"
TRAINING_EPISODE_ENV = "DODO_TRAINING_EPISODE"
TRAINING_PROGRESS_FILE_ENV = "DODO_PROGRESS_FILE"
SAMPLE_TOKEN_ENV = "DODO_SAMPLE_TOKEN"
LEARN_TOKEN_ENV = "DODO_LEARN_TOKEN"
CENTER_FINISHED_ENV = "DODO_CENTER_FINISHED"

_PROGRESS_KEYS = (
    TRAINING_STAGE_ENV,
    TRAINING_EPISODE_ENV,
    SAMPLE_TOKEN_ENV,
    LEARN_TOKEN_ENV,
    CENTER_FINISHED_ENV,
)
_PROGRESS_FILE_LOCK = threading.Lock()


def publish_training_progress(
        *,
        stage: str,
        episode: str,
        last_published: Optional[Tuple[str, str]],
) -> Optional[Tuple[str, str]]:
    """发布当前阶段和局数。无变化时返回 last_published 本身。"""
    progress = (stage, episode)
    os.environ[TRAINING_STAGE_ENV] = stage
    os.environ[TRAINING_EPISODE_ENV] = episode
    if progress == last_published:
        return last_published
    _persist_progress_file({
        TRAINING_STAGE_ENV: stage,
        TRAINING_EPISODE_ENV: episode,
    })
    return progress


def _sanitize_token_value(value: str) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def publish_worker_token(*, kind: str, value: str) -> None:
    """发布本进程当前执行的采样/学习令牌。"""
    if kind == "sample":
        env_name = SAMPLE_TOKEN_ENV
    elif kind == "learn":
        env_name = LEARN_TOKEN_ENV
    else:
        raise ValueError(f"worker token kind must be sample or learn, got {kind!r}")
    os.environ[env_name] = _sanitize_token_value(value)
    _persist_progress_file({env_name: os.environ[env_name]})


def publish_center_finished() -> None:
    """中心关停前把 DODO_CENTER_FINISHED=1 写入本进程环境（及跨进程快照）。"""
    os.environ[CENTER_FINISHED_ENV] = "1"
    _persist_progress_file({CENTER_FINISHED_ENV: "1"})


def is_center_finished() -> bool:
    if os.environ.get(CENTER_FINISHED_ENV, "").strip() == "1":
        return True
    progress_file = os.environ.get(TRAINING_PROGRESS_FILE_ENV, "").strip()
    if not progress_file:
        return False
    return _read_progress_file(progress_file).get(CENTER_FINISHED_ENV, "").strip() == "1"


def _read_progress_file(progress_file: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        with open(progress_file, "r", encoding="utf-8") as stream:
            for raw_line in stream:
                line = raw_line.rstrip("\n")
                if not line or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key] = value
    except OSError:
        return values
    return values


def _persist_progress_file(updates: Dict[str, str]) -> None:
    """有 DODO_PROGRESS_FILE 时把更新合并进快照，供父 shell / 其它进程读环境变量。"""
    progress_file = os.environ.get(TRAINING_PROGRESS_FILE_ENV, "").strip()
    if not progress_file:
        return
    progress_dir = os.path.dirname(os.path.abspath(progress_file))
    snapshot_path = f"{progress_file}.{uuid.uuid4().hex}.tmp"
    with _PROGRESS_FILE_LOCK:
        current = _read_progress_file(progress_file)
        current.update(updates)
        try:
            os.makedirs(progress_dir, exist_ok=True)
            with open(snapshot_path, "w", encoding="utf-8") as stream:
                for key in _PROGRESS_KEYS:
                    if key in current:
                        stream.write(f"{key}={current[key]}\n")
            os.replace(snapshot_path, progress_file)
        except OSError as exc:
            LOG.warning(
                f"Failed to publish training progress: file={progress_file}, error={exc}"
            )
            try:
                os.unlink(snapshot_path)
            except OSError:
                pass


_dispatch_snapshot_path: Optional[Path] = None
_dispatch_stages: dict[str, dict[str, int]] = {}
_dispatch_last_written: Optional[tuple[str, dict[str, int]]] = None


def publish_dispatch_progress(
        *, stage: str, generation: int, sample_dispatched: int, sample_completed: int,
        learn_dispatched: int, sample_total: int,
) -> None:
    """Publish the center's sampling completion and token dispatch counts by stage."""
    global _dispatch_snapshot_path, _dispatch_last_written
    configured = os.environ.get("DODO_DISPATCH_PROGRESS_FILE")
    if not configured:
        return
    target = Path(configured)
    counters = {
        "generation": generation,
        "sample_dispatched": sample_dispatched,
        "sample_completed": sample_completed,
        "learn_dispatched": learn_dispatched,
        "sample_total": sample_total,
    }
    with _PROGRESS_FILE_LOCK:
        if _dispatch_snapshot_path != target:
            _dispatch_snapshot_path = target
            _dispatch_stages.clear()
            _dispatch_last_written = None
        if _dispatch_last_written == (stage, counters) and target.is_file():
            return
        _dispatch_stages[stage] = counters
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": 1, "current_stage": stage, "stages": _dispatch_stages}
            temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            os.replace(temporary, target)
            _dispatch_last_written = (stage, counters)
        except OSError as exc:
            LOG.warning(f"Failed to publish token dispatch progress: file={target}, error={exc}")
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

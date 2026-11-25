"""产物契约、本地归档与 Role 持有的 ArtifactsManager（v0.14.2）。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
import copy
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol
from contextlib import ExitStack, contextmanager

from filelock import FileLock

from common.exception import ArtifactsArchiveException
from common.log_files import SharedLogFiles
from common.process_shutdown import process_close
from config import ArtifactsManagerConfig


class ArtifactType(str, Enum):
    MODEL = "model"
    RESULT = "result"
    RENDER_GIF = "render_gif"
    SAMPLE = "sample"
    REPLAY = "replay"
    AUTOML = "automl"
    REPORT = "report"
    CONFIG = "config"
    LOG = "log"
    CUSTOM = "custom"


@dataclass(frozen=True)
class Artifact:
    artifact_type: ArtifactType
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_type, ArtifactType):
            raise ArtifactsArchiveException(
                operation="contract", detail="artifact_type must be ArtifactType",
            )

    def to_dict(self) -> dict[str, Any]:
        """RPC 边界使用的固定字段；不以自定义字典键表达产物类型。"""
        return {
            "artifact_type": self.artifact_type.value,
            "path": self.path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Artifact:
        """从传输记录恢复契约；路径安全性由归档登记时校验。"""
        try:
            if set(payload) - {"artifact_type", "path", "metadata"}:
                raise ValueError("unexpected artifact fields")
            artifact_type = ArtifactType(payload["artifact_type"])
            path = payload["path"]
            metadata = payload.get("metadata", {})
            if not isinstance(path, str) or not path or not isinstance(metadata, dict):
                raise ValueError("artifact requires a nonempty path and object metadata")
            return cls(artifact_type, path, dict(metadata))
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactsArchiveException(operation="decode", detail=str(exc)) from exc


def ensure_output_parent(path: str | Path) -> Path:
    """本机写文件前建父目录；不写台账。"""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@dataclass(frozen=True)
class ArchivedTask:
    task_id: str
    task_type: str
    archive_dir: str
    stages: tuple[str, ...]
    artifact_count: int


class ArtifactStatus(str, Enum):
    PLANNED = "planned"
    GENERATED = "generated"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class ArtifactRegistration:
    """配置预测的产物壳及其实际文件；查询返回快照，不暴露可变台账。"""

    artifact_type: ArtifactType
    planned_path: str
    status: ArtifactStatus = ArtifactStatus.PLANNED
    artifacts: tuple[Artifact, ...] = ()


_TYPE_DIRECTORIES = {
    ArtifactType.MODEL: "models",
    ArtifactType.RESULT: "results",
    ArtifactType.RENDER_GIF: "recordings",
    ArtifactType.SAMPLE: "samples",
    ArtifactType.REPLAY: "replay",
    ArtifactType.AUTOML: "automl",
    ArtifactType.REPORT: "reports",
    ArtifactType.CONFIG: "config",
    ArtifactType.LOG: "logs",
    ArtifactType.CUSTOM: "custom",
}
_SAFE_COMPONENT = re.compile(r"^[0-9A-Za-z_.-]+$")


class _ArchiveStorage:
    """台账的私有本地存储分区，负责路径安全、原子归档及 manifest 查询。"""

    _ownership_lock = threading.RLock()
    _source_owners: dict[str, tuple[str, str]] = {}

    def __init__(
            self,
            archive_save_dir: str,
            *,
            allowed_runtime_dirs: Iterable[str] = (),
            task_type: str = "task",
    ) -> None:
        self.archive_save_dir = Path(archive_save_dir).expanduser().resolve()
        self.allowed_runtime_dirs = tuple(
            Path(path).expanduser().resolve()
            for path in allowed_runtime_dirs
            if path
        )
        self.task_type = str(task_type)
        self.retain_runtime_sources = False
        self._outputs: dict[tuple[str, str], dict[ArtifactType, set[Path]]] = {}
        self._pending: dict[tuple[str, str], dict[str, Artifact]] = {}
        self._registrations: dict[tuple[str, str], dict[tuple[ArtifactType, str], ArtifactRegistration]] = {}
        self._lock = threading.RLock()

    def allow_runtime_dir(self, path: str) -> None:
        """登记 producer 配置中明确声明的额外输出目录。"""
        root = Path(path).expanduser().resolve()
        if self._is_within(root, self.archive_save_dir) and not self._is_remote_cache_path(root):
            self._error("allow_runtime_dir", f"runtime directory overlaps archive output: {root}")
        with self._lock:
            if root not in self.allowed_runtime_dirs:
                self.allowed_runtime_dirs = (*self.allowed_runtime_dirs, root)

    @staticmethod
    def prepare_output_directory(path: str | Path) -> Path:
        """注册输出或实际写入缓存时准备目录；配置构造和输入路径不调用。"""
        directory = Path(path).expanduser()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ArtifactsArchiveException(
                operation="prepare_output", detail=f"cannot create output directory {directory}: {exc}",
            ) from exc
        return directory

    @classmethod
    def prepare_output(cls, artifact: Artifact, *, directory: bool = False) -> Path:
        """producer 写入前准备路径；文件只建父目录，目录产物显式指定 directory。"""
        path = Path(artifact.path).expanduser()
        cls.prepare_output_directory(path if directory else path.parent)
        return path

    def register(self, task_id: str, stage_name: str, artifact: Artifact) -> None:
        """入口根据配置注册产物壳；不把规划的共享目录本身当作待搬运产物。"""
        task = self._validate_component(task_id, "task_id")
        stage = self._validate_component(stage_name, "stage_name")
        is_config = artifact.artifact_type is ArtifactType.CONFIG
        source = (
            self.archive_save_dir / task / "config" if is_config
            else self._validate_source(artifact.path, must_exist=False, inspect_tree=False)
        )
        with self._lock:
            self._ensure_stage_open(task, stage)
            if not is_config:
                self.prepare_output_directory(source)
                outputs = self._outputs.setdefault((task, stage), {})
                outputs.setdefault(artifact.artifact_type, set()).add(source)
            rows = self._registrations.setdefault((task, stage), {})
            rows.setdefault(
                (artifact.artifact_type, str(source)),
                ArtifactRegistration(artifact.artifact_type, str(source)),
            )

    def get_registered_artifacts(self, task_id: str, stage_name: str) -> list[ArtifactRegistration]:
        with self._lock:
            return copy.deepcopy(list(self._registrations.get((task_id, stage_name), {}).values()))

    def record(self, task_id: str, stage_name: str, artifact: Artifact) -> None:
        """运行中记录实际生成的文件或代次目录；必须位于初始化时注册的对应类型目录下。"""
        task = self._validate_component(task_id, "task_id")
        stage = self._validate_component(stage_name, "stage_name")
        source = self._validate_source(artifact.path)
        self._ensure_stage_open(task, stage)
        normalized = Artifact(
            artifact_type=artifact.artifact_type,
            path=str(source),
            metadata=copy.deepcopy(artifact.metadata or {}),
        )
        try:
            json.dumps(normalized.metadata, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            self._error("record", f"artifact metadata is not JSON serializable: {exc}")

        owner = (task, stage)
        with self._lock:
            self._ensure_stage_open(task, stage)
            roots = self._outputs.get(owner, {}).get(artifact.artifact_type, set())
            if not any(source != root and self._is_within(source, root) for root in roots):
                self._error("record", f"artifact output is not registered for {artifact.artifact_type.value}: {source}")
            bucket = self._pending.setdefault(owner, {})
            existing = bucket.get(str(source))
            if existing is not None:
                if existing.artifact_type is not normalized.artifact_type:
                    self._error("record", f"source recorded with conflicting contract: {source}")
                normalized = Artifact(
                    normalized.artifact_type, normalized.path,
                    {**existing.metadata, **normalized.metadata},
                )
            with self._ownership_lock:
                existing_owner = self._source_owners.get(str(source))
                if existing_owner is not None and existing_owner != owner:
                    self._error(
                        "record",
                        f"source already belongs to task/stage {existing_owner}: {source}",
                    )
                self._source_owners[str(source)] = owner
                bucket[str(source)] = normalized
            rows = self._registrations[owner]
            root = max((root for root in roots if self._is_within(source, root)), key=lambda root: len(root.parts))
            key = (normalized.artifact_type, str(root))
            row = rows[key]
            files = {item.path: item for item in row.artifacts}
            files[normalized.path] = normalized
            rows[key] = ArtifactRegistration(row.artifact_type, row.planned_path, ArtifactStatus.GENERATED, tuple(files.values()))

    def archive_task(
            self, task_id: str, stage_name: str, *, raw_file: str | None, config_name: str | None,
    ) -> ArchivedTask:
        with self.archive_lock(task_id):
            return self._archive_task(task_id, stage_name, raw_file=raw_file, config_name=config_name)

    @contextmanager
    def archive_lock(self, task_id: str):
        task = self._validate_component(task_id, "task_id")
        # 锁是跨进程协调元数据，不属于归档产物；放在归档根目录之外，避免
        # `.task.archive.lock` 出现在 `artifacts_archive_save_dir` 的内容中。
        lock_dir = self.archive_save_dir.parent / f".{self.archive_save_dir.name or 'archives'}.locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(lock_dir / f".{task}.archive.lock")):
            yield

    def _archive_task(
            self,
            task_id: str,
            stage_name: str,
            *,
            raw_file: str | None,
            config_name: str | None,
    ) -> ArchivedTask:
        task = self._validate_component(task_id, "task_id")
        stage = self._validate_component(stage_name, "stage_name")
        owner = (task, stage)
        with self._lock:
            pending = list(self._pending.get(owner, {}).values())
            target = self.archive_save_dir / task
            manifest = self._load_manifest(target, allow_missing=True)
            if manifest is not None and stage in manifest.get("stages", {}):
                if pending:
                    self._error("archive", f"stage already archived and has new artifacts: {task}/{stage}")
                return self._archived_task(target, manifest)

            if manifest is None:
                manifest = {
                    "task_id": task,
                    "task_type": self.task_type,
                    "created_at": datetime.now().astimezone().isoformat(),
                    "stages": {},
                    "artifacts": [],
                }
            elif manifest.get("task_id") != task:
                self._error("archive", f"manifest task_id mismatch under {target}")
            elif manifest.get("task_type") != self.task_type:
                manifest["task_type"] = "mixed"

            staging = self.archive_save_dir / f".{task}.{stage}.{uuid.uuid4().hex}.tmp"
            committed: list[Path] = []
            new_entries: list[dict[str, Any]] = []
            target_existed = target.exists()
            try:
                self.archive_save_dir.mkdir(parents=True, exist_ok=True)
                staging.mkdir(parents=False, exist_ok=False)
                for artifact in pending:
                    source = self._validate_source(artifact.path)
                    relative = self._destination_relative(stage, artifact, manifest, new_entries)
                    staged_path = staging / relative
                    staged_path.parent.mkdir(parents=True, exist_ok=True)
                    if artifact.artifact_type is ArtifactType.LOG:
                        self._copy_log_snapshot(source, staged_path)
                    else:
                        self._copy_source(source, staged_path)
                    size, checksum = self._measure(staged_path)
                    new_entries.append({
                        "artifact_type": artifact.artifact_type.value,
                        "stage": stage,
                        "relative_path": relative.as_posix(),
                        "size": size,
                        "sha256": checksum,
                        "metadata": dict(artifact.metadata),
                    })

                config_entry = self._stage_raw_config(
                    staging,
                    manifest,
                    task=task,
                    stage=stage,
                    raw_file=raw_file,
                    config_name=config_name,
                )
                if config_entry is not None:
                    new_entries.append(config_entry)

                target.mkdir(parents=True, exist_ok=True)
                for entry in new_entries:
                    relative = Path(entry["relative_path"])
                    staged_path = staging / relative
                    final_path = target / relative
                    if not staged_path.exists():
                        continue
                    if final_path.exists():
                        self._error("archive", f"archive destination already exists: {final_path}")
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staged_path, final_path)
                    committed.append(final_path)

                archived_at = datetime.now().astimezone().isoformat()
                manifest["stages"][stage] = {
                    "archived_at": archived_at,
                    "artifact_count": len(new_entries),
                }
                manifest["artifacts"].extend(new_entries)
                self._write_manifest(target, manifest)
            except ArtifactsArchiveException:
                self._rollback_committed(committed)
                if not target_existed:
                    shutil.rmtree(target, ignore_errors=True)
                raise
            except Exception as exc:
                self._rollback_committed(committed)
                if not target_existed:
                    shutil.rmtree(target, ignore_errors=True)
                self._error("archive", str(exc))
            finally:
                shutil.rmtree(staging, ignore_errors=True)

            for artifact in pending:
                if self.retain_runtime_sources:
                    continue
                # 共享分片也可能被别的进程/阶段归档，保留源文件以免后续归档缺失。
                if artifact.artifact_type is not ArtifactType.LOG or (
                    artifact.metadata.get("closed", False) and not artifact.metadata.get("shared_log")
                ):
                    self._remove_runtime_source(Path(artifact.path))
            with self._ownership_lock:
                for artifact in pending:
                    self._source_owners.pop(artifact.path, None)
            self._pending.pop(owner, None)
            destinations = {
                artifact.path: Artifact(
                    artifact.artifact_type,
                    str(target / entry["relative_path"]),
                    dict(artifact.metadata),
                )
                for artifact, entry in zip(pending, new_entries)
            }
            rows = self._registrations.get(owner, {})
            for key, row in list(rows.items()):
                files = tuple(destinations[item.path] for item in row.artifacts)
                if row.artifact_type is ArtifactType.CONFIG:
                    files = tuple(
                        Artifact(ArtifactType.CONFIG, str(target / entry["relative_path"]), dict(entry["metadata"]))
                        for entry in manifest["artifacts"] if entry["artifact_type"] == ArtifactType.CONFIG.value
                    )
                if files:
                    rows[key] = ArtifactRegistration(row.artifact_type, row.planned_path, ArtifactStatus.ARCHIVED, files)
            return self._archived_task(target, manifest)

    def list_archived_tasks(self) -> list[ArchivedTask]:
        tasks: list[ArchivedTask] = []
        if not self.archive_save_dir.is_dir():
            return tasks
        for path in sorted(self.archive_save_dir.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or path.name.startswith("."):
                continue
            try:
                manifest = self._load_manifest(path, allow_missing=False)
            except ArtifactsArchiveException:
                continue
            tasks.append(self._archived_task(path, manifest))
        return tasks

    def get_task_artifacts(self, task_id: str) -> list[Artifact]:
        task = self._validate_component(task_id, "task_id")
        target = self.archive_save_dir / task
        manifest = self._load_manifest(target, allow_missing=False)
        return [
            Artifact(
                artifact_type=ArtifactType(entry["artifact_type"]),
                path=str((target / entry["relative_path"]).resolve()),
                metadata=dict(entry.get("metadata") or {}),
            )
            for entry in manifest["artifacts"]
        ]

    def _ensure_stage_open(self, task: str, stage: str) -> None:
        manifest = self._load_manifest(self.archive_save_dir / task, allow_missing=True)
        if manifest is not None and stage in manifest.get("stages", {}):
            self._error("register", f"stage already archived: {task}/{stage}")

    def _validate_source(
            self, raw_path: str, *, must_exist: bool = True, inspect_tree: bool = True,
    ) -> Path:
        source = Path(raw_path).expanduser()
        try:
            resolved = source.resolve(strict=must_exist)
        except (FileNotFoundError, OSError) as exc:
            self._error("register", f"artifact source does not exist: {source}: {exc}")
        if self._is_within(resolved, self.archive_save_dir) and not self._is_remote_cache_path(resolved):
            self._error("register", f"archive output cannot be registered as runtime input: {resolved}")
        # 空白名单：模块自 put 场景，只禁止写进归档根；有白名单时仍须落在允许目录内。
        if self.allowed_runtime_dirs and not any(
                self._is_within(resolved, root) for root in self.allowed_runtime_dirs
        ):
            self._error("register", f"artifact source is outside allowed runtime directories: {resolved}")
        if inspect_tree and resolved.is_dir() and self.allowed_runtime_dirs:
            for child in resolved.rglob("*"):
                try:
                    child_resolved = child.resolve(strict=True)
                except (FileNotFoundError, OSError) as exc:
                    self._error("register", f"invalid artifact tree entry {child}: {exc}")
                if not any(self._is_within(child_resolved, root) for root in self.allowed_runtime_dirs):
                    self._error("register", f"artifact symlink escapes allowed runtime directories: {child}")
        return resolved

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _is_remote_cache_path(self, path: Path) -> bool:
        """AM 自有 _remote_cache 在归档根下，不算生产者 runtime 目录。"""
        cache_root = (self.archive_save_dir / REMOTE_CACHE_DIRNAME).resolve()
        return path == cache_root or self._is_within(path, cache_root)

    @staticmethod
    def _copy_log_snapshot(source: Path, destination: Path) -> None:
        """只复制打开源文件时已有的字节，日志持续增长也不会拖住归档。"""
        with source.open("rb") as reader, destination.open("xb") as writer:
            remaining = os.fstat(reader.fileno()).st_size
            while remaining:
                chunk = reader.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise OSError(f"log truncated while archiving: {source}")
                writer.write(chunk)
                remaining -= len(chunk)

    @staticmethod
    def _copy_source(source: Path, destination: Path) -> None:
        if source.is_dir():
            # 台账硬约束：一文件一行；目录行在 record/put 入口已拒绝，这里是归档侧兜底。
            raise ArtifactsArchiveException(
                operation="archive",
                detail=f"directory artifacts must be put per-file, not as a directory: {source}",
            )
        shutil.copy2(source, destination)

    def _destination_relative(
            self,
            stage: str,
            artifact: Artifact,
            manifest: dict[str, Any],
            pending_entries: list[dict[str, Any]],
    ) -> Path:
        directory = _TYPE_DIRECTORIES[artifact.artifact_type]
        relative = Path(directory) / stage
        if artifact.artifact_type is ArtifactType.RESULT and artifact.metadata.get("category") == "plot_csv":
            relative /= "raw_data"
        relative /= Path(artifact.path).name
        existing = {entry["relative_path"] for entry in manifest.get("artifacts", [])}
        existing.update(entry["relative_path"] for entry in pending_entries)
        if relative.as_posix() in existing:
            self._error("archive", f"duplicate archive destination: {relative}")
        return relative

    def _stage_raw_config(
            self,
            staging: Path,
            manifest: dict[str, Any],
            *,
            task: str,
            stage: str,
            raw_file: str | None,
            config_name: str | None,
    ) -> dict[str, Any] | None:
        if raw_file is None:
            return None
        digest = hashlib.sha256(raw_file.encode("utf-8")).hexdigest()
        existing_digest = manifest.get("config_sha256")
        if existing_digest is not None:
            if existing_digest != digest:
                self._error("archive", f"raw TOML changed between stages for task {task}")
            return None
        stem = Path(config_name or "config.toml").stem or "config"
        safe_stem = re.sub(r"[^0-9A-Za-z_.-]+", "_", stem).strip("_") or "config"
        timestamp = datetime.now().astimezone().strftime("%Y%m%d%H%M%S")
        relative = Path("config") / f"{safe_stem}_{timestamp}.toml"
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(raw_file, encoding="utf-8")
        size, checksum = self._measure(destination)
        manifest["config_sha256"] = digest
        return {
            "artifact_type": ArtifactType.CONFIG.value,
            "stage": stage,
            "relative_path": relative.as_posix(),
            "size": size,
            "sha256": checksum,
            "metadata": {"source_name": str(config_name or "")},
        }

    @staticmethod
    def _measure(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        if path.is_file():
            data = path.read_bytes()
            return len(data), hashlib.sha256(data).hexdigest()
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(relative)
            with child.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
        return size, digest.hexdigest()

    def _remove_runtime_source(self, source: Path) -> None:
        try:
            if source.is_dir():
                shutil.rmtree(source)
            elif source.exists():
                source.unlink()
        except OSError:
            return
        parent = source.parent
        stop_roots = set(self.allowed_runtime_dirs)
        while parent not in stop_roots and any(
                self._is_within(parent, root) for root in self.allowed_runtime_dirs
        ):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    @staticmethod
    def _rollback_committed(paths: list[Path]) -> None:
        for path in reversed(paths):
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            except OSError:
                pass

    def _load_manifest(self, target: Path, *, allow_missing: bool) -> dict[str, Any] | None:
        path = target / "manifest.json"
        if not path.exists():
            if allow_missing and not target.exists():
                return None
            if allow_missing and target.is_dir() and not any(target.iterdir()):
                return None
            if allow_missing:
                self._error("load_manifest", f"archive directory has no manifest: {target}")
            self._error("get_task", f"task archive does not exist or has no manifest: {target.name}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(manifest.get("stages"), dict) or not isinstance(manifest.get("artifacts"), list):
                raise ValueError("missing stages/artifacts")
            if manifest.get("task_id") != target.name:
                raise ValueError("task_id does not match archive directory")
            for entry in manifest["artifacts"]:
                if not isinstance(entry, dict):
                    raise ValueError("artifact entry must be an object")
                ArtifactType(entry.get("artifact_type", ""))
                stage = str(entry.get("stage", ""))
                if stage not in manifest["stages"]:
                    raise ValueError(f"artifact references unknown stage: {stage!r}")
                relative = Path(str(entry.get("relative_path", "")))
                if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                    raise ValueError(f"unsafe artifact relative_path: {relative}")
                artifact_path = (target / relative).resolve(strict=True)
                if not self._is_within(artifact_path, target.resolve()):
                    raise ValueError(f"artifact path escapes task archive: {relative}")
            return manifest
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ArtifactsArchiveException) as exc:
            self._error("load_manifest", f"invalid manifest {path}: {exc}")

    @staticmethod
    def _write_manifest(target: Path, manifest: dict[str, Any]) -> None:
        temp = target / f".manifest.{uuid.uuid4().hex}.tmp"
        temp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, target / "manifest.json")

    @staticmethod
    def _archived_task(target: Path, manifest: dict[str, Any]) -> ArchivedTask:
        return ArchivedTask(
            task_id=str(manifest["task_id"]),
            task_type=str(manifest.get("task_type", "task")),
            archive_dir=str(target.resolve()),
            stages=tuple(manifest.get("stages", {}).keys()),
            artifact_count=len(manifest.get("artifacts", [])),
        )

    @staticmethod
    def _validate_component(value: str, name: str) -> str:
        normalized = str(value)
        if not normalized or not _SAFE_COMPONENT.fullmatch(normalized) or normalized in {".", ".."}:
            raise ArtifactsArchiveException(
                operation="validate",
                detail=f"unsafe {name}: {value!r}",
            )
        return normalized

    @staticmethod
    def _error(operation: str, detail: str) -> None:
        raise ArtifactsArchiveException(operation=operation, detail=detail)


class ArtifactsArchive:
    """进程级产物台账；入口注册计划，各模块补全，TaskManager 统一归档。"""

    prepare_output = _ArchiveStorage.prepare_output
    prepare_output_directory = staticmethod(_ArchiveStorage.prepare_output_directory)

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: dict[tuple[str, str], _ArchiveStorage] = {}
        self._raw_files: dict[tuple[str, str], tuple[str | None, str | None]] = {}
        # 日志早于入口预注册产生，且跨阶段持续写入；先按任务保留实际文件契约。
        self._task_logs: dict[str, dict[str, Artifact]] = {}

    def register_task(
            self, task_id: str, stage_name: str, *, archive_save_dir: str,
            allowed_runtime_dirs: Iterable[str], task_type: str,
            raw_file: str | None = None, config_name: str | None = None,
    ) -> None:
        """入口建立任务/阶段分区；导入单例和本方法本身均不创建归档目录。"""
        task = _ArchiveStorage._validate_component(task_id, "task_id")
        stage = _ArchiveStorage._validate_component(stage_name, "stage_name")
        owner = (task, stage)
        root = Path(archive_save_dir).expanduser().resolve()
        with self._lock:
            existing = self._tasks.get(owner)
            if existing is not None:
                if existing.archive_save_dir != root or existing.task_type != task_type:
                    _ArchiveStorage._error("register_task", f"conflicting task plan: {task}/{stage}")
                if self._raw_files[owner] != (raw_file, config_name):
                    _ArchiveStorage._error("register_task", f"configuration changed for task plan: {task}/{stage}")
                return
            self._tasks[owner] = _ArchiveStorage(
                str(root), allowed_runtime_dirs=allowed_runtime_dirs, task_type=task_type,
            )
            self._raw_files[owner] = (raw_file, config_name)

    def _store(self, task_id: str, stage_name: str) -> _ArchiveStorage:
        with self._lock:
            store = self._tasks.get((task_id, stage_name))
            if store is None:
                _ArchiveStorage._error(
                    "lookup", f"task artifacts are not planned: {task_id}/{stage_name}; register at the entry before creating Role",
                )
            return store

    def register(self, task_id: str, stage_name: str, artifact: Artifact) -> None:
        self._store(task_id, stage_name).register(task_id, stage_name, artifact)

    def record(self, task_id: str, stage_name: str, artifact: Artifact) -> None:
        """补全已有壳中的实际文件；同路径允许更新元数据，不允许改变产物类型。"""
        self._store(task_id, stage_name).record(task_id, stage_name, artifact)

    def record_log(self, task_id: str, artifact: Artifact) -> None:
        """首次写入或轮转时补全；共享文件的分片通过持久化索引供各进程查询。"""
        task = _ArchiveStorage._validate_component(task_id, "task_id")
        if artifact.artifact_type is not ArtifactType.LOG:
            _ArchiveStorage._error("record_log", "expected ArtifactType.LOG")
        with self._lock, ExitStack() as locks:
            roots = [
                Path(row.planned_path)
                for (owner_task, stage), store in self._tasks.items() if owner_task == task
                for row in store.get_registered_artifacts(task, stage)
                if row.artifact_type is ArtifactType.LOG
            ]
            shared_base = artifact.metadata.get("shared_log")
            if shared_base:
                base = Path(shared_base).expanduser().resolve()
                if roots and not any(_ArchiveStorage._is_within(base, root) for root in roots):
                    _ArchiveStorage._error("record_log", f"shared log is outside registered directories: {base}")
                # 获取台账锁后再获取文件锁，与归档端一致；避开其他进程轮转时的 rename 窗口。
                locks.enter_context(SharedLogFiles(base).locked())
            try:
                source = Path(artifact.path).expanduser().resolve(strict=True)
            except OSError as exc:
                raise ArtifactsArchiveException(operation="record_log", detail=str(exc)) from exc
            if not source.is_file():
                _ArchiveStorage._error("record_log", f"log is not a file: {source}")
            if roots and not any(_ArchiveStorage._is_within(source, root) for root in roots):
                _ArchiveStorage._error("record_log", f"log is outside registered directories: {source}")
            if any(str(source) in files for other, files in self._task_logs.items() if other != task):
                _ArchiveStorage._error("record_log", f"log already belongs to another task: {source}")
            self._task_logs.setdefault(task, {})[str(source)] = Artifact(
                ArtifactType.LOG, str(source), copy.deepcopy(artifact.metadata),
            )

    def _shared_log_paths(self, task_id: str, rows: list[ArtifactRegistration]) -> list[Path]:
        roots = [Path(row.planned_path) for row in rows if row.artifact_type is ArtifactType.LOG]
        return sorted({
            Path(item.metadata["shared_log"]).resolve()
            for item in self._task_logs.get(task_id, {}).values()
            if item.metadata.get("shared_log")
            and any(_ArchiveStorage._is_within(Path(item.metadata["shared_log"]).resolve(), root) for root in roots)
        })

    def _logs_for_stage(
        self, task_id: str, rows: list[ArtifactRegistration], *, files_locked: bool = False,
    ) -> list[Artifact]:
        roots = [Path(row.planned_path) for row in rows if row.artifact_type is ArtifactType.LOG]
        logs = {item.path: copy.deepcopy(item) for item in self._task_logs.get(task_id, {}).values()
                if not item.metadata.get("shared_log")
                and any(_ArchiveStorage._is_within(Path(item.path), root) for root in roots)}
        for base in self._shared_log_paths(task_id, rows):
            files = SharedLogFiles(base)
            with ExitStack() as locks:
                if not files_locked:
                    locks.enter_context(files.locked())
                for path in files.files():
                    logs[str(path)] = Artifact(
                        ArtifactType.LOG, str(path), {"shared_log": str(base), "closed": path != base},
                    )
        return list(logs.values())

    def get_registered_artifacts(self, task_id: str, stage_name: str) -> list[ArtifactRegistration]:
        with self._lock:
            rows = self._store(task_id, stage_name).get_registered_artifacts(task_id, stage_name)
            for index, row in enumerate(rows):
                if row.artifact_type is not ArtifactType.LOG or row.status is ArtifactStatus.ARCHIVED:
                    continue
                files = {item.path: item for item in row.artifacts}
                files.update({item.path: item for item in self._logs_for_stage(task_id, [row])})
                if files:
                    rows[index] = ArtifactRegistration(
                        row.artifact_type, row.planned_path, ArtifactStatus.GENERATED, tuple(files.values()),
                    )
            return rows

    def archive_task(self, task_id: str, stage_name: str) -> ArchivedTask:
        with self._lock:
            store = self._store(task_id, stage_name)
            with store.archive_lock(task_id), ExitStack() as locks:
                raw_file, config_name = self._raw_files[(task_id, stage_name)]
                manifest = store._load_manifest(store.archive_save_dir / task_id, allow_missing=True)
                # 先取得跨进程归档锁，再判断幂等，避免两个进程覆盖同一 manifest。
                if manifest and stage_name in manifest["stages"]:
                    return store._archived_task(store.archive_save_dir / task_id, manifest)
                rows = store.get_registered_artifacts(task_id, stage_name)
                for base in self._shared_log_paths(task_id, rows):
                    locks.enter_context(SharedLogFiles(base).locked())
                logs = self._logs_for_stage(task_id, rows, files_locked=True)
                for artifact in logs:
                    store.record(task_id, stage_name, artifact)
                archived = store._archive_task(task_id, stage_name, raw_file=raw_file, config_name=config_name)
                for artifact in logs:
                    if artifact.metadata.get("closed", False) and not artifact.metadata.get("shared_log"):
                        self._task_logs[task_id].pop(artifact.path, None)
                return archived

    def list_archived_tasks(self, archive_save_dir: str | None = None) -> list[ArchivedTask]:
        with self._lock:
            roots = {store.archive_save_dir for store in self._tasks.values()}
        if archive_save_dir is not None:
            roots = {Path(archive_save_dir).expanduser().resolve()}
        return [task for root in sorted(roots) for task in _ArchiveStorage(str(root)).list_archived_tasks()]

    def get_task_artifacts(self, task_id: str, archive_save_dir: str | None = None) -> list[Artifact]:
        with self._lock:
            roots = {store.archive_save_dir for (task, _), store in self._tasks.items() if task == task_id}
        if archive_save_dir is not None:
            roots = {Path(archive_save_dir).expanduser().resolve()}
        if not roots:
            _ArchiveStorage._error("query", f"task not found: {task_id}")
        return [artifact for root in sorted(roots) for artifact in _ArchiveStorage(str(root)).get_task_artifacts(task_id)]


# --- ArtifactsManager（Role 持有） ---

REMOTE_CACHE_DIRNAME = "_remote_cache"

# put_bytes 单次输送上限；GIF 等大产物超过该值直接拒绝，不进台账。
PUT_BYTES_MAX_BYTES = 100 * 1024 * 1024

_active_lock = threading.RLock()
_active_manager: Optional["ArtifactsManager"] = None


class ArtifactOrigin(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


class ArtifactLedgerStatus(str, Enum):
    REGISTERED = "registered"
    AVAILABLE = "available"
    ARCHIVED = "archived"


@dataclass(frozen=True)
class ArtifactIntent:
    """register 载荷：声明将产出的类型；不是文件清单。"""

    artifact_type: ArtifactType
    planned_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_type, ArtifactType):
            raise ArtifactsArchiveException(
                operation="contract", detail="artifact_type must be ArtifactType",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type.value,
            "planned_hint": self.planned_hint,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactIntent:
        try:
            if set(payload) - {"artifact_type", "planned_hint", "metadata"}:
                raise ValueError("unexpected artifact intent fields")
            artifact_type = ArtifactType(payload["artifact_type"])
            planned_hint = payload.get("planned_hint")
            metadata = payload.get("metadata", {})
            if planned_hint is not None and not isinstance(planned_hint, str):
                raise ValueError("planned_hint must be a string or null")
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be an object")
            return cls(artifact_type, planned_hint, dict(metadata))
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactsArchiveException(operation="decode", detail=str(exc)) from exc


@dataclass(frozen=True)
class ArtifactPut:
    """put 载荷：生产者本机上的单个文件路径。"""

    artifact_type: ArtifactType
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_type, ArtifactType):
            raise ArtifactsArchiveException(
                operation="contract", detail="artifact_type must be ArtifactType",
            )
        if not isinstance(self.path, str) or not self.path:
            raise ArtifactsArchiveException(
                operation="contract", detail="ArtifactPut.path must be a nonempty string",
            )
        if not isinstance(self.metadata, dict):
            raise ArtifactsArchiveException(
                operation="contract", detail="ArtifactPut.metadata must be a dict",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type.value,
            "path": self.path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactPut:
        try:
            if set(payload) - {"artifact_type", "path", "metadata"}:
                raise ValueError("unexpected artifact put fields")
            return cls(
                ArtifactType(payload["artifact_type"]),
                payload["path"],
                dict(payload.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError, ArtifactsArchiveException) as exc:
            if isinstance(exc, ArtifactsArchiveException):
                raise
            raise ArtifactsArchiveException(operation="decode", detail=str(exc)) from exc


class ArtifactRegisterCaller(Protocol):
    """Role 绑到 Manager.register 的闭包；生产者只传 Intent。"""

    def __call__(self, intent: ArtifactIntent) -> Any:
        ...


class ArtifactPutCaller(Protocol):
    """Role 绑到 Manager.put 的闭包；生产者只传已生成文件。"""

    def __call__(self, item: ArtifactPut) -> Any:
        ...


class ArtifactCompleteCaller(Protocol):
    """Role 绑到 Manager.complete 的闭包；本模块本阶段不会再 put。"""

    def __call__(self) -> Any:
        ...


class ArtifactCallerSupport:
    """产物模块侧 register/put/complete；未接线则跳过。生产者在 ``__init__`` 里调 ``_init_artifact_callers``。"""

    # Role 接到 ArtifactsManager.register 的闭包；未接线时登记意图直接跳过。
    register_caller: ArtifactRegisterCaller | None = None
    # Role 接到 ArtifactsManager.put 的闭包；未接线时输送文件直接跳过。
    put_caller: ArtifactPutCaller | None = None
    # Role 接到 ArtifactsManager.complete 的闭包；未接线时完成本阶段上报直接跳过。
    complete_caller: ArtifactCompleteCaller | None = None
    # 已登记过的 (产物类型, planned_hint)，避免同一意图重复 register。
    _registered_intent_keys: set[tuple[ArtifactType, str | None]]

    def _init_artifact_callers(self) -> None:
        self.register_caller = None
        self.put_caller = None
        self.complete_caller = None
        self._registered_intent_keys = set()

    def register_stage_intents(self) -> None:
        """装配完成、caller 已接上后登记本阶段产物意图。子类覆盖；默认无意图。"""

    def reset_stage_artifact_intents(self) -> None:
        """多阶段进入新阶段时清除本模块已登记意图，再按本阶段重新 register。"""
        self._registered_intent_keys.clear()
        self.register_stage_intents()

    def register_artifact_intent(
            self,
            artifact_type: ArtifactType,
            planned_hint: str | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> None:
        key = (artifact_type, planned_hint)
        if key in self._registered_intent_keys:
            return
        caller = getattr(self, "register_caller", None)
        if caller is None:
            return
        caller(ArtifactIntent(artifact_type, planned_hint, dict(metadata or {})))
        self._registered_intent_keys.add(key)

    def put_artifact_file(
            self,
            artifact_type: ArtifactType,
            path: str,
            metadata: dict[str, Any] | None = None,
    ) -> None:
        caller = getattr(self, "put_caller", None)
        if caller is None:
            return
        caller(ArtifactPut(artifact_type, str(path), dict(metadata or {})))

    def complete_artifact_stage(self) -> None:
        """本模块本阶段不会再 put；未接线则跳过。可重复调用。"""
        caller = getattr(self, "complete_caller", None)
        if caller is None:
            return
        caller()


@dataclass(frozen=True)
class LedgerEntry:
    """中心台账权威行；AVAILABLE 的 center_path 在 master 上可 open。"""

    entry_id: str
    task_id: str
    stage_name: str
    artifact_type: ArtifactType
    status: ArtifactLedgerStatus
    origin: ArtifactOrigin | None
    center_path: str | None
    producer_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "task_id": self.task_id,
            "stage_name": self.stage_name,
            "artifact_type": self.artifact_type.value,
            "status": self.status.value,
            "origin": None if self.origin is None else self.origin.value,
            "center_path": self.center_path,
            "producer_path": self.producer_path,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LedgerEntry:
        try:
            allowed = {
                "entry_id", "task_id", "stage_name", "artifact_type", "status",
                "origin", "center_path", "producer_path", "metadata",
            }
            if set(payload) - allowed:
                raise ValueError("unexpected ledger entry fields")
            origin_raw = payload.get("origin")
            return cls(
                entry_id=payload["entry_id"],
                task_id=payload["task_id"],
                stage_name=payload["stage_name"],
                artifact_type=ArtifactType(payload["artifact_type"]),
                status=ArtifactLedgerStatus(payload["status"]),
                origin=None if origin_raw is None else ArtifactOrigin(origin_raw),
                center_path=payload.get("center_path"),
                producer_path=payload.get("producer_path"),
                metadata=dict(payload.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactsArchiveException(operation="decode", detail=str(exc)) from exc


def get_active_artifacts_manager() -> Optional["ArtifactsManager"]:
    with _active_lock:
        return _active_manager


def set_active_artifacts_manager(manager: Optional["ArtifactsManager"]) -> None:
    global _active_manager
    with _active_lock:
        _active_manager = manager


def reset_active_artifacts_manager_for_tests(
        config: ArtifactsManagerConfig | None = None,
) -> "ArtifactsManager":
    """测试隔离：新建并激活空 Manager。"""
    manager = ArtifactsManager(config or ArtifactsManagerConfig())
    set_active_artifacts_manager(manager)
    return manager


def resolve_artifacts_manager_config(
        *,
        artifacts_manager_config: ArtifactsManagerConfig | None = None,
        artifacts_archive_save_dir: str | None = None,
) -> ArtifactsManagerConfig:
    """新节优先，否则旧 TaskConfig.save_dir，否则类默认。"""
    if artifacts_manager_config is not None:
        return ArtifactsManagerConfig(save_dir=artifacts_manager_config.save_dir)
    if artifacts_archive_save_dir:
        return ArtifactsManagerConfig(save_dir=artifacts_archive_save_dir)
    return ArtifactsManagerConfig()


def build_artifacts_manager(
        *,
        artifacts_manager_config: ArtifactsManagerConfig | None = None,
        artifacts_archive_save_dir: str | None = None,
        raw_file: str | None = None,
        config_name: str | None = None,
        activate: bool = True,
) -> "ArtifactsManager":
    config = resolve_artifacts_manager_config(
        artifacts_manager_config=artifacts_manager_config,
        artifacts_archive_save_dir=artifacts_archive_save_dir,
    )
    manager = ArtifactsManager(config)
    if raw_file is not None or config_name is not None:
        manager.bind_raw_config(raw_file, config_name)
    if activate:
        set_active_artifacts_manager(manager)
    return manager


class _StageArchiveGate:
    """本阶段产物模块 complete 门闩：expected 收齐后才 ARCHIVABLE。"""

    def __init__(self) -> None:
        self.expected: set[str] = set()
        self.completed: set[str] = set()

    def is_archivable(self) -> bool:
        return self.expected <= self.completed


class ArtifactsManager:
    """Role 持有的产物台账；TM 直接注入并调用 archive_task。"""

    prepare_output = staticmethod(ArtifactsArchive.prepare_output)
    prepare_output_directory = staticmethod(ArtifactsArchive.prepare_output_directory)

    def __init__(self, config: ArtifactsManagerConfig | None = None) -> None:
        self.config = config or ArtifactsManagerConfig()
        self._backend = ArtifactsArchive()
        self._lock = threading.RLock()
        self._ledger: dict[str, LedgerEntry] = {}
        self._path_index: dict[tuple[str, str, str], str] = {}
        self._intent_keys: set[tuple[str, str, ArtifactType, str | None]] = set()
        self._stage_gates: dict[tuple[str, str], _StageArchiveGate] = {}
        self._raw_file: str | None = None
        self._config_name: str | None = None
        self._default_task_type = "training"
        self._closed = False
        self._runtime_roots: dict[str, Path] = {}
        self._archived_stages: set[tuple[str, str]] = set()
        self._archive_ready = threading.Condition(self._lock)

    def bind_task_runtime_directory(self, task_id: str, path: str | Path) -> None:
        """声明任务独占目录；仅在全部已规划阶段归档成功、close 时删除。"""
        self._require_open()
        task = _ArchiveStorage._validate_component(task_id, "task_id")
        root = Path(path).expanduser().resolve()
        if root.name != task:
            _ArchiveStorage._error("bind_runtime", "runtime directory must be named after task_id")
        with self._lock:
            existing = self._runtime_roots.get(task)
            if existing is not None and existing != root:
                _ArchiveStorage._error("bind_runtime", f"conflicting runtime directory for {task}")
            self._runtime_roots[task] = root

    @property
    def backend(self) -> ArtifactsArchive:
        """兼容层 / 测试访问内部归档台账。"""
        return self._backend

    @property
    def remote_cache_dir(self) -> Path:
        return Path(self.config.save_dir).expanduser() / REMOTE_CACHE_DIRNAME

    def bind_raw_config(self, raw_file: str | None, config_name: str | None = None) -> None:
        self._raw_file = raw_file
        self._config_name = config_name

    def plan_task_stages(
            self, task_id: str, stage_names: Iterable[str], *, archive_save_dir: str,
    ) -> None:
        """提前登记本次运行的全部阶段；多阶段归档只复制，任务结束后统一清理。"""
        self._require_open()
        stages = tuple(stage_names)
        with self._lock:
            for stage in stages:
                self.ensure_task(task_id, stage, archive_save_dir=archive_save_dir)
            if len(stages) > 1:
                for stage in stages:
                    self._backend._store(task_id, stage).retain_runtime_sources = True

    def ensure_task(
            self,
            task_id: str,
            stage_name: str,
            *,
            save_dir: str | None = None,
            archive_save_dir: str | None = None,
            allowed_runtime_dirs: list[str] | tuple[str, ...] = (),
            task_type: str | None = None,
            raw_file: str | None = None,
            config_name: str | None = None,
    ) -> None:
        self._require_open()
        task = _ArchiveStorage._validate_component(task_id, "task_id")
        stage = _ArchiveStorage._validate_component(stage_name, "stage_name")
        owner = (task, stage)
        explicit_dir = archive_save_dir or save_dir
        with self._backend._lock:
            existing = self._backend._tasks.get(owner)
            if existing is not None:
                if explicit_dir is not None:
                    root = Path(explicit_dir).expanduser().resolve()
                    if existing.archive_save_dir != root:
                        raise ArtifactsArchiveException(
                            operation="register_task",
                            detail=f"conflicting task plan: {task}/{stage}",
                        )
                for directory in allowed_runtime_dirs:
                    existing.allow_runtime_dir(directory)
                incoming_raw = (
                    raw_file if raw_file is not None else self._raw_file,
                    config_name if config_name is not None else self._config_name,
                )
                stored_raw = self._backend._raw_files.get(owner, (None, None))
                if stored_raw == incoming_raw:
                    return
                if stored_raw == (None, None):
                    self._backend._raw_files[owner] = incoming_raw
                    return
                if incoming_raw != (None, None):
                    raise ArtifactsArchiveException(
                        operation="register_task",
                        detail=f"configuration changed for task plan: {task}/{stage}",
                    )
                return
        archive_dir = explicit_dir or self.config.save_dir
        self._backend.register_task(
            task_id,
            stage_name,
            archive_save_dir=archive_dir,
            allowed_runtime_dirs=allowed_runtime_dirs,
            task_type=task_type or self._default_task_type,
            raw_file=raw_file if raw_file is not None else self._raw_file,
            config_name=config_name if config_name is not None else self._config_name,
        )

    def register_task(self, task_id: str, stage_name: str, **kwargs) -> None:
        self.ensure_task(task_id, stage_name, **kwargs)

    def register(
            self,
            task_id: str,
            stage_name: str,
            artifact: ArtifactIntent | Artifact,
    ) -> LedgerEntry:
        """模块登记产物意图；兼容旧 Artifact 壳注册。"""
        self._require_open()
        if isinstance(artifact, Artifact):
            intent = ArtifactIntent(
                artifact.artifact_type,
                planned_hint=artifact.path,
                metadata=dict(artifact.metadata),
            )
        elif isinstance(artifact, ArtifactIntent):
            intent = artifact
        else:
            raise ArtifactsArchiveException(
                operation="register", detail="expected ArtifactIntent or Artifact",
            )
        self.ensure_task(task_id, stage_name)
        if intent.planned_hint:
            self._allow_runtime_dir_for_task(task_id, stage_name, intent.planned_hint)
            self._backend.register(
                task_id,
                stage_name,
                Artifact(intent.artifact_type, intent.planned_hint, dict(intent.metadata)),
            )
        key = (task_id, stage_name, intent.artifact_type, intent.planned_hint)
        with self._lock:
            if key in self._intent_keys:
                for entry in self._ledger.values():
                    if (
                        entry.task_id == task_id
                        and entry.stage_name == stage_name
                        and entry.artifact_type is intent.artifact_type
                        and entry.status is ArtifactLedgerStatus.REGISTERED
                        and entry.producer_path == intent.planned_hint
                    ):
                        return entry
            self._intent_keys.add(key)
            entry = LedgerEntry(
                entry_id=uuid.uuid4().hex,
                task_id=task_id,
                stage_name=stage_name,
                artifact_type=intent.artifact_type,
                status=ArtifactLedgerStatus.REGISTERED,
                origin=None,
                center_path=None,
                producer_path=intent.planned_hint,
                metadata=copy.deepcopy(intent.metadata),
            )
            self._ledger[entry.entry_id] = entry
            return entry

    def put(self, task_id: str, stage_name: str, item: ArtifactPut) -> LedgerEntry:
        """输送单个已生成文件 → AVAILABLE（本机 origin=LOCAL）。"""
        self._require_open()
        if not isinstance(item, ArtifactPut):
            raise ArtifactsArchiveException(operation="put", detail="expected ArtifactPut")
        source = self._resolve_put_file(item.path)
        self.ensure_task(task_id, stage_name)
        # 用生产者给出的父目录扩白名单，不用 resolve 后的父目录，避免符号链接把外部目录放进来。
        self._ensure_type_root(task_id, stage_name, item.artifact_type, Path(item.path).expanduser().parent)
        artifact = Artifact(item.artifact_type, str(source), copy.deepcopy(item.metadata))
        self._backend.record(task_id, stage_name, artifact)
        return self._upsert_available(
            task_id,
            stage_name,
            artifact_type=item.artifact_type,
            center_path=str(source),
            producer_path=item.path,
            origin=ArtifactOrigin.LOCAL,
            metadata=item.metadata,
        )

    def put_bytes(
            self,
            task_id: str,
            stage_name: str,
            item: ArtifactPut,
            data: bytes,
    ) -> LedgerEntry:
        """worker 侧文件 bytes 输送：落地 _remote_cache 后登记 AVAILABLE（origin=REMOTE）。"""
        self._require_open()
        if not isinstance(item, ArtifactPut):
            raise ArtifactsArchiveException(operation="put_bytes", detail="expected ArtifactPut")
        if isinstance(data, bytearray):
            data = bytes(data)
        if not isinstance(data, bytes):
            raise ArtifactsArchiveException(operation="put_bytes", detail="data must be bytes")
        if len(data) > PUT_BYTES_MAX_BYTES:
            raise ArtifactsArchiveException(
                operation="put_bytes",
                detail=f"payload exceeds limit: {len(data)} > {PUT_BYTES_MAX_BYTES}",
            )
        self.ensure_task(task_id, stage_name)
        with self._lock:
            for entry in self._ledger.values():
                if (
                    entry.task_id == task_id
                    and entry.stage_name == stage_name
                    and entry.status is ArtifactLedgerStatus.AVAILABLE
                    and entry.producer_path == item.path
                ):
                    return copy.deepcopy(entry)
        store = self._backend._store(task_id, stage_name)
        task_key = _ArchiveStorage._validate_component(task_id, "task_id")
        type_dir = _TYPE_DIRECTORIES[item.artifact_type]
        source_name = Path(item.path).name or f"{uuid.uuid4().hex}.bin"
        cache_dir = Path(store.archive_save_dir) / REMOTE_CACHE_DIRNAME / task_key / type_dir
        dest = cache_dir / source_name
        if dest.exists() or dest.is_symlink():
            dest = cache_dir / f"{dest.stem}_{uuid.uuid4().hex[:8]}{dest.suffix}"
        ensure_output_parent(dest)
        dest.write_bytes(data)
        dest_resolved = dest.resolve(strict=True)
        self._ensure_type_root(task_id, stage_name, item.artifact_type, dest.parent)
        artifact = Artifact(item.artifact_type, str(dest_resolved), copy.deepcopy(item.metadata))
        self._backend.record(task_id, stage_name, artifact)
        return self._upsert_available(
            task_id,
            stage_name,
            artifact_type=item.artifact_type,
            center_path=str(dest_resolved),
            producer_path=item.path,
            origin=ArtifactOrigin.REMOTE,
            metadata=item.metadata,
        )

    def put_config(self, task_id: str, stage_name: str) -> LedgerEntry | None:
        """把绑定的原始 TOML 副本 register+put 进运行期台账（Harness 查询用）。

        未绑定 raw TOML（纯编程任务）返回 None，不伪造配置。副本落
        ``_remote_cache/<task>/config/``；台账行不进归档 pending——归档的
        config/ 仍由 ``_stage_raw_config`` 按 raw 摘要唯一产出，缓存副本
        不随归档清源，保证运行期可 open。
        """
        self._require_open()
        if not self._raw_file:
            return None
        self.ensure_task(task_id, stage_name)
        store = self._backend._store(task_id, stage_name)
        task_key = _ArchiveStorage._validate_component(task_id, "task_id")
        source_name = Path(self._config_name or "config.toml").name or "config.toml"
        cache_dir = Path(store.archive_save_dir) / REMOTE_CACHE_DIRNAME / task_key / "config"
        dest = cache_dir / source_name
        ensure_output_parent(dest)
        dest.write_text(self._raw_file, encoding="utf-8")
        dest_resolved = dest.resolve(strict=True)
        self._ensure_type_root(task_id, stage_name, ArtifactType.CONFIG, dest.parent)
        return self._upsert_available(
            task_id,
            stage_name,
            artifact_type=ArtifactType.CONFIG,
            center_path=str(dest_resolved),
            producer_path=str(self._config_name or ""),
            origin=ArtifactOrigin.REMOTE,
            metadata={
                "source_name": str(self._config_name or ""),
                "sha256": hashlib.sha256(self._raw_file.encode("utf-8")).hexdigest(),
            },
        )

    def put_log(self, task_id: str, role_id: str, stage_name: str | None = None) -> list[LedgerEntry]:
        """按 role_id 读取共享 log_dir 下日志快照，写入 AM cache 并登记 AVAILABLE。"""
        self._require_open()
        role = str(role_id or "").strip()
        if not role:
            raise ArtifactsArchiveException(operation="put_log", detail="role_id is empty")
        from common.logger import CONF, resolve_role_log_files

        log_dir = Path(CONF.log_dir).expanduser().resolve()
        sources = resolve_role_log_files(role, task_id)
        stage = stage_name
        if not stage:
            planned = self._stages_for_task(task_id)
            stage = planned[0] if planned else "train"
        self.ensure_task(task_id, stage)
        store = self._backend._store(task_id, stage)
        task_key = _ArchiveStorage._validate_component(task_id, "task_id")
        cache_dir = Path(store.archive_save_dir) / REMOTE_CACHE_DIRNAME / task_key / "logs" / role
        entries: list[LedgerEntry] = []
        active_resolved = sources[-1].expanduser().resolve() if sources else None
        for source in sources:
            try:
                resolved = source.expanduser().resolve(strict=True)
            except OSError as exc:
                raise ArtifactsArchiveException(operation="put_log", detail=str(exc)) from exc
            if not resolved.is_file():
                raise ArtifactsArchiveException(
                    operation="put_log", detail=f"log is not a file: {resolved}",
                )
            if not _ArchiveStorage._is_within(resolved, log_dir):
                raise ArtifactsArchiveException(
                    operation="put_log", detail=f"log is outside log_dir: {resolved}",
                )
            ensure_output_parent(cache_dir / resolved.name)
            dest = cache_dir / resolved.name
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            _ArchiveStorage._copy_log_snapshot(resolved, dest)
            dest_resolved = dest.resolve(strict=True)
            self._ensure_type_root(task_id, stage, ArtifactType.LOG, dest.parent)
            metadata = {
                "closed": active_resolved is not None and resolved != active_resolved,
                "role_id": role,
                "shared_log": str(active_resolved or resolved),
            }
            self._backend.record(
                task_id,
                stage,
                Artifact(ArtifactType.LOG, str(dest_resolved), metadata),
            )
            entries.append(
                self._upsert_available(
                    task_id,
                    stage,
                    artifact_type=ArtifactType.LOG,
                    center_path=str(dest_resolved),
                    producer_path=str(source),
                    origin=ArtifactOrigin.REMOTE,
                    metadata=metadata,
                )
            )
        return entries

    def list_entries(
            self,
            task_id: str,
            stage_name: str | None = None,
    ) -> list[LedgerEntry]:
        with self._lock:
            rows = [
                copy.deepcopy(entry)
                for entry in self._ledger.values()
                if entry.task_id == task_id
                and (stage_name is None or entry.stage_name == stage_name)
            ]
        rows.sort(key=lambda item: (item.stage_name, item.artifact_type.value, item.entry_id))
        return rows

    def expect_producer(self, task_id: str, stage_name: str, producer_id: str) -> None:
        """Role 接线时登记本阶段产物模块；按模块实例计，不是 Intent 类型数。"""
        self._require_open()
        producer = str(producer_id)
        if not producer:
            raise ArtifactsArchiveException(operation="expect_producer", detail="producer_id is empty")
        with self._archive_ready:
            gate = self._stage_gates.setdefault((task_id, stage_name), _StageArchiveGate())
            gate.expected.add(producer)
            self._archive_ready.notify_all()

    def complete(self, task_id: str, stage_name: str, producer_id: str) -> None:
        """本模块本阶段产物上报完毕；未 expect 的 complete 仍记录，不单独构成可归档。"""
        self._require_open()
        producer = str(producer_id)
        if not producer:
            raise ArtifactsArchiveException(operation="complete", detail="producer_id is empty")
        with self._archive_ready:
            gate = self._stage_gates.setdefault((task_id, stage_name), _StageArchiveGate())
            gate.completed.add(producer)
            self._archive_ready.notify_all()

    def is_stage_archivable(self, task_id: str, stage_name: str) -> bool:
        with self._lock:
            gate = self._stage_gates.get((task_id, stage_name))
            return True if gate is None else gate.is_archivable()

    def _wait_until_archivable(self, task_id: str, stage_name: str) -> None:
        with self._archive_ready:
            while True:
                self._require_open()
                gate = self._stage_gates.get((task_id, stage_name))
                if gate is None or gate.is_archivable():
                    return
                self._archive_ready.wait()

    def _reset_stage_gate(self, task_id: str, stage_name: str) -> None:
        with self._archive_ready:
            self._stage_gates.pop((task_id, stage_name), None)
            self._archive_ready.notify_all()

    def archive_task(self, task_id: str, stage_name: str) -> ArchivedTask:
        self._require_open()
        self.ensure_task(task_id, stage_name)
        self._wait_until_archivable(task_id, stage_name)
        archived = self._backend.archive_task(task_id, stage_name)
        with self._lock:
            self._archived_stages.add((task_id, stage_name))
            for entry_id, entry in list(self._ledger.items()):
                if (
                    entry.task_id == task_id
                    and entry.stage_name == stage_name
                    and entry.status is ArtifactLedgerStatus.AVAILABLE
                ):
                    self._ledger[entry_id] = LedgerEntry(
                        entry_id=entry.entry_id,
                        task_id=entry.task_id,
                        stage_name=entry.stage_name,
                        artifact_type=entry.artifact_type,
                        status=ArtifactLedgerStatus.ARCHIVED,
                        origin=entry.origin,
                        center_path=entry.center_path,
                        producer_path=entry.producer_path,
                        metadata=dict(entry.metadata),
                    )
        self._reset_stage_gate(task_id, stage_name)
        return archived

    def list_archived_tasks(self, archive_save_dir: str | None = None) -> list[ArchivedTask]:
        return self._backend.list_archived_tasks(archive_save_dir)

    def get_task_artifacts(self, task_id: str, archive_save_dir: str | None = None) -> list[Artifact]:
        return self._backend.get_task_artifacts(task_id, archive_save_dir)

    def get_registered_artifacts(self, task_id: str, stage_name: str) -> list[ArtifactRegistration]:
        return self._backend.get_registered_artifacts(task_id, stage_name)

    # --- 兼容旧 ArtifactsArchive 方法名（facade / TM 过渡） ---

    def record(self, task_id: str, stage_name: str, artifact: Artifact) -> None:
        """兼容旧 record：仅接受文件，委托 put；目录拒绝（调用方逐文件 put，一文件一行）。"""
        self._require_open()
        if not isinstance(artifact, Artifact):
            raise ArtifactsArchiveException(operation="record", detail="expected Artifact")
        path = Path(artifact.path).expanduser()
        try:
            source = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ArtifactsArchiveException(
                operation="record", detail=f"artifact source does not exist: {path}: {exc}",
            ) from exc
        if source.is_dir():
            raise ArtifactsArchiveException(
                operation="record",
                detail=(
                    f"directory artifacts must be put per-file, not as a directory: {source}"
                ),
            )
        self.put(
            task_id,
            stage_name,
            ArtifactPut(artifact.artifact_type, str(source), dict(artifact.metadata)),
        )

    def record_log(self, task_id: str, artifact: Artifact) -> None:
        """兼容旧路径登记：仍走 backend.record_log，不经过 role_id 缓存采集。"""
        self._require_open()
        if not isinstance(artifact, Artifact):
            raise ArtifactsArchiveException(operation="record_log", detail="expected Artifact")
        self._backend.record_log(task_id, artifact)

    @process_close
    def close(self) -> None:
        """释放台账，清理缓存；全部阶段归档成功后清理任务独占运行目录。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # 收集本 AM 经手过的任务及其归档根；cache/锁都按 task 子目录清理，
            # 共享 save_dir 下其他任务的运行期目录不受影响。
            planned = [
                (task, Path(store.archive_save_dir))
                for (task, _stage), store in self._backend._tasks.items()
            ]
            runtime_roots = self._confirmed_runtime_roots()
            self._ledger.clear()
            self._path_index.clear()
            self._intent_keys.clear()
            self._stage_gates.clear()
            self._archive_ready.notify_all()
            self._backend = ArtifactsArchive()
        self._cleanup_task_residue(planned)
        for root in runtime_roots:
            from common.logger import retire_log_directory

            retire_log_directory(root)
            shutil.rmtree(root)
        with _active_lock:
            global _active_manager
            if _active_manager is self:
                _active_manager = None

    def _confirmed_runtime_roots(self) -> list[Path]:
        """归档在源目录内或任何阶段未成功时保留源目录，防止误删。"""
        confirmed = []
        for task, root in self._runtime_roots.items():
            stores = {
                owner: store for owner, store in self._backend._tasks.items()
                if owner[0] == task
            }
            if not root.exists() or root.is_symlink() or not stores:
                continue
            if not stores.keys() <= self._archived_stages:
                continue
            if any(
                _ArchiveStorage._is_within(store.archive_save_dir, root)
                or _ArchiveStorage._is_within(root, store.archive_save_dir)
                for store in self._backend._tasks.values()
            ):
                continue
            # close 可能晚于归档很久；删除前再次确认 manifest 和产物仍完整。
            if all(self._runtime_archive_is_complete(owner, store) for owner, store in stores.items()):
                confirmed.append(root)
        return confirmed

    @staticmethod
    def _runtime_archive_is_complete(owner: tuple[str, str], store: _ArchiveStorage) -> bool:
        task, stage = owner
        target = store.archive_save_dir / task
        try:
            manifest = store._load_manifest(target, allow_missing=True)
            return manifest is not None and stage in manifest["stages"] and all(
                (target / entry["relative_path"]).is_file()
                and store._measure(target / entry["relative_path"]) == (entry["size"], entry["sha256"])
                for entry in manifest["artifacts"]
            )
        except (ArtifactsArchiveException, OSError, KeyError, TypeError):
            return False

    def _cleanup_task_residue(self, planned: list[tuple[str, Path]]) -> None:
        """按任务清理 _remote_cache/<task>/ 与 .<save_dir>.locks/.<task>.archive.lock。"""
        for task, archive_save_dir in planned:
            try:
                task_key = _ArchiveStorage._validate_component(task, "task_id")
            except ArtifactsArchiveException:
                continue
            root = Path(archive_save_dir).expanduser()
            shutil.rmtree(root / REMOTE_CACHE_DIRNAME / task_key, ignore_errors=True)
            lock_dir = root.parent / f".{root.name or 'archives'}.locks"
            try:
                (lock_dir / f".{task}.archive.lock").unlink()
            except OSError:
                pass
            try:
                lock_dir.rmdir()
            except OSError:
                pass

    def _require_open(self) -> None:
        if self._closed:
            raise ArtifactsArchiveException(operation="closed", detail="ArtifactsManager is closed")

    @staticmethod
    def _resolve_put_file(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ArtifactsArchiveException(
                operation="put", detail=f"artifact source does not exist: {path}: {exc}",
            ) from exc
        if resolved.is_dir():
            raise ArtifactsArchiveException(
                operation="put", detail=f"ArtifactPut.path must be a file, not a directory: {resolved}",
            )
        if not resolved.is_file():
            raise ArtifactsArchiveException(
                operation="put", detail=f"ArtifactPut.path is not a file: {resolved}",
            )
        return resolved

    def _allow_runtime_dir_for_task(self, task_id: str, stage_name: str, path: str | Path) -> None:
        """把本阶段实际写出目录补进运行时白名单；已有其它类型目录时不得互斥。"""
        owner = (
            _ArchiveStorage._validate_component(task_id, "task_id"),
            _ArchiveStorage._validate_component(stage_name, "stage_name"),
        )
        with self._backend._lock:
            store = self._backend._tasks.get(owner)
        if store is not None:
            store.allow_runtime_dir(str(path))

    def _ensure_type_root(
            self,
            task_id: str,
            stage_name: str,
            artifact_type: ArtifactType,
            root: Path,
    ) -> None:
        resolved_root = root.expanduser().resolve()
        rows = self._backend.get_registered_artifacts(task_id, stage_name)
        for row in rows:
            if row.artifact_type is artifact_type:
                planned = Path(row.planned_path).expanduser().resolve()
                try:
                    resolved_root.relative_to(planned)
                    return
                except ValueError:
                    if planned == resolved_root:
                        return
        self._allow_runtime_dir_for_task(task_id, stage_name, resolved_root)
        self._backend.register(
            task_id,
            stage_name,
            Artifact(artifact_type, str(resolved_root)),
        )

    def _stages_for_task(self, task_id: str) -> list[str]:
        with self._backend._lock:
            return sorted({stage for (task, stage) in self._backend._tasks if task == task_id})

    def _upsert_available(
            self,
            task_id: str,
            stage_name: str,
            *,
            artifact_type: ArtifactType,
            center_path: str,
            producer_path: str,
            origin: ArtifactOrigin,
            metadata: dict[str, Any],
    ) -> LedgerEntry:
        index_key = (task_id, stage_name, center_path)
        with self._lock:
            existing_id = self._path_index.get(index_key)
            if existing_id and existing_id in self._ledger:
                previous = self._ledger[existing_id]
                entry = LedgerEntry(
                    entry_id=previous.entry_id,
                    task_id=task_id,
                    stage_name=stage_name,
                    artifact_type=artifact_type,
                    status=ArtifactLedgerStatus.AVAILABLE,
                    origin=origin,
                    center_path=center_path,
                    producer_path=producer_path,
                    metadata={**previous.metadata, **copy.deepcopy(metadata)},
                )
            else:
                entry = LedgerEntry(
                    entry_id=uuid.uuid4().hex,
                    task_id=task_id,
                    stage_name=stage_name,
                    artifact_type=artifact_type,
                    status=ArtifactLedgerStatus.AVAILABLE,
                    origin=origin,
                    center_path=center_path,
                    producer_path=producer_path,
                    metadata=copy.deepcopy(metadata),
                )
            self._ledger[entry.entry_id] = entry
            self._path_index[index_key] = entry.entry_id
            return entry


# 兼容门面：未显式注入时转发到 active ArtifactsManager。


class _ArtifactsArchiveProxy:
    """兼容门面：转发到当前激活的 ArtifactsManager。"""

    def _manager(self):
        manager = get_active_artifacts_manager()
        if manager is None:
            manager = build_artifacts_manager(activate=True)
        return manager

    def __getattr__(self, name: str):
        manager = self._manager()
        if hasattr(manager, name):
            return getattr(manager, name)
        return getattr(manager.backend, name)


artifacts_archive = _ArtifactsArchiveProxy()

__all__ = [
    "Artifact",
    "ArtifactIntent",
    "ArtifactPut",
    "ArtifactType",
    "ArtifactOrigin",
    "ArtifactLedgerStatus",
    "ArtifactRegistration",
    "ArtifactStatus",
    "ArchivedTask",
    "ArtifactsArchive",
    "ArtifactsManager",
    "LedgerEntry",
    "ArtifactCallerSupport",
    "ArtifactPutCaller",
    "ArtifactRegisterCaller",
    "ensure_output_parent",
    "REMOTE_CACHE_DIRNAME",
    "PUT_BYTES_MAX_BYTES",
    "_ArchiveStorage",
    "artifacts_archive",
    "build_artifacts_manager",
    "get_active_artifacts_manager",
    "reset_active_artifacts_manager_for_tests",
    "resolve_artifacts_manager_config",
    "set_active_artifacts_manager",
]

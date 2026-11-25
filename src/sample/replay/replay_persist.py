"""ReplayBuffer 阶段快照：DRBF 单文件容器（msgpack meta + 流式张量段）与本地/S3 I/O。"""
from __future__ import annotations

import importlib
import os
import shutil
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import msgpack
import numpy as np
import torch

from common.exception import S3ProtocolException
from common.logger import init_logger
from config import (
    LocalStorageConfig,
    ReplayBufferConfig,
    ResourceDir,
    S3StorageConfig,
    StorageType,
)
from sample.replay.step_tensor_pool import KIND_MMAP_DIR, KIND_TORCH_SAVE, TensorCheckpointInfo

if TYPE_CHECKING:
    pass

LOG = init_logger("replay_persist")

DRBF_MAGIC = b"DRBF"
DRBF_VERSION = 1
_HEADER_STRUCT = struct.Struct("<4sHHIQQ")  # magic, version, flags, reserved, meta_len, tensor_len
MMPK_MAGIC = b"MMPK"
_MMPK_HEADER = struct.Struct("<4sI")  # magic, version
_COPY_CHUNK = 8 * 1024 * 1024


def _run_s3_op(operation: str, storage: S3StorageConfig, func, *, key: str = ""):
    """执行一次 S3 调用；失败时抛 S3ProtocolException。"""
    try:
        return func()
    except S3ProtocolException:
        raise
    except Exception as exc:
        raise S3ProtocolException(
            operation=operation,
            bucket=storage.bucket_name,
            key=key,
            detail=exc,
        ) from exc


def resolve_replay_snapshot_path(path: str | Path, samples_dir: Optional[str] = None) -> Path:
    """绝对路径直用；相对路径相对 ResourceDir.SAMPLES（或 samples_dir）。"""
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute():
        return candidate
    root = Path(samples_dir or ResourceDir.SAMPLES.value)
    return root / candidate


def default_replay_snapshot_name(stage_name: str, task_id: str) -> str:
    safe_stage = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stage_name)
    safe_task = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(task_id))
    return f"replay_{safe_stage}_{safe_task}.bin"


def write_drbf(path: str | Path, meta: Dict[str, Any], tensor_path: Path) -> Path:
    """把已序列化的 meta dict 与张量文件流式写成单个 DRBF 文件。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = Path(tensor_path)
    tensor_nbytes = int(tensor_path.stat().st_size)
    meta_bytes = msgpack.packb(meta, use_bin_type=True)
    header = _HEADER_STRUCT.pack(
        DRBF_MAGIC,
        DRBF_VERSION,
        0,
        0,
        len(meta_bytes),
        tensor_nbytes,
    )
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        handle.write(header)
        handle.write(meta_bytes)
        with open(tensor_path, "rb") as src:
            shutil.copyfileobj(src, handle, length=_COPY_CHUNK)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)
    return target


def read_drbf(path: str | Path) -> Tuple[Dict[str, Any], int, int]:
    """读取 DRBF header/meta，返回 (meta_dict, tensor_offset, tensor_len)。不把张量段读进内存。"""
    source = Path(path)
    with open(source, "rb") as handle:
        header = handle.read(_HEADER_STRUCT.size)
        if len(header) != _HEADER_STRUCT.size:
            raise ValueError(f"DRBF header truncated: path={source}")
        magic, version, _flags, _reserved, meta_len, tensor_len = _HEADER_STRUCT.unpack(header)
        if magic != DRBF_MAGIC:
            raise ValueError(f"DRBF magic mismatch: path={source}, magic={magic!r}")
        if int(version) != DRBF_VERSION:
            raise ValueError(f"unsupported DRBF version={version}, path={source}")
        meta_bytes = handle.read(int(meta_len))
        if len(meta_bytes) != int(meta_len):
            raise ValueError(f"DRBF meta truncated: path={source}")
        tensor_offset = _HEADER_STRUCT.size + int(meta_len)
    meta = msgpack.unpackb(meta_bytes, raw=False)
    if not isinstance(meta, dict):
        raise ValueError(f"DRBF meta must be dict, path={source}")
    file_size = source.stat().st_size
    if tensor_offset + int(tensor_len) > file_size:
        raise ValueError(f"DRBF payload truncated: path={source}")
    return meta, tensor_offset, int(tensor_len)


def extract_drbf_tensors(path: str | Path, dest: Path) -> Tuple[Dict[str, Any], Path]:
    """把 DRBF 张量段流式拷到 dest 文件，返回 (meta, dest)。"""
    source = Path(path)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta, tensor_offset, tensor_len = read_drbf(source)
    with open(source, "rb") as src, open(dest, "wb") as out:
        src.seek(tensor_offset)
        remaining = tensor_len
        while remaining > 0:
            chunk = src.read(min(_COPY_CHUNK, remaining))
            if not chunk:
                raise ValueError(f"DRBF tensor truncated while copying: path={source}")
            out.write(chunk)
            remaining -= len(chunk)
        out.flush()
        os.fsync(out.fileno())
    return meta, dest


def pack_tensor_checkpoint(src_dir: Path, info: TensorCheckpointInfo, dest: Path) -> Path:
    """把 checkpoint 目录打成单个张量 blob，供写入 DRBF 张量段。"""
    src_dir = Path(src_dir)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if info.kind == KIND_TORCH_SAVE:
        if len(info.relative_files) != 1:
            raise ValueError(f"torch_save checkpoint must contain one file, got {info.relative_files}")
        shutil.copy2(src_dir / info.relative_files[0], dest)
        return dest
    if info.kind != KIND_MMAP_DIR:
        raise ValueError(f"unsupported tensor checkpoint kind={info.kind!r}")
    _pack_mmap_dir(src_dir, dest, info.relative_files)
    return dest


def unpack_tensor_checkpoint(blob: Path, dest_dir: Path, kind: str) -> Path:
    """把 DRBF 张量 blob 还原成 checkpoint 路径（文件或目录）。"""
    blob = Path(blob)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if kind == KIND_TORCH_SAVE:
        payload = dest_dir / "tensors.pt"
        shutil.copy2(blob, payload)
        return payload
    if kind != KIND_MMAP_DIR:
        raise ValueError(f"unsupported tensor checkpoint kind={kind!r}")
    _unpack_mmap_dir(blob, dest_dir)
    return dest_dir


def _pack_mmap_dir(src_dir: Path, dest: Path, relative_files: Sequence[str]) -> None:
    with open(dest, "wb") as out:
        out.write(_MMPK_HEADER.pack(MMPK_MAGIC, 1))
        out.write(struct.pack("<I", len(relative_files)))
        for rel in relative_files:
            path = src_dir / rel
            name = rel.encode("utf-8")
            out.write(struct.pack("<I", len(name)))
            out.write(name)
            size = int(path.stat().st_size)
            out.write(struct.pack("<Q", size))
            with open(path, "rb") as src:
                shutil.copyfileobj(src, out, length=_COPY_CHUNK)


def _unpack_mmap_dir(blob: Path, dest_dir: Path) -> None:
    with open(blob, "rb") as src:
        header = src.read(_MMPK_HEADER.size)
        if len(header) != _MMPK_HEADER.size:
            raise ValueError(f"mmap pack header truncated: {blob}")
        magic, version = _MMPK_HEADER.unpack(header)
        if magic != MMPK_MAGIC:
            raise ValueError(f"mmap pack magic mismatch: {blob}, magic={magic!r}")
        if int(version) != 1:
            raise ValueError(f"unsupported mmap pack version={version}, path={blob}")
        count_raw = src.read(4)
        if len(count_raw) != 4:
            raise ValueError(f"mmap pack truncated: {blob}")
        (file_count,) = struct.unpack("<I", count_raw)
        for _ in range(int(file_count)):
            name_len_raw = src.read(4)
            if len(name_len_raw) != 4:
                raise ValueError(f"mmap pack truncated: {blob}")
            (name_len,) = struct.unpack("<I", name_len_raw)
            name = src.read(int(name_len)).decode("utf-8")
            size_raw = src.read(8)
            if len(size_raw) != 8:
                raise ValueError(f"mmap pack truncated: {blob}")
            (size,) = struct.unpack("<Q", size_raw)
            dest = dest_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            remaining = int(size)
            with open(dest, "wb") as out:
                while remaining > 0:
                    chunk = src.read(min(_COPY_CHUNK, remaining))
                    if not chunk:
                        raise ValueError(f"mmap pack file truncated: {blob} name={name}")
                    out.write(chunk)
                    remaining -= len(chunk)


def build_replay_s3_client(storage: S3StorageConfig, *, ensure_bucket: bool = False):
    import boto3

    client = _run_s3_op(
        "create_client",
        storage,
        lambda: boto3.client(
            "s3",
            endpoint_url=storage.endpoint_url,
            aws_access_key_id=storage.access_key,
            aws_secret_access_key=storage.secret_key,
        ),
    )
    if ensure_bucket:
        def _ensure_bucket():
            try:
                client.head_bucket(Bucket=storage.bucket_name)
            except Exception:
                client.create_bucket(Bucket=storage.bucket_name)

        _run_s3_op("ensure_bucket", storage, _ensure_bucket)
    return client


def sync_snapshot_to_storage(local_path: Path, config: ReplayBufferConfig) -> None:
    """本地写完后按需上传到 S3；local 模式无需额外动作。"""
    if StorageType(config.storage_type) != StorageType.S3:
        return
    storage = config.storage
    if not isinstance(storage, S3StorageConfig):
        raise TypeError("ReplayBufferConfig.storage must be S3StorageConfig when storage_type=s3")
    client = build_replay_s3_client(storage, ensure_bucket=True)
    key = local_path.name
    _run_s3_op(
        "upload_file",
        storage,
        lambda: client.upload_file(str(local_path), storage.bucket_name, key),
        key=key,
    )
    LOG.info(
        f"ReplayBuffer snapshot uploaded: bucket={storage.bucket_name}, key={key}, path={local_path}"
    )


def ensure_local_snapshot(path: Path, config: Optional[ReplayBufferConfig] = None) -> Path:
    """确保快照在本地可读；S3 模式下本地缺失则 download_file。"""
    if path.exists():
        return path
    if config is None or StorageType(config.storage_type) != StorageType.S3:
        raise FileNotFoundError(f"ReplayBuffer snapshot not found: {path}")
    storage = config.storage
    if not isinstance(storage, S3StorageConfig):
        raise TypeError("ReplayBufferConfig.storage must be S3StorageConfig when storage_type=s3")
    client = build_replay_s3_client(storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    _run_s3_op(
        "download_file",
        storage,
        lambda: client.download_file(storage.bucket_name, path.name, str(path)),
        key=path.name,
    )
    LOG.info(
        f"ReplayBuffer snapshot downloaded: bucket={storage.bucket_name}, "
        f"key={path.name}, path={path}"
    )
    return path


def samples_dir_from_config(config: Optional[ReplayBufferConfig]) -> str:
    if config is not None and isinstance(config.storage, LocalStorageConfig) and config.storage.data_path:
        return str(config.storage.data_path)
    return ResourceDir.SAMPLES.value


# ---------------------------------------------------------------------------
# Meta 序列化：EpisodeMeta / StepMeta / TensorSchema / 嵌套模板
# ---------------------------------------------------------------------------

def serialize_persist_value(val: Any) -> Any:
    """把回放池元数据递归转为 msgpack 友好结构。"""
    if isinstance(val, np.ndarray):
        return {"__type__": "ndarray", "data": val.tolist(), "dtype": str(val.dtype)}
    if isinstance(val, torch.Tensor):
        return {
            "__type__": "tensor",
            "data": val.detach().cpu().numpy().tolist(),
            "dtype": str(val.dtype),
        }
    if isinstance(val, type):
        return {
            "__type__": "type",
            "__module__": val.__module__,
            "__class__": val.__name__,
        }
    if isinstance(val, Path):
        return {"__type__": "path", "data": str(val)}
    if hasattr(val, "_asdict") and isinstance(val, tuple):
        return {
            "__type__": "namedtuple",
            "__module__": val.__class__.__module__,
            "__class__": val.__class__.__name__,
            "data": {k: serialize_persist_value(v) for k, v in val._asdict().items()},
        }
    # frozen / 普通 dataclass：优先按字段构造，避免 __dict__.update 撞 frozen
    try:
        from dataclasses import is_dataclass, fields as dc_fields

        if is_dataclass(val) and not isinstance(val, type):
            payload = {f.name: serialize_persist_value(getattr(val, f.name)) for f in dc_fields(val)}
            return {
                "__type__": "dataclass",
                "__module__": val.__class__.__module__,
                "__class__": val.__class__.__name__,
                "data": payload,
            }
    except Exception:
        pass
    if hasattr(val, "__dict__") and not isinstance(val, type):
        return {
            "__type__": "custom_object",
            "__module__": val.__class__.__module__,
            "__class__": val.__class__.__name__,
            "data": {k: serialize_persist_value(v) for k, v in val.__dict__.items()},
        }
    if isinstance(val, dict):
        return {str(k) if not isinstance(k, (str, int, bytes)) else k:
                serialize_persist_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [serialize_persist_value(v) for v in val]
    if isinstance(val, tuple):
        return {"__type__": "pure_tuple", "data": [serialize_persist_value(v) for v in val]}
    if isinstance(val, set):
        return {"__type__": "pure_set", "data": [serialize_persist_value(v) for v in val]}
    return val


def deserialize_persist_value(val: Any) -> Any:
    """还原 serialize_persist_value 的输出。"""
    if isinstance(val, dict) and "__type__" in val:
        t = val["__type__"]
        if t == "ndarray":
            return np.array(val["data"], dtype=val["dtype"])
        if t == "tensor":
            dtype_str = str(val["dtype"]).split(".")[-1]
            torch_dtype = getattr(torch, dtype_str, torch.float32)
            return torch.tensor(val["data"], dtype=torch_dtype)
        if t == "type":
            module = importlib.import_module(val["__module__"])
            return getattr(module, val["__class__"])
        if t == "path":
            return Path(val["data"])
        if t == "pure_tuple":
            return tuple(deserialize_persist_value(x) for x in val["data"])
        if t == "pure_set":
            return {deserialize_persist_value(x) for x in val["data"]}
        if t in ("dataclass", "namedtuple", "custom_object", "slots_object"):
            module = importlib.import_module(val["__module__"])
            cls = getattr(module, val["__class__"])
            data = {k: deserialize_persist_value(v) for k, v in val["data"].items()}
            if t == "namedtuple":
                return cls(**data)
            if t == "dataclass":
                return cls(**data)
            obj = cls.__new__(cls)
            if t == "custom_object":
                from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition
                if isinstance(obj, (EpisodeTransition, SAEpisodeTransition)):
                    obj.__setstate__(data)
                else:
                    obj.__dict__.update(data)
            else:
                for key, value in data.items():
                    setattr(obj, key, value)
            return obj
        # 未知标签：当作普通 dict 继续
    if isinstance(val, dict):
        restored: Dict[Any, Any] = {}
        for key, value in val.items():
            # step_meta 等整型键经 msgpack 可能仍是 int；字符串数字键也尽量还原
            out_key: Any = key
            if isinstance(key, str) and key.isdigit():
                out_key = int(key)
            restored[out_key] = deserialize_persist_value(value)
        return restored
    if isinstance(val, list):
        return [deserialize_persist_value(v) for v in val]
    return val


# InMemory / LazyMemmap 通过 checkpoint/restore 落盘，DRBF 张量段只做文件流式 copy。

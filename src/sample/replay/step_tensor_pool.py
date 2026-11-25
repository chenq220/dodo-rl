"""ReplayBuffer 内部按 Step 存储的张量池：内存后端与 LazyMemmap 后端。"""
from __future__ import annotations

import gc
import heapq
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from common.info_bridge import INFO_TOML_CONFIG, info_bridge
from common.logger import init_logger
from config import LocalStorageConfig, ResourceDir, ReplayBufferConfig
from sample.replay.replay_payload import TensorOwnership
from sample.replay.replay_perf import timed
from utils.task_id import task_id

LOG = init_logger("step_tensor_pool")

KIND_TORCH_SAVE = "torch_save"
KIND_MMAP_DIR = "mmap_dir"

_DTYPE_TO_NUMPY = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class TensorCheckpointInfo:
    """张量池 checkpoint 的外壳描述，供 DRBF 按 kind 流式拼接。"""

    kind: str
    logical_bytes: int
    relative_files: Tuple[str, ...]


class StepTensorPool(Protocol):
    """ReplayBuffer 内部张量池协议；内存与 mmap 后端实现同一套接口。"""

    kind: str

    def append(self, step_tensors: List[Dict[str, torch.Tensor]]) -> List[int]:
        ...

    def append_batch(
            self,
            schema_id: int,
            step_tensors: Dict[str, torch.Tensor],
            ownership: TensorOwnership = TensorOwnership.BORROWED,
    ) -> List[int]:
        ...

    def get(self, tensor_indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        ...

    def get_batch(self, tensor_indices: List[int]) -> Dict[str, torch.Tensor]:
        ...

    def release(self, tensor_indices: List[int]) -> None:
        ...

    def clear(self) -> None:
        ...

    def checkpoint(self, dest: Path) -> TensorCheckpointInfo:
        ...

    def restore(self, src: Path) -> None:
        ...

    def close(self) -> None:
        ...

    @property
    def records(self) -> Dict[int, Dict[str, torch.Tensor]]:
        ...

    @property
    def memory_bytes(self) -> int:
        ...


@dataclass
class _TensorChunk:
    """一次 Episode 批量写入产生的列式张量块。"""

    schema_id: int
    tensors: Dict[str, torch.Tensor]
    active_count: int


class InMemoryStepTensorPool(StepTensorPool):
    """
    仅保存 CPU 张量的按 Step 内存池。

    Episode 归属、采样游标和业务 metadata 全部由 ReplayBuffer 元数据层维护。
    """

    kind = KIND_TORCH_SAVE

    def __init__(self):
        self._chunks: Dict[int, _TensorChunk] = {}
        self._locations: Dict[int, Tuple[int, int]] = {}
        self._next_chunk_id = 0
        self._free_indices: List[int] = []
        self._next_index = 0

    @property
    def records(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """只读测试/诊断视图；值结构中只允许出现 torch.Tensor。"""
        return {
            index: {
                key: tensor[offset]
                for key, tensor in self._chunks[chunk_id].tensors.items()
            }
            for index, (chunk_id, offset) in self._locations.items()
        }

    def append(self, step_tensors: List[Dict[str, torch.Tensor]]) -> List[int]:
        if not step_tensors:
            return []
        keys = tuple(step_tensors[0])
        if any(tuple(record) != keys for record in step_tensors):
            raise ValueError("all StepTensorPool records in a batch must have identical fields.")
        batch = {
            key: torch.stack([record[key] for record in step_tensors], dim=0)
            for key in keys
        }
        return self.append_batch(schema_id=0, step_tensors=batch)

    @timed("replay.tensor_write")
    def append_batch(
            self,
            schema_id: int,
            step_tensors: Dict[str, torch.Tensor],
            ownership: TensorOwnership = TensorOwnership.BORROWED,
    ) -> List[int]:
        if not step_tensors:
            return []
        if not all(isinstance(value, torch.Tensor) for value in step_tensors.values()):
            raise TypeError("StepTensorPool only accepts torch.Tensor values.")
        lengths = {int(value.shape[0]) for value in step_tensors.values()}
        if len(lengths) != 1:
            raise ValueError("all StepTensorPool columns must share the same leading Step dimension.")
        step_count = lengths.pop()
        indices: List[int] = []
        for _ in range(step_count):
            index = heapq.heappop(self._free_indices) if self._free_indices else self._next_index
            if index == self._next_index:
                self._next_index += 1
            indices.append(index)

        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        self._chunks[chunk_id] = _TensorChunk(
            schema_id=schema_id,
            tensors={
                key: self._prepare_tensor_for_storage(value, ownership)
                for key, value in step_tensors.items()
            },
            active_count=step_count,
        )
        for offset, index in enumerate(indices):
            self._locations[index] = (chunk_id, offset)
        return indices

    @staticmethod
    def _prepare_tensor_for_storage(
            value: torch.Tensor,
            ownership: TensorOwnership,
    ) -> torch.Tensor:
        detached = value.detach()
        if (
                ownership == TensorOwnership.TRANSFERRED
                and detached.device.type == "cpu"
                and detached.layout == torch.strided
                and detached.is_contiguous()
        ):
            return detached
        return detached.to("cpu").contiguous().clone()

    def get(self, tensor_indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        return [
            {
                key: tensor[offset]
                for key, tensor in self._chunks[chunk_id].tensors.items()
            }
            for chunk_id, offset in (self._locations[index] for index in tensor_indices)
        ]

    def get_batch(self, tensor_indices: List[int]) -> Dict[str, torch.Tensor]:
        if not tensor_indices:
            return {}
        locations = [self._locations[index] for index in tensor_indices]
        first_chunk_id = locations[0][0]
        first_chunk = self._chunks[first_chunk_id]
        keys = tuple(first_chunk.tensors)
        if all(chunk_id == first_chunk_id for chunk_id, _ in locations):
            offsets = [offset for _, offset in locations]
            if offsets == list(range(offsets[0], offsets[0] + len(offsets))):
                return {
                    key: tensor[offsets[0]:offsets[-1] + 1]
                    for key, tensor in first_chunk.tensors.items()
                }

        rows = self.get(tensor_indices)
        if any(tuple(row) != keys for row in rows):
            raise ValueError("get_batch requires all tensor indices to use one TensorSchema.")
        return {
            key: torch.stack([row[key] for row in rows], dim=0)
            for key in keys
        }

    def release(self, tensor_indices: List[int]) -> None:
        for index in tensor_indices:
            location = self._locations.pop(index, None)
            if location is None:
                continue
            chunk_id, _ = location
            chunk = self._chunks[chunk_id]
            chunk.active_count -= 1
            if chunk.active_count == 0:
                del self._chunks[chunk_id]
            heapq.heappush(self._free_indices, index)

    def clear(self) -> None:
        self._chunks.clear()
        self._locations.clear()
        self._free_indices.clear()
        self._next_index = 0
        self._next_chunk_id = 0

    def checkpoint(self, dest: Path) -> TensorCheckpointInfo:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        payload = dest / "tensors.pt"
        torch.save(
            {
                "chunks": self._chunks,
                "locations": self._locations,
                "free_indices": self._free_indices,
                "next_index": self._next_index,
                "next_chunk_id": self._next_chunk_id,
            },
            payload,
        )
        return TensorCheckpointInfo(
            kind=KIND_TORCH_SAVE,
            logical_bytes=int(payload.stat().st_size),
            relative_files=("tensors.pt",),
        )

    def restore(self, src: Path) -> None:
        src = Path(src)
        payload = src / "tensors.pt" if src.is_dir() else src
        state = torch.load(payload, map_location="cpu", weights_only=False)
        self._chunks = state["chunks"]
        self._locations = state["locations"]
        self._free_indices = list(state["free_indices"])
        heapq.heapify(self._free_indices)
        self._next_index = int(state["next_index"])
        self._next_chunk_id = int(state["next_chunk_id"])

    def close(self) -> None:
        self.clear()

    @property
    def memory_bytes(self) -> int:
        return sum(
            tensor.nelement() * tensor.element_size()
            for chunk in self._chunks.values()
            for tensor in chunk.tensors.values()
        )


@dataclass
class _MmapColumn:
    relpath: str
    mmap: np.memmap
    torch_dtype: torch.dtype
    row_shape: Tuple[int, ...]


class LazyMemmapStepTensorPool(StepTensorPool):
    """
    按 TensorSchema×字段把 Step 行映射到磁盘 mmap 文件。

    行号即 tensor_index，淘汰由 ReplayBuffer 调用 release；get/get_batch 必须 copy-out。
    """

    kind = KIND_MMAP_DIR

    def __init__(
            self,
            max_steps: int,
            scratch_dir: Optional[str | Path] = None,
    ):
        if int(max_steps) <= 0:
            raise ValueError("lazy_memmap tensor pool requires max_steps > 0")
        self._max_steps = int(max_steps)
        self._scratch_root = (
            Path(scratch_dir)
            if scratch_dir
            else Path(tempfile.gettempdir()) / "dodorl_replay_scratch"
        )
        self._scratch_dir: Optional[Path] = None
        self._columns: Dict[int, Dict[str, _MmapColumn]] = {}
        self._schema_of_index: Dict[int, int] = {}
        self._free_indices: List[int] = []
        self._next_index = 0
        self._closed = False

    def _ensure_scratch_dir(self) -> Path:
        """第一次写 mmap 才建目录，避免构造后尚未采样就被杀掉留下空壳。"""
        if self._scratch_dir is None:
            parent = self._scratch_root / str(task_id)
            parent.mkdir(parents=True, exist_ok=True)
            self._scratch_dir = Path(tempfile.mkdtemp(prefix="step_pool_", dir=str(parent)))
        return self._scratch_dir

    def reserve(self, max_steps: int) -> None:
        """只扩不缩；保留已有行号和内容，供阶段或 AutoML 增容使用。"""
        max_steps = int(max_steps)
        if max_steps <= self._max_steps:
            return
        if not self._columns:
            self._max_steps = max_steps
            return
        specs = [
            (schema_id, key, column.relpath, column.torch_dtype, column.row_shape)
            for schema_id, fields in self._columns.items()
            for key, column in fields.items()
        ]
        old_max_steps = self._max_steps
        self._close_maps()
        def reopen(rows: int, *, extend: bool) -> None:
            for schema_id, key, relpath, torch_dtype, row_shape in specs:
                path = self.scratch_dir / relpath
                if extend:
                    with path.open("r+b") as handle:
                        handle.truncate(
                            rows * int(np.dtype(_DTYPE_TO_NUMPY[torch_dtype]).itemsize)
                            * int(np.prod(row_shape, dtype=np.int64))
                        )
                mmap = np.memmap(
                    path, dtype=_DTYPE_TO_NUMPY[torch_dtype], mode="r+",
                    shape=(rows, *row_shape) if row_shape else (rows,),
                )
                self._columns.setdefault(schema_id, {})[key] = _MmapColumn(
                    relpath=relpath, mmap=mmap, torch_dtype=torch_dtype, row_shape=row_shape,
                )

        try:
            reopen(max_steps, extend=True)
        except Exception:
            self._close_maps()
            reopen(old_max_steps, extend=False)
            raise
        self._max_steps = max_steps

    def configure_empty_budget(self, max_steps: int) -> None:
        """首次写入前按 Role 已确定的分区容量修正预算，避免沿用默认 1000 的空映射。"""
        if self._columns or self._schema_of_index:
            raise RuntimeError("cannot resize a nonempty mmap tensor pool")
        if int(max_steps) <= 0:
            raise ValueError("lazy_memmap tensor pool requires max_steps > 0")
        self._max_steps = int(max_steps)

    def reserve_for_append(self, step_count: int) -> None:
        """先保留旧 active/rehearsal 行，再为整个原子 push 的真实步数预留暂存空间。"""
        if step_count < 0:
            raise ValueError("replay append step_count must be nonnegative")
        self.reserve(len(self._schema_of_index) + step_count)

    @property
    def created_scratch_dir(self) -> Optional[Path]:
        """已创建的 mkdtemp 目录；尚未写入时为 None，close 不要用 scratch_dir 去触发创建。"""
        return self._scratch_dir

    @property
    def scratch_dir(self) -> Path:
        if self._scratch_dir is None:
            raise RuntimeError("LazyMemmapStepTensorPool scratch dir is not created yet")
        return self._scratch_dir

    @property
    def records(self) -> Dict[int, Dict[str, torch.Tensor]]:
        return {
            index: self._copy_row(index)
            for index in sorted(self._schema_of_index)
        }

    def append(self, step_tensors: List[Dict[str, torch.Tensor]]) -> List[int]:
        if not step_tensors:
            return []
        keys = tuple(step_tensors[0])
        if any(tuple(record) != keys for record in step_tensors):
            raise ValueError("all StepTensorPool records in a batch must have identical fields.")
        batch = {
            key: torch.stack([record[key] for record in step_tensors], dim=0)
            for key in keys
        }
        return self.append_batch(schema_id=0, step_tensors=batch)

    @timed("replay.tensor_write")
    def append_batch(
            self,
            schema_id: int,
            step_tensors: Dict[str, torch.Tensor],
            ownership: TensorOwnership = TensorOwnership.BORROWED,
    ) -> List[int]:
        del ownership  # mmap 必须写入文件，无法接管调用方 Tensor。
        if self._closed:
            raise RuntimeError("LazyMemmapStepTensorPool is closed")
        if not step_tensors:
            return []
        if not all(isinstance(value, torch.Tensor) for value in step_tensors.values()):
            raise TypeError("StepTensorPool only accepts torch.Tensor values.")
        lengths = {int(value.shape[0]) for value in step_tensors.values()}
        if len(lengths) != 1:
            raise ValueError("all StepTensorPool columns must share the same leading Step dimension.")
        step_count = lengths.pop()
        live = len(self._schema_of_index)
        if live + step_count > self._max_steps:
            raise RuntimeError(
                "LazyMemmapStepTensorPool exhausted: "
                f"need {step_count} more steps, live={live}, max_steps={self._max_steps}. "
                "mmap row budget is (capacity + rehearsal) × max_step_per_episode."
            )
        indices = self._allocate_indices(step_count)
        self._ensure_schema(schema_id, step_tensors)
        columns = self._columns[schema_id]
        if tuple(columns) != tuple(step_tensors):
            raise ValueError(
                f"schema_id={schema_id} field set mismatch: "
                f"pool={tuple(columns)} batch={tuple(step_tensors)}"
            )
        cpu_batch = {
            key: value.detach().to("cpu").contiguous()
            for key, value in step_tensors.items()
        }
        if indices == list(range(indices[0], indices[0] + step_count)):
            start = indices[0]
            for key, tensor in cpu_batch.items():
                columns[key].mmap[start:start + step_count] = tensor.numpy()
        else:
            for offset, index in enumerate(indices):
                for key, tensor in cpu_batch.items():
                    columns[key].mmap[index] = tensor[offset].numpy()
        for index in indices:
            self._schema_of_index[index] = schema_id
        return indices

    def _allocate_indices(self, step_count: int) -> List[int]:
        indices: List[int] = []
        for _ in range(step_count):
            if self._free_indices:
                index = heapq.heappop(self._free_indices)
            else:
                index = self._next_index
                if index >= self._max_steps:
                    raise RuntimeError(
                        "LazyMemmapStepTensorPool exhausted: "
                        f"next_index={index}, max_steps={self._max_steps}"
                    )
                self._next_index += 1
            indices.append(index)
        return indices

    def _ensure_schema(self, schema_id: int, step_tensors: Dict[str, torch.Tensor]) -> None:
        if schema_id in self._columns:
            return
        scratch_dir = self._ensure_scratch_dir()
        schema_dir = scratch_dir / f"schema_{schema_id}"
        schema_dir.mkdir(parents=True, exist_ok=True)
        columns: Dict[str, _MmapColumn] = {}
        used_names: Dict[str, str] = {}
        for key, tensor in step_tensors.items():
            np_dtype = _DTYPE_TO_NUMPY.get(tensor.dtype)
            if np_dtype is None:
                raise TypeError(f"unsupported mmap dtype: {tensor.dtype} for field {key!r}")
            row_shape = tuple(int(dim) for dim in tensor.shape[1:])
            filename = self._unique_filename(key, used_names)
            relpath = f"schema_{schema_id}/{filename}"
            path = scratch_dir / relpath
            mmap = np.memmap(
                path,
                dtype=np_dtype,
                mode="w+",
                shape=(self._max_steps, *row_shape) if row_shape else (self._max_steps,),
            )
            columns[key] = _MmapColumn(
                relpath=relpath,
                mmap=mmap,
                torch_dtype=tensor.dtype,
                row_shape=row_shape,
            )
        self._columns[schema_id] = columns

    @staticmethod
    def _unique_filename(key: str, used_names: Dict[str, str]) -> str:
        base = _SAFE_NAME.sub("_", key).strip("._") or "field"
        name = f"{base}.dat"
        suffix = 1
        while name in used_names.values():
            suffix += 1
            name = f"{base}_{suffix}.dat"
        used_names[key] = name
        return name

    def _copy_row(self, index: int) -> Dict[str, torch.Tensor]:
        schema_id = self._schema_of_index[index]
        return {
            key: self._copy_slice(column, index, index + 1)[0]
            for key, column in self._columns[schema_id].items()
        }

    @staticmethod
    def _copy_slice(column: _MmapColumn, start: int, end: int) -> torch.Tensor:
        view = np.ascontiguousarray(column.mmap[start:end])
        return torch.from_numpy(view).clone()

    def get(self, tensor_indices: List[int]) -> List[Dict[str, torch.Tensor]]:
        return [self._copy_row(index) for index in tensor_indices]

    def get_batch(self, tensor_indices: List[int]) -> Dict[str, torch.Tensor]:
        if not tensor_indices:
            return {}
        schema_ids = [self._schema_of_index[index] for index in tensor_indices]
        first_schema = schema_ids[0]
        if any(schema_id != first_schema for schema_id in schema_ids):
            raise ValueError("get_batch requires all tensor indices to use one TensorSchema.")
        columns = self._columns[first_schema]
        if tensor_indices == list(range(tensor_indices[0], tensor_indices[0] + len(tensor_indices))):
            start = tensor_indices[0]
            end = tensor_indices[-1] + 1
            return {
                key: self._copy_slice(column, start, end)
                for key, column in columns.items()
            }
        indices = np.asarray(tensor_indices, dtype=np.intp)
        # NumPy 高级索引复制所选行，返回张量不会引用后续可能复用的 mmap 行。
        return {
            key: torch.from_numpy(column.mmap[indices])
            for key, column in columns.items()
        }

    def release(self, tensor_indices: List[int]) -> None:
        for index in tensor_indices:
            if self._schema_of_index.pop(index, None) is None:
                continue
            heapq.heappush(self._free_indices, index)

    def clear(self) -> None:
        self._schema_of_index.clear()
        self._free_indices.clear()
        self._next_index = 0

    def checkpoint(self, dest: Path) -> TensorCheckpointInfo:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        self._flush()
        relative_files: List[str] = ["state.pt"]
        for columns in self._columns.values():
            for column in columns.values():
                relative_files.append(column.relpath)
                dest_file = dest / column.relpath
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self._scratch_dir / column.relpath, dest_file)
        torch.save(self._state_dict(), dest / "state.pt")
        logical_bytes = sum((dest / rel).stat().st_size for rel in relative_files)
        return TensorCheckpointInfo(
            kind=KIND_MMAP_DIR,
            logical_bytes=int(logical_bytes),
            relative_files=tuple(relative_files),
        )

    def restore(self, src: Path) -> None:
        src = Path(src)
        if not src.is_dir():
            raise ValueError(f"mmap restore requires a directory, got {src}")
        state_path = src / "state.pt"
        if not state_path.exists():
            raise ValueError(f"mmap checkpoint missing state.pt: {src}")
        self._close_maps()
        scratch_dir = self._ensure_scratch_dir()
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if str(state.get("kind", KIND_MMAP_DIR)) != KIND_MMAP_DIR:
            raise ValueError(
                f"tensor pool kind mismatch: checkpoint={state.get('kind')!r}, current={KIND_MMAP_DIR!r}"
            )
        self._max_steps = int(state["max_steps"])
        self._schema_of_index = dict(state["schema_of_index"])
        self._free_indices = list(state["free_indices"])
        heapq.heapify(self._free_indices)
        self._next_index = int(state["next_index"])
        self._columns = {}
        for schema_id, fields in state["columns"].items():
            schema_id = int(schema_id)
            restored: Dict[str, _MmapColumn] = {}
            for key, spec in fields.items():
                relpath = str(spec["relpath"])
                dest_file = scratch_dir / relpath
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / relpath, dest_file)
                torch_dtype = getattr(torch, str(spec["dtype"]).split(".")[-1])
                np_dtype = _DTYPE_TO_NUMPY[torch_dtype]
                row_shape = tuple(int(dim) for dim in spec["row_shape"])
                mmap = np.memmap(
                    scratch_dir / relpath,
                    dtype=np_dtype,
                    mode="r+",
                    shape=(self._max_steps, *row_shape) if row_shape else (self._max_steps,),
                )
                restored[key] = _MmapColumn(
                    relpath=relpath,
                    mmap=mmap,
                    torch_dtype=torch_dtype,
                    row_shape=row_shape,
                )
            self._columns[schema_id] = restored
        self._closed = False

    def _state_dict(self) -> dict:
        columns = {}
        for schema_id, fields in self._columns.items():
            columns[schema_id] = {
                key: {
                    "relpath": column.relpath,
                    "dtype": str(column.torch_dtype).replace("torch.", ""),
                    "row_shape": list(column.row_shape),
                }
                for key, column in fields.items()
            }
        return {
            "kind": KIND_MMAP_DIR,
            "max_steps": self._max_steps,
            "schema_of_index": self._schema_of_index,
            "free_indices": self._free_indices,
            "next_index": self._next_index,
            "columns": columns,
        }

    def _flush(self) -> None:
        for columns in self._columns.values():
            for column in columns.values():
                column.mmap.flush()

    def _close_maps(self) -> None:
        for columns in self._columns.values():
            for column in columns.values():
                try:
                    column.mmap.flush()
                except Exception:
                    pass
                del column.mmap
        self._columns = {}
        gc.collect()

    def close(self) -> None:
        """释放 mmap 句柄；scratch 由 ReplayBuffer.close 按 keep_scratch 删除或保留本地。"""
        if self._closed:
            return
        self._close_maps()
        self.clear()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @property
    def memory_bytes(self) -> int:
        return sum(
            int(column.mmap.size) * int(column.mmap.dtype.itemsize)
            for columns in self._columns.values()
            for column in columns.values()
        )


def infer_lazy_memmap_max_steps(
        capacity: int,
        max_step_per_episode: int,
        rehearsal_archive_ratios: Sequence[float] = (),
) -> int:
    """mmap 行预算：(capacity + Σ round(capacity × rehearsal_ratio)) × 单局最大步数。"""
    capacity = int(capacity)
    max_step_per_episode = int(max_step_per_episode)
    if capacity <= 0 or max_step_per_episode <= 0:
        raise ValueError(
            "lazy_memmap max_steps requires capacity > 0 and max_step_per_episode > 0, "
            f"got capacity={capacity}, max_step_per_episode={max_step_per_episode}"
        )
    extra_episodes = sum(
        int(round(capacity * float(ratio)))
        for ratio in rehearsal_archive_ratios
    )
    return (capacity + extra_episodes) * max_step_per_episode


def lazy_memmap_max_steps_of(config: ReplayBufferConfig) -> int:
    """从桥上的 toml_config 字典读取 episode 上限与 rehearsal 比例，再按 capacity 推算。"""
    try:
        packed = info_bridge.get(INFO_TOML_CONFIG)
    except KeyError as exc:
        raise ValueError(
            "lazy_memmap requires info_bridge toml_config "
            "(publish_training_info after TomlTrainingConfig, or wrap tests in info_bridge.scope)"
        ) from exc
    env_config = packed["env_config"]
    stage_task = packed.get("stage_task_config")
    stages = () if stage_task is None else stage_task.stages
    ratios = [stage.rehearsal_archive_ratio for stage in stages]
    max_step = int(env_config.max_step_per_episode)
    if config.agent_capacities:
        # 所有分区共用物理行池，预算须相加；原子批量 stage 预留每分区一局。
        return sum(
            infer_lazy_memmap_max_steps(capacity, max_step, ratios)
            for capacity in config.agent_capacities.values()
        ) + len(config.agent_capacities) * max_step
    return infer_lazy_memmap_max_steps(config.capacity, max_step, ratios) + max_step


def resolve_mmap_scratch_dir(config: ReplayBufferConfig) -> Path:
    """mmap 工作目录：local 为 data_path/scratch_subdir；S3 回退到 samples/replay_scratch。"""
    storage = config.storage
    if isinstance(storage, LocalStorageConfig):
        subdir = str(storage.scratch_subdir or "replay_scratch").strip() or "replay_scratch"
        return Path(storage.data_path) / subdir
    return Path(ResourceDir.SAMPLES.value) / "replay_scratch"


def build_step_tensor_pool(config: ReplayBufferConfig) -> StepTensorPool:
    """按 ReplayBufferConfig.tensor_pool 构造后端；缺省为内存池。"""
    kind = config.tensor_pool
    if kind == "memory":
        return InMemoryStepTensorPool()
    if kind == "lazy_memmap":
        return LazyMemmapStepTensorPool(
            max_steps=lazy_memmap_max_steps_of(config),
            scratch_dir=resolve_mmap_scratch_dir(config),
        )
    raise ValueError(f"unsupported replay_buffer_config.tensor_pool={kind!r}")

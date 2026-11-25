"""Transition与ReplayBuffer之间的列式回放载荷协议。"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class TensorFieldSpec:
    """一个Step张量字段的共享类型描述，不包含张量值或物理存储位置。"""

    key: str
    source_kind: str
    source_dtype: Optional[str]
    shape: Tuple[int, ...]
    source_device: Optional[str] = None
    requires_grad: bool = False


@dataclass(frozen=True)
class TensorRef:
    key: str


@dataclass(frozen=True)
class InlineRef:
    index: int


@dataclass(frozen=True)
class ContainerTemplate:
    kind: str
    items: Tuple[Any, ...]
    module: Optional[str] = None
    class_name: Optional[str] = None


@dataclass(frozen=True)
class OptionalTemplate:
    present: TensorRef
    value: Any


class BatchEncodingUnsupported(ValueError):
    """逐Step结构不一致，无法安全合并为一个Episode列式张量批次。"""


class TensorOwnership(str, Enum):
    """张量写入池时的内部所有权语义，不属于ReplayBuffer公开接口。"""

    BORROWED = "borrowed"
    TRANSFERRED = "transferred"


@dataclass
class PreparedReplayEpisode:
    """
    Transition在采样进程完成的整局列式载荷。

    tensors的第一维统一为Step；template/fields只描述如何无损恢复公开Transition字段，
    inline_rows仅保留无法张量化的轻量业务元数据。
    """

    step_count: int
    tensors: Dict[str, torch.Tensor]
    template: Any
    fields: Tuple[TensorFieldSpec, ...]
    inline_rows: List[Tuple[Any, ...]]
    step_field_layout: Dict[str, Tuple[str, Tuple[Any, ...]]]
    static_fields: Dict[str, Any]


@dataclass(frozen=True)
class ReplayTransitionEnvelope:
    """一个Transition的RPC精简载荷，不携带原始逐Step公开字段。"""

    transition_type: type
    episode_num: int
    starts_with_episode: bool
    has_starts_with_episode: bool
    prepared_episode: PreparedReplayEpisode
    tensor_ownership: TensorOwnership = TensorOwnership.TRANSFERRED


@dataclass(frozen=True)
class ReplayPushEntryKey:
    """一次 push payload 内的轨迹传输条目键。

    entry_index 是本次 push 内同一 AgentName 的轨迹序号（0 起），不是组身份；
    服务端落池前只取 agent_name，不会保存成 ReplayBuffer 分区键。
    """

    agent_name: Any
    entry_index: int


@dataclass(frozen=True)
class ReplayPushEnvelope:
    """ReplayBuffer.push的内部RPC传输信封。

    v1：episodes 键为 AgentName，值为单条 Transition/信封。
    v2：已预封装条目改用 ReplayPushEntryKey，允许同一 AgentName 携带多条
    组轨迹；未预封装的原始值仍以 AgentName 为键（值可为列表）。
    """

    version: int
    episodes: Dict[Any, Any]

    CURRENT_VERSION = 2
    SUPPORTED_VERSIONS = (1, 2)


def encode_replay_push_payload(
        episode_data: Any,
        agent_name: Any = None,
) -> Any:
    """
    在RPC客户端把已预封装Transition替换为精简信封。

    未预封装、第三方或旧插件Transition保持原对象，以便服务端继续走兼容路径；
    同一 AgentName 的多条轨迹全部预封装时按 ReplayPushEntryKey 逐条编码，
    混有未预封装对象时该 AgentName 整体保持原值（列表或单条）整包发送。
    """
    if isinstance(episode_data, ReplayPushEnvelope):
        return episode_data

    if isinstance(episode_data, dict):
        episodes = episode_data
    elif agent_name is not None:
        episodes = {agent_name: episode_data}
    else:
        # 保留ReplayBuffer原有“单Transition必须提供agent_name”的服务端校验。
        return episode_data

    encoded: Dict[Any, Any] = {}
    has_envelope_entry = False
    for name, value in episodes.items():
        items = value if isinstance(value, list) else [value]
        prepared_items = [
            (item, getattr(item, "prepared_replay_episode", None))
            for item in items
        ]
        if items and all(
            isinstance(prepared, PreparedReplayEpisode) for _, prepared in prepared_items
        ):
            has_envelope_entry = True
            for entry_index, (item, prepared) in enumerate(prepared_items):
                encoded[ReplayPushEntryKey(name, entry_index)] = ReplayTransitionEnvelope(
                    transition_type=type(item),
                    episode_num=int(item.episode_num),
                    starts_with_episode=bool(item.starts_with_episode),
                    has_starts_with_episode=True,
                    prepared_episode=prepared,
                )
            continue
        encoded[name] = value

    if not has_envelope_entry:
        return episode_data
    return ReplayPushEnvelope(
        version=ReplayPushEnvelope.CURRENT_VERSION,
        episodes=encoded,
    )


def contains_tensor_data(value: Any) -> bool:
    if isinstance(value, (torch.Tensor, np.ndarray, np.generic)):
        return True
    if isinstance(value, dict):
        return any(contains_tensor_data(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_tensor_data(item) for item in value)
    if hasattr(value, "__dict__"):
        return any(contains_tensor_data(item) for item in vars(value).values())
    if hasattr(value, "__slots__"):
        return any(
            contains_tensor_data(getattr(value, slot))
            for slot in value.__slots__
            if hasattr(value, slot)
        )
    return False


def prepare_replay_episode(
        step_payloads: List[Dict[str, Any]],
        step_field_layout: Dict[str, Tuple[str, Tuple[Any, ...]]],
        static_fields: Dict[str, Any],
) -> PreparedReplayEpisode:
    """在Transition所在进程把完整Episode一次性转换成列式张量载荷。"""
    if not step_payloads:
        return PreparedReplayEpisode(
            step_count=0,
            tensors={},
            template=ContainerTemplate(kind="dict", items=()),
            fields=(),
            inline_rows=[],
            step_field_layout=step_field_layout,
            static_fields=static_fields,
        )

    tensor_static_fields = {
        field_name: value
        for field_name, value in static_fields.items()
        if contains_tensor_data(value)
    }
    if tensor_static_fields:
        step_payloads[0]["episode_tensor_fields"] = tensor_static_fields
        for field_name in tensor_static_fields:
            del static_fields[field_name]

    tensors: Dict[str, torch.Tensor] = {}
    field_specs: List[TensorFieldSpec] = []
    inline_rows: List[List[Any]] = [[] for _ in step_payloads]

    def encode(values: List[Any]) -> Any:
        field_key = f"v{len(field_specs)}"
        tensor_value = values_to_storage_tensor(values, field_key)
        if tensor_value is not None:
            tensor, spec = tensor_value
            tensors[spec.key] = tensor
            field_specs.append(spec)
            return TensorRef(spec.key)

        present = [value is not None for value in values]
        if any(present) and not all(present):
            exemplar = next(value for value in values if value is not None)
            mask_key = f"v{len(field_specs)}"
            tensors[mask_key] = torch.tensor(present, dtype=torch.bool)
            field_specs.append(TensorFieldSpec(
                key=mask_key,
                source_kind="python_bool",
                source_dtype="bool",
                shape=(),
            ))
            filled = [
                value if value is not None else empty_value_like(exemplar)
                for value in values
            ]
            return OptionalTemplate(
                present=TensorRef(mask_key),
                value=encode(filled),
            )

        first = values[0]
        if all(isinstance(value, dict) for value in values):
            keys = tuple(first)
            if any(tuple(value) != keys for value in values):
                if any(contains_tensor_data(value) for value in values):
                    raise BatchEncodingUnsupported("Step dict layouts are inconsistent.")
            else:
                return ContainerTemplate(
                    kind="dict",
                    items=tuple(
                        (copy.deepcopy(key), encode([value[key] for value in values]))
                        for key in keys
                    ),
                )
        if (
                isinstance(first, tuple)
                and hasattr(first, "_asdict")
                and all(type(value) is type(first) for value in values)
        ):
            keys = tuple(first._asdict())
            return ContainerTemplate(
                kind="namedtuple",
                items=tuple(
                    (key, encode([value._asdict()[key] for value in values]))
                    for key in keys
                ),
                module=first.__class__.__module__,
                class_name=first.__class__.__name__,
            )
        if isinstance(first, list) and all(isinstance(value, list) for value in values):
            item_count = len(first)
            if all(len(value) == item_count for value in values):
                return ContainerTemplate(
                    kind="list",
                    items=tuple(
                        encode([value[index] for value in values])
                        for index in range(item_count)
                    ),
                )
        if (
                isinstance(first, tuple)
                and all(type(value) is tuple and len(value) == len(first) for value in values)
        ):
            return ContainerTemplate(
                kind="tuple",
                items=tuple(
                    encode([value[index] for value in values])
                    for index in range(len(first))
                ),
            )
        if hasattr(first, "__dict__") and all(type(value) is type(first) for value in values):
            keys = tuple(vars(first))
            if all(tuple(vars(value)) == keys for value in values):
                return ContainerTemplate(
                    kind="custom_object",
                    items=tuple(
                        (key, encode([vars(value)[key] for value in values]))
                        for key in keys
                    ),
                    module=first.__class__.__module__,
                    class_name=first.__class__.__name__,
                )
        if hasattr(first, "__slots__") and all(type(value) is type(first) for value in values):
            slots = tuple(
                slot for slot in first.__slots__
                if all(hasattr(value, slot) for value in values)
            )
            return ContainerTemplate(
                kind="slots_object",
                items=tuple(
                    (slot, encode([getattr(value, slot) for value in values]))
                    for slot in slots
                ),
                module=first.__class__.__module__,
                class_name=first.__class__.__name__,
            )

        if any(contains_tensor_data(value) for value in values):
            raise BatchEncodingUnsupported(
                "heterogeneous Step values containing tensors require the compatibility encoder."
            )
        inline_index = len(inline_rows[0])
        for row, value in zip(inline_rows, values):
            row.append(copy.deepcopy(value))
        return InlineRef(inline_index)

    template = encode(step_payloads)
    if not tensors:
        # 纯字符串/对象metadata同样需要Step物理位置；该内部列只用于分配索引。
        tensors["__step__"] = torch.arange(len(step_payloads), dtype=torch.int64)
        field_specs.append(TensorFieldSpec(
            key="__step__",
            source_kind="python_int",
            source_dtype="int",
            shape=(),
        ))
    return PreparedReplayEpisode(
        step_count=len(step_payloads),
        tensors=tensors,
        template=template,
        fields=tuple(field_specs),
        inline_rows=[tuple(row) for row in inline_rows],
        step_field_layout=step_field_layout,
        static_fields=static_fields,
    )


def empty_value_like(value: Any) -> Any:
    """为可选字段构造仅用于列式占位的同构零值。"""
    if isinstance(value, torch.Tensor):
        return torch.zeros_like(value)
    if isinstance(value, np.ndarray):
        return np.zeros_like(value)
    if isinstance(value, np.generic):
        return type(value)(0)
    if type(value) is bool:
        return False
    if type(value) is int:
        return 0
    if type(value) is float:
        return 0.0
    if type(value) is complex:
        return 0j
    if isinstance(value, dict):
        return {
            copy.deepcopy(key): empty_value_like(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [empty_value_like(item) for item in value]
    if isinstance(value, tuple):
        return tuple(empty_value_like(item) for item in value)
    return copy.deepcopy(value)


def values_to_storage_tensor(
        values: List[Any],
        field_key: str,
) -> Optional[Tuple[torch.Tensor, TensorFieldSpec]]:
    """把同一字段的一整局Step值堆叠成一个CPU张量列。"""
    first = values[0]
    if isinstance(first, torch.Tensor) and all(isinstance(value, torch.Tensor) for value in values):
        if any(
                value.layout != torch.strided
                or value.dtype != first.dtype
                or tuple(value.shape) != tuple(first.shape)
                or value.device != first.device
                for value in values
        ):
            return None
        tensor = torch.stack(
            [value.detach().to("cpu").contiguous() for value in values],
            dim=0,
        )
        return tensor, TensorFieldSpec(
            key=field_key,
            source_kind="torch",
            source_dtype=str(first.dtype),
            shape=tuple(first.shape),
            source_device=str(first.device),
            requires_grad=bool(first.requires_grad),
        )
    if isinstance(first, np.ndarray) and all(isinstance(value, np.ndarray) for value in values):
        if any(
                value.dtype != first.dtype
                or tuple(value.shape) != tuple(first.shape)
                or value.dtype.hasobject
                for value in values
        ):
            return None
        array = np.stack(values)
        try:
            tensor = torch.from_numpy(np.ascontiguousarray(array))
        except (TypeError, ValueError):
            return None
        return tensor, TensorFieldSpec(
            key=field_key,
            source_kind="ndarray",
            source_dtype=str(first.dtype),
            shape=tuple(first.shape),
        )
    if isinstance(first, np.generic) and all(
            isinstance(value, np.generic) and value.dtype == first.dtype
            for value in values
    ):
        array = np.asarray(values)
        try:
            tensor = torch.from_numpy(np.ascontiguousarray(array))
        except (TypeError, ValueError):
            return None
        return tensor, TensorFieldSpec(
            key=field_key,
            source_kind="numpy_scalar",
            source_dtype=str(first.dtype),
            shape=(),
        )
    if type(first) is bool and all(type(value) is bool for value in values):
        return torch.tensor(values, dtype=torch.bool), TensorFieldSpec(
            key=field_key, source_kind="python_bool", source_dtype="bool", shape=()
        )
    if type(first) is int and all(type(value) is int for value in values):
        try:
            tensor = torch.tensor(values, dtype=torch.int64)
        except (RuntimeError, OverflowError):
            return None
        return tensor, TensorFieldSpec(
            key=field_key, source_kind="python_int", source_dtype="int", shape=()
        )
    if type(first) is float and all(type(value) is float for value in values):
        return torch.tensor(values, dtype=torch.float64), TensorFieldSpec(
            key=field_key, source_kind="python_float", source_dtype="float", shape=()
        )
    if type(first) is complex and all(type(value) is complex for value in values):
        return torch.tensor(values, dtype=torch.complex128), TensorFieldSpec(
            key=field_key, source_kind="python_complex", source_dtype="complex", shape=()
        )
    if (
            isinstance(first, list)
            and all(
                isinstance(value, list) and is_numeric_sequence(value)
                for value in values
            )
    ):
        array = np.asarray(values)
        if array.dtype.hasobject or array.shape[0] != len(values):
            return None
        try:
            tensor = torch.from_numpy(np.ascontiguousarray(array))
        except (TypeError, ValueError):
            return None
        return tensor, TensorFieldSpec(
            key=field_key,
            source_kind="numeric_list",
            source_dtype=str(array.dtype),
            shape=tuple(array.shape[1:]),
        )
    return None


def is_numeric_sequence(value: list | tuple) -> bool:
    if not value:
        return True
    for item in value:
        if isinstance(item, (list, tuple)):
            if not is_numeric_sequence(item):
                return False
        elif not isinstance(item, (bool, int, float, complex, np.generic)):
            return False
    return True

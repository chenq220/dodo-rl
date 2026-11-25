"""Torch RPC 内部的回放分片信封。业务层、dispatch、gRPC 都不引用本模块。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from sample.replay.replay_payload import (
    PreparedReplayEpisode,
    ReplayPushEnvelope,
    ReplayTransitionEnvelope,
    TensorOwnership,
)


@dataclass(frozen=True)
class ReplayAgentChunkHeader:
    """仅第 0 块携带的 agent 元数据，后续块只传张量和 inline_rows。"""

    transition_type: type
    episode_num: int
    starts_with_episode: bool
    has_starts_with_episode: bool
    tensor_ownership: TensorOwnership
    template: Any
    fields: Tuple[Any, ...]
    step_field_layout: Dict[str, Tuple[str, Tuple[Any, ...]]]
    static_fields: Dict[str, Any]
    total_step_count: int
    # 源 ReplayPushEnvelope 版本；重组器按该版本构造结果，不把旧键布局标成新协议。
    envelope_version: int = ReplayPushEnvelope.CURRENT_VERSION


@dataclass(frozen=True)
class ReplayAgentChunk:
    tensors: Dict[str, torch.Tensor]
    inline_rows: List[Tuple[Any, ...]]
    header: Optional[ReplayAgentChunkHeader] = None


@dataclass(frozen=True)
class ReplayPushChunkEnvelope:
    """一次 push_with_permit 调用中的步窗口载荷。"""

    version: int
    chunk_index: int
    chunk_count: int
    step_start: int
    step_count: int
    is_last: bool
    episodes: Dict[Any, ReplayAgentChunk]

    CURRENT_VERSION = 1


def envelope_tensor_nbytes(envelope: ReplayPushEnvelope) -> int:
    total = 0
    for item in envelope.episodes.values():
        if not isinstance(item, ReplayTransitionEnvelope):
            continue
        for tensor in item.prepared_episode.tensors.values():
            total += int(tensor.numel() * tensor.element_size())
    return total


def _prepared_episodes(envelope: ReplayPushEnvelope) -> Dict[Any, ReplayTransitionEnvelope]:
    prepared: Dict[Any, ReplayTransitionEnvelope] = {}
    for name, item in envelope.episodes.items():
        if not isinstance(item, ReplayTransitionEnvelope):
            return {}
        prepared[name] = item
    return prepared


def _max_step_count(prepared: Dict[Any, ReplayTransitionEnvelope]) -> int:
    if not prepared:
        return 0
    return max(item.prepared_episode.step_count for item in prepared.values())


def _slice_prepared(
        prepared: PreparedReplayEpisode,
        start: int,
        end: int,
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[Any, ...]]]:
    tensors = {
        key: tensor[start:end].contiguous()
        for key, tensor in prepared.tensors.items()
    }
    return tensors, list(prepared.inline_rows[start:end])


def split_replay_push_envelope(
        envelope: Any,
        chunk_bytes: int,
) -> List[ReplayPushChunkEnvelope]:
    """按整步切列式信封。无需分片时返回空列表，调用方应整包发送。"""
    if chunk_bytes <= 0 or not isinstance(envelope, ReplayPushEnvelope):
        return []
    prepared = _prepared_episodes(envelope)
    if not prepared:
        return []
    max_steps = _max_step_count(prepared)
    nbytes = envelope_tensor_nbytes(envelope)
    if max_steps <= 0 or nbytes <= chunk_bytes:
        return []
    window = max(1, (max_steps * chunk_bytes) // nbytes)
    if window >= max_steps:
        return []
    chunk_count = (max_steps + window - 1) // window
    chunks: List[ReplayPushChunkEnvelope] = []
    for chunk_index, step_start in enumerate(range(0, max_steps, window)):
        step_end = min(step_start + window, max_steps)
        episodes: Dict[Any, ReplayAgentChunk] = {}
        for name, item in prepared.items():
            episode = item.prepared_episode
            if step_start >= episode.step_count and chunk_index != 0:
                continue
            tensors, inline_rows = _slice_prepared(
                episode,
                min(step_start, episode.step_count),
                min(step_end, episode.step_count),
            )
            header = None
            if chunk_index == 0:
                header = ReplayAgentChunkHeader(
                    transition_type=item.transition_type,
                    episode_num=item.episode_num,
                    starts_with_episode=item.starts_with_episode,
                    has_starts_with_episode=item.has_starts_with_episode,
                    tensor_ownership=item.tensor_ownership,
                    template=episode.template,
                    fields=episode.fields,
                    step_field_layout=episode.step_field_layout,
                    static_fields=episode.static_fields,
                    total_step_count=episode.step_count,
                    envelope_version=envelope.version,
                )
            episodes[name] = ReplayAgentChunk(
                tensors=tensors,
                inline_rows=inline_rows,
                header=header,
            )
        chunks.append(ReplayPushChunkEnvelope(
            version=ReplayPushChunkEnvelope.CURRENT_VERSION,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            step_start=step_start,
            step_count=step_end - step_start,
            is_last=chunk_index == chunk_count - 1,
            episodes=episodes,
        ))
    return chunks


def assemble_replay_push_envelope(chunks: List[ReplayPushChunkEnvelope]) -> ReplayPushEnvelope:
    if not chunks:
        raise ValueError("cannot assemble replay push from empty chunks")
    expected = chunks[0].chunk_count
    if len(chunks) != expected:
        raise ValueError(f"replay push chunk count mismatch: got {len(chunks)}, expected {expected}")
    for index, chunk in enumerate(chunks):
        if chunk.version != ReplayPushChunkEnvelope.CURRENT_VERSION:
            raise ValueError(f"unsupported ReplayPushChunkEnvelope version: {chunk.version}")
        if chunk.chunk_index != index:
            raise ValueError(f"replay push chunks are out of order: index={chunk.chunk_index}, expected={index}")
        if chunk.chunk_count != expected:
            raise ValueError("replay push chunk_count changed during transfer")
        if chunk.is_last != (index == expected - 1):
            raise ValueError("replay push is_last flag does not match chunk index")

    headers: Dict[Any, ReplayAgentChunkHeader] = {}
    tensor_parts: Dict[Any, Dict[str, List[torch.Tensor]]] = {}
    inline_parts: Dict[Any, List[Tuple[Any, ...]]] = {}
    for chunk in chunks:
        for name, agent_chunk in chunk.episodes.items():
            if agent_chunk.header is not None:
                headers[name] = agent_chunk.header
            if name not in tensor_parts:
                tensor_parts[name] = {key: [] for key in agent_chunk.tensors}
                inline_parts[name] = []
            for key, tensor in agent_chunk.tensors.items():
                tensor_parts[name].setdefault(key, []).append(tensor)
            inline_parts[name].extend(agent_chunk.inline_rows)

    if set(headers) != set(tensor_parts):
        raise ValueError("replay push chunk agents do not match header agents")

    envelope_versions = {header.envelope_version for header in headers.values()}
    if len(envelope_versions) > 1:
        raise ValueError(
            f"replay push chunks carry mixed envelope versions: {sorted(envelope_versions)}"
        )
    envelope_version = next(iter(envelope_versions)) if envelope_versions else (
        ReplayPushEnvelope.CURRENT_VERSION
    )

    episodes: Dict[Any, ReplayTransitionEnvelope] = {}
    for name, header in headers.items():
        tensors = {
            key: torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            for key, parts in tensor_parts[name].items()
        }
        inline_rows = inline_parts[name]
        if len(inline_rows) != header.total_step_count:
            raise ValueError(
                f"assembled inline_rows length {len(inline_rows)} != total_step_count {header.total_step_count}"
            )
        for key, tensor in tensors.items():
            if tensor.shape[0] != header.total_step_count:
                raise ValueError(
                    f"assembled field {key!r} length {tensor.shape[0]} != total_step_count {header.total_step_count}"
                )
        episodes[name] = ReplayTransitionEnvelope(
            transition_type=header.transition_type,
            episode_num=header.episode_num,
            starts_with_episode=header.starts_with_episode,
            has_starts_with_episode=header.has_starts_with_episode,
            prepared_episode=PreparedReplayEpisode(
                step_count=header.total_step_count,
                tensors=tensors,
                template=header.template,
                fields=header.fields,
                inline_rows=inline_rows,
                step_field_layout=header.step_field_layout,
                static_fields=header.static_fields,
            ),
            tensor_ownership=header.tensor_ownership,
        )
    return ReplayPushEnvelope(
        version=envelope_version,
        episodes=episodes,
    )

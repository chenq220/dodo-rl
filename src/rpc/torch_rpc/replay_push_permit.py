from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from common.logger import init_logger
from config import TorchRpcConfig
from rpc.torch_rpc.replay_push_chunk import (
    ReplayPushChunkEnvelope,
    assemble_replay_push_envelope,
)


LOG = init_logger("replay_push_permit")
_OCCUPYING_STATES = frozenset({"active", "chunking", "pushing"})


@dataclass
class _Permit:
    token: str
    request_id: str
    sample_role_name: str
    episode_num: int | None
    generation: int | None
    requested_at: float
    expires_at: float
    state: str = "active"
    result: Any = None
    chunks: list[ReplayPushChunkEnvelope] = field(default_factory=list)
    expected_chunk_count: int | None = None


class TorchReplayBufferService:
    """Torch RPC 专属 ReplayBuffer 适配层，在大载荷进入 TensorPipe 前授予准入令牌。"""

    def __init__(self, target: Any, config: TorchRpcConfig) -> None:
        self._target = target
        self._max_concurrent = int(config.max_concurrent_replay_pushes)
        self._lease_timeout = float(config.replay_push_permit_lease_timeout_seconds)
        self._lock = threading.RLock()
        self._permits: dict[str, _Permit] = {}
        self._tokens_by_request: dict[str, str] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    @staticmethod
    def _validate_identity(
            sample_role_name: str,
            request_id: str,
            episode_num: int | None,
            generation: int | None,
    ) -> None:
        if not isinstance(sample_role_name, str) or not sample_role_name.strip():
            raise ValueError("sample_role_name must be a non-empty string")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if episode_num is not None and (isinstance(episode_num, bool) or not isinstance(episode_num, int)):
            raise TypeError("episode_num must be an integer or None")
        if generation is not None and (isinstance(generation, bool) or not isinstance(generation, int)):
            raise TypeError("generation must be an integer or None")

    @staticmethod
    def _same_identity(
            permit: _Permit,
            sample_role_name: str,
            episode_num: int | None,
            generation: int | None,
    ) -> bool:
        return (
            permit.sample_role_name == sample_role_name
            and permit.episode_num == episode_num
            and permit.generation == generation
        )

    def _cleanup_expired_locked(self, now: float) -> None:
        for token, permit in list(self._permits.items()):
            # pushing 已进入 ReplayBuffer.push，不能回收。
            # sample 在 TensorPipe 上传或分片中途崩溃时，permit 仍为 active / chunking。
            if permit.state == "pushing":
                continue
            if permit.expires_at > now:
                continue
            self._permits.pop(token, None)
            if self._tokens_by_request.get(permit.request_id) == token:
                self._tokens_by_request.pop(permit.request_id, None)
            if permit.state in {"active", "chunking"}:
                LOG.warning(
                    f"Expired ReplayBuffer push permit reclaimed: token={token}, "
                    f"sample_role={permit.sample_role_name}, episode={permit.episode_num}, "
                    f"generation={permit.generation}, state={permit.state}"
                )

    @staticmethod
    def _response(permit: _Permit, *, acquired: bool, reason: str = "") -> dict[str, Any]:
        return {
            "acquired": acquired,
            "token": permit.token if acquired else "",
            "sample_role_name": permit.sample_role_name,
            "episode_num": permit.episode_num,
            "generation": permit.generation,
            "requested_at": permit.requested_at,
            "expires_at": permit.expires_at,
            "lease_timeout_seconds": max(0.0, permit.expires_at - permit.requested_at),
            "state": permit.state,
            "reason": reason,
        }

    def acquire_replay_push_permit(
            self,
            sample_role_name: str,
            request_id: str,
            episode_num: int | None = None,
            generation: int | None = None,
    ) -> dict[str, Any]:
        self._validate_identity(sample_role_name, request_id, episode_num, generation)
        now = time.time()
        with self._lock:
            self._cleanup_expired_locked(now)
            existing_token = self._tokens_by_request.get(request_id)
            if existing_token is not None:
                existing = self._permits.get(existing_token)
                if existing is not None:
                    if not self._same_identity(existing, sample_role_name, episode_num, generation):
                        raise ValueError("request_id is already bound to another replay push identity")
                    if existing.state == "active":
                        return self._response(existing, acquired=True, reason="existing")
                    return self._response(existing, acquired=False, reason=existing.state)

            active_count = sum(
                permit.state in _OCCUPYING_STATES for permit in self._permits.values()
            )
            if active_count >= self._max_concurrent:
                denied = _Permit(
                    token="",
                    request_id=request_id,
                    sample_role_name=sample_role_name,
                    episode_num=episode_num,
                    generation=generation,
                    requested_at=now,
                    expires_at=now,
                    state="waiting",
                )
                return self._response(denied, acquired=False, reason="capacity")

            token = uuid.uuid4().hex
            permit = _Permit(
                token=token,
                request_id=request_id,
                sample_role_name=sample_role_name,
                episode_num=episode_num,
                generation=generation,
                requested_at=now,
                expires_at=now + self._lease_timeout,
            )
            self._permits[token] = permit
            self._tokens_by_request[request_id] = token
            return self._response(permit, acquired=True)

    def push_with_permit(
            self,
            permit_token: str,
            episode_data,
            *,
            sample_role_name: str,
            episode_num: int | None = None,
            generation: int | None = None,
            **push_kwargs,
    ) -> Any:
        now = time.time()
        with self._lock:
            self._cleanup_expired_locked(now)
            permit = self._permits.get(permit_token)
            if permit is None:
                raise ValueError("replay push permit is unknown or expired")
            if not self._same_identity(permit, sample_role_name, episode_num, generation):
                raise ValueError("replay push permit does not match sample role, episode, or generation")
            if permit.state == "completed":
                return permit.result
            if permit.state == "pushing":
                return {"__replay_push_permit__": True, "state": "pushing"}
            if isinstance(episode_data, ReplayPushChunkEnvelope):
                assembled = self._stage_chunk_locked(permit, episode_data)
                if assembled is None:
                    return {
                        "__replay_push_permit__": True,
                        "state": "chunking",
                        "chunk_index": episode_data.chunk_index,
                    }
            else:
                if permit.state != "active":
                    raise ValueError(f"replay push permit is not active: state={permit.state}")
                permit.state = "pushing"
                assembled = episode_data

        return self._run_target_push(
            permit,
            assembled,
            sample_role_name=sample_role_name,
            episode_num=episode_num,
            generation=generation,
            **push_kwargs,
        )

    def _stage_chunk_locked(
            self,
            permit: _Permit,
            chunk: ReplayPushChunkEnvelope,
    ) -> Any:
        if permit.state not in {"active", "chunking"}:
            raise ValueError(f"replay push permit is not accepting chunks: state={permit.state}")
        if chunk.version != ReplayPushChunkEnvelope.CURRENT_VERSION:
            raise ValueError(f"unsupported ReplayPushChunkEnvelope version: {chunk.version}")
        if permit.state == "active":
            if chunk.chunk_index != 0:
                raise ValueError("replay push chunks must start at index 0")
            permit.state = "chunking"
            permit.expected_chunk_count = chunk.chunk_count
            permit.chunks = []
        if permit.expected_chunk_count != chunk.chunk_count:
            raise ValueError("replay push chunk_count changed during transfer")
        if chunk.chunk_index != len(permit.chunks):
            raise ValueError(
                f"replay push chunks are out of order: index={chunk.chunk_index}, "
                f"expected={len(permit.chunks)}"
            )
        permit.chunks.append(chunk)
        permit.expires_at = time.time() + self._lease_timeout
        if not chunk.is_last:
            return None
        try:
            assembled = assemble_replay_push_envelope(list(permit.chunks))
        except Exception:
            permit.state = "failed"
            permit.chunks = []
            permit.expires_at = time.time() + self._lease_timeout
            raise
        permit.state = "pushing"
        permit.chunks = []
        return assembled

    def _run_target_push(
            self,
            permit: _Permit,
            episode_data,
            *,
            sample_role_name: str,
            episode_num: int | None,
            generation: int | None,
            **push_kwargs,
    ) -> Any:
        succeeded = False
        result = None
        try:
            result = self._target.push(
                episode_data,
                sample_role_name=sample_role_name,
                episode_num=episode_num,
                generation=generation,
                **push_kwargs,
            )
            succeeded = True
            return result
        finally:
            # handler 一旦开始，无论业务成功或异常都立即释放并保留短期 tombstone；
            # handler 尚未开始时则由 acquire 路径的租约清理回收。
            with self._lock:
                current = self._permits.get(permit.token)
                if current is permit:
                    current.state = "completed" if succeeded else "failed"
                    current.result = result
                    current.expires_at = time.time() + self._lease_timeout
                    current.chunks = []

    def replay_push_permit_result(
            self,
            permit_token: str,
            sample_role_name: str,
    ) -> dict[str, Any]:
        """大载荷调用超时重试撞到 in-flight token 时，供 sample 用轻量 RPC 查询结果。"""
        now = time.time()
        with self._lock:
            self._cleanup_expired_locked(now)
            permit = self._permits.get(permit_token)
            if permit is None:
                return {"__replay_push_permit__": True, "state": "unknown_or_expired"}
            if permit.sample_role_name != sample_role_name:
                return {"__replay_push_permit__": True, "state": "sample_role_mismatch"}
            return {
                "__replay_push_permit__": True,
                "state": permit.state,
                "result": permit.result if permit.state == "completed" else None,
            }

    def release_replay_push_permit(
            self,
            permit_token: str,
            sample_role_name: str,
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._cleanup_expired_locked(now)
            permit = self._permits.get(permit_token)
            if permit is None:
                return {"released": False, "reason": "unknown_or_expired"}
            if permit.sample_role_name != sample_role_name:
                return {"released": False, "reason": "sample_role_mismatch"}
            if permit.state in {"completed", "failed", "released"}:
                return {"released": False, "reason": permit.state}
            if permit.state == "pushing":
                return {"released": False, "reason": "pushing"}
            permit.state = "released"
            permit.expires_at = now + self._lease_timeout
            return {"released": True, "reason": "released"}

    def replay_push_permit_state(self) -> dict[str, Any]:
        """测试和诊断快照；不暴露大载荷。"""
        now = time.time()
        with self._lock:
            self._cleanup_expired_locked(now)
            active = [permit for permit in self._permits.values() if permit.state in _OCCUPYING_STATES]
            return {
                "active_count": len(active),
                "max_concurrent": self._max_concurrent,
                "active_tokens": [permit.token for permit in active],
            }

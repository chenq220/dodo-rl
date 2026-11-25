from __future__ import annotations

import time
import uuid
from typing import Any

import torch.distributed.rpc as torch_rpc

from common.exception import RpcCodecException, RpcServerGone, RpcTransportException, RpcWorldFailure
from common.logger import init_logger
from common.process_shutdown import process_close
from scheduler.artifact import Artifact
from rpc.artifacts import encode_artifacts
from config import TorchRpcConfig
from rpc.dispatch import (
    ARTIFACTS_MANAGER,
    MODEL_MANAGER,
    MONTE_CARLO_METHOD,
    REPLAY_BUFFER,
    SIM_MANAGER,
    TRAIN_TASK_MANAGER,
    RpcRemoteError,
    RpcResponseEnvelope,
    normalize_models_payload,
    prepare_torch_rpc_payload,
)
from rpc.center_lifecycle import is_rpc_center_finished
from rpc.torch_rpc.replay_push_chunk import split_replay_push_envelope
from rpc.torch_rpc.runtime import _remote_call, service_rank, worker_context
from sample.replay.replay_payload import encode_replay_push_payload
from sample.replay.replay_perf import now as perf_now, record_seconds
from utils.task_id import task_id

LOG = init_logger("torch_rpc_client")


class TorchRpcClient:
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_INTERVAL_SECONDS = 5.0

    def __init__(self, service_name: str, call_timeout: float | None = None,
                 method_timeouts: dict[str, float] | None = None):
        self.service_name = service_name
        self.method_timeouts = dict(method_timeouts or {})
        self.call_timeout = 0.0 if call_timeout is not None and call_timeout <= 0 else (
            -1.0 if call_timeout is None else float(call_timeout)
        )

    @staticmethod
    def _is_irrecoverable_agent_error(exc: Exception) -> bool:
        detail = str(exc).lower()
        if any(marker in detail for marker in (
            "epipe", "broken pipe", "eof", "pipe closed", "connection closed",
        )):
            return True
        return "invalid argument" in detail and any(marker in detail for marker in (
            "tensorpipe/transport/uv", "libuv", "connection_impl.cc",
        ))

    def call(self, method: str, *args, _rpc_context: dict[str, Any] | None = None, **kwargs):
        rpc_args = prepare_torch_rpc_payload(args)
        rpc_kwargs = prepare_torch_rpc_payload(kwargs)
        for attempt in range(1, self.DEFAULT_MAX_RETRIES + 1):
            try:
                response: RpcResponseEnvelope = torch_rpc.rpc_sync(
                    service_rank(self.service_name),
                    _remote_call,
                    args=(self.service_name, method, rpc_args, rpc_kwargs),
                    timeout=self.method_timeouts.get(method, self.call_timeout),
                )
                break
            except Exception as exc:
                if self._is_irrecoverable_agent_error(exc):
                    if is_rpc_center_finished():
                        LOG.info(
                            "Torch RPC center finished; treating irrecoverable pipe error "
                            f"as server gone: service={self.service_name}, method={method}, "
                            f"detail={exc}"
                        )
                        raise RpcServerGone(
                            service=self.service_name,
                            method=method,
                            attempts=attempt,
                            interval=0.0,
                            detail=str(exc),
                        ) from exc
                    context = _rpc_context or {}
                    worker, rank = worker_context()
                    failure = RpcWorldFailure(
                        task_id=str(task_id),
                        worker=worker,
                        rank="" if rank is None else rank,
                        sample_role=str(context.get("sample_role", "")),
                        service=self.service_name,
                        method=method,
                        episode=context.get("episode", ""),
                        generation=context.get("generation", ""),
                        detail=str(exc),
                    )
                    LOG.error(str(failure))
                    raise failure from exc
                if attempt < self.DEFAULT_MAX_RETRIES:
                    retry_delay = self.DEFAULT_RETRY_INTERVAL_SECONDS * (2 ** (attempt - 1))
                    LOG.warning(
                        f"Torch RPC unavailable, retry {attempt}/{self.DEFAULT_MAX_RETRIES}: "
                        f"service={self.service_name}, method={method}, "
                        f"retry_in={retry_delay}s, detail={exc}"
                    )
                    time.sleep(retry_delay)
                    continue
                if self.service_name == TRAIN_TASK_MANAGER:
                    raise RpcServerGone(
                        service=self.service_name,
                        method=method,
                        attempts=attempt,
                        interval=self.DEFAULT_RETRY_INTERVAL_SECONDS,
                        detail=str(exc),
                    ) from exc
                raise RpcTransportException(
                    service=self.service_name, method=method, detail=str(exc), status_code="torch_rpc",
                ) from exc
        if not response.ok:
            if response.error_type in {"RpcCodecException", "PicklingError", "UnpicklingError", "EOFError"}:
                raise RpcCodecException(
                    service=self.service_name,
                    method=method,
                    detail=f"{response.error_type}: {response.error_message}",
                )
            raise RpcRemoteError(
                response.error_type,
                response.error_message,
                response.remote_traceback,
                service=self.service_name,
                method=method,
            )
        return response.value

    @process_close
    def close(self) -> None:
        return None


class _ServiceProxy:
    SERVICE_NAME = "RPC"

    def __init__(self, call_timeout: float | None = None):
        self.client = TorchRpcClient(self.SERVICE_NAME, call_timeout=call_timeout)

    def __getattr__(self, method: str):
        if method.startswith("_"):
            raise AttributeError(method)
        return lambda *args, **kwargs: self.client.call(method, *args, **kwargs)

    @process_close
    def close(self) -> None:
        self.client.close()


class TrainTaskManagerTorchRpcClient(_ServiceProxy):
    SERVICE_NAME = TRAIN_TASK_MANAGER

    def report_sample_result(
            self,
            episode_num: int,
            total_rewards: dict[str, float],
            success: bool = True,
            artifacts: list[Artifact] | None = None,
            episode_metrics: dict[str, float] | None = None,
            episode_rewards_details: dict[str, float] | None = None,
            lease_id: str = "",
            step_count: int = 0,
    ):
        return self.client.call(
            "report_sample_result",
            episode_num,
            total_rewards=total_rewards,
            success=success,
            artifacts=encode_artifacts(artifacts),
            episode_metrics=episode_metrics,
            episode_rewards_details=episode_rewards_details,
            lease_id=lease_id,
            step_count=step_count,
        )

    @property
    def sample_task_process(self):
        return self.client.call("sample_task_process")

    @property
    def learning_process(self):
        return self.client.call("learning_process")


class ReplayBufferTorchRpcClient(_ServiceProxy):
    SERVICE_NAME = REPLAY_BUFFER
    requires_replay_push_permit = True

    def __init__(
            self,
            call_timeout: float | None = None,
            config: TorchRpcConfig | None = None,
    ) -> None:
        config = config or TorchRpcConfig()
        super().__init__(
            call_timeout=config.call_timeout_seconds if call_timeout is None else call_timeout
        )
        self.permit_poll_interval_seconds = float(
            config.replay_push_permit_poll_interval_seconds
        )
        self.replay_push_chunk_bytes = int(config.replay_push_chunk_bytes)

    def acquire_replay_push_permit(
            self,
            *,
            sample_role_name: str,
            episode_num: int | None,
            generation: int | None,
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        context = {
            "sample_role": sample_role_name,
            "episode": episode_num,
            "generation": generation,
        }
        while True:
            permit = self.client.call(
                "acquire_replay_push_permit",
                sample_role_name,
                request_id,
                episode_num=episode_num,
                generation=generation,
                _rpc_context=context,
            )
            if permit.get("acquired"):
                return permit
            if permit.get("reason") != "capacity":
                raise RpcTransportException(
                    service=self.SERVICE_NAME,
                    method="acquire_replay_push_permit",
                    detail=f"permit request rejected: {permit}",
                    status_code="torch_rpc_permit",
                )
            time.sleep(self.permit_poll_interval_seconds)

    def release_replay_push_permit(
            self,
            permit: dict[str, Any],
            *,
            sample_role_name: str,
            episode_num: int | None = None,
            generation: int | None = None,
    ) -> dict[str, Any]:
        return self.client.call(
            "release_replay_push_permit",
            permit["token"],
            sample_role_name,
            _rpc_context={
                "sample_role": sample_role_name,
                "episode": episode_num,
                "generation": generation,
            },
        )

    def push_with_permit(
            self,
            permit: dict[str, Any],
            episode_data,
            save=False,
            notify_learning=True,
            agent_name=None,
            sample_role_name="Serial",
            episode_num=None,
            sample_cost_secs=None,
            generation=None,
            sample_version=None,
    ):
        permit_token = permit["token"]
        context = {
            "sample_role": sample_role_name,
            "episode": episode_num,
            "generation": generation,
        }
        world_failed = False
        try:
            # permit 已授予后才进行可能昂贵的整局编码，避免等待者提前制造大载荷。
            encode_started = perf_now()
            rpc_episode_data = encode_replay_push_payload(
                episode_data,
                agent_name=agent_name,
            )
            record_seconds("rpc.encode_push", encode_started)
            push_started = perf_now()
            try:
                payloads = split_replay_push_envelope(
                    rpc_episode_data,
                    int(vars(self).get("replay_push_chunk_bytes", 0) or 0),
                ) or [rpc_episode_data]
                result = None
                for payload in payloads:
                    result = self.client.call(
                        "push_with_permit",
                        permit_token,
                        payload,
                        save=save,
                        notify_learning=notify_learning,
                        agent_name=agent_name,
                        sample_role_name=sample_role_name,
                        episode_num=episode_num,
                        sample_cost_secs=sample_cost_secs,
                        generation=generation,
                        **({"sample_version": sample_version} if sample_version is not None else {}),
                        _rpc_context=context,
                    )
                    while isinstance(result, dict) and result.get("__replay_push_permit__"):
                        state = result.get("state")
                        if state == "completed":
                            return result.get("result")
                        if state == "chunking":
                            break
                        if state != "pushing":
                            raise RpcTransportException(
                                service=self.SERVICE_NAME,
                                method="push_with_permit",
                                detail=f"permit push did not complete: state={state}, token={permit_token}",
                                status_code="torch_rpc_permit",
                            )
                        time.sleep(self.permit_poll_interval_seconds)
                        result = self.client.call(
                            "replay_push_permit_result",
                            permit_token,
                            sample_role_name,
                            _rpc_context=context,
                        )
                return result
            finally:
                record_seconds("rpc.push_wait", push_started)
        except RpcWorldFailure:
            world_failed = True
            raise
        finally:
            # push handler 自身也在 finally 中释放。这里覆盖编码失败/发送前失败；
            # agent 已断裂时不再向同一 pipe 发无意义的 release，交给租约回收。
            if not world_failed:
                try:
                    self.release_replay_push_permit(
                        permit,
                        sample_role_name=sample_role_name,
                        episode_num=episode_num,
                        generation=generation,
                    )
                except RpcWorldFailure:
                    raise
                except Exception as exc:
                    LOG.warning(
                        f"Replay push permit release failed; lease will reclaim it: "
                        f"token={permit_token}, sample_role={sample_role_name}, "
                        f"episode={episode_num}, generation={generation}, detail={exc}"
                    )

    def push(
            self,
            episode_data,
            save=False,
            notify_learning=True,
            agent_name=None,
            sample_role_name="Serial",
            episode_num=None,
            sample_cost_secs=None,
            generation=None,
            sample_version=None,
    ):
        permit = self.acquire_replay_push_permit(
            sample_role_name=sample_role_name,
            episode_num=episode_num,
            generation=generation,
        )
        return self.push_with_permit(
            permit,
            episode_data,
            save=save,
            notify_learning=notify_learning,
            agent_name=agent_name,
            sample_role_name=sample_role_name,
            episode_num=episode_num,
            sample_cost_secs=sample_cost_secs,
            generation=generation,
            **({"sample_version": sample_version} if sample_version is not None else {}),
        )


class ModelManagerTorchRpcClient(_ServiceProxy):
    SERVICE_NAME = MODEL_MANAGER

    @property
    def model_list(self):
        return self.client.call("model_list")

    def submit_model(self, model_name, algo_type, models, **kwargs):
        return self.client.call(
            "submit_model", model_name, algo_type, normalize_models_payload(models), **kwargs,
        )

    def get_model(self, model_name, *, version=None, output_type="state_dict", artifact_names=None):
        if output_type == "nn":
            raise ValueError("RPC ModelManager client does not support output_type='nn'; use 'state_dict' or 'info'.")
        return self.client.call(
            "get_model",
            model_name,
            version=version,
            output_type=output_type,
            artifact_names=artifact_names,
        )


class SimManagerTorchRpcClient(_ServiceProxy):
    SERVICE_NAME = SIM_MANAGER

    def __init__(self, call_timeout: float | None = None):
        self.client = TorchRpcClient(self.SERVICE_NAME, call_timeout=call_timeout,
                                     method_timeouts={"reacquire_simulation": 600.0})

    @process_close
    def close(self) -> None:
        """请求中心强制关闭仿真管理器，再关闭本地传输。"""
        if self.__dict__.get("_closed", False):
            return
        self._closed = True
        try:
            self.client.call("close")
        except RpcServerGone:
            # 中心已确认收口，远端管理器随中心释放；本地关闭仍须完成。
            pass
        finally:
            self.client.close()


class MonteCarloMethodTorchRpcClient(_ServiceProxy):
    SERVICE_NAME = MONTE_CARLO_METHOD


class ArtifactsManagerTorchRpcClient:
    """薄转发 stub：只暴露白名单方法，不读盘、不拼 content。"""

    def __init__(self, call_timeout: float | None = None):
        self.client = TorchRpcClient(ARTIFACTS_MANAGER, call_timeout=call_timeout)

    def register(self, task_id, stage_name, intent):
        return self.client.call("register", task_id, stage_name, intent)

    def put(self, task_id, stage_name, item):
        return self.client.call("put", task_id, stage_name, item)

    def put_bytes(self, task_id, stage_name, item, data):
        return self.client.call("put_bytes", task_id, stage_name, item, data)

    def put_log(self, task_id, role_id, stage_name=None):
        return self.client.call("put_log", task_id, role_id, stage_name)

    def expect_producer(self, task_id, stage_name, producer_id):
        return self.client.call("expect_producer", task_id, stage_name, producer_id)

    def complete(self, task_id, stage_name, producer_id):
        return self.client.call("complete", task_id, stage_name, producer_id)

    @process_close
    def close(self) -> None:
        self.client.close()

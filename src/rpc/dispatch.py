from __future__ import annotations

import traceback
import copy
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping

import torch
from torch import nn
from common.exception import RpcRemoteApplicationException
from rpc.artifacts import decode_artifacts


TRAIN_TASK_MANAGER = "TrainTaskManager"
REPLAY_BUFFER = "ReplayBuffer"
MODEL_MANAGER = "ModelManager"
SIM_MANAGER = "SimManager"
MONTE_CARLO_METHOD = "MonteCarloMethod"
ARTIFACTS_MANAGER = "ArtifactsManager"


@dataclass
class RpcResponseEnvelope:
    ok: bool
    value: Any = None
    error_type: str = ""
    error_message: str = ""
    remote_traceback: str = ""


class RpcRemoteError(RpcRemoteApplicationException):
    def __init__(self, error_type: str, error_message: str, remote_traceback: str = "",
                 service: str = "", method: str = ""):
        self.error_type = error_type
        self.error_message = error_message
        super().__init__(
            service=service,
            method=method,
            detail=f"{error_type}: {error_message}",
            remote_error_type=error_type,
            remote_traceback=remote_traceback,
        )


class RpcMethodDispatcher:
    """按显式方法白名单分发一次 RPC 调用。"""

    def __init__(self, target: Any, methods: Dict[str, Callable]):
        self.target = target
        self.methods = methods

    def invoke(self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if method not in self.methods:
            raise AttributeError(f"RPC method not exposed: {method}")
        return self.methods[method](*args, **kwargs)

    def invoke_enveloped(
            self, method: str, args: tuple[Any, ...], kwargs: dict[str, Any],
    ) -> RpcResponseEnvelope:
        try:
            return RpcResponseEnvelope(ok=True, value=self.invoke(method, args, kwargs))
        except Exception as exc:
            return RpcResponseEnvelope(
                ok=False,
                error_type=exc.__class__.__name__,
                error_message=str(exc),
                remote_traceback=traceback.format_exc(),
            )


def to_plain_data(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain_data(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_plain_data(item) for item in value)
    return value


def normalize_model_payload(model: nn.Module | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(model, nn.Module):
        model = model.state_dict()
    if isinstance(model, Mapping):
        return {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in model.items()
        }
    raise TypeError(f"Unsupported model payload for RPC: {type(model)}")


def normalize_models_payload(
        models: Mapping[str, nn.Module | Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {name: normalize_model_payload(model) for name, model in models.items()}


def prepare_torch_rpc_payload(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """复制 RPC 负载并把非 CPU Tensor 移到 CPU，避免修改调用方持有的训练对象。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if value is None or isinstance(value, (str, bytes, int, float, bool, Enum, type)):
        return value
    memo = {} if memo is None else memo
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if isinstance(value, dict):
        result = value.__class__()
        memo[value_id] = result
        for key, item in value.items():
            result[prepare_torch_rpc_payload(key, memo)] = prepare_torch_rpc_payload(item, memo)
        return result
    if isinstance(value, list):
        result: list[Any] = []
        memo[value_id] = result
        result.extend(prepare_torch_rpc_payload(item, memo) for item in value)
        return result
    if isinstance(value, tuple):
        items = tuple(prepare_torch_rpc_payload(item, memo) for item in value)
        result = value.__class__(*items) if hasattr(value, "_fields") else items
        memo[value_id] = result
        return result
    if isinstance(value, set):
        result = {prepare_torch_rpc_payload(item, memo) for item in value}
        memo[value_id] = result
        return result
    if hasattr(value, "__dict__"):
        result = copy.copy(value)
        memo[value_id] = result
        for name, item in vars(value).items():
            # frozen dataclass会拦截普通setattr；RPC复制属于底层重建，不改变源对象。
            object.__setattr__(result, name, prepare_torch_rpc_payload(item, memo))
        return result
    return value


def build_train_task_manager_methods(target: Any) -> dict[str, Callable]:
    def report_sample_result(
            episode_num, total_rewards, success=True, artifacts=None,
            episode_metrics=None, episode_rewards_details=None, lease_id="",
            step_count=0,
    ):
        return target.report_sample_result(
            episode_num,
            total_rewards=total_rewards,
            success=success,
            artifacts=decode_artifacts(artifacts),
            episode_metrics=episode_metrics,
            episode_rewards_details=episode_rewards_details,
            lease_id=lease_id,
            step_count=step_count,
        )

    return {
        "get_sample_task_info": target.get_sample_task_info,
        "sample_backpressure_state": target.sample_backpressure_state,
        "sample_gate_state": target.sample_gate_state,
        "report_sample_result": report_sample_result,
        "accept_sample_task_result": target.accept_sample_task_result,
        "save_reward_curve": target.save_reward_curve,
        "is_sample_completed": target.is_sample_completed,
        "get_learn_task_info": target.get_learn_task_info,
        "report_learning_result": target.report_learning_result,
        "is_learning_completed": target.is_learning_completed,
        "get_task_manager_info": target.get_task_manager_info,
        "sample_task_process": lambda: target.sample_task_process,
        "learning_process": lambda: target.learning_process,
        "custom_episode": target.custom_episode,
        "learn_batches_update": target.learn_batches_update,
        "restart_training_task": target.restart_training_task,
        "push_info": target.push_info,
        "pull_info": target.pull_info,
    }


def build_replay_buffer_methods(target: Any) -> dict[str, Callable]:
    sample_for_rpc = getattr(target, "sample_for_rpc", target.sample)
    methods = {
        "size": lambda: target.size,
        "state": target.state,
        "agent_size": target.agent_buffer_size,
        "agent_names": target.agent_names,
        "reset_sliding_idx": target.reset_sliding_idx,
        "ordered_buffer": target.ordered_buffer,
        "push": target.push,
        "sample": sample_for_rpc,
        "save_buffer": target.save_buffer,
    }
    for method_name in (
        "acquire_replay_push_permit",
        "push_with_permit",
        "release_replay_push_permit",
        "replay_push_permit_result",
        "replay_push_permit_state",
    ):
        method = getattr(target, method_name, None)
        if callable(method):
            methods[method_name] = method
    return methods


def build_model_manager_methods(target: Any) -> dict[str, Callable]:
    def submit_model(model_name, algo_type, models, **kwargs):
        info = target.submit_model(
            model_name=model_name,
            algo_type=algo_type,
            models=normalize_models_payload(models),
            **kwargs,
        )
        return info.to_dict()

    def get_model(model_name, *, version=None, output_type="state_dict", artifact_names=None):
        if output_type == "nn":
            raise ValueError("RPC ModelManager does not support output_type='nn'; use 'state_dict' or 'info'.")
        return to_plain_data(target.get_model(
            model_name=model_name,
            version=version,
            output_type=output_type,
            artifact_names=artifact_names,
        ))

    return {
        "model_list": lambda: [item.to_dict() for item in target.model_list],
        "submit_model": submit_model,
        "register_model": target.register_model,
        "get_model": get_model,
        "get_champion": lambda name: (
            None if (info := target.get_champion(name)) is None else info.to_dict()
        ),
        "promote_model": lambda name, version=None: target.promote_model(name, version).to_dict(),
        "archive_model": lambda name, version: target.archive_model(name, version).to_dict(),
        "quarantine_model": lambda name, version: target.quarantine_model(name, version).to_dict(),
        "mark_specialist": lambda name, version, tags=None: target.mark_specialist(
            name, version, tags=tags,
        ).to_dict(),
        "rollback_model": lambda name, version=None: target.rollback_model(
            name, version=version,
        ).to_dict(),
        "list_versions": lambda model_name=None, status=None: [
            item.to_dict() for item in target.list_versions(model_name=model_name, status=status)
        ],
        "get_agent_models": lambda: [item.to_dict() for item in target.get_agent_models()],
        "save_models": target.save_models,
    }


def build_sim_manager_methods(target: Any) -> dict[str, Callable]:
    methods = {
        "get_sim_info": target.get_sim_info,
        "release_sim": target.release_sim,
        "get_sim_list": target.get_sim_list,
    }
    for method in ("reacquire_simulation", "validate_sim_lease", "close"):
        if callable(getattr(target, method, None)):
            methods[method] = getattr(target, method)
    return methods


def build_monte_carlo_method_methods(target: Any) -> dict[str, Callable]:
    return {
        "receiver": target.receiver,
        "get_recorded_episode": target.get_recorded_episode,
        "state": lambda: {
            "task_id": target.task_id,
            "write_enabled": target.write_enabled,
            "current_sample_count": target.current_sample_count,
            "local_file": target.local_file,
            "storage_type": target.config.storage_type,
            "s3_key": target.s3_key,
        },
    }


ARTIFACTS_MANAGER_ALLOWED_METHODS = (
    "register",
    "put",
    "put_bytes",
    "put_log",
    "expect_producer",
    "complete",
)


def build_artifacts_manager_methods(target: Any) -> dict[str, Callable]:
    return {name: getattr(target, name) for name in ARTIFACTS_MANAGER_ALLOWED_METHODS}


SERVICE_METHOD_BUILDERS: dict[str, Callable[[Any], dict[str, Callable]]] = {
    TRAIN_TASK_MANAGER: build_train_task_manager_methods,
    REPLAY_BUFFER: build_replay_buffer_methods,
    MODEL_MANAGER: build_model_manager_methods,
    SIM_MANAGER: build_sim_manager_methods,
    MONTE_CARLO_METHOD: build_monte_carlo_method_methods,
    ARTIFACTS_MANAGER: build_artifacts_manager_methods,
}


def build_dispatcher(service_name: str, target: Any) -> RpcMethodDispatcher:
    try:
        methods = SERVICE_METHOD_BUILDERS[service_name](target)
    except KeyError as exc:
        raise ValueError(f"Unknown RPC service: {service_name}") from exc
    return RpcMethodDispatcher(target=target, methods=methods)

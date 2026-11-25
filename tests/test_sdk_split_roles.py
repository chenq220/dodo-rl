from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SDK_SOURCE_DIR = Path(__file__).resolve().parents[1] / "sdk" / "src"
if str(SDK_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_SOURCE_DIR))

from config import GrpcConfig, TorchRpcConfig
from dodorl_sdk.portal import (
    ControlSDK,
    EvalControlSDK,
    EvaluationSDK,
    LearnControlSDK,
    LearnSDK,
    ModelSDK,
    SampleSDK,
    SerialStageTrainSDK,
    SerialTrainSDK,
    StageControlSDK,
    StageLearnSDK,
)


class _RoleRecorder:
    def __init__(self, calls: list[dict], **kwargs) -> None:
        self._calls = calls
        self._calls.append(kwargs)

    def run(self) -> None:
        return None


# portal.run 把 SDK 持有的 ArtifactsManager 原样透传给 Role；绕过 __init__ 的测试用同一 stub 断言透传。
_SDK_ARTIFACTS_MANAGER = SimpleNamespace()


def _grpc_config() -> GrpcConfig:
    return GrpcConfig(
        task_manager_address="127.0.0.1:5001",
        replay_buffer_address="127.0.0.1:5002",
        model_manager_address="127.0.0.1:5003",
        sim_manager_address="127.0.0.1:5004",
        monte_carlo_method_address="127.0.0.1:5005",
    )


def _training_config(rpc_config) -> SimpleNamespace:
    return SimpleNamespace(
        rpc_config=rpc_config,
        stage_task_config=object(),
        train_config=object(),
        env_config=object(),
        sim_config=SimpleNamespace(config_source="local"),
        replay_buffer_config=object(),
        model_manager_config=object(),
        sim_manager_config=object(),
        monte_carlo_method_config=object(),
        step_agents_config=[object()],
        agents_config=[object()],
        learners_config=[object()],
        eval_config=object(),
    )


def _sdk_without_loading(sdk_class, config, **attributes):
    sdk = object.__new__(sdk_class)
    sdk.config = config
    sdk.artifacts_manager = attributes.pop("artifacts_manager", _SDK_ARTIFACTS_MANAGER)
    for name, value in attributes.items():
        setattr(sdk, name, value)
    return sdk


def test_control_sdk_runs_independent_control_role(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "ControlRole", lambda **kwargs: _RoleRecorder(calls, **kwargs))
    config = _training_config(_grpc_config())

    _sdk_without_loading(ControlSDK, config).run()

    assert calls == [{
        "train_config": config.train_config,
        "replay_buffer_config": config.replay_buffer_config,
        "policy_agent_configs": config.step_agents_config,
        "learner_configs": config.learners_config,
        "task_manager_address": "127.0.0.1:5001",
        "replay_buffer_address": "127.0.0.1:5002",
        "sim_manager_config": config.sim_manager_config,
        "sim_manager_address": "127.0.0.1:5004",
        "monte_carlo_method_config": config.monte_carlo_method_config,
        "monte_carlo_method_address": "127.0.0.1:5005",
        "rpc_config": config.rpc_config,
        "artifacts_manager": _SDK_ARTIFACTS_MANAGER,
    }]


def test_learn_sdk_runs_independent_learn_role(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "LearnRole", lambda **kwargs: _RoleRecorder(calls, **kwargs))
    config = _training_config(_grpc_config())

    _sdk_without_loading(LearnSDK, config, role_id="learn-test").run()

    assert calls == [{
        "role_id": "learn-test",
        "train_config": config.train_config,
        "env_config": config.env_config,
        "sim_config": config.sim_config,
        "step_agents_config": config.step_agents_config,
        "learners_config": config.learners_config,
        "task_manager_address": "127.0.0.1:5001",
        "replay_buffer_address": "127.0.0.1:5002",
        "model_manager_address": "127.0.0.1:5003",
        "rpc_config": config.rpc_config,
        "artifacts_manager": _SDK_ARTIFACTS_MANAGER,
    }]


def test_stage_control_sdk_runs_independent_stage_control_role(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "StageControlRole", lambda **kwargs: _RoleRecorder(calls, **kwargs))
    config = _training_config(_grpc_config())

    _sdk_without_loading(StageControlSDK, config, role_id="stage-control-test").run()

    assert calls == [{
        "role_id": "stage-control-test",
        "stage_task_config": config.stage_task_config,
        "env_config": config.env_config,
        "replay_buffer_config": config.replay_buffer_config,
        "task_manager_address": "127.0.0.1:5001",
        "replay_buffer_address": "127.0.0.1:5002",
        "sim_manager_config": config.sim_manager_config,
        "sim_manager_address": "127.0.0.1:5004",
        "monte_carlo_method_config": config.monte_carlo_method_config,
        "monte_carlo_method_address": "127.0.0.1:5005",
        "rpc_config": config.rpc_config,
        "artifacts_manager": _SDK_ARTIFACTS_MANAGER,
    }]


def test_model_sdk_runs_independent_model_role(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "ModelRole", lambda **kwargs: _RoleRecorder(calls, **kwargs))
    config = _training_config(_grpc_config())

    _sdk_without_loading(ModelSDK, config).run()

    assert calls == [{
        "model_manager_config": config.model_manager_config,
        "model_manager_address": "127.0.0.1:5003",
        "rpc_config": config.rpc_config,
        "artifacts_manager": _SDK_ARTIFACTS_MANAGER,
    }]


@pytest.mark.parametrize("sdk_class,role_name", [
    (ControlSDK, "ControlRole"),
    (StageControlSDK, "StageControlRole"),
    (LearnSDK, "LearnRole"),
    (ModelSDK, "ModelRole"),
])
def test_split_role_sdks_accept_torch_rpc_config(monkeypatch, sdk_class, role_name) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, role_name, lambda **kwargs: _RoleRecorder(calls, **kwargs))
    config = _training_config(TorchRpcConfig(world_size=2, init_address="127.0.0.1:53151"))
    attributes = (
        {"role_id": "learn-test"}
        if sdk_class in {LearnSDK, StageControlSDK}
        else {}
    )
    sdk = _sdk_without_loading(sdk_class, config, **attributes)

    sdk.run()

    assert len(calls) == 1
    assert calls[0]["rpc_config"] is config.rpc_config


def test_eval_control_sdk_runs_eval_control_role(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "resolve_eval_config", lambda conf: conf.eval_config)
    monkeypatch.setattr(
        portal,
        "resolve_distributed_evaluation_class",
        lambda conf: lambda **kwargs: _RoleRecorder(calls, **kwargs),
    )
    config = _training_config(_grpc_config())

    _sdk_without_loading(EvalControlSDK, config, role_id="eval-control-test").run()

    assert calls == [{
        "role_id": "eval-control-test",
        "eval_config": config.eval_config,
        "env_config": config.env_config,
        "step_agents_config": config.step_agents_config,
        "replay_buffer_config": config.replay_buffer_config,
        "model_manager_config": config.model_manager_config,
        "sim_manager_config": config.sim_manager_config,
        "monte_carlo_method_config": config.monte_carlo_method_config,
        "task_manager_address": "127.0.0.1:5001",
        "replay_buffer_address": "127.0.0.1:5002",
        "model_manager_address": "127.0.0.1:5003",
        "sim_manager_address": "127.0.0.1:5004",
        "monte_carlo_method_address": "127.0.0.1:5005",
        "rpc_config": config.rpc_config,
        "artifacts_manager": _SDK_ARTIFACTS_MANAGER,
    }]


def test_sample_sdk_eval_branch_runs_sample_role(monkeypatch) -> None:
    from dodorl_sdk import portal
    from config import EvalConfig, SampleType

    calls = []
    patched_agents = [object()]
    monkeypatch.setattr(portal, "resolve_eval_config", lambda conf: conf.eval_config)
    monkeypatch.setattr(portal, "SampleRole", lambda **kwargs: _RoleRecorder(calls, **kwargs))
    monkeypatch.setattr(
        portal,
        "apply_eval_model_file_paths",
        lambda eval_config, agents: patched_agents,
    )
    config = _training_config(_grpc_config())
    config.eval_config = EvalConfig(
        sample_type=SampleType.OFFLINE.value,
        record_sample=True,
    )

    _sdk_without_loading(
        SampleSDK,
        config,
        sample_role_name="sample-0",
        branch="eval",
    ).run()

    assert config.eval_config.sample_type == SampleType.ONLINE.value
    assert config.eval_config.record_sample is True
    assert calls == [{
        "role_id": "sample-0",
        "train_config": config.eval_config,
        "env_config": config.env_config,
        "sim_config": config.sim_config,
        "step_agents_config": patched_agents,
        "learners_config": [],
        "task_manager_address": "127.0.0.1:5001",
        "replay_buffer_address": "127.0.0.1:5002",
        "model_manager_address": "127.0.0.1:5003",
        "sim_manager_address": "127.0.0.1:5004",
        "monte_carlo_method_address": "127.0.0.1:5005",
        "rpc_config": config.rpc_config,
    }]


def test_learn_control_sdk_runs_central_learn_control_role(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "LearnControlRole", lambda **kwargs: _RoleRecorder(calls, **kwargs))
    config = _training_config(_grpc_config())

    _sdk_without_loading(LearnControlSDK, config, role_id="learn-control-test").run()

    assert len(calls) == 1
    assert calls[0]["role_id"] == "learn-control-test"
    assert calls[0]["train_config"] is config.train_config
    assert calls[0]["step_agents_config"] is config.step_agents_config
    assert calls[0]["learners_config"] is config.learners_config
    assert calls[0]["rpc_config"] is config.rpc_config


def test_stage_learn_sdk_runs_stage_learn_role(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "StageLearnRole", lambda **kwargs: _RoleRecorder(calls, **kwargs))
    config = _training_config(_grpc_config())

    _sdk_without_loading(StageLearnSDK, config, role_id="stage-learn-test").run()

    assert len(calls) == 1
    assert calls[0]["role_id"] == "stage-learn-test"
    assert calls[0]["stage_task_config"] is config.stage_task_config
    assert calls[0]["replay_buffer_config"] is config.replay_buffer_config
    assert calls[0]["rpc_config"] is config.rpc_config


def test_sample_sdk_train_branch_keeps_training_configuration(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "SampleRole", lambda **kwargs: _RoleRecorder(calls, **kwargs))
    config = _training_config(_grpc_config())

    _sdk_without_loading(
        SampleSDK,
        config,
        sample_role_name="sample-train-0",
        branch="train",
    ).run()

    assert len(calls) == 1
    assert calls[0]["role_id"] == "sample-train-0"
    assert calls[0]["train_config"] is config.train_config
    assert calls[0]["step_agents_config"] is config.step_agents_config
    assert calls[0]["learners_config"] is config.learners_config


def test_serial_stage_train_sdk_runs_serial_stage_role(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "SerialStageTrainRole", lambda **kwargs: _RoleRecorder(calls, **kwargs))
    config = _training_config(_grpc_config())

    _sdk_without_loading(SerialStageTrainSDK, config).run()

    assert calls == [{
        "stage_task_config": config.stage_task_config,
        "env_config": config.env_config,
        "sim_config": config.sim_config,
        "replay_buffer_config": config.replay_buffer_config,
        "model_manager_config": config.model_manager_config,
        "sim_manager_config": config.sim_manager_config,
        "artifacts_manager": _SDK_ARTIFACTS_MANAGER,
    }]


def test_serial_train_sdk_runs_serial_train_role(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "SerialTrainRole", lambda **kwargs: _RoleRecorder(calls, **kwargs))
    config = _training_config(_grpc_config())

    _sdk_without_loading(SerialTrainSDK, config).run()

    assert calls == [{
        "config": config.train_config,
        "env_config": config.env_config,
        "sim_config": config.sim_config,
        "replay_buffer_config": config.replay_buffer_config,
        "step_agents_config": config.step_agents_config,
        "learners_config": config.learners_config,
        "sim_manager_config": config.sim_manager_config,
        "artifacts_manager": _SDK_ARTIFACTS_MANAGER,
    }]


def test_evaluation_sdk_runs_resolved_evaluation_role(monkeypatch) -> None:
    from dodorl_sdk import portal

    calls = []
    monkeypatch.setattr(portal, "resolve_eval_config", lambda conf: conf.eval_config)
    monkeypatch.setattr(
        portal,
        "resolve_evaluation_class",
        lambda conf: lambda **kwargs: _RoleRecorder(calls, **kwargs),
    )
    config = _training_config(_grpc_config())

    _sdk_without_loading(EvaluationSDK, config).run()

    assert calls == [{
        "eval_config": config.eval_config,
        "agent_config": config.agents_config,
        "env_config": config.env_config,
        "sim_config": config.sim_config,
        "sim_manager_config": config.sim_manager_config,
        "artifacts_manager": _SDK_ARTIFACTS_MANAGER,
    }]

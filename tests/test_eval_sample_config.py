from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from common.portal_eval import bind_eval_config_for_sample
from config import (
    EnvConfig,
    EvalConfig,
    GrpcConfig,
    SampleType,
    TaskConfig,
    TrainConfig,
)
from roles.sample import SampleRole
from scheduler.train_task import TrainTaskManager
from utils.toml_config import TomlTrainingConfig


def _eval_config(tmp_path: Path, **overrides) -> EvalConfig:
    values = {
        "result_save_dir": str(tmp_path / "results"),
        "render_video_dir": str(tmp_path / "recordings"),
        "output_dir": str(tmp_path / "samples"),
        **overrides,
    }
    return EvalConfig(**values)


def test_task_config_declares_shared_sample_fields(
        tmp_path: Path,
) -> None:
    config = _eval_config(tmp_path)

    assert "sample_type" in TaskConfig.__annotations__
    assert "record_sample" in TaskConfig.__annotations__
    assert "sample_type" not in TrainConfig.__annotations__
    assert "record_sample" not in TrainConfig.__annotations__
    assert "sample_type" not in EvalConfig.__annotations__
    assert "record_sample" not in EvalConfig.__annotations__
    assert config.sample_type == SampleType.ONLINE.value
    assert config.record_sample is False


@pytest.mark.parametrize("record_sample", [False, True])
def test_eval_sample_binding_forces_online_and_preserves_recording(
        tmp_path: Path,
        record_sample: bool,
) -> None:
    config = _eval_config(
        tmp_path,
        sample_type=SampleType.OFFLINE.value,
        record_sample=record_sample,
    )

    bound = bind_eval_config_for_sample(config)

    assert bound is config
    assert config.sample_type == SampleType.ONLINE.value
    assert config.record_sample is record_sample


def test_eval_config_toml_fields_are_explicit_before_runtime_binding(
        tmp_path: Path,
) -> None:
    config_file = tmp_path / "eval.toml"
    config_file.write_text(
        "[eval_config]\n"
        'sample_type = "offline"\n'
        "record_sample = true\n"
        f'result_save_dir = "{tmp_path / "results"}"\n'
        f'render_video_dir = "{tmp_path / "recordings"}"\n'
        f'output_dir = "{tmp_path / "samples"}"\n',
        encoding="utf-8",
    )

    config = TomlTrainingConfig(config_file, project_root=tmp_path).eval_config

    assert config.sample_type == SampleType.OFFLINE.value
    assert config.record_sample is True

    bind_eval_config_for_sample(config)

    assert config.sample_type == SampleType.ONLINE.value
    assert config.record_sample is True


@pytest.mark.parametrize("removed_field", ["record_samples", "save_transitions"])
def test_eval_config_toml_rejects_removed_recording_aliases(
        tmp_path: Path,
        removed_field: str,
) -> None:
    config_file = tmp_path / "eval.toml"
    config_file.write_text(
        f"[eval_config]\n{removed_field} = true\n",
        encoding="utf-8",
    )

    with pytest.raises(
            AttributeError,
            match=rf"EvalConfig has no config field '{removed_field}'",
    ):
        TomlTrainingConfig(config_file, project_root=tmp_path)


def test_eval_task_manager_rejects_unbound_offline_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="EvalConfig.sample_type must be online"):
        TrainTaskManager(
            config=_eval_config(tmp_path, sample_type=SampleType.OFFLINE.value),
            policy_agent_configs=[],
            learner_configs=[],
        )


def test_eval_task_tokens_explicitly_use_online_sample_type(tmp_path: Path) -> None:
    task_manager = TrainTaskManager(
        config=_eval_config(tmp_path, max_episode=2),
        policy_agent_configs=[],
        learner_configs=[],
    )
    try:
        assert [token.sample_type for token in task_manager.sample_task_seq] == [
            SampleType.ONLINE.value,
            SampleType.ONLINE.value,
        ]
    finally:
        task_manager.close()


def test_sample_role_rejects_unbound_offline_eval_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="EvalConfig.sample_type must be online"):
        SampleRole(
            role_id="sample-0",
            train_config=_eval_config(tmp_path, sample_type=SampleType.OFFLINE.value),
            env_config=EnvConfig(),
            sim_config=None,
            step_agents_config=[],
            learners_config=[],
            task_manager_address="",
            replay_buffer_address="",
            model_manager_address="",
        )


@pytest.mark.parametrize("record_sample", [False, True])
def test_bound_eval_config_initializes_sample_role_without_dynamic_fields(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        record_sample: bool,
) -> None:
    from roles import sample as sample_module

    config = bind_eval_config_for_sample(
        _eval_config(tmp_path, record_sample=record_sample)
    )
    environment = SimpleNamespace(sim=None, record_caller=None)
    monte_carlo_method = SimpleNamespace(receiver=object())
    connect_sim_values: list[bool] = []

    monkeypatch.setattr(sample_module, "apply_train_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(sample_module, "init_rpc_client_runtime", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sample_module,
        "create_rpc_client",
        lambda rpc_config, service: monte_carlo_method,
    )
    monkeypatch.setattr(sample_module, "prepare_env_agent_names", lambda *args: None)
    monkeypatch.setattr(sample_module, "align_agent_configs", lambda *args: None)
    monkeypatch.setattr(sample_module, "assert_state_shape_matches_env", lambda *args: None)
    monkeypatch.setattr(sample_module, "build_policy_agents", lambda configs: [])
    monkeypatch.setattr(
        sample_module,
        "build_agent_compose",
        lambda sample_config, agents: SimpleNamespace(compose_type="order"),
    )

    def build_env(*args, connect_sim: bool, **kwargs):
        connect_sim_values.append(connect_sim)
        return environment

    monkeypatch.setattr(sample_module, "build_env", build_env)
    monkeypatch.setattr(
        sample_module,
        "RolloutSampleWorker",
        lambda **kwargs: SimpleNamespace(),
    )

    role = SampleRole(
        role_id="sample-0",
        train_config=config,
        env_config=EnvConfig(use_sim=False),
        sim_config=None,
        step_agents_config=[],
        learners_config=[],
        task_manager_address="127.0.0.1:5001",
        replay_buffer_address="127.0.0.1:5002",
        model_manager_address="127.0.0.1:5003",
        monte_carlo_method_address=("127.0.0.1:5005" if record_sample else None),
        rpc_config=GrpcConfig(),
    )

    assert role.train_config is config
    assert connect_sim_values == [True]
    assert environment.record_caller is (
        monte_carlo_method.receiver if record_sample else None
    )

"""ModelManager 自动晋升：按 Agent 独立的学习批次计数。"""
from __future__ import annotations

from pathlib import Path

import torch

from config import ModelManagerConfig
from models import ModelManager
from models.model import ModelStatus


def _submit(
        manager: ModelManager,
        name: str,
        value: int,
        *,
        learning_batch: bool = False,
        reward: float | None = None,
):
    return manager.submit_model(
        model_name=name,
        algo_type="ppo",
        models={"agent_net": {"weight": torch.tensor([value])}},
        is_learning_batch=learning_batch,
        metrics={} if reward is None else {"reward": reward},
    )


def test_auto_promote_counts_learning_batches_per_agent(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(
        tmp_local_dir=str(tmp_path), auto_promote_interval=2,
    ))
    first_a = _submit(manager, "agent_a", 0)
    first_b = _submit(manager, "agent_b", 0)
    candidate_a = _submit(manager, "agent_a", 1, learning_batch=True)
    candidate_b = _submit(manager, "agent_b", 1, learning_batch=True)

    assert manager.get_champion("agent_a").version == first_a.version
    assert manager.get_champion("agent_b").version == first_b.version
    assert candidate_a.status == ModelStatus.CANDIDATE.value
    assert candidate_b.status == ModelStatus.CANDIDATE.value

    champion_a = _submit(manager, "agent_a", 2, learning_batch=True)
    assert manager.get_champion("agent_a").version == champion_a.version
    assert manager.get_champion("agent_b").version == first_b.version

    champion_b = _submit(manager, "agent_b", 2, learning_batch=True)
    assert manager.get_champion("agent_b").version == champion_b.version


def test_auto_promote_minus_one_keeps_latest_learning_model_candidate(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(
        tmp_local_dir=str(tmp_path), auto_promote_interval=-1,
    ))
    first = _submit(manager, "agent", 0)
    candidate = _submit(manager, "agent", 1, learning_batch=True)

    assert manager.get_champion("agent").version == first.version
    assert candidate.status == ModelStatus.CANDIDATE.value
    state = manager.get_model("agent", version=first.version, output_type="state_dict")
    assert state["agent_net"]["weight"].item() == 0


def test_model_manager_restart_does_not_restore_runtime_registry(tmp_path: Path) -> None:
    config = ModelManagerConfig(tmp_local_dir=str(tmp_path), auto_promote_interval=2)
    manager = ModelManager(config)
    first = _submit(manager, "agent", 0)
    assert manager.get_champion("agent").version == first.version

    reloaded = ModelManager(config)

    assert reloaded.get_champion("agent") is None
    assert not (tmp_path / "model_registry.json").exists()


def test_model_pool_file_names_include_task_id_as_second_field(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(tmp_local_dir=str(tmp_path)))
    model_info = _submit(manager, "dynamic_threats", 0)

    artifact_name = Path(model_info.artifacts["agent_net"].local_path).name
    assert artifact_name == (
        f"dynamic_threats__{manager.task_id}__{model_info.version}__agent_net.pth"
    )
    assert manager._metadata_file_name("dynamic_threats", model_info.version) == (
        f"dynamic_threats__{manager.task_id}__{model_info.version}__metadata.json"
    )


def test_auto_promote_optimo_reward_selects_best_model_in_window(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(
        tmp_local_dir=str(tmp_path),
        auto_promote_interval=2,
        auto_promote_optimo_reward=True,
    ))
    _submit(manager, "agent", 0)
    best = _submit(manager, "agent", 1, learning_batch=True, reward=10.0)
    _submit(manager, "agent", 2, learning_batch=True, reward=2.0)

    assert manager.get_champion("agent").version == best.version


def test_auto_promote_without_optimo_reward_keeps_latest_model(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(
        tmp_local_dir=str(tmp_path),
        auto_promote_interval=2,
        auto_promote_optimo_reward=False,
    ))
    _submit(manager, "agent", 0)
    _submit(manager, "agent", 1, learning_batch=True, reward=10.0)
    latest = _submit(manager, "agent", 2, learning_batch=True, reward=2.0)

    assert manager.get_champion("agent").version == latest.version


def test_model_pool_prunes_old_archived_versions_but_keeps_champion(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(
        tmp_local_dir=str(tmp_path), auto_promote_interval=1, max_versions_per_model=2,
    ))
    versions = [_submit(manager, "agent", value, learning_batch=True) for value in range(4)]

    retained = manager.list_versions("agent")
    assert {info.version for info in retained} == {versions[2].version, versions[3].version}
    assert manager.get_champion("agent").version == versions[3].version
    assert not Path(versions[0].artifacts["agent_net"].local_path).exists()
    assert not (tmp_path / manager._metadata_file_name("agent", versions[0].version)).exists()


def test_model_pool_keeps_candidates_in_auto_promote_window(tmp_path: Path) -> None:
    manager = ModelManager(ModelManagerConfig(
        tmp_local_dir=str(tmp_path), auto_promote_interval=3, max_versions_per_model=1,
    ))
    champion = _submit(manager, "agent", 0)
    candidate_one = _submit(manager, "agent", 1, learning_batch=True)
    candidate_two = _submit(manager, "agent", 2, learning_batch=True)

    assert {info.version for info in manager.list_versions("agent")} == {
        champion.version, candidate_one.version, candidate_two.version,
    }
    assert Path(candidate_one.artifacts["agent_net"].local_path).exists()

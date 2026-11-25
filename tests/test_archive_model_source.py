from pathlib import Path

import pytest

from roles.common.archive_model_source import resolve_learner_archive_model_files


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_resolve_models_from_automl_archive_root(tmp_path: Path) -> None:
    models = tmp_path / "artifacts" / "models"
    version = "optimo_6fc524f7_deploy_hold_knight_5"
    actor = models / f"{version}_archer_agent_net.pth"
    critic = models / f"{version}_archer_critic_net.pth"
    _touch(actor)
    _touch(critic)

    resolved = resolve_learner_archive_model_files(
        str(tmp_path), version, "archer", ("agent_net", "critic_net"),
    )

    assert resolved == {"agent_net": str(actor), "critic_net": str(critic)}


def test_resolve_models_rejects_missing_or_ambiguous_components(tmp_path: Path) -> None:
    models = tmp_path / "artifacts" / "models"
    version = "optimo_a_deploy_hold_knight_5"
    _touch(models / f"{version}_knight_agent_net.pth")
    with pytest.raises(ValueError, match="critic_net"):
        resolve_learner_archive_model_files(
            str(tmp_path), version, "knight", ("agent_net", "critic_net"),
        )

    with pytest.raises(ValueError, match="archer_agent_net"):
        resolve_learner_archive_model_files(
            str(tmp_path), version, "archer", ("agent_net",),
        )

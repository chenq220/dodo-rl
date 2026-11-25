"""TOML optional-section enabled 语义。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.toml_config import TomlTrainingConfig


def _load(tmp_path: Path, content: str) -> TomlTrainingConfig:
    path = tmp_path / "training.toml"
    path.write_text(content, encoding="utf-8")
    return TomlTrainingConfig(path, project_root=PROJECT_ROOT)


def test_optional_module_sections_return_none_when_disabled(tmp_path: Path) -> None:
    config = _load(tmp_path, """
[auto_ml_config]
enabled = false
subject_name = "disabled"

[model_manager_config]
enabled = false

[monte_carlo_method_config]
enabled = false
""")

    assert config.base_auto_ml_config is None
    assert config.train_config.auto_ml_config is None
    assert config.model_manager_config is None
    assert config.monte_carlo_method_config is None
    assert config.sim_manager_config is None


def test_optional_module_sections_default_to_enabled(tmp_path: Path) -> None:
    config = _load(tmp_path, """
[auto_ml_config]
subject_name = "enabled"

[model_manager_config]
auto_promote_interval = 7

[monte_carlo_method_config]
sample_size = 123
""")

    assert config.base_auto_ml_config is not None
    assert config.base_auto_ml_config.monitor_enabled is True
    assert config.train_config.auto_ml_config is not None
    assert config.model_manager_config is not None
    assert config.model_manager_config.auto_promote_interval == 7
    assert config.monte_carlo_method_config is not None
    assert config.sim_manager_config is None


def test_disabled_simulation_sections_skip_driver_validation(tmp_path: Path) -> None:
    config = _load(tmp_path, '''
[sim_config]
enabled = false
[sim_manager_config]
enabled = false
[sim_manager_config.instances]
type = "missing"
''')
    assert config.sim_config is None
    assert config.sim_manager_config is None


def test_stage_automl_uses_disabled_global_values_as_defaults(tmp_path: Path) -> None:
    config = _load(tmp_path, """
[auto_ml_config]
enabled = false
subject_name = "stage-defaults"
poll_interval_episodes = 9

[stages.0]
name = "enabled-stage"

[stages.0.auto_ml_overrides]
min_episodes_before_check = 17

[stages.1]
name = "disabled-stage"

[stages.1.auto_ml_overrides]
enabled = false
""")

    enabled_stage, disabled_stage = config.stage_task_config.stages
    assert config.base_auto_ml_config is None
    assert enabled_stage.auto_ml_config is not None
    assert enabled_stage.auto_ml_config.subject_name == "stage-defaults"
    assert enabled_stage.auto_ml_config.poll_interval_episodes == 9
    assert enabled_stage.auto_ml_config.min_episodes_before_check == 17
    assert disabled_stage.auto_ml_config is None


def test_stage_runtime_entry_config_is_loaded_from_toml(tmp_path: Path) -> None:
    config = _load(tmp_path, """
[stage_task_config]
start_stage = 1
stage_model_versions = ["model-stage-0", "model-stage-1"]
skip_stage_model_load_dir = "/tmp/models"
skip_stage_model_version = "explicit-skip"
skip_stage_replay_path = "replay_stage0.bin"

[stages.0]
name = "first"
save_replay_buffer = true

[stages.1]
name = "second"
""")

    assert config.stage_task_config.start_stage == 1
    assert config.stage_task_config.stage_model_versions == ["model-stage-0", "model-stage-1"]
    assert config.stage_task_config.skip_stage_model_load_dir == "/tmp/models"
    assert config.stage_task_config.skip_stage_model_version == "explicit-skip"
    assert config.stage_task_config.skip_stage_replay_path == "replay_stage0.bin"
    assert config.stage_task_config.stages[0].save_replay_buffer is True
    assert [stage.train_config.model_version for stage in config.stage_task_config.stages] == [None, None]


def test_optional_section_enabled_must_be_boolean(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match=r"model_manager_config\].enabled must be bool"):
        _load(tmp_path, """
[model_manager_config]
enabled = "false"
""")


def test_automl_three_search_space_sections_are_parsed(tmp_path: Path) -> None:
    config = _load(tmp_path, """
[auto_ml_config]
subject_name = "three-spaces"

[auto_ml_config.learner_search_space.0]
agent_name = "knight"
path = "algo_config.lr"
low = 0.0001
high = 0.001

[auto_ml_config.env_metadata_search_space.0]
path = "reward_weights.navigation"
type = "float"
low = 0.05
high = 0.5

[auto_ml_config.seed_search]
enabled = true
low = 100
high = 200
""")

    automl = config.base_auto_ml_config
    assert automl.learner_search_space[0].agent_name == "knight"
    assert automl.learner_search_space[0].path == "algo_config.lr"
    assert automl.env_metadata_search_space[0].path == "reward_weights.navigation"
    assert automl.seed_search.low == 100
    assert automl.seed_search.high == 200


def test_automl_business_metric_sections_are_parsed(tmp_path: Path) -> None:
    config = _load(tmp_path, """
[auto_ml_config]
subject_name = "business-metric"

[auto_ml_config.metric_outputs.team_kills]
info_path = "combat.kills"
agent_reduce = "max"
episode_reduce = "sum"

[auto_ml_config.metric_outputs.max_zombies]
metadata_path = "game.max_zombies"

[auto_ml_config.business_metric]
enabled = true
name = "combat_score"
expression = "team_kills + max_zombies"
aggregation = "mean"
window = 25
target = 7.0
deadline_episodes = 100
""")
    automl = config.base_auto_ml_config
    assert automl.metric_outputs["team_kills"].info_path == "combat.kills"
    assert automl.metric_outputs["max_zombies"].metadata_path == "game.max_zombies"
    assert automl.business_metric.name == "combat_score"
    assert automl.business_metric.window == 25


def test_legacy_learner_search_space_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="numbered entries"):
        _load(tmp_path, """
[auto_ml_config]
subject_name = "legacy"

[auto_ml_config.learner_search_space.knight]
include = ["lr", "seed"]
""")

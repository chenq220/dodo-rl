"""TomlTrainingConfig 核心装配：design/eval/branch、编号表、阶段覆盖与字段归一。"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from common.exception import EnvAlgoDimException
from common.info_bridge import publish_training_info
from config import AGENT_ALGO_CONFIG, AgentConfig, AgentType, EnvConfig, LearnerConfig, LocalStorageConfig, ReplayBufferConfig, ResourceDir, TorchRpcConfig, TrainConfig
from roles.common.config_alignment import assert_state_shape_matches_env, inherit_replay_buffer_seed
from sample.replay.step_tensor_pool import (
    infer_lazy_memmap_max_steps,
    resolve_mmap_scratch_dir,
)
from sample.replay_buffer import ReplayBuffer
from utils.toml_config import TomlTrainingConfig


def _load(tmp_path: Path, content: str, *, branch: str = "train") -> TomlTrainingConfig:
    path = tmp_path / "training.toml"
    path.write_text(content, encoding="utf-8")
    return TomlTrainingConfig(path, project_root=tmp_path, branch=branch)


@pytest.mark.parametrize(("algo_type", "expected"), [
    ("qmix", "rnn"),
    ("mappo", "policy_rnn"),
    ("coma", "policy_rnn"),
    ("maddpg", "maddpg_mlp"),
    ("ppo", "sb3"),
    ("a2c", "sb3"),
    ("dqn", "sb3"),
])
def test_training_agent_defaults_to_its_named_learner(tmp_path: Path, algo_type: str, expected: str):
    config = _load(tmp_path, f"""
[agents.0]
name = "fixed"
agent_names = ["a"]
agent_type = "monte_carlo"

[agents.1]
name = "team"
agent_names = ["b"]
obs_dim = 8

[learners.0]
name = "team"
algo_type = "{algo_type}"

[stages.0]
name = "sample_only"
learners = []
""")
    fixed, team = config.agents_config
    assert fixed.agent_type == "monte_carlo"
    assert team.agent_type == expected
    assert type(team.algo_config) is AGENT_ALGO_CONFIG[AgentType(expected)]
    assert team.agent_names == ["b"] and team.obs_dim == 8
    assert config.stage_task_config.stages[0].agents_config[1].agent_type == expected


def test_explicit_agent_type_overrides_learner_default(tmp_path: Path):
    config = _load(tmp_path, """
[agents.0]
name = "team"
agent_names = ["a"]
agent_type = "q_rnn"

[learners.0]
name = "team"
algo_type = "qmix"
""")
    assert config.agents_config[0].agent_type == "q_rnn"
    assert type(config.agents_config[0].algo_config) is AGENT_ALGO_CONFIG[AgentType.Q_RNN]


def test_agent_without_learner_requires_explicit_type(tmp_path: Path):
    with pytest.raises(ValueError, match="Agent 'fixed' has no matching Learner.*agent_type"):
        _load(tmp_path, """
[agents.0]
name = "fixed"
agent_names = ["a"]
model_file_path = "actor.pth"
""", branch="eval")


def test_design_resolves_refs_eval_and_project_root(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """
[design]
root_dir = "{{project_root}}"
resource_dir = "${root_dir}/artifacts"
names = ["a", "b", "c"]
count = "eval: len(names)"
width = 10
area = "eval: width * count"
exact_count = "${count}"
label = "map-${width}"

[train_config]
model_save_dir = "${resource_dir}/models"
max_episode = "${area}"
""",
    )

    assert config.design["root_dir"] == str(tmp_path.resolve())
    assert config.design["resource_dir"] == f"{tmp_path.resolve()}/artifacts"
    assert config.design["count"] == 3
    assert config.design["area"] == 30
    assert config.design["exact_count"] == 3
    assert config.design["label"] == "map-10"
    assert config.train_config.model_save_dir == f"{tmp_path.resolve()}/artifacts/models"
    assert config.train_config.max_episode == 30


def test_unresolved_design_variable_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be resolved: a, b"):
        _load(
            tmp_path,
            """
[design]
a = "${b}"
b = "${a}"
""",
        )


def test_eval_rejects_unsupported_calls(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="Only len"):
        _load(
            tmp_path,
            """
[design]
n = "eval: abs(-1)"
""",
        )


def test_branch_selects_train_or_eval_arm(tmp_path: Path) -> None:
    content = """
[design]
train_sample_count = 3
eval_sample_count = 6

[torch_rpc_config]
world_size = "branch: train=1 + train_sample_count, eval=1 + eval_sample_count"
init_address = "127.0.0.1:53151"
"""
    train = _load(tmp_path, content, branch="train")
    eval_config = _load(tmp_path, content, branch="eval")

    assert train.branch == "train"
    assert isinstance(train.rpc_config, TorchRpcConfig)
    assert train.rpc_config.world_size == 4
    assert eval_config.branch == "eval"
    assert eval_config.rpc_config.world_size == 7


def test_branch_arm_supports_ref_and_eval_prefix(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """
[design]
n = 4
size = "branch: train=${n}, eval=eval: n * 2"

[train_config]
max_episode = "${size}"
""",
        branch="eval",
    )
    assert config.design["size"] == 8
    assert config.train_config.max_episode == 8


def test_missing_branch_arm_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="branch='eval' is not defined"):
        _load(
            tmp_path,
            """
[design]
size = "branch: train=2"
""",
            branch="eval",
        )


def test_invalid_branch_expression_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Invalid branch"):
        _load(
            tmp_path,
            """
[design]
size = "branch: 1 + 2"
""",
        )


def test_numbered_agents_actions_and_learner_alignment(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """
[env_config]
state_dim = 8

[agents.10]
name = "second"
agent_type = "sb3"
agent_names = ["second"]
state_shape = 8

[agents.2]
name = "first"
agent_type = "sb3"
agent_names = ["first"]
hidden_dim = 16
state_shape = 8

[agents.2.actions.0]
action_key = "heading"
add_target = false
allow_null_targets = false

[agents.2.actions.0.parameters.0]
key = "value"
values = [0, 1, 2]

[learners.0]
name = "first"
algo_type = "ppo"
state_shape = 8

[learners.1]
name = "second"
algo_type = "ppo"
state_shape = 8
""",
    )

    assert [agent.name for agent in config.agents_config] == ["first", "second"]
    first = config.agents_config[0]
    assert first.state_shape == 8
    assert first.agent_output_type == "q"
    assert first.define_action[0].action_key == "heading"
    assert first.define_action[0].parameters[0].key == "value"
    assert first.define_action[0].parameters[0].values == [0, 1, 2]
    assert [learner.name for learner in config.learners_config] == ["first", "second"]
    assert config.learners_config[0].state_shape == 8


def test_stage_overrides_and_skip_model_version_compat(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """
[train_config]
max_episode = 100
start_epsilon = 1.0

[agents.0]
name = "uav"
agent_type = "sb3"
agent_names = ["uav"]
hidden_dim = 32

[learners.0]
name = "uav"
algo_type = "ppo"

[learners.0.algo_config]
lr = 0.001

[stage_task_config]
start_stage = 1
stage_model_versions = ["from-stage-0"]

[stages.0]
name = "search"
learners = ["uav"]
clear_replay_buffer = false
rehearsal_archive_ratio = 0.2
rehearsal_archive_select = "newest"
rehearsal_sample_ratio = 0.1
save_replay_buffer = true

[stages.0.train_overrides]
max_episode = 20

[stages.0.agent_overrides.uav]
hidden_dim = 64

[stages.0.learner_overrides.uav.algo_config]
lr = 0.0005
""",
    )

    stage = config.stage_task_config.stages[0]
    assert config.stage_task_config.start_stage == 1
    assert config.stage_task_config.skip_stage_model_version == "from-stage-0"
    assert stage.stage_name == "search"
    assert stage.train_config.max_episode == 20
    assert config.train_config.max_episode == 100
    assert stage.agents_config[0].hidden_dim == 64
    assert config.agents_config[0].hidden_dim == 32
    assert stage.learners_config[0].algo_config.lr == pytest.approx(0.0005)
    assert config.learners_config[0].algo_config.lr == pytest.approx(0.001)
    assert stage.clear_replay_buffer is False
    assert stage.rehearsal_archive_ratio == pytest.approx(0.2)
    assert stage.rehearsal_archive_select == "newest"
    assert stage.rehearsal_sample_ratio == pytest.approx(0.1)
    assert stage.save_replay_buffer is True


def test_stage_learners_subset_swaps_who_learns(tmp_path: Path) -> None:
    """阶段共享全部 Agent；learners 子集决定谁上场学习（原分阶段设计第 6/15 条）。"""
    config = _load(
        tmp_path,
        """
[agents.0]
name = "agent_a"
agent_type = "sb3"
agent_names = ["agent_a"]

[agents.1]
name = "agent_b"
agent_type = "sb3"
agent_names = ["agent_b"]

[learners.0]
name = "agent_a"
algo_type = "ppo"

[learners.1]
name = "agent_b"
algo_type = "ppo"

[stages.0]
name = "a_learns"
learners = ["agent_a"]

[stages.1]
name = "b_learns"
learners = ["agent_b"]
""",
    )

    stage_a, stage_b = config.stage_task_config.stages
    assert [agent.name for agent in stage_a.agents_config] == ["agent_a", "agent_b"]
    assert [agent.name for agent in stage_b.agents_config] == ["agent_a", "agent_b"]
    assert [learner.name for learner in stage_a.learners_config] == ["agent_a"]
    assert [learner.name for learner in stage_b.learners_config] == ["agent_b"]
    assert [learner.name for learner in config.stage_task_config.bootstrap_learners_config] == [
        "agent_a",
        "agent_b",
    ]


def test_empty_stage_learners_means_sample_only(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """
[agents.0]
name = "uav"
agent_type = "sb3"
agent_names = ["uav"]

[learners.0]
name = "uav"
algo_type = "ppo"

[stages.0]
name = "sample_only"
learners = []

[stages.1]
name = "all_learners"
""",
    )

    sample_only, all_learners = config.stage_task_config.stages
    assert sample_only.learners_config == []
    assert [learner.name for learner in all_learners.learners_config] == ["uav"]


def test_stage_env_metadata_and_end_eval_are_isolated(tmp_path: Path) -> None:
    """阶段不再各持一份 EnvConfig，环境切换走 train_overrides.env_metadata。"""
    config = _load(
        tmp_path,
        """
[train_config]
max_episode = 100
mid_eval_enabled = false
end_eval_episodes = 0

[stages.0]
name = "search"

[stages.0.train_overrides]
max_episode = 10
mid_eval_enabled = true
end_eval_episodes = 3

[stages.0.train_overrides.env_metadata]
dynamic_threats_enabled = false

[stages.1]
name = "evade"

[stages.1.train_overrides]
max_episode = 20
end_eval_episodes = 2

[stages.1.train_overrides.env_metadata]
dynamic_threats_enabled = true
""",
    )

    search, evade = config.stage_task_config.stages
    assert search.train_config.env_metadata == {"dynamic_threats_enabled": False}
    assert evade.train_config.env_metadata == {"dynamic_threats_enabled": True}
    assert config.train_config.env_metadata == {}
    assert search.train_config.mid_eval_enabled is True
    assert search.train_config.end_eval_episodes == 3
    assert evade.train_config.mid_eval_enabled is False
    assert evade.train_config.end_eval_episodes == 2




def test_stage_rejects_unknown_learner_and_unsupported_fields(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="unknown learner 'missing'"):
        _load(
            tmp_path,
            """
[agents.0]
name = "uav"
agent_type = "sb3"
agent_names = ["uav"]

[learners.0]
name = "uav"
algo_type = "ppo"

[stages.0]
name = "search"
learners = ["missing"]
""",
        )

    with pytest.raises(AssertionError, match="unsupported fields"):
        _load(
            tmp_path,
            """
[stages.0]
name = "search"
max_episode = 10
""",
        )

    with pytest.raises(AssertionError, match="unknown agent 'ghost'"):
        _load(
            tmp_path,
            """
[agents.0]
name = "uav"
agent_type = "sb3"
agent_names = ["uav"]

[learners.0]
name = "uav"
algo_type = "ppo"

[stages.0]
name = "search"

[stages.0.agent_overrides.ghost]
hidden_dim = 8
""",
        )

    with pytest.raises(AssertionError, match="rehearsal_archive_ratio must be in"):
        _load(
            tmp_path,
            """
[stages.0]
name = "search"
rehearsal_archive_ratio = 1.5
""",
        )

    with pytest.raises(AssertionError, match="rehearsal_archive_select must be one of"):
        _load(
            tmp_path,
            """
[stages.0]
name = "search"
rehearsal_archive_select = "random"
""",
        )


def test_eval_config_defaults_model_path_and_parses_metrics(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """
[train_config]
model_save_dir = "/tmp/train-models"

[eval_config]
max_episode = 3

[eval_config.metric_outputs.success]
info_path = "task.success"
episode_reduce = "mean"
""",
    )

    assert config.eval_config.agent_model_path == ResourceDir.MODELS.value
    assert config.eval_config.max_episode == 3
    assert config.eval_config.metric_outputs["success"]["info_path"] == "task.success"

    explicit = _load(
        tmp_path,
        """
[train_config]
model_save_dir = "/tmp/train-models"

[eval_config]
agent_model_path = "/tmp/eval-only"
""",
    )
    assert explicit.eval_config.agent_model_path == "/tmp/eval-only"


def test_start_stage_zero_does_not_map_skip_model_version(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """
[stage_task_config]
start_stage = 0
stage_model_versions = ["from-stage-0"]

[stages.0]
name = "first"
""",
    )
    assert config.stage_task_config.skip_stage_model_version is None


def test_root_dir_alias_resolves_like_project_root(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """
[design]
root_dir = "{{root_dir}}"
""",
    )
    assert config.design["root_dir"] == str(tmp_path.resolve())


def test_seed_game_config_and_replay_buffer_seed_not_inherited_by_toml(tmp_path: Path) -> None:
    none_seed = _load(
        tmp_path,
        """
[train_config]
seed = "none"

[game_config]
map_width = 300
""",
    )
    assert none_seed.train_config.seed is None
    assert none_seed.replay_buffer_config.seed is None
    assert none_seed.env_config.metadata["game_config"]["map_width"] == 300

    loaded = _load(
        tmp_path,
        """
[train_config]
seed = 42
""",
    )
    assert loaded.train_config.seed == 42
    assert loaded.replay_buffer_config.seed is None

    clamped = _load(
        tmp_path,
        """
[train_config]
seed = 0
""",
    )
    assert clamped.train_config.seed == 1


def test_inherit_replay_buffer_seed_from_train_config() -> None:
    train_config = TrainConfig(seed=42)
    unset = ReplayBufferConfig()
    inherit_replay_buffer_seed(unset, train_config)
    assert unset.seed == 42

    explicit = ReplayBufferConfig(seed=7)
    inherit_replay_buffer_seed(explicit, train_config)
    assert explicit.seed == 7

    none_train = TrainConfig(seed=None)
    remains_none = ReplayBufferConfig()
    inherit_replay_buffer_seed(remains_none, none_train)
    assert remains_none.seed is None


def test_assert_state_shape_matches_env() -> None:
    env_config = EnvConfig(state_dim=8)
    agent = AgentConfig(name="uav", state_shape=8)
    learner = LearnerConfig(name="uav", state_shape=8)
    assert_state_shape_matches_env(env_config, [agent], [learner])

    agent.state_shape = 1
    with pytest.raises(EnvAlgoDimException, match="state_shape"):
        assert_state_shape_matches_env(env_config, [agent], [learner])


def test_read_plugins_config_parses_without_full_assembly(tmp_path: Path) -> None:
    path = tmp_path / "plugins.toml"
    path.write_text(
        """
[design]
plugin_module = "demo_plugin:DemoEnv"

[plugins_config]
disabled = ["environment.demo"]

[plugins_config.environment]
demo = "${plugin_module}"
""",
        encoding="utf-8",
    )

    plugins = TomlTrainingConfig.read_plugins_config(path, project_root=tmp_path)
    assert plugins.disabled == ["environment.demo"]
    assert plugins.locations["environment"]["demo"] == "demo_plugin:DemoEnv"


def test_legacy_top_level_auto_ml_search_space_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Top-level \[auto_ml_search_space\] is removed"):
        _load(
            tmp_path,
            """
[auto_ml_search_space]
""",
        )


def test_custom_agent_plugin_requires_registry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires an activated PluginRegistry"):
        _load(
            tmp_path,
            """
[agents.0]
name = "custom"
agent_type = "project_agent"
agent_names = ["custom"]
""",
        )


def test_infer_lazy_memmap_max_steps_covers_capacity_and_rehearsal() -> None:
    assert infer_lazy_memmap_max_steps(1000, 600, [0.25]) == 750000
    assert infer_lazy_memmap_max_steps(1000, 600, ()) == 600000
    assert infer_lazy_memmap_max_steps(1000, 600, [0.25, 0.25]) == 900000


def test_replay_buffer_storage_defaults_to_local_samples() -> None:
    config = ReplayBufferConfig()
    assert isinstance(config.storage, LocalStorageConfig)
    assert config.storage.data_path == ResourceDir.SAMPLES.value
    assert config.storage.scratch_subdir == "replay_scratch"
    assert config.keep_scratch is False
    assert "max_steps_per_agent" not in {item.name for item in fields(ReplayBufferConfig)}


def test_replay_buffer_nested_storage_table_sets_scratch_subdir(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """
[design]
root_dir = "{{project_root}}"
resource_dir = "${root_dir}/artifacts"

[replay_buffer_config]
tensor_pool = "lazy_memmap"
keep_scratch = true

[replay_buffer_config.storage]
data_path = "${resource_dir}/samples"
scratch_subdir = "mmap_work"

[agents.0]
name = "uav"
agent_type = "sb3"
agent_names = ["uav"]

[learners.0]
name = "uav"
algo_type = "ppo"
""",
    )
    storage = config.replay_buffer_config.storage
    assert isinstance(storage, LocalStorageConfig)
    assert storage.data_path == f"{tmp_path.resolve()}/artifacts/samples"
    assert storage.scratch_subdir == "mmap_work"
    assert config.replay_buffer_config.keep_scratch is True
    assert resolve_mmap_scratch_dir(config.replay_buffer_config) == (
        Path(tmp_path.resolve()) / "artifacts" / "samples" / "mmap_work"
    )


def test_lazy_memmap_max_steps_inferred_from_capacity_rehearsal_and_episode_limit(
        tmp_path: Path,
) -> None:
    config = _load(
        tmp_path,
        """
[env_config]
use_sim = false
max_step_per_episode = 600

[replay_buffer_config]
tensor_pool = "lazy_memmap"

[agents.0]
name = "uav"
agent_type = "sb3"
agent_names = ["uav"]

[learners.0]
name = "uav"
algo_type = "ppo"

[learners.0.algo_config]
capacity = 1000

[stages.0]
name = "search"
learners = ["uav"]
rehearsal_archive_ratio = 0.25
""",
    )
    publish_training_info(config)
    buffer = ReplayBuffer(config.replay_buffer_config)
    buffer.set_agent_capacity(
        config.learners_config[0].name, config.learners_config[0].algo_config.capacity,
    )
    try:
        assert buffer.tensor_pool._max_steps == 750600
    finally:
        buffer.close()


def test_replay_buffer_default_capacity_can_be_overridden(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        """
[replay_buffer_config]
capacity = 5

[agents.0]
name = "uav"
agent_type = "sb3"
agent_names = ["uav"]

[learners.0]
name = "uav"
algo_type = "ppo"
""",
    )
    assert config.replay_buffer_config.capacity == 5


def test_lazy_memmap_max_steps_toml_knob_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AttributeError, match="has no config field 'max_steps_per_agent'"):
        _load(
            tmp_path,
            """
[env_config]
max_step_per_episode = 600

[replay_buffer_config]
tensor_pool = "lazy_memmap"
max_steps_per_agent = 12345

[agents.0]
name = "uav"
agent_type = "sb3"
agent_names = ["uav"]

[learners.0]
name = "uav"
algo_type = "ppo"

[learners.0.algo_config]
capacity = 1000
""",
        )

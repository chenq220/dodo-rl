from pathlib import Path
import sys

import pytest

from common.plugins import (
    GlobalPluginRegistry,
    PluginRegistry,
    PluginsConfig,
    RETIRED_LEARNERS,
    RUNTIME_PLUGINS,
    activate_plugins,
)
from common.exception import PluginImplementationDuplicateException, PluginNameDuplicateException
from portal.plugins.registry import project_plugin_registry
from utils.toml_config import TomlTrainingConfig


def test_sdk_modules_export_extensible_baselines() -> None:
    sdk_source_dir = Path(__file__).resolve().parents[1] / "sdk" / "src"
    if str(sdk_source_dir) not in sys.path:
        sys.path.insert(0, str(sdk_source_dir))
    from dodorl_sdk.modules import (
        AutoML,
        LearnPermitPolicy,
        ModelManager,
        MonteCarloMethod,
        ReplayReadyLearnPermitPolicy,
        RolloutSampleWorker,
        StageTask,
    )

    assert RolloutSampleWorker.__name__ == "RolloutSampleWorker"
    assert StageTask.__name__ == "StageTask"
    assert MonteCarloMethod.__name__ == "MonteCarloMethod"
    assert AutoML.__name__ == "AutoML"
    assert ModelManager.__name__ == "ModelManager"
    assert LearnPermitPolicy.__name__ == "LearnPermitPolicy"
    assert ReplayReadyLearnPermitPolicy.__name__ == "ReplayReadyLearnPermitPolicy"


def test_plugins_config_is_exposed_by_toml(tmp_path: Path) -> None:
    config_file = tmp_path / "training.toml"
    config_file.write_text(
        """
[plugins_config]
disabled = ["environment.demo"]
[plugins_config.environment]
demo = "demo_plugin:DemoEnv"
""",
        encoding="utf-8",
    )
    config = TomlTrainingConfig(config_file, project_root=tmp_path)
    assert config.plugins_config.enabled == []
    assert config.plugins_config.disabled == ["environment.demo"]
    assert config.plugins_config.locations["environment"]["demo"] == "demo_plugin:DemoEnv"


def test_plugin_directory_registers_resources(tmp_path: Path) -> None:
    (tmp_path / "demo_plugin.py").write_text(
        """
class DemoEnv: pass
class DemoSim: pass
""",
        encoding="utf-8",
    )
    config = PluginsConfig(
        enabled=["environment.demo_env", "simulation.demo_sim"],
        locations={
            "environment": {"demo_env": "demo_plugin:DemoEnv"},
            "simulation": {"demo_sim": "demo_plugin:DemoSim"},
            "agent": {}, "learner": {}, "module": {},
        },
    )
    registry = RUNTIME_PLUGINS
    registry.activate(config, [tmp_path])
    assert registry.resolve("env", "demo_env").__name__ == "DemoEnv"
    assert registry.resolve("sim", "demo_sim").__name__ == "DemoSim"


def test_plugin_enabled_item_must_be_declared() -> None:
    config = PluginsConfig(
        enabled=["environment.missing"],
        locations={"environment": {}, "simulation": {}, "agent": {}, "learner": {}, "module": {}},
    )
    with pytest.raises(ValueError, match="undeclared plugin"):
        RUNTIME_PLUGINS.activate(config)


def test_project_global_registry_defaults_to_all_and_toml_can_disable(tmp_path: Path) -> None:
    (tmp_path / "demo_plugin.py").write_text(
        """
class VisibleEnv: pass
class HiddenEnv: pass
""",
        encoding="utf-8",
    )
    project_plugins = GlobalPluginRegistry(
        environment={
            "visible": "demo_plugin:VisibleEnv",
            "hidden": "demo_plugin:HiddenEnv",
        },
    )
    registry = activate_plugins(
        PluginsConfig(disabled=["environment.hidden"]),
        [tmp_path],
        default_plugins=project_plugins,
    )
    assert registry.has_resource("env", "visible")
    assert not registry.has_resource("env", "hidden")
    assert registry.active_plugins == ["environment.visible"]


def test_project_global_registry_contains_native_agents_and_learners() -> None:
    registry = activate_plugins(PluginsConfig(), default_plugins=project_plugin_registry())
    assert registry.resolve("agent", "rnn").__name__ == "RNNAgent"
    assert registry.resolve("agent", "sb3").__name__ == "SB3Agent"
    assert {name: registry.resolve("learner", name).__name__ for name in
            ("qmix", "coma", "mappo", "dqn")} == {
        "qmix": "QMixerLearner", "coma": "COMALearner",
        "mappo": "MAPPOActorCriticLearner", "dqn": "SB3DQNLearner",
    }


@pytest.mark.parametrize("name", sorted(RETIRED_LEARNERS))
def test_retired_learner_reports_migration(name: str) -> None:
    registry = PluginRegistry()
    with pytest.raises(ValueError, match="removed from DodoRL 0.17.0.*custom learner"):
        registry.resolve("learner", name)
    with pytest.raises(ValueError, match="removed from DodoRL 0.17.0"):
        registry.activate(PluginsConfig(enabled=[f"learner.{name}"]))
    assert name not in project_plugin_registry().learner


def test_custom_plugin_can_reuse_retired_name(tmp_path: Path) -> None:
    (tmp_path / "custom_retired_learner.py").write_text("class CustomLearner: pass\n", encoding="utf-8")
    config = PluginsConfig(
        enabled=["learner.iql"],
        locations={"learner": {"iql": "custom_retired_learner:CustomLearner"}},
    )
    registry = PluginRegistry()
    registry.activate(config, [tmp_path])
    assert registry.resolve("learner", "iql").__name__ == "CustomLearner"


def test_enabled_and_disabled_are_mutually_exclusive(tmp_path: Path) -> None:
    config_file = tmp_path / "training.toml"
    config_file.write_text(
        """
[plugins_config]
enabled = ["environment.demo"]
disabled = ["environment.demo"]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        TomlTrainingConfig(config_file, project_root=tmp_path)


def test_duplicate_plugin_name_in_toml_raises_plugin_name_exception(tmp_path: Path) -> None:
    config_file = tmp_path / "training.toml"
    config_file.write_text(
        """
[plugins_config.learner]
qmix = "demo_plugin:FirstLearner"
qmix = "demo_plugin:SecondLearner"
""",
        encoding="utf-8",
    )
    with pytest.raises(PluginNameDuplicateException, match="learner.*qmix"):
        TomlTrainingConfig(config_file, project_root=tmp_path)


def test_project_registry_and_toml_duplicate_name_raises_plugin_name_exception() -> None:
    defaults = GlobalPluginRegistry(learner={"qmix": "demo_plugin:DefaultLearner"})
    extra = PluginsConfig(
        locations={
            "learner": {"qmix": "demo_plugin:CustomLearner"},
            "simulation": {}, "environment": {}, "agent": {}, "module": {},
        },
    )
    with pytest.raises(PluginNameDuplicateException, match="learner.*qmix"):
        activate_plugins(extra, default_plugins=defaults)


def test_sdk_style_disabled_uses_the_same_toml_policy(tmp_path: Path) -> None:
    (tmp_path / "demo_plugin.py").write_text(
        """
class EnabledEnv: pass
class DisabledEnv: pass
""",
        encoding="utf-8",
    )
    config_file = tmp_path / "training.toml"
    config_file.write_text(
        """
[plugins_config]
disabled = ["environment.disabled"]

[plugins_config.environment]
enabled = "demo_plugin:EnabledEnv"
disabled = "demo_plugin:DisabledEnv"
""",
        encoding="utf-8",
    )
    config = TomlTrainingConfig(config_file, project_root=tmp_path)
    registry = activate_plugins(config.plugins_config, [tmp_path])
    assert registry.has_resource("env", "enabled")
    assert not registry.has_resource("env", "disabled")


def test_module_plugin_uses_extensible_parent_without_register_function(tmp_path: Path) -> None:
    (tmp_path / "rollout_plugin.py").write_text(
        """
from scheduler.rollout_sample import RolloutSampleWorker
class CustomRolloutWorker(RolloutSampleWorker):
    pass
""",
        encoding="utf-8",
    )
    config = PluginsConfig(
        enabled=["module.custom_rollout"],
        locations={
            "module": {"custom_rollout": "rollout_plugin:CustomRolloutWorker"},
            "environment": {}, "simulation": {}, "agent": {}, "learner": {},
        },
    )
    RUNTIME_PLUGINS.activate(config, [tmp_path])
    assert RUNTIME_PLUGINS.active_plugins == ["module.custom_rollout"]


def test_module_plugins_with_same_baseline_raise_implementation_exception(tmp_path: Path) -> None:
    (tmp_path / "rollout_plugins.py").write_text(
        """
from scheduler.rollout_sample import RolloutSampleWorker
class FirstRolloutWorker(RolloutSampleWorker): pass
class SecondRolloutWorker(RolloutSampleWorker): pass
""",
        encoding="utf-8",
    )
    config = PluginsConfig(
        locations={
            "module": {
                "first": "rollout_plugins:FirstRolloutWorker",
                "second": "rollout_plugins:SecondRolloutWorker",
            },
            "simulation": {}, "environment": {}, "agent": {}, "learner": {},
        },
    )
    with pytest.raises(PluginImplementationDuplicateException, match="RolloutSampleWorker"):
        RUNTIME_PLUGINS.activate(config, [tmp_path])


def test_custom_replay_and_transition_plugin_bundle(tmp_path: Path) -> None:
    from sample.replay_buffer import ReplayBuffer
    from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition

    (tmp_path / "custom_replay.py").write_text(
        "from sample.replay_buffer import ReplayBuffer\n"
        "from sample.sample_transfer import EpisodeTransition, SAEpisodeTransition\n"
        "class CustomEpisode(EpisodeTransition): pass\n"
        "class CustomSAEpisode(SAEpisodeTransition): pass\n"
        "class CustomReplay(ReplayBuffer):\n"
        "    __dodo_module_plugin_bundle__ = (CustomEpisode, CustomSAEpisode)\n"
    )
    config = PluginsConfig(enabled=["module.custom_replay"], locations={
        "module": {"custom_replay": "custom_replay:CustomReplay"},
        "environment": {}, "simulation": {}, "agent": {}, "learner": {},
    })
    RUNTIME_PLUGINS.activate(config, [tmp_path])
    assert type(ReplayBuffer.__new__(ReplayBuffer)).__name__ == "CustomReplay"
    assert type(EpisodeTransition()).__name__ == "CustomEpisode"
    assert type(SAEpisodeTransition()).__name__ == "CustomSAEpisode"


def test_training_plugin_selectors_dedupes_and_orders() -> None:
    from common.plugins import training_plugin_selectors

    assert training_plugin_selectors(
        sense_name="cooperative_search",
        agent_types=["rnn", "rnn"],
        algo_types=["qmix", "qmix"],
        sim_type="counter",
        module_names=["demo_rollout"],
    ) == [
        "simulation.counter",
        "environment.cooperative_search",
        "agent.rnn",
        "learner.qmix",
        "module.demo_rollout",
    ]


def test_bind_plugins_config_to_training_skips_when_whitelist_set() -> None:
    from common.plugins import PluginsConfig, bind_plugins_config_to_training

    original = PluginsConfig(enabled=["learner.qmix"])
    bound = bind_plugins_config_to_training(
        original,
        sense_name="cooperative_search",
        agent_types=["rnn"],
        algo_types=["qmix"],
        default_plugins=project_plugin_registry(),
    )
    assert bound.enabled == ["learner.qmix"]


def test_bind_plugins_config_to_training_infers_when_empty() -> None:
    from common.plugins import PluginsConfig, bind_plugins_config_to_training

    bound = bind_plugins_config_to_training(
        PluginsConfig(),
        sense_name="cooperative_search",
        agent_types=["rnn"],
        algo_types=["qmix"],
        sim_type=None,
        default_plugins=project_plugin_registry(),
    )
    assert bound.enabled == [
        "environment.cooperative_search",
        "agent.rnn",
        "learner.qmix",
    ]


def test_module_plugin_registration_via_toml_enabled(tmp_path: Path) -> None:
    """单元测试用临时 module 插件验证注册生效，不依赖项目默认 demo。"""
    (tmp_path / "demo_rollout_plugin.py").write_text(
        """
from scheduler.rollout_sample import RolloutSampleWorker


class DemoRolloutSampleWorker(RolloutSampleWorker):
    pass
""",
        encoding="utf-8",
    )
    config = PluginsConfig(
        enabled=["module.demo_rollout"],
        locations={
            "module": {
                "demo_rollout": "demo_rollout_plugin:DemoRolloutSampleWorker",
            },
            "environment": {},
            "simulation": {},
            "agent": {},
            "learner": {},
        },
    )
    registry = activate_plugins(config, [tmp_path])
    assert registry.active_plugins == ["module.demo_rollout"]
    from scheduler.rollout_sample import RolloutSampleWorker

    assert type(RolloutSampleWorker.__new__(RolloutSampleWorker)).__name__ == (
        "DemoRolloutSampleWorker"
    )

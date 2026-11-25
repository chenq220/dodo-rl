from pathlib import Path
import os
import sys

import pytest

from common.plugins import PluginsConfig, activate_plugins


SDK_SOURCE_DIR = Path(__file__).resolve().parents[1] / "sdk" / "src"
if str(SDK_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_SOURCE_DIR))

from dodorl_sdk.plugins import SDK_DEFAULT_PLUGIN_REGISTRY
from dodorl_sdk.portal import SDKPluginRegistry, SampleSDK, SerialTrainSDK


def _locations(**overrides) -> dict[str, dict[str, str]]:
    locations = {
        "simulation": {}, "environment": {}, "agent": {}, "learner": {}, "module": {},
    }
    locations.update(overrides)
    return locations


def _activate_sdk_plugins(config: PluginsConfig, plugin_dirs=()):
    return activate_plugins(
        config,
        plugin_dirs,
        default_plugins=SDK_DEFAULT_PLUGIN_REGISTRY,
        enable_default_plugins=True,
        allow_default_overrides=True,
    )


def test_sdk_defaults_remain_active_when_enabled_only_lists_project_plugins(tmp_path: Path) -> None:
    (tmp_path / "custom_plugins.py").write_text(
        "class CustomSimulation: pass\nclass CustomEnvironment: pass\n",
        encoding="utf-8",
    )
    config = PluginsConfig(
        enabled=["simulation.custom", "environment.custom"],
        locations=_locations(
            simulation={"custom": "custom_plugins:CustomSimulation"},
            environment={"custom": "custom_plugins:CustomEnvironment"},
        ),
    )

    registry = _activate_sdk_plugins(config, [tmp_path])

    assert registry.has_resource("sim", "custom")
    assert registry.has_resource("env", "custom")
    assert registry.resolve("agent", "rnn").__name__ == "RNNAgent"
    assert registry.resolve("learner", "qmix").__name__ == "QMixerLearner"
    assert registry.resolve("agent", "maddpg_mlp").__name__ == "MaddpgAgent"
    assert registry.resolve("learner", "maddpg").__name__ == "MADDPGLearner"
    assert registry.active_plugins[:15] == [
        "agent.rnn", "agent.policy_rnn", "agent.q_rnn", "agent.mappo_rnn", "agent.rnn_ns", "agent.monte_carlo", "agent.sb3",
        "agent.maddpg_mlp", "learner.qmix", "learner.mappo", "learner.ppo", "learner.a2c", "learner.dqn",
        "learner.coma", "learner.maddpg",
    ]


def test_toml_explicit_builtin_declarations_match_shared_defaults(tmp_path: Path, caplog) -> None:
    from portal.plugins.registry import project_plugin_registry
    from utils.toml_config import TomlTrainingConfig

    project_registry = project_plugin_registry()
    assert SDK_DEFAULT_PLUGIN_REGISTRY.agent is project_registry.agent
    assert SDK_DEFAULT_PLUGIN_REGISTRY.learner is project_registry.learner
    config_file = tmp_path / "explicit.toml"
    config_file.write_text(
        '''
[plugins_config]
enabled = ["agent.rnn", "learner.qmix"]

[plugins_config.agent]
rnn = "algos.modules.agents.rnn_agent:RNNAgent"

[plugins_config.learner]
qmix = "algos.training.qmix_learner:QMixerLearner"
''',
        encoding="utf-8",
    )
    config = TomlTrainingConfig(config_file, project_root=tmp_path)

    registry = SDKPluginRegistry().activate(config)

    assert registry.resolve("agent", "rnn").__name__ == "RNNAgent"
    assert registry.resolve("learner", "qmix").__name__ == "QMixerLearner"
    assert "Explicit plugin overrides default" not in caplog.text


def test_explicit_project_plugin_overrides_sdk_default(tmp_path: Path, caplog) -> None:
    (tmp_path / "custom_agent.py").write_text(
        "from algos.modules.agents.rnn_agent import RNNAgent\n"
        "class ProjectRNNAgent(RNNAgent): pass\n",
        encoding="utf-8",
    )
    config = PluginsConfig(
        enabled=["agent.rnn"],
        locations=_locations(agent={"rnn": "custom_agent:ProjectRNNAgent"}),
    )

    registry = _activate_sdk_plugins(config, [tmp_path])

    assert registry.resolve("agent", "rnn").__name__ == "ProjectRNNAgent"
    assert "Explicit plugin overrides default: agent.rnn" in caplog.text


def test_disabled_can_disable_sdk_default_plugin() -> None:
    registry = _activate_sdk_plugins(PluginsConfig(disabled=["agent.rnn"]))

    assert not registry.has_resource("agent", "rnn")
    with pytest.raises(ValueError, match="explicitly disabled.*agent.rnn"):
        registry.resolve("agent", "rnn")


def test_serial_train_sdk_loads_project_only_plugins_and_builds_builtin_qmix(
        tmp_path: Path,
        monkeypatch,
) -> None:
    from dodorl_sdk import portal

    runtime_entries = []
    monkeypatch.setattr(
        "common.portal_entry.log_startup_process",
        lambda role, config_file, role_id=None: runtime_entries.append(
            (role, config_file, role_id),
        ),
    )
    (tmp_path / "custom_plugins.py").write_text(
        "from envs.sense.cooperative_search_env import CooperativeSearchEnv\n"
        "class CustomSimulation: pass\n"
        "class CustomEnvironment(CooperativeSearchEnv): pass\n",
        encoding="utf-8",
    )
    source = (
        Path(__file__).resolve().parents[1]
        / "subjects/cooperative_search/cooperative_search_qmix.toml"
    ).read_text(encoding="utf-8")
    plugin_config = '''
[plugins_config]
enabled = ["environment.custom"]

[plugins_config.environment]
custom = "custom_plugins:CustomEnvironment"

'''
    source = plugin_config + source
    source = source.replace("[design]\n", '[design]\ntask_id = "sdk-task-id"\n', 1)
    source = source.replace('sense_name = "cooperative_search"', 'sense_name = "custom"')
    source = source.replace(
        '[sim_config]\nconfig_source = "local"',
        '[sim_config]\nsim_type = "custom"\nconfig_source = "local"',
    )
    config_file = tmp_path / "training.toml"
    config_file.write_text(source, encoding="utf-8")

    sdk = SerialTrainSDK(plugin_dirs=[tmp_path], config_file=config_file)
    assert str(portal.task_id) == "sdk-task-id"
    assert os.environ["DODO_TASK_ID"] == "sdk-task-id"
    assert runtime_entries == [("serial_train", str(config_file.resolve()), None)]
    SampleSDK(
        plugin_dirs=[tmp_path],
        config_file=config_file,
        sample_role_name="sample-2",
    )
    assert runtime_entries == [
        ("serial_train", str(config_file.resolve()), None),
        ("sample", str(config_file.resolve()), "sample-2"),
    ]
    registry = sdk.plugins.runtime
    assert registry is not None
    assert not registry.has_resource("sim", "custom")
    assert registry.resolve("env", "custom").__name__ == "CustomEnvironment"

    from common.creator import create_learner
    from utils.module_utils import build_env, build_policy_agents

    # 该测试只验证SDK插件装配，不依赖场景外部训练产出的固定威胁权重。
    fixed_agents = [
        config for config in sdk.config.step_agents_config if config.fixed_mode
    ]
    assert [config.name for config in fixed_agents] == ["dynamic_threats"]
    fixed_agents[0].model_file_path = None
    agents = build_policy_agents(sdk.config.step_agents_config)
    env = build_env(
        sdk.config.env_config,
        sdk.config.sim_config,
        agents,
        connect_sim=False,
    )
    try:
        learner = create_learner(
            learner_config=sdk.config.learners_config[0],
            agent=agents[0],
            env=env,
            device=sdk.config.learners_config[0].device,
            net_source="blanco",
        )
        assert type(agents[0]).__name__ == "RNNAgent"
        assert type(learner).__name__ == "QMixerLearner"
    finally:
        env.close()


@pytest.mark.parametrize("explicit_agent_type", [True, False])
def test_custom_agent_and_learner_parse_their_declared_config_classes(
    tmp_path: Path, explicit_agent_type: bool,
) -> None:
    (tmp_path / "custom_algorithms.py").write_text(
        "from dataclasses import dataclass\n"
        "from typing import ClassVar\n"
        "from dodorl_sdk.algorithms import agent_plugin, learner_plugin\n"
        "from config import RnnAgentConfig, QmixConfig\n"
        "from algos.modules.agents.rnn_agent import RNNAgent\n"
        "from algos.training.qmix_learner import QMixerLearner\n"
        "@dataclass\n"
        "class CustomAgentConfig(RnnAgentConfig):\n"
        "    custom_width: int = 7\n"
        "@agent_plugin(CustomAgentConfig, default_obs_agent_id=False)\n"
        "class CustomAgent(RNNAgent): pass\n"
        "@dataclass\n"
        "class CustomLearnerConfig(QmixConfig):\n"
        "    default_agent_type: ClassVar[str] = 'custom'\n"
        "    custom_scale: float = 1.0\n"
        "@learner_plugin(CustomLearnerConfig)\n"
        "class CustomLearner(QMixerLearner): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "custom_environment.py").write_text(
        "from envs.sense.cooperative_search_env import CooperativeSearchEnv\n"
        "class CustomEnvironment(CooperativeSearchEnv): pass\n",
        encoding="utf-8",
    )
    source = (
        Path(__file__).resolve().parents[1]
        / "subjects/cooperative_search/cooperative_search_qmix.toml"
    ).read_text(encoding="utf-8")
    source = '''
[plugins_config]
enabled = ["environment.custom", "agent.custom", "learner.custom"]

[plugins_config.environment]
custom = "custom_environment:CustomEnvironment"

[plugins_config.agent]
custom = "custom_algorithms:CustomAgent"

[plugins_config.learner]
custom = "custom_algorithms:CustomLearner"

''' + source
    source = source.replace('sense_name = "cooperative_search"', 'sense_name = "custom"')
    if explicit_agent_type:
        source = source.replace("[agents.0]\n", '[agents.0]\nagent_type = "custom"\n', 1)
    source = source.replace('obs_agent_id = true\n', '')
    source = source.replace('algo_type = "qmix"', 'algo_type = "custom"')
    source = source.replace(
        '[agents.0.actions.0]',
        '[agents.0.algo_config]\ncustom_width = 23\n\n[agents.0.actions.0]',
    )
    source = source.replace(
        '[learners.0.algo_config]',
        '[learners.0.algo_config]\ncustom_scale = 2.5',
    )
    config_file = tmp_path / "custom.toml"
    config_file.write_text(source, encoding="utf-8")

    sdk = SerialTrainSDK(plugin_dirs=[tmp_path], config_file=config_file)

    assert type(sdk.config.agents_config[0].algo_config).__name__ == "CustomAgentConfig"
    assert sdk.config.agents_config[0].algo_config.custom_width == 23
    assert sdk.config.agents_config[0].obs_agent_id is False
    assert type(sdk.config.learners_config[0].algo_config).__name__ == "CustomLearnerConfig"
    assert sdk.config.learners_config[0].algo_config.custom_scale == 2.5


def test_algorithm_plugin_without_config_declaration_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad_algorithm.py").write_text(
        "from algos.agents import Agent\n"
        "class BadAgent(Agent):\n"
        "    def get_actions(self, *args, **kwargs): return {}, {}\n",
        encoding="utf-8",
    )
    config = PluginsConfig(
        enabled=["agent.bad"],
        locations=_locations(agent={"bad": "bad_algorithm:BadAgent"}),
    )
    registry = _activate_sdk_plugins(config, [tmp_path])

    with pytest.raises(TypeError, match="agent plugin.*does not declare its config class"):
        registry.algorithm_spec("agent", "bad")

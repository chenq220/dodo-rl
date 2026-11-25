from pathlib import Path
import sys

import pytest

from utils.toml_config import TomlTrainingConfig


def test_file_and_text_configs_have_same_resolution(tmp_path):
    text = '[design]\nroot_dir = "{{project_root}}"\n[env_config]\nuse_sim = false\n[env_config.metadata]\nroot = "${root_dir}"\n'
    path = tmp_path / 'rl.toml'
    path.write_text(text)
    file = TomlTrainingConfig(path, project_root=tmp_path)
    memory = TomlTrainingConfig(config_text=text, project_root=tmp_path)
    assert file.env_config == memory.env_config
    assert memory.env_config.metadata['root'] == str(tmp_path)
    assert memory.raw_file == text
    assert memory.config_file == 'training.toml'
    assert TomlTrainingConfig.read_plugins_config(config_text=text).enabled == []


@pytest.mark.parametrize('kwargs', [{}, {'config_file': 'unused', 'config_text': ''}])
def test_source_is_exclusive(kwargs):
    with pytest.raises(ValueError, match='exactly one'):
        TomlTrainingConfig(**kwargs)
    with pytest.raises(ValueError, match='exactly one'):
        TomlTrainingConfig.prepare_runtime_bootstrap(**kwargs)


def test_unified_toml_delegates_only_requested_resolved_section(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import utils.toml_config as toml_module
    calls = []

    class Parser:
        @staticmethod
        def from_dict(values, **kwargs):
            calls.append(values)
            return SimpleNamespace(**values)

    monkeypatch.setattr(toml_module, 'load_sim_module',
                        lambda name: SimpleNamespace(TomlSimulationConfig=Parser))
    text = '''
[design]
root_dir = "{{project_root}}"
[sim_config.metadata]
count = "eval: 2 * 3"
[sim_manager_config.instances]
count = "branch: train=2, eval=1"
[sim_manager_config.instances.metadata]
runtime_dir = "${root_dir}/runtime"
'''
    path = tmp_path / 'combined.toml'
    path.write_text(text)
    file = TomlTrainingConfig(path, project_root=tmp_path, branch='eval')
    memory = TomlTrainingConfig(config_text=text, project_root=tmp_path, branch='eval')
    assert calls == []
    assert file.sim_config == memory.sim_config == {'metadata': {'count': 6}}
    assert all(set(values) == {'sim_config'} for values in calls)
    assert file.sim_manager_config == memory.sim_manager_config
    assert memory.sim_manager_config['instances'] == {
        'count': 1, 'metadata': {'runtime_dir': str(tmp_path / 'runtime')}}
    assert all(set(values) == {'sim_manager_config'} for values in calls[2:])
    assert memory.raw_file == text


def test_sdk_keeps_delegated_config_objects(monkeypatch):
    from types import SimpleNamespace
    import utils.toml_config as toml_module
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sdk/src'))
    from dodorl_sdk import portal
    sim, manager = object(), object()

    class Parser:
        @staticmethod
        def from_dict(values, **kwargs):
            return SimpleNamespace(sim_config=sim, sim_manager_config=manager)

    monkeypatch.setattr(toml_module, 'load_sim_module',
                        lambda name: SimpleNamespace(TomlSimulationConfig=Parser))
    monkeypatch.setattr(portal.SDKPluginRegistry, 'activate', lambda *args: None)
    monkeypatch.setattr(portal.SDKPluginRegistry, 'require_environment', lambda *args: None)
    monkeypatch.setattr(portal, 'initialize_role_runtime', lambda *args, **kwargs: None)
    sdk = portal.ControlSDK(config_text='[sim_config]\nconfig_source = "manager"\n[sim_manager_config]\nuse_sim = true')
    assert sdk.config.sim_config is sim
    assert sdk.config.sim_manager_config is manager


@pytest.mark.parametrize('name', ['ControlSDK', 'StageControlSDK', 'LearnSDK', 'ModelSDK',
                                 'LearnControlSDK', 'StageLearnSDK', 'SampleSDK',
                                 'SerialTrainSDK', 'SerialStageTrainSDK', 'EvaluationSDK', 'EvalControlSDK'])
def test_sdk_role_accepts_text_and_external_sim_objects(name, monkeypatch, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sdk/src'))
    from dodorl_sdk import portal
    monkeypatch.setattr(portal.SDKPluginRegistry, 'activate', lambda *args: None)
    monkeypatch.setattr(portal.SDKPluginRegistry, 'require_environment', lambda *args: None)
    monkeypatch.setattr(portal, 'initialize_role_runtime', lambda *args, **kwargs: None)
    text = '[env_config]\nuse_sim = false\n[log_config]\nenabled_handlers = []\n'
    sim, manager = object(), object()
    kwargs = {'config_text': text, 'sim_config': sim, 'sim_manager_config': manager}
    if name == 'SampleSDK':
        kwargs['sample_role_name'] = 'sample-0'
    sdk = getattr(portal, name)(**kwargs)
    assert sdk.config.raw_file == text
    assert sdk.config.sim_config is sim
    assert sdk.config.sim_manager_config is manager
    assert sdk.config_file is None


def test_optional_config_sections_are_independent_and_cached(monkeypatch):
    from types import SimpleNamespace
    import utils.toml_config as toml_module
    sim, manager = object(), object()
    calls = []

    class Parser:
        @staticmethod
        def from_dict(values, **kwargs):
            calls.append((values, kwargs))
            return SimpleNamespace(sim_config=sim, sim_manager_config=manager)

    def load(name):
        assert name == 'dodo_sim.toml_config'
        return SimpleNamespace(TomlSimulationConfig=Parser)

    monkeypatch.setattr(toml_module, 'load_sim_module', load)
    config = TomlTrainingConfig(config_text='[sim_config]\nconfig_source = "manager"\n[sim_manager_config]\nuse_sim = true')
    assert calls == []
    assert config.env_config.use_sim is True
    assert calls == []
    assert config.sim_manager_config is manager
    assert len(calls) == 1 and set(calls[0][0]) == {'sim_manager_config'}
    assert config.sim_config is sim
    assert config.sim_manager_config is manager
    assert config.sim_config is sim
    assert len(calls) == 2 and set(calls[1][0]) == {'sim_config'}


def test_explicit_simulation_overrides_do_not_load_sdk(monkeypatch):
    import utils.toml_config as toml_module

    def unexpected_load(name):
        pytest.fail(f'Explicitly provided configs must not load {name}')

    monkeypatch.setattr(toml_module, 'load_sim_module', unexpected_load)
    config = TomlTrainingConfig(config_text='[sim_config]\nconfig_source = "manager"')
    sim, manager = object(), object()
    config.sim_config = sim
    config.sim_manager_config = manager
    assert config.sim_config is sim and config.sim_manager_config is manager


@pytest.mark.parametrize('text,attribute', [
    ('', 'sim_manager_config'),
    ('[sim_config]\nenabled = false', 'sim_config'),
    ('[sim_manager_config]\nenabled = false', 'sim_manager_config'),
    ('[sim_manager_config]\nuse_sim = false', 'sim_manager_config'),
    ('[env_config]\nuse_sim = false\n[sim_config]\nunknown = 1', 'sim_config'),
])
def test_missing_or_disabled_optional_section_never_loads_sdk(monkeypatch, text, attribute):
    import utils.toml_config as toml_module
    def unexpected_load(name):
        pytest.fail(f'Inactive optional section tried to import {name}')
    monkeypatch.setattr(toml_module, 'load_sim_module', unexpected_load)
    config = TomlTrainingConfig(config_text=text)
    assert getattr(config, attribute) is None

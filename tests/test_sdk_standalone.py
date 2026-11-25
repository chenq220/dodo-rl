"""Build and exercise the standalone RL SDK without installed SDK dependencies."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def installed_sdk(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sdk-installed")
    subprocess.run(['bash', str(ROOT / 'sdk/build_sdk.sh')],
                   env=dict(os.environ, DODORL_BUILD_PYTHON=sys.executable),
                   check=True, capture_output=True, text=True)
    wheel = next((ROOT / 'sdk/dist').glob('dodorl_sdk-*.whl'))
    target = tmp_path / 'installed'
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
                    '--target', str(target), str(wheel)], check=True, capture_output=True, text=True)
    return target, tmp_path


def test_installed_wheel_excludes_internal_train_entry(installed_sdk):
    from zipfile import ZipFile

    target, tmp_path = installed_sdk
    for relative in ("portal/train.py", "roles/train.py"):
        assert (ROOT / "src" / relative).is_file()
    wheel = next((ROOT / 'sdk/dist').glob('dodorl_sdk-*.whl'))
    with ZipFile(wheel) as package:
        names = package.namelist()
        assert not any(name.startswith('dodorl_sdk/dodorl/portal/') for name in names)
        role_prefix = 'dodorl_sdk/dodorl/roles/'
        role_files = {
            name.removeprefix(role_prefix) for name in names
            if name.startswith(role_prefix) and '/' not in name.removeprefix(role_prefix)
        }
        assert role_files == {
            '__init__.py', 'control.py', 'stage_control.py', 'learn.py', 'model.py',
            'learn_control.py', 'stage_learn_control.py', 'sample.py', 'serial_train.py',
            'serial_stage_train.py', 'evaluation.py', 'eval_control.py',
        }
    probe = r'''
import importlib.util
import sys
sys.path.insert(0, sys.argv[1])
import dodorl_sdk
from dodorl_sdk import portal
assert not hasattr(dodorl_sdk, 'TrainSDK')
assert not hasattr(portal, 'TrainSDK')
assert importlib.util.find_spec('dodorl_sdk.dodorl.portal') is None
for module in ('roles.train', 'roles.random_sampling', 'roles.sample_learning_evaluation'):
    assert importlib.util.find_spec('dodorl_sdk.dodorl.' + module) is None
for module in ('roles.serial_train', 'roles.serial_stage_train', 'roles.control', 'roles.evaluation'):
    assert importlib.util.find_spec('dodorl_sdk.dodorl.' + module) is not None
from dodorl_sdk.portal import SerialTrainSDK, SerialStageTrainSDK, ControlSDK, EvaluationSDK
'''
    result = subprocess.run([sys.executable, '-I', '-c', probe, str(target)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr





def test_installed_wheel_without_sim_sdk(installed_sdk):
    target, tmp_path = installed_sdk
    probe = r'''
import importlib.abc
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])

class RejectOptionalAndSourceImports(importlib.abc.MetaPathFinder):
    attempts = []
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split('.')[0]
        if root == 'dodo_sim':
            self.attempts.append(fullname)
            raise ModuleNotFoundError('SIM is deliberately unavailable', name='dodo_sim')
        if root in {'config','common','envs','sims','roles','engine','utils','rpc','scheduler','sample','algos','models','portal'}:
            raise ImportError('unqualified runtime import: ' + fullname)
sys.meta_path.insert(0, RejectOptionalAndSourceImports())
import dodorl_sdk
from dodorl_sdk import portal, Evaluation, State, Action
from dodorl_sdk.dodorl.common.simulation import create_sim_manager, load_sim_module, simulation_exception_types
from dodorl_sdk.dodorl.common.info_bridge import publish_training_info
from dodorl_sdk.dodorl.common.process_shutdown import shutdown
from dodorl_sdk.dodorl.utils.toml_config import TomlTrainingConfig
from dodorl_sdk.dodorl.rpc.factory import RpcServiceHost
from dodorl_sdk.dodorl.config import GrpcConfig
from dodorl_sdk.dodorl.scheduler.rollout_sample import RolloutSampleWorker
assert create_sim_manager(None) is None
assert create_sim_manager(SimpleNamespace(use_sim=False)) is None
assert simulation_exception_types() == ()
assert RpcServiceHost(GrpcConfig(), 'disabled', {'sim_manager': None}).services == {}
config_file = Path('no-sim.toml')
config_file.write_text('[env_config]\nuse_sim = false\n[sim_config]\nsim_type = \"afsim\"\n[sim_manager_config]\nuse_sim = false\n[sim_manager_config.instances]\ntype = \"missing\"\n')
config = TomlTrainingConfig(config_file)
assert config.sim_config is None and config.sim_manager_config is None
# 只读取训练配置时，即使声明了尚未使用的仿真，也不导入可选 SDK。
pending = TomlTrainingConfig(config_text='[sim_config]\nconfig_source = "manager"')
assert pending.env_config.use_sim is True
assert pending.train_config.max_episode > 0
publish_training_info(config)
shutdown()
assert not any(name == 'dodo_sim' or name.startswith('dodo_sim.') for name in sys.modules)
assert RejectOptionalAndSourceImports.attempts == []
try:
    pending.sim_config
except ModuleNotFoundError as exc:
    assert exc.name == 'dodo_sim' and 'dodo-sim' in str(exc)
else:
    raise AssertionError('Accessing enabled simulation config must report the missing dependency')
try:
    create_sim_manager(SimpleNamespace(use_sim=True))
except ModuleNotFoundError as exc:
    assert exc.name == 'dodo_sim' and '场景工程' in str(exc)
else:
    raise AssertionError('SIM use must report the missing optional dependency')
print('installed SDK works without SIM')
'''
    result = subprocess.run([sys.executable, '-I', '-c', probe, str(target)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'installed SDK works without SIM' in result.stdout


@pytest.mark.parametrize('source', ['file', 'text'])
def test_installed_sdk_trains_local_game_without_sim(installed_sdk, source):
    """阻断可选 SDK 的全部导入，实际完成本地计数游戏的 PPO 采样、学习和收尾。"""
    import toml

    target, tmp_path = installed_sdk
    work = tmp_path / f'local-game-{source}'
    work.mkdir()
    (work / 'mock_game.py').write_text('''
from dodorl_sdk.dodorl.envs.sense.paralle_examples import CounterParallelEnvironment

steps = 0
closed = False

class LocalGame(CounterParallelEnvironment):
    def __init__(self, config, **kwargs):
        super().__init__(config.agent_names, max_steps=config.max_step_per_episode)

    def step(self, actions):
        global steps
        steps += 1
        return super().step(actions)

    def close(self):
        global closed
        closed = True
        super().close()
''')
    values = {
        'design': {'task_id': f'no-sim-{source}'},
        'train_config': {
            'max_episode': 4, 'seed': 123, 'record_sample': False,
            'mid_eval_enabled': False, 'end_eval_episodes': 0,
            'model_save_dir': str(work / 'models'), 'model_load_dir': str(work / 'models'),
            'result_save_dir': str(work / 'results'), 'artifacts_archive_enabled': False,
        },
        'env_config': {'sense_name': 'mock_game', 'agent_names': ['agent_0'],
                       'state_dim': 1, 'max_step_per_episode': 4, 'use_sim': False},
        'agents': {'0': {
            'name': 'counter', 'agent_type': 'sb3', 'agent_names': ['agent_0'],
            'obs_dim': 1, 'state_shape': 1, 'hidden_dim': 16, 'obs_last_action': False,
            'actions': {'0': {'action_key': 'counter', 'add_target': False,
                              'parameters': {'0': {'key': 'delta', 'values': [-1, 1]}}}},
        }},
        'learners': {'0': {
            'name': 'counter', 'device': 'cpu', 'algo_type': 'ppo', 'state_shape': 1,
            'algo_config': {'capacity': 32, 'batch_size': 8, 'n_steps': 8, 'n_epochs': 1,
                            'minibatch_size': 4, 'learn_times_per_episode': 1,
                            'sample_strategy': 'on_policy_recent_steps'},
        }},
        'plugins_config': {
            'environment': {'mock_game': 'mock_game:LocalGame'},
            # 未使用的仿真插件只登记路径；不要求安装它的依赖。
            'simulation': {'afsim': 'dodo_sim.simulations.afsim:AFSim'},
        },
        'log_config': {'log_dir': str(work / 'logs'), 'enabled_handlers': ['console']},
    }
    (work / 'scene.toml').write_text(toml.dumps(values))
    probe = r'''
import importlib.abc
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
attempts = []

class RejectSimulation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'dodo_sim':
            attempts.append(fullname)
            raise ModuleNotFoundError('SIM is deliberately unavailable', name='dodo_sim')

sys.meta_path.insert(0, RejectSimulation())
from dodorl_sdk import SerialTrainSDK, configure_sdk
from dodorl_sdk.dodorl.common.process_shutdown import shutdown
configure_sdk(project_root=Path.cwd())
path = Path('scene.toml')
kwargs = {'config_file': path} if sys.argv[2] == 'file' else {'config_text': path.read_text()}
sdk = SerialTrainSDK(plugin_dirs=[Path.cwd()], **kwargs)
assert sdk.config.sim_config is None and sdk.config.sim_manager_config is None
sdk.run()
import mock_game
assert mock_game.steps == 16, mock_game.steps
assert mock_game.closed
assert list(Path('models').glob('*.pth')), 'Training must save a final policy'
shutdown()
assert attempts == [], attempts
assert not any(name == 'dodo_sim' or name.startswith('dodo_sim.') for name in sys.modules)
print('Local game training completed without SIM')
'''
    result = subprocess.run(
        [sys.executable, '-I', '-c', probe, str(target), source], cwd=work,
        env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(work / 'mpl')),
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Local game training completed without SIM' in result.stdout


def test_installed_wheel_four_action_spaces(installed_sdk):
    target, tmp_path = installed_sdk
    probe = r'''
import json
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from dodorl_sdk import (
    ActionSpace, ActionTable, StepActionDefined, ActionParameterDefined,
    ContinuousParameter, ContinuousActionSpace, DiscreteActionSpace,
    MultiContinuousActionSpace, MultiDiscreteActionSpace, action_from_dict,
)
discrete = DiscreteActionSpace(ActionTable([
    StepActionDefined("move", [ActionParameterDefined("d", values=[0, 1])], False),
]))
point = ContinuousActionSpace("point", [ContinuousParameter("x", -10, 10)])
spaces = [discrete, point, MultiDiscreteActionSpace([("a", discrete), ("b", discrete)]),
          MultiContinuousActionSpace([("a", point), ("b", point)])]
for space in spaces:
    action = space.sample(np.random.default_rng(4))
    record = json.loads(json.dumps(space.record(action)))
    assert action_from_dict(record, space).to_dict() == action.to_dict()
    assert space.to_gym_space().contains(space.encode(action))
'''
    result = subprocess.run([sys.executable, '-I', '-c', probe, str(target)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr

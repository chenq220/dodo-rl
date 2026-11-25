"""进程环境的副作用边界及导入前线程配置验证。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from utils.environment import (bootstrap_environment, read_globe_env_vars,
    apply_globe_env_vars, bootstrap_environment_from_argv)
from utils.toml_config import TomlTrainingConfig


def test_explicit_override_literals_and_no_unrelated_change(monkeypatch):
    monkeypatch.setenv('DODORL_ENV_TEST', 'old')
    monkeypatch.setenv('DODORL_KEEP_TEST', 'keep')
    text='[globe_env_vars]\nDODORL_ENV_TEST="${literal}"\n'
    values=read_globe_env_vars(config_text=text)
    assert os.environ['DODORL_ENV_TEST']=='old'
    config=TomlTrainingConfig(config_text=text)
    assert config.globe_env_vars==values
    assert os.environ['DODORL_ENV_TEST']=='old'
    bootstrap_environment(config_text=text)
    assert os.environ['DODORL_ENV_TEST']=='${literal}'
    assert os.environ['DODORL_KEEP_TEST']=='keep'
    TomlTrainingConfig.prepare_runtime_bootstrap(config_text=text)
    assert os.environ['DODORL_ENV_TEST']=='${literal}'


@pytest.mark.parametrize('bad', [{'': 'x'}, {'BAD=KEY':'x'}, {'BAD\x00KEY':'x'},
    {'X':3}, {'X':True}, {'X':[]}, {'X':{}}, {'X':'bad\x00value'}, []])
def test_invalid_table_rejects_entire_update(monkeypatch,bad):
    monkeypatch.setenv('DODORL_ENV_TEST','old')
    values={'DODORL_ENV_TEST':'new',**bad} if isinstance(bad,dict) else bad
    with pytest.raises(ValueError):apply_globe_env_vars(values)
    assert os.environ['DODORL_ENV_TEST']=='old'


def test_argv_check_does_not_apply_and_empty_table_does_nothing(tmp_path,monkeypatch):
    monkeypatch.setenv('DODORL_ENV_TEST','old')
    p=tmp_path/'train.toml';p.write_text('[globe_env_vars]\nDODORL_ENV_TEST="new"\n')
    bootstrap_environment_from_argv(tmp_path,['-f',str(p),'--check'])
    assert os.environ['DODORL_ENV_TEST']=='old'
    bootstrap_environment(config_text='[design]\nx=1\n')
    assert os.environ['DODORL_ENV_TEST']=='old'
    bootstrap_environment_from_argv(tmp_path,['--config-file='+str(p)])
    assert os.environ['DODORL_ENV_TEST']=='new'


def test_fresh_sdk_import_bootstrap_and_child_inheritance(tmp_path):
    root=Path(__file__).resolve().parents[1]
    p=tmp_path/'train.toml';p.write_text('[globe_env_vars]\nOMP_NUM_THREADS="2"\nMKL_NUM_THREADS="2"\nDODORL_CHILD="yes"\n')
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'sdk/src'),str(root/'src')]))
    script='''import sys,os,json,subprocess
from dodorl_sdk import bootstrap_environment
assert 'torch' not in sys.modules
bootstrap_environment(sys.argv[1])
import torch
child=subprocess.check_output([sys.executable,'-c','import os;print(os.environ["DODORL_CHILD"])'],text=True).strip()
print(json.dumps(dict(threads=torch.get_num_threads(),child=child)))
'''
    result=subprocess.run([sys.executable,'-c',script,str(p)],cwd=tmp_path,env=env,text=True,capture_output=True,check=True)
    assert json.loads(result.stdout)==dict(threads=2,child='yes')

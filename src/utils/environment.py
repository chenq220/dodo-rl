"""仅用标准库读取/应用进程环境；可在导入计算库前调用。"""
from __future__ import annotations

import os
from pathlib import Path
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def validate_globe_env_vars(value):
    if not isinstance(value, dict):
        raise ValueError('[globe_env_vars] must be a TOML table')
    result = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or '=' in key or '\x00' in key:
            raise ValueError('globe_env_vars contains an invalid environment variable name')
        if not isinstance(item, str) or '\x00' in item:
            raise ValueError(f'globe_env_vars.{key} must be a string without NUL characters')
        os.fsencode(key)
        os.fsencode(item)
        result[key] = item
    return result


def read_globe_env_vars(config_file=None, *, config_text=None):
    """读取且校验字面字符串，不执行替换或修改环境。"""
    if (config_file is None) == (config_text is None):
        raise ValueError('Provide exactly one of config_file or config_text')
    text = Path(config_file).expanduser().read_text(encoding='utf-8') if config_file is not None else config_text
    if not isinstance(text, str):
        raise TypeError('config_text must be TOML text')
    raw = tomllib.loads(text)
    return validate_globe_env_vars(raw.get('globe_env_vars', {}))


def apply_globe_env_vars(value):
    """整体校验后覆盖当前进程环境；相同配置重复调用不会重复写入。"""
    values = validate_globe_env_vars(value)
    os.environ.update({k: v for k, v in values.items() if os.environ.get(k) != v})
    return values


def bootstrap_environment(config_file=None, *, config_text=None, apply=True):
    """在导入 PyTorch/OpenMP/CUDA 前调用；apply=False 只读取校验。"""
    values = read_globe_env_vars(config_file, config_text=config_text)
    if apply:
        apply_globe_env_vars(values)
    return values


def bootstrap_environment_from_argv(project_root, argv=None):
    """供源码 portal 的早期引导使用；不用于猜测第三方应用的参数。"""
    argv = list(sys.argv[1:] if argv is None else argv)
    filename = None
    for index, arg in enumerate(argv):
        if arg in ('-f', '--config-file') and index + 1 < len(argv):
            filename = argv[index + 1]
        elif arg.startswith('--config-file='):
            filename = arg.split('=', 1)[1]
    if filename is None:
        return {}
    path = Path(filename).expanduser()
    if not path.is_absolute():
        roots = [Path.cwd(), Path(project_root), Path(project_root) / 'subjects']
        path = next((root / path for root in roots if (root / path).is_file()), path)
    if not path.is_file():
        return {}  # 正常参数解析阶段给出原有路径错误。
    try:
        return bootstrap_environment(path, apply='--check' not in argv)
    except tomllib.TOMLDecodeError:
        return {}  # 完整配置解析器负责保留原有 TOML 错误诊断。

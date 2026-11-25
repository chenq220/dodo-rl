"""轻量进程环境预加载入口；导入本模块不会加载 PyTorch。"""
from utils.environment import (read_globe_env_vars, apply_globe_env_vars, bootstrap_environment)

__all__ = ["read_globe_env_vars", "apply_globe_env_vars", "bootstrap_environment"]

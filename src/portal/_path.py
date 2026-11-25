"""Portal 入口运行时路径引导（仅 stdlib，避免 shadow sample/train 等顶层包）。"""
from __future__ import annotations

import sys
from pathlib import Path

PORTAL_DIR = Path(__file__).resolve().parent
SRC_DIR = PORTAL_DIR.parent
PROJECT_ROOT = SRC_DIR.parent


def bootstrap_sys_path() -> Path:
    """将 ``src`` 置于 sys.path 首位，并移除 portal/ 直接脚本路径。"""
    portal_dir = str(PORTAL_DIR)
    src_dir = str(SRC_DIR)
    while portal_dir in sys.path:
        sys.path.remove(portal_dir)
    if src_dir in sys.path:
        sys.path.remove(src_dir)
    sys.path.insert(0, src_dir)
    from utils.environment import bootstrap_environment_from_argv
    bootstrap_environment_from_argv(PROJECT_ROOT)
    return PROJECT_ROOT


def setup_path() -> Path:
    """bootstrap_sys_path 的别名，供 `from portal import setup_path` 使用。"""
    return bootstrap_sys_path()


__all__ = [
    "PORTAL_DIR",
    "SRC_DIR",
    "PROJECT_ROOT",
    "bootstrap_sys_path",
    "setup_path",
]

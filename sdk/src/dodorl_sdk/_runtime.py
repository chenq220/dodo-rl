"""安装版 SDK 的内部运行时定位。"""
from __future__ import annotations

from pathlib import Path


def bootstrap_runtime() -> Path:
    """返回 wheel 内部运行时目录，不向 ``sys.path`` 注入顶层包。"""
    return Path(__file__).resolve().parent / "dodorl"


__all__ = ["bootstrap_runtime"]

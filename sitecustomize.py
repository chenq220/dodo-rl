"""从仓库根目录启动 Python 时，自动发现 ``src`` 下的源码。"""
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
_src_dir = str(SRC_DIR)

if SRC_DIR.is_dir() and _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

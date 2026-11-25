#!/usr/bin/env python3
"""Resolve the task ID used by distributed launcher scripts."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.task_id import task_id  # noqa: E402


def resolve_design_task_id(config_file: str | Path) -> str:
    """Apply ``design.task_id`` to the process singleton and return its value."""
    path = Path(config_file).expanduser().resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)

    design = config.get("design", {})
    if not isinstance(design, dict):
        raise TypeError("[design] must be a TOML table")
    task_id.configure(design.get("task_id"))
    return str(task_id)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: resolve_design_task_id.py <config-file>")
    print(resolve_design_task_id(sys.argv[1]))


if __name__ == "__main__":
    main()

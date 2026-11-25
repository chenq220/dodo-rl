from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOLVER = PROJECT_ROOT / "scripts" / "resolve_design_task_id.py"
LAUNCHERS = [
    PROJECT_ROOT / "scripts" / "start_distributed.sh",
    PROJECT_ROOT / "scripts" / "start_grpc_3n_distributed.sh",
    PROJECT_ROOT / "scripts" / "start_torch_rpc_distributed.sh",
    PROJECT_ROOT / "scripts" / "start_torch_rpc_3n_distributed.sh",
    PROJECT_ROOT / "scripts" / "start_distributed_eval.sh",
    PROJECT_ROOT / "scripts" / "start_compose_distributed.sh",
]


def _resolve(
        tmp_path: Path,
        design: str,
        *,
        environment_task_id: str | None,
) -> subprocess.CompletedProcess[str]:
    config = tmp_path / "training.toml"
    config.write_text(design, encoding="utf-8")
    environment = os.environ.copy()
    if environment_task_id is None:
        environment.pop("DODO_TASK_ID", None)
    else:
        environment["DODO_TASK_ID"] = environment_task_id
    return subprocess.run(
        [sys.executable, str(RESOLVER), str(config)],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_resolver_prefers_nonempty_design_task_id(tmp_path: Path) -> None:
    result = _resolve(
        tmp_path,
        '[design]\ntask_id = "  toml-task  "\n',
        environment_task_id="environment-task",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "toml-task"


@pytest.mark.parametrize(
    "design",
    [
        "",
        "[design]\n",
        '[design]\ntask_id = ""\n',
        '[design]\ntask_id = " none "\n',
        '[design]\ntask_id = "NULL"\n',
    ],
)
def test_resolver_preserves_environment_for_empty_design_values(
        tmp_path: Path,
        design: str,
) -> None:
    result = _resolve(
        tmp_path,
        design,
        environment_task_id="environment-task",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "environment-task"


def test_resolver_generates_eight_hex_characters_without_other_source(
        tmp_path: Path,
) -> None:
    result = _resolve(tmp_path, "[design]\n", environment_task_id=None)

    assert result.returncode == 0, result.stderr
    assert re.fullmatch(r"[0-9a-f]{8}", result.stdout.strip())


def test_resolver_rejects_non_string_design_task_id(tmp_path: Path) -> None:
    result = _resolve(
        tmp_path,
        "[design]\ntask_id = 123\n",
        environment_task_id="environment-task",
    )

    assert result.returncode != 0
    assert "task_id must be a string" in result.stderr


@pytest.mark.parametrize("launcher", LAUNCHERS, ids=lambda path: path.name)
def test_distributed_launcher_resolves_task_id_before_default_log_directory(
        launcher: Path,
) -> None:
    source = launcher.read_text(encoding="utf-8")

    resolver_position = source.index('export DODO_TASK_ID="$(')
    log_directory_position = source.index(
        'LOG_DIR="${DODO_LOG_DIR:-$PROJECT_ROOT/logs/distributed/$DODO_TASK_ID}"'
    )

    assert "resolve_design_task_id.py" in source
    assert resolver_position < log_directory_position

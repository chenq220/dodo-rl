from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCH_LAUNCHER = PROJECT_ROOT / "scripts" / "start_torch_rpc_3n_distributed.sh"
GRPC_LAUNCHER = PROJECT_ROOT / "scripts" / "start_grpc_3n_distributed.sh"


def _fake_python(tmp_path: Path) -> Path:
    path = tmp_path / "fake-python"
    path.write_text(
        """#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == *"resolve_design_task_id.py" ]]; then
  printf '%s\n' "$DODO_TASK_ID"
  exit 0
fi
if [[ "${1:-}" == "-c" ]]; then
  if [[ "$FAKE_PROTOCOL" == "torch" ]]; then
    printf '5\\n0\\n'
  else
    printf '0\\n'
  fi
  exit 0
fi
module="${2:-}"
printf '%s,%s\\n' "${RANK:-none}" "$module" >>"$FAKE_RECORD_FILE"
if [[ "$module" == "src.portal.control" || "$module" == "src.portal.stage_control" ]]; then
  printf 'DODO_TRAINING_STAGE=evade\nDODO_TRAINING_EPISODE=1828/3001\n' >"$DODO_PROGRESS_FILE"
  if [[ "${FAKE_CONTROL_MODE:-}" == "finish_after_marker" ]]; then
    # 故意延后标记，验证 worker 等待标记，而非依赖进程调度顺序。
    sleep 1.5
    printf 'DODO_CENTER_FINISHED=1\n' >>"$DODO_PROGRESS_FILE"
    sleep 2
    exit 0
  fi
  if [[ "${FAKE_CONTROL_MODE:-}" == "wait" ]]; then
    exec sleep 30
  fi
  sleep 1.2
  exit 0
fi
if [[ "${FAKE_WORKER_MODE:-}" == "exit_after_marker" ]]; then
  for attempt in {1..300}; do
    if [[ -r "$DODO_PROGRESS_FILE" ]] && grep -qx 'DODO_CENTER_FINISHED=1' "$DODO_PROGRESS_FILE"; then
      exit 0
    fi
    sleep 0.01
  done
  echo "Timed out waiting for center finished marker" >&2
  exit 3
fi
if [[ "${FAKE_WORKER_MODE:-}" == "exit" ]]; then
  exit 0
fi
exec sleep 30
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run_launcher(
        tmp_path: Path,
        protocol: str,
        *,
        control_mode: str = "",
        worker_mode: str = "",
) -> tuple[subprocess.CompletedProcess, list[str]]:
    config = tmp_path / "training.toml"
    config.write_text("# parsed by fake python\n", encoding="utf-8")
    record = tmp_path / "roles.csv"
    environment = os.environ.copy()
    environment.update({
        "PYTHON_BIN": str(_fake_python(tmp_path)),
        "DODO_TASK_ID": f"launcher-3n-{protocol}-{os.getpid()}",
        "FAKE_PROTOCOL": protocol,
        "FAKE_RECORD_FILE": str(record),
        "FAKE_CONTROL_MODE": control_mode,
        "FAKE_WORKER_MODE": worker_mode,
        "DODO_LOG_DIR": str(tmp_path / "distributed-logs"),
    })
    command = [
        "bash",
        str(TORCH_LAUNCHER if protocol == "torch" else GRPC_LAUNCHER),
        str(config),
    ]
    if protocol == "grpc":
        command.append("2")
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=8,
    )
    return result, record.read_text(encoding="utf-8").splitlines()


def test_torch_3n_launcher_assigns_contiguous_rank_environment(tmp_path: Path) -> None:
    result, records = _run_launcher(tmp_path, "torch")

    assert result.returncode == 0, result.stdout
    assert "训练进度：阶段=evade，局数=1828/3001" in result.stdout
    assert sorted(records) == sorted([
        "0,src.portal.control",
        "1,src.portal.learn",
        "2,src.portal.model",
        "3,src.portal.sample",
        "4,src.portal.sample",
    ])


def test_grpc_3n_launcher_starts_three_roles_and_samples_without_rank(tmp_path: Path) -> None:
    result, records = _run_launcher(tmp_path, "grpc")

    assert result.returncode == 0, result.stdout
    assert "训练进度：阶段=evade，局数=1828/3001" in result.stdout
    assert sorted(records) == sorted([
        "none,src.portal.control",
        "none,src.portal.learn",
        "none,src.portal.model",
        "none,src.portal.sample",
        "none,src.portal.sample",
    ])


def test_grpc_3n_launcher_allows_workers_to_exit_first_when_center_finished_marker_exists(tmp_path: Path) -> None:
    result, records = _run_launcher(
        tmp_path,
        "grpc",
        control_mode="finish_after_marker",
        worker_mode="exit_after_marker",
    )

    assert result.returncode == 0, result.stdout
    assert "已感知中心收口" in result.stdout
    assert "控制进程已正常退出" in result.stdout
    assert "在控制进程存活时提前退出" not in result.stdout
    assert any(line.endswith("src.portal.sample") for line in records)


def test_torch_3n_launcher_allows_workers_to_exit_first_when_center_finished_marker_exists(tmp_path: Path) -> None:
    result, _records = _run_launcher(
        tmp_path,
        "torch",
        control_mode="finish_after_marker",
        worker_mode="exit_after_marker",
    )

    assert result.returncode == 0, result.stdout
    assert "已感知中心收口" in result.stdout
    assert "控制进程已正常退出" in result.stdout
    assert "在控制进程存活时提前退出" not in result.stdout


@pytest.mark.parametrize("protocol", ["grpc", "torch"])
def test_3n_launcher_rejects_workers_exiting_without_center_finished_marker(tmp_path, protocol):
    result, _records = _run_launcher(tmp_path, protocol, control_mode="wait", worker_mode="exit")

    assert result.returncode == 1, result.stdout
    assert "在控制进程存活时提前退出" in result.stdout
    assert "已感知中心收口" not in result.stdout

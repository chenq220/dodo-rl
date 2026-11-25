from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "start_torch_rpc_distributed.sh"


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
  printf '2\\n0\\n'
  exit 0
fi
printf '%s\\n' "${RANK:-missing}" >>"$FAKE_RANK_FILE"
if [[ "${RANK:-}" == "0" ]]; then
  printf '%s' "$$" >"$FAKE_CONTROL_PID_FILE"
  printf '%s,%s' "${TP_SOCKET_IFNAME:-}" "${GLOO_SOCKET_IFNAME:-}" >"$FAKE_INTERFACE_FILE"
  if [[ "$FAKE_CONTROL_MODE" == "finish" ]]; then
    sleep 0.2
    exit 0
  fi
  if [[ "$FAKE_CONTROL_MODE" == "finish_after_marker" ]]; then
    printf 'DODO_CENTER_FINISHED=1\n' >"$DODO_PROGRESS_FILE"
    sleep 2
    exit 0
  fi
  exec sleep 30
fi
if [[ "$FAKE_SAMPLE_MODE" == "exit_after_marker" ]]; then
  for attempt in {1..300}; do
    if [[ -r "$DODO_PROGRESS_FILE" ]] && grep -qx 'DODO_CENTER_FINISHED=1' "$DODO_PROGRESS_FILE"; then
      exit 0
    fi
    sleep 0.01
  done
  echo "Timed out waiting for center finished marker" >&2
  exit 3
fi
if [[ "$FAKE_SAMPLE_MODE" == "exit" ]]; then
  exit 0
fi
exec sleep 0.4
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _run_launcher(tmp_path: Path, *, control_mode: str, sample_mode: str):
    config = tmp_path / "training.toml"
    config.write_text("[torch_rpc_config]\nworld_size=2\ninit_address='127.0.0.1:5555'\n", encoding="utf-8")
    control_pid_file = tmp_path / "control.pid"
    interface_file = tmp_path / "interfaces"
    rank_file = tmp_path / "ranks"
    environment = os.environ.copy()
    environment.update({
        "PYTHON_BIN": str(_fake_python(tmp_path)),
        "DODO_TASK_ID": f"launcher-{control_mode}-{sample_mode}-{os.getpid()}",
        "FAKE_CONTROL_MODE": control_mode,
        "FAKE_SAMPLE_MODE": sample_mode,
        "FAKE_CONTROL_PID_FILE": str(control_pid_file),
        "FAKE_INTERFACE_FILE": str(interface_file),
        "FAKE_RANK_FILE": str(rank_file),
        "TP_SOCKET_IFNAME": "user-tp",
        "GLOO_SOCKET_IFNAME": "user-gloo",
        "DODO_LOG_DIR": str(tmp_path / "distributed-logs"),
    })
    started = time.monotonic()
    result = subprocess.run(
        ["bash", str(LAUNCHER), str(config)],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=8,
    )
    return result, time.monotonic() - started, control_pid_file, interface_file, rank_file


def test_launcher_fails_fast_and_cleans_rank_zero_when_sample_exits_early(tmp_path):
    result, elapsed, control_pid_file, _, rank_file = _run_launcher(
        tmp_path, control_mode="wait", sample_mode="exit",
    )

    assert result.returncode != 0
    assert elapsed < 5
    assert "sample 在 Rank 0 仍存活时提前退出" in result.stdout
    assert sorted(rank_file.read_text(encoding="utf-8").splitlines()) == ["0", "1"]
    control_pid = int(control_pid_file.read_text(encoding="utf-8"))
    try:
        os.kill(control_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError(f"Rank 0 process was not cleaned: pid={control_pid}")


def test_launcher_allows_samples_to_exit_after_rank_zero_finishes(tmp_path):
    result, elapsed, _, interface_file, rank_file = _run_launcher(
        tmp_path, control_mode="finish", sample_mode="wait",
    )

    assert result.returncode == 0, result.stdout
    assert elapsed < 5
    assert "已退出。" in result.stdout
    assert "已正常退出" not in result.stdout
    assert "全部 sample 已感知中心关闭并自然退出" not in result.stdout
    assert interface_file.read_text(encoding="utf-8") == "user-tp,user-gloo"
    assert sorted(rank_file.read_text(encoding="utf-8").splitlines()) == ["0", "1"]


def test_launcher_allows_sample_to_exit_first_when_center_finished_marker_exists(tmp_path):
    result, elapsed, _, _, rank_file = _run_launcher(
        tmp_path, control_mode="finish_after_marker", sample_mode="exit_after_marker",
    )

    assert result.returncode == 0, result.stdout
    assert elapsed < 5
    assert "已感知中心收口" in result.stdout
    assert "已退出。" in result.stdout
    assert "已正常退出" not in result.stdout
    assert "全部 sample 已感知中心关闭并自然退出" not in result.stdout
    assert "sample 在 Rank 0 仍存活时提前退出" not in result.stdout
    assert sorted(rank_file.read_text(encoding="utf-8").splitlines()) == ["0", "1"]

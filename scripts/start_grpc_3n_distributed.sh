#!/usr/bin/env bash
# 启动 gRPC 3+N：Control + Learn + Model + N Sample。
set -Eeuo pipefail

if ! ulimit -f unlimited 2>/dev/null; then
  echo "无法解除单文件大小限制，请先提高当前终端的文件大小硬限制。" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
用法：
  ./scripts/start_grpc_3n_distributed.sh <科目配置文件> <采样进程数>
EOF
}

if [[ $# -ne 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=progress_env.sh
source "$SCRIPT_DIR/progress_env.sh"
CONFIG_FILE="$1"
SAMPLE_COUNT="$2"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

if [[ "$CONFIG_FILE" != /* ]]; then CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"; fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "配置文件不存在：$CONFIG_FILE" >&2
  exit 2
fi
if [[ ! "$SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "采样进程数必须是正整数：$SAMPLE_COUNT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$(command -v python)"; fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DODO_RPC="${DODO_RPC:-grpc}"
export DODO_TASK_ID="$(
  "$PYTHON_BIN" "$SCRIPT_DIR/resolve_design_task_id.py" "$CONFIG_FILE"
)"

CONFIG_INFO="$($PYTHON_BIN -c '
import pathlib, sys, tomllib
path = pathlib.Path(sys.argv[1])
with path.open("rb") as stream:
    data = tomllib.load(stream)
if not isinstance(data.get("grpc_config"), dict):
    raise SystemExit("配置文件缺少 [grpc_config]")
stages = data.get("stages", {})
print(len(stages) if isinstance(stages, dict) else 0)
' "$CONFIG_FILE")"
STAGE_COUNT="$(printf '%s\n' "$CONFIG_INFO" | sed -n '1p')"

LOG_DIR="${DODO_LOG_DIR:-$PROJECT_ROOT/logs/distributed/$DODO_TASK_ID}"
mkdir -p "$LOG_DIR"
cp "$CONFIG_FILE" "$LOG_DIR/$(basename "$CONFIG_FILE")"
PROGRESS_FILE="$LOG_DIR/training_progress.env"
export DODO_PROGRESS_FILE="$PROGRESS_FILE"
export DODO_TRAINING_STAGE=""
export DODO_TRAINING_EPISODE=""
PROGRESS_LINE_ACTIVE=0

print_task_elapsed() {
  local elapsed=$((SECONDS - TASK_START_SECONDS))
  local hours=$((elapsed / 3600))
  local minutes=$(((elapsed % 3600) / 60))
  local secs=$((elapsed % 60))
  printf '本次任务执行耗时：%d:%02d:%02d\n' "$hours" "$minutes" "$secs"
}

wait_for_control_after_finished_marker() {
  local deadline=$((SECONDS + 40))
  local control_status
  while kill -0 "$CONTROL_PID" 2>/dev/null && [[ "$SECONDS" -lt "$deadline" ]]; do
    sleep 1
  done
  if kill -0 "$CONTROL_PID" 2>/dev/null; then
    echo "控制进程未在中心收口后 40 秒内退出；将清理剩余进程。" >&2
    exit 1
  fi
  set +e
  wait "$CONTROL_PID"
  control_status="$?"
  set -e
  if [[ "$control_status" -eq 0 ]]; then
    echo "控制进程已正常退出，统一回收其余 gRPC 角色。"
  else
    echo "控制进程异常退出：status=${control_status}" >&2
  fi
  exit "$control_status"
}

if [[ "$STAGE_COUNT" -gt 0 ]]; then
  CONTROL_MODULE="src.portal.stage_control"
  CONTROL_NAME="stage_control"
else
  CONTROL_MODULE="src.portal.control"
  CONTROL_NAME="control"
fi

PIDS=()
ROLE_NAMES=()
STOPPING=0
TASK_START_SECONDS=$SECONDS
stop_children() {
  if [[ "$STOPPING" -eq 1 ]]; then return; fi
  STOPPING=1
  trap - INT TERM EXIT
  finish_training_progress_line
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "停止 gRPC 3+N 任务：DODO_TASK_ID=${DODO_TASK_ID}，进程数=${#PIDS[@]}"
    kill -TERM "${PIDS[@]}" 2>/dev/null || true
    sleep 1
    kill -KILL "${PIDS[@]}" 2>/dev/null || true
  fi
  print_task_elapsed
}
trap stop_children INT TERM EXIT

start_role() {
  local role_name="$1" module="$2"
  shift 2
  nohup "$PYTHON_BIN" -m "$module" "$@" >"$LOG_DIR/$role_name.log" 2>&1 &
  PIDS+=("$!")
  ROLE_NAMES+=("$role_name")
  echo "已启动：${role_name}"
}

cd "$PROJECT_ROOT"
echo "启动 gRPC 3+N：Control + Learn + Model + ${SAMPLE_COUNT} Sample"
echo "配置文件：$CONFIG_FILE"
echo "日志目录：$LOG_DIR"

start_role "$CONTROL_NAME" "$CONTROL_MODULE" -f "$CONFIG_FILE"
CONTROL_PID="${PIDS[0]}"
start_role "model-0" "src.portal.model" -f "$CONFIG_FILE"
start_role "learn-0" "src.portal.learn" --role-id learn-0 -f "$CONFIG_FILE"
for ((index = 0; index < SAMPLE_COUNT; index++)); do
  start_role "sample-$index" "src.portal.sample" -f "$CONFIG_FILE" "sample-$index"
done

while true; do
  print_training_progress_updates
  if ! kill -0 "$CONTROL_PID" 2>/dev/null; then
    finish_training_progress_line
    set +e
    wait "$CONTROL_PID"
    control_status="$?"
    set -e
    if [[ "$control_status" -eq 0 ]]; then
      echo "控制进程已正常退出，统一回收其余 gRPC 角色。"
    else
      echo "控制进程异常退出：status=${control_status}" >&2
    fi
    exit "$control_status"
  fi
  for ((index = 1; index < ${#PIDS[@]}; index++)); do
    if kill -0 "${PIDS[$index]}" 2>/dev/null; then continue; fi
    set +e
    wait "${PIDS[$index]}"
    role_status="$?"
    set -e
    finish_training_progress_line
    if is_center_finished_env; then
      echo "${ROLE_NAMES[$index]} 已感知中心收口：status=${role_status}；等待控制进程退出。"
      wait_for_control_after_finished_marker
    fi
    echo "${ROLE_NAMES[$index]} 在控制进程存活时提前退出：status=${role_status}" >&2
    exit 1
  done
  sleep 1
done

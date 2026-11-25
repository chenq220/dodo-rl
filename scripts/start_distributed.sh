#!/usr/bin/env bash
# 启动一个中心学习控制进程和 N 个分布式采样进程。
set -Eeuo pipefail

# 解除启动终端可能继承的单文件大小软限制，避免长日志和GIF在4 MiB处被截断。
if ! ulimit -f unlimited 2>/dev/null; then
  echo "无法解除单文件大小限制，请先提高当前终端的文件大小硬限制。" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
用法：
  ./scripts/start_distributed.sh <科目配置文件> <采样进程数>

示例：
  ./scripts/start_distributed.sh \
    subjects/cooperative_search/cooperative_search_qmix.toml 3
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

if [[ "$CONFIG_FILE" != /* ]]; then
  CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "配置文件不存在：$CONFIG_FILE" >&2
  exit 2
fi
if [[ ! "$SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "采样进程数必须是正整数：$SAMPLE_COUNT" >&2
  exit 2
fi

export DODO_RPC="${DODO_RPC:-grpc}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# 本地开发优先沿用项目虚拟环境；容器基础镜像则使用其已安装依赖的 Python。
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
export DODO_TASK_ID="$(
  "$PYTHON_BIN" "$SCRIPT_DIR/resolve_design_task_id.py" "$CONFIG_FILE"
)"

LOG_DIR="${DODO_LOG_DIR:-$PROJECT_ROOT/logs/distributed/$DODO_TASK_ID}"
mkdir -p "$LOG_DIR"
cp "$CONFIG_FILE" "$LOG_DIR/$(basename "$CONFIG_FILE")"
# 子进程不能回写父 shell 的环境变量；控制进程把同名变量写入进度快照。
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

# 多阶段配置必须使用StageLearnRole，否则普通LearnControlRole只会按
# [train_config] 运行一次，不会在第一阶段结束后切换课程。
STAGE_COUNT="$(awk '/^\[stages\.[0-9]+\][[:space:]]*$/ { count += 1 } END { print count + 0 }' "$CONFIG_FILE")"
if [[ "$STAGE_COUNT" -gt 1 ]]; then
  CONTROL_MODULE="src.portal.stage_learn"
  CONTROL_NAME="stage_learn"
else
  CONTROL_MODULE="src.portal.learn_control"
  CONTROL_NAME="learn_control"
fi

PIDS=()
SAMPLE_PIDS=()
STOPPING=0
TASK_START_SECONDS=$SECONDS

stop_children() {
  if [[ "$STOPPING" -eq 1 ]]; then
    return
  fi
  STOPPING=1
  trap - INT TERM EXIT
  finish_training_progress_line
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "停止分布式任务：DODO_TASK_ID=${DODO_TASK_ID}，进程数=${#PIDS[@]}"
    kill -TERM "${PIDS[@]}" 2>/dev/null || true
    sleep 1
    kill -KILL "${PIDS[@]}" 2>/dev/null || true
  fi
  print_task_elapsed
}

trap stop_children INT TERM EXIT

cd "$PROJECT_ROOT"
echo "启动分布式任务：DODO_TASK_ID=${DODO_TASK_ID}"
echo "配置文件：$CONFIG_FILE"
echo "采样进程数：$SAMPLE_COUNT"
echo "训练阶段数：$STAGE_COUNT"
echo "日志目录：$LOG_DIR"

nohup "$PYTHON_BIN" -m "$CONTROL_MODULE" \
  -f "$CONFIG_FILE" >"$LOG_DIR/learn_control.log" 2>&1 &
CONTROL_PID="$!"
PIDS+=("$CONTROL_PID")

for ((index = 0; index < SAMPLE_COUNT; index++)); do
  role_id="sample-$index"
  nohup "$PYTHON_BIN" -m src.portal.sample \
    -f "$CONFIG_FILE" "$role_id" >"$LOG_DIR/$role_id.log" 2>&1 &
  sample_pid="$!"
  PIDS+=("$sample_pid")
  SAMPLE_PIDS+=("$sample_pid")
done

echo "已启动：$CONTROL_NAME + $SAMPLE_COUNT 个 sample。按 Ctrl+C 统一停止。"

# macOS 默认 Bash 不保证 wait -n。中心异常退出时立即清采样，避免误连另一场训练；
# 中心正常退出后先等 sample 感知服务关闭，超时再由 EXIT trap 回收。
while true; do
  print_training_progress_updates
  if ! kill -0 "$CONTROL_PID" 2>/dev/null; then
    finish_training_progress_line
    set +e
    wait "$CONTROL_PID"
    control_status="$?"
    set -e
    if [[ "$control_status" -ne 0 ]]; then
      echo "学习控制进程异常退出：PID=${CONTROL_PID}，status=${control_status}，停止同组采样进程。" >&2
      exit "$control_status"
    fi
    echo "学习控制进程 ${CONTROL_PID} 已退出。"
    deadline=$((SECONDS + 40))
    while [[ "$SECONDS" -lt "$deadline" ]]; do
      alive=0
      for sample_pid in "${SAMPLE_PIDS[@]}"; do
        if kill -0 "$sample_pid" 2>/dev/null; then
          alive=1
          break
        fi
      done
      if [[ "$alive" -eq 0 ]]; then
        PIDS=()
        break
      fi
      sleep 1
    done
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
      echo "sample 未在中心正常结束后 40 秒内退出；将清理剩余进程。" >&2
      exit 1
    fi
    exit 0
  fi
  sleep 1
done

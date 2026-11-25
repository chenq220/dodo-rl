#!/usr/bin/env bash
# Compose 容器内角色入口：占槽、注入 RANK、归档日志，并把进度打到 compose console。
set -Eeuo pipefail

if ! ulimit -f unlimited 2>/dev/null; then
  echo "无法解除单文件大小限制，请先提高运行环境的文件大小硬限制。" >&2
  exit 2
fi

PROJECT_ROOT="${DODORL_PROJECT_ROOT:-/code}"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

SERVICE="${DODO_COMPOSE_SERVICE:-${DODO_COMPOSE_ROLE:-}}"
MODE="${DODO_MODE:-train}"
CONFIG_FILE="${DODO_CONFIG_FILE:-}"
LOG_DIR="${DODO_LOG_DIR:-}"
TASK_ID="${DODO_TASK_ID:-}"

if [[ -z "$SERVICE" || -z "$CONFIG_FILE" || -z "$LOG_DIR" || -z "$TASK_ID" ]]; then
  echo "compose_role.sh 需要 DODO_COMPOSE_SERVICE（或 DODO_COMPOSE_ROLE）、DODO_CONFIG_FILE、DODO_LOG_DIR、DODO_TASK_ID。" >&2
  exit 2
fi
if [[ "$CONFIG_FILE" != /* ]]; then
  CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
fi
if [[ "$LOG_DIR" != /* ]]; then
  LOG_DIR="$PROJECT_ROOT/$LOG_DIR"
fi
mkdir -p "$LOG_DIR"

PROGRESS_FILE="${DODO_PROGRESS_FILE:-$LOG_DIR/training_progress.env}"
if [[ "$PROGRESS_FILE" != /* ]]; then
  PROGRESS_FILE="$PROJECT_ROOT/$PROGRESS_FILE"
fi
export DODO_PROGRESS_FILE="$PROGRESS_FILE"

read_progress_kv() {
  local file="$1" key="$2"
  [[ -r "$file" ]] || return 0
  awk -F= -v k="$key" '$1==k {sub(/^[^=]+=/, ""); print; exit}' "$file"
}

print_role_banner() {
  echo "启动 compose 角色：${CONSOLE_LABEL}"
  echo "DODO_TASK_ID=${TASK_ID}"
  echo "配置文件：${CONFIG_FILE}"
  echo "协议：${DODO_RPC:-}  模式：${MODE}"
  echo "日志目录：${LOG_DIR}"
  if [[ "$SERVICE" == "control" ]]; then
    echo "中心模块：${MODULE}"
  fi
  if [[ -n "${RANK:-}" ]]; then
    echo "RANK=${RANK}"
  fi
}

zh_progress_token() {
  # 进度行 key 用中文；学习= 后面再保留 episode= 会变成「学习=episode=32」。
  local token="$1"
  local strip_episode="${2:-0}"
  if [[ "$strip_episode" == "1" ]]; then
    token="${token#episode=}"
  fi
  token="${token//episode=/局=}"
  token="${token//stage=/阶段=}"
  token="${token//learners=/学习器=}"
  token="${token//generation=/代次=}"
  printf '%s' "$token"
}

print_console_progress() {
  # Python 子进程改不了父 shell 环境变量，从 DODO_PROGRESS_FILE 读回同名键。
  # 进度还没写入时不要打「阶段=?」，那不是合法状态。
  local stage episode line sample_token learn_token
  stage="$(read_progress_kv "$PROGRESS_FILE" DODO_TRAINING_STAGE)"
  episode="$(read_progress_kv "$PROGRESS_FILE" DODO_TRAINING_EPISODE)"
  sample_token="$(read_progress_kv "$DODO_PROGRESS_FILE" DODO_SAMPLE_TOKEN)"
  learn_token="$(read_progress_kv "$DODO_PROGRESS_FILE" DODO_LEARN_TOKEN)"
  if [[ "$SERVICE" == "control" ]]; then
    [[ -n "$stage" && -n "$episode" ]] || return 0
    line="[${CONSOLE_LABEL}] 阶段=${stage} 采样=${episode}"
    if [[ -n "$learn_token" ]]; then
      line+=" 学习=$(zh_progress_token "$learn_token" 1)"
    fi
  else
    [[ -n "$sample_token" ]] || return 0
    line="[${CONSOLE_LABEL}] 正在采样 $(zh_progress_token "$sample_token" 0)"
  fi
  if [[ "$line" != "${LAST_CONSOLE_LINE:-}" ]]; then
    printf '%s\n' "$line"
    LAST_CONSOLE_LINE="$line"
  fi
}

run_with_console_progress() {
  local log_file="$1"
  shift
  python "$@" >>"$log_file" 2>&1 &
  PY_PID=$!
  trap 'kill "$PY_PID" 2>/dev/null || true; wait "$PY_PID" || true' TERM INT
  while kill -0 "$PY_PID" 2>/dev/null; do
    print_console_progress
    sleep 1
  done
  print_console_progress
  wait "$PY_PID"
}

if [[ "$SERVICE" == "control" ]]; then
  MODULE="${DODO_CONTROL_MODULE:-src.portal.stage_learn}"
  LOG_NAME="${DODO_CONTROL_LOG:-stage_learn.log}"
  CONSOLE_LABEL="control"
  if [[ "${DODO_RPC:-}" == "trpc" ]]; then
    export RANK=0
  fi
  print_role_banner
  run_with_console_progress "$LOG_DIR/$LOG_NAME" -m "$MODULE" -f "$CONFIG_FILE"
  exit $?
fi

if [[ "$SERVICE" != "sample" ]]; then
  echo "不支持的 DODO_COMPOSE_SERVICE：$SERVICE" >&2
  exit 2
fi

SAMPLE_COUNT="${DODO_SAMPLE_COUNT:-}"
if [[ ! "$SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "DODO_SAMPLE_COUNT 必须是正整数：$SAMPLE_COUNT" >&2
  exit 2
fi

SLOT_ROOT="$LOG_DIR/slots"
LOCK_ROOT="${DODO_SLOT_LOCK_DIR:-/var/dodorl/slot-lock}"
if ! mkdir -p "$LOCK_ROOT" 2>/dev/null; then
  LOCK_ROOT="$SLOT_ROOT"
fi
mkdir -p "$SLOT_ROOT" "$LOCK_ROOT"
INDEX=""
for ((i = 0; i < SAMPLE_COUNT; i++)); do
  if mkdir "$LOCK_ROOT/$i" 2>/dev/null; then
    INDEX="$i"
    break
  fi
done
if [[ -z "$INDEX" ]]; then
  echo "无法为 sample 占槽：已有 $SAMPLE_COUNT 个副本。" >&2
  exit 2
fi
mkdir -p "$SLOT_ROOT/$INDEX"

if [[ "${DODO_RPC:-}" == "trpc" ]]; then
  export RANK=$((INDEX + 1))
fi

if [[ "$MODE" == "eval" ]]; then
  SAMPLE_MODULE="src.portal.eval_sample"
else
  SAMPLE_MODULE="src.portal.sample"
fi

CONSOLE_LABEL="sample-$INDEX"
export DODO_PROGRESS_FILE="$SLOT_ROOT/$INDEX/progress.env"
print_role_banner
run_with_console_progress "$LOG_DIR/sample-$INDEX.log" -m "$SAMPLE_MODULE" -f "$CONFIG_FILE" "sample-$INDEX"
exit $?

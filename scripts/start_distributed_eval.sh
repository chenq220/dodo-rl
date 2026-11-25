#!/usr/bin/env bash
# 启动一个分布式评估中心进程和 N 个评估采样进程。
set -Eeuo pipefail

# 解除启动终端可能继承的单文件大小软限制，避免长日志和GIF在4 MiB处被截断。
if ! ulimit -f unlimited 2>/dev/null; then
  echo "无法解除单文件大小限制，请先提高当前终端的文件大小硬限制。" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
用法：
  ./scripts/start_distributed_eval.sh <科目配置文件>
  ./scripts/start_distributed_eval.sh <科目配置文件> <采样进程数>

Torch RPC 只接受配置文件，采样进程数由 branch=eval 解析后的 world_size - 1 决定。
gRPC 科目再传采样进程数，行为同 start_distributed.sh。

示例：
  ./scripts/start_distributed_eval.sh \
    subjects/cooperative_search/cooperative_search_qmix.toml
EOF
}

if [[ $# -lt 1 || $# -gt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=progress_env.sh
source "$SCRIPT_DIR/progress_env.sh"
CONFIG_FILE="$1"
REQUESTED_SAMPLE_COUNT="${2:-}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

if [[ "$CONFIG_FILE" != /* ]]; then
  CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "配置文件不存在：$CONFIG_FILE" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python)"
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export DODO_TASK_ID="$(
  "$PYTHON_BIN" "$SCRIPT_DIR/resolve_design_task_id.py" "$CONFIG_FILE"
)"

CONFIG_INFO="$($PYTHON_BIN -c '
import sys
from pathlib import Path
from config import GrpcConfig, TorchRpcConfig
from utils.toml_config import TomlTrainingConfig
config_file = Path(sys.argv[1])
project_root = Path(sys.argv[2])
conf = TomlTrainingConfig(config_file, project_root=project_root, branch="eval")
if isinstance(conf.rpc_config, TorchRpcConfig):
    print("torch")
    print(int(conf.rpc_config.world_size))
elif isinstance(conf.rpc_config, GrpcConfig):
    print("grpc")
    print(0)
else:
    raise SystemExit(f"unsupported rpc config: {type(conf.rpc_config).__name__}")
' "$CONFIG_FILE" "$PROJECT_ROOT")"

RPC_PROTOCOL="$(printf '%s\n' "$CONFIG_INFO" | sed -n '1p')"
WORLD_SIZE="$(printf '%s\n' "$CONFIG_INFO" | sed -n '2p')"

if [[ "$RPC_PROTOCOL" == "torch" ]]; then
  if [[ -n "$REQUESTED_SAMPLE_COUNT" ]]; then
    echo "Torch RPC 评估脚本不接受采样进程数，world_size 由配置决定。" >&2
    exit 2
  fi
  SAMPLE_COUNT=$((WORLD_SIZE - 1))
  export DODO_RPC="${DODO_RPC:-trpc}"
elif [[ "$RPC_PROTOCOL" == "grpc" ]]; then
  if [[ -z "$REQUESTED_SAMPLE_COUNT" ]]; then
    echo "gRPC 评估需要采样进程数。" >&2
    exit 2
  fi
  if [[ ! "$REQUESTED_SAMPLE_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "采样进程数必须是正整数：$REQUESTED_SAMPLE_COUNT" >&2
    exit 2
  fi
  SAMPLE_COUNT="$REQUESTED_SAMPLE_COUNT"
  export DODO_RPC="${DODO_RPC:-grpc}"
else
  echo "无法识别 RPC 协议：$RPC_PROTOCOL" >&2
  exit 2
fi

LOG_DIR="${DODO_LOG_DIR:-$PROJECT_ROOT/logs/distributed/$DODO_TASK_ID}"
mkdir -p "$LOG_DIR"
cp "$CONFIG_FILE" "$LOG_DIR/$(basename "$CONFIG_FILE")"
# 子进程不能回写父 shell 的环境变量；控制进程把同名变量写入进度快照。
PROGRESS_FILE="$LOG_DIR/training_progress.env"
export DODO_PROGRESS_FILE="$PROGRESS_FILE"
export DODO_TRAINING_STAGE=""
export DODO_TRAINING_EPISODE=""
PROGRESS_LINE_ACTIVE=0

if [[ "$RPC_PROTOCOL" == "torch" ]]; then
  # 本脚本只启动同机 1+N world。固定回环接口可避免 Wi-Fi/VPN 路由变化影响长连接；
  # 用户显式指定任一变量时分别保留，不覆盖。
  case "$(uname -s)" in
    Darwin) LOOPBACK_IFNAME="lo0" ;;
    Linux) LOOPBACK_IFNAME="lo" ;;
    *) LOOPBACK_IFNAME="" ;;
  esac
  if [[ -n "$LOOPBACK_IFNAME" ]]; then
    export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-$LOOPBACK_IFNAME}"
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$LOOPBACK_IFNAME}"
  fi
fi

print_task_elapsed() {
  local elapsed=$((SECONDS - TASK_START_SECONDS))
  local hours=$((elapsed / 3600))
  local minutes=$(((elapsed % 3600) / 60))
  local secs=$((elapsed % 60))
  printf '本次任务执行耗时：%d:%02d:%02d\n' "$hours" "$minutes" "$secs"
}

wait_remaining_samples_after_center_exit() {
  local deadline=$((SECONDS + 40))
  local alive sample_pid
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
      return 0
    fi
    sleep 1
  done
  echo "sample 未在中心正常结束后 40 秒内退出；将清理剩余进程。" >&2
  return 1
}

finish_after_center_exit() {
  local control_status="$1"
  if [[ "$control_status" -ne 0 ]]; then
    echo "中心进程异常退出：PID=${CONTROL_PID}，status=${control_status}" >&2
    exit "$control_status"
  fi
  echo "中心进程 ${CONTROL_PID} 已退出。"
  if ! wait_remaining_samples_after_center_exit; then
    exit 1
  fi
  exit 0
}

wait_for_center_after_finished_marker() {
  local deadline=$((SECONDS + 40))
  while kill -0 "$CONTROL_PID" 2>/dev/null && [[ "$SECONDS" -lt "$deadline" ]]; do
    sleep 1
  done
  if kill -0 "$CONTROL_PID" 2>/dev/null; then
    echo "中心未在收口后 40 秒内退出；将清理剩余进程。" >&2
    exit 1
  fi
  local control_status
  set +e
  wait "$CONTROL_PID"
  control_status="$?"
  set -e
  finish_after_center_exit "$control_status"
}

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
  finish_eval_progress_line
  if [[ "${#PIDS[@]}" -gt 0 ]]; then
    echo "停止分布式评估任务：DODO_TASK_ID=${DODO_TASK_ID}，进程数=${#PIDS[@]}"
    kill -TERM "${PIDS[@]}" 2>/dev/null || true
    sleep 1
    kill -KILL "${PIDS[@]}" 2>/dev/null || true
  fi
  print_task_elapsed
}

trap stop_children INT TERM EXIT

cd "$PROJECT_ROOT"
echo "启动分布式评估任务：DODO_TASK_ID=${DODO_TASK_ID}"
echo "配置文件：$CONFIG_FILE"
echo "RPC 协议：${RPC_PROTOCOL}"
echo "采样进程数：${SAMPLE_COUNT}"
echo "日志目录：${LOG_DIR}"
if [[ "$RPC_PROTOCOL" == "torch" ]]; then
  echo "world_size：${WORLD_SIZE}（中心 1 + sample ${SAMPLE_COUNT}）"
  if [[ -n "${TP_SOCKET_IFNAME:-}" ]]; then
    echo "TensorPipe 网卡：${TP_SOCKET_IFNAME}"
  fi
  if [[ -n "${GLOO_SOCKET_IFNAME:-}" ]]; then
    echo "Gloo 网卡：${GLOO_SOCKET_IFNAME}"
  fi
fi

CONTROL_MODULE="src.portal.eval_control"
CONTROL_NAME="eval_control"

if [[ "$RPC_PROTOCOL" == "torch" ]]; then
  RANK=0 nohup "$PYTHON_BIN" -m "$CONTROL_MODULE" \
    -f "$CONFIG_FILE" >"$LOG_DIR/${CONTROL_NAME}.log" 2>&1 &
else
  nohup "$PYTHON_BIN" -m "$CONTROL_MODULE" \
    -f "$CONFIG_FILE" >"$LOG_DIR/${CONTROL_NAME}.log" 2>&1 &
fi
CONTROL_PID="$!"
PIDS+=("$CONTROL_PID")
if [[ "$RPC_PROTOCOL" == "torch" ]]; then
  echo "已启动：${CONTROL_NAME}，RANK=0"
else
  echo "已启动：${CONTROL_NAME}"
fi

for ((index = 0; index < SAMPLE_COUNT; index++)); do
  role_id="sample-$index"
  if [[ "$RPC_PROTOCOL" == "torch" ]]; then
    rank=$((index + 1))
    RANK="$rank" nohup "$PYTHON_BIN" -m src.portal.eval_sample \
      -f "$CONFIG_FILE" "$role_id" >"$LOG_DIR/$role_id.log" 2>&1 &
  else
    nohup "$PYTHON_BIN" -m src.portal.eval_sample \
      -f "$CONFIG_FILE" "$role_id" >"$LOG_DIR/$role_id.log" 2>&1 &
  fi
  sample_pid="$!"
  PIDS+=("$sample_pid")
  SAMPLE_PIDS+=("$sample_pid")
  if [[ "$RPC_PROTOCOL" == "torch" ]]; then
    echo "已启动：${role_id}，RANK=${rank}"
  else
    echo "已启动：${role_id}"
  fi
done

echo "已启动：${CONTROL_NAME} + ${SAMPLE_COUNT} 个 sample。按 Ctrl+C 统一停止。"

while true; do
  print_eval_progress_updates
  if ! kill -0 "$CONTROL_PID" 2>/dev/null; then
    finish_eval_progress_line
    set +e
    wait "$CONTROL_PID"
    control_status="$?"
    set -e
    finish_after_center_exit "$control_status"
  fi
  if [[ "$RPC_PROTOCOL" == "torch" ]]; then
    for ((sample_index = 0; sample_index < ${#SAMPLE_PIDS[@]}; sample_index++)); do
      sample_pid="${SAMPLE_PIDS[$sample_index]}"
      if kill -0 "$sample_pid" 2>/dev/null; then
        continue
      fi
      if ! kill -0 "$CONTROL_PID" 2>/dev/null; then
        break
      fi
      set +e
      wait "$sample_pid"
      sample_status="$?"
      set -e
      finish_eval_progress_line
      sample_role="sample-$sample_index"
      if is_center_finished_env; then
        echo "sample 已感知中心收口：role=${sample_role}，PID=${sample_pid}，status=${sample_status}；等待中心退出。"
        wait_for_center_after_finished_marker
      fi
      echo "sample 在中心仍存活时提前退出：role=${sample_role}，PID=${sample_pid}，status=${sample_status}；Torch RPC world 失败。" >&2
      if [[ "$sample_status" -eq 0 ]]; then
        exit 1
      fi
      exit "$sample_status"
    done
  fi
  sleep 1
done

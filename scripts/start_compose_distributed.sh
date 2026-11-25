#!/usr/bin/env bash
# 用同目录的 docker-compose-distribute.yml 拉起 1 中心 + N 采样容器。
set -Eeuo pipefail

usage() {
  cat <<'EOF'
用法：
  ./scripts/start_compose_distributed.sh <科目配置文件> --rpc grpc|trpc [--mode train|eval] [--samples N] [-d]
  ./scripts/start_compose_distributed.sh --down

示例：
  ./scripts/start_compose_distributed.sh \
    subjects/cooperative_search/cooperative_search_qmix.toml --rpc trpc --mode train
  DODO_TASK_ID=<train_id> ./scripts/start_compose_distributed.sh \
    subjects/cooperative_search/cooperative_search_qmix.toml --rpc trpc --mode eval
EOF
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose-distribute.yml"
COMPOSE_GPU_FILE="$SCRIPT_DIR/docker-compose-distribute.gpu.yml"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

compose_has_nvidia() {
  docker info --format '{{range $k, $v := .Runtimes}}{{$k}} {{end}}' 2>/dev/null | grep -qw nvidia
}

compose() {
  local files=(-f "$COMPOSE_FILE")
  if [[ -f "$COMPOSE_GPU_FILE" ]] && compose_has_nvidia; then
    files+=(-f "$COMPOSE_GPU_FILE")
  fi
  docker compose "${files[@]}" --project-directory "$PROJECT_ROOT" "$@"
}

compose_slot_lock_volume() {
  printf '%s_compose_sample_slot_lock' "$(basename "$PROJECT_ROOT")"
}

if [[ "${1:-}" == "--down" ]]; then
  compose down --volumes
  docker volume rm -f "$(compose_slot_lock_volume)" 2>/dev/null || true
  exit 0
fi

CONFIG_FILE=""
RPC=""
MODE="train"
SAMPLES=""
DETACH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rpc)
      RPC="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --samples)
      SAMPLES="${2:-}"
      shift 2
      ;;
    -d|--detach)
      DETACH=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --down)
      compose down --volumes
      docker volume rm -f "$(compose_slot_lock_volume)" 2>/dev/null || true
      exit 0
      ;;
    -*)
      echo "不支持的选项：$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$CONFIG_FILE" ]]; then
        echo "科目配置文件只能指定一次：$1" >&2
        exit 2
      fi
      CONFIG_FILE="$1"
      shift
      ;;
  esac
done

if [[ -z "$CONFIG_FILE" || -z "$RPC" ]]; then
  usage >&2
  exit 2
fi
if [[ "$RPC" != "grpc" && "$RPC" != "trpc" ]]; then
  echo "--rpc 必须是 grpc 或 trpc：$RPC" >&2
  exit 2
fi
if [[ "$MODE" != "train" && "$MODE" != "eval" ]]; then
  echo "--mode 必须是 train 或 eval：$MODE" >&2
  exit 2
fi
if [[ "$CONFIG_FILE" != /* ]]; then
  CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "配置文件不存在：$CONFIG_FILE" >&2
  exit 2
fi
if [[ ! -x "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -n "${PYTHON_BIN:-}" ]] && ! PYTHONPATH="$PROJECT_ROOT/src" "$PYTHON_BIN" -c 'from config import TorchRpcConfig' >/dev/null 2>&1; then
  PYTHON_BIN=""
fi

run_toml_python() {
  if [[ -n "${PYTHON_BIN:-}" && -x "$PYTHON_BIN" ]]; then
    PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$@"
    return
  fi
  docker run --rm \
    -v "$PROJECT_ROOT:/code" \
    -w /code \
    -e PYTHONPATH=/code/src \
    -e DODO_RPC="$DODO_RPC" \
    -e DODO_TASK_ID="${DODO_TASK_ID:-}" \
    "${DODORL_IMAGE:-dodorl:base}" \
    python "$@"
}

export DODO_RPC="$RPC"
export DODO_MODE="$MODE"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG_RELATIVE="$CONFIG_FILE"
if [[ "$CONFIG_FILE" == "$PROJECT_ROOT/"* ]]; then
  CONFIG_RELATIVE="${CONFIG_FILE#"$PROJECT_ROOT"/}"
fi

toml_python_config="$CONFIG_FILE"
toml_python_root="$PROJECT_ROOT"
toml_task_id_resolver="$SCRIPT_DIR/resolve_design_task_id.py"
if [[ -z "${PYTHON_BIN:-}" || ! -x "${PYTHON_BIN:-}" ]]; then
  toml_python_config="/code/${CONFIG_RELATIVE}"
  toml_python_root="/code"
  toml_task_id_resolver="/code/scripts/resolve_design_task_id.py"
fi

export DODO_TASK_ID="$(
  run_toml_python "$toml_task_id_resolver" "$toml_python_config"
)"

BRANCH="$MODE"
CONFIG_INFO="$(run_toml_python -c '
import sys
from pathlib import Path
from config import GrpcConfig, TorchRpcConfig
from utils.toml_config import TomlTrainingConfig
config_file = Path(sys.argv[1])
project_root = Path(sys.argv[2])
branch = sys.argv[3]
want = sys.argv[4]
conf = TomlTrainingConfig(config_file, project_root=project_root, branch=branch)
if want == "trpc":
    if not isinstance(conf.rpc_config, TorchRpcConfig):
        raise SystemExit("DODO_RPC=trpc 需要 [torch_rpc_config]")
    print(int(conf.rpc_config.world_size))
    host, _, port = conf.rpc_config.init_address.rpartition(":")
    print(port)
else:
    if not isinstance(conf.rpc_config, GrpcConfig):
        raise SystemExit("DODO_RPC=grpc 需要 [grpc_config]")
    print(0)
    _, _, port = (conf.rpc_config.task_manager_address or "127.0.0.1:53151").rpartition(":")
    print(port or "53151")
print(len(conf.stage_task_config.stages))
' "$toml_python_config" "$toml_python_root" "$BRANCH" "$RPC")"

WORLD_SIZE="$(printf '%s\n' "$CONFIG_INFO" | sed -n '1p')"
HEALTH_PORT="$(printf '%s\n' "$CONFIG_INFO" | sed -n '2p')"
STAGE_COUNT="$(printf '%s\n' "$CONFIG_INFO" | sed -n '3p')"

if [[ "$RPC" == "trpc" ]]; then
  COMPUTED_SAMPLES=$((WORLD_SIZE - 1))
  if [[ -n "$SAMPLES" && "$SAMPLES" != "$COMPUTED_SAMPLES" ]]; then
    echo "Torch RPC --samples 必须等于 world_size-1=$COMPUTED_SAMPLES，得到 $SAMPLES。" >&2
    exit 2
  fi
  SAMPLE_COUNT="$COMPUTED_SAMPLES"
else
  if [[ -z "$SAMPLES" ]]; then
    echo "gRPC 需要 --samples N。" >&2
    exit 2
  fi
  if [[ ! "$SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
    echo "采样进程数必须是正整数：$SAMPLES" >&2
    exit 2
  fi
  SAMPLE_COUNT="$SAMPLES"
fi

if [[ "$MODE" == "eval" ]]; then
  CONTROL_MODULE="src.portal.eval_control"
  CONTROL_LOG="eval_control.log"
elif [[ "$STAGE_COUNT" -gt 1 ]]; then
  CONTROL_MODULE="src.portal.stage_learn"
  CONTROL_LOG="stage_learn.log"
else
  CONTROL_MODULE="src.portal.learn_control"
  CONTROL_LOG="learn_control.log"
fi

LOG_DIR="${DODO_LOG_DIR:-$PROJECT_ROOT/logs/distributed/$DODO_TASK_ID}"
mkdir -p "$LOG_DIR"
# 同 TASK_ID 再次 up 时清掉上次留下的占槽目录，否则 sample 会报「已有 N 个副本」。
rm -rf "$LOG_DIR/slots"
cp "$CONFIG_FILE" "$LOG_DIR/$(basename "$CONFIG_FILE")"
PROGRESS_FILE="$LOG_DIR/training_progress.env"

if [[ "$LOG_DIR" == "$PROJECT_ROOT/"* ]]; then
  LOG_DIR_RELATIVE="${LOG_DIR#"$PROJECT_ROOT"/}"
  PROGRESS_RELATIVE="${PROGRESS_FILE#"$PROJECT_ROOT"/}"
else
  LOG_DIR_RELATIVE="$LOG_DIR"
  PROGRESS_RELATIVE="$PROGRESS_FILE"
fi

export DODORL_PROJECT_ROOT="${DODORL_PROJECT_ROOT:-$PROJECT_ROOT}"
export DODORL_IMAGE="${DODORL_IMAGE:-dodorl:base}"
export DODO_SAMPLE_COUNT="$SAMPLE_COUNT"
export DODO_CONFIG_FILE="$CONFIG_RELATIVE"
export DODO_LOG_DIR="$LOG_DIR_RELATIVE"
export DODO_PROGRESS_FILE="$PROGRESS_RELATIVE"
export DODO_CONTROL_MODULE="$CONTROL_MODULE"
export DODO_CONTROL_LOG="$CONTROL_LOG"
export DODO_RPC_HEALTH_PORT="${HEALTH_PORT:-53151}"

if [[ -z "$(compose ps -q 2>/dev/null || true)" ]]; then
  docker volume rm -f "$(compose_slot_lock_volume)" 2>/dev/null || true
fi

echo "启动 compose 分布式任务：DODO_TASK_ID=${DODO_TASK_ID}"
echo "配置文件：$CONFIG_FILE"
echo "协议：$RPC  模式：$MODE  采样容器数：$SAMPLE_COUNT"
echo "中心模块：$CONTROL_MODULE"
echo "日志目录：$LOG_DIR"

COMPOSE_UP_ARGS=(up)
if [[ "$DETACH" -eq 1 ]]; then
  COMPOSE_UP_ARGS+=(-d)
fi
compose "${COMPOSE_UP_ARGS[@]}" --scale control=1 --scale sample="$SAMPLE_COUNT"

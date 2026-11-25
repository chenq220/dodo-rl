#!/usr/bin/env bash
# 统一角色启动入口：本地参考命令和容器 ENTRYPOINT 均可复用。
set -euo pipefail

# 解除容器或启动终端可能继承的单文件大小软限制，避免长日志和GIF被截断。
if ! ulimit -f unlimited 2>/dev/null; then
  echo "无法解除单文件大小限制，请先提高运行环境的文件大小硬限制。" >&2
  exit 1
fi

usage() {
  cat <<'EOF'
用法：./startup.sh <role> [入口参数...]

可用 role：
  eval                策略评估
  learn_control       单阶段分布式学习控制
  sample              分布式采样进程
  serial_stage_train  单进程分阶段训练
  stage_learn         分阶段分布式学习控制
  serial_train        单进程串行训练
  allinone_grpc       一键启动 gRPC 中心与采样进程
  allinone_trpc       一键启动 Torch RPC 中心与采样进程

示例：
  ./startup.sh serial_stage_train -f subjects/cooperative_search/cooperative_search_qmix.toml
  ./startup.sh stage_learn -f subjects/cooperative_search/cooperative_search_qmix.toml
  ./startup.sh sample sample-0 -f subjects/cooperative_search/cooperative_search_qmix.toml
  ./startup.sh allinone_grpc subjects/cooperative_search/cooperative_search_qmix.toml 3
  ./startup.sh allinone_trpc subjects/cooperative_search/cooperative_search_qmix.toml
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

role="$1"
shift

case "$role" in
  eval|learn_control|sample|serial_stage_train|stage_learn|serial_train)
    module="portal.$role"
    ;;
  allinone_grpc)
    launcher="scripts/start_distributed.sh"
    ;;
  allinone_trpc)
    launcher="scripts/start_torch_rpc_distributed.sh"
    ;;
  *)
    echo "不支持的 role：$role" >&2
    usage >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# 一键分布式启动脚本自行生成任务 ID，以便每个任务组拥有唯一标识。
if [[ -n "${launcher:-}" ]]; then
  exec "$PROJECT_ROOT/$launcher" "$@"
fi

# 容器编排可显式传入任务 ID；本地启动则按月、日、时、分生成八位默认值。
if [[ -z "${DODO_TASK_ID:-}" ]]; then
  export DODO_TASK_ID="$(date +%m%d%H%M)"
fi

# Python 在加载根目录 sitecustomize.py 前尚未把 cwd 放入 sys.path；
# 显式暴露仓库根，使其将 src/ 加入 sys.path，同时保证 subjects 可导入。
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec python -m "$module" "$@"

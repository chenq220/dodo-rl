#!/usr/bin/env bash
# QMIX 算法规范性发烟测试 demo：用 run_test.py 跑 qmix.toml。
# 用法（在仓库根目录或本目录均可）：
#   ./sdk/algo_conformance/demo.sh
#   ./sdk/algo_conformance/demo.sh --only learner
#   ./sdk/algo_conformance/demo.sh --only agent
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TEST="$SCRIPT_DIR/run_test.py"
CONFIG="$SCRIPT_DIR/qmix.toml"

if [[ "$PYTHON_BIN" == */* ]]; then
  if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$RUN_TEST" -f "$CONFIG" "$@"

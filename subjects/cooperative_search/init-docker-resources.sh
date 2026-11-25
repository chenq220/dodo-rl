#!/usr/bin/env bash
# 将 cooperative_search 训练所需的固定威胁策略初始化到工程 artifacts/。
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${DODORL_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts"
MODEL_RELATIVE_PATH="models/cooperative_search_threat/final_b1813237_avoid_targets_0_dynamic_threats_agent_net.pth"
# 随本脚本分发，便于在没有项目 artifacts/ 目录的机器上初始化威胁模型。
MODEL_FILE="$SCRIPT_DIR/final_b1813237_avoid_targets_0_dynamic_threats_agent_net.pth"

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "本地模型文件不存在：$MODEL_FILE" >&2
  exit 2
fi

target="$ARTIFACTS_DIR/$MODEL_RELATIVE_PATH"
if [[ -e "$target" ]]; then
  echo "模型已存在，跳过初始化：$target"
  exit 0
fi

mkdir -p "$(dirname "$target")"
cp "$MODEL_FILE" "$target"
echo "模型初始化完成：$target"
echo "产物目录初始化完成：$ARTIFACTS_DIR"

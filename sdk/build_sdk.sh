#!/usr/bin/env bash
# 构建可独立安装的训练 SDK；发布由 CI 使用其自身的凭据处理。
set -euo pipefail

SDK_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SDK_DIR/.." && pwd)"
SDK_PACKAGE_DIR="$SDK_DIR/src/dodorl_sdk"
DIST_DIR="$SDK_DIR/dist"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dodorl-sdk-build.XXXXXX")"

# 构建 SDK 使用的 Python 3.12 解释器（按需修改此路径）。
PYTHON_BIN="${DODORL_BUILD_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "解释器不存在或不可执行: ${PYTHON_BIN}，请修改本脚本顶部配置" >&2
  exit 1
fi

"$PYTHON_BIN" -c 'import sys; sys.exit("Python 3.12 is required, got %s" % sys.version.split()[0]) if sys.version_info[:2] != (3, 12) else None'

cleanup() {
  rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

mkdir -p "$BUILD_DIR/src" "$DIST_DIR"
# Keep previous versions available when building a new wheel.

# wheel 顶层只安装 dodorl_sdk；运行时实现收纳在 dodorl_sdk.dodorl 下。
cp -R "$SDK_PACKAGE_DIR" "$BUILD_DIR/src/dodorl_sdk"
mkdir -p "$BUILD_DIR/src/dodorl_sdk/dodorl"
cp "$SDK_DIR/runtime_package_init.py" "$BUILD_DIR/src/dodorl_sdk/dodorl/__init__.py"
for package in algos common engine envs models rpc sample scheduler utils; do
  cp -R "$PROJECT_ROOT/src/$package" "$BUILD_DIR/src/dodorl_sdk/dodorl/$package"
done
# SDK 公开入口由 sdk/src/dodorl_sdk/portal.py 的 __all__ 显式定义。
# 只分发这些入口依赖的角色实现；新角色不会因出现在 src/roles 下就自动进入 SDK。
SDK_ROLE_MODULES=(
  control stage_control learn model learn_control stage_learn_control sample
  serial_train serial_stage_train evaluation eval_control
)
mkdir -p "$BUILD_DIR/src/dodorl_sdk/dodorl/roles"
cp "$PROJECT_ROOT/src/roles/__init__.py" "$BUILD_DIR/src/dodorl_sdk/dodorl/roles/__init__.py"
cp -R "$PROJECT_ROOT/src/roles/common" "$BUILD_DIR/src/dodorl_sdk/dodorl/roles/common"
for role_module in "${SDK_ROLE_MODULES[@]}"; do
  cp "$PROJECT_ROOT/src/roles/$role_module.py" "$BUILD_DIR/src/dodorl_sdk/dodorl/roles/$role_module.py"
done
# src/portal 是源工程命令行入口，不复制进 SDK；SDK 用户使用 dodorl_sdk.portal。
cp "$PROJECT_ROOT/src/config.py" "$BUILD_DIR/src/dodorl_sdk/dodorl/config.py"
cp "$SDK_DIR/pyproject.toml" "$BUILD_DIR/pyproject.toml"
cp "$SDK_DIR/README.md" "$BUILD_DIR/src/dodorl_sdk/README.md"
cp "$SDK_DIR/Agent.md" "$BUILD_DIR/src/dodorl_sdk/Agent.md"

# 源工程保留既有顶层导入；只在 wheel 构建副本中限定到 SDK 私有命名空间。
"$PYTHON_BIN" "$SDK_DIR/rewrite_sdk_imports.py" "$BUILD_DIR/src/dodorl_sdk"

"$PYTHON_BIN" -m build --no-isolation --wheel --outdir "$DIST_DIR" "$BUILD_DIR"
echo "SDK wheel directory: $DIST_DIR"

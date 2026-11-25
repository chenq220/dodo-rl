#!/usr/bin/env bash
# 按标签含义打 dodorl 镜像：base 无代码；dev=develop；v*=git tag。
set -Eeuo pipefail

usage() {
  cat <<'EOF'
用法：
  ./scripts/docker-build.sh [base|dev|vX.Y.Z]

镜像规则：
  dodorl:base      无业务代码，Dockerfile-base
  dodorl:dev       有代码，对应 develop 分支
  dodorl:v0.11.0   有代码，对应 git tag v0.11.0

无参数时：当前分支是 develop 则打 dev；HEAD 正好是 v* tag 则打该版本。
也可写 dodorl:dev / dodorl:v0.11.0（会去掉仓库名前缀）。

示例：
  ./scripts/docker-build.sh base
  ./scripts/docker-build.sh
  ./scripts/docker-build.sh v0.11.0
EOF
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"

is_version_tag() {
  [[ "$1" =~ ^v[0-9]+(\.[0-9]+)*$ ]]
}

normalize_name() {
  local raw="$1"
  raw="${raw#dodorl:}"
  printf '%s' "$raw"
}

detect_code_name() {
  if ! git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "当前目录不是 git 仓库，请显式传入 base、dev 或版本 tag（如 v0.11.0）。" >&2
    exit 2
  fi
  local branch
  branch="$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD)"
  if [[ "$branch" == "develop" ]]; then
    printf '%s' "dev"
    return
  fi
  local tag=""
  if tag="$(git -C "$PROJECT_ROOT" describe --tags --exact-match HEAD 2>/dev/null)"; then
    if is_version_tag "$tag"; then
      printf '%s' "$tag"
      return
    fi
  fi
  echo "无法从 git 推断镜像标签（branch=${branch}）。请显式传入 base、dev 或版本 tag（如 v0.11.0）。" >&2
  exit 2
}

NAME=""
if [[ $# -eq 0 ]]; then
  NAME="$(detect_code_name)"
else
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "不支持的选项：$1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ $# -ne 1 ]]; then
        echo "只能指定一个镜像标签。" >&2
        usage >&2
        exit 2
      fi
      NAME="$(normalize_name "$1")"
      ;;
  esac
fi

if [[ "$NAME" != "base" && "$NAME" != "dev" ]] && ! is_version_tag "$NAME"; then
  echo "不支持的镜像标签：${NAME}。只接受 base、dev 或 v* 版本 tag（如 v0.11.0）。" >&2
  usage >&2
  exit 2
fi

IMAGE="dodorl:${NAME}"

if [[ "$NAME" == "base" ]]; then
  echo "构建无代码基础镜像：${IMAGE}（${PLATFORM}）"
  docker build \
    --platform "$PLATFORM" \
    --file "$PROJECT_ROOT/Dockerfile-base" \
    --tag "$IMAGE" \
    "$PROJECT_ROOT"
  exit 0
fi

if ! docker image inspect dodorl:base >/dev/null 2>&1; then
  echo "本地不存在 dodorl:base。请先执行：./scripts/docker-build.sh base" >&2
  exit 2
fi
echo "复用本地基础镜像：dodorl:base"
echo "构建带代码镜像：${IMAGE}（${PLATFORM}）"
docker build \
  --platform "$PLATFORM" \
  --file "$PROJECT_ROOT/Dockerfile" \
  --tag "$IMAGE" \
  "$PROJECT_ROOT"

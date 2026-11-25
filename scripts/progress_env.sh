# 进度快照是 KEY=value 环境变量镜像；Python 子进程改不了父 shell 环境。
read_progress_kv() {
  local file="${1:-}" key="${2:-}"
  [[ -n "$file" && -r "$file" ]] || return 0
  awk -F= -v k="$key" '$1==k {sub(/^[^=]+=/, ""); print; exit}' "$file"
}

print_training_progress_updates() {
  local progress_file="${DODO_PROGRESS_FILE:-}"
  local stage episode
  [[ -n "$progress_file" ]] || return 0
  stage="$(read_progress_kv "$progress_file" DODO_TRAINING_STAGE)"
  episode="$(read_progress_kv "$progress_file" DODO_TRAINING_EPISODE)"
  [[ -n "$stage" && -n "$episode" ]] || return 0
  if [[ "$stage" == "${DODO_TRAINING_STAGE:-}" && "$episode" == "${DODO_TRAINING_EPISODE:-}" ]]; then
    return 0
  fi
  export DODO_TRAINING_STAGE="$stage"
  export DODO_TRAINING_EPISODE="$episode"
  printf '\r训练进度：阶段=%s，局数=%s\033[K' "$stage" "$episode"
  PROGRESS_LINE_ACTIVE=1
}

finish_training_progress_line() {
  if [[ "${PROGRESS_LINE_ACTIVE:-0}" -eq 1 ]]; then
    printf '\n'
    PROGRESS_LINE_ACTIVE=0
  fi
}

print_eval_progress_updates() {
  local progress_file="${DODO_PROGRESS_FILE:-}"
  local episode
  [[ -n "$progress_file" ]] || return 0
  episode="$(read_progress_kv "$progress_file" DODO_TRAINING_EPISODE)"
  [[ -n "$episode" ]] || return 0
  if [[ "$episode" == "${DODO_TRAINING_EPISODE:-}" ]]; then
    return 0
  fi
  export DODO_TRAINING_STAGE="$(read_progress_kv "$progress_file" DODO_TRAINING_STAGE)"
  export DODO_TRAINING_EPISODE="$episode"
  printf '\r评估进度：局数=%s\033[K' "$episode"
  PROGRESS_LINE_ACTIVE=1
}

finish_eval_progress_line() {
  finish_training_progress_line
}

is_center_finished_env() {
  [[ "$(read_progress_kv "${DODO_PROGRESS_FILE:-}" DODO_CENTER_FINISHED)" == "1" ]]
}

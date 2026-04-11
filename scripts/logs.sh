#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

ensure_runtime_dirs
service="${1:-all}"
follow="${2:-}"

show_log() {
  local file="$1"
  local label="$2"
  if [[ ! -f "$file" ]]; then
    log_warning "$label 日志不存在: $file"
    return 0
  fi
  echo "===== $label ====="
  if [[ "$follow" == "-f" || "$follow" == "--follow" ]]; then
    tail -n 80 -f "$file"
  else
    tail -n 80 "$file"
  fi
}

case "$service" in
  backend) show_log "$BACKEND_RUNTIME_LOG" backend ;;
  frontend) show_log "$FRONTEND_RUNTIME_LOG" frontend ;;
  worker) show_log "$WORKER_RUNTIME_LOG" worker ;;
  all)
    show_log "$BACKEND_RUNTIME_LOG" backend
    echo
    show_log "$FRONTEND_RUNTIME_LOG" frontend
    echo
    show_log "$WORKER_RUNTIME_LOG" worker
    ;;
  *)
    log_error "未知服务: $service"
    echo "用法: $0 [backend|frontend|worker|all] [-f]"
    exit 1
    ;;
esac

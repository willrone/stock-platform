#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

ensure_runtime_dirs

if ! command_exists tmux; then
  log_error "未找到 tmux，当前开发编排无法启动。"
  exit 1
fi

if tmux_session_exists; then
  log_warning "tmux 会话 $DEV_SESSION_NAME 已存在。"
  echo "可执行: tmux attach -t $DEV_SESSION_NAME"
  exit 0
fi

: > "$BACKEND_RUNTIME_LOG"
: > "$FRONTEND_RUNTIME_LOG"
: > "$WORKER_RUNTIME_LOG"

log_info "先执行环境体检..."
"$PROJECT_ROOT/scripts/doctor.sh"

log_info "创建 tmux 开发会话: $DEV_SESSION_NAME"

tmux new-session -d -s "$DEV_SESSION_NAME" -n backend "cd '$PROJECT_ROOT' && bash '$PROJECT_ROOT/scripts/dev-backend.sh' 2>&1 | tee -a '$BACKEND_RUNTIME_LOG'"
tmux new-window -t "$DEV_SESSION_NAME" -n frontend "cd '$PROJECT_ROOT' && bash '$PROJECT_ROOT/scripts/dev-frontend.sh' 2>&1 | tee -a '$FRONTEND_RUNTIME_LOG'"
tmux new-window -t "$DEV_SESSION_NAME" -n worker "cd '$PROJECT_ROOT' && bash '$PROJECT_ROOT/scripts/dev-worker.sh' 2>&1 | tee -a '$WORKER_RUNTIME_LOG'"

sleep 2

log_success "开发会话已启动。"
echo ""
echo "  tmux 会话: $DEV_SESSION_NAME"
echo "  backend 日志: $BACKEND_RUNTIME_LOG"
echo "  frontend 日志: $FRONTEND_RUNTIME_LOG"
echo "  worker 日志: $WORKER_RUNTIME_LOG"
echo ""
echo "常用命令:"
echo "  tmux attach -t $DEV_SESSION_NAME"
echo "  $PROJECT_ROOT/status.sh"
echo "  $PROJECT_ROOT/stop.sh"

aif="${ATTACH_TMUX:-1}"
if [[ "$aif" == "1" && -t 1 ]]; then
  exec tmux attach -t "$DEV_SESSION_NAME"
fi

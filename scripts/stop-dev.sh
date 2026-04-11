#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if tmux_session_exists; then
  log_info "停止 tmux 会话: $DEV_SESSION_NAME"
  tmux kill-session -t "$DEV_SESSION_NAME"
  log_success "开发会话已停止。"
else
  log_warning "tmux 会话 $DEV_SESSION_NAME 不存在。"
fi

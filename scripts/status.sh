#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

BACKEND_PORT="${STOCK_PLATFORM_BACKEND_PORT:-$DEFAULT_DEV_BACKEND_PORT}"
FRONTEND_PORT="${STOCK_PLATFORM_FRONTEND_PORT:-$DEFAULT_DEV_FRONTEND_PORT}"

if tmux_session_exists; then
  log_success "tmux 会话运行中: $DEV_SESSION_NAME"
  tmux list-windows -t "$DEV_SESSION_NAME"
else
  log_warning "tmux 会话未运行: $DEV_SESSION_NAME"
fi

echo ""
for entry in backend:$BACKEND_PORT frontend:$FRONTEND_PORT; do
  name="${entry%%:*}"
  port="${entry##*:}"
  if is_port_in_use "$port"; then
    log_success "$name 端口监听中: $port"
    port_owner "$port" | sed 's/^/  /'
  else
    log_warning "$name 端口未监听: $port"
  fi
  echo ""
done

if command_exists curl; then
  if curl -fsS "http://127.0.0.1:${BACKEND_PORT}/api/v1/health" >/dev/null 2>&1; then
    log_success "backend 健康检查通过"
  else
    log_warning "backend 健康检查未通过"
  fi
fi

echo "访问地址:"
echo "  Frontend: http://127.0.0.1:${FRONTEND_PORT}"
echo "  Backend:  http://127.0.0.1:${BACKEND_PORT}"
echo "  API Docs: http://127.0.0.1:${BACKEND_PORT}/api/v1/docs"

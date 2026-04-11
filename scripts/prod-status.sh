#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

BACKEND_PORT="${STOCK_PLATFORM_BACKEND_PORT:-$DEFAULT_DEV_BACKEND_PORT}"
FRONTEND_PORT="${STOCK_PLATFORM_FRONTEND_PORT:-$DEFAULT_DEV_FRONTEND_PORT}"
services=(stock-platform-backend.service stock-platform-frontend.service stock-platform-worker.service)

if command_exists systemctl; then
  echo "systemd 服务状态:"
  for svc in "${services[@]}"; do
    state="$(systemctl is-active "$svc" 2>/dev/null || true)"
    enabled="$(systemctl is-enabled "$svc" 2>/dev/null || true)"
    printf '  %-32s active=%s enabled=%s\n' "$svc" "${state:-unknown}" "${enabled:-unknown}"
  done
else
  log_warning "当前机器未安装 systemctl，跳过 systemd 状态检查。"
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

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [[ -n "${LAN_IP:-}" ]]; then
  echo "局域网访问入口:"
  echo "  frontend: http://${LAN_IP}:${FRONTEND_PORT}"
  echo "  backend health: http://${LAN_IP}:${BACKEND_PORT}/api/v1/health"
  echo ""
fi

if command_exists curl; then
  if curl -fsS "http://127.0.0.1:${BACKEND_PORT}/api/v1/health" >/dev/null 2>&1; then
    log_success "backend 健康检查通过 (/api/v1/health)"
  else
    log_warning "backend 健康检查未通过 (/api/v1/health)"
  fi
fi

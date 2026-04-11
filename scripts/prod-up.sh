#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if ! command_exists systemctl; then
  log_error "未找到 systemctl，当前机器无法使用 systemd 生产启动。"
  exit 1
fi

services=(stock-platform-backend.service stock-platform-frontend.service stock-platform-worker.service)

log_info "通过 systemd 启动 stock-platform 服务..."
sudo systemctl start "${services[@]}"

log_success "已触发启动。"
exec "$PROJECT_ROOT/scripts/prod-status.sh"

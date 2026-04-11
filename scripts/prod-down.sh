#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if ! command_exists systemctl; then
  log_error "未找到 systemctl，当前机器无法使用 systemd 生产停机。"
  exit 1
fi

services=(stock-platform-frontend.service stock-platform-worker.service stock-platform-backend.service)

log_info "停止 stock-platform systemd 服务..."
sudo systemctl stop "${services[@]}"
log_success "已停止。"

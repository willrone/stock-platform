#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if ! command_exists systemctl; then
  log_error "未找到 systemctl，无法安装 systemd unit。"
  exit 1
fi

units=(
  stock-platform-backend.service
  stock-platform-frontend.service
  stock-platform-worker.service
)

log_info "安装 systemd unit 到 /etc/systemd/system ..."
for unit in "${units[@]}"; do
  sudo install -m 0644 "$PROJECT_ROOT/systemd/$unit" "/etc/systemd/system/$unit"
done

sudo systemctl daemon-reload
sudo systemctl enable "${units[@]}"

log_success "systemd unit 已安装并设置为开机启用。"
echo ""
echo "下一步:"
echo "  1. $PROJECT_ROOT/scripts/prod-build.sh"
echo "  2. $PROJECT_ROOT/scripts/prod-up.sh"
echo "  3. $PROJECT_ROOT/scripts/prod-status.sh"

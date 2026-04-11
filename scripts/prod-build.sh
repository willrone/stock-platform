#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

ensure_runtime_dirs
ensure_backend_env
ensure_frontend_env

BACKEND_PYTHON="$(backend_python_bin)"

if [[ ! -x "$BACKEND_PYTHON" ]]; then
  log_error "未找到 backend Python 虚拟环境: $BACKEND_PYTHON"
  echo "请先执行: $PROJECT_ROOT/scripts/setup-backend.sh"
  exit 1
fi

if ! command_exists npm; then
  log_error "未找到 npm，无法构建 frontend。"
  exit 1
fi

log_info "检查 backend 基础环境..."
"$BACKEND_PYTHON" -V

log_info "构建 frontend 生产产物..."
(
  cd "$PROJECT_ROOT/frontend"
  npm run build
)

log_success "生产构建完成。"
echo ""
echo "下一步:"
echo "  1. sudo $PROJECT_ROOT/scripts/install-systemd.sh"
echo "  2. $PROJECT_ROOT/scripts/prod-up.sh"

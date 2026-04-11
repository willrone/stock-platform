#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

ensure_runtime_dirs
ensure_frontend_env
load_env_file "$FRONTEND_ENV_FILE"

FRONTEND_HOST="${STOCK_PLATFORM_FRONTEND_HOST:-$DEFAULT_DEV_FRONTEND_HOST}"
FRONTEND_PORT="${STOCK_PLATFORM_FRONTEND_PORT:-${PORT:-$DEFAULT_DEV_FRONTEND_PORT}}"
BACKEND_HOST="${STOCK_PLATFORM_BACKEND_HOST:-$DEFAULT_DEV_BACKEND_HOST}"
BACKEND_PORT="${STOCK_PLATFORM_BACKEND_PORT:-$DEFAULT_DEV_BACKEND_PORT}"

if [[ ! -d "$PROJECT_ROOT/frontend/node_modules" ]]; then
  log_error "frontend/node_modules 不存在，请先运行 scripts/setup-frontend.sh"
  exit 1
fi

ensure_port_free "$FRONTEND_PORT" "frontend"

export PORT="$FRONTEND_PORT"
export NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-http://${BACKEND_HOST}:${BACKEND_PORT}}"
export NEXT_PUBLIC_WS_URL="${NEXT_PUBLIC_WS_URL:-ws://${BACKEND_HOST}:${BACKEND_PORT}}"

log_info "启动 frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}"
log_info "前端将连接 API: ${NEXT_PUBLIC_API_URL}"

cd "$PROJECT_ROOT/frontend"
exec npm run dev -- --hostname "$FRONTEND_HOST" --port "$FRONTEND_PORT"

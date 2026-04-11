#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

ensure_runtime_dirs
ensure_backend_env
load_env_file "$BACKEND_ENV_FILE"

BACKEND_HOST="${STOCK_PLATFORM_BACKEND_HOST:-${HOST:-$DEFAULT_DEV_BACKEND_HOST}}"
BACKEND_PORT="${STOCK_PLATFORM_BACKEND_PORT:-${PORT:-$DEFAULT_DEV_BACKEND_PORT}}"
METRICS_PORT="${STOCK_PLATFORM_METRICS_PORT:-${METRICS_PORT:-$DEFAULT_DEV_METRICS_PORT}}"
FRONTEND_PORT="${STOCK_PLATFORM_FRONTEND_PORT:-$DEFAULT_DEV_FRONTEND_PORT}"
BACKEND_PYTHON="$(backend_python_bin)"

if [[ ! -x "$BACKEND_PYTHON" ]]; then
  log_error "未找到后端虚拟环境，请先运行 scripts/setup-backend.sh"
  exit 1
fi

ensure_port_free "$BACKEND_PORT" "backend"

export HOST="$BACKEND_HOST"
export PORT="$BACKEND_PORT"
export DEBUG="${DEBUG:-true}"
export METRICS_PORT="$METRICS_PORT"
export CORS_ORIGINS="${CORS_ORIGINS:-http://127.0.0.1:${FRONTEND_PORT},http://localhost:${FRONTEND_PORT}}"

log_info "启动 backend: http://${BACKEND_HOST}:${BACKEND_PORT}"
log_info "API 文档: http://${BACKEND_HOST}:${BACKEND_PORT}/api/v1/docs"

cd "$PROJECT_ROOT/backend"
exec "$BACKEND_PYTHON" run.py

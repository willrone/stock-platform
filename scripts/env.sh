#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$PROJECT_ROOT/runtime"
RUNTIME_LOG_DIR="$RUNTIME_DIR/logs"
RUNTIME_PID_DIR="$RUNTIME_DIR/pids"
DEV_SESSION_NAME="stock-platform-dev"

BACKEND_ENV_FILE="$PROJECT_ROOT/backend/.env"
BACKEND_ENV_EXAMPLE="$PROJECT_ROOT/backend/.env.example"
FRONTEND_ENV_FILE="$PROJECT_ROOT/frontend/.env.local"
FRONTEND_ENV_EXAMPLE="$PROJECT_ROOT/frontend/.env.local.example"

DEFAULT_DEV_BACKEND_HOST="127.0.0.1"
DEFAULT_DEV_BACKEND_PORT="18082"
DEFAULT_DEV_FRONTEND_HOST="127.0.0.1"
DEFAULT_DEV_FRONTEND_PORT="13000"
DEFAULT_DEV_METRICS_PORT="19090"

BACKEND_RUNTIME_LOG="$RUNTIME_LOG_DIR/backend.log"
FRONTEND_RUNTIME_LOG="$RUNTIME_LOG_DIR/frontend.log"
WORKER_RUNTIME_LOG="$RUNTIME_LOG_DIR/worker.log"

RED='[0;31m'
GREEN='[0;32m'
YELLOW='[1;33m'
BLUE='[0;34m'
NC='[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERR]${NC} $*"; }

ensure_runtime_dirs() {
  mkdir -p "$RUNTIME_LOG_DIR" "$RUNTIME_PID_DIR"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

is_port_in_use() {
  local port="$1"
  if command_exists lsof; then
    lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  ss -ltn "( sport = :$port )" | tail -n +2 | grep -q .
}

port_owner() {
  local port="$1"
  if command_exists lsof; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN || true
    return
  fi
  ss -ltnp "( sport = :$port )" || true
}

ensure_port_free() {
  local port="$1"
  local name="$2"
  if is_port_in_use "$port"; then
    log_error "$name 需要的端口 $port 已被占用。"
    port_owner "$port"
    exit 1
  fi
}

tmux_session_exists() {
  tmux has-session -t "$DEV_SESSION_NAME" 2>/dev/null
}

backend_python_bin() {
  if [[ -x "$PROJECT_ROOT/backend/.venv/bin/python" ]]; then
    echo "$PROJECT_ROOT/backend/.venv/bin/python"
  elif [[ -x "$PROJECT_ROOT/backend/venv/bin/python" ]]; then
    echo "$PROJECT_ROOT/backend/venv/bin/python"
  else
    echo "$PROJECT_ROOT/backend/.venv/bin/python"
  fi
}

ensure_backend_env() {
  if [[ ! -f "$BACKEND_ENV_FILE" ]]; then
    if [[ -f "$BACKEND_ENV_EXAMPLE" ]]; then
      cp "$BACKEND_ENV_EXAMPLE" "$BACKEND_ENV_FILE"
      log_warning "backend/.env 不存在，已从示例文件创建。"
    else
      cat > "$BACKEND_ENV_FILE" <<'EOF'
APP_NAME="Stock Prediction Platform"
APP_VERSION="0.1.0"
DEBUG=true
LOG_LEVEL="INFO"
HOST="127.0.0.1"
PORT=18082
WORKERS=1
DATABASE_URL="sqlite:///./data/app.db"
REMOTE_DATA_SERVICE_URL="http://192.168.3.62:5002"
REMOTE_DATA_SERVICE_TIMEOUT=30
DATA_ROOT_PATH="../data"
PARQUET_DATA_PATH="../data/stocks"
MODEL_STORAGE_PATH="../data/models"
QLIB_DATA_PATH="../data/qlib_data"
QLIB_CACHE_PATH="../data/qlib_cache"
API_V1_PREFIX="/api/v1"
CORS_ORIGINS="http://127.0.0.1:13000,http://localhost:13000"
ENABLE_METRICS=true
METRICS_PORT=19090
EOF
      log_warning "backend/.env 不存在，已按开发默认值创建。"
    fi
  fi
}

ensure_frontend_env() {
  if [[ ! -f "$FRONTEND_ENV_FILE" ]]; then
    if [[ -f "$FRONTEND_ENV_EXAMPLE" ]]; then
      cp "$FRONTEND_ENV_EXAMPLE" "$FRONTEND_ENV_FILE"
      log_warning "frontend/.env.local 不存在，已从示例文件创建。"
    else
      cat > "$FRONTEND_ENV_FILE" <<'EOF'
PORT=13000
NEXT_PUBLIC_API_URL=http://127.0.0.1:18082
NEXT_PUBLIC_WS_URL=ws://127.0.0.1:18082
NEXT_PUBLIC_APP_NAME=股票预测平台
NEXT_PUBLIC_APP_VERSION=1.0.0
NODE_ENV=development
EOF
      log_warning "frontend/.env.local 不存在，已按开发默认值创建。"
    fi
  fi
}

load_env_file() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

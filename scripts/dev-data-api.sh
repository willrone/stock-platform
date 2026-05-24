#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

ensure_runtime_dirs

DATA_API_PORT="${STOCK_PLATFORM_DATA_API_PORT:-$DEFAULT_DEV_DATA_API_PORT}"
DATA_SERVICE_DIR="$PROJECT_ROOT/back_test_data_service"
START_SCRIPT="$DATA_SERVICE_DIR/start.sh"

if [[ ! -x "$START_SCRIPT" ]]; then
  log_error "未找到可执行的数据服务启动脚本: $START_SCRIPT"
  exit 1
fi

ensure_port_free "$DATA_API_PORT" "data-api"

export PARQUET_DATA_DIR="$PROJECT_ROOT/data/parquet"
export DATA_API_PORT="$DATA_API_PORT"

log_info "启动 data-api: http://0.0.0.0:${DATA_API_PORT}"
log_info "共享 parquet 数据目录: $PARQUET_DATA_DIR"

cd "$DATA_SERVICE_DIR"
exec "$START_SCRIPT" api

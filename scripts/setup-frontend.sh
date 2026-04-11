#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

ensure_runtime_dirs
ensure_frontend_env

cd "$PROJECT_ROOT/frontend"

if ! command_exists npm; then
  log_error "未找到 npm，无法安装前端依赖"
  exit 1
fi

if [[ -f package-lock.json ]]; then
  log_info "安装前端依赖: npm ci"
  npm ci
else
  log_info "安装前端依赖: npm install"
  npm install
fi

log_success "前端环境准备完成。"

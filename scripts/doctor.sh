#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

ensure_runtime_dirs

log_info "检查 stock-platform 开发环境..."

if command_exists python3; then
  log_success "Python: $(python3 --version 2>&1)"
else
  log_error "未找到 python3"
fi

if command_exists node; then
  log_success "Node.js: $(node --version 2>&1)"
else
  log_warning "未找到 node，前端无法启动"
fi

if command_exists npm; then
  log_success "npm: $(npm --version 2>&1)"
else
  log_warning "未找到 npm，前端无法启动"
fi

if command_exists tmux; then
  log_success "tmux: $(tmux -V)"
else
  log_error "未找到 tmux；当前开发编排依赖 tmux"
fi

ensure_backend_env
ensure_frontend_env

if [[ -x "$(backend_python_bin)" ]]; then
  log_success "后端虚拟环境: $(backend_python_bin)"
else
  log_warning "后端虚拟环境未就绪，运行 make setup-backend 或 scripts/setup-backend.sh"
fi

if [[ -d "$PROJECT_ROOT/frontend/node_modules" ]]; then
  log_success "前端依赖目录: frontend/node_modules"
else
  log_warning "前端依赖未安装，运行 make setup-frontend 或 scripts/setup-frontend.sh"
fi

for path in "$PROJECT_ROOT/data" "$PROJECT_ROOT/backend/data" "$PROJECT_ROOT/runtime"; do
  if [[ -d "$path" ]]; then
    log_success "目录存在: $path"
  else
    log_warning "目录缺失: $path"
  fi
done

for port_name in   "backend:$DEFAULT_DEV_BACKEND_PORT"   "frontend:$DEFAULT_DEV_FRONTEND_PORT"   "metrics:$DEFAULT_DEV_METRICS_PORT"; do
  service="${port_name%%:*}"
  port="${port_name##*:}"
  if is_port_in_use "$port"; then
    log_warning "$service 目标端口 $port 已被占用"
    port_owner "$port"
  else
    log_success "$service 目标端口 $port 空闲"
  fi
done

if [[ -f "$PROJECT_ROOT/backend/data/app.db" ]]; then
  log_success "检测到运行库: backend/data/app.db"
else
  log_warning "未检测到 backend/data/app.db"
fi

log_info "推荐命令:"
echo "  make setup           # 准备依赖"
echo "  make dev             # 用 tmux 拉起前后端"
echo "  ./status.sh          # 查看运行状态"

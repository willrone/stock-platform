#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

ensure_runtime_dirs
ensure_backend_env

cd "$PROJECT_ROOT/backend"

if [[ ! -d ".venv" && -d "venv" ]]; then
  log_warning "检测到旧的 backend/venv，当前脚本优先复用它。"
fi

PYTHON_BIN="$(backend_python_bin)"
if [[ ! -x "$PYTHON_BIN" ]]; then
  log_info "创建后端虚拟环境 .venv ..."
  python3 -m venv .venv
  PYTHON_BIN="$PROJECT_ROOT/backend/.venv/bin/python"
fi

log_info "升级 pip / wheel / setuptools ..."
"$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools

REQ_FILE="requirements.txt"
if [[ ! -f "$REQ_FILE" && -f "requirements-minimal.txt" ]]; then
  REQ_FILE="requirements-minimal.txt"
fi

log_info "安装后端依赖: $REQ_FILE"
"$PYTHON_BIN" -m pip install -r "$REQ_FILE"

mkdir -p "$PROJECT_ROOT/data" "$PROJECT_ROOT/data/logs" "$PROJECT_ROOT/backend/data"
log_success "后端环境准备完成。"

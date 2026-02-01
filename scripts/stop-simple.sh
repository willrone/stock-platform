#!/bin/bash

# 停止简单启动的服务
set -e

# 获取脚本所在目录的父目录作为项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

log_error() {
    echo -e "${RED}[错误]${NC} $1"
}

echo "========================================"
echo "    停止股票预测平台服务"
echo "========================================"
echo ""

cd "$PROJECT_ROOT"

# 校验 PID 是否有效（正整数，且不能为 0，避免误杀进程组）
is_valid_pid() {
    local pid=$1
    [[ -z "$pid" ]] && return 1
    [[ ! "$pid" =~ ^[1-9][0-9]*$ ]] && return 1
    return 0
}

# 递归停止进程及其所有子进程（先子后父）
kill_process_tree() {
    local pid=$1
    local force=${2:-false}
    is_valid_pid "$pid" || return 0
    kill -0 "$pid" 2>/dev/null || return 0

    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    if [ -n "$children" ]; then
        for child in $children; do
            kill_process_tree "$child" "$force"
        done
        sleep 0.5
    fi

    if [ "$force" = "true" ]; then
        kill -9 "$pid" 2>/dev/null || true
    else
        kill "$pid" 2>/dev/null || true
    fi
}

# 停止后端服务（先杀所有子进程，再杀主进程）
stop_backend() {
    if [ -f "data/backend.pid" ]; then
        backend_pid=$(cat data/backend.pid | tr -d '[:space:]')
        if ! is_valid_pid "$backend_pid"; then
            log_warning "后端PID文件内容无效: '$backend_pid'，已删除"
            rm -f data/backend.pid
        elif kill -0 "$backend_pid" 2>/dev/null; then
            log_info "停止后端服务 (PID: $backend_pid) 及其所有子进程..."
            kill_process_tree "$backend_pid" false
            sleep 2
            if kill -0 "$backend_pid" 2>/dev/null; then
                log_warning "进程未响应，强制停止..."
                kill_process_tree "$backend_pid" true
            fi
            rm -f data/backend.pid
            log_success "后端服务已停止"
        else
            log_warning "后端服务进程不存在"
            rm -f data/backend.pid
        fi
    else
        log_warning "未找到后端服务PID文件"
    fi
}

# 停止前端服务（先杀所有子进程，再杀主进程）
stop_frontend() {
    if [ -f "data/frontend.pid" ]; then
        frontend_pid=$(cat data/frontend.pid | tr -d '[:space:]')
        if ! is_valid_pid "$frontend_pid"; then
            log_warning "前端PID文件内容无效: '$frontend_pid'，已删除"
            rm -f data/frontend.pid
        elif kill -0 "$frontend_pid" 2>/dev/null; then
            log_info "停止前端服务 (PID: $frontend_pid) 及其所有子进程..."
            kill_process_tree "$frontend_pid" false
            sleep 1
            if kill -0 "$frontend_pid" 2>/dev/null; then
                kill_process_tree "$frontend_pid" true
            fi
            rm -f data/frontend.pid
            log_success "前端服务已停止"
        else
            log_warning "前端服务进程不存在"
            rm -f data/frontend.pid
        fi
    else
        log_warning "未找到前端服务PID文件"
    fi
}

# 清理进程
cleanup_processes() {
    log_info "清理相关进程..."
    
    # 查找并停止可能的残留进程
    pkill -f "python.*run.py" 2>/dev/null || true
    pkill -f "npm.*run.*dev" 2>/dev/null || true
    pkill -f "next.*dev" 2>/dev/null || true
    
    log_success "进程清理完成"
}

# 主函数
main() {
    stop_backend
    stop_frontend
    cleanup_processes
    
    echo ""
    log_success "所有服务已停止！"
    echo ""
}

main
#!/bin/bash

# 股票预测平台停止脚本
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 递归停止进程及其所有子进程
kill_process_tree() {
    local pid=$1
    local force=${2:-false}
    
    if [ -z "$pid" ] || ! kill -0 $pid 2>/dev/null; then
        return 0
    fi
    
    # 获取所有子进程
    local children=$(pgrep -P $pid 2>/dev/null || true)
    if [ -n "$children" ]; then
        for child in $children; do
            # 递归停止子进程
            kill_process_tree $child $force
        done
    fi
    
    # 停止当前进程
    if [ "$force" = "true" ]; then
        kill -9 $pid 2>/dev/null || true
    else
        kill $pid 2>/dev/null || true
    fi
}

# 停止直接运行的进程（非Docker方式）
stop_direct_processes() {
    log_info "查找并停止直接运行的进程..."
    
    # 查找主进程（通过PID文件）
    if [ -f "data/backend.pid" ]; then
        local backend_pid=$(cat data/backend.pid 2>/dev/null)
        if [ -n "$backend_pid" ] && kill -0 $backend_pid 2>/dev/null; then
            log_info "停止后端主进程 (PID: $backend_pid) 及其所有子进程..."
            # 先尝试优雅停止（包括所有子进程）
            kill_process_tree $backend_pid false
            sleep 2
            # 如果还在运行，强制停止
            if kill -0 $backend_pid 2>/dev/null; then
                log_warning "主进程未响应，强制停止..."
                kill_process_tree $backend_pid true
            fi
            rm -f data/backend.pid
        fi
    fi
    
    # 查找并停止所有相关进程（包括子进程）
    # 1. 通过进程名查找主进程
    local main_pids=$(pgrep -f "python.*run\.py" 2>/dev/null || true)
    if [ -n "$main_pids" ]; then
        for pid in $main_pids; do
            log_info "停止主进程 (PID: $pid) 及其所有子进程..."
            kill_process_tree $pid false
            sleep 1
            # 如果还在运行，强制停止
            if kill -0 $pid 2>/dev/null; then
                log_warning "进程未响应，强制停止..."
                kill_process_tree $pid true
            fi
        done
    fi
    
    # 2. 查找 uvicorn 进程
    local uvicorn_pids=$(pgrep -f "uvicorn.*main:app" 2>/dev/null || true)
    if [ -n "$uvicorn_pids" ]; then
        for pid in $uvicorn_pids; do
            log_info "停止 uvicorn 进程 (PID: $pid) 及其所有子进程..."
            kill_process_tree $pid false
            sleep 1
            if kill -0 $pid 2>/dev/null; then
                log_warning "进程未响应，强制停止..."
                kill_process_tree $pid true
            fi
        done
    fi
    
    # 3. 查找并停止所有 multiprocessing 子进程（进程池中的子进程）
    local mp_pids=$(pgrep -f "multiprocessing\.(spawn|resource_tracker)" 2>/dev/null || true)
    if [ -n "$mp_pids" ]; then
        log_info "停止进程池子进程: $mp_pids"
        kill $mp_pids 2>/dev/null || true
        sleep 1
        # 强制停止仍在运行的子进程
        for pid in $mp_pids; do
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid 2>/dev/null || true
            fi
        done
    fi
    
    # 4. 停止占用端口8000的进程
    local port_pid=$(lsof -ti :8000 2>/dev/null || true)
    if [ -n "$port_pid" ]; then
        log_info "停止占用端口8000的进程 (PID: $port_pid)..."
        kill $port_pid 2>/dev/null || true
        sleep 1
        if kill -0 $port_pid 2>/dev/null; then
            kill -9 $port_pid 2>/dev/null || true
        fi
    fi
    
    # 5. 停止前端进程（如果存在）
    if [ -f "data/frontend.pid" ]; then
        local frontend_pid=$(cat data/frontend.pid 2>/dev/null)
        if [ -n "$frontend_pid" ] && kill -0 $frontend_pid 2>/dev/null; then
            log_info "停止前端进程 (PID: $frontend_pid)..."
            kill $frontend_pid 2>/dev/null || true
            rm -f data/frontend.pid
        fi
    fi
    
    # 清理残留进程
    pkill -f "npm.*run.*dev" 2>/dev/null || true
    pkill -f "next.*dev" 2>/dev/null || true
    
    log_success "直接运行的进程已停止"
}

# 停止服务
stop_services() {
    local mode=${1:-production}
    
    log_info "停止服务 (模式: $mode)..."
    
    # 先停止直接运行的进程（非Docker方式）
    stop_direct_processes
    
    # 然后停止Docker容器（如果使用Docker）
    if [ "$mode" = "development" ]; then
        if [ -f docker-compose.dev.yml ]; then
            log_info "停止Docker开发环境..."
            docker-compose -f docker-compose.dev.yml down
        else
            log_warning "开发环境配置文件不存在"
        fi
    else
        if [ -f docker-compose.yml ]; then
            log_info "停止Docker生产环境..."
            docker-compose down
        else
            log_warning "生产环境配置文件不存在"
        fi
    fi
    
    log_success "服务停止完成"
}

# 清理资源
cleanup() {
    local clean_volumes=${1:-false}
    local clean_images=${2:-false}
    
    if [ "$clean_volumes" = "true" ]; then
        log_info "清理数据卷..."
        docker volume prune -f
        log_success "数据卷清理完成"
    fi
    
    if [ "$clean_images" = "true" ]; then
        log_info "清理镜像..."
        docker image prune -f
        log_success "镜像清理完成"
    fi
}

# 显示状态
show_status() {
    echo ""
    log_info "检查残留进程..."
    
    # 检查直接运行的进程
    local remaining_pids=$(pgrep -f "python.*run\.py|uvicorn.*main:app|multiprocessing\.(spawn|resource_tracker)" 2>/dev/null || true)
    if [ -n "$remaining_pids" ]; then
        log_warning "发现残留进程: $remaining_pids"
        log_info "进程详情:"
        ps -p $remaining_pids -o pid,ppid,cmd 2>/dev/null || true
    else
        log_success "没有发现残留进程"
    fi
    
    echo ""
    log_info "当前Docker容器状态："
    docker ps -a --filter "name=stock-prediction" 2>/dev/null || log_warning "Docker未运行或无法访问"
    
    echo ""
    log_info "当前Docker镜像："
    docker images | grep -E "(stock-prediction|<none>)" 2>/dev/null || echo "没有相关镜像"
    
    echo ""
    log_info "当前数据卷："
    docker volume ls | grep -E "(stock-prediction|prometheus|grafana)" 2>/dev/null || echo "没有相关数据卷"
}

# 主函数
main() {
    local mode=${1:-production}
    local clean_volumes=${2:-false}
    local clean_images=${3:-false}
    
    echo "========================================"
    echo "    股票预测平台停止脚本"
    echo "========================================"
    echo ""
    
    stop_services "$mode"
    cleanup "$clean_volumes" "$clean_images"
    show_status
    
    echo ""
    log_success "股票预测平台已停止"
}

# 处理命令行参数
case "${1:-}" in
    "dev"|"development")
        main "development" "${2:-false}" "${3:-false}"
        ;;
    "prod"|"production"|"")
        main "production" "${2:-false}" "${3:-false}"
        ;;
    "clean")
        main "production" "true" "true"
        ;;
    "help"|"-h"|"--help")
        echo "用法: $0 [mode] [clean_volumes] [clean_images]"
        echo ""
        echo "模式:"
        echo "  dev, development  - 停止开发环境"
        echo "  prod, production  - 停止生产环境 (默认)"
        echo "  clean            - 停止服务并清理所有资源"
        echo "  help             - 显示帮助信息"
        echo ""
        echo "参数:"
        echo "  clean_volumes    - 是否清理数据卷 (true/false)"
        echo "  clean_images     - 是否清理镜像 (true/false)"
        ;;
    *)
        log_error "未知模式: $1"
        echo "使用 '$0 help' 查看帮助信息"
        exit 1
        ;;
esac
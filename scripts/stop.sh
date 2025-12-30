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

# 停止服务
stop_services() {
    local mode=${1:-production}
    
    log_info "停止服务 (模式: $mode)..."
    
    if [ "$mode" = "development" ]; then
        if [ -f docker-compose.dev.yml ]; then
            docker-compose -f docker-compose.dev.yml down
        else
            log_warning "开发环境配置文件不存在"
        fi
    else
        if [ -f docker-compose.yml ]; then
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
    log_info "当前Docker容器状态："
    docker ps -a --filter "name=stock-prediction"
    
    echo ""
    log_info "当前Docker镜像："
    docker images | grep -E "(stock-prediction|<none>)" || echo "没有相关镜像"
    
    echo ""
    log_info "当前数据卷："
    docker volume ls | grep -E "(stock-prediction|prometheus|grafana)" || echo "没有相关数据卷"
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
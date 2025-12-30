#!/bin/bash

# 股票预测平台启动脚本
set -e

# 获取脚本所在目录的父目录作为项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

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

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    log_success "Docker环境检查通过"
}

# 创建必要的目录
create_directories() {
    log_info "创建必要的目录..."
    
    mkdir -p data/stocks
    mkdir -p data/models
    mkdir -p data/logs
    mkdir -p backend/logs
    mkdir -p nginx/ssl
    mkdir -p monitoring/grafana/dashboards
    mkdir -p monitoring/grafana/datasources
    
    log_success "目录创建完成"
}

# 复制环境变量文件
setup_env() {
    if [ ! -f .env ]; then
        log_info "复制环境变量配置文件..."
        cp .env.example .env
        log_warning "请编辑 .env 文件配置您的环境变量"
    else
        log_info "环境变量文件已存在"
    fi
}

# 构建镜像
build_images() {
    local mode=${1:-production}
    local minimal=${2:-false}
    
    log_info "构建Docker镜像 (模式: $mode, 最小化: $minimal)..."
    
    # 设置环境变量以避免权限问题
    export COMPOSE_DOCKER_CLI_BUILD=1
    export DOCKER_BUILDKIT=1
    
    # 设置Docker构建参数使用国内源
    export DOCKER_BUILDKIT_INLINE_CACHE=1
    
    if [ "$minimal" = "true" ]; then
        log_info "使用最小化依赖构建..."
        # 临时替换requirements文件
        if [ -f backend/requirements.txt ]; then
            cp backend/requirements.txt backend/requirements.txt.backup
            cp backend/requirements-minimal.txt backend/requirements.txt
        fi
    fi
    
    # 不使用.env文件，直接构建
    sudo docker-compose --env-file /dev/null build --no-cache \
        --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
        --build-arg NPM_REGISTRY=https://registry.npmmirror.com
    
    # 恢复原始requirements文件
    if [ "$minimal" = "true" ] && [ -f backend/requirements.txt.backup ]; then
        mv backend/requirements.txt.backup backend/requirements.txt
    fi
    
    log_success "镜像构建完成"
}

# 启动服务
start_services() {
    local mode=${1:-production}
    
    log_info "启动服务 (模式: $mode)..."
    
    if [ "$mode" = "development" ]; then
        sudo docker-compose --env-file /dev/null -f docker-compose.dev.yml up -d
    else
        sudo docker-compose --env-file /dev/null up -d
    fi
    
    log_success "服务启动完成"
}

# 检查服务状态
check_services() {
    log_info "检查服务状态..."
    
    # 等待服务启动
    sleep 10
    
    # 检查后端健康状态
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "后端服务运行正常"
    else
        log_error "后端服务启动失败"
        return 1
    fi
    
    # 检查前端服务
    if curl -f http://localhost:3000 &> /dev/null; then
        log_success "前端服务运行正常"
    else
        log_error "前端服务启动失败"
        return 1
    fi
    
    log_success "所有服务运行正常"
}

# 显示服务信息
show_info() {
    echo ""
    log_success "股票预测平台启动成功！"
    echo ""
    echo "服务访问地址："
    echo "  前端应用: http://localhost:3000"
    echo "  后端API: http://localhost:8000"
    echo "  API文档: http://localhost:8000/docs"
    echo "  Prometheus: http://localhost:9090 (如果启用)"
    echo "  Grafana: http://localhost:3001 (如果启用)"
    echo ""
    echo "常用命令："
    echo "  查看日志: docker-compose logs -f"
    echo "  停止服务: docker-compose down"
    echo "  重启服务: docker-compose restart"
    echo ""
}

# 主函数
main() {
    local mode=${1:-production}
    local minimal=${2:-false}
    
    echo "========================================"
    echo "    股票预测平台部署脚本"
    echo "========================================"
    echo ""
    
    check_docker
    create_directories
    setup_env
    build_images "$mode" "$minimal"
    start_services "$mode"
    check_services
    show_info
}

# 处理命令行参数
case "${1:-}" in
    "dev"|"development")
        main "development" "${2:-false}"
        ;;
    "prod"|"production"|"")
        main "production" "${2:-false}"
        ;;
    "minimal")
        main "production" "true"
        ;;
    "dev-minimal")
        main "development" "true"
        ;;
    "help"|"-h"|"--help")
        echo "用法: $0 [mode] [options]"
        echo ""
        echo "模式:"
        echo "  dev, development  - 开发模式"
        echo "  prod, production  - 生产模式 (默认)"
        echo "  minimal          - 最小化依赖模式（快速启动）"
        echo "  dev-minimal      - 开发模式 + 最小化依赖"
        echo "  help             - 显示帮助信息"
        echo ""
        echo "示例:"
        echo "  $0                # 生产模式，完整依赖"
        echo "  $0 minimal        # 生产模式，最小化依赖"
        echo "  $0 dev            # 开发模式，完整依赖"
        echo "  $0 dev-minimal    # 开发模式，最小化依赖"
        ;;
    *)
        log_error "未知模式: $1"
        echo "使用 '$0 help' 查看帮助信息"
        exit 1
        ;;
esac
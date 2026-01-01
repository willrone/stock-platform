#!/bin/bash

# 超简单启动脚本 - 无需Docker，直接本地运行
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
echo "    股票预测平台 - 超简单启动"
echo "========================================"
echo ""

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "未找到Python3，请先安装Python 3.9+"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_success "Python版本: $python_version"
    
    if ! command -v pip3 &> /dev/null; then
        log_error "未找到pip3，请先安装pip"
        exit 1
    fi
}

# 检查Node.js环境
check_node() {
    log_info "检查Node.js环境..."
    
    if ! command -v node &> /dev/null; then
        log_warning "未找到Node.js，将跳过前端启动"
        return 1
    fi
    
    node_version=$(node -v)
    log_success "Node.js版本: $node_version"
    
    if ! command -v npm &> /dev/null; then
        log_warning "未找到npm，将跳过前端启动"
        return 1
    fi
    
    return 0
}

# 创建Python虚拟环境
setup_venv() {
    log_info "设置Python虚拟环境..."
    
    cd "$PROJECT_ROOT/backend"
    
    if [ ! -d "venv" ]; then
        log_info "创建虚拟环境..."
        python3 -m venv venv
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
    
    log_success "虚拟环境设置完成"
}

# 安装最小化依赖
install_minimal_deps() {
    log_info "安装最小化Python依赖..."
    
    cd "$PROJECT_ROOT/backend"
    source venv/bin/activate
    
    # 使用最小化依赖文件
    pip install -r requirements-minimal.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
    
    log_success "Python依赖安装完成"
}

# 设置环境变量
setup_env() {
    log_info "设置环境变量..."
    
    cd "$PROJECT_ROOT"
    
    if [ ! -f .env ]; then
        cp .env.example .env
        log_warning "已创建.env文件，请根据需要修改配置"
    fi
    
    cd backend
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
        else
            # 创建基本的.env文件
            cat > .env << EOF
# 数据库配置
DATABASE_URL=sqlite:///./data/app.db

# API配置
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# 日志配置
LOG_LEVEL=INFO
EOF
        fi
        log_warning "已创建backend/.env文件"
    fi
}

# 创建必要目录
create_dirs() {
    log_info "创建必要目录..."
    
    cd "$PROJECT_ROOT"
    mkdir -p data/stocks
    mkdir -p data/models
    mkdir -p data/logs
    mkdir -p backend/data
    mkdir -p backend/logs
    
    log_success "目录创建完成"
}

# 启动后端服务
start_backend() {
    log_info "启动后端服务..."
    
    cd "$PROJECT_ROOT/backend"
    source venv/bin/activate
    
    # 后台启动后端服务
    # 注意：不要重定向stdout，让loguru自己管理日志文件
    # 如果需要查看实时日志，可以使用: tail -f ../data/logs/app.log
    nohup python run.py > /dev/null 2>&1 &
    backend_pid=$!
    echo $backend_pid > ../data/backend.pid
    
    # 等待服务启动
    sleep 5
    
    # 检查服务是否启动成功
    if curl -f http://localhost:8000/api/v1/health &> /dev/null; then
        log_success "后端服务启动成功 (PID: $backend_pid)"
        log_info "API文档: http://localhost:8000/api/v1/docs"
    else
        log_error "后端服务启动失败，请查看日志: data/logs/backend.log"
        return 1
    fi
}

# 安装前端依赖
install_frontend_deps() {
    log_info "安装前端依赖..."
    
    cd "$PROJECT_ROOT/frontend"
    
    # 配置npm使用国内源
    npm config set registry https://registry.npmmirror.com
    
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    
    # 设置前端环境变量
    if [ ! -f .env.local ]; then
        if [ -f .env.example ]; then
            cp .env.example .env.local
        else
            cat > .env.local << EOF
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
EOF
        fi
        log_warning "已创建frontend/.env.local文件"
    fi
    
    log_success "前端依赖安装完成"
}

# 启动前端服务
start_frontend() {
    log_info "启动前端服务..."
    
    cd "$PROJECT_ROOT/frontend"
    
    # 后台启动前端服务
    nohup npm run dev > ../data/logs/frontend.log 2>&1 &
    frontend_pid=$!
    echo $frontend_pid > ../data/frontend.pid
    
    # 等待服务启动
    sleep 10
    
    # 检查服务是否启动成功
    if curl -f http://localhost:3000 &> /dev/null; then
        log_success "前端服务启动成功 (PID: $frontend_pid)"
        log_info "前端应用: http://localhost:3000"
    else
        log_warning "前端服务启动可能失败，请查看日志: data/logs/frontend.log"
    fi
}

# 显示服务信息
show_info() {
    echo ""
    log_success "股票预测平台启动完成！"
    echo ""
    echo "🌐 服务访问地址："
    echo "  前端应用: http://localhost:3000"
    echo "  后端API: http://localhost:8000"
    echo "  API文档: http://localhost:8000/api/v1/docs"
    echo ""
    echo "📋 常用命令："
    echo "  查看后端日志: tail -f data/logs/backend.log"
    echo "  查看前端日志: tail -f data/logs/frontend.log"
    echo "  停止服务: ./scripts/stop-simple.sh"
    echo ""
    echo "📁 重要文件："
    echo "  后端进程ID: data/backend.pid"
    echo "  前端进程ID: data/frontend.pid"
    echo "  数据库文件: backend/data/app.db"
    echo ""
}

# 主函数
main() {
    local skip_frontend=${1:-false}
    
    check_python
    setup_venv
    install_minimal_deps
    setup_env
    create_dirs
    start_backend
    
    if [ "$skip_frontend" != "true" ] && check_node; then
        install_frontend_deps
        start_frontend
    else
        log_warning "跳过前端启动，仅运行后端服务"
    fi
    
    show_info
}

# 处理命令行参数
case "${1:-}" in
    "backend-only")
        main true
        ;;
    "help"|"-h"|"--help")
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  backend-only  - 仅启动后端服务"
        echo "  help          - 显示帮助信息"
        echo ""
        echo "默认启动前端和后端服务"
        ;;
    *)
        main false
        ;;
esac
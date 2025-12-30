#!/bin/bash

# Python依赖快速安装脚本（使用国内源）
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

# 配置pip国内源
configure_pip() {
    log_info "配置pip使用国内源..."
    
    # 创建pip配置目录
    mkdir -p ~/.pip
    
    # 配置pip使用清华源
    cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120

[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
    
    log_success "pip国内源配置完成"
}

# 升级pip
upgrade_pip() {
    log_info "升级pip..."
    python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/
    log_success "pip升级完成"
}

# 安装后端依赖
install_backend_deps() {
    log_info "安装后端Python依赖..."
    
    cd "$PROJECT_ROOT/backend"
    
    # 选择要安装的requirements文件
    local req_file=${1:-requirements.txt}
    
    if [ ! -f "$req_file" ]; then
        log_error "依赖文件 $req_file 不存在"
        return 1
    fi
    
    log_info "使用依赖文件: $req_file"
    
    # 使用国内源安装依赖
    pip install -r "$req_file" -i https://pypi.tuna.tsinghua.edu.cn/simple/
    
    log_success "后端依赖安装完成"
}

# 安装数据服务依赖
install_data_service_deps() {
    log_info "安装数据服务Python依赖..."
    
    cd "$PROJECT_ROOT/back_test_data_service"
    
    if [ ! -f "requirements.txt" ]; then
        log_error "数据服务依赖文件不存在"
        return 1
    fi
    
    # 使用国内源安装依赖
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
    
    log_success "数据服务依赖安装完成"
}

# 配置npm国内源
configure_npm() {
    log_info "配置npm使用国内源..."
    
    # 配置npm使用淘宝镜像
    npm config set registry https://registry.npmmirror.com
    
    log_success "npm国内源配置完成"
}

# 安装前端依赖
install_frontend_deps() {
    log_info "安装前端Node.js依赖..."
    
    cd "$PROJECT_ROOT/frontend"
    
    if [ ! -f "package.json" ]; then
        log_error "前端package.json文件不存在"
        return 1
    fi
    
    # 使用国内源安装依赖
    npm install --registry=https://registry.npmmirror.com
    
    log_success "前端依赖安装完成"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  backend [file]    - 安装后端Python依赖 (默认: requirements.txt)"
    echo "                      可选文件: requirements-minimal.txt, requirements-test.txt"
    echo "  data-service      - 安装数据服务Python依赖"
    echo "  frontend          - 安装前端Node.js依赖"
    echo "  all               - 安装所有依赖"
    echo "  config            - 仅配置国内源"
    echo "  help              - 显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 backend                    # 安装后端完整依赖"
    echo "  $0 backend requirements-test.txt  # 安装后端测试依赖"
    echo "  $0 all                        # 安装所有依赖"
}

# 主函数
main() {
    local action=${1:-all}
    
    echo "========================================"
    echo "    Python/Node.js 依赖快速安装"
    echo "========================================"
    echo ""
    
    case "$action" in
        "backend")
            configure_pip
            upgrade_pip
            install_backend_deps "${2:-requirements.txt}"
            ;;
        "data-service")
            configure_pip
            upgrade_pip
            install_data_service_deps
            ;;
        "frontend")
            configure_npm
            install_frontend_deps
            ;;
        "all")
            configure_pip
            configure_npm
            upgrade_pip
            install_backend_deps
            install_data_service_deps
            install_frontend_deps
            ;;
        "config")
            configure_pip
            configure_npm
            log_success "国内源配置完成"
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log_error "未知选项: $action"
            show_help
            exit 1
            ;;
    esac
    
    log_success "操作完成！"
}

# 执行主函数
main "$@"
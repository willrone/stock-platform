#!/bin/bash

# TA-Lib 安装脚本
# 用于安装TA-Lib C库和Python包

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# 检查是否已安装
check_talib_installed() {
    if python3 -c "import talib" 2>/dev/null; then
        log_success "TA-Lib Python包已安装"
        python3 -c "import talib; print(f'TA-Lib版本: {talib.__version__}')" 2>/dev/null || echo "无法获取版本信息"
        return 0
    fi
    return 1
}

# 安装依赖
install_dependencies() {
    log_info "安装编译依赖..."
    
    # 检查是否有sudo权限
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
    else
        SUDO=""
    fi
    
    # 安装必要的编译工具和库
    $SUDO apt-get update
    $SUDO apt-get install -y \
        wget \
        build-essential \
        gcc \
        g++ \
        make \
        cmake \
        libtool \
        autoconf \
        automake \
        pkg-config
    
    log_success "依赖安装完成"
}

# 安装TA-Lib C库
install_talib_c_library() {
    log_info "安装TA-Lib C库..."
    
    # 创建临时目录
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # 下载TA-Lib源码
    log_info "下载TA-Lib源码..."
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    
    # 解压
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    
    # 配置、编译和安装
    log_info "编译TA-Lib C库（这可能需要几分钟）..."
    ./configure --prefix=/usr
    make
    
    # 检查是否有sudo权限
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
    else
        SUDO=""
        log_warning "没有sudo权限，尝试使用用户权限安装..."
    fi
    
    $SUDO make install
    
    # 更新动态链接库缓存
    if [ -n "$SUDO" ]; then
        $SUDO ldconfig
    fi
    
    # 清理临时文件
    cd /
    rm -rf "$TEMP_DIR"
    
    log_success "TA-Lib C库安装完成"
}

# 安装Python包
install_talib_python() {
    log_info "安装TA-Lib Python包..."
    
    # 切换到后端目录
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$PROJECT_ROOT/backend"
    
    # 激活虚拟环境
    if [ -d "venv" ]; then
        source venv/bin/activate
        log_info "已激活虚拟环境"
    else
        log_warning "未找到虚拟环境，将安装到系统Python"
    fi
    
    # 安装Python包
    pip install TA-Lib
    
    log_success "TA-Lib Python包安装完成"
}

# 验证安装
verify_installation() {
    log_info "验证TA-Lib安装..."
    
    if python3 -c "import talib; print('TA-Lib安装成功！'); print(f'版本: {talib.__version__}')" 2>/dev/null; then
        log_success "TA-Lib安装验证成功！"
        return 0
    else
        log_error "TA-Lib安装验证失败"
        return 1
    fi
}

# 主函数
main() {
    echo "========================================"
    echo "    TA-Lib 安装脚本"
    echo "========================================"
    echo ""
    
    # 检查是否已安装
    if check_talib_installed; then
        log_warning "TA-Lib似乎已经安装，是否要重新安装？(y/N)"
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            log_info "取消安装"
            exit 0
        fi
    fi
    
    # 安装步骤
    install_dependencies
    install_talib_c_library
    install_talib_python
    
    # 验证
    if verify_installation; then
        echo ""
        log_success "TA-Lib安装完成！"
        echo ""
        echo "现在可以取消注释 requirements.txt 中的 ta-lib 依赖"
        echo "或者直接使用，代码会自动检测并使用TA-Lib"
        exit 0
    else
        log_error "安装验证失败，请检查错误信息"
        exit 1
    fi
}

# 运行主函数
main "$@"


#!/bin/bash

# 测试国内源配置脚本
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

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 测试pip源
test_pip_source() {
    log_info "测试pip国内源配置..."
    
    # 检查pip配置
    if pip config list | grep -q "tuna.tsinghua.edu.cn"; then
        log_success "pip国内源配置正确"
    else
        log_error "pip国内源配置失败"
        return 1
    fi
    
    # 测试安装一个小包
    log_info "测试安装requests包..."
    if pip install requests==2.31.0 --dry-run > /dev/null 2>&1; then
        log_success "pip源连接正常"
    else
        log_error "pip源连接失败"
        return 1
    fi
}

# 测试npm源
test_npm_source() {
    log_info "测试npm国内源配置..."
    
    # 检查npm配置
    if npm config get registry | grep -q "npmmirror.com"; then
        log_success "npm国内源配置正确"
    else
        log_error "npm国内源配置失败"
        return 1
    fi
    
    # 测试npm源连接
    log_info "测试npm源连接..."
    if npm ping --registry=https://registry.npmmirror.com > /dev/null 2>&1; then
        log_success "npm源连接正常"
    else
        log_error "npm源连接失败"
        return 1
    fi
}

# 显示当前配置
show_config() {
    echo ""
    log_info "当前配置信息："
    echo ""
    echo "pip配置："
    pip config list 2>/dev/null || echo "  未找到pip配置"
    echo ""
    echo "npm配置："
    echo "  registry: $(npm config get registry 2>/dev/null || echo '未配置')"
    echo ""
}

# 主函数
main() {
    echo "========================================"
    echo "    国内源配置测试"
    echo "========================================"
    echo ""
    
    show_config
    
    # 测试pip
    if command -v pip &> /dev/null; then
        test_pip_source
    else
        log_error "pip未安装"
    fi
    
    echo ""
    
    # 测试npm
    if command -v npm &> /dev/null; then
        test_npm_source
    else
        log_error "npm未安装"
    fi
    
    echo ""
    log_success "测试完成！"
}

# 执行主函数
main "$@"
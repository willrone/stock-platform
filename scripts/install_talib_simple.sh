#!/bin/bash

# TA-Lib 简化安装脚本（用户目录安装，不需要sudo）

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 检查是否已安装
if python3 -c "import talib" 2>/dev/null; then
    log_success "TA-Lib已经安装！"
    python3 -c "import talib; print(f'版本: {talib.__version__}')" 2>/dev/null
    exit 0
fi

log_info "开始安装TA-Lib..."

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_PREFIX="$HOME/.local"

# 创建临时目录
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

log_info "步骤1/3: 下载TA-Lib源码..."
wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz || {
    log_warning "下载失败，尝试备用链接..."
    wget -q https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz || {
        echo "下载失败，请检查网络连接"
        exit 1
    }
}

log_info "步骤2/3: 编译安装TA-Lib C库（这可能需要几分钟）..."
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/

# 配置并安装到用户目录
./configure --prefix="$INSTALL_PREFIX"
make -j$(nproc)
make install

# 设置环境变量
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# 添加到.bashrc（如果还没有）
if ! grep -q "LD_LIBRARY_PATH.*\.local/lib" "$HOME/.bashrc" 2>/dev/null; then
    echo "" >> "$HOME/.bashrc"
    echo "# TA-Lib库路径" >> "$HOME/.bashrc"
    echo "export LD_LIBRARY_PATH=\"\$HOME/.local/lib:\$LD_LIBRARY_PATH\"" >> "$HOME/.bashrc"
    echo "export PKG_CONFIG_PATH=\"\$HOME/.local/lib/pkgconfig:\$PKG_CONFIG_PATH\"" >> "$HOME/.bashrc"
    log_info "已将环境变量添加到 ~/.bashrc"
fi

# 清理临时文件
cd /
rm -rf "$TEMP_DIR"

log_info "步骤3/3: 安装Python包..."
cd "$PROJECT_ROOT/backend"

# 激活虚拟环境
if [ -d "venv" ]; then
    source venv/bin/activate
    log_info "已激活虚拟环境"
fi

# 设置环境变量
export LD_LIBRARY_PATH="$INSTALL_PREFIX/lib:$LD_LIBRARY_PATH"
export PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# 安装Python包
pip install TA-Lib

# 验证安装
log_info "验证安装..."
if python -c "import talib; print('TA-Lib安装成功！版本:', talib.__version__)" 2>/dev/null; then
    log_success "TA-Lib安装完成！"
    echo ""
    echo "注意：如果重新打开终端，请运行以下命令或重新加载.bashrc："
    echo "  source ~/.bashrc"
    echo "  或者"
    echo "  export LD_LIBRARY_PATH=\"\$HOME/.local/lib:\$LD_LIBRARY_PATH\""
    exit 0
else
    log_warning "安装可能不完整，请检查错误信息"
    exit 1
fi


#!/bin/bash

# 快速启动脚本 - 跳过重型依赖，快速启动基本服务
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

echo "========================================"
echo "    股票预测平台快速启动脚本"
echo "========================================"
echo ""

log_info "使用最小化依赖快速启动服务..."

# 检查是否已有镜像
if sudo docker images | grep -q "stock-prediction-platform"; then
    log_info "发现已存在的镜像，直接启动服务..."
    sudo docker-compose --env-file /dev/null up -d
else
    log_info "未发现镜像，使用最小化依赖构建..."
    ./scripts/start.sh minimal
fi

log_success "快速启动完成！"
echo ""
echo "服务访问地址："
echo "  前端应用: http://localhost:3000"
echo "  后端API: http://localhost:8000"
echo "  API文档: http://localhost:8000/docs"
echo ""
echo "注意：此为快速启动模式，部分高级功能可能不可用"
echo "如需完整功能，请运行: ./scripts/start.sh"
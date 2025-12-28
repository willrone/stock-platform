#!/bin/bash

echo "🔧 数据服务环境初始化脚本"
echo "==========================="

# 检查是否在正确的目录中
if [ ! -f "requirements.txt" ] || [ ! -d "data_service" ]; then
    echo "❌ 请在数据服务目录中运行此脚本"
    echo "   cd back_test_data_service"
    echo "   ./setup_data_service.sh"
    exit 1
fi

echo "📁 当前目录: $(pwd)"

# 检查Python
echo "🐍 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到python3，请安装Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python版本: $PYTHON_VERSION"

# 创建虚拟环境
echo "🔨 创建虚拟环境..."
if [ -d ".venv" ]; then
    echo "⚠️  虚拟环境已存在，跳过创建"
else
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "❌ 创建虚拟环境失败"
        exit 1
    fi
    echo "✅ 虚拟环境创建成功"
fi

# 激活虚拟环境并安装依赖
echo "📦 激活虚拟环境并安装依赖..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ 依赖安装成功"
else
    echo "❌ 依赖安装失败"
    exit 1
fi

echo ""
echo "🎉 数据服务环境初始化完成！"
echo ""
echo "💡 后续步骤:"
echo "   1. 配置环境变量 (TUSHARE_TOKEN, MYSQL_*, REDIS_*)"
echo "   2. 运行 ./start.sh 启动数据服务"
echo ""
echo "📖 详细说明请查看 README.md"

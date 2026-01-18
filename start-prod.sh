#!/bin/bash

# 一键启动脚本 - 生产环境
# 使用 Docker Compose 启动生产环境

echo "🚀 启动股票预测平台（生产环境）..."
echo ""

# 检查是否在项目根目录
if [ ! -f "scripts/start.sh" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 执行生产环境启动脚本
./scripts/start.sh production "$@"

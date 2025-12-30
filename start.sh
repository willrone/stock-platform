#!/bin/bash

# 一键启动脚本 - 放在项目根目录
# 这是最简单的启动方式，无需Docker

echo "🚀 启动股票预测平台..."
echo ""

# 检查是否在项目根目录
if [ ! -f "scripts/simple-start.sh" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 执行简单启动脚本
./scripts/simple-start.sh "$@"
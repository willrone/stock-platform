#!/bin/bash

# 一键停止脚本 - 放在项目根目录

echo "🛑 停止股票预测平台..."
echo ""

# 检查是否在项目根目录
if [ ! -f "scripts/stop-simple.sh" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    exit 1
fi

# 执行停止脚本
./scripts/stop-simple.sh
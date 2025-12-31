#!/bin/bash

# 远程数据服务安全重启脚本
# 用于重启远程数据服务以加载新的API端点

set -e  # 遇到错误立即退出

# 配置
REMOTE_HOST="192.168.3.62"
REMOTE_PORT="5002"
SERVICE_PATH="/home/willrone/stock-prediction-platform/back_test_data_service"
MAX_WAIT_TIME=30  # 最大等待时间（秒）

echo "🔄 开始远程数据服务重启流程..."

# 函数：检查服务状态
check_service_status() {
    local host=$1
    local port=$2
    
    if curl -s --connect-timeout 5 "http://${host}:${port}/api/data/health" > /dev/null 2>&1; then
        return 0  # 服务正常
    else
        return 1  # 服务不可用
    fi
}

# 函数：检查新API端点
check_new_endpoint() {
    local host=$1
    local port=$2
    
    # 测试新的股票数据API端点
    local test_url="http://${host}:${port}/api/data/stock/000001.SZ/daily?start_date=2024-12-01&end_date=2024-12-31"
    
    if curl -s --connect-timeout 5 "$test_url" | grep -q '"success"'; then
        return 0  # 新端点可用
    else
        return 1  # 新端点不可用
    fi
}

# 步骤1: 检查当前服务状态
echo "📋 步骤1: 检查当前服务状态..."
if check_service_status "$REMOTE_HOST" "$REMOTE_PORT"; then
    echo "✅ 远程数据服务当前正在运行"
    
    # 检查新端点是否已经可用
    if check_new_endpoint "$REMOTE_HOST" "$REMOTE_PORT"; then
        echo "🎉 新API端点已经可用，无需重启！"
        exit 0
    else
        echo "⚠️  新API端点不可用，需要重启服务"
    fi
else
    echo "❌ 远程数据服务当前不可用"
    echo "请手动检查服务状态或联系系统管理员"
    exit 1
fi

# 步骤2: 准备重启（这里只能提供指导，无法直接操作远程服务器）
echo ""
echo "📋 步骤2: 远程服务重启指导"
echo "由于安全限制，无法直接操作远程服务器。"
echo "请按照以下步骤手动重启远程数据服务："
echo ""
echo "1. 登录到远程服务器 ${REMOTE_HOST}"
echo "2. 进入服务目录: cd ${SERVICE_PATH}"
echo "3. 停止当前服务（如果有运行的进程）:"
echo "   pkill -f 'run_data_api.py' || true"
echo "   pkill -f 'run_data_service.py' || true"
echo "4. 重新启动服务:"
echo "   ./start.sh api"
echo ""

# 步骤3: 等待用户操作并验证
echo "📋 步骤3: 等待服务重启..."
echo "请在另一个终端执行上述重启命令，然后按回车键继续验证..."
read -p "按回车键继续验证服务状态..."

# 步骤4: 验证服务重启结果
echo ""
echo "📋 步骤4: 验证服务重启结果..."

wait_count=0
while [ $wait_count -lt $MAX_WAIT_TIME ]; do
    echo "🔍 检查服务状态... (${wait_count}/${MAX_WAIT_TIME})"
    
    if check_service_status "$REMOTE_HOST" "$REMOTE_PORT"; then
        echo "✅ 远程数据服务已恢复运行"
        
        # 检查新API端点
        if check_new_endpoint "$REMOTE_HOST" "$REMOTE_PORT"; then
            echo "🎉 新API端点验证成功！"
            echo ""
            echo "📊 测试新端点响应:"
            curl -s "http://${REMOTE_HOST}:${REMOTE_PORT}/api/data/stock/000001.SZ/daily?start_date=2024-12-01&end_date=2024-12-31" | head -200
            echo ""
            echo "✅ 远程数据服务重启完成，新API端点已可用"
            exit 0
        else
            echo "⚠️  服务已启动但新API端点仍不可用"
        fi
    else
        echo "⏳ 服务尚未启动，继续等待..."
    fi
    
    sleep 2
    wait_count=$((wait_count + 2))
done

echo "❌ 等待超时，服务可能启动失败"
echo "请检查远程服务器的日志文件："
echo "  - ${SERVICE_PATH}/logs/data_api.log"
echo "  - ${SERVICE_PATH}/logs/data_service.log"
exit 1
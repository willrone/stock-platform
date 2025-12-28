#!/bin/bash

# MySQL连接测试脚本
# 在App服务器上执行此脚本测试连接

echo "=========================================="
echo "MySQL远程连接测试脚本"
echo "=========================================="
echo ""

# 从环境变量读取配置
MYSQL_HOST="${REMOTE_MYSQL_HOST:-localhost}"
MYSQL_PORT="${REMOTE_MYSQL_PORT:-3306}"
MYSQL_USER="${REMOTE_MYSQL_USER:-stock_user}"
MYSQL_DATABASE="${REMOTE_MYSQL_DATABASE:-stock_data}"

echo "连接配置："
echo "  Host: $MYSQL_HOST"
echo "  Port: $MYSQL_PORT"
echo "  User: $MYSQL_USER"
echo "  Database: $MYSQL_DATABASE"
echo ""

# 检查MySQL客户端
if ! command -v mysql &> /dev/null; then
    echo "❌ MySQL客户端未安装"
    echo "安装方法："
    echo "  macOS: brew install mysql-client"
    echo "  Ubuntu: sudo apt-get install mysql-client"
    exit 1
fi

# 测试连接
echo "测试连接..."
mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p "$MYSQL_DATABASE" -e "SELECT 1 as connection_test;" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ MySQL连接成功！"
    echo ""
    echo "测试查询股票数据..."
    mysql -h "$MYSQL_HOST" -P "$MYSQL_PORT" -u "$MYSQL_USER" -p "$MYSQL_DATABASE" -e "SELECT COUNT(*) as total_stocks FROM stock_list; SELECT COUNT(*) as total_records FROM stock_data LIMIT 1;" 2>&1
else
    echo ""
    echo "❌ MySQL连接失败"
    echo ""
    echo "排查步骤："
    echo "1. 检查网络连接: ping $MYSQL_HOST"
    echo "2. 检查端口是否开放: telnet $MYSQL_HOST $MYSQL_PORT"
    echo "3. 检查MySQL用户权限"
    echo "4. 检查MySQL bind-address配置"
    echo "5. 检查防火墙设置"
    exit 1
fi


#!/bin/bash

# MySQL远程连接配置脚本
# 在数据服务器上执行此脚本

echo "=========================================="
echo "MySQL远程连接配置脚本"
echo "=========================================="
echo ""

# 检查MySQL是否运行
if ! command -v mysql &> /dev/null; then
    echo "❌ MySQL未安装或未在PATH中"
    exit 1
fi

echo "📋 配置步骤："
echo "1. 配置MySQL允许远程连接"
echo "2. 创建远程访问用户"
echo "3. 授予权限"
echo "4. 检查防火墙"
echo ""

# 获取App服务器IP
read -p "请输入App服务器IP地址（例如：192.168.3.89）: " APP_SERVER_IP

if [ -z "$APP_SERVER_IP" ]; then
    echo "❌ IP地址不能为空"
    exit 1
fi

echo ""
echo "🔧 步骤1: 配置MySQL允许远程连接"
echo "----------------------------------------"

# 查找MySQL配置文件
MYSQL_CONF=""
if [ -f "/etc/mysql/mysql.conf.d/mysqld.cnf" ]; then
    MYSQL_CONF="/etc/mysql/mysql.conf.d/mysqld.cnf"
elif [ -f "/etc/my.cnf" ]; then
    MYSQL_CONF="/etc/my.cnf"
elif [ -f "/usr/local/etc/my.cnf" ]; then
    MYSQL_CONF="/usr/local/etc/my.cnf"
elif [ -f "/opt/homebrew/etc/my.cnf" ]; then
    MYSQL_CONF="/opt/homebrew/etc/my.cnf"
fi

if [ -n "$MYSQL_CONF" ]; then
    echo "找到MySQL配置文件: $MYSQL_CONF"
    
    # 检查bind-address配置
    if grep -q "^bind-address" "$MYSQL_CONF"; then
        echo "当前bind-address配置:"
        grep "^bind-address" "$MYSQL_CONF"
        echo ""
        read -p "是否修改bind-address为0.0.0.0以允许远程连接？(y/n): " MODIFY_BIND
        if [ "$MODIFY_BIND" = "y" ]; then
            # 备份配置文件
            sudo cp "$MYSQL_CONF" "${MYSQL_CONF}.backup.$(date +%Y%m%d_%H%M%S)"
            echo "已备份配置文件"
            
            # 修改bind-address
            sudo sed -i '' "s/^bind-address.*/bind-address = 0.0.0.0/" "$MYSQL_CONF" 2>/dev/null || \
            sudo sed -i "s/^bind-address.*/bind-address = 0.0.0.0/" "$MYSQL_CONF"
            
            echo "✅ 已修改bind-address为0.0.0.0"
            echo "⚠️  需要重启MySQL服务才能生效"
        fi
    else
        echo "未找到bind-address配置，将添加..."
        read -p "是否添加bind-address = 0.0.0.0配置？(y/n): " ADD_BIND
        if [ "$ADD_BIND" = "y" ]; then
            echo "bind-address = 0.0.0.0" | sudo tee -a "$MYSQL_CONF" > /dev/null
            echo "✅ 已添加bind-address配置"
        fi
    fi
else
    echo "⚠️  未找到MySQL配置文件，请手动配置bind-address = 0.0.0.0"
fi

echo ""
echo "🔧 步骤2: 创建MySQL远程访问用户"
echo "----------------------------------------"

# 生成SQL脚本
SQL_FILE="/tmp/setup_mysql_remote_$$.sql"
cat > "$SQL_FILE" << EOF
-- 创建用户（允许从指定IP连接）
CREATE USER IF NOT EXISTS 'stock_user'@'$APP_SERVER_IP' IDENTIFIED BY 'stock_password_2024';

-- 授予权限
GRANT ALL PRIVILEGES ON stock_data.* TO 'stock_user'@'$APP_SERVER_IP';

-- 刷新权限
FLUSH PRIVILEGES;

-- 显示用户信息
SELECT User, Host FROM mysql.user WHERE User = 'stock_user';
EOF

echo "生成的SQL脚本:"
cat "$SQL_FILE"
echo ""

read -p "是否执行SQL脚本创建用户？(y/n): " EXECUTE_SQL
if [ "$EXECUTE_SQL" = "y" ]; then
    echo "请输入MySQL root密码:"
    mysql -u root -p < "$SQL_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✅ MySQL用户创建成功"
    else
        echo "❌ MySQL用户创建失败"
        rm -f "$SQL_FILE"
        exit 1
    fi
else
    echo "跳过SQL执行，请手动执行SQL脚本"
    echo "SQL文件位置: $SQL_FILE"
fi

rm -f "$SQL_FILE"

echo ""
echo "🔧 步骤3: 检查防火墙"
echo "----------------------------------------"

# 检查3306端口是否开放
if command -v ufw &> /dev/null; then
    echo "检测到ufw防火墙"
    if sudo ufw status | grep -q "3306"; then
        echo "✅ 3306端口已配置"
    else
        read -p "是否开放3306端口？(y/n): " OPEN_PORT
        if [ "$OPEN_PORT" = "y" ]; then
            sudo ufw allow 3306/tcp
            echo "✅ 已开放3306端口"
        fi
    fi
elif command -v firewall-cmd &> /dev/null; then
    echo "检测到firewalld防火墙"
    if sudo firewall-cmd --list-ports | grep -q "3306"; then
        echo "✅ 3306端口已配置"
    else
        read -p "是否开放3306端口？(y/n): " OPEN_PORT
        if [ "$OPEN_PORT" = "y" ]; then
            sudo firewall-cmd --permanent --add-port=3306/tcp
            sudo firewall-cmd --reload
            echo "✅ 已开放3306端口"
        fi
    fi
else
    echo "⚠️  未检测到常见防火墙，请手动检查3306端口是否开放"
fi

echo ""
echo "=========================================="
echo "配置完成！"
echo "=========================================="
echo ""
echo "📝 配置摘要："
echo "  - MySQL用户: stock_user"
echo "  - 允许IP: $APP_SERVER_IP"
echo "  - 密码: stock_password_2024"
echo "  - 数据库: stock_data"
echo ""
echo "⚠️  重要提示："
echo "1. 如果修改了bind-address，请重启MySQL服务："
echo "   macOS: brew services restart mysql"
echo "   Linux: sudo systemctl restart mysql"
echo ""
echo "2. 在App服务器上配置环境变量："
echo "   export REMOTE_MYSQL_HOST=\"数据服务器IP\""
echo "   export REMOTE_MYSQL_USER=\"stock_user\""
echo "   export REMOTE_MYSQL_PASSWORD=\"stock_password_2024\""
echo "   export REMOTE_MYSQL_DATABASE=\"stock_data\""
echo ""
echo "3. 测试连接："
echo "   mysql -h 数据服务器IP -u stock_user -p"


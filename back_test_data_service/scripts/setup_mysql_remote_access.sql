-- MySQL远程连接配置脚本
-- 在数据服务器上执行此脚本以允许App服务器远程连接

-- 1. 创建允许远程连接的用户（推荐方式）
-- 方式A：允许从任何IP连接（适合局域网环境）
CREATE USER IF NOT EXISTS 'stock_user'@'%' IDENTIFIED BY 'stock_password_2024';

-- 方式B：只允许从特定IP连接（更安全，替换为实际的App服务器IP）
-- CREATE USER IF NOT EXISTS 'stock_user'@'192.168.3.89' IDENTIFIED BY 'stock_password_2024';

-- 2. 授予权限
GRANT ALL PRIVILEGES ON stock_data.* TO 'stock_user'@'%';
-- 如果使用方式B，使用：
-- GRANT ALL PRIVILEGES ON stock_data.* TO 'stock_user'@'192.168.3.89';

-- 3. 刷新权限
FLUSH PRIVILEGES;

-- 4. 验证用户创建
SELECT User, Host FROM mysql.user WHERE User = 'stock_user';

-- 5. 查看权限
SHOW GRANTS FOR 'stock_user'@'%';


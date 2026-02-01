# MLOps配置和部署指南

## 概述

本指南详细介绍如何配置和部署股票预测平台的MLOps功能，包括环境准备、系统配置、部署流程和运维管理。

## 目录

1. [环境准备](#环境准备)
2. [系统要求](#系统要求)
3. [安装依赖](#安装依赖)
4. [配置系统](#配置系统)
5. [部署流程](#部署流程)
6. [运维管理](#运维管理)
7. [监控配置](#监控配置)
8. [安全配置](#安全配置)
9. [性能调优](#性能调优)
10. [备份恢复](#备份恢复)

## 环境准备

### 操作系统要求

**推荐系统**:
- Ubuntu 20.04 LTS 或更高版本
- CentOS 8 或更高版本
- macOS 10.15 或更高版本

**最低要求**:
- 64位操作系统
- 支持Python 3.8+
- 支持Docker (可选)

### 硬件要求

#### 最低配置
- **CPU**: 2核心
- **内存**: 4GB RAM
- **存储**: 20GB可用空间
- **网络**: 稳定的互联网连接

#### 推荐配置
- **CPU**: 4核心或更多
- **内存**: 8GB RAM或更多
- **存储**: 50GB SSD存储
- **网络**: 高速互联网连接

#### 生产环境配置
- **CPU**: 8核心或更多
- **内存**: 16GB RAM或更多
- **存储**: 100GB SSD存储
- **网络**: 专用网络连接
- **备份**: 独立备份存储

## 系统要求

### 必需软件

```bash
# Python 3.8+
python3 --version

# pip包管理器
pip3 --version

# Git版本控制
git --version

# curl网络工具
curl --version
```

### 可选软件

```bash
# Docker容器化 (推荐)
docker --version
docker-compose --version

# Node.js (前端开发)
node --version
npm --version

# Redis缓存 (性能优化)
redis-server --version

# PostgreSQL数据库 (生产环境推荐)
psql --version
```

## 安装依赖

### 1. 系统依赖安装

#### Ubuntu/Debian

```bash
# 更新包列表
sudo apt update

# 安装基础依赖
sudo apt install -y python3 python3-pip python3-venv git curl

# 安装编译工具 (某些Python包需要)
sudo apt install -y build-essential python3-dev

# 安装数据库依赖
sudo apt install -y sqlite3 libsqlite3-dev

# 可选: 安装PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# 可选: 安装Redis
sudo apt install -y redis-server
```

#### CentOS/RHEL

```bash
# 安装EPEL仓库
sudo yum install -y epel-release

# 安装基础依赖
sudo yum install -y python3 python3-pip git curl

# 安装编译工具
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3-devel

# 可选: 安装PostgreSQL
sudo yum install -y postgresql postgresql-server postgresql-contrib

# 可选: 安装Redis
sudo yum install -y redis
```

#### macOS

```bash
# 使用Homebrew安装
brew install python3 git curl

# 可选依赖
brew install postgresql redis
```

### 2. Python依赖安装

```bash
# 创建虚拟环境
python3 -m venv backend/venv

# 激活虚拟环境
source backend/venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装基础依赖
pip install -r backend/requirements.txt

# 安装MLOps特定依赖
pip install qlib shap scikit-learn lightgbm xgboost optuna mlflow psutil pyyaml
```

### 3. 前端依赖安装 (可选)

```bash
# 进入前端目录
cd frontend

# 安装Node.js依赖
npm install

# 或使用yarn
yarn install

# 返回根目录
cd ..
```

## 配置系统

### 1. 环境变量配置

创建 `.env` 文件：

```bash
# 复制示例配置
cp .env.example .env

# 编辑配置文件
nano .env
```

**基础配置**:
```bash
# 应用配置
APP_NAME="股票预测平台"
APP_VERSION="1.0.0"
DEBUG=false

# 数据库配置
DATABASE_URL="sqlite:///./backend/data/app.db"
# 生产环境使用PostgreSQL
# DATABASE_URL="postgresql://user:password@localhost:5432/stock_prediction"

# MLOps配置
MLOPS_ENABLED=true
QLIB_CACHE_DIR="backend/data/qlib_cache"
FEATURE_CACHE_ENABLED=true
MONITORING_ENABLED=true
AB_TESTING_ENABLED=true

# 缓存配置
REDIS_URL="redis://localhost:6379/0"
CACHE_TTL=3600

# 日志配置
LOG_LEVEL="INFO"
LOG_FILE="backend/logs/mlops.log"

# 安全配置
SECRET_KEY="your-secret-key-here"
API_RATE_LIMIT=100

# 外部服务配置
EXTERNAL_API_TIMEOUT=30
EXTERNAL_API_RETRIES=3
```

### 2. MLOps配置文件

编辑 `backend/config/mlops_config.yaml`：

```yaml
# 特征工程配置
feature_engineering:
  technical_indicators:
    enabled: true
    batch_size: 1000
    parallel_workers: 4
    cache_ttl: 3600

# Qlib集成配置
qlib_integration:
  data_provider:
    cache_enabled: true
    cache_dir: "backend/data/qlib_cache"
    alpha_factors_enabled: true

# 训练配置
training:
  default_config:
    validation_split: 0.2
    early_stopping_patience: 10
  hyperparameter_optimization:
    enabled: true
    max_trials: 20

# 监控配置
monitoring:
  performance:
    enabled: true
    metrics_collection_interval: 60
  alerting:
    enabled: true
    channels: ["email", "websocket"]
```

### 3. 数据库配置

#### SQLite (开发环境)

```bash
# 创建数据目录
mkdir -p backend/data

# 数据库会自动创建
```

#### PostgreSQL (生产环境)

```bash
# 创建数据库用户
sudo -u postgres createuser --interactive stock_user

# 创建数据库
sudo -u postgres createdb stock_prediction -O stock_user

# 设置密码
sudo -u postgres psql -c "ALTER USER stock_user PASSWORD 'your_password';"

# 更新.env文件中的DATABASE_URL
DATABASE_URL="postgresql://stock_user:your_password@localhost:5432/stock_prediction"
```

### 4. Redis配置 (可选)

```bash
# 编辑Redis配置
sudo nano /etc/redis/redis.conf

# 关键配置项
bind 127.0.0.1
port 6379
maxmemory 256mb
maxmemory-policy allkeys-lru

# 启动Redis服务
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

## 部署流程

### 1. 自动部署

使用提供的部署脚本：

```bash
# 完整部署
./scripts/deploy_mlops.sh

# 自定义部署选项
./scripts/deploy_mlops.sh --no-backup --create-service --skip-tests
```

### 2. 手动部署

#### 步骤1: 准备环境

```bash
# 创建必要目录
mkdir -p backend/data/{models,features,qlib_cache}
mkdir -p backend/logs
mkdir -p data/backups

# 设置权限
chmod 755 backend/data
chmod 755 backend/logs
```

#### 步骤2: 安装依赖

```bash
# 创建虚拟环境
python3 -m venv backend/venv
source backend/venv/bin/activate

# 安装Python依赖
pip install -r backend/requirements.txt
pip install qlib shap scikit-learn lightgbm xgboost optuna mlflow psutil pyyaml
```

#### 步骤3: 初始化数据库

```bash
cd backend
python -c "
from app.core.database import engine, Base
from app.models import task_models, stock_models
Base.metadata.create_all(bind=engine)
print('数据库初始化完成')
"
cd ..
```

#### 步骤4: 初始化Qlib

```bash
cd backend
python -c "
import qlib
from qlib.config import REG_CN
qlib.init(provider_uri='data/qlib_data', region=REG_CN)
print('Qlib初始化完成')
"
cd ..
```

#### 步骤5: 启动服务

```bash
# 启动后端服务
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 &
cd ..

# 启动前端服务 (可选)
cd frontend
npm run dev &
cd ..
```

### 3. Docker部署

#### 创建Dockerfile

```dockerfile
# backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装MLOps依赖
RUN pip install qlib shap scikit-learn lightgbm xgboost optuna mlflow psutil pyyaml

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p data/{models,features,qlib_cache} logs

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 创建docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/stock_prediction
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./backend/data:/app/data
      - ./backend/logs:/app/logs
    depends_on:
      - db
      - redis

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=stock_prediction
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

#### 启动Docker服务

```bash
# 构建并启动服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f backend
```

## 运维管理

### 1. 系统服务管理

#### 创建systemd服务

```bash
# 创建服务文件
sudo tee /etc/systemd/system/mlops-backend.service > /dev/null << EOF
[Unit]
Description=MLOps Backend Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)/backend
Environment=PATH=$(pwd)/backend/venv/bin
ExecStart=$(pwd)/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动服务
sudo systemctl daemon-reload
sudo systemctl enable mlops-backend.service
sudo systemctl start mlops-backend.service
```

#### 服务管理命令

```bash
# 启动服务
sudo systemctl start mlops-backend

# 停止服务
sudo systemctl stop mlops-backend

# 重启服务
sudo systemctl restart mlops-backend

# 查看状态
sudo systemctl status mlops-backend

# 查看日志
sudo journalctl -u mlops-backend -f
```

### 2. 日志管理

#### 日志轮转配置

```bash
# 创建logrotate配置
sudo tee /etc/logrotate.d/mlops << EOF
/path/to/backend/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload mlops-backend
    endscript
}
EOF
```

#### 日志监控

```bash
# 实时查看日志
tail -f backend/logs/mlops.log

# 查看错误日志
grep "ERROR" backend/logs/mlops.log | tail -20

# 日志统计
awk '/ERROR/ {error++} /WARNING/ {warning++} /INFO/ {info++} END {print "ERROR:", error, "WARNING:", warning, "INFO:", info}' backend/logs/mlops.log
```

### 3. 健康检查

#### 创建健康检查脚本

```bash
#!/bin/bash
# scripts/health_check.sh

# 检查后端服务
if curl -s http://localhost:8000/health > /dev/null; then
    echo "Backend: OK"
else
    echo "Backend: FAILED"
    exit 1
fi

# 检查数据库连接
cd backend
source venv/bin/activate
python -c "
from app.core.database import SessionLocal
try:
    session = SessionLocal()
    session.execute('SELECT 1')
    session.close()
    print('Database: OK')
except Exception as e:
    print(f'Database: FAILED - {e}')
    exit(1)
"
cd ..

# 检查Redis连接 (如果启用)
if redis-cli ping > /dev/null 2>&1; then
    echo "Redis: OK"
else
    echo "Redis: WARNING - Not available"
fi

echo "Health check completed"
```

#### 设置定时健康检查

```bash
# 添加到crontab
crontab -e

# 每5分钟执行一次健康检查
*/5 * * * * /path/to/scripts/health_check.sh >> /var/log/mlops_health.log 2>&1
```

## 监控配置

### 1. 系统监控

#### 安装监控工具

```bash
# 安装htop
sudo apt install htop

# 安装iotop
sudo apt install iotop

# 安装netstat
sudo apt install net-tools
```

#### 监控脚本

```bash
#!/bin/bash
# scripts/system_monitor.sh

echo "=== 系统监控报告 $(date) ==="

# CPU使用率
echo "CPU使用率:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}'

# 内存使用率
echo "内存使用率:"
free -h

# 磁盘使用率
echo "磁盘使用率:"
df -h

# 网络连接
echo "网络连接:"
netstat -tuln | grep -E ":(8000|3000|5432|6379)"

# 进程状态
echo "关键进程:"
ps aux | grep -E "(uvicorn|npm|postgres|redis)" | grep -v grep

echo "=========================="
```

### 2. 应用监控

#### Prometheus配置 (可选)

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'mlops-backend'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

#### Grafana仪表板 (可选)

创建自定义仪表板监控：
- API响应时间
- 模型训练进度
- 系统资源使用
- 错误率统计

## 安全配置

### 1. 网络安全

#### 防火墙配置

```bash
# Ubuntu UFW
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8000/tcp  # Backend API
sudo ufw allow 3000/tcp  # Frontend (如果需要)

# CentOS firewalld
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=3000/tcp
sudo firewall-cmd --reload
```

#### Nginx反向代理

```nginx
# /etc/nginx/sites-available/mlops
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/ {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 2. 数据安全

#### 数据库安全

```bash
# PostgreSQL安全配置
sudo nano /etc/postgresql/13/main/postgresql.conf

# 限制连接
listen_addresses = 'localhost'
max_connections = 100

# 启用SSL
ssl = on
ssl_cert_file = '/path/to/server.crt'
ssl_key_file = '/path/to/server.key'
```

#### 备份加密

```bash
# 加密备份脚本
#!/bin/bash
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# 创建备份
pg_dump stock_prediction > "$BACKUP_DIR/backup_$DATE.sql"

# 加密备份
gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 \
    --s2k-digest-algo SHA512 --s2k-count 65536 --symmetric \
    --output "$BACKUP_DIR/backup_$DATE.sql.gpg" \
    "$BACKUP_DIR/backup_$DATE.sql"

# 删除未加密文件
rm "$BACKUP_DIR/backup_$DATE.sql"
```

## 性能调优

### 1. 数据库优化

#### PostgreSQL调优

```sql
-- 创建索引
CREATE INDEX CONCURRENTLY idx_model_info_status ON model_info(status);
CREATE INDEX CONCURRENTLY idx_model_info_created_at ON model_info(created_at);
CREATE INDEX CONCURRENTLY idx_stock_data_code_date ON stock_data(stock_code, date);

-- 分析表统计信息
ANALYZE model_info;
ANALYZE stock_data;

-- 查看慢查询
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

#### 连接池配置

```python
# backend/app/core/database.py
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### 2. 应用优化

#### 缓存策略

```python
# Redis缓存配置
CACHES = {
    'default': {
        'BACKEND': 'redis_cache.RedisCache',
        'LOCATION': 'redis://localhost:6379/0',
        'OPTIONS': {
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 50,
                'retry_on_timeout': True,
            }
        },
        'TIMEOUT': 3600,  # 1小时
    }
}
```

#### 异步处理

```python
# 使用Celery处理长时间任务
from celery import Celery

celery_app = Celery(
    'mlops',
    broker='redis://localhost:6379/1',
    backend='redis://localhost:6379/2'
)

@celery_app.task
def train_model_async(model_config):
    # 异步模型训练
    pass
```

### 3. 系统优化

#### 内核参数调优

```bash
# /etc/sysctl.conf
# 网络优化
net.core.somaxconn = 65535
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_max_syn_backlog = 65535

# 内存优化
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# 应用配置
sysctl -p
```

#### 文件描述符限制

```bash
# /etc/security/limits.conf
* soft nofile 65535
* hard nofile 65535

# 重启后生效
```

## 备份恢复

### 1. 数据备份

#### 自动备份脚本

```bash
#!/bin/bash
# scripts/backup.sh

BACKUP_DIR="/data/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 备份数据库
if [ "$DATABASE_TYPE" = "postgresql" ]; then
    pg_dump stock_prediction > "$BACKUP_DIR/db_backup_$DATE.sql"
else
    cp backend/data/app.db "$BACKUP_DIR/db_backup_$DATE.db"
fi

# 备份模型文件
tar -czf "$BACKUP_DIR/models_backup_$DATE.tar.gz" backend/data/models/

# 备份配置文件
tar -czf "$BACKUP_DIR/config_backup_$DATE.tar.gz" .env backend/config/

# 清理旧备份
find "$BACKUP_DIR" -name "*backup*" -mtime +$RETENTION_DAYS -delete

echo "备份完成: $DATE"
```

#### 定时备份

```bash
# 添加到crontab
crontab -e

# 每天凌晨2点备份
0 2 * * * /path/to/scripts/backup.sh >> /var/log/backup.log 2>&1
```

### 2. 数据恢复

#### 恢复脚本

```bash
#!/bin/bash
# scripts/restore.sh

BACKUP_FILE=$1
BACKUP_DIR="/data/backups"

if [ -z "$BACKUP_FILE" ]; then
    echo "用法: $0 <backup_file>"
    echo "可用备份文件:"
    ls -la "$BACKUP_DIR"/*backup*
    exit 1
fi

# 停止服务
sudo systemctl stop mlops-backend

# 恢复数据库
if [[ "$BACKUP_FILE" == *.sql ]]; then
    # PostgreSQL恢复
    dropdb stock_prediction
    createdb stock_prediction
    psql stock_prediction < "$BACKUP_DIR/$BACKUP_FILE"
else
    # SQLite恢复
    cp "$BACKUP_DIR/$BACKUP_FILE" backend/data/app.db
fi

# 恢复模型文件
if [ -f "$BACKUP_DIR/models_backup_*.tar.gz" ]; then
    tar -xzf "$BACKUP_DIR/models_backup_*.tar.gz" -C /
fi

# 恢复配置文件
if [ -f "$BACKUP_DIR/config_backup_*.tar.gz" ]; then
    tar -xzf "$BACKUP_DIR/config_backup_*.tar.gz" -C /
fi

# 启动服务
sudo systemctl start mlops-backend

echo "恢复完成"
```

### 3. 灾难恢复

#### 完整系统恢复流程

1. **准备新环境**
   ```bash
   # 安装基础依赖
   ./scripts/deploy_mlops.sh --no-backup
   ```

2. **恢复数据**
   ```bash
   # 恢复最新备份
   ./scripts/restore.sh db_backup_latest.sql
   ```

3. **验证系统**
   ```bash
   # 检查系统状态
   ./scripts/status_mlops.sh full
   ```

4. **更新配置**
   ```bash
   # 根据新环境更新配置
   nano .env
   nano backend/config/mlops_config.yaml
   ```

## 故障排除

### 常见问题解决

#### 1. 服务启动失败

```bash
# 检查端口占用
sudo netstat -tulpn | grep :8000

# 检查权限
ls -la backend/data/
sudo chown -R $USER:$USER backend/data/

# 检查依赖
source backend/venv/bin/activate
pip list | grep -E "(fastapi|uvicorn|sqlalchemy)"
```

#### 2. 数据库连接问题

```bash
# 测试数据库连接
cd backend
python -c "
from app.core.database import SessionLocal
session = SessionLocal()
print('数据库连接成功')
session.close()
"
```

#### 3. 内存不足

```bash
# 检查内存使用
free -h
ps aux --sort=-%mem | head -10

# 清理缓存
echo 3 | sudo tee /proc/sys/vm/drop_caches

# 重启服务
sudo systemctl restart mlops-backend
```

### 日志分析

```bash
# 查看错误日志
grep -i error backend/logs/mlops.log | tail -20

# 分析访问模式
awk '{print $1}' access.log | sort | uniq -c | sort -nr

# 监控资源使用
sar -u 1 10  # CPU使用率
sar -r 1 10  # 内存使用率
sar -d 1 10  # 磁盘I/O
```

## 总结

本指南涵盖了MLOps系统的完整部署和配置流程。遵循这些步骤可以确保系统的稳定运行和最佳性能。

### 关键要点

1. **环境准备**: 确保满足系统要求
2. **安全配置**: 实施适当的安全措施
3. **性能优化**: 根据负载调整配置
4. **监控告警**: 建立完善的监控体系
5. **备份恢复**: 制定可靠的备份策略

### 后续维护

- 定期更新系统和依赖
- 监控系统性能和资源使用
- 定期测试备份和恢复流程
- 根据使用情况调整配置参数

---

*如有问题，请参考故障排除部分或联系技术支持。*
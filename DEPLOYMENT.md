# 股票预测平台部署指南

## 概述

本文档描述了如何部署股票预测平台，包括开发环境和生产环境的部署方式。

## 系统要求

### 硬件要求
- CPU: 4核心以上
- 内存: 8GB以上
- 存储: 50GB以上可用空间
- 网络: 稳定的互联网连接

### 软件要求
- Docker 20.10+
- Docker Compose 2.0+
- Git
- curl (用于健康检查)

## 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd stock-prediction-platform
```

### 2. 配置环境变量
```bash
cp .env.example .env
# 编辑 .env 文件，配置您的环境变量
```

### 3. 启动服务
```bash
# 生产环境
./scripts/start.sh

# 开发环境
./scripts/start.sh dev
```

### 4. 访问应用
- 前端应用: http://localhost:3000
- 后端API: http://localhost:8000
- API文档: http://localhost:8000/docs

## 详细部署步骤

### 开发环境部署

1. **准备环境**
   ```bash
   # 检查Docker版本
   docker --version
   docker-compose --version
   ```

2. **启动开发环境**
   ```bash
   ./scripts/start.sh development
   ```

3. **查看日志**
   ```bash
   docker-compose -f docker-compose.dev.yml logs -f
   ```

### 生产环境部署

1. **系统配置**
   ```bash
   # 创建应用目录
   sudo mkdir -p /opt/stock-prediction-platform
   sudo chown $USER:$USER /opt/stock-prediction-platform
   
   # 复制项目文件
   cp -r . /opt/stock-prediction-platform/
   cd /opt/stock-prediction-platform
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   # 编辑生产环境配置
   nano .env
   ```

3. **启动服务**
   ```bash
   ./scripts/start.sh production
   ```

4. **配置系统服务（可选）**
   ```bash
   # 复制服务文件
   sudo cp systemd/stock-prediction.service /etc/systemd/system/
   
   # 启用服务
   sudo systemctl daemon-reload
   sudo systemctl enable stock-prediction.service
   sudo systemctl start stock-prediction.service
   ```

## 配置说明

### 环境变量配置

主要环境变量说明：

| 变量名 | 描述 | 默认值 |
|--------|------|--------|
| `DATABASE_URL` | 数据库连接URL | `sqlite:///app/data/stock_prediction.db` |
| `DATA_SERVICE_URL` | 远端数据服务地址 | `http://192.168.3.62:8080` |
| `LOG_LEVEL` | 日志级别 | `INFO` |
| `CORS_ORIGINS` | 允许的跨域源 | `http://localhost:3000` |

### Docker配置

#### 后端配置
- 基础镜像: `python:3.11-slim`
- 端口: 8000
- 数据卷: `./data:/app/data`

#### 前端配置
- 基础镜像: `node:18-alpine`
- 端口: 3000
- 多阶段构建优化

### Nginx配置

Nginx作为反向代理，提供以下功能：
- 静态文件服务
- API请求代理
- WebSocket支持
- Gzip压缩
- SSL终止（可选）

## 监控和日志

### 日志管理

1. **应用日志**
   ```bash
   # 查看所有服务日志
   docker-compose logs -f
   
   # 查看特定服务日志
   docker-compose logs -f backend
   docker-compose logs -f frontend
   ```

2. **日志文件位置**
   - 后端日志: `./backend/logs/`
   - Nginx日志: `/var/log/nginx/`

### 监控配置

系统包含可选的监控组件：

1. **Prometheus** (端口: 9090)
   - 指标收集和存储
   - 配置文件: `monitoring/prometheus.yml`

2. **Grafana** (端口: 3001)
   - 可视化仪表板
   - 默认用户名/密码: admin/admin123

## 维护操作

### 服务管理

```bash
# 启动服务
./scripts/start.sh

# 停止服务
./scripts/stop.sh

# 重启服务
docker-compose restart

# 查看服务状态
docker-compose ps
```

### 数据备份

```bash
# 备份数据目录
tar -czf backup-$(date +%Y%m%d).tar.gz data/

# 备份数据库
sqlite3 data/stock_prediction.db ".backup backup-$(date +%Y%m%d).db"
```

### 更新部署

```bash
# 拉取最新代码
git pull

# 重新构建镜像
docker-compose build --no-cache

# 重启服务
docker-compose down && docker-compose up -d
```

## 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep :8000
   netstat -tulpn | grep :3000
   ```

2. **权限问题**
   ```bash
   # 修复数据目录权限
   sudo chown -R $USER:$USER data/
   chmod -R 755 data/
   ```

3. **内存不足**
   ```bash
   # 检查系统资源
   free -h
   df -h
   docker system df
   ```

4. **网络连接问题**
   ```bash
   # 测试远端数据服务连接
   curl -f http://192.168.3.62:8080/health
   ```

### 健康检查

```bash
# 检查后端健康状态
curl http://localhost:8000/health

# 检查前端服务
curl http://localhost:3000

# 检查所有容器状态
docker-compose ps
```

### 日志分析

```bash
# 查看错误日志
docker-compose logs backend | grep ERROR
docker-compose logs frontend | grep ERROR

# 实时监控日志
docker-compose logs -f --tail=100
```

## 安全考虑

1. **网络安全**
   - 使用防火墙限制端口访问
   - 配置SSL证书（生产环境）
   - 定期更新系统和依赖

2. **数据安全**
   - 定期备份数据
   - 加密敏感配置
   - 限制数据库访问权限

3. **应用安全**
   - 使用强密码
   - 启用CORS保护
   - 定期安全审计

## 性能优化

1. **资源配置**
   - 根据负载调整容器资源限制
   - 配置合适的工作进程数
   - 优化数据库连接池

2. **缓存策略**
   - 启用Nginx静态文件缓存
   - 配置API响应缓存
   - 使用Redis缓存（可选）

3. **监控指标**
   - CPU和内存使用率
   - 响应时间
   - 错误率
   - 数据库性能

## 支持和联系

如有问题，请查看：
1. 项目文档
2. 日志文件
3. GitHub Issues
4. 联系开发团队
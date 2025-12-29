# Parquet股票数据服务

独立的数据获取和存储服务，负责从Tushare获取股票数据并存储到Parquet文件中。

## 功能特性

- 定期从Tushare获取股票数据
- 存储到Parquet文件格式（高效的列式存储）
- 支持增量更新和全量更新
- 自动定时任务调度
- 提供数据状态查询API

## 项目结构

```
back_test_data_service/
├── data_service/          # 数据服务核心模块
│   ├── __init__.py
│   ├── config.py          # 配置管理
│   ├── parquet_dao.py     # Parquet数据访问层
│   ├── mysql_dao.py       # MySQL数据访问层（已废弃）
│   ├── fetcher.py         # 数据获取服务
│   ├── redis_sync.py      # Redis同步服务（已废弃）
│   ├── data_status_api.py # 数据状态API服务
│   └── scheduler.py       # 定时任务调度
├── scripts/               # 脚本目录
│   ├── run_data_service.py  # 数据获取服务启动脚本
│   └── run_data_api.py      # 数据API服务启动脚本
├── static/                # 前端静态文件
│   └── dashboard.js       # 监控面板JavaScript
├── templates/             # 前端模板
│   └── index.html         # 监控面板HTML
├── venv/                  # Python虚拟环境
├── logs/                  # 日志目录
├── requirements.txt       # Python依赖
└── README.md             # 本文档
```

## 安装配置

### 1. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 验证安装

```bash
# 激活虚拟环境
source venv/bin/activate

# 测试Parquet功能
python3 -c "from data_service.parquet_dao import create_dao; dao = create_dao(); print('✅ Parquet DAO创建成功')"
```

### 3. 配置环境变量

创建 `.env` 文件或设置环境变量：

```bash
# Tushare配置（必需）
export TUSHARE_TOKEN="your_tushare_token"

# Parquet存储配置（可选，默认使用 data/parquet/）
# export PARQUET_DATA_DIR="/path/to/custom/parquet/dir"
```

## 使用方法

### 启动数据服务

使用统一的启动脚本：

```bash
# 启动所有服务（数据获取 + API，默认）
./start.sh

# 或明确指定
./start.sh all

# 仅启动数据获取服务
./start.sh service

# 仅启动数据API服务
./start.sh api
```

启动脚本会自动：
- ✅ 检查并创建虚拟环境（如果需要）
- ✅ 验证并安装依赖（包括pyarrow）
- ✅ 修复数据目录权限
- ✅ 使用虚拟环境的Python运行服务

### 服务说明

#### 数据获取服务 (`run_data_service.py`)
- 负责从Tushare获取股票数据并保存到Parquet文件
- 自动执行以下定时任务：
  - **每周一凌晨2点**: 更新股票列表
  - **每天18:00**: 更新所有股票的最新数据
  - **每周日凌晨3点**: 全量更新股票数据

#### 数据API服务 (`run_data_api.py`)
- 提供数据状态查询API接口
- 前端监控面板：http://localhost:5002
- API端点：
  - `GET /api/data/health` - 健康检查
  - `GET /api/data/stock_data_status` - 股票数据状态
  - `GET /api/data/data_summary` - 数据汇总统计

### 手动执行任务

```python
from data_service.fetcher import DataFetcher
from data_service.redis_sync import RedisSync

# 获取并保存股票数据
fetcher = DataFetcher()
fetcher.fetch_and_save_stock_data('000001.SZ', '20200301', '20241231')

# 同步到Redis
sync = RedisSync()
sync.sync_stock_data('000001.SZ', '20200301', '20241231')
```

## 数据表结构

### stock_data 表

存储股票日线数据：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键 |
| ts_code | VARCHAR(20) | 股票代码 |
| trade_date | DATE | 交易日期 |
| open | DECIMAL(10,2) | 开盘价 |
| high | DECIMAL(10,2) | 最高价 |
| low | DECIMAL(10,2) | 最低价 |
| close | DECIMAL(10,2) | 收盘价 |
| volume | BIGINT | 成交量 |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### stock_list 表

存储股票列表：

| 字段 | 类型 | 说明 |
|------|------|------|
| ts_code | VARCHAR(20) | 股票代码（主键） |
| name | VARCHAR(100) | 股票名称 |
| updated_at | TIMESTAMP | 更新时间 |

## Redis键命名规范

股票数据键格式：`stock_data:{md5(ts_code:start_date:end_date)}`

示例：
- `stock_data:a1b2c3d4e5f6...` (000001.SZ:20200301:20241231的哈希值)

## 网络配置

### Mac Mini端

1. 确保Redis和MySQL服务正在运行
2. 开放端口：
   - Redis: 6379
   - MySQL: 3306
3. 配置防火墙（如果需要）

### 高性能服务器端

配置远程连接信息（环境变量或配置文件）：

```bash
# 远程Redis（Mac Mini的IP）
export REMOTE_REDIS_HOST="192.168.1.100"  # Mac Mini的IP地址
export REMOTE_REDIS_PORT="6379"

# 远程MySQL（Mac Mini的IP）
export REMOTE_MYSQL_HOST="192.168.1.100"
export REMOTE_MYSQL_PORT="3306"
export REMOTE_MYSQL_USER="root"
export REMOTE_MYSQL_PASSWORD="your_password"
export REMOTE_MYSQL_DATABASE="stock_data"
```

## 监控和日志

日志文件位置：`logs/data_service.log`

查看日志：

```bash
tail -f logs/data_service.log
```

## 故障排查

### Redis连接失败

1. 检查Redis服务是否运行：`redis-cli ping`
2. 检查防火墙配置
3. 检查Redis配置中的bind设置

### MySQL连接失败

1. 检查MySQL服务是否运行
2. 检查用户权限：`GRANT ALL PRIVILEGES ON stock_data.* TO 'user'@'%';`
3. 检查防火墙配置

### Tushare API调用失败

1. 检查TUSHARE_TOKEN是否正确
2. 检查网络连接
3. 检查API调用频率限制

## 开发说明

这是一个独立的Git仓库，可以单独开发和部署。

### 初始化Git仓库

```bash
cd back_test_data_service
git init
git add .
git commit -m "Initial commit: 数据服务"
```

## 许可证

MIT License


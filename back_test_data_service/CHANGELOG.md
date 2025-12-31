# 重构日志

## 版本 2.0.0 - 独立运行版本

### 主要变更

#### ✅ 独立运行能力
- **移除对backend的依赖**：不再从backend配置中获取Tushare Token
- **内置默认Token**：服务已内置Tushare Token，开箱即用
- **完全独立**：不依赖MySQL/Redis，使用Parquet文件存储

#### ✅ 配置优化
- **简化配置**：移除复杂的服务发现逻辑
- **环境变量支持**：支持通过环境变量覆盖默认配置
- **默认值优化**：提供合理的默认配置值

#### ✅ 文档更新
- **README更新**：详细说明独立运行特性
- **快速开始指南**：提供清晰的启动步骤
- **API文档**：完善API使用说明

#### ✅ 新增功能
- **主启动脚本**：`main.py` 提供统一的Python启动入口
- **配置测试脚本**：`test_config.py` 用于验证配置是否正确
- **环境变量模板**：`.env.example` 提供配置模板

### 文件变更

#### 修改的文件
- `data_service/config.py` - 移除backend依赖，使用内置Token
- `data_service/__init__.py` - 更新模块描述
- `README.md` - 更新文档，说明独立运行特性
- `start.sh` - 更新注释说明

#### 新增的文件
- `main.py` - Python主启动脚本
- `test_config.py` - 配置测试脚本
- `.env.example` - 环境变量配置模板
- `CHANGELOG.md` - 本文件

### 使用方式

#### 快速开始
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 测试配置
python test_config.py

# 3. 启动服务
python main.py          # 启动所有服务
python main.py service  # 仅启动数据获取服务
python main.py api      # 仅启动API服务
```

#### 或使用Shell脚本
```bash
./start.sh              # 启动所有服务
./start.sh service      # 仅启动数据获取服务
./start.sh api          # 仅启动API服务
```

### 配置说明

#### 默认配置
- **Tushare Token**: 已内置默认Token，无需配置即可使用
- **数据存储**: 默认使用 `data/parquet/` 目录
- **API端口**: 默认使用 5002 端口

#### 自定义配置
通过环境变量可以覆盖默认配置：
```bash
export TUSHARE_TOKEN="your_token"
export PARQUET_DATA_DIR="/path/to/data"
```

### 兼容性

- ✅ 完全向后兼容现有功能
- ✅ 现有数据文件无需迁移
- ✅ API接口保持不变

### 注意事项

1. **Token安全**：虽然内置了默认Token，但建议在生产环境中使用环境变量设置
2. **数据目录**：确保数据目录有写入权限
3. **端口占用**：如果5002端口被占用，API服务会自动查找可用端口


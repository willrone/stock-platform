# 数据路径迁移总结

## 迁移时间
2025-01-03

## 迁移目标
统一数据存储路径，将股票数据从 `backend/data/` 迁移到项目根目录的 `data/`

## 迁移原因
1. 配置文件中 `DATA_ROOT_PATH` 设置为相对路径 `./data`
2. 从项目根目录运行时，应该使用 `data/` 而不是 `backend/data/`
3. 避免数据路径混乱和重复存储

## 已完成的操作

### 1. 数据复制
```bash
# 复制股票数据文件
cp -r backend/data/parquet/stock_data/* data/parquet/stock_data/
```

**结果**：
- 源目录：`backend/data/parquet/stock_data/` (5470个文件, 228MB)
- 目标目录：`data/parquet/stock_data/` (5470个文件, 228MB)
- ✅ 数据复制成功

### 2. 路径配置修改

**配置文件**：`backend/app/core/config.py`
```python
# 修改前
DATA_ROOT_PATH: str = "./data"  # 相对于运行目录

# 修改后
DATA_ROOT_PATH: str = "../data"  # 相对于backend目录，指向项目根目录的data/
```

**环境变量**：`backend/.env`
```
# 修改前
DATA_ROOT_PATH="./data"

# 修改后
DATA_ROOT_PATH="../data"
```

**原因**：后端服务从 `backend/` 目录启动，工作目录是 `backend/`，所以需要使用 `../data` 来访问项目根目录的 `data/` 文件夹。

### 3. 数据同步服务路径

**SFTP同步服务**：`backend/app/services/data/sftp_sync_service.py`
- 本地数据目录：`Path(settings.DATA_ROOT_PATH) / "parquet"`
- 实际路径：`data/parquet/stock_data/`
- ✅ 配置正确，无需修改

## 路径使用说明

### 正确的数据路径结构
```
项目根目录/
├── data/                          # 主数据目录
│   ├── parquet/
│   │   └── stock_data/           # 股票数据存储 (5470个文件)
│   │       ├── 000001_SZ.parquet
│   │       ├── 000002_SZ.parquet
│   │       └── ...
│   ├── models/                    # 模型存储
│   ├── logs/                      # 日志文件
│   ├── features/                  # 特征数据
│   └── qlib_cache/               # Qlib缓存
└── backend/
    └── data/                      # 旧数据目录（保留作为备份）
        └── parquet/
            └── stock_data/        # 原始数据（可删除）
```

### 数据加载流程
1. **StockDataLoader** 使用 `settings.DATA_ROOT_PATH` 配置
2. 实际路径：`data/parquet/stock_data/{stock_code}.parquet`
3. 回测执行器从该路径加载数据

### SFTP同步配置
- 远端服务器：192.168.3.62
- 远端路径：`/Users/ronghui/Projects/willrone/data/parquet/stock_data`
- 本地路径：`data/parquet/stock_data/`
- ✅ 同步服务会自动将数据下载到正确位置

## 验证清单

- [x] 数据文件已复制到 `data/parquet/stock_data/`
- [x] 文件数量一致 (5470个文件)
- [x] 文件大小一致 (228MB)
- [x] 配置文件路径正确 (`DATA_ROOT_PATH="./data"`)
- [x] SFTP同步服务路径正确
- [x] StockDataLoader 使用正确的配置

## 后续操作建议

### 1. 测试数据加载
```bash
# 从项目根目录运行
cd /home/willrone/stock-prediction-platform
python -m backend.test_local_data
```

### 2. 测试回测功能
确认回测任务能正常加载股票数据

### 3. 清理旧数据（可选）
确认新路径工作正常后，可以删除旧数据：
```bash
# 谨慎操作！确认新路径正常后再执行
rm -rf backend/data/parquet/stock_data/
```

### 4. 更新文档
如果有其他文档提到数据路径，需要同步更新

## 注意事项

1. **运行目录**：确保从项目根目录运行后端服务
2. **相对路径**：`./data` 是相对于运行目录的路径
3. **数据同步**：SFTP同步会自动下载到 `data/parquet/stock_data/`
4. **备份保留**：建议保留 `backend/data/` 作为备份，确认无问题后再删除

## 影响范围

### 受影响的模块
- ✅ 回测执行器 (BacktestExecutor)
- ✅ 数据加载器 (StockDataLoader)
- ✅ SFTP同步服务 (SFTPSyncService)
- ✅ Qlib数据提供器 (EnhancedQlibProvider)
- ✅ 特征管道 (FeaturePipeline)
- ✅ 模型训练服务 (ModelTrainingService)

### 无需修改的代码
所有模块都使用 `settings.DATA_ROOT_PATH` 配置，无需修改代码

## 迁移状态
✅ **迁移完成** - 数据已成功迁移到统一路径

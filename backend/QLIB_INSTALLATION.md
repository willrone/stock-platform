# Qlib 安装说明

## 问题描述

如果看到日志提示 "Qlib未安装，某些功能将不可用"，需要安装 Microsoft Qlib 量化投资平台。

## 安装方法

### 方法1：从 GitHub 安装（推荐，支持 Python 3.13）

```bash
cd backend
source venv/bin/activate
pip install git+https://github.com/microsoft/qlib.git
```

### 方法2：使用 requirements.txt

由于 Qlib 需要从 GitHub 安装，requirements.txt 中已包含 `pyqlib>=0.9.8.dev18`，但实际安装时需要使用：

```bash
cd backend
source venv/bin/activate
pip install git+https://github.com/microsoft/qlib.git
```

## 验证安装

安装完成后，可以运行以下命令验证：

```bash
python -c "import qlib; from qlib.config import REG_CN; from qlib.data import D; print('Qlib安装成功！')"
```

或者在代码中检查：

```python
from app.services.models.model_training import QLIB_AVAILABLE
print(f"QLIB_AVAILABLE: {QLIB_AVAILABLE}")  # 应该输出 True
```

## 注意事项

1. **Python 版本兼容性**：Qlib 支持 Python 3.7+，在 Python 3.13 上也可以正常工作
2. **依赖项**：安装 Qlib 会自动安装所需的依赖项，包括：
   - numpy, pandas, pyarrow
   - lightgbm, scikit-learn
   - matplotlib, jupyter
   - mlflow, redis, pymongo
   - 等等

3. **安装时间**：从 GitHub 安装可能需要一些时间，因为需要编译一些组件

## 功能说明

安装 Qlib 后，以下功能将可用：
- 模型训练服务中的 QlibDataProvider
- 使用 Qlib 进行数据预处理和特征工程
- 量化投资相关的数据分析和回测功能

## 故障排除

如果安装失败，可以尝试：
1. 升级 pip: `pip install --upgrade pip setuptools wheel`
2. 使用国内镜像（如果网络问题）：
   ```bash
   pip install git+https://github.com/microsoft/qlib.git -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```
3. 检查 Python 版本：`python --version`（需要 3.7+）


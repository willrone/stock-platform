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

## 常见问题

### 问题1：缺少 setuptools_scm 模块

如果看到错误 "No module named 'setuptools_scm'"：

```bash
cd backend
source venv/bin/activate
pip install setuptools_scm
```

### 问题2：Qlib 安装后缺少依赖

如果 Qlib 已安装但缺少依赖（如 cvxpy, lightgbm 等），需要安装完整依赖：

```bash
cd backend
source venv/bin/activate

# 安装 Qlib 的完整依赖
pip install setuptools_scm cvxpy dill fire gym jupyter lightgbm matplotlib mlflow nbconvert pymongo python-redis-lock redis "ruamel.yaml>=0.17.38"
```

或者让 pip 自动安装所有缺失的依赖：

```bash
# 重新安装 Qlib，让 pip 自动安装所有依赖
pip install --force-reinstall --no-cache-dir git+https://gitee.com/mirrors/qlib.git
```

## 故障排除

### 方法1：使用 Gitee 镜像（推荐，适用于无法访问 GitHub 的情况）

如果遇到 "Failed to connect to github.com" 错误，可以使用 Gitee 镜像：

```bash
cd backend
source venv/bin/activate

# 方法1：直接从 Gitee 镜像安装
pip install git+https://gitee.com/mirrors/qlib.git

# 方法2：手动克隆后安装（如果直接安装失败）
git clone https://gitee.com/mirrors/qlib.git /tmp/qlib
pip install /tmp/qlib
rm -rf /tmp/qlib
```

### 方法2：使用代理（如果有可用的代理）

```bash
# 设置代理环境变量（示例）
export http_proxy=http://proxy.example.com:8080
export https_proxy=http://proxy.example.com:8080

# 然后再安装
pip install git+https://github.com/microsoft/qlib.git
```

### 方法3：其他解决方案

1. **升级 pip**: `pip install --upgrade pip setuptools wheel`

2. **使用国内 PyPI 镜像源**（注意：这不会解决 git clone 的问题，但可以加速依赖包的安装）：
   ```bash
   pip install git+https://gitee.com/mirrors/qlib.git -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. **增加超时时间**：
   ```bash
   pip install --timeout=600 git+https://gitee.com/mirrors/qlib.git
   ```

4. **检查 Python 版本**：`python --version`（需要 3.7+）

5. **手动下载源码安装**：
   ```bash
   # 从 Gitee 下载 ZIP 文件并解压
   # 然后进入目录
   cd qlib
   pip install .
   ```

### 方法4：解决 osqp 编译失败问题

如果安装过程中遇到 `osqp` 编译失败（需要从 GitHub 克隆依赖），可以尝试以下方案：

#### 方案A：先单独安装 osqp 预编译版本（推荐）

```bash
# 先尝试安装预编译的 osqp wheel（如果可用）
pip install osqp

# 然后再安装 qlib
pip install git+https://gitee.com/mirrors/qlib.git --no-deps
pip install -r <(pip show pyqlib | grep -A 100 "^Requires:" | sed 's/^Requires: //' | tr ',' '\n' | grep -v "^osqp$" | tr '\n' ' ')
```

#### 方案B：配置 git 使用 Gitee 镜像或代理

如果方案A不可行，需要让 CMake 在编译时使用镜像源。由于 `osqp` 是 `cvxpy` 的依赖，而 `cvxpy` 又是 `qlib` 的依赖，可以：

1. **手动克隆 osqp 依赖**：
   ```bash
   # 设置环境变量，让 CMake 使用本地克隆的代码
   export OSQP_SOURCE_DIR=/tmp/osqp-source
   git clone --depth 1 https://gitee.com/mirrors/osqp.git $OSQP_SOURCE_DIR 2>/dev/null || \
   git clone --depth 1 https://github.com/osqp/osqp.git $OSQP_SOURCE_DIR
   ```

2. **使用代理环境变量**（如果有代理）：
   ```bash
   export http_proxy=http://your-proxy:port
   export https_proxy=http://your-proxy:port
   export GIT_SSL_NO_VERIFY=1  # 如果需要
   pip install git+https://gitee.com/mirrors/qlib.git
   ```

#### 方案C：跳过 osqp 依赖（如果不需要优化功能）

如果您的使用场景不需要 `cvxpy` 的优化功能，可以：

```bash
# 安装 qlib 时跳过可选依赖
pip install git+https://gitee.com/mirrors/qlib.git --no-deps
# 然后手动安装必需的依赖（排除 cvxpy/osqp）
pip install numpy pandas scikit-learn lightgbm xgboost
```

注意：这可能会影响某些 Qlib 功能，特别是与优化相关的功能。


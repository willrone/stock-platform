#!/bin/bash

# Parquet数据服务统一启动脚本
# 支持启动数据获取服务和数据API服务

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"

# 检查虚拟环境是否存在
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ 错误: 虚拟环境未找到"
    echo "   正在创建虚拟环境并安装依赖..."
    cd "$SCRIPT_DIR"
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo "✅ 虚拟环境创建完成"
fi

# 检查Python可执行性
if [ ! -x "$VENV_PYTHON" ]; then
    echo "❌ 错误: Python解释器不可执行"
    exit 1
fi

# 验证pyarrow是否安装
echo "🔍 验证依赖..."
if ! "$VENV_PYTHON" -c "import pyarrow; print('✅ PyArrow version:', pyarrow.__version__)" 2>/dev/null; then
    echo "⚠️  PyArrow未安装，正在安装..."
    source "$SCRIPT_DIR/venv/bin/activate"
    pip install pyarrow -i https://pypi.tuna.tsinghua.edu.cn/simple
    if [ $? -ne 0 ]; then
        echo "❌ PyArrow安装失败"
        exit 1
    fi
    echo "✅ PyArrow安装成功"
fi

# 确保日志目录存在
mkdir -p "$SCRIPT_DIR/logs"

# 确保数据目录存在并有正确权限
DATA_DIR="$SCRIPT_DIR/../data/parquet"
mkdir -p "$DATA_DIR/stock_data"
if [ -d "$DATA_DIR" ]; then
    # 尝试修复权限（如果需要sudo）
    if [ ! -w "$DATA_DIR" ]; then
        echo "🔧 修复数据目录权限..."
        echo "101618" | sudo -S chown -R $(whoami):staff "$DATA_DIR" 2>/dev/null || true
        echo "101618" | sudo -S chmod -R 755 "$DATA_DIR" 2>/dev/null || true
    fi
fi

# 解析命令行参数
SERVICE_TYPE="${1:-all}"

# 设置环境变量，确保使用虚拟环境的Python
export PATH="$SCRIPT_DIR/venv/bin:$PATH"
export VIRTUAL_ENV="$SCRIPT_DIR/venv"
export PYTHONHOME=""

case "$SERVICE_TYPE" in
    "service"|"data")
        echo "🚀 启动Parquet数据获取服务..."
        echo "📋 日志文件: $SCRIPT_DIR/logs/data_service.log"
        echo "🐍 使用Python: $VENV_PYTHON"
        echo ""
        # 使用虚拟环境的Python，并设置工作目录
        cd "$SCRIPT_DIR"
        exec "$VENV_PYTHON" "$SCRIPT_DIR/scripts/run_data_service.py"
        ;;
    "api")
        echo "🚀 启动Parquet数据API服务..."
        echo "📋 日志文件: $SCRIPT_DIR/logs/data_api.log"
        echo "🌐 API服务地址: http://localhost:5002"
        echo "🐍 使用Python: $VENV_PYTHON"
        echo ""
        # 使用虚拟环境的Python，并设置工作目录
        cd "$SCRIPT_DIR"
        exec "$VENV_PYTHON" "$SCRIPT_DIR/scripts/run_data_api.py"
        ;;
    "all")
        echo "🚀 启动Parquet数据服务（数据获取 + API）..."
        echo "📋 数据服务日志: $SCRIPT_DIR/logs/data_service.log"
        echo "📋 API服务日志: $SCRIPT_DIR/logs/data_api.log"
        echo "🌐 API服务地址: http://localhost:5002"
        echo "🐍 使用Python: $VENV_PYTHON"
        echo ""
        
        # 设置工作目录
        cd "$SCRIPT_DIR"
        
        # 后台启动数据获取服务
        "$VENV_PYTHON" "$SCRIPT_DIR/scripts/run_data_service.py" > "$SCRIPT_DIR/logs/data_service.log" 2>&1 &
        DATA_SERVICE_PID=$!
        echo "✅ 数据获取服务已启动 (PID: $DATA_SERVICE_PID)"
        
        # 等待一下让数据服务启动
        sleep 2
        
        # 前台启动API服务（作为主进程）
        exec "$VENV_PYTHON" "$SCRIPT_DIR/scripts/run_data_api.py"
        ;;
    *)
        echo "用法: $0 [service|api|all]"
        echo ""
        echo "选项:"
        echo "  service, data  - 仅启动数据获取服务"
        echo "  api            - 仅启动数据API服务"
        echo "  all            - 同时启动数据获取服务和API服务（默认）"
        exit 1
        ;;
esac
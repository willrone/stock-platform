# 使用Python 3.11作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 配置pip使用国内源并分步安装依赖（优化缓存）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip install --upgrade pip

# 先安装基础依赖（较小的包）
RUN pip install --no-deps \
    fastapi==0.104.1 \
    uvicorn==0.24.0 \
    python-multipart==0.0.6 \
    websockets==12.0 \
    sqlalchemy==2.0.23 \
    alembic==1.12.1 \
    aiosqlite==0.19.0 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    python-dotenv==1.0.0 \
    loguru==0.7.2 \
    prometheus-client==0.19.0 \
    httpx==0.25.2

# 安装数据处理包
RUN pip install --no-deps \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    pyarrow==14.0.1

# 最后安装大型包（PyTorch等）
RUN pip install --no-deps \
    torch>=2.1.0 --index-url https://download.pytorch.org/whl/cpu

# 安装剩余依赖
RUN pip install -r requirements.txt

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/stocks /app/data/models /app/data/logs

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
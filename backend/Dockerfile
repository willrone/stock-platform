# 后端服务 Dockerfile
# 多阶段构建：分离依赖安装和应用代码，利用 Docker 缓存加速构建

# ========== 构建阶段 ==========
FROM python:3.13-slim AS builder

WORKDIR /build

# 最小系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 只复制 requirements.txt 以缓存 pip install 层
COPY requirements.txt .

# 换国内源，pip 缓存到本地
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip install --upgrade pip

# 安装依赖到 /install 目录，用于运行时阶段复制
RUN pip install --user -r requirements.txt

# 单独安装 Qlib（从 GitHub，需要编译）
RUN pip install --user git+https://github.com/microsoft/qlib.git

# ========== 运行时阶段 ==========
FROM python:3.13-slim AS runner

WORKDIR /app

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 系统运行时依赖（最小集）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 从 builder 复制已安装的包
COPY --from=builder /root/.local /root/.local

# 复制应用代码
COPY . .

# 创建运行时目录
RUN mkdir -p /app/data/stocks /app/data/models /app/data/logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

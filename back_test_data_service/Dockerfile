# 数据服务 Dockerfile
FROM python:3.12-slim

WORKDIR /app

ENV \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

# 创建数据目录和日志目录
RUN mkdir -p data logs

EXPOSE 5002

# 默认启动 all（数据获取+API）
CMD ["python", "main.py", "all"]

#!/bin/bash

# 测试运行脚本
# 运行所有测试并生成覆盖率报告

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}开始运行测试套件...${NC}\n"

# 运行后端测试
echo -e "${YELLOW}运行后端测试...${NC}"
cd backend

# 检查覆盖率阈值
COVERAGE_THRESHOLD=80

echo "  - 运行pytest测试..."
if pytest tests/ --cov=app --cov-report=html --cov-report=term --cov-report=json \
    --cov-fail-under=$COVERAGE_THRESHOLD -v; then
    echo -e "${GREEN}  ✓ 后端测试通过（覆盖率 ≥ ${COVERAGE_THRESHOLD}%）${NC}"
else
    echo -e "${RED}  ❌ 后端测试失败或覆盖率不足${NC}"
    exit 1
fi

cd ..

# 运行前端测试
echo -e "\n${YELLOW}运行前端测试...${NC}"
cd frontend

echo "  - 运行Jest测试..."
if npm run test:ci; then
    echo -e "${GREEN}  ✓ 前端测试通过${NC}"
else
    echo -e "${RED}  ❌ 前端测试失败${NC}"
    exit 1
fi

cd ..

echo -e "\n${GREEN}✅ 所有测试通过！${NC}"
echo -e "${BLUE}覆盖率报告已生成：${NC}"
echo -e "  - 后端: backend/htmlcov/index.html"
echo -e "  - 前端: frontend/coverage/index.html"

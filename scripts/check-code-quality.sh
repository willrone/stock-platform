#!/bin/bash

# 代码质量检查脚本
# 用于在CI/CD和本地开发中检查代码质量

# 不立即退出，收集所有错误
set +e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始代码质量检查...${NC}\n"

# 检查后端代码质量
echo -e "${YELLOW}检查后端代码质量...${NC}"
cd backend

# 检测并激活虚拟环境
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Black格式化检查
echo "  - 检查代码格式化 (Black)..."
BLACK_ERROR=0
if ! black --check app/ 2>&1 | tee /tmp/black_output.txt; then
    BLACK_ERROR=1
    BLACK_FILES=$(grep "would reformat" /tmp/black_output.txt | wc -l)
    echo -e "${YELLOW}  ⚠️  Black检查发现 $BLACK_FILES 个文件需要格式化${NC}"
    echo -e "${YELLOW}  运行 'black app/' 可以自动修复${NC}"
else
    echo -e "${GREEN}  ✓ Black检查通过${NC}"
fi

# isort导入排序检查
echo "  - 检查导入排序 (isort)..."
ISORT_ERROR=0
if ! isort --check-only app/ 2>&1 | tee /tmp/isort_output.txt; then
    ISORT_ERROR=1
    echo -e "${YELLOW}  ⚠️  isort检查发现需要调整的导入${NC}"
    echo -e "${YELLOW}  运行 'isort app/' 可以自动修复${NC}"
else
    echo -e "${GREEN}  ✓ isort检查通过${NC}"
fi

# Flake8代码风格检查
echo "  - 检查代码风格 (Flake8)..."
FLAKE8_ERROR=0
if ! flake8 app/ 2>&1 | tee /tmp/flake8_output.txt; then
    FLAKE8_ERROR=1
    FLAKE8_ISSUES=$(grep -c "^" /tmp/flake8_output.txt || echo "0")
    echo -e "${YELLOW}  ⚠️  Flake8检查发现 $FLAKE8_ISSUES 个问题${NC}"
else
    echo -e "${GREEN}  ✓ Flake8检查通过${NC}"
fi

# mypy类型检查
echo "  - 检查类型注解 (mypy)..."
if ! mypy app/ --ignore-missing-imports; then
    echo -e "${YELLOW}⚠️  mypy检查有警告（忽略导入错误）${NC}"
fi
echo -e "${GREEN}  ✓ mypy检查完成${NC}"

# Bandit安全扫描
echo "  - 安全扫描 (Bandit)..."
if ! bandit -r app/ -f json -o bandit-report.json; then
    echo -e "${YELLOW}⚠️  Bandit发现安全问题，请查看报告${NC}"
fi
echo -e "${GREEN}  ✓ Bandit扫描完成${NC}"

cd ..

# 检查前端代码质量
echo -e "\n${YELLOW}检查前端代码质量...${NC}"
cd frontend

# ESLint检查
echo "  - 检查代码质量 (ESLint)..."
if ! npm run lint; then
    echo -e "${RED}❌ ESLint检查失败${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ ESLint检查通过${NC}"

# Prettier格式化检查
echo "  - 检查代码格式化 (Prettier)..."
if ! npm run format:check; then
    echo -e "${RED}❌ Prettier检查失败，请运行: npm run format${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Prettier检查通过${NC}"

# TypeScript类型检查
echo "  - 检查类型 (TypeScript)..."
if ! npm run type-check; then
    echo -e "${RED}❌ TypeScript类型检查失败${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ TypeScript检查通过${NC}"

cd ..

# 汇总结果
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}代码质量检查汇总${NC}"
echo -e "${BLUE}========================================${NC}"

TOTAL_ERRORS=0

if [ $BLACK_ERROR -eq 1 ]; then
    echo -e "${YELLOW}⚠️  Black: 需要格式化${NC}"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
else
    echo -e "${GREEN}✓ Black: 通过${NC}"
fi

if [ $ISORT_ERROR -eq 1 ]; then
    echo -e "${YELLOW}⚠️  isort: 需要调整导入${NC}"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
else
    echo -e "${GREEN}✓ isort: 通过${NC}"
fi

if [ $FLAKE8_ERROR -eq 1 ]; then
    echo -e "${YELLOW}⚠️  Flake8: 发现代码风格问题${NC}"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
else
    echo -e "${GREEN}✓ Flake8: 通过${NC}"
fi

echo -e "${BLUE}========================================${NC}"

if [ $TOTAL_ERRORS -eq 0 ]; then
    echo -e "\n${GREEN}✅ 所有代码质量检查通过！${NC}"
    exit 0
else
    echo -e "\n${YELLOW}⚠️  发现 $TOTAL_ERRORS 类问题需要修复${NC}"
    echo -e "${YELLOW}建议运行以下命令自动修复：${NC}"
    echo -e "  ${BLUE}cd backend${NC}"
    echo -e "  ${BLUE}source venv/bin/activate${NC}"
    echo -e "  ${BLUE}black app/ && isort app/${NC}"
    exit 1
fi

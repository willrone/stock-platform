#!/bin/bash

# 代码质量检查脚本
# 用于在CI/CD和本地开发中检查代码质量

# 不立即退出，收集所有错误；同时确保管道中的失败不会被 tee 吞掉
set +e
set -o pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}开始代码质量检查...${NC}\n"

ROOT_DIR="$(pwd)"
BACKEND_DIR="${ROOT_DIR}/backend"
FRONTEND_DIR="${ROOT_DIR}/frontend"

BACKEND_BIN=""
if [ -x "${BACKEND_DIR}/venv/bin/python" ]; then
    BACKEND_BIN="${BACKEND_DIR}/venv/bin"
elif [ -x "${BACKEND_DIR}/.venv/bin/python" ]; then
    BACKEND_BIN="${BACKEND_DIR}/.venv/bin"
fi

run_backend_tool() {
    local tool="$1"
    shift

    if [ -z "${BACKEND_BIN}" ]; then
        echo "__CODE_QUALITY_MISSING_ENV__ 未找到 backend 虚拟环境（venv 或 .venv）"
        return 127
    fi

    if [ ! -x "${BACKEND_BIN}/${tool}" ]; then
        echo "__CODE_QUALITY_MISSING_TOOL__ ${tool}"
        return 127
    fi

    "${BACKEND_BIN}/${tool}" "$@"
}

backend_tool_failed() {
    local output_file="$1"
    local tool_name="$2"

    if grep -q "__CODE_QUALITY_MISSING_ENV__" "${output_file}"; then
        echo -e "${RED}  ❌ 未找到后端虚拟环境，无法执行 ${tool_name}${NC}"
    elif grep -q "__CODE_QUALITY_MISSING_TOOL__" "${output_file}"; then
        echo -e "${RED}  ❌ 缺少后端工具 ${tool_name}${NC}"
    fi
}

# 检查后端代码质量
echo -e "${YELLOW}检查后端代码质量...${NC}"
cd "${BACKEND_DIR}"

# Black格式化检查
echo "  - 检查代码格式化 (Black)..."
BLACK_ERROR=0
if ! run_backend_tool black --check app/ 2>&1 | tee /tmp/black_output.txt; then
    BLACK_ERROR=1
    backend_tool_failed /tmp/black_output.txt black
    if grep -q "would reformat" /tmp/black_output.txt; then
        BLACK_FILES=$(grep "would reformat" /tmp/black_output.txt | wc -l)
        echo -e "${YELLOW}  ⚠️  Black检查发现 $BLACK_FILES 个文件需要格式化${NC}"
        echo -e "${YELLOW}  运行 'black app/' 可以自动修复${NC}"
    fi
else
    echo -e "${GREEN}  ✓ Black检查通过${NC}"
fi

# isort导入排序检查
echo "  - 检查导入排序 (isort)..."
ISORT_ERROR=0
if ! run_backend_tool isort --check-only app/ 2>&1 | tee /tmp/isort_output.txt; then
    ISORT_ERROR=1
    backend_tool_failed /tmp/isort_output.txt isort
    if ! grep -q "__CODE_QUALITY_MISSING_" /tmp/isort_output.txt; then
        echo -e "${YELLOW}  ⚠️  isort检查发现需要调整的导入${NC}"
        echo -e "${YELLOW}  运行 'isort app/' 可以自动修复${NC}"
    fi
else
    echo -e "${GREEN}  ✓ isort检查通过${NC}"
fi

# Flake8代码风格检查
echo "  - 检查代码风格 (Flake8)..."
FLAKE8_ERROR=0
if ! run_backend_tool flake8 app/ 2>&1 | tee /tmp/flake8_output.txt; then
    FLAKE8_ERROR=1
    backend_tool_failed /tmp/flake8_output.txt flake8
    if ! grep -q "__CODE_QUALITY_MISSING_" /tmp/flake8_output.txt; then
        FLAKE8_ISSUES=$(grep -c "^" /tmp/flake8_output.txt || echo "0")
        echo -e "${YELLOW}  ⚠️  Flake8检查发现 $FLAKE8_ISSUES 个问题${NC}"
    fi
else
    echo -e "${GREEN}  ✓ Flake8检查通过${NC}"
fi

# mypy类型检查
echo "  - 检查类型注解 (mypy)..."
MYPY_ERROR=0
if ! run_backend_tool mypy app/ --ignore-missing-imports 2>&1 | tee /tmp/mypy_output.txt; then
    MYPY_ERROR=1
    backend_tool_failed /tmp/mypy_output.txt mypy
    if ! grep -q "__CODE_QUALITY_MISSING_" /tmp/mypy_output.txt; then
        echo -e "${YELLOW}⚠️  mypy检查未通过${NC}"
    fi
else
    echo -e "${GREEN}  ✓ mypy检查通过${NC}"
fi

# Bandit安全扫描
echo "  - 安全扫描 (Bandit)..."
BANDIT_ERROR=0
if ! run_backend_tool bandit -r app/ -f json -o bandit-report.json 2>&1 | tee /tmp/bandit_output.txt; then
    BANDIT_ERROR=1
    backend_tool_failed /tmp/bandit_output.txt bandit
    if ! grep -q "__CODE_QUALITY_MISSING_" /tmp/bandit_output.txt; then
        echo -e "${YELLOW}⚠️  Bandit发现安全问题，请查看报告${NC}"
    fi
else
    echo -e "${GREEN}  ✓ Bandit扫描通过${NC}"
fi

cd ..

# 检查前端代码质量
echo -e "\n${YELLOW}检查前端代码质量...${NC}"
cd "${FRONTEND_DIR}"

# ESLint检查
echo "  - 检查代码质量 (ESLint)..."
FRONTEND_LINT_ERROR=0
if ! npm run lint; then
    FRONTEND_LINT_ERROR=1
    echo -e "${RED}❌ ESLint检查失败${NC}"
else
    echo -e "${GREEN}  ✓ ESLint检查通过${NC}"
fi

# Prettier格式化检查
echo "  - 检查代码格式化 (Prettier)..."
FRONTEND_FORMAT_ERROR=0
if ! npm run format:check; then
    FRONTEND_FORMAT_ERROR=1
    echo -e "${RED}❌ Prettier检查失败，请运行: npm run format${NC}"
else
    echo -e "${GREEN}  ✓ Prettier检查通过${NC}"
fi

# TypeScript类型检查
echo "  - 检查类型 (TypeScript)..."
FRONTEND_TYPE_ERROR=0
if ! npm run type-check; then
    FRONTEND_TYPE_ERROR=1
    echo -e "${RED}❌ TypeScript类型检查失败${NC}"
else
    echo -e "${GREEN}  ✓ TypeScript检查通过${NC}"
fi

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

if [ $MYPY_ERROR -eq 1 ]; then
    echo -e "${YELLOW}⚠️  mypy: 未通过${NC}"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
else
    echo -e "${GREEN}✓ mypy: 通过${NC}"
fi

if [ $BANDIT_ERROR -eq 1 ]; then
    echo -e "${YELLOW}⚠️  Bandit: 未通过或未安装${NC}"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
else
    echo -e "${GREEN}✓ Bandit: 通过${NC}"
fi

if [ $FRONTEND_LINT_ERROR -eq 1 ]; then
    echo -e "${YELLOW}⚠️  ESLint: 未通过${NC}"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
else
    echo -e "${GREEN}✓ ESLint: 通过${NC}"
fi

if [ $FRONTEND_FORMAT_ERROR -eq 1 ]; then
    echo -e "${YELLOW}⚠️  Prettier: 未通过${NC}"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
else
    echo -e "${GREEN}✓ Prettier: 通过${NC}"
fi

if [ $FRONTEND_TYPE_ERROR -eq 1 ]; then
    echo -e "${YELLOW}⚠️  TypeScript: 未通过${NC}"
    TOTAL_ERRORS=$((TOTAL_ERRORS + 1))
else
    echo -e "${GREEN}✓ TypeScript: 通过${NC}"
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

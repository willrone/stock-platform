#!/bin/bash

# 生成质量报告脚本
# 生成代码质量、测试覆盖率、安全扫描等综合报告

set -e

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

REPORT_DIR="quality-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo -e "${BLUE}生成质量报告...${NC}\n"

# 创建报告目录
mkdir -p $REPORT_DIR

# 后端报告
echo "生成后端报告..."
cd backend

# 测试覆盖率报告
pytest tests/ --cov=app --cov-report=html --cov-report=term-missing --cov-report=json \
    -o junit_family=xunit2 --junitxml=../$REPORT_DIR/backend-test-results.xml \
    --cov-report=html:../$REPORT_DIR/backend-coverage-html \
    --cov-report=json:../$REPORT_DIR/backend-coverage.json

# 安全扫描报告
bandit -r app/ -f json -o ../$REPORT_DIR/backend-security-report.json || true

# 代码复杂度报告（使用radon）
if command -v radon &> /dev/null; then
    radon cc app/ -j -o ../$REPORT_DIR/backend-complexity.json || true
    radon mi app/ -j -o ../$REPORT_DIR/backend-maintainability.json || true
fi

cd ..

# 前端报告
echo "生成前端报告..."
cd frontend

# 测试覆盖率报告
npm run test:coverage -- --coverageDirectory=../$REPORT_DIR/frontend-coverage

# 依赖安全扫描
if [ -f package-lock.json ]; then
    npm audit --json > ../$REPORT_DIR/frontend-security-audit.json || true
fi

cd ..

# 生成汇总报告
echo "生成汇总报告..."
cat > $REPORT_DIR/README.md << EOF
# 代码质量报告

生成时间: $(date)

## 报告内容

### 后端
- 测试覆盖率: [backend-coverage-html/index.html](backend-coverage-html/index.html)
- 安全扫描: [backend-security-report.json](backend-security-report.json)
- 测试结果: [backend-test-results.xml](backend-test-results.xml)

### 前端
- 测试覆盖率: [frontend-coverage/index.html](frontend-coverage/index.html)
- 安全审计: [frontend-security-audit.json](frontend-security-audit.json)

## 查看报告

\`\`\`bash
# 后端覆盖率报告
open $REPORT_DIR/backend-coverage-html/index.html

# 前端覆盖率报告
open $REPORT_DIR/frontend-coverage/index.html
\`\`\`
EOF

echo -e "${GREEN}✅ 报告生成完成！${NC}"
echo -e "报告目录: ${BLUE}$REPORT_DIR${NC}"

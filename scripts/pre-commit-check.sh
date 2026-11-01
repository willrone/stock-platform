#!/bin/bash
# 提交前检查脚本
# 确保代码质量和安全性

set -e

echo "🔍 执行提交前检查..."

# 检查是否有敏感文件
echo "📋 检查敏感文件..."
SENSITIVE_FILES=(
    "*.env"
    "*.key" 
    "*.pem"
    "*.p12"
    "*.pfx"
    "config.ini"
    "secrets.json"
    "credentials.json"
    "*.db"
    "*.sqlite*"
)

for pattern in "${SENSITIVE_FILES[@]}"; do
    if git ls-files --cached | grep -q "$pattern"; then
        echo "❌ 发现敏感文件: $pattern"
        echo "请将其添加到.gitignore并从暂存区移除"
        exit 1
    fi
done

# 检查大文件
echo "📏 检查大文件..."
MAX_SIZE=10485760  # 10MB
large_files=$(git ls-files --cached | xargs ls -l | awk '$5 > '$MAX_SIZE' {print $9, $5}')
if [ -n "$large_files" ]; then
    echo "❌ 发现大文件 (>10MB):"
    echo "$large_files"
    echo "请考虑使用Git LFS或将其添加到.gitignore"
    exit 1
fi

# 后端检查
if [ -d "backend" ]; then
    echo "🐍 检查Python后端..."
    cd backend
    
    # 简单检查是否有明显的语法错误（只检查主要文件）
    if command -v python3 &> /dev/null; then
        echo "  - 快速语法检查..."
        for file in app/main.py run.py; do
            if [ -f "$file" ]; then
                python3 -m py_compile "$file" 2>/dev/null || {
                    echo "❌ $file 语法错误"
                    exit 1
                }
            fi
        done
    fi
    
    cd ..
fi

# 前端检查
if [ -d "frontend" ]; then
    echo "⚛️  检查React前端..."
    cd frontend
    
    # 快速检查TypeScript类型（跳过测试以节省时间）
    if [ -f "package.json" ] && command -v npm &> /dev/null; then
        echo "  - 快速TypeScript检查..."
        npm run type-check || {
            echo "❌ TypeScript类型错误"
            exit 1
        }
    fi
    
    cd ..
fi

# 检查提交信息格式（如果是通过git hooks调用）
if [ -n "$1" ]; then
    echo "📝 检查提交信息格式..."
    commit_msg=$(cat "$1")
    
    # 检查提交信息是否符合Conventional Commits格式
    if ! echo "$commit_msg" | grep -qE "^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .+"; then
        echo "❌ 提交信息格式不正确"
        echo "请使用Conventional Commits格式:"
        echo "  feat: 新功能"
        echo "  fix: 修复bug"
        echo "  docs: 文档更新"
        echo "  style: 代码格式"
        echo "  refactor: 重构"
        echo "  test: 测试"
        echo "  chore: 构建/工具"
        exit 1
    fi
fi

# ── 未定义名闸门（F821/F811，零基线）────────────────────────────────
# 为什么单独挂：check-code-quality.sh 的 flake8 只覆盖 app/，
# 而 cron/launchd 实际执行的是 scripts/ → 这一类必崩的错误没人看。
# 2026-11-01 实测：scripts/paper_portfolio.py 引用已删除的 LIMIT_UP →
# 纸面盘整轮崩（RC=1）→ 同一次运行里后面的 multi_asset_final 旗舰线轮不到执行。
echo "🔤 检查未定义名 (F821/F811)..."
UNDEF_RC=0
for PYBIN in backend/.venv/bin/python3 backend/.venv-py313/bin/python3 backend/venv/bin/python3; do
    if [ -x "$PYBIN" ]; then
        env -u PYTHONPATH -u PYTHONHOME "$PYBIN" backend/scripts/check_undefined_names.py || UNDEF_RC=$?
        break
    fi
done
if [ "$UNDEF_RC" -ne 0 ]; then
    echo "❌ 未定义名检查未通过（新增 F821/F811 会导致运行时 NameError）"
    exit 1
fi

# ── 闸门自身可运行性体检 ─────────────────────────────────────────────
# 门禁的门也必须是能跑的：pre-commit-check.sh / quick-check.sh 曾经是
# CRLF 行尾 → bash 语法错误 → 脚本第一行都没执行，却一直"存在"着。
echo "🚪 检查闸门自身可运行性..."
GATES_RC=0
for PYBIN in backend/.venv/bin/python3 backend/.venv-py313/bin/python3 backend/venv/bin/python3; do
    if [ -x "$PYBIN" ]; then
        env -u PYTHONPATH -u PYTHONHOME "$PYBIN" backend/scripts/check_gates_runnable.py || GATES_RC=$?
        break
    fi
done
if [ "$GATES_RC" -ne 0 ]; then
    echo "❌ 闸门可运行性体检未通过"
    exit 1
fi

echo "✅ 所有检查通过！"
echo "🚀 可以安全提交代码"
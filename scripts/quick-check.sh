#!/bin/bash
# 快速提交前检查脚本
# 只检查最关键的问题

set -e

echo "⚡ 执行快速检查..."

# 检查敏感文件
echo "🔒 检查敏感文件..."
if git ls-files --cached | grep -E "\.(env|key|pem|p12|pfx)$|config\.ini|secrets\.json|credentials\.json|\.db$|\.sqlite"; then
    echo "❌ 发现敏感文件，请检查.gitignore"
    exit 1
fi

# 检查超大文件 (>50MB)
echo "📏 检查超大文件..."
large_files=$(git ls-files --cached | xargs ls -l 2>/dev/null | awk '$5 > 52428800 {print $9, $5}' || true)
if [ -n "$large_files" ]; then
    echo "❌ 发现超大文件 (>50MB):"
    echo "$large_files"
    exit 1
fi

# 检查是否有Python语法错误（只检查主文件）
if [ -f "backend/app/main.py" ]; then
    echo "🐍 检查主要Python文件..."
    python3 -m py_compile backend/app/main.py 2>/dev/null || {
        echo "❌ main.py 语法错误"
        exit 1
    }
fi

# 检查前端TypeScript（如果存在且快速）
if [ -f "frontend/package.json" ] && command -v npm &> /dev/null; then
    echo "⚛️  检查TypeScript..."
    cd frontend
    timeout 30s npm run type-check || {
        echo "⚠️  TypeScript检查超时或失败，请手动检查"
    }
    cd ..
fi

echo "✅ 快速检查完成！"
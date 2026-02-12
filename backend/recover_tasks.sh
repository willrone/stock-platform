#!/bin/bash

# 5 个卡住的任务
tasks=(
    "54f14258-5406-4f3f-b854-2dce24dd9710"
    "b4da39e3-4730-4721-a157-a82d6c325c52"
    "55530c47-6b2c-4871-90a7-b4989d1d85e5"
    "9b6ebefe-9e72-485b-bd34-65f2a3b739ff"
    "5bcad3cc-e674-4701-b75a-d3697f6b0a92"
)

base_url="http://localhost:8000/api/v1"

echo "📊 开始恢复 ${#tasks[@]} 个任务..."
echo ""

recovered=0
failed=0

for task_id in "${tasks[@]}"; do
    short_id="${task_id:0:8}"
    
    # 1. 取消任务
    echo "🔄 取消任务: $short_id..."
    cancel_resp=$(curl -sS -X POST "$base_url/tasks/$task_id/cancel" -w "\n%{http_code}")
    status_code=$(echo "$cancel_resp" | tail -1)
    
    if [ "$status_code" = "200" ]; then
        echo "  ✅ 已取消"
    else
        echo "  ⚠️  取消失败: $status_code"
    fi
    
    # 2. 重试任务
    echo "🚀 重新提交任务: $short_id..."
    retry_resp=$(curl -sS -X POST "$base_url/tasks/$task_id/retry" -w "\n%{http_code}")
    status_code=$(echo "$retry_resp" | tail -1)
    
    if [ "$status_code" = "200" ]; then
        echo "  ✅ 提交成功"
        ((recovered++))
    else
        echo "  ❌ 提交失败: $status_code"
        echo "$retry_resp" | head -1
        ((failed++))
    fi
    
    echo ""
done

echo "============================================================"
echo "✅ 成功恢复: $recovered/${#tasks[@]}"
echo "❌ 失败: $failed/${#tasks[@]}"
echo "============================================================"

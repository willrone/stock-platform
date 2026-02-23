#!/bin/bash

TASK_ID="a5249513-5296-4711-8933-e74b6b525e85"
PROGRESS_FILE="/Users/ronghui/Projects/willrone/backend/PHASE2_PROGRESS.md"

echo "开始监控任务: $TASK_ID"
echo "按 Ctrl+C 停止监控"
echo ""

while true; do
    # 获取任务状态
    RESPONSE=$(curl -sS "http://localhost:8000/api/v1/tasks/$TASK_ID")
    
    # 解析关键字段
    STATUS=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data']['status'])" 2>/dev/null)
    PROGRESS=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data']['progress'])" 2>/dev/null)
    
    # 显示状态
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$TIMESTAMP] 状态: $STATUS | 进度: $PROGRESS%"
    
    # 如果任务完成，显示详细结果
    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
        echo ""
        echo "=========================================="
        echo "任务已完成！详细结果："
        echo "=========================================="
        echo "$RESPONSE" | python3 -m json.tool
        
        # 提取性能数据
        echo ""
        echo "=========================================="
        echo "性能分析："
        echo "=========================================="
        echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
result = data.get('data', {}).get('result', {})
perf = result.get('performance_breakdown', {})

if perf:
    print(f\"总耗时: {perf.get('total_time_s', 'N/A')} 秒\")
    print(f\"主循环: {perf.get('main_loop_s', 'N/A')} 秒\")
    print(f\"信号预计算: {perf.get('precompute_signals_s', 'N/A')} 秒\")
    print(f\"数组对齐: {perf.get('align_arrays_s', 'N/A')} 秒\")
    print(f\"\\n总收益率: {result.get('total_return', 'N/A')}\")
    print(f\"最终资产: {result.get('final_value', 'N/A')}\")
else:
    print('性能数据不可用')
" 2>/dev/null
        
        # 更新进度文件
        if [ "$STATUS" = "completed" ]; then
            TOTAL_TIME=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data']['result'].get('performance_breakdown', {}).get('total_time_s', 'N/A'))" 2>/dev/null)
            TOTAL_RETURN=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data']['result'].get('total_return', 'N/A'))" 2>/dev/null)
            
            cat >> "$PROGRESS_FILE" << EOF

### $(date '+%Y-%m-%d %H:%M')

**任���完成！**
- 总耗时: $TOTAL_TIME 秒
- 总收益率: $TOTAL_RETURN
- 目标达成: $(if (( $(echo "$TOTAL_TIME < 180" | bc -l 2>/dev/null || echo 0) )); then echo "✅ 是"; else echo "❌ 否"; fi)
EOF
        fi
        
        break
    fi
    
    # 等待30秒
    sleep 30
done

echo ""
echo "监控结束"

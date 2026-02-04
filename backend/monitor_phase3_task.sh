#!/bin/bash

TASK_ID="027be19e-0293-4e9d-9b34-3a8cac62d573"
START_TIME=$(date +%s)

echo "监控任务: $TASK_ID"
echo "开始时间: $(date)"
echo ""

while true; do
    # 查询任务状态
    RESPONSE=$(curl -sS "http://localhost:8000/api/v1/tasks/$TASK_ID")
    
    # 提取状态和进度
    STATUS=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data']['status'])")
    PROGRESS=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data']['progress'])")
    
    # 计算已用时间
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    echo "[${ELAPSED}s] 状态: $STATUS, 进度: $PROGRESS%"
    
    # 检查是否完成
    if [ "$STATUS" = "completed" ]; then
        echo ""
        echo "✅ 任务完成！"
        echo "总耗时: ${ELAPSED}秒"
        echo ""
        
        # 提取结果
        echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
result = data['data'].get('result', {})
print('回测结果:')
print(f\"  总收益率: {result.get('total_return_pct', 'N/A')}%\")
print(f\"  夏普比率: {result.get('sharpe_ratio', 'N/A')}\")
print(f\"  最大回撤: {result.get('max_drawdown_pct', 'N/A')}%\")
print(f\"  总交易: {result.get('total_trades', 'N/A')}\")

perf = result.get('perf_breakdown', {})
if perf:
    print('')
    print('性能分解:')
    print(f\"  数据加载: {perf.get('data_loading_s', 0):.2f}s\")
    print(f\"  信号预计算: {perf.get('precompute_signals_s', 0):.2f}s\")
    print(f\"  数组对齐: {perf.get('align_arrays_s', 0):.2f}s\")
    print(f\"  主循环: {perf.get('main_loop_s', 0):.2f}s\")
    print(f\"  指标计算: {perf.get('metrics_s', 0):.2f}s\")
    print(f\"  报告生成: {perf.get('report_generation_s', 0):.2f}s\")
    print(f\"  总耗时: {perf.get('total_wall_s', 0):.2f}s\")
"
        break
    elif [ "$STATUS" = "failed" ]; then
        echo ""
        echo "❌ 任务失败！"
        ERROR=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'].get('error_message', '未知错误'))")
        echo "错误信息: $ERROR"
        break
    fi
    
    sleep 5
done

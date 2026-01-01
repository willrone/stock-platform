#!/bin/bash

# 清理日志文件中的ANSI转义序列

LOG_FILE="/home/willrone/stock-prediction-platform/data/logs/backend.log"

if [ -f "$LOG_FILE" ]; then
    echo "清理日志文件中的ANSI转义序列..."
    # 使用sed删除ANSI转义序列
    sed -i 's/\x1b\[[0-9;]*m//g' "$LOG_FILE"
    sed -i 's/\x1b\[[0-9;]*[a-zA-Z]//g' "$LOG_FILE"
    echo "完成！已清理 $LOG_FILE"
else
    echo "日志文件不存在: $LOG_FILE"
fi


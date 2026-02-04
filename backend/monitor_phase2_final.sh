#!/bin/bash
# Phase 2 Final Validation Monitor
# Task ID: da10ac61-5828-4020-9287-e56b320b8c12

TASK_ID="da10ac61-5828-4020-9287-e56b320b8c12"
START_TIME=$(date +%s)

while true; do
    RESPONSE=$(curl -sS "http://localhost:8000/api/v1/tasks/$TASK_ID")
    STATUS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['data']['status'])")
    PROGRESS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['data']['progress'])")
    
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Status: $STATUS | Progress: $PROGRESS% | Elapsed: ${ELAPSED}s"
    
    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
        echo "Task finished with status: $STATUS"
        break
    fi
    
    sleep 30
done

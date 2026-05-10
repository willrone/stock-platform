#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

log_warning "当前版本未拆分独立 worker 进程；任务调度仍随 backend 生命周期启动。"
log_warning "如后续拆分 worker，可在此脚本中接入独立入口。"
exec tail -f /dev/null

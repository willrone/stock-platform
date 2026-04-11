#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_deprecated.sh"

warn_deprecated "scripts/simple-start.sh" "./start.sh" "旧的无 Docker / 强杀端口简单启动方案已下线，统一切到 tmux 开发流。"
exec "$PROJECT_ROOT/start.sh" "$@"

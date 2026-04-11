#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_deprecated.sh"

warn_deprecated "scripts/stop-simple.sh" "./stop.sh" "旧的 simple-start 停机脚本已下线，统一切到 tmux 开发流。"
exec "$PROJECT_ROOT/stop.sh" "$@"

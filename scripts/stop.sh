#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_deprecated.sh"

mode="${1:-dev}"
case "$mode" in
  dev|development|"")
    warn_deprecated "scripts/stop.sh" "./stop.sh" "本地开发默认入口已经切到 tmux 开发流。"
    exec "$PROJECT_ROOT/stop.sh" "${@:2}"
    ;;
  prod|production|systemd)
    warn_deprecated "scripts/stop.sh" "./scripts/prod-down.sh" "旧 Docker 停机入口不再作为默认维护路径。"
    exec "$PROJECT_ROOT/scripts/prod-down.sh"
    ;;
  help|-h|--help)
    cat <<'EOF'
Deprecated compatibility wrapper.

开发态:
  ./stop.sh

生产态:
  ./scripts/prod-down.sh
EOF
    ;;
  *)
    warn_deprecated "scripts/stop.sh" "./stop.sh 或 ./scripts/prod-down.sh"
    exec "$PROJECT_ROOT/stop.sh" "$@"
    ;;
esac

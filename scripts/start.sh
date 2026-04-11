#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_deprecated.sh"

mode="${1:-dev}"
case "$mode" in
  dev|development|"")
    warn_deprecated "scripts/start.sh" "./start.sh" "本地开发默认入口已经切到 tmux 开发流。"
    exec "$PROJECT_ROOT/start.sh" "${@:2}"
    ;;
  prod|production|systemd)
    warn_deprecated "scripts/start.sh" "./scripts/prod-build.sh && sudo ./scripts/install-systemd.sh && ./scripts/prod-up.sh" "旧 Docker 部署入口不再作为默认维护路径。"
    exec "$PROJECT_ROOT/scripts/prod-up.sh"
    ;;
  help|-h|--help)
    cat <<'EOF'
Deprecated compatibility wrapper.

开发态:
  ./start.sh

生产态:
  ./scripts/prod-build.sh
  sudo ./scripts/install-systemd.sh
  ./scripts/prod-up.sh
EOF
    ;;
  *)
    warn_deprecated "scripts/start.sh" "./start.sh 或 ./scripts/prod-up.sh" "未知参数将回退到新的默认开发入口。"
    exec "$PROJECT_ROOT/start.sh" "$@"
    ;;
esac

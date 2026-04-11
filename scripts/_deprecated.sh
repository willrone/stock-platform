#!/usr/bin/env bash
set -euo pipefail

warn_deprecated() {
  local old_path="$1"
  local new_path="$2"
  local note="${3:-}"

  echo "[DEPRECATED] $old_path 已不再是推荐入口。" >&2
  echo "[DEPRECATED] 请改用: $new_path" >&2
  if [[ -n "$note" ]]; then
    echo "[DEPRECATED] $note" >&2
  fi
  echo >&2
}

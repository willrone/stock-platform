#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-py313}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-snapshot}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SNAPSHOT_DIR="${QUALITY_SNAPSHOT_DIR:-$ROOT_DIR/reports/quality/$TIMESTAMP}"
PYTEST_TARGET="${PYTEST_TARGET:-tests/unit/infrastructure/test_basic_infrastructure.py}"
STRICT_BASELINE_MANIFEST_PATH="${STRICT_BASELINE_MANIFEST_PATH:-$ROOT_DIR/tests/golden/strict_baseline/manifest.json}"
STRICT_BASELINE_TASK_ID="${STRICT_BASELINE_TASK_ID:-}"
STRICT_BASELINE_STRATEGY="${STRICT_BASELINE_STRATEGY:-}"
STRICT_BASELINE_DB_PATH="${STRICT_BASELINE_DB_PATH:-}"
STRICT_BASELINE_NO_STRICT_HASHES="${STRICT_BASELINE_NO_STRICT_HASHES:-0}"
QUALITY_REQUIRE_MYPY="${QUALITY_REQUIRE_MYPY:-0}"

mkdir -p "$SNAPSHOT_DIR"

log() {
  printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

ensure_venv() {
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    log "创建虚拟环境: $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

ensure_quality_deps() {
  log "安装/校验质量依赖"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements-quality.txt"
}

run_and_capture() {
  local name="$1"
  shift
  local logfile="$SNAPSHOT_DIR/${name}.txt"
  log "运行 $name"
  set +e
  "$@" > >(tee "$logfile") 2>&1
  local status=$?
  set -e
  printf '%s=%s\n' "$name" "$status" >> "$SNAPSHOT_DIR/status.env"
  return "$status"
}

run_snapshot() {
  local pytest_status=0
  local flake8_status=0
  local mypy_status=0

  run_and_capture pytest "$VENV_DIR/bin/python" -m pytest "$PYTEST_TARGET" -q || pytest_status=$?
  run_and_capture flake8 "$VENV_DIR/bin/python" -m flake8 app tests || flake8_status=$?
  run_and_capture mypy "$VENV_DIR/bin/python" -m mypy app --ignore-missing-imports || mypy_status=$?

  cat > "$SNAPSHOT_DIR/summary.md" <<EOF
# Backend 质量快照

- 时间: $(date -Iseconds)
- 虚拟环境: $VENV_DIR
- pytest target: $PYTEST_TARGET
- pytest exit: $pytest_status
- flake8 exit: $flake8_status
- mypy exit: $mypy_status
- require mypy: $QUALITY_REQUIRE_MYPY

说明：
- pytest 这里默认跑 backend 基础 smoke，用来确认命令与环境可执行；
- flake8 是 snapshot 的硬门禁，用来防止基础 lint 回退；
- mypy 当前仍有存量类型债，snapshot 默认记录结果但不阻断；
- 如需把 mypy 作为硬门禁，设置 QUALITY_REQUIRE_MYPY=1 或直接运行 scripts/quality.sh mypy。
EOF

  log "质量快照输出目录: $SNAPSHOT_DIR"

  if [ "$pytest_status" -ne 0 ] || [ "$flake8_status" -ne 0 ]; then
    return 1
  fi
  if [ "$QUALITY_REQUIRE_MYPY" = "1" ] && [ "$mypy_status" -ne 0 ]; then
    return 1
  fi
}

run_strict_baseline() {
  local output_dir="$SNAPSHOT_DIR/strict-baseline"
  local -a command_args

  mkdir -p "$output_dir"
  command_args=(
    tests/scripts/run_strict_baseline_regression.py
    --manifest-path "$STRICT_BASELINE_MANIFEST_PATH"
    --summary-json "$output_dir/summary.json"
    --summary-md "$output_dir/summary.md"
    --junit-xml "$output_dir/junit.xml"
  )

  if [ -n "$STRICT_BASELINE_DB_PATH" ]; then
    command_args+=(--db-path "$STRICT_BASELINE_DB_PATH")
  fi
  if [ -n "$STRICT_BASELINE_TASK_ID" ]; then
    command_args+=(--task-id "$STRICT_BASELINE_TASK_ID")
  fi
  if [ -n "$STRICT_BASELINE_STRATEGY" ]; then
    command_args+=(--strategy "$STRICT_BASELINE_STRATEGY")
  fi
  if [ "$STRICT_BASELINE_NO_STRICT_HASHES" = "1" ]; then
    command_args+=(--no-strict-hashes)
  fi

  run_and_capture strict-baseline "$VENV_DIR/bin/python" "${command_args[@]}"
  log "strict-baseline 输出目录: $output_dir"
}

main() {
  set -e
  case "$MODE" in
    install)
      ensure_venv
      ensure_quality_deps
      ;;
    pytest)
      ensure_venv
      ensure_quality_deps
      exec "$VENV_DIR/bin/python" -m pytest "$PYTEST_TARGET" -q
      ;;
    flake8)
      ensure_venv
      ensure_quality_deps
      exec "$VENV_DIR/bin/python" -m flake8 app tests
      ;;
    mypy)
      ensure_venv
      ensure_quality_deps
      exec "$VENV_DIR/bin/python" -m mypy app --ignore-missing-imports
      ;;
    snapshot|all)
      ensure_venv
      ensure_quality_deps
      run_snapshot
      ;;
    baseline|strict-baseline)
      ensure_venv
      run_strict_baseline
      ;;
    backtest-guard|backtest-golden)
      ensure_venv
      exec "$ROOT_DIR/scripts/backtest_optimization_guard.sh"
      ;;
    *)
      echo "用法: $0 [install|pytest|flake8|mypy|snapshot|baseline|backtest-guard]" >&2
      exit 2
      ;;
  esac
}

main "$@"

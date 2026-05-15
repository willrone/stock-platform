#!/usr/bin/env bash
# Correctness guard that must pass before/after backtest performance changes.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-py313}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
CASE_NAME="${BACKTEST_GOLDEN_CASE:-all}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  echo "Set VENV_DIR or PYTHON_BIN to a prepared backend environment." >&2
  exit 2
fi

cd "$ROOT_DIR"

echo "[guard] py_compile golden tools"
"$PYTHON_BIN" -m py_compile \
  scripts/backtest_golden_runner.py \
  scripts/backtest_result_compare.py

echo "[guard] comparator unit tests"
"$PYTHON_BIN" -m pytest tests/unit/backtest/test_backtest_result_compare.py -q

echo "[guard] golden backtest verification: case=$CASE_NAME"
"$PYTHON_BIN" scripts/backtest_golden_runner.py verify --case "$CASE_NAME"

echo "[guard] PASS: backtest optimization correctness guard"

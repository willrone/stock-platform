"""Verify strict-baseline golden artifacts against SQLite tasks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from strict_baseline_common import (
    DEFAULT_MANIFEST_PATH,
    build_golden_document,
    compare_documents,
    load_task_row,
    open_db,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="校验 strict-baseline golden files")
    parser.add_argument("--db-path", type=Path, default=None, help="SQLite 数据库路径")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="manifest 路径",
    )
    parser.add_argument("--task-id", help="要校验的任务 ID；不传则执行 manifest 自检")
    parser.add_argument("--strategy", help="强制指定策略名")
    parser.add_argument(
        "--no-strict-hashes",
        action="store_true",
        help="忽略 sha256 指纹，只校验配置、指标与长度",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the manifest JSON."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_golden_document(manifest_path: Path, strategy_name: str) -> dict[str, Any]:
    """Load a strategy golden document by strategy name."""
    golden_path = manifest_path.parent / "strategies" / f"{strategy_name}.json"
    return json.loads(golden_path.read_text(encoding="utf-8"))


def resolve_strategy_name(
    manifest: dict[str, Any],
    task_id: str,
    fallback_strategy: str | None,
) -> str:
    """Resolve which strategy golden should be used for one task."""
    if fallback_strategy:
        return fallback_strategy
    for entry in manifest["task_id_mapping"]:
        if entry["task_id"] == task_id:
            return entry["strategy_name"]
    raise ValueError("请通过 --strategy 指定要比对的策略名")


def verify_one_task(
    db_path: Path | None,
    manifest_path: Path,
    task_id: str,
    strategy_name: str,
    strict_hashes: bool,
) -> list[str]:
    """Verify one task against the named golden document."""
    golden_document = load_golden_document(manifest_path, strategy_name)
    with open_db(db_path) as connection:
        row = load_task_row(connection, task_id)
        candidate_document = build_golden_document(row)
    return compare_documents(golden_document, candidate_document, strict_hashes)


def verify_manifest_self_check(
    db_path: Path | None,
    manifest: dict[str, Any],
    manifest_path: Path,
    strict_hashes: bool,
) -> dict[str, list[str]]:
    """Verify all source tasks listed in the manifest."""
    failures: dict[str, list[str]] = {}
    for entry in manifest["task_id_mapping"]:
        strategy_name = entry["strategy_name"]
        task_id = entry["task_id"]
        mismatches = verify_one_task(
            db_path=db_path,
            manifest_path=manifest_path,
            task_id=task_id,
            strategy_name=strategy_name,
            strict_hashes=strict_hashes,
        )
        if mismatches:
            failures[strategy_name] = mismatches
    return failures


def print_failures(failures: dict[str, list[str]]) -> None:
    """Print mismatch details in a readable format."""
    for strategy_name, mismatches in failures.items():
        print(f"[FAIL] {strategy_name}")
        for mismatch in mismatches:
            print(f"  - {mismatch}")


def main() -> None:
    """Run the verifier CLI."""
    args = parse_args()
    strict_hashes = not args.no_strict_hashes
    manifest = load_manifest(args.manifest_path)

    if args.task_id:
        strategy_name = resolve_strategy_name(manifest, args.task_id, args.strategy)
        mismatches = verify_one_task(
            db_path=args.db_path,
            manifest_path=args.manifest_path,
            task_id=args.task_id,
            strategy_name=strategy_name,
            strict_hashes=strict_hashes,
        )
        if mismatches:
            print_failures({strategy_name: mismatches})
            sys.exit(1)
        print(f"[PASS] {strategy_name}: {args.task_id}")
        return

    failures = verify_manifest_self_check(
        db_path=args.db_path,
        manifest=manifest,
        manifest_path=args.manifest_path,
        strict_hashes=strict_hashes,
    )
    if failures:
        print_failures(failures)
        sys.exit(1)
    print(f"[PASS] manifest self-check: {manifest['strategy_count']} strategies")


if __name__ == "__main__":
    main()

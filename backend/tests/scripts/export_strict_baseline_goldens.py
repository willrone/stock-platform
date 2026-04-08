"""Export strict-baseline golden artifacts from SQLite tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from strict_baseline_common import (
    DEFAULT_GOLDEN_DIR,
    DEFAULT_MANIFEST_PATH,
    SOURCE_TASK_IDS,
    build_golden_document,
    build_manifest,
    load_task_row,
    open_db,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="导出 strict-baseline golden files")
    parser.add_argument("--db-path", type=Path, default=None, help="SQLite 数据库路径")
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=DEFAULT_GOLDEN_DIR,
        help="golden 输出目录",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="manifest 输出路径",
    )
    return parser.parse_args()


def export_goldens(
    db_path: Path | None,
    golden_dir: Path,
    manifest_path: Path,
) -> list[dict[str, object]]:
    """Export all golden documents and return the manifest mapping rows."""
    documents: list[dict[str, object]] = []
    strategies_dir = golden_dir / "strategies"
    with open_db(db_path) as connection:
        for strategy_name, task_id in SOURCE_TASK_IDS.items():
            row = load_task_row(connection, task_id)
            document = build_golden_document(row)
            if document["strategy_name"] != strategy_name:
                raise ValueError(
                    f"策略映射不一致: {strategy_name} != {document['strategy_name']}"
                )
            write_json(strategies_dir / f"{strategy_name}.json", document)
            documents.append(document)
    manifest = build_manifest(documents)
    write_json(manifest_path, manifest)
    return manifest["task_id_mapping"]


def main() -> None:
    """Run the exporter CLI."""
    args = parse_args()
    mapping_rows = export_goldens(args.db_path, args.golden_dir, args.manifest_path)
    print(f"已导出 {len(mapping_rows)} 个 strict-baseline golden files")
    for row in mapping_rows:
        print(f"- {row['strategy_name']}: {row['task_id']}")


if __name__ == "__main__":
    main()

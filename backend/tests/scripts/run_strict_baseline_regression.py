"""Run strict-baseline regression checks with CI-friendly artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement

from strict_baseline_common import (
    DEFAULT_MANIFEST_PATH,
    build_golden_document,
    compare_documents,
    load_task_row,
    open_db,
)


@dataclass(frozen=True)
class RegressionCaseResult:
    """One strategy regression verification result."""

    strategy_name: str
    task_id: str
    mismatches: list[str]

    @property
    def passed(self) -> bool:
        """Return whether the regression case passed."""
        return not self.mismatches


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="运行 strict-baseline regression runner"
    )
    parser.add_argument("--db-path", type=Path, default=None, help="SQLite 数据库路径")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="manifest 路径",
    )
    parser.add_argument("--task-id", help="只校验一个 rerun task_id")
    parser.add_argument("--strategy", help="单任务模式下显式指定策略名")
    parser.add_argument(
        "--no-strict-hashes",
        action="store_true",
        help="忽略 sha256 指纹，只校验配置、指标与长度",
    )
    parser.add_argument("--summary-json", type=Path, default=None, help="JSON 摘要输出")
    parser.add_argument(
        "--summary-md", type=Path, default=None, help="Markdown 摘要输出"
    )
    parser.add_argument("--junit-xml", type=Path, default=None, help="JUnit XML 输出")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the strict-baseline manifest."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_single_strategy(
    manifest: dict[str, Any],
    task_id: str,
    fallback_strategy: str | None,
) -> str:
    """Resolve the strategy name for one rerun task."""
    if fallback_strategy:
        return fallback_strategy
    for entry in manifest["task_id_mapping"]:
        if entry["task_id"] == task_id:
            return entry["strategy_name"]
    raise ValueError("请通过 --strategy 指定 rerun task 对应的策略名")


def iter_requested_cases(
    manifest: dict[str, Any],
    task_id: str | None,
    strategy_name: str | None,
) -> list[tuple[str, str]]:
    """Build the strategy/task cases requested by the caller."""
    if task_id:
        return [(resolve_single_strategy(manifest, task_id, strategy_name), task_id)]
    return [
        (entry["strategy_name"], entry["task_id"])
        for entry in manifest["task_id_mapping"]
    ]


def load_golden_document(manifest_path: Path, strategy_name: str) -> dict[str, Any]:
    """Load one strategy golden document."""
    golden_path = manifest_path.parent / "strategies" / f"{strategy_name}.json"
    return json.loads(golden_path.read_text(encoding="utf-8"))


def run_one_case(
    manifest_path: Path,
    strategy_name: str,
    task_id: str,
    strict_hashes: bool,
) -> RegressionCaseResult:
    """Run one regression comparison against the target task."""
    golden_document = load_golden_document(manifest_path, strategy_name)
    with open_db() as connection:
        row = load_task_row(connection, task_id)
        candidate_document = build_golden_document(row)
    mismatches = compare_documents(golden_document, candidate_document, strict_hashes)
    return RegressionCaseResult(
        strategy_name=strategy_name,
        task_id=task_id,
        mismatches=mismatches,
    )


def run_cases(
    db_path: Path | None,
    manifest_path: Path,
    cases: list[tuple[str, str]],
    strict_hashes: bool,
) -> list[RegressionCaseResult]:
    """Run all requested regression checks."""
    results: list[RegressionCaseResult] = []
    with open_db(db_path) as connection:
        for strategy_name, task_id in cases:
            golden_document = load_golden_document(manifest_path, strategy_name)
            row = load_task_row(connection, task_id)
            candidate_document = build_golden_document(row)
            mismatches = compare_documents(
                golden_document,
                candidate_document,
                strict_hashes,
            )
            results.append(
                RegressionCaseResult(
                    strategy_name=strategy_name,
                    task_id=task_id,
                    mismatches=mismatches,
                )
            )
    return results


def build_summary(
    manifest_path: Path,
    strict_hashes: bool,
    results: list[RegressionCaseResult],
) -> dict[str, Any]:
    """Build a serializable summary for console and artifact output."""
    passed_count = sum(result.passed for result in results)
    failed_count = len(results) - passed_count
    return {
        "manifest_path": str(manifest_path),
        "strict_hashes": strict_hashes,
        "total": len(results),
        "passed": passed_count,
        "failed": failed_count,
        "results": [
            {
                "strategy_name": result.strategy_name,
                "task_id": result.task_id,
                "status": "passed" if result.passed else "failed",
                "mismatches": result.mismatches,
            }
            for result in results
        ],
    }


def ensure_parent(path: Path) -> None:
    """Create the parent directory for one output file."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    """Write the summary JSON artifact."""
    ensure_parent(path)
    content = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    path.write_text(f"{content}\n", encoding="utf-8")


def build_markdown_lines(summary: dict[str, Any]) -> list[str]:
    """Build markdown summary lines."""
    lines = [
        "# Strict Baseline Regression Summary",
        "",
        f"- manifest: `{summary['manifest_path']}`",
        f"- strict hashes: `{summary['strict_hashes']}`",
        f"- total: `{summary['total']}`",
        f"- passed: `{summary['passed']}`",
        f"- failed: `{summary['failed']}`",
        "",
    ]
    for result in summary["results"]:
        prefix = "PASS" if result["status"] == "passed" else "FAIL"
        lines.append(
            f"- [{prefix}] `{result['strategy_name']}` · `{result['task_id']}`"
        )
        for mismatch in result["mismatches"]:
            lines.append(f"  - {mismatch}")
    return lines


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    """Write the markdown summary artifact."""
    ensure_parent(path)
    content = "\n".join(build_markdown_lines(summary))
    path.write_text(f"{content}\n", encoding="utf-8")


def append_failure(test_case: Element, mismatches: list[str]) -> None:
    """Attach a failure node when one case drifts."""
    if not mismatches:
        return
    failure = SubElement(test_case, "failure", message="strict-baseline drift detected")
    failure.text = "\n".join(mismatches)


def build_testsuite(results: list[RegressionCaseResult]) -> Element:
    """Build the JUnit testsuite XML tree."""
    failures = sum(not result.passed for result in results)
    suite = Element(
        "testsuite",
        name="strict-baseline-regression",
        tests=str(len(results)),
        failures=str(failures),
        errors="0",
    )
    for result in results:
        case = SubElement(
            suite,
            "testcase",
            classname="strict_baseline",
            name=result.strategy_name,
        )
        SubElement(case, "properties")
        append_failure(case, result.mismatches)
    return suite


def write_junit_xml(path: Path, results: list[RegressionCaseResult]) -> None:
    """Write the JUnit XML artifact."""
    ensure_parent(path)
    tree = ElementTree.ElementTree(build_testsuite(results))
    tree.write(path, encoding="utf-8", xml_declaration=True)


def print_console_summary(summary: dict[str, Any]) -> None:
    """Print a readable console summary."""
    status = "PASS" if summary["failed"] == 0 else "FAIL"
    print(
        f"[{status}] strict baseline regression: "
        f"{summary['passed']}/{summary['total']} passed"
    )
    for result in summary["results"]:
        if result["status"] == "passed":
            print(f"  - [PASS] {result['strategy_name']}: {result['task_id']}")
            continue
        print(f"  - [FAIL] {result['strategy_name']}: {result['task_id']}")
        for mismatch in result["mismatches"]:
            print(f"      * {mismatch}")


def write_requested_artifacts(
    args: argparse.Namespace,
    summary: dict[str, Any],
    results: list[RegressionCaseResult],
) -> None:
    """Write all optional artifact outputs."""
    if args.summary_json:
        write_summary_json(args.summary_json, summary)
    if args.summary_md:
        write_summary_markdown(args.summary_md, summary)
    if args.junit_xml:
        write_junit_xml(args.junit_xml, results)


def main() -> None:
    """Run the strict-baseline regression runner."""
    args = parse_args()
    manifest = load_manifest(args.manifest_path)
    strict_hashes = not args.no_strict_hashes
    cases = iter_requested_cases(manifest, args.task_id, args.strategy)
    results = run_cases(args.db_path, args.manifest_path, cases, strict_hashes)
    summary = build_summary(args.manifest_path, strict_hashes, results)
    write_requested_artifacts(args, summary, results)
    print_console_summary(summary)
    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from pilot_lib import read_jsonl, write_csv


def as_bool(value):
    if value is True or value == "True" or value == "true" or value == 1:
        return True
    if value is False or value == "False" or value == "false" or value == 0:
        return False
    return None


def as_float(value, default=0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def summarize_group(rows: list[dict], label: str) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    attempted = [r for r in rows if not str(r.get("status", "")).startswith("skipped")]
    correct_values = [as_bool(r.get("correctness")) for r in ok]
    correct_values = [v for v in correct_values if v is not None]
    base_values = [as_bool(r.get("original_correctness")) for r in ok]
    base_values = [v for v in base_values if v is not None]
    improvements = 0
    regressions = 0
    preserved_correct = 0
    preserved_wrong = 0
    for row in ok:
        before = as_bool(row.get("original_correctness"))
        after = as_bool(row.get("correctness"))
        if before is False and after is True:
            improvements += 1
        elif before is True and after is False:
            regressions += 1
        elif before is True and after is True:
            preserved_correct += 1
        elif before is False and after is False:
            preserved_wrong += 1
    answer_changed = [as_bool(r.get("answer_changed")) for r in ok]
    answer_changed = [v for v in answer_changed if v is not None]
    calls = sum(int(as_float(r.get("calls"))) for r in ok)
    tokens = sum(int(as_float(r.get("total_tokens"))) for r in ok)
    latency = sum(as_float(r.get("latency_s")) for r in ok)
    accuracy = sum(correct_values) / len(correct_values) if correct_values else None
    base_accuracy = sum(base_values) / len(base_values) if base_values else None
    return {
        "group": label,
        "rows": len(rows),
        "ok_rows": len(ok),
        "attempted_rows": len(attempted),
        "coverage": len(ok) / len(rows) if rows else 0.0,
        "accuracy": accuracy,
        "original_accuracy_on_ok_rows": base_accuracy,
        "net_accuracy_delta": (accuracy - base_accuracy) if accuracy is not None and base_accuracy is not None else None,
        "improvements": improvements,
        "regressions": regressions,
        "preserved_correct": preserved_correct,
        "preserved_wrong": preserved_wrong,
        "answer_change_rate": sum(answer_changed) / len(answer_changed) if answer_changed else None,
        "calls": calls,
        "total_tokens": tokens,
        "mean_tokens_per_ok_row": tokens / len(ok) if ok else 0.0,
        "total_latency_s": latency,
        "mean_latency_s": latency / len(ok) if ok else 0.0,
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def markdown_table(rows: list[dict], cols: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_path = out_dir / "revision_results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}")
    rows = read_jsonl(results_path)
    errors = read_jsonl(out_dir / "revision_errors.jsonl") if (out_dir / "revision_errors.jsonl").exists() else []

    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    method_rows = [summarize_group(method_rows, method) for method, method_rows in sorted(by_method.items())]

    cross_rows = []
    for key_name in ["benchmark", "model"]:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row["method"], row.get(key_name, ""))].append(row)
        for (method, key), group_rows in sorted(grouped.items()):
            item = summarize_group(group_rows, f"{method}::{key}")
            item["method"] = method
            item[key_name] = key
            cross_rows.append(item)

    write_csv(out_dir / "revision_method_summary.csv", method_rows)
    write_csv(out_dir / "revision_cross_summary.csv", cross_rows)

    status_counts = Counter(r.get("status", "") for r in rows)
    error_counts = Counter((r.get("method"), r.get("model")) for r in errors)

    cols = [
        "group",
        "ok_rows",
        "coverage",
        "accuracy",
        "original_accuracy_on_ok_rows",
        "net_accuracy_delta",
        "improvements",
        "regressions",
        "answer_change_rate",
        "calls",
        "total_tokens",
        "mean_latency_s",
    ]
    report = [
        "# UATR Revision Summary",
        "",
        f"- result rows: {len(rows)}",
        f"- error rows: {len(errors)}",
        f"- status counts: {dict(status_counts)}",
        "",
        "## By Method",
        markdown_table(method_rows, cols),
        "",
        "## Error Counts",
    ]
    if error_counts:
        for key, count in error_counts.most_common():
            report.append(f"- {key}: {count}")
    else:
        report.append("- none")
    report.append("")
    report.append("## Interpretation Notes")
    report.append("- Compare each method's accuracy against `original_accuracy_on_ok_rows`, not only raw accuracy.")
    report.append("- `contradiction_only` has lower coverage by design because many high-risk rows are instability-only.")
    report.append("- UATR is successful if it improves or matches accuracy with fewer calls/tokens than generic baselines.")

    (out_dir / "revision_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(out_dir / "revision_summary.md")
    print("\n".join(report))


if __name__ == "__main__":
    main()

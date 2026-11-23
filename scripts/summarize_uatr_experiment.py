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
    ok = [row for row in rows if row.get("status") == "ok"]
    parsed = [row for row in ok if as_bool(row.get("correctness")) is not None]
    strict_correct = sum(1 for row in ok if as_bool(row.get("correctness")) is True)
    original_correct = sum(1 for row in ok if as_bool(row.get("original_correctness")) is True)
    improvements = sum(
        1
        for row in ok
        if as_bool(row.get("original_correctness")) is False and as_bool(row.get("correctness")) is True
    )
    regressions = sum(
        1
        for row in ok
        if as_bool(row.get("original_correctness")) is True and as_bool(row.get("correctness")) is False
    )
    answer_changed_values = [as_bool(row.get("answer_changed")) for row in ok]
    answer_changed_values = [value for value in answer_changed_values if value is not None]
    forced = sum(int(as_float(row.get("forced_answer_retry_count"))) for row in ok)
    calls = sum(int(as_float(row.get("calls"))) for row in ok)
    tokens = sum(int(as_float(row.get("total_tokens"))) for row in ok)
    latency = sum(as_float(row.get("latency_s")) for row in ok)
    return {
        "group": label,
        "rows": len(rows),
        "ok_rows": len(ok),
        "parsed_rows": len(parsed),
        "missing_rows": len(ok) - len(parsed),
        "coverage": len(ok) / len(rows) if rows else 0.0,
        "strict_accuracy": strict_correct / len(ok) if ok else None,
        "original_accuracy": original_correct / len(ok) if ok else None,
        "strict_net_delta": (strict_correct - original_correct) / len(ok) if ok else None,
        "improvements": improvements,
        "regressions": regressions,
        "answer_change_rate": sum(answer_changed_values) / len(answer_changed_values) if answer_changed_values else None,
        "forced_answer_retries": forced,
        "calls": calls,
        "total_tokens": tokens,
        "mean_tokens_per_ok_row": tokens / len(ok) if ok else 0.0,
        "mean_latency_s": latency / len(ok) if ok else 0.0,
    }


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    results_path = out_dir / "uatr_results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}")
    rows = read_jsonl(results_path)
    errors = read_jsonl(out_dir / "uatr_errors.jsonl") if (out_dir / "uatr_errors.jsonl").exists() else []

    by_method: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    method_rows = [summarize_group(group, method) for method, group in sorted(by_method.items())]

    cross_rows = []
    for key_name in ["model", "benchmark", "trigger_mode"]:
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            grouped[(row["method"], str(row.get(key_name, "")))].append(row)
        for (method, key), group in sorted(grouped.items()):
            item = summarize_group(group, f"{method}::{key_name}::{key}")
            item["method"] = method
            item[key_name] = key
            cross_rows.append(item)

    write_csv(out_dir / "uatr_method_summary.csv", method_rows)
    write_csv(out_dir / "uatr_cross_summary.csv", cross_rows)

    cols = [
        "group",
        "ok_rows",
        "parsed_rows",
        "missing_rows",
        "strict_accuracy",
        "original_accuracy",
        "strict_net_delta",
        "improvements",
        "regressions",
        "answer_change_rate",
        "forced_answer_retries",
        "calls",
        "total_tokens",
        "mean_latency_s",
    ]
    error_counts = Counter((row.get("method"), row.get("model")) for row in errors)
    report = [
        "# UATR Strict Revision Summary",
        "",
        f"- result rows: {len(rows)}",
        f"- error rows: {len(errors)}",
        f"- status counts: {dict(Counter(row.get('status') for row in rows))}",
        "",
        "## By Method",
        table(method_rows, cols),
        "",
        "## Error Counts",
    ]
    if error_counts:
        for key, count in error_counts.most_common():
            report.append(f"- {key}: {count}")
    else:
        report.append("- none")
    report.extend(
        [
            "",
            "## Scoring Rule",
            "- `strict_accuracy` counts missing/unparsed final answers as incorrect.",
            "- `forced_answer_retries` reports how often the uniform answer-format repair pass was used.",
            "- The UATR claim should compare strict net delta and calls/tokens against random-step and generic baselines.",
        ]
    )
    text = "\n".join(report)
    (out_dir / "uatr_summary.md").write_text(text, encoding="utf-8")
    print(out_dir / "uatr_summary.md")
    print(text)


if __name__ == "__main__":
    main()

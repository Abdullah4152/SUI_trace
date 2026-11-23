from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value):
    if value is True or value == "True" or value == "true" or value == 1:
        return True
    if value is False or value == "False" or value == "false" or value == 0:
        return False
    return None


def fnum(value, default=0.0) -> float:
    try:
        if value == "" or value is None:
            return default
        return float(value)
    except Exception:
        return default


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def bootstrap_ci(values: list[float], *, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    boots = [mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot)]
    return percentile(boots, 0.025), percentile(boots, 0.975)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def sign_test_p_value(plus: int, minus: int) -> float:
    n = plus + minus
    if n == 0:
        return 1.0
    k = min(plus, minus)
    # two-sided exact binomial p under p=0.5
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * prob)


def mcnemar_exact_p(b01: int, b10: int) -> float:
    return sign_test_p_value(b01, b10)


def utility_stats(sui_dir: Path, out_dir: Path) -> list[dict]:
    rows = read_jsonl(sui_dir / "pilot_utility_scores.jsonl")
    valid = [row for row in rows if row.get("valid_for_utility") is True]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in valid:
        groups[("overall", "all")].append(row)
        groups[("model", row["model"])].append(row)
        groups[("benchmark", row["benchmark"])].append(row)
        groups[("step_bin", row["step_bin"])].append(row)

    output = []
    for (group_type, group), group_rows in sorted(groups.items()):
        n = len(group_rows)
        decision = sum(1 for row in group_rows if row.get("step_bin") == "decision-critical")
        supportive = sum(1 for row in group_rows if row.get("step_bin") == "supportive")
        redundant = sum(1 for row in group_rows if row.get("step_bin") == "redundant")
        delta_ans_vals = [fnum(row.get("delta_ans")) for row in group_rows]
        delta_corr_vals = [fnum(row.get("delta_corr")) for row in group_rows]
        utility_vals = [fnum(row.get("utility_score")) for row in group_rows]
        decision_ci = wilson_ci(decision, n)
        output.append(
            {
                "group_type": group_type,
                "group": group,
                "n": n,
                "decision_critical_n": decision,
                "decision_critical_rate": decision / n if n else "",
                "decision_critical_ci_low": decision_ci[0],
                "decision_critical_ci_high": decision_ci[1],
                "supportive_n": supportive,
                "supportive_rate": supportive / n if n else "",
                "redundant_n": redundant,
                "redundant_rate": redundant / n if n else "",
                "mean_delta_ans": mean(delta_ans_vals),
                "mean_delta_corr": mean(delta_corr_vals),
                "mean_utility_score": mean(utility_vals),
            }
        )
    write_csv(out_dir / "sui_utility_ci_summary.csv", output)
    return output


def revision_stats(result_dir: Path, label: str, out_dir: Path) -> list[dict]:
    path = result_dir / "uatr_results.jsonl"
    if not path.exists():
        return []
    rows = read_jsonl(path)
    by_job: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row.get("status") == "ok":
            by_job[row["job_id"]][row["method"]] = row

    methods = sorted({row["method"] for row in rows})
    output = []
    for method in methods:
        method_rows = [row for row in rows if row.get("method") == method and row.get("status") == "ok"]
        if not method_rows:
            continue
        strict_values = [1.0 if as_bool(row.get("correctness")) is True else 0.0 for row in method_rows]
        orig_values = [1.0 if as_bool(row.get("original_correctness")) is True else 0.0 for row in method_rows]
        diffs = [a - b for a, b in zip(strict_values, orig_values)]
        acc_ci = bootstrap_ci(strict_values, seed=20260517 + len(method))
        diff_ci = bootstrap_ci(diffs, seed=20260617 + len(method))
        plus = sum(1 for d in diffs if d > 0)
        minus = sum(1 for d in diffs if d < 0)
        output.append(
            {
                "experiment": label,
                "method": method,
                "n": len(method_rows),
                "strict_accuracy": mean(strict_values),
                "strict_accuracy_ci_low": acc_ci[0],
                "strict_accuracy_ci_high": acc_ci[1],
                "original_accuracy_same_rows": mean(orig_values),
                "net_delta": mean(diffs),
                "net_delta_ci_low": diff_ci[0],
                "net_delta_ci_high": diff_ci[1],
                "improvements": plus,
                "regressions": minus,
                "sign_test_p_vs_original": sign_test_p_value(plus, minus),
                "calls": sum(int(fnum(row.get("calls"))) for row in method_rows),
                "total_tokens": sum(int(fnum(row.get("total_tokens"))) for row in method_rows),
            }
        )

    comparisons = []
    baseline_methods = [method for method in methods if method not in {"original"}]
    for a, b in [
        ("uatr", "random_step"),
        ("uatr", "self_refine2"),
        ("uatr", "debate2x2"),
        ("uatr", "self_consistency"),
        ("random_step", "self_refine2"),
        ("debate2x2", "self_consistency"),
    ]:
        if a not in baseline_methods or b not in baseline_methods:
            continue
        paired = []
        for job_id, methods_for_job in by_job.items():
            if a in methods_for_job and b in methods_for_job:
                ca = 1 if as_bool(methods_for_job[a].get("correctness")) is True else 0
                cb = 1 if as_bool(methods_for_job[b].get("correctness")) is True else 0
                paired.append((ca, cb))
        if not paired:
            continue
        b01 = sum(1 for ca, cb in paired if ca == 0 and cb == 1)
        b10 = sum(1 for ca, cb in paired if ca == 1 and cb == 0)
        diffs = [ca - cb for ca, cb in paired]
        diff_ci = bootstrap_ci(diffs, seed=20260717 + len(a) + len(b))
        comparisons.append(
            {
                "experiment": label,
                "method_a": a,
                "method_b": b,
                "paired_n": len(paired),
                "accuracy_delta_a_minus_b": mean(diffs),
                "delta_ci_low": diff_ci[0],
                "delta_ci_high": diff_ci[1],
                "b_wins_a_loses": b01,
                "a_wins_b_loses": b10,
                "mcnemar_exact_p": mcnemar_exact_p(b01, b10),
            }
        )
    write_csv(out_dir / f"{label}_revision_ci_summary.csv", output)
    write_csv(out_dir / f"{label}_revision_pairwise_tests.csv", comparisons)
    return output + comparisons


def format_float(value) -> str:
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def markdown_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(col)) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sui-dir", required=True)
    parser.add_argument("--uatr-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sui_summary = utility_stats(Path(args.sui_dir), out_dir)
    uatr = revision_stats(Path(args.uatr_dir) / "final", "uatr", out_dir)

    uatr_methods = [row for row in uatr if row.get("method")]
    overall_sui = [row for row in sui_summary if row["group_type"] == "overall"]

    report = [
        "# Confidence Intervals and Significance Tests",
        "",
        "## SUI Utility Summary",
        markdown_table(
            overall_sui,
            [
                "group",
                "n",
                "decision_critical_rate",
                "decision_critical_ci_low",
                "decision_critical_ci_high",
                "supportive_rate",
                "redundant_rate",
                "mean_delta_ans",
                "mean_delta_corr",
            ],
        ),
        "",
        "## UATR Revision",
        markdown_table(
            uatr_methods,
            [
                "method",
                "n",
                "strict_accuracy",
                "strict_accuracy_ci_low",
                "strict_accuracy_ci_high",
                "net_delta",
                "net_delta_ci_low",
                "net_delta_ci_high",
                "sign_test_p_vs_original",
                "calls",
            ],
        ),
        "",
        "## Test Definitions",
        "- Utility proportions use Wilson 95% confidence intervals.",
        "- Mean revision accuracies and deltas use nonparametric bootstrap 95% confidence intervals.",
        "- Per-method improvement over original uses an exact two-sided sign test over changed outcomes.",
        "- Pairwise method comparisons use exact McNemar/sign tests over paired job outcomes.",
    ]
    (out_dir / "confidence_intervals_and_tests.md").write_text("\n".join(report), encoding="utf-8")
    print(out_dir / "confidence_intervals_and_tests.md")
    print("\n".join(report))


if __name__ == "__main__":
    main()

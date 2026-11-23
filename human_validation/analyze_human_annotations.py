from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


FIELDS = [
    "semantic_change_score",
    "local_coherence_score",
    "fluency_score",
    "answer_leakage_label",
    "over_disruption_label",
    "valid_intervention_label",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_int(value):
    try:
        if value == "" or value is None:
            return None
        return int(float(value))
    except Exception:
        return None


def nominal_alpha(items: dict[str, list[int]]) -> float:
    values = [v for vals in items.values() for v in vals if v is not None]
    if len(values) < 2:
        return float("nan")
    categories = sorted(set(values))
    total_pairs = 0
    observed_disagreement = 0
    for vals in items.values():
        vals = [v for v in vals if v is not None]
        for a, b in itertools.combinations(vals, 2):
            total_pairs += 1
            observed_disagreement += 0 if a == b else 1
    if total_pairs == 0:
        return float("nan")
    do = observed_disagreement / total_pairs
    counts = Counter(values)
    n = len(values)
    expected = 0.0
    for a in categories:
        for b in categories:
            if a != b:
                expected += counts[a] * counts[b]
    de = expected / (n * (n - 1)) if n > 1 else 0
    if de == 0:
        return 1.0 if do == 0 else float("nan")
    return 1 - do / de


def interval_alpha(items: dict[str, list[int]]) -> float:
    values = [v for vals in items.values() for v in vals if v is not None]
    if len(values) < 2:
        return float("nan")
    total_pairs = 0
    observed = 0.0
    for vals in items.values():
        vals = [v for v in vals if v is not None]
        for a, b in itertools.combinations(vals, 2):
            total_pairs += 1
            observed += (a - b) ** 2
    if total_pairs == 0:
        return float("nan")
    do = observed / total_pairs
    mean = sum(values) / len(values)
    de = sum((v - mean) ** 2 for v in values) * 2 / (len(values) - 1) if len(values) > 1 else 0
    if de == 0:
        return 1.0 if do == 0 else float("nan")
    return 1 - do / de


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="main")
    args = parser.parse_args()

    rows = [row for row in read_csv(Path(args.annotations)) if row.get("split") == args.split]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for field in FIELDS:
        by_item: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            value = as_int(row.get(field))
            if value is not None:
                by_item[row["validation_item_id"]].append(value)
        alpha = nominal_alpha(by_item) if field in {"valid_intervention_label"} else interval_alpha(by_item)
        flat = [v for vals in by_item.values() for v in vals]
        report_rows.append(
            {
                "field": field,
                "split": args.split,
                "rated_items": len(by_item),
                "ratings": len(flat),
                "alpha": alpha,
                "mean": sum(flat) / len(flat) if flat else "",
                "distribution": dict(Counter(flat)),
            }
        )

    valid_items = defaultdict(list)
    leakage_items = defaultdict(list)
    over_items = defaultdict(list)
    for row in rows:
        item = row["validation_item_id"]
        valid_items[item].append(as_int(row.get("valid_intervention_label")))
        leakage_items[item].append(as_int(row.get("answer_leakage_label")))
        over_items[item].append(as_int(row.get("over_disruption_label")))

    def majority(vals: list[int | None]):
        clean = [v for v in vals if v is not None]
        if not clean:
            return None
        counts = Counter(clean)
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    item_summary = []
    for item_id in sorted(valid_items):
        valid = majority(valid_items[item_id])
        leakage = majority(leakage_items[item_id])
        over = majority(over_items[item_id])
        item_summary.append(
            {
                "validation_item_id": item_id,
                "majority_valid_intervention_label": valid,
                "majority_answer_leakage_label": leakage,
                "majority_over_disruption_label": over,
            }
        )

    valid_rate = sum(1 for row in item_summary if row["majority_valid_intervention_label"] == 2) / len(item_summary) if item_summary else float("nan")
    leakage_rate = sum(1 for row in item_summary if row["majority_answer_leakage_label"] == 2) / len(item_summary) if item_summary else float("nan")
    over_rate = sum(1 for row in item_summary if row["majority_over_disruption_label"] == 2) / len(item_summary) if item_summary else float("nan")

    with (out_dir / "human_annotation_reliability.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["field", "split", "rated_items", "ratings", "alpha", "mean", "distribution"])
        writer.writeheader()
        writer.writerows(report_rows)
    with (out_dir / "human_annotation_item_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(item_summary[0].keys()) if item_summary else ["validation_item_id"])
        writer.writeheader()
        writer.writerows(item_summary)

    report = {
        "split": args.split,
        "items": len(item_summary),
        "majority_valid_rate": valid_rate,
        "majority_clear_answer_leakage_rate": leakage_rate,
        "majority_clear_over_disruption_rate": over_rate,
        "reliability": report_rows,
    }
    (out_dir / "human_annotation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

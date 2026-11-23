from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


MODEL_FILTERED_STEMS = {
    "pilot_trace_metadata",
    "pilot_sampled_steps",
    "pilot_interventions",
    "pilot_utility_scores",
    "pilot_uatr_region_labels",
    "pilot_intervention_errors",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if isinstance(value, (dict, list)):
                    out[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(out)


def dedupe_by_key(rows: list[dict], key: str) -> list[dict]:
    latest = {}
    for row in rows:
        latest[str(row.get(key, ""))] = row
    return list(latest.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="+", required=True, help="Model display names to keep.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(args.models)

    for path in in_dir.glob("*"):
        if path.is_dir():
            continue
        stem = path.stem
        if path.suffix not in {".jsonl", ".csv"}:
            continue
        if stem == "pilot_editor_comparison":
            continue
        if stem == "pilot_instances":
            shutil.copy2(path, out_dir / path.name)
            continue
        if stem not in MODEL_FILTERED_STEMS:
            shutil.copy2(path, out_dir / path.name)
            continue
        if path.suffix != ".jsonl":
            continue

        rows = read_jsonl(path)
        rows = [row for row in rows if row.get("model") in keep]
        if stem == "pilot_trace_metadata":
            rows = dedupe_by_key(rows, "trace_id")
        write_jsonl(out_dir / f"{stem}.jsonl", rows)
        write_csv(out_dir / f"{stem}.csv", rows)

    print(f"Wrote filtered output to {out_dir}")
    print(f"Kept models: {', '.join(sorted(keep))}")


if __name__ == "__main__":
    main()

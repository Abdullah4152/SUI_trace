from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_id(*parts: str) -> str:
    return hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()[:12]


def as_text(value, max_chars: int = 1800) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > max_chars:
        return text[: max_chars - 20] + "\n...[truncated]"
    return text


def sample_balanced(rows: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_bin: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bin[row["step_bin"]].append(row)
    bins = ["decision-critical", "supportive", "redundant", "invalid-edit"]
    target_each = n // len(bins)
    selected: list[dict] = []
    leftovers: list[dict] = []
    for bin_name in bins:
        group = by_bin.get(bin_name, [])
        rng.shuffle(group)
        take = min(target_each, len(group))
        selected.extend(group[:take])
        leftovers.extend(group[take:])
    while len(selected) < n and leftovers:
        idx = rng.randrange(len(leftovers))
        selected.append(leftovers.pop(idx))
    rng.shuffle(selected)
    return selected[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sui-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--main-n", type=int, default=180)
    parser.add_argument("--calibration-n", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args()

    sui_dir = Path(args.sui_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampled_steps = {
        (row["model"], row["instance_id"], int(row["step_id"])): row
        for row in read_jsonl(sui_dir / "pilot_sampled_steps.jsonl")
    }
    interventions_all = read_jsonl(sui_dir / "pilot_interventions.jsonl")
    utility_rows = read_jsonl(sui_dir / "pilot_utility_scores.jsonl")
    region_rows = {
        (row["model"], row["instance_id"], int(row["step_id"])): row
        for row in read_jsonl(sui_dir / "pilot_uatr_region_labels.jsonl")
    }

    final_intervention: dict[tuple[str, str, int], dict] = {}
    for row in interventions_all:
        key = (row["model"], row["instance_id"], int(row["step_id"]))
        prev = final_intervention.get(key)
        if prev is None or int(row.get("attempt_id") or 0) >= int(prev.get("attempt_id") or 0):
            final_intervention[key] = row

    candidates: list[dict] = []
    for util in utility_rows:
        key = (util["model"], util["instance_id"], int(util["step_id"]))
        intervention = final_intervention.get(key)
        sampled = sampled_steps.get(key, {})
        region = region_rows.get(key, {})
        if intervention is None:
            continue
        step_bin = util.get("step_bin") or "unknown"
        candidates.append(
            {
                "validation_item_id": "HV-" + stable_id(util["model"], util["instance_id"], str(util["step_id"])),
                "model": util["model"],
                "benchmark": util["benchmark"],
                "instance_id": util["instance_id"],
                "step_id": int(util["step_id"]),
                "step_bin": step_bin,
                "norm_position": util.get("norm_position", ""),
                "valid_for_utility_auto": util.get("valid_for_utility", ""),
                "utility_score": util.get("utility_score", ""),
                "delta_ans": util.get("delta_ans", ""),
                "delta_corr": util.get("delta_corr", ""),
                "delta_cont": util.get("delta_cont", ""),
                "contradiction_flag_auto": region.get("contradiction_flag", ""),
                "instability_flag_auto": region.get("instability_flag", ""),
                "answer_critical_flag_auto": region.get("answer_critical_flag", ""),
                "original_step": as_text(intervention.get("original_step")),
                "edited_step": as_text(intervention.get("edited_step")),
                "baseline_answer": intervention.get("baseline_answer", ""),
                "edited_answer": intervention.get("edited_answer", ""),
                "baseline_response_tail": as_text(intervention.get("baseline_response_tail"), 900),
                "edited_response_tail": as_text(intervention.get("edited_response_tail"), 900),
                "segmentation_method": sampled.get("segmentation_method", ""),
                "num_trace_steps": sampled.get("num_trace_steps", ""),
                "stratum": step_bin,
            }
        )

    rng = random.Random(args.seed)
    calibration = sample_balanced(candidates, args.calibration_n, args.seed + 1)
    calibration_ids = {row["validation_item_id"] for row in calibration}
    remaining = [row for row in candidates if row["validation_item_id"] not in calibration_ids]
    main = sample_balanced(remaining, args.main_n, args.seed + 2)

    for row in calibration:
        row["split"] = "calibration"
    for row in main:
        row["split"] = "main"
    sample = calibration + main

    annotation_fields = [
        "validation_item_id",
        "split",
        "rater_id",
        "model",
        "benchmark",
        "instance_id",
        "step_id",
        "step_bin",
        "stratum",
        "original_step",
        "edited_step",
        "baseline_response_tail",
        "edited_response_tail",
        "semantic_change_score",
        "local_coherence_score",
        "fluency_score",
        "answer_leakage_label",
        "over_disruption_label",
        "valid_intervention_label",
        "human_failure_mode",
        "comment",
    ]
    annotation_rows = []
    for row in sample:
        annotation_rows.append({**row, "rater_id": ""})

    metadata_fields = [
        "validation_item_id",
        "split",
        "model",
        "benchmark",
        "instance_id",
        "step_id",
        "step_bin",
        "norm_position",
        "valid_for_utility_auto",
        "utility_score",
        "delta_ans",
        "delta_corr",
        "delta_cont",
        "contradiction_flag_auto",
        "instability_flag_auto",
        "answer_critical_flag_auto",
        "baseline_answer",
        "edited_answer",
        "segmentation_method",
        "num_trace_steps",
        "stratum",
    ]

    write_csv(out_dir / "human_validation_annotation_sheet.csv", annotation_rows, annotation_fields)
    write_csv(out_dir / "human_validation_metadata.csv", sample, metadata_fields)
    write_jsonl(out_dir / "human_validation_sample.jsonl", sample)

    counts = {
        "total_candidates": len(candidates),
        "sample_total": len(sample),
        "calibration_n": len(calibration),
        "main_n": len(main),
        "sample_by_split": Counter(row["split"] for row in sample),
        "sample_by_step_bin": Counter(row["step_bin"] for row in sample),
        "sample_by_model": Counter(row["model"] for row in sample),
        "sample_by_benchmark": Counter(row["benchmark"] for row in sample),
        "full_by_step_bin": Counter(row["step_bin"] for row in candidates),
        "full_by_model": Counter(row["model"] for row in candidates),
        "full_by_benchmark": Counter(row["benchmark"] for row in candidates),
    }
    serializable = {
        key: (dict(value) if isinstance(value, Counter) else value)
        for key, value in counts.items()
    }
    (out_dir / "human_validation_sample_summary.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(serializable, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

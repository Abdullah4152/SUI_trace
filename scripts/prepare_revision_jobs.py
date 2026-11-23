from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from pilot_lib import read_jsonl, split_trace_steps, trace_text, write_csv, write_jsonl


def stable_seed(*parts: str) -> int:
    text = "::".join(parts)
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def fnum(value) -> float:
    try:
        if value == "" or value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def sort_region_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            bool(r.get("answer_critical_flag")),
            bool(r.get("contradiction_flag")),
            bool(r.get("instability_flag")),
            fnum(r.get("utility_score")),
        ),
        reverse=True,
    )


def compact_step(row: dict, steps: list[str]) -> dict:
    step_id = int(row.get("step_id", -1))
    step_text = steps[step_id] if 0 <= step_id < len(steps) else ""
    return {
        "step_id": step_id,
        "utility_score": row.get("utility_score", ""),
        "contradiction_flag": bool(row.get("contradiction_flag")),
        "contradiction_source": row.get("contradiction_source", ""),
        "instability_flag": bool(row.get("instability_flag")),
        "answer_critical_flag": bool(row.get("answer_critical_flag")),
        "negative_support_signals": row.get("negative_support_signals", ""),
        "step_text": step_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sui-dir", required=True, help="Folder containing frozen SUI output files.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--max-per-model-benchmark", type=int, default=None)
    parser.add_argument("--max-flagged-steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260514)
    args = parser.parse_args()

    sui_dir = Path(args.sui_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = {row["instance_id"]: row for row in read_jsonl(sui_dir / "pilot_instances.jsonl")}
    traces = {
        (row["model"], row["instance_id"]): row
        for row in read_jsonl(sui_dir / "pilot_trace_metadata.jsonl")
        if row.get("status") == "ok"
    }
    region_rows = read_jsonl(sui_dir / "pilot_uatr_region_labels.jsonl")

    selected_by_trace: dict[tuple[str, str], list[dict]] = {}
    valid_by_trace: dict[tuple[str, str], list[dict]] = {}
    contradiction_by_trace: dict[tuple[str, str], list[dict]] = {}
    for row in region_rows:
        key = (row["model"], row["instance_id"])
        if row.get("valid_for_utility") is True:
            valid_by_trace.setdefault(key, []).append(row)
        if row.get("selected_for_revision") is True:
            selected_by_trace.setdefault(key, []).append(row)
            if row.get("contradiction_flag") is True:
                contradiction_by_trace.setdefault(key, []).append(row)

    jobs: list[dict] = []
    for key, selected_rows in selected_by_trace.items():
        trace = traces.get(key)
        if trace is None:
            continue
        inst = instances.get(key[1])
        if inst is None:
            continue

        steps, segmentation_method = split_trace_steps(trace_text(trace))
        selected_sorted = sort_region_rows(selected_rows)[: max(1, args.max_flagged_steps)]
        valid_candidates = valid_by_trace.get(key, [])
        selected_ids = {int(r.get("step_id", -1)) for r in selected_rows}
        random_pool = [r for r in valid_candidates if int(r.get("step_id", -1)) not in selected_ids]
        if not random_pool:
            random_pool = valid_candidates or selected_rows
        rng = random.Random(stable_seed(str(args.seed), key[0], key[1]))
        random_row = rng.choice(random_pool) if random_pool else selected_sorted[0]

        contradiction_sorted = sort_region_rows(contradiction_by_trace.get(key, []))
        job_id = f"{key[0]}::{key[1]}"
        jobs.append(
            {
                "job_id": job_id,
                "model": trace["model"],
                "model_id": trace["model_id"],
                "benchmark": trace["benchmark"],
                "instance_id": trace["instance_id"],
                "trace_id": trace["trace_id"],
                "original_answer": trace.get("answer", ""),
                "original_correctness": bool(trace.get("correctness")),
                "gold_answer": inst.get("gold_answer", trace.get("gold_answer", "")),
                "num_trace_steps": len(steps),
                "segmentation_method": segmentation_method,
                "uatr_steps": [compact_step(row, steps) for row in selected_sorted],
                "random_steps": [compact_step(random_row, steps)],
                "contradiction_steps": [
                    compact_step(row, steps)
                    for row in contradiction_sorted[: max(1, args.max_flagged_steps)]
                ],
                "selected_step_ids": [int(row.get("step_id", -1)) for row in selected_sorted],
                "random_step_ids": [int(random_row.get("step_id", -1))],
                "contradiction_step_ids": [int(row.get("step_id", -1)) for row in contradiction_sorted],
                "selection_reason": "selected_for_revision/high_risk",
            }
        )

    rng = random.Random(args.seed)
    jobs.sort(key=lambda r: (r["model"], r["benchmark"], r["instance_id"]))
    if args.max_per_model_benchmark is not None:
        groups: dict[tuple[str, str], list[dict]] = {}
        for job in jobs:
            groups.setdefault((job["model"], job["benchmark"]), []).append(job)
        capped: list[dict] = []
        for group_jobs in groups.values():
            rng.shuffle(group_jobs)
            capped.extend(group_jobs[: args.max_per_model_benchmark])
        jobs = sorted(capped, key=lambda r: (r["model"], r["benchmark"], r["instance_id"]))
    if args.max_jobs is not None:
        rng.shuffle(jobs)
        jobs = jobs[: args.max_jobs]
        jobs.sort(key=lambda r: (r["model"], r["benchmark"], r["instance_id"]))

    write_jsonl(out_dir / "revision_jobs.jsonl", jobs)
    csv_rows = []
    for job in jobs:
        csv_rows.append(
            {
                "job_id": job["job_id"],
                "model": job["model"],
                "benchmark": job["benchmark"],
                "instance_id": job["instance_id"],
                "original_answer": job["original_answer"],
                "original_correctness": job["original_correctness"],
                "gold_answer": job["gold_answer"],
                "num_trace_steps": job["num_trace_steps"],
                "uatr_step_ids": ",".join(str(x) for x in job["selected_step_ids"]),
                "random_step_ids": ",".join(str(x) for x in job["random_step_ids"]),
                "contradiction_step_ids": ",".join(str(x) for x in job["contradiction_step_ids"]),
            }
        )
    write_csv(out_dir / "revision_jobs.csv", csv_rows)
    print(f"Wrote {len(jobs)} revision jobs to {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from pilot_lib import read_jsonl, split_trace_steps, trace_text, write_csv, write_jsonl


def fnum(value) -> float:
    try:
        if value == "" or value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def stable_seed(*parts: str) -> int:
    return int(hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def is_uatr_trigger(row: dict, utility_threshold: float) -> bool:
    if row.get("valid_for_utility") is not True:
        return False

    utility_ok = fnum(row.get("utility_score")) >= utility_threshold
    answer_critical = row.get("answer_critical_flag") is True
    contradiction = row.get("contradiction_flag") is True
    unstable = row.get("instability_flag") is True
    return answer_critical or (contradiction and unstable and utility_ok)


def sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            bool(r.get("answer_critical_flag")),
            fnum(r.get("utility_score")),
            bool(r.get("instability_flag")),
        ),
        reverse=True,
    )


def compact_step(row: dict, steps: list[str]) -> dict:
    step_id = int(row.get("step_id", -1))
    text = steps[step_id] if 0 <= step_id < len(steps) else ""
    return {
        "step_id": step_id,
        "utility_score": row.get("utility_score", ""),
        "contradiction_flag": bool(row.get("contradiction_flag")),
        "contradiction_source": row.get("contradiction_source", ""),
        "instability_flag": bool(row.get("instability_flag")),
        "answer_critical_flag": bool(row.get("answer_critical_flag")),
        "negative_support_signals": row.get("negative_support_signals", ""),
        "step_text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sui-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--utility-threshold", type=float, default=1.0)
    parser.add_argument("--max-trigger-steps", type=int, default=2)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--max-per-model-benchmark", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260515)
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
    labels = read_jsonl(sui_dir / "pilot_uatr_region_labels.jsonl")

    valid_by_trace: dict[tuple[str, str], list[dict]] = {}
    trigger_by_trace: dict[tuple[str, str], list[dict]] = {}
    for row in labels:
        key = (row["model"], row["instance_id"])
        if row.get("valid_for_utility") is True:
            valid_by_trace.setdefault(key, []).append(row)
        if is_uatr_trigger(row, args.utility_threshold):
            trigger_by_trace.setdefault(key, []).append(row)

    jobs: list[dict] = []
    for key, trigger_rows in sorted(trigger_by_trace.items()):
        trace = traces.get(key)
        inst = instances.get(key[1])
        if trace is None or inst is None:
            continue
        steps, segmentation_method = split_trace_steps(trace_text(trace))
        trigger_rows = sort_rows(trigger_rows)[: max(1, args.max_trigger_steps)]
        trigger_ids = {int(row.get("step_id", -1)) for row in trigger_rows}
        random_pool = [
            row
            for row in valid_by_trace.get(key, [])
            if int(row.get("step_id", -1)) not in trigger_ids
        ]
        if not random_pool:
            random_pool = valid_by_trace.get(key, trigger_rows)
        rng = random.Random(stable_seed(str(args.seed), "uatr", key[0], key[1]))
        random_rows = rng.sample(random_pool, min(len(trigger_rows), len(random_pool)))

        jobs.append(
            {
                "job_id": f"uatr::{key[0]}::{key[1]}",
                "trigger_mode": "uatr",
                "utility_threshold": args.utility_threshold,
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
                "uatr_steps": [compact_step(row, steps) for row in trigger_rows],
                "random_steps": [compact_step(row, steps) for row in random_rows],
                "trigger_step_ids": [int(row.get("step_id", -1)) for row in trigger_rows],
                "random_step_ids": [int(row.get("step_id", -1)) for row in random_rows],
            }
        )

    rng = random.Random(args.seed)
    if args.max_per_model_benchmark is not None:
        groups: dict[tuple[str, str], list[dict]] = {}
        for job in jobs:
            groups.setdefault((job["model"], job["benchmark"]), []).append(job)
        capped: list[dict] = []
        for group_jobs in groups.values():
            rng.shuffle(group_jobs)
            capped.extend(group_jobs[: args.max_per_model_benchmark])
        jobs = sorted(capped, key=lambda row: (row["model"], row["benchmark"], row["instance_id"]))
    if args.max_jobs is not None:
        rng.shuffle(jobs)
        jobs = sorted(jobs[: args.max_jobs], key=lambda row: (row["model"], row["benchmark"], row["instance_id"]))

    write_jsonl(out_dir / "uatr_jobs.jsonl", jobs)
    csv_rows = [
        {
            "job_id": job["job_id"],
            "trigger_mode": job["trigger_mode"],
            "model": job["model"],
            "benchmark": job["benchmark"],
            "instance_id": job["instance_id"],
            "original_answer": job["original_answer"],
            "original_correctness": job["original_correctness"],
            "gold_answer": job["gold_answer"],
            "trigger_step_ids": ",".join(str(x) for x in job["trigger_step_ids"]),
            "random_step_ids": ",".join(str(x) for x in job["random_step_ids"]),
        }
        for job in jobs
    ]
    write_csv(out_dir / "uatr_jobs.csv", csv_rows)
    print(f"Wrote {len(jobs)} UATR jobs to {out_dir}")
    print(f"trigger=uatr, utility_threshold={args.utility_threshold}")


if __name__ == "__main__":
    main()

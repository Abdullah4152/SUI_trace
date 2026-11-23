from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pilot_lib import load_config, read_jsonl
from run_pilot_interventions import (
    build_step_jobs,
    filter_traces_for_target_panel,
    job_key_from_job,
    job_key_from_utility,
)


def read_jsonl_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="*", help="Optional target model names or model IDs.")
    args = parser.parse_args()

    config = load_config(args.config)
    instances = read_jsonl(args.instances)
    traces = filter_traces_for_target_panel(read_jsonl(args.traces), config, args.models)
    jobs = build_step_jobs(instances, traces, config, None)

    out_dir = Path(args.out_dir)
    utilities = read_jsonl_if_exists(out_dir / "pilot_utility_scores.jsonl")
    errors = read_jsonl_if_exists(out_dir / "pilot_intervention_errors.jsonl")

    done = {job_key_from_utility(row) for row in utilities}
    remaining = [job for job in jobs if job_key_from_job(job) not in done]
    duplicate_utility_rows = len(utilities) - len(done)

    valid = [row for row in utilities if row.get("valid_for_utility")]
    answer_complete = [row for row in valid if row.get("answer_complete_flag")]

    print("# Resume Status")
    print(f"active_traces: {len(traces)}")
    print(f"planned_jobs: {len(jobs)}")
    print(f"utility_rows: {len(utilities)}")
    print(f"unique_completed_jobs: {len(done)}")
    print(f"duplicate_utility_rows: {duplicate_utility_rows}")
    print(f"remaining_jobs: {len(remaining)}")
    print(f"error_rows_logged: {len(errors)}")
    print(f"valid_utility_rows: {len(valid)}")
    if utilities:
        print(f"valid_rate_so_far: {len(valid) / len(utilities):.3f}")
    if valid:
        print(f"answer_complete_rate_so_far: {len(answer_complete) / len(valid):.3f}")
    print("")
    print("planned_by_model:", dict(Counter(job["trace"]["model"] for job in jobs)))
    print("completed_by_model:", dict(Counter(row.get("model") for row in utilities)))
    print("remaining_by_model:", dict(Counter(job["trace"]["model"] for job in remaining)))
    print("remaining_by_benchmark:", dict(Counter(job["instance"]["benchmark"] for job in remaining)))
    print("errors_by_model:", dict(Counter(row.get("model") for row in errors)))
    print("errors_by_benchmark:", dict(Counter(row.get("benchmark") for row in errors)))

    if errors:
        print("")
        print("top_error_messages:")
        for message, count in Counter(str(row.get("error", ""))[:160] for row in errors).most_common(12):
            print(f"- {count}: {message}")


if __name__ == "__main__":
    main()

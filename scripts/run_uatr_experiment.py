from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from pilot_lib import (
    append_jsonl,
    extract_answer,
    final_answer_instruction,
    first_n_words,
    is_correct,
    load_config,
    normalize_text,
    read_jsonl,
    trace_text,
    write_csv,
)
from run_revision_experiment import (
    call_target,
    keyring_for,
    make_keyrings,
    make_result_row,
    response_full_text,
    run_method as run_baseline_method,
    total_tokens,
)


ALL_METHODS = ["original", "uatr", "random_step", "self_refine2", "debate2x2", "self_consistency"]


def step_block(steps: list[dict]) -> str:
    out = []
    for step in steps:
        signals = []
        if step.get("contradiction_flag"):
            signals.append(f"contradiction={step.get('contradiction_source', 'yes')}")
        if step.get("answer_critical_flag"):
            signals.append("answer-critical")
        if step.get("instability_flag"):
            signals.append("unstable")
        out.append(
            f"Step {step.get('step_id')} | utility={step.get('utility_score')} | signals={', '.join(signals)}\n"
            f"{str(step.get('step_text') or '').strip()}"
        )
    return "\n\n".join(out)


def make_uatr_prompt(instance: dict, trace: dict, job: dict, steps: list[dict], condition: str, max_trace_words: int) -> str:
    answer_instruction = final_answer_instruction(instance["benchmark"])
    return (
        "You are repairing a previous reasoning trace using localized diagnostic evidence.\n"
        "The flagged step(s) were selected because they appear to contradict the final answer and are high-value revision targets.\n"
        "Revise only what is necessary. If the original answer is still correct, preserve it. "
        "If the flagged contradiction changes the answer, correct the answer.\n"
        "You must end with exactly one parseable final-answer line.\n\n"
        f"{answer_instruction}\n\n"
        f"Revision condition: {condition}\n\n"
        f"Problem prompt:\n{instance['prompt']}\n\n"
        f"Original final answer: {job.get('original_answer')}\n"
        f"Original trace excerpt:\n{first_n_words(trace_text(trace), max_trace_words)}\n\n"
        f"Flagged step(s):\n{step_block(steps)}\n\n"
        "Now write the repaired reasoning and the required final-answer line."
    )


def force_answer_prompt(instance: dict, previous_response: str) -> str:
    answer_instruction = final_answer_instruction(instance["benchmark"])
    return (
        "The previous response did not contain a parseable final-answer line. "
        "Return a final answer now. Be concise and do not include extra commentary.\n\n"
        f"{answer_instruction}\n\n"
        f"Problem prompt:\n{instance['prompt']}\n\n"
        f"Previous response excerpt:\n{first_n_words(previous_response, 700)}\n\n"
        "Final-answer line only:"
    )


def apply_forced_answer_retry(
    *,
    row: dict,
    config: dict,
    keyring,
    job: dict,
    instance: dict,
    force_answer_retries: int,
) -> dict:
    if row.get("status") != "ok" or row.get("method") == "original":
        return row
    if str(row.get("prediction") or "").strip():
        row["forced_answer_retry_count"] = 0
        return row

    previous = str(row.get("response_tail") or "")
    keys = [k for k in str(row.get("api_key_names") or "").split(",") if k]
    calls = int(float(row.get("calls") or 0))
    tokens = int(float(row.get("total_tokens") or 0))
    latency = float(row.get("latency_s") or 0.0)
    forced_texts = []
    prediction = None
    for retry_id in range(force_answer_retries):
        started = time.time()
        result, key_name = call_target(
            config=config,
            keyring=keyring,
            job=job,
            messages=[
                {"role": "system", "content": "Return only the required final-answer line."},
                {"role": "user", "content": force_answer_prompt(instance, previous)},
            ],
        )
        full_text = response_full_text(result)
        forced_texts.append(full_text)
        previous = full_text or previous
        keys.append(key_name)
        calls += 1
        tokens += total_tokens(result)
        latency += time.time() - started
        prediction = extract_answer(job["benchmark"], full_text)
        if prediction:
            break

    row["forced_answer_retry_count"] = len(forced_texts)
    row["calls"] = calls
    row["total_tokens"] = tokens
    row["latency_s"] = round(latency, 3)
    row["api_key_names"] = ",".join(keys)
    row["forced_answer_tail"] = "\n\n".join(forced_texts)[-1200:]
    if prediction:
        row["prediction"] = prediction
        row["correctness"] = is_correct(job["benchmark"], prediction, job["gold_answer"])
        row["delta_corr"] = int(bool(row["correctness"])) - int(bool(job.get("original_correctness")))
        row["answer_changed"] = normalize_text(prediction) != normalize_text(job.get("original_answer"))
    return row


def run_uatr_method(
    *,
    method: str,
    job: dict,
    instance: dict,
    trace: dict,
    config: dict,
    keyring,
    max_trace_words: int,
    sc_samples: int,
    force_answer_retries: int,
    dry_run: bool,
) -> dict:
    if method == "original":
        row = make_result_row(
            job=job,
            method=method,
            status="ok",
            prediction=job.get("original_answer"),
            calls=0,
            selected_step_ids=[],
        )
        row["forced_answer_retry_count"] = 0
        return row

    if dry_run:
        row = make_result_row(
            job=job,
            method=method,
            status="ok",
            response_text=f"dry-run {method}",
            prediction=job.get("gold_answer") if method == "uatr" else job.get("original_answer"),
            calls=1 if method != "self_consistency" else sc_samples,
            selected_step_ids=job.get("trigger_step_ids", []),
        )
        row["forced_answer_retry_count"] = 0
        return row

    if method in {"uatr", "random_step"}:
        steps = job["uatr_steps"] if method == "uatr" else job["random_steps"]
        condition = (
            f"UATR strict trigger: {job.get('trigger_mode')}."
            if method == "uatr"
            else "Random valid-step revision baseline."
        )
        started = time.time()
        result, key_name = call_target(
            config=config,
            keyring=keyring,
            job=job,
            messages=[
                {"role": "system", "content": "You are a careful reasoning repair model. Always provide the final-answer line."},
                {"role": "user", "content": make_uatr_prompt(instance, trace, job, steps, condition, max_trace_words)},
            ],
        )
        full_text = response_full_text(result)
        prediction = extract_answer(job["benchmark"], full_text)
        row = make_result_row(
            job=job,
            method=method,
            status="ok",
            response_text=full_text,
            prediction=prediction,
            calls=1,
            latency_s=time.time() - started,
            total_token_count=total_tokens(result),
            api_key_names=[key_name],
            selected_step_ids=[int(step.get("step_id", -1)) for step in steps],
        )
        return apply_forced_answer_retry(
            row=row,
            config=config,
            keyring=keyring,
            job=job,
            instance=instance,
            force_answer_retries=force_answer_retries,
        )

    # Generic baselines are shared with the earlier revision runner, then given
    # the same final-answer repair pass for fair parsing.
    base_job = dict(job)
    base_job["uatr_steps"] = job.get("uatr_steps", [])
    base_job["contradiction_steps"] = job.get("uatr_steps", [])
    row = run_baseline_method(
        method=method,
        job=base_job,
        instance=instance,
        trace=trace,
        config=config,
        keyring=keyring,
        max_trace_words=max_trace_words,
        sc_samples=sc_samples,
        dry_run=dry_run,
    )
    return apply_forced_answer_retry(
        row=row,
        config=config,
        keyring=keyring,
        job=job,
        instance=instance,
        force_answer_retries=force_answer_retries,
    )


def result_key(row: dict) -> tuple[str, str]:
    return (row["job_id"], row["method"])


def reset_outputs(out_dir: Path) -> None:
    for name in [
        "uatr_results.jsonl",
        "uatr_results.csv",
        "uatr_errors.jsonl",
        "uatr_errors.csv",
    ]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sui-dir", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sc-samples", type=int, default=10)
    parser.add_argument("--max-trace-words", type=int, default=900)
    parser.add_argument("--force-answer-retries", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    unknown = [method for method in args.methods if method not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Choices: {ALL_METHODS}")

    config = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        reset_outputs(out_dir)

    jobs = read_jsonl(args.jobs)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    instances = {row["instance_id"]: row for row in read_jsonl(Path(args.sui_dir) / "pilot_instances.jsonl")}
    traces = {
        (row["model"], row["instance_id"]): row
        for row in read_jsonl(Path(args.sui_dir) / "pilot_trace_metadata.jsonl")
        if row.get("status") == "ok"
    }

    results_path = out_dir / "uatr_results.jsonl"
    completed: set[tuple[str, str]] = set()
    if results_path.exists() and not args.fresh:
        completed = {result_key(row) for row in read_jsonl(results_path)}
        print(f"Resuming: skipped {len(completed)} completed UATR rows.")

    keyrings = make_keyrings(config, jobs, args.dry_run)
    tasks = [
        (job, method)
        for job in jobs
        for method in args.methods
        if (job["job_id"], method) not in completed
    ]
    print(f"UATR tasks to run: {len(tasks)}")

    def worker(task: tuple[dict, str]) -> dict:
        job, method = task
        try:
            return run_uatr_method(
                method=method,
                job=job,
                instance=instances[job["instance_id"]],
                trace=traces[(job["model"], job["instance_id"])],
                config=config,
                keyring=keyring_for(keyrings, job),
                max_trace_words=args.max_trace_words,
                sc_samples=args.sc_samples,
                force_answer_retries=args.force_answer_retries,
                dry_run=args.dry_run,
            )
        except Exception as exc:
            return {
                "job_id": job["job_id"],
                "method": method,
                "model": job["model"],
                "benchmark": job["benchmark"],
                "instance_id": job["instance_id"],
                "status": "error",
                "error": str(exc),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            if row.get("status") == "error":
                append_jsonl(out_dir / "uatr_errors.jsonl", row)
                print(f"error: {row.get('method')} {row.get('model')} {row.get('benchmark')} {row.get('instance_id')}")
            else:
                append_jsonl(results_path, row)
                print(f"ok: {row.get('method')} {row.get('model')} {row.get('benchmark')} {row.get('instance_id')}")

    result_rows = read_jsonl(results_path) if results_path.exists() else []
    error_rows = read_jsonl(out_dir / "uatr_errors.jsonl") if (out_dir / "uatr_errors.jsonl").exists() else []
    if result_rows:
        write_csv(out_dir / "uatr_results.csv", result_rows)
    if error_rows:
        write_csv(out_dir / "uatr_errors.csv", error_rows)
    print(f"Wrote UATR outputs to {out_dir}")


if __name__ == "__main__":
    main()

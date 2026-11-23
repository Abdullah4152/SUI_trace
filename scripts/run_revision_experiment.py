from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

from pilot_lib import (
    KeyRing,
    append_jsonl,
    call_nvidia_chat,
    configured_key_names,
    extract_answer,
    final_answer_instruction,
    first_n_words,
    is_correct,
    load_api_keys,
    load_config,
    normalize_text,
    read_jsonl,
    runtime_settings,
    trace_text,
    write_csv,
)


ALL_METHODS = [
    "original",
    "uatr",
    "random_step",
    "contradiction_only",
    "self_refine2",
    "self_consistency",
    "debate2x2",
]


def make_keyrings(config: dict, jobs: list[dict], dry_run: bool) -> dict[str, KeyRing]:
    if dry_run:
        return {"dry_run": KeyRing([("dry_run", "dry_run")])}
    fallback = load_api_keys(config)
    keyrings: dict[str, KeyRing] = {}
    for job in jobs:
        name = f"target::{job['model']}::{job['model_id']}"
        if name in keyrings:
            continue
        names = configured_key_names(config, role="main", model_name=job["model"], model_id=job["model_id"])
        keys = load_api_keys(config, names)
        if not keys:
            keys = fallback
        keyrings[name] = KeyRing(keys)
        print(f"keypool: {name} uses {len(keys)} key(s): {', '.join(k for k, _ in keys)}")
    return keyrings


def keyring_for(keyrings: dict[str, KeyRing], job: dict) -> KeyRing:
    if "dry_run" in keyrings:
        return keyrings["dry_run"]
    return keyrings[f"target::{job['model']}::{job['model_id']}"]


def response_full_text(result: dict) -> str:
    reasoning = str(result.get("reasoning_content") or "").strip()
    response = str(result.get("response_text") or "").strip()
    if reasoning and response:
        return f"{reasoning}\n\nFinal response:\n{response}"
    return reasoning or response


def total_tokens(result: dict) -> int:
    usage = result.get("usage") or {}
    try:
        return int(usage.get("total_tokens") or 0)
    except Exception:
        return 0


def call_target(
    *,
    config: dict,
    keyring: KeyRing,
    job: dict,
    messages: list[dict[str, str]],
    sampling: bool = False,
) -> tuple[dict, str]:
    settings = runtime_settings(config, job["model_id"], "continuation")
    temperature = float(settings["temperature"])
    top_p = float(settings["top_p"])
    if sampling:
        decoding = config.get("decoding", {})
        temperature = float(decoding.get("sampling_temperature", 0.7))
        top_p = float(decoding.get("sampling_top_p", 0.95))
    with keyring.lease() as (key_name, api_key):
        result = call_nvidia_chat(
            base_url=config["base_url"],
            api_key=api_key,
            model=job["model_id"],
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=int(settings["max_tokens"]),
            extra=settings.get("extra", {}),
            timeout=int(config.get("request_timeout_s", 900)),
            retries=int(config.get("request_retries", 2)),
            retry_sleep=float(config.get("retry_sleep_s", 20)),
            response_cooldown=float(config.get("response_cooldown_s", 0)),
            stream=bool(settings.get("stream", True)),
        )
    return result, key_name


def step_block(steps: list[dict]) -> str:
    lines = []
    for step in steps:
        signals = []
        for key in ["answer_critical_flag", "contradiction_flag", "instability_flag"]:
            if step.get(key):
                signals.append(key.replace("_flag", ""))
        if step.get("negative_support_signals"):
            signals.append(f"negative={step.get('negative_support_signals')}")
        signals_text = ", ".join(signals) if signals else "none"
        lines.append(
            f"Step {step.get('step_id')} | utility={step.get('utility_score')} | signals={signals_text}\n"
            f"{step.get('step_text', '').strip()}"
        )
    return "\n\n".join(lines)


def make_targeted_prompt(instance: dict, trace: dict, job: dict, steps: list[dict], label: str, max_trace_words: int) -> str:
    answer_instruction = final_answer_instruction(instance["benchmark"])
    original_trace = first_n_words(trace_text(trace), max_trace_words)
    return (
        "You are revising a previous reasoning trace for one problem.\n"
        "Use the flagged step information as localized feedback. Do not rewrite unrelated parts unless needed.\n"
        "Check whether the flagged step causes an error or unsupported inference, repair the reasoning locally, "
        "and then provide the final answer in the required format.\n\n"
        f"{answer_instruction}\n\n"
        f"Revision condition: {label}\n\n"
        f"Problem prompt:\n{instance['prompt']}\n\n"
        f"Original final answer: {job.get('original_answer')}\n"
        f"Original trace excerpt:\n{original_trace}\n\n"
        f"Flagged step(s):\n{step_block(steps)}\n\n"
        "Now provide the revised reasoning and final answer."
    )


def make_self_refine_prompt(instance: dict, current_solution: str, round_id: int) -> str:
    answer_instruction = final_answer_instruction(instance["benchmark"])
    return (
        f"{instance['prompt']}\n\n"
        f"Current solution for refinement round {round_id}:\n{current_solution}\n\n"
        "Critique the current solution for mistakes, unsupported jumps, or answer-format issues. "
        "Then produce a corrected solution.\n"
        f"{answer_instruction}"
    )


def make_debate_critic_prompt(instance: dict, trace: str, stance: str) -> str:
    return (
        f"{instance['prompt']}\n\n"
        f"Previous solution:\n{trace}\n\n"
        f"You are critic {stance}. Find any reasoning or answer error. "
        "Be concise and focus only on issues that could change the final answer."
    )


def make_debate_final_prompt(instance: dict, trace: str, critiques: list[str]) -> str:
    answer_instruction = final_answer_instruction(instance["benchmark"])
    joined = "\n\n".join(f"Critique {i + 1}:\n{text}" for i, text in enumerate(critiques))
    return (
        f"{instance['prompt']}\n\n"
        f"Previous solution:\n{trace}\n\n"
        f"{joined}\n\n"
        "Adjudicate the critiques and produce the best final solution.\n"
        f"{answer_instruction}"
    )


def make_result_row(
    *,
    job: dict,
    method: str,
    status: str,
    response_text: str = "",
    prediction: str | None = None,
    calls: int = 0,
    latency_s: float = 0.0,
    total_token_count: int = 0,
    api_key_names: list[str] | None = None,
    selected_step_ids: list[int] | None = None,
    sample_answers: list[str | None] | None = None,
    error: str = "",
) -> dict:
    correct = "" if prediction is None else is_correct(job["benchmark"], prediction, job["gold_answer"])
    if correct == "":
        delta_corr = ""
    else:
        delta_corr = int(bool(correct)) - int(bool(job.get("original_correctness")))
    answer_changed = "" if prediction is None else normalize_text(prediction) != normalize_text(job.get("original_answer"))
    return {
        "job_id": job["job_id"],
        "method": method,
        "status": status,
        "model": job["model"],
        "model_id": job["model_id"],
        "benchmark": job["benchmark"],
        "instance_id": job["instance_id"],
        "original_answer": job.get("original_answer", ""),
        "original_correctness": bool(job.get("original_correctness")),
        "gold_answer": job.get("gold_answer", ""),
        "prediction": prediction or "",
        "correctness": correct,
        "delta_corr": delta_corr,
        "answer_changed": answer_changed,
        "calls": calls,
        "latency_s": round(latency_s, 3),
        "total_tokens": total_token_count,
        "api_key_names": ",".join(api_key_names or []),
        "selected_step_ids": ",".join(str(x) for x in (selected_step_ids or [])),
        "sample_answers_json": json.dumps(sample_answers or [], ensure_ascii=False),
        "response_tail": response_text[-1200:],
        "error": error,
    }


def run_method(
    *,
    method: str,
    job: dict,
    instance: dict,
    trace: dict,
    config: dict,
    keyring: KeyRing,
    max_trace_words: int,
    sc_samples: int,
    dry_run: bool,
) -> dict:
    if method == "original":
        return make_result_row(
            job=job,
            method=method,
            status="ok",
            response_text="",
            prediction=job.get("original_answer"),
            calls=0,
            selected_step_ids=[],
        )

    if dry_run:
        prediction = job.get("gold_answer") if method == "uatr" else job.get("original_answer")
        return make_result_row(
            job=job,
            method=method,
            status="ok",
            response_text=f"dry run response for {method}",
            prediction=prediction,
            calls=1 if method != "self_consistency" else sc_samples,
            selected_step_ids=job.get("selected_step_ids", []),
        )

    started = time.time()
    keys: list[str] = []
    token_count = 0

    if method in {"uatr", "random_step", "contradiction_only"}:
        if method == "uatr":
            steps = job.get("uatr_steps", [])
            label = "UATR high-risk/utility-selected step revision"
        elif method == "random_step":
            steps = job.get("random_steps", [])
            label = "Random-step targeted revision baseline"
        else:
            steps = job.get("contradiction_steps", [])
            label = "Contradiction-only revision baseline"
            if not steps:
                return make_result_row(
                    job=job,
                    method=method,
                    status="skipped_no_contradiction",
                    calls=0,
                    selected_step_ids=[],
                )
        prompt = make_targeted_prompt(instance, trace, job, steps, label, max_trace_words)
        result, key_name = call_target(
            config=config,
            keyring=keyring,
            job=job,
            messages=[
                {"role": "system", "content": "You are a careful reasoning reviser. Follow the requested final-answer format."},
                {"role": "user", "content": prompt},
            ],
        )
        keys.append(key_name)
        token_count += total_tokens(result)
        full_text = response_full_text(result)
        prediction = extract_answer(job["benchmark"], full_text)
        selected_ids = [int(s.get("step_id", -1)) for s in steps]
        return make_result_row(
            job=job,
            method=method,
            status="ok",
            response_text=full_text,
            prediction=prediction,
            calls=1,
            latency_s=time.time() - started,
            total_token_count=token_count,
            api_key_names=keys,
            selected_step_ids=selected_ids,
        )

    if method == "self_refine2":
        current = trace_text(trace)
        final_text = ""
        for round_id in [1, 2]:
            prompt = make_self_refine_prompt(instance, first_n_words(current, max_trace_words), round_id)
            result, key_name = call_target(
                config=config,
                keyring=keyring,
                job=job,
                messages=[
                    {"role": "system", "content": "You are a careful self-refinement solver. Follow the final-answer format."},
                    {"role": "user", "content": prompt},
                ],
            )
            keys.append(key_name)
            token_count += total_tokens(result)
            final_text = response_full_text(result)
            current = final_text
        prediction = extract_answer(job["benchmark"], final_text)
        return make_result_row(
            job=job,
            method=method,
            status="ok",
            response_text=final_text,
            prediction=prediction,
            calls=2,
            latency_s=time.time() - started,
            total_token_count=token_count,
            api_key_names=keys,
        )

    if method == "self_consistency":
        answers: list[str | None] = []
        responses: list[str] = []
        for _ in range(sc_samples):
            prompt = (
                f"{instance['prompt']}\n\n"
                "Solve independently. Give concise reasoning and the required final-answer line."
            )
            result, key_name = call_target(
                config=config,
                keyring=keyring,
                job=job,
                messages=[
                    {"role": "system", "content": "You are an independent reasoning solver. Follow the final-answer format."},
                    {"role": "user", "content": prompt},
                ],
                sampling=True,
            )
            keys.append(key_name)
            token_count += total_tokens(result)
            full_text = response_full_text(result)
            responses.append(full_text[-400:])
            answers.append(extract_answer(job["benchmark"], full_text))
        counts: dict[str, int] = {}
        first_seen: dict[str, str | None] = {}
        for answer in answers:
            key = normalize_text(answer)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, answer)
        prediction = None
        if counts:
            winner = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            prediction = first_seen[winner]
        return make_result_row(
            job=job,
            method=method,
            status="ok",
            response_text="\n\n".join(responses),
            prediction=prediction,
            calls=sc_samples,
            latency_s=time.time() - started,
            total_token_count=token_count,
            api_key_names=keys,
            sample_answers=answers,
        )

    if method == "debate2x2":
        original_trace = first_n_words(trace_text(trace), max_trace_words)
        critiques: list[str] = []
        for stance in ["A", "B"]:
            result, key_name = call_target(
                config=config,
                keyring=keyring,
                job=job,
                messages=[
                    {"role": "system", "content": "You are a strict reasoning critic."},
                    {"role": "user", "content": make_debate_critic_prompt(instance, original_trace, stance)},
                ],
            )
            keys.append(key_name)
            token_count += total_tokens(result)
            critiques.append(response_full_text(result))
        result, key_name = call_target(
            config=config,
            keyring=keyring,
            job=job,
            messages=[
                {"role": "system", "content": "You are a careful debate adjudicator. Follow the final-answer format."},
                {"role": "user", "content": make_debate_final_prompt(instance, original_trace, critiques)},
            ],
        )
        keys.append(key_name)
        token_count += total_tokens(result)
        final_text = response_full_text(result)
        prediction = extract_answer(job["benchmark"], final_text)
        return make_result_row(
            job=job,
            method=method,
            status="ok",
            response_text=final_text,
            prediction=prediction,
            calls=3,
            latency_s=time.time() - started,
            total_token_count=token_count,
            api_key_names=keys,
        )

    raise ValueError(f"Unknown method: {method}")


def result_key(row: dict) -> tuple[str, str]:
    return (row["job_id"], row["method"])


def reset_outputs(out_dir: Path) -> None:
    for name in [
        "revision_results.jsonl",
        "revision_results.csv",
        "revision_errors.jsonl",
        "revision_errors.csv",
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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sc-samples", type=int, default=10)
    parser.add_argument("--max-trace-words", type=int, default=1800)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    unknown = [m for m in args.methods if m not in ALL_METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. Choices: {ALL_METHODS}")

    config = load_config(args.config)
    sui_dir = Path(args.sui_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        reset_outputs(out_dir)

    jobs = read_jsonl(args.jobs)
    if args.limit is not None:
        jobs = jobs[: args.limit]
    instances = {row["instance_id"]: row for row in read_jsonl(sui_dir / "pilot_instances.jsonl")}
    traces = {
        (row["model"], row["instance_id"]): row
        for row in read_jsonl(sui_dir / "pilot_trace_metadata.jsonl")
        if row.get("status") == "ok"
    }

    completed: set[tuple[str, str]] = set()
    results_path = out_dir / "revision_results.jsonl"
    if results_path.exists() and not args.fresh:
        completed = {result_key(row) for row in read_jsonl(results_path)}
        print(f"Resuming: skipped {len(completed)} completed revision rows.")

    keyrings = make_keyrings(config, jobs, args.dry_run)

    tasks = []
    for job in jobs:
        for method in args.methods:
            if (job["job_id"], method) not in completed:
                tasks.append((job, method))
    print(f"Revision tasks to run: {len(tasks)}")

    def worker(task: tuple[dict, str]) -> dict:
        job, method = task
        instance = instances[job["instance_id"]]
        trace = traces[(job["model"], job["instance_id"])]
        try:
            return run_method(
                method=method,
                job=job,
                instance=instance,
                trace=trace,
                config=config,
                keyring=keyring_for(keyrings, job),
                max_trace_words=args.max_trace_words,
                sc_samples=args.sc_samples,
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
                append_jsonl(out_dir / "revision_errors.jsonl", row)
                print(f"error: {row.get('method')} {row.get('model')} {row.get('benchmark')} {row.get('instance_id')}")
            else:
                append_jsonl(results_path, row)
                print(f"ok: {row.get('method')} {row.get('model')} {row.get('benchmark')} {row.get('instance_id')}")

    result_rows = read_jsonl(results_path) if results_path.exists() else []
    error_rows = read_jsonl(out_dir / "revision_errors.jsonl") if (out_dir / "revision_errors.jsonl").exists() else []
    if result_rows:
        write_csv(out_dir / "revision_results.csv", result_rows)
    if error_rows:
        write_csv(out_dir / "revision_errors.csv", error_rows)
    print(f"Wrote revision outputs to {out_dir}")


if __name__ == "__main__":
    main()

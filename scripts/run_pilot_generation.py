from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

from pilot_lib import (
    KeyRing,
    call_nvidia_chat,
    configured_key_names,
    extract_answer,
    is_correct,
    load_api_keys,
    load_config,
    runtime_settings,
    read_jsonl,
    synthetic_trace,
    write_csv,
    write_jsonl,
    now_iso,
)


def make_model_keyrings(config: dict, target_models: dict[str, str], dry_run: bool) -> dict[str, KeyRing]:
    if dry_run:
        return {model_name: KeyRing([("dry_run", "dry_run")]) for model_name in target_models}
    fallback = load_api_keys(config)
    keyrings = {}
    for model_name, model_id in target_models.items():
        names = configured_key_names(config, role="main", model_name=model_name, model_id=model_id)
        keys = load_api_keys(config, names)
        if not keys:
            keys = fallback
        keyrings[model_name] = KeyRing(keys)
        print(f"keypool: {model_name} uses {len(keys)} key(s): {', '.join(name for name, _ in keys)}")
    return keyrings


def build_jobs(
    instances: list[dict],
    target_models: dict[str, str],
    limit: int | None,
    model_filter: list[str] | None = None,
) -> list[dict]:
    jobs = []
    if model_filter:
        allowed = set(model_filter)
        target_models = {
            name: model_id
            for name, model_id in target_models.items()
            if name in allowed or model_id in allowed
        }
        if not target_models:
            raise ValueError(f"No target models matched --models {model_filter}")
    selected = instances if limit is None else instances[:limit]
    for inst in selected:
        for model_name, model_id in target_models.items():
            jobs.append({"model_name": model_name, "model_id": model_id, "instance": inst})
    return jobs


def run_job(job: dict, config: dict, keyring: KeyRing, dry_run: bool) -> dict:
    inst = job["instance"]
    if dry_run:
        response_text = synthetic_trace(inst)
        result = {
            "response_text": response_text,
            "reasoning_content": "",
            "finish_reason": "dry_run",
            "response_model": job["model_id"],
            "usage": {},
            "latency_s": 0.0,
        }
        key_name = "dry_run"
    else:
        settings = runtime_settings(config, job["model_id"], "main")
        with keyring.lease() as (key_name, api_key):
            result = call_nvidia_chat(
                base_url=config["base_url"],
                api_key=api_key,
                model=job["model_id"],
                messages=[
                    {"role": "system", "content": "You are a careful reasoning model. Show concise step-by-step reasoning and then the final answer in the requested format."},
                    {"role": "user", "content": inst["prompt"]},
                ],
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
                max_tokens=int(settings["max_tokens"]),
                extra=settings.get("extra", {}),
                timeout=int(config.get("request_timeout_s", 900)),
                retries=int(config.get("request_retries", 2)),
                retry_sleep=float(config.get("retry_sleep_s", 20)),
                response_cooldown=float(config.get("response_cooldown_s", 0)),
                stream=bool(settings.get("stream", True)),
            )

    pred = extract_answer(inst["benchmark"], result["response_text"])
    correct = is_correct(inst["benchmark"], pred, inst["gold_answer"])
    return {
        "timestamp": now_iso(),
        "model": job["model_name"],
        "model_id": job["model_id"],
        "api_key_name": key_name,
        "benchmark": inst["benchmark"],
        "instance_id": inst["instance_id"],
        "trace_id": f"{job['model_name']}::{inst['instance_id']}",
        "prompt": inst["prompt"],
        "answer": pred or "",
        "gold_answer": inst["gold_answer"],
        "correctness": correct,
        "response_text": result["response_text"],
        "reasoning_content": result.get("reasoning_content", ""),
        "finish_reason": result.get("finish_reason"),
        "response_model": result.get("response_model"),
        "usage_json": result.get("usage", {}),
        "latency_s": result.get("latency_s"),
        "status": "ok",
        "error": "",
    }


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") == "ok":
                done.add(str(row.get("trace_id")))
    return done


def append_jsonl_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_trace_csv(jsonl_path: Path, out_dir: Path) -> None:
    all_rows_raw = read_jsonl(jsonl_path) if jsonl_path.exists() else []
    latest_by_trace: dict[str, dict] = {}
    for row in all_rows_raw:
        latest_by_trace[str(row.get("trace_id"))] = row
    csv_rows = []
    for row in latest_by_trace.values():
        slim = dict(row)
        slim["usage_json"] = str(slim.get("usage_json", {}))
        csv_rows.append(slim)
    write_csv(out_dir / "pilot_trace_metadata.csv", csv_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    instances = read_jsonl(args.instances)
    jobs = build_jobs(instances, config["target_models"], args.limit, args.models)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "pilot_trace_metadata.jsonl"
    done = load_done(jsonl_path)
    jobs = [job for job in jobs if f"{job['model_name']}::{job['instance']['instance_id']}" not in done]
    if done:
        print(f"Resuming: skipped {len(done)} completed trace rows.")

    selected_target_models = {job["model_name"]: job["model_id"] for job in jobs}
    keyrings = make_model_keyrings(config, selected_target_models, args.dry_run)
    rows: list[dict] = []

    def safe(job: dict) -> dict:
        try:
            return run_job(job, config, keyrings[job["model_name"]], args.dry_run)
        except Exception as exc:
            inst = job["instance"]
            return {
                "timestamp": now_iso(),
                "model": job["model_name"],
                "model_id": job["model_id"],
                "benchmark": inst["benchmark"],
                "instance_id": inst["instance_id"],
                "trace_id": f"{job['model_name']}::{inst['instance_id']}",
                "prompt": inst["prompt"],
                "answer": "",
                "gold_answer": inst["gold_answer"],
                "correctness": False,
                "response_text": "",
                "reasoning_content": "",
                "finish_reason": "",
                "response_model": job["model_id"],
                "usage_json": {},
                "latency_s": "",
                "status": "error",
                "error": str(exc),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(safe, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            rows.append(row)
            append_jsonl_row(jsonl_path, row)
            if row["status"] == "ok":
                print(f"ok: {row['model']} {row['benchmark']} {row['instance_id']} latency={row.get('latency_s')}")
            else:
                print(f"error: {row['model']} {row['benchmark']} {row['instance_id']} :: {row.get('error', '')[:500]}")
            if len(rows) % 10 == 0:
                write_trace_csv(jsonl_path, out_dir)

    write_trace_csv(jsonl_path, out_dir)
    all_rows = read_jsonl(jsonl_path)
    print(f"Wrote {len({str(row.get('trace_id')) for row in all_rows})} trace rows to {out_dir}")


if __name__ == "__main__":
    main()

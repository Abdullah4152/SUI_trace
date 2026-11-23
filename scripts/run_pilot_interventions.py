from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

from pilot_lib import (
    KeyRing,
    append_jsonl,
    call_nvidia_chat,
    configured_key_names,
    continuation_divergence,
    dry_edit_step,
    extract_answer,
    first_n_words,
    is_correct,
    is_intervenable_step,
    lexical_overlap,
    load_api_keys,
    load_config,
    runtime_settings,
    make_editor_prompt,
    make_editor_retry_prompt,
    make_resume_prompt,
    parse_edited_step,
    read_jsonl,
    rule_based_contradiction,
    sample_step_indices,
    split_trace_steps,
    synthetic_trace,
    text_similarity,
    trace_text,
    validate_intervention,
    write_csv,
)


def make_role_keyrings(config: dict, traces: list[dict], dry_run: bool) -> dict[str, KeyRing]:
    if dry_run:
        return {"dry_run": KeyRing([("dry_run", "dry_run")])}
    fallback = load_api_keys(config)
    keyrings: dict[str, KeyRing] = {}

    def add(name: str, role: str, model_name: str | None, model_id: str) -> None:
        names = configured_key_names(config, role=role, model_name=model_name, model_id=model_id)
        keys = load_api_keys(config, names)
        if not keys:
            keys = fallback
        keyrings[name] = KeyRing(keys)
        print(f"keypool: {name} uses {len(keys)} key(s): {', '.join(key_name for key_name, _ in keys)}")

    for trace in traces:
        if trace.get("status") != "ok":
            continue
        name = f"target::{trace['model']}::{trace['model_id']}"
        if name not in keyrings:
            add(name, "main", trace.get("model"), trace["model_id"])
    for editor_name, editor_id in (config.get("editor_models") or {}).items():
        add(f"editor::{editor_name}::{editor_id}", "editor", editor_name, editor_id)
    judge_model = config["judge_model"]
    add(f"judge::{judge_model}", "judge", None, judge_model)
    return keyrings


def keyring_for(keyrings: dict[str, KeyRing], name: str) -> KeyRing:
    if name in keyrings:
        return keyrings[name]
    if "dry_run" in keyrings:
        return keyrings["dry_run"]
    return next(iter(keyrings.values()))


def validation_failure_reasons(validity, schema_ok: bool) -> list[str]:
    reasons = []
    if not schema_ok:
        reasons.append("schema_invalid")
    if not validity.length_check:
        reasons.append("length_fail")
    if not validity.passed_semantic_change_check:
        reasons.append("semantic_fail")
    if validity.new_answer_leakage_flag:
        reasons.append("new_answer_leakage")
    if validity.meta_commentary_flag:
        reasons.append("meta_commentary")
    if validity.final_answer_step_flag:
        reasons.append("final_answer_step")
    return reasons


def has_answer(value) -> bool:
    return bool(str(value or "").strip())


def trace_by_key(traces: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["model"], row["instance_id"]): row for row in traces if row.get("status") == "ok"}


def judge_contradiction(
    *,
    config: dict,
    keyring: KeyRing,
    step: str,
    final_answer: str,
    dry_run: bool,
) -> tuple[bool, str]:
    rule = rule_based_contradiction(step, final_answer)
    if dry_run:
        return rule, "rule" if rule else "none"

    prompt = (
        "Return JSON only with key label. The label must be one of "
        "`contradiction`, `neutral`, or `entailed`.\n\n"
        f"Step:\n{step}\n\nFinal answer:\n{final_answer}\n\n"
        "Does the step contradict the final answer?"
    )
    try:
        settings = runtime_settings(config, config["judge_model"], "judge")
        with keyring.lease() as (_key_name, api_key):
            result = call_nvidia_chat(
                base_url=config["base_url"],
                api_key=api_key,
                model=config["judge_model"],
                messages=[
                    {"role": "system", "content": "You are a strict consistency judge. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=float(settings["temperature"]),
                top_p=float(settings["top_p"]),
                max_tokens=int(settings["max_tokens"]),
                extra=settings.get("extra", {}),
                timeout=int(config.get("request_timeout_s", 900)),
                retries=int(config.get("request_retries", 2)),
                retry_sleep=float(config.get("retry_sleep_s", 20)),
                response_cooldown=float(config.get("response_cooldown_s", 0)),
                stream=bool(settings.get("stream", False)),
            )
        text = result["response_text"] or result.get("reasoning_content", "")
        try:
            label = json.loads(text).get("label", "")
        except Exception:
            label = text.lower()
        judge = "contradiction" in str(label).lower()
    except Exception:
        judge = False

    if rule and judge:
        return True, "judge+rule"
    if rule:
        return True, "rule"
    if judge:
        return True, "judge"
    return False, "none"


def build_step_jobs(instances: list[dict], traces: list[dict], config: dict, limit: int | None) -> list[dict]:
    inst_by_id = {row["instance_id"]: row for row in instances}
    latest_by_trace: dict[str, dict] = {}
    for trace in traces:
        latest_by_trace[str(trace.get("trace_id"))] = trace
    rows = list(latest_by_trace.values())
    rows = rows if limit is None else rows[:limit]
    jobs = []
    for trace in rows:
        if trace.get("status") != "ok":
            continue
        inst = inst_by_id.get(trace["instance_id"])
        if inst is None:
            continue
        steps, method = split_trace_steps(trace_text(trace))
        eligible = [
            idx
            for idx, step in enumerate(steps)
            if is_intervenable_step(step, inst["benchmark"], inst["gold_answer"], inst.get("gold_answer_text"))
        ]
        if not eligible:
            continue
        seed = int(hashlib.md5(trace["trace_id"].encode("utf-8")).hexdigest()[:8], 16)
        local_indices = sample_step_indices(
            len(eligible),
            int(config["pilot"]["sampled_steps_per_trace"]),
            seed,
        )
        for local_step_id in local_indices:
            step_id = eligible[local_step_id]
            for editor_name, editor_id in config["editor_models"].items():
                jobs.append(
                    {
                        "instance": inst,
                        "trace": trace,
                        "steps": steps,
                        "segmentation_method": method,
                        "num_trace_steps": len(steps),
                        "num_eligible_steps": len(eligible),
                        "step_id": step_id,
                        "editor_name": editor_name,
                        "editor_id": editor_id,
                    }
                )
    return jobs


def filter_traces_for_target_panel(traces: list[dict], config: dict, model_filter: list[str] | None) -> list[dict]:
    if model_filter:
        allowed_names = set(model_filter)
        allowed_ids = set(model_filter)
    else:
        allowed_names = set((config.get("target_models") or {}).keys())
        allowed_ids = set((config.get("target_models") or {}).values())
    return [
        trace
        for trace in traces
        if trace.get("model") in allowed_names or trace.get("model_id") in allowed_ids
    ]


def stratified_limit_jobs(jobs: list[dict], max_jobs: int | None, seed: int) -> list[dict]:
    if max_jobs is None or len(jobs) <= max_jobs:
        return jobs
    import random

    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict]] = {}
    for job in jobs:
        key = (job["trace"]["model"], job["instance"]["benchmark"])
        groups.setdefault(key, []).append(job)
    for group_jobs in groups.values():
        rng.shuffle(group_jobs)
    keys = sorted(groups)
    selected: list[dict] = []
    while len(selected) < max_jobs and any(groups.values()):
        for key in keys:
            if groups[key] and len(selected) < max_jobs:
                selected.append(groups[key].pop())
    rng.shuffle(selected)
    return selected


def reset_outputs(out_dir: Path) -> None:
    for name in [
        "pilot_sampled_steps.jsonl",
        "pilot_interventions.jsonl",
        "pilot_utility_scores.jsonl",
        "pilot_uatr_region_labels.jsonl",
        "pilot_intervention_errors.jsonl",
        "pilot_sampled_steps.csv",
        "pilot_interventions.csv",
        "pilot_utility_scores.csv",
        "pilot_uatr_region_labels.csv",
        "pilot_editor_comparison.csv",
    ]:
        path = out_dir / name
        if path.exists():
            path.unlink()


def run_intervention_job(job: dict, config: dict, keyrings: dict[str, KeyRing], dry_run: bool) -> dict:
    inst = job["instance"]
    trace = job["trace"]
    steps = job["steps"]
    step_id = job["step_id"]
    original_step = steps[step_id]
    previous_step = steps[step_id - 1] if step_id > 0 else ""
    next_step = steps[step_id + 1] if step_id + 1 < len(steps) else ""
    thresholds = config["thresholds"]

    original_answer = trace.get("answer") or extract_answer(inst["benchmark"], trace.get("response_text", ""))
    original_correct = bool(trace.get("correctness"))
    max_attempts = int(config["pilot"]["max_attempts_per_step"])
    attempt_rows = []
    accepted_attempt = None
    use_baseline = bool(config.get("pilot", {}).get("use_unedited_continuation_baseline", True))

    def continue_from_prefix(prefix: str, force_answer: bool = False) -> tuple[str, str, str]:
        if dry_run:
            text = synthetic_trace(inst)
            return text, text, "dry_run"
        target_keyring_name = f"target::{trace['model']}::{trace['model_id']}"
        resume_prompt = make_resume_prompt(inst, prefix, force_answer=force_answer)
        settings = runtime_settings(config, trace["model_id"], "continuation")
        with keyring_for(keyrings, target_keyring_name).lease() as (key_name, api_key):
            cont = call_nvidia_chat(
                base_url=config["base_url"],
                api_key=api_key,
                model=trace["model_id"],
                messages=[
                    {"role": "system", "content": "Continue the provided reasoning trace and finish with the required final answer format."},
                    {"role": "user", "content": resume_prompt},
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
        return trace_text(cont), cont["response_text"] or trace_text(cont), key_name

    def continue_with_answer(prefix: str) -> tuple[str, str, str, str | None, bool, int]:
        text, answer_text, key_name = continue_from_prefix(prefix, force_answer=False)
        answer = extract_answer(inst["benchmark"], answer_text)
        retry_count = 0
        max_answer_retries = int(config.get("pilot", {}).get("continuation_answer_retries", 0))
        while not has_answer(answer) and retry_count < max_answer_retries:
            retry_count += 1
            text, answer_text, key_name = continue_from_prefix(prefix, force_answer=True)
            answer = extract_answer(inst["benchmark"], answer_text)
        correct = is_correct(inst["benchmark"], answer, inst["gold_answer"])
        return text, answer_text, key_name, answer, correct, retry_count

    def attempt_row(
        *,
        attempt_id: int,
        edited_step: str,
        schema_ok: bool,
        validity,
        editor_key_name: str,
        target_key_name: str = "",
        baseline_key_name: str = "",
        delta_cont: float | str = "",
        cont_metric: str = "",
        delta_ans: int | str = "",
        delta_corr: int | str = "",
        edited_answer: str = "",
        edited_correct: bool | str = "",
        baseline_answer: str = "",
        baseline_correct: bool | str = "",
        baseline_answer_retry_count: int | str = "",
        edited_answer_retry_count: int | str = "",
        baseline_response_tail: str = "",
        edited_response_tail: str = "",
    ) -> dict:
        answer_complete = has_answer(baseline_answer) and has_answer(edited_answer)
        return {
            "model": trace["model"],
            "benchmark": inst["benchmark"],
            "instance_id": inst["instance_id"],
            "step_id": step_id,
            "editor_model": job["editor_name"],
            "editor_model_id": job["editor_id"],
            "attempt_id": attempt_id,
            "original_step": original_step,
            "edited_step": edited_step,
            "schema_valid": schema_ok,
            "passed_fluency_check": bool(edited_step.strip()),
            "passed_coherence_check": True,
            "passed_semantic_change_check": validity.passed_semantic_change_check,
            "semantic_similarity": validity.semantic_similarity,
            "lexical_overlap": validity.lexical_overlap,
            "length_ratio": validity.length_ratio,
            "length_check": validity.length_check,
            "meta_commentary_flag": validity.meta_commentary_flag,
            "final_answer_step_flag": validity.final_answer_step_flag,
            "original_answer_leakage_flag": validity.original_answer_leakage_flag,
            "answer_leakage_flag": validity.answer_leakage_flag,
            "new_answer_leakage_flag": validity.new_answer_leakage_flag,
            "delta_cont": delta_cont,
            "continuation_metric": cont_metric,
            "delta_ans": delta_ans,
            "delta_corr": delta_corr,
            "baseline_answer": baseline_answer,
            "baseline_correctness": baseline_correct,
            "edited_answer": edited_answer,
            "edited_correctness": edited_correct,
            "answer_complete_flag": answer_complete if delta_cont != "" else "",
            "answer_missing_flag": (not answer_complete) if delta_cont != "" else "",
            "baseline_answer_retry_count": baseline_answer_retry_count,
            "edited_answer_retry_count": edited_answer_retry_count,
            "baseline_response_tail": baseline_response_tail,
            "edited_response_tail": edited_response_tail,
            "passed_uptake_check": validity.passed_uptake_check,
            "over_disruption_flag": validity.over_disruption_flag,
            "accepted": validity.accepted,
            "valid_for_utility": bool(validity.accepted and delta_cont != ""),
            "unresolved_after_retries": False,
            "editor_key_name": editor_key_name,
            "baseline_key_name": baseline_key_name,
            "target_key_name": target_key_name,
        }

    previous_edit = ""
    previous_failure_reasons: list[str] = []
    for attempt_id in range(1, max_attempts + 1):
        if dry_run:
            editor_response = dry_edit_step(original_step)
            edited_step = editor_response
            schema_ok = True
            editor_key_name = "dry_run"
        else:
            editor_keyring_name = f"editor::{job['editor_name']}::{job['editor_id']}"
            if previous_failure_reasons:
                editor_prompt = make_editor_retry_prompt(
                    inst,
                    previous_step,
                    original_step,
                    next_step,
                    previous_edit,
                    previous_failure_reasons,
                )
            else:
                editor_prompt = make_editor_prompt(inst, previous_step, original_step, next_step)
            settings = runtime_settings(config, job["editor_id"], "editor")
            with keyring_for(keyrings, editor_keyring_name).lease() as (editor_key_name, api_key):
                result = call_nvidia_chat(
                    base_url=config["base_url"],
                    api_key=api_key,
                    model=job["editor_id"],
                    messages=[
                        {"role": "system", "content": "You are a controlled text editor. Return valid JSON only."},
                        {"role": "user", "content": editor_prompt},
                    ],
                    temperature=float(settings["temperature"]),
                    top_p=float(settings["top_p"]),
                    max_tokens=int(settings["max_tokens"]),
                    extra=settings.get("extra", {}),
                    timeout=int(config.get("request_timeout_s", 900)),
                    retries=int(config.get("request_retries", 2)),
                    retry_sleep=float(config.get("retry_sleep_s", 20)),
                    response_cooldown=float(config.get("response_cooldown_s", 0)),
                    stream=bool(settings.get("stream", False)),
                )
            editor_raw = result["response_text"] or result.get("reasoning_content", "")
            edited_step, schema_ok = parse_edited_step(editor_raw)

        pre_validity = validate_intervention(
            benchmark=inst["benchmark"],
            original_step=original_step,
            edited_step=edited_step,
            schema_valid=schema_ok,
            delta_cont=None,
            gold_answer=inst["gold_answer"],
            gold_answer_text=inst.get("gold_answer_text"),
            thresholds=thresholds,
        )
        if not pre_validity.accepted:
            previous_edit = edited_step
            previous_failure_reasons = validation_failure_reasons(pre_validity, schema_ok)
            attempt_rows.append(
                attempt_row(
                    attempt_id=attempt_id,
                    edited_step=edited_step,
                    schema_ok=schema_ok,
                    validity=pre_validity,
                    editor_key_name=editor_key_name,
                )
            )
            continue

        edited_steps = list(steps)
        edited_steps[step_id] = edited_step
        original_prefix = "\n\n".join(steps[: step_id + 1])
        edited_prefix = "\n\n".join(edited_steps[: step_id + 1])

        if use_baseline:
            (
                baseline_text,
                baseline_answer_text,
                baseline_key_name,
                baseline_answer,
                baseline_correct,
                baseline_answer_retry_count,
            ) = continue_with_answer(original_prefix)
        else:
            baseline_text = "\n\n".join(steps[step_id + 1 :])
            baseline_answer = original_answer
            baseline_correct = original_correct
            baseline_key_name = "original_trace"
            baseline_answer_text = baseline_text
            baseline_answer_retry_count = 0

        (
            continuation_text,
            continuation_answer_text,
            target_key_name,
            edited_answer,
            edited_correct,
            edited_answer_retry_count,
        ) = continue_with_answer(edited_prefix)
        delta_cont, cont_metric = continuation_divergence(
            first_n_words(baseline_text, 128),
            first_n_words(continuation_text, 128),
        )
        answer_complete = has_answer(baseline_answer) and has_answer(edited_answer)
        delta_ans = 1 if answer_complete and (edited_answer or "") != (baseline_answer or "") else 0
        delta_corr = 1 if answer_complete and bool(edited_correct) != bool(baseline_correct) else 0

        validity = validate_intervention(
            benchmark=inst["benchmark"],
            original_step=original_step,
            edited_step=edited_step,
            schema_valid=schema_ok,
            delta_cont=delta_cont,
            gold_answer=inst["gold_answer"],
            gold_answer_text=inst.get("gold_answer_text"),
            thresholds=thresholds,
        )

        row = attempt_row(
            attempt_id=attempt_id,
            edited_step=edited_step,
            schema_ok=schema_ok,
            validity=validity,
            editor_key_name=editor_key_name,
            target_key_name=target_key_name,
            baseline_key_name=baseline_key_name,
            delta_cont=delta_cont,
            cont_metric=cont_metric,
            delta_ans=delta_ans,
            delta_corr=delta_corr,
            baseline_answer=baseline_answer or "",
            baseline_correct=baseline_correct,
            edited_answer=edited_answer or "",
            edited_correct=edited_correct,
            baseline_answer_retry_count=baseline_answer_retry_count,
            edited_answer_retry_count=edited_answer_retry_count,
            baseline_response_tail=baseline_answer_text[-1000:],
            edited_response_tail=continuation_answer_text[-1000:],
        )
        attempt_rows.append(row)
        if validity.accepted:
            accepted_attempt = row
            break
        previous_edit = edited_step
        previous_failure_reasons = validation_failure_reasons(validity, schema_ok)

    if accepted_attempt is None and attempt_rows:
        attempt_rows[-1]["unresolved_after_retries"] = True
        accepted_attempt = attempt_rows[-1]

    valid_for_utility = bool(accepted_attempt.get("valid_for_utility"))
    answer_complete = bool(accepted_attempt.get("answer_complete_flag"))
    if valid_for_utility:
        delta_ans = int(accepted_attempt["delta_ans"])
        delta_corr = int(accepted_attempt["delta_corr"])
        delta_cont = float(accepted_attempt["delta_cont"])
        utility_score = float(delta_ans) + float(delta_corr) + delta_cont
        if answer_complete and (delta_ans or delta_corr):
            step_bin = "decision-critical"
        elif delta_cont > float(thresholds["instability_delta_cont"]):
            step_bin = "supportive"
        else:
            step_bin = "redundant"
    else:
        utility_score = ""
        step_bin = "invalid-edit"
    contradiction_flag, contradiction_source = judge_contradiction(
        config=config,
        keyring=keyring_for(keyrings, f"judge::{config['judge_model']}"),
        step=original_step,
        final_answer=str(original_answer or ""),
        dry_run=dry_run,
    )
    answer_critical_flag = bool(
        valid_for_utility
        and answer_complete
        and (int(accepted_attempt["delta_ans"]) == 1 or int(accepted_attempt["delta_corr"]) == 1)
    )
    instability_flag = bool(
        valid_for_utility
        and float(accepted_attempt["delta_cont"]) >= float(thresholds["instability_delta_cont"])
    )
    high_risk_flag = bool(
        valid_for_utility
        and (
            answer_critical_flag
            or (
                contradiction_flag
                and instability_flag
                and float(utility_score) >= float(thresholds["high_risk_utility"])
            )
        )
    )
    low_priority_flag = bool(valid_for_utility and not high_risk_flag)

    utility_row = {
        "model": trace["model"],
        "benchmark": inst["benchmark"],
        "instance_id": inst["instance_id"],
        "step_id": step_id,
        "norm_position": step_id / max(1, len(steps) - 1),
        "valid_for_utility": valid_for_utility,
        "delta_ans": accepted_attempt["delta_ans"],
        "delta_corr": accepted_attempt["delta_corr"],
        "delta_cont": accepted_attempt["delta_cont"],
        "utility_score": utility_score,
        "step_bin": step_bin,
        "answer_complete_flag": answer_complete,
        "answer_critical_flag": answer_critical_flag,
        "editor_model": job["editor_name"],
    }
    region_row = {
        "model": trace["model"],
        "benchmark": inst["benchmark"],
        "instance_id": inst["instance_id"],
        "step_id": step_id,
        "valid_for_utility": valid_for_utility,
        "utility_score": utility_score,
        "contradiction_flag": contradiction_flag,
        "contradiction_source": contradiction_source,
        "instability_flag": instability_flag,
        "answer_complete_flag": answer_complete,
        "answer_critical_flag": answer_critical_flag,
        "high_risk_flag": high_risk_flag,
        "negative_support_signals": ",".join(
            name
            for name, flag in [
                ("answer_critical", answer_critical_flag),
                ("contradiction", contradiction_flag),
                ("instability", instability_flag),
            ]
            if flag
        ),
        "low_priority_flag": low_priority_flag,
        "selected_for_revision": bool(high_risk_flag),
        "selection_priority": "invalid-edit" if not valid_for_utility else "high-risk" if high_risk_flag else "low-priority",
    }
    sampled_step_row = {
        "model": trace["model"],
        "benchmark": inst["benchmark"],
        "instance_id": inst["instance_id"],
        "step_id": step_id,
        "norm_position": step_id / max(1, len(steps) - 1),
        "step_text": original_step,
        "step_token_len": len(original_step.split()),
        "segmentation_method": job["segmentation_method"],
        "num_trace_steps": job["num_trace_steps"],
        "num_eligible_steps": job["num_eligible_steps"],
    }
    return {
        "attempt_rows": attempt_rows,
        "utility_row": utility_row,
        "region_row": region_row,
        "sampled_step_row": sampled_step_row,
    }


def build_editor_comparison(interventions: list[dict], utilities: list[dict]) -> list[dict]:
    rows = []
    by_editor: dict[str, list[dict]] = {}
    for row in interventions:
        by_editor.setdefault(row["editor_model"], []).append(row)
    for editor, items in sorted(by_editor.items()):
        final_attempts = {}
        for row in items:
            key = (row["model"], row["benchmark"], row["instance_id"], row["step_id"], row["editor_model"])
            final_attempts[key] = row
        finals = list(final_attempts.values())
        valid_finals = [row for row in finals if row.get("valid_for_utility") in (True, "True", "true", 1, "1")]
        rows.append(
            {
                "editor_model": editor,
                "schema_validity_rate": mean(row.get("schema_valid") for row in items),
                "valid_edit_rate": mean(row.get("accepted") for row in finals),
                "valid_for_utility_rate": mean(row.get("valid_for_utility") for row in finals),
                "answer_complete_rate": mean(row.get("answer_complete_flag") for row in valid_finals),
                "human_valid_intervention_rate": "",
                "over_disruption_rate": mean(row.get("over_disruption_flag") for row in finals),
                "answer_leakage_rate": mean(row.get("answer_leakage_flag") for row in finals),
                "new_answer_leakage_rate": mean(row.get("new_answer_leakage_flag") for row in finals),
                "meta_commentary_rate": mean(row.get("meta_commentary_flag") for row in finals),
                "spearman_vs_reference_on_shared_subset": "",
                "mean_latency": "",
                "mean_editor_tokens": "",
                "selected_for_main_run": "",
            }
        )
    return rows


def mean(values) -> float:
    vals = [1.0 if v in (True, "True", "true", 1, "1") else 0.0 for v in values]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def read_jsonl_if_exists(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


def job_key(*, model: str, benchmark: str, instance_id: str, step_id: int | str, editor_model: str) -> str:
    return f"{model}::{benchmark}::{instance_id}::{step_id}::{editor_model}"


def job_key_from_job(job: dict) -> str:
    return job_key(
        model=job["trace"]["model"],
        benchmark=job["instance"]["benchmark"],
        instance_id=job["instance"]["instance_id"],
        step_id=job["step_id"],
        editor_model=job["editor_name"],
    )


def job_key_from_utility(row: dict) -> str:
    return job_key(
        model=str(row.get("model", "")),
        benchmark=str(row.get("benchmark", "")),
        instance_id=str(row.get("instance_id", "")),
        step_id=str(row.get("step_id", "")),
        editor_model=str(row.get("editor_model", "")),
    )


def flush_outputs(out_dir: Path) -> None:
    interventions = read_jsonl_if_exists(out_dir / "pilot_interventions.jsonl")
    utilities = read_jsonl_if_exists(out_dir / "pilot_utility_scores.jsonl")
    regions = read_jsonl_if_exists(out_dir / "pilot_uatr_region_labels.jsonl")
    sampled_steps = read_jsonl_if_exists(out_dir / "pilot_sampled_steps.jsonl")
    write_csv(out_dir / "pilot_sampled_steps.csv", sampled_steps)
    write_csv(out_dir / "pilot_interventions.csv", interventions)
    write_csv(out_dir / "pilot_utility_scores.csv", utilities)
    write_csv(out_dir / "pilot_uatr_region_labels.csv", regions)
    write_csv(out_dir / "pilot_editor_comparison.csv", build_editor_comparison(interventions, utilities))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--models", nargs="*", help="Optional target model names or model IDs to include.")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    instances = read_jsonl(args.instances)
    traces = filter_traces_for_target_panel(read_jsonl(args.traces), config, args.models)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        reset_outputs(out_dir)

    jobs = build_step_jobs(instances, traces, config, args.limit)
    jobs = stratified_limit_jobs(jobs, args.max_jobs, args.seed)
    done = {job_key_from_utility(row) for row in read_jsonl_if_exists(out_dir / "pilot_utility_scores.jsonl")}
    if done:
        jobs = [job for job in jobs if job_key_from_job(job) not in done]
        print(f"Resuming: skipped {len(done)} completed intervention rows.")

    keyrings = make_role_keyrings(config, traces, args.dry_run)

    def safe(job: dict) -> dict:
        try:
            return run_intervention_job(job, config, keyrings, args.dry_run)
        except Exception as exc:
            inst = job["instance"]
            return {
                "error": str(exc),
                "model": job["trace"]["model"],
                "benchmark": inst["benchmark"],
                "instance_id": inst["instance_id"],
                "step_id": job["step_id"],
                "editor": job["editor_name"],
            }

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(safe, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if "error" in result:
                append_jsonl(out_dir / "pilot_intervention_errors.jsonl", result)
                print(
                    f"error: {result['model']} {result['benchmark']} "
                    f"{result['instance_id']} step={result['step_id']} {result['editor']} {result['error']}"
                )
                continue
            append_jsonl(out_dir / "pilot_sampled_steps.jsonl", result["sampled_step_row"])
            for row in result["attempt_rows"]:
                append_jsonl(out_dir / "pilot_interventions.jsonl", row)
            append_jsonl(out_dir / "pilot_utility_scores.jsonl", result["utility_row"])
            append_jsonl(out_dir / "pilot_uatr_region_labels.jsonl", result["region_row"])
            completed += 1
            print(
                "ok:",
                result["utility_row"]["model"],
                result["utility_row"]["benchmark"],
                result["utility_row"]["instance_id"],
                result["utility_row"]["editor_model"],
                result["utility_row"]["step_id"],
            )
            if completed % 10 == 0:
                flush_outputs(out_dir)

    flush_outputs(out_dir)
    print(f"Wrote pilot intervention outputs to {out_dir}")


if __name__ == "__main__":
    main()

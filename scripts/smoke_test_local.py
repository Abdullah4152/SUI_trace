from __future__ import annotations

import argparse
from pathlib import Path

from pilot_lib import (
    dry_edit_step,
    extract_answer,
    first_n_words,
    is_correct,
    rule_based_contradiction,
    sample_step_indices,
    split_trace_steps,
    synthetic_trace,
    validate_intervention,
    write_csv,
    read_jsonl,
)


DEFAULT_THRESHOLDS = {
    "uptake_delta_cont_min": 0.10,
    "semantic_similarity_max": 0.95,
    "lexical_overlap_max": 0.90,
    "instability_delta_cont": 0.25,
    "over_disruption_delta_cont": 0.60,
    "high_risk_utility": 0.50,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    instances = read_jsonl(args.instances)[: args.limit]
    rows = []
    for inst in instances:
        trace = synthetic_trace(inst)
        pred = extract_answer(inst["benchmark"], trace)
        correct = is_correct(inst["benchmark"], pred, inst["gold_answer"])
        steps, method = split_trace_steps(trace)
        indices = sample_step_indices(len(steps), 4, 17)
        for idx in indices:
            edited = dry_edit_step(steps[idx])
            delta_cont = 0.30
            validity = validate_intervention(
                benchmark=inst["benchmark"],
                original_step=steps[idx],
                edited_step=edited,
                schema_valid=True,
                delta_cont=delta_cont,
                gold_answer=inst["gold_answer"],
                gold_answer_text=inst.get("gold_answer_text"),
                thresholds=DEFAULT_THRESHOLDS,
            )
            contradiction = rule_based_contradiction(steps[idx], pred)
            rows.append(
                {
                    "benchmark": inst["benchmark"],
                    "instance_id": inst["instance_id"],
                    "predicted_answer": pred,
                    "correct": correct,
                    "num_steps": len(steps),
                    "segmentation_method": method,
                    "sampled_step_id": idx,
                    "step_preview": first_n_words(steps[idx], 20),
                    "accepted": validity.accepted,
                    "over_disruption_flag": validity.over_disruption_flag,
                    "answer_leakage_flag": validity.answer_leakage_flag,
                    "contradiction_flag": contradiction,
                    "instability_flag": delta_cont > DEFAULT_THRESHOLDS["instability_delta_cont"],
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "local_smoke_checks.csv", rows)
    print(f"Wrote {len(rows)} smoke rows to {out_dir / 'local_smoke_checks.csv'}")
    print("Local smoke test complete.")


if __name__ == "__main__":
    main()

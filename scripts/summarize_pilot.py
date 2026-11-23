from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_rows(out_dir: Path, stem: str, dedupe_field: str | None = None) -> list[dict]:
    rows = read_jsonl(out_dir / f"{stem}.jsonl")
    if not rows:
        rows = read_csv(out_dir / f"{stem}.csv")
    if dedupe_field:
        latest = {}
        for row in rows:
            latest[str(row.get(dedupe_field))] = row
        rows = list(latest.values())
    return rows


def as_bool(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def mean_bool(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if as_bool(row.get(field))) / len(rows)


def mean_float(rows: list[dict], field: str) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(field, "")))
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else 0.0


def count_value(rows: list[dict], field: str, value: str) -> int:
    return sum(1 for row in rows if str(row.get(field, "")) == value)


def group_by(rows: list[dict], field: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in rows:
        out.setdefault(str(row.get(field, "")), []).append(row)
    return out


def final_attempts(rows: list[dict]) -> list[dict]:
    latest = {}
    for row in rows:
        key = (
            row.get("model"),
            row.get("benchmark"),
            row.get("instance_id"),
            row.get("step_id"),
            row.get("editor_model"),
        )
        latest[key] = row
    return list(latest.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    traces = read_rows(out_dir, "pilot_trace_metadata", "trace_id")
    interventions = read_rows(out_dir, "pilot_interventions")
    utilities = read_rows(out_dir, "pilot_utility_scores")
    regions = read_rows(out_dir, "pilot_uatr_region_labels")
    errors = read_rows(out_dir, "pilot_intervention_errors")
    editor_rows = read_rows(out_dir, "pilot_editor_comparison")

    lines = []
    lines.append("# SUI Pilot Summary")
    lines.append("")
    lines.append(f"- trace rows: {len(traces)}")
    lines.append(f"- intervention attempt rows: {len(interventions)}")
    lines.append(f"- utility rows: {len(utilities)}")
    lines.append(f"- intervention error rows: {len(errors)}")
    lines.append(f"- UATR region rows: {len(regions)}")
    lines.append("")

    if traces:
        ok = mean_bool(traces, "status") if "status" in traces[0] else 0.0
        lines.append("## Trace Generation")
        lines.append(f"- ok rows: {sum(1 for r in traces if r.get('status') == 'ok')} / {len(traces)}")
        lines.append("")

    if interventions:
        finals = final_attempts(interventions)
        valid_utils = [row for row in finals if as_bool(row.get("valid_for_utility"))]
        lines.append("## Intervention Validity")
        lines.append(f"- final attempted step rows: {len(finals)}")
        if finals or errors:
            lines.append(f"- target/API error rate: {len(errors) / max(1, len(finals) + len(errors)):.3f}")
        lines.append(f"- schema validity rate over attempts: {mean_bool(interventions, 'schema_valid'):.3f}")
        lines.append(f"- valid edit rate over final rows: {mean_bool(finals, 'accepted'):.3f}")
        lines.append(f"- valid-for-utility rate over final rows: {mean_bool(finals, 'valid_for_utility'):.3f}")
        lines.append(f"- answer-complete rate over valid utility rows: {mean_bool(valid_utils, 'answer_complete_flag'):.3f}")
        lines.append(f"- new answer leakage rate over final rows: {mean_bool(finals, 'new_answer_leakage_flag'):.3f}")
        lines.append(f"- meta-commentary rate over final rows: {mean_bool(finals, 'meta_commentary_flag'):.3f}")
        lines.append(f"- over-disruption rate over valid utility rows: {mean_bool(valid_utils, 'over_disruption_flag'):.3f}")
        lines.append(f"- mean delta_cont over valid utility rows: {mean_float(valid_utils, 'delta_cont'):.3f}")
        lines.append("")
        lines.append("### By Editor")
        for editor, rows in sorted(group_by(interventions, "editor_model").items()):
            editor_finals = final_attempts(rows)
            editor_valid = [row for row in editor_finals if as_bool(row.get("valid_for_utility"))]
            lines.append(
                f"- {editor}: valid_edit={mean_bool(editor_finals, 'accepted'):.3f}, "
                f"valid_for_utility={mean_bool(editor_finals, 'valid_for_utility'):.3f}, "
                f"answer_complete={mean_bool(editor_valid, 'answer_complete_flag'):.3f}, "
                f"new_leakage={mean_bool(editor_finals, 'new_answer_leakage_flag'):.3f}, "
                f"meta={mean_bool(editor_finals, 'meta_commentary_flag'):.3f}, "
                f"over_disruption_valid={mean_bool(editor_valid, 'over_disruption_flag'):.3f}"
            )
        lines.append("")

    if utilities:
        valid_utilities = [row for row in utilities if as_bool(row.get("valid_for_utility"))]
        lines.append("## Utility")
        lines.append(f"- valid utility rows: {len(valid_utilities)} / {len(utilities)}")
        lines.append(f"- mean utility score over valid rows: {mean_float(valid_utilities, 'utility_score'):.3f}")
        lines.append(f"- mean delta_ans over valid rows: {mean_float(valid_utilities, 'delta_ans'):.3f}")
        lines.append(f"- mean delta_corr over valid rows: {mean_float(valid_utilities, 'delta_corr'):.3f}")
        lines.append(f"- mean delta_cont over valid rows: {mean_float(valid_utilities, 'delta_cont'):.3f}")
        lines.append(f"- decision-critical rows: {count_value(valid_utilities, 'step_bin', 'decision-critical')} / {len(valid_utilities)}")
        lines.append(f"- supportive rows: {count_value(valid_utilities, 'step_bin', 'supportive')} / {len(valid_utilities)}")
        lines.append(f"- redundant rows: {count_value(valid_utilities, 'step_bin', 'redundant')} / {len(valid_utilities)}")
        lines.append("")

    if regions:
        lines.append("## UATR Region Labels")
        lines.append(f"- contradiction rate: {mean_bool(regions, 'contradiction_flag'):.3f}")
        lines.append(f"- instability rate: {mean_bool(regions, 'instability_flag'):.3f}")
        lines.append(f"- high-risk rate: {mean_bool(regions, 'high_risk_flag'):.3f}")
        lines.append(f"- low-priority rate: {mean_bool(regions, 'low_priority_flag'):.3f}")
        lines.append(f"- selected-for-revision rate: {mean_bool(regions, 'selected_for_revision'):.3f}")
        lines.append("")

    if editor_rows:
        lines.append("## Editor Comparison Rows")
        for row in editor_rows:
            lines.append(
                f"- {row.get('editor_model')}: schema={row.get('schema_validity_rate')}, "
                f"valid_edit={row.get('valid_edit_rate')}, "
                f"valid_for_utility={row.get('valid_for_utility_rate')}, "
                f"answer_complete={row.get('answer_complete_rate')}, "
                f"over={row.get('over_disruption_rate')}, "
                f"new_leak={row.get('new_answer_leakage_rate')}, "
                f"meta={row.get('meta_commentary_rate')}"
            )
        lines.append("")

    lines.append("## Full-Run Gate Reminder")
    lines.append("- valid edit rate should be at least 0.70 before full rerun")
    lines.append("- target API error rate should be below 0.05")
    lines.append("- answer-complete rate over valid utility rows should be at least 0.90")
    lines.append("- new answer leakage rate should be at most 0.02")
    lines.append("- meta-commentary rate should be at most 0.05")
    lines.append("- selected-for-revision rate should be materially below 0.50 for UATR to count as targeted")

    summary_path = out_dir / "pilot_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

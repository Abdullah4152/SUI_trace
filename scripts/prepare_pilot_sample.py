from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from pilot_lib import (
    build_gpqa_prompt,
    build_logiqa_prompt,
    build_math_prompt,
    extract_boxed_answer,
    load_config,
    read_jsonl,
    write_csv,
    write_jsonl,
)


def sample_rows(df: pd.DataFrame, n: int, seed: int, strata: list[str] | None = None) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    if not strata:
        return df.sample(n=n, random_state=seed).copy()

    parts = []
    per_group = max(1, n // max(1, df.groupby(strata).ngroups))
    for _, group in df.groupby(strata, dropna=False):
        parts.append(group.sample(n=min(per_group, len(group)), random_state=seed))
    sampled = pd.concat(parts) if parts else df.head(0)
    if len(sampled) < n:
        remaining = df.drop(index=sampled.index, errors="ignore")
        extra = remaining.sample(n=min(n - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra])
    return sampled.sample(n=min(n, len(sampled)), random_state=seed).copy()


def load_math(path: str, n: int, seed: int) -> list[dict]:
    df = pd.read_parquet(path)
    df = df.copy()
    df["ground_truth"] = df["solution"].apply(extract_boxed_answer)
    df = df[df["ground_truth"].notna()].reset_index(drop=False).rename(columns={"index": "source_index"})
    sample = sample_rows(df, n, seed, ["level", "type"])
    rows = []
    for _, row in sample.iterrows():
        rows.append(
            {
                "benchmark": "MATH-500",
                "instance_id": f"math_{int(row['source_index'])}",
                "source_path": path,
                "prompt": build_math_prompt(row["problem"]),
                "question": row["problem"],
                "gold_answer": row["ground_truth"],
                "gold_answer_text": row["ground_truth"],
                "metadata": {
                    "level": row.get("level"),
                    "type": row.get("type"),
                    "source_index": int(row["source_index"]),
                    "note": "Kaggle folder says MATH-500, file contains full MATH export.",
                },
            }
        )
    return rows


def load_gpqa(path: str, n: int, seed: int) -> list[dict]:
    df = pd.read_csv(path)
    sample = sample_rows(df, n, seed, ["High-level domain"])
    rows = []
    for idx, row in sample.reset_index(drop=True).iterrows():
        rng = random.Random(f"{seed}:{row['Record ID']}")
        options = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        rng.shuffle(options)
        correct_letter = chr(65 + options.index(row["Correct Answer"]))
        rows.append(
            {
                "benchmark": "GPQA Diamond",
                "instance_id": f"gpqa_{row['Record ID']}",
                "source_path": path,
                "prompt": build_gpqa_prompt(row["Question"], options),
                "question": row["Question"],
                "gold_answer": correct_letter,
                "gold_answer_text": row["Correct Answer"],
                "metadata": {
                    "record_id": row["Record ID"],
                    "domain": row.get("High-level domain"),
                    "subdomain": row.get("Subdomain"),
                    "option_A": options[0],
                    "option_B": options[1],
                    "option_C": options[2],
                    "option_D": options[3],
                },
            }
        )
    return rows


def load_logiqa(path: str, n: int, seed: int) -> list[dict]:
    data = read_jsonl(path)
    df = pd.DataFrame(data).reset_index(drop=False).rename(columns={"index": "source_index"})
    sample = sample_rows(df, n, seed, ["label"])
    rows = []
    for _, row in sample.iterrows():
        rows.append(
            {
                "benchmark": "LogiQA 2.0",
                "instance_id": f"logiqa_{int(row['source_index'])}",
                "source_path": path,
                "prompt": build_logiqa_prompt(row["premise"], row.get("hypothesis", "")),
                "question": f"Premise: {row['premise']}\nHypothesis: {row.get('hypothesis', '')}",
                "gold_answer": row["label"],
                "gold_answer_text": row["label"],
                "metadata": {"id": row["id"], "source_index": int(row["source_index"])},
            }
        )
    return rows


def flatten_for_csv(row: dict) -> dict:
    flat = dict(row)
    meta = flat.pop("metadata", {}) or {}
    for key, value in meta.items():
        flat[f"metadata_{key}"] = value
    return flat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.out_dir)
    n = int(config["pilot"]["instances_per_benchmark"])
    seed = int(config["pilot"]["seed"])
    paths = config["dataset_paths"]

    rows: list[dict] = []
    rows.extend(load_math(paths["math"], n, seed))
    rows.extend(load_gpqa(paths["gpqa_diamond"], n, seed + 1))
    rows.extend(load_logiqa(paths["logiqa_test"], n, seed + 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "pilot_instances.jsonl", rows)
    write_csv(out_dir / "pilot_instances.csv", [flatten_for_csv(r) for r in rows])

    counts = {}
    for row in rows:
        counts[row["benchmark"]] = counts.get(row["benchmark"], 0) + 1
    print(f"Wrote {len(rows)} pilot instances to {out_dir}")
    print(counts)


if __name__ == "__main__":
    main()

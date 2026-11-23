# Runbook

Step-by-step execution guide for a hosted notebook environment (the setup the original
long-running experiment used). For a plain local run, follow the README instead — the commands
are the same, only the paths and secret handling differ.

This artifact ships no generated outputs. Every stage below writes its own results into the
output directory you choose.

## Cell 0: Copy Artifact To Working Directory

```python
import os, shutil

src = "/kaggle/input/YOUR-UPLOADED-ARTIFACT/sui_trace_artifact"
dst = "/kaggle/working/sui_trace_artifact"

if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
os.chdir(dst)
print(os.getcwd())
```

If your benchmark datasets are mounted separately, edit `config/sui_final_config.json`:

```json
"dataset_paths": {
  "math": "/kaggle/input/.../MATH-500/train-00000-of-00001-7320a6f3aba8ebd2.parquet",
  "gpqa_diamond": "/kaggle/input/.../GPQA Diamond/gpqa_diamond.csv",
  "logiqa_test": "/kaggle/input/.../LogiQA 2.0/test.jsonl"
}
```

## Cell 1: Load API Keys

Use the notebook secret store. The artifact expects generic anonymous names.

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
key_names = [f"NVIDIA_API_KEY_{i}" for i in range(1, 9)]

for name in key_names:
    os.environ[name] = secrets.get_secret(name)

print("Loaded", len(key_names), "keys")
```

Supply fewer keys by shortening this list, lowering `--workers` to match, and editing
`api_key_env_names` in the config.

## Cell 2: Install Dependencies

```python
!pip install -q -r requirements.txt
```

## Cell 3: Human Validation Analysis

This is the only analysis that runs without API access, because the annotations are included in
the artifact. It is a useful first check that the environment is working.

```python
!python human_validation/analyze_human_annotations.py \
  --annotations human_validation/human_validation_annotation_sheet_filled.csv \
  --out-dir /kaggle/working/outputs/human_validation_main \
  --split main

!python human_validation/analyze_human_annotations.py \
  --annotations human_validation/human_validation_annotation_sheet_filled.csv \
  --out-dir /kaggle/working/outputs/human_validation_calibration \
  --split calibration
```

## Cell 4: SUI Audit

```python
!python scripts/prepare_pilot_sample.py \
  --config config/sui_final_config.json \
  --out-dir /kaggle/working/outputs/sui

!python scripts/run_pilot_generation.py \
  --config config/sui_final_config.json \
  --instances /kaggle/working/outputs/sui/pilot_instances.jsonl \
  --out-dir /kaggle/working/outputs/sui \
  --workers 8

!python scripts/run_pilot_interventions.py \
  --config config/sui_final_config.json \
  --instances /kaggle/working/outputs/sui/pilot_instances.jsonl \
  --traces /kaggle/working/outputs/sui/pilot_trace_metadata.jsonl \
  --out-dir /kaggle/working/outputs/sui \
  --workers 8

!python scripts/summarize_pilot.py --out-dir /kaggle/working/outputs/sui
```

The interventions stage is the long one and is resumable. If a session times out, re-run the
same command and it continues from existing output. Check progress with:

```python
!python scripts/report_resume_status.py \
  --config config/sui_final_config.json \
  --instances /kaggle/working/outputs/sui/pilot_instances.jsonl \
  --traces /kaggle/working/outputs/sui/pilot_trace_metadata.jsonl \
  --out-dir /kaggle/working/outputs/sui
```

## Cell 5: UATR Revision

```python
!python scripts/prepare_uatr_jobs.py \
  --sui-dir /kaggle/working/outputs/sui \
  --out-dir /kaggle/working/outputs/uatr

!python scripts/run_uatr_experiment.py \
  --config config/sui_final_config.json \
  --sui-dir /kaggle/working/outputs/sui \
  --jobs /kaggle/working/outputs/uatr/uatr_jobs.jsonl \
  --out-dir /kaggle/working/outputs/uatr \
  --workers 8

!python scripts/summarize_uatr_experiment.py --out-dir /kaggle/working/outputs/uatr
```

Before comparing methods, check how many jobs actually completed. `prepare_uatr_jobs.py`
accepts `--max-jobs` and `--max-per-model-benchmark`, so a job set can be much smaller and less
balanced across models than the SUI audit it was derived from. Report the per-method row count
and the model/benchmark breakdown alongside any comparison.

## Cell 6: Aggregate Statistics

```python
!python human_validation/compute_emnlp_statistics.py \
  --sui-dir /kaggle/working/outputs/sui \
  --uatr-dir /kaggle/working/outputs/uatr \
  --out-dir /kaggle/working/outputs/statistics
```

## Cell 7: Save Outputs

```python
import shutil

shutil.make_archive("/kaggle/working/outputs_archive", "zip", "/kaggle/working/outputs")
```

## Runtime Notes

- The configuration uses a 1800 second request timeout.
- Each transient failure is retried twice.
- Workers sleep for 30 seconds after successful API responses.
- With 8 independent API keys, use `--workers 8`. With fewer keys, lower `--workers` to match.
- Use `--dry-run` and `--limit` on the generation, intervention, and revision stages to size a
  run before committing budget to it.
- `scripts/smoke_test_local.py` exercises segmentation and scoring without API access.

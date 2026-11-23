# Human Validation Instructions

## Purpose

This annotation checks whether the step-level interventions used in the SUI audit are valid enough to support the paper's methodological claims.

You are not judging whether the model is generally intelligent. You are judging whether a single edited reasoning step is a valid intervention on the original reasoning trace.

## Files

Annotate:

- `human_validation_annotation_sheet.csv`

Reference-only metadata:

- `human_validation_metadata.csv`
- `human_validation_sample.jsonl`

Do not edit metadata files.

## Raters

Use 3 independent raters.

All raters should annotate:

1. The `calibration` split first.
2. Discuss disagreements after calibration.
3. Then annotate the `main` split independently.

For the final reliability report, compute agreement on the `main` split. Calibration items are for alignment, not final claims.

## What You See

Each row contains:

- `original_step`: the original model reasoning step.
- `edited_step`: the edited/intervened step.
- `baseline_response_tail`: the final portion of the model continuation from the original trace/prefix.
- `edited_response_tail`: the final portion of the model continuation after the edited step.
- automatic labels such as `step_bin` are included for later analysis, but raters should not use them as ground truth.

## Annotation Fields

Fill these columns.

### `rater_id`

Use a stable anonymized ID:

- `R1`
- `R2`
- `R3`

### `semantic_change_score`

Question: Does the edited step meaningfully change the intermediate reasoning content?

Use:

- `1`: no meaningful semantic change; mostly formatting/paraphrase.
- `2`: very small change; unlikely to affect reasoning.
- `3`: moderate change to an intermediate claim or inference.
- `4`: clear substantive change.
- `5`: major change that substantially alters the reasoning path.

Recommended validity threshold for analysis: `>= 3`.

### `local_coherence_score`

Question: Does the edited step remain locally coherent with the surrounding task and step content?

Use:

- `1`: incoherent or unrelated.
- `2`: mostly incoherent.
- `3`: partly coherent but awkward or underspecified.
- `4`: coherent with minor issues.
- `5`: fully coherent and plausible as a reasoning step.

Recommended validity threshold for analysis: `>= 3`.

### `fluency_score`

Question: Is the edited step grammatical and readable?

Use:

- `1`: unreadable.
- `2`: hard to read.
- `3`: understandable but rough.
- `4`: fluent with minor issues.
- `5`: fully fluent.

Recommended validity threshold for analysis: `>= 3`.

### `answer_leakage_label`

Question: Does the edited step reveal or copy the benchmark gold answer in a way that would unfairly help the model?

Use:

- `0`: no answer leakage.
- `1`: possible leakage or suspicious answer cue.
- `2`: clear answer leakage.

For MATH, direct appearance of the final numeric/symbolic answer can be leakage if it was not already present in the original step.

For GPQA, direct assertion of the correct option letter can be leakage.

For LogiQA, direct assertion of `entailment` or `not-entailment` as the final label can be leakage if it is not part of legitimate local reasoning.

### `over_disruption_label`

Question: Does the edited step disrupt the trace so much that it no longer isolates a local step contribution?

Use:

- `0`: not over-disruptive; local intervention.
- `1`: borderline; the edit may be too broad.
- `2`: clearly over-disruptive; changes the problem, introduces a new solution, or breaks local isolation.

### `valid_intervention_label`

Overall judgment: Is this a valid intervention for the SUI audit?

Use:

- `0`: invalid.
- `1`: borderline/uncertain.
- `2`: valid.

Use `2` only if:

- semantic change score is at least 3,
- local coherence score is at least 3,
- fluency score is at least 3,
- answer leakage is not clear,
- over-disruption is not clear.

### `human_failure_mode`

Choose one primary failure mode if `valid_intervention_label` is `0` or `1`.

Allowed labels:

- `none`
- `paraphrase_only`
- `incoherent_edit`
- `ungrammatical`
- `answer_leakage`
- `over_disruptive`
- `wrong_scope`
- `other`

Use `none` for valid interventions.

### `comment`

Optional short explanation. Use this for unusual cases, especially when marking an intervention invalid.

## Examples

### Valid

Original step: "Since the object accelerates uniformly, use v = u + at."

Edited step: "Since the acceleration is not uniform, a constant-acceleration equation is not directly applicable."

This is a meaningful, local, fluent change. It may alter downstream reasoning. Mark as valid unless it leaks the final answer.

### Invalid: Paraphrase Only

Original step: "Therefore x equals 4."

Edited step: "So x = 4."

This is fluent and coherent but not a meaningful semantic change. Mark `semantic_change_score = 1`, `human_failure_mode = paraphrase_only`.

### Invalid: Answer Leakage

Original step: "We now solve for the unknown."

Edited step: "The correct final answer is B, so the intermediate claim should support B."

This leaks the answer. Mark answer leakage as `2`.

### Borderline: Over-Disruptive

Original step: a single local algebra step.

Edited step: an entirely new multi-paragraph solution with a different method.

Even if correct, this is too broad for a step-level intervention. Mark over-disruption as `2`.

## Reliability Targets

Report:

- Krippendorff's alpha for `valid_intervention_label`.
- Krippendorff's alpha or weighted kappa for ordinal fields:
  - `semantic_change_score`
  - `local_coherence_score`
  - `fluency_score`
  - `answer_leakage_label`
  - `over_disruption_label`

Minimum desirable agreement:

- `valid_intervention_label`: alpha >= 0.67 acceptable, >= 0.80 strong.
- Ordinal fields: alpha >= 0.60 acceptable for exploratory validation.

## Reporting

In the paper, report:

- number of annotated items,
- number of raters,
- agreement score,
- valid intervention rate,
- answer leakage rate,
- over-disruption rate,
- examples of common invalid edits.

Use the `main` split for final numbers. Use the `calibration` split only to describe the calibration process if needed.

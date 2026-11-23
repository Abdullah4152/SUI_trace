from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def configured_key_names(
    config: dict[str, Any],
    *,
    role: str = "main",
    model_name: str | None = None,
    model_id: str | None = None,
) -> list[str]:
    lookup_keys = [key for key in [model_name, model_id] if key]
    mapping_names = {
        "main": ["model_api_key_env_names"],
        "editor": ["editor_api_key_env_names", "model_api_key_env_names"],
        "judge": ["judge_api_key_env_names", "editor_api_key_env_names", "model_api_key_env_names"],
    }.get(role, ["model_api_key_env_names"])
    for mapping_name in mapping_names:
        mapping = config.get(mapping_name, {}) or {}
        for key in lookup_keys:
            names = mapping.get(key)
            if names:
                return list(names)
        names = mapping.get("default")
        if names:
            return list(names)
    return list(config.get("api_key_env_names", []))


def load_api_keys(config: dict[str, Any], names: list[str] | None = None) -> list[tuple[str, str]]:
    explicit_names = names is not None
    names = list(names if names is not None else config.get("api_key_env_names", []))
    if not explicit_names:
        names.extend(sorted(name for name in os.environ if name.startswith("NVIDIA_API_KEY")))
    keys: list[tuple[str, str]] = []
    seen: set[str] = set()

    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        UserSecretsClient = None  # type: ignore

    secret_client = None
    if UserSecretsClient is not None:
        try:
            secret_client = UserSecretsClient()
        except Exception:
            secret_client = None

    for name in names:
        value = os.environ.get(name)
        if not value and secret_client is not None:
            try:
                value = secret_client.get_secret(name)
            except Exception:
                value = None
        if value and value not in seen:
            keys.append((name, value))
            seen.add(value)
    return keys


def runtime_settings(config: dict[str, Any], model_id: str, profile: str) -> dict[str, Any]:
    """Resolve model-specific runtime settings.

    profile is one of: main, editor, judge. Missing model-specific values fall
    back to the protocol-level decoding defaults.
    """
    decoding = config.get("decoding", {})
    defaults = {
        "main": {
            "temperature": decoding.get("main_temperature", 0.0),
            "top_p": decoding.get("main_top_p", 1.0),
            "max_tokens": decoding.get("main_max_tokens", 8192),
            "stream": True,
            "extra": {},
        },
        "continuation": {
            "temperature": decoding.get("continuation_temperature", decoding.get("main_temperature", 0.0)),
            "top_p": decoding.get("continuation_top_p", decoding.get("main_top_p", 1.0)),
            "max_tokens": decoding.get("continuation_max_tokens", 4096),
            "stream": True,
            "extra": {},
        },
        "editor": {
            "temperature": decoding.get("editor_temperature", 0.2),
            "top_p": decoding.get("editor_top_p", 0.9),
            "max_tokens": decoding.get("editor_max_tokens", 256),
            "stream": False,
            "extra": {},
        },
        "judge": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 256,
            "stream": False,
            "extra": {},
        },
    }
    resolved = dict(defaults[profile])
    resolved["extra"] = dict(resolved.get("extra", {}))
    override = config.get("model_settings", {}).get(model_id, {}).get(profile, {})
    for key, value in override.items():
        if key == "extra":
            resolved["extra"] = dict(value or {})
        else:
            resolved[key] = value
    return resolved


class KeyRing:
    def __init__(self, keys: list[tuple[str, str]]):
        if not keys:
            raise RuntimeError("No NVIDIA API keys found. Set the configured NVIDIA_API_KEY_* Kaggle secrets.")
        self.keys = keys
        self.index = 0
        self.lock = threading.Lock()
        self.semaphore = threading.BoundedSemaphore(len(keys))

    def next(self) -> tuple[str, str]:
        with self.lock:
            key = self.keys[self.index % len(self.keys)]
            self.index += 1
            return key

    @contextmanager
    def lease(self):
        self.semaphore.acquire()
        try:
            yield self.next()
        finally:
            self.semaphore.release()


def call_nvidia_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int = 900,
    retries: int = 2,
    retry_sleep: float = 20.0,
    response_cooldown: float = 0.0,
    stream: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if extra:
        payload.update(extra)

    data: dict[str, Any] | None = None
    started = time.time()
    retryable_http = {408, 409, 425, 429, 500, 502, 503, 504}
    for attempt in range(1, retries + 2):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if stream:
                    content_parts: list[str] = []
                    reasoning_parts: list[str] = []
                    finish_reason = None
                    response_model = model
                    usage: dict[str, Any] = {}
                    saw_event = False
                    for raw_line in response:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        if line.startswith("data:"):
                            line = line[len("data:") :].strip()
                        if line == "[DONE]":
                            break
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        saw_event = True
                        response_model = event.get("model", response_model)
                        if event.get("usage"):
                            usage = event.get("usage", {})
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        choice = choices[0] or {}
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or choice.get("message") or {}
                        if delta.get("reasoning_content"):
                            reasoning_parts.append(str(delta.get("reasoning_content")))
                        if delta.get("content"):
                            content_parts.append(str(delta.get("content")))
                    if not saw_event:
                        raise RuntimeError(f"No stream events returned for {model}")
                    if response_cooldown > 0:
                        time.sleep(response_cooldown)
                    return {
                        "response_text": "".join(content_parts).strip(),
                        "reasoning_content": "".join(reasoning_parts).strip(),
                        "finish_reason": finish_reason,
                        "response_model": response_model,
                        "usage": usage,
                        "latency_s": round(time.time() - started, 3),
                    }
                data = json.loads(response.read().decode("utf-8"))
                if response_cooldown > 0:
                    time.sleep(response_cooldown)
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = f"HTTP {exc.code} for {model} on attempt {attempt}: {body[:1200]}"
            if exc.code in retryable_http and attempt <= retries:
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(message) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            message = f"Request failure for {model} on attempt {attempt}: {exc}"
            if attempt <= retries:
                time.sleep(retry_sleep)
                continue
            raise RuntimeError(message) from exc
    if data is None:
        raise RuntimeError(f"No response data returned for {model}")
    choice = data["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return {
        "response_text": content.strip(),
        "reasoning_content": reasoning.strip(),
        "finish_reason": choice.get("finish_reason"),
        "response_model": data.get("model", model),
        "usage": data.get("usage", {}),
        "latency_s": round(time.time() - started, 3),
    }


def trace_text(row: dict[str, Any]) -> str:
    """Return the full auditable trace emitted by NVIDIA-style reasoning APIs."""
    reasoning = str(row.get("reasoning_content") or "").strip()
    response = str(row.get("response_text") or "").strip()
    if reasoning and response:
        if reasoning in response:
            return response
        return f"{reasoning}\n\nFinal response:\n{response}"
    return reasoning or response


def has_final_answer_cue(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(
            r"\b(final\s+answer|answer\s+is|correct\s+answer|correct\s+option|therefore|thus|so\s+the\s+answer|choose\s+option)\b",
            lowered,
        )
    ) or "\\boxed" in lowered


def has_meta_commentary(text: str) -> bool:
    lowered = text.lower().strip()
    meta_patterns = [
        r"^okay[, ]+let'?s",
        r"^let me",
        r"\bi need to edit\b",
        r"\bthe current step\b",
        r"\boriginal current step\b",
        r"\bfor this intervention study\b",
        r"\bthe task is to\b",
        r"\brequirements are\b",
        r"\breturn json\b",
        r"\bedited_step\b",
    ]
    return any(re.search(pattern, lowered) for pattern in meta_patterns)


def is_intervenable_step(
    step: str,
    benchmark: str | None = None,
    gold_answer: str | None = None,
    gold_answer_text: str | None = None,
) -> bool:
    text = step.strip()
    if len(text.split()) < 8:
        return False
    lowered = text.lower()
    if lowered.startswith(("final answer", "final response", "answer:", "\\boxed")):
        return False
    if benchmark and gold_answer and contains_answer_leak(benchmark, text, gold_answer, gold_answer_text):
        return False
    return True


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[^a-z0-9.+\-/%= ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_boxed_answer(text: str | None) -> str | None:
    if not text:
        return None
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    out: list[str] = []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        out.append(ch)
        i += 1
    return "".join(out).strip() if depth == 0 else None


def extract_answer(benchmark: str, response: str) -> str | None:
    if benchmark == "MATH-500":
        boxed = extract_boxed_answer(response)
        if boxed:
            return boxed
        match = re.search(r"final answer\s*[:=]\s*([^\n.]+)", response, re.I)
        return match.group(1).strip() if match else None
    if benchmark == "GPQA Diamond":
        match = re.search(r"final answer\s*[:=]?\s*([A-D])\b", response, re.I)
        if match:
            return match.group(1).upper()
        matches = re.findall(r"\b([A-D])\b", response)
        return matches[-1].upper() if matches else None
    if benchmark == "LogiQA 2.0":
        text = response.lower()
        matches = re.findall(r"\bnot[- ]entailment\b|\bentailment\b", text)
        if matches:
            last = matches[-1].replace(" ", "-")
            return "not-entailment" if last == "not-entailment" else "entailment"
        return None
    return None


def is_correct(benchmark: str, pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    if benchmark == "MATH-500":
        p = normalize_text(pred)
        g = normalize_text(gold)
        if p == g:
            return True
        try:
            return math.isclose(float(p), float(g), rel_tol=1e-9, abs_tol=1e-9)
        except Exception:
            return False
    return normalize_text(pred) == normalize_text(gold)


def build_math_prompt(problem: str) -> str:
    return (
        "You are an expert mathematics solver. Solve the problem step by step. "
        "At the end, write the final answer inside \\boxed{}.\n\n"
        f"Problem:\n{problem}\n\nSolution:"
    )


def build_gpqa_prompt(question: str, options: list[str]) -> str:
    return (
        "Answer the following multiple-choice science question. Think step by step, "
        "then end with exactly 'Final answer: X' where X is A, B, C, or D.\n\n"
        f"Question: {question}\n\n"
        f"A) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}\n\n"
        "Reasoning:"
    )


def build_logiqa_prompt(premise: str, hypothesis: str) -> str:
    hypothesis = hypothesis.strip() or "[empty hypothesis]"
    return (
        "Premise:\n"
        f"{premise.strip()}\n\n"
        "Hypothesis:\n"
        f"{hypothesis}\n\n"
        "First reason step by step. Then end with exactly "
        "'Final answer: entailment' or 'Final answer: not-entailment'.\n\n"
        "Reasoning:"
    )


def extract_think_content(text: str) -> tuple[str, bool]:
    blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.I | re.S)
    if blocks:
        return "\n\n".join(block.strip() for block in blocks if block.strip()), True
    return text, False


def split_trace_steps(text: str) -> tuple[list[str], str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text, had_think = extract_think_content(text)
    chunks = [c.strip() for c in re.split(r"\n\s*\n+", text) if c.strip()]
    if len(chunks) >= 4:
        return chunks, "think_block_double_newline" if had_think else "double_newline"

    marker_chunks = [c.strip() for c in re.split(r"(?:^|\n)\s*(?:step\s*\d+[:.)-]?|\d+[:.)-]|\-|\*)\s+", text, flags=re.I) if c.strip()]
    if len(marker_chunks) >= 4:
        return marker_chunks, "marker_fallback"

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    merged: list[str] = []
    buf: list[str] = []
    for sentence in sentences:
        buf.append(sentence)
        if len(" ".join(buf).split()) >= 25:
            merged.append(" ".join(buf))
            buf = []
    if buf:
        merged.append(" ".join(buf))
    return (merged if merged else [text]), "sentence_merge_fallback"


def sample_step_indices(num_steps: int, n: int, seed: int) -> list[int]:
    if num_steps <= n:
        return list(range(num_steps))
    rng = random.Random(seed)
    if n == 4:
        bins = [(0.0, 0.33, 1), (0.33, 0.66, 1), (0.66, 1.0, 2)]
    else:
        bins = [(0.0, 0.33, 2), (0.33, 0.66, 3), (0.66, 1.0, max(1, n - 5))]
    selected: set[int] = set()
    for lo, hi, count in bins:
        start = int(math.floor(lo * num_steps))
        end = int(math.ceil(hi * num_steps))
        choices = list(range(start, min(end, num_steps)))
        if choices:
            selected.update(rng.sample(choices, min(count, len(choices))))
    while len(selected) < min(n, num_steps):
        selected.add(rng.randrange(num_steps))
    return sorted(selected)


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def lexical_overlap(a: str, b: str) -> float:
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def continuation_divergence(original: str, edited: str) -> tuple[float, str]:
    try:
        from bert_score import score  # type: ignore

        _, _, f1 = score([edited], [original], lang="en", verbose=False, rescale_with_baseline=False)
        similarity = float(f1[0])
        return max(0.0, min(1.0, 1.0 - similarity)), "bertscore"
    except Exception:
        return max(0.0, min(1.0, 1.0 - text_similarity(original, edited))), "sequence_matcher"


def first_n_words(text: str, n: int = 128) -> str:
    return " ".join(text.split()[:n])


def contains_answer_leak(
    benchmark: str,
    text: str,
    gold_answer: str,
    gold_answer_text: str | None = None,
) -> bool:
    raw = text or ""
    lowered = raw.lower()
    hay = normalize_text(raw)
    gold = normalize_text(gold_answer)
    gold_text = normalize_text(gold_answer_text or "")
    if not gold:
        return False

    if benchmark == "GPQA Diamond":
        letter = re.escape(gold.upper())
        explicit_letter = re.search(
            rf"\b(final\s+answer|answer\s+is|correct\s+(?:answer|option)|choose\s+option|option)\s*[:\-]?\s*{letter}\b",
            raw,
            flags=re.I,
        )
        if explicit_letter:
            return True
        if gold_text and len(gold_text.split()) >= 3 and gold_text in hay and has_final_answer_cue(raw):
            return True
        return False

    if benchmark == "LogiQA 2.0":
        label = "not[- ]entailment" if gold == "not-entailment" else "entailment"
        return bool(
            re.search(
                rf"\b(final\s+answer|answer\s+is|therefore|thus|so\s+the\s+answer)\s*[:\-]?\s*{label}\b",
                lowered,
            )
        )

    if benchmark == "MATH-500":
        if re.search(rf"\\boxed\s*\{{\s*{re.escape(str(gold_answer).strip())}\s*\}}", raw):
            return True
        if not has_final_answer_cue(raw):
            return False
        if len(gold) <= 2:
            return bool(re.search(rf"(?<![a-z0-9]){re.escape(gold)}(?![a-z0-9])", hay))
        return gold in hay or bool(gold_text and gold_text in hay)

    return bool(gold in hay and has_final_answer_cue(raw))


def contains_new_answer_leak(
    benchmark: str,
    original_step: str,
    edited_step: str,
    gold_answer: str,
    gold_answer_text: str | None = None,
) -> bool:
    return contains_answer_leak(benchmark, edited_step, gold_answer, gold_answer_text) and not contains_answer_leak(
        benchmark,
        original_step,
        gold_answer,
        gold_answer_text,
    )


def rule_based_contradiction(step: str, final_answer: str | None) -> bool:
    if not final_answer:
        return False
    step_n = normalize_text(step)
    ans_n = normalize_text(final_answer)
    if not ans_n:
        return False
    answer_patterns = re.findall(r"final answer\s*[:=]?\s*([a-d]|entailment|not[- ]entailment|[-+]?\d+(?:\.\d+)?)", step_n)
    if answer_patterns and normalize_text(answer_patterns[-1]) != ans_n:
        return True
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", step_n)
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", ans_n) and numbers:
        return ans_n not in numbers and any("therefore" in step_n or word in step_n for word in ["answer", "equals", "="])
    return False


def make_editor_prompt(instance: dict[str, Any], previous_step: str, current_step: str, next_step: str) -> str:
    words = len(current_step.split())
    min_words = max(8, int(words * 0.4))
    max_words = max(min_words + 1, int(words * 1.8))
    return (
        "Edit exactly one reasoning step for a controlled intervention study.\n"
        "Return exactly one JSON object and nothing else: {\"edited_step\":\"...\"}\n"
        "Do not explain your edit. Do not mention this prompt, the current step, JSON, or the intervention study.\n"
        "The edited_step must read like it naturally belongs between the previous and next steps.\n\n"
        "Hard requirements:\n"
        "- change one substantive intermediate claim, assumption, operation, or comparison,\n"
        "- preserve grammar, fluency, and local coherence,\n"
        "- do not introduce a final-answer phrase,\n"
        "- do not copy or reveal the correct final answer,\n"
        "- do not include meta-commentary such as 'I need to edit' or 'the original step',\n"
        f"- keep length roughly 0.4x to 1.8x the original step ({min_words}-{max_words} words).\n"
        "- do not summarize a long step into a short conclusion; keep the same local role and surface form where possible.\n\n"
        f"Benchmark: {instance['benchmark']}\n"
        f"Problem prompt:\n{instance['prompt'][:3000]}\n\n"
        f"Previous step:\n{previous_step}\n\n"
        f"Current step to edit:\n{current_step}\n\n"
        f"Next step:\n{next_step}\n\n"
        "JSON only:"
    )


def make_editor_retry_prompt(
    instance: dict[str, Any],
    previous_step: str,
    current_step: str,
    next_step: str,
    previous_edit: str,
    failure_reasons: list[str],
) -> str:
    reasons = ", ".join(failure_reasons) if failure_reasons else "unknown validity failure"
    base = make_editor_prompt(instance, previous_step, current_step, next_step)
    return (
        base
        + "\n\nThe previous edited_step failed validation for: "
        + reasons
        + ". Return a corrected JSON object only. "
        + "If the failure was schema_invalid, keep LaTeX simple and still return only the edited_step field. "
        + "If the failure was length_fail, match the original step's word count more closely. "
        + "If the failure was semantic_fail, change exactly one substantive claim more clearly. "
        + "If the failure was meta_commentary, remove all discussion of the editing task.\n\n"
        + f"Previous rejected edit:\n{previous_edit[:1200]}\n\n"
        + "Corrected JSON only:"
    )


def _decode_jsonish_string(value: str) -> str:
    """Decode safe JSON escapes while preserving LaTeX backslashes."""
    value = value.replace(r"\/", "/")
    value = value.replace(r"\\", "\\")
    value = value.replace(r"\"", '"')
    value = value.replace(r"\n", "\n")
    value = value.replace(r"\t", "\t")
    return value.strip()


def _extract_edited_step_jsonish(text: str) -> str | None:
    marker = re.search(r'"edited_step"\s*:\s*"', text, flags=re.S)
    if not marker:
        return None
    start = marker.end()
    tail = text[start:]

    # Prefer the final quote before the closing brace. This tolerates invalid
    # JSON caused by unescaped LaTeX sequences like \(, \[, or \frac.
    end_match = re.search(r'"\s*}\s*$', tail, flags=re.S)
    if end_match:
        return _decode_jsonish_string(tail[: end_match.start()])

    chars: list[str] = []
    escaped = False
    for ch in tail:
        if ch == '"' and not escaped:
            break
        chars.append(ch)
        escaped = ch == "\\" and not escaped
        if ch != "\\":
            escaped = False
    return _decode_jsonish_string("".join(chars)) if chars else None


def parse_edited_step(response_text: str) -> tuple[str, bool]:
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
        text = re.sub(r"```$", "", text.strip()).strip()
    try:
        data = json.loads(text)
        edited = str(data.get("edited_step", "")).strip()
        return edited, bool(edited)
    except Exception:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                data, _ = decoder.raw_decode(text[match.start() :])
                if not isinstance(data, dict):
                    continue
                edited = str(data.get("edited_step", "")).strip()
                return edited, bool(edited)
            except Exception:
                continue
        edited = _extract_edited_step_jsonish(text)
        if edited:
            return edited, True
    return text, False


def final_answer_instruction(benchmark: str) -> str:
    if benchmark == "MATH-500":
        return "End with exactly one line of the form: Final answer: \\boxed{...}"
    if benchmark == "GPQA Diamond":
        return "End with exactly one line of the form: Final answer: A, B, C, or D."
    if benchmark == "LogiQA 2.0":
        return "End with exactly one line of the form: Final answer: entailment or Final answer: not-entailment."
    return "End with exactly one explicit final-answer line."


def make_resume_prompt(instance: dict[str, Any], edited_prefix: str, force_answer: bool = False) -> str:
    answer_instruction = final_answer_instruction(instance["benchmark"])
    extra = (
        "\nDo not stop before the final-answer line. The final-answer line is required for parsing."
        if force_answer
        else ""
    )
    return (
        f"{instance['prompt']}\n\n"
        "A partial reasoning trace is provided below. Continue from it without restarting, "
        "then provide the final answer in the required format.\n"
        f"{answer_instruction}{extra}\n\n"
        f"Partial reasoning trace:\n{edited_prefix}\n\nContinue:"
    )


def synthetic_trace(instance: dict[str, Any]) -> str:
    gold = instance["gold_answer"]
    if instance["benchmark"] == "MATH-500":
        final = f"Final answer: \\boxed{{{gold}}}"
    else:
        final = f"Final answer: {gold}"
    return (
        "Step 1. I restate the problem and identify the relevant quantities.\n\n"
        "Step 2. I apply the main rule needed for this benchmark item.\n\n"
        "Step 3. I compare the candidate conclusion against the problem constraints.\n\n"
        f"Step 4. Therefore I choose the answer. {final}"
    )


def dry_edit_step(step: str) -> str:
    if "not" in step.lower():
        return re.sub(r"\bnot\b", "", step, flags=re.I).strip()
    return step + " This intermediate claim is adjusted for the intervention."


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class ValidityResult:
    passed_semantic_change_check: bool
    passed_uptake_check: bool
    over_disruption_flag: bool
    answer_leakage_flag: bool
    original_answer_leakage_flag: bool
    new_answer_leakage_flag: bool
    meta_commentary_flag: bool
    final_answer_step_flag: bool
    length_ratio: float
    length_check: bool
    accepted: bool
    semantic_similarity: float
    lexical_overlap: float


def validate_intervention(
    *,
    benchmark: str,
    original_step: str,
    edited_step: str,
    schema_valid: bool,
    delta_cont: float | None,
    gold_answer: str,
    gold_answer_text: str | None,
    thresholds: dict[str, float],
) -> ValidityResult:
    sem_sim = text_similarity(original_step, edited_step)
    lex = lexical_overlap(original_step, edited_step)
    passed_sem = sem_sim < thresholds["semantic_similarity_max"] and lex < thresholds["lexical_overlap_max"]
    passed_uptake = bool(delta_cont is not None and delta_cont >= thresholds["uptake_delta_cont_min"])
    over = bool(delta_cont is not None and delta_cont > thresholds["over_disruption_delta_cont"])
    original_leak = contains_answer_leak(benchmark, original_step, gold_answer, gold_answer_text)
    edited_leak = contains_answer_leak(benchmark, edited_step, gold_answer, gold_answer_text)
    new_leak = edited_leak and not original_leak
    meta = has_meta_commentary(edited_step)
    final_answer_step = not is_intervenable_step(original_step, benchmark, gold_answer, gold_answer_text)
    original_len = max(1, len(original_step.split()))
    edited_len = len(edited_step.split())
    length_ratio = edited_len / original_len
    length_ok = 0.4 <= length_ratio <= 1.8
    accepted = bool(
        schema_valid
        and bool(edited_step.strip())
        and passed_sem
        and length_ok
        and not new_leak
        and not meta
        and not final_answer_step
    )
    return ValidityResult(
        passed_sem,
        passed_uptake,
        over,
        edited_leak,
        original_leak,
        new_leak,
        meta,
        final_answer_step,
        round(length_ratio, 4),
        length_ok,
        accepted,
        sem_sim,
        lex,
    )

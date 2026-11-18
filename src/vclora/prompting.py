"""Prompts and output parsing shared by the prompting baselines, training and serving.

Every LLM system (zero-shot, 5-shot, fine-tuned, quantized) sees the same system prompt with the
closed label list, so the only thing that changes between them is the weights or the examples.
"""

from __future__ import annotations

import random
import re

INVALID = "__INVALID__"
MAX_NARRATIVE_WORDS = 250  # keeps system prompt + narrative + answer under 512 tokens for almost all rows
SHOT_WORDS = 80


def truncate_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " ..."


def system_prompt(labels: list[str]) -> str:
    return (
        "You classify US vehicle safety complaints by the vehicle component that failed.\n"
        "Answer with exactly one category from this list and nothing else:\n"
        + "\n".join(labels)
    )


def user_prompt(text: str, max_words: int = MAX_NARRATIVE_WORDS) -> str:
    return f"Complaint: {truncate_words(text, max_words)}\nCategory:"


def build_messages(text: str, labels: list[str], shots: list[dict] | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt(labels)}]
    for ex in shots or []:
        msgs.append({"role": "user", "content": user_prompt(ex["text"], SHOT_WORDS)})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": user_prompt(text)})
    return msgs


def pick_shots(train: list[dict], k: int, seed: int) -> list[dict]:
    """k examples with distinct labels, drawn from the training split with a fixed seed."""
    rng = random.Random(seed)
    pool = list(train)
    rng.shuffle(pool)
    shots, seen = [], set()
    for ex in pool:
        if ex["label"] not in seen:
            shots.append(ex)
            seen.add(ex["label"])
        if len(shots) == k:
            break
    return shots


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9/ ]+", " ", s.upper()).strip()


def parse_label(output: str, labels: list[str]) -> str:
    """Map raw model output to one label in the closed list, or INVALID.

    Accepts the exact label (case and punctuation ignored), or output whose first line starts with
    exactly one label, or a first line that is a prefix of exactly one label (e.g. 'SERVICE
    BRAKES' for 'SERVICE BRAKES, HYDRAULIC' is not ambiguous only if one label matches).
    Anything else, including empty output or two labels, counts as INVALID.
    """
    if not output:
        return INVALID
    first = _norm(output.strip().splitlines()[0] if output.strip() else "")
    if not first:
        return INVALID
    normed = {lab: _norm(lab) for lab in labels}
    exact = [lab for lab, n in normed.items() if n == first]
    if exact:
        return exact[0]
    starts = [lab for lab, n in normed.items() if first.startswith(n)]
    if len(starts) == 1:
        return starts[0]
    if len(starts) > 1:  # e.g. "ENGINE" and "ENGINE AND ENGINE COOLING": take the longest match
        best = max(starts, key=lambda lab: len(normed[lab]))
        if sum(len(normed[s]) == len(normed[best]) for s in starts) == 1:
            return best
    prefix = [lab for lab, n in normed.items() if n.startswith(first) and len(first) >= 4]
    if len(prefix) == 1:
        return prefix[0]
    return INVALID

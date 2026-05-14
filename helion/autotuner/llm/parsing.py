"""Extract JSON config payloads from raw LLM responses."""

from __future__ import annotations

import json
import re


def fix_python_json(text: str) -> str:
    """Normalize Python literals in LLM output to valid JSON literals."""
    text = re.sub(r"\bNone\b", "null", text)
    text = re.sub(r"\bTrue\b", "true", text)
    return re.sub(r"\bFalse\b", "false", text)


def extract_balanced_block(text: str, opener: str, closer: str) -> str | None:
    """Extract the first balanced JSON-like block, respecting quoted strings."""

    start = text.find(opener)
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find(opener, start + 1)
    return None


def iter_jsonish_candidates(text: str) -> list[str]:
    """Yield likely JSON-ish substrings from raw LLM output."""

    candidates: list[str] = []
    stripped = text.strip()
    if stripped:
        candidates.append(stripped)
    for match in re.finditer(
        r"```(?:json|python)?\s*([\s\S]*?)```", text, re.IGNORECASE
    ):
        candidate = match.group(1).strip()
        if candidate:
            candidates.append(candidate)
    for opener, closer in (("{", "}"), ("[", "]")):
        if candidate := extract_balanced_block(text, opener, closer):
            candidates.append(candidate.strip())
    return list(dict.fromkeys(candidates))


def parse_jsonish(text: str) -> object | None:
    """Parse JSON output with light extraction from wrapped LLM responses."""

    for candidate in iter_jsonish_candidates(fix_python_json(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None

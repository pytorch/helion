"""
LLM-based configuration ranker for Helion autotuning.

Sends kernel features + candidate configs to an LLM and gets back
a ranking of configurations from most-to-least promising.

Supports both the Anthropic (Claude) and OpenAI APIs.
Set the environment variable AUTOTUNER_LLM_BACKEND to "anthropic" or "openai"
(default: "anthropic").
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helion.runtime.config import Config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a GPU kernel optimization expert specializing in Triton kernels.

You will be given:
1. A description of a GPU kernel workload (operation type, tensor shape, memory access pattern, etc.).
2. A list of candidate kernel configurations, each with parameters like block_sizes, num_warps, and num_stages.

Your task: rank the configurations from best (fastest) to worst based on expected GPU performance.

Key heuristics:
- For memory-bound elementwise kernels, larger block sizes generally improve memory throughput up to a point.
- num_warps should be balanced: too few leaves SMs underutilized, too many causes register pressure.
- num_stages > 1 helps hide memory latency via software pipelining but adds register pressure.
- Total threads per block = num_warps * 32. The product of block_sizes should not vastly exceed what the threads can handle.
- For simple elementwise ops, moderate configurations (block_size 256-1024, num_warps 4-8) tend to work well.

Return ONLY a JSON array of integer indices (0-based) representing the ranking from best to worst.
Example: [2, 0, 4, 1, 3]
"""


def _configs_to_dicts(configs: list[Config]) -> list[dict[str, object]]:
    """Convert Config objects to plain dicts for JSON serialization."""
    result = []
    for cfg in configs:
        d: dict[str, object] = {}
        for key in ("block_sizes", "num_warps", "num_stages"):
            if key in cfg:
                d[key] = cfg[key]
        result.append(d)
    return result


def _build_user_prompt(
    kernel_features: dict[str, object],
    config_dicts: list[dict[str, object]],
) -> str:
    return (
        f"Kernel description:\n{json.dumps(kernel_features, indent=2)}\n\n"
        f"Candidate configurations (0-indexed):\n{json.dumps(config_dicts, indent=2)}\n\n"
        "Rank them from best to worst. Return ONLY a JSON array of indices."
    )


def _parse_ranking(text: str, n: int) -> list[int]:
    """Parse a JSON array of indices from the LLM response.

    Falls back to extracting all integers if JSON parsing fails.
    Validates that the result is a permutation of range(n).
    """
    text = text.strip()
    # Try to find a JSON array in the response
    match = re.search(r"\[[\d\s,]+\]", text)
    if match:
        try:
            ranking = json.loads(match.group())
            if isinstance(ranking, list) and all(isinstance(i, int) for i in ranking):
                if sorted(ranking) == list(range(n)):
                    return ranking
        except json.JSONDecodeError:
            pass

    # Fallback: extract all integers
    indices = [int(x) for x in re.findall(r"\d+", text)]
    # Deduplicate while preserving order
    seen: set[int] = set()
    unique: list[int] = []
    for i in indices:
        if 0 <= i < n and i not in seen:
            seen.add(i)
            unique.append(i)

    # Append any missing indices at the end
    for i in range(n):
        if i not in seen:
            unique.append(i)

    return unique


def _rank_with_anthropic(
    kernel_features: dict[str, object],
    config_dicts: list[dict[str, object]],
    model: str,
) -> list[int]:
    import anthropic

    client = anthropic.Anthropic()
    user_prompt = _build_user_prompt(kernel_features, config_dicts)

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text
    log.info("LLM response: %s", text)
    return _parse_ranking(text, len(config_dicts))


def _rank_with_openai(
    kernel_features: dict[str, object],
    config_dicts: list[dict[str, object]],
    model: str,
) -> list[int]:
    import urllib.request

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    user_prompt = _build_user_prompt(kernel_features, config_dicts)

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_completion_tokens": 8192,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"LLM API {e.code}: {body}") from e

    log.debug("Raw API response: %s", json.dumps(data, indent=2))
    text = data["choices"][0]["message"].get("content") or ""
    log.info("LLM response: %s", text)
    return _parse_ranking(text, len(config_dicts))


def llm_rank_configs(
    configs: list[Config],
    kernel_features: dict[str, object],
    *,
    backend: str | None = None,
    model: str | None = None,
) -> list[Config]:
    """Rank configs using an LLM and return them in best-to-worst order.

    Args:
        configs: Candidate configurations to rank.
        kernel_features: Structured description of the kernel workload.
        backend: "anthropic" or "openai". Defaults to env AUTOTUNER_LLM_BACKEND or "anthropic".
        model: Model name override. Defaults depend on backend.

    Returns:
        The same configs reordered from most to least promising.
    """
    if backend is None:
        backend = os.environ.get("AUTOTUNER_LLM_BACKEND", "openai")

    config_dicts = _configs_to_dicts(configs)
    log.info(
        "Sending %d configs to LLM (%s) for ranking...", len(configs), backend
    )

    if backend == "anthropic":
        model = model or os.environ.get("AUTOTUNER_LLM_MODEL", "claude-sonnet-4-20250514")
        ranking = _rank_with_anthropic(kernel_features, config_dicts, model)
    elif backend == "openai":
        model = model or os.environ.get("AUTOTUNER_LLM_MODEL", "gpt-5-mini-2025-08-07")
        ranking = _rank_with_openai(kernel_features, config_dicts, model)
    else:
        raise ValueError(f"Unknown LLM backend: {backend!r}. Use 'anthropic' or 'openai'.")

    ranked = [configs[i] for i in ranking]
    log.info("LLM ranking (best first): %s", ranking)
    return ranked

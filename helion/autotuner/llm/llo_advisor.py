"""LLM-search integration: turn LLO dumps + measurements into bottleneck hints.

Wraps `scripts/tpu_roofline.py` static analysis so the LLM-guided autotune
search gets directional feedback ("MXU-bound, try larger inner-K block")
instead of just config+timing tuples.

Per-config feedback expanded from:
    {config: ..., measured_us: 22360}
to:
    {config: ..., measured_us: 22360, predicted_us: 22128, binding_lane: "MXU",
     regime: "compute-bound", register_pressure: "high",
     suggestions: ["try larger inner-K block", "in-loop Q-scale"]}

The LLM can then reason about *why* each config performs the way it does and
propose targeted edits rather than random walks through the config space.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from helion.autotuner.base_search import BenchmarkResult


# Path to the predictor script. Resolved at import time relative to repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PREDICTOR_SCRIPT = _REPO_ROOT / "scripts" / "tpu_roofline.py"


@dataclass
class BottleneckHints:
    """Structured advisor output for one (config, llo) pair."""

    predicted_us: float | None
    regime: str  # "compute-bound (MXU)" / "memory-bound" / "overhead-bound"
    binding_lane: str  # "MXU" / "XLU" / "VALU" / ...
    mxu_busy_pct: float | None
    register_pressure: str  # "low" / "moderate" / "high"
    counterfactual_savings_us: dict[str, float]  # {"MXU=1.0": 1252.5, ...}
    suggestions: list[str]

    def format_for_prompt(self) -> str:
        """Render as a compact natural-language block for LLM consumption."""
        lines: list[str] = []
        if self.predicted_us is not None:
            lines.append(f"  predicted: {self.predicted_us:.1f} µs ({self.regime})")
        if self.binding_lane:
            mxu_note = f"binding lane {self.binding_lane}" + (
                f" @ {self.mxu_busy_pct:.0f}% busy" if self.mxu_busy_pct else ""
            )
            lines.append(f"  {mxu_note}")
        if self.register_pressure != "low":
            lines.append(f"  register pressure: {self.register_pressure}")
        if self.counterfactual_savings_us:
            top = sorted(
                self.counterfactual_savings_us.items(),
                key=lambda kv: -kv[1],
            )[:2]
            cfacts = "; ".join(f"{k} → save {v:.0f}µs" for k, v in top)
            lines.append(f"  ceilings: {cfacts}")
        if self.suggestions:
            for s in self.suggestions[:3]:
                lines.append(f"  • {s}")
        return "\n".join(lines)


def _parse_advise_output(stdout: str) -> BottleneckHints:
    """Parse `scripts/tpu_roofline.py --advise` text output into structured hints."""
    predicted_us: float | None = None
    regime = ""
    binding_lane = ""
    mxu_busy: float | None = None
    counterfactuals: dict[str, float] = {}
    suggestions: list[str] = []
    register_pressure = "low"

    in_suggestions = False
    in_counterfactuals = False
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("Predicted:") and "← max(" in s:
            with contextlib.suppress(IndexError, ValueError):
                predicted_us = float(s.split("Predicted:")[1].split("µs")[0].strip())
        elif s.startswith("Regime:"):
            regime = s.split("Regime:", 1)[1].strip()
        elif "← binding" in s:
            # e.g. "MXU     2  66,219,406    83.1%  ███... ← binding"
            parts = s.split()
            if parts:
                binding_lane = parts[0]
            for tok in parts:
                if tok.endswith("%"):
                    with contextlib.suppress(ValueError):
                        mxu_busy = float(tok.rstrip("%"))
                    break
        elif "VLOAD:FILL" in s or "VSTORE:SPILL" in s:
            parts = s.split()
            for tok in parts:
                if tok.endswith("%"):
                    try:
                        pct = float(tok.rstrip("%"))
                        if pct > 30:
                            register_pressure = "high"
                        elif pct > 10 and register_pressure == "low":
                            register_pressure = "moderate"
                    except ValueError:
                        pass
                    break
        elif s.startswith("=== Counterfactuals"):
            in_counterfactuals = True
            continue
        elif s.startswith("=== Suggestions"):
            in_counterfactuals = False
            in_suggestions = True
            continue
        elif s.startswith("==="):
            in_counterfactuals = False
            in_suggestions = False
        elif in_counterfactuals and "→" in s and "save" in s:
            try:
                lhs = s.split("→")[0].strip()
                rhs = s.split("save")[1]
                us_token = rhs.split("µs")[0].split("+")[-1].strip()
                counterfactuals[lhs] = float(us_token)
            except (IndexError, ValueError):
                pass
        elif in_suggestions and s.startswith(("- ", "• ", "⚠")):
            suggestion = s.lstrip("-•⚠ ").strip()
            if suggestion:
                suggestions.append(suggestion)

    return BottleneckHints(
        predicted_us=predicted_us,
        regime=regime,
        binding_lane=binding_lane,
        mxu_busy_pct=mxu_busy,
        register_pressure=register_pressure,
        counterfactual_savings_us=counterfactuals,
        suggestions=suggestions,
    )


def advise_from_llo(
    llo_dir: Path | str,
    *,
    inputs: str,
    outputs: str,
    measured_us: float | None = None,
    timeout_s: int = 30,
) -> BottleneckHints | None:
    """Run the predictor's `--advise` mode against an LLO entry and parse the output.

    Returns None if the predictor isn't reachable or errored.
    """
    if not _PREDICTOR_SCRIPT.is_file():
        return None
    cmd = [
        sys.executable,
        str(_PREDICTOR_SCRIPT),
        str(llo_dir),
        "--inputs",
        inputs,
        "--outputs",
        outputs,
        "--advise",
    ]
    if measured_us is not None:
        cmd += ["--measured-us", str(measured_us)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    return _parse_advise_output(result.stdout)


def summarize_bottlenecks_for_llm(
    config_hints: list[tuple[str, BottleneckHints]],
) -> str:
    """Format a per-config bottleneck table for inclusion in the LLM prompt.

    `config_hints` is an ordered list of `(config_label, hints)` pairs — typically
    one entry per "anchor" config currently being refined.
    """
    if not config_hints:
        return "  No bottleneck analysis available."
    blocks: list[str] = []
    for label, hints in config_hints:
        block = f"  {label}:\n" + hints.format_for_prompt()
        blocks.append(block)
    return "\n\n".join(blocks)


# Optional JSON serialization for cross-process use (e.g. autotune workers).
def hints_to_json(hints: BottleneckHints) -> str:
    return json.dumps(
        {
            "predicted_us": hints.predicted_us,
            "regime": hints.regime,
            "binding_lane": hints.binding_lane,
            "mxu_busy_pct": hints.mxu_busy_pct,
            "register_pressure": hints.register_pressure,
            "counterfactual_savings_us": hints.counterfactual_savings_us,
            "suggestions": hints.suggestions,
        }
    )


def hints_from_json(s: str) -> BottleneckHints:
    d = json.loads(s)
    return BottleneckHints(
        predicted_us=d.get("predicted_us"),
        regime=d.get("regime", ""),
        binding_lane=d.get("binding_lane", ""),
        mxu_busy_pct=d.get("mxu_busy_pct"),
        register_pressure=d.get("register_pressure", "low"),
        counterfactual_savings_us=d.get("counterfactual_savings_us", {}),
        suggestions=d.get("suggestions", []),
    )


# ─────────────────────────────────────────────────────────────────────────
# Integration with the autotuner
# ─────────────────────────────────────────────────────────────────────────


_TORCH_DTYPE_TO_PREDICTOR = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.float32": "fp32",
    "torch.float8_e4m3fn": "fp8",
    "torch.float8_e5m2": "fp8",
    "torch.int8": "i8",
    "torch.int32": "i32",
    "torch.int64": "i64",
    "torch.uint8": "u8",
}


def predictor_inputs_from_args(args: Iterable[object] | None) -> str:
    """Build the predictor's `--inputs` string from a kernel's argument tuple.

    Skips non-tensor args (scalars, configs). Returns "" if no tensors found.
    """
    try:
        import torch  # pyrefly: ignore[import-not-found]
    except ImportError:
        return ""
    parts: list[str] = []
    for a in args or ():
        if not isinstance(a, torch.Tensor):
            continue
        dtype_key = str(a.dtype)
        dtype_str = _TORCH_DTYPE_TO_PREDICTOR.get(dtype_key)
        if dtype_str is None:
            continue
        shape_str = "x".join(str(d) for d in a.shape)
        parts.append(f"{dtype_str}:{shape_str}")
    return ",".join(parts)


def llo_dump_dir_from_env() -> str | None:
    """Parse `LIBTPU_INIT_ARGS` env var for `--xla_jf_dump_to=<dir>`.

    Returns None if not set. This is the canonical way to enable LLO dumping
    for autotune — the user must set the env var BEFORE the autotune process
    starts (libtpu reads it at load time).
    """
    import os

    libtpu_args = os.environ.get("LIBTPU_INIT_ARGS", "")
    for token in libtpu_args.split():
        if token.startswith("--xla_jf_dump_to="):
            return token.split("=", 1)[1]
    return None


def _resolve_predictor_dir(llo_dir: Path) -> Path | None:
    """Locate a directory the predictor can ingest.

    Supports both layouts that show up in practice:
      1. Flat libtpu dump dir — `--xla_jf_dump_to=<dir>` writes all kernels'
         `*-final_bundles.txt` and matching utilization files directly into
         `<dir>`. The predictor's `find_bundle_and_util_files` already picks
         the largest `*-final_bundles.txt` (excluding schedule-analysis), so
         we return the dump dir itself.
      2. Per-compile subdirs — for harnesses that arrange dumps as
         `<llo_dir>/<config_hash>/*-final_bundles.txt`, we return the newest
         such subdir.
    """
    if not llo_dir.is_dir():
        return None

    def _has_bundles(d: Path) -> bool:
        try:
            return any(
                p.name.endswith("-final_bundles.txt")
                and "schedule-analysis" not in p.name
                for p in d.iterdir()
            )
        except OSError:
            return False

    if _has_bundles(llo_dir):
        return llo_dir
    candidates = sorted(
        (p for p in llo_dir.iterdir() if p.is_dir()),
        key=lambda p: -p.stat().st_mtime,
    )
    for c in candidates[:10]:
        if _has_bundles(c):
            return c
    return None


def refinement_strategy_from_hints(
    config_hints: list[tuple[str, BottleneckHints]],
) -> list[str]:
    """Translate advisor diagnoses into refinement instructions.

    Returned lines override `_DEFAULT_REFINEMENT_LINES` for one round so
    the LLM is pushed toward edits that target the diagnosed bottleneck
    instead of always emitting 1-field mutations of the current best.
    The trade-off: when the LLO signal is clear, exploration depth in the
    direction the signal indicates is worth more than the safety of
    near-anchor mutations.

    Empty list = no override (caller keeps the default strategy lines).
    """
    if not config_hints:
        return []
    _, best = config_hints[0]
    lines: list[str] = []
    lane = best.binding_lane or ""
    if lane == "MXU":
        lines.extend(
            [
                (
                    "Best config is compute-bound on MXU. Per-tile MXU work is "
                    "the lever — try larger block sizes on reduction loops to "
                    "amortize MXU pipeline setup over more matmul work."
                ),
                (
                    "Consult the Configuration Space to identify which "
                    "block_sizes entries are reduction (inner) vs grid (outer) "
                    "loops; do not assume a specific position."
                ),
            ]
        )
    elif lane in ("VSTORE", "VLOAD"):
        lines.extend(
            [
                (
                    f"Best config is memory-bound on {lane}. Reduce per-tile "
                    "working set: smaller block sizes on grid loops and "
                    "different loop_orders to cut HBM traffic."
                ),
            ]
        )
    elif lane == "VALU":
        lines.append(
            "Best config is VALU-bound (scalar/vector lane). Per-tile "
            "vector-ALU bookkeeping dominates; try a structurally different "
            "scheduling family (change pallas_loop_type or loop_orders) "
            "rather than tile-size tweaks."
        )
    if lines:
        lines.append(
            "Multi-field changes are encouraged this round; do not constrain "
            "to 1-field mutations of the anchors."
        )
    if best.register_pressure == "high":
        lines.append(
            "Register pressure is high. Reduce per-tile working set with "
            "smaller block sizes, or set pallas_pre_broadcast=False if "
            "present, to lower spills."
        )
    if best.regime.startswith("overhead-bound"):
        lines.append(
            "Anchor is at the small-shape min-time floor — block_sizes have "
            "little leverage. Use this round to try a structurally different "
            "family (e.g. change pallas_loop_type) rather than near-anchor "
            "mutations."
        )
    return lines


def collect_bottlenecks_for_anchors(
    results: list[BenchmarkResult],
    default_config_dict: dict[str, object],
    *,
    inputs: str | None = None,
    outputs: str | None = None,
    llo_subdir_by_config: dict[str, str] | None = None,
    config_key_fn: object | None = None,
    max_anchors: int = 2,
) -> tuple[str, list[str]]:
    """Run the LLO advisor against the top-N anchor configs.

    Returns (prompt_section, refinement_strategy_override). Both are empty
    when the advisor can't run (no LLO map, no shapes) so the search stays
    backward compatible. When the advisor does run, the strategy override
    replaces the default "1-field mutations of Anchor 1" instructions for
    one round — this is how the LLO signal escapes anchor-stickiness.

    Args:
      results: BenchmarkResult list (same as feedback.py consumers)
      default_config_dict: same shape feedback.py passes around
      inputs/outputs: shape specs passed to the predictor (e.g.
        "bf16:8x32x8192x256,bf16:8x32x8192x256,bf16:8x32x8192x256")
      llo_subdir_by_config: {config_key: llo_subdir} populated by the
        autotuner's per-config dump routing. Anchors without an entry are
        skipped to avoid mis-identifying the kernel.
      config_key_fn: callable that turns a Config into the same string key
        used for `llo_subdir_by_config`. Required when that map is provided.
      max_anchors: limit advisor invocations (each is ~50ms)
    """
    if (
        inputs is None
        or outputs is None
        or not llo_subdir_by_config
        or config_key_fn is None
    ):
        return "", []

    from .feedback import finite_results
    from .feedback import format_config_diff

    finite = finite_results(results)
    if not finite:
        return "", []

    hint_blocks: list[tuple[str, BottleneckHints]] = []
    for cfg, perf in finite[:max_anchors]:
        diff = format_config_diff(default_config_dict, cfg)
        key = config_key_fn(cfg)  # pyrefly: ignore[not-callable]
        subdir = llo_subdir_by_config.get(key)
        if subdir is None:
            continue
        predictor_dir = _resolve_predictor_dir(Path(subdir))
        if predictor_dir is None:
            continue
        hints = advise_from_llo(
            predictor_dir,
            inputs=inputs,
            outputs=outputs,
            measured_us=perf * 1000.0,  # perf is in ms; predictor expects µs
        )
        if hints is None:
            continue
        label = f"{perf:.4f} ms — {diff}"
        hint_blocks.append((label, hints))

    return summarize_bottlenecks_for_llm(hint_blocks), refinement_strategy_from_hints(
        hint_blocks
    )

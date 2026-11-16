"""The CuTe backend's prose for the LLM autotuner's prompts.

Role: heuristic (prompt prose for the LLM autotuner).  The CuTe config space has no ``num_warps`` /
``num_stages`` / ``maxnreg``: threads are ``num_threads`` per tiled dimension
(0 = automatic layout) and the pipeline depth is whatever staging controls the
kernel's search surface exposes, so every line of the prompts that names those
controls is rephrased here against the displayed Configuration Space
(``Backend.llm_prompt_profile``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....autotuner.llm.profile import STREAMING_REDUCTION_HINT
from ....autotuner.llm.profile import LlmPromptProfile

if TYPE_CHECKING:
    from collections.abc import Mapping


class CuteLlmPromptProfile(LlmPromptProfile):
    """Prompt prose for the CuTe config space."""

    def __init__(self) -> None:
        super().__init__("CuTe")

    # -- initial search strategy ---------------------------------------------

    families_line = "Use block_sizes to define families: include at least 3 materially different tiling families instead of tiny perturbations of one tile, where the displayed bounds permit it. Keep fixed fields and fixed coordinates unchanged."
    aggressive_knobs_line = "Avoid configs that simultaneously max out several aggressive knobs, unless strongly justified."
    reduction_span_line = "This is reduction-like: span the space — include conservative configs (small blocks) AND aggressive configs (large blocks up to the reduction dim), within the displayed bounds. Memory-bound reductions on modern accelerators often prefer the aggressive end."

    def tiling_family_lines(self, flat_fields: Mapping[str, object]) -> list[str]:
        # A flash search pins the tiles: no family lines without block sizes.
        if "block_sizes" not in flat_fields:
            return []
        return super().tiling_family_lines(flat_fields)

    def threading_lines(self, flat_fields: Mapping[str, object]) -> list[str]:
        if "num_threads" not in flat_fields:
            return []
        return [
            "num_threads specifies thread counts per tiled dimension: 0 selects automatic layout; positive values are powers of two within the displayed bounds. Coordinate these with tile sizes and preserve the array length."
        ]

    # -- refinement ------------------------------------------------------------

    back_off_line = "Back off aggressive settings first: smaller thread counts or pipeline depths where tunable, pointer indexing where available, fewer advanced toggles."
    attributable_edits_line = "Prefer edits with attributable effects: change exposed tiling, threading, pipeline, scheduling, indexing, or l2_groupings controls instead of rewriting every field."

    # -- system prompt -----------------------------------------------------------

    threads_knob_line = "num_threads: thread counts per tiled dimension. 0 selects automatic layout; positive values are powers of two within the displayed bounds."
    pipeline_knob_line = (
        "Pipeline depth: use the staging controls exposed in the Configuration Space."
    )
    secondary_knobs_line = "l2_groupings and other exposed controls are secondary knobs; change them selectively after choosing a coherent tiling family."
    powers_of_two_line = "block_sizes and positive num_threads values should be powers of 2 when present."
    persistent_heuristic_line = "persistent pid_type, when available, is often worth trying when total tile count is comparable to or larger than SM count, and it may also be required for some kernels."
    aggressive_depth_line = "higher pipeline depths and multi-buffering are more aggressive and should be used selectively."

    # -- workload hints ------------------------------------------------------------

    attention_aggressive_settings = "settings"
    attention_bounds_prefix = "Within the displayed block_sizes bounds, a"

    def attention_balanced_lines(self) -> list[str]:
        return []

    def attention_shape_hints_apply(self, flat_fields: Mapping[str, object]) -> bool:
        # The flash search fixes the tiles; without block sizes there is
        # nothing to start from.
        return "block_sizes" in flat_fields and not any(
            key.startswith("cute_flash_") for key in flat_fields
        )

    def attention_stage_variant_lines(self) -> list[str]:
        return []

    def reduction_hints(self, flat_fields: Mapping[str, object]) -> list[str]:
        settings = []
        if "block_sizes" in flat_fields:
            settings.append("one row per program (small leading block_size, e.g. 1)")
        if "reduction_loops" in flat_fields:
            settings.append(
                "no reduction looping (reduction_loops null) when the reduction dim fits"
            )
        if "num_stages" in flat_fields:
            settings.append("num_stages 1-2")
        conservative = "The best configs are usually conservative"
        if settings:
            conservative += ": " + ", ".join(settings)
        hints = [
            STREAMING_REDUCTION_HINT
            + conservative
            + " — not large tiles or deep pipelining."
        ]
        if "load_eviction_policies" in flat_fields:
            hints.append(
                "The main remaining speedup for such streaming reductions is "
                "cache-eviction hints, which the search often overlooks: set "
                "load_eviction_policies to 'last' (or 'first') on the streamed input "
                "loads instead of leaving them empty, so a value read once is not "
                "kept in cache. Include several configs that try 'last' on every "
                "load_eviction_policies entry."
            )
        if "num_warps" in flat_fields:
            hints.append(
                "num_warps 4 is usually the balanced choice here; reserve 8+ warps "
                "and any aggressive tiling for a small minority of exploratory "
                "configs."
            )
        return hints

    large_gemm_guidance = False

    def matmul_persistent_hint(self, flat_fields: Mapping[str, object]) -> str | None:
        if "pid_type" not in flat_fields:
            return None
        grouping = " with l2_groupings" if "l2_groupings" in flat_fields else ""
        return (
            "Problem large enough for persistent kernels - "
            f"try persistent pid_type choices{grouping} when available in the config space"
        )

    def matmul_starting_tile_hint(
        self, flat_fields: Mapping[str, object], m: int, n: int, k: int
    ) -> str | None:
        if "block_sizes" not in flat_fields:
            return None
        return (
            f"Try block_sizes near [{min(m, 128)}, {min(n, 128)}, "
            f"{min(k, 64)}] as starting point if that matches the config space"
        )

    def matmul_tips(self, flat_fields: Mapping[str, object]) -> list[str]:
        hints = []
        if "block_sizes" in flat_fields:
            hints.append(
                "High-perf matmul tips: try asymmetric tiles like [64,128,64] "
                "if that matches the config space"
            )
        if "load_eviction_policies" in flat_fields:
            hints.append(
                "High-perf matmul tips: try load_eviction_policies with 'first' or 'last'"
            )
        return hints

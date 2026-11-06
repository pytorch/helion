"""The backend-specific prose of the LLM autotuner's prompts.

``LlmPromptProfile`` holds every line of the prompts that differs between
backends.  The builders in ``prompting.py`` and ``workload.py`` ask the
backend for its profile (``Backend.llm_prompt_profile``) and splice the
profile's lines into their backend-neutral sections.  This default profile
describes the Triton-style config space (``num_warps`` / ``num_stages`` /
``maxnreg`` and the range toggles); a backend with a different space (CuTe:
``num_threads`` and the staged pipeline controls exposed per kernel) overrides
the attributes and the methods whose guidance depends on which controls the
config space exposes (``flat_fields``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._compiler.backend_registry import get_backend
from ..._compiler.backend_registry import registered_backends

if TYPE_CHECKING:
    from collections.abc import Mapping

STREAMING_REDUCTION_HINT = (
    "This kernel is a memory-bound reduction/normalization that streams "
    "each input row once. "
)


def prompt_profile(backend: str) -> LlmPromptProfile:
    """The prose of the prompts for the backend named ``backend``; a name no
    registered backend answers to gets this default profile under that name."""
    if backend not in registered_backends():
        return LlmPromptProfile(backend)
    return get_backend(backend).llm_prompt_profile()


class LlmPromptProfile:
    """Prompt prose for a backend with the Triton-style config space."""

    def __init__(self, label: str) -> None:
        # How the system prompt names the backend.
        self.label = label

    # -- initial search strategy ---------------------------------------------

    families_line = "Use block_sizes to define families: include at least 3 materially different tiling families instead of tiny perturbations of one tile."
    aggressive_knobs_line = "Avoid configs that simultaneously max out several aggressive knobs such as num_warps, num_stages, and maxnreg when present, unless strongly justified."
    reduction_span_line = "This is reduction-like: span the space — include conservative configs (small blocks, num_stages=1-2) AND aggressive configs (large blocks up to the reduction dim, num_stages=4-8 with deep range_num_stages pipelining). Memory-bound reductions on modern accelerators often prefer the aggressive end."

    def tiling_family_lines(self, flat_fields: Mapping[str, object]) -> list[str]:
        """The block-size family guidance, when the space has block sizes to vary."""
        return [
            self.families_line,
            "Vary block_sizes coherently across dimensions rather than by arbitrary skew.",
        ]

    def threading_lines(self, flat_fields: Mapping[str, object]) -> list[str]:
        """Guidance on the thread / pipeline controls of the space."""
        return [
            (
                "Explicitly diversify num_stages and num_warps across the batch: don't cluster all suggestions at num_stages=1; include several with num_stages in {2, 4, 8} so LFBO refinement has aggressive seeds to anchor on."
            )
        ]

    # -- refinement ------------------------------------------------------------

    back_off_line = "Back off aggressive settings first: smaller num_stages/num_warps, pointer indexing, fewer advanced toggles."
    attributable_edits_line = "Prefer edits with attributable effects: change block_sizes, num_warps, num_stages, pid_type, indexing, l2_groupings, or maxnreg instead of rewriting every field."

    # -- system prompt -----------------------------------------------------------

    threads_knob_line = (
        "num_warps: threads/32. 4-8 is typical; 16+ is mainly for clearly larger tiles."
    )
    pipeline_knob_line = "num_stages: pipeline depth. 2-4 is common for streaming loops; 1 is safer when unsure."
    secondary_knobs_line = "l2_groupings, maxnreg, num_sm_multiplier, and advanced range toggles are secondary knobs; change them selectively after choosing a coherent tiling family."
    powers_of_two_line = "block_sizes and num_warps should be powers of 2 when present."
    persistent_heuristic_line = "persistent pid_type is often worth trying when total tile count is comparable to or larger than SM count, and it may also be required for some kernels."
    aggressive_depth_line = "higher num_stages and multi-buffering are more aggressive and should be used selectively."

    # -- workload hints ------------------------------------------------------------

    # What the attention hint calls the settings to keep for a minority of configs.
    attention_aggressive_settings = "num_warps/num_stages or advanced toggles"
    attention_bounds_prefix = "A"

    def attention_balanced_lines(self) -> list[str]:
        return [
            (
                "Keep most balanced configs on moderate warps and stages; "
                "for attention-style streaming kernels, 4 warps is often the "
                "balanced choice, while 8+ warps or 4+ stages are "
                "exploratory unless tiles are clearly large and compile "
                "cleanly."
            )
        ]

    def attention_shape_hints_apply(self, flat_fields: Mapping[str, object]) -> bool:
        """Whether the shape-derived starting-tile hints suit the space."""
        return True

    def attention_stage_variant_lines(self) -> list[str]:
        return [
            (
                "Within that starting family, include a couple of "
                "balanced variants that keep block_sizes fixed and "
                "vary num_stages through 2-3 before moving to much "
                "larger tiles or higher warps."
            )
        ]

    def reduction_hints(self, flat_fields: Mapping[str, object]) -> list[str]:
        """Guidance for a memory-bound streaming reduction."""
        return [
            (
                STREAMING_REDUCTION_HINT
                + "The best configs are usually conservative: one "
                "row per program (small leading block_size, e.g. 1), no reduction "
                "looping (reduction_loops null) when the reduction dim fits, and "
                "num_stages 1-2 — not large tiles or deep pipelining."
            ),
            (
                "The main remaining speedup for such streaming reductions is "
                "cache-eviction hints, which the search often overlooks: set "
                "load_eviction_policies to 'last' (or 'first') on the streamed input "
                "loads instead of leaving them empty, so a value read once is not "
                "kept in cache. Include several configs that try 'last' on every "
                "load_eviction_policies entry; leaving them all empty typically "
                "leaves ~15-20% on the table for these kernels."
            ),
            (
                "num_warps 4 is usually the balanced choice here; reserve 8+ warps "
                "and any aggressive tiling for a small minority of exploratory "
                "configs."
            ),
        ]

    # Whether a large balanced standalone GEMM takes the flat / K=32 guidance
    # (``_large_matmul_hints``) instead of the generic matmul tail.
    large_gemm_guidance = True

    def matmul_persistent_hint(self, flat_fields: Mapping[str, object]) -> str | None:
        """The persistent-scheduling hint for a problem with many tiles."""
        return (
            "Problem large enough for persistent kernels - "
            "try pid_type='persistent_blocked' with l2_groupings=8-64"
        )

    def matmul_starting_tile_hint(
        self, flat_fields: Mapping[str, object], m: int, n: int, k: int
    ) -> str | None:
        return (
            f"Try block_sizes near [{min(m, 128)}, {min(n, 128)}, "
            f"{min(k, 64)}] as starting point"
        )

    def matmul_tips(self, flat_fields: Mapping[str, object]) -> list[str]:
        return [
            (
                "High-perf matmul tips: try asymmetric tiles like [64,128,64], "
                "num_stages=3-4, maxnreg=128 or 256, "
                "range_multi_buffers=[true,true] for double-buffering, "
                "load_eviction_policies with 'first' or 'last'"
            )
        ]

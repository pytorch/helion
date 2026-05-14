"""Build prompts for the LLM-guided autotuner."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from .configs import describe_config_space
from .feedback import MAX_CHANGED_FIELDS_PER_CONFIG
from .feedback import format_config_for_prompt
from .workload import compute_workload_hints
from .workload import describe_kernel
from .workload import detect_workload_traits

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..base_search import _AutotunableKernel
    from ..config_spec import ConfigSpec

RETURN_JSON_ONLY = 'Return minified JSON only: {"configs":[...]}'
SHAPE_RULE = (
    "Do not guess field structure: for list-valued fields, emit an explicit JSON "
    "array of the exact required length; if that length is unclear, omit the "
    "field."
)
_ADVANCED_TOGGLE_FIELDS = (
    "range_warp_specializes",
    "range_multi_buffers",
    "range_flattens",
)
_INITIAL_STRATEGY_BASE_LINES = (
    "First analyze the kernel source, input tensors, GPU hardware, and config space.",
    "Cover 3 config families with a rough mix of about 40% near-default safe, 40% balanced throughput, and 20% aggressive configs, while keeping most candidates valid and compilable.",
    "If the kernel structure is unclear, stay closer to default and avoid aggressive coupled changes.",
    (
        "Keep each config sparse: usually 2-6 changed fields, omit unchanged defaults, "
        "and exceed 6 only when several coupled changes are needed for a distinct family."
    ),
    "Use block_sizes to define families: include at least 3 materially different tiling families instead of tiny perturbations of one tile.",
    "Vary block_sizes coherently across dimensions rather than by arbitrary skew.",
    SHAPE_RULE,
    "Do not pretty-print or repeat unchanged defaults.",
    "Avoid configs that simultaneously max out several aggressive knobs such as num_warps, num_stages, and maxnreg when present, unless strongly justified.",
)
_FAILURE_HEAVY_REFINEMENT_LINES = (
    "Recent rounds had many failures. Use only the best 1-2 anchors.",
    "At least 80% of configs should be 1-2 field mutations of those anchors.",
    "Back off aggressive settings first: smaller num_stages/num_warps, pointer indexing, fewer advanced toggles.",
)
_DEFAULT_REFINEMENT_LINES = (
    "About two thirds of configs should be 1-field mutations of Anchor 1.",
    "Use most of the rest for 1-2 field mutations of Anchor 2.",
    "Reserve at most a small minority for one clearly different family, not random noise.",
)
_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert GPU kernel autotuner for Helion/Triton kernels.

    Use the provided Configuration Space and Default Configuration as the source of truth for:
    - allowed field names and enum values
    - which fields are scalar vs list-valued
    - required list lengths
    - valid ranges and defaults

    Key knobs:
    - block_sizes: per-dimension tile sizes. Good families usually change them coherently. For kernels with an inner or reduction dimension, very small values there are a separate aggressive family, not the default.
    - num_warps: threads/32. 4-8 is typical; 16+ is mainly for clearly larger tiles.
    - num_stages: pipeline depth. 2-4 is common for streaming loops; 1 is safer when unsure.
    - pid_type: flat, persistent_blocked, and persistent_interleaved are distinct scheduling families when available.
    - indexing: pointer and tensor_descriptor are distinct families.
    - l2_groupings, maxnreg, num_sm_multiplier, and advanced range toggles are secondary knobs; change them selectively after choosing a coherent tiling family.

    General heuristics:
    - analyze the kernel source, input tensors, GPU hardware, and config space to infer likely optimization traits from the code itself and target hardware; if unsure, stay closer to default.
    - block_sizes and num_warps should be powers of 2 when present.
    - persistent pid_type is often worth trying when total tile count is comparable to or larger than SM count, and it may also be required for some kernels.
    - tensor_descriptor is a distinct family from pointer indexing.
    - higher num_stages and multi-buffering are more aggressive and should be used selectively.

    Output contract:
    - Return minified JSON on a single line. No markdown, code fences, comments, pretty-printing, or trailing commas.
    - Emit exactly one top-level object: {"configs":[...]} and make every config unique.
    - Do not use Python syntax or expressions such as single-quoted strings or list multiplication like ["pointer"] * 4.
    - Only specify fields you want to change; unspecified = default.
    - Use only field names and enum values that appear in the config space.
    - For list-valued fields, emit an explicit JSON array with the exact required length shown in the config space.
    - Never use a scalar as shorthand for a list-valued field, and never wrap scalar-valued fields in single-element lists.
    - If you are unsure about a field's structure, required list length, or allowed values, omit that field instead of guessing.
    - Use null not None, true/false not True/False.
    - Return ONLY minified JSON: {"configs":[...]}""")


def _section(title: str, body: str) -> str:
    """Render a titled prompt section."""
    return f"## {title}\n{body}"


def _bullet_section(title: str, lines: Sequence[str]) -> str:
    """Render a titled prompt section whose body is a bullet list."""
    return _section(title, "\n".join(f"  - {line}" for line in lines))


def _join_sections(*sections: str) -> str:
    """Join non-empty prompt sections with a blank line."""
    return "\n\n".join(section for section in sections if section)


def _initial_strategy_lines(
    *,
    configs_per_round: int,
    compile_timeout_s: int | None,
    flat_fields: Mapping[str, object],
) -> list[str]:
    """Build the bullet list used for the initial search-strategy section."""
    lines = [
        (
            f"Generate up to {configs_per_round} UNIQUE candidate configs. "
            "Fewer is better than invalid JSON."
        ),
        *_INITIAL_STRATEGY_BASE_LINES,
    ]
    if compile_timeout_s is not None:
        lines.append(
            f"Compile timeout is {compile_timeout_s}s, so avoid candidates that are likely to compile very slowly."
        )
    if "indexing" in flat_fields:
        lines.append(
            "If tensor_descriptor is available, treat it as a separate family: include a few configs using it, but keep some pure pointer configs too."
        )
    if "pid_type" in flat_fields:
        lines.append(
            "Include both flat and persistent scheduling families when plausible; do not put every config on the same pid_type."
        )
    if "reduction_loops" in flat_fields:
        lines.append(
            "This is reduction-like: keep most configs conservative and avoid very large block_sizes or maxed num_warps/num_stages."
        )
    if any(name in flat_fields for name in _ADVANCED_TOGGLE_FIELDS):
        lines.append(
            "Use advanced toggles like warp_specialize, multi_buffer, and flatten in only a minority of otherwise sane configs."
        )
    return lines


def _refinement_strategy_lines(
    *,
    compile_timeout_s: int | None,
    failed_count: int,
    total_count: int,
) -> list[str]:
    """Build the bullet list used for the refinement-step section."""
    if total_count > 0 and failed_count * 3 >= total_count:
        lines = list(_FAILURE_HEAVY_REFINEMENT_LINES)
    else:
        lines = list(_DEFAULT_REFINEMENT_LINES)
    lines.append(
        "Prefer edits with attributable effects: change block_sizes, num_warps, num_stages, pid_type, indexing, l2_groupings, or maxnreg instead of rewriting every field."
    )
    lines.append(
        "Keep each config sparse: usually 1-4 changed fields, and no more than "
        f"{MAX_CHANGED_FIELDS_PER_CONFIG} unless absolutely necessary."
    )
    lines.append(SHAPE_RULE)
    if compile_timeout_s is not None:
        lines.append(
            f"Keep compile cost in mind: avoid candidates that are likely to exceed the {compile_timeout_s}s compile timeout."
        )
    lines.append(
        "If unsure, return fewer valid configs instead of verbose or malformed JSON."
    )
    return lines


def build_system_prompt() -> str:
    """Return the global instruction block shared by every LLM request."""
    return _SYSTEM_PROMPT


def build_initial_search_guidance(
    *,
    configs_per_round: int,
    compile_timeout_s: int | None,
    flat_fields: Mapping[str, object],
) -> str:
    """Build the search-strategy section of the initial prompt."""
    return _bullet_section(
        "Search Strategy",
        _initial_strategy_lines(
            configs_per_round=configs_per_round,
            compile_timeout_s=compile_timeout_s,
            flat_fields=flat_fields,
        ),
    )


def build_initial_prompt(
    *,
    kernel: _AutotunableKernel,
    args: Sequence[object],
    config_spec: ConfigSpec,
    configs_per_round: int,
    compile_timeout_s: int | None,
) -> str:
    """Build the full initial user prompt sent to the LLM."""
    default_config = config_spec.default_config()
    workload_hints = compute_workload_hints(
        args,
        workload_traits=detect_workload_traits(kernel, config_spec=config_spec),
    )
    guidance = build_initial_search_guidance(
        configs_per_round=configs_per_round,
        compile_timeout_s=compile_timeout_s,
        flat_fields=config_spec._flat_fields(),
    )
    default_section = (
        _section("Default Configuration", format_config_for_prompt(default_config))
        + workload_hints
    )
    task_section = (
        "Suggest the first batch of configs. Include both near-default and exploratory candidates. "
        f"{RETURN_JSON_ONLY}"
    )
    return _join_sections(
        describe_kernel(kernel, args),
        _section("Configuration Space", describe_config_space(config_spec)),
        default_section,
        guidance,
        _section("Task", task_section),
    )


def build_refinement_prompt(
    *,
    configs_per_round: int,
    compile_timeout_s: int | None,
    failed_count: int,
    total_count: int,
    search_state: str,
    anchor_configs: str,
    results: str,
    top_patterns: str,
    failed_patterns: str,
) -> str:
    """Build the refinement prompt sent after each benchmarking round."""
    task_section = (
        f"Suggest up to {configs_per_round} NEW UNIQUE configs around the anchors above. "
        "Avoid the failed patterns above and favor targeted edits with attributable effects. "
        f"{RETURN_JSON_ONLY}"
    )
    return _join_sections(
        _section("Search State", search_state),
        _section("Anchor Configs", anchor_configs),
        _section("Results (best first)", results),
        _section("Top Config Patterns", top_patterns),
        _section("Failed Config Patterns", failed_patterns),
        _bullet_section(
            "Next Step",
            _refinement_strategy_lines(
                compile_timeout_s=compile_timeout_s,
                failed_count=failed_count,
                total_count=total_count,
            ),
        ),
        _section("Task", task_section),
    )

"""Infer workload traits and render kernel context for LLM prompts."""

from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING

import torch

from ..._compat import get_device_name
from ..._compat import num_compute_units
from ..._compiler.autotuner_heuristics.common import REDUCTION_TARGET_NAMES
from ..._compiler.autotuner_heuristics.common import op_name_parts

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from ..base_search import _AutotunableKernel
    from ..config_spec import ConfigSpec


# M*N*K threshold above which a standalone 2D GEMM is "large" enough to want the
# flat / small-K-tile hint below. ~16G sits between a 2048^3 GEMM (8.6G, left
# untouched) and a 4096^3 GEMM (69G, measurably helped — see the ablation). A
# necessary but not sufficient condition: see _is_large_balanced_gemm.
_LARGE_MATMUL_MNK = 16_000_000_000

MATMUL_TARGETS = frozenset(
    {
        torch.matmul,
        torch.ops.aten.mm.default,
        torch.ops.aten.addmm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
    }
)
MATMUL_API_NAMES = frozenset({"dot", "dot_scaled"})
BATCH_MATMUL_TARGET_NAMES = frozenset({"bmm", "baddbmm"})
EXP_TARGET_NAMES = frozenset({"exp", "exp2"})


def _kernel_source_text(kernel: _AutotunableKernel) -> str:
    """Extract the underlying kernel source when it is available."""
    try:
        inner_kernel = getattr(kernel, "kernel", None)
        if inner_kernel is None or not hasattr(inner_kernel, "fn"):
            return "# Source unavailable"
        raw_source = inspect.getsource(inner_kernel.fn)
    except (OSError, TypeError):
        return "# Source unavailable"

    source_lines = textwrap.dedent(raw_source).splitlines()
    start_idx = 0
    while start_idx < len(source_lines) and not source_lines[
        start_idx
    ].lstrip().startswith("def "):
        start_idx += 1
    return "\n".join(source_lines[start_idx:])


def _tensor_args(args: Sequence[object]) -> list[torch.Tensor]:
    """Return only the tensor-valued runtime arguments."""
    return [arg for arg in args if isinstance(arg, torch.Tensor)]


def _input_tensor_lines(args: Sequence[object]) -> list[str]:
    """Render tensor argument shapes and dtypes for the prompt."""
    lines: list[str] = []
    for index, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            lines.append(f"  arg[{index}]: shape={list(arg.shape)}, dtype={arg.dtype}")
    return lines


def _gpu_hardware_lines(device: torch.device) -> list[str]:
    """Render a short hardware summary for the prompt."""
    device_name = get_device_name(device) or str(device)
    lines = [
        f"  Device: {device_name}",
        f"  Compute units (SMs): {num_compute_units()}",
    ]
    if device.type != "cuda" or not torch.cuda.is_available():
        return lines
    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:
        return lines
    lines.extend(
        [
            f"  Total memory: {props.total_memory / (1024**3):.1f} GB",
            f"  Max threads per SM: {props.max_threads_per_multi_processor}",
        ]
    )
    return lines


def describe_kernel(kernel: _AutotunableKernel, args: Sequence[object]) -> str:
    """Build a description of the kernel, its inputs, and the target GPU."""
    parts = [f"## Kernel Source Code\n```python\n{_kernel_source_text(kernel)}\n```"]

    if tensor_lines := _input_tensor_lines(args):
        parts.append("## Input Tensors\n" + "\n".join(tensor_lines))

    parts.append(
        "## GPU Hardware\n" + "\n".join(_gpu_hardware_lines(kernel.env.device))
    )
    return "\n\n".join(parts)


def _iter_call_targets(kernel: _AutotunableKernel) -> Iterator[object]:
    """Yield traced call targets from compiler-generated FX graphs."""
    host_function = getattr(kernel, "host_function", None)
    device_ir = getattr(host_function, "device_ir", None)
    for graph_info in getattr(device_ir, "graphs", ()):
        graph = getattr(graph_info, "graph", None)
        if not isinstance(graph, torch.fx.Graph):
            continue
        for node in graph.nodes:
            if node.op == "call_function":
                yield node.target


def detect_workload_traits(
    kernel: _AutotunableKernel | None,
    *,
    config_spec: ConfigSpec | None = None,
) -> frozenset[str]:
    """Infer coarse workload traits from compiler-traced graphs."""
    if kernel is None:
        return frozenset()

    saw_matmul = False
    saw_batched_matmul = False
    saw_reduction = bool(config_spec is not None and config_spec.reduction_loops)
    saw_exp = False

    for target in _iter_call_targets(kernel):
        name_parts = op_name_parts(target)
        if target in MATMUL_TARGETS or name_parts & MATMUL_API_NAMES:
            saw_matmul = True
        if name_parts & BATCH_MATMUL_TARGET_NAMES:
            saw_batched_matmul = True
        if name_parts & REDUCTION_TARGET_NAMES:
            saw_reduction = True
        if name_parts & EXP_TARGET_NAMES:
            saw_exp = True

    traits: set[str] = set()
    if saw_matmul:
        traits.add("matmul")
    if saw_reduction:
        traits.add("reduction")
    if saw_matmul and saw_reduction and (saw_batched_matmul or saw_exp):
        traits.add("attention_reduction")
    return frozenset(traits)


def _summary_hints(
    tensors: Sequence[torch.Tensor],
    *,
    workload_traits: frozenset[str],
) -> list[str]:
    """Summarize input size and compiler-detected workload traits."""
    hints: list[str] = []
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if total_bytes > 0:
        hints.append(f"Total input data: {total_bytes / (1024**2):.1f} MB")
    if workload_traits:
        hints.append("Compiler-detected traits: " + ", ".join(sorted(workload_traits)))
    return hints


def _attention_reduction_hints(tensors: Sequence[torch.Tensor]) -> list[str]:
    """Suggest conservative but diverse starting families for attention-like kernels."""
    hints = [
        (
            "Compiler detected matmul-family ops with reductions/"
            "normalization; keep at least one attention/reduction-style "
            "family with moderate streaming tiles, and reserve very "
            "aggressive num_warps/num_stages or advanced toggles for a "
            "minority of configs."
        ),
        (
            "For this workload, distinct families can still stay near "
            "moderate tiles: use scheduling or indexing changes to create "
            "diversity too, instead of forcing every family to jump to a "
            "much larger tile."
        ),
        (
            "Keep most balanced configs on moderate warps and stages; "
            "for attention-style streaming kernels, 4 warps is often the "
            "balanced choice, while 8+ warps or 4+ stages are "
            "exploratory unless tiles are clearly large and compile "
            "cleanly."
        ),
        (
            "Include at least one persistent scheduling family when "
            "available for long streaming dimensions."
        ),
    ]
    shapes = [list(tensor.shape) for tensor in tensors]
    if len(tensors) < 3 or not all(len(shape) >= 2 for shape in shapes[:3]):
        return hints

    seq = shapes[0][-2]
    head_dim = shapes[0][-1]
    if not isinstance(seq, int) or not isinstance(head_dim, int):
        return hints

    mid_tile = min(seq, 64)
    inner_tile = min(head_dim, 64)
    hints.extend(
        [
            (
                "Attention/reduction-style starting point: try a "
                f"family near [1, {mid_tile}, {inner_tile}] if that matches the "
                "config space."
            ),
            (
                "Within that starting family, include a couple of "
                "balanced variants that keep block_sizes fixed and "
                "vary num_stages through 2-3 before moving to much "
                "larger tiles or higher warps."
            ),
        ]
    )
    if inner_tile >= 32:
        hints.append(
            "A nearby family like "
            f"[1, {mid_tile}, {max(16, inner_tile // 2)}] is already "
            "distinct for this shape; prefer that kind of inner-tile "
            "change before doubling the streaming tile, and keep tiles "
            f"above {mid_tile} on the streaming axis to at most a "
            "small minority."
        )
    return hints


def _reduction_hints() -> list[str]:
    """Cache-eviction guidance for pure (non-attention) reductions.

    Memory-bound row reductions recover their remaining headroom from
    ``load_eviction_policies`` (which the LLM tends to leave at default), not
    larger tiles. Family-general: no specific kernel or shape.
    """
    return [
        (
            "This kernel is a memory-bound reduction/normalization that streams "
            "each input row once. The best configs are usually conservative: one "
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


def _matmul_hints(
    tensors: Sequence[torch.Tensor],
    *,
    workload_traits: frozenset[str],
) -> list[str]:
    """Suggest matmul-oriented starting tiles when the traced graph looks matmul-like."""
    if "matmul" not in workload_traits:
        return []

    shapes = [list(tensor.shape) for tensor in tensors]
    ndims = [len(shape) for shape in shapes]
    is_2d_compatible = len(tensors) >= 2 and all(dim == 2 for dim in ndims[:2])
    if not is_2d_compatible:
        if "attention_reduction" in workload_traits:
            return []
        return [
            (
                "Compiler detected matmul-family ops; keep at least one coherent "
                "matmul-style tiling family in the search."
            )
        ]

    m, k = shapes[0]
    k2, n = shapes[1]
    total_tiles_64 = (m // 64) * (n // 64)
    hints = [f"Matmul-like: [{m}x{k}] @ [{k2}x{n}], ~{total_tiles_64} tiles at 64x64"]

    # Large *balanced* standalone 2D GEMM: emit only the flat / K=32 guidance and
    # early-return, dropping the default tail below (which pushes persistent
    # scheduling and a K<=64 tile that are counterproductive at this size). Gated
    # to a pure matmul so a reduction/attention-fused kernel keeps its own hints.
    is_pure_matmul = not (workload_traits & {"reduction", "attention_reduction"})
    if is_pure_matmul and _is_large_balanced_gemm(m, n, k, total_tiles_64):
        hints.extend(_large_matmul_hints(m, n))
        return hints

    if total_tiles_64 > num_compute_units() * 4:
        hints.append(
            "Problem large enough for persistent kernels - "
            "try pid_type='persistent_blocked' with l2_groupings=8-64"
        )
    hints.extend(
        [
            (
                f"Try block_sizes near [{min(m, 128)}, {min(n, 128)}, "
                f"{min(k, 64)}] as starting point"
            ),
            (
                "High-perf matmul tips: try asymmetric tiles like [64,128,64], "
                "num_stages=3-4, maxnreg=128 or 256, "
                "range_multi_buffers=[true,true] for double-buffering, "
                "load_eviction_policies with 'first' or 'last'"
            ),
        ]
    )
    return hints


def _is_large_balanced_gemm(m: int, n: int, k: int, total_tiles_64: int) -> bool:
    """Whether a rank-2 GEMM is large *and* has a balanced large-tile output grid.

    Requires both a large M*N*K and an M*N grid with at least one 64x64 tile per
    compute unit. The grid guard rejects skinny shapes (e.g. [1, 5e5] @ [5e5,
    32768]) that clear the FLOP threshold but have a degenerate output grid, where
    "keep M/N tiles large" advice would be wrong.
    """
    return m * n * k >= _LARGE_MATMUL_MNK and total_tiles_64 >= num_compute_units()


def _large_matmul_hints(m: int, n: int) -> list[str]:
    """Pipelining/scheduling guidance for a large *standalone* 2D GEMM.

    At this size LFBO prefers a small K-tile (32) and plain `flat` scheduling,
    while the default guidance pushes a K-tile of 64 and persistent scheduling;
    the caller suppresses that default for this branch. Keyed purely on operand
    size (no kernel name or specific shape).
    """
    return [
        (
            f"This is a large compute-bound GEMM ([{m}x*] @ [*x{n}]). For a plain "
            "standalone GEMM at this size, prefer pid_type='flat' as the primary "
            "scheduling family — flat is frequently faster here than persistent "
            "scheduling even though the tile count exceeds the SM count, so make "
            "most configs flat and reserve persistent for a minority probe."
        ),
        (
            "The reduction (K) tile is the key pipelining knob at this size: make "
            "the K-tile 32 (not 64) your primary choice — e.g. block_sizes "
            "[128, 256, 32] / [256, 256, 32] — which shrinks the per-stage working "
            "set so the K-loop overlaps better at pipeline depth num_stages 4-5. "
            "Keep M/N tiles large (128-256)."
        ),
    ]


def _format_workload_analysis(hints: list[str]) -> str:
    """Render workload hints as an optional prompt section."""
    if not hints:
        return ""
    return "\n\n## Workload Analysis\n" + "\n".join(f"  {hint}" for hint in hints)


def compute_workload_hints(
    args: Sequence[object], *, workload_traits: frozenset[str] = frozenset()
) -> str:
    """Analyze the kernel workload and produce optimization hints."""
    tensors = _tensor_args(args)
    hints = _summary_hints(tensors, workload_traits=workload_traits)
    if "attention_reduction" in workload_traits:
        hints.extend(_attention_reduction_hints(tensors))
    elif "reduction" in workload_traits and "matmul" not in workload_traits:
        hints.extend(_reduction_hints())
    hints.extend(_matmul_hints(tensors, workload_traits=workload_traits))
    return _format_workload_analysis(hints)

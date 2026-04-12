"""Infer workload traits and render kernel context for LLM prompts."""

from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING

import torch

from ..._compat import get_device_name
from ..._compat import num_compute_units

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..base_search import _AutotunableKernel
    from ..config_spec import ConfigSpec


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
REDUCTION_TARGET_NAMES = frozenset({"amax", "sum", "softmax", "logsumexp"})
EXP_TARGET_NAMES = frozenset({"exp", "exp2"})


def describe_kernel(kernel: _AutotunableKernel, args: Sequence[object]) -> str:
    """Build a description of the kernel, its inputs, and the target GPU."""
    parts: list[str] = []

    try:
        inner_kernel = getattr(kernel, "kernel", None)
        if inner_kernel is not None and hasattr(inner_kernel, "fn"):
            raw_source = inspect.getsource(inner_kernel.fn)
            source_lines = textwrap.dedent(raw_source).splitlines()
            start_idx = 0
            while start_idx < len(source_lines) and not source_lines[
                start_idx
            ].lstrip().startswith("def "):
                start_idx += 1
            kernel_source = "\n".join(source_lines[start_idx:])
        else:
            kernel_source = "# Source unavailable"
    except (OSError, TypeError):
        kernel_source = "# Source unavailable"

    parts.append(f"## Kernel Source Code\n```python\n{kernel_source}\n```")

    tensor_info: list[str] = []
    for index, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            tensor_info.append(
                f"  arg[{index}]: shape={list(arg.shape)}, dtype={arg.dtype}"
            )
    if tensor_info:
        parts.append("## Input Tensors\n" + "\n".join(tensor_info))

    device = kernel.env.device
    device_name = get_device_name(device) or str(device)
    hw_lines = [
        f"  Device: {device_name}",
        f"  Compute units (SMs): {num_compute_units()}",
    ]
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(device)
            hw_lines.extend(
                [
                    f"  Total memory: {props.total_memory / (1024**3):.1f} GB",
                    f"  Max threads per SM: {props.max_threads_per_multi_processor}",
                ]
            )
        except Exception:
            pass
    parts.append("## GPU Hardware\n" + "\n".join(hw_lines))

    return "\n\n".join(parts)


def _target_name_parts(target: object) -> frozenset[str]:
    """Extract coarse name tokens for a traced call target."""
    parts: set[str] = set()
    for raw in (
        getattr(target, "__name__", None),
        getattr(target, "name", None),
        str(target),
    ):
        if not isinstance(raw, str):
            continue
        parts.add(raw)
        parts.update(piece for piece in raw.split(".") if piece)
    return frozenset(parts)


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

    host_function = getattr(kernel, "host_function", None)
    device_ir = getattr(host_function, "device_ir", None)
    for graph_info in getattr(device_ir, "graphs", ()):
        graph = getattr(graph_info, "graph", None)
        if not isinstance(graph, torch.fx.Graph):
            continue
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            name_parts = _target_name_parts(node.target)
            if node.target in MATMUL_TARGETS or name_parts & MATMUL_API_NAMES:
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


def compute_workload_hints(
    args: Sequence[object], *, workload_traits: frozenset[str] = frozenset()
) -> str:
    """Analyze the kernel workload and produce optimization hints."""
    hints: list[str] = []
    tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    if total_bytes > 0:
        hints.append(f"Total input data: {total_bytes / (1024**2):.1f} MB")
    if workload_traits:
        hints.append("Compiler-detected traits: " + ", ".join(sorted(workload_traits)))

    shapes = [list(tensor.shape) for tensor in tensors]
    ndims = [len(shape) for shape in shapes]
    is_2d_compatible = len(tensors) >= 2 and all(dim == 2 for dim in ndims[:2])

    if "attention_reduction" in workload_traits:
        hints.extend(
            [
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
        )
        if len(tensors) >= 3 and all(len(shape) >= 2 for shape in shapes[:3]):
            seq = shapes[0][-2]
            head_dim = shapes[0][-1]
            if isinstance(seq, int) and isinstance(head_dim, int):
                mid_tile = min(seq, 64)
                inner_tile = min(head_dim, 64)
                hints.extend(
                    [
                        (
                            "Attention/reduction-style starting point: try a "
                            "family near "
                            f"[1, {mid_tile}, {inner_tile}] if that matches the "
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
    elif "matmul" in workload_traits and not is_2d_compatible:
        hints.append(
            "Compiler detected matmul-family ops; keep at least one coherent "
            "matmul-style tiling family in the search."
        )

    if "matmul" in workload_traits and is_2d_compatible:
        m, k = shapes[0]
        k2, n = shapes[1]
        total_tiles_64 = (m // 64) * (n // 64)
        hints.append(
            f"Matmul-like: [{m}x{k}] @ [{k2}x{n}], ~{total_tiles_64} tiles at 64x64"
        )
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
    if not hints:
        return ""
    return "\n\n## Workload Analysis\n" + "\n".join(f"  {hint}" for hint in hints)

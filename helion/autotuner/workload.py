# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Infer coarse workload traits for autotune policy selection."""

from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from .base_search import _AutotunableKernel
    from .config_spec import ConfigSpec


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
SOFTMAX_TARGET_NAMES = frozenset({"softmax", "_softmax"})
LOGSUMEXP_TARGET_NAMES = frozenset({"logsumexp"})
SUM_TARGET_NAMES = frozenset({"sum"})


def kernel_source_text(kernel: _AutotunableKernel) -> str:
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
    saw_softmax = False
    saw_logsumexp = False
    saw_sum = False

    for target in _iter_call_targets(kernel):
        name_parts = _target_name_parts(target)
        if target in MATMUL_TARGETS or name_parts & MATMUL_API_NAMES:
            saw_matmul = True
        if name_parts & BATCH_MATMUL_TARGET_NAMES:
            saw_batched_matmul = True
        if name_parts & REDUCTION_TARGET_NAMES:
            saw_reduction = True
        if name_parts & EXP_TARGET_NAMES:
            saw_exp = True
        if name_parts & SOFTMAX_TARGET_NAMES:
            saw_softmax = True
        if name_parts & LOGSUMEXP_TARGET_NAMES:
            saw_logsumexp = True
        if name_parts & SUM_TARGET_NAMES:
            saw_sum = True

    traits: set[str] = set()
    if saw_matmul:
        traits.add("matmul")
    if saw_reduction:
        traits.add("reduction")
    if saw_exp:
        traits.add("exp")
    if saw_softmax:
        traits.add("softmax")
    if saw_logsumexp:
        traits.add("cross_entropy")
    if saw_sum:
        traits.add("sum_reduction")
    if saw_matmul and saw_reduction and (saw_batched_matmul or saw_exp):
        traits.add("attention_reduction")
    return frozenset(traits)

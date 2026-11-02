"""Codegen side of the tcgen05 prologue trims (see
``_tcgen05_prologue_runtime`` for the device helpers and the measurements).

The TMA-load warp of a one-tile-per-CTA tcgen05 kernel runs its whole
prologue on the critical path of the CTAs the hardware dispatches last,
after the dependent-launch release.  These helpers render the cheaper forms:

* :func:`floordiv_nonneg_expr` / :func:`mod_nonneg_expr` for tile indices,
  tile offsets and virtual pids (never negative): a shift / mask or a
  truncating division instead of the DSL's floor-division fix-up chains.
* :func:`masked_cluster_rank_expr`: the cluster-local CTA rank masked to its
  static range, so the coordinate math the DSL derives from it folds to
  shifts (the compiler cannot see the range through ``make_warp_uniform``).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from ..ast_extension import statement_from_string

if TYPE_CHECKING:
    from ..device_function import DeviceFunction

PROLOGUE_RUNTIME = "_helion_tc_pro"
_PREAMBLE = (
    f"import helion._compiler.cute._tcgen05_prologue_runtime as {PROLOGUE_RUNTIME}"
)


def split_barrier_init_supported() -> bool:
    """Whether the clustered kernels may build their pipelines through the
    runtime's no-init constructors: the installed DSL must be the release
    whose private pipeline surface they mirror
    (``_tcgen05_prologue_runtime.SPLIT_BARRIER_INIT_SUPPORTED``); otherwise
    codegen keeps the DSL constructors with their serial warp-0 inits."""
    from ._tcgen05_prologue_runtime import SPLIT_BARRIER_INIT_SUPPORTED

    return SPLIT_BARRIER_INIT_SUPPORTED


def prologue_runtime(device_function: DeviceFunction) -> str:
    """Return the module alias of the prologue runtime, importing it once."""
    statements = device_function.codegen.module_statements
    if not any(ast.unparse(stmt) == _PREAMBLE for stmt in statements):
        statements.append(statement_from_string(_PREAMBLE))
    return PROLOGUE_RUNTIME


def floordiv_nonneg_expr(
    device_function: DeviceFunction, expr: str, divisor: object
) -> str:
    """``expr // divisor`` for a non-negative ``expr`` (a tile index, tile
    offset or virtual pid) and a positive ``divisor`` (an int or an
    expression the DSL resolves to one)."""
    return f"{prologue_runtime(device_function)}.floordiv_nonneg({expr}, {divisor})"


def mod_nonneg_expr(device_function: DeviceFunction, expr: str, divisor: object) -> str:
    """``expr % divisor`` for a non-negative ``expr`` (see
    :func:`floordiv_nonneg_expr`)."""
    return f"{prologue_runtime(device_function)}.mod_nonneg({expr}, {divisor})"


def masked_cluster_rank_expr(rank_expr: str, cluster_size: int) -> str:
    """The cluster-local CTA rank ``rank_expr`` (in ``[0, cluster_size)``)
    masked to that range when it is a power of two: a no-op the compiler uses
    to fold the floor divisions of the coordinate math into shifts."""
    if cluster_size > 1 and cluster_size & (cluster_size - 1) == 0:
        return f"({rank_expr}) & cutlass.Int32({cluster_size - 1})"
    return rank_expr


def inline_floordiv_nonneg_expr(
    expr: str, divisor_expr: str, divisor: int | None
) -> str:
    """``expr // divisor`` for a non-negative ``expr`` as an inline shift when
    ``divisor`` is a static power of two, else the DSL's ``//``.  The inline
    form keeps the persistent-kernel splitter's side-effect-free allowlist
    (``_tcgen05_unsafe_shared_stmts``) satisfied where a helper call would not."""
    if divisor == 1:
        return f"({expr})"
    if divisor is not None and divisor > 1 and divisor & (divisor - 1) == 0:
        return f"(({expr}) >> cutlass.Int32({divisor.bit_length() - 1}))"
    return f"(({expr}) // ({divisor_expr}))"


def inline_mod_nonneg_expr(expr: str, divisor_expr: str, divisor: int | None) -> str:
    """``expr % divisor`` for a non-negative ``expr`` (see
    :func:`inline_floordiv_nonneg_expr`): an inline mask for a static power
    of two, else the DSL's ``%``."""
    if divisor is not None and divisor > 1 and divisor & (divisor - 1) == 0:
        return f"(({expr}) & cutlass.Int32({divisor - 1}))"
    return f"(({expr}) % ({divisor_expr}))"

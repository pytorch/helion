"""Shared configuration-dependent fragment extents and storage accounting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
import torch
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource

from ... import exc
from ..._compat import shape_env_size_hint
from ..host_function import HostFunction
from ..variable_origin import BlockSizeOrigin

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..compile_environment import CompileEnvironment


def aligned_shared_bytes(count: int, dtype: torch.dtype) -> int:
    return (count * dtype.itemsize + 15) // 16 * 16


def metadata_guarded(env: CompileEnvironment, expr: sympy.Expr) -> bool:
    if "input_tensor_metadata" not in env.compiler_fact_specialization_facts:
        return False
    # The exact metadata guard covers sizes and strides, not runtime scalar
    # arguments, device values, or storage offsets.
    return all(
        isinstance(symbol, sympy.Symbol)
        and any(
            isinstance(source, TensorPropertySource)
            and source.prop in (TensorProperty.SIZE, TensorProperty.STRIDE)
            for source in env.shape_env.var_to_sources.get(symbol, ())
        )
        for symbol in expr.free_symbols
    )


def configured_fragment_expr(
    env: CompileEnvironment,
    value: sympy.Basic,
    resolve_block_size: Callable[[int], int | torch.SymInt | None],
) -> sympy.Basic:
    """Resolve logical dimensions using their owning block, not aliases.

    Reduction blocks can reuse a tile symbol or have a derived full-axis
    extent. Fragment roots own their iteration/reduction geometry, so a
    reduction's logical numel is required rather than a padded tracing hint.
    Runtime symbols with no block-size origin remain symbolic.
    """
    origins = HostFunction.current().expr_to_origin

    def resolve(expr: sympy.Basic, visiting: frozenset[sympy.Basic]) -> sympy.Basic:
        substitutions = {}
        for symbol in expr.free_symbols:
            info = origins.get(symbol)
            if info is None or not isinstance(info.origin, BlockSizeOrigin):
                continue
            if symbol in visiting:
                raise exc.InvalidConfig(
                    f"cyclic computed fragment block-size extent: {symbol}"
                )
            block = env.block_sizes[info.origin.block_id]
            replacement = (
                block.numel if block.reduction else resolve_block_size(block.block_id)
            )
            if replacement is None:
                continue
            if isinstance(replacement, torch.SymInt):
                replacement = replacement._sympy_()
            resolved = resolve(sympy.sympify(replacement), visiting | {symbol})
            if block.reduction:
                logical = env.specialize_expr(sympy.sympify(resolved))
                if logical.free_symbols and metadata_guarded(env, logical):
                    # Storage has a static capacity; masks and scans still
                    # use block.numel's runtime logical bound. Exact tensor
                    # metadata in the binding key makes this hint a proof,
                    # including when a direct BoundKernel call is replayed.
                    resolved = sympy.Integer(
                        env.backend.static_rdim_size(
                            max(1, shape_env_size_hint(env.shape_env, logical))
                        )
                    )
            substitutions[symbol] = resolved
        return expr.xreplace(substitutions)

    return resolve(value, frozenset())

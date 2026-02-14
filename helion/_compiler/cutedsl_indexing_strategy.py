"""CuteDSL-native indexing strategy for load/store operations.

Instead of tl.load(ptr + offset, mask) / tl.store(ptr + offset, value, mask),
CuteDSL uses direct tensor indexing: tensor[flat_offset] / tensor[flat_offset] = value.

In the per-thread model, each thread computes a single scalar flat offset.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .indexing_strategy import IndexingStrategy
from .indexing_strategy import SubscriptIndexing

if TYPE_CHECKING:
    import torch

    from .inductor_lowering import CodegenState


class CuteDSLIndexingStrategy(IndexingStrategy):
    """CuteDSL-native load/store using direct tensor indexing.

    Reuses SubscriptIndexing for offset computation, then generates:
    - Load:  tensor[flat_offset]  (with conditional for mask)
    - Store: tensor[flat_offset] = value  (with if-guard for mask)
    """

    def codegen_load(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
        eviction_policy: ast.AST | None,
    ) -> ast.AST:
        indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
        name = state.device_function.tensor_arg(fake_tensor).name

        load_expr = expr_from_string(
            f"{name}[{{offset}}]",
            offset=indexing.index_expr,
        )

        if indexing.has_mask():
            # Wrap in conditional: val if mask else 0
            if fake_tensor.dtype in (
                __import__("torch").float8_e4m3fn,
                __import__("torch").float8_e5m2,
            ):
                other = "0.0"
            else:
                other = "0"
            load_expr = expr_from_string(
                f"({{load}} if {{mask}} else {other})",
                load=load_expr,
                mask=indexing.mask_expr,
            )

        return load_expr

    def codegen_store(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> ast.AST:
        indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
        name = state.device_function.tensor_arg(fake_tensor).name

        store_stmt = statement_from_string(
            f"{name}[{{offset}}] = {{value}}",
            offset=indexing.index_expr,
            value=value,
        )

        if indexing.has_mask():
            # Wrap in if-guard: if mask: tensor[offset] = value
            from .ast_extension import create

            if_stmt = create(
                ast.If,
                test=indexing.mask_expr,
                body=[store_stmt],
                orelse=[],
            )
            state.add_statement(if_stmt)
            return expr_from_string("None")

        state.add_statement(store_stmt)
        return expr_from_string("None")

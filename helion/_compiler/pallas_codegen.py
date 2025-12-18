"""Pallas-specific code generation for TPU support.

This module provides indexing strategies and code generation utilities for
generating Pallas/JAX code instead of Triton code.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import torch

from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .indexing_strategy import IndexingStrategy

if TYPE_CHECKING:
    from .inductor_lowering import CodegenState


class PallasIndexingStrategy(IndexingStrategy):
    """Pallas indexing strategy that generates JAX/Pallas array access code.

    Instead of Triton's pointer arithmetic (tl.load/tl.store with offset+mask),
    this generates JAX-style array indexing with Pallas dynamic slices.
    """

    def codegen_load(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        extra_mask: ast.AST | None,
        eviction_policy: ast.AST | None,
    ) -> ast.AST:
        """Generate Pallas load code.

        In Pallas with BlockSpec, the kernel receives pre-sliced blocks.
        We access the entire block using slice notation [:].
        """
        # Name already includes _ref suffix for Pallas backend
        name = state.device_function.tensor_arg(fake_tensor).name

        # With BlockSpec, we access the entire block - the slicing is done by pallas_call
        # For 1D tensors, use [:] to get the whole block
        return expr_from_string(f"{name}[:]")

    def codegen_store(
        self,
        state: CodegenState,
        fake_tensor: torch.Tensor,
        subscript: list[object],
        value: ast.AST,
        extra_mask: ast.AST | None,
    ) -> None:
        """Generate Pallas store code.

        In Pallas with BlockSpec, the kernel receives pre-sliced blocks.
        We write to the entire block using slice notation [:].

        Since assignments are statements (not expressions), we add the statement
        directly and return None.
        """
        # Name already includes _ref suffix for Pallas backend
        name = state.device_function.tensor_arg(fake_tensor).name

        # With BlockSpec, we write to the entire block - the slicing is done by pallas_call
        # For 1D tensors, use [:] to write the whole block
        state.add_statement(
            statement_from_string(
                f"{name}[:] = {{value}}",
                value=value,
            )
        )
        return None

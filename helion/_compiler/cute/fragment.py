"""Typed, thread-local SSA values and their explicit register materialization.

A fragment's extent describes its value, not a memory instruction's width.
Packing a register buffer produces an immutable CuTe TensorSSA: subsequent
writes to the buffer cannot change that value. These primitives are shared
by initialization, scalar operation emission, and control-flow lowering.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment

if TYPE_CHECKING:
    import torch

    from ..generate_ast import GenerateAST


@dataclasses.dataclass(frozen=True)
class FragmentType:
    """A flat, rank-one bundle of typed per-thread register slots."""

    dtype: torch.dtype
    elements: int


@dataclasses.dataclass(frozen=True)
class SSAFragment:
    name: str
    type: FragmentType

    def element(self, index: str) -> ast.AST:
        # The index is a flat register slot, not a logical CuTe coordinate.
        # A vector view needs no lazily cached logical-layout IR, whose
        # definition could otherwise escape the first projection's region.
        return expr_from_string(f"{self.name}.to_vector()[{index}]")


@dataclasses.dataclass(frozen=True)
class RegisterBuffer:
    name: str
    type: FragmentType

    def element(self, index: str) -> ast.AST:
        return expr_from_string(f"{self.name}[{index}]")


class FragmentEmitter:
    def __init__(self, cg: GenerateAST) -> None:
        self.cg = cg

    def allocate(self, prefix: str, fragment_type: FragmentType) -> RegisterBuffer:
        buffer = RegisterBuffer(
            self.cg.device_function.new_var(prefix, dce=False), fragment_type
        )
        dtype = CompileEnvironment.current().backend.dtype_str(fragment_type.dtype)
        self.cg.add_statement(
            f"{buffer.name} = cute.make_rmem_tensor(({fragment_type.elements},), {dtype})"
        )
        return buffer

    def pack(self, buffer: RegisterBuffer, prefix: str) -> SSAFragment:
        value = SSAFragment(self.cg.device_function.new_var(prefix), buffer.type)
        self.cg.add_statement(f"{value.name} = {buffer.name}.load()")
        return value

    def bind(self, value: SSAFragment, prefix: str) -> SSAFragment:
        """Give a value a distinct control-flow binding, without copying storage."""
        result = SSAFragment(self.cg.device_function.new_var(prefix), value.type)
        self.assign(result, value)
        return result

    def assign(self, target: SSAFragment, value: SSAFragment) -> None:
        assert target.type == value.type
        self.cg.add_statement(f"{target.name} = {value.name}")

    def store(self, buffer: RegisterBuffer, index: str, value: object) -> None:
        if isinstance(value, (bool, int, float)):
            value = ast.Constant(value=value)
        assert isinstance(value, ast.AST)
        value = CompileEnvironment.current().backend.cast_ast(value, buffer.type.dtype)
        self.cg.add_statement(
            statement_from_string(f"{buffer.name}[{index}] = {{value}}", value=value)
        )

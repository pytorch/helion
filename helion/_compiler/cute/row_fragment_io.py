"""Guarded scalar and vector memory access for subgroup-owned row fragments."""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING

import torch
from torch.fx import Node

from ..compile_environment import CompileEnvironment
from .row_fragment import RowFragment

if TYPE_CHECKING:
    from ..generate_ast import GenerateAST
    from .row_fragment import RowFragmentLayout


class RowFragmentIO:
    """Transfer proven row-local loads/stores without changing fragment ownership.

    The graph planner proves indexing and storage disjointness. This emitter
    additionally proves complete contiguous vectors and checks actual pointer
    alignment; all other accesses retain scalar bounds checks.
    """

    def __init__(self, cg: GenerateAST, *, row: str, lane: str, valid_row: str) -> None:
        self.cg = cg
        self.row = row
        self.lane = lane
        self.valid_row = valid_row

    def _add(self, source: str) -> None:
        for statement in ast.parse(source).body:
            self.cg.add_statement(statement)

    def _tensor(self, node: Node) -> tuple[torch.Tensor, str, list[object]]:
        tensor_node, subscript = node.args[:2]
        assert isinstance(tensor_node, Node)
        tensor = tensor_node.meta["val"]
        assert isinstance(tensor, torch.Tensor)
        assert isinstance(subscript, (tuple, list))
        name = self.cg.device_function.tensor_arg(tensor).name
        self.cg.device_function.placeholder_args.add(name)
        return tensor, name, [index for index in subscript if index is not None]

    def expression(self, node: Node, column: str) -> str:
        tensor, name, subscript = self._tensor(node)
        indices = []
        for dimension, index in enumerate(subscript):
            if isinstance(index, Node):
                indices.append(self.row)
            elif index == slice(None):
                indices.append("0" if tensor.size(dimension) == 1 else column)
            else:
                assert isinstance(index, int)
                indices.append(str(index))
        return f"{name}[{', '.join(indices)}]"

    def _vector_offset(self, node: Node, vector: int, column: str) -> str | None:
        tensor, _name, subscript = self._tensor(node)
        if vector <= 1 or tensor.dtype is torch.bool:
            return None
        if tensor.ndim == 1 and subscript == [slice(None)] and tensor.stride(0) == 1:
            return f"cutlass.Int64({column})"
        if (
            tensor.ndim == 2
            and len(subscript) == 2
            and isinstance(subscript[0], Node)
            and subscript[1] == slice(None)
            and tensor.stride(1) == 1
            and isinstance(tensor.stride(0), int)
            and tensor.stride(0) % vector == 0
        ):
            return (
                f"cutlass.Int64({self.row}) * cutlass.Int64({tensor.stride(0)})"
                f" + cutlass.Int64({column})"
            )
        return None

    def _transfer(self, node: Node, fragment: RowFragment, *, store: bool) -> None:
        tensor, name, _subscript = self._tensor(node)
        backend = CompileEnvironment.current().backend
        fn = self.cg.device_function
        extent = tensor.size(1) if tensor.ndim == 2 else tensor.size(0)
        if not store:
            extent = fragment.extent
        elif tensor.ndim == 1:
            extent = 1
        single_owner = store and fragment.replicated and extent == 1
        registers = 1 if single_owner else fragment.layout.num_registers(extent)
        if fragment.replicated and not store:
            registers = 1
        register = fn.new_var("row_store_register" if store else "row_load_register")
        column = (
            "0"
            if fragment.replicated and extent == 1
            else fragment.layout.column(register)
        )
        owner = f"({self.lane} == 0)" if single_owner else f"(({column}) < {extent})"
        address = self.expression(node, column)
        value = ast.unparse(backend.cast_ast(fragment.element(register), tensor.dtype))
        assignment = (
            f"{address} = {value}"
            if store
            else f"{fragment.name}[{register}] = {address}"
        )
        scalar = f"""
for {register} in cutlass.range_constexpr({registers}):
    if {self.valid_row} & {owner}:
        {assignment}
"""
        vector = fragment.layout.vector_width
        group = fn.new_var("row_transfer_group")
        column = fragment.layout.column(f"{group} * {vector}")
        offset_expression = self._vector_offset(node, vector, column)
        if single_owner or extent % vector or offset_expression is None:
            self._add(scalar)
            return
        item = fn.new_var("row_transfer_item")
        packed = fn.new_var("row_transfer_vector")
        memory = fn.new_var("row_transfer_memory")
        offset = fn.new_var("row_transfer_offset")
        dtype = backend.dtype_str(tensor.dtype)
        scalar_value = ast.unparse(
            backend.cast_ast(
                fragment.element(f"{group} * {vector} + {item}"), tensor.dtype
            )
        )
        transfer = (
            f"for {item} in cutlass.range_constexpr({vector}):\n"
            f"    {packed}[{item}] = {scalar_value}\n"
            f"cute.autovec_copy({packed}, {memory})"
            if store
            else f"cute.autovec_copy({memory}, {packed})\n"
            f"for {item} in cutlass.range_constexpr({vector}):\n"
            f"    {fragment.name}[{group} * {vector} + {item}] = {packed}[{item}]"
        )
        wide = f"""
for {group} in cutlass.range_constexpr({registers // vector}):
    if {self.valid_row} & (({column}) < {extent}):
        {packed} = cute.make_rmem_tensor({vector}, {dtype})
        {offset} = {offset_expression}
        {offset} = cute.assume({offset}, divby={vector})
        {memory} = cute.make_tensor({name}.iterator + {offset}, cute.make_layout({vector}, stride=1))
{textwrap.indent(transfer, "        ")}
"""
        self._add(
            f"if cutlass.const_expr({name}.iterator.alignment >= {min(16, vector * tensor.element_size())}):\n"
            + textwrap.indent(wide, "    ")
            + "else:\n"
            + textwrap.indent(scalar, "    ")
        )

    def load(self, node: Node, layout: RowFragmentLayout) -> RowFragment:
        tensor, _name, subscript = self._tensor(node)
        extent = 1
        for dimension, index in enumerate(subscript):
            if index == slice(None):
                extent = tensor.size(dimension)
        assert isinstance(extent, int)
        value = node.meta["val"]
        assert isinstance(value, torch.Tensor)
        fragment = RowFragment(
            self.cg.device_function.new_var("row_load"),
            value.dtype,
            extent,
            layout,
            replicated=extent == 1,
        )
        dtype = CompileEnvironment.current().backend.dtype_str(value.dtype)
        self._add(
            f"{fragment.name} = cute.make_rmem_tensor({fragment.num_registers}, {dtype})\n"
            f"{fragment.name}.fill({dtype}(0))"
        )
        self._transfer(node, fragment, store=False)
        return fragment

    def store(self, node: Node, fragment: RowFragment) -> None:
        self._transfer(node, fragment, store=True)

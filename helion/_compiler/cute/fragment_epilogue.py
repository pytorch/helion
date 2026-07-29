"""Execution of a generic FX epilogue on a tcgen05 register fragment."""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING

import torch
from torch._inductor.virtualized import V
from torch.fx.node import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_index
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .cute_epilogue import _fragment_elementwise_inputs
from .cute_reshape import _coords_from_flat_index
from .cute_reshape import _flat_index_from_coords
from .cute_reshape import _get_tile_shape
from .cute_reshape import broadcast_logical_flat_index
from .cute_reshape import logical_coords_from_flat
from .cute_reshape import logical_flat_from_coords
from .cute_reshape import resolve_cute_logical_coordinate

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState
    from .cute_epilogue import Tcgen05EpilogueStorePlan


@dataclasses.dataclass(frozen=True)
class FragmentEpilogueCode:
    prelude: str
    expression: str


def _shape(state: CodegenState, node: Node) -> list[int]:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"fragment node {node.name} is not tensor-valued")
    return _get_tile_shape(
        value, CompileEnvironment.current(), state.device_function.config
    )


def _elementwise_inputs(node: Node) -> tuple[Node, ...]:
    inputs = _fragment_elementwise_inputs(node)
    if inputs is None:
        raise RuntimeError(f"invalid fragment elementwise inputs for {node.name}")
    return inputs


def _capture_elementwise(
    state: CodegenState,
    node: Node,
    inputs: list[ast.AST],
    *,
    preserve_tensor_container: bool = False,
) -> tuple[list[ast.AST], ast.AST]:
    from ..aten_lowering import LoweringContext
    from ..inductor_lowering import APIFuncLowering
    from ..inductor_lowering import PointwiseLowering

    lowering = node.meta.get("lowering")
    if not isinstance(lowering, (PointwiseLowering, APIFuncLowering)):
        raise TypeError(f"{node.name} is not an elementwise lowering")
    ctx = LoweringContext.__new__(LoweringContext)
    ctx.cg = state.codegen
    ctx.env = state.env
    statements: list[ast.AST] = []
    with state.codegen.set_statements(statements), V.set_current_node(node):
        container_template: ast.AST | None = None
        if preserve_tensor_container:
            if not inputs:
                raise TypeError(f"tensor fragment {node.name} has no inputs")
            inputs = [state.codegen.lift(value) for value in inputs]
            container_template = inputs[0]
        if isinstance(lowering, PointwiseLowering):
            result = lowering.codegen_from_input_asts(
                ctx,
                node,
                inputs,
                preserve_tensor_container=preserve_tensor_container,
            )
        else:
            result = lowering.codegen_from_input_asts(ctx, node, inputs)
    if not isinstance(result, ast.AST):
        raise TypeError(f"elementwise fragment result for {node.name}: {type(result)}")
    if container_template is not None and not isinstance(lowering, PointwiseLowering):
        result = expr_from_string(
            "_cute_ensure_tensor_ssa({value}, {container})",
            value=result,
            container=container_template,
        )
    return statements, result


def _render_ast_statements(statements: list[ast.AST], indent: str) -> str:
    return "".join(
        f"{indent}{line}\n"
        for statement in statements
        for line in ast.unparse(ast.fix_missing_locations(statement)).splitlines()
    )


def _whole_fragment(
    state: CodegenState,
    store: Tcgen05EpilogueStorePlan,
    *,
    carrier_name: str,
    target_dtype: str,
    load_locals: dict[Node, str],
    indent: str,
) -> FragmentEpilogueCode:
    lines: list[str] = []
    memo: dict[Node, ast.AST] = {}
    boundary_set = set(store.boundary_nodes)
    carrier_loaded = state.device_function.new_var("tcgen05_epi_acc")
    lines.append(f"{indent}{carrier_loaded} = {carrier_name}.load()\n")

    def evaluate(node: Node) -> ast.AST:
        if node in memo:
            return memo[node]
        if node in boundary_set:
            result = expr_from_string(carrier_loaded)
        elif node.target is memory_ops.load:
            local = load_locals.get(node)
            if local is None:
                raise RuntimeError(f"missing staged fragment load for {node.name}")
            result = expr_from_string(
                "_cute_ensure_tensor_ssa({value}, {container})",
                value=expr_from_string(local),
                container=expr_from_string(carrier_loaded),
            )
        else:
            from ..inductor_lowering import APIFuncLowering
            from ..inductor_lowering import PointwiseLowering

            lowering = node.meta.get("lowering")
            if not isinstance(lowering, (PointwiseLowering, APIFuncLowering)):
                raise RuntimeError(
                    f"whole-fragment plan contains non-elementwise node {node.name}"
                )
            inputs = [evaluate(input_node) for input_node in _elementwise_inputs(node)]
            statements, expression = _capture_elementwise(
                state, node, inputs, preserve_tensor_container=True
            )
            lines.append(_render_ast_statements(statements, indent))
            if isinstance(expression, ast.Name):
                result = expression
            else:
                name = state.device_function.new_var("tcgen05_epi_value")
                assignment = statement_from_string(
                    f"{name} = {{value}}", value=expression
                )
                lines.append(_render_ast_statements([assignment], indent))
                result = expr_from_string(name)
        memo[node] = result
        return result

    expression = evaluate(store.value_node)
    value = store.value_node.meta.get("val")
    rendered = ast.unparse(expression)
    if not isinstance(value, torch.Tensor) or value.dtype != store.output_tensor.dtype:
        rendered = f"({rendered}).to({target_dtype})"
    return FragmentEpilogueCode("".join(lines), rendered)


class _ScalarEvaluator:
    def __init__(
        self,
        state: CodegenState,
        store: Tcgen05EpilogueStorePlan,
        *,
        carrier_name: str,
        coordinate_name: str,
        body_indent: str,
    ) -> None:
        self.state = state
        self.carrier_name = carrier_name
        self.coordinate_name = coordinate_name
        self.body_indent = body_indent
        self.output_block_ids = store.output_block_ids
        self.boundaries = set(store.boundary_nodes)
        self.memo: dict[tuple[Node, str, int | None], ast.AST] = {}
        self.index_memo: dict[str, str] = {}
        self.lines: list[str] = []

    def _bind(self, prefix: str, expression: ast.AST) -> ast.AST:
        name = self.state.device_function.new_var(prefix)
        self.lines.append(
            _render_ast_statements(
                [statement_from_string(f"{name} = {{value}}", value=expression)],
                self.body_indent,
            )
        )
        return expr_from_string(name)

    def _index(self, expression: str) -> str:
        if expression in self.index_memo:
            return self.index_memo[expression]
        name = self.state.device_function.new_var("tcgen05_epi_flat")
        self.lines.append(f"{self.body_indent}{name} = cutlass.Int32({expression})\n")
        self.index_memo[expression] = name
        return name

    def _broadcast_flat(self, output: Node, source: Node, flat: str) -> str:
        source_flat = broadcast_logical_flat_index(
            flat,
            output_shape=_shape(self.state, output),
            source_shape=_shape(self.state, source),
        )
        if source_flat is None:
            raise RuntimeError(f"unsupported fragment broadcast into {output.name}")
        return self._index(str(source_flat))

    def _boundary(self, node: Node, flat: str) -> ast.AST:
        coords = _coords_from_flat_index(flat, _shape(self.state, node))
        if len(coords) == 2:
            target_m, target_n = coords
        elif len(coords) == 3:
            target_h, target_m, target_n = coords
            if target_h != "cutlass.Int32(0)":
                raise RuntimeError("tcgen05 fragment cannot cross a leading tile")
        else:
            raise RuntimeError("tcgen05 fragment boundary must be rank 2 or 3")
        result = self.state.device_function.new_var("tcgen05_epi_acc_scalar")
        matches = self.state.device_function.new_var("tcgen05_epi_matches")
        scan = self.state.device_function.new_var("tcgen05_epi_scan")
        coord = self.state.device_function.new_var("tcgen05_epi_scan_coord")
        is_match = self.state.device_function.new_var("tcgen05_epi_is_match")
        indent = self.body_indent
        self.lines.extend(
            [
                f"{indent}{result} = {self.carrier_name}[0]\n",
                f"{indent}{matches} = cutlass.Int32(0)\n",
                f"{indent}for {scan} in range(cute.size({self.carrier_name}.shape)):\n",
                f"{indent}    {coord} = {self.coordinate_name}[{scan}]\n",
                (
                    f"{indent}    {is_match} = {coord}[0] == {target_m} and "
                    f"{coord}[1] == {target_n}\n"
                ),
                (
                    f"{indent}    {result} = {self.carrier_name}[{scan}] if "
                    f"{is_match} else {result}\n"
                ),
                f"{indent}    {matches} = {matches} + cutlass.Int32({is_match})\n",
                (
                    f"{indent}cute.testing.assert_({matches} == cutlass.Int32(1), "
                    '"tcgen05 fragment coordinate must have exactly one local owner")\n'
                ),
            ]
        )
        return expr_from_string(result)

    def _tile_index(self, node: Node, flat: str) -> ast.AST:
        from ...language.memory_ops import _cute_tile_begin_expr

        size_node = node.args[0] if node.args else None
        size = size_node.meta.get("val") if isinstance(size_node, Node) else size_node
        base = _cute_tile_begin_expr(self.state, size)
        coord = _coords_from_flat_index(flat, _shape(self.state, node))[0]
        return expr_from_string(f"cutlass.Int32({base}) + cutlass.Int32({coord})")

    def _host_load(self, node: Node, flat: str) -> ast.AST:
        from .matmul_utils import cute_rematerialize_scalar_load

        source_node = node.args[0] if node.args else None
        indices = node.args[1] if len(node.args) > 1 else None
        if not isinstance(source_node, Node) or not isinstance(indices, (list, tuple)):
            raise RuntimeError(f"malformed fragment load {node.name}")
        tensor = source_node.meta.get("val")
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"fragment load {node.name} has no host tensor")
        output_value = node.meta.get("val")
        if not isinstance(output_value, torch.Tensor):
            raise RuntimeError(f"fragment load {node.name} has no tensor metadata")
        output_shape = _shape(self.state, node)
        output_coords = logical_coords_from_flat(flat, output_shape)
        advanced = [
            index
            for index in indices
            if isinstance(index, Node)
            and isinstance(index.meta.get("val"), torch.Tensor)
        ]
        index_overrides: dict[int, str] = {}
        env_overrides: dict[Node, ast.AST] = {}
        if advanced:
            if not all(isinstance(index, Node) for index in indices):
                raise RuntimeError(
                    "mixed basic/advanced fragment loads are unsupported"
                )
            for index in indices:
                assert isinstance(index, Node)
                index_flat = self._broadcast_flat(node, index, flat)
                env_overrides[index] = self.evaluate(index, index_flat)
        else:
            from ...language.memory_ops import _cute_tile_begin_expr

            output_dim = 0
            env = CompileEnvironment.current()
            index_block_ids = [
                env.get_block_id(value)
                if isinstance(index, Node)
                and isinstance(value := index.meta.get("val"), torch.SymInt)
                else None
                for index in indices
            ]
            represented_block_ids = tuple(
                block_id
                for block_id in self.output_block_ids
                if block_id in index_block_ids
            )
            for tensor_dim, index in enumerate(indices):
                if env.known_equal(tensor.shape[tensor_dim], 1) and not (
                    isinstance(index, slice) and index == slice(None)
                ):
                    if isinstance(index, Node) and isinstance(
                        index.meta.get("val"), torch.SymInt
                    ):
                        block_id = env.get_block_id(index.meta["val"])
                        if block_id is not None:
                            index_overrides[block_id] = "cutlass.Int32(0)"
                        output_dim += 1
                    continue
                if isinstance(index, Node):
                    value = index.meta.get("val")
                    if isinstance(value, torch.SymInt):
                        base = _cute_tile_begin_expr(self.state, value)
                        block_id = env.get_block_id(value)
                        if block_id is None:
                            raise RuntimeError(
                                f"fragment load index has no block id: {node.name}"
                            )
                        coord_dim = (
                            represented_block_ids.index(block_id)
                            if block_id in represented_block_ids
                            and len(represented_block_ids) == len(output_coords)
                            else output_dim
                        )
                        index_overrides[block_id] = (
                            f"({base}) + ({output_coords[coord_dim]})"
                        )
                        output_dim += 1
                    elif not isinstance(value, int):
                        raise RuntimeError(f"unsupported fragment load index {value!r}")
                elif isinstance(index, slice) and index == slice(None):
                    output_size = output_value.shape[output_dim]
                    block_id = env.resolve_block_id(output_size)
                    if block_id is None:
                        raise RuntimeError(
                            f"fragment slice has no block id: {node.name}"
                        )
                    base = _cute_tile_begin_expr(self.state, output_size)
                    index_overrides[block_id] = (
                        f"({base}) + ({output_coords[output_dim]})"
                    )
                    output_dim += 1
                elif not isinstance(index, int):
                    raise RuntimeError(f"unsupported fragment load index {index!r}")
        rematerialized = cute_rematerialize_scalar_load(
            self.state,
            node,
            index_overrides=index_overrides,
            env_overrides=env_overrides,
        )
        if rematerialized is None:
            raise RuntimeError(f"failed to rematerialize fragment load {node.name}")
        statements, expression = rematerialized
        self.lines.append(_render_ast_statements(statements, self.body_indent))
        return self._bind("tcgen05_epi_load", expression)

    def evaluate(self, node: Node, flat: str, projection: int | None = None) -> ast.AST:
        key = (node, flat, projection)
        if key in self.memo:
            return self.memo[key]

        def leaf(
            current: Node, current_flat: int | str, current_projection: int | None
        ) -> ast.AST:
            if not isinstance(current_flat, str):
                current_flat = str(current_flat)
            if current in self.boundaries:
                return self._boundary(current, current_flat)
            if current.target is memory_ops.load:
                source = current.args[0] if current.args else None
                if (
                    isinstance(source, Node)
                    and source.target is _tracing_ops._host_tensor
                ):
                    return self._host_load(current, current_flat)
                if isinstance(source, Node):
                    indices = current.args[1] if len(current.args) > 1 else None
                    if not isinstance(indices, (list, tuple)):
                        raise RuntimeError(
                            f"malformed device-value load {current.name}"
                        )
                    output_coords = iter(
                        logical_coords_from_flat(
                            current_flat, _shape(self.state, current)
                        )
                    )
                    source_coords: list[int | str] = []
                    for index in indices:
                        coord = next(output_coords)
                        if index is None:
                            continue
                        if isinstance(index, slice) and index == slice(None):
                            source_coords.append(coord)
                        else:
                            raise RuntimeError(
                                f"unsupported device-value load index {index!r}"
                            )
                    source_flat = logical_flat_from_coords(
                        source_coords, _shape(self.state, source)
                    )
                    return self.evaluate(source, self._index(str(source_flat)))
                raise RuntimeError(f"malformed fragment load {current.name}")
            if current.target is tile_index:
                return self._tile_index(current, current_flat)
            from ..inductor_lowering import APIFuncLowering
            from ..inductor_lowering import PointwiseLowering

            lowering = current.meta.get("lowering")
            if not isinstance(lowering, (PointwiseLowering, APIFuncLowering)):
                raise RuntimeError(
                    f"unsupported fragment node {current.name}: {current.target}"
                )
            inputs = [
                self.evaluate(
                    input_node,
                    self._broadcast_flat(current, input_node, current_flat),
                )
                for input_node in _elementwise_inputs(current)
            ]
            statements, expression = _capture_elementwise(self.state, current, inputs)
            self.lines.append(_render_ast_statements(statements, self.body_indent))
            return self._bind("tcgen05_epi_value", expression)

        def select(selector: int | str, choices: tuple[ast.AST, ...]) -> ast.AST:
            if not isinstance(selector, str) or len(choices) != 2:
                raise RuntimeError("invalid fragment coordinate selection")
            return self._bind(
                "tcgen05_epi_join",
                expr_from_string(
                    "({left} if ({selector}) == cutlass.Int32(0) else {right})",
                    selector=expr_from_string(selector),
                    left=choices[0],
                    right=choices[1],
                ),
            )

        result = resolve_cute_logical_coordinate(
            node,
            flat,
            config=self.state.device_function.config,
            leaf=leaf,
            select=select,
            projection=projection,
        )
        self.memo[key] = result
        return result


def _scalar_fragment(
    state: CodegenState,
    store: Tcgen05EpilogueStorePlan,
    *,
    carrier_name: str,
    coordinate_name: str,
    target_dtype: str,
    indent: str,
) -> FragmentEpilogueCode:
    if store.required_pair_width is not None and (
        store.pair_local is None or store.pair_local.width != store.required_pair_width
    ):
        raise RuntimeError("scalar fragment reached rendering without pair-local proof")
    df = state.device_function
    output = df.new_var("tcgen05_epi_output")
    index = df.new_var("tcgen05_epi_index")
    coord = df.new_var("tcgen05_epi_coord")
    body_indent = indent + "    "
    output_shape = _shape(state, store.value_node)
    if len(output_shape) not in (2, 3):
        raise RuntimeError("tcgen05 scalar fragment output must be rank 2 or 3")
    matrix_shape = output_shape[-2:]
    matrix_flat = _flat_index_from_coords([f"{coord}[0]", f"{coord}[1]"], matrix_shape)
    evaluator = _ScalarEvaluator(
        state,
        store,
        carrier_name=carrier_name,
        coordinate_name=coordinate_name,
        body_indent=body_indent,
    )
    value = evaluator.evaluate(store.value_node, evaluator._index(matrix_flat))
    prelude = (
        f"{indent}{output} = cute.make_rmem_tensor("
        f"cute.make_layout({carrier_name}.shape), {target_dtype})\n"
        f"{indent}for {index} in range(cute.size({carrier_name}.shape)):\n"
        f"{body_indent}{coord} = {coordinate_name}[{index}]\n"
        f"{''.join(evaluator.lines)}"
        f"{body_indent}{output}[{index}] = {ast.unparse(value)}\n"
    )
    return FragmentEpilogueCode(prelude, f"{output}.load()")


def render_fragment_epilogue(
    state: CodegenState,
    store: Tcgen05EpilogueStorePlan,
    *,
    carrier_name: str,
    target_dtype: str,
    load_locals: dict[Node, str],
    coordinate_name: str | None,
    indent: str,
) -> FragmentEpilogueCode:
    if not store.requires_scalar_fragment:
        return _whole_fragment(
            state,
            store,
            carrier_name=carrier_name,
            target_dtype=target_dtype,
            load_locals=load_locals,
            indent=indent,
        )
    if coordinate_name is None:
        raise RuntimeError("scalar tcgen05 fragment requires logical coordinates")
    return _scalar_fragment(
        state,
        store,
        carrier_name=carrier_name,
        coordinate_name=coordinate_name,
        target_dtype=target_dtype,
        indent=indent,
    )

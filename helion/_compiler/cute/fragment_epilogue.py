"""Execution of a generic FX epilogue on a tcgen05 register fragment."""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import Any

import sympy
import torch
from torch._inductor.virtualized import V
from torch.fx.node import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import tile_index
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from .cute_epilogue import Tcgen05PairTraversal
from .cute_epilogue import _fragment_elementwise_inputs
from .cute_reshape import CuteLogicalCoordinate
from .cute_reshape import _flat_index_from_coords
from .cute_reshape import _get_tile_shape
from .cute_reshape import bounded_logical_coordinate_choices
from .cute_reshape import broadcast_logical_flat_index
from .cute_reshape import logical_coords_from_flat
from .cute_reshape import logical_flat_from_coords
from .cute_reshape import pair_local_target_coordinates
from .cute_reshape import resolve_cute_logical_coordinate

if TYPE_CHECKING:
    from ..inductor_lowering import CodegenState
    from .cute_epilogue import Tcgen05EpilogueStorePlan


@dataclasses.dataclass(frozen=True)
class FragmentEpilogueCode:
    prelude: str
    expression: str


_FragmentCoordinate = int | str | CuteLogicalCoordinate


@dataclasses.dataclass(frozen=True)
class _TerminalKey:
    kind: str
    node: Node
    coordinates: tuple[object, ...]
    predicate: Node | None
    dtype: torch.dtype | None


@dataclasses.dataclass(frozen=True)
class _PairSlot:
    index_name: str
    partner_index_name: str
    coordinates: tuple[CuteLogicalCoordinate, CuteLogicalCoordinate]
    partner_coordinates: tuple[CuteLogicalCoordinate, CuteLogicalCoordinate]


def _coordinate_expression(value: _FragmentCoordinate) -> str:
    if isinstance(value, CuteLogicalCoordinate):
        return value.expression
    if isinstance(value, int):
        return str(value)
    return value


def _coordinate_semantic(
    value: _FragmentCoordinate,
) -> Any | None:  # noqa: ANN401 - SymPy's operator stubs are incomplete.
    if isinstance(value, CuteLogicalCoordinate):
        return value.semantic
    if isinstance(value, int):
        return sympy.Integer(value)
    return None


def _proven_coordinate_choices(
    target: tuple[CuteLogicalCoordinate, ...],
    choices: tuple[tuple[CuteLogicalCoordinate, ...], ...],
) -> frozenset[int]:
    try:
        return bounded_logical_coordinate_choices(target, choices)
    except ValueError as exc:
        raise AssertionError(
            "rendered coordinate contradicts the planned pair-local proof"
        ) from exc


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
    # The carrier is the f32 TMEM accumulator whatever the FX value's dtype
    # says, so the target conversion is unconditional. A same-dtype ``to`` is a
    # no-op; keying it off ``value.dtype`` instead drops the cast whenever the
    # user seeds the accumulator at the output dtype (``hl.zeros(..., x.dtype)``).
    return FragmentEpilogueCode(
        "".join(lines), f"({ast.unparse(expression)}).to({target_dtype})"
    )


class _ScalarEvaluator:
    def __init__(
        self,
        state: CodegenState,
        store: Tcgen05EpilogueStorePlan,
        *,
        carrier_name: str,
        index_name: str,
        body_indent: str,
        pair_slot: _PairSlot | None = None,
        terminal_cache: dict[_TerminalKey, ast.AST] | None = None,
    ) -> None:
        self.state = state
        self.carrier_name = carrier_name
        self.index_name = index_name
        self.body_indent = body_indent
        self.pair_slot = pair_slot
        self.terminal_cache = terminal_cache
        self.output_block_ids = store.output_block_ids
        self.boundaries = set(store.boundary_nodes)
        self.memo: dict[tuple[Node, _FragmentCoordinate, int | None], ast.AST] = {}
        self.index_memo: dict[_FragmentCoordinate, _FragmentCoordinate] = {}
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

    def _index(self, expression: _FragmentCoordinate) -> _FragmentCoordinate:
        if expression in self.index_memo:
            return self.index_memo[expression]
        rendered = _coordinate_expression(expression)
        if rendered.isidentifier():
            # Already an ``Int32`` local this evaluator bound; rebinding it just
            # chains ``flat_n = cutlass.Int32(flat_n-1)`` down the recursion.
            self.index_memo[expression] = expression
            return expression
        name = self.state.device_function.new_var("tcgen05_epi_flat")
        self.lines.append(f"{self.body_indent}{name} = cutlass.Int32({rendered})\n")
        result: _FragmentCoordinate = name
        if isinstance(expression, CuteLogicalCoordinate):
            result = CuteLogicalCoordinate(
                name,
                expression.semantic,
                expression.bounds,
                expression.symbol_expressions,
            )
        self.index_memo[expression] = result
        return result

    def _broadcast_flat(
        self, output: Node, source: Node, flat: _FragmentCoordinate
    ) -> _FragmentCoordinate:
        source_flat = broadcast_logical_flat_index(
            flat,
            output_shape=_shape(self.state, output),
            source_shape=_shape(self.state, source),
        )
        if source_flat is None:
            raise RuntimeError(f"unsupported fragment broadcast into {output.name}")
        return self._index(source_flat)

    def _cached_terminal(self, key: _TerminalKey | None) -> ast.AST | None:
        if key is None or self.terminal_cache is None:
            return None
        return self.terminal_cache.get(key)

    def _remember_terminal(self, key: _TerminalKey | None, value: ast.AST) -> ast.AST:
        if key is not None and self.terminal_cache is not None:
            self.terminal_cache.setdefault(key, value)
        return value

    @staticmethod
    def _terminal_dtype(node: Node) -> torch.dtype | None:
        value = node.meta.get("val")
        return value.dtype if isinstance(value, torch.Tensor) else None

    def _boundary(self, node: Node, flat: _FragmentCoordinate) -> ast.AST:
        coords = logical_coords_from_flat(flat, _shape(self.state, node))
        if len(coords) == 2:
            target_m, target_n = coords
        elif len(coords) == 3:
            target_h, target_m, target_n = coords
            if isinstance(target_h, CuteLogicalCoordinate):
                leading_choices = _proven_coordinate_choices(
                    (target_h,),
                    (
                        (
                            dataclasses.replace(
                                target_h,
                                expression="cutlass.Int32(0)",
                                semantic=sympy.Integer(0),
                            ),
                        ),
                    ),
                )
                if leading_choices != frozenset({0}):
                    raise RuntimeError("tcgen05 fragment cannot cross a leading tile")
            else:
                target_h_semantic = _coordinate_semantic(target_h)
                if (
                    target_h_semantic is not None
                    and sympy.simplify(target_h_semantic) != 0
                ) or (
                    target_h_semantic is None
                    and _coordinate_expression(target_h) != "cutlass.Int32(0)"
                ):
                    raise RuntimeError("tcgen05 fragment cannot cross a leading tile")
        else:
            raise RuntimeError("tcgen05 fragment boundary must be rank 2 or 3")
        semantic_coords = tuple(_coordinate_semantic(coord) for coord in coords)
        terminal_key = (
            _TerminalKey(
                "boundary",
                node,
                tuple(semantic_coords),
                None,
                self._terminal_dtype(node),
            )
            if all(coord is not None for coord in semantic_coords)
            else None
        )
        if (cached := self._cached_terminal(terminal_key)) is not None:
            return cached
        if self.pair_slot is not None:
            if not isinstance(target_m, CuteLogicalCoordinate) or not isinstance(
                target_n, CuteLogicalCoordinate
            ):
                raise RuntimeError("pair-local boundary lost typed coordinates")
            target = (target_m, target_n)
            pair_choices = _proven_coordinate_choices(
                target,
                (
                    self.pair_slot.coordinates,
                    self.pair_slot.partner_coordinates,
                ),
            )
            if pair_choices == frozenset({0}):
                index = self.pair_slot.index_name
                value = expr_from_string(f"{self.carrier_name}[{index}]")
            elif pair_choices == frozenset({1}):
                index = self.pair_slot.partner_index_name
                value = expr_from_string(f"{self.carrier_name}[{index}]")
            elif pair_choices == frozenset({0, 1}):
                is_current = " and ".join(
                    f"({_coordinate_expression(target_coordinate)}) == "
                    f"({_coordinate_expression(current_coordinate)})"
                    for target_coordinate, current_coordinate in zip(
                        target, self.pair_slot.coordinates, strict=True
                    )
                )
                value = expr_from_string(
                    f"{self.carrier_name}[{self.pair_slot.index_name}] if "
                    f"({is_current}) else "
                    f"{self.carrier_name}[{self.pair_slot.partner_index_name}]"
                )
            else:
                raise RuntimeError(
                    "pair-local boundary has an invalid bounded choice relation"
                )
            return self._remember_terminal(
                terminal_key,
                self._bind(
                    "tcgen05_epi_acc_scalar",
                    value,
                ),
            )
        # Without a pair slot the plan carries ``boundary_identity``: every
        # boundary demand resolves to the coordinate the loop is already
        # standing on, which is this register.
        return self._remember_terminal(
            terminal_key, expr_from_string(f"{self.carrier_name}[{self.index_name}]")
        )

    def _tile_index(self, node: Node, flat: _FragmentCoordinate) -> ast.AST:
        from ...language.memory_ops import _cute_tile_begin_expr

        size_node = node.args[0] if node.args else None
        size = size_node.meta.get("val") if isinstance(size_node, Node) else size_node
        base = _cute_tile_begin_expr(self.state, size)
        coord = logical_coords_from_flat(flat, _shape(self.state, node))[0]
        semantic = _coordinate_semantic(coord)
        terminal_key = (
            _TerminalKey(
                "tile_index",
                node,
                (semantic,),
                None,
                self._terminal_dtype(node),
            )
            if semantic is not None
            else None
        )
        if (cached := self._cached_terminal(terminal_key)) is not None:
            return cached
        result = expr_from_string(
            f"cutlass.Int32({base}) + cutlass.Int32({_coordinate_expression(coord)})"
        )
        if terminal_key is not None and self.terminal_cache is not None:
            result = self._bind("tcgen05_epi_tile_index", result)
        return self._remember_terminal(terminal_key, result)

    def _host_load(self, node: Node, flat: _FragmentCoordinate) -> ast.AST:
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
        flat_semantic = _coordinate_semantic(flat)
        extra_mask = node.args[2] if len(node.args) > 2 else None
        terminal_key = (
            _TerminalKey(
                "load",
                node,
                (flat_semantic,),
                extra_mask if isinstance(extra_mask, Node) else None,
                output_value.dtype,
            )
            if flat_semantic is not None
            else None
        )
        if (cached := self._cached_terminal(terminal_key)) is not None:
            return cached
        advanced = [
            index
            for index in indices
            if isinstance(index, Node)
            and isinstance(index.meta.get("val"), torch.Tensor)
        ]
        index_overrides: dict[int, str] = {}
        env_overrides: dict[Node, ast.AST] = {}
        advanced_indices: list[tuple[Node, _FragmentCoordinate]] = []
        if advanced:
            if not all(isinstance(index, Node) for index in indices):
                raise RuntimeError(
                    "mixed basic/advanced fragment loads are unsupported"
                )
            for index in indices:
                assert isinstance(index, Node)
                index_flat = broadcast_logical_flat_index(
                    flat,
                    output_shape=output_shape,
                    source_shape=_shape(self.state, index),
                )
                if index_flat is None:
                    raise RuntimeError(
                        f"unsupported fragment broadcast into {node.name}"
                    )
                advanced_indices.append((index, index_flat))
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
                            f"({base}) + "
                            f"({_coordinate_expression(output_coords[coord_dim])})"
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
                        f"({base}) + "
                        f"({_coordinate_expression(output_coords[output_dim])})"
                    )
                    output_dim += 1
                elif not isinstance(index, int):
                    raise RuntimeError(f"unsupported fragment load index {index!r}")
        for index, index_flat in advanced_indices:
            env_overrides[index] = self.evaluate(index, self._index(index_flat))
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
        return self._remember_terminal(
            terminal_key, self._bind("tcgen05_epi_load", expression)
        )

    def evaluate(
        self,
        node: Node,
        flat: _FragmentCoordinate,
        projection: int | None = None,
    ) -> ast.AST:
        key = (node, flat, projection)
        if key in self.memo:
            return self.memo[key]

        def leaf(
            current: Node,
            current_flat: _FragmentCoordinate,
            current_projection: int | None,
        ) -> ast.AST:
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
                    source_coords: list[_FragmentCoordinate] = []
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
                    return self.evaluate(source, self._index(source_flat))
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

        def select(
            selector: _FragmentCoordinate, choices: tuple[ast.AST, ...]
        ) -> ast.AST:
            if len(choices) != 2:
                raise RuntimeError("invalid fragment coordinate selection")
            semantic = _coordinate_semantic(selector)
            if semantic is not None and semantic.is_Integer:
                selected = int(semantic)
                if selected not in (0, 1):
                    raise RuntimeError("fragment coordinate selected an invalid input")
                return choices[selected]
            return self._bind(
                "tcgen05_epi_join",
                expr_from_string(
                    "({left} if ({selector}) == cutlass.Int32(0) else {right})",
                    selector=expr_from_string(_coordinate_expression(selector)),
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


def _identity_scalar_fragment(
    state: CodegenState,
    store: Tcgen05EpilogueStorePlan,
    *,
    carrier_name: str,
    coordinate_name: str,
    target_dtype: str,
    indent: str,
) -> FragmentEpilogueCode:
    """Render a scalar fragment whose accumulator reads stay register-local.

    Reached when the plan proved ``boundary_identity`` but at least one load
    could not be staged output-aligned, so the tile is walked one register at a
    time to rematerialize that load at its own coordinate.
    """
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
        index_name=index,
        body_indent=body_indent,
    )
    value = evaluator.evaluate(store.value_node, evaluator._index(matrix_flat))
    prelude = (
        f"{indent}{output} = cute.make_rmem_tensor("
        f"cute.make_layout({carrier_name}.shape), {target_dtype})\n"
        f"{indent}for {index} in range(cute.size({carrier_name}.shape)):\n"
        f"{body_indent}{coord} = {coordinate_name}[{index}]\n"
        f"{''.join(evaluator.lines)}"
        f"{body_indent}{output}[{index}] = {target_dtype}({ast.unparse(value)})\n"
    )
    return FragmentEpilogueCode(prelude, f"{output}.load()")


def _pair_scalar_fragment(
    state: CodegenState,
    store: Tcgen05EpilogueStorePlan,
    *,
    carrier_name: str,
    coordinate_name: str,
    target_dtype: str,
    indent: str,
) -> FragmentEpilogueCode:
    capability = store.pair_local
    if (
        capability is None
        or capability.width != 2
        or capability.traversal is not Tcgen05PairTraversal.CONTIGUOUS_EVEN_ODD_N_R2S
    ):
        raise RuntimeError("pair fragment requires contiguous even/odd R2S traversal")
    df = state.device_function
    output = df.new_var("tcgen05_epi_output")
    index = df.new_var("tcgen05_epi_index")
    partner_index = df.new_var("tcgen05_epi_partner_index")
    coord = df.new_var("tcgen05_epi_coord")
    body_indent = indent + "    "
    output_shape = _shape(state, store.value_node)
    if len(output_shape) not in (2, 3):
        raise RuntimeError("tcgen05 pair fragment output must be rank 2 or 3")
    if len(output_shape) == 3 and output_shape[0] != 1:
        raise RuntimeError("tcgen05 pair fragment requires one leading tile")
    matrix_shape = output_shape[-2:]
    # Same construction the planner proved the epilogue against; see
    # cute_reshape.pair_local_target_coordinates.
    targets = pair_local_target_coordinates(
        matrix_shape,
        width=capability.width,
        row_expression=f"{coord}[0]",
        column_expression=f"{coord}[1]",
    )
    if targets is None or len(targets) != capability.width:
        raise RuntimeError("tcgen05 pair fragment requires complete register pairs")
    current_coordinates, partner_coordinates = targets
    current_flat = logical_flat_from_coords(current_coordinates, matrix_shape)
    partner_flat = logical_flat_from_coords(partner_coordinates, matrix_shape)
    if not isinstance(current_flat, CuteLogicalCoordinate) or not isinstance(
        partner_flat, CuteLogicalCoordinate
    ):
        raise RuntimeError("pair fragment lost typed coordinate provenance")

    terminal_cache: dict[_TerminalKey, ast.AST] = {}
    evaluator = _ScalarEvaluator(
        state,
        store,
        carrier_name=carrier_name,
        index_name=index,
        body_indent=body_indent,
        pair_slot=_PairSlot(
            index,
            partner_index,
            current_coordinates,
            partner_coordinates,
        ),
        terminal_cache=terminal_cache,
    )
    partner_evaluator = _ScalarEvaluator(
        state,
        store,
        carrier_name=carrier_name,
        index_name=partner_index,
        body_indent=body_indent,
        pair_slot=_PairSlot(
            partner_index,
            index,
            partner_coordinates,
            current_coordinates,
        ),
        terminal_cache=terminal_cache,
    )
    value = evaluator.evaluate(store.value_node, evaluator._index(current_flat))
    partner_value = partner_evaluator.evaluate(
        store.value_node, partner_evaluator._index(partner_flat)
    )
    # ``CONTIGUOUS_EVEN_ODD_N_R2S`` -- register ``2k`` holds an even N and
    # register ``2k + 1`` its odd partner -- is established by the MMA-side
    # config gate and covered by the pair-swap layout tests over that gate's
    # envelope. It is deliberately not re-checked per iteration: the loop is
    # unrolled over the whole fragment, so a device assertion here would cost
    # one comparison per register in every gated kernel.
    prelude = (
        f"{indent}assert cute.size({carrier_name}.shape) % "
        f'{capability.width} == 0, "tcgen05 pair carrier must be complete"\n'
        f"{indent}{output} = cute.make_rmem_tensor("
        f"cute.make_layout({carrier_name}.shape), {target_dtype})\n"
        f"{indent}for {index} in range(0, cute.size({carrier_name}.shape), "
        f"{capability.width}):\n"
        f"{body_indent}{partner_index} = {index} + 1\n"
        f"{body_indent}{coord} = {coordinate_name}[{index}]\n"
        f"{''.join(evaluator.lines)}"
        f"{body_indent}{output}[{index}] = {target_dtype}({ast.unparse(value)})\n"
        f"{''.join(partner_evaluator.lines)}"
        f"{body_indent}{output}[{partner_index}] = "
        f"{target_dtype}({ast.unparse(partner_value)})\n"
    )
    return FragmentEpilogueCode(prelude, f"{output}.load()")


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
    if store.pair_local is not None:
        return _pair_scalar_fragment(
            state,
            store,
            carrier_name=carrier_name,
            coordinate_name=coordinate_name,
            target_dtype=target_dtype,
            indent=indent,
        )
    return _identity_scalar_fragment(
        state,
        store,
        carrier_name=carrier_name,
        coordinate_name=coordinate_name,
        target_dtype=target_dtype,
        indent=indent,
    )


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

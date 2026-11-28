"""Read a pure row accumulator at the coordinates of its final reduction."""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch.fx import Node

from ... import exc
from ...language import _tracing_ops
from ...language import creation_ops
from ...language import memory_ops
from ...language import view_ops
from ..ast_extension import statement_from_string
from ..device_ir import ForLoopGraphInfo
from ..device_ir import control_flow_child_graph_ids
from .fragment_context import Fragment
from .fragment_vector_io import static_int
from .register_loads import has_opaque_inline_asm
from .register_loads import host_load_is_readonly
from .row_carry_terms import carry_tile
from .row_carry_terms import evaluate_terms
from .warp_row_carry import warp_row_carries

if TYPE_CHECKING:
    import ast

    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from .computed_fragment import FragmentCompiler


@dataclass(frozen=True)
class RowCarry:
    slot: int
    nodes: tuple[Node, ...]
    term: Node
    reduction: Node

    def after_last_tile(self, compiler: FragmentCompiler, value: Fragment) -> Fragment:
        operand = cast("Node", self.reduction.args[0])
        if operand.target is _tracing_ops._phi:
            return value
        return cast(
            "Fragment", compiler.node(operand, {cast("Node", operand.args[0]): value})
        )


@dataclass(frozen=True)
class RowLoop:
    block_id: int
    begin: int
    end: int
    placeholders: tuple[Node, ...]
    results: tuple[RowCarry, ...]
    unmasked_rows: tuple[int, int] | None = None


@dataclass(frozen=True)
class RowLoopStructure:
    """A proven row-carry body; a concrete interval is required to lower it."""

    block_id: int
    begin: object
    end: object
    placeholders: tuple[Node, ...]
    results: tuple[RowCarry, ...]
    implicit_row_axis: int | None
    native_terms: bool

    def bounded(self, env: CompileEnvironment) -> RowLoop | None:
        begin, end = self.begin, self.end
        if (
            not self.native_terms
            or type(begin) is not int
            or type(end) is not int
            or not 0 <= begin < end < 2**31
        ):
            return None
        unmasked_rows = None
        if (row := self.implicit_row_axis) is not None:
            rows = static_int(env.block_sizes[row].size)
            if rows is None or rows <= 0:
                return None
            unmasked_rows = (row, rows)
        return RowLoop(
            self.block_id, begin, end, self.placeholders, self.results, unmasked_rows
        )


def row_value_before_casts(value: object) -> object:
    # A conversion can disappear with a different bound dtype. Only the
    # schema may look through it; the lowering reads the original FX nodes.
    while (
        isinstance(value, Node)
        and value.target is torch.ops.prims.convert_element_type.default
    ):
        value = value.args[0]
    return value


def phase_region(graphs: list[GraphInfo], root_id: int) -> list[GraphInfo]:
    """Include both arms of a branch, even when codegen visits only one."""
    infos = {g.graph_id: g for g in graphs}
    todo = [root_id]
    seen: set[int] = set()
    while todo:
        gid = todo.pop()
        if gid in seen:
            continue
        seen.add(gid)
        todo.extend(
            child
            for node in infos[gid].graph.nodes
            for child in control_flow_child_graph_ids(node)
        )
    return [g for g in graphs if g.graph_id in seen]


def row_loop_structure(
    env: CompileEnvironment, call: Node, region: list[GraphInfo]
) -> RowLoopStructure | None:
    # Every result must be an independent FP32 accumulation. A use of the
    # matrix before the row sum would require materializing it as before.
    if call.target is not _tracing_ops._for_loop or len(call.args) != 4:
        return None
    gid, starts, ends, captures = call.args
    inner = next((g for g in region if g.graph_id == gid), None)
    if (
        not isinstance(inner, ForLoopGraphInfo)
        or len(inner.block_ids) != 1
        or not isinstance(starts, (tuple, list))
        or not isinstance(ends, (tuple, list))
        or len(starts) != 1
        or len(ends) != 1
        or not isinstance(captures, (tuple, list))
    ):
        return None
    (axis,) = inner.block_ids
    placeholders = tuple(inner.graph.find_nodes(op="placeholder"))
    if len(placeholders) != len(captures):
        return None
    outputs = inner.graph.find_nodes(op="output")[0].args[0]
    if (
        not isinstance(outputs, (tuple, list))
        or not outputs
        or len(outputs) != len(call.users)
    ):
        return None
    implicit = False
    native_terms = True
    row_axes: set[int | None] = set()
    slots: dict[int, int] = {}
    reductions: dict[int, Node] = {}
    for getter in call.users:
        if (
            getter.target is not operator.getitem
            or type(output_i := getter.args[1]) is not int
            or not 0 <= output_i < len(outputs)
            or output_i in slots
            or len(getter.users) != 1
        ):
            return None
        (phi,) = getter.users
        if (
            phi.target is not _tracing_ops._phi
            or phi.args[1] is not getter
            or len(phi.users) != 1
        ):
            return None
        initial = phi.args[0]
        if (
            not isinstance(initial, Node)
            or initial not in captures
            or initial.target is not creation_ops.full
            or initial.args[1:3] != (0, torch.float32)
        ):
            return None
        val = initial.meta.get("val")
        if not isinstance(val, torch.Tensor) or val.ndim != 2:
            return None
        row_axis = env.resolve_block_id(val.shape[0])
        if (
            row_axis is None
            or row_axis == axis
            or env.resolve_block_id(val.shape[1]) != axis
        ):
            # Both emitters reduce trailing K. A masked [K, row] carry does
            # not own a row projection and must not suppress a valid sibling.
            return None
        row_axes.add(row_axis)
        (user,) = phi.users
        if user.target is _tracing_ops._mask_to:
            if user.args != (phi, 0) or user.kwargs or len(user.users) != 1:
                return None
            (reduction,) = user.users
            operand = user
        else:
            # Fast math may prove the padded value is zero and omit the mask.
            implicit = True
            operand, reduction = phi, user
        if (
            reduction.target is not torch.ops.aten.sum.dim_IntList
            or len(reduction.args) != 2
            or reduction.args[0] is not operand
            or reduction.kwargs
            or reduction.args[1] not in ([-1], [1], (-1,), (1,))
        ):
            return None
        slots[output_i] = captures.index(initial)
        reductions[output_i] = reduction
    if len(set(slots.values())) != len(slots):
        return None
    allowed = {
        _tracing_ops._host_tensor,
        _tracing_ops._get_symnode,
        _tracing_ops._new_var,
        _tracing_ops._mask_select,
        torch.ops.aten.sym_size.int,
        torch.ops.prims.convert_element_type.default,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.add.Tensor,
        view_ops.subscript,
        memory_ops.load,
    }
    for node in inner.graph.nodes:
        if node.op == "call_function" and (
            node.target not in allowed
            or node.target is memory_ops.load
            and node.args[2:] != (None, None)
        ):
            return None
    accumulators = {placeholders[i] for i in slots.values()}
    results = []
    for i, output in enumerate(outputs):
        slot = slots[i]
        if (
            not isinstance(output, Node)
            or output.target is not torch.ops.aten.add.Tensor
            or len(output.args) != 2
            or output.kwargs
        ):
            return None
        previous, term = output.args
        if (
            not isinstance(previous, Node)
            or previous.target is not _tracing_ops._new_var
            or previous.args != (placeholders[slot],)
        ):
            return None
        if not isinstance(term, Node):
            return None
        # Both emitters read the term using the carry's (row, K)
        # coordinates. A zero mask preserves its operand rank, and an add
        # may implicitly broadcast a rank-one term into a rank-two carry.
        initial = cast("Node", captures[slot])
        acc_type = initial.meta.get("val")
        term_type = term.meta.get("val")
        if (
            not isinstance(term_type, torch.Tensor)
            or not isinstance(acc_type, torch.Tensor)
            or term_type.ndim != 2
            or acc_type.ndim != 2
            or any(
                env.resolve_block_id(x) is None
                or env.resolve_block_id(x) != env.resolve_block_id(y)
                for x, y in zip(term_type.shape, acc_type.shape, strict=True)
            )
        ):
            return None
        shape_term = row_value_before_casts(term)
        if not isinstance(shape_term, Node):
            return None
        # The schema can see a product through a cast. The native carry
        # still needs the original FP32 addend in its supported form.
        native_terms &= term_type.dtype == torch.float32 and shape_term is term
        if shape_term.target is _tracing_ops._mask_select:
            if (
                len(shape_term.args) != 3
                or shape_term.kwargs
                or shape_term.args[1] != 0
                or shape_term.args[2] not in ([axis], (axis,))
            ):
                return None
        elif shape_term.target is torch.ops.aten.mul.Tensor:
            implicit = True
        else:
            return None
        needed: set[Node] = set()
        todo = [term]
        while todo:
            n = todo.pop()
            if n in needed:
                continue
            needed.add(n)
            if n in accumulators:
                return None
            # sym_size observes a shape, never the value of the carry.
            if n.target is not torch.ops.aten.sym_size.int:
                todo.extend(n.all_input_nodes)
        nodes = tuple(
            n for n in inner.graph.nodes if n.op == "call_function" and n in needed
        )
        results.append(RowCarry(slot, nodes, term, reductions[i]))
    row = None
    if implicit:
        if not env.settings.fast_math or len(row_axes) != 1:
            return None
        (row,) = row_axes
        if row is None or row == axis:
            return None
    return RowLoopStructure(
        axis, starts[0], ends[0], placeholders, tuple(results), row, native_terms
    )


def row_loop_plan(
    env: CompileEnvironment, call: Node, region: list[GraphInfo]
) -> RowLoop | None:
    structure = row_loop_structure(env, call, region)
    return structure.bounded(env) if structure is not None else None


def row_phase_plans(
    env: CompileEnvironment, region: list[GraphInfo]
) -> list[tuple[Node, RowLoop]]:
    return [
        (n, plan)
        for g in region
        for n in g.graph.nodes
        if (plan := row_loop_plan(env, n, region)) is not None
    ]


def row_phase_reads_are_safe(
    env: CompileEnvironment, region: list[GraphInfo], *, allow_unbound: bool = False
) -> bool:
    return not has_opaque_inline_asm(region) and all(
        host_load_is_readonly(n, env, region, allow_unbound=allow_unbound)
        for g in region
        for n in g.graph.nodes
        if n.target is memory_ops.load
    )


def row_phase_supported(env: CompileEnvironment, region: list[GraphInfo]) -> bool:
    # Codegen rechecks the final projected graph and its bound buffers.
    return bool(row_phase_plans(env, region)) and row_phase_reads_are_safe(env, region)


def folded_results(
    c: FragmentCompiler, call: Node, existing: dict[Node, object]
) -> list[Fragment] | None:
    root = c.cg.current_root_graph_info
    if root is None:
        return None
    region = phase_region(c.graphs, root.graph_id)
    plan = row_loop_plan(c.env, call, region)
    if plan is None:
        return None
    axis = plan.block_id
    if axis in c.offsets:
        # A single block ID cannot distinguish outer and K coordinates,
        # including the lanes used by the reduction's final mask.
        if c.checked_root:
            return None  # Retain the original device-assertion diagnostic.
        raise exc.NestedDeviceLoopsConflict
    tile = c.df.resolved_block_size(axis)
    if plan.unmasked_rows is not None:
        row, rows = plan.unmasked_rows
        row_tile = c.df.resolved_block_size(row)
        if row_tile is None or row_tile <= 0 or rows % row_tile:
            # Other row contractions in this root can still use the warp
            # path. Lower this partial one through the original loop.
            return None
    if tile is None or tile <= 0 or (plan.end - plan.begin) % tile:
        raise exc.InvalidConfig("row carry needs whole contraction tiles")
    caps = cast("list[Node]", call.args[3])
    mapped = {
        ph: existing[cap] for ph, cap in zip(plan.placeholders, caps, strict=True)
    }
    states = [mapped[plan.placeholders[item.slot]] for item in plan.results]
    if any(
        not isinstance(s, Fragment) or len(s.shape) != 2 or s.shape[-1] != tile
        for s in states
    ):
        raise exc.InvalidConfig("row carry needs a trailing contraction tile")
    if (
        isinstance(tile, int)
        and c.df.config.get("cute_fragment_reduction", "serial") == "warp"
    ):
        collective = warp_row_carries(
            c, region, plan, mapped, [cast("Fragment", s) for s in states], tile
        )
        if collective is not None:
            return collective
    dependencies = tuple(v for v in mapped.values() if isinstance(v, Fragment))

    def result(item: RowCarry, state: Fragment) -> Fragment:
        def evaluate(coords: tuple[str, ...]) -> str:
            acc = c.df.new_var("fragment_row_carry")
            step = c.df.new_var("fragment_row_tile")
            c.emit(f"{acc} = cutlass.Float32(0)")
            body: list[ast.AST] = []
            with carry_tile(c, plan, step, tile), c.cg.set_statements(body):
                values = evaluate_terms(c, mapped, item.nodes, region)
                value = cast("Fragment", values[item.term]).read(coords)
                c.emit(f"{acc} = {acc} + cutlass.Float32({value})")
            loop = cast(
                "ast.For",
                statement_from_string(
                    f"for {step} in range({plan.begin}, {plan.end}, {tile}):\n    pass"
                ),
            )
            loop.body = cast("list[ast.stmt]", body)
            c.cg.add_statement(loop)
            return acc

        return Fragment(
            state.shape,
            state.dtype,
            evaluate,
            dependencies=dependencies,
            logical_domain=state.logical_domain,
        )

    return [
        result(item, cast("Fragment", s))
        for item, s in zip(plan.results, states, strict=True)
    ]

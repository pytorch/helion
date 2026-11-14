"""Prove seeded entry and the final O/LSE envelope around a page recurrence.

This certificate does not admit an emitter. The page arithmetic, readonly
accesses, output ownership and final projected graphs need independent proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import operator
from typing import TYPE_CHECKING
from typing import cast

import sympy
import torch
from torch.fx import Node
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.functions import PythonMod

from ....language import _tracing_ops as tracing
from ....language import creation_ops
from ....language import memory_ops
from ....language import tile_ops
from ....language import view_ops
from ...device_ir import ForLoopGraphInfo
from ...variable_origin import GridOrigin
from ...variable_origin import TileIdOrigin
from .conditional_denominator import require_denominator_tree
from .conditional_regions import ConditionalRegions
from .conditional_regions import prove_conditional_regions
from .match_common import symnode_sympy

if TYPE_CHECKING:
    from ...compile_environment import CompileEnvironment
    from ...device_ir import DeviceIR


@dataclass(frozen=True)
class ConditionalContinuation:
    regions: ConditionalRegions
    loop: ForLoopGraphInfo
    seed: Node
    rescale_seed: Node
    rescale_branch: Node
    initial_states: tuple[Node, ...]
    loop_outputs: tuple[Node, ...]
    imported_reads: tuple[Node, ...]
    denominator: Node
    output_store: Node
    lse_store: Node
    empty: Node
    proof_nodes: frozenset[Node]


class _Refused(Exception):
    pass


def _require(condition: bool) -> None:
    if not condition:
        raise _Refused


class _Envelope:
    def __init__(
        self, ir: DeviceIR, regions: ConditionalRegions, env: CompileEnvironment
    ) -> None:
        self.ir = ir
        self.env = env
        self.regions = regions
        self.captures = {
            capture.placeholder: capture.source
            for region in regions.regions
            for capture in region.captures
        }
        self.nodes: set[Node] = set()
        self.integer_active: set[Node] = set()

    def resolve(self, value: object) -> Node:
        if not isinstance(value, Node):
            raise _Refused
        seen = set()
        while value in self.captures or value.target is tracing._new_var:
            _require(value not in seen)
            seen.add(value)
            self.nodes.add(value)
            if value in self.captures:
                value = self.captures[value]
            else:
                _require(len(value.args) == 1 and not value.kwargs)
                value = value.args[0]
            if not isinstance(value, Node):
                raise _Refused
        self.nodes.add(value)
        return value

    def call(self, value: object, target: object, count: int) -> tuple:
        node = self.resolve(value)
        _require(node.target is target and len(node.args) == count and not node.kwargs)
        return node.args

    def same(self, left: object, right: object) -> None:
        _require(self.resolve(left) is self.resolve(right))

    def shape(self, value: object, dtype: torch.dtype) -> tuple[int, ...]:
        fake = self.resolve(value).meta.get("val")
        if not isinstance(fake, torch.Tensor) or fake.dtype != dtype:
            raise _Refused
        _require(all(type(n) is int for n in fake.shape))
        return cast("tuple[int, ...]", tuple(fake.shape))

    def scalar(self, value: object, expected: float) -> None:
        node = self.resolve(value)
        _require(node.target is torch.ops.aten.scalar_tensor.default)
        _require(node.args == (expected,) and self.shape(node, torch.float32) == ())
        if expected == 0:
            actual = node.args[0]
            if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                raise _Refused
            _require(math.copysign(1, actual) == 1)

    def expanded(self, value: object, column: bool) -> Node:
        source, indices = self.call(value, view_ops.subscript, 2)
        _require(indices == ([slice(None), None] if column else [None, slice(None)]))
        return self.resolve(source)

    def integer(self, value: object) -> sympy.Expr:
        if type(value) is int:
            return sympy.Integer(value)
        node = self.resolve(value)
        if node.target is tracing._get_symnode:
            expression = symnode_sympy(node)
            # A declared integer config value is uniform. Keep its symbol in
            # the identity proof so entry and rescale must agree for every
            # setting; never replace it with a default or size hint.
            if (
                isinstance(node.meta.get("val"), torch.SymInt)
                and expression is not None
                and expression.free_symbols
                and expression.free_symbols <= self.env.tunable_symbols
                and all(
                    symbol in self.env.config_value_expressions
                    for symbol in expression.free_symbols
                )
            ):
                return expression
            raise _Refused
        if node.target is tile_ops.tile_id:
            expression = symnode_sympy(node)
            host = self.ir.host_function
            if host is None or expression is None:
                raise _Refused
            origin = host.expr_to_origin.get(expression)
            if origin is None or not isinstance(origin.origin, TileIdOrigin):
                raise _Refused
            _require(
                origin.origin.block_id
                in self.ir.grid_block_ids[self.regions.root.phase_index]
            )
            return expression
        _require(len(node.args) == 2 and not node.kwargs)
        _require(node not in self.integer_active)
        self.integer_active.add(node)
        try:
            left, right = (self.integer(x) for x in node.args)
            if node.target is operator.add:
                result = sympy.Add(left, right)
            elif node.target is operator.sub:
                result = sympy.Add(left, sympy.Mul(-1, right))
            elif node.target is operator.mul:
                result = sympy.Mul(left, right)
            elif node.target is operator.floordiv:
                result = FloorDiv(left, right)
            elif node.target is operator.mod:
                result = PythonMod(left, right)
            else:
                raise _Refused
            return sympy.expand(result)
        finally:
            self.integer_active.remove(node)

    def denominator(self, value: Node, lanes: Node, rows: int) -> None:
        require_denominator_tree(
            value, lanes, rows, call=self.call, same=self.same, require=_require
        )

    def prove(self) -> ConditionalContinuation:
        loops = [
            r.graph
            for r in self.regions.regions
            if isinstance(r.graph, ForLoopGraphInfo)
        ]
        _require(len(loops) == 1)
        loop = loops[0]
        carries = sorted(self.regions.carries, key=lambda item: item.slot)
        _require(len(carries) == 3 and [c.slot for c in carries] == [0, 1, 2])
        _require(all(c.call is carries[0].call for c in carries))
        loop_id = carries[0].call.args[0]
        _require(type(loop_id) is int and loop_id == loop.graph_id)
        initial = tuple(c.initial for c in carries)
        output = tuple(c.phi for c in carries)
        shape = self.shape(initial[0], torch.float32)
        _require(len(shape) == 1)
        rows = shape[0]
        _require(self.shape(initial[1], torch.float32) == (rows, 128))
        _require(self.shape(initial[2], torch.float32) == (rows, 128))
        joins = {b.phi: b for b in self.regions.branches}
        _require(all(node in joins for node in initial))
        entry = [joins[node] for node in initial]
        _require(all(b.call is entry[0].call for b in entry))
        _require([b.slot for b in entry] == [0, 1, 2])
        for b, size, fill in zip(
            entry,
            ([rows], [rows, 128], [rows, 128]),
            (-math.inf, 0.0, 0.0),
            strict=True,
        ):
            sizes, value, dtype, layout = self.call(b.when_false, creation_ops.full, 4)
            _require(
                list(sizes) == size
                and value == fill
                and dtype is torch.float32
                and layout is None
            )
            if fill == 0:
                _require(math.copysign(1, value) == 1)
        maximum, accumulator, denominator_lanes = [
            self.resolve(b.when_true) for b in entry
        ]
        m_host, m_index, mask, other = self.call(maximum, memory_ops.load, 4)
        _require(
            mask is None and other is None and len(m_index) == 3 and m_index[2] == 0
        )
        nonempty, imported_o, zero = self.call(
            accumulator, torch.ops.aten.where.self, 3
        )
        nonempty = self.expanded(nonempty, True)
        tested, negative_inf = self.call(nonempty, torch.ops.aten.ne.Scalar, 2)
        self.same(tested, maximum)
        _require(negative_inf == -math.inf)
        self.scalar(zero, 0.0)
        lane_mask, imported_l, zero = self.call(
            denominator_lanes, torch.ops.aten.where.self, 3
        )
        self.scalar(zero, 0.0)
        lane_zero, lane_nonempty = self.call(
            lane_mask, torch.ops.aten.bitwise_and.Tensor, 2
        )
        self.same(self.expanded(lane_nonempty, True), nonempty)
        lane, zero = self.call(lane_zero, torch.ops.aten.eq.Scalar, 2)
        _require(type(zero) is int and zero == 0)
        lane = self.expanded(lane, False)
        _require(lane.target is torch.ops.prims.iota.default and lane.args == (128,))
        _require(type(lane.kwargs.get("start")) is int and lane.kwargs["start"] == 0)
        _require(type(lane.kwargs.get("step")) is int and lane.kwargs["step"] == 1)
        _require(self.shape(lane, torch.int32) == (128,))
        imported_l = self.expanded(imported_l, True)
        l_host, l_index, mask, other = self.call(imported_l, memory_ops.load, 4)
        _require(mask is None and other is None and l_host is m_host)
        _require(len(l_index) == 3 and l_index[:2] == m_index[:2] and l_index[2] == 1)
        _, o_index, mask, other = self.call(imported_o, memory_ops.load, 4)
        _require(mask is None and other is None and len(o_index) == 1)
        _require(self.shape(maximum, torch.float32) == (rows,))
        _require(self.shape(imported_l, torch.float32) == (rows,))
        _require(self.shape(imported_o, torch.float32) == (rows, 128))
        seed = self.resolve(entry[0].call.args[0])
        left, right = self.call(seed, operator.ne, 2)
        seed_expression = sympy.Ne(self.integer(left), self.integer(right))
        candidates = {
            b.call
            for b in self.regions.branches
            if self.resolve(b.call.args[0]).target is operator.or_
        }
        _require(len(candidates) == 1)
        rescale = candidates.pop()
        iteration_test, rescale_seed = self.call(rescale.args[0], operator.or_, 2)
        iteration, zero = self.call(iteration_test, operator.gt, 2)
        _require(type(zero) is int and zero == 0)
        iteration = self.resolve(iteration)
        _require(iteration.target is tracing._get_symnode)
        symbol = symnode_sympy(iteration)
        host = self.ir.host_function
        if host is None or symbol is None:
            raise _Refused
        origin = host.expr_to_origin.get(symbol)
        _require(
            origin is not None
            and isinstance(origin.origin, GridOrigin)
            and loop.block_ids == [origin.origin.block_id]
        )
        rescale_seed = self.resolve(rescale_seed)
        _require(
            rescale_seed.target is tracing._get_symnode and not rescale_seed.kwargs
        )
        # _get_symnode's argument is a debug label. Its lowering reads the
        # typed symbolic metadata, so compare that semantic value with the
        # expression reconstructed from the actual outer scalar operations.
        seeded = rescale_seed.meta.get("val")
        if not isinstance(seeded, torch.SymBool):
            raise _Refused
        # Tracing can retain factored integer products while the reconstructed
        # operations distribute them. Compare exact expanded identities,
        # including inside floor divisions, without evaluating any tunable.
        _require(sympy.expand(seeded._sympy_()) == sympy.expand(seed_expression))
        stores = [
            n
            for r in self.regions.regions
            for n in r.graph.graph.nodes
            if n.target is memory_ops.store
        ]
        _require(len(stores) == 2)
        out_store, lse_store = stores
        for store in stores:
            _require(
                len(store.args) == 4 and store.args[3] is None and not store.kwargs
            )
        selected, dtype = self.call(
            out_store.args[2], torch.ops.prims.convert_element_type.default, 2
        )
        _require(dtype is torch.bfloat16)
        empty, zero, normalized = self.call(selected, torch.ops.aten.where.self, 3)
        empty = self.expanded(empty, True)
        self.scalar(zero, 0.0)
        query_position, zero = self.call(empty, torch.ops.aten.lt.Scalar, 2)
        _require(zero == 0 and self.shape(query_position, torch.int32) == (rows,))
        accumulated, scale = self.call(normalized, torch.ops.aten.mul.Tensor, 2)
        self.same(accumulated, output[1])
        scale, factor = self.call(
            self.expanded(scale, True), torch.ops.aten.mul.Tensor, 2
        )
        _require(type(factor) is int and factor == 64)
        reciprocal, factor = self.call(scale, torch.ops.aten.mul.Tensor, 2)
        _require(type(factor) is float and factor == 1.0)
        (denominator,) = self.call(reciprocal, torch.ops.aten.reciprocal.default, 1)
        denominator = self.resolve(denominator)
        self.denominator(denominator, output[2], rows)
        test, negative_inf, lse = self.call(
            lse_store.args[2], torch.ops.aten.where.self, 3
        )
        self.same(test, empty)
        self.scalar(negative_inf, -math.inf)
        added, factor = self.call(lse, torch.ops.aten.mul.Tensor, 2)
        _require(type(factor) is float and factor == math.log(2))
        maximum, logarithm = self.call(added, torch.ops.aten.add.Tensor, 2)
        self.same(maximum, output[0])
        (logged,) = self.call(logarithm, torch.ops.aten.log2.default, 1)
        self.same(logged, denominator)
        return ConditionalContinuation(
            self.regions,
            loop,
            seed,
            rescale_seed,
            rescale,
            initial,
            output,
            (self.resolve(entry[0].when_true), imported_l, self.resolve(imported_o)),
            denominator,
            out_store,
            lse_store,
            empty,
            frozenset(self.nodes),
        )


def prove_conditional_continuation(
    ir: DeviceIR, env: CompileEnvironment
) -> ConditionalContinuation | None:
    if len(ir.root_ids) != 1:
        return None
    regions = prove_conditional_regions(ir.graphs, ir.root_ids[0])
    if regions is None:
        return None
    try:
        return _Envelope(ir, regions, env).prove()
    except _Refused:
        return None

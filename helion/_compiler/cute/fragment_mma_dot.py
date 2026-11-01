"""Trace-time rewrite of a GEMV's scalar dot fallback onto ``mma.sync`` fragments.

``hl.dot(lhs[m, k], rhs[k, 1])`` takes the scalar fallback: every thread
converts the 16-bit elements of its packets to fp32 and runs one FMA per
element, then the K threads of a row combine through warp shuffles::

    for lane_1 in range(L):                                   # K lane loop
        lane_base_1 = ...
        _tile_unroll_vec_1_0 = <packet load>(w_ptr, ir.VectorType.get([16], cutlass.Uint16.mlir_type))
        _tile_unroll_vec_1_1 = <packet load>(x_ptr, ir.VectorType.get([16], cutlass.Uint16.mlir_type))
        for vec_lane_1 in cutlass.range_constexpr(16):
            gathered = cutlass.Uint16(_tile_unroll_vec_1_0[vec_lane_1]).bitcast(cutlass.Float16) if PRED else cutlass.Float16(0)
            load_1 = cutlass.Uint16(_tile_unroll_vec_1_1[vec_lane_1]).bitcast(cutlass.Float16)
            _helion_vfold_acc_0 = _helion_vfold_acc_0 + cutlass.Float32(cutlass.Float32(gathered) * cutlass.Float32(load_1))
    dot = _cute_grouped_reduce_warp(_helion_vfold_acc_0, 'sum', cutlass.Float32(0), <lane>, pre=8, group_span=32)

That is two converts, a select and an FMA per element (64 instructions per
32-byte packet) plus 40 shuffles per chunk.  A 128 MB-per-call GEMV is
bandwidth-bound at boost clocks and issue-bound once the sustained power cap
pulls the SM clock down (B200: 1965 -> ~900 MHz under a 256 MB stream;
gather_gemv (8, 8192, 2): memory-only body 46 us, the fallback 53).

With ``cute_dot_fragment_mma`` on a ``(8, 4, 1)`` thread block (eight row
threads on x, four K threads on y) the pass remaps the lanes so that the
hardware fragment layout of ``mma.sync.m16n8k16`` (row group ``lane // 4``,
K quarter ``lane % 4``) holds one row per group and the row's four K
packets across the group, and replaces the element loop by tensor-core
instructions::

    _fmma_a0 = cutlass.Uint32(_tile_unroll_vec_1_0[0]) if PRED else cutlass.Uint32(0)
    ...
    c0, c1, c2, c3 = _cute_mma_f16(
        _fmma_a0, 0, _fmma_a1, 0, _fmma_b0, _fmma_b1, c0, c1, c2, c3
    )
    ...  # one MMA per 4 words
    dot = c0

The A fragment's upper eight rows are zero (one row per thread), B holds the
vector packet in every lane, so every column of D is the row's dot and the
MMA also performs the cross-thread K reduction: 4 MMAs and 8 selects per
32-byte packet, no shuffles.  The logical-to-physical K permutation is the
same for A and B (both packets come from the same ``lane_base``), so the
contraction is exact: products are exact in fp32 and accumulated in fp32,
as Triton's ``tl.dot`` does for the same kernel.

The pass fires only on the complete pattern (two 16-bit packet loads, the
element loop made of the two extractions and the fp32 fold, the grouped sum
over the K threads right after the lane loop, no other cross-lane primitive
in the kernel) and reports whether it did so the caller can regenerate with
the knob off (``FragmentMmaNotApplied``): the knob alone never changes the
code.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from ...runtime.config import Config
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import ReadWrites
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES

if TYPE_CHECKING:
    from collections.abc import Callable

FRAGMENT_MMA_KEY = "cute_dot_fragment_mma"
# The thread block the pass accepts: eight row threads on x, four K threads
# on y; one warp whose lanes the pass permutes into the fragment layout.
FRAGMENT_MMA_BLOCK_DIMS = (8, 4, 1)
FRAGMENT_MMA_ROW_THREADS = 8
FRAGMENT_MMA_K_THREADS = 4
# 16-bit element type spelled in the extraction -> the MMA helper.
_MMA_HELPERS = {
    "cutlass.Float16": "_cute_mma_f16",
    "cutlass.BFloat16": "_cute_mma_bf16",
}
_GROUPED_REDUCE = "_cute_grouped_reduce_warp"
# Names of calls that communicate across lanes; a body holding one after the
# rewrite cannot have its lanes permuted.
_CROSS_LANE_MARKERS = (
    "warp_reduction",
    "grouped_reduce",
    "shuffle",
    "shfl",
    "sync_threads",
    "barrier",
    "smem",
    "cluster",
    "elect",
    "vote",
    "ballot",
    "cp_async",
    "tma",
)


class FragmentMmaNotApplied(exc.Base):
    """``cute_dot_fragment_mma`` asked for the fragment rewrite but no dot took it.

    The knob must not change the code by itself, so ``generate_ast``
    generates the kernel again with the knob off (``fragment_mma_off_config``).
    """


def fragment_mma_off_config(config: Config) -> Config:
    """``config`` with the fragment MMA off."""
    return Config.from_dict({**config.config, FRAGMENT_MMA_KEY: False})


def _is_lane_loop(stmt: ast.AST) -> bool:
    return isinstance(stmt, ast.For) and bool(
        getattr(stmt, HELION_LANE_LOOP_VAR_ATTR, None)
    )


def _constexpr_extent(loop: ast.For) -> int | None:
    """``for v in cutlass.range_constexpr(N)`` -> N."""
    it = loop.iter
    if (
        isinstance(it, ast.Call)
        and ast.unparse(it.func) == "cutlass.range_constexpr"
        and len(it.args) == 1
        and not it.keywords
        and isinstance(it.args[0], ast.Constant)
        and isinstance(it.args[0].value, int)
        and isinstance(loop.target, ast.Name)
        and not loop.orelse
    ):
        return it.args[0].value
    return None


def _plain_range_extent(loop: ast.For) -> int | None:
    """``for v in range(N)`` (the rolled lane loop) -> N."""
    it = loop.iter
    if (
        isinstance(it, ast.Call)
        and ast.unparse(it.func) == "range"
        and len(it.args) == 1
        and not it.keywords
        and isinstance(it.args[0], ast.Constant)
        and isinstance(it.args[0].value, int)
        and it.args[0].value >= 1
    ):
        return it.args[0].value
    return None


def _is_load_call(call: ast.Call) -> bool:
    name = ast.unparse(call.func)
    return name == "cute.arch.load" or name in _CUTE_CACHE_LOAD_HELPER_NAMES


@dataclasses.dataclass
class _Packet:
    """One 16-bit packet load of the lane body."""

    stmt: ast.Assign
    name: str
    vector_type: ast.Call  # ``ir.VectorType.get([V], cutlass.Uint16.mlir_type)``
    width: int


def _packet_load(stmt: ast.AST) -> _Packet | None:
    """``NAME = <load>(ptr, ir.VectorType.get([V], cutlass.Uint16.mlir_type))``."""
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Call)
        and _is_load_call(stmt.value)
        and len(stmt.value.args) == 2
        and not stmt.value.keywords
    ):
        return None
    vector_type = stmt.value.args[1]
    if not (
        isinstance(vector_type, ast.Call)
        and ast.unparse(vector_type.func) == "ir.VectorType.get"
        and len(vector_type.args) == 2
        and ast.unparse(vector_type.args[1]) == "cutlass.Uint16.mlir_type"
        and isinstance(vector_type.args[0], ast.List)
        and len(vector_type.args[0].elts) == 1
        and isinstance(vector_type.args[0].elts[0], ast.Constant)
    ):
        return None
    width = vector_type.args[0].elts[0].value
    if width not in (8, 16):
        return None
    return _Packet(stmt, stmt.targets[0].id, vector_type, width)


@dataclasses.dataclass
class _Extraction:
    """``NAME = cutlass.Uint16(PKT[VEC]).bitcast(ELEM) [if PRED else ELEM(0)]``."""

    name: str
    packet: str
    element: str
    predicate: ast.expr | None


def _extraction(stmt: ast.AST, vec_var: str) -> _Extraction | None:
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return None
    value = stmt.value
    predicate: ast.expr | None = None
    if isinstance(value, ast.IfExp):
        predicate = value.test
        zero = value.orelse
        value = value.body
        if not (
            isinstance(zero, ast.Call)
            and len(zero.args) == 1
            and isinstance(zero.args[0], ast.Constant)
            and zero.args[0].value == 0
            and not zero.keywords
        ):
            return None
        zero_type = ast.unparse(zero.func)
    else:
        zero_type = None
    # cutlass.Uint16(PKT[VEC]).bitcast(ELEM)
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "bitcast"
        and len(value.args) == 1
        and not value.keywords
    ):
        return None
    element = ast.unparse(value.args[0])
    if element not in _MMA_HELPERS or (zero_type is not None and zero_type != element):
        return None
    inner = value.func.value
    if not (
        isinstance(inner, ast.Call)
        and ast.unparse(inner.func) == "cutlass.Uint16"
        and len(inner.args) == 1
        and not inner.keywords
        and isinstance(inner.args[0], ast.Subscript)
        and isinstance(inner.args[0].value, ast.Name)
        and ast.unparse(inner.args[0].slice) == vec_var
    ):
        return None
    return _Extraction(stmt.targets[0].id, inner.args[0].value.id, element, predicate)


def _product_operands(product: ast.AST) -> tuple[str, str] | None:
    """``cutlass.Float32(cutlass.Float32(a) * cutlass.Float32(b))`` (the outer
    cast optional) -> ``(a, b)``."""
    if (
        isinstance(product, ast.Call)
        and ast.unparse(product.func) == "cutlass.Float32"
        and len(product.args) == 1
        and not product.keywords
    ):
        product = product.args[0]
    if not (isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)):
        return None
    operands = []
    for factor in (product.left, product.right):
        if not (
            isinstance(factor, ast.Call)
            and ast.unparse(factor.func) == "cutlass.Float32"
            and len(factor.args) == 1
            and isinstance(factor.args[0], ast.Name)
        ):
            return None
        operands.append(factor.args[0].id)
    return operands[0], operands[1]


def _fold_operands(
    stmt: ast.AST, definitions: dict[str, ast.expr]
) -> tuple[str, tuple[str, str]] | None:
    """``ACC = ACC + <product>`` where the product is spelled inline or is a
    name the element loop bound to it -> ``(ACC, (a, b))``."""
    if not (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.BinOp)
        and isinstance(stmt.value.op, ast.Add)
        and isinstance(stmt.value.left, ast.Name)
        and stmt.value.left.id == stmt.targets[0].id
    ):
        return None
    acc = stmt.targets[0].id
    product: ast.AST = stmt.value.right
    if isinstance(product, ast.Name) and product.id in definitions:
        product = definitions[product.id]
    operands = _product_operands(product)
    if operands is None:
        return None
    return acc, operands


def _is_pure_assignment(stmt: ast.AST) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and not any(
            isinstance(node, ast.Call)
            and (
                _is_load_call(node)
                or any(
                    marker in ast.unparse(node.func).lower()
                    for marker in _CROSS_LANE_MARKERS
                )
                or ast.unparse(node.func).endswith(".store")
            )
            for node in ast.walk(stmt.value)
        )
    )


@dataclasses.dataclass
class _Match:
    lane_loop: ast.For
    row_packet: _Packet
    vec_packet: _Packet
    row_predicate: ast.expr | None
    vec_predicate: ast.expr | None
    element: str
    vec_loop: ast.For
    acc: str
    init: ast.Assign  # ``ACC = Float32(0)``, in this or an enclosing list
    reduce_stmt: ast.stmt  # the statement holding the grouped sum
    reduce_call: ast.Call  # ``_cute_grouped_reduce_warp(ACC, 'sum', ...)``
    # The grouped sum closes the lane body (the running-sum route reduces
    # once per lane) rather than following the loop.
    reduce_in_body: bool
    # Trip count of the K lane loop: with the four K threads and the packet
    # width it is the K chunk the vector packet is staged for.
    lane_extent: int


def _grouped_sum_call(stmt: ast.AST, acc: str) -> ast.Call | None:
    """The one ``_cute_grouped_reduce_warp(ACC, 'sum', Float32(0), lane, pre=8,
    group_span=32)`` of an assignment (``DOT = <call>`` or
    ``acc = base + <call>``), or None."""
    if not isinstance(stmt, ast.Assign):
        return None
    calls = [
        node
        for node in ast.walk(stmt.value)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == _GROUPED_REDUCE
    ]
    if len(calls) != 1:
        return None
    call = calls[0]
    if not (
        len(call.args) == 4
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == acc
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "sum"
    ):
        return None
    keywords = {kw.arg: kw.value for kw in call.keywords}
    if set(keywords) != {"pre", "group_span"}:
        return None
    pre = keywords["pre"]
    span = keywords["group_span"]
    if not (
        isinstance(pre, ast.Constant)
        and pre.value == FRAGMENT_MMA_ROW_THREADS
        and isinstance(span, ast.Constant)
        and span.value == FRAGMENT_MMA_ROW_THREADS * FRAGMENT_MMA_K_THREADS
    ):
        return None
    return call


def _is_zero_init(stmt: ast.AST, acc: str) -> bool:
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == acc
        and ast.unparse(stmt.value) in ("cutlass.Float32(0)", "cutlass.Float32(0.0)")
    )


def _find_zero_init(
    statements: list[ast.stmt], index: int, stack: list[list[ast.stmt]], acc: str
) -> ast.Assign | None:
    """The zero init of ``acc`` that dominates ``statements[index]``: the last
    assignment to it before the lane loop in its own list, else in the
    enclosing lists (the running sum of a K-chunked dot is zeroed once before
    the chunk loop and reduced inside it)."""
    for scope in (statements[:index], *reversed(stack)):
        for stmt in reversed(scope):
            if isinstance(stmt, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == acc
                for target in stmt.targets
            ):
                return stmt if _is_zero_init(stmt, acc) else None
    return None


def _mentions(stmt: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id == name for node in ast.walk(stmt)
    )


def _simple_statements(statements: list[ast.stmt]) -> list[ast.stmt]:
    """Every non-compound statement of a body, in order."""
    result: list[ast.stmt] = []
    for stmt in statements:
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With)):
            for field in ("body", "orelse"):
                result.extend(_simple_statements(getattr(stmt, field, []) or []))
        else:
            result.append(stmt)
    return result


def _element_bound(
    name: str,
    vec_definitions: dict[str, ast.expr],
    vec_var: str,
    width: int,
) -> tuple[str, int] | None:
    """``NAME = IDX < K`` with ``IDX = LANE_BASE + cutlass.Int32(VEC)`` (the
    element loop's K bound) -> ``(LANE_BASE, K)`` when ``K`` is a whole number
    of packets, so the bound holds for all or none of a packet's elements and
    ``LANE_BASE < K`` decides it at packet level; None for any other shape."""
    bound = vec_definitions.get(name)
    if not (
        isinstance(bound, ast.Compare)
        and len(bound.ops) == 1
        and isinstance(bound.ops[0], ast.Lt)
        and isinstance(bound.left, ast.Name)
        and len(bound.comparators) == 1
        and isinstance(bound.comparators[0], ast.Constant)
        and isinstance(bound.comparators[0].value, int)
    ):
        return None
    index = vec_definitions.get(bound.left.id)
    if not (
        isinstance(index, ast.BinOp)
        and isinstance(index.op, ast.Add)
        and isinstance(index.left, ast.Name)
        and ast.unparse(index.right) == f"cutlass.Int32({vec_var})"
    ):
        return None
    extent = bound.comparators[0].value
    if extent % width:
        return None
    return index.left.id, extent


def _packet_predicate(
    predicate: ast.expr | None,
    vec_definitions: dict[str, ast.expr],
    vec_var: str,
    width: int,
) -> tuple[ast.expr | None, bool]:
    """The extraction predicate lifted to the packet: every name the element
    loop defines must be its K bound, replaced by the lane-base bound.
    ``(predicate, ok)``; ``ok`` False when an element-level name has another
    shape (the rewrite then declines)."""
    if predicate is None:
        return None, True
    reads = set(ReadWrites.from_ast(predicate).reads)
    if vec_var in reads:
        return None, False
    substitutions: dict[str, str] = {}
    for name in reads:
        if name not in vec_definitions:
            continue
        bound = _element_bound(name, vec_definitions, vec_var, width)
        if bound is None:
            return None, False
        lane_base, extent = bound
        substitutions[name] = f"{lane_base} < {extent}"
    lifted = cast("ast.expr", expr_from_string(ast.unparse(predicate)))
    for name, text in substitutions.items():
        lifted = cast("ast.expr", _NameSubstitution(name, f"({text})").visit(lifted))
    return lifted, True


def _match_lane_loop(
    statements: list[ast.stmt],
    index: int,
    stack: list[list[ast.stmt]],
    root: list[ast.stmt],
) -> _Match | None:
    row_names = _row_prologue_names(stack)
    loop = statements[index]
    if not _is_lane_loop(loop):
        return None
    assert isinstance(loop, ast.For)
    lane_extent = _plain_range_extent(loop)
    if lane_extent is None:
        return None
    packets: list[_Packet] = []
    vec_loop: ast.For | None = None
    vec_position = -1
    others: list[tuple[int, ast.stmt]] = []
    for position, stmt in enumerate(loop.body):
        packet = _packet_load(stmt)
        if packet is not None:
            packets.append(packet)
            continue
        if isinstance(stmt, ast.For) and _constexpr_extent(stmt) is not None:
            if vec_loop is not None:
                return None
            vec_loop = stmt
            vec_position = position
            continue
        others.append((position, stmt))
    if vec_loop is None or len(packets) != 2:
        return None
    width = _constexpr_extent(vec_loop)
    if width is None or any(packet.width != width for packet in packets):
        return None
    assert isinstance(vec_loop.target, ast.Name)
    vec_var = vec_loop.target.id
    extractions: dict[str, _Extraction] = {}
    fold: tuple[str, str] | None = None
    acc: str | None = None
    definitions: dict[str, ast.expr] = {}
    for stmt in vec_loop.body:
        extraction = _extraction(stmt, vec_var)
        if extraction is not None:
            if extraction.name in extractions:
                return None
            extractions[extraction.name] = extraction
            continue
        folded = _fold_operands(stmt, definitions)
        if folded is not None:
            if fold is not None:
                return None
            acc, fold = folded
            continue
        if not _is_pure_assignment(stmt):
            return None
        assert isinstance(stmt, ast.Assign)
        definitions[stmt.targets[0].id] = stmt.value  # type: ignore[union-attr]
    if fold is None or acc is None or len(extractions) != 2:
        return None
    # Every other name the element loop binds is a product temporary the
    # fold consumed (or dead); anything read outside the loop pins it.
    for name in definitions:
        for stmt in loop.body:
            if stmt is vec_loop:
                continue
            if name in ReadWrites.from_ast(stmt).reads:
                return None
    lhs_name, rhs_name = fold
    if lhs_name not in extractions or rhs_name not in extractions:
        return None
    # The extraction predicates may read the element loop's K bound (a K
    # tail); at packet level it becomes the lane-base bound.  Any other
    # element-level name (or a K that is not whole packets) declines.
    assert isinstance(vec_loop.target, ast.Name)
    lifted: dict[str, ast.expr | None] = {}
    for extraction in (extractions[lhs_name], extractions[rhs_name]):
        predicate, ok = _packet_predicate(
            extraction.predicate, definitions, vec_loop.target.id, width
        )
        if not ok:
            return None
        lifted[extraction.name] = predicate
    lhs = extractions[lhs_name]
    rhs = extractions[rhs_name]
    by_name = {packet.name: packet for packet in packets}
    if (
        lhs.packet not in by_name
        or rhs.packet not in by_name
        or lhs.packet == rhs.packet
    ):
        return None
    if lhs.element != rhs.element:
        return None
    # The accumulator is zeroed once before the lane loop (in this list or an
    # enclosing one), folded in the element loop, reduced over the K threads
    # right after the lane loop, and used nowhere else in the kernel.
    init = _find_zero_init(statements, index, stack, acc)
    if init is None:
        return None
    reduce_stmt: ast.stmt | None = None
    reduce_call: ast.Call | None = None
    for position, stmt in others:
        if _is_pure_assignment(stmt):
            continue
        call = _grouped_sum_call(stmt, acc)
        if call is None or reduce_stmt is not None or position < vec_position:
            return None
        reduce_stmt, reduce_call = stmt, call
    reduce_in_body = reduce_stmt is not None
    if reduce_stmt is None:
        if index + 1 >= len(statements):
            return None
        reduce_stmt = statements[index + 1]
        reduce_call = _grouped_sum_call(reduce_stmt, acc)
        if reduce_call is None:
            return None
    assert reduce_call is not None
    fold_stmts = [stmt for stmt in vec_loop.body if _fold_operands(stmt, definitions)]
    allowed = {id(init), id(reduce_stmt), *(id(stmt) for stmt in fold_stmts)}
    for stmt in _simple_statements(root):
        if id(stmt) not in allowed and _mentions(stmt, acc):
            return None
    # The vector operand must not vary with the row: the pass puts it in
    # every lane's B fragment, so its address may read only what the K lane
    # loop defines (lane base, K offsets, tensor arguments), never the row
    # prologue (``indices_0``, a gathered row index) or the row thread.
    vec_packet = by_name[rhs.packet]
    lane_names = {
        stmt.targets[0].id
        for stmt in loop.body
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    }
    pending = set(ReadWrites.from_ast(vec_packet.stmt.value).reads)
    vec_predicate = lifted[rhs_name]
    if vec_predicate is not None:
        pending |= set(ReadWrites.from_ast(vec_predicate).reads)
    seen: set[str] = set()
    lane_definitions = {
        stmt.targets[0].id: stmt.value
        for stmt in loop.body
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    }
    row_thread = "cute.arch.thread_idx()[0]"
    texts = [ast.unparse(vec_packet.stmt.value)]
    if vec_predicate is not None:
        texts.append(ast.unparse(vec_predicate))
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in row_names:
            return None
        if name in lane_names and name in lane_definitions:
            texts.append(ast.unparse(lane_definitions[name]))
            pending |= set(ReadWrites.from_ast(lane_definitions[name]).reads)
    # Neither the address nor anything it is computed from reads the row
    # thread (the lane body's own definitions included).
    if any(row_thread in text for text in texts):
        return None
    return _Match(
        lane_loop=loop,
        row_packet=by_name[lhs.packet],
        vec_packet=vec_packet,
        row_predicate=lifted[lhs_name],
        vec_predicate=vec_predicate,
        element=lhs.element,
        vec_loop=vec_loop,
        acc=acc,
        init=init,
        reduce_stmt=reduce_stmt,
        reduce_call=reduce_call,
        reduce_in_body=reduce_in_body,
        lane_extent=lane_extent,
    )


# Bytes of the vector chunk one CTA stages in shared memory at most (the K
# chunk of 16-bit elements times two bytes); larger chunks keep the vector
# packets in registers.  14 one-warp CTAs per SM at 8 KB leave ~110 KB of
# the unified 228 KB to L1; a 16 KB stage per CTA (a full 8192-element K)
# squeezed L1 out and read 82 us instead of 52 on gather_gemv (8, 8192, 2).
_SMEM_CHUNK_BYTES = 8 * 1024


class _NameSubstitution(ast.NodeTransformer):
    def __init__(self, name: str, replacement: str) -> None:
        self.name = name
        self.replacement = replacement

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            return expr_from_string(self.replacement)
        return node


def _lane_base_name(match: _Match) -> str | None:
    """The name of the K lane base the vector packet's address reads: the
    lane body statement ``NAME = <chunk offset> + (lane * 4 + k_thread) * V``."""
    reads = set(ReadWrites.from_ast(match.vec_packet.stmt.value).reads)
    lane_var = cast("ast.Name", match.lane_loop.target).id
    for stmt in match.lane_loop.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in reads
            and lane_var in ReadWrites.from_ast(stmt.value).reads
            and "cute.arch.thread_idx()[1]" in ast.unparse(stmt.value)
        ):
            return stmt.targets[0].id
    return None


def _rewrite(
    match: _Match, new_var: Callable[[str], str], prologue: list[ast.stmt]
) -> list[ast.stmt]:
    """Replace the element loop by MMAs, the zero init by the four fragment
    accumulators and the grouped sum by the first accumulator column.

    The vector packets (the same for every row group of the warp) are staged
    once per K chunk in shared memory by the warp and read back as two
    16-byte shared loads right before the MMAs, so the registers that hold
    bytes in flight from DRAM carry only the row packets (a thread holding
    both read 108 registers for four packets each; the vector packets are
    L2-resident and cost nothing to re-read)."""
    helper = _MMA_HELPERS[match.element]
    words = match.row_packet.width // 2
    packet_bytes = words * 4
    chunk_bytes = match.lane_extent * FRAGMENT_MMA_K_THREADS * packet_bytes
    lane_base = _lane_base_name(match)
    # The 32 lanes fill the stage one packet each per step: the chunk must
    # hold whole steps (a shorter chunk keeps the vector packets in
    # registers; a partial step left lanes' packets unwritten).
    stage = (
        lane_base is not None
        and chunk_bytes <= _SMEM_CHUNK_BYTES
        and chunk_bytes % (32 * packet_bytes) == 0
    )
    for packet in (match.row_packet, match.vec_packet):
        packet.vector_type.args = [
            ast.List(elts=[ast.Constant(words)], ctx=ast.Load()),
            cast("ast.expr", expr_from_string("cutlass.Uint32.mlir_type")),
        ]
    zero = new_var("_fmma_zero")
    smem_words: list[str] = []
    fill: list[ast.stmt] = []
    smem_addr = chunk_offset = ""
    if stage:
        assert lane_base is not None
        smem = new_var("_fmma_xs")
        smem_addr = new_var("_fmma_xs_addr")
        # The allocation is a kernel top-level statement (prepended once the
        # walk is done).
        prologue.extend(
            [
                statement_from_string(
                    f"{smem} = cute.arch.alloc_smem(cutlass.Uint32, {chunk_bytes // 4})"
                ),
                statement_from_string(f"{smem_addr} = cutlass.Int32({smem}.toint())"),
            ]
        )
        # The fill: the 32 lanes copy the chunk's vector packets (``steps``
        # packets each) through the vector packet's own address expression
        # (bounds select included) with the lane base of packet ``i`` of
        # lane ``l``: ``<chunk offset> + (i * 32 + l) * V``.  The hardware
        # lane spelled as ``thread_idx()[0] * 4 + thread_idx()[1]`` is the
        # identity after the lane permutation below.
        lane_var = cast("ast.Name", match.lane_loop.target).id
        steps = chunk_bytes // (32 * packet_bytes)
        lane_id = (
            "(cutlass.Int32(cute.arch.thread_idx()[0]) * 4 + "
            "cutlass.Int32(cute.arch.thread_idx()[1]))"
        )
        chunk_offset = new_var("_fmma_xs_off")
        fill.append(statement_from_string("cute.arch.sync_warp()"))
        # The lane base of lane 0's packet 0 is the chunk offset: evaluate the
        # lane base statement with lane 0 and K thread 0.
        base_stmt = next(
            stmt
            for stmt in match.lane_loop.body
            if isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == lane_base
        )
        base_text = ast.unparse(
            _NameSubstitution(lane_var, "0").visit(
                statement_from_string(ast.unparse(base_stmt))
            )
        )
        base_text = base_text.split("=", 1)[1].strip()
        base_text = base_text.replace(
            "cutlass.Int32(cute.arch.thread_idx()[1])", "cutlass.Int32(0)"
        )
        fill.append(statement_from_string(f"{chunk_offset} = {base_text}"))
        for i in range(steps):
            element = (
                f"({chunk_offset} + ({i} * 32 + {lane_id}) * {match.vec_packet.width})"
            )
            load_text = (
                ast.unparse(
                    _NameSubstitution(lane_base, element).visit(
                        statement_from_string(ast.unparse(match.vec_packet.stmt))
                    )
                )
                .split("=", 1)[1]
                .strip()
            )
            packet_name = new_var("_fmma_xs_pk")
            fill.append(statement_from_string(f"{packet_name} = {load_text}"))
            # A packet the vector predicate excludes (a K tail) is stored as
            # zeros: its lanes' MMAs must not see the clamped load's bytes.
            word_texts = [f"cutlass.Uint32({packet_name}[{w}])" for w in range(words)]
            if match.vec_predicate is not None:
                fill_predicate = ast.unparse(
                    _NameSubstitution(lane_base, element).visit(
                        cast(
                            "ast.expr",
                            expr_from_string(ast.unparse(match.vec_predicate)),
                        )
                    )
                )
                word_texts = [
                    f"({text} if {fill_predicate} else cutlass.Uint32(0))"
                    for text in word_texts
                ]
            byte_offset = f"({i} * 32 + {lane_id}) * {packet_bytes}"
            for half in range(words // 4):
                fill.append(
                    statement_from_string(
                        f"_cute_sts_v4_b32({smem_addr} + {byte_offset} + {half * 16}, "
                        + ", ".join(word_texts[half * 4 : half * 4 + 4])
                        + ")"
                    )
                )
        fill.append(statement_from_string("cute.arch.sync_warp()"))
        smem_words = [new_var(f"_fmma_b{i}") for i in range(words)]
    # One accumulator chain per MMA slot of a packet: the slots' MMAs then
    # issue back to back instead of waiting on each other (a single chain
    # read 10% slower in the hand-written kernels); the chains are summed
    # where the grouped sum stood.
    chains = max(1, words // 2)
    accs = [[new_var(f"_fmma_c{j}_{i}") for i in range(4)] for j in range(chains)]
    flat_accs = [acc for chain in accs for acc in chain]
    # The grouped sum over the K threads is the MMA's own contraction: its
    # value is the first accumulator column (every column holds the row).
    # The marker is stripped when the statement is copied (in the lane body
    # or, by the caller, after the loop).
    match.reduce_call.func = ast.Name("_helion_fragment_mma_result", ast.Load())
    match.reduce_call.args = [
        cast("ast.expr", expr_from_string(" + ".join(chain[0] for chain in accs)))
    ]
    match.reduce_call.keywords = []
    body: list[ast.stmt] = []
    for stmt in match.lane_loop.body:
        if stage and stmt is match.vec_packet.stmt:
            # The vector packet comes from shared memory, right before the
            # MMAs (emitted with them below).
            continue
        if stmt is match.vec_loop:
            body.append(statement_from_string(f"{zero} = cutlass.Uint32(0)"))
            a_words: list[str] = []
            b_words: list[str] = []
            roles = [("a", match.row_packet, match.row_predicate, a_words)]
            if stage:
                assert lane_base is not None
                offset = f"({lane_base} - {chunk_offset}) * 2"
                for half in range(words // 4):
                    body.append(
                        statement_from_string(
                            ", ".join(smem_words[half * 4 : half * 4 + 4])
                            + f" = _cute_lds_v4_b32({smem_addr} + {offset} + {half * 16})"
                        )
                    )
                b_words.extend(smem_words)
            else:
                roles.append(("b", match.vec_packet, match.vec_predicate, b_words))
            for role, packet, predicate, names in roles:
                for i in range(words):
                    name = new_var(f"_fmma_{role}{i}")
                    names.append(name)
                    text = f"cutlass.Uint32({packet.name}[{i}])"
                    if predicate is not None:
                        text = f"{text} if {{pred}} else {zero}"
                        body.append(
                            statement_from_string(f"{name} = {text}", pred=predicate)
                        )
                    else:
                        body.append(statement_from_string(f"{name} = {text}"))
            for j in range(words // 2):
                targets = ", ".join(accs[j % chains])
                body.append(
                    statement_from_string(
                        f"{targets} = {helper}({a_words[2 * j]}, {zero}, "
                        f"{a_words[2 * j + 1]}, {zero}, {b_words[2 * j]}, "
                        f"{b_words[2 * j + 1]}, {targets})"
                    )
                )
            continue
        body.append(_strip_result_marker(stmt) if stmt is match.reduce_stmt else stmt)
    match.lane_loop.body = body
    # The accumulators are zeroed where the running sum was (once per row,
    # before a chunk loop that reduces inside it, or right before the lane
    # loop): a tuple assignment keeps the statement in place.
    init_target = statement_from_string(
        f"{', '.join(flat_accs)} = "
        f"({', '.join('cutlass.Float32(0.0)' for _ in flat_accs)})"
    )
    assert isinstance(init_target, ast.Assign)
    match.init.targets = init_target.targets
    match.init.value = init_target.value
    return fill


class _ResultMarker(ast.NodeTransformer):
    """``_helion_fragment_mma_result(c0)`` -> ``c0``."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "_helion_fragment_mma_result"
            and len(node.args) == 1
        ):
            return node.args[0]
        return node


def _strip_result_marker(stmt: ast.stmt) -> ast.stmt:
    return _ResultMarker().visit(stmt)


class _LanePermutation(ast.NodeTransformer):
    """``thread_idx()[0]`` -> the fragment row group, ``[1]`` -> the K quarter."""

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == "cute.arch.thread_idx"
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in (0, 1)
        ):
            lane = (
                "(cutlass.Int32(cute.arch.thread_idx()[0]) + "
                f"cutlass.Int32(cute.arch.thread_idx()[1]) * {FRAGMENT_MMA_ROW_THREADS})"
            )
            if node.slice.value == 0:
                return expr_from_string(f"({lane} // {FRAGMENT_MMA_K_THREADS})")
            return expr_from_string(f"({lane} % {FRAGMENT_MMA_K_THREADS})")
        return node


def _row_prologue_names(stack: list[list[ast.stmt]]) -> set[str]:
    """Names the enclosing lane loops define: the row prologue a vector
    operand must not read."""
    names: set[str] = set()
    for statements in stack:
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif _is_lane_loop(stmt):
                assert isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name)
                names.add(stmt.target.id)
    return names


def _apply(
    statements: list[ast.stmt],
    stack: list[list[ast.stmt]],
    new_var: Callable[[str], str],
    root: list[ast.stmt],
    prologue: list[ast.stmt],
) -> tuple[list[ast.stmt], int]:
    fired = 0
    result: list[ast.stmt] = []
    index = 0
    while index < len(statements):
        stmt = statements[index]
        if _is_lane_loop(stmt):
            match = _match_lane_loop(statements, index, stack, root)
            if match is not None:
                # The smem fill of the vector chunk runs right before the lane
                # loop (once per K chunk).
                result.extend(_rewrite(match, new_var, prologue))
                result.append(stmt)
                if match.reduce_in_body:
                    index += 1
                else:
                    result.append(_strip_result_marker(match.reduce_stmt))
                    index += 2
                fired += 1
                continue
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With)):
            for field in ("body", "orelse"):
                nested = getattr(stmt, field, None)
                if nested:
                    # The enclosing statements and this loop (its target is
                    # a row lane when it is a lane loop) are the row prologue.
                    rewritten, count = _apply(
                        nested,
                        [*stack, [*statements[:index], stmt]],
                        new_var,
                        root,
                        prologue,
                    )
                    setattr(stmt, field, rewritten)
                    fired += count
        result.append(stmt)
        index += 1
    return result, fired


def fragment_mma_dot(
    body: list[ast.stmt],
    *,
    thread_block_dims: tuple[int, int, int] | None,
    new_var: Callable[[str], str],
) -> tuple[list[ast.stmt], bool]:
    """Rewrite every matching scalar-dot lane loop of ``body`` onto MMA
    fragments; ``(body, fired)``.  Declines (``fired`` False, body unchanged)
    when the thread block is not ``(8, 4, 1)``, when no lane loop matches or
    when the permuted body would still hold a cross-lane primitive."""
    if thread_block_dims != FRAGMENT_MMA_BLOCK_DIMS:
        return body, False
    # The lanes are permuted below, so nothing else in the kernel may
    # communicate across lanes: only the grouped sums the rewrite consumes.
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                name = ast.unparse(node.func).lower()
                if name != _GROUPED_REDUCE and any(
                    marker in name for marker in _CROSS_LANE_MARKERS
                ):
                    return body, False
    body = list(body)
    prologue: list[ast.stmt] = []
    rewritten, fired = _apply(body, [], new_var, body, prologue)
    if not fired:
        return body, False
    rewritten = [*prologue, *rewritten]
    for stmt in rewritten:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == _GROUPED_REDUCE:
                # A grouped sum the rewrite did not consume.
                raise FragmentMmaNotApplied
    permuted = [_LanePermutation().visit(stmt) for stmt in rewritten]
    return permuted, True

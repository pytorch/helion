"""Trace-time rewrite of a GEMV's scalar dot fallback onto a cp.async ring
feeding ``mma.sync`` (``cute_dot_ring_mma``).

``hl.dot(lhs[m, k], rhs[k, 1])`` with 16-bit operands lowers to the scalar
fallback: the rows of a CTA's tile are a rolled lane loop, each row walks K
in device-loop chunks, and inside a chunk every thread loads its 16-bit
packets of the row and of the vector, folds the products in fp32 and the
threads combine through a reduction::

    for lane_0 in range(R):                       # row lane loop
        indices_0 = tile_offset_0 + lane_0
        acc = cutlass.Float32(0.0)
        ... row prologue (a gathered row index, say) ...
        for tile_offset_1 in range(0, K, BK):     # K device loop
            _helion_vfold_acc_0 = cutlass.Float32(0)
            for lane_1 in range(L):               # K lane loop
                lane_base_1 = tile_offset_1 + thread_idx()[0] * V + lane_1 * V
                ... row index statements ...
                row packet = <load>(w.iterator + ROW * stride[0] + lane_base_1 * stride[1] ...)
                vec packet = <load>(x.iterator + lane_base_1 * stride[0] ...)
                for vec_lane_1 in range_constexpr(V): fold
            dot = <reduce>(_helion_vfold_acc_0, ...)
            acc = acc + dot
        out[indices_0] = acc

Every byte in flight from DRAM then sits in a register of the thread that
loaded it, which caps the bytes in flight per SM (gather_gemv (8, 8192, 2)
on a power-capped B200: scalar fallback 53 us, register fragments 52,
helion-triton's TMA/cp.async ring + tl.dot 49, the memory floor 45-47).

With ``cute_dot_ring_mma`` on a ``(T, 1, 1)`` thread block the pass replaces
the row lane loop by the structure Triton's pipeliner builds for that
kernel: the CTA's ``R x BK`` row tile streams through a ``STAGES``-deep
``cp.async.cg`` ring in shared memory (register-free bytes in flight), the
vector chunk rides along in the ring, and the warps consume each chunk with
``ldmatrix.x4`` + ``mma.sync.m16n8k16`` (B holds the vector in every column
so each D column is the row's dot; N is padded to 8 the way Triton pads
``tl.dot`` to 16).  ``cute_dot_ring_split`` says whether the warps split
the chunk's K steps (``k``: a shared-memory sum of the warps' partials at
the end) or the tile's 16-row atoms (``m``).  The row prologue and the row
index statements are re-instantiated per row the thread issues or stores,
so any row index expression (a gather, a modulus) is supported; the K tail
beyond the last whole chunk is zero-filled by cp.async's ``src-size``.

The kernel's dependents (``cute_pdl``) are released after the ring's last
MMA whatever ``cute_pdl_release`` says (the normalizer pins the key to its
default under the ring): released at entry, the dependents' CTAs take the
slots the ring leaves free and the kernel reads 55 us instead of 46.

Products are exact in fp32 and accumulated in fp32 (``tl.dot``'s numerics
for the same kernel).  The pass fires only on the complete pattern and
reports whether it did so the caller can regenerate with the knob off
(``RingMmaNotApplied``): the knob alone never changes the code.
"""

from __future__ import annotations

import ast
import dataclasses
import logging
import textwrap
from typing import TYPE_CHECKING

from ... import exc
from ...runtime.config import Config
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import ReadWrites
from .cute_warp_mma_gemm import _swizzled_chunk
from .fragment_mma_dot import _CROSS_LANE_MARKERS
from .fragment_mma_dot import _MMA_HELPERS
from .fragment_mma_dot import _constexpr_extent
from .fragment_mma_dot import _extraction
from .fragment_mma_dot import _fold_operands
from .fragment_mma_dot import _is_lane_loop
from .fragment_mma_dot import _is_load_call
from .fragment_mma_dot import _is_pure_assignment
from .fragment_mma_dot import _plain_range_extent

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    import torch


RING_MMA_KEY = "cute_dot_ring_mma"
RING_STAGES_KEY = "cute_dot_ring_stages"
RING_SPLIT_KEY = "cute_dot_ring_split"
RING_STAGE_CHOICES: tuple[int, ...] = (2, 3, 4, 6, 8)
RING_SPLIT_CHOICES: tuple[str, ...] = ("k", "m")
RING_DEFAULT_STAGES = 3
RING_DEFAULT_SPLIT = "k"
RING_MIN_ROWS = 16
RING_MAX_ROWS = 256
RING_MIN_THREADS = 32
RING_MAX_THREADS = 512
# Shared memory a CTA may stage when the device cannot be queried (the B200's
# 227 KB opt-in limit); ``ring_mma_smem_budget`` reads the real device.
RING_SMEM_BUDGET = 227 * 1024
# The K lane loop must load 16-bit packets (a width of 1 is a scalar load).
RING_MIN_K_WIDTH = 2
# Row pointers a thread keeps in registers (one 64-bit pointer and one
# shared-memory offset per packet row it issues).
RING_MAX_ROWS_PER_THREAD = 32
_PACKET_BYTES = 16
_ATOM_ROWS = 16
_K_STEP = 16
log = logging.getLogger(__name__)


def _no(reason: str) -> None:
    """Decline the match with ``reason`` (logged at debug level)."""
    log.debug("cute_dot_ring_mma: no match: %s", reason)
    return None


class RingMmaNotApplied(exc.Base):
    """``cute_dot_ring_mma`` asked for the ring rewrite but no dot took it.

    The knob must not change the code by itself, so ``generate_ast``
    generates the kernel again with the knob off (``ring_mma_off_config``).
    """


def ring_mma_off_config(config: Config) -> Config:
    """``config`` with the ring MMA off."""
    return Config.from_dict({**config.config, RING_MMA_KEY: False})


def ring_mma_smem_budget(device: torch.device | None) -> int:
    """Shared memory one CTA may stage on ``device``: the opt-in per-block
    capacity (the B200's 227 KB), the B200 figure when the device cannot be
    queried (CPU code generation)."""
    from .tcgen05_config import CuteTcgen05Config

    if device is None:
        return RING_SMEM_BUDGET
    return CuteTcgen05Config.per_cta_smem_capacity_bytes(device) or RING_SMEM_BUDGET


def ring_mma_smem_bytes(
    *, rows: int, chunk: int, stages: int, threads: int, split: str, elem_bytes: int
) -> int:
    """Shared memory of the ring: ``stages`` row tiles and vector chunks plus
    the warps' partial sums of a K split."""
    stage = (rows + 1) * chunk * elem_bytes
    reduce = threads // 32 * rows * 4 if split == "k" else 0
    return stages * stage + reduce


def ring_mma_geometry_reason(
    *,
    rows: int,
    chunk: int,
    threads: int,
    stages: int,
    split: str,
    elem_bytes: int,
    k_width: int | None,
    smem_budget: int = RING_SMEM_BUDGET,
) -> str | None:
    """Why ``rows x chunk`` on ``threads`` threads with ``stages`` ring slots
    and the warp ``split`` cannot run the ring (None: it can).  Shared by the
    pass and the config normalizer so no admitted config fails late.
    ``k_width`` is the K vector width of the config (None when the caller does
    not know it): the K lane loop must be one packet per thread per chunk
    (``chunk == threads * k_width``), the layout whose lane base is the plain
    K position the matcher streams."""
    if elem_bytes != 2:
        return "needs fp16 or bf16 operands"
    if k_width is not None and k_width < RING_MIN_K_WIDTH:
        return f"needs K packets of at least {RING_MIN_K_WIDTH} elements"
    if k_width is not None and chunk != threads * k_width:
        return "needs one K packet per thread per chunk (block_k == num_threads * vector width)"
    if threads % 32 or not RING_MIN_THREADS <= threads <= RING_MAX_THREADS:
        return f"needs a whole number of warps, {RING_MIN_THREADS}-{RING_MAX_THREADS} threads"
    if rows % _ATOM_ROWS or not RING_MIN_ROWS <= rows <= RING_MAX_ROWS:
        return f"needs a row block that is a multiple of {_ATOM_ROWS} in [{RING_MIN_ROWS}, {RING_MAX_ROWS}]"
    if chunk % _K_STEP or chunk * elem_bytes % _PACKET_BYTES:
        return f"needs a K block that is a multiple of {_K_STEP}"
    packets_per_row = chunk * elem_bytes // _PACKET_BYTES
    if (rows * packets_per_row) % threads:
        return "the row tile's 16-byte packets must split evenly over the threads"
    if threads % packets_per_row and packets_per_row % threads:
        return "the threads must divide or be a multiple of a row's 16-byte packets"
    rows_per_thread = (
        rows * packets_per_row // threads if threads % packets_per_row == 0 else rows
    )
    if rows_per_thread > RING_MAX_ROWS_PER_THREAD:
        return f"a thread would hold more than {RING_MAX_ROWS_PER_THREAD} row pointers"
    if stages not in RING_STAGE_CHOICES:
        return f"{RING_STAGES_KEY} must be one of {RING_STAGE_CHOICES}"
    warps = threads // 32
    if split == "k":
        if (chunk // _K_STEP) % warps:
            return "a K split needs the chunk's MMA steps to divide over the warps"
    elif split == "m":
        if (rows // _ATOM_ROWS) % warps:
            return "an M split needs the tile's 16-row atoms to divide over the warps"
    else:
        return f"{RING_SPLIT_KEY} must be one of {RING_SPLIT_CHOICES}"
    smem = ring_mma_smem_bytes(
        rows=rows,
        chunk=chunk,
        stages=stages,
        threads=threads,
        split=split,
        elem_bytes=elem_bytes,
    )
    if smem > smem_budget:
        return f"the ring stages {smem} B of shared memory, over {smem_budget}"
    return None


# ---------------------------------------------------------------------------
# matching


def _assigned_name(stmt: ast.AST) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    return None


def _reads(node: ast.AST) -> set[str]:
    return set(ReadWrites.from_ast(node).reads)


def _mentions_thread(node: ast.AST) -> bool:
    return "cute.arch.thread_idx" in ast.unparse(node)


def _has_cross_lane(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            name = ast.unparse(sub.func).lower()
            if any(marker in name for marker in _CROSS_LANE_MARKERS):
                return True
    return False


def _is_store_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "store"
    )


# The cross-thread sums the scalar dot fallback emits (``matmul_fallback.py``
# and the one-warp ``warp_reduction_sum`` of ``backend.py``); the first
# positional argument is the folded value, the grouped forms name the
# operation as their second argument.
_WARP_SUM = "cute.arch.warp_reduction_sum"
_GROUPED_SUMS = (
    "_cute_grouped_reduce_warp",
    "_cute_grouped_reduce_shared_two_stage",
    "_cute_grouped_reduce_shared_tree",
)


def _is_sum_reduction(call: ast.Call, folded: str) -> bool:
    """Whether ``call`` is the fallback's cross-thread sum of ``folded``."""
    if not (
        call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == folded
    ):
        return False
    func = ast.unparse(call.func)
    if func == _WARP_SUM:
        return True
    return (
        func in _GROUPED_SUMS
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "sum"
    )


def _is_zero_float(expr: ast.AST) -> bool:
    return ast.unparse(expr) in ("cutlass.Float32(0)", "cutlass.Float32(0.0)")


def _int32_const(expr: ast.AST) -> int | None:
    """``cutlass.Int32(N)`` or ``N`` -> N."""
    if (
        isinstance(expr, ast.Call)
        and ast.unparse(expr.func) == "cutlass.Int32"
        and len(expr.args) == 1
        and not expr.keywords
    ):
        expr = expr.args[0]
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
        return expr.value
    return None


def _strip_int32(expr: ast.expr) -> ast.expr:
    while (
        isinstance(expr, ast.Call)
        and ast.unparse(expr.func) == "cutlass.Int32"
        and len(expr.args) == 1
        and not expr.keywords
    ):
        expr = expr.args[0]
    return expr


@dataclasses.dataclass
class _DeviceLoop:
    loop: ast.For
    var: str
    start: int
    stop: int


def _match_device_loop(
    stmt: ast.AST, k_block_var: str | None, chunk: int
) -> _DeviceLoop | None:
    """``for KOFF in range(cutlass.Int32(a), cutlass.Int32(b), cutlass.Int32(BK))``."""
    if not (
        isinstance(stmt, ast.For)
        and isinstance(stmt.target, ast.Name)
        and not stmt.orelse
        and isinstance(stmt.iter, ast.Call)
        and ast.unparse(stmt.iter.func) == "range"
        and len(stmt.iter.args) == 3
        and not stmt.iter.keywords
    ):
        return None
    start = _int32_const(stmt.iter.args[0])
    stop = _int32_const(stmt.iter.args[1])
    step = _strip_int32(stmt.iter.args[2])
    if start is None or stop is None:
        return None
    if isinstance(step, ast.Name):
        if k_block_var is None or step.id != k_block_var:
            return None
    elif not (isinstance(step, ast.Constant) and step.value == chunk):
        return None
    return _DeviceLoop(stmt, stmt.target.id, start, stop)


@dataclasses.dataclass
class _Packet:
    stmt: ast.Assign
    name: str
    width: int
    address: ast.expr
    predicate: ast.expr | None


def _packet_load(stmt: ast.AST) -> _Packet | None:
    """``NAME = <load>(ADDR [if PRED else ADDR0], ir.VectorType.get([V], cutlass.Uint16.mlir_type))``."""
    name = _assigned_name(stmt)
    if name is None:
        return None
    assert isinstance(stmt, ast.Assign)
    call = stmt.value
    if not (
        isinstance(call, ast.Call)
        and _is_load_call(call)
        and len(call.args) == 2
        and not call.keywords
    ):
        return None
    vector_type = call.args[1]
    if not (
        isinstance(vector_type, ast.Call)
        and ast.unparse(vector_type.func) == "ir.VectorType.get"
        and len(vector_type.args) == 2
        and ast.unparse(vector_type.args[1]) == "cutlass.Uint16.mlir_type"
        and isinstance(vector_type.args[0], ast.List)
        and len(vector_type.args[0].elts) == 1
        and isinstance(vector_type.args[0].elts[0], ast.Constant)
        and isinstance(vector_type.args[0].elts[0].value, int)
    ):
        return None
    width = vector_type.args[0].elts[0].value
    address = call.args[0]
    predicate: ast.expr | None = None
    if isinstance(address, ast.IfExp):
        predicate = address.test
        address = address.body
    return _Packet(stmt, name, width, address, predicate)


@dataclasses.dataclass
class _Address:
    tensor: str
    terms: list[tuple[ast.expr, int]]  # (coordinate, dim)


def _parse_address(expr: ast.expr) -> _Address | None:
    """``T.iterator + cutlass.Int32(E0) * cutlass.Int32(T.layout.stride[0]) + ...``."""
    parts: list[ast.expr] = []

    def flatten(node: ast.expr) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            flatten(node.left)
            flatten(node.right)
        else:
            parts.append(node)

    flatten(expr)
    if not parts:
        return None
    base = parts[0]
    if not (
        isinstance(base, ast.Attribute)
        and base.attr == "iterator"
        and isinstance(base.value, ast.Name)
    ):
        return None
    tensor = base.value.id
    terms: list[tuple[ast.expr, int]] = []
    for part in parts[1:]:
        if not (isinstance(part, ast.BinOp) and isinstance(part.op, ast.Mult)):
            return None
        stride = _strip_int32(part.right)
        if not (
            isinstance(stride, ast.Subscript)
            and ast.unparse(stride.value) == f"{tensor}.layout.stride"
            and isinstance(stride.slice, ast.Constant)
            and isinstance(stride.slice.value, int)
        ):
            return None
        terms.append((_strip_int32(part.left), stride.slice.value))
    return _Address(tensor, terms)


def _conjuncts(expr: ast.expr | None) -> list[ast.expr]:
    if expr is None:
        return []
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.And):
        result: list[ast.expr] = []
        for value in expr.values:
            result.extend(_conjuncts(value))
        return result
    return [expr]


def _lane_base_axis(
    value: ast.expr, *, k_var: str, lane_var: str, width: int
) -> int | None:
    """The thread axis of the plain K lane base ``KOFF + thread_idx()[a] * V +
    lane * V`` (the packet's own K position), or None for any other form."""
    parts: list[ast.expr] = []

    def flatten(node: ast.expr) -> None:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            flatten(node.left)
            flatten(node.right)
        else:
            parts.append(node)

    flatten(value)
    if len(parts) != 3:
        return None
    seen_k = seen_lane = False
    axis: int | None = None
    for part in parts:
        bare = _strip_int32(part)
        if isinstance(bare, ast.Name) and bare.id == k_var:
            seen_k = True
            continue
        if not (
            isinstance(part, ast.BinOp)
            and isinstance(part.op, ast.Mult)
            and isinstance(part.right, ast.Constant)
            and part.right.value == width
        ):
            return None
        factor = _strip_int32(part.left)
        if isinstance(factor, ast.Name) and factor.id == lane_var:
            seen_lane = True
        elif (
            isinstance(factor, ast.Subscript)
            and ast.unparse(factor.value) == "cute.arch.thread_idx()"
            and isinstance(factor.slice, ast.Constant)
            and isinstance(factor.slice.value, int)
        ):
            axis = factor.slice.value
        else:
            return None
    return axis if seen_k and seen_lane and axis is not None else None


def _k_bound(conjunct: ast.expr, lane_base: str) -> int | None:
    """``LANE_BASE < N`` (the generic K bounds mask) -> N."""
    if (
        isinstance(conjunct, ast.Compare)
        and len(conjunct.ops) == 1
        and isinstance(conjunct.ops[0], ast.Lt)
        and len(conjunct.comparators) == 1
    ):
        left = _strip_int32(conjunct.left)
        bound = _int32_const(conjunct.comparators[0])
        if isinstance(left, ast.Name) and left.id == lane_base and bound is not None:
            return bound
    return None


@dataclasses.dataclass
class _Match:
    row_loop: ast.For
    row_var: str
    pre: list[ast.stmt]  # the row prologue without the accumulator init
    acc: str
    k_loop: _DeviceLoop
    post: list[ast.stmt]
    row_index_slice: list[ast.stmt]
    row_index: ast.expr
    row_predicate: ast.expr | None
    w_name: str
    w_row_dim: int
    w_k_dim: int
    x_name: str
    x_k_dim: int
    element: str
    k_width: int


def _closure(
    seeds: set[str], definitions: dict[str, ast.expr], stop: set[str]
) -> set[str]:
    """Names reachable from ``seeds`` through ``definitions`` (``stop`` names
    are not expanded)."""
    pending = set(seeds)
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in stop:
            continue
        if name in definitions:
            pending |= _reads(definitions[name])
    return seen


def _match_row_loop(
    stmt: ast.AST,
    *,
    rows: int,
    chunk: int,
    k_block_var: str | None,
    tensors: dict[str, torch.Tensor],
    aligned: set[str],
    thread_axis: int,
) -> _Match | None:
    if not (_is_lane_loop(stmt) and isinstance(stmt, ast.For)):
        return _no("not a lane loop")
    if _plain_range_extent(stmt) != rows or not isinstance(stmt.target, ast.Name):
        return _no(
            "if _plain_range_extent(stmt) != rows or not isinstance(stmt.target, as..."
        )
    row_var = stmt.target.id
    k_loops = [
        (index, loop)
        for index, body_stmt in enumerate(stmt.body)
        if (loop := _match_device_loop(body_stmt, k_block_var, chunk)) is not None
    ]
    if len(k_loops) != 1:
        return _no("the row loop must hold exactly one K device loop")
    k_index, k_loop = k_loops[0]
    pre = stmt.body[:k_index]
    post = stmt.body[k_index + 1 :]
    # The row prologue: single-name assignments (gathers allowed), nothing
    # cross-lane, nothing per thread, no stores.
    for pre_stmt in pre:
        if _assigned_name(pre_stmt) is None:
            return _no("a row prologue statement is not a single-name assignment")
        if _has_cross_lane(pre_stmt) or _mentions_thread(pre_stmt):
            return _no("a row prologue statement is per thread or cross-lane")
        if any(_is_store_call(node) for node in ast.walk(pre_stmt)):
            return _no("a row prologue statement stores")
    # ---- the K loop body: the K lane loop, with the fold's zero init before
    # it (chunked form) or in the row prologue (running-sum form, which also
    # reduces inside the lane loop), and the reduce + sum after the lane loop
    # or at the end of its body.
    body = k_loop.loop.body
    lane_positions = [i for i, s in enumerate(body) if _is_lane_loop(s)]
    if len(lane_positions) != 1:
        return _no("the K loop must hold one lane loop")
    lane_position = lane_positions[0]
    lane_loop = body[lane_position]
    assert isinstance(lane_loop, ast.For) and isinstance(lane_loop.target, ast.Name)
    if _plain_range_extent(lane_loop) is None:
        return _no("the K lane loop must have a static trip count")
    lane_var = lane_loop.target.id
    inits = body[:lane_position]
    trailing = list(body[lane_position + 1 :])
    if len(inits) > 1:
        return _no("more than one statement before the K lane loop")
    init_name = _assigned_name(inits[0]) if inits else None
    if inits and (init_name is None or not _is_zero_float(inits[0].value)):  # type: ignore[union-attr]
        return _no("the statement before the K lane loop is not a zero init")
    # ---- the lane body: packets, the element loop, the row index statements
    # before it, the reduce + sum (running-sum form) after it.
    packets: list[_Packet] = []
    vec_loop: ast.For | None = None
    lane_defs: dict[str, ast.expr] = {}
    lane_stmts: list[ast.stmt] = []
    lane_tail: list[ast.stmt] = []
    for lane_stmt in lane_loop.body:
        if vec_loop is not None:
            lane_tail.append(lane_stmt)
            continue
        packet = _packet_load(lane_stmt)
        if packet is not None:
            packets.append(packet)
            continue
        if isinstance(lane_stmt, ast.For) and _constexpr_extent(lane_stmt) is not None:
            vec_loop = lane_stmt
            continue
        name = _assigned_name(lane_stmt)
        if name is None or not _is_pure_assignment(lane_stmt):
            return _no("an impure lane body statement before the element loop")
        assert isinstance(lane_stmt, ast.Assign)
        lane_defs[name] = lane_stmt.value
        lane_stmts.append(lane_stmt)
    if vec_loop is None or len(packets) != 2:
        return _no("the lane body needs two 16-bit packet loads and the element loop")
    width = _constexpr_extent(vec_loop)
    assert width is not None
    if width < RING_MIN_K_WIDTH or any(p.width != width for p in packets):
        return _no("the K packets must hold at least two elements")
    assert isinstance(vec_loop.target, ast.Name)
    vec_var = vec_loop.target.id
    extractions = {}
    fold = None
    fold_acc = None
    definitions: dict[str, ast.expr] = {}
    for vec_stmt in vec_loop.body:
        extraction = _extraction(vec_stmt, vec_var)
        if extraction is not None:
            if extraction.name in extractions:
                return _no("an element is extracted twice")
            extractions[extraction.name] = extraction
            continue
        folded = _fold_operands(vec_stmt, definitions)
        if folded is not None:
            if fold is not None:
                return _no("two folds in the element loop")
            fold_acc, fold = folded
            continue
        if not _is_pure_assignment(vec_stmt):
            return _no("an impure element loop statement")
        assert isinstance(vec_stmt, ast.Assign)
        definitions[vec_stmt.targets[0].id] = vec_stmt.value  # type: ignore[union-attr]
    if fold is None or fold_acc is None or len(extractions) != 2:
        return _no("the element loop is not the two-operand fold")
    vfold = fold_acc
    if init_name is not None and init_name != vfold:
        return _no("the zero init before the lane loop is not the fold's")
    if init_name is None and not any(
        _assigned_name(s) == vfold and _is_zero_float(s.value)  # type: ignore[union-attr]
        for s in pre
    ):
        return _no("the fold accumulator is not zeroed before the K loop")
    lhs_name, rhs_name = fold
    if lhs_name not in extractions or rhs_name not in extractions:
        return _no("the fold operands are not the extracted elements")
    lhs, rhs = extractions[lhs_name], extractions[rhs_name]
    if lhs.element != rhs.element or lhs.element not in _MMA_HELPERS:
        return _no("the operands are not one fp16/bf16 dtype")
    by_name = {p.name: p for p in packets}
    if (
        lhs.packet not in by_name
        or rhs.packet not in by_name
        or lhs.packet == rhs.packet
    ):
        return _no("the extracted elements do not come from the two packets")
    # ---- K-dependent names: the lane variable, the chunk offset, anything
    # read from the thread index, and what depends on them.
    row_names = {row_var} | {name for s in pre if (name := _assigned_name(s))}
    k_dependent = {lane_var, k_loop.var}
    changed = True
    while changed:
        changed = False
        for name, value in lane_defs.items():
            if name in k_dependent:
                continue
            if _mentions_thread(value) or _reads(value) & k_dependent:
                k_dependent.add(name)
                changed = True

    def depends_on_row(packet: _Packet) -> bool:
        seeds = _reads(packet.address)
        if packet.predicate is not None:
            seeds |= _reads(packet.predicate)
        return bool(_closure(seeds, lane_defs, set()) & row_names)

    def depends_on_k(expr: ast.expr) -> bool:
        return bool(_closure(_reads(expr), lane_defs, set()) & k_dependent)

    # The lane base: the one lane name of the plain K position
    # ``KOFF + thread_idx()[axis] * V + lane * V``; any other K indexing (a
    # shift, a stride, a reversal) is not the ring's streaming order.
    lane_bases = [
        name
        for name, value in lane_defs.items()
        if _lane_base_axis(value, k_var=k_loop.var, lane_var=lane_var, width=width)
        == thread_axis
    ]
    if len(lane_bases) != 1:
        return _no("the lane body needs one plain K lane base on the thread axis")
    lane_base = lane_bases[0]

    def is_lane_base(coordinate: ast.expr) -> bool:
        bare = _strip_int32(coordinate)
        return isinstance(bare, ast.Name) and bare.id == lane_base

    lhs_packet, rhs_packet = by_name[lhs.packet], by_name[rhs.packet]
    if depends_on_row(lhs_packet) and not depends_on_row(rhs_packet):
        row_packet, vec_packet = lhs_packet, rhs_packet
    elif depends_on_row(rhs_packet) and not depends_on_row(lhs_packet):
        row_packet, vec_packet = rhs_packet, lhs_packet
    else:
        return _no("exactly one packet must depend on the row")
    w_address = _parse_address(row_packet.address)
    x_address = _parse_address(vec_packet.address)
    if w_address is None or x_address is None or len(w_address.terms) != 2:
        return _no("the packet addresses are not base + coordinate * stride sums")
    w_row_term = w_k_term = None
    for coordinate, dim in w_address.terms:
        if is_lane_base(coordinate):
            if w_k_term is not None:
                return _no("the row packet address has two K coordinates")
            w_k_term = (coordinate, dim)
        elif depends_on_k(coordinate):
            return _no("a row packet coordinate depends on K but is not the K position")
        else:
            w_row_term = (coordinate, dim)
    if w_row_term is None or w_k_term is None:
        return _no("the row packet address needs one row and one K coordinate")
    x_k_dim = None
    for coordinate, dim in x_address.terms:
        if is_lane_base(coordinate):
            if x_k_dim is not None:
                return _no("the vector address has two K coordinates")
            x_k_dim = dim
        elif depends_on_k(coordinate):
            return _no("a vector coordinate depends on K but is not the K position")
        elif _int32_const(coordinate) != 0:
            return _no("the vector address has a non-zero column coordinate")
    if x_k_dim is None:
        return _no("the vector address has no K coordinate")
    w_fake = tensors.get(w_address.tensor)
    x_fake = tensors.get(x_address.tensor)
    if w_fake is None or x_fake is None:
        return _no("the packet tensors are not kernel arguments")
    if w_fake.dtype.itemsize != 2 or x_fake.dtype != w_fake.dtype:
        return _no("the operand tensors are not one 16-bit dtype")
    if w_fake.stride(w_k_term[1]) != 1 or x_fake.stride(x_k_dim) != 1:
        return _no("the operands are not K-contiguous")
    # 16-byte packets: the row stride and both bases must be 16-byte aligned.
    if (w_fake.stride(w_row_term[1]) * w_fake.dtype.itemsize) % _PACKET_BYTES:
        return _no("the row stride is not 16-byte aligned")
    if w_address.tensor not in aligned or x_address.tensor not in aligned:
        return _no("an operand base is not provably 16-byte aligned")
    if (k_loop.start * w_fake.dtype.itemsize) % _PACKET_BYTES:
        return _no("the K loop does not start at a 16-byte boundary")
    # The K bounds masks: every K-dependent conjunct of the row packet and
    # every conjunct of the vector packet must be ``LANE_BASE < K_stop``,
    # the K loop's own extent, which the ring's zero-filled tail reproduces;
    # an operand without a mask must hold the whole K range.
    row_predicate_parts: list[ast.expr] = []
    k_bounds: dict[str, list[int]] = {"row": [], "vector": []}
    for part in _conjuncts(row_packet.predicate):
        if depends_on_k(part):
            bound = _k_bound(part, lane_base)
            if bound is None:
                return _no("a K-dependent row packet mask is not the plain K bound")
            k_bounds["row"].append(bound)
        else:
            row_predicate_parts.append(part)
    for part in _conjuncts(vec_packet.predicate):
        bound = _k_bound(part, lane_base)
        if bound is None:
            return _no("a vector packet mask is not the plain K bound")
        k_bounds["vector"].append(bound)
    for role, fake, dim in (
        ("row", w_fake, w_k_term[1]),
        ("vector", x_fake, x_k_dim),
    ):
        if k_bounds[role]:
            if any(bound != k_loop.stop for bound in k_bounds[role]):
                return _no(f"the {role} packet's K bound is not the K loop's extent")
        else:
            size = fake.size(dim)
            if not isinstance(size, int) or size < k_loop.stop:
                return _no(f"the unmasked {role} operand does not hold the K range")
    row_predicate: ast.expr | None = None
    if row_predicate_parts:
        row_predicate = (
            row_predicate_parts[0]
            if len(row_predicate_parts) == 1
            else ast.BoolOp(op=ast.And(), values=row_predicate_parts)
        )
    seeds = _reads(w_row_term[0])
    if row_predicate is not None:
        seeds |= _reads(row_predicate)
    needed = _closure(seeds, lane_defs, set())
    if needed & k_dependent:
        return _no("the row index depends on K")
    row_index_slice = [s for s in lane_stmts if _assigned_name(s) in needed]
    # ---- the reduce of the fold and ``ACC = ACC + DOT``: the statements
    # after the element loop (in the lane body and after the lane loop).
    tail = [*lane_tail, *trailing]
    aliases: dict[str, str] = {}
    for pre_stmt in pre:
        if isinstance(pre_stmt, ast.Assign) and isinstance(pre_stmt.value, ast.Name):
            name = _assigned_name(pre_stmt)
            if name is not None:
                aliases[name] = pre_stmt.value.id
    reduce_target: str | None = None
    if not tail:
        return _no("no statements after the element loop")
    for t_stmt in tail[:-1]:
        name = _assigned_name(t_stmt)
        if name is None:
            return _no("a statement after the element loop is not an assignment")
        assert isinstance(t_stmt, ast.Assign)
        value = t_stmt.value
        if isinstance(value, ast.Call) and _is_sum_reduction(value, vfold):
            if reduce_target is not None:
                return _no("two reductions of the fold")
            reduce_target = name
            continue
        if isinstance(value, ast.Name):
            aliases[name] = value.id
            continue
        if not _is_pure_assignment(t_stmt):
            return _no("an impure statement after the element loop")
    acc_stmt = tail[-1]
    acc = _assigned_name(acc_stmt)
    if (
        acc is None
        or reduce_target is None
        or not isinstance(acc_stmt, ast.Assign)
        or not isinstance(acc_stmt.value, ast.BinOp)
        or not isinstance(acc_stmt.value.op, ast.Add)
    ):
        return _no("the chunk does not end with ACC = ACC + DOT")

    def resolve(name: str) -> str:
        seen: set[str] = set()
        while name in aliases and name not in seen:
            seen.add(name)
            name = aliases[name]
        return name

    value = acc_stmt.value
    if not (isinstance(value.left, ast.Name) and isinstance(value.right, ast.Name)):
        return _no("the accumulate operands are not names")
    operands = {resolve(value.left.id), resolve(value.right.id)}
    if operands != {acc, reduce_target}:
        return _no("the accumulate is not ACC + the reduced fold")
    # The accumulator starts in the row prologue (any pure value: it is added
    # to the ring's dot at the end) and is stored after the K loop.
    if not any(_assigned_name(s) == acc for s in pre):
        return _no("the accumulator is not initialized in the row prologue")
    if not post or not any(acc in _reads(s) for s in post):
        return _no("the accumulator is not read after the K loop")
    # The stores may read the accumulator (the epilogue provides it) but no
    # other value the K loop computed (those are gone with the loop).
    forbidden = {k_loop.var, *_assigned_names(k_loop.loop.body)} - {acc}
    for post_stmt in post:
        if not isinstance(post_stmt, (ast.Assign, ast.Expr, ast.If)):
            return _no("a statement after the K loop is not an assignment, call or if")
        if _has_cross_lane(post_stmt) or _mentions_thread(post_stmt):
            return _no("a statement after the K loop communicates across lanes")
        if _reads(post_stmt) & forbidden:
            return _no("a statement after the K loop reads the K loop's values")
        for node in ast.walk(post_stmt):
            if isinstance(node, (ast.For, ast.While)):
                return _no("a loop after the K loop")
    return _Match(
        row_loop=stmt,
        row_var=row_var,
        pre=pre,
        acc=acc,
        k_loop=k_loop,
        post=post,
        row_index_slice=row_index_slice,
        row_index=w_row_term[0],
        row_predicate=row_predicate,
        w_name=w_address.tensor,
        w_row_dim=w_row_term[1],
        w_k_dim=w_k_term[1],
        x_name=x_address.tensor,
        x_k_dim=x_k_dim,
        element=lhs.element,
        k_width=width,
    )


# ---------------------------------------------------------------------------
# rewriting


class _Renamer(ast.NodeTransformer):
    """Rename the names of ``mapping`` in place (the nodes stay what they are:
    the kernel's own statements carry source locations)."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.mapping:
            node.id = self.mapping[node.id]
        return node


_PDL_WAIT = "cute.arch.griddepcontrol_wait"
_PDL_RELEASE = "cute.arch.griddepcontrol_launch_dependents"


def _is_call_statement(stmt: ast.AST, name: str) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and ast.unparse(stmt.value.func) == name
    )


def _uses_pdl(body: Sequence[ast.stmt]) -> bool:
    return any(
        isinstance(node, ast.Call) and ast.unparse(node.func) == _PDL_WAIT
        for stmt in body
        for node in ast.walk(stmt)
    )


def _without_entry_release(body: list[ast.stmt]) -> list[ast.stmt]:
    """``body`` without the generic early release of the dependents: the
    plain ``griddepcontrol_launch_dependents()`` after the kernel's wait
    (``cute_pdl_release`` entry) or the last-wave guard around it.  The
    ring releases after its last MMA instead (as the warp_mma family does):
    released at entry, the dependents' CTAs take the slots the ring's CTAs
    leave free and the kernel reads 55 us instead of 46 on gather_gemv
    (8, 8192, 2) with 7 CTAs per SM."""
    result: list[ast.stmt] = []
    for stmt in body:
        if _is_call_statement(stmt, _PDL_RELEASE):
            continue
        if (
            isinstance(stmt, ast.If)
            and len(stmt.body) == 1
            and _is_call_statement(stmt.body[0], _PDL_RELEASE)
            and all(isinstance(s, ast.Pass) for s in stmt.orelse)
        ):
            continue
        result.append(stmt)
    return result


class _AddressSelect(ast.NodeTransformer):
    """``X = (ADDR).load() if COND else ZERO`` ->
    ``T = (ADDR if COND else BASE).load(); X = T if COND else ZERO``.

    The row prologue's gathers are masked on the value, which the DSL lowers
    to a branch around the load; the rewrite loads unconditionally from the
    masked address (the tensor's own base with zero offsets, in bounds for
    any tensor) and keeps the value select, so every mask -- the bounds mask
    and an ``extra_mask`` alike -- yields exactly what the lowering yielded,
    and the prologue is straight-line."""

    def __init__(self, new_var: Callable[[str], str]) -> None:
        self.new_var = new_var

    def visit_Assign(self, node: ast.Assign) -> ast.AST | list[ast.AST]:
        value = node.value
        if not (
            isinstance(value, ast.IfExp)
            and isinstance(value.body, ast.Call)
            and isinstance(value.body.func, ast.Attribute)
            and value.body.func.attr == "load"
            and not value.body.args
            and not value.body.keywords
            and isinstance(value.orelse, ast.Call)
            and len(value.orelse.args) == 1
            and isinstance(value.orelse.args[0], ast.Constant)
            and value.orelse.args[0].value == 0
        ):
            return node
        address = _parse_address(value.body.func.value)
        if address is None or not address.terms:
            return node
        base = f"{address.tensor}.iterator" + "".join(
            f" + cutlass.Int32(0) * cutlass.Int32({address.tensor}.layout.stride[{dim}])"
            for _, dim in address.terms
        )
        safe = ast.parse(base, mode="eval").body
        loaded = self.new_var("_ring_gather")
        load = ast.Assign(
            targets=[ast.Name(id=loaded, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.IfExp(
                        test=value.test, body=value.body.func.value, orelse=safe
                    ),
                    attr="load",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            ),
            lineno=node.lineno,
        )
        node.value = ast.IfExp(
            test=value.test,
            body=ast.Name(id=loaded, ctx=ast.Load()),
            orelse=value.orelse,
        )
        return [load, node]


def _copy_statement(stmt: ast.stmt) -> ast.stmt:
    """A plain-AST copy of ``stmt`` (through its source text; the kernel's
    extended nodes are not deep-copyable)."""
    return ast.parse(ast.unparse(stmt)).body[0]


def _instantiate(
    statements: Sequence[ast.stmt],
    mapping: dict[str, str],
    indent: str,
    new_var: Callable[[str], str],
) -> list[str]:
    lines: list[str] = []
    renamer = _Renamer(mapping)
    select = _AddressSelect(new_var)
    for stmt in statements:
        module = ast.Module(body=[_copy_statement(stmt)], type_ignores=[])
        module = select.visit(module)
        for copied in module.body:
            renamer.visit(copied)
            for line in ast.unparse(copied).split("\n"):
                lines.append(indent + line)
    return lines


def _rename_expr(expr: ast.expr, mapping: dict[str, str]) -> str:
    copied = ast.parse(ast.unparse(expr), mode="eval").body
    return ast.unparse(_Renamer(mapping).visit(copied))


def _assigned_names(statements: Sequence[ast.stmt]) -> list[str]:
    names: list[str] = []
    for stmt in statements:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id not in names:
                    names.append(node.id)
    return names


def _emit_ring(
    match: _Match,
    *,
    rows: int,
    chunk: int,
    stages: int,
    split: str,
    threads: int,
    thread_axis: int,
    pdl: bool,
    new_var: Callable[[str], str],
) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """``(prologue, replacement)``: the kernel-level shared-memory allocation
    and the statements that replace the row lane loop (``pdl``: the kernel
    is a programmatic dependent and releases its own dependents after the
    ring's last MMA)."""
    es = 2
    warps = threads // 32
    bkb = chunk * es
    cpr = bkb // _PACKET_BYTES
    pk = rows * cpr
    pt = pk // threads
    ks = chunk // _K_STEP
    am = rows // _ATOM_ROWS
    a_stage = rows * bkb
    x_stage = bkb
    x_off = stages * a_stage
    red_off = x_off + stages * x_stage
    k0, k1 = match.k_loop.start, match.k_loop.stop
    k_extent = k1 - k0
    nchunk = -(-k_extent // chunk)
    k_bytes = k_extent * es
    if split == "k":
        ksw = ks // warps
        atoms = list(range(am))
    else:
        ksw = ks
        atoms = list(range(am // warps))
    helper = _MMA_HELPERS[match.element]
    v = new_var
    smem, sb = v("_ring_smem"), v("_ring_sb")
    tidx, warp, lane = v("_ring_tidx"), v("_ring_warp"), v("_ring_lane")
    g, c, l8, lq = v("_ring_g"), v("_ring_c"), v("_ring_l8"), v("_ring_lq")
    wg, xg, wrb = v("_ring_wg"), v("_ring_xg"), v("_ring_wrb")
    smem_bytes = ring_mma_smem_bytes(
        rows=rows,
        chunk=chunk,
        stages=stages,
        threads=threads,
        split=split,
        elem_bytes=es,
    )
    prologue = [
        f"{smem} = cute.arch.alloc_smem(cutlass.Uint8, {smem_bytes}, alignment=128)",
        f"{sb} = cutlass.Int32({smem}.toint())",
    ]
    L: list[str] = []
    emit = L.append
    emit(f"{tidx} = cutlass.Int32(cute.arch.thread_idx()[{thread_axis}])")
    if warps > 1:
        emit(f"{warp} = cute.arch.make_warp_uniform(cute.arch.warp_idx())")
    else:
        emit(f"{warp} = cutlass.Int32(0)")
    emit(f"{lane} = {tidx} % 32")
    emit(f"{g} = {lane} // 4")
    emit(f"{c} = {lane} % 4")
    emit(f"{l8} = {lane} % 8")
    emit(f"{lq} = {lane} // 8")
    emit(f"{wg} = {match.w_name}.iterator.toint()")
    emit(f"{xg} = {match.x_name}.iterator.toint()")
    emit(
        f"{wrb} = cutlass.Int64(cutlass.Int32({match.w_name}.layout.stride[{match.w_row_dim}])) * {es}"
    )
    sync = "cute.arch.barrier()" if warps > 1 else "cute.arch.sync_warp()"
    # ---- the thread's packets: packet ``p = tidx + j * threads`` is row
    # ``p // cpr``, 16-byte column ``p % cpr`` (a warp instruction reads one
    # row's contiguous bytes).  With whole rows per pass the column is one
    # per thread and the rows sit at immediate offsets; with whole passes per
    # row the rows are constants.
    r0, pc0 = v("_ring_r0"), v("_ring_pc0")
    emit(f"{r0} = {tidx} // {cpr}")
    emit(f"{pc0} = {tidx} % {cpr}")
    if threads % cpr == 0:
        rpp = threads // cpr
        row_exprs = [f"({r0} + {j * rpp})" if j else r0 for j in range(pt)]
        col_exprs = [pc0] * pt
    else:
        q = cpr // threads
        row_exprs = [str(j // q) for j in range(pt)]
        col_exprs = [
            f"({tidx} + {(j % q) * threads})" if j % q else tidx for j in range(pt)
        ]
    # ---- the row pointers: the row prologue and the row index statements
    # instantiated once per distinct packet row of the thread (a row past the
    # tile reads row 0; it is never stored).
    slice_names = _assigned_names([*match.pre, *match.row_index_slice])
    rowptr_of: dict[str, str] = {}
    for jr, row in enumerate(dict.fromkeys(row_exprs)):
        mapping = {name: v(f"{name}_p{jr}") for name in slice_names}
        mapping[match.row_var] = v(f"_ring_prow{jr}")
        emit(f"{mapping[match.row_var]} = {row}")
        L.extend(_instantiate([*match.pre, *match.row_index_slice], mapping, "", v))
        rowidx = v(f"_ring_rowidx{jr}")
        index_text = _rename_expr(match.row_index, mapping)
        if match.row_predicate is not None:
            pred_text = _rename_expr(match.row_predicate, mapping)
            emit(
                f"{rowidx} = cutlass.Int32({index_text}) if {pred_text} else cutlass.Int32(0)"
            )
        else:
            emit(f"{rowidx} = cutlass.Int32({index_text})")
        rowptr = v(f"_ring_rowptr{jr}")
        emit(
            f"{rowptr} = {wg} + cutlass.Int64({rowidx}) * {wrb} + cutlass.Int64({k0 * es})"
        )
        rowptr_of[row] = rowptr
    soffs: list[str] = []
    for j in range(pt):
        soff = v(f"_ring_soff{j}")
        swz = _swizzled_chunk(row_exprs[j], col_exprs[j], cpr)
        emit(f"{soff} = {row_exprs[j]} * {bkb} + {swz} * {_PACKET_BYTES}")
        soffs.append(soff)
    xpt = -(-cpr // threads)
    xidx: list[tuple[str, bool]] = []
    for jx in range(xpt):
        name = v(f"_ring_xi{jx}")
        emit(f"{name} = {tidx} + {jx * threads}" if jx else f"{name} = {tidx}")
        xidx.append((name, jx * threads + threads > cpr))
    x_base = v("_ring_xbase")
    emit(f"{x_base} = {xg} + cutlass.Int64({k0 * es})")

    def issue(
        chunk_text: str, slot: int, indent: str, static_chunk: int | None
    ) -> None:
        # Bytes of the chunk inside K (the last chunk may be partial).
        remaining = None
        if static_chunk is not None:
            remaining = min(bkb, k_bytes - static_chunk * bkb)
        partial = remaining is not None and remaining < bkb
        for j in range(pt):
            col = col_exprs[j]
            dst = f"{sb} + {slot * a_stage} + {soffs[j]}"
            src = (
                f"{rowptr_of[row_exprs[j]]} "
                f"+ cutlass.Int64(({chunk_text}) * {bkb} + {col} * {_PACKET_BYTES})"
            )
            if partial:
                assert remaining is not None
                nbytes = (
                    f"(cutlass.Int32({_PACKET_BYTES}) if {col} * {_PACKET_BYTES} + {_PACKET_BYTES} <= {remaining} "
                    f"else (cutlass.Int32({remaining}) - {col} * {_PACKET_BYTES} "
                    f"if {col} * {_PACKET_BYTES} < {remaining} else cutlass.Int32(0)))"
                )
                emit(f"{indent}_cute_ring_cp_async_16_zfill({dst}, {src}, {nbytes})")
            else:
                emit(f"{indent}_cute_ring_cp_async_16({dst}, {src})")
        for name, guard in xidx:
            inner = indent
            if guard:
                emit(f"{indent}if {name} < {cpr}:")
                inner = indent + "    "
            dst = f"{sb} + {x_off + slot * x_stage} + {name} * {_PACKET_BYTES}"
            src = f"{x_base} + cutlass.Int64(({chunk_text}) * {bkb} + {name} * {_PACKET_BYTES})"
            if partial:
                assert remaining is not None
                nbytes = (
                    f"(cutlass.Int32({_PACKET_BYTES}) if {name} * {_PACKET_BYTES} + {_PACKET_BYTES} <= {remaining} "
                    f"else (cutlass.Int32({remaining}) - {name} * {_PACKET_BYTES} "
                    f"if {name} * {_PACKET_BYTES} < {remaining} else cutlass.Int32(0)))"
                )
                emit(f"{inner}_cute_ring_cp_async_16_zfill({dst}, {src}, {nbytes})")
            else:
                emit(f"{inner}_cute_ring_cp_async_16({dst}, {src})")
        emit(f"{indent}cute.arch.cp_async_commit_group()")

    # ---- accumulators and fragment lane terms.
    accs = {a: [v(f"_ring_acc{a}_{i}") for i in range(4)] for a in atoms}
    for a in atoms:
        emit("; ".join(f"{name} = cutlass.Float32(0.0)" for name in accs[a]))
    arow, achunk = v("_ring_arow"), v("_ring_achunk")
    emit(f"{arow} = {l8} + 8 * ({lq} % 2)")
    emit(f"{achunk} = {lq} // 2")
    aoffs: dict[int, str] = {}
    kst0 = wam0 = ""
    if split == "k":
        kst0 = v("_ring_kst0")
        emit(f"{kst0} = {warp} * {ksw}")
        for a in atoms:
            aoffs[a] = v(f"_ring_aoff{a}")
            emit(f"{aoffs[a]} = ({a * _ATOM_ROWS} + {arow}) * {bkb}")
    else:
        wam0 = v("_ring_wam0")
        emit(f"{wam0} = {warp} * {len(atoms)}")
        for a in atoms:
            aoffs[a] = v(f"_ring_aoff{a}")
            emit(f"{aoffs[a]} = (({wam0} + {a}) * {_ATOM_ROWS} + {arow}) * {bkb}")
    # The ldmatrix column swizzle depends on the lane row only (atom rows are
    # multiples of 16): one term per K step of the warp.
    swzs: list[str] = []
    for st in range(ksw):
        st_expr = f"({kst0} + {st})" if split == "k" else str(st)
        name = v(f"_ring_swz{st}")
        emit(
            f"{name} = {_swizzled_chunk(arow, f'(2 * {st_expr} + {achunk})', cpr)} * {_PACKET_BYTES}"
        )
        swzs.append(name)
    boff = v("_ring_boff")
    emit(f"{boff} = {c} * 4")
    a_regs = [v(f"_ring_a{i}") for i in range(4)]
    b0, b1 = v("_ring_b0"), v("_ring_b1")

    def compute(slot: int, indent: str) -> None:
        a_slot = f"{sb} + {slot * a_stage}"
        x_slot = f"{sb} + {x_off + slot * x_stage}"
        for st in range(ksw):
            if split == "k":
                b_base = f"{x_slot} + ({kst0} + {st}) * 32 + {boff}"
            else:
                b_base = f"{x_slot} + {st * 32} + {boff}"
            emit(f"{indent}{b0} = _cute_ring_lds_b32({b_base})")
            emit(f"{indent}{b1} = _cute_ring_lds_b32({b_base} + 16)")
            for a in atoms:
                emit(
                    f"{indent}{', '.join(a_regs)} = _cute_ring_ldmatrix_x4("
                    f"{a_slot} + {aoffs[a]} + {swzs[st]})"
                )
                acc = ", ".join(accs[a])
                emit(
                    f"{indent}{acc} = {helper}({', '.join(a_regs)}, {b0}, {b1}, {acc})"
                )

    # ---- the ring: STAGES - 1 chunks ahead, a dynamic loop over whole groups
    # of STAGES chunks, then a peeled tail with static slot indices.
    for s in range(stages - 1):
        if s < nchunk:
            issue(str(s), s, "", s)
        else:
            emit("cute.arch.cp_async_commit_group()")
    tail = stages + (nchunk - stages) % stages if nchunk >= stages else nchunk
    main = nchunk - tail
    if main > 0:
        i = v("_ring_i")
        emit(
            f"for {i} in range(cutlass.Int32(0), cutlass.Int32({main}), cutlass.Int32({stages})):"
        )
        for s in range(stages):
            emit(f"    cute.arch.cp_async_wait_group({stages - 2})")
            emit(f"    {sync}")
            issue(f"{i} + {s + stages - 1}", (s + stages - 1) % stages, "    ", None)
            compute(s, "    ")
    for s in range(tail):
        ch = main + s
        emit(f"cute.arch.cp_async_wait_group({stages - 2})")
        emit(sync)
        if ch + stages - 1 < nchunk:
            issue(str(ch + stages - 1), (s + stages - 1) % stages, "", ch + stages - 1)
        else:
            emit("cute.arch.cp_async_commit_group()")
        compute(s % stages, "")
    if pdl:
        # The last MMA is issued: let the next kernel launch while this CTA
        # runs its epilogue (its own wait orders it after our stores).
        emit(f"{_PDL_RELEASE}()")
    # ---- epilogue: every row's value through the row prologue (which sets
    # the accumulator's start) and the stores.
    # The stores need the accumulator's start and whatever row prologue
    # values they read (the row index, a bias): that backward slice of the
    # prologue, not the row gather the packets needed.
    pre_defs = {
        name: stmt for stmt in match.pre if (name := _assigned_name(stmt)) is not None
    }
    needed_by_post = _closure(
        {match.acc} | {name for stmt in match.post for name in _reads(stmt)},
        {name: stmt.value for name, stmt in pre_defs.items()},  # type: ignore[attr-defined]
        set(),
    )
    epilogue_pre = [
        stmt for stmt in match.pre if _assigned_name(stmt) in needed_by_post
    ]
    post_names = _assigned_names([*epilogue_pre, *match.post])

    def epilogue(row_text: str, acc_text: str, tag: str, indent: str) -> None:
        mapping = {name: v(f"{name}_{tag}") for name in post_names}
        mapping[match.row_var] = v(f"_ring_row_{tag}")
        emit(f"{indent}{mapping[match.row_var]} = {row_text}")
        L.extend(_instantiate(epilogue_pre, mapping, indent, v))
        acc_name = mapping[match.acc]
        emit(f"{indent}{acc_name} = {acc_name} + {acc_text}")
        L.extend(_instantiate(match.post, mapping, indent, v))

    if split == "k":
        red = v("_ring_red")
        emit(f"{red} = {sb} + {red_off} + {warp} * {rows * 4}")
        emit(f"if {c} == 0:")
        for a in atoms:
            emit(
                f"    _cute_ring_sts_f32({red} + ({a * _ATOM_ROWS} + {g}) * 4, {accs[a][0]})"
            )
            emit(
                f"    _cute_ring_sts_f32({red} + ({a * _ATOM_ROWS + 8} + {g}) * 4, {accs[a][2]})"
            )
        emit(sync)
        for jr in range(-(-rows // threads)):
            row = f"({tidx} + {jr * threads})" if jr else tidx
            indent = ""
            if jr * threads + threads > rows:
                emit(f"if {row} < {rows}:")
                indent = "    "
            tot = v(f"_ring_tot{jr}")
            emit(f"{indent}{tot} = _cute_ring_lds_f32({sb} + {red_off} + {row} * 4)")
            for w in range(1, warps):
                emit(
                    f"{indent}{tot} = {tot} + _cute_ring_lds_f32({sb} + {red_off + w * rows * 4} + {row} * 4)"
                )
            epilogue(row, tot, f"e{jr}", indent)
    else:
        emit(f"if {c} == 0:")
        for a in atoms:
            row_lo = f"(({wam0} + {a}) * {_ATOM_ROWS} + {g})"
            epilogue(row_lo, accs[a][0], f"e{a}lo", "    ")
            epilogue(f"({row_lo} + 8)", accs[a][2], f"e{a}hi", "    ")
    prologue_stmts = ast.parse("\n".join(prologue)).body
    body_stmts = ast.parse(textwrap.dedent("\n".join(L))).body
    return prologue_stmts, body_stmts


def _candidate_row_loops(
    statements: list[ast.stmt], *, rows: int, chunk: int, k_block_var: str | None
) -> list[tuple[list[ast.stmt], int]]:
    """``(parent list, index)`` of every lane loop of ``rows`` trips holding a
    K device loop of ``chunk`` steps (a structural scan that touches nothing)."""
    found: list[tuple[list[ast.stmt], int]] = []
    for index, stmt in enumerate(statements):
        if (
            _is_lane_loop(stmt)
            and isinstance(stmt, ast.For)
            and _plain_range_extent(stmt) == rows
            and any(
                _match_device_loop(inner, k_block_var, chunk) is not None
                for inner in stmt.body
            )
        ):
            found.append((statements, index))
            continue
        if isinstance(stmt, (ast.For, ast.While, ast.If, ast.With)):
            for field in ("body", "orelse"):
                nested = getattr(stmt, field, None)
                if nested:
                    found.extend(
                        _candidate_row_loops(
                            nested, rows=rows, chunk=chunk, k_block_var=k_block_var
                        )
                    )
    return found


def _attributed_copy(stmt: ast.stmt) -> ast.stmt | None:
    """A plain-AST copy of ``stmt`` carrying the lane-loop markers of its
    loops (the kernel's extended nodes are not deep-copyable; the match works
    on the copy so a declined candidate is never touched)."""
    copied = ast.parse(ast.unparse(stmt)).body[0]
    originals = [node for node in ast.walk(stmt) if isinstance(node, ast.For)]
    copies = [node for node in ast.walk(copied) if isinstance(node, ast.For)]
    if len(originals) != len(copies):
        return None
    for original, copy_node in zip(originals, copies, strict=True):
        if ast.unparse(original.target) != ast.unparse(copy_node.target):
            return None
        marker = getattr(original, HELION_LANE_LOOP_VAR_ATTR, None)
        if marker:
            setattr(copy_node, HELION_LANE_LOOP_VAR_ATTR, marker)
    return copied


def ring_mma_dot(
    body: list[ast.stmt],
    *,
    thread_block_dims: tuple[int, int, int] | None,
    rows: int | None,
    chunk: int | None,
    stages: int,
    split: str,
    k_block_var: str | None,
    tensors: dict[str, torch.Tensor],
    aligned: set[str],
    new_var: Callable[[str], str],
    rename_groups: dict[str, str] | None = None,
    smem_budget: int = RING_SMEM_BUDGET,
) -> tuple[list[ast.stmt], bool]:
    """Rewrite the matching GEMV row loop of ``body`` onto the cp.async ring;
    ``(body, fired)``.  Declines (``fired`` False, ``body`` untouched) when
    the thread block is not ``(T, 1, 1)``, when the geometry cannot run the
    ring, when the body does not hold exactly one candidate row loop or when
    that loop does not match.  ``tensors`` maps the kernel's tensor argument
    names to their fake tensors, ``aligned`` names those whose base is
    provably 16-byte aligned, ``rename_groups`` maps variable names to the
    canonical names the final print gives them, and ``smem_budget`` is the
    device's per-CTA shared memory."""
    if thread_block_dims is None or rows is None or chunk is None:
        return body, False
    # The K threads are the one non-unit thread axis (the generic lowering
    # drops unit axes, so a ``num_threads`` of ``[1, T]`` is a ``(T, 1, 1)``
    # block, but the rewrite also takes the axis where it stands).
    axes = [i for i, dim in enumerate(thread_block_dims) if dim != 1]
    if len(axes) > 1:
        return body, False
    thread_axis = axes[0] if axes else 0
    threads = thread_block_dims[thread_axis]
    if (
        ring_mma_geometry_reason(
            rows=rows,
            chunk=chunk,
            threads=threads,
            stages=stages,
            split=split,
            elem_bytes=2,
            k_width=None,
            smem_budget=smem_budget,
        )
        is not None
    ):
        return body, False
    candidates = _candidate_row_loops(
        body, rows=rows, chunk=chunk, k_block_var=k_block_var
    )
    if len(candidates) != 1:
        _no(f"{len(candidates)} candidate row loops")
        return body, False
    parent, index = candidates[0]
    candidate = _attributed_copy(parent[index])
    if candidate is None:
        return body, False
    if rename_groups:
        # The final print maps every variable to its rename group's canonical
        # name; the match reads the names as they will print.
        _Renamer(rename_groups).visit(candidate)
    match = _match_row_loop(
        candidate,
        rows=rows,
        chunk=chunk,
        k_block_var=k_block_var,
        tensors=tensors,
        aligned=aligned,
        thread_axis=thread_axis,
    )
    if match is None:
        return body, False
    pdl = _uses_pdl(body)
    smem_stmts, replacement = _emit_ring(
        match,
        rows=rows,
        chunk=chunk,
        stages=stages,
        split=split,
        threads=threads,
        thread_axis=thread_axis,
        pdl=pdl,
        new_var=new_var,
    )
    # The match holds: only now is anything rewritten.  The kernel's own
    # list is copied; a nested parent list is spliced in place.
    rewritten = _without_entry_release(list(body)) if pdl else list(body)
    target = parent[index]
    if parent is body:
        position = next(i for i, stmt in enumerate(rewritten) if stmt is target)
        rewritten[position : position + 1] = replacement
    else:
        parent[index : index + 1] = replacement
    return [*smem_stmts, *rewritten], True

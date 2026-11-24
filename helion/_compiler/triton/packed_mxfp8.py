"""Triton lowering of packed MXFP8 ``hl.dot_scaled`` on sm100/sm103.

A packed MXFP8 GEMM loads E4M3 operands and int32 words holding four E8M0
scale bytes, and extracts each K group's byte in the kernel
(``mxfp8.match_packed_scale``).  The generic lowering loads the scale words
once per byte and leaves every operand on pointer loads.  Here the operands
and scale words are loaded through host tensor descriptors instead, the words
are unpacked once into the order tcgen05's UTCCP scale copy expects, and a
16- or 32-row tile is issued as the transposed MMA (``swap``).

Each rewrite is used only when it reads exactly the values of the load it
replaces:

* every descriptor offset is the start of an ``iota + scalar`` index,
* every mask on an operand only clips lanes outside the tensor (a descriptor
  zero fills those) -- a bound at or past the extent, or the end of a loop
  whose tiles never cross it before the end of the tensor.  That includes
  the tile masks codegen adds itself, on the load and in the ``_mask_to`` a
  dot puts on its operands, for a dimension the size of a loop's tile,
* the K tiles of an unpacked scale start on a word (a multiple of 4 groups),
  and the scale lanes the word load's masks clip (its own and its tile
  masks) are zeroed after unpacking, as the generic load's ``other=0`` does.

These are proved for the config being compiled: a split-K kernel's partition
tunable decides several of them.

Inside a warp-specialized loop the dot is lowered natively only as a whole,
every operand and scale word through a descriptor: a pointer-loaded operand
beside descriptor loads and the TMA output store gives wrong,
nondeterministic results there (Triton 3.9).  A config whose proofs do not
cover the whole dot keeps the generic loads in such a loop (still issued as
the transposed MMA when it swaps), and its output keeps pointer stores.
``lowers_whole`` runs the same proofs for a seed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import functools
import math
import operator
from typing import TYPE_CHECKING
from typing import cast
import weakref

import sympy
import torch

from ..._compat import tensor_descriptor_supports_dtype
from ...language import _tracing_ops
from ...language import view_ops
from ...language.atomic_ops import ATOMIC_OPS
from ...language.distributed_ops import remote_barrier
from ...language.distributed_ops import start_async_remote_copy_descriptor
from ...language.distributed_ops import wait_async_remote_copy
from ...language.distributed_ops import wait_recv_async_remote_copy
from ...language.distributed_ops import wait_send_async_remote_copy
from ...language.inline_asm_ops import inline_asm_elementwise
from ...language.inline_triton_ops import inline_triton
from ...language.pdl_ops import pdl_launch_dependents
from ...language.pdl_ops import pdl_wait
from ..ast_extension import expr_from_string
from ..compile_environment import CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE
from ..compile_environment import CompileEnvironment
from ..host_function import HostFunction
from ..indexing_strategy import _contiguous_integer_tensor_index
from ..indexing_strategy import _contiguous_integer_tensor_index_base
from ..indexing_strategy import _in_warp_specialized_loop
from ..loop_dependency_checker import _THROUGH_POINTERS
from ..loop_dependency_checker import _storage_id
from ..loop_dependency_checker import graph_accesses
from ..matmul_utils import _resolve_dim_size
from ..mxfp8 import PackedMxfp8Operand
from ..mxfp8 import PackedScaleMatch
from ..mxfp8 import find_packed_mxfp8_dot
from ..mxfp8 import native_packed_mxfp8_kernel
from ..mxfp8 import native_packed_mxfp8_target
from ..mxfp8 import packed_mxfp8_operands
from ..node_masking import skip_copies
from ..variable_origin import BlockSizeOrigin
from ..variable_origin import GridOrigin
from ..variable_origin import TensorSizeOrigin
from ..variable_origin import TileBeginOrigin

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Sequence

    from ...runtime.config import Config
    from ..device_function import DeviceFunction
    from ..device_ir import DeviceIR
    from ..device_ir import GraphInfo
    from ..inductor_lowering import CodegenState

# All-127 bytes: E8M0 scale 1.0 for padded words outside the scale tensor.
_UNIT_SCALE_WORDS = 0x7F7F7F7F
# A TMA store stages its whole tile in shared memory beside the main loop's
# operand ring: at most 64 KiB (a 128x256 bf16 tile), and with the ring's
# stages within the shared memory of an sm100 block.
_MAX_STORE_TILE_BYTES = 64 * 1024
# The shared memory an sm100 block can allocate.
SM100_SHARED_MEMORY_BYTES = 227 * 1024


def operand_ring_stage_bytes(tiles: Iterable[tuple[int, int]]) -> int:
    """Shared memory one stage of a packed MXFP8 dot's operand ring holds:
    for each operand tile of ``(rows, groups)``, its E4M3 values (32 bytes
    per row and group) and its E8M0 scale bytes (one per row and group)."""
    return sum(rows * groups * 33 for rows, groups in tiles)


@dataclass(frozen=True)
class PackedMxfp8Dot:
    """Replacements for a packed MXFP8 ``dot_scaled``'s argument ASTs.

    A ``None`` operand keeps its generic load.  ``swap`` issues the dot as
    ``(rhs^T @ lhs^T)^T`` so a 16- or 32-row tile fills tcgen05's 128-row M.
    """

    lhs: ast.AST | None
    lhs_scale: ast.AST | None
    rhs: ast.AST | None
    rhs_scale: ast.AST | None
    swap: bool


@dataclass(frozen=True)
class _Operand:
    """A matched operand of the dot and the size of its unpacked scale tile
    in the config compiled: operand rows by K groups."""

    match: PackedMxfp8Operand
    tile: tuple[int, int]

    @property
    def load(self) -> torch.fx.Node:
        return self.match.load

    @property
    def tensor(self) -> torch.Tensor:
        return self.match.tensor

    @property
    def indices(self) -> tuple[object, object]:
        return self.match.indices

    @property
    def scale(self) -> PackedScaleMatch:
        return self.match.scale

    @property
    def scale_value(self) -> torch.Tensor:
        """The unpacked scale tile passed to the dot."""
        return self.match.scale_value


@dataclass(frozen=True)
class _Proof:
    """What the proofs read: the config compiled, its graphs, and whether
    codegen masks a block's tile past the end of its loop."""

    config: Config
    graphs: Sequence[GraphInfo]
    tile_masked: Callable[[int], bool]


@dataclass(frozen=True)
class _Plan:
    """The parts of a packed MXFP8 dot the proofs let codegen lower natively."""

    lhs: _Operand
    rhs: _Operand
    swap: bool
    # How each scale's packed words load ("descriptor" or "pointer"); None
    # keeps the generic scale loads, and then the operands' loads too.
    lhs_scale: str | None
    rhs_scale: str | None
    # Whether each operand loads through a descriptor.
    lhs_operand: bool
    rhs_operand: bool

    @property
    def whole(self) -> bool:
        """Every operand and scale word through a descriptor."""
        return (
            self.lhs_operand
            and self.rhs_operand
            and self.lhs_scale == "descriptor"
            and self.rhs_scale == "descriptor"
        )


def _codegen_proof(state: CodegenState) -> _Proof:
    codegen = state.codegen
    return _Proof(
        state.config,
        codegen.codegen_graphs,
        lambda block_id: codegen.mask_var(block_id) is not None,
    )


def packed_mxfp8_dot(state: CodegenState) -> PackedMxfp8Dot | None:
    """Lower the ``dot_scaled`` being generated natively, where proved."""
    # The bound kernel's alignment keys and the function's async-proxy fences
    # cover the tensors of the kernel's only packed dot.  A prologue fused
    # under torch.compile reaches the generic loads only (``load_transform``).
    if (
        not native_packed_mxfp8_kernel(
            CompileEnvironment.current(), HostFunction.current().device_ir.graphs
        )
        or state.codegen.load_transform is not None
    ):
        return None
    plan = _plan(_codegen_proof(state), state.fx_node)
    if plan is None:
        return None
    # The generic loads, without a descriptor load beside them.
    generic = PackedMxfp8Dot(None, None, None, None, plan.swap)
    warp_specialized = _in_warp_specialized_loop(state)
    if warp_specialized and not plan.whole:
        return generic
    lhs_scale = _packed_scale_ast(state, plan.lhs, plan.lhs_scale, prefix="mxfp8_lhs")
    rhs_scale = _packed_scale_ast(state, plan.rhs, plan.rhs_scale, prefix="mxfp8_rhs")
    if lhs_scale is None or rhs_scale is None:
        return generic
    lhs_operand = _operand_ast(state, plan.lhs) if plan.lhs_operand else None
    rhs_operand = _operand_ast(state, plan.rhs) if plan.rhs_operand else None
    whole = plan.whole and lhs_operand is not None and rhs_operand is not None
    if warp_specialized and not whole:
        return generic
    # Only a dot loaded whole through descriptors lets its output take a TMA
    # store (``flat_output_descriptor_store``), which stages its tile beside
    # the operand ring: per stage an operand tile and the scale words of each.
    state.device_function.packed_mxfp8_whole_dot = whole
    state.device_function.packed_mxfp8_stage_bytes = operand_ring_stage_bytes(
        (plan.lhs.tile, plan.rhs.tile)
    )
    if rhs_operand is not None:
        rhs_operand = expr_from_string("tl.permute({value}, [1, 0])", value=rhs_operand)
    return PackedMxfp8Dot(lhs_operand, lhs_scale, rhs_operand, rhs_scale, plan.swap)


def lowers_whole(device_ir: DeviceIR, config: Config) -> bool:
    """Whether codegen of ``config`` lowers the kernel's packed MXFP8 dot as a
    whole (``_Plan.whole``), which warp-specializing its loop requires.

    Runs codegen's proofs on the device IR, taking every block a loop tiles
    as masked past the loop's end.
    """
    from ..device_ir import ForLoopGraphInfo

    env = CompileEnvironment.current()
    host_function = device_ir.host_function
    # A prologue fused under torch.compile keeps the generic loads.
    if (
        host_function is None
        or not native_packed_mxfp8_target(env)
        or env.settings.torch_compile_fusion
    ):
        return False
    looped = {
        block_id
        for info in device_ir.graphs
        if isinstance(info, ForLoopGraphInfo)
        for block_id in info.block_ids
    }
    looped.update(
        block_id
        for block_ids in device_ir.grid_block_ids
        for block_id in block_ids
        if env.block_sizes[block_id].from_config(config) != 1
    )
    looped.update(info.block_id for info in env.block_sizes if info.reduction)
    canonical = {env.canonical_block_id(block_id) for block_id in looped}
    proof = _Proof(
        config,
        device_ir.graphs,
        lambda block_id: env.canonical_block_id(block_id) in canonical,
    )
    with host_function:
        dot = find_packed_mxfp8_dot(device_ir.graphs)
        plan = None if dot is None else _plan(proof, dot)
        return plan is not None and plan.whole


# ``render_drops_whole_dot`` per environment, by config and call facts.
_RENDER_DROPS_WHOLE: weakref.WeakKeyDictionary[
    CompileEnvironment, dict[tuple[object, ...], bool]
] = weakref.WeakKeyDictionary()


def render_drops_whole_dot(config: Config) -> bool:
    """Whether the code being generated serves an Inductor graph's calls
    (``inductor_call_facts``: inputs aligned only to ``GPU_ALIGN_BYTES``, or
    no dispatch guards and strides the trace may not have seen) under which
    the kernel's packed MXFP8 dot is not lowered whole (``lowers_whole``),
    though the binding's own launches lower it whole.

    The binding's seed warp-specializes a loop, such as an outer persistent
    one, that Triton cannot compile around the dot's generic loads only
    where the dot is lowered whole, so such a render's loops are not
    warp-specialized (``TritonBackend.range_warp_specializes``).  Where the
    binding's own code keeps the generic loads too, the render emits that
    same code and keeps its warp specialization.  False outside a compile
    environment (a backend asked about a config alone).
    """
    if not CompileEnvironment.has_current():
        return False
    env = CompileEnvironment.current()
    if env.dispatch_guards_enforced and env.input_alignment_limit is None:
        return False
    key = (
        repr(config),
        env.dispatch_guards_enforced,
        env.input_alignment_limit,
        env.symbolic_stride_sources,
    )
    cache = _RENDER_DROPS_WHOLE.setdefault(env, {})
    if key not in cache:
        device_ir = HostFunction.current().device_ir
        if not native_packed_mxfp8_kernel(env, device_ir.graphs):
            cache[key] = False
        else:
            with env.binding_call_facts():
                binding_whole = lowers_whole(device_ir, config)
            cache[key] = binding_whole and not lowers_whole(device_ir, config)
    return cache[key]


def _plan(proof: _Proof, dot: object) -> _Plan | None:
    operands = _match_operands(proof, dot)
    if operands is None:
        return None
    lhs, rhs = operands
    # The native loads are issued at the dot, not where the loads they
    # replace stand: nothing of the dot's phase may write what they read, nor
    # synchronize with other threads.
    written = _phase_writes(proof, cast("torch.fx.Node", dot))
    if (
        written is None
        or _phase_synchronizes(proof, cast("torch.fx.Node", dot))
        or _THROUGH_POINTERS in written
        or any(
            _storage_id(tensor) in written
            for tensor in (lhs.tensor, rhs.tensor, lhs.scale.tensor, rhs.scale.tensor)
        )
    ):
        return None
    swap = lhs.tile[0] in (16, 32) and rhs.tile[0] >= 128
    lhs_scale, rhs_scale = (_scale_source(proof, operand) for operand in operands)
    if lhs_scale is None or rhs_scale is None:
        return _Plan(lhs, rhs, swap, None, None, False, False)
    return _Plan(
        lhs,
        rhs,
        swap,
        lhs_scale,
        rhs_scale,
        _operand_descriptor(proof, lhs),
        _operand_descriptor(proof, rhs),
    )


# Ops that may order a thread's memory accesses against other threads': the
# native loads must not move past one (``_phase_synchronizes``).
_SYNCHRONIZING_OPS = frozenset(
    {
        *ATOMIC_OPS,
        remote_barrier,
        start_async_remote_copy_descriptor,
        wait_async_remote_copy,
        wait_send_async_remote_copy,
        wait_recv_async_remote_copy,
        pdl_wait,
        pdl_launch_dependents,
        inline_asm_elementwise,
        inline_triton,
    }
)


def _phase_root(proof: _Proof, graph_id: int) -> int:
    """The root graph (phase) that runs graph ``graph_id``."""
    from ..device_ir import control_flow_parent_ids

    parents = control_flow_parent_ids(proof.graphs)
    while graph_id in parents:
        graph_id = parents[graph_id]
    return graph_id


def _node_graph_id(proof: _Proof, node: torch.fx.Node) -> int | None:
    return next(
        (info.graph_id for info in proof.graphs if info.graph is node.graph), None
    )


def _phase_writes(proof: _Proof, node: torch.fx.Node) -> set[int] | None:
    """The storages the root graph (phase) holding ``node`` writes, with
    ``_THROUGH_POINTERS`` for a write through device pointers (a stack
    tensor's, which may reach any storage)."""
    graph_id = _node_graph_id(proof, node)
    if graph_id is None:
        return None
    phase = _phase_root(proof, graph_id)
    return graph_accesses(list(proof.graphs), phase).writes().storages()


def _phase_synchronizes(proof: _Proof, node: torch.fx.Node) -> bool:
    """Whether the phase holding ``node`` has an op that may synchronize with
    other threads (``_SYNCHRONIZING_OPS``: an atomic, a remote barrier or
    copy, a PDL wait, inline code), which a load moved to the dot may cross:
    a value read after a release another thread acquires may already be
    overwritten, one read after an acquire may be newer than the program's.
    True when ``node``'s graph is not known."""
    graph_id = _node_graph_id(proof, node)
    if graph_id is None:
        return True
    phase = _phase_root(proof, graph_id)
    return any(
        other.target in _SYNCHRONIZING_OPS
        for info in proof.graphs
        if _phase_root(proof, info.graph_id) == phase
        for other in info.graph.nodes
    )


def _match_operands(proof: _Proof, dot: object) -> tuple[_Operand, _Operand] | None:
    """The operands of a packed MXFP8 dot, with scale tiles of one K extent
    in the config compiled."""
    matched = packed_mxfp8_operands(dot)
    if matched is None:
        return None
    operands = []
    for operand in matched:
        rows, groups = (_configured(proof, size) for size in operand.scale_value.shape)
        if not (
            isinstance(rows, sympy.Integer)
            and isinstance(groups, sympy.Integer)
            and rows > 0
            and groups > 0
        ):
            return None
        operands.append(_Operand(operand, (int(rows), int(groups))))
    lhs, rhs = operands
    return (lhs, rhs) if lhs.tile[1] == rhs.tile[1] else None


# -- proofs ----------------------------------------------------------------


def _expr(value: object) -> sympy.Expr | None:
    if isinstance(value, torch.fx.Node):
        value = value.meta.get("val")
    if isinstance(value, torch.SymInt):
        value = value._sympy_()
    if type(value) is int:
        return sympy.Integer(value)
    if isinstance(value, sympy.Expr):
        return CompileEnvironment.current().shape_env.replace(value)
    return None


def static_extent(value: object) -> int | None:
    """A tensor size or stride every input of this bound kernel shares."""
    expr = _expr(value)
    if expr is None:
        return None
    expr = CompileEnvironment.current().specialize_expr(expr)
    return int(expr) if isinstance(expr, sympy.Integer) else None


def _tensor_extent(value: object) -> sympy.Expr | None:
    """A tensor dimension, specialized when the bound kernel keys on it."""
    expr = _expr(value)
    if expr is None:
        return None
    return CompileEnvironment.current().specialize_expr(expr)


def _configured(proof: _Proof, value: object) -> sympy.Expr | None:
    """``value`` with the block sizes and tunables of the config compiled."""
    expr = _expr(value)
    if expr is None:
        return None
    env = CompileEnvironment.current()
    origins = HostFunction.current().expr_to_origin
    expr = env.specialize_expr(expr)
    substitutions: dict[sympy.Basic, sympy.Basic] = {}
    for symbol in expr.free_symbols:
        origin = origins.get(symbol)
        if symbol in env.config_value_expressions:
            substitutions[symbol] = sympy.Integer(
                env.config_value_expressions[symbol].evaluate(proof.config)
            )
        elif origin is not None and isinstance(origin.origin, BlockSizeOrigin):
            size = _block_size(proof, origin.origin.block_id)
            if size is not None:
                substitutions[symbol] = sympy.Integer(size)
    # Grid and tile-begin symbols stay free: a proof holds for every tile.
    return cast("sympy.Expr", expr.xreplace(substitutions))


def _known_nonnegative(expr: sympy.Expr) -> bool:
    return sympy.simplify(expr).is_nonnegative is True


def _known_nonnegative_index(expr: sympy.Expr) -> bool:
    """``_known_nonnegative`` with the grid indices and tile begins of loops
    whose coordinates start at zero and step by one (root or device loops
    outside ``noncanonical_task_origin_block_ids``) taken as nonnegative."""
    host_function = HostFunction.current()
    noncanonical = host_function.device_ir.noncanonical_task_origin_block_ids
    nonnegative: dict[sympy.Basic, sympy.Basic] = {}
    for symbol in expr.free_symbols:
        origin = host_function.expr_to_origin.get(symbol)
        if (
            origin is not None
            and isinstance(origin.origin, GridOrigin)
            and origin.origin.block_id not in noncanonical
        ):
            nonnegative[symbol] = sympy.Dummy(integer=True, nonnegative=True)
    return _known_nonnegative(cast("sympy.Expr", expr.xreplace(nonnegative)))


def _known_multiple(expr: sympy.Expr, divisor: int) -> bool:
    return sympy.Mod(sympy.expand(expr), divisor) == 0


def _parent(proof: _Proof, graph: torch.fx.Graph) -> tuple[torch.fx.Node, int] | None:
    """The one control-flow node capturing ``graph``'s inputs, and its slot."""
    # device_ir imports this backend's codegen modules; import it on use.
    from ..device_ir import control_flow_child_graph_ids
    from ..device_ir import control_flow_parent_entries

    graphs = proof.graphs
    info = next((info for info in graphs if info.graph is graph), None)
    if info is None:
        return None
    references = sum(
        control_flow_child_graph_ids(node).count(info.graph_id)
        for other in graphs
        for node in other.graph.nodes
    )
    if references != 1:
        return None
    return control_flow_parent_entries(graphs).get(info.graph_id)


def _definition(proof: _Proof, node: object) -> object:
    """Follow ``_new_var`` copies and loop-invariant captures to a value's
    definition in an enclosing graph."""
    for _ in range(8):
        node = skip_copies(node)
        if not isinstance(node, torch.fx.Node) or node.op != "placeholder":
            return node
        entry = _parent(proof, node.graph)
        if entry is None:
            return node
        parent, slot = entry
        captured = parent.args[slot]
        position = list(node.graph.find_nodes(op="placeholder")).index(node)
        if not isinstance(captured, (list, tuple)) or position >= len(captured):
            return node
        source = captured[position]
        # A loop carry rebinds the placeholder after the first iteration.
        if isinstance(source, torch.fx.Node) and any(
            user.target is _tracing_ops._phi
            and len(user.args) == 2
            and user.args[0] is source
            for user in source.users
        ):
            return node
        node = source
    return node


def _contiguous(proof: _Proof, index: object) -> tuple[sympy.Expr, sympy.Expr] | None:
    """``(start, extent)`` of an ``iota(extent) + start`` index vector."""
    definition = _definition(proof, index)
    value = (
        definition.meta.get("val") if isinstance(definition, torch.fx.Node) else None
    )
    if not isinstance(value, torch.Tensor):
        return None
    info = _contiguous_integer_tensor_index(value, definition)
    if info is None:
        return None
    start: sympy.Expr = sympy.Integer(0)
    for sign, term in info.base_terms:
        term_expr = _expr(term)
        if term_expr is None:
            # A device-produced (tensor) base: contiguous, but not affine in
            # the loop symbols this module reasons about.
            return None
        start = sympy.Add(start, sympy.Mul(sign, term_expr))
    extent = _expr(info.extent)
    return (start, extent) if extent is not None else None


def _loop_bounds(proof: _Proof, block_id: int) -> tuple[sympy.Expr, sympy.Expr] | None:
    """The configured begin and end of the device loop over ``block_id``."""
    from ..device_ir import ForLoopGraphInfo
    from ..device_ir import device_loop_bounds

    loops = [
        info
        for info in proof.graphs
        if isinstance(info, ForLoopGraphInfo) and block_id in info.block_ids
    ]
    if len(loops) != 1:
        return None
    entry = _parent(proof, loops[0].graph)
    if entry is None:
        return None
    bounds = device_loop_bounds(loops[0], {loops[0].graph_id: entry[0]}, block_id)
    if bounds is None:
        return None
    begin, end = (_configured(proof, value) for value in bounds)
    return (begin, end) if begin is not None and end is not None else None


def _block_size(proof: _Proof, block_id: int) -> int | None:
    size = CompileEnvironment.current().block_sizes[block_id].from_config(proof.config)
    return size if type(size) is int else None


def _tile_begins(start: sympy.Expr) -> tuple[int, int] | None:
    """``(block_id, c)`` when ``start`` is ``c * tile.begin`` of a K tile."""
    origins = HostFunction.current().expr_to_origin
    for symbol in start.free_symbols:
        origin = origins.get(symbol)
        if origin is None or not isinstance(origin.origin, TileBeginOrigin):
            continue
        coefficient = sympy.simplify(sympy.Mul(start, sympy.Pow(symbol, -1)))
        if coefficient.is_Integer and int(coefficient) > 0:
            return origin.origin.block_id, int(coefficient)
    return None


def _tile_begin_symbol(start: sympy.Expr, block_id: int) -> sympy.Symbol | None:
    """The ``tile.begin`` of ``block_id``'s loop among ``start``'s symbols."""
    env = CompileEnvironment.current()
    origins = HostFunction.current().expr_to_origin
    for symbol in start.free_symbols:
        origin = origins.get(symbol)
        if (
            isinstance(symbol, sympy.Symbol)
            and origin is not None
            and isinstance(origin.origin, TileBeginOrigin)
            and env.canonical_block_id(origin.origin.block_id)
            == env.canonical_block_id(block_id)
        ):
            return symbol
    return None


def _loop_end_outside(
    begin: sympy.Expr,
    end: sympy.Expr,
    size: int,
    index_at: Callable[[sympy.Expr], sympy.Expr],
    extent: int | sympy.Expr,
) -> bool:
    """Whether every arm of a loop's end (a minimum of arms) ends a whole
    number of tiles after ``begin``, so no tile crosses it, or puts the index
    there (``index_at(arm)``) at or past ``extent``."""
    arms = cast(
        "tuple[sympy.Expr, ...]", end.args if isinstance(end, sympy.Min) else (end,)
    )
    return all(
        _known_multiple(sympy.Add(arm, sympy.Mul(-1, begin)), size)
        or _known_nonnegative(sympy.Add(index_at(arm), -extent))
        for arm in arms
    )


def _bound_clips_only_outside(
    proof: _Proof, index: object, bound: object, extent: int | sympy.Expr
) -> bool:
    """Whether ``index < bound`` is false only where ``index >= extent``,
    on every tile the config executes: a bound at least ``extent`` (a
    symbolic one, of a dimension the bound kernel does not specialize, when
    the bound is that same extent), or the end of a split-K partition no
    tile crosses."""
    configured = _configured(proof, bound)
    if configured is None:
        return False
    if _known_nonnegative(sympy.Add(configured, -extent)):
        return True
    # A split-K partition ends inside the tensor: ``k < c * loop_end`` for an
    # index ``c * tile.begin + iota(c * block)`` of that loop is outside the
    # loop's own tiles unless a tile crosses the partition end.
    contiguous = _contiguous(proof, index)
    if contiguous is None:
        return False
    start, width = (_configured(proof, value) for value in contiguous)
    tile = None if start is None else _tile_begins(start)
    if tile is None:
        return False
    block_id, scale = tile
    size = _block_size(proof, block_id)
    bounds = _loop_bounds(proof, block_id)
    if size is None or bounds is None or width != scale * size:
        return False
    begin, end = bounds
    if sympy.simplify(sympy.Add(configured, sympy.Mul(-scale, end))) != 0:
        return False
    return _loop_end_outside(
        begin, end, size, lambda arm: sympy.Mul(scale, arm), extent
    )


def _tile_end_outside(
    proof: _Proof, index: object, block_id: int, extent: int | sympy.Expr
) -> bool:
    """Whether the lanes of ``index`` (``tile.index`` of ``block_id``'s loop
    plus an offset) past the loop's end lie at or past ``extent``."""
    contiguous = _contiguous(proof, index)
    size = _block_size(proof, block_id)
    bounds = _loop_bounds(proof, block_id)
    if contiguous is None or size is None or bounds is None:
        return False
    start = _configured(proof, contiguous[0])
    begin_symbol = None if start is None else _tile_begin_symbol(start, block_id)
    if start is None or begin_symbol is None:
        return False
    offset = sympy.expand(sympy.Add(start, sympy.Mul(-1, begin_symbol)))
    if begin_symbol in offset.free_symbols:
        return False
    begin, end = bounds
    return _loop_end_outside(
        begin, end, size, lambda arm: sympy.Add(arm, offset), extent
    )


def _tile_masks_clip_only_outside(
    proof: _Proof,
    sizes: Sequence[int | torch.SymInt],
    indices: tuple[object, object],
    shape: tuple[int | sympy.Expr, int | sympy.Expr],
) -> bool:
    """Whether the tile masks of a load's dimensions only clip lanes outside
    the tensor.

    Generic codegen masks a dimension the size of a loop's tile with that
    tile's mask (``tile.index < end``), on the load and in the ``_mask_to``
    a dot puts on its operands, which ``unmask_zero_operand`` drops.
    """
    env = CompileEnvironment.current()
    for dim, size in enumerate(sizes):
        for block_id in (env.get_block_id(size), env.resolve_block_id(size)):
            if (
                block_id is not None
                and proof.tile_masked(block_id)
                and not _tile_end_outside(proof, indices[dim], block_id, shape[dim])
            ):
                return False
    return True


def _mask_conjuncts(mask: object) -> list[object] | None:
    mask = skip_copies(mask)
    if mask is None:
        return []
    if (
        isinstance(mask, torch.fx.Node)
        and mask.target in (operator.and_, torch.ops.aten.bitwise_and.Tensor)
        and len(mask.args) == 2
        and not mask.kwargs
    ):
        left, right = (_mask_conjuncts(arg) for arg in mask.args)
        return None if left is None or right is None else [*left, *right]
    return [mask]


def _broadcast_source(node: object, dim: int) -> object:
    """``x`` for ``x[:, None]`` (dim 0) or ``x[None, :]`` (dim 1)."""
    node = skip_copies(node)
    pattern = [slice(None), None] if dim == 0 else [None, slice(None)]
    if (
        isinstance(node, torch.fx.Node)
        and node.target is view_ops.subscript
        and len(node.args) == 2
        and not node.kwargs
        and isinstance(node.args[1], (list, tuple))
        and list(node.args[1]) == pattern
    ):
        return skip_copies(node.args[0])
    return None


def _index_bounds(
    proof: _Proof, mask: object, indices: tuple[object, object]
) -> list[tuple[tuple[int, ...], object] | None] | None:
    """The conjuncts of a 2-D access's mask: for an ``index[:, None] < bound``
    or ``index[None, :] < bound`` test, the dimensions whose index it bounds
    and the bound; None for any other conjunct, and overall when ``mask`` is
    not a conjunction."""
    conjuncts = _mask_conjuncts(mask)
    if conjuncts is None:
        return None
    definitions = [_definition(proof, index) for index in indices]
    result: list[tuple[tuple[int, ...], object] | None] = []
    for conjunct in conjuncts:
        if (
            not isinstance(conjunct, torch.fx.Node)
            or conjunct.target
            not in (operator.lt, torch.ops.aten.lt.Scalar, torch.ops.aten.lt.Tensor)
            or len(conjunct.args) != 2
            or conjunct.kwargs
        ):
            result.append(None)
            continue
        lhs, bound = conjunct.args
        dims = tuple(
            dim
            for dim in (0, 1)
            if (source := _broadcast_source(lhs, dim)) is not None
            and _definition(proof, source) is definitions[dim]
        )
        result.append((dims, bound))
    return result


def mask_clips_only_outside(
    proof: _Proof,
    mask: object,
    indices: tuple[object, object],
    shape: tuple[int | sympy.Expr, int | sympy.Expr],
) -> bool:
    """Whether a 2-D access's mask equals its out-of-bounds lanes.

    ``mask`` must be a conjunction of ``index[:, None] < bound`` and
    ``index[None, :] < bound`` tests (``_bound_clips_only_outside``).
    """
    bounds = _index_bounds(proof, mask, indices)
    return bounds is not None and all(
        entry is not None
        and any(
            _bound_clips_only_outside(proof, indices[dim], entry[1], shape[dim])
            for dim in entry[0]
        )
        for entry in bounds
    )


def _mask_keeps_only_inside(
    proof: _Proof,
    mask: object,
    indices: tuple[object, object],
    shape: tuple[int | sympy.Expr, int | sympy.Expr],
) -> bool:
    """Whether a 2-D access's mask is false on every lane at or past the end
    of either dimension: each has an ``index < bound`` conjunct with a bound
    at most its extent."""
    bounds = _index_bounds(proof, mask, indices)
    if bounds is None:
        return False
    bounded: set[int] = set()
    for entry in bounds:
        configured = None if entry is None else _configured(proof, entry[1])
        if entry is None or configured is None:
            continue
        bounded.update(
            dim
            for dim in entry[0]
            if _known_nonnegative(sympy.Add(shape[dim], sympy.Mul(-1, configured)))
        )
    return bounded == {0, 1}


def _scales_word_aligned(proof: _Proof, operand: _Operand) -> bool:
    """Whether every K tile of the scale starts on a whole packed word."""
    block_id = operand.scale.group_block_id
    if block_id is None:
        return True  # a full ``arange(groups)`` starts at group 0
    size = _block_size(proof, block_id)
    bounds = _loop_bounds(proof, block_id)
    return (
        size is not None
        and size % 4 == 0
        and bounds is not None
        and _known_multiple(bounds[0], 4)
    )


def _scale_word_index_fits_int32(proof: _Proof, operand: _Operand) -> bool:
    """``_offsets_fit_int32`` for the scale's word index, ``group // 4`` of
    the K-group tile (``_scales_word_aligned``): an int32 vector whose loop
    bounds are int32 expressions."""
    if not _int32_index_vector(_definition(proof, operand.scale.word_index)):
        return False
    block_id = operand.scale.group_block_id
    if block_id is None:
        return True  # a full ``arange(groups)``
    bounds = _loop_bounds(proof, block_id)
    return bounds is not None and all(
        _int32_expression(proof, bound) for bound in bounds
    )


def _power_of_two(size: int) -> bool:
    return size > 0 and size & (size - 1) == 0


def _extent_above_one(tensor: torch.Tensor, size: int | torch.SymInt) -> bool:
    """Whether a dimension of ``tensor`` exceeds one on every call of this
    bound kernel: a static extent, or a dynamic one of a kernel input, whose
    class (empty, one or more) the input's descriptor layout guard keys --
    a dispatch guard, which an Inductor render's calls skip."""
    static = static_extent(size)
    if static is not None:
        return static > 1
    env = CompileEnvironment.current()
    return (
        env.dispatch_guards_enforced
        and env.tensor_input_source(tensor) is not None
        and env.size_hint(size) > 1
    )


def _layout_holds(tensor: torch.Tensor) -> bool:
    """Whether the traced strides of ``tensor``, and so the layout signature
    they give, are its strides on every call of the code.  An Inductor render
    of a dynamic-shape graph (``inductor_call_facts``) runs it at strides
    the trace may not have seen for an input the graph keeps symbolic
    (``traced_strides_hold``)."""
    return CompileEnvironment.current().traced_strides_hold(tensor)


def _descriptor_block(tensor: torch.Tensor, block_shape: tuple[int, int]) -> bool:
    """Whether a host descriptor over ``tensor`` can load ``block_shape``: the
    config-independent checks (its layout signature and base alignment are
    part of the bound kernel's key) and the block's."""
    env = CompileEnvironment.current()
    layout = env.tensor_descriptor_layout_signature(tensor)
    return (
        tensor.ndim == 2
        and _layout_holds(tensor)
        # A descriptor needs positive extents, and generic indexing
        # broadcasts a dimension of one, which a descriptor does not.
        and all(_extent_above_one(tensor, size) for size in tensor.shape)
        # Offsets are cast to int32, which keeps every index of an
        # int32-indexed kernel and not necessarily one of an int64-indexed one.
        and env.index_dtype != torch.int64
        and tensor_descriptor_supports_dtype(tensor.dtype)
        and layout is not None
        and layout[0] is not None
        and all(aligned or dim == layout[0] for dim, aligned in enumerate(layout[1]))
        and env.tensor_descriptor_base_is_aligned(tensor)
        and all(
            size <= CUDA_TENSOR_DESCRIPTOR_MAX_BLOCK_SIZE and _power_of_two(size)
            for size in block_shape
        )
        and block_shape[layout[0]] * tensor.element_size() >= 16
    )


_INT32_MAX = 2**31 - 1


def _int32_expression(proof: _Proof, expr: sympy.Expr, depth: int = 0) -> bool:
    """Whether the generated code computes ``expr`` in int32 on every tile:
    integer constants in the int32 range over tensor sizes (which fit it in
    an int32-indexed kernel), grid indices of loops starting at zero, and
    tile begins of device loops whose configured bounds satisfy the same.
    Triton widens a term outside that set (a larger literal, or a scalar
    argument at or past ``2**31``) to int64."""
    if any(abs(int(atom)) > _INT32_MAX for atom in expr.atoms(sympy.Integer)):
        return False
    host_function = HostFunction.current()
    device_ir = host_function.device_ir
    noncanonical = device_ir.noncanonical_task_origin_block_ids
    int32_roots = device_ir.int32_literal_task_origin_block_ids
    for symbol in expr.free_symbols:
        origin = host_function.expr_to_origin.get(symbol)
        kind = None if origin is None else origin.origin
        if isinstance(kind, TensorSizeOrigin):
            continue
        if not isinstance(kind, GridOrigin):
            return False
        bounds = _loop_bounds(proof, kind.block_id)
        if bounds is None:
            # A root loop: its indices are ``begin + pid * step``, int32 for
            # a zero begin and unit step or for int32 literals.
            if kind.block_id in noncanonical and kind.block_id not in int32_roots:
                return False
            continue
        if depth >= 4 or not all(
            _int32_expression(proof, bound, depth + 1) for bound in bounds
        ):
            return False
    return True


def _int32_index_vector(index: object) -> bool:
    """Whether ``index`` is an integer vector of at most 32 bits."""
    value = index.meta.get("val") if isinstance(index, torch.fx.Node) else None
    return (
        isinstance(value, torch.Tensor)
        and not value.dtype.is_floating_point
        and not value.dtype.is_complex
        and value.dtype != torch.bool
        and value.dtype.itemsize <= 4
    )


def _offsets_fit_int32(proof: _Proof, indices: Sequence[object]) -> bool:
    """Whether the first lane of each ``iota + start`` index, which a
    descriptor takes cast to int32 (``descriptor_tile``), is computed in
    int32, so the cast keeps it.  The mask proofs show that a lane the
    generic access clips lies outside the tensor; that carries over to the
    descriptor only if the cast does not wrap the lane back inside."""
    for index in indices:
        contiguous = _contiguous(proof, index)
        start = None if contiguous is None else _configured(proof, contiguous[0])
        if (
            start is None
            or not _int32_index_vector(_definition(proof, index))
            or not _int32_expression(proof, start)
        ):
            return False
    return True


def _start_aligned(
    proof: _Proof, tensor: torch.Tensor, indices: tuple[object, object]
) -> bool:
    """Whether a block's start along ``tensor``'s contiguous dimension is a
    multiple of 16 bytes, as Triton requires of a descriptor offset."""
    layout = CompileEnvironment.current().tensor_descriptor_layout_signature(tensor)
    if layout is None or layout[0] is None:
        return False
    contiguous = _contiguous(proof, indices[layout[0]])
    start = None if contiguous is None else _configured(proof, contiguous[0])
    if start is None:
        return False
    # A tile begins a whole number of tiles after its loop's begin.
    origins = HostFunction.current().expr_to_origin
    tile_begins: dict[sympy.Basic, sympy.Basic] = {}
    for symbol in start.free_symbols:
        origin = origins.get(symbol)
        if origin is None or not isinstance(origin.origin, TileBeginOrigin):
            continue
        size = _block_size(proof, origin.origin.block_id)
        bounds = _loop_bounds(proof, origin.origin.block_id)
        if size is not None and bounds is not None:
            tile_begins[symbol] = sympy.Add(
                bounds[0], sympy.Mul(size, sympy.Dummy(integer=True))
            )
    return _known_multiple(
        cast("sympy.Expr", start.xreplace(tile_begins)),
        max(1, 16 // tensor.element_size()),
    )


def _pointer_words(tensor: torch.Tensor) -> tuple[int, int, int] | None:
    """``(rows, words per row, word stride)`` of MN-major scale words for the
    pointer form, which spells them out (so every input of this bound kernel
    must share them) and indexes in int32."""
    rows, words_per_row = (static_extent(size) for size in tensor.shape)
    row_stride, word_stride = (static_extent(stride) for stride in tensor.stride())
    # Generic indexing broadcasts a dimension of one; this form does not.
    if (
        not _layout_holds(tensor)
        or rows is None
        or words_per_row is None
        or rows < 2
        or words_per_row < 2
        or row_stride != 1
        or word_stride is None
        or word_stride < rows
        or words_per_row * word_stride >= 2**31
    ):
        return None
    return rows, words_per_row, word_stride


def _scale_source(proof: _Proof, operand: _Operand) -> str | None:
    """How the packed words of ``operand``'s scale load natively, if they do."""
    match = operand.scale
    block_rows, block_groups = operand.tile
    # A tile that is not a power of two pads its aranges, which the unpacked
    # words (``[K/128, rows, 4 bytes]``) cannot.
    if (
        block_rows % 128
        or block_groups % 4
        or not _power_of_two(block_rows)
        or not _power_of_two(block_groups)
        or not _scales_word_aligned(proof, operand)
    ):
        return None
    indices = (match.row, match.word_index)
    layout = CompileEnvironment.current().tensor_descriptor_layout_signature(
        match.tensor
    )
    # MN-major words (rows contiguous) arrive in descriptor order [words, rows].
    # The layout signature is part of the bound kernel's key.
    if (
        layout is not None
        and layout[0] == 0
        and _contiguous(proof, match.row) is not None
        and _offsets_fit_int32(proof, (match.row,))
        and _scale_word_index_fits_int32(proof, operand)
        and _start_aligned(proof, match.tensor, indices)
        and _descriptor_block(match.tensor, (block_rows, block_groups // 4))
    ):
        return "descriptor"
    # The pointer form indexes rows with the row vector's lanes; a tile
    # (``sa[tile, ...]``) has no such vector in the generated code.
    row = match.row.meta.get("val") if isinstance(match.row, torch.fx.Node) else None
    if (
        not isinstance(row, torch.Tensor)
        or row.ndim != 1
        or _pointer_words(match.tensor) is None
    ):
        return None
    return "pointer"


def _operand_descriptor(proof: _Proof, operand: _Operand) -> bool:
    """Whether a descriptor load of the operand tile reads the same values."""
    rows, groups = operand.tile
    # A dimension the bound kernel does not specialize (for example the K of
    # ``b`` when only ``a``'s K is passed to ``hl.specialize``) stays symbolic:
    # the mask proofs then have to hold for every value of it.
    extents = [_tensor_extent(size) for size in operand.tensor.shape]
    value = operand.load.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or tuple(_configured(proof, size) for size in value.shape)
        != (rows, groups * 32)
        or extents[0] is None
        or extents[1] is None
    ):
        return False
    shape = (extents[0], extents[1])
    return (
        all(_contiguous(proof, index) is not None for index in operand.indices)
        and mask_clips_only_outside(proof, operand.load.args[2], operand.indices, shape)
        and _tile_masks_clip_only_outside(proof, value.shape, operand.indices, shape)
        and _offsets_fit_int32(proof, operand.indices)
        and _start_aligned(proof, operand.tensor, operand.indices)
        and _descriptor_block(operand.tensor, (rows, groups * 32))
    )


# -- codegen ---------------------------------------------------------------


def _node_ast(state: CodegenState, node: object) -> ast.AST | None:
    if not isinstance(node, torch.fx.Node):
        return None
    value = state.env.get(node)
    return value if isinstance(value, ast.AST) else None


def _first_lane(state: CodegenState, index: object, index_ast: ast.AST) -> ast.AST:
    """An index vector's first lane: the scalar base of an ``iota + base``
    computed in this graph (as ``TensorDescriptorIndexingStrategy``), else the
    minimum of its lanes."""
    value = index.meta.get("val") if isinstance(index, torch.fx.Node) else None
    info = (
        _contiguous_integer_tensor_index(value, index)
        if isinstance(value, torch.Tensor)
        else None
    )
    base = None if info is None else _contiguous_integer_tensor_index_base(state, info)
    if base is not None:
        return base
    return expr_from_string("tl.min({index})", index=index_ast)


def descriptor_tile(
    state: CodegenState,
    tensor: torch.Tensor,
    indices: tuple[object, object],
    block_shape: tuple[int, int],
    *,
    physical_layout: bool = False,
    load: bool = True,
) -> tuple[str, list[ast.AST], list[int] | None] | None:
    """A host descriptor over ``tensor`` for a 2-D block at ``indices``.

    Returns the descriptor name, its offsets in descriptor-dimension order,
    and the inverse permutation that restores the tensor-dimension order
    (``None`` when the descriptor is not permuted or ``physical_layout`` asks
    for the descriptor order).  An offset is its index's first lane
    (``_first_lane``): callers prove operand and store indices contiguous, and
    a scale's word index (``group // 4``, each word repeated) starts on a word.
    """
    env = CompileEnvironment.current()
    index_asts = [
        index for index in map(functools.partial(_node_ast, state), indices) if index
    ]
    if (
        len(index_asts) != 2
        # Fences around cross-thread syncs only cover TMA the function knows of.
        or not state.device_function.uses_async_proxy_global
        or not native_packed_mxfp8_target(env)
        or not _descriptor_block(tensor, block_shape)
    ):
        return None
    descriptor = state.device_function.tensor_descriptor_arg(
        tensor, list(block_shape), force_host=True
    )
    if load is False:
        state.device_function.note_tma_store(
            warp_specialized=_in_warp_specialized_loop(state)
        )
    order = descriptor.permutation or [0, 1]
    offsets = [
        expr_from_string(
            "tl.cast({offset}, tl.int32)",
            offset=_first_lane(state, indices[dim], index_asts[dim]),
        )
        for dim in order
    ]
    inverse = (
        None
        if descriptor.permutation is None or physical_layout
        else descriptor.inverse_permutation
    )
    return descriptor.name, offsets, inverse


def _descriptor_load(
    state: CodegenState,
    tensor: torch.Tensor,
    indices: tuple[object, object],
    block_shape: tuple[int, int],
    *,
    physical_layout: bool = False,
) -> ast.AST | None:
    """A host descriptor load of ``block_shape`` at ``indices``, or None.
    The load it replaces loses its ``load_eviction_policies`` entry:
    Triton's descriptor load takes no eviction policy."""
    descriptor = descriptor_tile(
        state, tensor, indices, block_shape, physical_layout=physical_layout
    )
    if descriptor is None:
        return None
    name, offsets, inverse = descriptor
    placeholders = {f"offset_{i}": offset for i, offset in enumerate(offsets)}
    value = expr_from_string(
        f"{name}.load([{', '.join(f'{{{key}}}' for key in placeholders)}])",
        **placeholders,
    )
    if inverse is not None:
        value = expr_from_string(f"tl.permute({{value}}, {inverse!r})", value=value)
    return value


def _operand_ast(state: CodegenState, operand: _Operand) -> ast.AST | None:
    """The operand tile through a descriptor (``_operand_descriptor``)."""
    rows, groups = operand.tile
    return _descriptor_load(state, operand.tensor, operand.indices, (rows, groups * 32))


def _load_tile_mask(
    state: CodegenState, sizes: Sequence[int | torch.SymInt]
) -> str | None:
    """The tile masks generic codegen puts on a load of ``sizes``: a
    dimension the size of a loop's tile takes that tile's mask."""
    env = CompileEnvironment.current()
    masks = []
    for dim, size in enumerate(sizes):
        block_id = env.get_block_id(size)
        mask = None if block_id is None else state.codegen.mask_var(block_id)
        if mask is not None:
            masks.append(f"({mask}{state.tile_strategy.expand_str([*sizes], dim)})")
    return " & ".join(masks) if masks else None


def _packed_scale_ast(
    state: CodegenState, operand: _Operand, source: str | None, *, prefix: str
) -> ast.AST | None:
    """Load packed words once and expose scales in tcgen05's UTCCP order."""
    match = operand.scale
    block_rows, block_groups = operand.tile
    row_ast = _node_ast(state, match.row)
    word_index_ast = _node_ast(state, match.word_index)
    if source is None or row_ast is None or word_index_ast is None:
        return None
    tensor = match.tensor
    words = block_groups // 4
    if source == "descriptor":
        loaded = _descriptor_load(
            state,
            tensor,
            (match.row, match.word_index),
            (block_rows, words),
            physical_layout=True,
        )
    else:
        pointer_words = _pointer_words(tensor)
        if pointer_words is None:
            return None
        rows, words_per_row, word_stride = pointer_words
        # Every address read lies inside the tensor; the masks of the load
        # this replaces are applied to the unpacked scales below.
        loaded = expr_from_string(
            (
                "tl.load({pointer} + "
                f"({{word_start}} + tl.arange(0, {words}))[:, None] * {word_stride} + "
                "{rows}[None, :], "
                f"mask=(({{word_start}} + tl.arange(0, {words}))[:, None] >= 0) "
                f"& (({{word_start}} + tl.arange(0, {words}))[:, None] < {words_per_row}) "
                f"& ({{rows}}[None, :] >= 0) & ({{rows}}[None, :] < {rows}), "
                f"other={_UNIT_SCALE_WORDS})"
            ),
            pointer=expr_from_string(state.device_function.tensor_arg(tensor).name),
            word_start=expr_from_string(
                "tl.min({word_index})", word_index=word_index_ast
            ),
            rows=row_ast,
        )
    if loaded is None:
        return None
    loaded = state.codegen.lift(loaded, dce=True, prefix=f"{prefix}_words")
    unpacked = expr_from_string(
        (
            "(({words}[:, :, None] >> "
            "(tl.arange(0, 4)[None, None, :] * 8)) & 255).to(tl.uint8)"
        ),
        words=loaded,
    )
    unpacked = state.codegen.lift(unpacked, dce=True, prefix=f"{prefix}_unpacked")
    # The words arrive as [K/128, rows, 4 bytes].  The reshape and transposes
    # are the identity on the logical [rows, groups] scale tile; they match
    # Triton's block-scaled tutorial so the scales reach TMEM through UTCCP.
    scale = expr_from_string(
        (
            f"{{scale}}.reshape({words}, {block_rows // 128}, 4, 32, 4)"
            ".trans(1, 0, 3, 2, 4)"
            ".trans(0, 3, 2, 1, 4)"
            f".reshape({block_rows}, {block_groups})"
        ),
        scale=unpacked,
    )
    # The generic word load reads 0 on the lanes its masks clip: its own mask
    # and its tile masks (a K tile past the loop's end).
    masks: list[ast.AST] = []
    if match.mask is not None:
        mask_ast = _node_ast(state, match.mask)
        if mask_ast is None:
            return None
        masks.append(mask_ast)
    tile_mask = _load_tile_mask(state, operand.scale_value.shape)
    if tile_mask is not None:
        masks.append(expr_from_string(tile_mask))
    if not masks:
        return scale
    mask = functools.reduce(
        lambda left, right: expr_from_string(
            "{left} & {right}", left=left, right=right
        ),
        masks,
    )
    return expr_from_string("tl.where({mask}, {scale}, 0)", mask=mask, scale=scale)


def _flattened_indices(
    index: object, row_stride: int | torch.SymInt
) -> tuple[object, object] | None:
    """``(rows, cols)`` of ``rows[:, None] * row_stride + cols[None, :]``."""
    index = skip_copies(index)
    if (
        not isinstance(index, torch.fx.Node)
        or index.target not in (operator.add, torch.ops.aten.add.Tensor)
        or len(index.args) != 2
        or index.kwargs
    ):
        return None
    env = CompileEnvironment.current()
    for row_term, col_term in (index.args, index.args[::-1]):
        row_term = skip_copies(row_term)
        cols = _broadcast_source(col_term, 1)
        if (
            cols is None
            or not isinstance(row_term, torch.fx.Node)
            or row_term.target not in (operator.mul, torch.ops.aten.mul.Tensor)
            or len(row_term.args) != 2
            or row_term.kwargs
        ):
            continue
        for rows_term, stride in (row_term.args, row_term.args[::-1]):
            rows = _broadcast_source(rows_term, 0)
            if (
                rows is not None
                and isinstance(stride, (int, torch.SymInt))
                and env.known_equal(stride, row_stride)
            ):
                return rows, cols
    return None


def without_cross_thread_sync(device_ir: DeviceIR) -> bool:
    """Whether the kernel emits no cross-thread sync, which drains TMA stores
    (``DeviceFunction.async_store_drain``) and cannot drain those of a
    warp-specialized loop: one phase, no barrier (``hl.barrier`` or
    ``remote_barrier``), no atomics, no remote copy, and no storage both read
    and written (a read-after-write barrier)."""
    env = CompileEnvironment.current()
    if (
        len(device_ir.root_ids) != 1
        or env.has_barrier
        or any(
            node.target in ATOMIC_OPS
            or node.target in (remote_barrier, start_async_remote_copy_descriptor)
            for graph in device_ir.graphs
            for node in graph.graph.nodes
        )
    ):
        return False
    # An access through device pointers (a stack tensor's) may reach any
    # storage.
    accesses = graph_accesses(device_ir.graphs, device_ir.root_ids[0])
    reads = {access.storage for access in accesses.accesses if access.kind == "read"}
    writes = accesses.writes().storages()
    return not (
        reads & writes
        or (_THROUGH_POINTERS in reads and writes)
        or (_THROUGH_POINTERS in writes and reads)
    )


def stores_once(device_function: DeviceFunction, device_ir: DeviceIR) -> bool:
    """Whether the kernel issues one store op, of one storage, under the
    config being compiled (a branch the config rules out, on a block size or
    a tunable, stores nothing: ``DeviceFunction.constexpr_taken_graph_id``).
    A TMA store's staging tile is budgeted beside the operand ring
    (``flat_output_descriptor_store``); another store, of any output, needs
    shared memory of its own (its own staging tile, or the scratch a pointer
    store converts its tile's layout through), which leaves a seed's ring no
    room for that tile."""
    from ..device_ir import control_flow_child_graph_ids

    graphs = device_ir.graphs
    dead: set[int] = set()
    for info in graphs:
        for node in info.graph.nodes:
            if node.target is _tracing_ops._if:
                taken = device_function.constexpr_taken_graph_id(node, tunables=True)
                if taken is not None:
                    _, if_graph_id, else_graph_id, *_ = node.args
                    dead.add(
                        cast(
                            "int",
                            else_graph_id if taken == if_graph_id else if_graph_id,
                        )
                    )
    pending = list(dead)
    while pending:
        for node in graphs[pending.pop()].graph.nodes:
            for child in control_flow_child_graph_ids(node):
                if child not in dead:
                    dead.add(child)
                    pending.append(child)
    live = {id(info.graph) for info in graphs if info.graph_id not in dead}
    writes = [
        access
        for access in graph_accesses(graphs, device_ir.root_ids[0]).writes().accesses
        if id(access.node.graph) in live
    ]
    return (
        len({access.node for access in writes}) == 1
        and len({access.storage for access in writes}) == 1
    )


def _pipeline_stages(config: Config) -> int:
    """The most stages any loop of ``config`` pipelines (``num_stages`` and
    the per-loop ``range_num_stages``)."""
    return max([config.num_stages, *config.range_num_stages])


def flat_output_descriptor_store(
    state: CodegenState, tensor: torch.Tensor, value: ast.AST
) -> ast.AST | None:
    """Store a tile of a flattened 2-D output through a 2-D host descriptor.

    ``out.view(m * n)[rows[:, None] * n + cols[None, :]] = tile`` under the
    mask ``(rows < m) & (cols < n)`` writes the ``[rows, cols]`` block of
    ``out``.  In a warp-specialized loop whose packed MXFP8 dot was lowered
    whole (``packed_mxfp8_dot``), a TMA store of that block lets the epilogue
    overlap the next tile's main loop.
    """
    node = state.fx_node
    base = tensor._base
    env = CompileEnvironment.current()
    if (
        node is None
        or not state.device_function.packed_mxfp8_whole_dot
        or base is None
        or tensor.ndim != 1
        or base.ndim != 2
        or not base.is_contiguous()
        or base not in HostFunction.current().tensor_to_origin
        or tensor.stride() != (1,)
        or not env.known_equal(tensor.storage_offset(), base.storage_offset())
        or not env.known_equal(tensor.numel(), base.numel())
        or not _in_warp_specialized_loop(state)
        # The bound kernel keys no alignment for the base of a flattened
        # output; the kernel's own allocation is aligned, an input may not be.
        or env.tensor_input_source(base) is not None
        or not without_cross_thread_sync(HostFunction.current().device_ir)
        or not stores_once(state.device_function, HostFunction.current().device_ir)
    ):
        return None
    subscript = node.args[1]
    value_proxy = state.proxy_arg(2)
    if (
        not isinstance(subscript, (list, tuple))
        or len(subscript) != 1
        or not isinstance(value_proxy, torch.Tensor)
        or value_proxy.ndim != 2
        or value_proxy.dtype != base.dtype
    ):
        return None
    indices = _flattened_indices(subscript[0], base.shape[1])
    block = tuple(_resolve_dim_size(size) for size in value_proxy.shape)
    shape = tuple(_resolve_dim_size(size) for size in base.shape)
    if indices is None or not all(type(size) is int for size in (*block, *shape)):
        return None
    proof = _codegen_proof(state)
    rows_range, cols_range = (_contiguous(proof, index) for index in indices)
    if rows_range is None or cols_range is None:
        return None
    extents = cast("tuple[int, int]", shape)
    column_start = _configured(proof, cols_range[0])
    device_function = state.device_function
    tile_bytes = math.prod(cast("tuple[int, int]", block)) * base.element_size()
    if (
        # The value fills the [rows, cols] block rather than broadcasting.
        any(
            _configured(proof, index_range[1]) != size
            for index_range, size in zip((rows_range, cols_range), block, strict=True)
        )
        or tile_bytes > _MAX_STORE_TILE_BYTES
        # Its staging tile fits beside the operand ring (``stores_once`` above
        # rules out a second store needing shared memory of its own).
        or _pipeline_stages(state.config) * device_function.packed_mxfp8_stage_bytes
        + tile_bytes
        > SM100_SHARED_MEMORY_BYTES
        # The block store writes exactly the lanes the flat store writes:
        # every lane its masks (its own and its tile masks) clip lies outside
        # ``base``, and every lane they keep lies inside it.  A row index is
        # never negative there; a negative column would write the row before.
        or not mask_clips_only_outside(proof, node.args[3], indices, extents)
        or not _tile_masks_clip_only_outside(proof, value_proxy.shape, indices, extents)
        or not _mask_keeps_only_inside(proof, node.args[3], indices, extents)
        or column_start is None
        or not _known_nonnegative_index(column_start)
        or not _offsets_fit_int32(proof, indices)
        or not _start_aligned(proof, base, indices)
    ):
        return None
    # ``base`` is contiguous and its extents exceed one, so the descriptor is
    # not permuted.
    descriptor = descriptor_tile(
        state, base, indices, cast("tuple[int, int]", block), load=False
    )
    if descriptor is None:
        return None
    name, (row, col), _ = descriptor
    return expr_from_string(
        f"{name}.store([{{row}}, {{col}}], {{value}})", row=row, col=col, value=value
    )

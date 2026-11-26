from __future__ import annotations

import dataclasses
from itertools import starmap
import logging
import math
import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import sympy
import torch

from .. import exc
from .. import language as hl
from .._compat import shape_env_size_hint
from ..autotuner.config_spec import AccumulatorFact
from ..autotuner.config_spec import DotAxes
from ..autotuner.config_spec import DotAxisKind
from ..autotuner.config_spec import DotSite
from ..autotuner.config_spec import KernelGridFact
from ..autotuner.config_spec import KernelMatmulFact
from ..autotuner.config_spec import LiveTile
from ..autotuner.config_spec import LoopAxisFact
from ..autotuner.config_spec import MemoryOpFact
from ..autotuner.config_spec import PipelinedRegion
from ..autotuner.config_spec import PointwiseElementwiseFact
from ..autotuner.config_spec import ResidentRegion
from ..autotuner.config_spec import ResolvedMatmulFact
from ..autotuner.config_spec import RootGridFact
from ..autotuner.config_spec import SymbolicLoopBound
from ..language import _tracing_ops
from ..language._decorators import is_api_func
from .compile_environment import FixedBlockSizeSource
from .compile_environment import _has_unbacked
from .compile_environment import _symint_free_symbols
from .compile_environment import _symint_sympy_expr
from .compile_environment import _to_sympy
from .fx_side_effects import is_impure_node
from .indexing_strategy import _contiguous_integer_tensor_index
from .indexing_strategy import is_block_size_index
from .indexing_strategy import subscript_index_scale
from .indexing_strategy import subscript_tile_info
from .tile_dependency import TILE_ACCESS_META
from .tile_dependency import _relation_product_is_within_budget
from .utils import compute_slice_size
from .utils import is_full_slice
from .variable_origin import GridOrigin
from .variable_origin import TileBeginOrigin
from .variable_origin import TileIdOrigin

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Container
    from collections.abc import Iterable
    from collections.abc import Sequence

    from ..autotuner.config_spec import ConfigSpec
    from .compile_environment import CompileEnvironment
    from .device_ir import DeviceIR
    from .device_ir import GraphInfo
    from .host_function import CompilerState
    from .host_function import HostFunction
    from .tile_dependency import AffineSubscriptRange
    from .tile_dependency import IntegerExpression
    from .tile_dependency import TileAccess


log = logging.getLogger(__name__)


def matmul_operand_positions() -> dict[object, tuple[int, int]]:
    """Matmul/dot FX targets mapped to their lhs/rhs argument positions."""
    from ..language import matmul_ops

    return {
        matmul_ops.dot_scaled: (0, 3),
        matmul_ops.dot: (0, 1),
        torch.ops.aten.mm.default: (0, 1),
        torch.ops.aten.bmm.default: (0, 1),
        torch.ops.aten.bmm.dtype: (0, 1),
        torch.ops.aten.addmm.default: (1, 2),
        torch.ops.aten.baddbmm.default: (1, 2),
    }


def trace_back_to_load(arg: object, load_op: object) -> torch.fx.Node | None:
    """Follow a matmul operand through pass-through ops to one load node."""
    cur = arg
    for _ in range(8):
        if not isinstance(cur, torch.fx.Node):
            return None
        if cur.target is load_op:
            return cur
        tensor_inputs = [
            value
            for value in cur.args
            if isinstance(value, torch.fx.Node)
            and isinstance(value.meta.get("val"), torch.Tensor)
        ]
        if len(tensor_inputs) != 1:
            return None
        cur = tensor_inputs[0]
    return None


def _rank_reduction_scaled_baddbmm_batch_block_id(
    node: torch.fx.Node,
    env: CompileEnvironment,
) -> int | None:
    """Detect the narrow Triton bug pattern for the H100 matmul heuristic.

    Triton's SM90 layout solver fails above ``num_stages=1`` when a dot-derived
    row reduction rescales a loop-carried ``baddbmm`` accumulator. Return its
    batch axis so the heuristic can recognize the singleton-batch WGMMA form
    and force one stage; this fact is not a general hardware constraint.
    """
    if node.target is not torch.ops.aten.baddbmm.default:
        return None
    output = node.meta.get("val")
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        return None
    scaled_acc = node.args[0]
    if not isinstance(scaled_acc, torch.fx.Node):
        return None
    if scaled_acc.target is not torch.ops.aten.mul.Tensor:
        return None

    def is_carried(value: object) -> bool:
        if not isinstance(value, torch.fx.Node):
            return False
        tensor = value.meta.get("val")
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != output.ndim
            or not all(map(env.known_equal, tensor.shape, output.shape))
        ):
            return False
        return value.op == "placeholder" or (
            value.target is _tracing_ops._new_var
            and bool(value.args)
            and isinstance(value.args[0], torch.fx.Node)
            and value.args[0].op == "placeholder"
        )

    accumulator, scale = scaled_acc.args[:2]
    if not is_carried(accumulator):
        accumulator, scale = scale, accumulator
    if not is_carried(accumulator) or not isinstance(scale, torch.fx.Node):
        return None

    from .inductor_lowering import ReductionLowering

    dot_targets = matmul_operand_positions()
    pending = [(scale, False)]
    seen: set[tuple[torch.fx.Node, bool]] = set()
    while pending:
        candidate, reduced = pending.pop()
        value = candidate.meta.get("val")
        reduced |= (
            isinstance(candidate.meta.get("lowering"), ReductionLowering)
            and isinstance(value, torch.Tensor)
            and len(value.shape) == 2
            and all(map(env.known_equal, value.shape, output.shape[:-1]))
        )
        state = (candidate, reduced)
        if state in seen:
            continue
        seen.add(state)
        if reduced and candidate.target in dot_targets:
            return env.get_block_id(output.shape[0])
        pending.extend((parent, reduced) for parent in candidate.all_input_nodes)
    return None


def _immovable_extent(
    env: CompileEnvironment,
    spec: ConfigSpec,
    block_id: int,
) -> int | None:
    """Return the per-program extent of an axis the config cannot move: not
    one an inline host helper over tunables sizes, which each config gives
    its own value (``FixedBlockSizeSource.varies_with_config``)."""
    if not (0 <= block_id < len(env.block_sizes)):
        return None
    info = env.block_sizes[block_id]
    if (
        not isinstance(info.block_size_source, FixedBlockSizeSource)
        or info.block_size_source.varies_with_config
    ):
        return None
    try:
        value = info.block_size_source.from_config(
            spec._base_default_config(),
            info,
        )
    except Exception:
        return None
    if value is None:
        return None
    if type(value) is int:
        return max(1, value)
    if not isinstance(value, torch.SymInt):
        return None
    expression = env.shape_env.simplify(_symint_sympy_expr(value))
    if expression.free_symbols or not isinstance(expression, sympy.Integer):
        return None
    return max(1, int(expression))


def _load_needs_eviction_tunable(node: torch.fx.Node) -> bool:
    """A load gets an eviction-policy slot only when the user did not pass one."""
    eviction_policy_arg = node.kwargs.get("eviction_policy")
    if eviction_policy_arg is None and len(node.args) >= 4:
        eviction_policy_arg = node.args[3]
    return eviction_policy_arg is None


# Ops that touch no memory: they route tensors through SSA values and subgraphs,
# read metadata, only debug or only warm a cache (a prefetch orders nothing).
TENSOR_ROUTING_OPS = (
    _tracing_ops._phi,
    _tracing_ops._new_var,
    _tracing_ops._for_loop,
    _tracing_ops._for_loop_step,
    _tracing_ops._while_loop,
    _tracing_ops._if,
    operator.getitem,
    torch.ops.aten.sym_size.int,
    torch.ops.aten.sym_stride.int,
    hl.device_print,
    hl.breakpoint,
    hl.prefetch,
)

# Ops that keep a prefetching root's loads ordered although they touch no
# memory: any loop (conservatively: its loads may decide the trip count), a
# print or breakpoint that shows them.
_HINT_BREAKING_OPS = (
    _tracing_ops._for_loop,
    _tracing_ops._for_loop_step,
    _tracing_ops._while_loop,
    hl.device_print,
    hl.breakpoint,
)


def _may_access_memory_unseen(node: torch.fx.Node, state: CompilerState) -> bool:
    """Whether a side-effecting Helion op or an op on symmetric memory hides accesses."""
    if node.target in TENSOR_ROUTING_OPS:
        return False
    # Peer views, and the local copies other ranks reach through theirs.
    symmetric = {*state.peer_views, *(v[0] for v in state.peer_views.values() if v)}
    return (
        is_api_func(node.target) and is_impure_node(node, impure_random=False)
    ) or any(
        not symmetric.isdisjoint(state.ssa_storages(value))
        for arg in node.all_input_nodes
        if isinstance(value := arg.meta.get("val"), torch.Tensor)
    )


def _accessed_tensor_fake(node: torch.fx.Node) -> torch.Tensor | None:
    """Fake tensor of the buffer a load/store accesses."""
    arg = node.args[0] if node.args else None
    if isinstance(arg, torch.fx.Node):
        value = arg.meta.get("val")
        if isinstance(value, torch.Tensor):
            return value
    return None


_CONJUNCTIONS = (
    torch.ops.aten.bitwise_and.Tensor,
    torch.ops.aten.logical_and.default,
    operator.and_,
)


def _launch_uniform_terms(
    mask: object,
    uniform_tensor: Callable[[torch.Tensor], object | None],
) -> tuple[frozenset[object], bool]:
    """The conjuncts of an extra mask that hold one value for the whole launch
    (keys equal across roots for the same predicate), and whether every
    conjunct does.

    A launch-uniform conjunct is one element, reads no tile index or program
    id and reads some data: it is built from constants, one-element loads at
    constant indices of tensors the kernel never writes (``uniform_tensor``
    keys a load's tensor by name and view, None when it may change during the
    launch), broadcasts and pointwise ops (``torch.Tag.pointwise``) drawing no
    random numbers. Keys hold only names and numbers, so the rank digest
    compares them across ranks (a gate built from a rank constant differs
    there).
    """
    from ..language import memory_ops
    from ..language import view_ops

    def broadcast_only(index: object) -> bool:
        return isinstance(index, (list, tuple)) and all(
            item is None or is_full_slice(item) for item in index
        )

    def constant_index(item: object) -> object | None:
        if type(item) is int:
            return item
        if (
            isinstance(item, torch.fx.Node)
            and item.target is torch.ops.prims.iota.default
            and item.args[:1] == (1,)
            and item.kwargs.get("start", 0) == 0
        ):
            return 0
        return None

    memo: dict[torch.fx.Node, object | None] = {}

    def single(node: torch.fx.Node) -> bool:
        """One element: a value the whole launch shares, not one per lane."""
        fake = node.meta.get("val")
        return isinstance(fake, torch.Tensor) and fake.numel() == 1

    def key(value: object) -> object | None:
        if isinstance(value, (bool, int, float)):
            return ("constant", type(value).__name__, value)
        if isinstance(value, torch.dtype):
            return ("dtype", str(value))
        if (
            not isinstance(value, torch.fx.Node)
            or value.op != "call_function"
            or not single(value)
        ):
            return None
        if value in memo:
            return memo[value]
        memo[value] = None
        target, args = value.target, value.args
        result: object | None = None
        if target is view_ops.subscript and len(args) >= 2 and broadcast_only(args[1]):
            result = key(args[0])
        elif (
            isinstance(target, torch._ops.OpOverload)
            and torch.Tag.pointwise in target.tags
            and torch.Tag.nondeterministic_seeded not in target.tags
            and not value.kwargs.keys() - {"dtype"}
        ):
            operands = [key(arg) for arg in args]
            if all(operand is not None for operand in operands):
                result = (str(target), *operands, value.kwargs.get("dtype"))
        elif target is memory_ops.load and len(args) >= 2:
            fake = _accessed_tensor_fake(value)
            tensor = None if fake is None else uniform_tensor(fake)
            index = args[1]
            extra_mask = args[2] if len(args) >= 3 else None
            if (
                tensor is not None
                and extra_mask is None
                and isinstance(index, (list, tuple))
            ):
                items = [constant_index(item) for item in index]
                if all(item is not None for item in items):
                    result = ("load", tensor, tuple(items))
        memo[value] = result
        return result

    def boolean(value: object) -> bool:
        fake = value.meta.get("val") if isinstance(value, torch.fx.Node) else None
        return isinstance(fake, torch.Tensor) and fake.dtype is torch.bool

    def conjuncts(value: object) -> list[object]:
        if isinstance(value, torch.fx.Node):
            if (
                value.target in _CONJUNCTIONS
                and len(value.args) == 2
                and all(boolean(arg) for arg in value.args)
            ):
                return [*conjuncts(value.args[0]), *conjuncts(value.args[1])]
            if (
                value.target is view_ops.subscript
                and len(value.args) >= 2
                and broadcast_only(value.args[1])
                and isinstance(value.args[0], torch.fx.Node)
                and value.args[0].target in _CONJUNCTIONS
            ):
                return conjuncts(value.args[0])
        return [value]

    def reads_data(term: object) -> bool:
        return isinstance(term, tuple) and (
            term[:1] == ("load",) or any(reads_data(part) for part in term)
        )

    keys = [key(term) for term in conjuncts(mask)]
    keys = [k if reads_data(k) else None for k in keys]
    uniform = frozenset(k for k in keys if k is not None)
    return uniform, bool(keys) and all(k is not None for k in keys)


def _subscript_block_id(env: CompileEnvironment, subscript: object) -> int | None:
    """Return the block axis indexed by a tile-provenance subscript."""
    info = subscript_tile_info(env, subscript)
    return info.block_id if info is not None else None


# A cast preserves an index only into a 32/64-bit integer.
_INDEX_CASTS = (
    torch.ops.prims.convert_element_type.default,
    torch.ops.aten._to_copy.default,
)
_INDEX_DTYPES = (torch.int32, torch.int64)


def _is_gather_subscript(env: CompileEnvironment, subscript: object) -> bool:
    """Whether ``subscript`` is an index tensor that no tile axis resolves: the
    ``ids[tile]`` of ``table[ids[tile], tile_d]``, but not ``2 * tile.index``."""
    return (
        isinstance(subscript, torch.fx.Node)
        and isinstance(subscript.meta.get("val"), torch.Tensor)
        and subscript_index_scale(env, subscript)[0] is None
    )


def _placeholder_resolver(
    device_ir: DeviceIR,
) -> Callable[[torch.fx.Node], torch.fx.Node]:
    """A function from a node to itself, or for a placeholder of a rolled or
    nested graph to the value its enclosing graph passes in."""
    from .device_ir import HelperFunctionGraphInfo
    from .device_ir import NodeArgsGraphInfo

    graph_info_by_graph = {
        graph_info.graph: graph_info for graph_info in device_ir.graphs
    }

    def resolve_placeholder(node: torch.fx.Node) -> torch.fx.Node:
        seen: set[torch.fx.Node] = set()
        while node.op == "placeholder" and node not in seen:
            seen.add(node)
            graph_info = graph_info_by_graph.get(node.graph)
            if not isinstance(graph_info, NodeArgsGraphInfo) or isinstance(
                graph_info, HelperFunctionGraphInfo
            ):
                break
            try:
                node = graph_info.placeholder_to_outer_arg(node)
            except KeyError:
                break
        return node

    return resolve_placeholder


def _reads_memory(
    subscript: object, resolve: Callable[[torch.fx.Node], torch.fx.Node]
) -> bool:
    """Whether the value of ``subscript`` is computed from a load of a tensor in
    memory: ``ids[tile]`` and anything derived from it. Not a tile, ``hl.arange``
    or ``tile.index``, however that is subscripted (``tile.index[:, None]`` is a
    load too, of the index itself), and not a size: the shape of a loaded tensor
    is not read from memory. ``resolve`` is ``_placeholder_resolver``'s."""
    pending = [subscript] if isinstance(subscript, torch.fx.Node) else []
    seen: set[torch.fx.Node] = set()
    while pending:
        node = resolve(pending.pop())
        if node in seen or not isinstance(node.meta.get("val"), torch.Tensor):
            continue
        seen.add(node)
        if node.target is hl.load:
            base = node.args[0]
            if (
                isinstance(base, torch.fx.Node)
                and resolve(base).target is _tracing_ops._host_tensor
            ):
                return True
        pending.extend(node.all_input_nodes)
    return False


def _index_dims(subscript: object) -> tuple[int | torch.SymInt, ...]:
    """The dims of an index tensor subscript that are not of size one (none for
    a scalar index)."""
    val = subscript.meta.get("val") if isinstance(subscript, torch.fx.Node) else None
    if not isinstance(val, torch.Tensor):
        return ()
    return tuple(
        size for size in val.shape if not (isinstance(size, int) and size == 1)
    )


def _indexed_region_sizes(
    env: CompileEnvironment, tensor: torch.Tensor, index_list: torch.fx.node.Argument
) -> list[int | torch.SymInt]:
    """The sizes of the region ``tensor[index_list]`` addresses.

    The dims ``SubscriptIndexing.compute_shape`` gives the subscript, but a
    slice contributes its own length: the reduction dim that stands for it is
    allocated when the subscript is typed, not here.
    """
    index = cast(
        "list[object]", torch.fx.map_arg(index_list, lambda node: node.meta["val"])
    )
    tensors = [k for k in index if isinstance(k, torch.Tensor)]
    broadcast = env.should_broadcast_tensor_indexers(index)
    sizes: list[int | torch.SymInt] = []
    dims = iter(tensor.shape)
    for k in index:
        if k is None:
            continue  # A new axis of size one: it addresses no dim of the tensor.
        size = next(dims)
        if isinstance(k, slice):
            sizes.append(compute_slice_size(k, size))
        elif isinstance(k, torch.Tensor):
            # Adjacent index tensors address one group of dims (the Cartesian
            # product of 1-D ones, else their broadcast); index tensors that
            # something else separates each address their own.
            if not broadcast:
                sizes.extend(env.tensor_indexer_dims(k))
            elif k is tensors[0]:
                sizes.extend(env.tensor_indexer_broadcast_shape(tensors))
        elif isinstance(k, torch.SymInt) and is_block_size_index(k):
            sizes.append(k)
        # Else a scalar (an int, a symbolic value, a grid index): one element
        # along this dim.
    return sizes


def _block_extents(env: CompileEnvironment) -> dict[sympy.Basic, sympy.Basic]:
    """Each block's symbol mapped to the extent of the loop it iterates. A block
    without a single extent has no value here: its symbol stays."""
    return {
        info.symbol(): _to_sympy(info.size)
        for info in env.block_sizes
        if isinstance(info.size, (int, torch.SymInt))
    }


def _spans_several(
    env: CompileEnvironment, sizes: Sequence[int | torch.SymInt]
) -> bool:
    """Whether a region of ``sizes`` has more than one element with every block at
    its loop's extent. A size that is not a number then (a dynamic shape, a
    registered tunable) is taken to."""
    numel = sympy.Mul(*(_to_sympy(size) for size in sizes))
    numel = numel.xreplace(_block_extents(env))
    return not numel.is_number or int(numel) > 1


def _numel_at_unit_tiles(
    env: CompileEnvironment, sizes: Sequence[int | torch.SymInt]
) -> int | None:
    """The numel of a region of ``sizes`` for one element of the tunable tiles:
    every tunable block size set to one, every other block at its loop's extent,
    a dynamic input size at its hint.

    ``None`` when the region has no such numel:

    * it is not linear in a tunable block size: ``tile // 2`` has no value at a
      tile of one element, and ``tile * tile`` (two index tensors over one
      tile) grows with the tile;
    * it depends on an unbacked value that is not a block size, such as a
      registered tunable, which has no value before a config picks one.
    """
    tunable = {spec.block_id for spec in env.config_spec.block_sizes}
    unit_tiles: dict[sympy.Basic, sympy.Basic] = {
        info.symbol(): sympy.Integer(1)
        for info in env.block_sizes
        if info.block_id in tunable
    }
    numel = sympy.Mul(*(_to_sympy(size) for size in sizes))
    for tile in numel.free_symbols & unit_tiles.keys():
        if not numel.is_polynomial(tile) or sympy.degree(numel, tile) > 1:
            return None
    # Every other block is as large as the loop it iterates; a region that
    # needs a block without one extent is not counted.
    numel = numel.xreplace({**_block_extents(env), **unit_tiles})
    if _has_unbacked(numel):
        return None
    return shape_env_size_hint(env.shape_env, numel)


def _scalar_subscript_affine(
    env: CompileEnvironment,
    host: HostFunction,
    value: int | torch.SymInt,
) -> tuple[int | None, int, int | None, bool]:
    """Classify a scalar subscript from its symbolic value, not its FX chain."""
    expression = (
        env.shape_env.simplify(_symint_sympy_expr(value))
        if isinstance(value, torch.SymInt)
        else sympy.Integer(value)
    )
    free_symbols = expression.free_symbols
    if len(free_symbols) == 1:
        (symbol,) = free_symbols
        origin_info = host.expr_to_origin.get(symbol)
        origin = origin_info.origin if origin_info is not None else None
        offset = env.shape_env.simplify(expression - symbol)
        # tile.id + c and a unit-step index + c (grid, tile.begin) are points. One
        # point per block of size > 1 is not representable: whole dimension.
        if isinstance(origin, GridOrigin):
            source = env.block_sizes[origin.block_id].block_size_source
            if offset.is_Integer and (
                type(origin) is TileIdOrigin
                or (
                    type(origin) in (GridOrigin, TileBeginOrigin)
                    and isinstance(source, FixedBlockSizeSource)
                    and env.known_equal(source.value, 1)
                )
            ):
                return origin.block_id, 1, int(offset), type(origin) is TileIdOrigin
            return None, 1, None, True
    # A block-size symbol is the tile slice itself.
    block_id = env.get_block_id(value)
    if block_id is not None:
        return block_id, 1, 0, False
    if expression.is_Integer:
        return None, 1, int(expression), True
    return None, 1, None, True


def _subscript_affine(
    env: CompileEnvironment,
    host: HostFunction,
    subscript: object,
) -> tuple[int | None, int, int | None, bool]:
    """Return ``(block_id, scale, offset, is_scalar)`` for one subscript.

    Tensor chains step only through integer add/mul and value-preserving nodes;
    anything else is unknown (no block, offset ``None``), i.e. the whole dimension.
    """
    if type(subscript) is int:
        return None, 1, subscript, True
    if not isinstance(subscript, torch.fx.Node):
        return None, 1, None, False
    value = subscript.meta.get("val")
    if isinstance(value, (int, torch.SymInt)):
        return _scalar_subscript_affine(env, host, value)
    scale = 1
    offset = 0
    node: object = subscript
    while isinstance(node, torch.fx.Node):
        info = subscript_tile_info(env, node)
        if info is not None:
            if isinstance(info.offset, int):
                return info.block_id, scale, offset + scale * info.offset, False
            if env.known_equal(info.offset, 0):
                return info.block_id, scale, offset, False
            break
        target = node.target
        args = node.args
        if target is torch.ops.prims.iota.default:
            start = node.kwargs.get("start", 0)
            if scale == 1 and type(start) is int and node.kwargs.get("step", 1) == 1:
                return None, 1, offset + start, False
            break
        if (
            target in (torch.ops.aten.add.Tensor, torch.ops.aten.mul.Tensor)
            and len(args) == 2
            and not node.kwargs
        ):
            constant, operand = args[1], args[0]
            if type(constant) is not int:
                constant, operand = operand, constant
            if type(constant) is not int:
                break
            if target is torch.ops.aten.add.Tensor:
                offset += scale * constant
            elif constant >= 1:
                scale *= constant
            else:
                break
            node = operand
        elif (
            target is hl.tile_index
            or (target in _INDEX_CASTS and node.meta["val"].dtype in _INDEX_DTYPES)
            or (
                target is hl.subscript
                and len(args) == 2
                and isinstance(args[1], (list, tuple))
                and all(item is None or is_full_slice(item) for item in args[1])
            )
        ):
            node = args[0]
        else:
            break
    return None, 1, None, False


def _subscript_static_extent(subscript: object) -> int | None:
    """Return the width of a proved contiguous static index vector."""
    if isinstance(subscript, int):
        return 1
    if not isinstance(subscript, torch.fx.Node):
        return None
    fake = subscript.meta.get("val")
    if not isinstance(fake, torch.Tensor):
        return None
    info = _contiguous_integer_tensor_index(fake, subscript)
    return info.extent if info is not None and isinstance(info.extent, int) else None


def _subscript_dense_span(
    env: CompileEnvironment,
    host: HostFunction,
    subscript: object,
) -> tuple[int, int, int] | None:
    """Prove ``tile.begin * scale + arange(block_size * scale)``.

    This compact fact preserves the logical tile axis for a contiguous vector
    subscript without materializing its lanes.  It is intentionally narrower
    than the general affine-index model: one tile-begin symbol (a grid index
    is the begin of a tile of its step), a positive integer scale, and an
    integer offset.
    """
    if not isinstance(subscript, torch.fx.Node):
        return None
    fake = subscript.meta.get("val")
    if not isinstance(fake, torch.Tensor):
        return None
    contiguous = _contiguous_integer_tensor_index(fake, subscript)
    if contiguous is None:
        return None

    def scalar_expression(value: object) -> sympy.Expr | None:
        if type(value) is int:
            return sympy.Integer(value)
        if isinstance(value, torch.fx.Node):
            value = value.meta.get("val")
        if type(value) is int:
            return sympy.Integer(value)
        if isinstance(value, torch.SymInt):
            return env.shape_env.simplify(_symint_sympy_expr(value))
        return None

    base: sympy.Expr = sympy.Integer(0)
    for sign, value in contiguous.base_terms:
        term = scalar_expression(value)
        if term is None:
            return None
        base = env.shape_env.simplify(
            base + sign * term  # pyrefly: ignore[unsupported-operation]
        )
    tile_symbols = tuple(
        symbol
        for symbol in base.free_symbols
        if (origin := host.expr_to_origin.get(symbol)) is not None
        and type(origin.origin) in (GridOrigin, TileBeginOrigin)
    )
    if len(tile_symbols) != 1:
        return None
    tile_symbol = tile_symbols[0]
    origin = host.expr_to_origin[tile_symbol].origin
    assert isinstance(origin, GridOrigin)
    coefficient = sympy.expand(base).coeff(tile_symbol)
    remainder = env.shape_env.simplify(base - coefficient * tile_symbol)
    if (
        not isinstance(coefficient, sympy.Integer)
        or int(coefficient) <= 0
        or not isinstance(remainder, sympy.Integer)
    ):
        return None
    axis = origin.block_id
    info = env.block_sizes[env.canonical_block_id(axis)]
    # A fixed block size keeps a symbolic var, but arange(value * scale) traces
    # as a constant extent; accept either spelling.
    block_sizes: list[object] = [info.var]
    source = info.block_size_source
    if isinstance(source, FixedBlockSizeSource) and type(source.value) is int:
        block_sizes.append(source.value)
    extent = scalar_expression(contiguous.extent)
    if extent is None or not any(
        (size := scalar_expression(block_size)) is not None
        and env.shape_env.simplify(
            cast("Any", extent) - cast("Any", coefficient) * size
        )
        == 0
        for block_size in block_sizes
    ):
        return None
    return axis, int(coefficient), int(remainder)


@dataclasses.dataclass(frozen=True)
class _AffineIndexScalar:
    """One scalar index in the bounded quasi-affine access subset.

    Each term is ``(root axis, coefficient, static divisor)``.  Divisor one is
    the ordinary affine case; larger divisors are admitted only for a direct,
    unshifted scalar ``tile.id // divisor``.
    """

    coefficients: tuple[tuple[int, IntegerExpression, int], ...]
    offset: IntegerExpression

    def scaled(self, factor: IntegerExpression) -> _AffineIndexScalar:
        return _AffineIndexScalar(
            tuple(
                (axis, sympy.simplify(coefficient * factor), divisor)
                for axis, coefficient, divisor in self.coefficients
            ),
            sympy.simplify(self.offset * factor),
        )

    def plus(self, other: _AffineIndexScalar) -> _AffineIndexScalar:
        coefficients: dict[tuple[int, int], int | sympy.Expr] = {
            (axis, divisor): coefficient
            for axis, coefficient, divisor in self.coefficients
        }
        for axis, coefficient, divisor in other.coefficients:
            key = axis, divisor
            coefficients[key] = sympy.simplify(coefficients.get(key, 0) + coefficient)
        return _AffineIndexScalar(
            tuple(
                sorted(
                    (
                        (axis, value, divisor)
                        for (axis, divisor), value in coefficients.items()
                        if value
                    ),
                    key=lambda item: (
                        item[0],
                        item[2],
                        sympy.default_sort_key(item[1]),
                    ),
                )
            ),
            sympy.simplify(self.offset + other.offset),
        )


@dataclasses.dataclass(frozen=True)
class _AffineIndexTensor:
    """A small, statically shaped tensor of quasi-affine scalar indices."""

    shape: tuple[int, ...]
    values: tuple[_AffineIndexScalar, ...]


def _flat_coordinate(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinates = [0] * len(shape)
    for dimension in range(len(shape) - 1, -1, -1):
        coordinates[dimension] = index % shape[dimension]
        index //= shape[dimension]
    return tuple(coordinates)


def _flat_index(coordinates: tuple[int, ...], shape: tuple[int, ...]) -> int:
    result = 0
    for coordinate, size in zip(coordinates, shape, strict=True):
        result = result * size + coordinate
    return result


def _broadcast_affine_value(
    value: _AffineIndexTensor,
    coordinates: tuple[int, ...],
) -> _AffineIndexScalar | None:
    if len(value.shape) > len(coordinates):
        return None
    padding = len(coordinates) - len(value.shape)
    projected: list[int] = []
    for dimension, size in enumerate(value.shape):
        coordinate = coordinates[padding + dimension]
        if size == 1:
            projected.append(0)
        elif coordinate < size:
            projected.append(coordinate)
        else:
            return None
    return value.values[_flat_index(tuple(projected), value.shape)]


def _broadcast_shape(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...] | None:
    result: list[int] = []
    for offset in range(1, max(len(left), len(right)) + 1):
        left_size = left[-offset] if offset <= len(left) else 1
        right_size = right[-offset] if offset <= len(right) else 1
        if left_size != right_size and left_size != 1 and right_size != 1:
            return None
        result.append(max(left_size, right_size))
    shape = tuple(reversed(result))
    return shape if _relation_product_is_within_budget(math.prod(shape)) else None


def _broadcast_only_index_shape(
    source_shape: tuple[int, ...],
    index: object,
) -> tuple[int, ...] | None:
    """Apply only ``None`` insertion and full slices to a small shape."""
    if not isinstance(index, (list, tuple)):
        return None
    source_dimension = 0
    result: list[int] = []
    for item in index:
        if item is None:
            result.append(1)
        elif is_full_slice(item) and source_dimension < len(source_shape):
            result.append(source_shape[source_dimension])
            source_dimension += 1
        else:
            return None
    if source_dimension != len(source_shape):
        return None
    return tuple(result)


def _affine_shape_matches_fake(
    env: CompileEnvironment,
    modeled_shape: tuple[int, ...],
    fake: torch.Tensor,
) -> bool:
    """Prove each modeled logical lane dimension matches its FX dimension.

    A reduction iota is represented here by its full logical extent even when
    lowering later tiles that extent.  Padded physical lanes are masked by the
    reduction lowering; the logical extent is the exact accessed index set.
    """
    if len(modeled_shape) != fake.ndim:
        return False
    for modeled_extent, fake_extent in zip(modeled_shape, fake.shape, strict=True):
        if env.known_equal(fake_extent, modeled_extent):
            continue
        block_id = env.resolve_block_id(fake_extent)
        if block_id is None or not (0 <= block_id < len(env.block_sizes)):
            return False
        info = env.block_sizes[block_id]
        if isinstance(info.size, int | torch.SymInt) and env.known_equal(
            info.size, modeled_extent
        ):
            continue
        fixed_extent = _immovable_extent(env, env.config_spec, block_id)
        if fixed_extent is None or fixed_extent != modeled_extent:
            return False
    return True


def _affine_subscript_ranges(
    env: CompileEnvironment,
    subscript: object,
) -> tuple[AffineSubscriptRange, ...] | None:
    """Recover an exact bounded quasi-affine set for a flattened tensor index.

    This handles the common source idiom ``view(-1)[affine_offsets]`` without
    changing the emitted memory operation.  Only fixed-width tile indices,
    direct scalar ``tile.id // constant``, static iotas, broadcasting,
    host-backed nonnegative integer scaling, and addition are accepted.
    Unsupported expressions decline to the existing root-barrier fallback.
    """
    from ..language import memory_ops
    from ..language import view_ops
    from ..language.tile_ops import tile_index

    memo: dict[torch.fx.Node, _AffineIndexTensor | None] = {}

    def host_scalar_expression(value: object) -> sympy.Expr | None:
        if type(value) is int:
            return sympy.Integer(value)
        if isinstance(value, torch.fx.Node):
            value = value.meta.get("val")
        if type(value) is int:
            return sympy.Integer(value)
        if not isinstance(value, torch.SymInt):
            return None
        expression = env.shape_env.simplify(_symint_sympy_expr(value))
        if _has_unbacked(expression) or expression.is_integer is not True:
            return None
        return expression

    def evaluate(value: object) -> _AffineIndexTensor | None:
        scalar_expression = host_scalar_expression(value)
        if scalar_expression is not None:
            return _AffineIndexTensor(
                (),
                (_AffineIndexScalar((), scalar_expression),),
            )
        if not isinstance(value, torch.fx.Node):
            return None
        if value in memo:
            return memo[value]
        memo[value] = None
        target = value.target
        args = value.args
        result: _AffineIndexTensor | None = None
        if target is torch.ops.prims.iota.default:
            length = args[0] if args else None
            start = value.kwargs.get("start", 0)
            step = value.kwargs.get("step", 1)
            if (
                isinstance(length, int)
                and isinstance(start, int)
                and isinstance(step, int)
                and length >= 0
                and step > 0
                and _relation_product_is_within_budget(length)
            ):
                result = _AffineIndexTensor(
                    (length,),
                    tuple(
                        _AffineIndexScalar((), start + lane * step)
                        for lane in range(length)
                    ),
                )
        elif target is tile_index and args and isinstance(args[0], torch.fx.Node):
            info = subscript_tile_info(env, args[0])
            extent = (
                None
                if info is None
                else _immovable_extent(env, env.config_spec, info.block_id)
            )
            if (
                info is not None
                and extent is not None
                and isinstance(info.offset, int)
                and _relation_product_is_within_budget(extent)
            ):
                result = _AffineIndexTensor(
                    (extent,),
                    tuple(
                        _AffineIndexScalar(
                            ((info.block_id, extent, 1),),
                            info.offset + lane,
                        )
                        for lane in range(extent)
                    ),
                )
        elif target is hl.tile_id and args and isinstance(args[0], torch.fx.Node):
            info = subscript_tile_info(env, args[0])
            if info is not None and env.known_equal(info.offset, 0):
                result = _AffineIndexTensor(
                    (),
                    (_AffineIndexScalar(((info.block_id, 1, 1),), 0),),
                )
        elif target is view_ops.subscript and args:
            index = args[1] if len(args) >= 2 else None
            if isinstance(index, (list, tuple)) and all(
                item is None or is_full_slice(item) for item in index
            ):
                base = evaluate(args[0])
                if base is not None:
                    indexed_shape = _broadcast_only_index_shape(base.shape, index)
                    if indexed_shape is not None:
                        result = _AffineIndexTensor(indexed_shape, base.values)
        elif target is memory_ops.load and args:
            index = args[1] if len(args) >= 2 else None
            extra_mask = args[2] if len(args) >= 3 else value.kwargs.get("extra_mask")
            if (
                extra_mask is None
                and isinstance(index, (list, tuple))
                and all(item is None or is_full_slice(item) for item in index)
            ):
                base = evaluate(args[0])
                if base is not None:
                    indexed_shape = _broadcast_only_index_shape(base.shape, index)
                    if indexed_shape is not None:
                        result = _AffineIndexTensor(indexed_shape, base.values)
        elif target in (torch.ops.aten.add.Tensor, operator.add) and len(args) == 2:
            alpha = value.kwargs.get("alpha", 1)
            left = evaluate(args[0])
            right = evaluate(args[1])
            if alpha == 1 and left is not None and right is not None:
                shape = _broadcast_shape(left.shape, right.shape)
                if shape is None:
                    return None
                values: list[_AffineIndexScalar] = []
                for index in range(math.prod(shape)):
                    coordinates = _flat_coordinate(index, shape)
                    left_value = _broadcast_affine_value(left, coordinates)
                    right_value = _broadcast_affine_value(right, coordinates)
                    if left_value is None or right_value is None:
                        break
                    values.append(left_value.plus(right_value))
                if len(values) == math.prod(shape):
                    result = _AffineIndexTensor(shape, tuple(values))
        elif target in (torch.ops.aten.mul.Tensor, operator.mul) and len(args) == 2:
            left = evaluate(args[0])
            right = evaluate(args[1])

            def scalar_factor(
                value: _AffineIndexTensor | None,
            ) -> IntegerExpression | None:
                if (
                    value is None
                    or value.shape
                    or len(value.values) != 1
                    or value.values[0].coefficients
                ):
                    return None
                return sympy.sympify(value.values[0].offset)

            factor = scalar_factor(right)
            base = left
            if factor is None:
                factor = scalar_factor(left)
                base = right
            if (
                factor is not None
                and factor.is_nonnegative is True
                and base is not None
            ):
                result = _AffineIndexTensor(
                    base.shape,
                    tuple(item.scaled(factor) for item in base.values),
                )
        elif (
            target
            in (
                operator.floordiv,
                torch.ops.aten.floor_divide.default,
                torch.ops.aten.floor_divide.Scalar,
            )
            and len(args) == 2
        ):
            dividend, divisor = args
            if (
                isinstance(dividend, torch.fx.Node)
                and dividend.target is hl.tile_id
                and isinstance(divisor, int)
                and divisor > 0
            ):
                base = evaluate(dividend)
                if (
                    base is not None
                    and base.shape == ()
                    and len(base.values) == 1
                    and base.values[0].offset == 0
                    and len(base.values[0].coefficients) == 1
                ):
                    axis, coefficient, previous_divisor = base.values[0].coefficients[0]
                    if coefficient == 1 and previous_divisor == 1:
                        result = _AffineIndexTensor(
                            (),
                            (_AffineIndexScalar(((axis, 1, divisor),), 0),),
                        )
        if result is not None:
            fake = value.meta.get("val")
            if (
                (
                    result.shape
                    and (
                        not isinstance(fake, torch.Tensor)
                        or not _affine_shape_matches_fake(env, result.shape, fake)
                    )
                )
                or (not result.shape and isinstance(fake, torch.Tensor) and fake.ndim)
                or len(result.values) != math.prod(result.shape)
            ):
                result = None
        memo[value] = result
        return result

    affine = evaluate(subscript)
    if affine is None or not affine.values:
        return None
    offsets_by_coefficients: dict[
        tuple[tuple[int, IntegerExpression, int], ...],
        set[IntegerExpression],
    ] = {}
    for value in affine.values:
        if any(
            sympy.sympify(coefficient).is_nonnegative is not True or divisor <= 0
            for _axis, coefficient, divisor in value.coefficients
        ):
            return None
        coefficients = tuple(
            (axis, env.shape_env.simplify(sympy.sympify(coefficient)), divisor)
            for axis, coefficient, divisor in value.coefficients
        )
        offset = env.shape_env.simplify(sympy.sympify(value.offset))
        if offset.is_integer is not True:
            return None
        offsets_by_coefficients.setdefault(coefficients, set()).add(offset)

    ranges: list[
        tuple[
            tuple[tuple[int, int | sympy.Expr, int], ...],
            int | sympy.Expr,
            int | sympy.Expr,
            int,
        ]
    ] = []
    for coefficients, offset_set in sorted(
        offsets_by_coefficients.items(),
        key=lambda item: tuple(
            (axis, sympy.default_sort_key(coefficient), divisor)
            for axis, coefficient, divisor in item[0]
        ),
    ):
        offsets_by_symbolic_base: dict[Any, set[int]] = {}
        for offset in offset_set:
            constant, symbolic_base = sympy.expand(offset).as_coeff_Add()
            if not isinstance(constant, sympy.Integer):
                return None
            offsets_by_symbolic_base.setdefault(symbolic_base, set()).add(int(constant))
        for symbolic_base, offset_constants in sorted(
            offsets_by_symbolic_base.items(),
            key=lambda item: sympy.default_sort_key(item[0]),
        ):
            offsets = sorted(offset_constants)
            begin = previous = offsets[0]
            step: int | None = None
            for offset in offsets[1:]:
                difference = offset - previous
                if step is None:
                    step = difference
                elif difference != step:
                    ranges.append(
                        (
                            coefficients,
                            symbolic_base + begin,
                            symbolic_base + previous + step,
                            step,
                        )
                    )
                    begin = offset
                    step = None
                previous = offset
            ranges.append(
                (
                    coefficients,
                    symbolic_base + begin,
                    (
                        symbolic_base + begin + 1
                        if step is None
                        else symbolic_base + previous + step
                    ),
                    1 if step is None else step,
                )
            )
    return tuple(ranges)


def _store_axis_key(
    env: CompileEnvironment,
    store_node: torch.fx.Node,
) -> tuple[int | None, ...]:
    """Return the full block-axis key of one store subscript."""
    fake = _accessed_tensor_fake(store_node)
    index_list = store_node.args[1] if len(store_node.args) >= 2 else None
    if fake is None or not isinstance(index_list, (list, tuple)):
        return ()
    key: list[int | None] = []
    for position, subscript in enumerate(index_list):
        if isinstance(subscript, int) or position >= fake.ndim:
            continue
        block_id = _subscript_block_id(env, subscript)
        if block_id is None:
            block_id = env.resolve_block_id(fake.shape[position])
        key.append(block_id)
    return tuple(key)


def _classify_load_dataflow(
    load_node: torch.fx.Node,
    reduction_nodes: Container[int],
    env: CompileEnvironment,
) -> tuple[set[int], tuple[tuple[int | None, ...], ...]]:
    """Trace a load forward to reductions and stores without crossing either."""
    from ..language import memory_ops

    reductions_fed: set[int] = set()
    stores_fed: set[tuple[int | None, ...]] = set()
    seen: set[int] = set()
    stack = list(load_node.users)
    while stack:
        user = stack.pop()
        if id(user) in reduction_nodes:
            reductions_fed.add(id(user))
            continue
        if id(user) in seen:
            continue
        seen.add(id(user))
        if user.op == "call_function" and user.target is memory_ops.store:
            stores_fed.add(_store_axis_key(env, user))
            continue
        stack.extend(user.users)
    return reductions_fed, tuple(
        sorted(
            stores_fed,
            key=lambda axes: tuple(-1 if axis is None else axis for axis in axes),
        )
    )


def tile_rank(dims: tuple[int | None, ...]) -> int:
    """Number of block dimensions spanned by a tile."""
    return sum(dim is not None for dim in dims)


def _live_tile_kind(node: torch.fx.Node, dot_targets: frozenset[object]) -> str:
    from ..language import memory_ops

    if node.op == "placeholder":
        return "carry"
    if node.op == "call_function":
        if node.target is _tracing_ops._host_tensor:
            return "global"
        if node.target in dot_targets:
            return "dot_out"
        if node.target is memory_ops.load:
            return "load"
    return "other"


def _tile_from_tensor(
    value: object,
    env: CompileEnvironment,
    *,
    kind: str,
    stageable: bool | None = None,
) -> LiveTile | None:
    if not (isinstance(value, torch.Tensor) and value.shape):
        return None
    dims = tuple(env.resolve_block_id(size) for size in value.shape)
    static_dims: list[int | None] = []
    for size, block_id in zip(value.shape, dims, strict=False):
        if block_id is not None:
            static_dims.append(None)
            continue
        try:
            static_dims.append(int(env.size_hint(size)))
        except Exception:
            static_dims.append(None)
    return LiveTile(
        dim_block_ids=dims,
        static_dims=tuple(static_dims),
        itemsize=value.dtype.itemsize,
        kind=kind,
        stageable=stageable,
    )


def _index_depends_on_loop(
    obj: object,
    env: CompileEnvironment,
    loop_block_ids: frozenset[int],
    seen: set[torch.fx.Node] | None = None,
) -> bool | None:
    """Whether an FX index is proven to depend on an enclosing loop axis."""
    if isinstance(obj, (list, tuple)):
        states = [
            _index_depends_on_loop(item, env, loop_block_ids, seen) for item in obj
        ]
        if any(state is True for state in states):
            return True
        if any(state is None for state in states):
            return None
        return False
    if isinstance(obj, dict):
        return _index_depends_on_loop(tuple(obj.values()), env, loop_block_ids, seen)
    if isinstance(obj, torch.fx.Node):
        seen = set() if seen is None else seen
        if obj in seen:
            return False
        seen.add(obj)
        value = obj.meta.get("val")
        known_origin = False
        try:
            block_id = env.resolve_block_id(value)
        except Exception:
            block_id = None
        if block_id is not None:
            known_origin = True
            if block_id in loop_block_ids:
                return True
        if isinstance(value, torch.Tensor):
            for dim in value.shape:
                try:
                    dim_block_id = env.resolve_block_id(dim)
                except Exception:
                    dim_block_id = None
                if dim_block_id is not None:
                    known_origin = True
                    if dim_block_id in loop_block_ids:
                        return True
        if obj.op == "placeholder":
            return False if known_origin else None
        children = (*obj.args, *obj.kwargs.values())
        if children:
            state = _index_depends_on_loop(children, env, loop_block_ids, seen)
            if state is not False:
                return state
            return False
        if obj.op == "get_attr":
            return False
        return False if known_origin else None
    if isinstance(obj, torch.SymInt):
        try:
            block_id = env.resolve_block_id(obj)
        except Exception:
            return None
        return block_id in loop_block_ids if block_id is not None else None
    if isinstance(obj, torch.Tensor):
        return None
    return False


def _live_node_steps(
    nodes: tuple[torch.fx.Node, ...],
    tracked_nodes: Container[torch.fx.Node],
    last_use: dict[torch.fx.Node, int],
    *,
    last_use_default: int = -1,
) -> Iterable[set[torch.fx.Node]]:
    """Yield the tracked nodes live after each graph node is introduced."""
    live: set[torch.fx.Node] = set()
    for index, node in enumerate(nodes):
        if node in tracked_nodes:
            live.add(node)
        yield live
        live = {
            value for value in live if last_use.get(value, last_use_default) > index
        }


@dataclasses.dataclass
class GraphAnalysis:
    """Shared structural and liveness observations for one DeviceIR graph."""

    graph_id: int
    nodes: tuple[torch.fx.Node, ...]
    is_reduction_loop: bool
    block_ids: frozenset[int]
    block_id_order: tuple[int, ...]
    live_tile_steps: tuple[tuple[LiveTile, ...], ...]
    peak_dot_output_tiles: tuple[LiveTile, ...]
    peak_promoted_lhs_tiles: tuple[LiveTile, ...]
    dot_nodes: tuple[torch.fx.Node, ...]
    reduction_occurrences: tuple[int, ...]
    reduction_axis_by_node_id: dict[int, int]
    memory_tiles: tuple[tuple[torch.fx.Node, LiveTile], ...]
    _memory_tiles_by_loop_axes: dict[frozenset[int], tuple[LiveTile, ...]] = (
        dataclasses.field(
            default_factory=dict,
            repr=False,
        )
    )

    @classmethod
    def build(
        cls,
        graph_info: GraphInfo,
        env: CompileEnvironment,
        *,
        is_reduction_loop: bool,
        dot_targets: frozenset[object],
        resolve_placeholder: Callable[[torch.fx.Node], torch.fx.Node],
    ) -> GraphAnalysis:
        from ..language import memory_ops
        from .inductor_lowering import ReductionLowering

        graph = graph_info.graph
        nodes = tuple(graph.nodes)
        last_use: dict[torch.fx.Node, int] = {}
        tile_details: dict[torch.fx.Node, LiveTile] = {}
        dot_details: dict[torch.fx.Node, LiveTile] = {}
        promoted_lhs_details: dict[torch.fx.Node, LiveTile] = {}
        dot_nodes: list[torch.fx.Node] = []
        reduction_occurrences: list[int] = []
        seen_reductions: set[int] = set()
        reduction_axis_by_node_id: dict[int, int] = {}
        memory_tiles: list[tuple[torch.fx.Node, LiveTile]] = []
        operand_positions = matmul_operand_positions()
        promoted_lhs_nodes: set[torch.fx.Node] = set()
        for node in nodes:
            positions = operand_positions.get(node.target)
            if (
                node.op != "call_function"
                or positions is None
                or positions[0] >= len(node.args)
            ):
                continue
            lhs = node.args[positions[0]]
            if (
                isinstance(lhs, torch.fx.Node)
                and resolve_placeholder(lhs).target is not memory_ops.load
            ):
                promoted_lhs_nodes.add(lhs)

        for index, node in enumerate(nodes):
            for input_node in node.all_input_nodes:
                last_use[input_node] = index

            kind_node = resolve_placeholder(node) if node.op == "placeholder" else node
            kind = _live_tile_kind(kind_node, dot_targets)
            if node.op == "placeholder" and kind != "global":
                kind = "carry"
            tile = _tile_from_tensor(node.meta.get("val"), env, kind=kind)
            if tile is not None and kind == "dot_out":
                dot_details[node] = tile
            if tile is not None and node in promoted_lhs_nodes:
                tile = tile._replace(promoted_lhs=True)
                promoted_lhs_details[node] = tile
            if tile is not None and any(
                block_id is not None for block_id in tile.dim_block_ids
            ):
                tile_details[node] = tile

            if node.op == "call_function" and node.target in dot_targets:
                dot_nodes.append(node)

            lowering = node.meta.get("lowering")
            if isinstance(lowering, ReductionLowering):
                block_id = getattr(lowering, "block_index", None)
                if isinstance(block_id, int):
                    # Codegen gives an untiled registered extent its own
                    # reduction alias. Facts retain the registered owner: its
                    # extent is still unknown until a configuration is chosen.
                    if (owner := env.registered_extent_owner(block_id)) is not None:
                        block_id = owner
                    reduction_axis_by_node_id[id(node)] = block_id
                    if block_id not in seen_reductions:
                        seen_reductions.add(block_id)
                        reduction_occurrences.append(block_id)

            if node.op != "call_function":
                continue
            if node.target is memory_ops.load:
                memory_tile = _tile_from_tensor(
                    node.meta.get("val"),
                    env,
                    kind="load",
                )
            elif node.target is memory_ops.store:
                stored_value = None
                for arg in node.args:
                    if isinstance(arg, torch.fx.Node):
                        candidate = arg.meta.get("val")
                        if isinstance(candidate, torch.Tensor) and candidate.shape:
                            stored_value = candidate
                memory_tile = _tile_from_tensor(stored_value, env, kind="store")
            else:
                memory_tile = None
            if memory_tile is not None:
                memory_tiles.append((node, memory_tile))

        live_tile_steps: list[tuple[LiveTile, ...]] = []
        seen_steps: set[frozenset[int]] = set()
        for live in _live_node_steps(nodes, tile_details, last_use):
            if live:
                step_key = frozenset(id(value) for value in live)
                step = tuple(tile_details[value] for value in live)
                if step_key not in seen_steps:
                    seen_steps.add(step_key)
                    live_tile_steps.append(step)

        def peak_role_tiles(
            details: dict[torch.fx.Node, LiveTile],
            *,
            last_use_default: int = -1,
        ) -> tuple[LiveTile, ...]:
            best: tuple[LiveTile, ...] = ()
            best_key = (-1, -1)
            for live in _live_node_steps(
                nodes,
                details,
                last_use,
                last_use_default=last_use_default,
            ):
                tiles = tuple(details[value] for value in live)
                key = (
                    sum(tile_rank(tile.dim_block_ids) for tile in tiles),
                    len(tiles),
                )
                if key > best_key:
                    best_key = key
                    best = tiles
            return best

        peak_dot_output_tiles = peak_role_tiles(
            dot_details,
            last_use_default=len(nodes),
        )
        peak_promoted_lhs_tiles = peak_role_tiles(promoted_lhs_details)

        return cls(
            graph_id=graph_info.graph_id,
            nodes=nodes,
            is_reduction_loop=is_reduction_loop,
            block_ids=frozenset(getattr(graph_info, "block_ids", ()) or ()),
            block_id_order=tuple(getattr(graph_info, "block_ids", ()) or ()),
            live_tile_steps=tuple(live_tile_steps),
            peak_dot_output_tiles=peak_dot_output_tiles,
            peak_promoted_lhs_tiles=peak_promoted_lhs_tiles,
            dot_nodes=tuple(dot_nodes),
            reduction_occurrences=tuple(reduction_occurrences),
            reduction_axis_by_node_id=reduction_axis_by_node_id,
            memory_tiles=tuple(memory_tiles),
        )

    def reaches_output(self, node: torch.fx.Node, limit: int = 64) -> bool:
        """Whether a value reaches the graph output under the legacy bounded walk."""
        frontier = [node]
        seen: set[torch.fx.Node] = {node}
        steps = 0
        while frontier and steps < limit:
            steps += 1
            current = frontier.pop()
            for user in current.users:
                if user.op == "output":
                    return True
                if user not in seen:
                    seen.add(user)
                    frontier.append(user)
        return False

    def memory_tiles_for_loop_axes(
        self,
        env: CompileEnvironment,
        loop_block_ids: frozenset[int] = frozenset(),
    ) -> tuple[LiveTile, ...]:
        """Memory tiles with load stageability resolved for the given loop axes."""
        cached = self._memory_tiles_by_loop_axes.get(loop_block_ids)
        if cached is not None:
            return cached
        out: list[LiveTile] = []
        for node, tile in self.memory_tiles:
            if tile.kind == "load":
                stageable = (
                    _index_depends_on_loop(node.args[1], env, loop_block_ids)
                    if len(node.args) > 1 and loop_block_ids
                    else False
                )
            else:
                stageable = False
            out.append(tile._replace(stageable=stageable))
        result = tuple(out)
        self._memory_tiles_by_loop_axes[loop_block_ids] = result
        return result


@dataclasses.dataclass
class DeviceIRAnalysis:
    """Shared graph interpretation for all fact builders of one DeviceIR."""

    graphs: tuple[GraphAnalysis, ...]
    by_id: dict[int, GraphAnalysis]
    child_loops: dict[int, tuple[tuple[int, frozenset[int]], ...]]
    parent_of: dict[int, int]
    loop_block_ids: dict[int, frozenset[int]]
    loop_calls: dict[int, tuple[torch.fx.Node, ...]]
    dot_nodes: tuple[tuple[int, torch.fx.Node], ...]
    accumulator_inputs: tuple[torch.fx.Node, ...]
    kernel_grid_fact: KernelGridFact

    @classmethod
    def build(
        cls,
        device_ir: DeviceIR,
        env: CompileEnvironment,
    ) -> DeviceIRAnalysis:
        from .device_ir import ForLoopGraphInfo
        from .device_ir import ReductionLoopGraphInfo
        from .device_ir import control_flow_parent_ids

        dot_targets = frozenset(matmul_operand_positions())
        resolve_placeholder = _placeholder_resolver(device_ir)

        graphs = tuple(
            GraphAnalysis.build(
                graph_info,
                env,
                is_reduction_loop=isinstance(
                    graph_info,
                    ReductionLoopGraphInfo,
                ),
                dot_targets=dot_targets,
                resolve_placeholder=resolve_placeholder,
            )
            for graph_info in device_ir.graphs
        )
        by_id = {graph.graph_id: graph for graph in graphs}
        child_loops: dict[int, tuple[tuple[int, frozenset[int]], ...]] = {}
        loop_calls: dict[int, list[torch.fx.Node]] = {}
        graph_parent_of = control_flow_parent_ids(device_ir.graphs)
        for graph in graphs:
            if graph.is_reduction_loop:
                continue
            edges: list[tuple[int, frozenset[int]]] = []
            for node in graph.nodes:
                if (
                    node.op != "call_function"
                    or not _tracing_ops.is_for_loop_target(node.target)
                    or not node.args
                    or not isinstance(node.args[0], int)
                ):
                    continue
                body_id = node.args[0]
                loop_calls.setdefault(body_id, []).append(node)
                body = by_id.get(body_id)
                if body is not None and not body.is_reduction_loop:
                    edges.append((body_id, body.block_ids))
            if edges:
                child_loops[graph.graph_id] = tuple(edges)

        parent_of: dict[int, int] = {}
        loop_block_ids: dict[int, frozenset[int]] = {}
        for parent_id, loop_edges in child_loops.items():
            for body_id, block_ids in loop_edges:
                parent_of[body_id] = parent_id
                loop_block_ids[body_id] = block_ids

        dot_nodes = tuple(
            (graph.graph_id, node)
            for graph in graphs
            if not graph.is_reduction_loop
            for node in graph.dot_nodes
        )
        # A rolled reduction's lane-loop graph (``ReductionLoopGraphInfo``)
        # is the compiler's: its node args are the body's live-ins (a
        # broadcast factor read by every lane), never a loop-carried
        # accumulator, so they must not pull a pure pointwise kernel out of
        # the pointwise family.
        accumulator_inputs = tuple(
            node
            for graph_info in device_ir.graphs
            if isinstance(graph_info, ForLoopGraphInfo)
            and not isinstance(graph_info, ReductionLoopGraphInfo)
            for node in graph_info.node_args
        )
        roots = tuple(
            RootGridFact(graph_id, tuple(block_ids))
            for graph_id, block_ids in zip(
                device_ir.root_ids,
                device_ir.grid_block_ids,
                strict=True,
            )
        )
        root_ids = {root.root_graph_id for root in roots}
        graph_to_root: list[tuple[int, int]] = []
        for graph in graphs:
            current = graph.graph_id
            seen: set[int] = set()
            while current not in root_ids and current not in seen:
                seen.add(current)
                current = graph_parent_of.get(current, -1)
            if current in root_ids:
                graph_to_root.append((graph.graph_id, current))
        kernel_grid_fact = KernelGridFact(roots, tuple(graph_to_root))
        return cls(
            graphs=graphs,
            by_id=by_id,
            child_loops=child_loops,
            parent_of=parent_of,
            loop_block_ids=loop_block_ids,
            loop_calls={
                graph_id: tuple(calls) for graph_id, calls in loop_calls.items()
            },
            dot_nodes=dot_nodes,
            accumulator_inputs=accumulator_inputs,
            kernel_grid_fact=kernel_grid_fact,
        )

    @property
    def non_reduction_graphs(self) -> tuple[GraphAnalysis, ...]:
        return tuple(graph for graph in self.graphs if not graph.is_reduction_loop)

    def original_reductions(self) -> tuple[tuple[int, int], ...]:
        """Ordered, deduplicated reduction occurrences in original graphs."""
        return tuple(
            (graph.graph_id, block_id)
            for graph in self.non_reduction_graphs
            for block_id in graph.reduction_occurrences
        )

    def kernel_live_tile_steps(self) -> tuple[tuple[LiveTile, ...], ...]:
        """Every original graph step, kept separate across sequential regions.

        Loop-body placeholders already represent the values carried into that
        body, including enclosing-loop carries. Flattening the graph-local
        timelines therefore preserves complete body residency without summing
        state from sequential graphs or mutually exclusive branches.
        """
        return tuple(
            step
            for graph in self.non_reduction_graphs
            for step in graph.live_tile_steps
        )

    def _kernel_peak_role_tiles(self, field: str) -> tuple[LiveTile, ...]:
        """Peak role-specific tile set after adding ancestor loop graphs."""
        tiles_of = {
            graph.graph_id: cast("tuple[LiveTile, ...]", getattr(graph, field))
            for graph in self.non_reduction_graphs
        }
        best: tuple[LiveTile, ...] = ()
        best_key = (-1, -1)
        for graph_id, own in tiles_of.items():
            chain = list(own)
            current = self.parent_of.get(graph_id, -1)
            seen = {graph_id}
            while current in tiles_of and current not in seen:
                seen.add(current)
                chain.extend(tiles_of[current])
                current = self.parent_of.get(current, -1)
            key = (
                sum(tile_rank(tile.dim_block_ids) for tile in chain),
                len(chain),
            )
            if key > best_key:
                best_key = key
                best = tuple(chain)
        return best

    def kernel_peak_dot_outputs(self) -> tuple[LiveTile, ...]:
        """Peak dot-output set after adding accumulator ancestor chains."""
        return self._kernel_peak_role_tiles("peak_dot_output_tiles")

    def kernel_peak_promoted_lhs(self) -> tuple[LiveTile, ...]:
        """Peak transformed-LHS set after adding ancestor loop graphs."""
        return self._kernel_peak_role_tiles("peak_promoted_lhs_tiles")

    def accumulator_facts(
        self,
        env: CompileEnvironment,
    ) -> list[AccumulatorFact]:
        """Describe every loop-carried tensor accumulator."""
        from .._utils import next_power_of_2

        def resolve_dim(size: object) -> int | None:
            block_id = env.resolve_block_id(size)
            if block_id is not None:
                return block_id
            extent = env.size_hint(cast("int | torch.SymInt", size))
            matches = [
                info.block_id
                for info in env.block_sizes
                if info.reduction
                and env.registered_extent_owner(info.block_id) is None
                and next_power_of_2(info.size_hint()) == extent
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                log.warning(
                    "accumulator dim (padded extent %s) matches %d reduction axes %s; "
                    "cannot differentiate by extent -- left unresolved "
                    "(per_feature_accumulator may miss it)",
                    extent,
                    len(matches),
                    matches,
                )
            return None

        facts: list[AccumulatorFact] = []
        for node in self.accumulator_inputs:
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor):
                facts.append(
                    AccumulatorFact(
                        dim_block_ids=tuple(resolve_dim(size) for size in value.shape),
                        itemsize=value.element_size(),
                    )
                )
        return facts

    def writes_input_storage(self, env: CompileEnvironment) -> bool:
        """Whether a store can overwrite an input, including through a host view."""
        from ..language.memory_ops import store

        input_storages = {tensor.untyped_storage() for tensor in env.input_sources}
        return any(
            node.op == "call_function"
            and node.target is store
            and (tensor := _accessed_tensor_fake(node)) is not None
            and tensor.untyped_storage() in input_storages
            for graph in self.graphs
            for node in graph.nodes
        )

    def memory_op_facts(
        self,
        device_ir: DeviceIR,
        env: CompileEnvironment,
        host: HostFunction,
    ) -> list[MemoryOpFact]:
        """Record one fact for every load/store, in the order code generation
        lowers them.

        Code generation numbers the ``indexing`` and ``load_eviction_policies``
        slots in the order it lowers the loads and stores
        (``codegen_node_order``): a root's loads ahead of a nested loop take
        their slots before the loop's.  This is the order of the configs that
        roll no reduction; ``DeviceIR.config_memory_op_facts`` gives the
        order of any config.  Codegen lowers a ``while`` condition ahead of the
        loop and again at the end of its body: a load there has a fact for
        each, unless the backend's loads keep their slots when lowered again
        (``Backend.relowered_memory_ops_take_new_slots``).  A load of a
        ``tile.index`` value or of a stack tensor takes no ``indexing`` slot
        (``takes_indexing_slot``): its fact keeps its place and its
        ``load_eviction_policies`` slot (the Pallas lowering gives it none,
        but reads neither list), and has one of the ``indexing`` slots after
        the others, which code generation never reads.  Each node keeps its
        facts in ``MEMORY_OP_FACT_META``, which the copies a config's codegen
        graphs make of it carry.

        Where a config's block sizes decide an ``if``'s condition, the CuTe
        lowering emits only the live branch: the loads of the other branch
        take no ``load_eviction_policies`` slots in its code, and the loads
        after them take earlier ones than here.
        """
        from ..language import memory_ops
        from .device_ir import MEMORY_OP_FACT_META
        from .device_ir import memory_op_slots

        load_op = memory_ops.load
        store_op = memory_ops.store
        resolve = _placeholder_resolver(device_ir)
        operand_positions = matmul_operand_positions()
        operands: dict[torch.fx.Node, str] = {}
        records: list[tuple[torch.fx.Node, MemoryOpFact]] = []
        eviction_index = 0

        for graph_analysis in self.graphs:
            for node in graph_analysis.nodes:
                positions = (
                    operand_positions.get(node.target)
                    if node.op == "call_function"
                    else None
                )
                if positions is None:
                    continue
                for arg_index, operand in (
                    (positions[0], "lhs"),
                    (positions[1], "rhs"),
                ):
                    if arg_index < len(node.args):
                        load = trace_back_to_load(node.args[arg_index], load_op)
                        if load is not None:
                            operands.setdefault(load, operand)

        for graph_id, node, memory_op_index in memory_op_slots(
            device_ir.graphs, device_ir.root_ids
        ):
            is_load = node.target is load_op
            if not (is_load or node.target is store_op):
                continue
            graph_analysis = self.by_id[graph_id]
            reduction_axis_by_id = graph_analysis.reduction_axis_by_node_id
            this_eviction_index: int | None = None
            if is_load and _load_needs_eviction_tunable(node):
                this_eviction_index = eviction_index
                eviction_index += 1

            fake = _accessed_tensor_fake(node)
            origin = host.tensor_to_origin.get(fake) if fake is not None else None

            reductions_fed: tuple[tuple[int, int], ...] = ()
            stores_fed: tuple[tuple[int | None, ...], ...] = ()
            if is_load:
                feeds, stores_fed = _classify_load_dataflow(
                    node,
                    reduction_axis_by_id.keys(),
                    env,
                )
                per_axis: dict[int, int] = {}
                for fed_id in feeds:
                    axis = reduction_axis_by_id[fed_id]
                    per_axis[axis] = per_axis.get(axis, 0) + 1
                reductions_fed = tuple(sorted(per_axis.items()))

            indexed_block_ids: tuple[int | None, ...] = ()
            subscript_block_ids: tuple[int | None, ...] = ()
            subscript_strides: tuple[int, ...] = ()
            subscript_index_scales: tuple[int, ...] = ()
            subscript_affine_block_ids: tuple[int | None, ...] = ()
            subscript_extents: tuple[int, ...] = ()
            gather_numel: int | None = None
            row_gather: bool | None = None
            loaded_index_block_ids: tuple[int | None, ...] = ()
            inner_extent: int | None = None
            accessed_numel = 0
            if fake is not None:
                accessed_numel = 1
                fake_strides = fake.stride()
                for dim_index, dim_size in enumerate(fake.shape):
                    if fake_strides[dim_index] != 0:
                        accessed_numel *= env.size_hint(dim_size)
                index_list = node.args[1] if len(node.args) >= 2 else None
                if isinstance(index_list, (list, tuple)):
                    indexed_positions = [
                        position
                        for position, subscript in enumerate(index_list)
                        if not isinstance(subscript, int) and position < fake.ndim
                    ]
                    indexed_block_ids = tuple(
                        env.resolve_block_id(fake.shape[position])
                        for position in indexed_positions
                    )
                    subscript_block_ids = tuple(
                        _subscript_block_id(env, index_list[position])
                        for position in indexed_positions
                    )
                    subscript_strides = tuple(
                        env.size_hint(fake_strides[position])
                        for position in indexed_positions
                    )
                    affine = [
                        subscript_index_scale(env, index_list[position])
                        for position in indexed_positions
                    ]
                    subscript_affine_block_ids = tuple(
                        block_id for block_id, _scale in affine
                    )
                    subscript_index_scales = tuple(scale for _block_id, scale in affine)
                    subscript_extents = tuple(
                        env.size_hint(fake.shape[position])
                        for position in indexed_positions
                    )
                    # ``None`` adds an axis and addresses no dim of the tensor.
                    # In a rolled or nested graph a subscript computed outside
                    # it is a placeholder: read what it stands for.
                    subscripts = [
                        resolve(k) if isinstance(k, torch.fx.Node) else k
                        for k in index_list
                        if k is not None
                    ]
                    if any(_is_gather_subscript(env, k) for k in subscripts):
                        gather_numel = _numel_at_unit_tiles(
                            env, _indexed_region_sizes(env, fake, index_list)
                        )
                    # An address that waits on another load. It takes rows
                    # of what the loaded index picks when the other
                    # subscripts span more than one element and no loaded
                    # index addresses a stride-1 dim, which scatters the
                    # elements themselves. Either way it fetches once for
                    # each element of its loaded indices.
                    loaded = [_reads_memory(k, resolve) for k in subscripts]
                    if any(loaded):
                        row_gather = not any(
                            is_loaded and env.size_hint(fake_strides[dim]) == 1
                            for dim, is_loaded in enumerate(loaded)
                        ) and _spans_several(
                            env,
                            _indexed_region_sizes(
                                env,
                                fake,
                                [
                                    0 if is_loaded else k
                                    for k, is_loaded in zip(
                                        subscripts, loaded, strict=True
                                    )
                                ],
                            ),
                        )
                        loaded_index_block_ids = tuple(
                            env.resolve_block_id(size)
                            for dim, k in enumerate(subscripts)
                            if loaded[dim]
                            for size in _index_dims(k)
                        )
                if fake.ndim >= 2:
                    inner_extent = env.size_hint(fake.shape[-1])

            records.append(
                (
                    node,
                    MemoryOpFact(
                        indexing_index=memory_op_index,
                        kind="load" if is_load else "store",
                        eviction_index=this_eviction_index,
                        tensor_name=origin.root_rw_name() if origin else None,
                        dtype=fake.dtype if fake is not None else None,
                        ndim=fake.ndim if fake is not None else 0,
                        num_reuses=len(node.users) if is_load else 0,
                        matmul_operand=None,
                        graph_id=graph_analysis.graph_id,
                        reductions_fed=reductions_fed,
                        stores_fed=stores_fed,
                        indexed_block_ids=indexed_block_ids,
                        inner_extent=inner_extent,
                        subscript_block_ids=subscript_block_ids,
                        subscript_strides=subscript_strides,
                        subscript_index_scales=subscript_index_scales,
                        subscript_affine_block_ids=subscript_affine_block_ids,
                        subscript_extents=subscript_extents,
                        accessed_numel=accessed_numel,
                        gather_numel=gather_numel,
                        row_gather=row_gather,
                        loaded_index_block_ids=loaded_index_block_ids,
                    ),
                )
            )

        facts = []
        node_facts: dict[torch.fx.Node, list[MemoryOpFact]] = {}
        for node, fact in records:
            fact = fact._replace(matmul_operand=operands.get(node))
            node_facts.setdefault(node, []).append(fact)
            facts.append(fact)
        for node, lowered in node_facts.items():
            node.meta[MEMORY_OP_FACT_META] = tuple(lowered)
        return facts

    def tile_accesses(
        self,
        device_ir: DeviceIR,
        env: CompileEnvironment,
        host: HostFunction,
    ) -> tuple[TileAccess, ...]:
        """Record allocation-coordinate accesses used for cross-root scheduling."""
        from ..language import memory_ops
        from ..language.atomic_ops import ATOMIC_OPS
        from ..language.reduce_ops import _reduce
        from ..language.scan_ops import _associative_scan
        from .device_ir import memory_op_slots
        from .loop_dependency_checker import flat_offset_loads
        from .tile_dependency import TileAccess
        from .tile_dependency import owner_roots_by_graph_id

        if len(device_ir.root_ids) <= 1:
            return ()
        # Loads of the tensor they are passed at flat offsets (a footprint of
        # the whole tensor), not memory ops of the indexing config.
        flat_loads = flat_offset_loads()

        graph_owners = owner_roots_by_graph_id(device_ir)

        def owners_of(graph_id: int) -> tuple[int, ...]:
            return graph_owners[graph_id] if graph_id < len(graph_owners) else ()

        # A root whose only effects are prefetches warms caches and nothing
        # else: when its loads only steer prefetch addresses (a newer value
        # warms another line and changes no result), later writes need not
        # wait for them (``TileAccess.hint``).  So a root with a loop, a print
        # or an effect is no such root, nor one where a loaded value reaches a
        # load's operands (an index, a mask: a stale or newer value could make
        # it read out of bounds) or a load sits in a branch (its condition may
        # be such a value).
        prefetching = {
            root
            for graph_analysis in self.graphs
            if any(
                node.op == "call_function" and node.target is memory_ops.prefetch
                for node in graph_analysis.nodes
            )
            for root in owners_of(graph_analysis.graph_id)
        }
        combine_callers = {
            node.args[0]: graph_analysis.graph_id
            for graph_analysis in self.graphs
            for node in graph_analysis.nodes
            if node.op == "call_function"
            and node.target in (_reduce, _associative_scan)
        }

        def hint_owners_of(graph_id: int) -> tuple[int, ...]:
            # A reduction's or scan's combine function is a graph no loop
            # owns: it runs in the roots that call it.
            while graph_id in combine_callers:
                graph_id = combine_callers[graph_id]
            return owners_of(graph_id)

        effectful: set[int] = set()
        if prefetching:
            for graph_analysis in self.graphs:
                owners = hint_owners_of(graph_analysis.graph_id)
                if not prefetching.intersection(owners):
                    continue
                for node in graph_analysis.nodes:
                    if node.op == "call_function" and (
                        node.target in _HINT_BREAKING_OPS
                        or _may_access_memory_unseen(node, host.compiler_state)
                    ):
                        effectful.update(owners)
                        break

        def steers_only(root: int) -> bool:
            top = device_ir.root_ids[root]
            loaded: set[torch.fx.Node] = set()
            for graph_analysis in self.graphs:
                graph_id = graph_analysis.graph_id
                if root not in hint_owners_of(graph_id):
                    continue
                for node in graph_analysis.nodes:
                    if node.op != "call_function":
                        continue
                    reads = node.all_input_nodes
                    if node.target is memory_ops.load or node.target in flat_loads:
                        if graph_id != top or not loaded.isdisjoint(reads):
                            return False
                        loaded.add(node)
                    elif graph_id == top and not loaded.isdisjoint(reads):
                        loaded.add(node)
            return True

        hint_roots = {root for root in prefetching - effectful if steers_only(root)}
        allocation_ids: dict[torch.UntypedStorage, int] = {}
        accesses: list[TileAccess] = []
        # The indexing slot of each load and store: codegen numbers them in
        # the order it lowers them (one in a ``while`` condition keeps its
        # first slot here).
        slots: dict[torch.fx.Node, int] = {}
        for _graph_id, node, slot in memory_op_slots(
            device_ir.graphs, device_ir.root_ids
        ):
            if node.target in (memory_ops.load, memory_ops.store):
                slots.setdefault(node, slot)

        def allocation_of(
            fake: torch.Tensor,
        ) -> tuple[torch.UntypedStorage, int | None] | None:
            """The local allocation a tensor views and the rank whose copy a
            peer view reaches (None: this rank's), or None when unknown."""
            source = host.compiler_state.ssa_source(fake)
            if source is None:
                return None
            storage = source.untyped_storage()
            return host.compiler_state.peer_views.get(storage, (storage, None))

        # Allocations some memory op of the kernel writes: a load of this
        # rank's copy of any other one reads one value for the whole launch.
        written: set[torch.UntypedStorage] = set()
        for graph_analysis in self.graphs:
            for node in graph_analysis.nodes:
                if node.op == "call_function" and (
                    node.target is memory_ops.store or node.target in ATOMIC_OPS
                ):
                    fake = _accessed_tensor_fake(node)
                    allocation = None if fake is None else allocation_of(fake)
                    if allocation is not None:
                        written.add(allocation[0])

        def uniform_tensor(fake: torch.Tensor) -> object | None:
            allocation = allocation_of(fake)
            origin = host.tensor_to_origin.get(fake)
            if (
                allocation is None
                or allocation[1] is not None
                or allocation[0] in written
                or origin is None
                or origin.root_rw_name() is None
            ):
                return None
            # The whole host path (gates[0] and gates[1] differ).
            return (
                origin.host_str(),
                str(fake.dtype),
                tuple(map(str, fake.shape)),
                tuple(map(str, fake.stride())),
                str(fake.storage_offset()),
            )

        for graph_analysis in self.graphs:
            for graph_node_index, node in enumerate(graph_analysis.nodes):
                if node.op != "call_function":
                    continue
                is_flat_load = node.target in flat_loads
                is_load = node.target is memory_ops.load or is_flat_load
                is_store = node.target is memory_ops.store
                is_atomic = node.target in ATOMIC_OPS
                if not (is_load or is_store or is_atomic) or is_flat_load:
                    # A flat load of symmetric memory is no peer load either.
                    if env.process_group_name is not None and (
                        _may_access_memory_unseen(node, host.compiler_state)
                    ):
                        raise exc.CrossLoopSchedulingError(
                            f"because {node.name} may access another rank's "
                            "memory outside a load or store"
                        )
                    if not is_flat_load:
                        continue

                fake = _accessed_tensor_fake(node)
                if fake is not None:
                    # An SSA copy stands for its source; an ambiguous join is unknown.
                    fake = host.compiler_state.ssa_source(fake)
                origin = host.tensor_to_origin.get(fake) if fake is not None else None
                allocation_id = -1
                owner_rank: int | None = None
                tensor_shape: tuple[sympy.Expr, ...] = ()
                tensor_strides: tuple[sympy.Expr, ...] = ()
                storage_offset: sympy.Expr = sympy.Integer(0)
                subscript_dims: tuple[int, ...] = ()
                subscript_affine_block_ids: tuple[int | None, ...] = ()
                subscript_index_scales: tuple[int, ...] = ()
                subscript_offsets: tuple[int | None, ...] = ()
                subscript_is_scalar: tuple[bool, ...] = ()
                subscript_is_full_slice: tuple[bool, ...] = ()
                subscript_static_extents: tuple[int | None, ...] = ()
                subscript_dense_spans: tuple[tuple[int, int, int] | None, ...] = ()
                affine_subscript_ranges = None
                layout_is_symbolically_exact = False

                if fake is not None:
                    # A peer view stands for the local allocation on its owner
                    # rank; an unmapped peer view has none (allocation_id
                    # stays -1 and the check below rejects the kernel).
                    allocation = allocation_of(fake)
                    if allocation is not None:
                        storage, owner_rank = allocation
                        allocation_id = allocation_ids.setdefault(
                            storage, len(allocation_ids)
                        )

                    def symbolic_layout_value(
                        value: int | torch.SymInt,
                    ) -> sympy.Expr:
                        if type(value) is int:
                            return sympy.Integer(value)
                        if not isinstance(value, torch.SymInt):
                            raise TypeError(
                                "tensor layout values must be integers or SymInts"
                            )
                        return env.shape_env.simplify(_symint_sympy_expr(value))

                    raw_layout = (
                        *fake.shape,
                        *fake.stride(),
                        fake.storage_offset(),
                    )
                    symbolic_layout = tuple(
                        symbolic_layout_value(value) for value in raw_layout
                    )
                    layout_is_symbolically_exact = not any(
                        _has_unbacked(expression) for expression in symbolic_layout
                    )
                    rank = fake.ndim
                    tensor_shape = symbolic_layout[:rank]
                    tensor_strides = symbolic_layout[rank : 2 * rank]
                    storage_offset = symbolic_layout[-1]
                    # Local views of a symmetric allocation are placed from its
                    # base, as its peer views are.
                    base = host.compiler_state.symmetric_bases.get(
                        fake.untyped_storage()
                    )
                    if base is not None:
                        storage_offset = env.shape_env.simplify(
                            storage_offset - sympy.sympify(base)
                        )
                    index_list = node.args[1] if len(node.args) >= 2 else None
                    if isinstance(index_list, (list, tuple)):
                        subscript_dims = tuple(range(min(len(index_list), fake.ndim)))
                        affine = tuple(
                            _subscript_affine(env, host, index_list[position])
                            for position in subscript_dims
                        )
                        subscript_affine_block_ids = tuple(item[0] for item in affine)
                        subscript_index_scales = tuple(item[1] for item in affine)
                        subscript_offsets = tuple(item[2] for item in affine)
                        subscript_is_scalar = tuple(item[3] for item in affine)
                        subscript_is_full_slice = tuple(
                            is_full_slice(index_list[position])
                            for position in subscript_dims
                        )
                        subscript_static_extents = tuple(
                            _subscript_static_extent(index_list[position])
                            for position in subscript_dims
                        )
                        subscript_dense_spans = tuple(
                            _subscript_dense_span(env, host, index_list[position])
                            for position in subscript_dims
                        )
                        if fake.ndim == 1 and subscript_dims == (0,):
                            affine_subscript_ranges = _affine_subscript_ranges(
                                env,
                                index_list[0],
                            )

                if allocation_id < 0 and env.process_group_name is not None:
                    # A barrier cannot order an unknown access against other ranks.
                    raise exc.CrossLoopSchedulingError(
                        "because a memory operation's allocation identity is "
                        "unavailable"
                    )
                has_explicit_mask = (
                    not is_atomic
                    and len(node.args) > (2 if is_load else 3)
                    and node.args[2 if is_load else 3] is not None
                )
                mask_uniform_terms: frozenset[object] = frozenset()
                mask_is_uniform = False
                if has_explicit_mask and not is_flat_load:
                    mask_uniform_terms, mask_is_uniform = _launch_uniform_terms(
                        node.args[2 if is_load else 3], uniform_tensor
                    )
                owner_roots = owners_of(graph_analysis.graph_id)
                if owner_roots:
                    node.meta[TILE_ACCESS_META] = tuple(
                        range(len(accesses), len(accesses) + len(owner_roots))
                    )
                for owner_root in owner_roots:
                    accesses.append(
                        TileAccess(
                            access_id=len(accesses),
                            memory_op_index=(
                                -1 if is_atomic or is_flat_load else slots[node]
                            ),
                            graph_id=graph_analysis.graph_id,
                            root=owner_root,
                            allocation_id=allocation_id,
                            owner_rank=owner_rank,
                            dtype=fake.dtype if fake is not None else None,
                            kind="load" if is_load else "store",
                            tensor_name=origin.root_rw_name() if origin else None,
                            tensor_shape=tensor_shape,
                            tensor_strides=tensor_strides,
                            storage_offset=storage_offset,
                            subscript_dims=subscript_dims,
                            subscript_affine_block_ids=subscript_affine_block_ids,
                            subscript_index_scales=subscript_index_scales,
                            subscript_offsets=subscript_offsets,
                            subscript_is_scalar=subscript_is_scalar,
                            has_explicit_mask=has_explicit_mask,
                            mask_uniform_terms=mask_uniform_terms,
                            mask_is_uniform=mask_is_uniform,
                            subscript_is_full_slice=subscript_is_full_slice,
                            subscript_static_extents=subscript_static_extents,
                            subscript_dense_spans=subscript_dense_spans,
                            is_atomic=is_atomic,
                            layout_is_symbolically_exact=layout_is_symbolically_exact,
                            graph_node_index=graph_node_index,
                            affine_subscript_ranges=affine_subscript_ranges,
                            hint=is_load and owner_root in hint_roots,
                        )
                    )

        return tuple(accesses)

    def pointwise_fact(
        self,
        spec: ConfigSpec,
    ) -> PointwiseElementwiseFact | None:
        """Build the fact for a pure, tile-independent elementwise kernel.

        Any reduction rules it out, including the ones the reduction seeds do
        not size (``SIZED_REDUCTION_CATEGORIES``): a partial one along a grid
        tile (``tile.amax(0)``) depends on the tile.
        """
        from ..language import memory_ops
        from ..language.atomic_ops import ATOMIC_OPS
        from ..language.inline_asm_ops import inline_asm_elementwise
        from ..language.inline_triton_ops import inline_triton
        from ..language.inline_triton_ops import triton_kernel
        from ..language.reduce_ops import _reduce
        from ..language.scan_ops import _associative_scan

        reduction_fact = spec.reduction_kernel_fact
        if (
            (reduction_fact is not None and reduction_fact.reductions)
            or spec.matmul_facts
            or spec.accumulator_facts
        ):
            return None
        if not spec.memory_op_facts or not spec.block_sizes:
            return None

        unsafe_pointwise_ops = frozenset(
            {
                _associative_scan,
                _reduce,
                inline_triton,
                triton_kernel,
                inline_asm_elementwise,
                *ATOMIC_OPS,
            }
        )
        total_numel = 1
        for block_size in spec.block_sizes:
            total_numel *= block_size.size_hint
        total_numel = max(1, total_numel)

        def value_itemsizes(value: object) -> list[int]:
            if isinstance(value, torch.Tensor):
                return [value.dtype.itemsize]
            if isinstance(value, (list, tuple)):
                return [
                    itemsize for item in value for itemsize in value_itemsizes(item)
                ]
            return []

        data_nodes: set[int] = set()
        data_path_buffer_width = 1
        for graph in self.graphs:
            for node in reversed(graph.nodes):
                if node.op != "call_function":
                    continue
                is_load = node.target is memory_ops.load
                if is_load or node.target is memory_ops.store:
                    for position in (2, 3) if not is_load else (2,):
                        if len(node.args) > position and isinstance(
                            node.args[position],
                            torch.fx.Node,
                        ):
                            data_nodes.add(id(node.args[position]))
                    accessed = _accessed_tensor_fake(node)
                    if accessed is not None:
                        data_path_buffer_width = max(
                            data_path_buffer_width,
                            accessed.dtype.itemsize,
                        )
                    if is_load:
                        continue
                if id(node) in data_nodes:
                    data_nodes.update(id(arg) for arg in node.all_input_nodes)

        sfu_ops = frozenset(
            {
                "sin",
                "cos",
                "tan",
                "tanh",
                "asin",
                "acos",
                "atan",
                "sinh",
                "cosh",
                "atanh",
                "asinh",
                "acosh",
                "exp",
                "exp2",
                "expm1",
                "log",
                "log2",
                "log10",
                "log1p",
                "sqrt",
                "rsqrt",
                "sigmoid",
                "erf",
                "erfc",
                "pow",
                "reciprocal",
            }
        )
        compute_itemsize = 1
        sfu_op_count = 0
        for graph in self.graphs:
            for node in graph.nodes:
                if id(node) in data_nodes:
                    for itemsize in value_itemsizes(node.meta.get("val")):
                        compute_itemsize = max(compute_itemsize, itemsize)
                if node.op == "call_function":
                    if node.target in unsafe_pointwise_ops:
                        if not spec.backend.pointwise_op_is_pure_scalar(node):
                            return None
                    base = getattr(node.target, "__name__", str(node.target)).split(
                        "."
                    )[0]
                    if base in sfu_ops:
                        sfu_op_count += 1
        if compute_itemsize == 1:
            compute_itemsize = max(1, data_path_buffer_width)

        tiled_ids = {block_size.block_id for block_size in spec.block_sizes}
        contiguous_block_ids: set[int] = set()
        slab_numel = 0
        max_op_slab_numel = 1
        storage_itemsize = 1
        gather_stride = 1
        # The loads behind a loaded index: along which blocks do the row loads
        # pick their rows, and along which do the element gathers fetch.
        # Whatever the slab rules below make of such a load, it is one: a
        # table smaller than the output included.
        row_load = False
        row_block_ids: set[int | None] = set()
        element_block_ids: list[int | None] = []
        # The rolled alternative of a full slice's loop lists every op a second
        # time, and a config runs one of the two. The copy reads a subscript
        # computed outside the loop as a placeholder (KNOWN_GAPS.md,
        # rolled-graph-subscripts-unresolved), so the widest op is the widest of
        # the ops outside it.
        rolled_graph_ids = {
            graph.graph_id for graph in self.graphs if graph.is_reduction_loop
        }
        for memory_fact in spec.memory_op_facts:
            if memory_fact.kind == "load" and memory_fact.row_gather is not None:
                if memory_fact.row_gather:
                    row_load = True
                    row_block_ids.update(memory_fact.loaded_index_block_ids)
                else:
                    element_block_ids.extend(memory_fact.loaded_index_block_ids)
            # A broadcast operand is amortized. This stays ahead of the gather
            # accounting: an op that touches less than the problem is not
            # counted, however it is indexed.
            if memory_fact.dtype is None or memory_fact.accessed_numel < total_numel:
                continue
            if memory_fact.gather_numel is not None:
                # A gathered op is addressed once per index element, however
                # long the gathered axis is. The tensor-size ratio would
                # charge it the table's rows per index element (vocab /
                # tokens for an embedding lookup), and the extents below the
                # whole gathered axis.
                op_slab = memory_fact.gather_numel
                slab_numel += op_slab
            else:
                slab_numel += memory_fact.accessed_numel // total_numel
                op_slab = 1
                for block_id, extent in zip(
                    memory_fact.subscript_affine_block_ids,
                    memory_fact.subscript_extents,
                    strict=True,
                ):
                    if block_id is None or block_id not in tiled_ids:
                        op_slab *= max(1, extent)
            if memory_fact.graph_id not in rolled_graph_ids:
                max_op_slab_numel = max(max_op_slab_numel, op_slab)
            storage_itemsize = max(storage_itemsize, memory_fact.dtype.itemsize)
            for block_id, stride in zip(
                memory_fact.subscript_block_ids,
                memory_fact.subscript_strides,
                strict=True,
            ):
                if block_id is not None and stride == 1 and block_id in tiled_ids:
                    contiguous_block_ids.add(block_id)
            for block_id, scale, stride in zip(
                memory_fact.subscript_affine_block_ids,
                memory_fact.subscript_index_scales,
                memory_fact.subscript_strides,
                strict=True,
            ):
                if block_id is not None and block_id in tiled_ids and stride == 1:
                    gather_stride = max(gather_stride, scale)

        return PointwiseElementwiseFact(
            total_numel=total_numel,
            slab_numel=slab_numel,
            storage_itemsize=storage_itemsize,
            compute_itemsize=compute_itemsize,
            contig_block_ids=tuple(sorted(contiguous_block_ids)),
            sfu_ops=sfu_op_count,
            gather_stride=gather_stride,
            max_op_slab_numel=max_op_slab_numel,
            # An element gather beside the rows fetches once per element of
            # its index. Along the blocks the rows are picked along
            # (``scales[ids[tile]]``, one per row) that is nothing next to the
            # rows; along any other axis it may be as much as they move.
            row_gather=row_load
            and all(
                block_id is not None and block_id in row_block_ids
                for block_id in element_block_ids
            ),
        )

    def kernel_matmul_fact(
        self,
        env: CompileEnvironment,
        spec: ConfigSpec,
    ) -> KernelMatmulFact | None:
        """Compose the whole-kernel contraction fact from analyzed FX graphs."""
        facts = spec.matmul_facts
        if not facts:
            return None

        def axis_extent_or_none(block_id: int | None) -> int | None:
            if block_id is not None and 0 <= block_id < len(env.block_sizes):
                size = env.block_sizes[block_id].size
                if isinstance(size, (int, torch.SymInt)):
                    return max(1, env.size_hint(size))
            return None

        from ..language.matmul_ops import MATMUL_DIM_BLOCK_IDS_META
        from ..language.matmul_ops import MATMUL_FACT_ID_META

        nodes_by_fact_id: dict[int, tuple[int, torch.fx.Node]] = {}
        for graph_id, node in self.dot_nodes:
            fact_id = node.meta.get(MATMUL_FACT_ID_META)
            if (
                isinstance(fact_id, int)
                and 0 <= fact_id < len(facts)
                and fact_id not in nodes_by_fact_id
            ):
                nodes_by_fact_id[fact_id] = (graph_id, node)
        attribution_complete = (
            len(self.dot_nodes) == len(facts) == len(nodes_by_fact_id)
        )

        if attribution_complete:
            operand_positions = matmul_operand_positions()

            def shape_block_ids(node: torch.fx.Node) -> tuple[int | None, ...]:
                value = node.meta.get("val")
                if not isinstance(value, torch.Tensor):
                    return ()
                # Do not use ``resolve_block_id``: matching a concrete padded extent
                # could assign an unrelated fixed dimension to a reduction axis.
                result = [env.get_block_id(size) for size in value.shape]
                annotated = node.meta.get(MATMUL_DIM_BLOCK_IDS_META)
                if isinstance(annotated, (list, tuple)) and len(annotated) == len(
                    result
                ):
                    for index, block_id in enumerate(annotated):
                        if isinstance(block_id, int):
                            result[index] = block_id
                return tuple(result)

            def choose_axis(
                original: int | None,
                *inferred: int | None,
            ) -> int | None:
                if original is not None:
                    return original
                candidates = {value for value in inferred if value is not None}
                return candidates.pop() if len(candidates) == 1 else None

            # Keep block identity separate from representative fake sizes. A dot
            # establishes the identity of its output M/N dimensions; shape-preserving
            # pointwise operations carry it until a later dot consumes that tensor.
            for graph in self.graphs:
                for node in graph.nodes:
                    value = node.meta.get("val")
                    if not isinstance(value, torch.Tensor):
                        continue
                    result_ids = list(shape_block_ids(node))
                    fact_id = node.meta.get(MATMUL_FACT_ID_META)
                    if isinstance(fact_id, int) and fact_id in nodes_by_fact_id:
                        positions = operand_positions.get(node.target)
                        if positions is None:
                            continue
                        lhs = node.args[positions[0]]
                        rhs = node.args[positions[1]]
                        assert isinstance(lhs, torch.fx.Node)
                        assert isinstance(rhs, torch.fx.Node)
                        lhs_ids = shape_block_ids(lhs)
                        rhs_ids = shape_block_ids(rhs)
                        fact = facts[fact_id]
                        m_block_id = choose_axis(
                            fact.m_block_id,
                            lhs_ids[-2] if len(lhs_ids) >= 2 else None,
                        )
                        n_block_id = choose_axis(
                            fact.n_block_id,
                            rhs_ids[-1] if rhs_ids else None,
                        )
                        k_block_id = choose_axis(
                            fact.k_block_id,
                            lhs_ids[-1] if lhs_ids else None,
                            rhs_ids[-2] if len(rhs_ids) >= 2 else None,
                        )
                        fact = fact._replace(
                            m_block_id=m_block_id,
                            n_block_id=n_block_id,
                            k_block_id=k_block_id,
                            static_m=axis_extent_or_none(m_block_id) or fact.static_m,
                            static_n=axis_extent_or_none(n_block_id) or fact.static_n,
                            static_k=axis_extent_or_none(k_block_id) or fact.static_k,
                        )
                        facts[fact_id] = fact
                        if len(result_ids) >= 2:
                            result_ids[-2:] = [m_block_id, n_block_id]
                    elif torch.Tag.pointwise in getattr(node.target, "tags", ()):
                        shape = tuple(map(str, value.shape))
                        matching_inputs = [
                            shape_block_ids(input_node)
                            for input_node in node.all_input_nodes
                            if isinstance(
                                input_value := input_node.meta.get("val"), torch.Tensor
                            )
                            and tuple(map(str, input_value.shape)) == shape
                        ]
                        for index in range(len(result_ids)):
                            if result_ids[index] is not None:
                                continue
                            candidates = {
                                input_ids[index]
                                for input_ids in matching_inputs
                                if len(input_ids) == len(result_ids)
                                and input_ids[index] is not None
                            }
                            if len(candidates) == 1:
                                result_ids[index] = candidates.pop()
                    if any(block_id is not None for block_id in result_ids):
                        node.meta[MATMUL_DIM_BLOCK_IDS_META] = tuple(result_ids)

        valid_block_ids = set(spec.block_sizes.valid_block_ids())

        def classify_axis(
            block_id: int | None,
            static_extent: int | None,
        ) -> tuple[DotAxisKind, int | None]:
            if block_id is not None and block_id in valid_block_ids:
                return DotAxisKind.TUNABLE_TILED, static_extent
            if block_id is not None:
                extent = _immovable_extent(env, spec, block_id)
                if extent is not None:
                    return DotAxisKind.FIXED_FULL_EXTENT, extent
            if static_extent is None:
                return DotAxisKind.UNKNOWN, None
            return DotAxisKind.FIXED_FULL_EXTENT, static_extent

        axes: list[DotAxes] = []
        for fact in facts:
            m_kind, m_extent = classify_axis(fact.m_block_id, fact.static_m)
            n_kind, n_extent = classify_axis(fact.n_block_id, fact.static_n)
            k_kind, k_extent = classify_axis(fact.k_block_id, fact.static_k)
            axes.append(
                DotAxes(
                    m_kind,
                    n_kind,
                    k_kind,
                    m_extent,
                    n_extent,
                    k_extent,
                )
            )

        knob_users: dict[int, list[tuple[int, str]]] = {}
        for index, fact in enumerate(facts):
            for axis, block_id in (
                ("m", fact.m_block_id),
                ("n", fact.n_block_id),
                ("k", fact.k_block_id),
            ):
                if block_id is not None and block_id in valid_block_ids:
                    knob_users.setdefault(block_id, []).append((index, axis))

        from .host_function import HostFunction
        from .variable_origin import BlockSizeOrigin

        origins = HostFunction.current().expr_to_origin

        def symbolic_loop_bound(block_id: int) -> SymbolicLoopBound | None:
            """Preserve candidate-dependent symbolic loop bounds."""
            size = env.block_sizes[block_id].size
            if not isinstance(size, torch.SymInt):
                return None
            expr = _symint_sympy_expr(size)

            block_size_symbols: list[tuple[sympy.Symbol, int]] = []
            tile_id_symbols: list[tuple[sympy.Symbol, int]] = []
            for symbol in sorted(
                _symint_free_symbols(size), key=lambda item: item.name
            ):
                origin_info = origins.get(symbol)
                if origin_info is None:
                    continue
                origin = origin_info.origin
                if isinstance(origin, BlockSizeOrigin):
                    block_size_symbols.append((symbol, origin.block_id))
                elif isinstance(origin, TileIdOrigin):
                    tile_id_symbols.append((symbol, origin.block_id))
            if not block_size_symbols and not tile_id_symbols:
                return None
            return SymbolicLoopBound(
                expr,
                tuple(block_size_symbols),
                tuple(tile_id_symbols),
            )

        symbolic_loop_bounds = {
            block_id: symbolic_loop_bound(block_id)
            for block_id in range(len(env.block_sizes))
        }

        grid_block_ids = set(spec.grid_block_ids)
        matmul_output_block_ids = {fact.m_block_id for fact in facts} | {
            fact.n_block_id for fact in facts
        }
        outer_grid = 1
        for block_id in spec.grid_block_ids:
            if block_id not in matmul_output_block_ids:
                outer_grid *= axis_extent_or_none(block_id) or 1

        bounded_by: dict[int, int] = {}
        bounded_extent: dict[int, int] = {}
        for slot in range(len(spec.block_sizes)):
            block_spec = spec.block_sizes[slot]
            parent = block_spec.bounded_by_block_id
            if parent is not None:
                bounded_by[block_spec.block_id] = parent
                if 0 <= parent < len(env.block_sizes):
                    source = env.block_sizes[parent].block_size_source
                    if isinstance(source, FixedBlockSizeSource) and isinstance(
                        source.value,
                        int,
                    ):
                        bounded_extent[block_spec.block_id] = max(1, source.value)

        def loop_axes_for(graph_id: int) -> tuple[LoopAxisFact, ...]:
            axes_out: list[LoopAxisFact] = []
            seen_axes: set[int] = set()
            current = graph_id
            while current in self.loop_block_ids:
                for block_id in sorted(self.loop_block_ids[current]):
                    if block_id in grid_block_ids or block_id in seen_axes:
                        continue
                    seen_axes.add(block_id)
                    bound = (
                        None
                        if block_id in bounded_by
                        else symbolic_loop_bounds.get(block_id)
                    )
                    axes_out.append(
                        LoopAxisFact(
                            block_id=block_id,
                            extent=(
                                None
                                if bound is not None
                                else axis_extent_or_none(block_id)
                            ),
                            bounded_by_block_id=bounded_by.get(block_id),
                            bounded_extent=bounded_extent.get(block_id),
                            symbolic_bound=bound,
                        )
                    )
                current = self.parent_of.get(current, -1)
            return tuple(axes_out)

        def max_trips_for(graph_id: int) -> int | None:
            trips = 1
            for axis in loop_axes_for(graph_id):
                extent = axis.extent
                if axis.symbolic_bound is not None:
                    # There is no candidate-independent upper bound for arbitrary
                    # symbolic arithmetic.
                    return None
                if axis.bounded_by_block_id is not None:
                    extent = (
                        axis.bounded_extent
                        or _immovable_extent(
                            env,
                            spec,
                            axis.bounded_by_block_id,
                        )
                        or axis_extent_or_none(axis.bounded_by_block_id)
                    )
                if extent is None:
                    return None
                block = _immovable_extent(env, spec, axis.block_id)
                trips *= max(1, -(-extent // block)) if block else extent
            return trips

        def strip_index_cast(value: object) -> object:
            for _ in range(4):
                if not isinstance(value, torch.fx.Node) or value.target not in (
                    torch.ops.prims.convert_element_type.default,
                    torch.ops.aten._to_copy.default,
                ):
                    break
                value = value.args[0]
            return value

        def is_single_segment_loop(body_graph_id: int, block_id: int) -> bool:
            calls = self.loop_calls.get(body_graph_id, ())
            if len(calls) != 1:
                return False
            block_id_order = self.by_id[body_graph_id].block_id_order
            if block_id not in block_id_order:
                return False
            position = block_id_order.index(block_id)
            call = calls[0]
            ends = call.args[2] if len(call.args) > 2 else None
            if not isinstance(ends, (list, tuple)) or position >= len(ends):
                return False

            difference = strip_index_cast(ends[position])
            if (
                not isinstance(difference, torch.fx.Node)
                or difference.target is not torch.ops.aten.sub.Tensor
            ):
                return False
            end = strip_index_cast(difference.args[0])
            begin = strip_index_cast(difference.args[1])

            from ..language import memory_ops

            if (
                not isinstance(end, torch.fx.Node)
                or end.target is not memory_ops.load
                or not isinstance(begin, torch.fx.Node)
                or begin.target is not memory_ops.load
                or end.args[0] is not begin.args[0]
            ):
                return False
            offsets = _accessed_tensor_fake(end)
            if (
                offsets is None
                or offsets.ndim != 1
                or env.size_hint(offsets.shape[0]) != 2
            ):
                return False
            end_indices = end.args[1] if len(end.args) > 1 else None
            begin_indices = begin.args[1] if len(begin.args) > 1 else None
            if (
                not isinstance(end_indices, (list, tuple))
                or len(end_indices) != 1
                or not isinstance(begin_indices, (list, tuple))
                or len(begin_indices) != 1
            ):
                return False
            end_index = strip_index_cast(end_indices[0])
            begin_index = strip_index_cast(begin_indices[0])
            if not isinstance(end_index, torch.fx.Node):
                return False
            if end_index.target not in (
                operator.add,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.add.Scalar,
            ):
                return False
            lhs, rhs = end_index.args[:2]
            return (lhs is begin_index and isinstance(rhs, int) and rhs == 1) or (
                rhs is begin_index and isinstance(lhs, int) and lhs == 1
            )

        def direct_operand_trip_count(
            node: torch.fx.Node,
            operand_index: int,
        ) -> tuple[int, int] | None:
            from ..language import memory_ops

            if operand_index >= len(node.args):
                return None
            operand = node.args[operand_index]
            load = trace_back_to_load(operand, memory_ops.load)
            if load is None or not isinstance(operand, torch.fx.Node):
                return None
            source = _accessed_tensor_fake(load)
            operand_value = operand.meta.get("val")
            if source is None or not isinstance(operand_value, torch.Tensor):
                return None

            source_numel = 1
            for size, stride in zip(source.shape, source.stride(), strict=True):
                if stride != 0:
                    source_numel *= max(1, env.size_hint(size))
            operand_numel = 1
            for size in operand_value.shape:
                operand_numel *= max(1, env.size_hint(size))
            denominator = max(1, outer_grid) * operand_numel
            if source_numel % denominator:
                return None
            return id(load), max(1, source_numel // denominator)

        dot_targets = matmul_operand_positions()
        dots_by_graph: dict[int, list[torch.fx.Node]] = {}
        for graph_id, node in self.dot_nodes:
            dots_by_graph.setdefault(graph_id, []).append(node)

        inferred_work_trips: dict[tuple[int, int], int] = {}
        for body_graph_id, block_ids in self.loop_block_ids.items():
            unresolved = [
                block_id
                for block_id in block_ids
                if block_id not in grid_block_ids
                and axis_extent_or_none(block_id) is None
                and block_id not in bounded_by
                and symbolic_loop_bounds.get(block_id) is None
            ]
            enclosing_axes = loop_axes_for(body_graph_id)
            if len(block_ids) != 1 or len(unresolved) != 1 or len(enclosing_axes) != 1:
                continue
            block_id = unresolved[0]
            if not is_single_segment_loop(body_graph_id, block_id):
                continue
            block = _immovable_extent(env, spec, block_id)
            if block is None:
                continue
            trip_by_load: dict[int, int] = {}
            for node in dots_by_graph.get(body_graph_id, ()):
                positions = dot_targets.get(node.target)
                if positions is None:
                    continue
                for operand_index in positions:
                    proof = direct_operand_trip_count(node, operand_index)
                    if proof is not None:
                        trip_by_load[proof[0]] = proof[1]
            trips = set(trip_by_load.values())
            if len(trip_by_load) >= 2 and len(trips) == 1:
                inferred_work_trips[(body_graph_id, block_id)] = trips.pop()

        sites: list[DotSite] = []
        if attribution_complete:
            for index, fact in enumerate(facts):
                graph_id, node = nodes_by_fact_id[index]
                value = node.meta.get("val")
                if isinstance(value, torch.Tensor) and value.ndim >= 2:
                    observed_extents: list[int] = []
                    annotated = node.meta.get(MATMUL_DIM_BLOCK_IDS_META)
                    annotated_tail = (
                        annotated[-2:]
                        if isinstance(annotated, (list, tuple))
                        and len(annotated) == value.ndim
                        else (None, None)
                    )
                    for dimension, annotated_block_id in zip(
                        value.shape[-2:],
                        annotated_tail,
                        strict=True,
                    ):
                        block_id = (
                            annotated_block_id
                            if isinstance(annotated_block_id, int)
                            else env.resolve_block_id(dimension)
                        )
                        if block_id is not None and 0 <= block_id < len(
                            env.block_sizes
                        ):
                            dimension = env.block_sizes[block_id].size
                        if not isinstance(dimension, (int, torch.SymInt)):
                            observed_extents.append(-1)
                            continue
                        observed_extents.append(env.size_hint(dimension))
                    expected_extents = [
                        fact.static_m,
                        fact.static_n,
                    ]
                    if any(
                        expected is not None and observed != -1 and expected != observed
                        for expected, observed in zip(
                            expected_extents,
                            observed_extents,
                            strict=False,
                        )
                    ):
                        attribution_complete = False
                updates_carry = (
                    self.by_id[graph_id].reaches_output(node)
                    and graph_id in self.loop_block_ids
                )
                rank_reduction_scaled_accumulator_batch_block_id = (
                    _rank_reduction_scaled_baddbmm_batch_block_id(
                        node,
                        env,
                    )
                    if updates_carry
                    else None
                )
                loop_axes = loop_axes_for(graph_id)
                exact_loop_trips = (
                    inferred_work_trips.get((graph_id, loop_axes[0].block_id))
                    if len(loop_axes) == 1
                    else None
                )
                sites.append(
                    DotSite(
                        graph_id,
                        updates_carry,
                        loop_axes,
                        exact_loop_trips,
                        max_trips_for(graph_id),
                        rank_reduction_scaled_accumulator_batch_block_id,
                    )
                )
        if not attribution_complete:
            sites = [DotSite(-1, False, (), None, None) for _ in facts]

        sequential_loop_trips = 1
        for body_graph_id, block_ids in self.loop_block_ids.items():
            if any(block_id in spec.grid_block_ids for block_id in block_ids):
                continue
            max_trips = max_trips_for(body_graph_id)
            if max_trips is not None:
                sequential_loop_trips = max(sequential_loop_trips, max_trips)

        pipelined_regions: list[PipelinedRegion] = []
        resident_regions: list[ResidentRegion] = []
        for graph in self.non_reduction_graphs:
            loop_axes = (
                loop_axes_for(graph.graph_id)
                if graph.graph_id in self.loop_block_ids
                else ()
            )
            tiles = graph.memory_tiles_for_loop_axes(
                env,
                frozenset(axis.block_id for axis in loop_axes),
            )
            if not tiles:
                continue
            if graph.graph_id in self.loop_block_ids:
                pipelined_regions.append(PipelinedRegion(loop_axes, tuple(tiles)))
            else:
                resident_regions.append(ResidentRegion(tuple(tiles)))

        resolved = tuple(
            starmap(ResolvedMatmulFact, zip(facts, axes, sites, strict=True))
        )
        return KernelMatmulFact(
            matmuls=resolved,
            knob_users=tuple(
                (block_id, tuple(knob_users[block_id]))
                for block_id in sorted(knob_users)
            ),
            sequential_loop_trips=sequential_loop_trips,
            live_dot_outputs=tuple(self.kernel_peak_dot_outputs()),
            live_promoted_lhs=tuple(self.kernel_peak_promoted_lhs()),
            live_tile_steps=tuple(self.kernel_live_tile_steps()),
            pipelined_regions=tuple(pipelined_regions),
            resident_regions=tuple(resident_regions),
            attribution_complete=attribution_complete,
        )

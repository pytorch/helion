"""Drop the host fill of a zero-initialized output the kernel writes completely.

``loss = torch.zeros([n], ...)`` followed by a device loop that stores every
element of ``loss`` costs one memset launch per output. On the small shapes
where a Helion kernel is a few microseconds, those memsets (1-2 us each in a
CUDA graph) can exceed the kernel itself: grpo_loss forward allocates four
such outputs. The fill is dead when the kernel provably overwrites every
element, so the factory becomes the matching ``empty`` call.

The proof is deliberately narrow. The host statement must be a top-level
``name = torch.zeros(...)`` / ``torch.zeros_like(...)`` / ``x.new_zeros(...)``
assignment whose name is bound once and is not read by any host statement
before the last device loop. On the device, every access to that tensor must
be an unmasked ``store`` whose indices are tiles, one per tensor dimension:
tiles of the root grid, or of enclosing ``hl.tile(extent)`` loops that are
not under a device ``if`` and run over their whole ``[0, extent)`` range.
Each tile spans ``[0, extent)``; partial tiles only mask out-of-range
elements, so such a store writes a full prefix of every dimension. When one
dimension is longer than its tile extent by a constant, a top-level host
assignment after the device loops must fill exactly that remainder
(``grad[:, -1, :] = 0`` after a loop over ``L = L_ADD_1 - 1`` rows); no host
statement may read the tensor's values before that fill. Any view, atomic,
masked or conditional store keeps the original fill.

A dimension may also be indexed by a root tile's ``.id`` (a split-K
workspace ``tmp[tile_m, tile_n, tile_k.id]``): the ids cover ``[0, tiles)``
where ``tiles = cdiv(extent, block_size)`` is evaluated for the configuration
being compiled (the block size and the workspace extent are usually
expressions of one ``hl.register_tunable`` value), so the proof is per
config and the fill stays for a config that leaves rows unwritten.

Only root tiles that start at zero with unit step count (a root
``hl.tile(begin, end)`` indexes ``[begin, end)`` and numbers its tiles from
``begin // block_size``), and only blocks with a static extent.

Device loads of the tensor are admitted only from a later ``hl.barrier()``
phase than every store: the grid-wide barrier orders the complete stores
before the first read, exactly as it does for the original program. A load
in the same phase as a store (or an earlier one) keeps the fill.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..ast_extension import statement_from_string
from ..device_ir import ForLoopGraphInfo
from ..device_ir import ReductionLoopGraphInfo
from ..device_ir import RootGraphInfo
from ..device_ir import control_flow_parent_entries
from ..type_info import LiteralType
from ..type_info import TensorType
from ..variable_origin import BlockSizeOrigin
from ..variable_origin import TileIdOrigin

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..compile_environment import CompileEnvironment
    from ..device_ir import GraphInfo
    from ..host_function import HostFunction
    from ..source_location import SourceLocation

_EMPTY_FACTORY = {
    "zeros": "empty",
    "zeros_like": "empty_like",
    "new_zeros": "new_empty",
}


@dataclass(frozen=True)
class ZeroFill:
    location: SourceLocation
    statement: str
    target: str
    factory: str


def _zero_factory(statement: ast.AST) -> tuple[str, str, ast.Call] | None:
    """``(target, factory, call)`` for ``target = <zero factory>(...)``."""
    if not (
        isinstance(statement, ast.Assign)
        and isinstance(statement, ExtendedAST)
        and len(statement.targets) == 1
        and isinstance(target := statement.targets[0], ast.Name)
        and isinstance(call := statement.value, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in _EMPTY_FACTORY
        # Store-context names carry no type; the call's result type binds the
        # fake tensor that device ``_host_tensor`` nodes reference.
        and isinstance(call, ExtendedAST)
        and isinstance(target_type := call._type_info, TensorType)
    ):
        return None
    owner = call.func.value
    owner_type = owner._type_info if isinstance(owner, ExtendedAST) else None
    if call.func.attr == "new_zeros":
        if not isinstance(owner_type, TensorType):
            return None
    elif not (isinstance(owner_type, LiteralType) and owner_type.value is torch):
        return None
    if any(keyword.arg in ("requires_grad", "out") for keyword in call.keywords):
        # ``out=`` fills another name's storage; ``requires_grad`` wants the
        # zeros to be a leaf of the autograd graph.
        return None
    if target_type.fake_value.ndim == 0:
        return None
    return target.id, call.func.attr, call


def _same_extent(numel: object, size: object) -> bool:
    if isinstance(size, torch.SymInt):
        size = size.node.expr
    if isinstance(numel, torch.SymInt):
        numel = numel.node.expr
    if isinstance(numel, int) and isinstance(size, int):
        return numel == size
    try:
        return sympy.simplify(sympy.sympify(numel) - sympy.sympify(size)) == 0
    except (TypeError, ValueError):
        return False


_METADATA_ATTRIBUTES = frozenset(
    {
        "shape",
        "size",
        "dtype",
        "device",
        "stride",
        "ndim",
        "dim",
        "numel",
        "element_size",
        "is_contiguous",
        "layout",
    }
)
_LIKE_FACTORIES = frozenset(
    {"empty_like", "zeros_like", "ones_like", "full_like", "rand_like", "randn_like"}
)


def _reads(statement: ast.AST, name: str) -> bool:
    """Whether ``statement`` can observe the values of ``name``.

    Metadata reads (``name.shape``, ``torch.empty_like(name)``) never touch
    the storage, so an output allocated from another output's metadata does
    not keep the fill alive.
    """
    metadata_only: set[int] = set()
    for node in ast.walk(statement):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in _METADATA_ATTRIBUTES
            and isinstance(node.value, ast.Name)
        ):
            metadata_only.add(id(node.value))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LIKE_FACTORIES
            and isinstance(owner := node.func.value, ast.Name)
            and owner.id == "torch"
            and isinstance(
                owner_type := getattr(owner, "_type_info", None), LiteralType
            )
            and owner_type.value is torch
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            metadata_only.add(id(node.args[0]))
    return any(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Load)
        and id(node) not in metadata_only
        for node in ast.walk(statement)
    )


def _assigns(statement: ast.AST, name: str) -> int:
    return sum(
        isinstance(node, ast.Name)
        and node.id == name
        and isinstance(node.ctx, ast.Store)
        for node in ast.walk(statement)
    )


def _static_numel(env: CompileEnvironment, block_id: int) -> sympy.Expr | None:
    """The block's extent, or ``None`` when it has no static size (a
    data-dependent bound, or an ``hl.register_block_size`` block reused at
    two extents)."""
    size = env.block_sizes[block_id].size
    if not isinstance(size, (int, torch.SymInt)):
        return None
    return env.block_sizes[block_id].numel


def _constant_remainder(numel: object, size: object) -> int | None:
    """``size - numel`` when it is a non-negative integer constant."""
    if isinstance(size, torch.SymInt):
        size = size.node.expr
    if isinstance(numel, torch.SymInt):
        numel = numel.node.expr
    try:
        difference = sympy.simplify(sympy.sympify(size) - sympy.sympify(numel))
    except (TypeError, ValueError):
        return None
    if difference.is_Integer and int(difference) >= 0:
        return int(difference)
    return None


def _config_int(value: object, env: CompileEnvironment, config: Config) -> int | None:
    """``value`` as the integer it takes for ``config``.

    Host integer helpers over ``hl.register_tunable`` values trace as unbacked
    symbols whose config expression the environment keeps; every other symbol
    (a dynamic input size) makes the value unknown.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, torch.SymInt):
        expr = value.node.expr
    elif isinstance(value, sympy.Basic):
        expr = value
    else:
        return None
    substitutions: dict[sympy.Basic, sympy.Integer] = {}
    for symbol in expr.free_symbols:
        expression = (
            env.config_value_expressions.get(symbol)
            if isinstance(symbol, sympy.Symbol)
            else None
        )
        if expression is None:
            return None
        substitutions[symbol] = sympy.Integer(expression.evaluate(config))
    resolved = sympy.simplify(expr.xreplace(substitutions)) if substitutions else expr
    if isinstance(resolved, sympy.Integer):
        return int(resolved)
    return None


def _index_block(index: torch.SymInt, host: HostFunction) -> tuple[int, bool] | None:
    """``(block_id, is_tile_id)`` for a tile index, else ``None``.

    A tile indexes its whole extent only through its block-size symbol
    (``BlockSizeOrigin``); ``tile.begin`` / ``tile.end`` / ``tile.count`` are
    scalars that also resolve to the block id through ``get_block_id`` but
    write a single element, so they are not coverage. ``tile.id`` is the
    one scalar with a known range.
    """
    origin = host.expr_to_origin.get(index.node.expr)
    if origin is None:
        return None
    if isinstance(origin.origin, BlockSizeOrigin):
        return origin.origin.block_id, False
    if isinstance(origin.origin, TileIdOrigin):
        return origin.origin.block_id, True
    return None


def _tile_id_remainder(
    block_id: int,
    size: object,
    env: CompileEnvironment,
    config: Config,
) -> int | None:
    """``size - tiles`` for a dimension indexed by a tile's ``.id``.

    The ids of an unconditional whole-extent loop run over ``[0, tiles)`` with
    ``tiles = cdiv(extent, block_size)``, both evaluated for ``config``.
    """
    info = env.block_sizes[block_id]
    extent = _config_int(info.size, env, config)
    block_size = _config_int(info.from_config(config), env, config)
    size_value = _config_int(size, env, config)
    if extent is None or block_size is None or size_value is None or block_size <= 0:
        return None
    tiles = -(-extent // block_size)
    if size_value < tiles:
        return None
    return size_value - tiles


def _top_graph(
    node: Node,
    graphs: list[GraphInfo],
    parents: dict[int, tuple[Node, int]],
) -> GraphInfo:
    """The outermost graph reachable from ``node`` through the loop/if calls."""
    by_graph = {info.graph: info for info in graphs}
    info = by_graph[node.graph]
    while info.graph_id in parents:
        call, _slot = parents[info.graph_id]
        info = by_graph[call.graph]
    return info


def _store_coverage(
    node: Node,
    tensor: Node,
    fake: torch.Tensor,
    env: CompileEnvironment,
    graphs: list[GraphInfo],
    parents: dict[int, tuple[Node, int]],
    roots: dict[torch.fx.Graph, set[int]],
    host: HostFunction,
    config: Config,
) -> tuple[int, ...] | None:
    """Per-dimension remainder a complete tile store leaves unwritten.

    ``None`` when the store is masked, conditional, inside a partial-range or
    stepped loop, or indexed by anything but tiles (or tile ids) of the
    enclosing loops.
    """
    if node.target is not memory_ops.store or node.args[0] is not tensor:
        return None
    if len(node.args) > 3 and node.args[3] is not None:
        return None
    if any(arg is tensor for arg in node.args[1:]):
        return None
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)) or len(indices) != fake.ndim:
        return None
    # Walk to the root: every enclosing graph must be an unconditional
    # ``hl.tile`` loop over its whole extent; collect the tiles in scope.
    by_graph = {info.graph: info for info in graphs}
    info = by_graph[node.graph]
    allowed: set[int] = set()
    while info.graph_id in parents:
        call, _slot = parents[info.graph_id]
        if (
            call.target is not _tracing_ops._for_loop
            or not isinstance(info, ForLoopGraphInfo)
            or len(call.args) < 4
        ):
            return None
        begins, ends = call.args[1], call.args[2]
        if not (
            isinstance(begins, (list, tuple))
            and isinstance(ends, (list, tuple))
            and len(begins) == len(ends) == len(info.block_ids)
        ):
            return None
        for block_id, begin, end in zip(info.block_ids, begins, ends, strict=True):
            if isinstance(begin, Node):
                begin = begin.meta.get("val")
            if isinstance(end, Node):
                end = end.meta.get("val")
            if env.block_sizes[block_id].size is None:
                # A jagged tile (``hl.jagged_tile``) has no static extent to
                # cover: the rows it walks are per-sequence.
                return None
            numel = _static_numel(env, block_id)
            if (
                numel is None
                or not _same_extent(begin, 0)
                or not _same_extent(numel, end)
            ):
                return None
            allowed.add(block_id)
        info = by_graph[call.graph]
    grid = roots.get(info.graph)
    if grid is None:
        return None
    allowed |= grid
    remainders: list[int] = []
    block_ids: list[int] = []
    for index, size in zip(indices, fake.shape, strict=True):
        if isinstance(index, Node):
            # Tile indices are traced as symbolic-size nodes.
            index = index.meta.get("val")
        if not isinstance(index, torch.SymInt):
            return None
        found = _index_block(index, host)
        if found is None:
            return None
        block_id, is_tile_id = found
        if block_id not in allowed or block_id in block_ids:
            return None
        if is_tile_id:
            remainder = _tile_id_remainder(block_id, size, env, config)
        else:
            numel = _static_numel(env, block_id)
            remainder = _constant_remainder(numel, size) if numel is not None else None
        if remainder is None:
            return None
        block_ids.append(block_id)
        remainders.append(remainder)
    return tuple(remainders)


def _remainder_fill(statement: ast.AST, name: str, remainders: tuple[int, ...]) -> bool:
    """``name[:, ..., -c, ...] = <value>`` writing exactly the uncovered rows."""
    if not (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(target := statement.targets[0], ast.Subscript)
        and isinstance(target.value, ast.Name)
        and target.value.id == name
        and not _reads(statement.value, name)
    ):
        return False
    items = (
        list(target.slice.elts)
        if isinstance(target.slice, ast.Tuple)
        else [target.slice]
    )
    if len(items) != len(remainders):
        return False
    for item, remainder in zip(items, remainders, strict=True):
        if remainder == 0:
            if not (
                isinstance(item, ast.Slice)
                and item.lower is None
                and item.upper is None
                and item.step is None
            ):
                return False
            continue
        negative = (
            isinstance(item, ast.UnaryOp)
            and isinstance(item.op, ast.USub)
            and isinstance(item.operand, ast.Constant)
            and item.operand.value == remainder
        )
        if remainder == 1 and negative:
            continue
        if (
            isinstance(item, ast.Slice)
            and item.upper is None
            and item.step is None
            and isinstance(item.lower, ast.UnaryOp)
            and isinstance(item.lower.op, ast.USub)
            and isinstance(item.lower.operand, ast.Constant)
            and item.lower.operand.value == remainder
        ):
            continue
        return False
    return True


def _is_load(user: Node, tensor: Node) -> bool:
    return (
        user.target is memory_ops.load
        and user.args[0] is tensor
        and not any(arg is tensor for arg in user.args[1:])
    )


def prove_host_zero_fills(
    host: HostFunction, env: CompileEnvironment, config: Config
) -> list[ZeroFill]:
    device_ir = host.device_ir
    graphs = device_ir.graphs
    # A root ``hl.tile(begin, end)`` with a nonzero begin (or a step) indexes
    # ``[begin, end)`` and numbers its tiles from ``begin // block_size``; its
    # blocks never count as coverage of ``[0, extent)``.
    skewed = device_ir.noncanonical_task_origin_block_ids
    roots = {
        graphs[root_id].graph: set(block_ids) - skewed
        for root_id, block_ids in zip(
            device_ir.root_ids, device_ir.grid_block_ids, strict=True
        )
        if isinstance(graphs[root_id], RootGraphInfo)
    }
    if not roots:
        return []
    # Keyed by graph object: a config that rolls a reduction replaces a root's
    # graph payload and appends a second info for it without the phase.
    root_phases: dict[torch.fx.Graph, int] = {}
    for root_id in device_ir.root_ids:
        root = graphs[root_id]
        if isinstance(root, RootGraphInfo):
            root_phases[root.graph] = root.phase_index
    device_loops = [
        index
        for index, statement in enumerate(host.body)
        if isinstance(statement, ExtendedAST) and statement._loop_type == LoopType.GRID
    ]
    if not device_loops:
        return []
    last_device_loop = device_loops[-1]
    host_tensors: dict[torch.Tensor, list[Node]] = {}
    for info in graphs:
        for node in info.graph.nodes:
            if node.target is _tracing_ops._host_tensor:
                value = node.meta.get("val")
                if isinstance(value, torch.Tensor):
                    host_tensors.setdefault(value, []).append(node)
    parents = control_flow_parent_entries(graphs)
    result: list[ZeroFill] = []
    for index, statement in enumerate(host.body):
        if index >= last_device_loop:
            break
        found = _zero_factory(statement)
        if found is None:
            continue
        name, factory, _call = found
        if sum(_assigns(other, name) for other in host.body) != 1:
            continue
        if any(
            _reads(other, name)
            for other in host.body[index + 1 : last_device_loop + 1]
            if not (
                isinstance(other, ExtendedAST) and other._loop_type == LoopType.GRID
            )
        ):
            continue
        assert isinstance(statement, ast.Assign) and isinstance(statement, ExtendedAST)
        assert isinstance(statement.value, ExtendedAST)
        target_type = statement.value._type_info
        assert isinstance(target_type, TensorType)
        fake = target_type.fake_value
        nodes = host_tensors.get(fake, [])
        if not nodes:
            continue
        coverages: set[tuple[int, ...] | None] = set()
        store_phases: list[int] = []
        load_phases: list[int] = []
        ordered = True
        for tensor in nodes:
            for user in tensor.users:
                top = _top_graph(user, graphs, parents)
                phase = root_phases.get(top.graph)
                if phase is None:
                    if isinstance(top, ReductionLoopGraphInfo):
                        # An unreferenced rolled copy of a root made while
                        # tracing: its accesses mirror the root's own.
                        continue
                    ordered = False
                elif _is_load(user, tensor):
                    load_phases.append(phase)
                    continue
                else:
                    store_phases.append(phase)
                coverages.add(
                    _store_coverage(
                        user, tensor, fake, env, graphs, parents, roots, host, config
                    )
                )
        if not ordered or len(coverages) != 1 or None in coverages:
            continue
        if load_phases and (not store_phases or max(store_phases) >= min(load_phases)):
            # Only a grid-wide barrier orders the loads after every store.
            continue
        (remainders,) = coverages
        assert remainders is not None
        if any(remainders):
            # One dimension may stop short of the tensor by a constant when a
            # host assignment after the device loops writes exactly that
            # remainder, with no value read of the tensor in between.
            if sum(1 for remainder in remainders if remainder) != 1:
                continue
            fill = next(
                (
                    position
                    for position in range(last_device_loop + 1, len(host.body))
                    if _remainder_fill(host.body[position], name, remainders)
                ),
                None,
            )
            if fill is None or any(
                _reads(other, name) for other in host.body[last_device_loop + 1 : fill]
            ):
                continue
        result.append(
            ZeroFill(
                statement._location,
                ast.unparse(statement),
                name,
                factory,
            )
        )
    return result


def lower_host_zero_fills(
    host: HostFunction, body: list[ast.AST], env: CompileEnvironment, config: Config
) -> list[ast.AST]:
    """Rewrite the proven fills whose statements survived generation unchanged."""
    proven = {fill.location: fill for fill in prove_host_zero_fills(host, env, config)}
    if not proven:
        return body
    result: list[ast.AST] = []
    for statement in body:
        fill = (
            proven.get(statement._location)
            if isinstance(statement, ExtendedAST)
            else None
        )
        if fill is None or ast.unparse(statement) != fill.statement:
            result.append(statement)
            continue
        assert isinstance(statement, ast.Assign)
        value = statement.value
        assert isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
        result.append(
            statement_from_string(
                f"{fill.target} = {_factory_call_source(value, _EMPTY_FACTORY[fill.factory])}"
            )
        )
    return result


def _factory_call_source(call: ast.Call, attr: str) -> str:
    arguments = [ast.unparse(argument) for argument in call.args]
    arguments.extend(
        f"{keyword.arg}={ast.unparse(keyword.value)}"
        if keyword.arg is not None
        else f"**{ast.unparse(keyword.value)}"
        for keyword in call.keywords
    )
    assert isinstance(call.func, ast.Attribute)
    return f"{ast.unparse(call.func.value)}.{attr}({', '.join(arguments)})"

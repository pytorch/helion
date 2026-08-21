from __future__ import annotations

import ast
import dataclasses
import enum
import itertools
from typing import TYPE_CHECKING

from .. import exc
from .ast_read_writes import ReadWrites

if TYPE_CHECKING:
    from collections.abc import Callable

    from .device_ir import GraphInfo


# fx node meta key marking a load that must be preceded by ``tl.debug_barrier()``.
INTRA_LOOP_RAW_BARRIER_META = "_needs_debug_barrier_before"


class TileDependencyKind(enum.Enum):
    READ_AFTER_WRITE = "read_after_write"
    WRITE_AFTER_READ = "write_after_read"
    WRITE_AFTER_WRITE = "write_after_write"


class TileDependencySynchronization(enum.Enum):
    UNSYNCHRONIZED = "unsynchronized"
    SOURCE_BARRIER = "source_barrier"


class TileAccessKind(enum.Enum):
    READ = "read"
    WRITE = "write"


@dataclasses.dataclass(frozen=True)
class TileAccess:
    """A loop-local tensor access normalized to its host base storage."""

    root: int
    storage: str
    kind: TileAccessKind
    # A stable AST representation is retained until the affine normalizer is
    # introduced. ``None`` means the access footprint could not be isolated.
    index: str | None


@dataclasses.dataclass(frozen=True)
class TileDependency:
    producer_root: int
    consumer_root: int
    name: str
    kinds: frozenset[TileDependencyKind]
    synchronization: TileDependencySynchronization


@dataclasses.dataclass(frozen=True)
class TileDependencyAnalysis:
    tile_dependencies: tuple[TileDependency, ...]
    accesses_by_root: tuple[tuple[TileAccess, ...], ...]
    source_phase_starts: frozenset[int]
    root_count: int

    @property
    def unsynchronized_tile_dependencies(self) -> tuple[TileDependency, ...]:
        return tuple(
            tile_dependency
            for tile_dependency in self.tile_dependencies
            if tile_dependency.synchronization
            is TileDependencySynchronization.UNSYNCHRONIZED
        )


def analyze_top_level_tile_dependencies(
    body: list[ast.stmt],
) -> TileDependencyAnalysis:
    """Analyze dependencies between top-level device loops without lowering them."""
    from .ast_extension import ExtendedAST
    from .ast_extension import LoopType
    from .type_info import BarrierResultType

    aliases: dict[str, str] = {}
    checker = LoopDependencyChecker(raise_on_dependency=False, aliases=aliases)
    root_index = 0

    for stmt in body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt, ExtendedAST):
            value = stmt.value
            if isinstance(value, ExtendedAST) and isinstance(
                value._type_info, BarrierResultType
            ):
                checker.insert_barrier_after_root(root_index - 1)
                continue

        if not (
            isinstance(stmt, ast.For)
            and isinstance(stmt, ExtendedAST)
            and stmt._loop_type == LoopType.GRID
        ):
            _update_host_aliases(stmt, aliases)
            continue

        checker.register_loop(stmt, root_index)
        root_index += 1

    return checker.analysis


def collect_host_tensor_aliases(body: list[ast.stmt]) -> dict[str, str]:
    """Collect conservative base-storage aliases from host-wrapper statements."""
    aliases: dict[str, str] = {}
    for stmt in body:
        if not isinstance(stmt, ast.For):
            _update_host_aliases(stmt, aliases)
    return aliases


def _access_base_name(expr: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(expr, ast.Name):
        return _canonical_alias(expr.id, aliases)
    if isinstance(expr, ast.Subscript):
        return _access_base_name(expr.value, aliases)
    if isinstance(expr, ast.Attribute) and expr.attr in {"T", "mT", "data"}:
        return _access_base_name(expr.value, aliases)
    return None


def _access_index(expr: ast.expr | None) -> str | None:
    return None if expr is None else ast.dump(expr, include_attributes=False)


class _TileAccessVisitor(ast.NodeVisitor):
    """Collect explicit tensor accesses without treating scalar names as storage."""

    def __init__(
        self,
        root: int,
        aliases: dict[str, str],
        device_local_names: frozenset[str],
    ) -> None:
        super().__init__()
        self.root = root
        self.aliases = aliases
        self.device_local_names = device_local_names
        self.accesses: list[TileAccess] = []

    def _record(
        self,
        base_expr: ast.expr,
        kind: TileAccessKind,
        index_expr: ast.expr | None,
    ) -> None:
        raw_storage = _access_base_name(base_expr, {})
        # Device SSA tensors such as an index vector may themselves be
        # subscripted while building an address. They are assigned inside this
        # root and cannot represent storage shared with another root.
        if raw_storage in self.device_local_names:
            return
        storage = _access_base_name(base_expr, self.aliases)
        if storage is None:
            return
        self.accesses.append(
            TileAccess(
                root=self.root,
                storage=storage,
                kind=kind,
                index=_access_index(index_expr),
            )
        )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, ast.Load):
            self._record(node.value, TileAccessKind.READ, node.slice)
        elif isinstance(node.ctx, ast.Store):
            self._record(node.value, TileAccessKind.WRITE, node.slice)
        self.visit(node.slice)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Subscript):
            self._record(node.target.value, TileAccessKind.READ, node.target.slice)
            self._record(node.target.value, TileAccessKind.WRITE, node.target.slice)
            self.visit(node.target.slice)
            self.visit(node.value)
            return
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "hl"
            and node.args
            and (
                func.attr == "load"
                or func.attr == "store"
                or func.attr.startswith("atomic_")
            )
        ):
            kind = TileAccessKind.READ if func.attr == "load" else TileAccessKind.WRITE
            index_expr = node.args[1] if len(node.args) > 1 else None
            self._record(node.args[0], kind, index_expr)
            if func.attr.startswith("atomic_"):
                self._record(node.args[0], TileAccessKind.READ, index_expr)
            for argument in node.args[1:]:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        self.generic_visit(node)


def _collect_tile_accesses(
    loop_node: ast.For,
    root: int,
    aliases: dict[str, str],
) -> tuple[TileAccess, ...]:
    device_local_names = frozenset(
        node.id
        for statement in loop_node.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    )
    visitor = _TileAccessVisitor(root, aliases, device_local_names)
    for statement in loop_node.body:
        visitor.visit(statement)
    return tuple(visitor.accesses)


def canonical_host_tensor_name(name: str, aliases: dict[str, str]) -> str:
    """Resolve ``name`` through aliases collected by collect_host_tensor_aliases."""
    return _canonical_alias(name, aliases)


_ALIAS_PRESERVING_METHODS = frozenset(
    {
        "as_strided",
        "detach",
        "expand",
        "flatten",
        "movedim",
        "narrow",
        "permute",
        "reshape",
        "select",
        "squeeze",
        "swapaxes",
        "swapdims",
        "transpose",
        "unbind",
        "unflatten",
        "unsqueeze",
        "view",
    }
)


def _canonical_alias(name: str, aliases: dict[str, str]) -> str:
    """Resolve a host name to the base storage name tracked by the analysis."""
    path: list[str] = []
    while (base := aliases.get(name)) is not None and base != name:
        path.append(name)
        name = base
    for alias in path:
        aliases[alias] = name
    return name


def _alias_base_name(expr: ast.expr, aliases: dict[str, str]) -> str | None:
    """Return the conservative base name for a host-side tensor alias expression.

    Basic slicing and the standard view-like tensor methods preserve storage.
    Treating an advanced-indexing subscript as an alias is conservative: it can
    add a dependency but cannot remove one.
    """
    if isinstance(expr, ast.Name):
        return _canonical_alias(expr.id, aliases)
    if isinstance(expr, ast.Subscript):
        return _alias_base_name(expr.value, aliases)
    if isinstance(expr, ast.Attribute) and expr.attr in {"T", "mT", "data"}:
        return _alias_base_name(expr.value, aliases)
    if isinstance(expr, ast.Call):
        if (
            isinstance(expr.func, ast.Attribute)
            and isinstance(expr.func.value, ast.Name)
            and expr.func.value.id == "torch"
            and expr.func.attr in _ALIAS_PRESERVING_METHODS
            and expr.args
        ):
            return _alias_base_name(expr.args[0], aliases)
        if (
            isinstance(expr.func, ast.Attribute)
            and expr.func.attr in _ALIAS_PRESERVING_METHODS
        ):
            return _alias_base_name(expr.func.value, aliases)
    return None


def _target_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.List, ast.Tuple)):
        return tuple(name for element in target.elts for name in _target_names(element))
    return ()


def _definitely_assigned_names(statement: ast.stmt) -> set[str]:
    """Names assigned whenever normal execution continues past ``statement``."""
    if isinstance(statement, ast.Assign):
        return {name for target in statement.targets for name in _target_names(target)}
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        return set(_target_names(statement.target))
    if isinstance(statement, ast.AugAssign):
        return set(_target_names(statement.target))
    if isinstance(statement, ast.If) and statement.orelse:
        return _definitely_assigned_in_block(
            statement.body
        ) & _definitely_assigned_in_block(statement.orelse)
    return set()


def _definitely_assigned_in_block(body: list[ast.stmt]) -> set[str]:
    result: set[str] = set()
    for statement in body:
        result.update(_definitely_assigned_names(statement))
    return result


def _live_in_read_names(body: list[ast.stmt]) -> set[str]:
    """Return names read before a definite root-local assignment.

    Branch-local definitions count only when every branch assigns the name.
    Loops remain conservative because they may execute zero times.
    """
    defined: set[str] = set()
    live_in: set[str] = set()
    for statement in body:
        rw = ReadWrites.from_ast(statement)
        loop_targets = {
            name
            for node in ast.walk(statement)
            if isinstance(node, (ast.For, ast.comprehension))
            for name in _target_names(node.target)
        }
        reads = {
            name
            for name, count in rw.reads.items()
            if count > rw.inplace_writes.get(name, 0)
        } - loop_targets
        live_in.update(reads - defined)
        defined.update(_definitely_assigned_names(statement))
    return live_in


def _update_alias_target(
    target: ast.expr,
    value: ast.expr,
    aliases: dict[str, str],
) -> None:
    """Apply one host assignment to the conservative storage-alias map."""
    if isinstance(target, ast.Name):
        base = _alias_base_name(value, aliases)
        if base is None:
            aliases.pop(target.id, None)
        else:
            aliases[target.id] = base
        return

    if not isinstance(target, (ast.List, ast.Tuple)):
        return

    # Pairwise tuple/list assignment preserves the more precise base for each
    # element.  Calls such as ``q, k, v = qkv.unbind(0)`` return multiple views
    # of one storage, so every unpacked name conservatively aliases the call's
    # receiver even though the RHS is not an AST tuple.
    if isinstance(value, (ast.List, ast.Tuple)) and len(target.elts) == len(value.elts):
        for target_element, value_element in zip(target.elts, value.elts, strict=True):
            _update_alias_target(target_element, value_element, aliases)
        return

    base = _alias_base_name(value, aliases)
    for name in _target_names(target):
        if base is None:
            aliases.pop(name, None)
        else:
            aliases[name] = base


def _update_host_aliases(stmt: ast.stmt, aliases: dict[str, str]) -> None:
    """Update base-storage aliases from a host-wrapper assignment."""
    value: ast.expr | None = None
    targets: tuple[ast.expr, ...] = ()
    if isinstance(stmt, ast.Assign):
        targets = tuple(stmt.targets)
        value = stmt.value
    elif isinstance(stmt, ast.AnnAssign):
        targets = (stmt.target,)
        value = stmt.value
    if value is None:
        return
    for target in targets:
        _update_alias_target(target, value, aliases)


def mark_intra_loop_raw_barriers(
    graphs: list[GraphInfo], root_graph_ids: list[int]
) -> None:
    """Mark loads that read storage written earlier in a device-loop body.

    Within a single device-loop body, ``qkv[a] = v`` followed by ``qkv[b]`` is a
    read-after-write on the same storage. When the store and the load use
    different-shaped index tensors their Triton thread->element layouts differ, so
    an element written by one thread is read back by another with no
    synchronization in between -- a data race (observed corrupting ~0.8% of
    outputs on B200). Helion already inserts ``tl.debug_barrier()`` for the
    analogous hazard *between* sequential top-level loops
    (``needs_inter_loop_debug_barrier_for_global_raw``); this extends the same
    guarantee to a store->load *within* one loop body.

    We mark the load's FX node; the Triton ``load`` codegen emits a
    ``tl.debug_barrier()`` before it. The barrier flushes every prior write in the
    block, so once emitted the pending-write set is cleared and later loads need a
    new store to re-arm. Storage identity comes from the fake tensor's underlying
    storage, so distinct FX nodes for aliases and views compare equal. The walk
    follows Helion control-flow subgraphs and merges pending writes at joins.
    """
    marker = _IntraLoopRawBarrierMarker(graphs)
    for graph_id in root_graph_ids:
        marker.run_graph(graph_id, set())


class _IntraLoopRawBarrierMarker:
    def __init__(self, graphs: list[GraphInfo]) -> None:
        self.graphs = graphs

    def _storage_ids(self, graph_id: int, obj: object) -> set[int]:
        import torch

        from .device_ir import NodeArgsGraphInfo

        if isinstance(obj, (list, tuple)):
            result: set[int] = set()
            for item in obj:
                result.update(self._storage_ids(graph_id, item))
            return result
        if not isinstance(obj, torch.fx.Node):
            return set()
        graph_info = self.graphs[graph_id]
        value = obj.meta.get("val")
        result = (
            {id(value.untyped_storage())} if isinstance(value, torch.Tensor) else set()
        )
        if (
            obj.op == "placeholder"
            and obj.graph is graph_info.graph
            and isinstance(graph_info, NodeArgsGraphInfo)
        ):
            result.update(
                self._storage_ids(graph_id, graph_info.placeholder_to_outer_arg(obj))
            )
        if result:
            return result
        for arg in obj.args:
            result.update(self._storage_ids(graph_id, arg))
        return result

    def run_graph(self, graph_id: int, written: set[int]) -> set[int]:
        from ..language import memory_ops
        from ..language._tracing_ops import _for_loop
        from ..language._tracing_ops import _for_loop_step
        from ..language._tracing_ops import _if
        from ..language._tracing_ops import _while_loop
        from .device_ir import ForLoopGraphInfo
        from .device_ir import IfGraphInfo
        from .device_ir import WhileLoopGraphInfo

        pending = set(written)
        for node in self.graphs[graph_id].graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is memory_ops.store:
                pending.update(self._storage_ids(graph_id, node.args[0]))
                continue
            if node.target is memory_ops.load:
                if node.meta.get(INTRA_LOOP_RAW_BARRIER_META):
                    pending.clear()
                elif pending & self._storage_ids(graph_id, node.args[0]):
                    node.meta[INTRA_LOOP_RAW_BARRIER_META] = True
                    pending.clear()
                continue
            if node.target is _if:
                if_graph_id = node.args[1]
                assert isinstance(if_graph_id, int)
                if_info = self.graphs[if_graph_id]
                assert isinstance(if_info, IfGraphInfo)
                if_pending = self.run_graph(if_graph_id, pending)
                if if_info.else_branch is None:
                    pending |= if_pending
                else:
                    else_graph_id = (
                        if_info.else_branch
                        if isinstance(if_info.else_branch, int)
                        else if_info.else_branch.graph_id
                    )
                    else_pending = self.run_graph(else_graph_id, pending)
                    pending = if_pending | else_pending
                continue
            if node.target in (_for_loop, _for_loop_step):
                loop_graph_id = node.args[0]
                assert isinstance(loop_graph_id, int)
                loop_info = self.graphs[loop_graph_id]
                assert isinstance(loop_info, ForLoopGraphInfo)
                loop_input = set() if loop_info.needs_barrier_before else pending
                loop_pending = self.run_graph(loop_graph_id, loop_input)
                # Without a pre-loop barrier, zero iterations preserve the input state.
                pending = (
                    loop_pending
                    if loop_info.needs_barrier_before
                    else pending | loop_pending
                )
                continue
            if node.target is _while_loop:
                body_graph_id = node.args[1]
                assert isinstance(body_graph_id, int)
                body_info = self.graphs[body_graph_id]
                assert isinstance(body_info, WhileLoopGraphInfo)
                condition_pending = self.run_graph(body_info.cond_graph_id, pending)
                body_pending = self.run_graph(body_graph_id, condition_pending)
                # The condition executes at least once; the body may not execute.
                pending = condition_pending | body_pending
        return pending


def needs_inter_loop_debug_barrier_for_global_raw(
    prev_global_writes: set[str],
    host_loop_reads: frozenset[str],
    *,
    global_barrier_tensor_names: Callable[[frozenset[str]], set[str]],
) -> bool:
    """Whether to emit ``tl.debug_barrier()`` before the next sequential device loop.

    Returns True when the union of host-named global writes accumulated from
    all prior siblings (since the last emitted barrier) intersects the current
    loop's host-named read set.
    """
    cur_global_reads = global_barrier_tensor_names(host_loop_reads)
    return bool(prev_global_writes & cur_global_reads)


class LoopDependencyChecker:
    """
    A class to check dependencies between top-level for loops in a Helion kernel.

    This class tracks memory accesses (reads and writes) for each top-level for
    loop. The legacy mode raises on an unsynchronized dependency; analysis mode
    returns TileDependency records for later scheduling passes.
    """

    def __init__(
        self,
        *,
        raise_on_dependency: bool = True,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.tile_dependencies: list[TileDependency] = []
        self.accesses_by_root: list[tuple[TileAccess, ...]] = []
        self._barrier_after_root: set[int] = set()
        self._writer_roots: dict[str, tuple[int, int]] = {}
        self._reader_roots: dict[str, dict[int, int]] = {}
        self._source_phase_starts: set[int] = set()
        self._generation: int = 0
        self._root_counter: int = 0
        self.raise_on_dependency = raise_on_dependency
        self.aliases = aliases if aliases is not None else {}
        self.disabled: bool = False

    def _canonical_name(self, name: str) -> str:
        return _canonical_alias(name, self.aliases)

    @property
    def analysis(self) -> TileDependencyAnalysis:
        return TileDependencyAnalysis(
            tile_dependencies=tuple(self.tile_dependencies),
            accesses_by_root=tuple(self.accesses_by_root),
            source_phase_starts=frozenset(self._source_phase_starts),
            root_count=self._root_counter,
        )

    def insert_barrier_after_root(self, root_id: int) -> None:
        """Record that a barrier separates root_id and root_id+1."""
        self._barrier_after_root.add(root_id)

    def register_loop(
        self, loop_node: ast.For, root_id: int | None = None
    ) -> tuple[TileDependency, ...]:
        if self.disabled:
            return ()
        current_root = root_id if root_id is not None else self._root_counter
        if (current_root - 1) in self._barrier_after_root:
            self.reads.clear()
            self.writes.clear()
            self._generation += 1
            self._source_phase_starts.add(current_root)
            self._barrier_after_root.discard(current_root - 1)
        rw = ReadWrites.from_list(loop_node.body)
        accesses = _collect_tile_accesses(loop_node, current_root, self.aliases)
        self.accesses_by_root.append(accesses)

        read_names = {
            self._canonical_name(name) for name in _live_in_read_names(loop_node.body)
        }
        read_names.update(
            access.storage for access in accesses if access.kind is TileAccessKind.READ
        )
        write_names = {self._canonical_name(name) for name in rw.writes}
        storage_write_names = {
            access.storage for access in accesses if access.kind is TileAccessKind.WRITE
        }
        dependencies = self._tile_dependencies(
            read_names, storage_write_names, current_root
        )
        self.tile_dependencies.extend(dependencies)
        unsynchronized = tuple(
            dependency
            for dependency in dependencies
            if dependency.synchronization
            is TileDependencySynchronization.UNSYNCHRONIZED
        )
        if self.raise_on_dependency and unsynchronized:
            raise exc.LoopDependencyError(unsynchronized[0].name)

        self.reads |= read_names
        self.writes |= write_names
        for name in write_names:
            self._writer_roots[name] = (current_root, self._generation)
            self._reader_roots.pop(name, None)
        for name in read_names - write_names:
            self._reader_roots.setdefault(name, {})[current_root] = self._generation
        self._root_counter = current_root + 1
        return dependencies

    def _tile_dependencies(
        self,
        read_names: set[str],
        write_names: set[str],
        current_root: int,
    ) -> tuple[TileDependency, ...]:
        """Return structured dependencies against prior top-level loops."""
        dependencies: dict[
            tuple[int, str, TileDependencySynchronization], set[TileDependencyKind]
        ] = {}

        def record(
            producer_root: int,
            producer_generation: int,
            name: str,
            kind: TileDependencyKind,
        ) -> None:
            synchronization = (
                TileDependencySynchronization.SOURCE_BARRIER
                if producer_generation < self._generation
                else TileDependencySynchronization.UNSYNCHRONIZED
            )
            dependencies.setdefault((producer_root, name, synchronization), set()).add(
                kind
            )

        for name in sorted(set(itertools.chain(read_names, write_names))):
            writer = self._writer_roots.get(name)
            if writer is not None:
                producer_root, producer_generation = writer
                if name in read_names:
                    record(
                        producer_root,
                        producer_generation,
                        name,
                        TileDependencyKind.READ_AFTER_WRITE,
                    )
                if name in write_names:
                    record(
                        producer_root,
                        producer_generation,
                        name,
                        TileDependencyKind.WRITE_AFTER_WRITE,
                    )
            if name in write_names:
                for producer_root, producer_generation in self._reader_roots.get(
                    name, {}
                ).items():
                    record(
                        producer_root,
                        producer_generation,
                        name,
                        TileDependencyKind.WRITE_AFTER_READ,
                    )

        return tuple(
            TileDependency(
                producer_root=producer_root,
                consumer_root=current_root,
                name=name,
                kinds=frozenset(kinds),
                synchronization=synchronization,
            )
            for (producer_root, name, synchronization), kinds in dependencies.items()
        )

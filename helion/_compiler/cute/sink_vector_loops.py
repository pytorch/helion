"""Sink a grid constexpr vector loop into the serial loops it encloses.

A grid tile axis with ``cute_vector_widths[block] = V > 1`` is partitioned
into an outer lane loop and a constexpr V-loop that wraps the WHOLE grid body
(``DeviceGridState.wrap_body``).  When that body reduces another axis with a
serial device loop and a per-thread lane loop (a column sum: ``for tile_m:
acc += sum(x[tile_m, tile_n], dim=0)``), the emitted nest is::

    for vec_lane in cutlass.range_constexpr(V):  # grid axis element
        acc = 0
        for tile_offset in range(0, m, BLOCK):  # serial device loop
            lane_acc = 0
            for lane in range(ROWS):  # per-thread rows
                lane_acc += x[row(lane), base + vec_lane]  # 2-byte load
            acc += combine_across_threads(lane_acc)
        store(acc)

so every thread walks its row strip V times with scalar loads and one load
in flight, and pays V cross-thread combines per row tile.  This pass
distributes the V-loop over the body and interchanges it with the loops::

    acc = make_rmem_tensor(V)
    for vec_lane in range_constexpr(V): acc[vec_lane] = 0
    for tile_offset in range(0, m, BLOCK):
        lane_acc = make_rmem_tensor(V)
        for vec_lane in range_constexpr(V): lane_acc[vec_lane] = 0
        ptr = &x[row(0), base]                       # strength-reduced address
        for lane in range(ROWS):                     # optionally unrolled
            vec = cute.arch.load(ptr + lane * row_stride, V)   # ONE vector load
            for vec_lane in range_constexpr(V):
                lane_acc[vec_lane] += vec[vec_lane]
        reduce_fragment(lane_acc -> reduced)         # one grouped combine
        for vec_lane in range_constexpr(V): acc[vec_lane] += reduced[vec_lane]
    for vec_lane in range_constexpr(V): store(acc[vec_lane])

Per-V-lane scalars that live across a loop boundary are expanded into
V-element register fragments (or recomputed where they are cheap functions
of the lane), V-lane-invariant pure statements run once outside the sunk
V-loops, and everything else keeps its original per-lane order inside a
constexpr V-loop, so the observable effects (stores, barriers) run in the
same order as before within each segment.

The pass is driven by facts the strategy and the load lowering record
(``CuteVloopWrapperFact`` / ``CuteVloopLoadFact``); it fires only for the
constexpr loop of a recorded wrapper, only when at least one scalar load
nested in an inner loop becomes a vector load, and it leaves the body
untouched whenever any statement or value is outside the proven forms (a
load under a per-lane condition, a hoist whose operands only the V-loop
defines, ...).  When the knob shaped the thread layout but nothing was sunk,
``generate_ast`` regenerates the knob-off code.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import Callable
from typing import cast

from ...autotuner.config_spec import _CUTE_LANE_UNROLL_CHOICES
from ...language.memory_ops import _CUTE_VECTOR_UNROLL_CARRIER
from ...language.memory_ops import _CUTE_VECTOR_UNROLL_DTYPES
from ...language.memory_ops import _cute_unroll_vec_load_expr
from ..ast_extension import create
from ..ast_extension import expr_from_string
from ..ast_extension import statement_from_string
from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ..ast_read_writes import ReadWrites
from ..ast_read_writes import ast_rename
from ..compile_environment import CompileEnvironment
from .vector_reduction_packets import _clone

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

    import torch

    from .device_state import CuteVloopLoadFact
    from .device_state import CuteVloopWrapperFact

# Marks vector-load statements this pass emitted (an attribute on the node).
_VSINK_VECTOR_LOAD_ATTR = "_helion_vsink_vector_load"
_TWO_STAGE_REDUCE = "_cute_grouped_reduce_shared_two_stage"
_TWO_STAGE_REDUCE_FRAGMENT = "_cute_grouped_reduce_shared_two_stage_fragment"
# Calls whose result carries the dtype of their first argument.
_DTYPE_PRESERVING_CALLS = (
    "cute.arch.warp_reduction",
    "_cute_grouped_reduce_",
    "cute.arch.fmax",
    "cute.arch.fmin",
    "cute.math.",
    "math.",
)
_UNKNOWN = "__helion_unknown__"
_CONFLICT = "__conflict__"


def _names_read(node: ast.AST) -> set[str]:
    return set(ReadWrites.from_ast(node).reads)


def _names_written(node: ast.AST) -> set[str]:
    return set(ReadWrites.from_ast(node).writes)


def _names_written_list(stmts: Sequence[ast.AST]) -> set[str]:
    return set().union(*(_names_written(s) for s in stmts)) if stmts else set()


def _plain_target(stmt: ast.AST) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    return None


def _is_constexpr_loop(stmt: ast.AST) -> tuple[str, int] | None:
    """``for X in cutlass.range_constexpr(V)`` -> ``(X, V)``."""
    if not isinstance(stmt, ast.For) or not isinstance(stmt.target, ast.Name):
        return None
    call = stmt.iter
    if (
        isinstance(call, ast.Call)
        and ast.unparse(call.func) == "cutlass.range_constexpr"
        and len(call.args) == 1
        and not call.keywords
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, int)
        and not stmt.orelse
    ):
        return stmt.target.id, call.args[0].value
    return None


def _lane_loop_extent(loop: ast.For) -> int | None:
    """Extent of an ascending ``range(N)`` lane loop, else None."""
    call = loop.iter
    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "range"
        and len(call.args) == 1
        and not call.keywords
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, int)
    ):
        return call.args[0].value
    return None


def _is_dtype_ctor(name: str) -> bool:
    member = name.rsplit(".", 1)[-1]
    return name.startswith("cutlass.") and bool(member) and member[0].isupper()


def _is_relocatable_statement(
    stmt: ast.AST, *, allow_load: bool, tensor_names: Collection[str]
) -> bool:
    """A plain assignment whose calls are all proven effect-free and that
    reads memory only if ``allow_load``.

    A load is a ``.load()`` / ``cute.arch.load`` call or a bare subscript of
    a kernel tensor argument (``tensor_names``): the gather lowering emits
    ``t[i] if mask else 0`` with no load call, and moving that read past a
    store or atomic to ``t`` would change the value it sees.
    """
    from ..tile_strategy import _is_proven_relocatable_call

    if not isinstance(stmt, ast.Assign) or _plain_target(stmt) is None:
        return False

    def relocatable(node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            return _is_proven_relocatable_call(node, allow_load=allow_load)
        if isinstance(node, ast.Subscript):
            return allow_load or not (
                isinstance(node.value, ast.Name) and node.value.id in tensor_names
            )
        return True

    return all(relocatable(node) for node in ast.walk(stmt))


def _constexpr_loop(var: str, width: int, body: list[ast.stmt]) -> ast.For:
    return create(
        ast.For,
        target=create(ast.Name, id=var, ctx=ast.Store()),
        iter=expr_from_string(f"cutlass.range_constexpr({width})"),
        body=body,
        orelse=[],
        type_comment=None,
    )


class _Unsupported(Exception):
    """The body is outside the proven forms; leave it untouched."""


@dataclasses.dataclass
class _Analysis:
    varying: set[str]
    expanded: dict[str, str]  # name -> fragment name
    fragment_dtypes: dict[str, str]  # fragment name -> dtype constructor
    # Cheap per-lane definitions recomputed where a segment needs them.
    rematerialized: dict[str, ast.Assign]
    definition_counts: dict[str, int]


class _Sinker:
    def __init__(
        self,
        vloop: ast.For,
        fact: CuteVloopWrapperFact,
        loads: Mapping[str, CuteVloopLoadFact],
        rename_groups: dict[str, str],
        new_var: Callable[..., str],
        lane_unroll: int,
        tensor_dtypes: Mapping[str, torch.dtype],
    ) -> None:
        self.vec_var = fact.vec_lane_var
        self.vec_width = fact.vec_width
        self.fact = fact
        self.loads = loads
        # Kernel tensor arguments by name: a subscript of one is a load.
        self.tensor_dtypes = tensor_dtypes
        self.new_var = new_var
        self.lane_unroll = lane_unroll
        self.fragment_allocs: list[ast.stmt] = []
        self.fragment_names: set[str] = set()
        self.vector_loads_emitted = 0
        # Canonicalize loop-carried aliases (``v_1`` -> ``col_acc``) on a
        # private copy so the dataflow sees one name per carried value.
        body: list[ast.stmt] = []
        for stmt in vloop.body:
            if isinstance(stmt, ast.Pass):
                continue
            cloned = _clone(cast("ast.stmt", stmt))
            ast_rename(cloned, rename_groups)
            body.append(cloned)
        self.body = body

    # ------------------------------------------------------------------ analysis

    def run(self) -> list[ast.stmt] | None:
        body = self.body
        self._check_statement_forms(body)
        self._uniformize_vector_mask(body)
        if not self._has_sinkable_nested_load(body):
            return None
        varying = self._varying_names(body)
        exposed = self._exposed_reads(body, set())
        if exposed & _names_written_list(body):
            # A scalar carried across the V lanes (read before it is written
            # in the body) has no per-lane meaning after distribution.
            return None
        # Once the V-loop is distributed, a name a segment reads before
        # writing it is carried across that segment's own V-loop.  Per-lane
        # values get fragments below; a V-invariant carried scalar (a
        # row-only accumulator ``wsum += w[row]``) would instead repeat its
        # update V times, so it becomes per-lane as well.
        carried = self._segment_carried_names(body) - varying
        if carried:
            varying = self._varying_names(body, carried)
        definitions = self._definitions(body)
        definition_counts = self._definition_counts(body)
        rematerialized = self._rematerializable(body, varying, definition_counts)
        expanded: dict[str, str] = {}
        fragment_dtypes: dict[str, str] = {}
        for name in sorted(self._segment_exposed_names(body, varying)):
            if name in rematerialized:
                continue
            dtype = self._infer_dtype(name, definitions, {}, set())
            if dtype is None:
                return None
            fragment = self.new_var(f"{name}_frag", dce=False)
            expanded[name] = fragment
            fragment_dtypes[fragment] = dtype
            self.fragment_names.add(fragment)
            self.fragment_allocs.append(
                statement_from_string(
                    f"{fragment} = cute.make_rmem_tensor({self.vec_width}, {dtype})"
                )
            )
        self.analysis = _Analysis(
            varying, expanded, fragment_dtypes, rematerialized, definition_counts
        )
        emitted = self._emit(body)
        if self.vector_loads_emitted == 0:
            return None
        return [*self.fragment_allocs, *emitted]

    def _walk_statements(self, stmts: Sequence[ast.AST]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for stmt in stmts:
            result.append(cast("ast.stmt", stmt))
            for field in ("body", "orelse"):
                inner = getattr(stmt, field, None)
                if isinstance(inner, list):
                    result.extend(self._walk_statements(inner))
        return result

    def _definition_counts(self, stmts: Sequence[ast.AST]) -> dict[str, int]:
        """How many simple statements of ``stmts`` (nested ``if`` bodies and
        loop bodies included) write each name."""
        counts: dict[str, int] = {}
        for stmt in self._walk_statements(stmts):
            if isinstance(stmt, (ast.For, ast.If)):
                continue
            for name in _names_written(stmt):
                counts[name] = counts.get(name, 0) + 1
        return counts

    def _check_statement_forms(self, stmts: Sequence[ast.AST]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                if stmt.orelse or not isinstance(stmt.target, ast.Name):
                    raise _Unsupported
                self._check_statement_forms(stmt.body)
            elif isinstance(stmt, ast.If):
                if any(
                    isinstance(node, (ast.For, ast.While))
                    for node in ast.walk(stmt)
                    if node is not stmt
                ):
                    raise _Unsupported
            elif isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(
                    stmt.targets[0], (ast.Name, ast.Subscript)
                ):
                    raise _Unsupported
            elif isinstance(stmt, ast.AugAssign):
                if not isinstance(stmt.target, (ast.Name, ast.Subscript)):
                    raise _Unsupported
            elif not isinstance(stmt, (ast.Expr, ast.Pass)):
                raise _Unsupported
            if any(
                isinstance(node, (ast.NamedExpr, ast.Lambda, ast.ListComp, ast.Yield))
                for node in ast.walk(stmt)
            ):
                raise _Unsupported

    def _uniformize_vector_mask(self, body: list[ast.stmt]) -> None:
        """Evaluate the vectorized axis' bounds mask at the chunk base.

        With the tile extent a multiple of V and the chunk base V-aligned,
        ``base + v < extent`` holds for every ``v`` in ``[0, V)`` exactly when
        ``base < extent`` does, so the per-element mask is uniform across the
        V lanes.  Rewriting its definition makes it lane-invariant: it is then
        computed once and can guard whole vector loads.
        """
        fact = self.fact
        if fact.mask_var is None or not fact.uniform_vector_mask:
            return
        definitions = [
            stmt
            for stmt in self._walk_statements(body)
            if fact.mask_var in _names_written(stmt)
        ]
        if len(definitions) != 1 or definitions[0] not in body:
            return
        definition = definitions[0]
        if not isinstance(definition, ast.Assign) or _plain_target(definition) is None:
            return
        _RenameLoads({fact.index_var: fact.base_index_var}).visit(definition)

    def _index_definition_ok(self, body: list[ast.stmt]) -> bool:
        """The per-element index is ``base + cutlass.Int32(vec_lane)``."""
        fact = self.fact
        definitions = [
            stmt
            for stmt in self._walk_statements(body)
            if fact.index_var in _names_written(stmt)
        ]
        if len(definitions) != 1 or definitions[0] not in body:
            return False
        definition = definitions[0]
        if not isinstance(definition, ast.Assign):
            return False
        expected = f"{fact.base_index_var} + cutlass.Int32({fact.vec_lane_var})"
        return ast.unparse(definition.value) == expected

    def _sinkable_load(self, node: ast.AST) -> CuteVloopLoadFact | None:
        """The load fact when ``node`` is a scalar load this pass may widen."""
        if not isinstance(node, ast.Call):
            return None
        pointer = _scalar_load_pointer(node)
        if pointer is None:
            return None
        fact = self.loads.get(ast.unparse(pointer))
        if fact is None or fact.vec_lane_var != self.vec_var:
            return None
        if self.fact.index_var not in _names_read(pointer):
            return None
        return fact

    def _has_sinkable_nested_load(self, body: list[ast.stmt]) -> bool:
        if not self._index_definition_ok(body):
            return False
        return any(
            isinstance(stmt, ast.For)
            and any(self._sinkable_load(node) is not None for node in ast.walk(stmt))
            for stmt in body
        )

    def _definitions(self, body: list[ast.stmt]) -> dict[str, list[ast.expr]]:
        definitions: dict[str, list[ast.expr]] = {}
        for stmt in self._walk_statements(body):
            if isinstance(stmt, (ast.For, ast.If)):
                # Compound statements only aggregate their nested writes.
                continue
            if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                definitions.setdefault(stmt.targets[0].id, []).append(stmt.value)
            elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
                definitions.setdefault(stmt.target.id, []).append(
                    ast.BinOp(
                        left=ast.Name(id=stmt.target.id, ctx=ast.Load()),
                        op=stmt.op,
                        right=stmt.value,
                    )
                )
            else:
                for name in _names_written(stmt):
                    # A structured write (tensor element, tuple) has no
                    # scalar definition; record it as unknown.
                    definitions.setdefault(name, []).append(
                        ast.Name(id=_UNKNOWN, ctx=ast.Load())
                    )
        return definitions

    def _varying_names(
        self, body: list[ast.stmt], seeds: set[str] | None = None
    ) -> set[str]:
        """Names whose value may differ between V lanes: the fixpoint from
        the V-lane variable and ``seeds``."""
        units: list[tuple[set[str], set[str]]] = []

        def collect(stmts: Sequence[ast.AST], control: set[str]) -> None:
            for stmt in stmts:
                if isinstance(stmt, ast.For):
                    units.append((_names_read(stmt.iter) | control, {"__loop_iter__"}))
                    collect(stmt.body, control)
                elif isinstance(stmt, ast.If):
                    test_reads = _names_read(stmt.test) | control
                    collect(stmt.body, test_reads)
                    collect(stmt.orelse, test_reads)
                else:
                    units.append((_names_read(stmt) | control, _names_written(stmt)))

        collect(body, set())
        varying = {self.vec_var, *(seeds or ())}
        changed = True
        while changed:
            changed = False
            for reads, writes in units:
                if reads & varying:
                    if "__loop_iter__" in writes:
                        # A loop bound that depends on the V lane cannot be
                        # interchanged with the V-loop.
                        raise _Unsupported
                    new = writes - varying
                    if new:
                        varying |= new
                        changed = True
        varying.discard(self.vec_var)
        return varying

    def _exposed_reads(self, stmts: Sequence[ast.AST], defined: set[str]) -> set[str]:
        """Names read in ``stmts`` before ``stmts`` (or ``defined``) write them."""
        exposed: set[str] = set()
        defined = set(defined)
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                exposed |= _names_read(stmt.iter) - defined
                target = cast("ast.Name", stmt.target).id
                exposed |= self._exposed_reads(stmt.body, defined | {target}) - defined
                defined |= _names_written(stmt)
            elif isinstance(stmt, ast.If):
                exposed |= _names_read(stmt.test) - defined
                exposed |= self._exposed_reads(stmt.body, defined) - defined
                exposed |= self._exposed_reads(stmt.orelse, defined) - defined
                if stmt.orelse:
                    defined |= _names_written_list(stmt.body) & _names_written_list(
                        stmt.orelse
                    )
            else:
                exposed |= _names_read(stmt) - defined
                defined |= _names_written(stmt)
        return exposed

    def _segments(self, stmts: Sequence[ast.AST]) -> list[list[ast.stmt] | ast.For]:
        groups: list[list[ast.stmt] | ast.For] = []
        current: list[ast.stmt] = []
        for stmt in stmts:
            if isinstance(stmt, ast.For):
                if current:
                    groups.append(current)
                    current = []
                groups.append(stmt)
            elif not isinstance(stmt, ast.Pass):
                current.append(cast("ast.stmt", stmt))
        if current:
            groups.append(current)
        return groups

    def _segment_exposed_names(
        self, body: list[ast.stmt], varying: set[str]
    ) -> set[str]:
        """V-lane-varying names that flow between segments.

        Each segment becomes its own constexpr V-loop, so a varying name that
        a segment reads before writing it (loop-carried, or produced by an
        earlier segment) must be recomputed or live in a per-lane fragment.
        """
        exposed: set[str] = set()

        def visit(stmts: Sequence[ast.AST], loop_targets: set[str]) -> None:
            for group in self._segments(stmts):
                if isinstance(group, ast.For):
                    target = cast("ast.Name", group.target).id
                    visit(group.body, loop_targets | {target})
                else:
                    exposed.update(self._exposed_reads(group, loop_targets))

        visit(body, set())
        return exposed & varying

    def _segment_carried_names(self, body: list[ast.stmt]) -> set[str]:
        """Names some segment both reads before writing and writes."""
        carried: set[str] = set()

        def visit(stmts: Sequence[ast.AST], loop_targets: set[str]) -> None:
            for group in self._segments(stmts):
                if isinstance(group, ast.For):
                    target = cast("ast.Name", group.target).id
                    visit(group.body, loop_targets | {target})
                else:
                    carried.update(
                        self._exposed_reads(group, loop_targets)
                        & _names_written_list(group)
                    )

        visit(body, set())
        return carried

    def _rematerializable(
        self, body: list[ast.stmt], varying: set[str], counts: dict[str, int]
    ) -> dict[str, ast.Assign]:
        """Varying names with one pure definition over lane-external inputs.

        Such a definition (the per-element index ``base + vec_lane``, a mask
        derived from it) can be recomputed at the start of any segment that
        reads it instead of occupying a register fragment.
        """
        written = _names_written_list(body)
        candidates: dict[str, ast.Assign] = {}
        for stmt in body:
            name = _plain_target(stmt)
            if (
                name is not None
                and isinstance(stmt, ast.Assign)
                and name in varying
                and counts.get(name) == 1
                and _is_relocatable_statement(
                    stmt, allow_load=False, tensor_names=self.tensor_dtypes
                )
            ):
                candidates[name] = stmt
        result: dict[str, ast.Assign] = {}
        changed = True
        while changed:
            changed = False
            for name, stmt in candidates.items():
                if name in result:
                    continue
                reads = _names_read(stmt) - {self.vec_var}
                if all(read not in written or read in result for read in reads):
                    result[name] = stmt
                    changed = True
        return result

    def _infer_dtype(
        self,
        name: str,
        definitions: dict[str, list[ast.expr]],
        cache: dict[str, str | None],
        visiting: set[str],
    ) -> str | None:
        if name in cache:
            return cache[name]
        if name in visiting or name not in definitions:
            return None
        visiting.add(name)
        result: str | None = None
        for value in definitions[name]:
            dtype = self._expr_dtype(value, definitions, cache, visiting)
            if dtype == _CONFLICT or (
                dtype is not None and result is not None and dtype != result
            ):
                result = _CONFLICT
                break
            if dtype is not None:
                result = dtype
        visiting.discard(name)
        if result == _CONFLICT:
            result = None
        cache[name] = result
        return result

    def _expr_dtype(
        self,
        node: ast.expr,
        definitions: dict[str, list[ast.expr]],
        cache: dict[str, str | None],
        visiting: set[str],
    ) -> str | None:
        def combine(*dtypes: str | None) -> str | None:
            known = {dtype for dtype in dtypes if dtype is not None}
            if _CONFLICT in known or len(known) > 1:
                return _CONFLICT
            return next(iter(known), None)

        if isinstance(node, ast.Name):
            if node.id == _UNKNOWN:
                return _CONFLICT
            return self._infer_dtype(node.id, definitions, cache, visiting)
        if isinstance(node, ast.Call):
            func = ast.unparse(node.func)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "bitcast":
                return ast.unparse(node.args[0]) if node.args else _CONFLICT
            if _is_dtype_ctor(func):
                return func
            if func.startswith(_DTYPE_PRESERVING_CALLS) and node.args:
                return self._expr_dtype(node.args[0], definitions, cache, visiting)
            if func in ("max", "min"):
                return combine(
                    *(
                        self._expr_dtype(arg, definitions, cache, visiting)
                        for arg in node.args
                    )
                )
            return _CONFLICT
        if isinstance(node, ast.BinOp):
            return combine(
                self._expr_dtype(node.left, definitions, cache, visiting),
                self._expr_dtype(node.right, definitions, cache, visiting),
            )
        if isinstance(node, ast.IfExp):
            return combine(
                self._expr_dtype(node.body, definitions, cache, visiting),
                self._expr_dtype(node.orelse, definitions, cache, visiting),
            )
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return "cutlass.Boolean"
            return self._expr_dtype(node.operand, definitions, cache, visiting)
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            return "cutlass.Boolean"
        if isinstance(node, ast.Constant):
            return None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            # An element of a kernel tensor argument (``t[i]``) has the
            # tensor's element type.
            tensor_dtype = self.tensor_dtypes.get(node.value.id)
            if tensor_dtype is not None:
                return CompileEnvironment.current().backend.dtype_str(tensor_dtype)
        return _CONFLICT

    # ------------------------------------------------------------------ emission

    def _emit(self, stmts: Sequence[ast.AST]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for group in self._segments(stmts):
            if isinstance(group, ast.For):
                result.extend(self._emit_loop(group))
            else:
                result.extend(self._emit_segment(group))
        return result

    def _emit_loop(self, loop: ast.For) -> list[ast.stmt]:
        inner = self._emit(loop.body)
        lane_var = getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None)
        target = cast("ast.Name", loop.target).id
        pre: list[ast.stmt] = []
        vector_loads = [
            stmt for stmt in inner if getattr(stmt, _VSINK_VECTOR_LOAD_ATTR, False)
        ]
        if vector_loads:
            pre.extend(self._strength_reduce(inner, target, vector_loads))
        new_loop = create(
            ast.For,
            target=create(ast.Name, id=target, ctx=ast.Store()),
            iter=_clone(loop.iter),
            body=inner,
            orelse=[],
            type_comment=None,
        )
        if lane_var is not None:
            setattr(new_loop, HELION_LANE_LOOP_VAR_ATTR, lane_var)
            extent = _lane_loop_extent(loop)
            if vector_loads and extent is not None:
                new_loop = self._unroll_lane_loop(new_loop, extent)
        return [*pre, new_loop]

    def _emit_segment(self, segment: list[ast.stmt]) -> list[ast.stmt]:
        analysis = self.analysis
        vec_var = self.vec_var
        pre: list[ast.stmt] = []
        inside: list[ast.stmt] = []
        vector_cache: dict[str, str] = {}
        # Distribution turns the per-lane execution of a hoisted statement
        # into one execution before this segment's V-loop.  For that to be
        # exact, the statement must be the only definition of the names it
        # writes in the segment (nested ``if`` bodies included): a second
        # definition kept inside, before or after it, would reach the later
        # lanes.  It may only read names defined outside the segment or by
        # statements hoisted ahead of it, and it must not redefine a name a
        # statement kept inside has read.  The same holds for the operands of
        # a widened load.
        definition_counts = self._definition_counts(segment)
        segment_writes = set(definition_counts)
        # Names the hoisted statements have defined so far.
        pre_writes: set[str] = set()
        # Names the statements kept inside so far have read.
        inside_reads: set[str] = set()
        # In the original order every effect of the segment in lane ``v``
        # precedes the loads of lanes ``v + 1, ...``, so a load moved before
        # the V-loop crosses the segment's effects in both directions: loads
        # only move out of a segment without any memory effect.
        has_effects = any(self._has_memory_effect(stmt) for stmt in segment)
        # Recompute cheap per-lane values this segment reads before defining.
        needed = self._exposed_reads(segment, set()) & set(analysis.rematerialized)
        for name in self._rematerialization_order(needed):
            recomputed = _clone(analysis.rematerialized[name])
            inside.append(recomputed)
            inside_reads |= _names_read(recomputed)
            segment_writes |= _names_written(recomputed)
        for stmt in segment:
            reads = _names_read(stmt)
            writes = _names_written(stmt)
            rewritten = _clone(stmt)
            if (
                vec_var not in reads
                and not reads & analysis.varying
                and not writes & analysis.varying
            ):
                # Lane-invariant: run once before the V-loop when its calls
                # are relocatable and nothing kept inside is ordered against
                # it; otherwise keep its per-lane execution and order.
                if (
                    _is_relocatable_statement(
                        rewritten,
                        allow_load=not has_effects,
                        tensor_names=self.tensor_dtypes,
                    )
                    and all(definition_counts.get(name) == 1 for name in writes)
                    and not reads & (segment_writes - pre_writes)
                    and not writes & inside_reads
                ):
                    pre.append(rewritten)
                    pre_writes |= writes
                    continue
            else:
                self._vectorize_loads(
                    rewritten,
                    vector_cache,
                    pre,
                    segment_writes - pre_writes,
                    has_effects,
                )
                _ExpandNames(analysis.expanded, vec_var).visit(rewritten)
            inside.append(rewritten)
            inside_reads |= reads
        inside = self._hoist_fragment_reductions(inside, pre)
        result = list(pre)
        if inside:
            result.append(_constexpr_loop(vec_var, self.vec_width, inside))
        return result

    def _has_memory_effect(self, stmt: ast.stmt) -> bool:
        """Whether ``stmt`` stores, synchronizes or otherwise acts on memory;
        writes into this pass's register fragments do not count.

        An assignment counts when it writes a tensor in place or calls
        anything outside the proven pure / load set: an atomic whose result
        is bound to a name (``old = cute.arch.atomic_add(...)``) or a
        shared-memory reduce is such a call.  Every other statement form
        (``if`` blocks, bare expressions) counts.
        """
        from ..tile_strategy import _has_side_effect
        from ..tile_strategy import _is_proven_relocatable_call

        if isinstance(stmt, ast.Assign):
            inplace = set(ReadWrites.from_ast(stmt).inplace_writes)
            if inplace - self.fragment_names:
                return True
            return not all(
                _is_proven_relocatable_call(node, allow_load=True)
                for node in ast.walk(stmt)
                if isinstance(node, ast.Call)
            )
        return _has_side_effect(stmt)

    def _rematerialization_order(self, names: set[str]) -> list[str]:
        """``names`` plus their rematerialized inputs, producers first."""
        table = self.analysis.rematerialized
        ordered: list[str] = []
        seen: set[str] = set()

        def visit(name: str) -> None:
            if name in seen or name not in table:
                return
            seen.add(name)
            for read in sorted(_names_read(table[name])):
                visit(read)
            ordered.append(name)

        for name in sorted(names):
            visit(name)
        return ordered

    def _vectorize_loads(
        self,
        stmt: ast.stmt,
        vector_cache: dict[str, str],
        pre: list[ast.stmt],
        unavailable: set[str],
        has_effects: bool,
    ) -> None:
        """Replace sinkable scalar loads in ``stmt`` by vector-lane extracts.

        The vector load itself is V-lane-invariant (it addresses the chunk
        base) and is appended to ``pre``.  A load is only widened when every
        V lane executes it: unconditionally, or under a V-invariant guard,
        which then selects a safe anchor pointer for the transaction while
        the per-lane select keeps its meaning.  A load under a per-lane
        condition (a per-element bounds mask) has no whole-chunk equivalent,
        so the body is left untouched.  ``unavailable`` names the segment's
        definitions that are not hoisted ahead of ``stmt``; ``has_effects``
        says whether the segment has any memory effect.
        """
        sinker = self
        varying = self.analysis.varying

        def contains_load(node: ast.AST) -> bool:
            return any(
                sinker._sinkable_load(child) is not None for child in ast.walk(node)
            )

        def invariant(node: ast.expr) -> bool:
            reads = _names_read(node)
            return sinker.vec_var not in reads and not reads & varying

        class Rewrite(ast.NodeTransformer):
            def __init__(self) -> None:
                self.guard: ast.expr | None = None

            def visit_If(self, node: ast.If) -> ast.AST:
                if contains_load(node):
                    raise _Unsupported
                return node

            def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
                # Operands after the first are evaluated conditionally.
                if any(contains_load(value) for value in node.values[1:]):
                    raise _Unsupported
                node.values[0] = self.visit(node.values[0])
                return node

            def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
                if contains_load(node.test) or contains_load(node.orelse):
                    raise _Unsupported
                if not invariant(node.test):
                    if contains_load(node.body):
                        raise _Unsupported
                    node.test = self.visit(node.test)
                    return node
                outer = self.guard
                self.guard = (
                    node.test
                    if outer is None
                    else ast.BoolOp(op=ast.And(), values=[outer, node.test])
                )
                node.body = self.visit(node.body)
                self.guard = outer
                return node

            def visit_Call(self, node: ast.Call) -> ast.AST:
                load_fact = sinker._sinkable_load(node)
                if load_fact is None:
                    return self.generic_visit(node)
                return sinker._emit_vector_load(
                    node,
                    load_fact,
                    self.guard,
                    vector_cache,
                    pre,
                    unavailable,
                    has_effects,
                )

        Rewrite().visit(stmt)

    def _emit_vector_load(
        self,
        call: ast.Call,
        load_fact: CuteVloopLoadFact,
        mask: ast.expr | None,
        vector_cache: dict[str, str],
        pre: list[ast.stmt],
        unavailable: set[str],
        has_effects: bool,
    ) -> ast.expr:
        if has_effects:
            # The transaction would move across an effect of this segment.
            raise _Unsupported
        pointer = _scalar_load_pointer(call)
        assert pointer is not None
        vector_pointer = _clone(pointer)
        _RenameLoads({self.fact.index_var: self.fact.base_index_var}).visit(
            vector_pointer
        )
        needed = _names_read(vector_pointer)
        if mask is not None:
            needed = needed | _names_read(mask)
        if (
            self.vec_var in needed
            or needed & self.analysis.varying
            or needed & unavailable
        ):
            # The address or guard differs per lane, or needs a value this
            # segment defines inside its V-loop (before or after the load).
            raise _Unsupported
        pointer_text = ast.unparse(vector_pointer)
        guard = ast.unparse(mask) if mask is not None else ""
        key = f"{pointer_text}|{guard}"
        vector_var = vector_cache.get(key)
        if vector_var is None:
            if mask is not None:
                # A masked thread still issues the (unconditional) vector
                # transaction; point it at the tensor's first chunk, which
                # exists for any non-empty extent, and let the per-lane
                # select keep discarding the bytes.  The explicit zero offset
                # gives the anchor the same pointer type as the offset form.
                guarded = (
                    f"({pointer_text} if {guard} else "
                    f"{load_fact.tensor_name}.iterator + cutlass.Int32(0))"
                )
            else:
                guarded = pointer_text
            vector_var = self.new_var("_vsink_vec", dce=False)
            load_stmt = statement_from_string(
                f"{vector_var} = "
                + _cute_unroll_vec_load_expr(
                    guarded,
                    load_fact.dtype,
                    load_fact.vec_width,
                    load_fact.eviction_suffix,
                )
            )
            setattr(load_stmt, _VSINK_VECTOR_LOAD_ATTR, True)
            pre.append(load_stmt)
            vector_cache[key] = vector_var
            self.vector_loads_emitted += 1
        carrier = _CUTE_VECTOR_UNROLL_CARRIER[load_fact.dtype]
        element = _CUTE_VECTOR_UNROLL_DTYPES[load_fact.dtype]
        return cast(
            "ast.expr",
            expr_from_string(
                f"{carrier}({vector_var}[{self.vec_var}]).bitcast({element})"
            ),
        )

    def _hoist_fragment_reductions(
        self, inside: list[ast.stmt], pre: list[ast.stmt]
    ) -> list[ast.stmt]:
        """Combine all V accumulators of a segment with one grouped reduce.

        ``R = wrap(_cute_grouped_reduce_shared_two_stage(acc_frag[v], ...))``
        inside the V-loop pays the two-barrier two-stage reduce V times.
        When the input is an expanded fragment and no other statement of the
        segment has a memory effect, the fragment form reduces all V values
        with one pair of barriers before the V-loop and the loop reads
        ``R = wrap(R_frag[v])``.
        """
        analysis = self.analysis
        result: list[ast.stmt] = []
        # The hoisted reduce runs before the V-loop: it cannot read a name
        # any statement kept inside defines (before or after it: the later
        # lanes would have seen that definition), and it does not move
        # across a memory effect of the segment.
        inside_writes = _names_written_list(inside)
        if any(
            self._has_memory_effect(stmt)
            for stmt in inside
            if _fragment_reduce_match(stmt, self.vec_var, self.fragment_names) is None
        ):
            return inside
        for stmt in inside:
            match = _fragment_reduce_match(stmt, self.vec_var, self.fragment_names)
            if match is not None:
                call, source_fragment, pre_count, group_span, group_count = match
                if (
                    pre_count > 0
                    and not pre_count & (pre_count - 1)
                    and pre_count <= 32
                    and group_span % 32 == 0
                    and pre_count * self.vec_width <= group_span
                    and not any(
                        self.vec_var in _names_read(arg)
                        or _names_read(arg) & analysis.varying
                        for arg in call.args[1:]
                    )
                    and not (_names_read(call) - {self.vec_var}) & inside_writes
                ):
                    target = cast("str", _plain_target(stmt))
                    result_fragment = self.new_var(f"{target}_frag", dce=False)
                    dtype = analysis.fragment_dtypes[source_fragment]
                    self.fragment_allocs.append(
                        statement_from_string(
                            f"{result_fragment} = cute.make_rmem_tensor("
                            f"{self.vec_width}, {dtype})"
                        )
                    )
                    args = ", ".join(ast.unparse(arg) for arg in call.args[1:])
                    pre.append(
                        statement_from_string(
                            f"{_TWO_STAGE_REDUCE_FRAGMENT}({source_fragment}, "
                            f"{result_fragment}, {args}, count={self.vec_width}, "
                            f"pre={pre_count}, group_span={group_span}, "
                            f"group_count={group_count})"
                        )
                    )
                    replaced = _clone(stmt)
                    _ReplaceCall(call, f"{result_fragment}[{self.vec_var}]").visit(
                        replaced
                    )
                    result.append(replaced)
                    continue
            result.append(stmt)
        return result

    def _strength_reduce(
        self, body: list[ast.stmt], lane_var: str, vector_loads: list[ast.stmt]
    ) -> list[ast.stmt]:
        """Address the loop's vector loads from a hoisted base pointer.

        ``T.iterator + Int32(A + Int32(lane) * C) * Int32(s_r) + rest`` with
        ``A``, ``C`` and ``rest`` lane-invariant becomes ``base + Int32(lane) *
        step`` with ``base = T.iterator + Int32(A) * Int32(s_r) + rest`` and
        ``step = Int32(C) * Int32(s_r)`` computed once before the loop, which
        drops the per-row index arithmetic to one multiply-add.
        """
        pre: list[ast.stmt] = []
        body_writes = _names_written_list(body)
        definitions = {
            name: stmt.value
            for stmt in body
            if isinstance(stmt, ast.Assign)
            and (name := _plain_target(stmt)) is not None
        }
        for load_stmt in vector_loads:
            assert isinstance(load_stmt, ast.Assign)
            for pointer in _maximal_pointer_sums(load_stmt.value):
                reduced = _reduce_pointer(
                    pointer, lane_var, definitions, body_writes, self.new_var
                )
                if reduced is None:
                    continue
                base_stmt, step_stmt, replacement = reduced
                pre.extend((base_stmt, step_stmt))
                _ReplaceNode(pointer, replacement).visit(load_stmt)
        return pre

    def _unroll_lane_loop(self, loop: ast.For, extent: int) -> ast.For:
        """Unroll by the ``cute_lane_unroll`` factor, issuing every copy's
        vector loads before any copy's compute."""
        unroll = 1
        for choice in _CUTE_LANE_UNROLL_CHOICES:
            if choice <= self.lane_unroll and extent % choice == 0:
                unroll = choice
        if unroll <= 1:
            return loop
        lane_var = cast("ast.Name", loop.target).id
        body = cast("list[ast.stmt]", loop.body)
        writes = _names_written_list(body)
        carried = self._exposed_reads(body, {lane_var}) & writes
        if carried - self.fragment_names:
            # A scalar carried across lane iterations cannot be renamed per
            # copy; keep the rolled loop.
            return loop
        renamable = sorted(writes - self.fragment_names)
        heads: list[list[ast.stmt]] = []
        tails: list[list[ast.stmt]] = []
        for copy_index in range(unroll):
            renames = {
                name: self.new_var(f"{name}_u{copy_index}", dce=True)
                for name in renamable
            }
            copy: list[ast.stmt] = []
            for stmt in body:
                cloned = _clone(stmt)
                ast_rename(cloned, renames)
                _SubstituteLane(lane_var, unroll, copy_index).visit(cloned)
                copy.append(cloned)
            head_indices = {
                index
                for index, stmt in enumerate(copy)
                if getattr(stmt, _VSINK_VECTOR_LOAD_ATTR, False)
            }
            needed: set[str] = set()
            for index in head_indices:
                needed |= _names_read(copy[index])
            for index in range(len(copy) - 1, -1, -1):
                if index in head_indices:
                    continue
                stmt_writes = _names_written(copy[index])
                if stmt_writes & needed and not stmt_writes & self.fragment_names:
                    head_indices.add(index)
                    needed |= _names_read(copy[index])
            heads.append([copy[i] for i in range(len(copy)) if i in head_indices])
            tails.append([copy[i] for i in range(len(copy)) if i not in head_indices])
        new_body: list[ast.stmt] = []
        for head in heads:
            new_body.extend(head)
        for tail in tails:
            new_body.extend(tail)
        new_loop = create(
            ast.For,
            target=create(ast.Name, id=lane_var, ctx=ast.Store()),
            iter=expr_from_string(f"range({extent // unroll})"),
            body=new_body,
            orelse=[],
            type_comment=None,
        )
        setattr(
            new_loop,
            HELION_LANE_LOOP_VAR_ATTR,
            getattr(loop, HELION_LANE_LOOP_VAR_ATTR),
        )
        return new_loop


def _scalar_load_pointer(call: ast.Call) -> ast.expr | None:
    """Pointer expression of ``(ptr).load()`` / ``cute.arch.load(ptr, dtype)``."""
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "load"
        and not call.args
        and not call.keywords
    ):
        return call.func.value
    if (
        ast.unparse(call.func) == "cute.arch.load"
        and len(call.args) == 2
        and isinstance(call.args[1], (ast.Name, ast.Attribute))
    ):
        return call.args[0]
    return None


def _pointer_root(node: ast.expr) -> str | None:
    """Tensor name of a ``T.iterator + ...`` pointer sum."""
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        node = node.left
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    ):
        return node.value.id
    return None


def _maximal_pointer_sums(node: ast.AST) -> list[ast.BinOp]:
    """Outermost ``T.iterator + ...`` sums in ``node``."""
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and _pointer_root(node) is not None
    ):
        return [node]
    result: list[ast.BinOp] = []
    for child in ast.iter_child_nodes(node):
        result.extend(_maximal_pointer_sums(child))
    return result


def _additive_terms(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_additive_terms(node.left), *_additive_terms(node.right)]
    return [node]


def _split_affine(
    node: ast.expr, lane_var: str
) -> tuple[list[ast.expr], ast.expr | None] | None:
    """Split ``A + Int32(lane) * C`` (any term order) into ``([A terms], C)``.

    ``C`` is None for a bare ``cutlass.Int32(lane)`` term.  Returns None
    when the lane variable appears other than in exactly one such term.
    """
    invariant: list[ast.expr] = []
    scale: ast.expr | None = None
    found = False
    lane_call = f"cutlass.Int32({lane_var})"
    for term in _additive_terms(node):
        if lane_var not in _names_read(term):
            invariant.append(term)
            continue
        if found:
            return None
        found = True
        if ast.unparse(term) == lane_call:
            scale = None
        elif (
            isinstance(term, ast.BinOp)
            and isinstance(term.op, ast.Mult)
            and ast.unparse(term.left) == lane_call
            and lane_var not in _names_read(term.right)
        ):
            scale = term.right
        elif (
            isinstance(term, ast.BinOp)
            and isinstance(term.op, ast.Mult)
            and ast.unparse(term.right) == lane_call
            and lane_var not in _names_read(term.left)
        ):
            scale = term.left
        else:
            return None
    if not found:
        return None
    return invariant, scale


def _reduce_pointer(
    pointer: ast.BinOp,
    lane_var: str,
    definitions: dict[str, ast.expr],
    body_writes: set[str],
    new_var: Callable[..., str],
) -> tuple[ast.stmt, ast.stmt, ast.expr] | None:
    terms = _additive_terms(pointer)
    root = terms[0]
    lane_term: tuple[int, list[ast.expr], ast.expr | None, ast.expr] | None = None
    for index, term in enumerate(terms[1:], start=1):
        # ``cutlass.Int32(idx) * cutlass.Int32(T.layout.stride[d])``
        if not (
            isinstance(term, ast.BinOp)
            and isinstance(term.op, ast.Mult)
            and isinstance(term.left, ast.Call)
            and ast.unparse(term.left.func) == "cutlass.Int32"
            and len(term.left.args) == 1
        ):
            return None
        reads = _names_read(term)
        if lane_var in reads:
            return None
        if not reads & body_writes:
            continue
        index_node = term.left.args[0]
        if not isinstance(index_node, ast.Name) or index_node.id not in definitions:
            return None
        definition = definitions[index_node.id]
        split = _split_affine(definition, lane_var)
        if (
            split is None
            or (_names_read(definition) - {lane_var}) & body_writes
            or _names_read(term.right) & body_writes
            or lane_term is not None
        ):
            return None
        lane_term = (index, split[0], split[1], term.right)
    if lane_term is None:
        return None
    index, invariant_terms, scale, stride = lane_term
    base_index = " + ".join(ast.unparse(t) for t in invariant_terms) or "0"
    stride_text = ast.unparse(stride)
    step_text = (
        f"cutlass.Int32({ast.unparse(scale)}) * {stride_text}"
        if scale is not None
        else stride_text
    )
    other_terms = [
        ast.unparse(term) for i, term in enumerate(terms) if i not in (0, index)
    ]
    base_var = new_var("_vsink_ptr", dce=False)
    step_var = new_var("_vsink_step", dce=False)
    base_text = " + ".join(
        [
            ast.unparse(root),
            f"cutlass.Int32({base_index}) * {stride_text}",
            *other_terms,
        ]
    )
    base_stmt = statement_from_string(f"{base_var} = {base_text}")
    step_stmt = statement_from_string(f"{step_var} = {step_text}")
    replacement = expr_from_string(
        f"{base_var} + cutlass.Int32({lane_var}) * {step_var}"
    )
    return base_stmt, step_stmt, cast("ast.expr", replacement)


def _fragment_reduce_match(
    stmt: ast.stmt, vec_var: str, fragments: set[str]
) -> tuple[ast.Call, str, int, int, int] | None:
    """``R = wrap(_cute_grouped_reduce_shared_two_stage(frag[vec], ...))``."""
    if _plain_target(stmt) is None or not isinstance(stmt, ast.Assign):
        return None
    calls = [
        node
        for node in ast.walk(stmt.value)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _TWO_STAGE_REDUCE
    ]
    if len(calls) != 1:
        return None
    call = calls[0]
    if len(call.args) != 6 or len(call.keywords) != 3:
        return None
    source = call.args[0]
    if not (
        isinstance(source, ast.Subscript)
        and isinstance(source.value, ast.Name)
        and source.value.id in fragments
        and isinstance(source.slice, ast.Name)
        and source.slice.id == vec_var
    ):
        return None
    keywords = {kw.arg: kw.value for kw in call.keywords}
    values: dict[str, int] = {}
    for key in ("pre", "group_span", "group_count"):
        node = keywords.get(key)
        if not (isinstance(node, ast.Constant) and isinstance(node.value, int)):
            return None
        values[key] = node.value
    return (
        call,
        source.value.id,
        values["pre"],
        values["group_span"],
        values["group_count"],
    )


class _RenameLoads(ast.NodeTransformer):
    """Rename ``ast.Name`` loads (not stores) by a mapping."""

    def __init__(self, renames: dict[str, str]) -> None:
        self.renames = renames

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.renames:
            node.id = self.renames[node.id]
        return node


class _ExpandNames(ast.NodeTransformer):
    """Replace expanded scalars by their fragment element ``frag[vec]``."""

    def __init__(self, expanded: dict[str, str], vec_var: str) -> None:
        self.expanded = expanded
        self.vec_var = vec_var

    def visit_Name(self, node: ast.Name) -> ast.AST:
        fragment = self.expanded.get(node.id)
        if fragment is None:
            return node
        return ast.copy_location(
            ast.Subscript(
                value=ast.Name(id=fragment, ctx=ast.Load()),
                slice=ast.Name(id=self.vec_var, ctx=ast.Load()),
                ctx=node.ctx,
            ),
            node,
        )


class _SubstituteLane(ast.NodeTransformer):
    """``lane`` -> ``lane * unroll + copy`` in load contexts."""

    def __init__(self, lane_var: str, unroll: int, copy_index: int) -> None:
        self.lane_var = lane_var
        self.unroll = unroll
        self.copy_index = copy_index

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id == self.lane_var:
            return ast.copy_location(
                cast(
                    "ast.expr",
                    expr_from_string(
                        f"{self.lane_var} * {self.unroll} + {self.copy_index}"
                    ),
                ),
                node,
            )
        return node


class _ReplaceCall(ast.NodeTransformer):
    """Replace one call node (matched by source text) with an expression."""

    def __init__(self, call: ast.Call, replacement: str) -> None:
        self.text = ast.unparse(call)
        self.replacement = replacement

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if ast.unparse(node) == self.text:
            return cast("ast.AST", expr_from_string(self.replacement))
        return self.generic_visit(node)


class _ReplaceNode(ast.NodeTransformer):
    def __init__(self, target: ast.AST, replacement: ast.expr) -> None:
        self.target = target
        self.replacement = replacement

    def visit(self, node: ast.AST) -> ast.AST:
        if node is self.target:
            return self.replacement
        return super().visit(node)


def sink_grid_vector_loops(
    body: list[ast.AST],
    *,
    wrappers: Mapping[str, CuteVloopWrapperFact],
    loads: Mapping[str, CuteVloopLoadFact],
    rename_groups: dict[str, str],
    lane_unroll: int,
    new_var: Callable[..., str],
    tensor_dtypes: Mapping[str, torch.dtype],
) -> tuple[list[ast.AST], bool]:
    """Sink every recorded grid V-loop that encloses a vectorizable serial
    reduction nest; bodies outside the proven forms are left untouched.

    ``tensor_dtypes`` maps every kernel tensor argument to its dtype: a
    subscript of one of these names is a memory read.

    Returns the body and whether any V-loop was sunk."""
    if not wrappers or not loads:
        return body, False
    fired = False

    def visit(stmts: list[ast.AST]) -> list[ast.AST]:
        nonlocal fired
        result: list[ast.AST] = []
        for stmt in stmts:
            constexpr = _is_constexpr_loop(stmt)
            fact = wrappers.get(constexpr[0]) if constexpr is not None else None
            if (
                fact is not None
                and constexpr is not None
                and constexpr[1] == fact.vec_width
                and isinstance(stmt, ast.For)
            ):
                try:
                    replacement = _Sinker(
                        stmt,
                        fact,
                        loads,
                        rename_groups,
                        new_var,
                        lane_unroll,
                        tensor_dtypes,
                    ).run()
                except _Unsupported:
                    replacement = None
                if replacement is not None:
                    result.extend(replacement)
                    fired = True
                    continue
            for field in ("body", "orelse"):
                inner = getattr(stmt, field, None)
                if isinstance(inner, list) and inner:
                    setattr(stmt, field, visit(inner))
            result.append(stmt)
        return result

    return visit(body), fired

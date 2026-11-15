"""Remove the first-pass stores and atomics duplicated by CuTe reduction
interchange, and decline the interchanged nests whose passes would reorder
their memory accesses."""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING
from typing import cast

from ... import exc
from ..ast_extension import clone_ast
from ..ast_read_writes import ReadWrites
from ._ast_pass_utils import _dependency_names
from ._ast_pass_utils import _is_proven_relocatable_assignment
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import exposed_reads
from ._ast_pass_utils import tensor_roots_may_alias
from .fuse_two_pass_loads import _is_store_call
from .fuse_two_pass_loads import _store_tensor_roots
from .fuse_two_pass_loads import _tensor_arg_roots
from .lane_loops.markers import _HELION_LANE_REDUCE_MARKER
from .lane_loops.markers import _is_lane_reduce_marker_assign
from .lane_loops.markers import _static_lane_loop_extent
from .lane_loops.owner_guards import _is_atomic_call
from .lane_loops.owner_guards import _is_isolated_store_statement
from .lane_loops.owner_guards import _memory_write_calls
from .lane_loops.owner_guards import _store_enclosing_predicates
from .persistent_branch_vec import _binding_write_roots
from .persistent_branch_vec import _definition_snapshots
from .persistent_branch_vec import _freeze_definition
from .persistent_branch_vec import _memory_address_definition_snapshots
from .persistent_branch_vec import _memory_load_calls
from .persistent_branch_vec import _plain_scalar_load_pointer
from .persistent_branch_vec import _plain_scalar_store_pointer
from .persistent_branch_vec import _pointer_integer_affine_form
from .persistent_branch_vec import _single_tensor_iterator_root
from .pure_lane_packets import is_pure_inline_asm_call

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Mapping


def _freeze_coverage(
    expressions: list[ast.expr], definitions: dict[str, ast.expr]
) -> list[ast.expr] | None:
    result = []
    for expression in expressions:
        frozen = _freeze_definition(expression, definitions)
        if frozen is None:
            return None
        result.append(frozen)
    return result


def _reads(statement: ast.AST) -> set[str]:
    result = set(ReadWrites.from_ast(statement).reads)
    marker = _is_lane_reduce_marker_assign(statement)
    if marker is not None and marker.group_lane_expr:
        result.update(
            ReadWrites.from_ast(ast.parse(marker.group_lane_expr, mode="eval")).reads
        )
    return result


def _plain_assignment(statement: ast.AST) -> tuple[str, ...]:
    name = assigned_name(statement)
    return () if name is None else (name,)


def _live_in(statements: list[ast.stmt]) -> set[str]:
    """Names read before a plain assignment binds them; compound statements
    (loops, branches) count as single statements that define nothing."""
    return set(
        exposed_reads(
            statements,
            reads=_reads,
            defines=_plain_assignment,
            loops="opaque",
            branches="opaque",
        )
    )


def _single_store(statement: ast.AST) -> ast.Call | None:
    stores = [
        node
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and _is_store_call(node)
    ]
    if len(stores) != 1:
        return None
    store = stores[0]
    if _plain_scalar_store_pointer(store) is None or not _is_isolated_store_statement(
        statement, store
    ):
        return None
    if any(
        isinstance(node, ast.Call)
        and node is not store
        and not _is_proven_relocatable_call(node, allow_load=False)
        for node in ast.walk(statement)
    ):
        return None
    return store


def _known_statements(statements: list[ast.stmt]) -> bool:
    """Reject control transfers, opaque calls, atomics, compound writes, and
    calls the passes cannot resolve (``_unclassified_accesses``: a load
    helper the relocation proof accepts but no load list holds)."""
    for statement in statements:
        if _unclassified_accesses(statement, writes=False):
            return False
        # Only metadata/coordinate subscripts are known scalar reads. Other
        # subscripts may hide an aliasing access without an explicit load call.
        if any(
            isinstance(node, (ast.NamedExpr, ast.Lambda))
            or isinstance(node, ast.Subscript)
            and not _known_subscript(node)
            for node in ast.walk(statement)
        ):
            return False
        if isinstance(statement, ast.Pass) or _single_store(statement) is not None:
            continue
        if _is_lane_reduce_marker_assign(statement) is not None:
            continue
        if not _is_proven_relocatable_assignment(
            statement, allow_load=True, allow_reduction=True
        ):
            return False
    return True


def _known_subscript(node: ast.Subscript) -> bool:
    is_stride = (
        isinstance(node.value, ast.Attribute)
        and node.value.attr == "stride"
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == "layout"
        and isinstance(node.value.value.value, ast.Name)
    )
    is_coordinate = isinstance(node.value, ast.Call) and call_path_str(
        node.value.func
    ) in {"cute.arch.thread_idx", "cute.arch.block_idx"}
    return is_stride or is_coordinate


def _coordinate_expression(
    expression: ast.expr, mutable_names: set[str], iteration_names: set[str]
) -> bool:
    """An address/bound may use immutable inputs and the two iteration IDs."""
    if set(ReadWrites.from_ast(expression).reads) & (mutable_names - iteration_names):
        return False
    for node in ast.walk(expression):
        if isinstance(node, ast.Call):
            if call_path_str(node.func) in {
                "range",
                "cutlass.range_constexpr",
            }:
                continue
            if not _is_proven_relocatable_call(node, allow_load=False):
                return False
        elif isinstance(node, ast.Subscript):
            if not _known_subscript(node):
                return False
    return True


def _iterator_tensor_names(nodes: Iterable[ast.AST]) -> set[str]:
    """The kernel tensor arguments ``nodes`` address as ``NAME.iterator``."""
    return {
        node.value.id
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    }


def _coverage_expressions(statement: ast.stmt, store: ast.Call) -> list[ast.expr]:
    pointer = _plain_scalar_store_pointer(store)
    assert pointer is not None
    return [
        pointer,
        *cast(
            "list[ast.expr]",
            _store_enclosing_predicates(statement, store),
        ),
    ]


def _matching_final_nest(
    lane_loop: ast.For, serial_loop: ast.For, final_nest: list[ast.AST]
) -> bool:
    """Whether ``final_nest`` ends in a loop with ``serial_loop``'s target
    and iterator whose body ends in a loop with ``lane_loop``'s, neither with
    an ``else``: the final pass the interchange builds from the two loops,
    in the shape ``_final_flat`` reads."""
    if not final_nest or not isinstance(final_nest[-1], ast.For):
        return False
    final_serial = final_nest[-1]
    if (
        final_serial.orelse
        or not final_serial.body
        or not isinstance(final_serial.body[-1], ast.For)
        or ast.unparse(final_serial.target) != ast.unparse(serial_loop.target)
        or ast.unparse(final_serial.iter) != ast.unparse(serial_loop.iter)
    ):
        return False
    final_lane = final_serial.body[-1]
    return not (
        final_lane.orelse
        or ast.unparse(final_lane.target) != ast.unparse(lane_loop.target)
        or ast.unparse(final_lane.iter) != ast.unparse(lane_loop.iter)
    )


def _final_flat(final_nest: list[ast.AST]) -> list[ast.stmt]:
    """The final pass as straight-line code: its prefix, its serial loop's
    head and its inner lane loop's body."""
    final_serial = final_nest[-1]
    assert isinstance(final_serial, ast.For) and final_serial.body
    final_lane = final_serial.body[-1]
    assert isinstance(final_lane, ast.For)
    return [
        *cast("list[ast.stmt]", final_nest[:-1]),
        *final_serial.body[:-1],
        *final_lane.body,
    ]


def _overwritten_stores(
    lane_loop: ast.For,
    serial_loop: ast.For,
    final_nest: list[ast.AST],
    indices: Iterable[int],
    protected_names: set[str],
    *,
    known_effects: bool = True,
) -> dict[int, ast.Call]:
    """The stores of the statements of ``serial_loop``'s body at ``indices``
    that the final pass stores again over the same elements.

    Restrict the proof to scalar stores in a rectangular pair of loops with
    the same bounds and without control transfers, and with ``known_effects``
    to statements whose effects are known (``_known_statements``, which
    removing a store and its producers needs; the coverage alone does not).
    Match the cloned final statement, expand assignment-time address/mask
    definitions, and require immutable, identical coverage: built from inputs
    no binding mutates and the two iteration variables only, so it is
    independent of the reduction and of anything either pass computes.
    Loop-carried coverage expressions and names with pending compiler aliases
    leave a store unproven.
    """
    if lane_loop.orelse or serial_loop.orelse:
        return {}
    if not isinstance(lane_loop.target, ast.Name) or not isinstance(
        serial_loop.target, ast.Name
    ):
        return {}
    if _static_lane_loop_extent(lane_loop) is None:
        return {}
    if (
        not isinstance(serial_loop.iter, ast.Call)
        or call_path_str(serial_loop.iter.func)
        not in {"range", "cutlass.range_constexpr"}
        or not 1 <= len(serial_loop.iter.args) <= 3
        or serial_loop.iter.keywords
    ):
        return {}
    if not _matching_final_nest(lane_loop, serial_loop, final_nest):
        return {}
    final_serial = cast("ast.For", final_nest[-1])

    serial_index = lane_loop.body.index(serial_loop)
    prefix = lane_loop.body[:serial_index]
    suffix = lane_loop.body[serial_index + 1 :]
    original_flat = [*prefix, *serial_loop.body]
    final_prefix = cast("list[ast.stmt]", final_nest[:-1])
    final_flat = _final_flat(final_nest)
    statements = [*original_flat, *suffix, *final_flat]
    if known_effects and not _known_statements(statements):
        return {}
    if any(
        isinstance(
            node, (ast.Break, ast.Continue, ast.Return, ast.Raise, ast.Try, ast.With)
        )
        for statement in statements
        for node in ast.walk(statement)
    ):
        return {}
    mutable_names = _binding_write_roots(lane_loop)
    iteration_names = {lane_loop.target.id, serial_loop.target.id}
    body_writes = set(ReadWrites.from_list(serial_loop.body).writes)
    carried_names = body_writes & _live_in(serial_loop.body)

    # Freeze prefixes at the loop entry. A suffix/backedge mutation of a free
    # bound is caught by mutable_names; lane-dependent bounds are rejected too.
    before_snapshots = _definition_snapshots([*prefix, ast.Pass()])
    after_snapshots = _definition_snapshots([*final_prefix, ast.Pass()])
    if before_snapshots is None or after_snapshots is None:
        return {}
    before_bounds = before_snapshots[-1]
    after_bounds = after_snapshots[-1]
    before_bound = _freeze_definition(serial_loop.iter, before_bounds)
    after_bound = _freeze_definition(final_serial.iter, after_bounds)
    if before_bound is None or after_bound is None:
        return {}
    bounds = [before_bound, after_bound]
    if any(not _coordinate_expression(bound, mutable_names, set()) for bound in bounds):
        return {}
    if ast.unparse(bounds[0]) != ast.unparse(bounds[1]):
        return {}
    if (
        _dependency_names(
            cast("list[ast.AST]", original_flat), _reads(serial_loop.iter)
        )
        & protected_names
    ):
        return {}

    original_definitions = _definition_snapshots(original_flat)
    final_definitions = _definition_snapshots(final_flat)
    if original_definitions is None or final_definitions is None:
        return {}
    proven: dict[int, ast.Call] = {}
    for index in indices:
        statement = serial_loop.body[index]
        store = _single_store(statement)
        if store is None:
            continue
        matches = [
            (position, other, final_store)
            for position, other in enumerate(final_flat)
            if ast.unparse(other) == ast.unparse(statement)
            and (final_store := _single_store(other)) is not None
        ]
        if len(matches) != 1:
            continue
        final_index, final_statement, final_store = matches[0]
        original_coverage = _coverage_expressions(statement, store)
        final_coverage = _coverage_expressions(final_statement, final_store)
        dependency_names = _dependency_names(
            cast("list[ast.AST]", original_flat),
            {name for expression in original_coverage for name in _reads(expression)},
        )
        if dependency_names & (carried_names | protected_names):
            continue
        before = _freeze_coverage(
            original_coverage, original_definitions[len(prefix) + index]
        )
        after = _freeze_coverage(final_coverage, final_definitions[final_index])
        if before is None or after is None:
            continue
        if not all(
            _coordinate_expression(expression, mutable_names, iteration_names)
            for expression in [*before, *after]
        ):
            continue
        if [ast.unparse(expression) for expression in before] != [
            ast.unparse(expression) for expression in after
        ]:
            continue
        proven[index] = store
    return proven


def eliminate_interchanged_stores(
    lane_loop: ast.For,
    serial_loop: ast.For,
    final_nest: list[ast.AST],
    candidate_indices: set[int],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    protected_names: set[str],
    *,
    read_back: bool = False,
) -> int:
    """Prune first-pass stores that an interchanged final pass overwrites.

    A store goes when the final pass stores again over the same elements
    (``_overwritten_stores``) and no load or other store may touch its
    destination, using cache-safe compiler facts. Unknown memory operations
    leave both passes unchanged. Producers are removed only when the final
    pass redefines their scalar bindings before use and no remaining
    first-pass operation reads them, and a serial loop left empty goes too
    (``_drop_emptied_serial_loop``). Names with pending compiler aliases
    remain protected until final renaming. ``read_back`` drops the load
    condition for stores a load reads back (``drop_read_back_partial_stores``,
    which checks the first pass's remaining loads itself).
    """
    if not candidate_indices:
        return 0
    proven = _overwritten_stores(
        lane_loop, serial_loop, final_nest, candidate_indices, protected_names
    )
    if not proven:
        return 0
    final_flat = _final_flat(final_nest)
    writes = [
        node
        for node in ast.walk(lane_loop)
        if isinstance(node, ast.Call) and _is_store_call(node)
    ]
    loads = [
        *_memory_load_calls(lane_loop),
        *(load for statement in final_flat for load in _memory_load_calls(statement)),
    ]
    tensor_names = _iterator_tensor_names([lane_loop])
    # Alias facts describe kernel argument bindings. A local reassignment or a
    # pending scalar rename cannot inherit the original argument's storage proof.
    if tensor_names & (_binding_write_roots(lane_loop) | protected_names):
        return 0
    removed: set[int] = set()
    for index, store in proven.items():
        roots = _store_tensor_roots(store, tensor_names)
        if roots is None or len(roots) != 1:
            continue
        if any(
            tensor_roots_may_alias(
                roots,
                _store_tensor_roots(other, tensor_names),
                proven_disjoint_tensor_pairs,
            )
            for other in writes
            if other is not store
        ):
            continue
        if not read_back and any(
            (pointer := _plain_scalar_load_pointer(load)) is None
            or tensor_roots_may_alias(
                roots,
                _tensor_arg_roots(pointer, tensor_names),
                proven_disjoint_tensor_pairs,
            )
            for load in loads
        ):
            continue
        removed.add(index)

    if not removed:
        return 0
    serial_loop.body = [
        statement
        for index, statement in enumerate(serial_loop.body)
        if index not in removed
    ]

    # The final pass executes the same nonempty Cartesian iteration set whenever
    # the first pass executes. These unconditional definitions therefore kill
    # the corresponding first-pass bindings, including after either loop exits.
    redefined = _redefined_names(final_flat, protected_names)
    _prune_redefined_bindings(lane_loop, serial_loop, redefined)
    _drop_emptied_serial_loop(lane_loop, serial_loop, redefined)
    return len(removed)


def _drop_emptied_serial_loop(
    lane_loop: ast.For, serial_loop: ast.For, redefined: set[str]
) -> None:
    """Remove ``serial_loop`` from the first pass once nothing is left in it.

    The bindings of the first pass that the final pass redefines and nothing
    left reads go with it; the lane loop's body may end up empty, which the
    caller takes as the whole first pass being dead.  A loop whose variable
    the rest of the first pass reads stays, with a ``pass`` body.
    """
    if serial_loop.body:
        return
    rest = [statement for statement in lane_loop.body if statement is not serial_loop]
    if (
        not isinstance(serial_loop.target, ast.Name)
        or serial_loop.target.id in ReadWrites.from_list(rest).reads
    ):
        serial_loop.body = [ast.Pass()]
        return
    lane_loop.body = rest
    _prune_redefined_bindings(lane_loop, lane_loop, redefined)


def _redefined_names(final_flat: list[ast.stmt], protected_names: set[str]) -> set[str]:
    """Names the final pass binds unconditionally before any read of its own."""
    return {
        name
        for statement in final_flat
        if (name := assigned_name(statement)) is not None
    } - (_live_in(final_flat) | protected_names)


def _prune_redefined_bindings(
    lane_loop: ast.For, owner: ast.For, redefined: set[str]
) -> None:
    """Drop ``owner``'s plain bindings that the final pass redefines and no
    statement of the first pass (``lane_loop``) still reads."""
    while True:
        reads = ReadWrites.from_ast(lane_loop).reads
        dead = [
            statement
            for statement in owner.body
            if (name := assigned_name(statement)) is not None
            and name in redefined
            and reads.get(name, 0) == ReadWrites.from_ast(statement).reads.get(name, 0)
            and _is_proven_relocatable_assignment(statement, allow_load=True)
        ]
        if not dead:
            break
        owner.body = [statement for statement in owner.body if statement not in dead]


def overwrites_later_first_pass_write(
    lane_loop: ast.For,
    serial_loop: ast.For,
    final_nest: list[ast.AST],
    consumers: list[ast.AST],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    protected_names: set[str],
) -> bool:
    """Whether a write of the final pass may land on one the tile program
    orders after it.

    The final pass runs after the whole first pass (``lane_loop``), so each
    of its writes lands after every write left there.  A first-pass copy of
    one of the reduction's consumer stores (``consumers``) writes a partial
    value the final pass overwrites when it stores again over the same
    elements (``_overwritten_stores``: coverage independent of the
    reduction).  Any other first-pass write that may reach a tensor the final
    pass writes may come after it in the tile program (a store only the first
    pass runs, after a consumer store of the same element), and so may a copy
    whose coverage depends on the reduction (``if reduced < 5.0:``): the
    final pass need not store again where it stored.  The alias facts hold
    only for the tensor names ``_nest_tensor_names`` checks; a write
    through any other name may reach every tensor, a subscript write reaches
    what ``_subscript_roots`` says (thread-local storage none), and so does
    an access the passes cannot resolve (``_unclassified_accesses``).  Loads
    are ``drop_read_back_partial_stores``'s and ``decline_stale_reads``'s.
    """
    tensor_names, unchecked = _nest_tensor_names(
        [lane_loop, *final_nest], protected_names
    )
    subscript_roots = _subscript_roots(
        [lane_loop, *final_nest], tensor_names, unchecked
    )
    overwritten: set[int] = set()
    if any(node is serial_loop for node in lane_loop.body):
        copies = [
            index
            for index, statement in enumerate(serial_loop.body)
            if any(statement is consumer for consumer in consumers)
        ]
        overwritten = {
            id(store)
            for store in _overwritten_stores(
                lane_loop,
                serial_loop,
                final_nest,
                copies,
                protected_names,
                known_effects=False,
            ).values()
        }

    def writes(nodes: list[ast.AST]) -> list[frozenset[str] | None]:
        roots = [
            _store_tensor_roots(call, tensor_names)
            for node in nodes
            for call in _write_calls(node)
            if id(call) not in overwritten
        ]
        roots += [
            subscript_roots(name)
            for node in nodes
            for name in _subscript_names(node, ast.Store)
        ]
        roots += [
            access
            for node in nodes
            for access in _unclassified_accesses(node, writes=True)
        ]
        return [access for access in roots if access != frozenset()]

    return _any_may_alias(
        writes(final_nest), writes([lane_loop]), proven_disjoint_tensor_pairs
    )


def atomic_write_calls(statement: ast.AST) -> list[ast.Call]:
    """The atomic read-modify-writes among ``statement``'s memory writes."""
    return [call for call in _memory_write_calls(statement) if _is_atomic_call(call)]


# Calls building thread-local storage, whose subscripts reach no tensor.
_LOCAL_STORAGE_CALLS = frozenset(
    {
        "cute.make_rmem_tensor",
        "cute.make_rmem_tensor_like",
        "cute.make_fragment",
        "cute.make_fragment_like",
    }
)


def _subscript_names(node: ast.AST, context: type[ast.expr_context]) -> list[str]:
    """The names ``node`` reads (``context`` ``ast.Load``) or writes
    (``ast.Store``) through a subscript."""
    return [
        child.value.id
        for child in ast.walk(node)
        if isinstance(child, ast.Subscript)
        and isinstance(child.ctx, context)
        and isinstance(child.value, ast.Name)
    ]


def _subscript_roots(
    nodes: list[ast.AST], tensor_names: set[str], unchecked: set[str]
) -> Callable[[str], frozenset[str] | None]:
    """For a name subscripted in ``nodes``, the tensors the subscript may
    access: of the alias-checked ``tensor_names``, or any (``None``) for one
    of the ``unchecked`` tensor arguments.  Any other name reaches what its
    bindings in ``nodes`` do: none when each binds thread-local storage (a
    list, tuple or constant, a register tensor, a loaded value, the values
    of inline assembly proven to be scalar arithmetic), what the
    name reaches for each binding to a name (``alias = column``), and any
    tensor when one binds something else or the name has no binding there
    (a tensor argument ``nodes`` never addresses as ``NAME.iterator``).  An
    element write (``frag[0] = value``) stores into what the name reaches
    and binds nothing."""
    bindings: dict[str, list[ast.expr | None]] = {}
    for node in nodes:
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and all(
                isinstance(target, ast.Name) for target in child.targets
            ):
                for target in child.targets:
                    bindings.setdefault(cast("ast.Name", target).id, []).append(
                        child.value
                    )
                continue
            if isinstance(child, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.For)):
                targets = (
                    child.targets if isinstance(child, ast.Assign) else [child.target]
                )
                for target in targets:
                    for name in ast.walk(target):
                        if isinstance(name, ast.Name) and isinstance(
                            name.ctx, ast.Store
                        ):
                            bindings.setdefault(name.id, []).append(None)
    memo: dict[str, frozenset[str] | None] = {}

    def roots(name: str) -> frozenset[str] | None:
        if name in tensor_names:
            return frozenset((name,))
        if name in unchecked or name not in bindings:
            return None
        if name in memo:
            return memo[name]
        memo[name] = frozenset()  # a cycle adds nothing of its own
        combined: frozenset[str] = frozenset()
        for value in bindings[name]:
            if isinstance(value, ast.Name):
                reached = roots(value.id)
            elif isinstance(
                value,
                (ast.List, ast.Tuple, ast.ListComp, ast.Dict, ast.Set, ast.Constant),
            ) or (
                isinstance(value, ast.Call)
                and (
                    call_path_str(value.func) in _LOCAL_STORAGE_CALLS
                    or any(load is value for load in _memory_load_calls(value))
                    or is_pure_inline_asm_call(value)
                )
            ):
                reached = frozenset()
            else:
                reached = None
            if reached is None:
                memo[name] = None
                return None
            combined |= reached
        memo[name] = combined
        return combined

    return roots


def _write_calls(node: ast.AST) -> list[ast.Call]:
    """``node``'s memory writes: the stores and atomics of
    ``_memory_write_calls`` and the bulk and helper writes ``_is_store_call``
    adds."""
    calls = _memory_write_calls(node)
    seen = {id(call) for call in calls}
    return [
        *calls,
        *(
            call
            for call in ast.walk(node)
            if isinstance(call, ast.Call)
            and id(call) not in seen
            and _is_store_call(call)
        ),
    ]


# Calls that bound a loop or mark a lane reduction, accessing no memory.
_EFFECT_FREE_CALLS = frozenset(
    {"range", "cutlass.range", "cutlass.range_constexpr", _HELION_LANE_REDUCE_MARKER}
)
# The ``cute`` calls that access memory; the rest of the namespace is layout
# and tensor-view algebra (``cute.crd2idx``, ``cute.local_tile``), whose
# views the subscripts and loads through them access.
_CUTE_MEMORY_CALLS = frozenset(
    {"cute.autovec_copy", "cute.basic_copy", "cute.copy", "cute.gemm"}
)


def _reads_no_memory(node: ast.Subscript) -> bool:
    """Whether subscript ``node`` reads a coordinate or layout metadata
    (``cute.arch.thread_idx()[0]``, ``x.layout.stride[1]``, ``x.shape[0]``)."""
    return _known_subscript(node) or (
        isinstance(node.value, ast.Attribute) and node.value.attr in {"shape", "stride"}
    )


def _unclassified_accesses(node: ast.AST, *, writes: bool) -> list[None]:
    """An access of any tensor (``None``) for each access in ``node`` the
    passes cannot resolve to tensors: a call that is no load or write they
    collect (``_memory_load_calls``, ``_write_calls``) and not proven free of
    memory accesses (``_is_proven_relocatable_call`` without loads, the
    layout algebra of the ``cute`` namespace; thread-local storage, loop
    ranges and lane-reduce markers aside), which may read and write any
    memory (an impure ``hl.inline_asm_elementwise``, a packet load helper),
    and a subscript read (with ``writes``, write) whose base is no name
    (``aliases[0][col]``) unless it reads metadata."""
    known = {id(call) for call in [*_memory_load_calls(node), *_write_calls(node)]}
    context = ast.Store if writes else ast.Load
    result: list[None] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            path = call_path_str(child.func)
            if not (
                id(child) in known
                or path in _LOCAL_STORAGE_CALLS
                or path in _EFFECT_FREE_CALLS
                or (
                    path is not None
                    and path.startswith("cute.")
                    and path.count(".") == 1
                    and path not in _CUTE_MEMORY_CALLS
                )
                or _is_proven_relocatable_call(
                    child, allow_load=False, allow_reduction=True
                )
            ):
                result.append(None)
        elif (
            isinstance(child, ast.Subscript)
            and isinstance(child.ctx, context)
            and not isinstance(child.value, ast.Name)
            and not _reads_no_memory(child)
        ):
            result.append(None)
    return result


def _nest_tensor_names(
    nodes: list[ast.AST], protected_names: set[str]
) -> tuple[set[str], set[str]]:
    """The tensor arguments ``nodes`` address as ``NAME.iterator``: those the
    alias facts describe there (as in ``eliminate_interchanged_stores``, no
    statement of ``nodes`` mutates the name or a value reached through it,
    ``_binding_write_roots``, and it has no pending rename), and the others,
    an access through which may access every tensor."""
    names = _iterator_tensor_names(nodes)
    mutated = set().union(*map(_binding_write_roots, nodes))
    checked = names - mutated - protected_names
    return checked, names - checked


def _resolved_access_roots(
    statements: list[ast.stmt],
    unstable_names: list[set[str]],
    tensor_names: set[str],
    *,
    writes: bool,
    unchecked: set[str],
) -> list[list[frozenset[str] | None]]:
    """For each of ``statements``, the tensors each of its loads (or, with
    ``writes``, each of its memory writes) may access, ``None`` for one that
    may access any, then its subscript accesses of tensor arguments (of the
    alias-checked ``tensor_names``, or of the ``unchecked`` ones, which may
    access any) and last the accesses it makes that the passes cannot
    resolve (``_unclassified_accesses``).

    An address that names no tensor argument itself (``pointer.load()`` after
    ``pointer = output.iterator + lane``) is resolved through the
    definitions reaching its statement in ``statements`` as straight-line
    code.  That takes a top-level assignment or expression, where no nested
    statement can rebind a name ahead of the access, and an address none of
    whose dependencies is among the statement's ``unstable_names``: names
    whose definitions ahead of it may not be the ones reaching it at run
    time (``_first_nest_flat``, ``_final_nest_flat``).
    """
    snapshots: list[dict[str, ast.expr]] | None = None
    snapshots_built = False
    subscript_roots = _subscript_roots(
        cast("list[ast.AST]", statements), tensor_names, unchecked
    )
    result: list[list[frozenset[str] | None]] = []
    for index, statement in enumerate(statements):
        accesses: list[frozenset[str] | None] = []
        for call in (
            _write_calls(statement) if writes else _memory_load_calls(statement)
        ):
            if writes:
                roots = _store_tensor_roots(call, tensor_names)
                pointer = _plain_scalar_store_pointer(call)
            else:
                roots = _tensor_arg_roots(call, tensor_names)
                pointer = _plain_scalar_load_pointer(call)
            if (
                roots is None
                and pointer is not None
                and isinstance(statement, (ast.Assign, ast.Expr))
                and not (
                    _dependency_names(
                        cast("list[ast.AST]", statements[:index]), _reads(pointer)
                    )
                    & unstable_names[index]
                )
            ):
                if not snapshots_built:
                    snapshots = _memory_address_definition_snapshots(statements)
                    snapshots_built = True
                if snapshots is not None:
                    frozen = _freeze_definition(pointer, snapshots[index])
                    if frozen is not None:
                        roots = _tensor_arg_roots(frozen, tensor_names)
            accesses.append(roots)
        for name in _subscript_names(statement, ast.Store if writes else ast.Load):
            roots = subscript_roots(name)
            if roots != frozenset():
                accesses.append(roots)
        accesses += _unclassified_accesses(statement, writes=writes)
        result.append(accesses)
    return result


def _carried_names(loop: ast.For, flat_body: list[ast.stmt]) -> set[str]:
    """The names ``loop`` carries from one iteration to the next: rebound in
    its body and read in ``flat_body`` (that body as straight-line code)
    before a plain assignment binds them."""
    return set(ReadWrites.from_list(loop.body).writes) & _live_in(flat_body)


_COMPOUND_STATEMENTS = (ast.If, ast.For, ast.While, ast.With, ast.Try)


def carried_lane_updates(
    lane_loop: ast.For,
    serial_body: list[ast.AST],
    rename_groups: Mapping[str, str] | None = None,
) -> Callable[[ast.AST], set[str]]:
    """For a statement of ``serial_body`` (the body of a serial loop of
    ``lane_loop``), the values it updates that the loop carries from one
    iteration to the next: a name it rebinds, or the thread-local storage
    (``_subscript_roots``) it writes an element of, that ``serial_body``
    may read before an update (``exposed_reads``: a branch updates what both
    its arms update, a loop nothing after it).  Names one pending rename
    merges (``rename_groups``) are one value."""
    renames = rename_groups or {}
    local = _subscript_roots([lane_loop], set(), _iterator_tensor_names([lane_loop]))

    def updated(statement: ast.AST) -> set[str]:
        return {
            renames.get(node.id, node.id)
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        } | {
            name
            for name in _subscript_names(statement, ast.Store)
            if local(name) == frozenset()
        }

    def reads(statement: ast.AST) -> set[str]:
        # A plain element write (``frag[0] = value``) reads no element of the
        # storage it writes; ``frag[0] += value`` does.
        bases = [
            target.value
            for node in ast.walk(statement)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
        ]
        loaded = {
            node.id
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and not any(node is base for base in bases)
        }
        return _reads(statement) - (
            {cast("ast.Name", base).id for base in bases} - loaded
        )

    carried = set().union(*map(updated, serial_body)) & exposed_reads(
        serial_body,
        renames=renames,
        reads=reads,
        defines=lambda statement: (
            set() if isinstance(statement, _COMPOUND_STATEMENTS) else updated(statement)
        ),
    )
    return lambda statement: updated(statement) & carried


def _first_nest_flat(
    lane_loop: ast.For, serial_loop: ast.For, protected_names: set[str]
) -> tuple[list[ast.stmt], list[set[str]]]:
    """The first nest as straight-line code (the lane loop's prefix, its
    serial loop's body and its suffix) and, for each statement, the names
    whose definitions ahead of it there may not be the ones reaching it: in
    and after the serial loop, the names it carries, which a definition
    ahead of the loop reaches in its first iteration only; after it, every
    name its body binds (it may run no iteration); and throughout, the names
    with a pending rename (``protected_names``)."""
    serial_index = lane_loop.body.index(serial_loop)
    prefix = lane_loop.body[:serial_index]
    suffix = lane_loop.body[serial_index + 1 :]
    within = _carried_names(serial_loop, serial_loop.body) | protected_names
    after = within | set(ReadWrites.from_list(serial_loop.body).writes)
    return (
        [*prefix, *serial_loop.body, *suffix],
        [
            *(protected_names for _ in prefix),
            *(within for _ in serial_loop.body),
            *(after for _ in suffix),
        ],
    )


def _final_nest_flat(
    final_nest: list[ast.AST], protected_names: set[str]
) -> tuple[list[ast.stmt], list[set[str]]]:
    """The final nest as straight-line code (its prefix, its serial loop's
    head and its inner lane loop's body) and, for each statement, the names
    whose definitions ahead of it there may not be the ones reaching it: in
    the loops, the names either carries, and throughout, the names with a
    pending rename (``protected_names``)."""
    flat = _final_flat(final_nest)
    final_serial = cast("ast.For", final_nest[-1])
    final_lane = cast("ast.For", final_serial.body[-1])
    prefix_size = len(final_nest) - 1
    looped = flat[prefix_size:]
    carried = (
        _carried_names(final_lane, final_lane.body)
        | _carried_names(final_serial, looped)
        | protected_names
    )
    return (
        flat,
        [*(protected_names for _ in range(prefix_size)), *(carried for _ in looped)],
    )


def _any_may_alias(
    reads: Iterable[frozenset[str] | None],
    writes: Iterable[frozenset[str] | None],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
) -> bool:
    writes = list(writes)
    return any(
        tensor_roots_may_alias(read, write, proven_disjoint_tensor_pairs)
        for read in reads
        for write in writes
    )


_OFFSET_MODULUS = 1 << 32
# Integer casts that wrap modulo a power of two, with the values each keeps.
_NARROWING_CASTS = {
    f"cutlass.{name}{bits}": (
        (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        if name == "Int"
        else (0, (1 << bits) - 1)
    )
    for name in ("Int", "Uint")
    for bits in (8, 16, 32)
}
# The 32-bit casts, which all agree modulo 2**32, and the operations that
# commute with narrowing modulo 2**32.
_CASTS_MOD_2_32 = frozenset({"cutlass.Int32", "cutlass.Uint32"})
_RING_OPERATORS = (ast.Add, ast.Sub, ast.Mult)


def _narrowing_cast(expression: ast.AST) -> str | None:
    """The cast (``cutlass.Int32``) ``expression`` applies when it is one of
    ``_NARROWING_CASTS``."""
    if (
        isinstance(expression, ast.Call)
        and len(expression.args) == 1
        and not expression.keywords
        and call_path_str(expression.func) in _NARROWING_CASTS
    ):
        return call_path_str(expression.func)
    return None


def _instances_never_meet(
    serial_step: int, serial_span: int, lane_step: int, lane_span: int
) -> bool:
    """Whether two instances of an access at offset ``serial_step * serial +
    lane_step * lane + C`` always touch distinct elements, the serial
    variable's values at most ``serial_span`` apart and the lane's at most
    ``lane_span``.

    Generated offsets may narrow through ``cutlass.Int32``, so two offsets
    meet when they agree modulo 2**32 (which covers Int64 offsets as well),
    and each step counts at its smallest value modulo 2**32: an offset of
    ``Int32(serial * 2**32) + lane`` does not move with the serial variable.
    When ``|A| * serial_span + |B| * lane_span`` stays below 2**32, a
    difference ``A * ds + B * dl`` is zero modulo 2**32 only when it is zero,
    so ``(ds, dl)`` is a multiple of ``(B, -A) / gcd(A, B)``: the instances
    never meet when that step leaves either range.  This proves a serial
    stride that outweighs every lane difference (a row-major layout) as well
    as a lane stride that outweighs every serial one (a transposed layout).
    """

    def smallest(step: int) -> int:
        step %= _OFFSET_MODULUS
        return step - _OFFSET_MODULUS if step > _OFFSET_MODULUS // 2 else step

    serial_step, lane_step = abs(smallest(serial_step)), abs(smallest(lane_step))
    if serial_step * serial_span + lane_step * lane_span >= _OFFSET_MODULUS:
        return False
    if serial_step == 0 or lane_step == 0:
        return (serial_span == 0 or serial_step > 0) and (
            lane_span == 0 or lane_step > 0
        )
    divisor = math.gcd(serial_step, lane_step)
    return lane_step // divisor > serial_span or serial_step // divisor > lane_span


def _serial_span(
    serial_loop: ast.For,
    prefix: list[ast.stmt],
    mutable_names: set[str],
    protected_names: set[str],
    constexpr_value: Callable[[str], int | None] | None,
) -> int | None:
    """The largest difference between two values of ``serial_loop``'s
    variable across all lanes, from its ``range`` bounds as they stand after
    the lane loop's ``prefix``, or ``None`` when the bounds may move with the
    lane or do not differ by a known integer.

    Every lane runs the same range: as in ``eliminate_interchanged_stores``,
    the bounds read no name the lane loop binds (``mutable_names``, its
    variable included), no memory and no name with a pending rename.  A
    constexpr (a tile's block size) takes its value; any other name must
    cancel out between the bounds, as the tile offset of a device loop over
    its grid tile does (``range(offset, offset + _BLOCK_SIZE_0, ...)``).  An
    end written as a ``min`` (``_min_arms``), as the masked tail of such a
    loop's dynamic extent is, counts through whichever arm the begin cancels
    out of, in a range whose step is a known positive integer: a descending
    one stays above its end.

    The difference of the bounds is computed as if integer casts were exact.
    A narrowing cast (``_NARROWING_CASTS``) of a known integer it keeps is;
    any other may wrap.  A 32-bit cast around both bounds
    (``range(cutlass.Int32(a), cutlass.Int32(b), step)``) keeps them
    within 2**32 of each other and, with only 32-bit casts inside and only
    ``+``, ``-`` and ``*``, their difference modulo 2**32: the range spans at
    most what it would unwrapped or, when the difference wrapped, runs no
    iteration, provided its step goes the way of the difference; so does a
    ``min`` end whose arms are all 32-bit values themselves.  A bound that
    may wrap otherwise (``range(cutlass.Int32(n), n + 1, n)`` runs twice at
    ``n = 2**32``) leaves the span unknown, and so does a step that may wrap
    in the cases that need its direction (``cutlass.Int32(2147483648)``
    counts down).
    """
    entry = _definition_snapshots([*prefix, ast.Pass()])
    if (
        entry is None
        or _dependency_names(cast("list[ast.AST]", prefix), _reads(serial_loop.iter))
        & protected_names
    ):
        return None
    iterator = _freeze_definition(serial_loop.iter, entry[-1])
    if (
        not isinstance(iterator, ast.Call)
        or call_path_str(iterator.func) not in {"range", "cutlass.range_constexpr"}
        or iterator.keywords
        or not 1 <= len(iterator.args) <= 3
    ):
        return None
    if len(iterator.args) == 1:
        begin, end = ast.Constant(0), iterator.args[0]
    else:
        begin, end = iterator.args[:2]
    if not all(
        _coordinate_expression(bound, mutable_names, set()) for bound in (begin, end)
    ):
        return None

    class _Constexprs(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            value = None if constexpr_value is None else constexpr_value(node.id)
            return node if value is None else ast.Constant(value)

    def constant(expression: ast.expr) -> int | None:
        """``expression``'s value when it is a known integer."""
        # The bounds read no loop variable, so there is none to track.
        form = _pointer_integer_affine_form(
            _Constexprs().visit(cast("ast.expr", clone_ast(expression))),
            "",
            {},
            set(),
            "range",
        )
        return None if form is None or form[1] else form[2]

    class _KeptConstants(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.expr:
            self.generic_visit(node)
            cast_path = _narrowing_cast(node)
            if cast_path is None:
                return node
            value = constant(node.args[0])
            low, high = _NARROWING_CASTS[cast_path]
            return (
                node
                if value is None or not low <= value <= high
                else ast.Constant(value)
            )

    begin, end = (
        cast("ast.expr", _KeptConstants().visit(cast("ast.expr", clone_ast(bound))))
        for bound in (begin, end)
    )
    casts: list[str] = []
    while (
        cast_path := _narrowing_cast(begin)
    ) is not None and cast_path == _narrowing_cast(end):
        begin, end = cast("ast.Call", begin).args[0], cast("ast.Call", end).args[0]
        casts.append(cast_path)
    wrapped = bool(casts)
    casts += [
        cast_path
        for bound in (begin, end)
        for node in ast.walk(bound)
        if (cast_path := _narrowing_cast(node)) is not None
    ]
    if casts and not (
        wrapped
        and set(casts) <= _CASTS_MOD_2_32
        and all(
            isinstance(node.op, _RING_OPERATORS)
            for bound in (begin, end)
            for node in ast.walk(bound)
            if isinstance(node, ast.BinOp)
        )
    ):
        return None

    def difference(bound: ast.expr) -> int | None:
        """``bound - begin`` when it is a known integer."""
        return constant(ast.BinOp(left=bound, op=ast.Sub(), right=begin))

    # The step's direction decides which way the range runs; a narrowing cast
    # of a value it does not keep may flip it (``cutlass.Int32(2**31)`` is
    # negative), so such a step has no known value.
    step = (
        cast(
            "ast.expr",
            _KeptConstants().visit(cast("ast.expr", clone_ast(iterator.args[2]))),
        )
        if len(iterator.args) == 3
        else ast.Constant(1)
    )
    step_value = (
        None
        if any(_narrowing_cast(node) is not None for node in ast.walk(step))
        else constant(step)
    )
    exact = difference(end)
    if exact is not None:
        if (
            wrapped
            and exact
            and (step_value is None or (step_value > 0) != (exact > 0))
        ):
            return None
        return max(abs(exact) - 1, 0)
    # A device loop over its grid tile's masked tail ends at ``min(offset +
    # block, size)``, generated as ``a if a < b else b``: the serial variable
    # of an ascending range stays below either arm, so an arm a known integer
    # past ``begin`` bounds the span (an empty range has no two values to
    # compare).  With the bounds wrapped, every arm must be a 32-bit value
    # itself: the min is then one too, the cast around it changes nothing,
    # and an arm lies within 2**32 of ``begin`` as the end does.  An arm that
    # may wrap would let the end wrap past the arm counted
    # (``cutlass.Int32(min(cutlass.Int32(t + 128), n))`` for a very negative
    # ``n``).
    if step_value is None or step_value <= 0:
        return None
    arms = _min_arms(end)
    if wrapped and not all(_narrowing_cast(arm) in _CASTS_MOD_2_32 for arm in arms):
        return None
    spans = [
        span for arm in arms if (span := difference(arm)) is not None and span >= 0
    ]
    return max(min(spans) - 1, 0) if spans else None


def _min_arms(expression: ast.expr) -> list[ast.expr]:
    """The arms of ``expression`` written as ``min(a, b)`` (``a if a < b else
    b`` or ``b if a > b else a``, through integer casts), else none."""
    while (
        isinstance(expression, ast.Call)
        and len(expression.args) == 1
        and not expression.keywords
        and call_path_str(expression.func) in {"cutlass.Int32", "cutlass.Int64"}
    ):
        expression = expression.args[0]
    if not (
        isinstance(expression, ast.IfExp)
        and isinstance(expression.test, ast.Compare)
        and len(expression.test.ops) == 1
        and len(expression.test.comparators) == 1
    ):
        return []
    left, right = expression.test.left, expression.test.comparators[0]
    operator = expression.test.ops[0]
    body, orelse = ast.dump(expression.body), ast.dump(expression.orelse)
    if isinstance(operator, (ast.Lt, ast.LtE)):
        smaller, larger = left, right
    elif isinstance(operator, (ast.Gt, ast.GtE)):
        smaller, larger = right, left
    else:
        return []
    if body != ast.dump(smaller) or orelse != ast.dump(larger):
        return []
    return [expression.body, expression.orelse]


def _reads_only_its_own_element(
    load: ast.Call,
    load_definitions: dict[str, ast.expr],
    store: ast.Call,
    store_definitions: dict[str, ast.expr],
    *,
    lane_var: str,
    lane_extent: int,
    serial_var: str,
    serial_span: int,
    mutable_names: set[str],
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
) -> bool:
    """Whether each instance of ``load`` reads an element only its own
    instance of ``store`` writes.

    An instance is one lane's pass through one serial iteration of the
    first pass, whose loops the caller knows give their variables a distinct
    value per iteration, the serial variable's values at most
    ``serial_span`` apart.  With the definitions reaching them inlined, both
    addresses must be one expression over a single tensor, immutable values
    and the two loop variables (``_coordinate_expression``), integer-affine
    in each: ``A * serial + B * lane + C`` with ``C`` the same at every
    instance, which two instances never share (``_instances_never_meet``,
    modulo 2**32).  This is one thread's program: the barrier pass already
    declines a body with lane reduction markers whose threads could race on
    a tensor (``lane_loops/barriers.py``).
    """
    load_pointer = _plain_scalar_load_pointer(load)
    store_pointer = _plain_scalar_store_pointer(store)
    if load_pointer is None or store_pointer is None:
        return False
    address = _freeze_definition(load_pointer, load_definitions)
    store_address = _freeze_definition(store_pointer, store_definitions)
    if (
        address is None
        or store_address is None
        or ast.dump(address) != ast.dump(store_address)
        or _single_tensor_iterator_root(address) is None
        or not _coordinate_expression(address, mutable_names, {lane_var, serial_var})
    ):
        return False
    lane_form = _pointer_integer_affine_form(
        address, lane_var, proven_tensor_stride_values, set(), "load"
    )
    serial_form = _pointer_integer_affine_form(
        address, serial_var, proven_tensor_stride_values, set(), "load"
    )
    if lane_form is None or serial_form is None:
        return False
    return _instances_never_meet(
        serial_form[0], serial_span, lane_form[0], lane_extent - 1
    )


def drop_read_back_partial_stores(
    lane_loop: ast.For,
    serial_loop: ast.For,
    final_nest: list[ast.AST],
    consumers: list[ast.AST],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    protected_names: set[str],
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    constexpr_value: Callable[[str], int | None] | None = None,
    duplicated: Iterable[ast.AST] = (),
) -> None:
    """Drop the first pass's stores whose value a load reads back out of order.

    A consumer store left in the first pass writes the reduction's per-lane
    partial value, which the final pass overwrites with the complete one and
    no load of the tile program reads.  An in-place update reads it all the
    same: the final pass loading the tensor the store updates, or a
    first-pass load of it that, in the tile program, reads what the store
    left in an earlier serial iteration, in its own one after the store, or
    (one in the lane loop's prefix) in an earlier lane, where the tile
    program's prefix reads the values the nest started from.  A
    ``duplicated`` store (one the final pass needs that does not consume the
    reduction, so both passes run it) writes the complete value, but a
    final-pass load of its tensor would read it before the tile program
    stores it.  Such a store has to go.  It is dropped through
    ``eliminate_interchanged_stores(..., read_back=True)``, which proves the
    overwrite with the load condition waived, when a load may read its
    tensor (its address resolved through the definitions reaching it,
    ``_resolved_access_roots``; one naming no tensor may read any): one in
    the final pass, or, for a consumer, one in the first pass other than a
    load ahead of the store that reads only its own instance's element
    (``_reads_only_its_own_element``: that element keeps its starting value
    until then, store kept or not).  The final pass then reads the values
    the nest started from, or its own earlier iterations' complete ones, as
    the tile program does, and so does the prefix.

    Once the store is dropped, every load left in the first pass's serial
    loop and after it that may read its destination must be one ahead of it
    reading only its own instance's element: any other would see the
    starting value where the tile program sees a complete one.  Raises
    ``BackendUnsupported`` for a store that cannot be dropped and for such a
    load: the nest is declined.  ``constexpr_value`` resolves the constexprs
    bounding the serial loop (``DeviceFunction.resolved_int_constexpr``).
    """
    body = list(serial_loop.body)
    positions = {id(statement): index for index, statement in enumerate(body)}
    copies = {id(statement) for statement in duplicated}
    candidates = [
        statement
        for statement in [*consumers, *duplicated]
        if id(statement) in positions and not atomic_write_calls(statement)
    ]
    if not candidates:
        return
    pairs = proven_disjoint_tensor_pairs
    tensor_names, unchecked = _nest_tensor_names(
        [lane_loop, *final_nest], protected_names
    )
    serial_index = lane_loop.body.index(serial_loop)
    prefix = lane_loop.body[:serial_index]
    suffix = lane_loop.body[serial_index + 1 :]
    flat, unstable = _first_nest_flat(lane_loop, serial_loop, protected_names)
    flat_index = {id(statement): index for index, statement in enumerate(flat)}
    reads = _resolved_access_roots(
        flat, unstable, tensor_names, writes=False, unchecked=unchecked
    )
    writes = _resolved_access_roots(
        flat, unstable, tensor_names, writes=True, unchecked=unchecked
    )
    destinations = {
        id(statement): writes[flat_index[id(statement)]] for statement in candidates
    }
    final_flat, final_unstable = _final_nest_flat(final_nest, protected_names)
    final_reads = [
        roots
        for accesses in _resolved_access_roots(
            final_flat, final_unstable, tensor_names, writes=False, unchecked=unchecked
        )
        for roots in accesses
    ]
    looped = flat[: len(prefix) + len(body)]
    snapshots = _memory_address_definition_snapshots(looped)
    lane_extent = _static_lane_loop_extent(lane_loop)
    mutable_names = _binding_write_roots(lane_loop)
    # A ``range`` gives the serial variable a distinct value per iteration,
    # those of all lanes at most ``serial_span`` apart.
    serial_span = _serial_span(
        serial_loop, prefix, mutable_names, protected_names, constexpr_value
    )
    body_unstable = unstable[len(prefix)]

    def own_element(load: ast.Call, index: int, statement: ast.AST) -> bool:
        store = _single_store(statement)
        holder = body[index]
        if (
            store is None
            or snapshots is None
            or lane_extent is None
            or serial_span is None
            or not isinstance(lane_loop.target, ast.Name)
            or not isinstance(serial_loop.target, ast.Name)
            # The definitions reaching ``holder`` describe its load only when
            # no nested statement can rebind a name ahead of it.
            or not isinstance(holder, ast.Assign)
            or len(holder.targets) != 1
            or not isinstance(holder.targets[0], ast.Name)
            or not any(node is load for node in ast.walk(holder.value))
        ):
            return False
        pointers = [
            _plain_scalar_load_pointer(load),
            _plain_scalar_store_pointer(store),
        ]
        if any(pointer is None for pointer in pointers) or (
            _dependency_names(
                cast("list[ast.AST]", looped),
                {
                    name
                    for pointer in pointers
                    for name in _reads(cast("ast.expr", pointer))
                },
            )
            & body_unstable
        ):
            return False
        return _reads_only_its_own_element(
            load,
            snapshots[len(prefix) + index],
            store,
            snapshots[len(prefix) + positions[id(statement)]],
            lane_var=lane_loop.target.id,
            lane_extent=lane_extent,
            serial_var=serial_loop.target.id,
            serial_span=serial_span,
            mutable_names=mutable_names,
            proven_tensor_stride_values=proven_tensor_stride_values,
        )

    def misreads(node: ast.AST, index: int | None, statement: ast.AST) -> bool:
        """Whether a load in the first pass's ``node`` (at ``index`` in the
        serial loop's body, ``None`` after the loop) may read what
        ``statement`` stores where the tile program reads a value the store
        left: every load that may read its destination but one ahead of it
        reading only its own instance's element."""
        roots = destinations[id(statement)]
        ahead = index is not None and index < positions[id(statement)]
        loads = _memory_load_calls(node)
        return any(
            tensor_roots_may_alias(write, read, pairs)
            and not (
                position < len(loads)
                and ahead
                and own_element(loads[position], cast("int", index), statement)
            )
            for position, read in enumerate(reads[flat_index[id(node)]])
            for write in roots
        )

    first_pass = [
        *enumerate(body),
        *((None, node) for node in suffix),
    ]
    prefix_reads = [roots for accesses in reads[: len(prefix)] for roots in accesses]
    read_back = [
        statement
        for statement in candidates
        if _any_may_alias(final_reads, destinations[id(statement)], pairs)
        or id(statement) not in copies
        and (
            _any_may_alias(prefix_reads, destinations[id(statement)], pairs)
            or any(
                node is not statement and misreads(node, index, statement)
                for index, node in first_pass
            )
        )
    ]
    if not read_back:
        return
    eliminate_interchanged_stores(
        lane_loop,
        serial_loop,
        final_nest,
        {positions[id(statement)] for statement in read_back},
        proven_disjoint_tensor_pairs,
        protected_names,
        read_back=True,
    )
    if any(statement in serial_loop.body for statement in read_back):
        raise exc.BackendUnsupported(
            "cute",
            "interchanged lane reduction cannot drop a partial store a load reads back",
        )
    # The prefix reads the values the nest started from once the store is
    # gone, as the tile program's prefix does.
    left = [
        (index, node)
        for index, node in first_pass
        if node in serial_loop.body or node in lane_loop.body
    ]
    if any(
        misreads(node, index, statement)
        for statement in read_back
        for index, node in left
    ):
        raise exc.BackendUnsupported(
            "cute",
            "a first-pass load of an interchanged lane reduction may read what a "
            "dropped partial store writes, after the store or in another serial "
            "iteration",
        )


def decline_stale_reads(
    first_nest: ast.For,
    serial_loop: ast.For,
    final_nest: list[ast.AST],
    per_iteration: list[ast.AST],
    prefix: list[ast.AST],
    proven_disjoint_tensor_pairs: set[frozenset[str]],
    protected_names: set[str],
    head_moves: Iterable[tuple[ast.AST, list[ast.AST]]] = (),
    rename_groups: Mapping[str, str] | None = None,
) -> None:
    """Decline passes that may load a value the tile program never saw at
    that point.

    The final pass runs after the whole first pass (``first_nest``, with its
    serial loop ``serial_loop`` unless that is gone; its body is empty once
    all of it is), so a load in it that may read a tensor the first pass
    still writes may read what the tile program stores only later, or the
    first pass's partial value.  It also reruns the lane-varying prefix
    statements it needs (``per_iteration``) in every serial iteration, where
    the tile program runs them once, before its serial loop: a load among
    them that may read a tensor the final pass writes may read what an
    earlier iteration stored.  And the final pass alone applies the atomics
    whose first-pass copies ``eliminate_interchanged_atomics`` drops, after
    the whole first pass: a first-pass load that may read a tensor one
    updates, other than one in the lane loop's ``prefix`` (which runs before
    every serial iteration in the tile program too), reads the value the
    nest started from where the tile program reads what the atomics of the
    earlier serial iterations added (or of all of them, after the loop).
    Last, the final pass runs the lane-invariant statements of its serial
    loop's body once per iteration, in the loop's head, ahead of its inner
    lane loop: each statement of ``head_moves`` with the lane-varying ones
    it followed in the serial body, which a load or write of it that may
    reach what they write or (for a write) read would now precede, and so
    would a name it binds that they read or bind (``k = 0`` after a load of
    ``k * 4 + lane``; names one pending rename merges, ``rename_groups``,
    are one name).
    Addresses are resolved through the definitions reaching them
    (``_resolved_access_roots``), and one that names no tensor may access
    any.  Each raises ``BackendUnsupported``.
    """
    pairs = proven_disjoint_tensor_pairs
    tensor_names, unchecked = _nest_tensor_names(
        [first_nest, *final_nest], protected_names
    )
    if any(node is serial_loop for node in first_nest.body):
        first_flat, first_unstable = _first_nest_flat(
            first_nest, serial_loop, protected_names
        )
    else:
        first_flat = list(first_nest.body)
        first_unstable = [protected_names for _ in first_flat]
    first_writes = [
        roots
        for accesses in _resolved_access_roots(
            first_flat, first_unstable, tensor_names, writes=True, unchecked=unchecked
        )
        for roots in accesses
    ]
    final_flat, final_unstable = _final_nest_flat(final_nest, protected_names)
    final_reads = _resolved_access_roots(
        final_flat, final_unstable, tensor_names, writes=False, unchecked=unchecked
    )
    final_write_lists = _resolved_access_roots(
        final_flat, final_unstable, tensor_names, writes=True, unchecked=unchecked
    )
    final_writes = [roots for accesses in final_write_lists for roots in accesses]
    renames = rename_groups or {}

    def names(nodes: Iterable[str]) -> set[str]:
        return {renames.get(name, name) for name in nodes}

    final_index = {id(node): index for index, node in enumerate(final_flat)}
    for head, earlier in head_moves:
        assert id(head) in final_index
        assert all(id(node) in final_index for node in earlier)
        bound = names(ReadWrites.from_ast(head).writes)
        if any(
            bound & names([*rw.reads, *rw.writes])
            for rw in map(ReadWrites.from_ast, earlier)
        ):
            raise exc.BackendUnsupported(
                "cute",
                "an interchanged lane reduction would move a lane-invariant "
                "assignment ahead of a statement that uses the name it binds",
            )
        positions = [final_index[id(node)] for node in earlier]
        head_reads = final_reads[final_index[id(head)]]
        head_writes = final_write_lists[final_index[id(head)]]
        if _any_may_alias(
            head_reads,
            (roots for position in positions for roots in final_write_lists[position]),
            pairs,
        ) or _any_may_alias(
            head_writes,
            (
                roots
                for position in positions
                for accesses in (final_reads[position], final_write_lists[position])
                for roots in accesses
            ),
            pairs,
        ):
            raise exc.BackendUnsupported(
                "cute",
                "an interchanged lane reduction would move a lane-invariant access "
                "ahead of a write or read it followed",
            )
    rerun = {id(node) for node in per_iteration}
    assert rerun <= final_index.keys()
    if _any_may_alias(
        (
            roots
            for node, accesses in zip(final_flat, final_reads, strict=True)
            if id(node) in rerun
            for roots in accesses
        ),
        final_writes,
        pairs,
    ):
        raise exc.BackendUnsupported(
            "cute",
            "an interchanged lane reduction would reload a value of the lane "
            "loop's prefix after a store may have overwritten it",
        )
    if _any_may_alias(
        (roots for accesses in final_reads for roots in accesses), first_writes, pairs
    ):
        raise exc.BackendUnsupported(
            "cute",
            "a final-pass load of an interchanged lane reduction may read what the "
            "first pass stores",
        )
    first_reads = _resolved_access_roots(
        first_flat, first_unstable, tensor_names, writes=False, unchecked=unchecked
    )
    ahead = {id(node) for node in prefix}
    if _any_may_alias(
        (
            roots
            for node, accesses in zip(first_flat, first_reads, strict=True)
            if id(node) not in ahead
            for roots in accesses
        ),
        [
            _store_tensor_roots(call, tensor_names)
            for node in final_flat
            for call in atomic_write_calls(node)
        ],
        pairs,
    ):
        raise exc.BackendUnsupported(
            "cute",
            "a first-pass load of an interchanged lane reduction may read a tensor "
            "an atomic of the final pass updates",
        )


def eliminate_interchanged_atomics(
    lane_loop: ast.For,
    serial_loop: ast.For,
    final_nest: list[ast.AST],
    candidates: list[ast.AST],
    protected_names: set[str],
) -> int:
    """Drop the first pass's copies of atomics the final pass applies in full.

    The first pass keeps each reduction consumer over the marker's raw
    per-lane input.  A store there writes a partial value the final pass
    overwrites; an atomic read-modify-write would instead add every lane's
    partial on top of the full reduction the final pass applies once per
    serial iteration (a column sum over a lane-looped row tile added into
    ``out[column]`` came out exactly doubled), and an atomic both passes
    need without consuming the reduction would apply twice.  A candidate is
    dropped when its one memory write is an atomic the statement does
    nothing else around and the final pass holds exactly one clone of it;
    the number dropped is returned so the caller can decline a nest it could
    not make exact.  The first-pass bindings the final pass redefines and
    nothing left in the first pass reads are pruned afterwards, the serial
    loop goes when its body is empty and the lane loop's body may end up
    empty too, which the caller takes as the whole first pass being dead.
    """
    if (
        not candidates
        or lane_loop.orelse
        or serial_loop.orelse
        or not _matching_final_nest(lane_loop, serial_loop, final_nest)
    ):
        return 0
    final_flat = _final_flat(final_nest)
    final_texts = [ast.unparse(statement) for statement in final_flat]
    removed: list[ast.AST] = []
    for statement in candidates:
        if statement not in serial_loop.body:
            continue
        writes = _memory_write_calls(statement)
        if (
            len(writes) != 1
            or not _is_atomic_call(writes[0])
            or not _is_isolated_store_statement(statement, writes[0])
            or final_texts.count(ast.unparse(statement)) != 1
        ):
            continue
        removed.append(statement)
    if not removed:
        return 0
    serial_loop.body = [
        statement for statement in serial_loop.body if statement not in removed
    ]

    # Both nests run the lane loop's static extent times the same serial
    # bounds, which nothing in the first pass rebinds, so the final pass's
    # unconditional definitions kill the first pass's bindings
    # (``eliminate_interchanged_stores``).
    if _static_lane_loop_extent(lane_loop) is None or (
        _reads(serial_loop.iter) & set(ReadWrites.from_ast(lane_loop).writes)
    ):
        if not serial_loop.body:
            serial_loop.body = [ast.Pass()]
        return len(removed)
    redefined = _redefined_names(final_flat, protected_names)
    _prune_redefined_bindings(lane_loop, serial_loop, redefined)
    _drop_emptied_serial_loop(lane_loop, serial_loop, redefined)
    return len(removed)

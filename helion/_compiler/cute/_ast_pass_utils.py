"""Shared AST analysis and rewrite helpers for the CuTe rewrite passes.

Role: analysis (shared AST walkers; no codegen, no config reads).  The post-codegen
AST passes under ``helion/_compiler/cute`` used to carry byte-identical private
copies of these walkers; the canonical versions live here.  Semantics are
pinned by ``test/test_cute_ast_pass_utils.py`` against the legacy copies:

* name collectors -- ``names_read`` (Load context), ``names_written`` (Store
  context) and ``names_stored_or_deleted`` (Store or Del), all over
  ``ast.walk``; ``is_name_assign`` / ``assigned_name`` / ``name_assignment``
  for ``NAME = value``; ``resolve_alias`` follows a rename map to its root.
* call paths -- ``call_path`` (tuple) and ``call_path_str`` (dotted) of a
  ``Name``/``Attribute`` chain, ``callee_name`` (its last component);
  ``cutlass_scalar_type`` for ``cutlass.X``.
* expressions -- ``same_expr`` (structural equality without positions),
  ``parse_expr`` / ``reparse_expr`` (plain-``ast`` expression clones; use
  ``expr_from_string`` when Helion ``ExtendedAST`` metadata is required),
  ``flatten_addition`` (the addends of ``a + b + c``) and
  ``comparison_is_always_true`` (interval proof of one ``ast.Compare``).
* loops -- ``loop_extent`` with the one canonical literal-extent predicate and
  ``is_vloop`` for ``cutlass.range_constexpr(<int>)`` V-loops;
  ``constexpr_v_loop`` keeps the looser grammar of the warp-reduce hoist and
  the sibling V-loop merge; ``is_staged_loop`` / ``is_trace_time_test``
  tell the loops and tests the DSL preprocessor stages from the ones it
  runs as Python.
* memory -- ``tensor_roots_may_alias`` over the passes' tensor-root facts.
* rewrites -- ``replace_node`` (identity), ``replace_call`` (structural or
  textual match), ``substitute_names`` / ``rename_names`` / ``expand_lane``;
  every rewrite that inserts a subtree takes the caller's clone callable so
  pass-specific markers survive.  ``deep_clone`` copies a subtree with those
  markers and ``dealias_shared_nodes`` gives every position of a tree its own
  node.
* dataflow -- ``exposed_reads`` (forward, with explicit branch/loop/define
  policies) and ``live_in`` (backward, zero-trip loops).
* device body -- ``body_end_follows_all_tiles``: whether a statement appended
  to a device body runs once per CTA, after all of its tiles.

Do not import this module from device-helper modules that generated code
imports (``reduce_helpers``, ``vec_utils``, ``*_runtime``): they are part of
the disk-cache key.

The relocation-purity proofs at the bottom (``_is_proven_relocatable_call``
and friends) came from ``tile_strategy.py`` and are shared by the lane split
(``lane_loops/``), the chunk-recurrence hoist (``passes/``) and the memory /
epilogue passes.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Literal
from typing import TypeGuard
from typing import TypeVar
from typing import cast

from ..ast_extension import ExtendedAST
from ..ast_extension import expr_from_string
from ..ast_read_writes import ReadWrites
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .cache_policy_loads import MASKED_PACKET_LOAD_HELPER_NAMES
from .reduce_call import CTA_REDUCES

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from ..device_ir import DeviceIR

_A = TypeVar("_A", bound=ast.AST)


# --------------------------------------------------------------------------
# Names
# --------------------------------------------------------------------------


def names_read(node: ast.AST) -> set[str]:
    """Ids of every ``ast.Name`` in Load context anywhere under ``node``."""
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def names_written(node: ast.AST) -> set[str]:
    """Ids of every ``ast.Name`` in Store context anywhere under ``node``."""
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }


def names_stored_or_deleted(node: ast.AST) -> set[str]:
    """Ids of every ``ast.Name`` in Store or Del context under ``node``."""
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del))
    }


def is_name_assign(stmt: ast.AST) -> TypeGuard[ast.Assign]:
    """``stmt`` is ``NAME = value`` with a single ``ast.Name`` target."""
    return (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    )


def assigned_name(stmt: ast.AST) -> str | None:
    """If ``stmt`` is ``NAME = value`` with a single ``ast.Name`` target, its id."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def name_assignment(stmt: ast.AST) -> tuple[str, ast.expr] | None:
    """``(NAME, value)`` of ``NAME = value`` with a single ``ast.Name`` target."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id, stmt.value
    return None


def resolve_alias(name: str, aliases: Mapping[str, str]) -> str:
    """Follow ``aliases`` from ``name`` to its root; a cycle stops at the
    first repeated name."""
    seen: set[str] = set()
    while name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    return name


def _bound_names(tree: ast.AST) -> set[str]:
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".", 1)[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            names.add(node.name)
    return names


def _fresh_prefix(
    hint: str, occupied: Collection[str], fresh_name: Callable[[str], str]
) -> str:
    """Reserve a template prefix whose derived identifiers cannot capture names.

    ``fresh_name`` reserves a new identifier on each call. Templates derive
    locals by appending underscores and suffixes, so reserving only the prefix
    does not protect an existing argument or local such as ``prefix_value``.
    The caller supplies all original AST names and external boundaries, and
    registers the emitted identifiers before later passes allocate more names.
    """
    while True:
        prefix = fresh_name(hint)
        if not any(
            name == prefix or name.startswith(f"{prefix}_") for name in occupied
        ):
            return prefix


# --------------------------------------------------------------------------
# Call paths and expressions
# --------------------------------------------------------------------------


def call_path(node: ast.AST) -> tuple[str, ...] | None:
    """``a.b.c`` -> ``("a", "b", "c")``; None unless a Name/Attribute chain."""
    path: list[str] = []
    while isinstance(node, ast.Attribute):
        path.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    path.append(node.id)
    return tuple(reversed(path))


def call_path_str(node: ast.AST) -> str | None:
    """``a.b.c`` -> ``"a.b.c"``; None unless a Name/Attribute chain."""
    path = call_path(node)
    return None if path is None else ".".join(path)


def callee_name(node: ast.AST) -> str | None:
    """The last component of a ``Name`` / ``Attribute`` callee (``a().b``
    included), the way the DSL preprocessor matches loops and tests."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def cutlass_scalar_type(node: ast.AST) -> str | None:
    """``cutlass.Float32`` -> ``"Float32"``; None unless a two-part
    ``cutlass.<X>`` path (callers test membership in their own type sets)."""
    path = call_path(node)
    if path is None or len(path) != 2 or path[0] != "cutlass":
        return None
    return path[1]


def same_expr(left: ast.AST, right: ast.AST) -> bool:
    """Structural equality ignoring source positions."""
    return ast.dump(left, include_attributes=False) == ast.dump(
        right, include_attributes=False
    )


def parse_expr(source: str) -> ast.expr:
    """Parse one expression into plain ``ast`` nodes (no ``ExtendedAST``)."""
    return ast.parse(source, mode="eval").body


def reparse_expr(node: ast.AST) -> ast.expr:
    """Clone an expression through its source text (drops every marker)."""
    return parse_expr(ast.unparse(node))


def flatten_addition(node: ast.expr) -> tuple[ast.expr, ...]:
    """The addends of a nested ``a + b + c`` in source order (the original
    nodes, so identity tests against them still hold)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return (*flatten_addition(node.left), *flatten_addition(node.right))
    return (node,)


def comparison_is_always_true(
    node: ast.Compare, interval: Callable[[ast.expr], tuple[int, int] | None]
) -> bool:
    """Whether a single-operator comparison holds for every value of its
    operands' integer intervals (``interval`` returns ``(low, high)``
    inclusive, or None when an operand is not bounded)."""
    if len(node.ops) != 1 or len(node.comparators) != 1:
        return False
    left = interval(node.left)
    right = interval(node.comparators[0])
    if left is None or right is None:
        return False
    op = node.ops[0]
    if isinstance(op, ast.Lt):
        return left[1] < right[0]
    if isinstance(op, ast.LtE):
        return left[1] <= right[0]
    if isinstance(op, ast.Gt):
        return left[0] > right[1]
    if isinstance(op, ast.GtE):
        return left[0] >= right[1]
    if isinstance(op, ast.Eq):
        return left[0] == left[1] == right[0] == right[1]
    if isinstance(op, ast.NotEq):
        return left[1] < right[0] or right[1] < left[0]
    return False


# --------------------------------------------------------------------------
# Loops
# --------------------------------------------------------------------------

# The DSL preprocessor stages a ``for`` over a call of these (``range``,
# ``cutlass.range``, ``range_dynamic``, under any prefix) as a dynamic loop
# and unrolls every other ``for`` (``cutlass.range_constexpr``, a Python
# iterable) while tracing; an ``if`` / ``while`` whose test calls one of
# ``_TRACE_TIME_TESTS`` (``cutlass.const_expr(...)``) runs as Python.
_STAGED_RANGE_CALLS = frozenset({"range", "range_dynamic"})
_TRACE_TIME_TESTS = frozenset({"const_expr", "target_version"})
# The register helpers of the exact fp32 division (``exact_division.py``,
# ``device/ptx.py``), by the names generated code calls them: pure
# arithmetic.  A pass that meets them after the exact pass
# (``jam_grid_lanes``) copies them like any other arithmetic; the passes
# that emit them register each name here.
EXACT_DIVISION_HELPERS = frozenset(
    {
        "_cute_exact_div_bounds_f32",
        "_cute_exact_div_fast_f32",
        "_cute_exact_div_fast_f32x2",
        "_cute_exact_div_redo_f32",
    }
)


def is_staged_loop(node: ast.For) -> bool:
    """Whether the DSL stages ``node`` as a dynamic loop, its body traced
    once, rather than unrolling it while tracing (see above)."""
    return (
        isinstance(node.iter, ast.Call)
        and callee_name(node.iter.func) in _STAGED_RANGE_CALLS
    )


def is_trace_time_test(test: ast.expr) -> bool:
    """Whether the DSL runs an ``if`` / ``while`` on ``test`` as Python
    while tracing (``cutlass.const_expr(...)``) rather than staging it."""
    return isinstance(test, ast.Call) and callee_name(test.func) in _TRACE_TIME_TESTS


LoopKind = Literal["range", "constexpr"]
_LOOP_CALLEES: dict[str, tuple[str, ...]] = {
    "range": ("range",),
    "constexpr": ("cutlass", "range_constexpr"),
}


def loop_extent(
    node: ast.AST,
    *,
    kinds: Collection[LoopKind] = ("range", "constexpr"),
    minimum: int | None = 1,
) -> int | None:
    """Literal trip count of a generated lane or V loop, else None.

    The one canonical predicate: an ``ast.For`` with a plain ``Name`` target,
    no ``else`` clause, iterating ``range(<int>)`` (kind ``"range"``, a lane
    loop) or ``cutlass.range_constexpr(<int>)`` (kind ``"constexpr"``, a
    V-loop) with exactly one positional ``int`` literal (``bool`` excluded) and
    no keywords.  ``minimum`` rejects smaller extents (``None`` accepts any).
    The multi-argument and reversed ``range_constexpr(start, stop, step)``
    spellings are deliberately not matched; their parsers stay with the
    passes that emit them.
    """
    if not isinstance(node, ast.For) or node.orelse:
        return None
    if not isinstance(node.target, ast.Name):
        return None
    call = node.iter
    if not isinstance(call, ast.Call) or len(call.args) != 1 or call.keywords:
        return None
    path = call_path(call.func)
    if path is None or not any(path == _LOOP_CALLEES[kind] for kind in kinds):
        return None
    extent = call.args[0]
    if not isinstance(extent, ast.Constant) or type(extent.value) is not int:
        return None
    if minimum is not None and extent.value < minimum:
        return None
    return extent.value


def is_constexpr_range(loop: ast.AST) -> bool:
    """Whether ``loop`` is a ``for`` over ``cutlass.range_constexpr``, which
    the trace unrolls whatever the spelling of its bounds."""
    return (
        isinstance(loop, ast.For)
        and isinstance(loop.iter, ast.Call)
        and call_path(loop.iter.func) == _LOOP_CALLEES["constexpr"]
    )


def is_vloop(node: ast.AST) -> bool:
    """``for v in cutlass.range_constexpr(<int>)`` with a literal extent."""
    return loop_extent(node, kinds=("constexpr",), minimum=None) is not None


def constexpr_v_loop(node: ast.AST) -> tuple[ast.For, int] | None:
    """``for X in cutlass.range_constexpr(V, ...)`` with a literal ``int``
    first argument -> ``(loop, V)``, else None.

    This is the looser grammar the warp-reduce hoist and the sibling V-loop
    merge match (any target, ``else`` clauses, keywords, ``bool`` literals and
    the multi-argument spelling, where ``V`` is the start); ``loop_extent`` is
    the canonical predicate for everything else.  Pinned by
    ``test_cute_ast_pass_utils.LEGACY_LOOP_TABLE``.
    """
    if not isinstance(node, ast.For):
        return None
    call = node.iter
    if (
        not isinstance(call, ast.Call)
        or call_path(call.func) != ("cutlass", "range_constexpr")
        or not call.args
    ):
        return None
    extent = call.args[0]
    if not (isinstance(extent, ast.Constant) and isinstance(extent.value, int)):
        return None
    return node, extent.value


# --------------------------------------------------------------------------
# Device body
# --------------------------------------------------------------------------


def body_end_follows_all_tiles(body: Sequence[ast.AST], device_ir: DeviceIR) -> bool:
    """Whether every thread of every CTA reaches the end of the device
    ``body`` once, after all of the CTA's tiles: the body has no early
    ``return`` and the launch no grid barrier (``hl.barrier()`` phases; a
    stage of the ordered-phase lowering is a launch of its own, with one
    phase).  The device remainder fills (``host_remainder_fills.py``) and
    the flush of privatized atomics (``atomic_privatization.py``) are
    appended there."""
    return len(device_ir.phases) <= 1 and not any(
        isinstance(node, ast.Return)
        for statement in body
        for node in ast.walk(statement)
    )


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------


def tensor_roots_may_alias(
    left: frozenset[str] | None,
    right: frozenset[str] | None,
    proven_disjoint_tensor_pairs: Collection[frozenset[str]],
) -> bool:
    """Whether two accesses over the tensor roots ``left`` / ``right`` may
    touch the same memory: unknown roots (None) alias, and otherwise some
    pair of roots is the same tensor or is not proven disjoint."""
    if left is None or right is None:
        return True
    return any(
        left_name == right_name
        or frozenset((left_name, right_name)) not in proven_disjoint_tensor_pairs
        for left_name in left
        for right_name in right
    )


# --------------------------------------------------------------------------
# Rewrites
# --------------------------------------------------------------------------


# ``ExtendedAST.new`` carries these itself; every other non-field attribute
# of a node (the ``_helion_*`` markers passes leave on loops and statements)
# is copied by ``deep_clone``.
_EXTENDED_AST_INTERNALS = frozenset(
    {"_location", "_type_info", "_loop_type", "_is_kernel_call", "_root_id"}
)


def deep_clone(node: _A) -> _A:
    """A fresh copy of ``node``'s whole subtree that also keeps the marker
    attributes set on its nodes (``clone_ast`` keeps Helion's source metadata
    but not those; ``scalar_recipe._clone`` copies fields only), so the copy
    is indistinguishable from the original to every later pass."""

    def clone_field(value: object) -> object:
        if isinstance(value, ast.AST):
            return deep_clone(value)
        if isinstance(value, list):
            return [clone_field(item) for item in value]
        return value

    fields = {name: clone_field(value) for name, value in ast.iter_fields(node)}
    if isinstance(node, ExtendedAST):
        result = node.new(fields)
    else:
        result = ast.copy_location(type(node)(**fields), node)
    skip = set(node._fields) | set(node._attributes) | _EXTENDED_AST_INTERNALS
    for key, value in vars(node).items():
        if key not in skip and key not in vars(result):
            setattr(result, key, value)
    return cast("_A", result)


def dealias_shared_nodes(body: list[ast.stmt]) -> None:
    """Give every position in the tree its own node object, in place.

    The passes key analyses by node identity and rewrite nodes in place, so
    one statement object placed at two positions (say in two sweeps) would
    share one position's facts and take both positions' rewrites.  Every
    repeated node reached through a field or a body list is replaced by a
    ``deep_clone``, and a body list held by two nodes by its own list; the
    rendered text is unchanged.
    """
    seen: set[int] = set()
    seen_lists: set[int] = set()

    def visit(node: ast.AST) -> None:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                if id(value) in seen_lists:
                    value = list(value)
                    setattr(node, field, value)
                seen_lists.add(id(value))
                for index, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        value[index] = fresh(item)
            elif isinstance(value, ast.AST):
                setattr(node, field, fresh(value))

    def fresh(node: ast.AST) -> ast.AST:
        if id(node) in seen:
            # The clone's subtree is new to the tree; no need to walk it.
            return deep_clone(node)
        seen.add(id(node))
        visit(node)
        return node

    for index, stmt in enumerate(body):
        body[index] = cast("ast.stmt", fresh(stmt))


def statement_lists(
    statements: list[ast.stmt], *, outer_first: bool = False
) -> list[list[ast.stmt]]:
    """``statements`` and every statement list nested in it (the ``body`` and
    ``orelse`` of its statements, at any depth), each nested list before the
    list holding it, or after it with ``outer_first``."""
    found: list[list[ast.stmt]] = [statements] if outer_first else []
    for stmt in statements:
        for field in ("body", "orelse"):
            nested = getattr(stmt, field, None)
            if isinstance(nested, list):
                found.extend(statement_lists(nested, outer_first=outer_first))
    if not outer_first:
        found.append(statements)
    return found


class _ReplaceNode(ast.NodeTransformer):
    def __init__(self, target: ast.AST, replacement: ast.AST) -> None:
        self.target = target
        self.replacement = replacement

    def visit(self, node: ast.AST) -> ast.AST:
        if node is self.target:
            return self.replacement
        return super().visit(node)


def replace_node(tree: _A, target: ast.AST, replacement: ast.AST) -> _A:
    """Replace the node that *is* ``target`` (identity) with ``replacement``."""
    result = _ReplaceNode(target, replacement).visit(tree)
    assert isinstance(result, ast.AST)
    return result  # type: ignore[return-value]


class _ReplaceCall(ast.NodeTransformer):
    def __init__(
        self,
        call: ast.Call,
        replacement: Callable[[], ast.AST],
        *,
        match: Literal["dump", "text"],
        first_only: bool,
    ) -> None:
        self.match = match
        self.key = self._key(call)
        self.replacement = replacement
        self.first_only = first_only
        self.count = 0

    def _key(self, node: ast.AST) -> str:
        if self.match == "text":
            return ast.unparse(node)
        return ast.dump(node, include_attributes=False)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (not self.first_only or self.count == 0) and self._key(node) == self.key:
            self.count += 1
            return self.replacement()
        return self.generic_visit(node)


def replace_call(
    tree: ast.AST,
    call: ast.Call,
    replacement: Callable[[], ast.AST],
    *,
    match: Literal["dump", "text"] = "dump",
    first_only: bool = False,
) -> tuple[ast.AST, int]:
    """Replace every call equal to ``call`` (structurally, or by source text
    with ``match="text"``) by a fresh ``replacement()``; returns the rewritten
    tree and the number of replacements.  Matched calls are not descended."""
    transformer = _ReplaceCall(call, replacement, match=match, first_only=first_only)
    result = transformer.visit(tree)
    assert isinstance(result, ast.AST)
    return result, transformer.count


class _SubstituteNames(ast.NodeTransformer):
    def __init__(
        self,
        mapping: Mapping[str, ast.AST],
        clone: Callable[[ast.AST], ast.AST],
        *,
        loads_only: bool,
        keep_ctx: bool,
    ) -> None:
        self.mapping = mapping
        self.clone = clone
        self.loads_only = loads_only
        self.keep_ctx = keep_ctx

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.loads_only and not isinstance(node.ctx, ast.Load):
            return node
        value = self.mapping.get(node.id)
        if value is None:
            return node
        replacement = self.clone(value)
        if self.keep_ctx and isinstance(replacement, ast.Name):
            replacement.ctx = node.ctx
        return ast.copy_location(replacement, node)


def substitute_names(
    tree: _A,
    mapping: Mapping[str, ast.AST],
    *,
    clone: Callable[[ast.AST], ast.AST],
    loads_only: bool = True,
    keep_ctx: bool = False,
) -> _A:
    """Replace ``Name`` nodes by ``clone(mapping[id])``.

    ``loads_only`` leaves Store/Del names alone; ``keep_ctx`` makes a ``Name``
    replacement inherit the replaced node's context.  The clone callable is the
    caller's (``scalar_recipe._clone``, ``vector_reduction_packets._clone``,
    ``reparse_expr`` ...) so pass markers on the substituted subtree survive or
    are dropped exactly as before.
    """
    result = _SubstituteNames(
        mapping, clone, loads_only=loads_only, keep_ctx=keep_ctx
    ).visit(tree)
    assert isinstance(result, ast.AST)
    return result  # type: ignore[return-value]


class _RenameNames(ast.NodeVisitor):
    def __init__(self, renames: Mapping[str, str], *, loads_only: bool) -> None:
        self.renames = renames
        self.loads_only = loads_only

    def visit_Name(self, node: ast.Name) -> None:
        if self.loads_only and not isinstance(node.ctx, ast.Load):
            return
        node.id = self.renames.get(node.id, node.id)


def rename_names(
    tree: _A, renames: Mapping[str, str], *, loads_only: bool = False
) -> _A:
    """Rename ``Name`` ids in place (node identity and metadata preserved)."""
    _RenameNames(renames, loads_only=loads_only).visit(tree)
    return tree


def expand_lane(tree: _A, lane_var: str, unroll: int, copy_index: int) -> _A:
    """``lane`` -> ``lane * unroll + copy_index`` in Load contexts."""
    expression = f"{lane_var} * {unroll} + {copy_index}"
    return substitute_names(
        tree,
        {lane_var: expr_from_string(expression)},
        clone=lambda _node: expr_from_string(expression),
        loads_only=True,
    )


# --------------------------------------------------------------------------
# Dataflow
# --------------------------------------------------------------------------


def _rw_reads(node: ast.AST) -> Collection[str]:
    return ReadWrites.from_ast(node).reads


DefinesPolicy = Literal["assign", "writes"]
LoopsPolicy = Literal["opaque", "may_skip", "runs_once"]
BranchesPolicy = Literal["analyze", "opaque"]


def exposed_reads(
    statements: Sequence[ast.AST],
    *,
    defined: Collection[str] = (),
    renames: Mapping[str, str] | None = None,
    reads: Callable[[ast.AST], Collection[str]] = _rw_reads,
    defines: DefinesPolicy | Callable[[ast.AST], Collection[str]] = "assign",
    loops: LoopsPolicy = "may_skip",
    branches: BranchesPolicy = "analyze",
) -> frozenset[str]:
    """Canonical names ``statements`` may read before assigning them.

    ``reads`` collects a statement's reads (default ``ReadWrites.reads``) and
    ``renames`` canonicalizes every name.  What a statement *defines* is a
    policy: ``"assign"`` counts only the ``Name`` targets of a plain
    ``ast.Assign``, ``"writes"`` every ``ReadWrites.writes`` name (subscript
    and attribute stores excluded, augmented and nested writes included); a
    callable gives the caller's own predicate.  ``branches="analyze"`` visits
    both arms of an ``if`` and defines what both arms define;
    ``"opaque"`` treats the whole ``if`` as one statement.  Loops:
    ``"opaque"`` treats a loop as one statement that defines nothing;
    ``"may_skip"`` reads the header, visits body and ``else`` with the target
    defined, and defines nothing afterwards (the body may never run);
    ``"runs_once"`` visits the body the same way but afterwards defines what
    the loop defines (the body ran at least once).  A ``while`` is only
    visited under ``"may_skip"``.
    """
    renames = renames or {}

    def canonical(names: Iterable[str]) -> frozenset[str]:
        return frozenset(renames.get(name, name) for name in names)

    def defined_by(statement: ast.AST) -> frozenset[str]:
        if callable(defines):
            return canonical(defines(statement))
        if defines == "assign":
            if isinstance(statement, ast.Assign):
                return canonical(
                    target.id
                    for target in statement.targets
                    if isinstance(target, ast.Name)
                )
            return frozenset()
        return canonical(ReadWrites.from_ast(statement).writes)

    def defined_by_block(block: Sequence[ast.AST]) -> frozenset[str]:
        result: frozenset[str] = frozenset()
        for statement in block:
            result |= defined_by(statement)
        return result

    def walk(block: Sequence[ast.AST], defined: frozenset[str]) -> frozenset[str]:
        exposed: set[str] = set()
        for statement in block:
            if branches == "analyze" and isinstance(statement, ast.If):
                exposed |= canonical(reads(statement.test)) - defined
                exposed |= walk(statement.body, defined)
                exposed |= walk(statement.orelse, defined)
                defined |= defined_by_block(statement.body) & defined_by_block(
                    statement.orelse
                )
            elif loops != "opaque" and isinstance(statement, ast.For):
                exposed |= canonical(reads(statement.iter)) - defined
                body = (
                    [*statement.body, *statement.orelse]
                    if loops == "may_skip"
                    else statement.body
                )
                exposed |= walk(
                    body,
                    defined | canonical(ReadWrites.from_ast(statement.target).writes),
                )
                if loops == "runs_once":
                    defined |= defined_by(statement)
            elif loops == "may_skip" and isinstance(statement, ast.While):
                exposed |= canonical(reads(statement.test)) - defined
                exposed |= walk([*statement.body, *statement.orelse], defined)
            else:
                exposed |= canonical(reads(statement)) - defined
                defined |= defined_by(statement)
        return frozenset(exposed)

    return walk(statements, frozenset(defined))


def live_in(
    statements: Sequence[ast.AST],
    live_out: Collection[str],
    *,
    reads: Callable[[ast.AST], Collection[str]] = _rw_reads,
    assigned: Callable[[ast.AST], str | None] = assigned_name,
) -> set[str]:
    """Backward scalar liveness with zero-trip loops and branch dominance.

    A plain assignment (``assigned``) kills its target before adding its
    reads; an ``if`` is live-in for either arm plus its test; a ``for`` with a
    ``Name`` target runs its ``else`` after the last iteration (a ``break``
    that skips it keeps what is live after the loop live at the end of its
    body, ``_breaks_out``), iterates its body to a fixed point (a zero-trip
    loop keeps what is live after it live) and then reads its iterator;
    anything else only reads.
    """
    live = set(live_out)
    for statement in reversed(statements):
        name = assigned(statement)
        if name is not None:
            live.discard(name)
            live.update(reads(statement))
        elif isinstance(statement, ast.If):
            live = (
                live_in(statement.body, live, reads=reads, assigned=assigned)
                | live_in(statement.orelse, live, reads=reads, assigned=assigned)
                | set(reads(statement.test))
            )
        elif isinstance(statement, ast.For) and isinstance(statement.target, ast.Name):
            after_else = live_in(statement.orelse, live, reads=reads, assigned=assigned)
            live = live | after_else if _breaks_out(statement) else after_else
            live = loop_live_end(statement, live, reads=reads, assigned=assigned)
            live |= set(reads(statement.iter))
        else:
            live.update(reads(statement))
    return live


def _breaks_out(loop: ast.For) -> bool:
    """Whether a ``break`` of ``loop`` may skip its ``else``: one in its body
    outside the bodies of nested loops (a nested loop's ``else`` runs in
    ``loop``) and of nested functions."""

    def breaks(node: ast.AST) -> bool:
        if isinstance(node, ast.Break):
            return True
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            return any(breaks(child) for child in node.orelse)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return False
        return any(breaks(child) for child in ast.iter_child_nodes(node))

    return any(breaks(statement) for statement in loop.body)


def loop_live_end(
    loop: ast.For,
    live_after: Collection[str],
    *,
    reads: Callable[[ast.AST], Collection[str]] = _rw_reads,
    assigned: Callable[[ast.AST], str | None] = assigned_name,
) -> set[str]:
    """Names live at the end of ``loop``'s body: live after the loop (a
    zero-trip loop keeps them) or read by a later iteration before it writes
    them, the fixed point over the back edge (``live_in``)."""
    assert isinstance(loop.target, ast.Name)
    end = set(live_after)
    while True:
        head = live_in(loop.body, end, reads=reads, assigned=assigned)
        grown = end | (head - {loop.target.id})
        if grown == end:
            return end
        end = grown


# ---------------------------------------------------------------------------
# Relocation-purity proofs shared by the lane split, the chunk-recurrence hoist
# and the memory / epilogue passes: which generated calls and assignments may
# change lane-loop scope (run once per chunk instead of once per lane, move
# across a lane pass) without observable effect.
# ---------------------------------------------------------------------------


def _backward_slice(
    body: list[ast.AST],
    roots: set[str],
    rename_groups: Mapping[str, str] | None = None,
) -> tuple[list[int], set[str]]:
    """Return the indices of the statements in ``body`` that (transitively)
    produce any name in ``roots``, plus the set of all names those statements
    write (as spelled).  Statements are scanned in reverse so a producer is
    included once a later consumer (already selected) reads its output.

    Names are compared through ``rename_groups``, the device function's
    aliases of each carried value: a device loop of ``body`` that
    accumulates a value writes it under its loop-output name (``v_8 =
    row_sums_copy_0 + sum_1`` for ``row_sums``), which only the final rename
    pass folds into the accumulator, so a slice over the names as spelled
    would take the accumulator's initializer for its only producer and
    leave the loop that accumulates it out.
    """
    from ..ast_read_writes import ReadWrites

    renames = rename_groups or {}

    def canonical(names: Iterable[str]) -> set[str]:
        return {renames.get(name, name) for name in names}

    needed = canonical(roots)
    selected: list[int] = []
    written: set[str] = set()
    for idx in range(len(body) - 1, -1, -1):
        stmt = body[idx]
        rw = ReadWrites.from_ast(stmt)
        if canonical(rw.writes) & needed:
            selected.append(idx)
            written |= set(rw.writes)
            needed |= canonical(rw.reads)
    selected.reverse()
    return selected, written


# The synthetic-lane matmul fold (``fold_helpers.fold_dot``): a pure K loop
# over operand re-reads.  Duplicating it into every lane pass is correct but
# repeats the whole contraction per pass, so the staged splitter stashes its
# per-lane value like a collective's instead.
_FOLD_DOT_HELPER = "_cute_fold_dot"


_PURE_RELOCATABLE_NAMES = frozenset(
    {
        "_cute_python_mod",
        "abs",
        "bool",
        "cutlass.max",
        "cutlass.min",
        "float",
        "int",
        "len",
        "max",
        "min",
    }
)
_PURE_OPERATOR_CALLS = frozenset(
    {
        "add",
        "and_",
        "eq",
        "floordiv",
        "ge",
        "getitem",
        "gt",
        "invert",
        "le",
        "lshift",
        "lt",
        "mod",
        "mul",
        "ne",
        "neg",
        "not_",
        "or_",
        "pos",
        "pow",
        "rshift",
        "sub",
        "truediv",
        "xor",
    }
)
# The CTA-local scalar reduces the relocation proofs and the chunk hoist
# recognize (the lane-split and chunk-recurrence passes import this name).
_CHUNK_REDUCTION_HELPERS = CTA_REDUCES


def is_pure_operator_call(path: str) -> bool:
    """Whether the call path ``path`` (``call_path_str``) is an ``operator``
    function of ``_PURE_OPERATOR_CALLS``, one that computes a value from its
    operands and does nothing else."""
    return (
        path.startswith("operator.") and path.rsplit(".", 1)[-1] in _PURE_OPERATOR_CALLS
    )


def is_cta_reduce_call(call: ast.Call) -> bool:
    """Whether ``call`` combines a value across the threads of one CTA: a
    scalar reduce helper (``reduce_call.CTA_REDUCES``) or the DSL's warp
    reduction (``cute.arch.warp_reduction*``, emitted for a row reduced by
    at most 32 threads).  These are the reduces the relocation proofs accept
    with ``allow_reduction``."""
    name = call_path_str(call.func)
    return name is not None and (
        name in CTA_REDUCES or name.startswith("cute.arch.warp_reduction")
    )


def _is_proven_relocatable_call(
    call: ast.Call,
    *,
    allow_load: bool,
    allow_reduction: bool = False,
) -> bool:
    """Whether evaluating ``call`` is known not to have observable effects."""
    # ``scalar_recipe`` is a user of this utility layer (it imports
    # ``names_read`` at module level), and the backend registry imports this
    # module through the CuTe backend, so bind these at call time.
    from ..compile_environment import CompileEnvironment
    from .scalar_recipe import PURE_HELPERS
    from .scalar_recipe import is_rounded_multiply

    if isinstance(call.func, ast.Attribute):
        if call.func.attr == "load":
            return allow_load
        if call.func.attr == "bitcast":
            return True
    if is_rounded_multiply(call):
        # ``mul.rn.{f32,f16,bf16}`` over two registers: a product kept
        # separately rounded (``scalar_recipe_rounding``), as pure as the
        # ``*`` it replaces.
        return True
    name = call_path_str(call.func)
    if name is None:
        return False
    if name == "_cute_inline_asm_elementwise":
        from .pure_lane_packets import is_pure_inline_asm_call

        # The packet pass's scalar-arithmetic proof of the asm text, packed
        # or not: declared purity alone does not exclude hidden memory
        # accesses or synchronization.
        return is_pure_inline_asm_call(call)
    if name in _PURE_RELOCATABLE_NAMES or name in PURE_HELPERS:
        return True
    if name.startswith("cutlass."):
        member = name.rsplit(".", 1)[-1]
        # Type conversions.
        return bool(member) and member[0].isupper()
    if name.startswith(("math.", "cute.math.")):
        return True
    if name == "ir.VectorType.get":
        # The MLIR vector type of a hoisted packet load: a pure type value.
        return True
    if name.startswith("operator."):
        return is_pure_operator_call(name)
    if name in {
        "cute.arch.block_dim",
        "cute.arch.block_idx",
        "cute.arch.fmax",
        "cute.arch.fmin",
        "cute.arch.grid_dim",
        "cute.arch.lane_idx",
        "cute.arch.thread_idx",
        "cute.arch.warp_idx",
    }:
        return True
    if (
        name == "cute.arch.load"
        or name in _CUTE_CACHE_LOAD_HELPER_NAMES
        or name in MASKED_PACKET_LOAD_HELPER_NAMES
    ):
        # The packet load, its L2-policy helper forms and the predicated
        # packet loads of the V-loop sink, the load-side twin of the
        # ``_cute_store_*`` convention ``_is_store_call`` relies on.
        # Only the listed helpers: the memory passes collect and measure
        # exactly these, so any other ``_cute_load_*`` stays unproven.
        return allow_load
    if name == _FOLD_DOT_HELPER:
        # The synthetic-lane matmul fold: pure arithmetic over reads of
        # operand tensors the kernel never writes (``_cute_matmul_reread_unsafe``
        # proved that before the fold was emitted), so it relocates exactly
        # like a load of the same tensors.
        return allow_load
    if (
        CompileEnvironment.has_current()
        and name in CompileEnvironment.current().backend.pure_device_calls
    ):
        # Another backend's pure device calls (Metal's casts); the AST passes
        # also run outside a compile, in their unit tests.
        return True
    return allow_reduction and is_cta_reduce_call(call)


def _is_proven_relocatable_assignment(
    stmt: ast.AST,
    *,
    allow_load: bool,
    allow_reduction: bool = False,
) -> bool:
    """Whether a simple assignment may safely change lane-loop scope."""
    if assigned_name(stmt) is None:
        return False
    return all(
        _is_proven_relocatable_call(
            node,
            allow_load=allow_load,
            allow_reduction=allow_reduction,
        )
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
    )


def _self_addend(stmt: ast.AST, target: str) -> ast.AST | None:
    """Return the non-self operand of ``target = target + value``."""
    if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.BinOp):
        return None
    if not isinstance(stmt.value.op, ast.Add):
        return None
    if isinstance(stmt.value.left, ast.Name) and stmt.value.left.id == target:
        return stmt.value.right
    if isinstance(stmt.value.right, ast.Name) and stmt.value.right.id == target:
        return stmt.value.left
    return None


def _canonical_name(name: str, rename_groups: dict[str, str]) -> str:
    return rename_groups.get(name, name)


def merged_names(rename_groups: Mapping[str, str]) -> dict[str, frozenset[str]]:
    """Each name ``rename_groups`` (``ast_rename``'s map to a shared name)
    merges with others, with all of them: once renamed they are one
    variable, so a write to any is a write to all."""
    groups: dict[str, set[str]] = {}
    for name, canonical in rename_groups.items():
        groups.setdefault(canonical, {canonical}).add(name)
    return {name: frozenset(group) for group in groups.values() for name in group}


def with_merged_names(
    names: Iterable[str], merged: Mapping[str, frozenset[str]]
) -> set[str]:
    """``names`` and every name ``merged`` (from ``merged_names``) merges
    with one of them."""
    return {alias for name in names for alias in merged.get(name, (name,))}


def _dependency_names(body: list[ast.AST], roots: set[str]) -> set[str]:
    """Return every name in the backward slice producing ``roots``."""
    from ..ast_read_writes import ReadWrites

    indices, _ = _backward_slice(body, roots)
    names = set(roots)
    for idx in indices:
        rw = ReadWrites.from_ast(body[idx])
        names.update(rw.reads)
        names.update(rw.writes)
    return names

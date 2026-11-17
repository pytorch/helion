"""Late branch-local vectorization for exact persistent-reduction fragments.

Persistent reduction lanes wrap the whole grid body.  A load whose address is
defined inside a runtime branch therefore cannot use the normal wrapper-level
vector hoist.  Memory lowering marks only sites that already passed the
stride-1, dtype, and exact-fragment checks; this pass places their vector
transaction immediately outside the materialized constexpr lane loop, where
branch-local indices dominate it.
"""

from __future__ import annotations

import ast
from collections import Counter
import dataclasses
from functools import cache
from functools import partial
from itertools import pairwise
import math
import operator
import re
from typing import TYPE_CHECKING
from typing import cast

from ..ast_read_writes import ReadWrites
from ._ast_pass_utils import loop_extent
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_written
from ._ast_pass_utils import reparse_expr as _clone_expr
from ._ast_pass_utils import replace_call
from .cache_policy_loads import _CUTE_CACHE_LOAD_HELPER_NAMES
from .cache_policy_loads import CachePolicy
from .packet_access import PERSISTENT_BRANCH_VEC_LOAD
from .packet_access import PERSISTENT_BRANCH_VEC_STORE
from .packet_access import call_write_roots
from .packet_access import fragment_writes
from .packet_access import match_packet_access
from .packet_access import render_packet_load

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

_LOAD_MARKER = PERSISTENT_BRANCH_VEC_LOAD
_STORE_MARKER = PERSISTENT_BRANCH_VEC_STORE
_SYNTHETIC_LANE_RE = re.compile(r"synthetic_lane_(\d+)")
_DTYPE_INFO = {
    "cutlass.BFloat16": ("cutlass.Uint16", 2, "_cute_store_u16_vec"),
    "cutlass.Float16": ("cutlass.Uint16", 2, "_cute_store_u16_vec"),
    "cutlass.Float32": ("cutlass.Uint32", 4, "_cute_store_u32_vec"),
    "cutlass.Int32": ("cutlass.Uint32", 4, "_cute_store_u32_vec"),
    "cutlass.Uint32": ("cutlass.Uint32", 4, "_cute_store_u32_vec"),
}
# FP8 stores collect unconverted lanes, which their flush helper converts as
# one packet; bool stores collect converted ``cutlass.Uint8`` lanes, which
# ``_cute_store_u8_vec`` only packs.  Loads of these dtypes keep their scalar
# path.
_STORE_DTYPE_INFO = {
    **_DTYPE_INFO,
    "cutlass.Float8E4M3FN": ("cutlass.Uint8", 1, "_cute_store_fp8e4m3fn_vec"),
    "cutlass.Float8E5M2": ("cutlass.Uint8", 1, "_cute_store_fp8e5m2_vec"),
    "cutlass.Boolean": ("cutlass.Uint8", 1, "_cute_store_u8_vec"),
}
# Element bytes of each generated ``NAME(pointer, values)`` packet flush; the
# lane passes treat exactly these helpers as one store of ``values``.
PACKET_STORE_BYTES = {
    helper: itemsize for _carrier, itemsize, helper in _STORE_DTYPE_INFO.values()
}


def _marker_call(node: ast.AST, name: str, nargs: int) -> ast.Call | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and len(node.args) == nargs
        and not node.keywords
    ):
        return node
    return None


def _literal(node: ast.AST, typ: type[int | str]) -> int | str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, typ):
        return node.value
    return None


def _range_extent(loop: ast.For) -> int | None:
    return loop_extent(loop, kinds=("constexpr",), minimum=2)


def _single_name_definitions(statements: list[ast.stmt]) -> dict[str, ast.expr]:
    definitions: dict[str, ast.expr] = {}
    for stmt in statements:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            definitions[stmt.targets[0].id] = stmt.value
    return definitions


def _binding_write_roots(node: ast.AST) -> set[str]:
    """Return names whose bindings or reachable values may be mutated."""

    def root_name(target: ast.AST) -> str | None:
        while isinstance(target, (ast.Attribute, ast.Subscript)):
            target = target.value
        return target.id if isinstance(target, ast.Name) else None

    result: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            result.add(child.id)
        elif isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(
            child.ctx, (ast.Store, ast.Del)
        ):
            if (name := root_name(child)) is not None:
                result.add(name)
        elif isinstance(child, ast.ExceptHandler) and child.name is not None:
            result.add(child.name)
    return result


def _mutated_value_roots(node: ast.AST) -> set[str]:
    """Return base names mutated through an attribute or subscript."""

    def root_name(target: ast.AST) -> str | None:
        while isinstance(target, (ast.Attribute, ast.Subscript)):
            target = target.value
        return target.id if isinstance(target, ast.Name) else None

    return {
        name
        for child in ast.walk(node)
        if isinstance(child, (ast.Attribute, ast.Subscript))
        and isinstance(child.ctx, (ast.Store, ast.Del))
        and (name := root_name(child)) is not None
    }


def _freeze_definition(
    node: ast.expr,
    definitions: dict[str, ast.expr],
) -> ast.expr | None:
    """Inline known values so a definition keeps assignment-time semantics."""

    class _TooLarge(Exception):
        pass

    class _FreezeNames(ast.NodeTransformer):
        def __init__(self) -> None:
            super().__init__()
            self.expanding: set[str] = set()
            self.remaining = 4096

        def visit(self, node: ast.AST) -> ast.AST | list[ast.stmt] | None:
            # Repeated scalar arithmetic can encode an exponentially large
            # expression DAG. Callers must decline the entire proof when this
            # budget is exceeded: omitting a definition could hide its lane
            # dependence and make an aliased address appear injective.
            self.remaining -= 1
            if self.remaining < 0:
                raise _TooLarge
            return super().visit(node)

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if not isinstance(node.ctx, ast.Load) or node.id in self.expanding:
                return node
            definition = definitions.get(node.id)
            if definition is None:
                return node
            self.expanding.add(node.id)
            replacement = self.visit(_clone_expr(definition))
            self.expanding.remove(node.id)
            assert isinstance(replacement, ast.expr)
            return ast.copy_location(replacement, node)

    try:
        result = _FreezeNames().visit(_clone_expr(node))
    except _TooLarge:
        return None
    assert isinstance(result, ast.expr)
    return result


def _needed_definition_names(
    statements: list[ast.stmt], required_names: set[str]
) -> set[str]:
    """Close the requested names over all possible assignment dependencies."""
    dependencies: dict[str, set[str]] = {}
    needed = set(required_names)
    for statement in statements:
        # Retain aliases used by structured mutations even when the alias is
        # absent from the requested expressions. Their referents are needed by
        # the normal invalidation below, e.g. alias = source; alias[0] = value.
        needed.update(_mutated_value_roots(statement))
        for name, value in _single_name_definitions([statement]).items():
            dependencies.setdefault(name, set()).update(
                ReadWrites.from_ast(value).reads
            )
    pending = list(needed)
    while pending:
        for dependency in dependencies.get(pending.pop(), ()):
            if dependency not in needed:
                needed.add(dependency)
                pending.append(dependency)
    return needed


def _definition_snapshots(
    statements: list[ast.stmt], *, required_names: set[str] | None = None
) -> list[dict[str, ast.expr]] | None:
    """Return the definitions dominating each top-level statement.

    A dependency-closed subset avoids expanding irrelevant scalar arithmetic.
    Keep every write invalidation, including mutations through aliases, so the
    selected definitions match the corresponding full snapshots. An oversized
    definition in the selected dependency closure still declines the proof.
    """
    if required_names is not None:
        required_names = _needed_definition_names(statements, required_names)
    definitions: dict[str, ast.expr] = {}
    definition_reads: dict[str, set[str]] = {}
    result: list[dict[str, ast.expr]] = []
    for statement in statements:
        result.append(dict(definitions))
        exact_definitions = _single_name_definitions([statement])
        # Definitions retain source expressions rather than assignment-time
        # values.  Invalidate every definition that transitively depends on a
        # written name before recording the new exact definitions.  Otherwise
        # a later reassignment could make two historically different address
        # aliases expand to the same current expression.  Unmodelled writes
        # (AugAssign, structured assignment, and control flow) also kill the
        # name itself because they have no unconditional post-value to inline.
        invalidated = {
            *ReadWrites.from_ast(statement).writes,
            *_binding_write_roots(statement),
        }
        # A structured write through ``alias`` also mutates the object named
        # by ``alias = original``.  Frozen definitions retain that referent as
        # a read, so invalidate it (and all definitions depending on it) too.
        # This is deliberately conservative for non-name RHS expressions.
        for name in _mutated_value_roots(statement):
            definition = definitions.get(name)
            if definition is not None:
                invalidated.update(definition_reads[name])
        while invalidated:
            newly_invalidated: set[str] = set()
            for name in tuple(definitions):
                if name in invalidated or (definition_reads[name] & invalidated):
                    definitions.pop(name)
                    definition_reads.pop(name)
                    newly_invalidated.add(name)
            invalidated = newly_invalidated
        for name, value in exact_definitions.items():
            if required_names is not None and name not in required_names:
                continue
            frozen = _freeze_definition(value, definitions)
            if frozen is None:
                return None
            definitions[name] = frozen
            definition_reads[name] = set(ReadWrites.from_ast(frozen).reads)
    return result


def _unstable_address_names(
    body: list[ast.stmt], lane_var: str, load_index: int, store_index: int
) -> set[str]:
    """Names whose value may differ across a load/store loop backedge."""
    definitely_written: set[str] = set()
    live_in: set[str] = set()
    may_writes: set[str] = set()
    for statement in body:
        effects = ReadWrites.from_ast(statement)
        live_in.update(set(effects.reads) - definitely_written - {lane_var})
        may_writes.update(effects.writes)
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            definitely_written.update(effects.writes)
    result = live_in & may_writes
    if store_index > load_index:
        for crossed_statement in body[load_index + 1 : store_index + 1]:
            result.update(ReadWrites.from_ast(crossed_statement).writes)
    return result


class _CollectMemoryLoads(ast.NodeVisitor):
    """Collect logical loads without double-counting a marker's scalar fallback."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        if _marker_call(node, _LOAD_MARKER, 6) is not None:
            self.calls.append(node)
            return
        if isinstance(node.func, ast.Attribute) and node.func.attr == "load":
            self.calls.append(node)
            return
        if isinstance(node.func, ast.Name) and node.func.id in _VECTOR_LOAD_HELPERS:
            self.calls.append(node)
            return
        self.generic_visit(node)


def _memory_load_calls(node: ast.AST) -> list[ast.Call]:
    collector = _CollectMemoryLoads()
    collector.visit(node)
    return collector.calls


def _lane_dependent_alias_expansion(
    node: ast.expr,
    definitions: dict[str, ast.expr],
    lane_var: str,
) -> ast.expr:
    dependency_cache: dict[str, bool] = {}
    resolving: set[str] = set()

    def depends_on_lane(name: str) -> bool:
        if name == lane_var:
            return True
        cached = dependency_cache.get(name)
        if cached is not None:
            return cached
        definition = definitions.get(name)
        if definition is None or name in resolving:
            dependency_cache[name] = False
            return False
        resolving.add(name)
        result = any(
            depends_on_lane(read) for read in ReadWrites.from_ast(definition).reads
        )
        resolving.remove(name)
        dependency_cache[name] = result
        return result

    class _ExpandLaneAliases(ast.NodeTransformer):
        def __init__(self) -> None:
            super().__init__()
            self.expanding: set[str] = set()

        def visit_Name(self, node: ast.Name) -> ast.AST:
            if (
                not isinstance(node.ctx, ast.Load)
                or node.id == lane_var
                or not depends_on_lane(node.id)
                or node.id in self.expanding
            ):
                return node
            definition = definitions.get(node.id)
            if definition is None:
                return node
            self.expanding.add(node.id)
            replacement = self.visit(_clone_expr(definition))
            self.expanding.remove(node.id)
            return ast.copy_location(replacement, node)

    expanded = _ExpandLaneAliases().visit(_clone_expr(node))
    assert isinstance(expanded, ast.expr)
    return expanded


def _is_affine_in_lane(node: ast.AST, lane_var: str) -> tuple[bool, bool]:
    """Return ``(depends_on_lane, is_affine)`` for pointer arithmetic."""
    if isinstance(node, ast.Name):
        return node.id == lane_var, True
    if isinstance(node, ast.UnaryOp):
        depends, valid = _is_affine_in_lane(node.operand, lane_var)
        return depends, valid and isinstance(node.op, (ast.UAdd, ast.USub))
    if isinstance(node, ast.BinOp):
        left_depends, left_valid = _is_affine_in_lane(node.left, lane_var)
        right_depends, right_valid = _is_affine_in_lane(node.right, lane_var)
        depends = left_depends or right_depends
        if not depends:
            return False, left_valid and right_valid
        if not left_valid or not right_valid or left_depends and right_depends:
            return True, False
        if isinstance(node.op, ast.Add):
            return True, True
        if isinstance(node.op, ast.Sub):
            return True, left_depends
        if isinstance(node.op, ast.Mult):
            return True, True
        return True, False
    if isinstance(node, ast.Call):
        # The callee counts too: ``cutlass.Int32(v[lane]).bitcast(...)``
        # depends on the lane through its receiver.
        func_depends, func_valid = _is_affine_in_lane(node.func, lane_var)
        dependencies = [
            _is_affine_in_lane(arg, lane_var)
            for arg in [*node.args, *(keyword.value for keyword in node.keywords)]
        ]
        depends = func_depends or any(item[0] for item in dependencies)
        if not depends:
            return False, func_valid and all(item[1] for item in dependencies)
        is_cutlass_cast = (
            not func_depends
            and len(node.args) == 1
            and not node.keywords
            and ast.unparse(node.func).startswith("cutlass.")
        )
        return True, is_cutlass_cast and all(item[1] for item in dependencies)
    dependencies = [
        _is_affine_in_lane(child, lane_var) for child in ast.iter_child_nodes(node)
    ]
    depends = any(item[0] for item in dependencies)
    return depends, not depends and all(item[1] for item in dependencies)


def _plain_scalar_load_pointer(call: ast.Call) -> ast.expr | None:
    """Return the pointer of a generated scalar ``PTR.load()`` call."""
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "load"
        and not call.args
        and not call.keywords
    ):
        return call.func.value
    return None


def _plain_scalar_store_pointer(call: ast.Call) -> ast.expr | None:
    """Return the pointer of a generated scalar ``PTR.store(value)`` call."""
    if (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "store"
        and len(call.args) == 1
        and not call.keywords
    ):
        return call.func.value
    return None


def _memory_definition_names(statements: list[ast.stmt]) -> set[str]:
    """Select addresses and their predicates, never unrelated value arithmetic."""
    names: set[str] = set()

    def visit(node: ast.AST, guards: tuple[ast.expr, ...]) -> None:
        if isinstance(node, (ast.If, ast.IfExp)):
            visit(node.test, guards)
            branches = (
                [*node.body, *node.orelse]
                if isinstance(node, ast.If)
                else [node.body, node.orelse]
            )
            for child in branches:
                visit(child, (*guards, node.test))
            return
        if isinstance(node, ast.For):
            names.update(names_read(node.iter))
        if isinstance(node, ast.Call):
            pointer = _plain_scalar_load_pointer(node)
            if pointer is None:
                pointer = _plain_scalar_store_pointer(node)
            if pointer is not None:
                names.update(names_read(pointer))
                for guard in guards:
                    names.update(names_read(guard))
        for child in ast.iter_child_nodes(node):
            visit(child, guards)

    for statement in statements:
        visit(statement, ())
    # A nested address can refer to an enclosing definition through its local
    # aliases. Close over every definition version before selecting each
    # scope's snapshots; their ordinary write invalidation remains unchanged.
    return _needed_definition_names(
        [
            node
            for statement in statements
            for node in ast.walk(statement)
            if isinstance(node, ast.stmt)
        ],
        names,
    )


def _memory_definition_snapshots(
    statements: list[ast.stmt],
) -> list[dict[str, ast.expr]] | None:
    """Freeze only the definitions needed by scalar addresses and their guards."""
    return _definition_snapshots(
        statements, required_names=_memory_definition_names(statements)
    )


# Widest element count one 16-byte flush helper can write.
_VECTOR_FLUSH_MAX_WIDTH = {
    **{helper: 16 // itemsize for helper, itemsize in PACKET_STORE_BYTES.items()},
    "_cute_store_u16x8_l2_evict_last": 8,
    "_cute_store_u32x4_l2_evict_last": 4,
    "_cute_store_u16x8_streaming": 8,
    "_cute_store_u32x4_streaming": 4,
}
# The generated vector loads besides ``cute.arch.load``: the L2-policy helpers
# (16- and 8-byte forms), called the same way
# (``NAME(PTR, ir.VectorType.get([V], ...))``).
_VECTOR_LOAD_HELPERS = _CUTE_CACHE_LOAD_HELPER_NAMES
# Word carriers of byte-packed loads (several sub-word elements per word).  The
# element count depends on the tensor dtype, which the call does not show.
_PACKED_LOAD_CARRIERS = frozenset(
    {"cutlass.Uint16", "cutlass.Uint32", "cutlass.Uint64"}
)


def _generated_access_pointer(call: ast.Call) -> tuple[ast.expr, int] | None:
    """Pointer and element width of a generated scalar or vector access.

    Recognizes ``PTR.load()`` / ``PTR.store(value)`` (one element), the
    hoisted ``cute.arch.load(PTR, ir.VectorType.get([V], ...))`` packet load
    and its L2-policy helper forms (``V`` elements), a scalar-typed
    ``cute.arch.load(PTR, cutlass.Float32, ...)`` (one element) and the
    packet flush (``PACKET_STORE_BYTES``) of a V-loop's values.  The
    flush's element count is not visible in the call, so it
    reports the widest fragment its helper can write.  Anything else returns
    ``None``: atomics, unknown helpers, the predicated packet loads of the
    V-loop sink (which runs later), and byte-packed loads through a word
    carrier (``cute.arch.load(PTR, cutlass.Uint64)`` reads as many elements
    as fit the word, a count the call does not show).
    """
    access = match_packet_access(call)
    if access is None:
        return None
    if access.kind in ("load", "store"):
        return access.pointer, 1
    if access.kind in ("packet", "hinted_packet"):
        return None if access.width is None else (access.pointer, access.width)
    if access.kind == "arch_load":
        if access.carrier in _PACKED_LOAD_CARRIERS:
            return None
        return access.pointer, 1
    if access.kind == "flush" and access.helper in _VECTOR_FLUSH_MAX_WIDTH:
        return access.pointer, _VECTOR_FLUSH_MAX_WIDTH[access.helper]
    return None


def _memory_address_definition_snapshots(
    statements: list[ast.stmt],
) -> list[dict[str, ast.expr]] | None:
    """Snapshot canonical addresses used by the lane-independence proofs."""
    required_names: set[str] = set()
    for statement in statements:
        for call in ast.walk(statement):
            if not isinstance(call, ast.Call):
                continue
            if (load := _marker_call(call, _LOAD_MARKER, 6)) is not None:
                pointer = load.args[4]
            elif (store := _marker_call(call, _STORE_MARKER, 6)) is not None:
                pointer = store.args[3]
            else:
                pointer = _plain_scalar_load_pointer(call)
                if pointer is None:
                    pointer = _plain_scalar_store_pointer(call)
            if pointer is not None:
                required_names.update(ReadWrites.from_ast(pointer).reads)
    return _definition_snapshots(statements, required_names=required_names)


def _single_tensor_iterator_root(node: ast.AST) -> str | None:
    roots = {
        child.value.id
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and child.attr == "iterator"
        and isinstance(child.value, ast.Name)
    }
    return next(iter(roots)) if len(roots) == 1 else None


def _tensor_stride_reference(node: ast.AST) -> tuple[str, int] | None:
    while (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) in ("cutlass.Int32", "cutlass.Int64")
    ):
        node = node.args[0]
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "stride"
        and isinstance(node.value.value, ast.Attribute)
        and node.value.value.attr == "layout"
        and isinstance(node.value.value.value, ast.Name)
    ):
        return None
    return node.value.value.value.id, node.slice.value


def _lane_scale_value(
    node: ast.AST,
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
) -> int | None:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return node.value
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _lane_scale_value(node.operand, proven_tensor_stride_values)
        if value is None:
            return None
        return value if isinstance(node.op, ast.UAdd) else -value
    if (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) in ("cutlass.Int32", "cutlass.Int64")
    ):
        return _lane_scale_value(node.args[0], proven_tensor_stride_values)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult)
    ):
        left = _lane_scale_value(node.left, proven_tensor_stride_values)
        right = _lane_scale_value(node.right, proven_tensor_stride_values)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    stride = _tensor_stride_reference(node)
    return proven_tensor_stride_values.get(stride) if stride is not None else None


def _scalar_pointer_calls_are_known(node: ast.AST) -> bool:
    """Whether every call has transparent integer-coordinate semantics."""
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = ast.unparse(child.func)
        if name in ("cutlass.Int32", "cutlass.Int64"):
            if len(child.args) != 1 or child.keywords:
                return False
            continue
        if name in (
            "cute.arch.block_idx",
            "cute.arch.thread_idx",
            "cute.arch.warp_idx",
        ):
            if child.args or child.keywords:
                return False
            continue
        return False
    return True


def _lane_step_separates(step: int, modulus: int, width: int) -> bool:
    """Whether a lane offset of ``step`` (mod ``modulus``) cannot overlap a
    ``width``-element access at the other lane."""
    remainder = step % modulus
    return min(remainder, modulus - remainder) >= width


def _exact_divisor(
    node: ast.AST, proven_tensor_stride_values: Mapping[tuple[str, int], int]
) -> int | None:
    """The value of a divisor whose runtime value is exactly known: a literal
    or a cache-safe exact tensor stride (possibly cast), in [1, 2**31), where
    Int32 arithmetic cannot have wrapped it."""
    while (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and not node.keywords
        and ast.unparse(node.func) in ("cutlass.Int32", "cutlass.Int64")
    ):
        node = node.args[0]
    if isinstance(node, ast.Constant):
        value = node.value
        if not isinstance(value, int) or isinstance(value, bool):
            return None
    else:
        stride = _tensor_stride_reference(node)
        value = proven_tensor_stride_values.get(stride) if stride is not None else None
    return value if value is not None and 1 <= value < 1 << 31 else None


def _lane_offsets(
    node: ast.AST,
    lane_var: str,
    lanes: Sequence[int],
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
) -> list[int] | None:
    """The offset ``node`` takes at each of ``lanes`` (the lane variable's
    values), up to a shift all lanes share, or None.

    ``node`` must be integer arithmetic of the lane (``+``, ``-``, ``*``,
    ``//``, ``%`` and ``cutlass.Int32`` / ``Int64`` casts) whose
    lane-independent terms are known (literals, cache-safe exact tensor
    strides) wherever they scale, divide or reduce a lane-dependent one; an
    unknown term may only be added, which shifts every lane alike.  A
    division or remainder needs the lane-dependent operand's own values, not
    shifted, non-negative and below 2**31, computed without leaving the Int32
    range on the way (the Int32 arithmetic agrees with them only modulo
    2**32), and an exactly known divisor (``_exact_divisor``).  That covers
    the chunked persistent lanes' ``lane // V * threads * V + lane % V``,
    which is not affine in the lane.
    """
    lanes = list(lanes)
    if not lanes:
        return []

    def depends(expr: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Name) and child.id == lane_var
            for child in ast.walk(expr)
        )

    # ``(values, exact)``: per-lane values, exact unless shifted by an
    # unknown addend; ``(None, value)`` for a lane-independent term, ``value``
    # None when it is not known.
    def visit(expr: ast.AST) -> tuple[list[int] | None, object] | None:
        if isinstance(expr, ast.Name) and expr.id == lane_var:
            return lanes, True
        if not depends(expr):
            return None, _lane_scale_value(expr, proven_tensor_stride_values)
        if isinstance(expr, ast.Call):
            if (
                len(expr.args) == 1
                and not expr.keywords
                and ast.unparse(expr.func) in ("cutlass.Int32", "cutlass.Int64")
            ):
                return visit_in_range(expr.args[0])
            return None
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            inner = visit_in_range(expr.operand)
            if inner is None or (values := inner[0]) is None:
                return None
            if isinstance(expr.op, ast.USub):
                values = [-v for v in values]
            return values, inner[1]
        if not isinstance(expr, ast.BinOp):
            return None
        left, right = visit_in_range(expr.left), visit_in_range(expr.right)
        if left is None or right is None:
            return None
        if isinstance(expr.op, (ast.Add, ast.Sub)):
            sign = 1 if isinstance(expr.op, ast.Add) else -1
            if left[0] is not None and right[0] is not None:
                return (
                    [a + sign * b for a, b in zip(left[0], right[0], strict=True)],
                    bool(left[1]) and bool(right[1]),
                )
            if left[0] is not None:
                if isinstance(right[1], int):
                    return [a + sign * right[1] for a in left[0]], left[1]
                return left[0], False
            assert right[0] is not None
            values = [sign * b for b in right[0]]
            if isinstance(left[1], int):
                return [left[1] + v for v in values], right[1]
            return values, False
        if isinstance(expr.op, ast.Mult):
            if left[0] is not None and right[0] is not None:
                if left[1] is True and right[1] is True:
                    return (
                        [a * b for a, b in zip(left[0], right[0], strict=True)],
                        True,
                    )
                return None
            lane_side, other = (left, right) if left[0] is not None else (right, left)
            if not isinstance(other[1], int) or lane_side[0] is None:
                return None
            return [v * other[1] for v in lane_side[0]], lane_side[1]
        if isinstance(expr.op, (ast.FloorDiv, ast.Mod)):
            divisor = _exact_divisor(expr.right, proven_tensor_stride_values)
            if (
                left[0] is None
                or right[0] is not None
                or left[1] is not True
                or divisor is None
                or min(left[0]) < 0
                or max(left[0]) >= 1 << 31
            ):
                return None
            if isinstance(expr.op, ast.FloorDiv):
                return [v // divisor for v in left[0]], True
            return [v % divisor for v in left[0]], True
        return None

    def visit_in_range(expr: ast.AST) -> tuple[list[int] | None, object] | None:
        result = visit(expr)
        if result is None:
            return None
        # Int32 arithmetic would have wrapped a value outside its range: what
        # follows agrees with these values only modulo 2**32.
        values, exact = result
        if (
            values is not None
            and exact is True
            and not all(-(1 << 31) <= v < 1 << 31 for v in values)
        ):
            return values, False
        return result

    result = visit_in_range(node)
    if result is None or result[0] is None:
        return None
    return result[0]


def _lane_offsets_separate(
    node: ast.AST,
    lane_var: str,
    lanes: Sequence[int],
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    width: int = 1,
) -> bool:
    """Whether distinct lanes' ``width``-element accesses at ``node``
    (``_lane_offsets``) are apart, modulo the Int32 index range.  ``lanes``
    are the lane variable's values."""
    values = _lane_offsets(node, lane_var, lanes, proven_tensor_stride_values)
    if values is None:
        return False
    if not values:
        return True
    modulus = 1 << 32
    ordered = sorted(value % modulus for value in values)
    gaps = [b - a for a, b in pairwise(ordered)]
    gaps.append(ordered[0] + modulus - ordered[-1])
    return min(gaps) >= width


def _pointer_integer_affine_form(
    node: ast.AST,
    lane_var: str,
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    unstable_names: set[str],
    site: str,
) -> tuple[int, dict[str, int], int] | None:
    """Return ``(lane coefficient, opaque coefficients, constant)``.

    Generated pointer offsets are integer-affine combinations of coordinates
    and exact tensor strides.  Lane-independent coordinates need not be equal
    at two accesses: retaining them as opaque integer atoms lets the caller
    prove that every possible base difference is a multiple of a safe row
    stride.  Unknown tensor strides and lane-dependent nonlinear expressions
    fail closed.
    """

    def add_forms(
        left: tuple[int, dict[str, int], int],
        right: tuple[int, dict[str, int], int],
        sign: int = 1,
    ) -> tuple[int, dict[str, int], int]:
        lane = left[0] + sign * right[0]
        opaque = dict(left[1])
        for key, coefficient in right[1].items():
            updated = opaque.get(key, 0) + sign * coefficient
            if updated:
                opaque[key] = updated
            else:
                opaque.pop(key, None)
        return lane, opaque, left[2] + sign * right[2]

    def scale_form(
        form: tuple[int, dict[str, int], int], coefficient: int
    ) -> tuple[int, dict[str, int], int]:
        return (
            form[0] * coefficient,
            {
                key: value * coefficient
                for key, value in form[1].items()
                if value * coefficient
            },
            form[2] * coefficient,
        )

    def opaque_atom(expr: ast.AST) -> tuple[int, dict[str, int], int] | None:
        if lane_var in ReadWrites.from_ast(expr).reads:
            return None
        if not _scalar_pointer_calls_are_known(expr):
            return None
        key = ast.dump(expr, include_attributes=False)
        if set(ReadWrites.from_ast(expr).reads) & unstable_names:
            key = f"{site}:{key}"
        return 0, {key: 1}, 0

    def visit(expr: ast.AST) -> tuple[int, dict[str, int], int] | None:
        stride = _tensor_stride_reference(expr)
        if stride is not None:
            value = proven_tensor_stride_values.get(stride)
            return None if value is None else (0, {}, value)
        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, int) and not isinstance(expr.value, bool):
                return 0, {}, expr.value
            return None
        if isinstance(expr, ast.Name):
            if expr.id == lane_var:
                return 1, {}, 0
            return opaque_atom(expr)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.UAdd, ast.USub)):
            operand = visit(expr.operand)
            if operand is None:
                return None
            return operand if isinstance(expr.op, ast.UAdd) else scale_form(operand, -1)
        if (
            isinstance(expr, ast.Call)
            and len(expr.args) == 1
            and not expr.keywords
            and ast.unparse(expr.func) in ("cutlass.Int32", "cutlass.Int64")
        ):
            return visit(expr.args[0])
        if isinstance(expr, ast.BinOp) and isinstance(
            expr.op, (ast.Add, ast.Sub, ast.Mult)
        ):
            if isinstance(expr.op, ast.Mult):
                left_constant = _lane_scale_value(
                    expr.left, proven_tensor_stride_values
                )
                right_constant = _lane_scale_value(
                    expr.right, proven_tensor_stride_values
                )
                if left_constant is not None:
                    right = visit(expr.right)
                    return None if right is None else scale_form(right, left_constant)
                if right_constant is not None:
                    left = visit(expr.left)
                    return None if left is None else scale_form(left, right_constant)
                return None
            left = visit(expr.left)
            right = visit(expr.right)
            if left is None or right is None:
                return None
            return add_forms(left, right, -1 if isinstance(expr.op, ast.Sub) else 1)
        return opaque_atom(expr)

    return visit(node)


def _scalar_pointers_are_modularly_lane_disjoint(
    load_pointer: ast.expr,
    store_pointer: ast.expr,
    *,
    lane_var: str,
    lane_extent: int,
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    unstable_names: set[str],
    width: int = 1,
    lane_step: int = 1,
) -> bool:
    """Prove cross-lane disjointness despite different row/checkpoint bases.

    If both pointers advance by the same nonzero lane step and every possible
    difference between their lane-independent bases is a multiple of ``M``, a
    cross-iteration dependence is impossible when ``step * delta`` is nonzero
    modulo ``M`` for every live lane delta (at least ``width`` elements away
    from zero for vector accesses): the ``lane_extent`` lane values are
    ``lane_step`` apart.  Including the Int32 modulus keeps the proof sound
    under generated index narrowing.
    """
    load_root = _single_tensor_iterator_root(load_pointer)
    store_root = _single_tensor_iterator_root(store_pointer)
    if load_root is None or load_root != store_root:
        return False
    if not _scalar_pointer_calls_are_known(
        load_pointer
    ) or not _scalar_pointer_calls_are_known(store_pointer):
        return False
    load_form = _pointer_integer_affine_form(
        load_pointer,
        lane_var,
        proven_tensor_stride_values,
        unstable_names,
        "load",
    )
    store_form = _pointer_integer_affine_form(
        store_pointer,
        lane_var,
        proven_tensor_stride_values,
        unstable_names,
        "store",
    )
    if load_form is None or store_form is None:
        return False
    lane_coefficient = load_form[0]
    if lane_coefficient == 0 or lane_coefficient != store_form[0]:
        return False
    base_coefficients = dict(load_form[1])
    for key, coefficient in store_form[1].items():
        updated = base_coefficients.get(key, 0) - coefficient
        if updated:
            base_coefficients[key] = updated
        else:
            base_coefficients.pop(key, None)
    modulus = 1 << 32
    modulus = math.gcd(modulus, abs(load_form[2] - store_form[2]))
    for coefficient in base_coefficients.values():
        modulus = math.gcd(modulus, abs(coefficient))
    return modulus > 1 and all(
        _lane_step_separates(lane_coefficient * lane_step * delta, modulus, width)
        for delta in range(1, lane_extent)
    )


def _same_stable_address_aliases(
    node: ast.AST,
    load_definitions: dict[str, ast.expr],
    store_definitions: dict[str, ast.expr],
    lane_var: str,
    unstable_names: set[str],
) -> bool:
    """Ensure opaque lane-independent locals mean the same value at both sites."""
    visiting: set[str] = set()
    checked: set[str] = set()

    def check(name: str) -> bool:
        if name == lane_var or name in checked:
            return True
        if name in unstable_names:
            return False
        load_definition = load_definitions.get(name)
        store_definition = store_definitions.get(name)
        if load_definition is None or store_definition is None:
            return load_definition is store_definition
        if name in visiting:
            return False
        if ast.dump(load_definition, include_attributes=False) != ast.dump(
            store_definition, include_attributes=False
        ):
            return False
        if not _scalar_pointer_calls_are_known(load_definition):
            return False
        visiting.add(name)
        result = all(
            check(dependency)
            for dependency in ReadWrites.from_ast(load_definition).reads
        )
        visiting.remove(name)
        if result:
            checked.add(name)
        return result

    return all(
        check(name) for name in ReadWrites.from_ast(node).reads if name != lane_var
    )


def _lane_accesses_are_iteration_independent(
    load_call: ast.Call,
    store_call: ast.Call,
    *,
    lane_var: str,
    lane_extent: int | None,
    load_definitions: dict[str, ast.expr],
    store_definitions: dict[str, ast.expr],
    proven_tensor_stride_values: Mapping[tuple[str, int], int] | None = None,
    loop_carried_names: set[str] | None = None,
    lane_start: int = 0,
    lane_step: int = 1,
) -> bool:
    """Prove an exact load/store pair cannot communicate across lane iterations.

    Both paths need the load and the store to share one lane-dependent
    pointer function over stable address aliases.  Distinct lanes then reach
    distinct elements when their enumerated offsets are apart
    (``_lane_offsets_separate``, backed by literals and cache-safe exact
    tensor-stride values: an affine pointer with an injective step, or a
    chunked persistent row's ``lane // V * threads * V + lane % V``, which
    is not affine in the lane), or else when the pointers are modularly
    lane-disjoint (``_scalar_pointers_are_modularly_lane_disjoint``).  The
    lanes are the lane variable's values: ``lane_extent`` of them from
    ``lane_start`` in steps of ``lane_step``.  Plain scalar accesses also
    need one known tensor iterator and known pointer calls.
    Persistent vector markers need matching blocks, widths and dtypes, a
    width dividing the lane extent (one fragment per lane loop, or one per
    chunk), and take the affine proof only for one fragment covering every
    lane.  Both paths preserve same-iteration load-before-store order while
    proving that iteration ``i`` cannot access iteration ``j``'s element.
    Shifted/unknown scalar accesses and atomics deliberately fail closed.
    """
    if lane_extent is None:
        return False
    lanes = range(lane_start, lane_start + lane_step * lane_extent, lane_step)
    load_marker = _marker_call(load_call, _LOAD_MARKER, 6)
    store_marker = _marker_call(store_call, _STORE_MARKER, 6)
    if (load_marker is None) != (store_marker is None):
        return False
    if load_marker is None:
        load_access = _generated_access_pointer(load_call)
        store_access = _generated_access_pointer(store_call)
        if load_access is None or store_access is None:
            return False
        load_pointer, load_width = load_access
        store_pointer, store_width = store_access
        # A hoisted packet or flush touches ``width`` consecutive elements
        # from its pointer; the lanes' fragments must not overlap.
        width = max(load_width, store_width)
        unstable_names = loop_carried_names or set()
        expanded_load = _lane_dependent_alias_expansion(
            load_pointer, load_definitions, lane_var
        )
        expanded_store = _lane_dependent_alias_expansion(
            store_pointer, store_definitions, lane_var
        )
        same_mapping = (
            ast.dump(expanded_load, include_attributes=False)
            == ast.dump(expanded_store, include_attributes=False)
            and _single_tensor_iterator_root(expanded_load) is not None
            and not bool(
                (set(ReadWrites.from_ast(expanded_load).reads) - {lane_var})
                & unstable_names
            )
            and _same_stable_address_aliases(
                expanded_load,
                load_definitions,
                store_definitions,
                lane_var,
                unstable_names,
            )
            and _scalar_pointer_calls_are_known(expanded_load)
        )
        if same_mapping and _lane_offsets_separate(
            expanded_load,
            lane_var,
            lanes,
            proven_tensor_stride_values or {},
            width,
        ):
            return True
        return _scalar_pointers_are_modularly_lane_disjoint(
            expanded_load,
            expanded_store,
            lane_var=lane_var,
            lane_extent=lane_extent,
            proven_tensor_stride_values=proven_tensor_stride_values or {},
            unstable_names=unstable_names,
            width=width,
            lane_step=lane_step,
        )
    assert store_marker is not None
    load_block = _literal(load_marker.args[0], int)
    load_width = _literal(load_marker.args[1], int)
    load_dtype = _literal(load_marker.args[2], str)
    store_block = _literal(store_marker.args[0], int)
    store_width = _literal(store_marker.args[1], int)
    store_dtype = _literal(store_marker.args[2], str)
    lane_match = _SYNTHETIC_LANE_RE.fullmatch(lane_var)
    if (
        lane_match is None
        or not isinstance(load_block, int)
        or load_block != store_block
        or load_block != int(lane_match.group(1))
        or not isinstance(load_width, int)
        or load_width != store_width
        # One fragment per lane loop, or one per chunk of a chunked
        # persistent row (``CuteLaneState.chunked_reduction_lanes``).
        or lane_extent % load_width
        or load_dtype != store_dtype
    ):
        return False

    load_pointer = load_marker.args[4]
    store_pointer = store_marker.args[3]
    if not isinstance(load_pointer, ast.expr) or not isinstance(
        store_pointer, ast.expr
    ):
        return False
    expanded_load = _lane_dependent_alias_expansion(
        load_pointer, load_definitions, lane_var
    )
    expanded_store = _lane_dependent_alias_expansion(
        store_pointer, store_definitions, lane_var
    )
    unstable_names = loop_carried_names or set()
    if ast.dump(expanded_load, include_attributes=False) == ast.dump(
        expanded_store, include_attributes=False
    ) and _same_stable_address_aliases(
        expanded_load,
        load_definitions,
        store_definitions,
        lane_var,
        unstable_names,
    ):
        depends, affine = _is_affine_in_lane(expanded_load, lane_var)
        if depends and affine and load_width == lane_extent:
            return True
        # The chunks of a chunked row: each lane one element, the lanes' own
        # elements distinct (``lane // V * threads * V + lane % V``).
        if _lane_offsets_separate(
            expanded_load, lane_var, lanes, proven_tensor_stride_values or {}
        ):
            return True
    return _scalar_pointers_are_modularly_lane_disjoint(
        expanded_load,
        expanded_store,
        lane_var=lane_var,
        lane_extent=lane_extent,
        proven_tensor_stride_values=proven_tensor_stride_values or {},
        unstable_names=unstable_names,
        lane_step=lane_step,
    )


def _store_accesses_are_iteration_independent(
    first_call: ast.Call,
    second_call: ast.Call,
    *,
    lane_var: str,
    lane_extent: int,
    first_definitions: dict[str, ast.expr],
    second_definitions: dict[str, ast.expr],
    proven_tensor_stride_values: Mapping[tuple[str, int], int],
    unstable_names: set[str],
) -> bool:
    """Prove two ordinary exact-fragment stores independent across lanes.

    Only markers as wide as the lane loop qualify (one vector per thread):
    two stores of a chunked row that may alias keep their scalar stores,
    which is safe.  The enumerated proof that admits a chunked row's load
    and store (``_lane_offsets_separate``) is not applied to two stores."""
    first_marker = _marker_call(first_call, _STORE_MARKER, 6)
    second_marker = _marker_call(second_call, _STORE_MARKER, 6)
    if first_marker is None or second_marker is None:
        return False
    lane_match = _SYNTHETIC_LANE_RE.fullmatch(lane_var)
    signatures = [
        (
            _literal(marker.args[0], int),
            _literal(marker.args[1], int),
            _literal(marker.args[2], str),
        )
        for marker in (first_marker, second_marker)
    ]
    if (
        lane_match is None
        or signatures[0] != signatures[1]
        or signatures[0][0] != int(lane_match.group(1))
        or signatures[0][1] != lane_extent
    ):
        return False
    first_pointer = first_marker.args[3]
    second_pointer = second_marker.args[3]
    if not isinstance(first_pointer, ast.expr) or not isinstance(
        second_pointer, ast.expr
    ):
        return False
    expanded_first = _lane_dependent_alias_expansion(
        first_pointer, first_definitions, lane_var
    )
    expanded_second = _lane_dependent_alias_expansion(
        second_pointer, second_definitions, lane_var
    )
    if ast.dump(expanded_first, include_attributes=False) == ast.dump(
        expanded_second, include_attributes=False
    ) and _same_stable_address_aliases(
        expanded_first,
        first_definitions,
        second_definitions,
        lane_var,
        unstable_names,
    ):
        depends, affine = _is_affine_in_lane(expanded_first, lane_var)
        if depends and affine:
            return True
    return _scalar_pointers_are_modularly_lane_disjoint(
        expanded_first,
        expanded_second,
        lane_var=lane_var,
        lane_extent=lane_extent,
        proven_tensor_stride_values=proven_tensor_stride_values,
        unstable_names=unstable_names,
    )


class _ExpandLocalAliases(ast.NodeTransformer):
    def __init__(
        self,
        definitions: dict[str, ast.expr],
        lane_var: str,
        lane_replacement: ast.expr | None,
    ) -> None:
        super().__init__()
        self.definitions = definitions
        self.lane_var = lane_var
        self.lane_replacement = lane_replacement
        self.expanding: set[str] = set()

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if not isinstance(node.ctx, ast.Load):
            return node
        if node.id == self.lane_var:
            if self.lane_replacement is None:
                return node
            return ast.copy_location(_clone_expr(self.lane_replacement), node)
        value = self.definitions.get(node.id)
        if value is None or node.id in self.expanding:
            return node
        self.expanding.add(node.id)
        replacement = self.visit(_clone_expr(value))
        self.expanding.remove(node.id)
        return ast.copy_location(replacement, node)


def _expand(
    node: ast.expr,
    definitions: dict[str, ast.expr],
    lane_var: str,
    lane_replacement: ast.expr | None = None,
) -> ast.expr:
    result = _ExpandLocalAliases(definitions, lane_var, lane_replacement).visit(
        _clone_expr(node)
    )
    assert isinstance(result, ast.expr)
    return result


def _and_terms(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return [term for value in node.values for term in _and_terms(value)]
    return [node]


def _uniform_guard(
    mask: ast.expr | None,
    definitions: dict[str, ast.expr],
    lane_var: str,
    width: int,
    local_writes: set[str],
    canonical: Callable[[str], str],
    first_lane: int = 0,
) -> tuple[bool, ast.expr | None]:
    """Whether ``mask`` admits the fragment of ``width`` lanes starting at
    ``first_lane`` as a whole, and the guard that decides it outside the
    loop (None when every lane is live).  ``local_writes`` holds canonical
    names (``canonical``)."""
    if mask is None:
        return True, None
    expanded = _expand(mask, definitions, lane_var)
    uniform: list[ast.expr] = []
    for term in _and_terms(expanded):
        reads = set(ReadWrites.from_ast(term).reads)
        if lane_var in reads:
            if not isinstance(term, ast.Compare):
                return False, None
            # Prove the complete contiguous fragment is live by checking both
            # endpoints outside the loop.  The candidate's earlier unit-affine
            # proof makes this sufficient for generated tensor-bound compares.
            # User masks (an FP8 packet store's ``extra_mask``) lower to
            # ``operator`` calls, so a lane-dependent one is rejected above.
            endpoints = (
                _expand(term, definitions, lane_var, ast.Constant(value=first_lane)),
                _expand(
                    term,
                    definitions,
                    lane_var,
                    ast.Constant(value=first_lane + width - 1),
                ),
            )
            if any(
                {canonical(name) for name in ReadWrites.from_ast(endpoint).reads}
                & local_writes
                for endpoint in endpoints
            ):
                return False, None
            uniform.extend(endpoints)
            continue
        if {canonical(name) for name in reads} & local_writes:
            return False, None
        uniform.append(term)
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("load", "store")
        for term in uniform
        for node in ast.walk(term)
    ):
        return False, None
    if not uniform:
        return True, None
    if len(uniform) == 1:
        return True, uniform[0]
    return True, ast.BoolOp(op=ast.And(), values=uniform)


def _flatten_add(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return [*_flatten_add(node.left), *_flatten_add(node.right)]
    return [node]


def _sum_terms(terms: list[ast.expr]) -> ast.expr:
    assert terms
    result = _clone_expr(terms[0])
    for term in terms[1:]:
        result = ast.BinOp(left=result, op=ast.Add(), right=_clone_expr(term))
    return result


def _safe_base_pointer(
    pointer: ast.expr,
    definitions: dict[str, ast.expr],
    lane_var: str,
    local_writes: set[str],
    guard: ast.expr | None,
    canonical: Callable[[str], str],
    lane: int = 0,
) -> ast.expr | None:
    """The address ``pointer`` takes at lane ``lane`` (a packet's first lane),
    or None when it does not vary with the lane or reads names the loop
    writes.  ``local_writes`` holds canonical names (``canonical``)."""
    expanded_with_lane = _expand(pointer, definitions, lane_var)
    if lane_var not in set(ReadWrites.from_ast(expanded_with_lane).reads):
        return None
    base = _expand(pointer, definitions, lane_var, ast.Constant(value=lane))
    if {canonical(name) for name in ReadWrites.from_ast(base).reads} & local_writes:
        return None
    if any(
        isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in ("load", "store")
            or isinstance(node.func, ast.Name)
            and node.func.id in (_LOAD_MARKER, _STORE_MARKER)
        )
        for node in ast.walk(base)
    ):
        return None
    if guard is None:
        return base

    terms = _flatten_add(base)
    iterator_indices = [
        index
        for index, term in enumerate(terms)
        if isinstance(term, ast.Attribute)
        and term.attr == "iterator"
        and isinstance(term.value, ast.Name)
    ]
    if len(iterator_indices) != 1 or len(terms) == 1:
        return None
    iterator_index = iterator_indices[0]
    iterator = terms[iterator_index]
    offset = _sum_terms(
        [term for index, term in enumerate(terms) if index != iterator_index]
    )
    return ast.BinOp(
        left=_clone_expr(iterator),
        op=ast.Add(),
        right=ast.IfExp(
            test=_clone_expr(guard),
            body=offset,
            orelse=ast.parse("cutlass.Int32(0)", mode="eval").body,
        ),
    )


_INT_FOLDS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


class _FoldIntConstants(ast.NodeTransformer):
    """Fold integer arithmetic on literals: a chunk's address substitutes its
    first lane into ``lane // V * stride + lane % V``."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        fold = _INT_FOLDS.get(type(node.op))
        if (
            fold is not None
            and isinstance(node.left, ast.Constant)
            and isinstance(node.right, ast.Constant)
            and type(node.left.value) is int
            and type(node.right.value) is int
            and not (
                isinstance(node.op, (ast.FloorDiv, ast.Mod)) and node.right.value == 0
            )
        ):
            return ast.copy_location(
                ast.Constant(value=fold(node.left.value, node.right.value)), node
            )
        return node


def _chunk_guards_and_bases(
    mask: object,
    pointer: object,
    *,
    chunks: int,
    width: int,
    definitions: dict[str, ast.expr],
    lane_var: str,
    local_writes: set[str],
    canonical: Callable[[str], str],
    guarded_bases: bool,
) -> tuple[list[ast.expr | None], list[ast.expr | None]]:
    """Each chunk's packet guard and base pointer, decided at its first lane
    (``chunk * width``): its packet of ``width`` lanes is live as a whole
    under the guard (``_uniform_guard``), and its base is None when no guard
    decides that or ``pointer`` has no safe base (``_safe_base_pointer``).
    With ``guarded_bases`` a base reads its offset under the guard (a load's
    packet); otherwise it is formed unguarded (a store's flush tests the
    guard itself).  Each chunk substitutes its first lane into
    ``lane // V * stride + lane % V``; with more than one chunk, the literal
    arithmetic of every chunk's guard and base is folded."""
    guards: list[ast.expr | None] = []
    bases: list[ast.expr | None] = []
    for chunk in range(chunks):
        first_lane = chunk * width
        guard_ok, guard = (
            _uniform_guard(
                mask,
                definitions,
                lane_var,
                width,
                local_writes,
                canonical,
                first_lane=first_lane,
            )
            if isinstance(mask, (ast.expr, type(None)))
            else (False, None)
        )
        base = (
            _safe_base_pointer(
                pointer,
                definitions,
                lane_var,
                local_writes,
                guard if guarded_bases else None,
                canonical,
                lane=first_lane,
            )
            if guard_ok and isinstance(pointer, ast.expr)
            else None
        )
        if chunks > 1:
            guard = None if guard is None else _fold_int_constants(guard)
            base = None if base is None else _fold_int_constants(base)
        guards.append(guard)
        bases.append(base)
    return guards, bases


def _fold_int_constants(expr: ast.expr) -> ast.expr:
    return cast("ast.expr", _FoldIntConstants().visit(_clone_expr(expr)))


def _scalar_load_and_mask(node: ast.expr) -> tuple[ast.Call, ast.expr | None] | None:
    if isinstance(node, ast.IfExp):
        value = node.body
        mask = node.test
    else:
        value = node
        mask = None
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "load"
    ):
        return value, mask
    return None


class _RestoreLoadMarkers(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        marker = _marker_call(node, _LOAD_MARKER, 6)
        if marker is not None:
            return self.visit(_clone_expr(marker.args[5]))
        return self.generic_visit(node)


@dataclasses.dataclass(frozen=True)
class _HoistedPacket:
    """A vector load placed before a lane loop, reusable by a later sibling
    loop that loads the same packet."""

    name: str
    # Tensors the packet reads (None: unresolved) and names its address reads.
    roots: frozenset[str] | None
    names: frozenset[str]
    # Whether its address reads an element of a name (``offs[0]``; a layout's
    # ``stride[0]`` is no fragment).
    reads_elements: bool


class _BranchLocalPersistentVectorizer:
    def __init__(
        self,
        body: list[ast.stmt],
        proven_disjoint_tensor_pairs: set[frozenset[str]],
        proven_tensor_stride_values: Mapping[tuple[str, int], int],
        wide_packet_hints: bool = False,
        rename_groups: Mapping[str, str] | None = None,
        chunked_lanes: Mapping[str, int] | None = None,
    ) -> None:
        self.name_counts = Counter(
            node.id
            for stmt in body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Name)
        )
        self.used_names = set(self.name_counts)
        self.counter = 0
        self.proven_disjoint_tensor_pairs = proven_disjoint_tensor_pairs
        self.proven_tensor_stride_values = proven_tensor_stride_values
        # Hinted 32-byte packets take their ``_32b`` helpers
        # (``cache_policy_loads.hint_wide_packets`` of the config).
        self.wide_packet_hints = wide_packet_hints
        # Lane loops walking several V-wide chunks per thread, by lane
        # variable: V (``CuteLaneState.chunked_reduction_lanes``).
        self.chunked_lanes = chunked_lanes or {}
        # Packets hoisted earlier in the statement list being transformed,
        # by (block, width, dtype, address): the consume loop of a split
        # persistent reduction reads the packet its accumulate loop loaded.
        self.hoisted: dict[tuple[int, int, str, str], _HoistedPacket] = {}
        # The final rename merges each group into one name (``lane_base_1``
        # into ``lane_base``): writes and reads compare canonical names.
        self.rename_groups = rename_groups or {}

    def _canonical(self, name: str) -> str:
        return self.rename_groups.get(name, name)

    def _packet_chunks(self, lane_var: str, extent: int, width: object) -> int:
        """The packets of a marker of ``width`` lanes in a lane loop of
        ``extent``: one when the marker covers the loop, one per chunk of a
        chunked persistent row, else 0 (the marker stays scalar)."""
        if width == extent:
            return 1
        if (
            isinstance(width, int)
            and self.chunked_lanes.get(lane_var) == width
            and extent % width == 0
        ):
            return extent // width
        return 0

    def _fresh(self, prefix: str) -> str:
        while True:
            name = f"{prefix}_{self.counter}"
            self.counter += 1
            if name not in self.used_names:
                self.used_names.add(name)
                return name

    @staticmethod
    def _store_marker(stmt: ast.stmt) -> ast.Call | None:
        if not isinstance(stmt, ast.Expr):
            return None
        return _marker_call(stmt.value, _STORE_MARKER, 6)

    @staticmethod
    def _restore_store(marker: ast.Call) -> list[ast.stmt]:
        pointer, value, mask = marker.args[3:]
        store = ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=_clone_expr(pointer), attr="store", ctx=ast.Load()
                ),
                args=[_clone_expr(value)],
                keywords=[],
            )
        )
        if isinstance(mask, ast.Constant) and mask.value is None:
            return [store]
        return [
            ast.If(test=_clone_expr(mask), body=[store], orelse=[]),
        ]

    @staticmethod
    def _pointer_roots(node: ast.AST) -> frozenset[str] | None:
        roots: set[str] = set()
        unresolved = False
        for child in ast.walk(node):
            if not isinstance(child, ast.Attribute) or child.attr != "iterator":
                continue
            if isinstance(child.value, ast.Name):
                roots.add(child.value.id)
            else:
                unresolved = True
        if unresolved:
            return None
        return frozenset(roots) if roots else None

    def _may_alias(
        self,
        left: frozenset[str] | None,
        right: frozenset[str] | None,
    ) -> bool:
        if left is None or right is None:
            return True
        return any(
            left_name == right_name
            or frozenset((left_name, right_name))
            not in self.proven_disjoint_tensor_pairs
            for left_name in left
            for right_name in right
        )

    @staticmethod
    def _contains_arch_attribute(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Attribute) and child.attr == "arch"
            for child in ast.walk(node)
        )

    @classmethod
    def _load_roots(cls, node: ast.Call) -> tuple[bool, frozenset[str] | None]:
        """Whether ``node`` loads, and the tensors it may read (None: any)."""
        marker = _marker_call(node, _LOAD_MARKER, 6)
        if marker is not None:
            return True, cls._pointer_roots(marker.args[4])
        if isinstance(node.func, ast.Name) and node.func.id in _VECTOR_LOAD_HELPERS:
            # ``<L2 helper>(PTR, ir.VectorType.get(...))``: a packet load the
            # loop's aliasing checks must see like any other load.  One of a
            # form the parser does not know may read anything.
            access = match_packet_access(node)
            return True, None if access is None else cls._pointer_roots(access.pointer)
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "load"):
            return False, None
        if node.args and cls._contains_arch_attribute(node.func.value):
            return True, cls._pointer_roots(node.args[0])
        return True, cls._pointer_roots(node.func.value)

    def _has_aliasing_write_in_loop(
        self,
        body: list[ast.stmt],
        load_marker: ast.Call,
        load_index: int,
        lane_var: str,
        lane_extent: int,
        definition_snapshots: list[dict[str, ast.expr]] | None,
    ) -> bool:
        """Whether any loop iteration can write storage read by this load.

        A write lexically after the load still runs before that load in the
        next constexpr-lane iteration.  Hoisting the vector load above the
        loop would observe the pre-write value for every lane. Such a write is
        a barrier unless its tensor is disjoint or both exact accesses have the
        same injective lane mapping.  ``definition_snapshots`` are ``body``'s
        ``_memory_address_definition_snapshots`` (``None``, a definition
        reaching an address too large to freeze, declines).
        """
        load_roots = self._pointer_roots(load_marker.args[4])
        if definition_snapshots is None:
            return True
        for store_index, stmt in enumerate(body):
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Call):
                    continue
                written = call_write_roots(node, self._pointer_roots)
                if written == frozenset() or not self._may_alias(written, load_roots):
                    continue
                # A later exact store preserves this load's same-iteration
                # ordering. Its identical injective lane mapping additionally
                # rules out a dependence through the loop backedge.
                if (
                    store_index > load_index
                    and _lane_accesses_are_iteration_independent(
                        load_marker,
                        node,
                        lane_var=lane_var,
                        lane_extent=lane_extent,
                        load_definitions=definition_snapshots[load_index],
                        store_definitions=definition_snapshots[store_index],
                        proven_tensor_stride_values=(self.proven_tensor_stride_values),
                        loop_carried_names=_unstable_address_names(
                            body,
                            lane_var,
                            load_index,
                            store_index,
                        ),
                    )
                ):
                    continue
                return True
        return False

    def _has_other_aliasing_access_in_loop(
        self,
        body: list[ast.stmt],
        store_marker: ast.Call,
        store_index: int,
        lane_var: str,
        lane_extent: int,
        definition_snapshots: list[dict[str, ast.expr]] | None,
    ) -> bool:
        """Whether sinking a store can cross an aliasing memory access.

        A load lexically before the store executes after that store on the
        next lane iteration.  Likewise, another write on either side can race
        the sunk vector transaction across iterations.  Exclude only the
        candidate marker itself; every other load/store/atomic in the repeated
        loop must be proven disjoint, except a preceding exact load with the
        same injective lane mapping and another store marker with it
        (``_store_accesses_are_iteration_independent``).  A later such store
        must be sunk as well, or this one is put back
        (``_unsink_overwritten_stores``).  ``definition_snapshots`` are as for
        ``_has_aliasing_write_in_loop``.
        """
        store_roots = self._pointer_roots(store_marker.args[3])
        if definition_snapshots is None:
            return True
        for access_index, stmt in enumerate(body):
            for node in _memory_load_calls(stmt):
                loads, roots = self._load_roots(node)
                if loads and self._may_alias(store_roots, roots):
                    # Sinking the store retains a preceding load's within-lane
                    # order. Matching exact lane mappings rule out only the
                    # cross-iteration dependence introduced by the sink.
                    if (
                        access_index < store_index
                        and _lane_accesses_are_iteration_independent(
                            node,
                            store_marker,
                            lane_var=lane_var,
                            lane_extent=lane_extent,
                            load_definitions=definition_snapshots[access_index],
                            store_definitions=definition_snapshots[store_index],
                            proven_tensor_stride_values=(
                                self.proven_tensor_stride_values
                            ),
                            loop_carried_names=_unstable_address_names(
                                body,
                                lane_var,
                                access_index,
                                store_index,
                            ),
                        )
                    ):
                        continue
                    return True
            for node in ast.walk(stmt):
                if node is store_marker or not isinstance(node, ast.Call):
                    continue
                written = call_write_roots(node, self._pointer_roots)
                if written != frozenset() and self._may_alias(store_roots, written):
                    if _store_accesses_are_iteration_independent(
                        store_marker,
                        node,
                        lane_var=lane_var,
                        lane_extent=lane_extent,
                        first_definitions=definition_snapshots[store_index],
                        second_definitions=definition_snapshots[access_index],
                        proven_tensor_stride_values=(self.proven_tensor_stride_values),
                        unstable_names=_unstable_address_names(
                            body,
                            lane_var,
                            min(store_index, access_index),
                            max(store_index, access_index),
                        ),
                    ):
                        continue
                    return True
        return False

    def _vectorize_loop(
        self, loop: ast.For, block_id: int, width: int
    ) -> tuple[list[ast.stmt], ast.For, list[ast.stmt]]:
        assert isinstance(loop.target, ast.Name)
        lane_var = loop.target.id
        # ``names_written`` adds the targets of the loops nested in the lane
        # loop, which ``ReadWrites`` skips: an address reading an inner loop's
        # index changes inside the lane loop, so it stays there.
        all_local_writes = {
            self._canonical(name)
            for stmt in loop.body
            for name in set(ReadWrites.from_ast(stmt).writes) | names_written(stmt)
        }
        fragments = {
            self._canonical(name)
            for stmt in loop.body
            for node in ast.walk(stmt)
            if isinstance(node, ast.Call)
            for name in fragment_writes(node)
        }
        if fragments or any(
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            for stmt in loop.body
            for node in ast.walk(stmt)
        ):
            # A store into an element, through any name, may change an element
            # another name reads (``alias = offs; alias[0] = alias[0] + 1``),
            # as between loops (``_invalidate_hoisted``): every fragment the
            # loop reads an element of, or a call stores into in place
            # (``fragment_writes``), counts as written.
            all_local_writes |= fragments | {
                self._canonical(node.value.id)
                for stmt in loop.body
                for node in ast.walk(stmt)
                if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            }
        definitions: dict[str, ast.expr] = {}
        before: list[ast.stmt] = []
        after: list[ast.stmt] = []
        rewritten_body: list[ast.stmt] = []
        # Each sunk store's collector setup and flushes, by statement index
        # (``_unsink_overwritten_stores``).
        sunk: dict[int, tuple[ast.stmt, list[ast.stmt]]] = {}
        # The aliasing checks compare the addresses of this loop's markers as
        # the body reads before any rewrite: hoisting a load edits its
        # statement in place, but every rewrite below runs a check first, so
        # the first call, which the cache keeps, comes before all of them.
        address_definitions = cache(
            partial(_memory_address_definition_snapshots, loop.body)
        )

        for stmt_index, stmt in enumerate(loop.body):
            collector: ast.stmt | None = None
            flushes_from = len(after)
            load_markers: list[ast.Call] = []
            for node in ast.walk(stmt):
                marker_call = _marker_call(node, _LOAD_MARKER, 6)
                if marker_call is not None:
                    load_markers.append(marker_call)
            if len(load_markers) == 1:
                marker = load_markers[0]
                marker_block = _literal(marker.args[0], int)
                marker_width = _literal(marker.args[1], int)
                dtype = _literal(marker.args[2], str)
                eviction = _literal(marker.args[3], str)
                original = marker.args[5]
                load_info = (
                    _scalar_load_and_mask(original)
                    if isinstance(original, ast.expr)
                    else None
                )
                dtype_info = _DTYPE_INFO.get(dtype) if isinstance(dtype, str) else None
                chunks = self._packet_chunks(lane_var, width, marker_width)
                if (
                    marker_block == block_id
                    and chunks
                    and isinstance(eviction, str)
                    and dtype_info is not None
                    and load_info is not None
                    and not self._has_aliasing_write_in_loop(
                        loop.body,
                        marker,
                        stmt_index,
                        lane_var,
                        width,
                        address_definitions(),
                    )
                ):
                    assert isinstance(marker_width, int)
                    scalar_load, mask = load_info
                    _guards, bases = _chunk_guards_and_bases(
                        mask,
                        marker.args[4],
                        chunks=chunks,
                        width=marker_width,
                        definitions=definitions,
                        lane_var=lane_var,
                        local_writes=all_local_writes,
                        canonical=self._canonical,
                        guarded_bases=True,
                    )
                    if all(base is not None for base in bases):
                        carrier, itemsize, _store_helper = dtype_info
                        assert isinstance(dtype, str)
                        vec_names: list[str] = []
                        for base in bases:
                            assert base is not None
                            key = (block_id, marker_width, dtype, ast.unparse(base))
                            hoisted = self.hoisted.get(key)
                            if hoisted is not None:
                                # Nothing since the earlier loop's packet was
                                # loaded may have written it or rebound a
                                # name of its address (``_invalidate_hoisted``).
                                vec_names.append(hoisted.name)
                                continue
                            vec_name = self._fresh("_persistent_branch_vec")
                            vec_names.append(vec_name)
                            # An L2-policy hint is not a ``cute.arch.load``
                            # kwarg: the packet goes through the inline-PTX
                            # helper of its width (32 bytes: nc_l2_first, and
                            # the L2 policies under ``cute_wide_packet_hints``).
                            policy = CachePolicy.from_suffix(eviction)
                            load_source = render_packet_load(
                                ast.unparse(base),
                                marker_width,
                                carrier,
                                helper=policy.vector_helper(
                                    marker_width * itemsize,
                                    wide_packets=self.wide_packet_hints,
                                ),
                                kwargs=policy.load_kwargs,
                            )
                            vector_load = ast.parse(load_source, mode="eval").body
                            before.append(
                                ast.Assign(
                                    targets=[ast.Name(id=vec_name, ctx=ast.Store())],
                                    value=vector_load,
                                )
                            )
                            self.hoisted[key] = _HoistedPacket(
                                vec_name,
                                self._pointer_roots(base),
                                frozenset(map(self._canonical, names_read(base))),
                                any(
                                    isinstance(node, ast.Subscript)
                                    and isinstance(node.value, ast.Name)
                                    for node in ast.walk(base)
                                ),
                            )
                        element = (
                            f"{vec_names[0]}[{lane_var}]"
                            if chunks == 1
                            else f"({', '.join(vec_names)})"
                            f"[{lane_var} // {marker_width}][{lane_var} % {marker_width}]"
                        )
                        extract = ast.parse(
                            f"{carrier}({element}).bitcast({dtype})",
                            mode="eval",
                        ).body
                        replacement, _ = replace_call(
                            _clone_expr(original),
                            scalar_load,
                            partial(_clone_expr, extract),
                            first_only=True,
                        )
                        assert isinstance(replacement, ast.expr)
                        stmt, _ = replace_call(
                            stmt,
                            marker,
                            partial(_clone_expr, replacement),
                            first_only=True,
                        )
                        assert isinstance(stmt, ast.stmt)

            store_marker = self._store_marker(stmt)
            if store_marker is not None:
                marker_block = _literal(store_marker.args[0], int)
                marker_width = _literal(store_marker.args[1], int)
                dtype = _literal(store_marker.args[2], str)
                dtype_info = (
                    _STORE_DTYPE_INFO.get(dtype) if isinstance(dtype, str) else None
                )
                mask_arg = store_marker.args[5]
                mask = (
                    None
                    if isinstance(mask_arg, ast.Constant) and mask_arg.value is None
                    else mask_arg
                )
                chunks = self._packet_chunks(lane_var, width, marker_width)
                # Each chunk's flush decides its own lanes.
                guards, bases = _chunk_guards_and_bases(
                    mask,
                    store_marker.args[3],
                    chunks=chunks,
                    width=cast("int", marker_width),
                    definitions=definitions,
                    lane_var=lane_var,
                    local_writes=all_local_writes,
                    canonical=self._canonical,
                    guarded_bases=False,
                )
                if (
                    marker_block == block_id
                    and chunks
                    and dtype_info is not None
                    and all(base is not None for base in bases)
                    and not self._has_other_aliasing_access_in_loop(
                        loop.body,
                        store_marker,
                        stmt_index,
                        lane_var,
                        width,
                        address_definitions(),
                    )
                ):
                    carrier, _itemsize, store_helper = dtype_info
                    lane_value = _clone_expr(store_marker.args[4])
                    if dtype == "cutlass.Boolean":
                        lane_value = ast.Call(
                            func=ast.parse(carrier, mode="eval").body,
                            args=[lane_value],
                            keywords=[],
                        )
                    elif carrier == "cutlass.Uint8":
                        assert isinstance(dtype, str)
                        fp8_source = self._fp8_lane_value(
                            lane_value, dtype, rewritten_body
                        )
                        if fp8_source is None:
                            rewritten_body.extend(self._restore_store(store_marker))
                            continue
                        lane_value = fp8_source
                    else:
                        lane_value = ast.Call(
                            func=ast.Attribute(
                                value=lane_value, attr="bitcast", ctx=ast.Load()
                            ),
                            args=[ast.parse(carrier, mode="eval").body],
                            keywords=[],
                        )
                    values_name = self._fresh("_persistent_branch_store_values")
                    collector = ast.Assign(
                        targets=[ast.Name(id=values_name, ctx=ast.Store())],
                        value=ast.List(elts=[], ctx=ast.Load()),
                    )
                    before.append(collector)
                    append = ast.Expr(
                        value=ast.Call(
                            func=ast.Attribute(
                                value=ast.Name(id=values_name, ctx=ast.Load()),
                                attr="append",
                                ctx=ast.Load(),
                            ),
                            args=[lane_value],
                            keywords=[],
                        )
                    )
                    stmt = append
                    assert isinstance(marker_width, int)
                    for chunk, (base, guard) in enumerate(
                        zip(bases, guards, strict=True)
                    ):
                        assert base is not None
                        # Several chunks flush constant slices of the one
                        # collector (``packet_store_policy._store_call``).
                        values: ast.expr = ast.Name(id=values_name, ctx=ast.Load())
                        if chunks > 1:
                            values = ast.Subscript(
                                value=values,
                                slice=ast.Slice(
                                    lower=ast.Constant(value=chunk * marker_width),
                                    upper=ast.Constant(
                                        value=(chunk + 1) * marker_width
                                    ),
                                ),
                                ctx=ast.Load(),
                            )
                        flush = ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id=store_helper, ctx=ast.Load()),
                                args=[base, values],
                                keywords=[],
                            )
                        )
                        after.append(
                            flush
                            if guard is None
                            else ast.If(
                                test=_clone_expr(guard), body=[flush], orelse=[]
                            )
                        )
                else:
                    rewritten_body.extend(self._restore_store(store_marker))
                    continue

            rewritten_body.append(stmt)
            definitions.update(_single_name_definitions([stmt]))
            if collector is not None:
                sunk[stmt_index] = (collector, after[flushes_from:])

        self._unsink_overwritten_stores(loop.body, rewritten_body, before, after, sunk)
        loop.body = self.transform_body(rewritten_body)
        return before, loop, after

    def _unsink_overwritten_stores(
        self,
        body: list[ast.stmt],
        rewritten_body: list[ast.stmt],
        before: list[ast.stmt],
        after: list[ast.stmt],
        sunk: dict[int, tuple[ast.stmt, list[ast.stmt]]],
    ) -> None:
        """Put back in the loop each sunk store that a later write of the
        loop may overwrite there.

        ``_has_other_aliasing_access_in_loop`` lets a store sink past a later
        store marker with the same lane addresses, or with lane addresses
        only proven apart across lanes (modular lane-disjointness: in the same
        lane the two may still meet).  That store runs after the sunk one in
        every lane, so it must flush after it, as a packet of its own; one
        that stays in the loop (its guard varies across the lanes, or it is
        nested) would be overwritten by the earlier flush.  The sunk store
        then keeps its scalar form, and counts as a write of the loop for the
        stores before it.  ``body`` holds the loop's marked statements,
        ``rewritten_body`` their rewrites, one each; ``sunk`` maps the index
        of each sunk store to its collector setup in ``before`` and its
        flushes in ``after``.
        """
        assert len(body) == len(rewritten_body)
        if not sunk:
            # Nothing to put back: the walk below would read every call.
            return
        later_writes: list[frozenset[str] | None] = []
        for index in reversed(range(len(body))):
            if index in sunk:
                marker = self._store_marker(body[index])
                assert marker is not None
                roots = self._pointer_roots(marker.args[3])
                if not any(self._may_alias(roots, other) for other in later_writes):
                    continue
                collector, flushes = sunk[index]
                rewritten_body[index : index + 1] = self._restore_store(marker)
                before[:] = [stmt for stmt in before if stmt is not collector]
                after[:] = [
                    stmt for stmt in after if not any(stmt is f for f in flushes)
                ]
            for node in ast.walk(body[index]):
                if isinstance(node, ast.Call):
                    written = call_write_roots(node, self._pointer_roots)
                    if written != frozenset():
                        later_writes.append(written)

    def _fp8_lane_value(
        self, value: ast.expr, dtype: str, loop_body: list[ast.stmt]
    ) -> ast.expr | None:
        """The input of the store's FP8 cast; the flush helper converts it.

        When that input is a single-use ``name = <dtype>(x)`` of this loop,
        collect ``x`` instead so FP32/FP16/BF16 lanes reach the vector
        conversion; ``name = x`` keeps the value from the assignment's position.
        Returns ``None`` for a value that is not an FP8 cast, which keeps the
        scalar stores.  ``memory_ops`` wraps every FP8 store value in the cast,
        so a copied E4M3FN value (a raw ``Uint8`` load) still reaches the
        helper, which converts it the way the scalar store does.
        """
        if not (
            isinstance(value, ast.Call)
            and ast.unparse(value.func) == dtype
            and len(value.args) == 1
            and not value.keywords
        ):
            return None
        (value,) = value.args
        if isinstance(value, ast.Name) and self.name_counts[value.id] == 2:
            for stmt in loop_body:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == value.id
                    and isinstance(stmt.value, ast.Call)
                    and ast.unparse(stmt.value.func) == dtype
                    and len(stmt.value.args) == 1
                    and not stmt.value.keywords
                ):
                    stmt.value = stmt.value.args[0]
                    break
        return value

    def _invalidate_hoisted(self, statement: ast.stmt) -> None:
        """Forget the hoisted packets ``statement`` may write (a call that
        may write memory it does not name, ``call_write_roots``, writes all
        of them) or whose address names it rebinds.  A store into an element,
        through any name, ends the packets whose address reads an element of
        a name: another name may hold the fragment
        (``alias = offs; alias[0] = 8``)."""
        if not self.hoisted:
            return
        written = {
            self._canonical(name)
            for name in set(ReadWrites.from_ast(statement).writes)
            | names_written(statement)
        }
        write_roots = [
            roots
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and (roots := call_write_roots(node, self._pointer_roots)) != frozenset()
        ]
        # A call that stores into a fragment in place (``fragment_writes``)
        # writes its elements too.
        fragments = {
            self._canonical(name)
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            for name in fragment_writes(node)
        }
        written |= fragments
        stores_elements = bool(fragments) or any(
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, (ast.Store, ast.Del))
            for node in ast.walk(statement)
        )
        self.hoisted = {
            key: packet
            for key, packet in self.hoisted.items()
            if not packet.names & written
            and not (stores_elements and packet.reads_elements)
            and not any(self._may_alias(roots, packet.roots) for roots in write_roots)
        }

    def transform_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        # Packets are reused only between sibling loops of one statement list.
        outer, self.hoisted = self.hoisted, {}
        result: list[ast.stmt] = []
        for stmt in body:
            if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
                match = _SYNTHETIC_LANE_RE.fullmatch(stmt.target.id)
                extent = _range_extent(stmt)
                if match is not None and extent is not None:
                    before, loop, after = self._vectorize_loop(
                        stmt, int(match.group(1)), extent
                    )
                    result.extend(before)
                    result.append(loop)
                    result.extend(after)
                    for emitted in (loop, *after):
                        self._invalidate_hoisted(emitted)
                    continue

            store_marker = self._store_marker(stmt)
            if store_marker is not None:
                restored = self._restore_store(store_marker)
                result.extend(restored)
                for emitted in restored:
                    self._invalidate_hoisted(emitted)
                continue
            for field in ("body", "orelse", "finalbody"):
                child = getattr(stmt, field, None)
                if isinstance(child, list) and all(
                    isinstance(child_stmt, ast.stmt) for child_stmt in child
                ):
                    setattr(stmt, field, self.transform_body(child))
            result.append(stmt)
            self._invalidate_hoisted(stmt)
        self.hoisted = outer
        return result


def vectorize_branch_local_persistent_fragments(
    body: list[ast.stmt],
    *,
    proven_disjoint_tensor_pairs: set[frozenset[str]] | None = None,
    proven_tensor_stride_values: Mapping[tuple[str, int], int] | None = None,
    wide_packet_hints: bool = False,
    rename_groups: Mapping[str, str] | None = None,
    chunked_lanes: Mapping[str, int] | None = None,
) -> list[ast.stmt]:
    """Vectorize marked exact fragments and erase every unhandled marker.

    ``wide_packet_hints`` (``cache_policy_loads.hint_wide_packets`` of the
    config) lets a hinted 32-byte packet take its ``_32b`` helper.
    ``rename_groups`` maps each name to the one the final rename gives it;
    ``chunked_lanes`` names the lane loops of chunked persistent rows (lane
    variable -> V), which take one packet per V-wide chunk."""
    if not any(
        isinstance(node, ast.Name) and node.id in (_LOAD_MARKER, _STORE_MARKER)
        for stmt in body
        for node in ast.walk(stmt)
    ):
        # Without a load or store marker the walk below restores nothing and
        # vectorizes nothing. Keep the location fix-up the full path applies,
        # since later passes unparse statements created without locations.
        return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])).body
    vectorizer = _BranchLocalPersistentVectorizer(
        body,
        proven_disjoint_tensor_pairs or set(),
        proven_tensor_stride_values or {},
        wide_packet_hints,
        rename_groups,
        chunked_lanes,
    )
    transformed = vectorizer.transform_body(body)
    # Most kernels have no persistent-fragment load markers.  Avoid running an
    # AST ``NodeTransformer`` over those bodies: FX can preserve an
    # ``immutable_list`` in constructs such as inline-asm operand lists, while
    # ``NodeTransformer.generic_visit`` mutates list fields in place.
    if not any(
        isinstance(node, ast.Call) and _marker_call(node, _LOAD_MARKER, 6) is not None
        for stmt in transformed
        for node in ast.walk(stmt)
    ):
        module = ast.Module(body=transformed, type_ignores=[])
        ast.fix_missing_locations(module)
        return module.body
    restored = _RestoreLoadMarkers().visit(
        ast.Module(body=transformed, type_ignores=[])
    )
    assert isinstance(restored, ast.Module)
    ast.fix_missing_locations(restored)
    return restored.body

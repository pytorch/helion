"""Vectorize affine scalar memory operations in constexpr tile-lane loops.

Address analysis happens after scalar code generation, where indirect index
arithmetic and its definitions are visible together. A complete-fragment
guard selects the vector path; partial tails retain the original scalar loop.
Unknown effects, aliased stores, and non-unit lane mappings are left intact.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import re
from typing import TYPE_CHECKING

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ..ast_read_writes import ReadWrites
from ..compile_environment import CompileEnvironment
from .memory_ops import tensor_has_specialized_base_alignment
from .persistent_branch_vec import _DTYPE_INFO
from .persistent_branch_vec import _and_terms
from .persistent_branch_vec import _binding_write_roots
from .persistent_branch_vec import _clone_expr
from .persistent_branch_vec import _definition_snapshots
from .persistent_branch_vec import _expand
from .persistent_branch_vec import _lane_scale_value
from .persistent_branch_vec import _plain_scalar_load_pointer
from .persistent_branch_vec import _plain_scalar_store_pointer
from .persistent_branch_vec import _pointer_integer_affine_form
from .persistent_branch_vec import _range_extent
from .persistent_branch_vec import _single_tensor_iterator_root
from .scalar_integer import _integer_expression

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..device_function import DeviceFunction

_LANE = re.compile(r"vec_lane_(\d+)")
_CASTS = frozenset(
    f"cutlass.{name}"
    for name in (
        "Int32",
        "Int64",
        "Uint32",
        "Uint64",
        "Float32",
        "Float16",
        "BFloat16",
        "Boolean",
    )
)
_COORDINATE_CALLS = frozenset(("cute.arch.thread_idx", "cute.arch.block_idx"))
_PURE = frozenset(
    f"cute.math.{name}"
    for name in (
        "abs",
        "exp",
        "exp2",
        "log",
        "log2",
        "max",
        "min",
        "rcp",
        "rsqrt",
        "sqrt",
        "tanh",
    )
)


@dataclasses.dataclass
class _Access:
    call: ast.Call
    pointer: ast.expr
    mask: ast.expr | None
    statement: int
    store: bool


class _Decline(Exception):
    pass


def _pure_boolean_guard(
    node: ast.expr, *, boolean_names: frozenset[str] = frozenset()
) -> bool:
    """Recognize effect-free Boolean scalar guards without evaluating them.

    Arithmetic can still raise (for example, division or a NaN conversion).
    The caller must retain operand order and short-circuit evaluation.
    """

    def scalar(value: ast.expr) -> bool:
        if isinstance(value, ast.Constant):
            return type(value.value) in (bool, int, float)
        if isinstance(value, ast.Name):
            return True
        if isinstance(value, ast.UnaryOp) and isinstance(
            value.op, (ast.UAdd, ast.USub, ast.Invert, ast.Not)
        ):
            return scalar(value.operand)
        if isinstance(value, ast.BinOp) and isinstance(
            value.op,
            (
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.FloorDiv,
                ast.Mod,
                ast.BitAnd,
                ast.BitOr,
                ast.BitXor,
                ast.LShift,
                ast.RShift,
            ),
        ):
            return scalar(value.left) and scalar(value.right)
        if isinstance(value, ast.Compare) and all(
            isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE))
            for op in value.ops
        ):
            return all(scalar(part) for part in (value.left, *value.comparators))
        if isinstance(value, ast.BoolOp):
            return all(
                _pure_boolean_guard(part, boolean_names=boolean_names)
                for part in value.values
            )
        if isinstance(value, ast.Call):
            return (
                ast.unparse(value.func) in _CASTS
                and len(value.args) == 1
                and not value.keywords
                and scalar(value.args[0])
            )
        if (
            isinstance(value, ast.Subscript)
            and isinstance(value.slice, ast.Constant)
            and type(value.slice.value) is int
            and value.slice.value >= 0
        ):
            base = value.value
            if (
                isinstance(base, ast.Attribute)
                and base.attr == "stride"
                and isinstance(base.value, ast.Attribute)
                and base.value.attr == "layout"
                and isinstance(base.value.value, ast.Name)
            ):
                return True
            return (
                isinstance(base, ast.Call)
                and ast.unparse(base.func) in _COORDINATE_CALLS
                and not base.args
                and not base.keywords
                and value.slice.value < 3
            )
        return False

    if isinstance(node, ast.Constant):
        return type(node.value) is bool
    if isinstance(node, ast.Name):
        return node.id in boolean_names
    if isinstance(node, ast.Compare):
        return scalar(node)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return scalar(node)
    if isinstance(node, ast.BoolOp):
        return all(
            _pure_boolean_guard(value, boolean_names=boolean_names)
            for value in node.values
        )
    return (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cutlass.Boolean"
        and scalar(node)
    )


def _conjunction(values: list[ast.expr]) -> ast.expr | None:
    unique = {ast.dump(value, include_attributes=False): value for value in values}
    if not unique:
        return None
    if len(unique) == 1:
        return next(iter(unique.values()))
    terms = list(unique.values())
    if len(terms) > 2 and all(_pure_boolean_guard(value) for value in terms):
        # CuTe's BoolOp preprocessor duplicates its accumulated LHS four times
        # per term. A flat conjunction therefore expands exponentially during
        # ordinary Python AST traversal. A right-associated tree keeps each
        # duplicated LHS small while preserving every term's order, Boolean
        # value and short-circuit protection. Numeric-valued/unknown guards
        # retain the original representation and conversion order.
        result = terms[-1]
        for value in reversed(terms[:-1]):
            result = ast.BoolOp(op=ast.And(), values=[value, result])
        return result
    return ast.BoolOp(op=ast.And(), values=terms)


def _forget_definitions(definitions: dict[str, ast.expr], writes: set[str]) -> None:
    """Drop expressions that no longer denote their assignment-time value."""
    pending = set(writes)
    while pending:
        invalidated = {
            name
            for name, value in definitions.items()
            if name in pending or pending.intersection(ReadWrites.from_ast(value).reads)
        }
        for name in invalidated:
            definitions.pop(name)
        pending = invalidated


def _i32_offset(node: ast.expr, names: frozenset[str] = frozenset()) -> bool:
    """Require one final modular index width before reasoning about adjacency.

    Widening an intermediate signed wrap can create a discontinuity even when
    its algebraic lane coefficient is one. Generated Int32 pointer offsets
    avoid this: an aligned power-of-two fragment cannot cross their wrap.
    """
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return -(1 << 31) <= node.value < (1 << 31)
    if isinstance(node, ast.Name):
        return node.id in names
    if isinstance(node, ast.Call):
        return (
            ast.unparse(node.func) == "cutlass.Int32"
            and len(node.args) == 1
            and not node.keywords
        )
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult)
    ):
        return _i32_offset(node.left, names) and _i32_offset(node.right, names)
    return False


def _specialize_fragment_mask(
    loop: ast.For, predicate: ast.expr, lane: str, value: bool
) -> ast.For:
    """Keep scalar live-outs and arithmetic while folding a uniform mask.

    In particular, adding masked zero to a reduction carry is retained. Simply
    dropping the inactive loop could change signed-zero or NaN behavior.
    """
    key = ast.dump(predicate, include_attributes=False)
    result = ast.parse(ast.unparse(loop)).body[0]
    assert isinstance(result, ast.For)

    # Truthiness of a numeric predicate does not identify its value or type.
    # Only replace expressions whose value itself is known to be Boolean.
    # Otherwise an arithmetic use of a mask such as Int32(2) would become 1.
    def boolean_value(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return type(node.value) is bool
        if isinstance(node, ast.Compare):
            return True
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return True
        if isinstance(node, ast.BoolOp):
            return all(boolean_value(value) for value in node.values)
        return (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cutlass.Boolean"
            and len(node.args) == 1
            and not node.keywords
        )

    if not boolean_value(predicate):
        return result
    snapshots = _definition_snapshots(result.body)
    if snapshots is None:
        return result

    class Fold(ast.NodeTransformer):
        def __init__(self, definitions: dict[str, ast.expr]) -> None:
            self.definitions = definitions

        def visit(self, node: ast.AST) -> ast.AST | list[ast.stmt] | None:
            if isinstance(node, ast.expr):
                expanded = _expand(node, self.definitions, lane)
                if ast.dump(expanded, include_attributes=False) == key:
                    return ast.Constant(value=value)
            return super().visit(node)

        def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
            test = self.visit(node.test)
            if isinstance(test, ast.Constant) and type(test.value) is bool:
                branch = self.visit(node.body if test.value else node.orelse)
                assert isinstance(branch, ast.expr)
                return branch
            return self.generic_visit(node)

        def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
            test = self.visit(node.test)
            if isinstance(test, ast.Constant) and type(test.value) is bool:
                body = node.body if test.value else node.orelse
                return [item for item in body if not isinstance(item, ast.Pass)]
            return self.generic_visit(node)

    body: list[ast.stmt] = []
    for statement, definitions in zip(result.body, snapshots, strict=True):
        rewritten = Fold(definitions).visit(statement)
        if isinstance(rewritten, list):
            body.extend(rewritten)
        elif isinstance(rewritten, ast.stmt):
            body.append(rewritten)
    result.body = body or [ast.Pass()]
    return result


def _accesses(
    body: list[ast.stmt],
    extra_pure_call: Callable[[ast.Call], bool] | None = None,
    private_access: Callable[[ast.Subscript], bool] | None = None,
) -> list[_Access]:
    result: list[_Access] = []

    # Named CuTe tensor subscripts can read global memory. Only the pointer
    # operations below participate in our alias and ordering proof, so reject
    # every other subscript except immutable layout and thread coordinates.
    # Inspect predicates and pointer expressions too: visit() handles those
    # specially and does not recursively traverse all their children.
    for statement in body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Subscript):
                continue
            if private_access is not None and private_access(node):
                continue
            if not (
                isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Constant)
                and type(node.slice.value) is int
                and node.slice.value >= 0
            ):
                raise _Decline
            value = node.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "stride"
                and isinstance(value.value, ast.Attribute)
                and value.value.attr == "layout"
                and isinstance(value.value.value, ast.Name)
            ):
                continue
            if (
                isinstance(value, ast.Call)
                and ast.unparse(value.func)
                in ("cute.arch.thread_idx", "cute.arch.block_idx")
                and not value.args
                and not value.keywords
                and node.slice.value < 3
            ):
                continue
            raise _Decline

    def visit(node: ast.AST, statement: int, guards: list[ast.expr]) -> None:
        if isinstance(node, ast.IfExp):
            # A predicate may not itself perform memory accesses or effects.
            if any(
                isinstance(item, ast.Call)
                and ast.unparse(item.func) not in _CASTS
                and not (extra_pure_call is not None and extra_pure_call(item))
                for item in ast.walk(node.test)
            ):
                raise _Decline
            visit(node.body, statement, [*guards, node.test])
            visit(
                node.orelse,
                statement,
                [*guards, ast.UnaryOp(op=ast.Not(), operand=node.test)],
            )
            return
        if isinstance(node, ast.If):
            if not (
                len(node.body) == 1
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Call)
                and _plain_scalar_store_pointer(node.body[0].value) is not None
                and all(isinstance(item, ast.Pass) for item in node.orelse)
            ):
                raise _Decline
            visit(node.body[0], statement, [*guards, node.test])
            return
        if isinstance(
            node,
            (ast.For, ast.While, ast.Break, ast.Continue, ast.Return, ast.AugAssign),
        ):
            raise _Decline
        if isinstance(node, ast.Assign) and not all(
            isinstance(target, ast.Name)
            or isinstance(target, ast.Subscript)
            and private_access is not None
            and private_access(target)
            for target in node.targets
        ):
            raise _Decline
        if isinstance(node, ast.Call):
            load = _plain_scalar_load_pointer(node)
            store = _plain_scalar_store_pointer(node)
            if load is not None or store is not None:
                pointer = load if load is not None else store
                assert pointer is not None
                result.append(
                    _Access(
                        node,
                        pointer,
                        _conjunction(guards),
                        statement,
                        store is not None,
                    )
                )
                for arg in node.args:
                    visit(arg, statement, guards)
                return
            name = ast.unparse(node.func)
            if name not in _CASTS | _PURE and name not in (
                "cute.arch.thread_idx",
                "cute.arch.block_idx",
            ):
                if not (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "bitcast"
                    and len(node.args) == 1
                    and not node.keywords
                ) and not (extra_pure_call is not None and extra_pure_call(node)):
                    raise _Decline
        for child in ast.iter_child_nodes(node):
            visit(child, statement, guards)

    for index, statement in enumerate(body):
        visit(statement, index, [])
    return result


class _Vectorizer:
    def __init__(
        self,
        body: list[ast.stmt],
        *,
        strides: dict[tuple[str, int], int],
        alignments: dict[str, int],
        dtypes: dict[str, str],
        disjoint: set[frozenset[str]],
        constexpr: dict[str, int],
        integer_parameters: frozenset[str] = frozenset(),
        integer_tensors: frozenset[str] = frozenset(),
        private_accesses: frozenset[str] = frozenset(),
        register_accesses: frozenset[str] = frozenset(),
    ) -> None:
        self.strides = strides
        self.alignments = alignments
        self.dtypes = dtypes
        self.disjoint = disjoint
        self.constexpr = constexpr
        # These prove integer kind, not a narrowing width. Any write removes
        # the parameter fact, even when it occurs in a conditional scope.
        writes = {
            name for statement in body for name in _binding_write_roots(statement)
        }
        self.integer_parameters = integer_parameters - writes
        self.integer_tensors = integer_tensors - writes
        self.private_accesses = private_accesses
        self.register_accesses = register_accesses
        self.used = {
            node.id
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name)
        }
        self.changed = 0

    def fresh(self, prefix: str) -> str:
        name = prefix
        while name in self.used:
            name += "_"
        self.used.add(name)
        return name

    def accesses(self, body: list[ast.stmt]) -> list[_Access]:
        if not self.private_accesses and not self.register_accesses:
            return _accesses(body)
        return _accesses(
            body,
            extra_pure_call=lambda call: (
                bool(self.private_accesses)
                and (
                    ast.unparse(call.func) in {"cute.arch.fmax", "cute.arch.fmin"}
                    or ast.unparse(call.func) == "float"
                    and len(call.args) == 1
                    and not call.keywords
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value in {"-inf", "inf", "nan"}
                )
                # The RNG scalar comparison is still an operator call here;
                # these are exactly the already supported ast.Compare forms.
                or bool(self.register_accesses)
                and ast.unparse(call.func)
                in {f"operator.{op}" for op in ("lt", "le", "gt", "ge", "eq", "ne")}
                and len(call.args) == 2
                and not call.keywords
            ),
            private_access=lambda node: (
                ast.dump(node, include_attributes=False)
                in self.private_accesses | self.register_accesses
            ),
        )

    def snapshots(self, body: list[ast.stmt]) -> list[dict[str, ast.expr]] | None:
        return _definition_snapshots(body)

    def memory_width(self, width: int, itemsize: int) -> int:
        if self.register_accesses:
            # An eight-lane RNG packet may need two sixteen-byte transactions.
            # Alignment and the complete-fragment guard still cover all lanes.
            return min(width, 16 // itemsize)
        return width

    def specialize_mask(
        self, loop: ast.For, predicate: ast.expr, lane: str, value: bool
    ) -> ast.For:
        return _specialize_fragment_mask(loop, predicate, lane, value)

    def divisible(
        self,
        node: ast.expr,
        width: int,
        definitions: dict[str, ast.expr],
        multiples: dict[str, int],
        active: frozenset[str] = frozenset(),
    ) -> bool:
        exact = _lane_scale_value(node, self.strides)
        if exact is not None:
            return exact % width == 0
        if isinstance(node, ast.Name):
            if node.id in multiples and multiples[node.id] % width == 0:
                return True
            if node.id in self.constexpr:
                return self.constexpr[node.id] % width == 0
            value = definitions.get(node.id)
            return (
                value is not None
                and node.id not in active
                and self.divisible(
                    value, width, definitions, multiples, active | {node.id}
                )
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return self.divisible(node.operand, width, definitions, multiples, active)
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func)
            in ("cutlass.Int32", "cutlass.Int64", "cutlass.Uint32", "cutlass.Uint64")
            and len(node.args) == 1
            and not node.keywords
        ):
            return self.divisible(node.args[0], width, definitions, multiples, active)
        if isinstance(node, ast.BinOp):
            left = self.divisible(node.left, width, definitions, multiples, active)
            right = self.divisible(node.right, width, definitions, multiples, active)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return left and right
            if isinstance(node.op, ast.Mult):
                return left or right
        return False

    def integer(self, node: ast.expr, definitions: dict[str, ast.expr]) -> int | None:
        value = _expand(
            node,
            {
                **{
                    name: ast.Constant(value=value)
                    for name, value in self.constexpr.items()
                },
                **definitions,
            },
            "",
        )
        return _lane_scale_value(value, self.strides)

    def prefix_divisible(
        self,
        node: ast.expr,
        width: int,
        definitions: dict[str, ast.expr],
        int32_names: frozenset[str],
        active: frozenset[str] = frozenset(),
    ) -> bool:
        """Prove alignment of an unchanged pointer-prefix offset.

        Do not infer divisibility through a float-to-integer conversion:
        Int32(x * 4) need not be divisible by four when x is floating point.
        Separate Int32 multiplication can wrap, but its modulus is a multiple
        of every supported power-of-two vector width.
        """
        exact = _lane_scale_value(node, self.strides)
        if exact is not None:
            return exact % width == 0
        if isinstance(node, ast.Name):
            if node.id in self.constexpr:
                return self.constexpr[node.id] % width == 0
            value = definitions.get(node.id)
            return (
                value is not None
                and node.id not in active
                and self.prefix_divisible(
                    value, width, definitions, int32_names, active | {node.id}
                )
            )
        names = int32_names | frozenset(self.constexpr)
        if not _i32_offset(node, names):
            return False
        if isinstance(node, ast.Call):
            return _i32_offset(node.args[0], names) and self.prefix_divisible(
                node.args[0], width, definitions, int32_names, active
            )
        if isinstance(node, ast.BinOp):
            left = self.prefix_divisible(
                node.left, width, definitions, int32_names, active
            )
            right = self.prefix_divisible(
                node.right, width, definitions, int32_names, active
            )
            if isinstance(node.op, ast.Mult):
                return left or right
            return left and right
        return False

    def pointer_prefix_guards(
        self,
        node: ast.expr,
        width: int,
        lane: str,
        writes: set[str],
        definitions: dict[str, ast.expr],
        int32_names: frozenset[str],
    ) -> list[ast.expr]:
        """Keep independent aligned row/batch additions in pointer arithmetic.

        Combining them into an Int32 scalar offset would introduce a different
        overflow boundary. Retain each original pointer addition and prove its
        alignment independently; only the final offset varies across lanes.
        """
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "iterator"
            and isinstance(node.value, ast.Name)
        ):
            return []
        if (
            not isinstance(node, ast.BinOp)
            or not isinstance(node.op, ast.Add)
            or set(ReadWrites.from_ast(node.right).reads) & (writes | {lane})
            or not _i32_offset(node.right, int32_names | frozenset(self.constexpr))
            or not _integer_expression(
                node.right,
                lane,
                int32_names | self.constexpr.keys(),
                integer_tensors=self.integer_tensors,
            )
        ):
            raise _Decline
        guards = self.pointer_prefix_guards(
            node.left, width, lane, writes, definitions, int32_names
        )
        if not self.prefix_divisible(node.right, width, definitions, int32_names):
            exact = _lane_scale_value(node.right, self.strides)
            if exact is not None:
                raise _Decline
            # Unlike the varying offset's existing alignment guard, this
            # expression may contain a masked conversion or division. The
            # caller appends these checks after every original memory mask.
            guards.append(
                ast.Compare(
                    left=ast.BinOp(
                        left=_clone_expr(node.right),
                        op=ast.Mod(),
                        right=ast.Constant(value=width),
                    ),
                    ops=[ast.Eq()],
                    comparators=[ast.Constant(value=0)],
                )
            )
        return guards

    def guard(
        self,
        mask: ast.expr | None,
        local: dict[str, ast.expr],
        outer: dict[str, ast.expr],
        multiples: dict[str, int],
        lane: str,
        width: int,
        writes: set[str],
        int32_names: frozenset[str],
    ) -> tuple[list[ast.expr], bool]:
        if mask is None:
            return [], True
        expanded = _expand(mask, local, lane)
        result = []
        uniform = True
        for term in _and_terms(expanded):
            if lane not in ReadWrites.from_ast(term).reads:
                candidates = [term]
            elif (
                isinstance(term, ast.Compare)
                and len(term.ops) == 1
                and isinstance(term.ops[0], ast.Lt)
                and lane not in ReadWrites.from_ast(term.comparators[0]).reads
                and (
                    form := _pointer_integer_affine_form(
                        term.left, lane, self.strides, set(), "mask"
                    )
                )
                is not None
                and form[0] == 1
                and _i32_offset(_expand(term.left, outer, lane), int32_names)
                and all(
                    _integer_expression(
                        _expand(value, outer, lane),
                        lane,
                        int32_names | self.constexpr.keys() | self.integer_parameters,
                        integer_tensors=self.integer_tensors,
                    )
                    for value in (term.left, term.comparators[0])
                )
                and self.divisible(
                    _expand(term.left, {}, lane, ast.Constant(value=0)),
                    width,
                    outer,
                    multiples,
                )
                and self.divisible(term.comparators[0], width, outer, multiples)
            ):
                # Two aligned boundaries make this predicate identical for
                # every lane in the fragment, including signed index wrapping.
                candidates = [_expand(term, {}, lane, ast.Constant(value=0))]
            else:
                # Checking all finite lanes also handles non-monotone masks;
                # endpoints alone would be unsound for predicates with holes.
                candidates = [
                    _expand(term, {}, lane, ast.Constant(value=value))
                    for value in range(width)
                ]
                uniform = False
            for candidate in candidates:
                if set(ReadWrites.from_ast(candidate).reads) & writes or any(
                    isinstance(item, ast.Call)
                    and ast.unparse(item.func) not in _CASTS
                    and not (
                        ast.unparse(item.func) in _COORDINATE_CALLS
                        and not item.args
                        and not item.keywords
                    )
                    for item in ast.walk(candidate)
                ):
                    raise _Decline
                result.append(candidate)
        return result, uniform

    def loop(
        self,
        original: ast.For,
        outer: dict[str, ast.expr],
        multiples: dict[str, int],
        int32_names: frozenset[str],
    ) -> list[ast.stmt] | None:
        width = _range_extent(original)
        if (
            not isinstance(original.target, ast.Name)
            or _LANE.fullmatch(original.target.id) is None
            or width not in (2, 4, 8)
        ):
            return None
        lane = original.target.id
        # Work on a private clone. A failed proof leaves the original intact.
        loop = ast.parse(ast.unparse(original)).body[0]
        assert isinstance(loop, ast.For)
        try:
            sites = self.accesses(loop.body)
            if not sites or sum(site.store for site in sites) > 1:
                return None
            roots = [_single_tensor_iterator_root(site.pointer) for site in sites]
            if any(root is None or root not in self.dtypes for root in roots):
                return None
            for left, site in enumerate(sites):
                if site.store and any(
                    index != left
                    and frozenset((roots[left], root)) not in self.disjoint
                    for index, root in enumerate(roots)
                ):
                    return None
            snapshots = self.snapshots(loop.body)
            if snapshots is None:
                return None
            writes = {
                name
                for statement in loop.body
                for name in ReadWrites.from_ast(statement).writes
            }
            before: list[ast.stmt] = []
            after: list[ast.stmt] = []
            guards: list[ast.expr] = []
            alignment_guards: list[ast.expr] = []
            prefix_guards: list[ast.expr] = []
            replacements: dict[int, ast.expr] = {}
            stores: set[int] = set()
            vectors: dict[str, list[str]] = {}
            uniform_masks: list[ast.expr] = []
            inactive_proven = True
            for site, root in zip(sites, roots, strict=True):
                assert root is not None
                dtype = self.dtypes[root]
                info = _DTYPE_INFO.get(dtype)
                if info is None:
                    raise _Decline
                carrier, itemsize, helper = info
                memory_width = self.memory_width(width, itemsize)
                if memory_width * itemsize > 16 or self.alignments.get(root, 1) % (
                    memory_width * itemsize
                ):
                    raise _Decline
                local = snapshots[site.statement]
                pointer = _expand(site.pointer, local, lane)
                if not isinstance(pointer, ast.BinOp) or not isinstance(
                    pointer.op, ast.Add
                ):
                    raise _Decline
                prefix_guards.extend(
                    self.pointer_prefix_guards(
                        pointer.left, width, lane, writes, outer, int32_names
                    )
                )
                form = _pointer_integer_affine_form(
                    pointer.right, lane, self.strides, set(), "pointer"
                )
                # A final integer conversion does not make floating lane
                # arithmetic affine: truncation or rounding can merge lanes.
                # Require the arithmetic before each varying cast to be integral.
                if (
                    form is None
                    or form[0] != 1
                    or not _i32_offset(pointer.right)
                    or not _integer_expression(
                        _expand(pointer.right, outer, lane),
                        lane,
                        int32_names | self.constexpr.keys() | self.integer_parameters,
                        integer_tensors=self.integer_tensors,
                    )
                ):
                    raise _Decline
                offset = _expand(pointer.right, {}, lane, ast.Constant(value=0))
                if set(ReadWrites.from_ast(offset).reads) & writes:
                    raise _Decline
                if not self.divisible(offset, width, outer, multiples):
                    inactive_proven = False
                    alignment_guards.append(
                        ast.Compare(
                            left=ast.BinOp(
                                left=_clone_expr(offset),
                                op=ast.Mod(),
                                right=ast.Constant(value=width),
                            ),
                            ops=[ast.Eq()],
                            comparators=[ast.Constant(value=0)],
                        )
                    )
                site_guards, uniform = self.guard(
                    site.mask,
                    local,
                    outer,
                    multiples,
                    lane,
                    width,
                    writes,
                    int32_names,
                )
                guards.extend(site_guards)
                inactive_proven &= uniform and site.mask is not None
                if site.mask is not None:
                    uniform_masks.append(_expand(site.mask, local, lane))
                base = ast.BinOp(
                    left=_clone_expr(pointer.left), op=ast.Add(), right=offset
                )
                if site.store:
                    values = self.fresh("_helion_affine_store")
                    before.append(
                        ast.Assign(
                            targets=[ast.Name(id=values, ctx=ast.Store())],
                            value=ast.List(elts=[], ctx=ast.Load()),
                        )
                    )
                    replacements[id(site.call)] = ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=values, ctx=ast.Load()),
                            attr="append",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Call(
                                func=ast.Attribute(
                                    value=site.call.args[0],
                                    attr="bitcast",
                                    ctx=ast.Load(),
                                ),
                                args=[ast.parse(carrier, mode="eval").body],
                                keywords=[],
                            )
                        ],
                        keywords=[],
                    )
                    stores.add(id(site.call))
                    for start in range(0, width, memory_width):
                        pointer = ast.unparse(base)
                        selected = values
                        if memory_width != width:
                            pointer = f"({pointer}) + {start}"
                            selected = f"{values}[{start}:{start + memory_width}]"
                        after.extend(ast.parse(f"{helper}({pointer}, {selected})").body)
                else:
                    key = ast.dump(base, include_attributes=False) + dtype
                    loaded = vectors.get(key)
                    if loaded is None:
                        loaded = []
                        vectors[key] = loaded
                        for start in range(0, width, memory_width):
                            vector = self.fresh("_helion_affine_load")
                            loaded.append(vector)
                            pointer = ast.unparse(base)
                            if start:
                                pointer = f"({pointer}) + {start}"
                            before.extend(
                                ast.parse(
                                    f"{vector} = cute.arch.load({pointer}, ir.VectorType.get([{memory_width}], {carrier}.mlir_type))"
                                ).body
                            )
                    selected = (
                        f"{loaded[0]}[{lane}]"
                        if len(loaded) == 1
                        else f"({', '.join(loaded)})[{lane} // {memory_width}][{lane} % {memory_width}]"
                    )
                    replacements[id(site.call)] = ast.parse(
                        f"{carrier}({selected}).bitcast({dtype})", mode="eval"
                    ).body

            class Replace(ast.NodeTransformer):
                def visit_Call(self, node: ast.Call) -> ast.AST:
                    replacement = replacements.get(id(node))
                    if replacement is not None:
                        return replacement
                    return self.generic_visit(node)

                def visit_If(self, node: ast.If) -> ast.AST | list[ast.stmt]:
                    if (
                        len(node.body) == 1
                        and isinstance(node.body[0], ast.Expr)
                        and id(node.body[0].value) in stores
                    ):
                        return [self.visit(node.body[0])]
                    return self.generic_visit(node)

            # When every access has the same lane-uniform predicate, a false
            # guard means all lanes are inactive, not a partially active tail.
            # Fold that predicate in the inactive path, retaining every scalar
            # live-out and the original zero arithmetic.
            fallback = original
            if (
                inactive_proven
                and not prefix_guards
                and len(
                    {ast.dump(mask, include_attributes=False) for mask in uniform_masks}
                )
                == 1
            ):
                specialized = self.specialize_mask(
                    original, uniform_masks[0], lane, False
                )
                if not self.accesses(specialized.body):
                    fallback = specialized
            rewritten = Replace().visit(loop)
            assert isinstance(rewritten, ast.For)
            # The vector guard has checked every memory predicate for every
            # finite lane, even when the predicate is not lane-uniform. Inside
            # this branch each is true, so remove redundant scalar selects.
            for mask in uniform_masks:
                rewritten = self.specialize_mask(rewritten, mask, lane, True)
            fast = [*before, rewritten, *after]
            # Both final offsets and pointer prefixes may contain arithmetic
            # protected by the original memory masks. Prove all lanes active
            # before evaluating either alignment check.
            guard = _conjunction([*guards, *alignment_guards, *prefix_guards])
            self.changed += 1
            return (
                fast
                if guard is None
                else [ast.If(test=guard, body=fast, orelse=[fallback])]
            )
        except _Decline:
            return None

    def body(
        self,
        statements: list[ast.stmt],
        outer: dict[str, ast.expr] | None = None,
        multiples: dict[str, int] | None = None,
        int32_names: frozenset[str] = frozenset(),
    ) -> list[ast.stmt]:
        outer = dict(outer or {})
        multiples = dict(multiples or {})
        snapshots = self.snapshots(statements)
        if snapshots is None:
            return statements
        result = []
        for index, statement in enumerate(statements):
            definitions = dict(outer)
            active_int32_names = set(int32_names)
            active_multiples = dict(multiples)
            # Outer definitions whose bindings are overwritten in this scope
            # are not valid after that write, even if it occurred in a branch.
            for previous in statements[:index]:
                writes = _binding_write_roots(previous)
                _forget_definitions(definitions, writes)
                for name in writes:
                    active_int32_names.discard(name)
                    active_multiples.pop(name, None)
            definitions.update(snapshots[index])
            if isinstance(statement, ast.For):
                replacement = self.loop(
                    statement,
                    definitions,
                    active_multiples,
                    frozenset(active_int32_names),
                )
                if replacement is not None:
                    result.extend(replacement)
                    continue
                child_definitions = dict(definitions)
                child_multiples = dict(active_multiples)
                child_int32_names = set(active_int32_names)
                # A later write in the loop may reach its next iteration.
                # Forget enclosing facts for all loop-carried bindings and
                # shadowed targets before learning fresh induction facts.
                loop_writes = _binding_write_roots(statement)
                _forget_definitions(child_definitions, loop_writes)
                for name in loop_writes:
                    child_multiples.pop(name, None)
                    child_int32_names.discard(name)
                if (
                    isinstance(statement.target, ast.Name)
                    and isinstance(statement.iter, ast.Call)
                    and isinstance(statement.iter.func, ast.Name)
                    and statement.iter.func.id == "range"
                ):
                    args = statement.iter.args
                    if len(args) in (1, 2, 3):
                        if all(
                            _i32_offset(
                                _expand(arg, definitions, ""),
                                frozenset(active_int32_names),
                            )
                            for arg in args
                        ):
                            child_int32_names.add(statement.target.id)
                        start = (
                            self.integer(args[0], definitions) if len(args) > 1 else 0
                        )
                        step = (
                            self.integer(args[2], definitions) if len(args) == 3 else 1
                        )
                        if start is not None and step is not None:
                            child_multiples[statement.target.id] = math.gcd(start, step)
                statement.body = self.body(
                    statement.body,
                    child_definitions,
                    child_multiples,
                    frozenset(child_int32_names),
                )
            elif isinstance(statement, ast.If):
                statement.body = self.body(
                    statement.body,
                    definitions,
                    active_multiples,
                    frozenset(active_int32_names),
                )
                statement.orelse = self.body(
                    statement.orelse,
                    definitions,
                    active_multiples,
                    frozenset(active_int32_names),
                )
            result.append(statement)
        return result


def vectorize_affine_tile_lanes(
    body: list[ast.stmt],
    device_function: DeviceFunction,
    constexpr_values: dict[str, int],
    *,
    register_accesses: frozenset[str] = frozenset(),
    integer_names: frozenset[str] = frozenset(),
) -> list[ast.stmt]:
    from ..device_function import TensorArg
    from ..device_function import TensorPropertyArg

    env = CompileEnvironment.current()
    inputs = list(env.input_sources)
    input_storages = {id(tensor.untyped_storage()) for tensor in inputs}
    strides = device_function.proven_tensor_stride_values()
    alignments = {}
    dtypes = {}
    for arg in device_function.arguments:
        if not isinstance(arg, TensorArg) or arg.fake_value.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            continue
        tensor = arg.fake_value
        dtypes[arg.name] = env.backend.dtype_str(tensor.dtype)
        for dim, stride in enumerate(tensor.stride()):
            if isinstance(stride, int):
                strides[arg.name, dim] = stride
        storage = id(tensor.untyped_storage())
        offset = tensor.storage_offset() * tensor.element_size()
        for alignment in (16, 8, 4):
            aligned = storage not in input_storages and statically_known_true(
                offset % alignment == 0
            )
            if not aligned:
                aligned = any(
                    id(source.untyped_storage()) == storage
                    and tensor_has_specialized_base_alignment(env, source, alignment)
                    and statically_known_true(
                        (offset - source.storage_offset() * source.element_size())
                        % alignment
                        == 0
                    )
                    for source in inputs
                )
            if aligned:
                alignments[arg.name] = alignment
                break
    if env.config_spec.pointwise_facts:
        from .pure_lane_packets import _PacketVectorizer

        packetizer = _PacketVectorizer(
            body,
            strides=strides,
            alignments=alignments,
            dtypes=dtypes,
            disjoint=device_function.proven_disjoint_tensor_pairs(),
            constexpr=constexpr_values,
        )
        body = packetizer.body(body)
    private_accesses: frozenset[str] = frozenset()
    renames = {
        name: aliases[0] for name, aliases in device_function._variable_renames.items()
    }
    renamed_writes = {
        renames.get(name, name)
        for statement in body
        for name in _binding_write_roots(statement)
    }
    integer_parameters = (
        {
            argument.name
            for expression, argument in device_function._expr_args.items()
            if expression.assumptions0.get("integer") is True
        }
        | {
            argument.name
            for argument in device_function.arguments
            if isinstance(argument, TensorPropertyArg)
        }
        | integer_names
    )
    integer_tensors = {
        argument.name
        for argument in device_function.arguments
        if isinstance(argument, TensorArg)
        and argument.fake_value.dtype
        in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        )
    }
    vectorizer = _Vectorizer(
        body,
        strides=strides,
        alignments=alignments,
        dtypes=dtypes,
        disjoint=device_function.proven_disjoint_tensor_pairs(),
        constexpr=constexpr_values,
        integer_parameters=frozenset(
            name
            for name in integer_parameters
            if renames.get(name, name) not in renamed_writes
        ),
        integer_tensors=frozenset(
            name
            for name in integer_tensors
            if renames.get(name, name) not in renamed_writes
        ),
        private_accesses=private_accesses,
        register_accesses=register_accesses,
    )
    return ast.fix_missing_locations(
        ast.Module(body=vectorizer.body(body), type_ignores=[])
    ).body

"""Stage independent pointwise lanes between vector loads and vector stores.

Keep the scalar program and its lane order unchanged. Only ordinary reads and
one disjoint output store move across lanes. The existing affine vectorizer
owns address reconstruction and its full-fragment/scalar-fallback selection.
"""

from __future__ import annotations

import ast
from collections import Counter
import re
from typing import TYPE_CHECKING

from .affine_vector_io import _Access
from .affine_vector_io import _accesses
from .affine_vector_io import _Decline
from .affine_vector_io import _Vectorizer
from .persistent_branch_vec import _binding_write_roots
from .persistent_branch_vec import _definition_snapshots
from .persistent_branch_vec import _expand
from .persistent_branch_vec import _lane_scale_value
from .persistent_branch_vec import _needed_definition_names
from .persistent_branch_vec import _plain_scalar_load_pointer
from .persistent_branch_vec import _plain_scalar_store_pointer
from .persistent_branch_vec import _range_extent
from .scalar_integer import _INTEGER_CASTS
from .scalar_integer import _integer_expression
from .scalar_recipe import _GLOBALS
from .scalar_recipe import _OPERATOR_CALLS
from .scalar_recipe import _NotRecipe
from .scalar_recipe import _PureExpression
from .scalar_recipe import _read_names

if TYPE_CHECKING:
    from collections.abc import Set as AbstractSet

_PACKET_GLOBALS = _GLOBALS | {"_cute_inline_asm_elementwise", "ir"}
_PTX_ARITHMETIC = frozenset(
    [
        "abs",
        "add",
        "and",
        "bfe",
        "bfi",
        "brev",
        "clz",
        "cos",
        "cvt",
        "div",
        "ex2",
        "fma",
        "lg2",
        "mad",
        "max",
        "min",
        "mov",
        "mul",
        "neg",
        "not",
        "or",
        "popc",
        "rcp",
        "rem",
        "rsqrt",
        "selp",
        "set",
        "setp",
        "shl",
        "shr",
        "sin",
        "sqrt",
        "sub",
        "xor",
    ]
)
_OPERATORS = frozenset(f"operator.{operation}" for operation in _OPERATOR_CALLS)


def _pure_packet_call(call: ast.Call) -> bool:
    name = ast.unparse(call.func)
    if name in _OPERATORS:
        return True
    if name != "_cute_inline_asm_elementwise" or len(call.args) != 1:
        return False
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    if set(keywords) != {"asm", "constraints", "dtype", "is_pure"}:
        return False
    pure, asm, constraints = (
        keywords["is_pure"],
        keywords["asm"],
        keywords["constraints"],
    )
    return (
        isinstance(pure, ast.Constant)
        and isinstance(asm, ast.Constant)
        and isinstance(constraints, ast.Constant)
        and is_pure_scalar_assembly(asm.value, constraints.value, pure.value)
    )


def is_pure_scalar_assembly(asm: object, constraints: object, is_pure: object) -> bool:
    if not (
        is_pure is True
        and isinstance(asm, str)
        and isinstance(constraints, str)
        and re.fullmatch(r"[=+&rlhfdn, ]+", constraints)
        and len(asm) <= 16384
    ):
        return False
    from .philox_stream import SCALAR_ASM
    from .philox_stream import SCALAR_CONSTRAINTS

    if asm == SCALAR_ASM and constraints == SCALAR_CONSTRAINTS:
        return True
    # A declared-pure asm can still describe a memory read. Accept only scalar
    # arithmetic instructions; no pointers, special registers, branches,
    # barriers, memory clobbers, or opaque instructions enter this envelope.
    text = asm.strip().removeprefix("{").removesuffix("}").strip()
    if not re.fullmatch(r"[A-Za-z0-9_$.,+;\s-]*", text):
        return False
    for instruction in text.split(";"):
        instruction = instruction.strip()
        if not instruction:
            continue
        if instruction.startswith(".reg "):
            if re.fullmatch(
                r"\.reg\s+\.[busf](8|16|32|64)\s+[A-Za-z_][\w, ]*", instruction
            ):
                continue
            return False
        if instruction.split(maxsplit=1)[0].split(".")[0] not in _PTX_ARITHMETIC:
            return False
    return True


class _PacketExpression(_PureExpression):
    def visit_Call(self, node: ast.Call) -> None:
        if _pure_packet_call(node):
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        super().visit_Call(node)


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
            names.update(_read_names(node.iter))
        if isinstance(node, ast.Call):
            pointer = _plain_scalar_load_pointer(node)
            if pointer is None:
                pointer = _plain_scalar_store_pointer(node)
            if pointer is not None:
                names.update(_read_names(pointer))
                for guard in guards:
                    names.update(_read_names(guard))
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


def _scalar_leaf(node: ast.expr) -> bool:
    return isinstance(node, (ast.Name, ast.Constant)) or (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, (ast.Name, ast.Constant))
    )


def _safe_inactive_pointer(node: ast.expr, integer_names: AbstractSet[str]) -> bool:
    """Do not speculate operations undefined for an inactive lane's inputs.

    Keep the packet proof's conservative envelope of modular arithmetic,
    even though the shared vectorizer now checks memory predicates before
    alignment. Integer kind alone does not prove defined division, remainder,
    or shifts for arbitrary inactive inputs.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.BinOp) and not isinstance(
            child.op, (ast.Add, ast.Sub, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)
        ):
            return False
        if (
            isinstance(child, ast.Call)
            and ast.unparse(child.func) in _INTEGER_CASTS
            and len(child.args) == 1
            and isinstance(child.args[0], ast.Name)
            and child.args[0].id not in integer_names
        ):
            # A direct conversion proves the resulting type, but speculating
            # a float input could newly evaluate a NaN/out-of-range cast.
            return False
    return True


def _exact_predicate(node: ast.expr, lane: str, known: AbstractSet[str]) -> bool:
    """Replay only scalar leaves, comparisons, and proved integer arithmetic.

    Expanding floating assignments into a predicate could introduce an FMA
    absent from the original predicate. A captured scalar itself retains its
    value; keep those simple comparisons without replaying float arithmetic.
    """
    if _scalar_leaf(node):
        return True
    if isinstance(node, ast.Compare):
        return all(
            _exact_predicate(value, lane, known)
            for value in (node.left, *node.comparators)
        )
    if isinstance(node, ast.BoolOp):
        return all(_exact_predicate(value, lane, known) for value in node.values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _exact_predicate(node.operand, lane, known)
    if isinstance(node, ast.Call) and not node.keywords:
        name = ast.unparse(node.func)
        if name == "cutlass.Boolean" and len(node.args) == 1:
            return _exact_predicate(node.args[0], lane, known)
        if (
            name in {"cutlass.Float32", "cutlass.Float16", "cutlass.BFloat16"}
            and len(node.args) == 1
        ):
            return _scalar_leaf(node.args[0])
    return _integer_expression(node, lane, known)


class _PacketVectorizer(_Vectorizer):
    def __init__(
        self,
        body: list[ast.stmt],
        *,
        strides: dict[tuple[str, int], int],
        alignments: dict[str, int],
        dtypes: dict[str, str],
        disjoint: set[frozenset[str]],
        constexpr: dict[str, int],
    ) -> None:
        super().__init__(
            body,
            strides=strides,
            alignments=alignments,
            dtypes=dtypes,
            disjoint=disjoint,
            constexpr=constexpr,
        )
        self.read_counts = Counter(
            node.id
            for statement in body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
        self.shadowed = bool(self.dtypes.keys() & _PACKET_GLOBALS) or any(
            _binding_write_roots(statement) & (_PACKET_GLOBALS | self.dtypes.keys())
            for statement in body
        )

    def accesses(self, body: list[ast.stmt]) -> list[_Access]:
        return _accesses(body, _pure_packet_call)

    def snapshots(self, body: list[ast.stmt]) -> list[dict[str, ast.expr]] | None:
        return _definition_snapshots(
            body, required_names=_memory_definition_names(body)
        )

    def memory_width(self, width: int, itemsize: int) -> int:
        return min(width, 16 // itemsize)

    def divisible(
        self,
        node: ast.expr,
        width: int,
        definitions: dict[str, ast.expr],
        multiples: dict[str, int],
        active: frozenset[str] = frozenset(),
    ) -> bool:
        value = _lane_scale_value(node, self.strides)
        # Dynamic alignment is checked on the exact original offset. Do not
        # infer divisibility through a potentially floating intermediate.
        return value is not None and value % width == 0

    def specialize_mask(
        self, loop: ast.For, predicate: ast.expr, lane: str, value: bool
    ) -> ast.For:
        # Keep every arithmetic use and live value of numeric predicates.
        # Memory operations themselves have already been replaced/guarded.
        return loop

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
            return [], False
        expression = _expand(mask, local, lane)
        if not _exact_predicate(
            _expand(expression, outer, lane), lane, int32_names | self.constexpr.keys()
        ):
            raise _Decline
        guards = []
        for value in range(width):
            guard = _expand(expression, {}, lane, ast.Constant(value))
            if _read_names(guard) & writes:
                raise _Decline
            try:
                pure = _PacketExpression()
                pure.visit(guard)
                if pure.reads_memory:
                    raise _Decline
            except _NotRecipe:
                raise _Decline from None
            guards.append(guard)
        return guards, False

    def loop(
        self,
        original: ast.For,
        outer: dict[str, ast.expr],
        multiples: dict[str, int],
        int32_names: frozenset[str],
    ) -> list[ast.stmt] | None:
        if (
            self.shadowed
            or not isinstance(original.target, ast.Name)
            or _range_extent(original) not in (4, 8)
            or original.orelse
        ):
            return None
        writes = _binding_write_roots(original)
        inside = Counter(
            node.id
            for node in ast.walk(original)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        )
        if any(self.read_counts[name] > inside[name] for name in writes):
            return None
        available = {original.target.id}
        try:
            sites = self.accesses(original.body)
            if sum(site.store for site in sites) != 1:
                return None
            for statement in original.body:
                if _read_names(statement) & (writes - available):
                    return None
                if isinstance(statement, ast.Assign):
                    if len(statement.targets) != 1 or not isinstance(
                        statement.targets[0], ast.Name
                    ):
                        return None
                    target = statement.targets[0].id
                    if target in available:
                        return None
                    _PacketExpression().visit(statement.value)
                    available.add(target)
                elif isinstance(statement, (ast.If, ast.Expr)):
                    # _accesses validates the sole conditional store's shape.
                    if statement is not original.body[-1]:
                        return None
                elif not isinstance(statement, ast.Pass):
                    return None
            snapshots = self.snapshots(original.body)
            if snapshots is None:
                return None
            for site in sites:
                if site.store:
                    _PacketExpression().visit(site.call.args[0])
                pointer = _expand(
                    site.pointer, snapshots[site.statement], original.target.id
                )
                pointer = _expand(pointer, outer, original.target.id)
                if not (
                    isinstance(pointer, ast.BinOp)
                    and isinstance(pointer.op, ast.Add)
                    and _safe_inactive_pointer(
                        pointer.right,
                        int32_names | self.constexpr.keys() | {original.target.id},
                    )
                    and _integer_expression(
                        pointer.right,
                        original.target.id,
                        int32_names | self.constexpr.keys(),
                    )
                ):
                    return None
        except (_Decline, _NotRecipe):
            return None
        return super().loop(original, outer, multiples, int32_names)

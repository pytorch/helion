"""Conservative scalar AST recipes for CuTe collective operand staging.

The caller supplies a dominating statement sequence and the names available at
the replay site, including every coordinate it intends to replace.  This module
does not prove that moving a recipe across a control-flow or alias boundary is
legal; the caller must keep the replay in the same proven memory/control region.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast

from ..ast_extension import ExtendedAST

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

_NUMERIC_TYPES = frozenset(
    {
        "Boolean",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Uint8",
        "Uint16",
        "Uint32",
        "Uint64",
        "BFloat16",
        "Float16",
        "Float32",
        "Float64",
        "Float8E4M3FN",
        "Float8E5M2",
        "Float8E8M0FNU",
        "Float4E2M1FN",
    }
)
_MATH_CALLS = frozenset(
    {
        "absf",
        "acos",
        "asin",
        "atan",
        "atan2",
        "ceil",
        "copysign",
        "cos",
        "cosh",
        "div",
        "erf",
        "exp",
        "exp2",
        "fabs",
        "floor",
        "fma",
        "fmax",
        "fmin",
        "fmod",
        "isfinite",
        "isinf",
        "isnan",
        "ldexp",
        "log",
        "log2",
        "log10",
        "max",
        "min",
        "pow",
        "rcp",
        "rsqrt",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "trunc",
    }
)
_INDEX_CALLS = frozenset(
    {
        "block_idx",
        "thread_idx",
        "lane_idx",
        "warp_idx",
        "block_dim",
        "grid_dim",
        "cluster_idx",
        "cluster_dim",
        "cluster_rank",
    }
)
_OPERATOR_CALLS = frozenset(
    {
        "abs",
        "add",
        "and_",
        "eq",
        "floordiv",
        "ge",
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
_BUILTINS = frozenset({"abs", "bool", "float", "int", "max", "min", "round"})
_HELPERS = frozenset(
    {"_cute_float4_e2m1fn_x2_to_float32", "_cute_fp8e4m3fn_to_float32"}
)
_GLOBALS = _BUILTINS | _HELPERS | {"cutlass", "cute", "math", "operator"}
_METADATA = frozenset({"iterator", "layout", "shape", "stride", "element_type"})
_AstT = TypeVar("_AstT", bound=ast.AST)


class _NotRecipe(Exception):
    pass


def _clone(node: _AstT) -> _AstT:
    """Copy ordinary and ExtendedAST nodes without invoking pickle/deepcopy."""

    def clone_field(value: object) -> object:
        if isinstance(value, ast.AST):
            return _clone(value)
        if isinstance(value, list):
            return [clone_field(item) for item in value]
        return value

    fields = {name: clone_field(value) for name, value in ast.iter_fields(node)}
    if isinstance(node, ExtendedAST):
        return cast("_AstT", node.new(fields))
    return ast.copy_location(type(node)(**fields), node)


def _path(node: ast.expr) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        prefix = _path(node.value)
        if prefix is not None:
            return (*prefix, node.attr)
    return None


def _numeric_type(node: ast.expr) -> bool:
    path = _path(node)
    return (
        path is not None
        and len(path) == 2
        and path[0] == "cutlass"
        and path[1] in _NUMERIC_TYPES
    )


class _PureExpression(ast.NodeVisitor):
    """Validate the small scalar language, tracking ordinary memory reads."""

    def __init__(self) -> None:
        self.reads_memory = False

    def generic_visit(self, node: ast.AST) -> None:
        raise _NotRecipe

    def visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            raise _NotRecipe

    def visit_Constant(self, node: ast.Constant) -> None:
        pass

    def visit_Tuple(self, node: ast.Tuple) -> None:
        if not isinstance(node.ctx, ast.Load):
            raise _NotRecipe
        for item in node.elts:
            self.visit(item)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not isinstance(node.ctx, ast.Load):
            raise _NotRecipe
        if node.attr in _METADATA or _numeric_type(node):
            self.visit(node.value)
            return
        if _path(node) in {("math", "inf"), ("math", "nan"), ("math", "pi")}:
            self.visit(node.value)
            return
        raise _NotRecipe

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if not isinstance(node.ctx, ast.Load):
            raise _NotRecipe
        # A named container may be a tensor, so conservatively classify it as
        # a read. Metadata and architecture index tuples are immutable.
        path = _path(node.value.func) if isinstance(node.value, ast.Call) else None
        immutable = (
            isinstance(node.value, ast.Tuple)
            or isinstance(node.value, ast.Attribute)
            and node.value.attr in {"shape", "stride"}
            or path is not None
            and path[:2] == ("cute", "arch")
            and path[-1] in _INDEX_CALLS
        )
        self.reads_memory |= not immutable
        self.visit(node.value)
        self.visit(node.slice)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            raise _NotRecipe
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        self.visit(node.operand)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        for value in node.values:
            self.visit(value)

    def visit_Compare(self, node: ast.Compare) -> None:
        if any(isinstance(op, (ast.In, ast.NotIn)) for op in node.ops):
            raise _NotRecipe
        self.visit(node.left)
        for value in node.comparators:
            self.visit(value)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Call(self, node: ast.Call) -> None:
        path = _path(node.func)
        pure_global = path is not None and (
            len(path) == 1
            and path[0] in _BUILTINS | _HELPERS
            or len(path) == 2
            and path[0] == "cutlass"
            and path[1] in _NUMERIC_TYPES | {"min", "max"}
            or len(path) == 2
            and path[0] == "math"
            and path[1] in _MATH_CALLS
            or len(path) == 2
            and path[0] == "operator"
            and path[1] in _OPERATOR_CALLS
            or len(path) == 3
            and path[:2] == ("cute", "math")
            and path[2] in _MATH_CALLS
            or path == ("cute", "where")
            or path == ("cute", "crd2idx")
            and len(node.args) == 2
            and not node.keywords
        )
        index_call = (
            path is not None
            and len(path) == 3
            and path[:2] == ("cute", "arch")
            and path[2] in _INDEX_CALLS
            and not node.args
            and not node.keywords
        )
        if (
            path == ("cute", "arch", "load")
            and len(node.args) == 2
            and not node.keywords
            and _numeric_type(node.args[1])
        ):
            self.reads_memory = True
        elif pure_global or index_call:
            # The global root's reaching definition is checked separately.
            pass
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr == "load" and not node.args and not node.keywords:
                self.reads_memory = True
                self.visit(node.func.value)
            elif (
                node.func.attr in {"bitcast", "to"}
                and len(node.args) == 1
                and not node.keywords
                and _numeric_type(node.args[0])
            ):
                self.visit(node.func.value)
            else:
                raise _NotRecipe
        else:
            raise _NotRecipe
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise _NotRecipe
            self.visit(keyword.value)


def _memory_reads(value: ast.expr) -> bool:
    visitor = _PureExpression()
    visitor.visit(value)
    return visitor.reads_memory


def _read_names(node: ast.AST) -> set[str]:
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


@dataclasses.dataclass(frozen=True, eq=False)
class _Definition:
    name: str
    order: int
    statement: ast.Assign | None
    references: Mapping[str, _Definition]
    dependencies: frozenset[str]
    epoch: int
    reads_memory: bool = False


@dataclasses.dataclass(frozen=True)
class _Expression:
    value: ast.expr
    references: Mapping[str, str | _Definition]


@dataclasses.dataclass(frozen=True)
class ScalarRecipe:
    """A proven dependency slice; AST inputs and emitted copies stay independent."""

    boundary_names: frozenset[str]
    _assignments: tuple[tuple[_Definition, _Expression], ...]
    _result: _Expression

    @property
    def source_statement_indices(self) -> tuple[int, ...]:
        """Pure assignments selected from the supplied dominating sequence."""
        return tuple(definition.order for definition, _ in self._assignments)

    def emit(
        self,
        replacements: Mapping[str, ast.expr],
        fresh_name: Callable[[str], str],
    ) -> tuple[list[ast.Assign], ast.expr]:
        """Replay once per definition, substituting boundary coordinates only.

        ``fresh_name(hint)`` must return a name unused by the insertion scope.
        Unused replacement keys are harmless. Replacement trees are copied and
        are not recursively substituted into one another.
        """
        names: dict[_Definition, str] = {}
        occupied = set(self.boundary_names | _GLOBALS)
        for value in replacements.values():
            occupied.update(_read_names(value))
        statements: list[ast.Assign] = []
        for definition, expression in self._assignments:
            name = fresh_name(f"{definition.name}_remat")
            if not name.isidentifier() or name in occupied:
                raise ValueError("fresh_name must return an unused Python identifier")
            occupied.add(name)
            names[definition] = name
            assert definition.statement is not None
            statement = _clone(definition.statement)
            target = statement.targets[0]
            assert isinstance(target, ast.Name)
            target.id = name
            statement.value = _emit_expression(expression, names, replacements)
            statements.append(ast.fix_missing_locations(statement))
        return statements, _emit_expression(self._result, names, replacements)


def _emit_expression(
    expression: _Expression,
    names: Mapping[_Definition, str],
    replacements: Mapping[str, ast.expr],
) -> ast.expr:
    class Rewrite(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            reference = expression.references[node.id]
            if isinstance(reference, _Definition):
                replacement: ast.expr = _clone(node)
                assert isinstance(replacement, ast.Name)
                replacement.id = names[reference]
            elif reference in replacements and reference not in _GLOBALS:
                replacement = _clone(replacements[reference])
            else:
                replacement = _clone(node)
            return ast.copy_location(replacement, node)

    result = Rewrite().visit(_clone(expression.value))
    assert isinstance(result, ast.expr)
    return ast.fix_missing_locations(result)


def build_recipe(
    value_expr: ast.expr,
    dominating_statements: Sequence[ast.stmt],
    boundary_names: Collection[str],
    *,
    mutable_names: Collection[str] = (),
) -> ScalarRecipe | None:
    """Capture a scalar value using its precise reaching definitions.

    Names listed in ``boundary_names`` stop expansion at their *current*
    definition. Include kernel arguments, constexprs, tile origins, and names
    whose coordinates will be replaced. Earlier shadowed versions are resolved
    through their original definition, never through the final name binding.

    Only single-name assignments are replayed. Control flow and unknown calls
    cannot supply definitions. Stores, barriers, and other unknown statements
    invalidate preceding memory reads. Explicit ``mutable_names`` and detected
    scalar recurrences are never replayed, even when also listed as boundaries.
    The caller should list accumulators whose updates lie outside this prefix.
    """
    boundaries = frozenset(boundary_names)
    mutable = set(mutable_names)
    environment: dict[str, _Definition] = {}
    epoch = 0

    def lookup(name: str) -> _Definition:
        if name not in environment:
            environment[name] = _Definition(name, -1, None, {}, frozenset(), 0)
        return environment[name]

    globals_at_entry = {name: lookup(name) for name in _GLOBALS}

    def reads_memory_here(value: ast.expr) -> bool:
        result = _memory_reads(value)
        if any(
            lookup(name) is not globals_at_entry[name]
            for name in _read_names(value) & _GLOBALS
        ):
            raise _NotRecipe
        return result

    for order, statement in enumerate(dominating_statements):
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            name = statement.targets[0].id
            references = {key: lookup(key) for key in _read_names(statement.value)}
            dependencies = frozenset(references).union(
                *(item.dependencies for item in references.values())
            )
            if name in dependencies:
                mutable.add(name)
            try:
                reads_memory = reads_memory_here(statement.value)
            except _NotRecipe:
                epoch += 1
                environment[name] = _Definition(
                    name, order, None, references, dependencies, epoch
                )
            else:
                environment[name] = _Definition(
                    name,
                    order,
                    _clone(statement),
                    references,
                    dependencies,
                    epoch,
                    reads_memory,
                )
            continue
        if isinstance(statement, ast.Expr):
            try:
                reads_memory_here(statement.value)
            except _NotRecipe:
                pass
            else:
                continue
        # Never flatten a branch/loop into the dominating sequence. Its writes
        # kill reaching definitions even if the statement is otherwise unused.
        epoch += 1
        for node in ast.walk(statement):
            if (
                isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Import,
                        ast.ImportFrom,
                        ast.Global,
                        ast.Nonlocal,
                    ),
                )
                or isinstance(node, ast.Attribute)
                and isinstance(node.ctx, (ast.Store, ast.Del))
            ):
                return None
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                mutable.add(node.target.id)
            if isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                environment[node.id] = _Definition(
                    node.id, order, None, {}, frozenset(), epoch
                )

    assignments: dict[_Definition, _Expression] = {}
    required_boundaries: set[str] = set()

    def resolve(definition: _Definition) -> str | _Definition:
        name = definition.name
        if name in mutable:
            raise _NotRecipe
        if name in globals_at_entry:
            if (
                definition is not globals_at_entry[name]
                or lookup(name) is not definition
            ):
                raise _NotRecipe
            return name
        if name in boundaries and lookup(name) is definition:
            required_boundaries.add(name)
            return name
        if definition.statement is None:
            raise _NotRecipe
        if definition.reads_memory and definition.epoch != epoch:
            raise _NotRecipe
        if definition not in assignments:
            references = {
                key: resolve(value) for key, value in definition.references.items()
            }
            assignments[definition] = _Expression(
                definition.statement.value, references
            )
        return definition

    try:
        reads_memory_here(value_expr)
        references = {name: resolve(lookup(name)) for name in _read_names(value_expr)}
    except _NotRecipe:
        return None
    ordered = tuple(sorted(assignments.items(), key=lambda item: item[0].order))
    return ScalarRecipe(
        frozenset(required_boundaries),
        ordered,
        _Expression(_clone(value_expr), references),
    )

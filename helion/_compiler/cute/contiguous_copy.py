"""Optional 128-bit copies for proven contiguous scalar operand recipes.

The caller owns the shared layout, vector coordinates, alias proof, and async
commit/wait/barrier schedule. This helper only replaces a vector of scalar copies at
one aligned coordinate by a direct global-to-shared copy when every lane has
the original load predicate. Partial vectors retain the original scalar code.
"""

from __future__ import annotations

import ast
import dataclasses
import itertools
import math
from typing import TYPE_CHECKING
from typing import Literal

from .scalar_recipe import _clone
from .scalar_recipe import _read_names
from .scalar_recipe import build_recipe

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

_INTEGER_CASTS = frozenset({"Int32", "Int64", "Uint32", "Uint64"})
_DTYPE_BYTES = {"cutlass.Float16": 2, "cutlass.BFloat16": 2, "cutlass.Float32": 4}
_DTYPES = frozenset(_DTYPE_BYTES)
_MAX_ANALYSIS_NODES = 4096
_MAX_ANALYSIS_DEPTH = 64
_ValueKind = Literal["integer", "pointer"] | None


@dataclasses.dataclass(frozen=True)
class CopyTensorFacts:
    """Facts guaranteed by the caller's tensor specialization/guards."""

    dtype: str
    strides: tuple[int, ...]
    alignment_bytes: int


def _path(value: ast.expr) -> str | None:
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        base = _path(value.value)
        if base is not None:
            return f"{base}.{value.attr}"
    return None


def _replace(value: ast.AST, replacements: Mapping[str, ast.expr]) -> ast.AST:
    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id in replacements:
                result = _clone(replacements[node.id])
                if isinstance(result, ast.Name):
                    result.ctx = node.ctx
                return result
            return node

    return ast.fix_missing_locations(Substitute().visit(_clone(value)))


def _expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def _within_analysis_budget(statements: Sequence[ast.Assign], value: ast.expr) -> bool:
    """Bound the original syntax and dependency depth before recursive proofs."""
    nodes = 0
    depths: dict[str, int] = {}
    for expression, target in itertools.chain(
        ((statement.value, statement.targets[0]) for statement in statements),
        ((value, None),),
    ):
        depth = 0
        pending: list[tuple[ast.AST, int]] = [(expression, 1)]
        while pending:
            node, current = pending.pop()
            nodes += 1
            if nodes > _MAX_ANALYSIS_NODES:
                return False
            dependent = depths.get(node.id, 0) if isinstance(node, ast.Name) else 0
            depth = max(depth, current + dependent)
            if depth > _MAX_ANALYSIS_DEPTH:
                return False
            pending.extend((child, current + 1) for child in ast.iter_child_nodes(node))
        if target is not None:
            assert isinstance(target, ast.Name)
            depths[target.id] = depth
    return True


class _Analysis:
    def __init__(
        self,
        definitions: Mapping[str, ast.expr],
        coordinate: str,
        tensors: Mapping[str, CopyTensorFacts],
        aligned_names: Mapping[str, int],
    ) -> None:
        self.definitions = definitions
        self.coordinate = coordinate
        self.tensors = tensors
        self.aligned_names = aligned_names
        self._resolved: dict[str, ast.expr] = {}
        self._dependencies: dict[ast.expr, bool] = {}
        self._kinds: dict[ast.expr, _ValueKind] = {}
        self._alignments: dict[ast.expr, int] = {}
        self._deltas: dict[ast.expr, int | None] = {}
        self._intervals: dict[ast.expr, bool] = {}
        self._directions: dict[ast.expr, int | None] = {}
        self._uniform_predicates: dict[ast.expr, bool] = {}
        self._expanded_sizes: dict[ast.expr, int] = {}
        self._expanded: dict[ast.expr, ast.expr] = {}
        # Resolved after identifying the terminal load's guarded source dtype.
        self.width = 8

    def resolve(self, value: ast.expr) -> ast.expr:
        if isinstance(value, ast.Name) and value.id in self.definitions:
            if value.id not in self._resolved:
                self._resolved[value.id] = self.resolve(self.definitions[value.id])
            return self._resolved[value.id]
        return value

    def depends(self, value: ast.expr) -> bool:
        if value not in self._dependencies:
            self._dependencies[value] = self._depends(value)
        return self._dependencies[value]

    def _depends(self, value: ast.expr) -> bool:
        for name in _read_names(value):
            if name == self.coordinate:
                return True
            if name in self.definitions and self.depends(self.definitions[name]):
                return True
        return False

    def constant(self, value: ast.expr) -> int | None:
        value = self.resolve(value)
        if isinstance(value, ast.Constant) and type(value.value) is int:
            return value.value
        if self.integer_cast(value):
            assert isinstance(value, ast.Call)
            # Strides and literal constants in this proof must already fit
            # their source cast. Do not infer a larger value from narrowing.
            result = self.constant(value.args[0])
            if result is not None and -(1 << 31) <= result < 1 << 31:
                return result
        if isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant):
            dim = value.slice.value
            path = _path(value.value)
            if path is not None and type(dim) is int:
                source, _, attr = path.partition(".")
                facts = self.tensors.get(source)
                if facts is not None and attr in {"stride", "layout.stride"}:
                    if 0 <= dim < len(facts.strides):
                        return facts.strides[dim]
        return None

    @staticmethod
    def integer_cast(value: ast.expr) -> bool:
        return (
            isinstance(value, ast.Call)
            and _path(value.func) in {f"cutlass.{name}" for name in _INTEGER_CASTS}
            and len(value.args) == 1
            and not value.keywords
        )

    def alignment(self, value: ast.expr) -> int:
        value = self.resolve(value)
        if value not in self._alignments:
            self._alignments[value] = self._alignment(value)
        return self._alignments[value]

    def kind(self, value: ast.expr) -> _ValueKind:
        value = self.resolve(value)
        if value not in self._kinds:
            self._kinds[value] = self._kind(value)
        return self._kinds[value]

    def _kind(self, value: ast.expr) -> _ValueKind:
        if self.constant(value) is not None or self.integer_cast(value):
            return "integer"
        if isinstance(value, ast.Name):
            # These names are integer coordinates supplied by the caller;
            # arbitrary scalar boundaries carry no implicit integer proof.
            if value.id == self.coordinate or value.id in self.aligned_names:
                return "integer"
        if isinstance(value, ast.Attribute) and value.attr == "iterator":
            source = _path(value.value)
            if source in self.tensors:
                return "pointer"
        if isinstance(value, ast.UnaryOp) and isinstance(
            value.op, (ast.UAdd, ast.USub, ast.Invert)
        ):
            return "integer" if self.kind(value.operand) == "integer" else None
        if isinstance(value, ast.IfExp):
            if self.kind(value.body) == self.kind(value.orelse) == "integer":
                return "integer"
        if isinstance(value, ast.BinOp):
            left, right = self.kind(value.left), self.kind(value.right)
            if left == right == "integer" and isinstance(
                value.op,
                (
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.FloorDiv,
                    ast.Mod,
                    ast.BitAnd,
                    ast.BitOr,
                    ast.BitXor,
                    ast.LShift,
                    ast.RShift,
                ),
            ):
                return "integer"
            if (
                left == "pointer"
                and right == "integer"
                and isinstance(value.op, (ast.Add, ast.Sub))
            ):
                return "pointer"
            if (
                left == "integer"
                and right == "pointer"
                and isinstance(value.op, ast.Add)
            ):
                return "pointer"
        return None

    def _alignment(self, value: ast.expr) -> int:
        if (constant := self.constant(value)) is not None:
            return math.gcd(self.width, constant)
        if isinstance(value, ast.Name):
            return math.gcd(self.width, self.aligned_names.get(value.id, 1))
        if isinstance(value, ast.Attribute) and value.attr == "iterator":
            source = _path(value.value)
            facts = self.tensors.get(source) if source is not None else None
            if facts is not None and facts.dtype in _DTYPE_BYTES:
                return math.gcd(
                    self.width, facts.alignment_bytes // _DTYPE_BYTES[facts.dtype]
                )
        if self.integer_cast(value):
            assert isinstance(value, ast.Call)
            # An integer cast does not preserve divisibility of a floating
            # expression: Int32(Float32(0.125) * 8) is one, not a multiple of 8.
            return (
                self.alignment(value.args[0])
                if self.kind(value.args[0]) == "integer"
                else 1
            )
        if isinstance(value, ast.UnaryOp) and isinstance(
            value.op, (ast.UAdd, ast.USub)
        ):
            return self.alignment(value.operand)
        if isinstance(value, ast.BinOp) and self.kind(value) is not None:
            left, right = self.alignment(value.left), self.alignment(value.right)
            if isinstance(value.op, (ast.Add, ast.Sub)):
                return math.gcd(left, right)
            if isinstance(value.op, ast.Mult):
                return math.gcd(self.width, left * right)
        return 1

    def inactive_pointer_is_safe(self, value: ast.expr, source_tensor: str) -> bool:
        """Allow zero-fill to evaluate only inert address arithmetic.

        A zero-byte copy does not dereference its source, but its source argument
        is still evaluated. An inline metadata load, division, or subscription
        may therefore be invalid on an inactive lane. Names already computed by
        the original recipe are safe; keep all other expressions behind the
        original copy predicate unless their arithmetic is explicitly harmless.
        """
        if isinstance(value, ast.Name):
            return True
        if isinstance(value, ast.Constant):
            return type(value.value) is int
        if isinstance(value, ast.Subscript):
            return self.constant(value) is not None
        if _path(value) == f"{source_tensor}.iterator":
            return True
        if self.integer_cast(value):
            assert isinstance(value, ast.Call)
            return self.inactive_pointer_is_safe(value.args[0], source_tensor)
        if isinstance(value, ast.UnaryOp) and isinstance(
            value.op, (ast.UAdd, ast.USub)
        ):
            return self.inactive_pointer_is_safe(value.operand, source_tensor)
        if isinstance(value, ast.BinOp) and isinstance(
            value.op, (ast.Add, ast.Sub, ast.Mult)
        ):
            return self.inactive_pointer_is_safe(
                value.left, source_tensor
            ) and self.inactive_pointer_is_safe(value.right, source_tensor)
        return False

    def delta(self, value: ast.expr) -> int | None:
        value = self.resolve(value)
        if value not in self._deltas:
            self._deltas[value] = self._delta(value)
        return self._deltas[value]

    def _delta(self, value: ast.expr) -> int | None:
        if not self.depends(value):
            return 0
        if isinstance(value, ast.Name) and value.id == self.coordinate:
            return 1
        if self.integer_cast(value):
            assert isinstance(value, ast.Call)
            # Integer wrap boundaries are multiples of the vector width. An
            # aligned vector cannot cross one inside this cast's elements.
            if (
                self.kind(value.args[0]) != "integer"
                or self.alignment(value.args[0]) != self.width
            ):
                return None
            return self.delta(value.args[0])
        if isinstance(value, ast.BinOp) and self.kind(value) is not None:
            left, right = self.delta(value.left), self.delta(value.right)
            if left is None or right is None or self.alignment(value) != self.width:
                return None
            if isinstance(value.op, ast.Add):
                delta = left + right
                return delta if delta in (0, 1) else None
            if isinstance(value.op, ast.Sub):
                delta = left - right
                return delta if delta in (0, 1) else None
            if isinstance(value.op, ast.Mult):
                if left == 0 and (constant := self.constant(value.left)) is not None:
                    delta = constant * right
                    return delta if delta in (0, 1) else None
                if right == 0 and (constant := self.constant(value.right)) is not None:
                    delta = constant * left
                    return delta if delta in (0, 1) else None
        return None

    def interval(self, value: ast.expr) -> bool:
        value = self.resolve(value)
        if value not in self._intervals:
            self._intervals[value] = self._interval(value)
        return self._intervals[value]

    def _interval(self, value: ast.expr) -> bool:
        if not self.depends(value):
            return True
        if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And):
            return all(self.interval(part) for part in value.values)
        if isinstance(value, ast.Compare):
            arguments = [value.left, *value.comparators]
            if not all(
                isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in value.ops
            ):
                return False
            pairs = itertools.pairwise(arguments)
        elif (
            isinstance(value, ast.Call)
            and _path(value.func)
            in {"operator.lt", "operator.le", "operator.gt", "operator.ge"}
            and len(value.args) == 2
            and not value.keywords
        ):
            pairs = iter(((value.args[0], value.args[1]),))
        else:
            return False
        for left, right in pairs:
            a, b = self.delta(left), self.delta(right)
            if a is None or b is None or (a, b) not in {(1, 0), (0, 1)}:
                return False
        return True

    def expansion_fits(self, values: Sequence[ast.expr]) -> bool:
        """Count every substituted tree before creating any expanded copies.

        Cached sizes keep an alias DAG linear to analyze. Counting both final
        trees and cached intermediate trees also bounds the cloning work for
        long dependency chains whose final expression would fit on its own.
        """
        for value in values:
            self._expanded_size(value)
        return sum(self._expanded_sizes.values()) <= _MAX_ANALYSIS_NODES

    def _expanded_size(self, value: ast.expr) -> int:
        if value in self._expanded_sizes:
            return self._expanded_sizes[value]
        size = 0
        for node in ast.walk(value):
            size += 1
            if (
                isinstance(node, ast.Name)
                and node.id in self.definitions
                and self.depends(self.definitions[node.id])
            ):
                # A Load name and its context are replaced by the entire tree.
                size += self._expanded_size(self.definitions[node.id]) - 2
            if size > _MAX_ANALYSIS_NODES:
                size = _MAX_ANALYSIS_NODES + 1
                break
        self._expanded_sizes[value] = size
        return size

    def expand_variant(self, value: ast.expr) -> ast.expr:
        if value in self._expanded:
            return self._expanded[value]
        replacements = {
            name: self.expand_variant(self.definitions[name])
            for name in _read_names(value)
            if name in self.definitions and self.depends(self.definitions[name])
        }
        result = _replace(value, replacements)
        assert isinstance(result, ast.expr)
        self._expanded[value] = result
        return result

    def predicate_direction(self, value: ast.expr) -> int | None:
        """Prove monotonic truth values within one aligned vector.

        Equal endpoint predicates imply a wholly active or inactive vector
        only for a monotone predicate. An interval can otherwise contain valid
        middle lanes while both endpoints are inactive.
        """
        value = self.resolve(value)
        if value not in self._directions:
            self._directions[value] = self._predicate_direction(value)
        return self._directions[value]

    def packet_uniform_predicate(self, value: ast.expr) -> bool:
        """Prove that a predicate cannot split an aligned vector.

        For an increasing integer coordinate, an aligned strict upper bound
        or inclusive lower bound lies between packets. Reversing the operands
        reverses the accepted comparisons. The existing unit-delta proof also
        excludes casts or arithmetic that could wrap inside a packet.
        """
        value = self.resolve(value)
        if value not in self._uniform_predicates:
            self._uniform_predicates[value] = self._packet_uniform_predicate(value)
        return self._uniform_predicates[value]

    def _packet_uniform_predicate(self, value: ast.expr) -> bool:
        if not self.depends(value):
            return True
        if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And):
            return all(self.packet_uniform_predicate(part) for part in value.values)
        if isinstance(value, ast.Compare):
            pairs = list(itertools.pairwise([value.left, *value.comparators]))
            operators = list(value.ops)
        elif (
            isinstance(value, ast.Call) and len(value.args) == 2 and not value.keywords
        ):
            path = _path(value.func)
            if path is None:
                return False
            operator_type = {
                "operator.lt": ast.Lt,
                "operator.le": ast.LtE,
                "operator.gt": ast.Gt,
                "operator.ge": ast.GtE,
            }.get(path)
            if operator_type is None:
                return False
            pairs = [(value.args[0], value.args[1])]
            operators = [operator_type()]
        else:
            return False
        for (left, right), op in zip(pairs, operators, strict=True):
            if (
                self.kind(left) != "integer"
                or self.kind(right) != "integer"
                or self.alignment(left) != self.width
                or self.alignment(right) != self.width
            ):
                return False
            deltas = (self.delta(left), self.delta(right))
            if not (
                deltas == (1, 0)
                and isinstance(op, (ast.Lt, ast.GtE))
                or deltas == (0, 1)
                and isinstance(op, (ast.Gt, ast.LtE))
            ):
                return False
        return True

    def _predicate_direction(self, value: ast.expr) -> int | None:
        if not self.depends(value):
            return 0
        if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.And):
            directions = {self.predicate_direction(part) for part in value.values}
        else:
            if isinstance(value, ast.Compare):
                pairs = list(itertools.pairwise([value.left, *value.comparators]))
                operators = [type(op) for op in value.ops]
            elif isinstance(value, ast.Call) and len(value.args) == 2:
                path = _path(value.func)
                if path is None:
                    return None
                operator_type = {
                    "operator.lt": ast.Lt,
                    "operator.le": ast.LtE,
                    "operator.gt": ast.Gt,
                    "operator.ge": ast.GtE,
                }.get(path)
                if operator_type is None or value.keywords:
                    return None
                pairs = [(value.args[0], value.args[1])]
                operators = [operator_type]
            else:
                return None
            directions = set()
            for (left, right), op in zip(pairs, operators, strict=True):
                left_delta, right_delta = self.delta(left), self.delta(right)
                if (left_delta, right_delta) not in {(1, 0), (0, 1)}:
                    return None
                if op in (ast.Lt, ast.LtE):
                    directions.add(-1 if left_delta else 1)
                elif op in (ast.Gt, ast.GtE):
                    directions.add(1 if left_delta else -1)
                else:
                    return None
        directions.discard(0)
        return next(iter(directions), 0) if len(directions) <= 1 else None


class _IntegerRecipeAnalysis(_Analysis):
    def __init__(
        self,
        definitions: Mapping[str, ast.expr],
        integer_tensor_names: Collection[str],
    ) -> None:
        super().__init__(definitions, "", {}, {})
        self.integer_tensor_names = integer_tensor_names

    def _kind(self, value: ast.expr) -> _ValueKind:
        if (kind := super()._kind(value)) is not None:
            return kind
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "load"
            and not value.args
            and not value.keywords
        ):
            pointer = self.resolve(value.func.value)
            while isinstance(pointer, ast.BinOp) and isinstance(
                pointer.op, (ast.Add, ast.Sub)
            ):
                if self.kind(pointer.right) != "integer":
                    return None
                pointer = self.resolve(pointer.left)
            if (
                isinstance(pointer, ast.Attribute)
                and pointer.attr == "iterator"
                and isinstance(pointer.value, ast.Name)
                and pointer.value.id in self.integer_tensor_names
            ):
                return "integer"
        return None


def recipe_has_integer_result(
    statements: Sequence[ast.Assign],
    value: ast.expr,
    integer_tensor_names: Collection[str],
) -> bool:
    """Retain an emitted scalar recipe's integer type when it becomes a boundary.

    The caller supplies declared integer tensor dtypes and an already proven
    pure, single-assignment recipe. A typed load proves only its result type;
    it grants no alignment or permission to evaluate a masked pointer.
    """
    if not _within_analysis_budget(statements, value):
        return False
    definitions = {
        statement.targets[0].id: statement.value
        for statement in statements
        if isinstance(statement.targets[0], ast.Name)
    }
    return (
        _IntegerRecipeAnalysis(definitions, integer_tensor_names).kind(value)
        == "integer"
    )


@dataclasses.dataclass(frozen=True)
class ContiguousCopy:
    """A proven direct load; emission retains exact scalar tails.

    Shared memory must have 16-byte alignment and ``width`` physically contiguous
    elements along ``coordinate``. A swizzle preserving the low four byte
    address bits, as used by the collective MMA staging layouts, satisfies this.
    The caller must exclude vectors outside the allocated shared tile.
    """

    source_tensor: str
    coordinate: str
    _statements: tuple[ast.Assign, ...]
    _value: ast.expr
    _pointer: ast.expr
    _predicate: ast.expr
    _invariants: frozenset[str]
    _zero_fill_monotone: bool
    width: int = 8
    _uniform_positive_zero: bool = False

    def emit_from_registers(
        self, registers: str, fresh_name: Callable[[str], str]
    ) -> list[ast.stmt]:
        """Store a typed register vector using this address/mask proof.

        The caller builds the proof from a synthetic load at its original
        store address, with the original store predicate. No synthetic load
        executes: only its invariant address/mask definitions are replayed.
        All vector predicates must hold; partial vectors keep scalar stores.
        """
        names = {
            statement.targets[0].id: _expr(
                fresh_name(f"{statement.targets[0].id}_store")
            )
            for statement in self._statements
            if isinstance(statement.targets[0], ast.Name)
        }
        common: list[ast.stmt] = []
        for statement in self._statements:
            target = statement.targets[0]
            assert isinstance(target, ast.Name)
            if target.id in self._invariants:
                copied = _replace(statement, names)
                assert isinstance(copied, ast.stmt)
                common.append(copied)
        low = _replace(self._predicate, names)
        high = _replace(
            self._predicate,
            {**names, self.coordinate: _expr(f"{self.coordinate} + {self.width - 1}")},
        )
        pointer = _replace(self._pointer, names)
        assert isinstance(low, ast.expr) and isinstance(high, ast.expr)
        condition = (
            low
            if ast.dump(low) == ast.dump(high)
            else ast.BoolOp(op=ast.And(), values=[low, high])
        )
        target = fresh_name("epilogue_destination")
        vector = ast.parse(
            f"{target} = cute.make_tensor(cute.make_ptr("
            f"{self.source_tensor}.element_type, ({ast.unparse(pointer)}).toint(), "
            "cute.AddressSpace.gmem, assumed_align=16), "
            f"cute.make_layout(({self.width},), stride=(1,)))\n"
            f"cute.autovec_copy({registers}, {target})"
        ).body
        lane = fresh_name("epilogue_store_lane")
        replacements = {
            **names,
            self.coordinate: _expr(f"{self.coordinate} + {lane}"),
        }
        scalar_pointer = _replace(self._pointer, replacements)
        scalar_predicate = _replace(self._predicate, replacements)
        scalar = ast.parse(
            f"for {lane} in cutlass.range_constexpr({self.width}):\n"
            f"    if {ast.unparse(scalar_predicate)}:\n"
            f"        ({ast.unparse(scalar_pointer)}).store({registers}[{lane}])"
        ).body
        common.append(ast.If(test=condition, body=vector, orelse=scalar))
        return [ast.fix_missing_locations(statement) for statement in common]

    def emit_to_registers(
        self, fresh_name: Callable[[str], str]
    ) -> tuple[list[ast.stmt], str]:
        """Load one aligned vector, preserving every scalar masked tail.

        The source's guarded dtype and pointer alignment are the same facts
        used for async shared copies. No pointer is evaluated outside its
        original load predicate. Callers own the surrounding read-only region.
        """
        registers = fresh_name("recipe_registers")
        source = fresh_name("recipe_source")
        lane = fresh_name("recipe_tail_lane")
        names = {
            statement.targets[0].id: _expr(
                fresh_name(f"{statement.targets[0].id}_register")
            )
            for statement in self._statements
            if isinstance(statement.targets[0], ast.Name)
        }
        replacements = {
            **names,
            self.coordinate: _expr(f"{self.coordinate} + {lane}"),
        }
        common: list[ast.stmt] = ast.parse(
            f"{registers} = cute.make_rmem_tensor("
            f"cute.make_layout(({self.width},)), "
            f"{self.source_tensor}.element_type)"
        ).body
        scalar: list[ast.stmt] = []
        for statement in self._statements:
            target = statement.targets[0]
            assert isinstance(target, ast.Name)
            invariant = target.id in self._invariants
            copied = _replace(statement, names if invariant else replacements)
            assert isinstance(copied, ast.stmt)
            (common if invariant else scalar).append(copied)
        low = _replace(self._predicate, names)
        high = _replace(
            self._predicate,
            {**names, self.coordinate: _expr(f"{self.coordinate} + {self.width - 1}")},
        )
        pointer = _replace(self._pointer, names)
        assert isinstance(low, ast.expr) and isinstance(high, ast.expr)
        assert isinstance(pointer, ast.expr)
        condition = (
            low
            if self._uniform_positive_zero or ast.dump(low) == ast.dump(high)
            else ast.BoolOp(op=ast.And(), values=[low, high])
        )
        vector = ast.parse(
            f"{source} = cute.make_tensor(cute.make_ptr("
            f"{self.source_tensor}.element_type, ({ast.unparse(pointer)}).toint(), "
            "cute.AddressSpace.gmem, assumed_align=16), "
            f"cute.make_layout(({self.width},), stride=(1,)))\n"
            f"cute.autovec_copy({source}, {registers})"
        ).body
        scalar_value = _replace(self._value, replacements)
        assert isinstance(scalar_value, ast.expr)
        scalar.append(
            ast.Assign(
                targets=[
                    ast.Subscript(
                        value=_expr(registers), slice=_expr(lane), ctx=ast.Store()
                    )
                ],
                value=scalar_value,
            )
        )
        common.append(
            ast.If(
                test=condition,
                body=vector,
                orelse=ast.parse(f"{registers}.fill(0)").body
                if self._uniform_positive_zero
                else [
                    ast.For(
                        target=ast.Name(id=lane, ctx=ast.Store()),
                        iter=_expr(f"cutlass.range_constexpr({self.width})"),
                        body=scalar,
                        orelse=[],
                    )
                ],
            )
        )
        return [ast.fix_missing_locations(node) for node in common], registers

    def emit_to_aligned_smem(
        self,
        destination_tensor: str,
        destination_indices: tuple[ast.expr, ...],
        fresh_name: Callable[[str], str],
        *,
        preserve_pointer_swizzle: bool = False,
    ) -> list[ast.stmt]:
        """Emit one 128-bit copy, without committing or waiting on it.

        A swizzle attached to the shared pointer requires a typed copy atom.
        The raw architecture copy converts its destination to an LLVM pointer,
        which drops this swizzle. Layout-carried swizzles already participate
        in ``crd2idx`` and can use the raw copy.
        """
        names = {
            statement.targets[0].id: ast.Name(
                id=fresh_name(f"{statement.targets[0].id}_copy"), ctx=ast.Load()
            )
            for statement in self._statements
            if isinstance(statement.targets[0], ast.Name)
        }
        lane = fresh_name("copy_tail_lane")
        coordinate = _expr(f"{self.coordinate} + {lane}")
        scalar_replacements = {**names, self.coordinate: coordinate}
        common: list[ast.stmt] = []
        scalar: list[ast.stmt] = []
        for statement in self._statements:
            target = statement.targets[0]
            assert isinstance(target, ast.Name)
            invariant = target.id in self._invariants
            result = _replace(statement, names if invariant else scalar_replacements)
            assert isinstance(result, ast.stmt)
            (common if invariant else scalar).append(result)

        low = _replace(self._predicate, names)
        high = _replace(
            self._predicate,
            {**names, self.coordinate: _expr(f"{self.coordinate} + {self.width - 1}")},
        )
        pointer = _replace(self._pointer, names)
        assert isinstance(low, ast.expr) and isinstance(high, ast.expr)
        assert isinstance(pointer, ast.expr)
        condition = (
            low
            if ast.dump(low) == ast.dump(high)
            else ast.BoolOp(op=ast.And(), values=[low, high])
        )
        assert isinstance(condition, ast.expr)
        index = ast.Tuple(elts=list(destination_indices), ctx=ast.Load())
        destination_offset = (
            f"cute.crd2idx({ast.unparse(index)}, {destination_tensor}.layout)"
        )
        if preserve_pointer_swizzle:
            # The caller proves an aligned contiguous shared vector, and the
            # recipe proves source alignment. Keep the original shared pointer
            # while making those alignment facts explicit to the copy atom.
            # Pointer.align() is unsuitable: it rounds and rebuilds a pointer,
            # discarding its swizzle even when the address was aligned.
            destination = (
                f"{destination_tensor}.iterator + "
                f"cute.assume({destination_offset}, divby={self.width})"
            )
            source = (
                f"cute.make_ptr({self.source_tensor}.element_type, "
                f"({ast.unparse(pointer)}).toint(), cute.AddressSpace.gmem, "
                "assumed_align=16)"
            )
            async_copy = ast.Expr(
                value=_expr(
                    "cute.copy(cute.make_copy_atom("
                    "cute.nvgpu.cpasync.CopyG2SOp(cute.nvgpu.LoadCacheMode.GLOBAL), "
                    f"{self.source_tensor}.element_type, num_bits_per_copy=128), "
                    f"cute.make_tensor({source}, cute.make_layout(({self.width},), stride=(1,))), "
                    f"cute.make_tensor({destination}, cute.make_layout(({self.width},), stride=(1,))))"
                )
            )
        else:
            async_copy = ast.Expr(
                value=ast.Call(
                    func=_expr("cute.arch.cp_async_shared_global"),
                    args=[
                        _expr(f"{destination_tensor}.iterator + {destination_offset}"),
                        pointer,
                        ast.Constant(value=16),
                        ast.Constant(value="cg"),
                    ],
                    keywords=[],
                )
            )
        scalar_index = _replace(index, {self.coordinate: coordinate})
        scalar_value = _replace(self._value, scalar_replacements)
        assert isinstance(scalar_index, ast.expr) and isinstance(scalar_value, ast.expr)
        scalar.append(
            ast.Assign(
                targets=[
                    ast.Subscript(
                        value=ast.Name(id=destination_tensor, ctx=ast.Load()),
                        slice=scalar_index,
                        ctx=ast.Store(),
                    )
                ],
                value=scalar_value,
            )
        )
        tail = ast.For(
            target=ast.Name(id=lane, ctx=ast.Store()),
            iter=_expr(f"cutlass.range({self.width}, unroll=1)"),
            body=scalar,
            orelse=[],
        )
        if self._zero_fill_monotone and not preserve_pointer_swizzle:
            # src-size zero performs no global read and initializes all sixteen
            # destination bytes. Keep partial vectors on their exact scalar
            # path, including lower-bound masks and gathered row guards.
            assert isinstance(async_copy.value, ast.Call)
            low_truth = ast.Compare(
                left=low, ops=[ast.NotEq()], comparators=[ast.Constant(0)]
            )
            async_copy.value.keywords.append(
                ast.keyword(
                    arg="cp_size",
                    value=ast.BinOp(
                        left=ast.Call(
                            func=_expr("cutlass.Int32"), args=[low_truth], keywords=[]
                        ),
                        op=ast.Mult(),
                        right=ast.Constant(16),
                    ),
                )
            )
            if ast.dump(low) == ast.dump(high):
                common.append(async_copy)
            else:
                high_truth = ast.Compare(
                    left=high, ops=[ast.NotEq()], comparators=[ast.Constant(0)]
                )
                common.append(
                    ast.If(
                        test=ast.Compare(
                            left=_clone(low_truth),
                            ops=[ast.Eq()],
                            comparators=[high_truth],
                        ),
                        body=[async_copy],
                        orelse=[tail],
                    )
                )
        else:
            common.append(ast.If(test=condition, body=[async_copy], orelse=[tail]))
        return [ast.fix_missing_locations(statement) for statement in common]


def plan_contiguous_copy(
    statements: Sequence[ast.Assign],
    value: ast.expr,
    *,
    coordinate: str,
    tensors: Mapping[str, CopyTensorFacts],
    aligned_names: Mapping[str, int],
) -> ContiguousCopy | None:
    """Prove a direct FP16/BF16/FP32 load for an aligned vector copy.

    ``statements`` and ``value`` are the output of ``ScalarRecipe.emit``.
    Alignment facts are in elements, except tensor base alignments in bytes.
    Unknown transformations, strides, masks, or alignment facts fail closed.
    """
    targets: set[str] = set()
    for statement in statements:
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
            or statement.targets[0].id in targets
        ):
            return None
        targets.add(statement.targets[0].id)
    if coordinate in targets:
        return None
    if not _within_analysis_budget(statements, value):
        return None
    boundaries = (
        set().union(
            *(_read_names(statement.value) for statement in statements),
            _read_names(value),
        )
        - targets
    )
    recipe = build_recipe(value, statements, boundaries)
    if recipe is None:
        return None
    counter = itertools.count()
    copied, result = recipe.emit({}, lambda name: f"_vector_{next(counter)}_{name}")
    definitions = {
        statement.targets[0].id: statement.value
        for statement in copied
        if isinstance(statement.targets[0], ast.Name)
    }
    analysis = _Analysis(definitions, coordinate, tensors, aligned_names)
    terminal = analysis.resolve(result)
    predicates: list[ast.expr] = []
    casts: list[str] = []
    positive_zero = True
    while True:
        terminal = analysis.resolve(terminal)
        if isinstance(terminal, ast.IfExp):
            zero = analysis.resolve(terminal.orelse)
            if (
                isinstance(zero, ast.Call)
                and _path(zero.func) in _DTYPES
                and len(zero.args) == 1
                and not zero.keywords
            ):
                casts.append(str(_path(zero.func)))
                zero = analysis.resolve(zero.args[0])
            if (
                not isinstance(zero, ast.Constant)
                or type(zero.value) not in (int, float)
                or zero.value != 0
            ):
                return None
            positive_zero = positive_zero and math.copysign(1.0, zero.value) > 0
            predicates.append(terminal.test)
            terminal = terminal.body
        elif (
            isinstance(terminal, ast.Call)
            and _path(terminal.func) in _DTYPES
            and len(terminal.args) == 1
            and not terminal.keywords
        ):
            casts.append(str(_path(terminal.func)))
            terminal = terminal.args[0]
        else:
            break
    if (
        not isinstance(terminal, ast.Call)
        or not isinstance(terminal.func, ast.Attribute)
        or terminal.func.attr != "load"
        or terminal.args
        or terminal.keywords
    ):
        return None
    pointer = terminal.func.value
    sources = [
        node.value.id
        for node in ast.walk(pointer)
        if isinstance(node, ast.Attribute)
        and node.attr == "iterator"
        and isinstance(node.value, ast.Name)
    ]
    if not sources:
        # A pointer itself may be a scalar recipe assignment.
        pointer = analysis.resolve(pointer)
        sources = [
            node.value.id
            for node in ast.walk(pointer)
            if isinstance(node, ast.Attribute)
            and node.attr == "iterator"
            and isinstance(node.value, ast.Name)
        ]
    if len(sources) != 1 or (facts := tensors.get(sources[0])) is None:
        return None
    if (
        facts.dtype not in _DTYPES
        or facts.alignment_bytes < 16
        or any(dtype != facts.dtype for dtype in casts)
    ):
        return None
    analysis.width = 16 // _DTYPE_BYTES[facts.dtype]
    if (
        aligned_names.get(coordinate, 1) % analysis.width
        or analysis.delta(pointer) != 1
        or analysis.alignment(pointer) != analysis.width
    ):
        return None
    predicate: ast.expr = (
        ast.Constant(value=True)
        if not predicates
        else predicates[0]
        if len(predicates) == 1
        else ast.BoolOp(op=ast.And(), values=predicates)
    )
    if not analysis.expansion_fits((pointer, predicate)):
        return None
    if not analysis.interval(predicate):
        return None
    invariants = frozenset(
        name
        for name, expression in definitions.items()
        if not analysis.depends(expression)
    )
    expanded_pointer = analysis.expand_variant(pointer)
    return ContiguousCopy(
        sources[0],
        coordinate,
        tuple(copied),
        result,
        expanded_pointer,
        analysis.expand_variant(predicate),
        invariants,
        positive_zero
        and analysis.predicate_direction(predicate) is not None
        and analysis.inactive_pointer_is_safe(expanded_pointer, sources[0]),
        width=analysis.width,
        _uniform_positive_zero=positive_zero
        and analysis.packet_uniform_predicate(predicate),
    )

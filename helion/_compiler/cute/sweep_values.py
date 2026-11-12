"""Value numbers for the sweeps of a reduction, to carry expensive pure values.

A persistent reduction whose row is spread over synthetic lanes runs one lane
loop per sweep (the mean, variance and store sweeps of a layer norm), and each
sweep re-emits the row's whole elementwise chain.  ``fuse_two_pass_loads``
serves the later sweeps' loads from per-thread fragments, but they still
recompute every pure value derived from those loads.  That is cheap for a few
ALU operations, not for a word0 ``hl.rand`` draw: a Philox4x32-10 chain of
about forty ``_cute_inline_asm_elementwise`` multiplies per element, evaluated
again by every sweep that needs the mask.

``SweepValues`` numbers the values computed by the sweeps of one fusion group
so that equal numbers mean equal values, for the same iteration of the sweeps'
identical loops.  It walks the statements forward with an environment from
names to numbers.  Leaves are constants, globals and kernel arguments, the
loop indices (which stand for the iteration and so compare equal across
sweeps: the fuser groups only sweeps over the same indices) and a V-loop's
lane index likewise, and loads the fuser proved equal to a first-sweep load.
A V-loop's body (a packet's lanes) is numbered statement by statement, and
after the loop each name it wrote is a leaf of the loop.  A pure expression
(operators, casts, ``cute.math``, pure inline asm) is numbered by its
operator and its operands' numbers.  Anything else (another load, a call with
effects, a name written by a nested statement or carried around a sweep or
lane loop) is a leaf of its own definition, equal only to reads of that
definition.  Names compare through the final rename's groups: a loop carry
written as ``v_2 = acc_copy_0 + sum_1`` rebinds ``acc``.  A carried value's
element type follows the DSL's promotion rules, Python scalars included.

``plan_cut`` then chooses which recomputed values a later sweep reads from a
fragment: walking the sweep backward, the first live value that the first
sweep also computes and whose heavy operations (inline asm, by its
instructions, and the enabled transcendentals) cost at least
``MIN_SAVED_COST`` is cut, which drops everything only it needed, its loads
included.  Cutting the latest such value keeps the fragments few: a layer
norm's variance sweep needs only its dropped-out row, not both dropout masks
and the two loads behind it.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import functools
import itertools
from typing import TYPE_CHECKING
from typing import cast

from ..ast_read_writes import ReadWrites
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import assigned_name
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import live_in
from ._ast_pass_utils import loop_extent
from ._ast_pass_utils import names_written
from .pure_lane_packets import inline_asm_operands
from .pure_lane_packets import is_pure_scalar_assembly

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Collection
    from collections.abc import Mapping
    from collections.abc import Sequence

# Transcendentals worth carrying when the reload policy accepts the registers.
TRANSCENDENTALS = frozenset(
    (
        "cute.math.exp",
        "cute.math.exp2",
        "cute.math.log",
        "cute.math.log2",
        "cute.math.tanh",
        "cute.math.erf",
    )
)
# Estimated instructions of one transcendental (range reduction, MUFU and
# fixup); every other operator counts one, an inline asm its instructions.
_TRANSCENDENTAL_COST = 16
# A later sweep reads carried values only when that saves at least this many
# estimated instructions per element, and only values whose heavy operations
# cost that much: a few ALU operations are cheaper to recompute than a
# fragment's registers are to hold.
MIN_SAVED_COST = 16
# A value this cheap is recomputed from operands the sweep computes anyway.
_RECOMPUTE_COST = 2

_MODULES = frozenset(("cutlass", "cute", "operator", "math"))
_INT_RANGES = {
    **{
        f"cutlass.Int{bits}": (-(1 << (bits - 1)), (1 << (bits - 1)) - 1)
        for bits in (8, 16, 32, 64)
    },
    **{f"cutlass.Uint{bits}": (0, (1 << bits) - 1) for bits in (8, 16, 32, 64)},
}
_FLOATS = frozenset(
    ("cutlass.Float16", "cutlass.BFloat16", "cutlass.Float32", "cutlass.Float64")
)
_BOOLEAN = "cutlass.Boolean"
# Element types a carried value may have: a fragment of one of them holds the
# value bit-exactly.
CACHE_DTYPES = frozenset((_BOOLEAN, *_INT_RANGES, *_FLOATS))
_COMPARISONS = frozenset(
    f"operator.{name}" for name in ("lt", "le", "gt", "ge", "eq", "ne")
)
_ARITHMETIC = {
    "operator.add": ast.Add(),
    "operator.sub": ast.Sub(),
    "operator.mul": ast.Mult(),
    "operator.and_": ast.BitAnd(),
    "operator.or_": ast.BitOr(),
    "operator.xor": ast.BitXor(),
}
_FLOAT_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_BITWISE_OPS = (ast.BitAnd, ast.BitOr, ast.BitXor)
_INT_OPS = (ast.Add, ast.Sub, ast.Mult, *_BITWISE_OPS, ast.LShift, ast.RShift)
# The dtype of a Python scalar constant, which the DSL converts before an
# operation with a DSL value (``_scalar_dtype``).
_PY_INT = "int"
_PY_FLOAT = "float"
_FLOAT_WIDTHS = {
    "cutlass.Float16": 16,
    "cutlass.BFloat16": 16,
    "cutlass.Float32": 32,
    "cutlass.Float64": 64,
}
_FLOAT_OF_WIDTH = {16: "cutlass.Float16", 32: "cutlass.Float32", 64: "cutlass.Float64"}
# ``cute.math`` functions whose result takes their first operand's type (the
# DSL wraps it like that operand).  Not the FP predicates (``isnan`` and the
# like return an ``i1``) or ``sincos`` (a pair).
_SAME_TYPE_MATH = frozenset(
    f"cute.math.{name}"
    for name in (
        "abs",
        "absf",
        "atan2",
        "ceil",
        "copysign",
        "cos",
        "div",
        "erf",
        "exp",
        "exp2",
        "floor",
        "fma",
        "log",
        "log2",
        "max",
        "min",
        "rcp",
        "roundeven",
        "rsqrt",
        "sin",
        "sqrt",
        "tanh",
        "trunc",
    )
)


@functools.cache
def _pure_asm_instructions(asm: str, constraints: object, pure: object) -> int | None:
    if not is_pure_scalar_assembly(asm, constraints, pure):
        return None
    instructions = [
        line.strip() for line in asm.replace("{", ";").replace("}", ";").split(";")
    ]
    return max(
        1, sum(1 for line in instructions if line and not line.startswith(".reg"))
    )


def asm_instruction_count(call: ast.Call) -> int | None:
    """Instructions of a pure ``_cute_inline_asm_elementwise`` call, or None
    when the call is not one (``is_pure_scalar_assembly`` admits only scalar
    arithmetic, the compiler's own Philox stream included)."""
    operands = inline_asm_operands(call)
    return None if operands is None else _pure_asm_instructions(*operands)


def heavy_cost(call: ast.Call, *, transcendentals: bool) -> int:
    """The estimated instructions of ``call`` when it is an operation worth
    carrying from the first sweep, else 0: a pure inline asm, or a
    transcendental when ``transcendentals`` (the explicit ``register``
    reload policies)."""
    if (count := asm_instruction_count(call)) is not None:
        return count
    if transcendentals and call_path_str(call.func) in TRANSCENDENTALS:
        return _TRANSCENDENTAL_COST
    return 0


def pure_value_call(call: ast.Call) -> bool:
    """A call whose result depends only on its operands and has no effect."""
    if call_path_str(call.func) == "_cute_inline_asm_elementwise":
        return asm_instruction_count(call) is not None
    return _is_proven_relocatable_call(call, allow_load=False)


def is_pure_expression(node: ast.expr) -> bool:
    return all(
        not isinstance(child, (ast.NamedExpr, ast.Lambda, ast.Await, ast.Yield))
        and (not isinstance(child, ast.Call) or pure_value_call(child))
        for child in ast.walk(node)
    )


def expression_cost(node: ast.expr, *, transcendentals: bool) -> int:
    """Estimated instructions to evaluate ``node`` once its names are known."""
    cost = 0
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            cost += heavy_cost(child, transcendentals=transcendentals) or 1
        elif isinstance(child, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.IfExp)):
            cost += 1
        elif isinstance(child, ast.Compare):
            cost += len(child.ops)
    return cost


def _python_dtype(value: object) -> str | None:
    if type(value) is int:
        return _PY_INT
    if type(value) is float:
        return _PY_FLOAT
    return None


def _scalar_dtype(value: object) -> str | None:
    """The DSL type a Python scalar operand converts to
    (``Numeric._from_python_value``): a float is a Float32, an int an Int32,
    or an Int64 outside the Int32 range."""
    if type(value) is float:
        return "cutlass.Float32"
    if type(value) is int:
        for dtype in ("cutlass.Int32", "cutlass.Int64"):
            low, high = _INT_RANGES[dtype]
            if low <= value <= high:
                return dtype
    return None


def _int_width(dtype: str) -> tuple[int, bool]:
    """The bits and signedness of an integer fragment type."""
    signed = dtype.startswith("cutlass.Int")
    bits = dtype.removeprefix("cutlass.Int" if signed else "cutlass.Uint")
    return int(bits), signed


def _promoted(a: str, b: str) -> str | None:
    """The type the DSL computes an operation on ``a`` and ``b`` values in
    (``_binary_op_type_promote``), or None where it refuses the pair; for
    the fragment types other than Boolean."""
    if a == b:
        return a
    if a in _FLOATS or b in _FLOATS:
        a_width = _FLOAT_WIDTHS[a] if a in _FLOATS else _int_width(a)[0]
        b_width = _FLOAT_WIDTHS[b] if b in _FLOATS else _int_width(b)[0]
        # An integer operand takes the float type at the wider width.
        promoted_a = a if a in _FLOATS else _FLOAT_OF_WIDTH.get(max(a_width, b_width))
        promoted_b = b if b in _FLOATS else _FLOAT_OF_WIDTH.get(max(a_width, b_width))
        if a_width > b_width and a_width >= 16:
            return promoted_a
        if b_width > a_width and b_width >= 16:
            return promoted_b
        if a_width == b_width:
            for dtype in ("cutlass.Float64", "cutlass.Float32", "cutlass.Float16"):
                if dtype in (promoted_a, promoted_b):
                    return dtype
        return None
    (a_width, a_signed), (b_width, b_signed) = _int_width(a), _int_width(b)
    if a_signed != b_signed:
        # The unsigned type when it is at least as wide, else the signed one.
        unsigned_width, signed_width = (
            (b_width, a_width) if a_signed else (a_width, b_width)
        )
        if unsigned_width >= signed_width:
            return b if a_signed else a
        return a if a_signed else b
    return a if a_width >= b_width else b


@dataclass(frozen=True)
class Value:
    # The DSL scalar type, when the expression proves it.
    dtype: str | None
    # The heavy operations (``heavy_cost``) its computation includes, by
    # number.
    heavy: frozenset[int]
    # Differs between iterations of the sweep loop.
    variant: bool


class SweepValues:
    """Hash-consed value numbers shared by the sweeps of one fusion group."""

    def __init__(
        self,
        *,
        transcendentals: bool,
        cache_dtypes: Mapping[str, str],
        binding_dtypes: Mapping[str, str],
        canonical: Callable[[str], str],
    ) -> None:
        self.transcendentals = transcendentals
        # Fragments the fuser filled in an earlier sweep: ``cache[slot]`` is
        # the value written there, the same in every later sweep.
        self._cache_dtypes = cache_dtypes
        # The DSL type of names a statement binds without showing it (the
        # uniform cutoffs' tuple assignment).
        self._binding_dtypes = binding_dtypes
        # The name each name renames to (the final rename merges a loop
        # carry's ``acc`` and ``v_2``): the environment holds canonical names.
        self.canonical = canonical
        self._numbers: dict[tuple[object, ...], int] = {}
        self.values: list[Value] = []
        # The estimated instructions of each heavy operation, by number.
        self._costs: dict[int, int] = {}
        self._constants: dict[int, object] = {}
        self._loads: set[int] = set()
        self._carried: set[int] = set()
        self._fresh = itertools.count()

    def _intern(
        self,
        key: tuple[object, ...],
        dtype: str | None,
        heavy: frozenset[int],
        variant: bool,
        cost: int = 0,
    ) -> int:
        number = self._numbers.get(key)
        if number is None:
            number = self._numbers[key] = len(self.values)
            if cost:
                self._costs[number] = cost
                heavy |= {number}
            self.values.append(Value(dtype, heavy, variant))
        return number

    def leaf(
        self, *key: object, dtype: str | None = None, variant: bool = False
    ) -> int:
        return self._intern(("leaf", *key), dtype, frozenset(), variant)

    def unique(self) -> int:
        """A value equal to nothing else: an effect or an unmodeled form."""
        return self.leaf("unique", next(self._fresh), variant=True)

    def load(self, key: str, dtype: str | None, *, variant: bool) -> int:
        """The value of the first sweep's load ``key`` (at this iteration)."""
        number = self.leaf("load", key, dtype=dtype, variant=variant)
        self._loads.add(number)
        return number

    def carried(self, loop: ast.For, name: str) -> int:
        """The value ``name`` holds in an iteration of ``loop`` before the
        iteration writes it: whatever the previous iteration left."""
        number = self.leaf("carried", id(loop), name, variant=True)
        self._carried.add(number)
        return number

    def index(self, depth: int) -> int:
        """The index of the sweep's ``depth``-th nested loop."""
        return self.leaf("index", depth, variant=True)

    def is_load(self, number: int | None) -> bool:
        return number in self._loads

    def heavy_instructions(self, number: int) -> int:
        """The estimated instructions of the heavy operations ``number``'s
        computation includes."""
        return sum(self._costs[heavy] for heavy in self.values[number].heavy)

    def cacheable(self, number: int) -> bool:
        return (
            self.values[number].dtype in CACHE_DTYPES
            and self.heavy_instructions(number) >= MIN_SAVED_COST
        )

    def _compound(
        self,
        key: tuple[object, ...],
        operands: Sequence[int],
        dtype: str | None,
        *,
        cost: int = 0,
    ) -> int:
        values = [self.values[operand] for operand in operands]
        return self._intern(
            (*key, *operands),
            dtype,
            frozenset().union(*(value.heavy for value in values)),
            any(value.variant for value in values),
            cost,
        )

    def _dtype(self, number: int) -> str | None:
        return self.values[number].dtype

    def _binary_dtype(self, op: ast.operator, left: int, right: int) -> str | None:
        """The DSL type of ``left op right``: a Python scalar operand converts
        first (``_scalar_dtype``), then both promote to one type
        (``_promoted``), so ``BFloat16 * 2.0`` is a Float32.  None where that
        is not provable: two Python scalars combine in Python, and an
        operation the numbering does not model."""
        left_dtype, right_dtype = self._dtype(left), self._dtype(right)
        if left_dtype in (_PY_INT, _PY_FLOAT) and right_dtype in CACHE_DTYPES:
            left_dtype = _scalar_dtype(self._constants.get(left))
        elif right_dtype in (_PY_INT, _PY_FLOAT) and left_dtype in CACHE_DTYPES:
            right_dtype = _scalar_dtype(self._constants.get(right))
        if left_dtype not in CACHE_DTYPES or right_dtype not in CACHE_DTYPES:
            return None
        assert left_dtype is not None and right_dtype is not None
        if _BOOLEAN in (left_dtype, right_dtype):
            same = left_dtype == right_dtype
            return _BOOLEAN if same and isinstance(op, _BITWISE_OPS) else None
        dtype = _promoted(left_dtype, right_dtype)
        if dtype in _FLOATS:
            return dtype if isinstance(op, _FLOAT_OPS) else None
        if dtype in _INT_RANGES:
            return dtype if isinstance(op, _INT_OPS) else None
        return None

    def expression(self, node: ast.expr, env: Mapping[str, int]) -> int:
        """Number ``node`` evaluated with the names in ``env``."""
        if isinstance(node, ast.Name):
            name = self.canonical(node.id)
            number = env.get(name)
            if number is not None:
                return number
            # A name bound outside the walked statements: the cutoffs
            # ``lower_uniform_comparisons`` hoists to the kernel's root, above
            # a persistent pid's loop, keep their DSL type.
            return self.leaf("global", name, dtype=self._binding_dtypes.get(node.id))
        if isinstance(node, ast.Constant):
            number = self.leaf(
                "constant",
                type(node.value).__name__,
                repr(node.value),
                dtype=_python_dtype(node.value),
            )
            self._constants[number] = node.value
            return number
        if isinstance(node, ast.Attribute):
            return self._compound(
                ("attr", node.attr), (self.expression(node.value, env),), None
            )
        if isinstance(node, ast.Subscript):
            return self._subscript(node, env)
        if isinstance(node, ast.Call):
            return self._call(node, env)
        if isinstance(node, ast.Tuple):
            return self._compound(
                ("tuple",), [self.expression(item, env) for item in node.elts], None
            )
        if isinstance(node, ast.BinOp):
            left = self.expression(node.left, env)
            right = self.expression(node.right, env)
            return self._compound(
                ("binop", type(node.op).__name__),
                (left, right),
                self._binary_dtype(node.op, left, right),
            )
        if isinstance(node, ast.UnaryOp):
            operand = self.expression(node.operand, env)
            operand_dtype = self._dtype(operand)
            if isinstance(node.op, ast.Not):
                dtype = _BOOLEAN if operand_dtype in CACHE_DTYPES else None
            else:
                dtype = operand_dtype if operand_dtype in CACHE_DTYPES else None
            return self._compound(("unary", type(node.op).__name__), (operand,), dtype)
        if isinstance(node, ast.Compare):
            operands = [
                self.expression(item, env) for item in (node.left, *node.comparators)
            ]
            dsl = any(self._dtype(operand) in CACHE_DTYPES for operand in operands)
            return self._compound(
                ("compare", *(type(op).__name__ for op in node.ops)),
                operands,
                _BOOLEAN if dsl else None,
            )
        if isinstance(node, ast.BoolOp):
            operands = [self.expression(item, env) for item in node.values]
            boolean = all(self._dtype(operand) == _BOOLEAN for operand in operands)
            return self._compound(
                ("boolop", type(node.op).__name__),
                operands,
                _BOOLEAN if boolean else None,
            )
        if isinstance(node, ast.IfExp):
            operands = [
                self.expression(item, env)
                for item in (node.test, node.body, node.orelse)
            ]
            body_dtype, else_dtype = self._dtype(operands[1]), self._dtype(operands[2])
            return self._compound(
                ("ifexp",),
                operands,
                body_dtype
                if body_dtype == else_dtype and body_dtype in CACHE_DTYPES
                else None,
            )
        return self.unique()

    def _subscript(self, node: ast.Subscript, env: Mapping[str, int]) -> int:
        index = self.expression(node.slice, env)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self._cache_dtypes
            # Not the sweep that fills it, where the name is carried.
            and env.get(self.canonical(node.value.id)) not in self._carried
        ):
            return self._compound(
                ("cache", node.value.id), (index,), self._cache_dtypes[node.value.id]
            )
        base = node.value
        # Only immutable bases: a pure call's result (``thread_idx()[0]``), a
        # layout's strides or shape, or a loaded packet.  A tensor or a
        # register fragment may change between the sweeps.
        immutable = (
            isinstance(base, ast.Call)
            or isinstance(base, ast.Attribute)
            and base.attr in ("stride", "shape")
            or isinstance(base, ast.Name)
            and env.get(self.canonical(base.id)) in self._loads
        )
        if not immutable:
            return self.unique()
        return self._compound(("item",), (self.expression(base, env), index), None)

    def _call(self, node: ast.Call, env: Mapping[str, int]) -> int:
        if not pure_value_call(node) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            return self.unique()
        func = node.func
        path = call_path_str(func)
        receiver: list[int] = []
        if isinstance(func, ast.Attribute) and (
            path is None or path.split(".", 1)[0] not in _MODULES
        ):
            # A method of a value, ``x.bitcast(cutlass.Uint32)``.
            receiver.append(self.expression(func.value, env))
            path = f".{func.attr}"
        if path is None:
            return self.unique()
        arguments = [self.expression(argument, env) for argument in node.args]
        keywords = [
            (keyword.arg, self.expression(keyword.value, env))
            for keyword in node.keywords
        ]
        operands = [*receiver, *arguments, *(number for _, number in keywords)]
        return self._compound(
            ("call", path, tuple(name for name, _ in keywords)),
            operands,
            self._call_dtype(node, path, receiver, arguments),
            cost=heavy_cost(node, transcendentals=self.transcendentals),
        )

    def _call_dtype(
        self, node: ast.Call, path: str, receiver: list[int], arguments: list[int]
    ) -> str | None:
        if path in CACHE_DTYPES:
            return path
        if path == ".bitcast" and node.args:
            target = call_path_str(node.args[0])
            return target if target in CACHE_DTYPES else None
        if path == "_cute_inline_asm_elementwise":
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            constraints = keywords.get("constraints")
            declared = call_path_str(keywords["dtype"]) if "dtype" in keywords else None
            single = (
                isinstance(constraints, ast.Constant)
                and isinstance(constraints.value, str)
                and constraints.value.count("=") == 1
            )
            return declared if single and declared in CACHE_DTYPES else None
        operand_dtypes = [self._dtype(operand) for operand in (*receiver, *arguments)]
        if path in _COMPARISONS:
            dsl = any(dtype in CACHE_DTYPES for dtype in operand_dtypes)
            return _BOOLEAN if dsl else None
        if path in _ARITHMETIC and len(arguments) == 2:
            return self._binary_dtype(_ARITHMETIC[path], *arguments)
        if path in ("cute.arch.fmax", "cute.arch.fmin"):
            # Both operands convert to Float32, and so does the result.
            return "cutlass.Float32"
        if path in ("abs", "operator.neg") and len(arguments) == 1:
            return operand_dtypes[0] if operand_dtypes[0] in CACHE_DTYPES else None
        if path in _SAME_TYPE_MATH or path in ("max", "min"):
            known = set(operand_dtypes)
            return operand_dtypes[0] if len(known) == 1 and known <= _FLOATS else None
        return None

    def run(
        self,
        statements: Sequence[ast.stmt],
        env: dict[str, int],
        *,
        loads: Mapping[int, int],
        scope: int | None,
    ) -> dict[int, int]:
        """Walk ``statements`` forward, updating ``env``.

        Returns the number of each ``NAME = value`` statement by statement id.
        ``loads`` gives proven load statements their load's number.  An impure
        value, and every name another statement writes, becomes a leaf of
        that statement in ``scope`` (one sweep's iteration, or None for the
        statements around the sweeps): two reads compare equal only when they
        read the same definition.
        """
        numbers: dict[int, int] = {}
        for statement in statements:
            target = assigned_name(statement)
            if target is not None:
                assert isinstance(statement, ast.Assign)
                name = self.canonical(target)
                number = loads.get(id(statement))
                if number is None:
                    number = (
                        self.expression(statement.value, env)
                        if is_pure_expression(statement.value)
                        else self.leaf("site", scope, id(statement), name)
                    )
                env[name] = numbers[id(statement)] = number
                continue
            self.overwrite(statement, env, scope=scope)
        return numbers

    def overwrite(
        self, statement: ast.stmt, env: dict[str, int], *, scope: int | None
    ) -> None:
        """Make every name ``statement`` writes a leaf of that statement."""
        for written in bound_names(statement):
            name = self.canonical(written)
            env[name] = self.leaf(
                "site",
                scope,
                id(statement),
                name,
                dtype=self._binding_dtypes.get(written),
            )


def bound_names(node: ast.AST) -> set[str]:
    """Names ``node`` may bind, loop targets included."""
    return set(ReadWrites.from_ast(node).writes) | names_written(node)


def number_sweep(
    values: SweepValues,
    loops: Sequence[ast.For],
    prefix: Sequence[ast.stmt],
    container: Sequence[ast.stmt],
    env: Mapping[str, int],
    *,
    loads: Mapping[int, int],
) -> dict[int, int]:
    """Number one sweep's ``container`` statements, and those of the bodies
    of its V-loops (``vector_lane_loop``), by statement id.

    ``loops`` are the sweep loop and its lane loop, outermost first; ``env``
    holds the enclosing names at the sweep loop and ``prefix`` the sweep
    loop's statements before the lane loop.  A name a loop writes reads,
    before its first write in an iteration, a value carried from the previous
    iteration: a leaf of this sweep, equal to nothing in another one.  That
    holds for the lane loop too, so a name the prefix assigns and the lane
    loop rebinds is carried from the prefix's value on.  Names are compared
    through ``values.canonical``.
    """
    env = dict(env)
    outer, lane = loops[0], loops[-1]

    def carry(loop: ast.For, skip: Collection[str] = ()) -> None:
        for name in {values.canonical(name) for name in bound_names(loop)}:
            if name not in skip:
                env[name] = values.carried(loop, name)

    carry(outer)
    for depth, loop in enumerate(loops):
        assert isinstance(loop.target, ast.Name)
        env[values.canonical(loop.target.id)] = values.index(depth)
    values.run(prefix, env, loads={}, scope=id(outer))
    if lane is not outer:
        assert isinstance(lane.target, ast.Name)
        carry(lane, skip={values.canonical(lane.target.id)})
    numbers: dict[int, int] = {}
    for statement in container:
        found = vector_lane_loop(statement)
        if found is None:
            numbers.update(values.run([statement], env, loads=loads, scope=id(outer)))
            continue
        # A V-loop's body is numbered too: its lane index compares equal
        # across sweeps like the loop indices, and a name it writes reads,
        # before its first write in a lane, what the lane before left.
        # After the loop each name it wrote holds its last lane's value.
        vloop, lane, _width = found
        lane_index = values.canonical(lane)
        carry(vloop, skip={lane_index})
        env[lane_index] = values.index(len(loops))
        numbers.update(values.run(vloop.body, env, loads=loads, scope=id(outer)))
        values.overwrite(vloop, env, scope=id(outer))
    return numbers


def vector_lane_loop(statement: ast.stmt) -> tuple[ast.For, str, int] | None:
    """``statement`` when it is a V-loop, ``for v in
    cutlass.range_constexpr(V)`` (the lanes of a packet the DSL unrolls while
    tracing), with its lane variable and width."""
    width = loop_extent(statement, kinds=("constexpr",), minimum=None)
    if width is None:
        return None
    loop = cast("ast.For", statement)
    return loop, cast("ast.Name", loop.target).id, width


@dataclass
class Cut:
    # Statements replaced by a read of the first sweep's value, by id, with
    # the value's number.
    cached: dict[int, int]
    # Statements no longer needed, by id.
    dead: set[int]
    # Estimated instructions per element the sweep no longer executes.
    saved: int


def plan_cut(
    values: SweepValues,
    statements: Sequence[ast.stmt],
    numbers: Mapping[int, int],
    first: Mapping[int, ast.stmt],
    *,
    live_after: Collection[str],
    loop_targets: Collection[str],
    reads: Callable[[ast.AST], set[str]],
    assigned: Callable[[ast.AST], str | None],
    removable: Callable[[ast.stmt], bool],
) -> Cut:
    """Choose which of a later sweep's values to read from the first sweep.

    ``statements`` is the sweep's lane-loop body, ``numbers`` their value
    numbers and ``first`` the first sweep's statement computing each number.
    ``live_after`` is what is live after the loop, ``loop_targets`` the
    loop's index, and ``reads`` and ``assigned`` give a statement's names
    the way ``live_after`` spells them.  Backward liveness runs to its fixed
    point over the loop's back edge; a live statement whose value the first
    sweep computes and is ``cacheable`` is cut, and a removable statement
    nothing reads any more is dead.
    """

    def backward(
        live_end: set[str], cut: bool
    ) -> tuple[dict[int, int], set[int], set[str]]:
        live = set(live_end)
        cached: dict[int, int] = {}
        dead: set[int] = set()
        for statement in reversed(statements):
            name = assigned(statement)
            if name is None:
                live |= reads(statement)
                continue
            if name not in live and removable(statement):
                dead.add(id(statement))
                continue
            live.discard(name)
            number = numbers.get(id(statement))
            if (
                cut
                and number is not None
                and number in first
                and values.cacheable(number)
            ):
                cached[id(statement)] = number
                continue
            live |= reads(statement)
        return cached, dead, live

    def solve(cut: bool) -> tuple[dict[int, int], set[int]]:
        live_end = set(live_after)
        while True:
            cached, dead, live_head = backward(live_end, cut)
            grown = live_end | (live_head - set(loop_targets))
            if grown == live_end:
                return cached, dead
            live_end = grown

    cached, dead = solve(True)
    # A cheap value of operands the sweep still computes (``Float32(keep)``
    # beside a carried ``keep`` the sweep also stores) is recomputed instead
    # of carried in a fragment of its own.  Its operands stay defined, so
    # recomputing it revives nothing.
    removed = {
        target
        for statement in statements
        if id(statement) in dead and (target := assigned(statement)) is not None
    }
    for statement in statements:
        if (
            id(statement) in cached
            and isinstance(statement, ast.Assign)
            and expression_cost(statement.value, transcendentals=values.transcendentals)
            <= _RECOMPUTE_COST
            and not reads(statement) & removed
        ):
            del cached[id(statement)]
    _, dead_before = solve(False)
    saved = sum(
        expression_cost(statement.value, transcendentals=values.transcendentals)
        for statement in statements
        if isinstance(statement, ast.Assign)
        # A load the fuser matched would be one fragment read, not this.
        and not values.is_load(numbers.get(id(statement)))
        and (
            id(statement) in cached
            or id(statement) in dead
            and id(statement) not in dead_before
        )
    )
    return Cut(cached, dead, saved)


def eliminate_dead(
    body: list[ast.stmt],
    region: range,
    live_out: Collection[str],
    *,
    removable: Callable[[ast.stmt], bool],
    reads: Callable[[ast.AST], set[str]],
    assigned: Callable[[ast.AST], str | None],
) -> list[ast.stmt]:
    """``body`` without the removable assignments in ``region`` (indices)
    whose value nothing reads before ``live_out``."""
    live = set(live_out)
    kept: list[ast.stmt] = []
    for index in range(len(body) - 1, -1, -1):
        statement = body[index]
        target = assigned(statement)
        if (
            index in region
            and target is not None
            and target not in live
            and removable(statement)
        ):
            continue
        live = live_in([statement], live, reads=reads, assigned=assigned)
        kept.append(statement)
    kept.reverse()
    return kept

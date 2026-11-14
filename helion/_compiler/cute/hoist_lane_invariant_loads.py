"""Hoist lane-invariant integer loads out of CuTe synthetic lane loops.

A lane loop runs its body once per lane, and the lane split
(``lane_loops/split.py``) re-runs in a persistent reduction's accumulate
sweep every statement the reduced value depends on, lane-invariant ones
included.  A dropout row whose seed is read from a tensor (``seeds[54]``, a
scalar load at a literal offset) therefore loads the seed once per lane::

    for synthetic_lane_1 in range(32):
        indices_1 = cutlass.Int32(cute.arch.thread_idx()[0]) + ...
        seed = (seeds.iterator + cutlass.Int32(54) * ...).load()
        ...
        v_20 = seed & v_19  # the Philox key schedule
        ...

LLVM keeps the load in the loop, and with it the integer key schedule the
Philox rounds derive from the seed: better-benchmark's
dual_dropout_residual_layernorm spent 56 integer adds per element on it.
Once the load sits in front of the loop, LLVM hoists that arithmetic itself.

The pass moves such a load in front of its loop.  A statement moves when

* it is a top-level ``name = value`` of a lane loop over a literal
  ``range(N)`` or ``cutlass.range_constexpr(N)`` with ``N >= 2``,
* ``value`` loads at least one scalar (a pointer ``.load()`` or a scalar
  ``cute.arch.load``) and no packet, every load from an integer or boolean
  kernel tensor through its iterator (``T.iterator`` plus offsets,
  ``_address_roots``: a pointer held in a name may reach any tensor); its
  other calls are pure and load nothing
  (``_is_proven_relocatable_call``), it reads memory through no subscript
  but launch coordinates and layout strides, it spells no float literal,
  conversion or math call, and it is provably an integer or a boolean
  (``_integral``), as is every arithmetic operation in it (``_rounds``), an
  integer conversion's operand included, with the names it reads typed by
  their assignments (``_integral_names``),
* no name ``value`` reads is written in the loop (a store into an element
  of a register, or a staging append, writes the register's name), except
  by a statement that moved before it (a load addressed by another moved
  load's value),
* ``name`` is written only there in the loop (its rename group included) and
  read nowhere in the loop before it, so no iteration reads a value an
  earlier one left behind, and
* every memory write of the loop goes through a kernel tensor's iterator,
  none an atomic, each proven disjoint from the loaded tensors
  (``_write_roots``), the loop's subscript stores go to
  ``cute.make_rmem_tensor`` registers, every call of the loop is pure, a
  load or a recognized store, so no barrier or helper of unknown effect sits
  in it (``operator.getitem``, which reads memory the subscript checks do
  not see, counts as unknown), and no ``break``, ``continue``, ``return`` or
  ``raise`` cuts a lane short;

and, for the loop's moves together, all arithmetic of the loop that they
make loop-invariant (an operation reading a moved name, directly or through
such a value, a stored one read back included, and nothing else the loop
writes that is not invariant) is integer arithmetic
(``_only_integers_become_invariant``).

The loop runs every lane, and a top-level statement runs in each of them, so
the moved load reads an address every lane read, and nothing the loop does
changes what is there: each lane read the value the moved load reads.  A
moved value is an integer, so the arithmetic the loop keeps that reads it is
integer arithmetic; no float arithmetic becomes loop-invariant for LLVM to
move after it, so every floating-point operation stays in its loop with its
operands, contracts as it did, and the results are unchanged bit for bit.
That holds for arithmetic written as a call (``operator.mul``, a math
function) as for an operator, and for a value that reaches arithmetic
through memory: LLVM forwards a stored value to the loads of the same
register element or address that follow the store, so a read that may read
back one of the loop's stores counts as reading the stored value and the
tests of the branches around the store.
Inner lane loops are rewritten first, so a load an inner loop gives up can
leave an outer lane loop too.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import TypeGuard

from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ._ast_pass_utils import _FOLD_DOT_HELPER
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import loop_extent
from ._ast_pass_utils import name_assignment
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_stored_or_deleted
from ._ast_pass_utils import resolve_alias
from ._ast_pass_utils import tensor_roots_may_alias
from .address_maps import _LOOP_CALLS
from .dtypes import CUTLASS_INTEGER_TYPE_NAMES
from .fuse_two_pass_loads import _is_store_call
from .fuse_two_pass_loads import _tensor_arg_roots
from .hoist_lane_invariant_reductions import _is_atomic_call
from .hoist_lane_invariant_reductions import _is_private_store_staging_append
from .hoist_lane_invariant_reductions import _mutation_roots
from .hoist_lane_invariant_reductions import _rmem_declaration
from .packet_access import call_write_roots
from .packet_access import match_packet_access
from .packet_access import store_pointer
from .scalar_integer import _INTEGER_OPERATORS
from .scalar_integer import _integer_expression

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterator
    from collections.abc import Mapping

# Calls that compute floats: a moved statement holds none.
_FLOAT_CALL_PREFIXES = (
    "cutlass.Float",
    "cutlass.BFloat",
    "cute.math.",
    "math.",
    "cute.arch.fmax",
    "cute.arch.fmin",
)
# Calls that compute a binary operator: the ``operator`` module's
# (``operator.mul(a, b)`` is ``a * b``) and Python's modulo helper.
_BINARY_CALLS: dict[str, ast.operator] = {
    "_cute_python_mod": ast.Mod(),
    "operator.add": ast.Add(),
    "operator.and_": ast.BitAnd(),
    "operator.floordiv": ast.FloorDiv(),
    "operator.lshift": ast.LShift(),
    "operator.mod": ast.Mod(),
    "operator.mul": ast.Mult(),
    "operator.or_": ast.BitOr(),
    "operator.pow": ast.Pow(),
    "operator.rshift": ast.RShift(),
    "operator.sub": ast.Sub(),
    "operator.truediv": ast.Div(),
    "operator.xor": ast.BitXor(),
}
# The other pure helpers of ``_is_proven_relocatable_call`` that compute
# arithmetic: the fp4 dot and the matmul fold.
_ARITHMETIC_HELPERS = frozenset(
    {"_cute_float4_e2m1fn_x16_dot_float32", _FOLD_DOT_HELPER}
)
# Where a memory read or store reaches (``_memory_reads``,
# ``_may_read_back``): a register, by rename group, or kernel tensors; None
# is any memory for a read and any tensor for a store.
_Where = str | frozenset[str] | None


class _Context:
    def __init__(
        self,
        body: list[ast.stmt],
        tensor_dtypes: Mapping[str, str],
        proven_disjoint_tensor_pairs: Collection[frozenset[str]],
        rename_groups: Mapping[str, str],
        rmem_names: set[str],
        integer_names: Collection[str],
        float_names: Collection[str],
    ) -> None:
        self.body = body
        self.tensor_names = set(tensor_dtypes)
        # The integer and boolean tensors, the only ones whose loads move.
        self.integer_tensors = {
            name
            for name, dtype in tensor_dtypes.items()
            if dtype in CUTLASS_INTEGER_TYPE_NAMES
        }
        self.disjoint = set(proven_disjoint_tensor_pairs)
        self.rename_groups = rename_groups
        self.rmem_names = rmem_names
        self.integer_names = frozenset(integer_names)
        self.float_names = frozenset(float_names)
        # The statements moved in front of a loop so far, which the body may
        # no longer reach (in front of a top-level loop).
        self.moved: list[ast.stmt] = []
        self._integral_names: set[str] | None = None

    def canonical(self, names: set[str]) -> set[str]:
        return {resolve_alias(name, self.rename_groups) for name in names}

    def integral_names(self) -> set[str]:
        """``_integral_names`` of the kernel body, computed on first use."""
        if self._integral_names is None:
            self._integral_names = _integral_names(self)
        return self._integral_names


def _integral_names(ctx: _Context) -> set[str]:
    """The names that hold an integer or a boolean wherever the kernel reads
    them: the constexprs and integer parameters (``integer_names``) it does
    not assign, and each name whose every assignment in the body, its rename
    group's included, is integral or a loop over a ``range``.  A name bound
    any other way (a tuple target, a loop over anything else) or sharing a
    rename group with a float parameter is not."""
    module = ast.Module(body=[*ctx.body, *ctx.moved], type_ignores=[])
    values: dict[str, list[ast.expr]] = {}
    bound: set[int] = set()
    for node in ast.walk(module):
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            target = node.target
            value = ast.BinOp(
                left=ast.Name(id=node.target.id, ctx=ast.Load()),
                op=node.op,
                right=node.value,
            )
        elif (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and call_path_str(node.iter.func) in _LOOP_CALLS
        ):
            target, value = node.target, ast.Constant(0)
        if isinstance(target, ast.Name) and value is not None:
            bound.add(id(target))
            group = resolve_alias(target.id, ctx.rename_groups)
            values.setdefault(group, []).append(value)
    members: dict[str, set[str]] = {}
    untyped = ctx.canonical(set(ctx.float_names))
    for node in ast.walk(module):
        if isinstance(node, ast.Name):
            group = resolve_alias(node.id, ctx.rename_groups)
            members.setdefault(group, set()).add(node.id)
            if isinstance(node.ctx, (ast.Store, ast.Del)) and id(node) not in bound:
                untyped.add(group)
    base = {
        name
        for name in ctx.integer_names
        if resolve_alias(name, ctx.rename_groups) not in values.keys() | untyped
    }
    # Assume every candidate integral, then drop the ones an assignment
    # disproves until none does: a loop-carried integer stays integral.
    candidates = set(values) - untyped
    while True:
        known = base | {name for group in candidates for name in members[group]}
        disproved = {
            group
            for group in candidates
            if not all(
                _integral(value, known, ctx.integer_tensors) for value in values[group]
            )
        }
        if not disproved:
            return known
        candidates -= disproved


def _scalar_load_pointer(node: ast.AST) -> ast.AST | None:
    """The pointer of ``node`` when it is a scalar load (a pointer ``.load()``
    or a scalar ``cute.arch.load``), else None."""
    access = match_packet_access(node)
    if access is None or access.kind not in ("load", "arch_load"):
        return None
    return access.pointer


def _binary(node: ast.AST) -> tuple[ast.operator, ast.expr, ast.expr] | None:
    """``(op, left, right)`` of a binary operation, written as an operator or
    as a call that computes one (``_BINARY_CALLS``), else None."""
    if isinstance(node, ast.BinOp):
        return node.op, node.left, node.right
    if (
        isinstance(node, ast.Call)
        and (op := _BINARY_CALLS.get(call_path_str(node.func) or "")) is not None
        and len(node.args) == 2
        and not node.keywords
    ):
        return op, node.args[0], node.args[1]
    return None


def _rounds(node: ast.AST) -> TypeGuard[ast.expr]:
    """Whether ``node`` is arithmetic, which rounds unless its operands are
    integers: a binary operation (``_binary``), a math call or an arithmetic
    helper.  Conversions, comparisons, selects, extrema, bitcasts and loads
    round nothing a contraction could change."""
    if _binary(node) is not None:
        return True
    if not isinstance(node, ast.Call):
        return False
    path = call_path_str(node.func) or ""
    return path.startswith(("math.", "cute.math.")) or path in _ARITHMETIC_HELPERS


def _store_root(target: ast.AST) -> ast.AST:
    """The name an assignment target ``R[i].x`` stores into: ``R``."""
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    return target


def _address_roots(pointer: ast.AST, tensor_names: set[str]) -> frozenset[str] | None:
    """The kernel tensors an address reaches (``_tensor_arg_roots``) when it
    is a kernel tensor's ``T.iterator`` plus offsets, else None.
    ``_tensor_arg_roots`` collects the iterators an address spells, so it
    would resolve a pointer held in a name plus an offset loaded from
    ``seeds`` (``pointer + (seeds.iterator + 0).load()``) to ``seeds``."""
    base = pointer
    while (binary := _binary(base)) is not None and isinstance(
        binary[0], (ast.Add, ast.Sub)
    ):
        base = binary[1]
    if not (
        isinstance(base, ast.Attribute)
        and base.attr == "iterator"
        and isinstance(base.value, ast.Name)
        and base.value.id in tensor_names
    ):
        return None
    return _tensor_arg_roots(pointer, tensor_names)


def _write_roots(call: ast.Call, ctx: _Context) -> frozenset[str] | None:
    """The kernel tensors a store call writes (``call_write_roots`` over
    ``_address_roots``), or None: an atomic, or an address not resolved."""
    if _is_atomic_call(call):
        return None
    return call_write_roots(
        call, lambda pointer: _address_roots(pointer, ctx.tensor_names)
    )


def _loop_admits_hoisting(loop: ast.For, ctx: _Context) -> list[frozenset[str]] | None:
    """The destination roots of ``loop``'s memory writes when every write is
    resolved (``_write_roots``), every call is known and every lane runs the
    whole body, else None.  ``operator.getitem`` is not known: it reads
    memory that the subscript checks do not see."""
    store_roots: list[frozenset[str]] = []
    for node in ast.walk(ast.Module(body=loop.body, type_ignores=[])):
        if isinstance(node, (ast.Break, ast.Continue, ast.Return, ast.Raise)):
            return None
        if isinstance(node, ast.Call):
            path = call_path_str(node.func)
            if _is_store_call(node):
                roots = _write_roots(node, ctx)
                if roots is None:
                    return None
                store_roots.append(roots)
            elif path == "operator.getitem" or not (
                _is_proven_relocatable_call(node, allow_load=True)
                or _is_private_store_staging_append(node)
                or path in _LOOP_CALLS
            ):
                return None
        elif isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(
            node.ctx, (ast.Store, ast.Del)
        ):
            root = _store_root(node)
            if not (isinstance(root, ast.Name) and root.id in ctx.rmem_names):
                return None
    return store_roots


def _float_valued(value: ast.AST) -> bool:
    """Whether ``value`` spells float arithmetic: a float literal, a float
    conversion or a math call."""
    for node in ast.walk(value):
        if isinstance(node, ast.Constant) and type(node.value) is float:
            return True
        if isinstance(node, ast.Call):
            path = call_path_str(node.func) or ""
            if path.startswith(_FLOAT_CALL_PREFIXES):
                return True
    return False


def _hoist_from_loop(loop: ast.For, ctx: _Context) -> list[ast.stmt]:
    """``loop`` preceded by the statements it gives up (see the module
    docstring), or ``[loop]``."""
    if (
        getattr(loop, HELION_LANE_LOOP_VAR_ATTR, None) is None
        or loop_extent(loop, minimum=2) is None
    ):
        return [loop]
    store_roots = _loop_admits_hoisting(loop, ctx)
    if store_roots is None:
        return [loop]
    # Every name the loop writes (its lane variable included), by rename
    # group, with the number of writes.  A register an element store or a
    # staging append writes counts twice: it is never a value that moves, and
    # nothing reading it moves either.
    writes: dict[str, int] = {}
    for node in ast.walk(loop):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            name = resolve_alias(node.id, ctx.rename_groups)
            writes[name] = writes.get(name, 0) + 1
    for root in _mutation_roots(loop):
        name = resolve_alias(root, ctx.rename_groups)
        writes[name] = writes.get(name, 0) + 2
    moved: list[ast.stmt] = []
    moved_names: set[str] = set()
    # The moved statements' targets by source name, as ``_integral`` reads
    # them.
    integral: set[str] = set()
    read_before: set[str] = set()
    for stmt in loop.body:
        assignment = name_assignment(stmt)
        if assignment is not None:
            name, value = assignment
            target = resolve_alias(name, ctx.rename_groups)
            if _moves(
                value,
                target,
                writes,
                moved_names,
                integral,
                read_before,
                store_roots,
                ctx,
            ):
                moved.append(stmt)
                moved_names.add(target)
                integral.add(name)
        read_before |= ctx.canonical(names_read(stmt))
    if (
        not moved
        or len(moved) == len(loop.body)
        or not _only_integers_become_invariant(
            loop, moved, moved_names, integral, writes, ctx
        )
    ):
        return [loop]
    loop.body = [stmt for stmt in loop.body if all(stmt is not m for m in moved)]
    ctx.moved.extend(moved)
    return [*moved, loop]


def _only_integers_become_invariant(
    loop: ast.For,
    moved: list[ast.stmt],
    moved_names: set[str],
    moved_integral: set[str],
    writes: dict[str, int],
    ctx: _Context,
) -> bool:
    """Whether all arithmetic of ``loop`` that the ``moved`` loads make
    loop-invariant (it reads a moved name, directly or through such a value,
    and nothing else the loop writes that is not invariant) is integer
    arithmetic.

    LLVM's loop-invariant code motion may move such arithmetic out of the
    loop once its operands are; a float product moved that way would no
    longer contract with an add the loop keeps, changing a rounding.
    Integer arithmetic carries no rounding, so moving it changes no result.
    A value reaches arithmetic through memory too: LLVM forwards a stored
    value to the loads of the same register element or address that follow
    the store.

    Invariance is over-approximated (``_loop_invariant_names``), which only
    makes this check stricter: a name the loop writes more than once, through
    a tuple target or an augmented assignment counts as invariant when one of
    its writes could be, and a memory read that may read back one of the
    loop's stores when its address or one of those stores could be.
    """
    memory, variant, affected = _loop_invariant_names(
        loop, moved, moved_names, writes, ctx
    )
    # By source name, as ``_integral`` reads them; ``_moves`` proved each
    # moved value integral.
    integral = ctx.integral_names() | moved_integral

    def made_invariant(node: ast.AST) -> bool:
        reads = _reads(node, memory, ctx)
        return bool(reads & affected) and not reads & variant

    # Any arithmetic, not only whole statements: LLVM moves single
    # instructions.  An augmented assignment computes its operator on its
    # target.
    for node in ast.walk(ast.Module(body=loop.body, type_ignores=[])):
        if isinstance(node, ast.AugAssign):
            target = (
                ast.Name(id=node.target.id, ctx=ast.Load())
                if isinstance(node.target, ast.Name)
                else node.target
            )
            node = ast.BinOp(left=target, op=node.op, right=node.value)
        if (
            _rounds(node)
            and made_invariant(node)
            and not _integral(node, integral, ctx.integer_tensors)
        ):
            return False
    return True


def _paired(target: ast.expr, value: ast.expr) -> list[tuple[ast.expr, ast.expr]]:
    """``(target, value)`` pairs of an assignment: a tuple target's elements
    paired with a literal tuple's (nested ones too), else the whole value."""
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(elt, ast.Starred) for elt in target.elts)
        and not any(isinstance(elt, ast.Starred) for elt in value.elts)
    ):
        return [
            pair
            for target_elt, value_elt in zip(target.elts, value.elts, strict=True)
            for pair in _paired(target_elt, value_elt)
        ]
    return [(target, value)]


def _loop_invariant_names(
    loop: ast.For,
    moved: list[ast.stmt],
    moved_names: set[str],
    writes: dict[str, int],
    ctx: _Context,
) -> tuple[dict[int, str], set[str], set[str]]:
    """``(memory, variant, affected)`` for ``loop`` once the ``moved``
    statements (writing ``moved_names``) sit in front of it: ``memory`` names
    the reads that may read back one of the loop's stores (by node id, as
    ``_reads`` takes it), ``variant`` holds the groups the loop writes that
    are not invariant, and ``affected`` the invariant ones that one of their
    invariant writes makes so by reading a moved name, directly or through
    another affected group.

    A rename group the loop writes counts as invariant unless each of its
    writes reads a group that is not: the greatest fixed point, so a name
    written twice in one iteration (``t = c(seed)``, then ``t = t * f``), in
    both branches of an ``if``, by a tuple target (its names paired with a
    literal tuple's elements, else each reading the whole value) or by an
    augmented assignment (which also reads its target) is invariant when one
    of its writes could be: the values an earlier write leaves in the group
    may be (``t = c(seed); product = t * f`` before ``t = c(lane)``).  A
    write reads the tests of the branches and loops around it too, which
    decide what it writes.  The lane variable is not invariant.

    A memory read that may read back one of the loop's stores
    (``_memory_reads``, ``_may_read_back``) is a group of its own.  LLVM
    forwards a stored value to the loads of the same register element or
    address that follow the store, whatever that address, so each such store
    writes the read's group, reading the stored value and the tests around
    the store, and so does the read's address, for the value it finds
    unforwarded.  The names of the registers the loop stores into, which
    only such reads read, are not invariant.
    """
    # The tests of the branches and loops around each node.
    control: dict[int, list[ast.expr]] = {}

    def mark(stmts: list[ast.stmt], tests: list[ast.expr]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.If, ast.While, ast.For)):
                test = stmt.iter if isinstance(stmt, ast.For) else stmt.test
                mark(stmt.body, [*tests, test])
                mark(stmt.orelse, [*tests, test])
            for node in ast.walk(stmt):
                control.setdefault(id(node), tests)

    mark(loop.body, [])
    body = ast.Module(body=loop.body, type_ignores=[])
    # The loop's stores, as (where, values, store): a register's group, or
    # the kernel tensors a store call writes (None for a staging append, whose
    # store the loop does not show).
    stores: list[tuple[_Where, list[ast.expr], ast.AST]] = []
    for node in ast.walk(body):
        if isinstance(node, ast.Call) and _is_store_call(node):
            # The stored value and the operands that decide whether it is
            # stored, not the address: a forwarded value is the same whatever
            # address it went through.  Where the address is not one operand,
            # any operand may be the stored value.
            where = _write_roots(node, ctx)
            if (pointer := store_pointer(node)) is not None or not node.args:
                values = [arg for arg in node.args if arg is not pointer]
                stores.append((where, values, node))
            else:
                stores.extend((where, [arg], node) for arg in node.args)
        elif isinstance(node, ast.Call) and _is_private_store_staging_append(node):
            stores.append((None, node.args, node))
        elif (
            isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign))
            and node.value is not None
        ):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            stores.extend(
                (
                    register,
                    [value, part] if isinstance(node, ast.AugAssign) else [value],
                    node,
                )
                for target in targets
                for part, value in _paired(target, node.value)
                for register in ctx.canonical(_mutation_roots(part))
            )
    # The reads outside the moved statements that may read one of those
    # stores back, with the stores.
    reads_back: list[tuple[ast.AST, _Where, list[tuple[list[ast.expr], ast.AST]]]] = []
    staying = [stmt for stmt in loop.body if all(stmt is not m for m in moved)]
    for read, where in _memory_reads(staying, ctx):
        matched = [
            (values, store)
            for written, values, store in stores
            if _may_read_back(where, written, ctx.disjoint)
        ]
        if matched:
            reads_back.append((read, where, matched))
    memory = {
        id(read): f"<memory read {index}>"
        for index, (read, _, _) in enumerate(reads_back)
    }

    def reads_of(node: ast.AST) -> set[str]:
        return _reads(node, memory, ctx)

    def tests(node: ast.AST) -> set[str]:
        return set().union(*map(reads_of, control.get(id(node), [])))

    reads_of_writes: dict[str, list[set[str]]] = {}

    def record(target: ast.expr, value: ast.expr, extra: set[str]) -> None:
        for part, part_value in _paired(target, value):
            value_reads = reads_of(part_value) | extra
            for name in names_stored_or_deleted(part):
                group = resolve_alias(name, ctx.rename_groups)
                reads_of_writes.setdefault(group, []).append(value_reads)

    for node in ast.walk(body):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                record(target, node.value, tests(node))
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)) and node.value:
            extra = (
                {resolve_alias(node.target.id, ctx.rename_groups)}
                if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                else set()
            )
            record(node.target, node.value, extra | tests(node))
        elif isinstance(node, (ast.For, ast.comprehension)):
            record(node.target, node.iter, tests(node))
        elif isinstance(node, ast.NamedExpr):
            record(node.target, node.value, tests(node))
    for read, where, matched in reads_back:
        address = set().union(*map(reads_of, ast.iter_child_nodes(read)))
        if isinstance(where, str):
            address.discard(where)
        reads_of_writes[memory[id(read)]] = [
            address,
            *(
                set().union(*map(reads_of, values)) | tests(store)
                for values, store in matched
            ),
        ]
    written = set(writes) | set(memory.values())
    invariant = set(reads_of_writes) - ctx.canonical(_mutation_roots(loop))
    while dropped := {
        group
        for group in invariant
        if all(reads & written - invariant for reads in reads_of_writes[group])
    }:
        invariant -= dropped
    invariant |= moved_names
    variant = written - invariant
    affected = set(moved_names)
    while grown := {
        group
        for group in invariant - affected
        if any(
            reads & affected and not reads & variant
            for reads in reads_of_writes.get(group, ())
        )
    }:
        affected |= grown
    return memory, variant, affected


def _memory_reads(
    stmts: list[ast.stmt], ctx: _Context
) -> Iterator[tuple[ast.AST, _Where]]:
    """Each memory read under ``stmts``, with where it reads: a register's
    group for an element of it (an augmented assignment's target included),
    a load of it or its name passed on whole; the kernel tensors of a
    tensor's subscript or of a load through an address ``_address_roots``
    resolves; or anything (None) for a subscript of another name or a load
    through any other address."""
    # The names a subscript or a load reaches its register through.
    heads: set[int] = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                heads.add(id(node.value))
            if isinstance(node, ast.AugAssign) and isinstance(
                node.target, ast.Subscript
            ):
                yield node.target, _subscript_reads(node.target, ctx)
            elif _memory_subscript(node):
                yield node, _subscript_reads(node, ctx)
            elif (
                isinstance(node, ast.Call)
                and _is_proven_relocatable_call(node, allow_load=True)
                and not _is_proven_relocatable_call(node, allow_load=False)
            ):
                access = match_packet_access(node)
                pointer = None if access is None else access.pointer
                if isinstance(pointer, ast.Name) and pointer.id in ctx.rmem_names:
                    heads.add(id(pointer))
                    yield node, resolve_alias(pointer.id, ctx.rename_groups)
                else:
                    yield (
                        node,
                        None
                        if pointer is None
                        else _address_roots(pointer, ctx.tensor_names),
                    )
            elif (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in ctx.rmem_names
                and id(node) not in heads
            ):
                yield node, resolve_alias(node.id, ctx.rename_groups)


def _subscript_reads(node: ast.Subscript, ctx: _Context) -> _Where:
    """Where a subscript reads (``_memory_reads``)."""
    base = node.value
    if isinstance(base, ast.Name) and base.id in ctx.rmem_names:
        return resolve_alias(base.id, ctx.rename_groups)
    if isinstance(base, ast.Name) and base.id in ctx.tensor_names:
        return frozenset({base.id})
    return None


def _may_read_back(
    read: _Where, written: _Where, disjoint: set[frozenset[str]]
) -> bool:
    """Whether a read of ``read`` (``_memory_reads``) may find what a store
    into ``written`` wrote: the same register, or tensors that may alias.  A
    read of anything finds any store; a store into any tensor (a staging
    append) is found by every read but a register's."""
    if isinstance(read, str) or isinstance(written, str):
        return read is None or read == written
    return tensor_roots_may_alias(read, written, disjoint)


def _reads(node: ast.AST, memory: Mapping[int, str], ctx: _Context) -> set[str]:
    """The groups ``node`` reads: the rename groups of its names, with a read
    of ``memory`` (by node id) read as its group instead of the names under
    it."""
    groups: set[str] = set()
    pending = [node]
    while pending:
        child = pending.pop()
        if (group := memory.get(id(child))) is not None:
            groups.add(group)
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            groups.add(resolve_alias(child.id, ctx.rename_groups))
        pending.extend(ast.iter_child_nodes(child))
    return groups


def _memory_subscript(node: ast.AST) -> TypeGuard[ast.Subscript]:
    """A subscript read other than of a launch coordinate
    (``thread_idx()[k]``, ``block_idx()[k]``) or a layout stride
    (``T.layout.stride[k]``), which read no memory."""
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and not _integer_expression(node, "", frozenset())
    )


def _integral(node: ast.expr, integral: set[str], integer_tensors: set[str]) -> bool:
    """Whether ``node`` yields an integer or a boolean: an ``integral`` name,
    an integer literal, an integer or boolean conversion (whatever it
    converts: arithmetic inside it is checked as its own operation), a load
    of ``integer_tensors`` or a scalar ``cute.arch.load`` of an integer
    carrier, an address, or integer arithmetic (an operator or a call that
    computes one, ``_binary``), comparisons and selects over such values."""
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, bool)
    if isinstance(node, ast.Name):
        return node.id in integral
    if (
        isinstance(node, ast.Call)
        and call_path_str(node.func) in CUTLASS_INTEGER_TYPE_NAMES
    ):
        return True
    access = match_packet_access(node)
    if access is not None and access.kind == "arch_load":
        # A scalar ``cute.arch.load`` yields the carrier type it names.
        return access.carrier in CUTLASS_INTEGER_TYPE_NAMES
    if (binary := _binary(node)) is not None:
        op, left, right = binary
        if isinstance(op, (ast.Add, ast.Sub)) and _is_pointer(
            left, integral, integer_tensors
        ):
            # An address: a tensor's iterator plus integer offsets.
            return _integral(right, integral, integer_tensors)
        return isinstance(op, _INTEGER_OPERATORS) and all(
            _integral(side, integral, integer_tensors) for side in (left, right)
        )
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, (ast.Not, ast.Invert, ast.USub, ast.UAdd)) and (
            _integral(node.operand, integral, integer_tensors)
        )
    if isinstance(node, ast.Compare):
        return all(
            _integral(operand, integral, integer_tensors)
            for operand in (node.left, *node.comparators)
        )
    if isinstance(node, ast.BoolOp):
        return all(_integral(value, integral, integer_tensors) for value in node.values)
    if isinstance(node, ast.IfExp):
        return all(
            _integral(part, integral, integer_tensors)
            for part in (node.test, node.body, node.orelse)
        )
    # Launch coordinates, layout strides and loads of integer tensors.
    return _integer_expression(node, "", integral, integer_tensors=integer_tensors)


def _is_pointer(node: ast.expr, integral: set[str], integer_tensors: set[str]) -> bool:
    """``T.iterator`` plus integer offsets."""
    if isinstance(node, ast.Attribute) and node.attr == "iterator":
        return isinstance(node.value, ast.Name)
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.Add, ast.Sub))
        and _is_pointer(node.left, integral, integer_tensors)
        and _integral(node.right, integral, integer_tensors)
    )


def _moves(
    value: ast.expr,
    target: str,
    writes: dict[str, int],
    moved_names: set[str],
    integral: set[str],
    read_before: set[str],
    store_roots: list[frozenset[str]],
    ctx: _Context,
) -> bool:
    """Whether the top-level ``target = value`` leaves its lane loop."""
    if writes.get(target) != 1 or target in read_before:
        return False
    if any(
        name in writes and name not in moved_names
        for name in ctx.canonical(names_read(value))
    ):
        return False
    # Its only loads are scalar ones, whose tensors are checked below; every
    # other call is pure, and it reads memory through no subscript (a kernel
    # tensor's element, a register through another name), which no check
    # below would see.
    calls = [node for node in ast.walk(value) if isinstance(node, ast.Call)]
    # The value and each arithmetic operation in it, an integer conversion's
    # operand included, are integers, with the names from outside the loop
    # typed by their assignments.
    known = integral | ctx.integral_names()
    if (
        _float_valued(value)
        or any(_memory_subscript(node) for node in ast.walk(value))
        or not _integral(value, known, ctx.integer_tensors)
        or not all(
            _integral(node, known, ctx.integer_tensors)
            for node in ast.walk(value)
            if _rounds(node)
        )
        or not all(
            _scalar_load_pointer(node) is not None
            or _is_proven_relocatable_call(node, allow_load=False)
            for node in calls
        )
    ):
        return False
    pointers = [
        pointer for node in calls if (pointer := _scalar_load_pointer(node)) is not None
    ]
    if not pointers:
        return False
    # Each load reads integer tensors through one's iterator, and no store of
    # the loop may write them.
    for pointer in pointers:
        roots = _address_roots(pointer, ctx.tensor_names)
        if (
            roots is None
            or any(root not in ctx.integer_tensors for root in roots)
            or any(
                tensor_roots_may_alias(roots, destination, ctx.disjoint)
                for destination in store_roots
            )
        ):
            return False
    return True


def _rewrite(body: list[ast.stmt], ctx: _Context) -> list[ast.stmt]:
    result: list[ast.stmt] = []
    for stmt in body:
        for field in ("body", "orelse", "finalbody"):
            child = getattr(stmt, field, None)
            if isinstance(child, list) and all(
                isinstance(item, ast.stmt) for item in child
            ):
                setattr(stmt, field, _rewrite(child, ctx))
        if isinstance(stmt, ast.For):
            result.extend(_hoist_from_loop(stmt, ctx))
        else:
            result.append(stmt)
    return result


def hoist_lane_invariant_loads(
    body: list[ast.stmt],
    *,
    tensor_dtypes: Mapping[str, str],
    proven_disjoint_tensor_pairs: Collection[frozenset[str]] = (),
    rename_groups: Mapping[str, str] | None = None,
    integer_names: Collection[str] = (),
    float_names: Collection[str] = (),
) -> list[ast.stmt]:
    """Move lane-invariant integer loads in front of their lane loops (see
    the module docstring).  ``integer_names`` and ``float_names`` are the
    kernel's integer and float names that ``body`` reads without assigning
    them: constexprs and parameters."""
    rmem_names = {
        declaration[0]
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, ast.stmt)
        and (declaration := _rmem_declaration(node)) is not None
    }
    ctx = _Context(
        body,
        tensor_dtypes,
        proven_disjoint_tensor_pairs,
        rename_groups or {},
        rmem_names,
        integer_names,
        float_names,
    )
    return _rewrite(body, ctx)

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

A load that varies with an inner lane but not with the lane loop around it
cannot leave its loop, and yet every outer lane repeats it.  A softmax
backward under a document mask (better-benchmark sum_02fe2c82b1fd) gives each
thread two rows of 128 keys; for each key it reads the key's position and
the document id of that position, the same in both rows::

    for vec_lane_2 in cutlass.range_constexpr(2):        # the thread's rows
        ...
        for synthetic_lane_3 in cutlass.range_constexpr(8):   # its keys
            indices_3 = reduction_lane_base_3 + cutlass.Int32(synthetic_lane_3 ...)
            tmp16 = _cute_scalar_policy_load(in2.iterator + ... indices_3 ..., cutlass.Int64, 'last')
            ...
            tmp20 = _cute_element_policy_load_if(in3.iterator + ... tmp19 ..., ..., cutlass.Int64, 'l2_last')

The policy loads are side-effecting inline PTX, so LLVM keeps every copy.
``_tabulate_inner_loads`` reads them once per key, before the row loop, into
a table the inner loop indexes by its lane::

    indices_3_lanes = []
    tmp16_lanes = []
    tmp20_lanes = []
    for synthetic_lane_3_table in cutlass.range_constexpr(8):
        indices_3_by_lane = reduction_lane_base_3 + cutlass.Int32(synthetic_lane_3_table ...)
        tmp16_by_lane = _cute_scalar_policy_load(in2.iterator + ... indices_3_by_lane ...)
        ...
        indices_3_lanes.append(indices_3_by_lane)
        tmp16_lanes.append(tmp16_by_lane)
        tmp20_lanes.append(tmp20_by_lane)
    for vec_lane_2 in cutlass.range_constexpr(2):
        ...
        for synthetic_lane_3 in cutlass.range_constexpr(8):
            indices_3 = indices_3_lanes[synthetic_lane_3]
            tmp16 = tmp16_lanes[synthetic_lane_3]
            ...

The table loop walks the lanes with a variable of its own, so the outer
loop's reads of a name of the inner lane keep their values.  Only the names
the rows (or code outside the outer loop) still read get a table: an address
temporary that only the table loads read is computed in the table loop
alone.

Both loops must run over
``cutlass.range_constexpr``: the DSL unrolls them at trace time, so the table
is a list of the traced values and each index a constant.  A statement goes
in the table when it is a top-level ``name = value`` of the inner loop that
reads only the inner lane, names the outer loop does not write, the outer
loop's leading pure integer definitions of such names (copied in front of
it, ``reduction_lane_base_3`` above) and earlier table statements; when
``name`` is written once in the inner loop and read there by nothing before
it; and when ``value`` is integer arithmetic, scalar loads of integer tensors
(the policy loads among them) and pure calls, as for a moved statement.  The
outer loop must let every lane run to its end, write memory only through
kernel tensors its stores name, none of them one a table load reads, and call
nothing of unknown effect (CTA reductions are known).  The table loads read
the addresses each outer lane read, with nothing in between writing there,
and only integers move, so the results are unchanged bit for bit.
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import TypeGuard

from ..ast_read_writes import HELION_LANE_LOOP_VAR_ATTR
from ._ast_pass_utils import _FOLD_DOT_HELPER
from ._ast_pass_utils import _is_proven_relocatable_call
from ._ast_pass_utils import call_path_str
from ._ast_pass_utils import deep_clone
from ._ast_pass_utils import loop_extent
from ._ast_pass_utils import name_assignment
from ._ast_pass_utils import names_read
from ._ast_pass_utils import names_stored_or_deleted
from ._ast_pass_utils import rename_names
from ._ast_pass_utils import resolve_alias
from ._ast_pass_utils import tensor_roots_may_alias
from .address_maps import _LOOP_CALLS
from .cache_policy_loads import PACKET_LOAD_HELPERS
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
        self._trace_lists: set[str] | None = None
        # The tables of integers made so far (``doc_lanes``): a name read
        # from one (``doc = doc_lanes[lane]``) is integral.
        self.integer_tables: set[str] = set()

    def canonical(self, names: set[str]) -> set[str]:
        return {resolve_alias(name, self.rename_groups) for name in names}

    def integral_names(self) -> set[str]:
        """``_integral_names`` of the kernel body, computed on first use."""
        if self._integral_names is None:
            self._integral_names = _integral_names(self)
        return self._integral_names

    def trace_lists(self) -> set[str]:
        """``_trace_lists`` of the kernel body and the statements moved out of
        it (the tables), computed on first use."""
        if self._trace_lists is None:
            self._trace_lists = _trace_lists([*self.body, *self.moved])
        return self._trace_lists

    def forget_derived_names(self) -> None:
        """Compute ``integral_names`` and ``trace_lists`` again at their next
        use: a rewrite bound new names (a table's lists among them)."""
        self._integral_names = None
        self._trace_lists = None


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
            if (
                isinstance(value, ast.Subscript)
                and isinstance(value.value, ast.Name)
                and value.value.id in ctx.integer_tables
            ):
                # A read of a table of integers (``doc = doc_lanes[lane]``).
                value = ast.Constant(0)
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


def _loop_admits_hoisting(
    loop: ast.For, ctx: _Context, *, tables: bool = False
) -> list[frozenset[str]] | None:
    """The destination roots of ``loop``'s memory writes when every write is
    resolved (``_write_roots``), every call is known and every lane runs the
    whole body, else None.  ``operator.getitem`` is not known: it reads
    memory that the subscript checks do not see.  An append to a trace-time
    list (a load table, a lane stash: ``_trace_lists``) is known: it writes
    no memory, so a loop around one whose loads moved to tables still gives
    up its own invariants.  For an outer loop whose inner loads move to a
    table (``tables``), policy loads and CTA reductions are known too."""
    lists = ctx.trace_lists()
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
                _is_proven_relocatable_call(
                    node, allow_load=True, allow_reduction=tables
                )
                or (tables and _policy_load(node) is not None)
                or _is_private_store_staging_append(node)
                or path in _LOOP_CALLS
                or _trace_list_append(node, lists) is not None
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


def _loop_writes(loop: ast.For, ctx: _Context) -> dict[str, int]:
    """Every name ``loop`` writes (its lane variable included), by rename
    group, with the number of writes.  A register an element store or a
    staging append writes counts twice: it is never a value that moves, and
    nothing reading it moves either."""
    writes: dict[str, int] = {}
    for node in ast.walk(loop):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            name = resolve_alias(node.id, ctx.rename_groups)
            writes[name] = writes.get(name, 0) + 1
    for root in _mutation_roots(loop):
        name = resolve_alias(root, ctx.rename_groups)
        writes[name] = writes.get(name, 0) + 2
    return writes


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
    writes = _loop_writes(loop, ctx)
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
    lists = ctx.trace_lists()
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
        elif (appended := _trace_list_append(node, lists)) is not None:
            # A trace-time list's append writes the list: of a table's
            # ``doc_lanes.append(doc_by_lane)``, ``doc = doc_lanes[lane]``
            # reads what was appended, so ``cutlass.Float32(doc) * 0.5``
            # becomes invariant when the appended values do.
            assert isinstance(node, ast.Call)
            group = resolve_alias(appended, ctx.rename_groups)
            reads_of_writes.setdefault(group, []).append(
                set().union(*map(reads_of, node.args)) | {group} | tests(node)
            )
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
    if not _integer_arithmetic(value, known, ctx) or not all(
        _scalar_load_pointer(node) is not None
        or _is_proven_relocatable_call(node, allow_load=False)
        for node in calls
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


# The scalar element loads with a cache hint (``_cute_scalar_policy_load``,
# ``_cute_element_policy_load_if``, ...): inline PTX with side effects, so
# LLVM keeps every copy.  Their first argument is the address and their dtype
# argument follows the predicate of the ``_if`` forms.
_SCALAR_POLICY_LOADS: dict[str, int] = {
    helper.name: (2 if helper.name.endswith("_if") else 1)
    for helper in PACKET_LOAD_HELPERS
    if helper.family == "scalar"
}


def _policy_load(node: ast.AST) -> tuple[ast.expr, ast.expr] | None:
    """``(address, dtype)`` of a scalar policy load call, else None."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _SCALAR_POLICY_LOADS
        and len(node.args) > _SCALAR_POLICY_LOADS[node.func.id]
    ):
        return None
    return node.args[0], node.args[_SCALAR_POLICY_LOADS[node.func.id]]


def _table_load_pointer(node: ast.AST) -> ast.AST | None:
    """The address of a scalar load, a policy load included, else None."""
    if (policy := _policy_load(node)) is not None:
        return policy[0]
    return _scalar_load_pointer(node)


def _table_integral(value: ast.expr, known: set[str], ctx: _Context) -> bool:
    """``_integral``, with a policy load of an integer dtype and an
    ``operator`` comparison of integers counted as integers."""
    if (policy := _policy_load(value)) is not None:
        return call_path_str(policy[1]) in CUTLASS_INTEGER_TYPE_NAMES
    if (
        isinstance(value, ast.Call)
        and call_path_str(value.func)
        in (
            "operator.eq",
            "operator.ne",
            "operator.lt",
            "operator.le",
            "operator.gt",
            "operator.ge",
        )
        and len(value.args) == 2
        and not value.keywords
    ):
        return all(_integral(arg, known, ctx.integer_tensors) for arg in value.args)
    return _integral(value, known, ctx.integer_tensors)


def _trace_lists(body: list[ast.stmt]) -> set[str]:
    """The names bound to a list literal in ``body``: trace-time lists of
    values (a lane stash), whose ``append`` touches no memory."""
    return {
        assignment[0]
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if (assignment := name_assignment(node)) is not None
        and isinstance(assignment[1], ast.List)
    }


def _trace_list_append(node: ast.AST, lists: set[str]) -> str | None:
    """The list of ``lists`` that ``node`` appends to (``name.append(v)``),
    else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in lists
    ):
        return node.func.value.id
    return None


def _integer_arithmetic(
    value: ast.expr, known: set[str], ctx: _Context, *, tables: bool = False
) -> bool:
    """Whether ``value`` reads memory through no subscript (a kernel tensor's
    element, a register through another name, which no address check would
    see) and is an integer, as is each arithmetic operation in it, an integer
    conversion's operand included, with the names in ``known`` integral:
    the terms of every move of this pass (``_moves``, ``_tabulable``,
    ``_copyable_definition``).  ``tables``: a policy load of an integer
    dtype and a comparison of integers count (``_table_integral``)."""
    return (
        not _float_valued(value)
        and not any(_memory_subscript(node) for node in ast.walk(value))
        and (
            _table_integral(value, known, ctx)
            if tables
            else _integral(value, known, ctx.integer_tensors)
        )
        and all(
            _integral(node, known, ctx.integer_tensors)
            for node in ast.walk(value)
            if _rounds(node)
        )
    )


def _tabulable(
    value: ast.expr,
    known: set[str],
    store_roots: list[frozenset[str]],
    ctx: _Context,
) -> bool:
    """Whether ``value`` may be computed once per inner lane in front of the
    outer loop: integer arithmetic, scalar loads of integer tensors that no
    store of the outer loop may write, and pure calls (``_moves``'s terms)."""
    if not _integer_arithmetic(value, known, ctx, tables=True):
        return False
    for node in ast.walk(value):
        if not isinstance(node, ast.Call):
            continue
        pointer = _table_load_pointer(node)
        if pointer is None:
            if _policy_load(node) is None and not _is_proven_relocatable_call(
                node, allow_load=False
            ):
                return False
            continue
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


def _copyable_definition(value: ast.expr, known: set[str], ctx: _Context) -> bool:
    """Whether a leading definition of an outer loop that the tables read
    may be copied in front of the loop: integer arithmetic over pure calls,
    ``_tabulable``'s terms without its loads. It reads no memory: copied in
    front of a loop that stores to ``view``, ``base = cutlass.Int32(view[0])
    * 2`` would hand every row the value the first row read."""
    return _integer_arithmetic(value, known, ctx, tables=True) and all(
        _is_proven_relocatable_call(node, allow_load=False)
        for node in ast.walk(value)
        if isinstance(node, ast.Call)
    )


# The most values a loop's load tables keep per thread (_tabulate_inner_loads):
# inner lanes times tabulated names, summed over the loop's tables.  It counts
# values, not registers: 32 Int64 values take 64 registers.
_MAX_TABLE_VALUES = 32


def _names_read_outside(body: list[ast.stmt], loop: ast.For) -> set[str]:
    """The names ``body`` reads outside ``loop``, a statement of it."""
    reads: set[str] = set()
    pending: list[ast.AST] = list(body)
    while pending:
        node = pending.pop()
        if node is loop:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            reads.add(node.id)
        pending.extend(ast.iter_child_nodes(node))
    return reads


def _read_tables(
    body: list[ast.stmt],
    kept: Mapping[int, str],
    lists: Mapping[str, str],
    lane: str,
    chain: Collection[int],
) -> list[ast.stmt]:
    """``body``, an inner lane loop's, reading the ``kept`` names (by index
    in ``body``) from their tables ``lists`` at ``lane``, and without the
    other statements of the ``chain`` (indices), which the tables compute."""
    out: list[ast.stmt] = []
    for index, inner in enumerate(body):
        if index in kept:
            name = kept[index]
            out.append(
                ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=ast.Subscript(
                            value=ast.Name(id=lists[name], ctx=ast.Load()),
                            slice=ast.Name(id=lane, ctx=ast.Load()),
                            ctx=ast.Load(),
                        ),
                    ),
                    inner,
                )
            )
        elif index not in chain:
            out.append(inner)
    return out


def _tabulate_inner_loads(outer: ast.For, ctx: _Context) -> list[ast.stmt]:
    """``outer`` preceded by the tables of its inner lane loops' repeated
    loads (see the module docstring), or ``[outer]``."""
    if loop_extent(outer, kinds=("constexpr",), minimum=2) is None:
        return [outer]
    store_roots = _loop_admits_hoisting(outer, ctx, tables=True)
    if store_roots is None:
        return [outer]
    written = ctx.canonical(names_stored_or_deleted(outer) | _mutation_roots(outer))
    taken = {
        node.id
        for node in ast.walk(ast.Module(body=[*ctx.body, *ctx.moved], type_ignores=[]))
        if isinstance(node, ast.Name)
    }
    known = set(ctx.integral_names())
    # Names read anywhere but in ``outer`` (after it, or by an enclosing
    # loop's next iteration): a chain name among them keeps its binding in
    # the rows, from its table, or ``last_key = key`` after the loop would
    # read a name the rows no longer bind.
    read_outside = ctx.canonical(_names_read_outside([*ctx.body, *ctx.moved], outer))
    prefix: list[ast.stmt] = []
    prefix_names: set[str] = set()
    tables: list[ast.stmt] = []
    # Names the outer body reads before the statement at hand (apart from
    # each inner loop's own history below): a leading definition copied in
    # front of the loop must not be one of them (of
    # ``first = base + row; base = tid * 2``, the copy would hand ``first``
    # the first iteration's tid * 2 instead of the value base had before).
    outer_reads: set[str] = set()
    table_values = 0
    # The lane variables of the table loops made here, which later table
    # loops may bind again.
    table_lanes: set[str] = set()
    for stmt in outer.body:
        read_here = ctx.canonical(names_read(stmt))
        assignment = name_assignment(stmt)
        if assignment is not None and not tables:
            name, value = assignment
            target = resolve_alias(name, ctx.rename_groups)
            reads = ctx.canonical(names_read(value))
            if (
                target not in outer_reads
                and sum(
                    1
                    for node in ast.walk(outer)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Store)
                    and resolve_alias(node.id, ctx.rename_groups) == target
                )
                == 1
                and all(read not in written or read in prefix_names for read in reads)
                and _copyable_definition(value, known, ctx)
            ):
                prefix.append(stmt)
                prefix_names.add(target)
                known.add(name)
            outer_reads |= read_here
            continue
        outer_reads |= read_here
        if not (
            isinstance(stmt, ast.For)
            and isinstance(stmt.target, ast.Name)
            and (lane_extent := loop_extent(stmt, kinds=("constexpr",), minimum=1))
            is not None
        ):
            continue
        lane = stmt.target.id
        inner_writes: dict[str, int] = {}
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                key = resolve_alias(node.id, ctx.rename_groups)
                inner_writes[key] = inner_writes.get(key, 0) + 1
        chain: list[tuple[int, str]] = []
        chain_names: set[str] = set()
        read_before: set[str] = set()
        loads = False
        for index, inner_stmt in enumerate(stmt.body):
            assignment = name_assignment(inner_stmt)
            if assignment is not None:
                name, value = assignment
                target = resolve_alias(name, ctx.rename_groups)
                reads = ctx.canonical(names_read(value))
                if (
                    inner_writes.get(target) == 1
                    and target not in read_before
                    and all(
                        read == lane
                        or read in chain_names
                        or read in prefix_names
                        or read not in written
                        for read in reads
                    )
                    and _tabulable(value, known | {lane}, store_roots, ctx)
                ):
                    chain.append((index, name))
                    chain_names.add(target)
                    known.add(name)
                    loads |= any(
                        _table_load_pointer(node) is not None
                        for node in ast.walk(value)
                    )
            read_before |= ctx.canonical(names_read(inner_stmt))
        chain_indices = {index for index, _ in chain}
        # The names of the chain the rows still read. An address temporary
        # that only table loads read (``key`` of ``doc = load(seeds + key)``)
        # goes dead once the loads come from tables: it gets no table and
        # leaves the row loop, and takes no value of the budget below.
        read_by_rows = set().union(
            *(
                ctx.canonical(names_read(inner))
                for index, inner in enumerate(stmt.body)
                if index not in chain_indices
            ),
            *(
                ctx.canonical(names_read(other))
                for other in outer.body
                if other is not stmt
            ),
            read_outside,
        )
        kept = {
            index: name
            for index, name in chain
            if resolve_alias(name, ctx.rename_groups) in read_by_rows
        }
        # Every table entry stays live across the whole outer loop: at most
        # _MAX_TABLE_VALUES of them.
        if (
            not loads
            or not kept
            or table_values + lane_extent * len(kept) > _MAX_TABLE_VALUES
        ):
            continue
        # Every spelling of a chain name's rename group that the chain reads
        # (``key_1 = base + lane`` read as ``key``) reads the table loop's
        # value, and the table loop walks the lanes with a variable of its
        # own: in front of the row loop, it must not rebind a ``lane`` the
        # rows read before their inner loop.
        renames = {name: f"{name}_by_lane" for _, name in chain}
        by_group = {resolve_alias(name, ctx.rename_groups): name for _, name in chain}
        for index, _ in chain:
            for read in names_read(stmt.body[index]):
                group = resolve_alias(read, ctx.rename_groups)
                if group in by_group and read not in renames:
                    renames[read] = renames[by_group[group]]
        lists = {name: f"{name}_lanes" for name in kept.values()}
        table_lane = f"{lane}_table"
        if any(new in taken for new in (*renames.values(), *lists.values())) or (
            table_lane in taken and table_lane not in table_lanes
        ):
            continue
        renames[lane] = table_lane

        # Read from the tables, the values are the same for every row: as
        # the existing move asks of what it moves (the module docstring), no
        # float arithmetic of the rows may become row-invariant through them.
        # ``cutlass.Float32(doc) * 0.7 + cache[lane]`` over doc =
        # doc_lanes[lane] would compute one product for both rows, which LLVM
        # may then round apart from the add it contracted with.
        trial = deep_clone(outer)
        trial_inner = trial.body[outer.body.index(stmt)]
        assert isinstance(trial_inner, ast.For)
        trial_inner.body = _read_tables(
            trial_inner.body, kept, lists, lane, chain_indices
        )
        if not _only_integers_become_invariant(
            trial,
            [],
            set(lists.values()),
            set(),
            _loop_writes(trial, ctx),
            ctx,
        ):
            continue
        table_values += lane_extent * len(kept)
        table_body: list[ast.stmt] = [
            rename_names(deep_clone(stmt.body[index]), renames) for index, _ in chain
        ]
        table_body.extend(
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=lists[name], ctx=ast.Load()),
                        attr="append",
                        ctx=ast.Load(),
                    ),
                    args=[ast.Name(id=renames[name], ctx=ast.Load())],
                    keywords=[],
                )
            )
            for name in kept.values()
        )
        new_tables: list[ast.stmt] = [
            ast.Assign(
                targets=[ast.Name(id=lists[name], ctx=ast.Store())],
                value=ast.List(elts=[], ctx=ast.Load()),
            )
            for name in kept.values()
        ]
        new_tables.append(
            ast.For(
                target=ast.Name(id=table_lane, ctx=ast.Store()),
                iter=deep_clone(stmt.iter),
                body=table_body,
                orelse=[],
            )
        )
        stmt.body = _read_tables(stmt.body, kept, lists, lane, chain_indices)
        for table in new_tables:
            ast.copy_location(table, stmt)
            ast.fix_missing_locations(table)
        tables.extend(new_tables)
        ctx.integer_tables |= set(lists.values())
        table_lanes.add(table_lane)
        taken |= set(renames.values()) | set(lists.values())
    if not tables:
        return [outer]
    # The leading definitions the tables read, and the ones those read.
    needed = set().union(*(ctx.canonical(names_read(table)) for table in tables))
    used: list[ast.stmt] = []
    for stmt in reversed(prefix):
        assignment = name_assignment(stmt)
        assert assignment is not None
        if resolve_alias(assignment[0], ctx.rename_groups) in needed:
            used.insert(0, deep_clone(stmt))
            needed |= ctx.canonical(names_read(assignment[1]))
    ast.fix_missing_locations(outer)
    ctx.moved.extend([*used, *tables])
    # The tables bind new names (key_by_lane), integral as the names they
    # stand for: computed again, the integral names include them, so a loop
    # around the tables can tell their address arithmetic is integer.
    ctx.forget_derived_names()
    return [*used, *tables, outer]


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
            hoisted = _hoist_from_loop(stmt, ctx)
            result.extend(hoisted[:-1])
            loop = hoisted[-1]
            assert isinstance(loop, ast.For)
            result.extend(_tabulate_inner_loads(loop, ctx))
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

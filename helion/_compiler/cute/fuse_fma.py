# pyrefly: ignore-errors
"""Contract ``t = a * b; w = t + c`` into ``w = cute.math.fma(a, b, c)``.

The CuTe DSL's ``*``/``+`` lower to plain ``arith.mulf``/``arith.addf``
without fastmath contract flags, so the NVVM/ptxas pipeline may not fuse
them into FFMA (measured: a bf16 gelu ran 26% more SASS instructions than
the equivalent handwritten kernel, most of it unfused FMUL+FADD chains).
Triton contracts by default (``allow_fp_fusion``), so fusing here brings
cute numerics CLOSER to the helion-triton backend, not further away.

Scope and safety:

- Only fuses when the multiply's result is a PROVEN-fp32 SSA name (a
  forward dataflow over ``cutlass.Float32(...)``, ``cute.math.*`` calls
  other than those the last item excludes, float constants and the fp32
  fragment elements below) that is read exactly once in the whole kernel —
  integer index arithmetic never matches.
- Operands must be plain names or constants, and any name operand must be
  assigned at most once in the kernel (codegen temporaries are SSA;
  loop-carried accumulators are reassigned and therefore excluded).
- An element ``f[i]`` of an fp32 register fragment counts as such a name
  when the fragment is stored by a single statement, whose value is proven
  fp32 as the name's value would have to be, and used only through element
  subscripts (``_single_store_fragments``: an int literal or the variable of
  a ``range`` loop; any other index, such as a view ``f[None]`` or a name
  that may hold one, may be written through), and neither ``f`` nor ``i``
  is written between the multiply and its consumer (a loop binding ``i``
  writes it whatever its target's ctx): the vector-loop sinking pass
  (``cute_vloop_sink``) moves the single-assignment per-lane values a row
  loop reads into such fragments, filled once before that loop, and a
  config knob must not change which products contract.  A plain fp32 load
  is proven in neither form (``.load()``, or the bitcast of a packet lane
  that fills the fragment), so neither contracts it.
- The consumer must be a top-level ``w = t + c`` / ``c + t`` / ``t - c`` /
  ``c - t`` assignment (``w`` a name or such a fragment element) in the
  SAME statement list as the multiply.
- Separately, marked distributed-scale subtractions from the existing scale
  hoister are contracted after a definite FP32 type proof. This makes that
  already chosen contraction independent of masked-load scheduling.
- A ``cute.math`` call returns its operands' type, so it counts as fp32
  only as follows.  ``_NON_FLOAT_MATH`` (``isnan``, ``absi`` and the other
  integer and bool results) never does.  ``_OPERAND_TYPED_MATH`` (``abs``,
  ``absf``, ``max``, ``min``, ``clamp``, which also take integers) does when
  each operand, positional or keyword, is proven fp32.  Any other call
  (``exp``, ``div``, ``fma``) does unless an operand is proven not to be
  fp32 (``_proven_non_fp32``: ``cute.math.exp(cutlass.Float64(x))`` is a
  Float64, and so is ``cute.math.exp(v)`` after ``v = cutlass.Float64(x)``).
"""

from __future__ import annotations

import ast
import dataclasses

from ..ast_extension import create
from ..ast_extension import expr_from_string
from .scaled_sub_fusion import contract_distributed_scale_subtractions


def _collect_counts(
    body: list[ast.stmt],
) -> tuple[dict[str, int], dict[str, int]]:
    reads: dict[str, int] = {}
    writes: dict[str, int] = {}

    class _Counter(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load):
                reads[node.id] = reads.get(node.id, 0) + 1
            else:
                writes[node.id] = writes.get(node.id, 0) + 1

    counter = _Counter()
    for stmt in body:
        counter.visit(stmt)
    return reads, writes


def _proven_non_fp32(node: ast.expr) -> bool:
    """Whether ``node`` is proven to hold a value other than fp32: a cast to
    another ``cutlass`` type, a name assigned such a value
    (``_mark_non_fp32_reads``), arithmetic, a negation, a selection or a
    ``cute.math`` call with such an operand, whose type the result may take (counted
    conservatively, also where it promotes to fp32), or a ``_NON_FLOAT_MATH``
    call.  ``fpowi`` takes its base's type, its exponent being an integer.  A
    plain load, whose type the code does not show, is not proven."""
    if isinstance(node, ast.Name):
        return getattr(node, _NON_FP32_ATTR, False)
    if isinstance(node, ast.UnaryOp):
        return _proven_non_fp32(node.operand)
    if isinstance(node, ast.BinOp):
        return _proven_non_fp32(node.left) or _proven_non_fp32(node.right)
    if isinstance(node, ast.IfExp):
        # ``torch.where`` of two such values.
        return _proven_non_fp32(node.body) or _proven_non_fp32(node.orelse)
    if not isinstance(node, ast.Call):
        return False
    func = ast.unparse(node.func)
    if func.startswith(_NUMERIC_CASTS):
        return func != "cutlass.Float32"
    name = func.removeprefix("cute.math.")
    if name == func:
        return False
    if name in _NON_FLOAT_MATH:
        return True
    if name == "fpowi":
        operands = [
            *node.args[:1],
            *(kw.value for kw in node.keywords if kw.arg in (None, "base")),
        ]
    else:
        operands = [*node.args, *(kw.value for kw in node.keywords)]
    return any(_proven_non_fp32(arg) for arg in operands)


# Set on a read of a name that is assigned a proven non-fp32 value.
_NON_FP32_ATTR = "_helion_fuse_fma_non_fp32"


def _mark_non_fp32_reads(body: list[ast.stmt]) -> None:
    """Mark, in program order, the reads of each name assigned a value
    ``_proven_non_fp32`` holds, so that a value computed from one is proven
    too: codegen casts into a temporary (``v_0 = cutlass.Float64(load)``,
    then ``cute.math.tanh(v_0)``), and a call that is no fp32 value
    (``v_2 = cute.math.max(cutlass.Float64(v_0), ...)``) feeds the next
    (``cute.math.sqrt(v_2)``).  A name assigned such a value anywhere counts
    from then on, however else it is assigned."""
    names: set[str] = set()

    def mark(node: ast.AST) -> None:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Name)
                and isinstance(sub.ctx, ast.Load)
                and sub.id in names
            ):
                setattr(sub, _NON_FP32_ATTR, True)

    def visit(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            if isinstance(stmt, (ast.For, ast.While, ast.If)):
                mark(stmt.iter if isinstance(stmt, ast.For) else stmt.test)
                visit(stmt.body)
                visit(stmt.orelse)
                continue
            mark(stmt)
            if isinstance(stmt, ast.Assign) and _proven_non_fp32(stmt.value):
                # The names assigned, not a subscript target's base or index.
                names.update(
                    sub.id
                    for target in stmt.targets
                    for sub in ast.walk(target)
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store)
                )

    visit(body)


_FLOAT_CALL_PREFIXES = ("cutlass.Float32", "cute.math.")
_NUMERIC_CASTS = (
    "cutlass.Boolean",
    "cutlass.BFloat",
    "cutlass.Float",
    "cutlass.Int",
    "cutlass.TFloat",
    "cutlass.Uint",
)
# The operand parameters of the ``cute.math`` functions that also take
# integers, and then return an integer (``absf`` is ``abs``).  They count as
# fp32 only when every operand is proven fp32, though their operands share
# one type and one would do.
_OPERAND_TYPED_MATH = {
    "abs": ("x",),
    "absf": ("x",),
    "clamp": ("x", "lo", "hi"),
    "max": ("x", "y"),
    "min": ("x", "y"),
}
# The ``cute.math`` functions that never return a float scalar: integer and
# bool results whatever their operands, and ``sincos``'s pair.
_NON_FLOAT_MATH = frozenset(
    {"absi", "ipowi", "isfinite", "isinf", "isnan", "isnormal", "sincos"}
)


def _callee(call: ast.Call, float_names: set[str], fragments: _Fragments) -> str:
    """The function ``call`` calls, for ``_FLOAT_CALL_PREFIXES`` to match, or
    "" for a ``cute.math`` call that may return something other than fp32,
    which ``math.fma`` cannot take with fp32 operands (see the module
    docstring): ``cute.math.max(p, q)`` on Int32 values is an integer, and
    ``cute.math.exp(cutlass.Float64(x))`` a Float64."""
    func = ast.unparse(call.func)
    name = func.removeprefix("cute.math.")
    if name == func:
        return func
    params = _OPERAND_TYPED_MATH.get(name)
    if params is not None:
        # A ``**`` argument, which has no keyword name, may hold an operand.
        operands = [
            *call.args[: len(params)],
            *(kw.value for kw in call.keywords if kw.arg in (None, *params)),
        ]
        if not all(_is_float_expr(arg, float_names, fragments) for arg in operands):
            return ""
    return "" if _proven_non_fp32(call) else func


def _store_base(target: ast.expr) -> str | None:
    """The name a subscript or attribute store target writes into."""
    while isinstance(target, (ast.Subscript, ast.Attribute)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


# The loops whose variable holds one integer index.
_RANGE_CALLS = frozenset({"range", "cutlass.range", "cutlass.range_constexpr"})


@dataclasses.dataclass(frozen=True)
class _Fragments:
    """The single-store fp32 fragments (``_single_store_fragments``), the
    names that index one of their elements, and the fragments whose store
    ``_process_list`` has met with a value proven fp32."""

    names: frozenset[str]
    indices: frozenset[str]
    proven: set[str] = dataclasses.field(default_factory=set)

    def element(self, node: ast.expr) -> bool:
        """``f[i]`` of such a fragment, indexed by one element."""
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.names
            and _element_index(node.slice, self.indices)
        )

    def fp32_element(self, node: ast.expr) -> bool:
        """Such an element of a fragment stored with a proven-fp32 value."""
        return self.element(node) and node.value.id in self.proven


def _integer_indices(body: list[ast.stmt]) -> frozenset[str]:
    """Names only ever bound by ``range`` loops, which hold one integer."""
    module = ast.Module(body=body, type_ignores=[])
    ranges: set[str] = set()
    other: set[str] = set()
    targets: set[int] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            # Generated loops do not always mark their target as a store.
            targets.add(id(node.target))
            if (
                isinstance(node.iter, ast.Call)
                and ast.unparse(node.iter.func) in _RANGE_CALLS
            ):
                ranges.add(node.target.id)
            else:
                other.add(node.target.id)
    other.update(
        node.id
        for node in ast.walk(module)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Store)
        and id(node) not in targets
    )
    return frozenset(ranges - other)


def _single_store_fragments(body: list[ast.stmt], writes: dict[str, int]) -> _Fragments:
    """fp32 register fragments whose elements behave like single-assignment
    names: ``f = cute.make_rmem_tensor(n, cutlass.Float32)`` bound once, used
    only through element subscripts ``f[i]`` (an alias, a helper call, a
    view such as ``f[None]`` or an index that may be one could write it
    unseen), and stored by at most one statement, so an accumulator (an
    initial store and an update) is excluded as its scalar form is."""
    indices = _integer_indices(body)
    module = ast.Module(body=body, type_ignores=[])
    fragments = {
        node.targets[0].id
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and writes.get(node.targets[0].id) == 1
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "cute.make_rmem_tensor"
        and len(node.value.args) == 2
        and ast.unparse(node.value.args[1]) == "cutlass.Float32"
    }
    subscripted = {
        id(node.value)
        for node in ast.walk(module)
        if isinstance(node, ast.Subscript) and _element_index(node.slice, indices)
    }
    stores: dict[str, int] = {}
    escaped: set[str] = set()
    for node in ast.walk(module):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and id(node) not in subscripted
        ):
            escaped.add(node.id)
        elif isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(
            node.ctx, ast.Store
        ):
            base = _store_base(node)
            if base is not None:
                stores[base] = stores.get(base, 0) + 1
    names = frozenset(name for name in fragments - escaped if stores.get(name, 0) <= 1)
    return _Fragments(names, indices)


def _element_index(index: ast.expr, indices: frozenset[str]) -> bool:
    """An index that selects one element: an int literal or one of
    ``indices``."""
    if isinstance(index, ast.Name):
        return index.id in indices
    return isinstance(index, ast.Constant) and type(index.value) is int


def _is_float_expr(
    node: ast.expr, float_names: set[str], fragments: _Fragments
) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float)
    if isinstance(node, ast.Name):
        return node.id in float_names
    if isinstance(node, ast.BinOp):
        return _is_float_expr(node.left, float_names, fragments) and _is_float_expr(
            node.right, float_names, fragments
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_float_expr(node.operand, float_names, fragments)
    if isinstance(node, ast.Call):
        func = _callee(node, float_names, fragments)
        return func.startswith(_FLOAT_CALL_PREFIXES)
    return fragments.fp32_element(node)


def _simple_operand(
    node: ast.expr,
    float_names: set[str],
    writes: dict[str, int],
    fragments: _Fragments,
) -> bool:
    """A name/constant that provably holds the same fp32 value at any later
    point in the kernel (constants, or names written exactly once), or an
    element of a single-store fragment (``_drop_stale_muls`` drops a product
    whose fragment or index is written before its consumer)."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float)
    if isinstance(node, ast.Name):
        return node.id in float_names and writes.get(node.id, 0) <= 1
    return fragments.fp32_element(node)


def _neg(node: ast.expr) -> ast.expr:
    if isinstance(node, ast.Constant) and isinstance(node.value, float):
        return create(ast.Constant, value=-node.value, kind=None)
    return create(ast.UnaryOp, op=ast.USub(), operand=node)


def _fma_call(a: ast.expr, b: ast.expr, c: ast.expr) -> ast.expr:
    return expr_from_string("cute.math.fma({a}, {b}, {c})", a=a, b=b, c=c)


def _drop_stale_muls(
    muls: dict[str, tuple[int, ast.expr, ast.expr]],
    written: str,
    canon: dict[str, str],
) -> None:
    """Moving an operand read down to the consumer is only safe while no
    other member of the operand's RENAME group is written in between (the
    pass runs pre-rename: two SSA names may collapse into one loop-carried
    variable, so a group write would change the value read at the fma).
    A fragment element's operand names are its fragment and its index."""
    written_c = canon.get(written, written)
    for name in [*muls]:
        a, b = muls[name][1], muls[name][2]
        if any(
            canon.get(node.id, node.id) == written_c
            for operand in (a, b)
            for node in ast.walk(operand)
            if isinstance(node, ast.Name)
        ):
            del muls[name]


def _process_list(
    stmts: list[ast.stmt],
    reads: dict[str, int],
    writes: dict[str, int],
    float_names: set[str],
    canon: dict[str, str],
    fragments: _Fragments,
) -> None:
    # Forward float-name dataflow + collect fusible multiplies as we go.
    muls: dict[str, tuple[int, ast.expr, ast.expr]] = {}
    to_delete: list[int] = []

    def _drop_for_subtree_writes(node: ast.AST) -> None:
        if not muls:
            # Nothing pending to invalidate: skip walking the nested body.
            return
        for sub in ast.walk(node):
            if not muls:
                return
            if isinstance(sub, ast.For):
                # Generated loops do not always mark their target as a store.
                for name in ast.walk(sub.target):
                    if isinstance(name, ast.Name):
                        _drop_stale_muls(muls, name.id, canon)
            elif isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                _drop_stale_muls(muls, sub.id, canon)
            elif isinstance(sub, (ast.Subscript, ast.Attribute)) and isinstance(
                sub.ctx, ast.Store
            ):
                base = _store_base(sub)
                if base is not None:
                    _drop_stale_muls(muls, base, canon)

    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, (ast.For, ast.While)):
            _process_list(stmt.body, reads, writes, float_names, canon, fragments)
            _drop_for_subtree_writes(stmt)
            continue
        if isinstance(stmt, ast.If):
            _process_list(stmt.body, reads, writes, float_names, canon, fragments)
            _process_list(stmt.orelse, reads, writes, float_names, canon, fragments)
            _drop_for_subtree_writes(stmt)
            continue
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and (
                isinstance(stmt.targets[0], ast.Name)
                or fragments.element(stmt.targets[0])
            )
        ):
            _drop_for_subtree_writes(stmt)
            continue
        target = stmt.targets[0]
        # A fragment element store writes its fragment and holds no product.
        to_name = isinstance(target, ast.Name)
        name = target.id if to_name else target.value.id
        value = stmt.value
        if _is_float_expr(value, float_names, fragments):
            (float_names if to_name else fragments.proven).add(name)
        if (
            to_name
            and isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Mult)
            and writes.get(name, 0) == 1
            and reads.get(name, 0) == 1
            and _simple_operand(value.left, float_names, writes, fragments)
            and _simple_operand(value.right, float_names, writes, fragments)
        ):
            _drop_stale_muls(muls, name, canon)
            muls[name] = (idx, value.left, value.right)
            continue
        if not (
            isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub))
        ):
            _drop_stale_muls(muls, name, canon)
            continue
        left, right = value.left, value.right
        fused: ast.expr | None = None
        mul_name: str | None = None
        for t_node, c_node, t_on_left in ((left, right, True), (right, left, False)):
            if not (isinstance(t_node, ast.Name) and t_node.id in muls):
                continue
            if not _simple_operand(c_node, float_names, writes, fragments):
                continue
            a, b = muls[t_node.id][1], muls[t_node.id][2]
            if isinstance(value.op, ast.Add):
                fused = _fma_call(a, b, c_node)
            elif t_on_left:  # w = t - c  ->  fma(a, b, -c)
                fused = _fma_call(a, b, _neg(c_node))
            else:  # w = c - t  ->  fma(-a, b, c)
                fused = _fma_call(_neg(a), b, c_node)
            mul_name = t_node.id
            break
        if fused is None or mul_name is None:
            _drop_stale_muls(muls, name, canon)
            continue
        mul_idx = muls.pop(mul_name)[0]
        stmt.value = fused
        if to_name:
            float_names.add(name)
        to_delete.append(mul_idx)
        _drop_stale_muls(muls, name, canon)
    for idx in sorted(to_delete, reverse=True):
        del stmts[idx]
    if not stmts:
        stmts.append(create(ast.Pass))


def fuse_fma(
    body: list[ast.stmt], rename_groups: dict[str, str] | None = None
) -> list[ast.stmt]:
    reads, writes = _collect_counts(body)
    _mark_non_fp32_reads(body)
    float_names: set[str] = set()
    _process_list(
        body,
        reads,
        writes,
        float_names,
        rename_groups or {},
        _single_store_fragments(body, writes),
    )
    return contract_distributed_scale_subtractions(body, rename_groups or {})

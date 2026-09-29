"""Row-local fused output epilogues for the CuTe flash-attention forward path.

The flash detector historically accepted only ``out = (acc / l_i).to(dtype)``
(optionally wrapped in ``relu``) as the output store.  Attention variants that
post-process the normalized accumulator per row -- gating, projections,
normalization, residual terms -- therefore fell back to the scalar SIMT path.

This module admits a small *program* between the normalized accumulator and
the store:

* pointwise arithmetic and math on ``[tile_b, tile_m, head_dim]`` values
  ("vec" values) and on ``[tile_b, tile_m]`` / ``[tile_b, tile_m, 1]`` values
  ("row" values, one scalar per row),
* reductions over ``head_dim`` (``sum`` / ``amax`` / ``amin``),
* extra loads of host tensors that share the output's
  ``(batch*heads, seq, head_dim)`` geometry, indexed by the same
  ``[tile_b, tile_m, :]`` rows as the output store,
* symbolic scalar kernel arguments (e.g. ``eps``) and literal constants.

Every op is per-row, so the flash emitters evaluate the program inside the
thread that already owns the row in the output epilogue.  The evaluation walks
the row in the epilogue's column chunks straight out of TMEM: each reduction
accumulates across the chunk loop, and every value that depends on a reduction
is computed in a later *pass* over the same chunks.  The final pass writes the
output chunk exactly where the identity epilogue stored ``acc / l_i``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import sympy
import torch
from torch.fx import Node

from ...language import _tracing_ops
from ...language import memory_ops
from ...language import view_ops

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence


VEC = "vec"
ROW = "row"

FLASH_OUTPUT_EPILOGUE_ROW_PROGRAM = "row_program"

_REDUCTIONS: dict[object, str] = {
    torch.ops.aten.sum.dim_IntList: "sum",
    torch.ops.aten.amax.default: "amax",
    torch.ops.aten.amin.default: "amin",
}

# aten target -> (op name, arity).  ``clamp_min``/``clamp_max`` share torch's
# NaN-propagating ``maximum``/``minimum`` semantics.
_BINARY: dict[object, str] = {
    torch.ops.aten.add.Tensor: "add",
    torch.ops.aten.sub.Tensor: "sub",
    torch.ops.aten.mul.Tensor: "mul",
    torch.ops.aten.div.Tensor: "div",
    torch.ops.aten.maximum.default: "maximum",
    torch.ops.aten.minimum.default: "minimum",
    torch.ops.aten.clamp_min.default: "maximum",
    torch.ops.aten.clamp_min.Tensor: "maximum",
    torch.ops.aten.clamp_max.default: "minimum",
    torch.ops.aten.clamp_max.Tensor: "minimum",
    torch.ops.aten.gt.Tensor: "gt",
    torch.ops.aten.gt.Scalar: "gt",
    torch.ops.aten.lt.Tensor: "lt",
    torch.ops.aten.lt.Scalar: "lt",
    torch.ops.aten.ge.Tensor: "ge",
    torch.ops.aten.ge.Scalar: "ge",
    torch.ops.aten.le.Tensor: "le",
    torch.ops.aten.le.Scalar: "le",
    torch.ops.aten.eq.Tensor: "eq",
    torch.ops.aten.eq.Scalar: "eq",
    torch.ops.aten.ne.Tensor: "ne",
    torch.ops.aten.ne.Scalar: "ne",
}
_UNARY: dict[object, str] = {
    torch.ops.aten.neg.default: "neg",
    torch.ops.aten.abs.default: "abs",
    torch.ops.aten.sqrt.default: "sqrt",
    torch.ops.aten.rsqrt.default: "rsqrt",
    torch.ops.aten.exp.default: "exp",
    torch.ops.aten.exp2.default: "exp2",
    torch.ops.aten.log.default: "log",
    torch.ops.aten.log2.default: "log2",
    torch.ops.aten.relu.default: "relu",
    torch.ops.aten.tanh.default: "tanh",
    torch.ops.aten.reciprocal.default: "reciprocal",
    torch.ops.aten.square.default: "square",
}
_IDENTITY_TARGETS: frozenset[object] = frozenset(
    {
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.expand.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
        _tracing_ops._mask_to,
    }
)
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DTYPE_STR = {
    torch.float32: "cutlass.Float32",
    torch.float16: "cutlass.Float16",
    torch.bfloat16: "cutlass.BFloat16",
}


@dataclasses.dataclass(frozen=True)
class RowEpilogueOp:
    """One value of the program.

    ``op`` is one of the leaf kinds ``o_norm`` (normalized accumulator chunk),
    ``aux`` (extra tensor chunk, ``attrs=(index,)``), ``scalar`` (symbolic
    kernel argument, ``attrs=(index,)``), ``const`` (``attrs=(value,)``), or a
    computed kind: ``cast`` (``attrs=(torch.dtype,)``), ``reduce``
    (``attrs=("sum"|"amax"|"amin",)``), ``where``, ``clamp``, or a name from
    ``_BINARY`` / ``_UNARY`` / ``pow`` (``attrs=(exponent,)``).
    """

    name: str
    kind: str
    op: str
    inputs: tuple[str, ...] = ()
    attrs: tuple[object, ...] = ()
    # First pass in which the value is computable (vec) or available (row).
    avail: int = 0
    # Reductions accumulate during this pass's chunk loop; ``avail`` is one more.
    reduce_pass: int = -1


@dataclasses.dataclass(frozen=True)
class FlashRowEpilogueProgram:
    ops: tuple[RowEpilogueOp, ...]
    output: str
    aux_names: tuple[str, ...]
    aux_dtypes: tuple[torch.dtype, ...]
    scalar_exprs: tuple[sympy.Expr, ...]
    store_pass: int

    @property
    def passes(self) -> int:
        return self.store_pass + 1

    def op(self, name: str) -> RowEpilogueOp:
        return next(op for op in self.ops if op.name == name)

    def describe(self) -> str:
        parts = []
        for op in self.ops:
            parts.append(
                f"{op.name}={op.op}{op.attrs or ''}"
                f"({','.join(op.inputs)})@{op.avail}"
                + (f"/r{op.reduce_pass}" if op.reduce_pass >= 0 else "")
            )
        return f"passes={self.passes} out={self.output} " + " ".join(parts)


class _Unsupported(Exception):
    pass


def _attr_int(op: RowEpilogueOp, index: int = 0) -> int:
    value = op.attrs[index]
    assert isinstance(value, int)
    return value


def _attr_float(op: RowEpilogueOp, index: int = 0) -> float:
    value = op.attrs[index]
    assert isinstance(value, (int, float))
    return float(value)


# Diagnostics: the most recent node that rejected a program (tests/debugging).
LAST_REJECTED: list[object] = []


def _tensor_val(node: Node) -> torch.Tensor | None:
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _value_kind(node: Node, head_dim: int) -> str:
    value = _tensor_val(node)
    if value is None:
        raise _Unsupported
    if value.dtype not in _FLOAT_DTYPES and value.dtype is not torch.bool:
        raise _Unsupported
    shape = value.shape
    if value.ndim == 3:
        last = shape[2]
        if isinstance(last, int) and last == head_dim:
            return VEC
        if isinstance(last, int) and last == 1:
            return ROW
        if isinstance(last, torch.SymInt) and last.node.hint == head_dim:
            return VEC
        raise _Unsupported
    if value.ndim in (0, 2):
        return ROW
    raise _Unsupported


def _reduction_dims(node: Node) -> tuple[int, ...] | None:
    if len(node.args) < 2:
        return None
    dims = node.args[1]
    if isinstance(dims, int):
        dims = [dims]
    if not isinstance(dims, (list, tuple)):
        return None
    resolved: list[int] = []
    for dim in dims:
        if not isinstance(dim, int):
            return None
        resolved.append(dim)
    return tuple(resolved)


def _is_identity_subscript(node: Node) -> bool:
    if node.target is not view_ops.subscript or len(node.args) < 2:
        return False
    indices = node.args[1]
    if not isinstance(indices, (list, tuple)):
        return False
    return all(
        index is None
        or (
            isinstance(index, slice)
            and index.start is None
            and index.stop is None
            and index.step is None
        )
        for index in indices
    )


def _symnode_expr(node: Node) -> sympy.Expr | None:
    if node.target is not _tracing_ops._get_symnode:
        return None
    value = node.meta.get("val")
    if not isinstance(value, (torch.SymInt, torch.SymFloat)):
        return None
    expr = value.node.expr
    return expr if isinstance(expr, sympy.Expr) else None


def match_flash_row_epilogue(
    store: Node,
    *,
    head_dim: int,
    is_o_norm: Callable[[Node], bool],
    is_row_load: Callable[[Node], str | None],
) -> FlashRowEpilogueProgram | None:
    """Extract the per-row program feeding ``store`` or return None.

    ``is_o_norm(node)`` recognizes the normalized-accumulator node
    (``div(acc_phi, l_i[:, :, None])``); ``is_row_load(node)`` returns the host
    tensor name of a ``[tile_b, tile_m, :]`` load with output geometry, or None.
    Anything else outside the supported op set rejects the whole program.
    """
    if store.op != "call_function" or store.target is not memory_ops.store:
        return None
    if len(store.args) < 3 or not isinstance(store.args[2], Node):
        return None
    ops: list[RowEpilogueOp] = []
    memo: dict[Node, str] = {}
    aux_names: list[str] = []
    aux_dtypes: list[torch.dtype] = []
    scalar_exprs: list[sympy.Expr] = []
    counter = [0]

    def emit(
        kind: str, op: str, inputs: tuple[str, ...] = (), attrs: tuple[object, ...] = ()
    ) -> str:
        name = f"v{counter[0]}"
        counter[0] += 1
        ops.append(RowEpilogueOp(name, kind, op, inputs, attrs))
        return name

    def const(value: object) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _Unsupported
        return emit(ROW, "const", (), (float(value),))

    def visit(node: Node) -> str:
        if node in memo:
            return memo[node]
        try:
            return _visit(node)
        except _Unsupported as error:
            if not error.args:
                raise _Unsupported(node) from None
            raise

    def _visit(node: Node) -> str:
        if node.op != "call_function":
            raise _Unsupported
        target = node.target
        if is_o_norm(node):
            name = emit(VEC, "o_norm")
        elif (aux_name := is_row_load(node)) is not None:
            value = _tensor_val(node)
            if value is None or value.dtype not in _FLOAT_DTYPES:
                raise _Unsupported
            if aux_name in aux_names:
                index = aux_names.index(aux_name)
            else:
                index = len(aux_names)
                aux_names.append(aux_name)
                aux_dtypes.append(value.dtype)
            name = emit(VEC, "aux", (), (index,))
        elif (expr := _symnode_expr(node)) is not None:
            if expr in scalar_exprs:
                index = scalar_exprs.index(expr)
            else:
                index = len(scalar_exprs)
                scalar_exprs.append(expr)
            name = emit(ROW, "scalar", (), (index,))
        elif target is torch.ops.aten.scalar_tensor.default and node.args:
            name = const(node.args[0])
        elif target in (torch.ops.aten.full.default, _tracing_ops._new_var) or (
            target is torch.ops.aten.full and len(node.args) >= 2
        ):
            if target is _tracing_ops._new_var:
                raise _Unsupported
            name = const(node.args[1])
        elif target in _IDENTITY_TARGETS or _is_identity_subscript(node):
            if not node.args or not isinstance(node.args[0], Node):
                raise _Unsupported
            name = visit(node.args[0])
            memo[node] = name
            return name
        elif target is torch.ops.prims.convert_element_type.default:
            if len(node.args) != 2 or not isinstance(node.args[0], Node):
                raise _Unsupported
            dtype = node.args[1]
            if dtype not in _FLOAT_DTYPES:
                raise _Unsupported
            source = visit(node.args[0])
            kind = _value_kind(node, head_dim)
            name = emit(kind, "cast", (source,), (dtype,))
        elif target in _REDUCTIONS:
            dims = _reduction_dims(node)
            if dims is None or len(dims) != 1 or dims[0] not in (-1, 2):
                raise _Unsupported
            if not node.args or not isinstance(node.args[0], Node):
                raise _Unsupported
            if node.kwargs and set(node.kwargs) - {"keepdim", "dtype"}:
                raise _Unsupported
            if node.kwargs.get("dtype") not in (None, torch.float32):
                raise _Unsupported
            source = visit(node.args[0])
            if _lookup(ops, source).kind != VEC:
                raise _Unsupported
            if _value_kind(node, head_dim) != ROW:
                raise _Unsupported
            name = emit(ROW, "reduce", (source,), (_REDUCTIONS[target],))
        elif target is torch.ops.aten.where.self:
            if len(node.args) != 3 or node.kwargs:
                raise _Unsupported
            operands = _operands(node)
            kind = _value_kind(node, head_dim)
            cond = _lookup(ops, operands[0])
            if cond.kind != VEC and kind == VEC:
                raise _Unsupported
            name = emit(kind, "where", operands)
        elif target is torch.ops.aten.clamp.default:
            if not node.args or not isinstance(node.args[0], Node) or node.kwargs:
                raise _Unsupported
            source = visit(node.args[0])
            low = node.args[1] if len(node.args) > 1 else None
            high = node.args[2] if len(node.args) > 2 else None
            name = source
            kind = _value_kind(node, head_dim)
            if low is not None:
                name = emit(kind, "maximum", (name, _operand(low)))
            if high is not None:
                name = emit(kind, "minimum", (name, _operand(high)))
            if low is None and high is None:
                raise _Unsupported
        elif target is torch.ops.aten.pow.Tensor_Scalar:
            if len(node.args) != 2 or node.kwargs or not isinstance(node.args[0], Node):
                raise _Unsupported
            exponent = node.args[1]
            if not isinstance(exponent, (int, float)) or isinstance(exponent, bool):
                raise _Unsupported
            if float(exponent) not in (2.0, 0.5, -1.0, -0.5, 1.0):
                raise _Unsupported
            source = visit(node.args[0])
            name = emit(
                _value_kind(node, head_dim), "pow", (source,), (float(exponent),)
            )
        elif target in _BINARY:
            if len(node.args) != 2 or node.kwargs:
                raise _Unsupported
            operands = _operands(node)
            name = emit(_value_kind(node, head_dim), _BINARY[target], operands)
        elif target in _UNARY:
            if len(node.args) != 1 or node.kwargs or not isinstance(node.args[0], Node):
                raise _Unsupported
            name = emit(
                _value_kind(node, head_dim), _UNARY[target], (visit(node.args[0]),)
            )
        else:
            raise _Unsupported
        memo[node] = name
        return name

    def _operand(arg: object) -> str:
        if isinstance(arg, Node):
            return visit(arg)
        return const(arg)

    def _operands(node: Node) -> tuple[str, ...]:
        """Operands of a lowered pointwise node.

        ``inductor_lowering.strip_unused_inputs`` masks every repeated operand
        after its first occurrence with ``None`` (``x * x`` becomes
        ``mul(x, None)``); the masked operand is the node's unique tensor input.
        """
        nodes = {arg for arg in node.args if isinstance(arg, Node)}
        resolved = []
        for arg in node.args:
            if arg is None:
                if len(nodes) != 1:
                    raise _Unsupported
                arg = next(iter(nodes))
            resolved.append(_operand(arg))
        return tuple(resolved)

    try:
        output = visit(store.args[2])
    except _Unsupported as error:
        LAST_REJECTED[:] = [error.args[0] if error.args else None]
        return None
    except RecursionError:
        return None
    output_op = _lookup(ops, output)
    if output_op.kind != VEC:
        return None
    if not any(op.op == "o_norm" for op in ops):
        return None
    # A trailing cast to the output dtype is performed by the store itself.
    store_value = _tensor_val(store.args[2])
    output_tensor = (
        _tensor_val(store.args[0]) if isinstance(store.args[0], Node) else None
    )
    if (
        output_op.op == "cast"
        and store_value is not None
        and output_tensor is not None
        and output_op.attrs[0] == output_tensor.dtype
    ):
        output = output_op.inputs[0]
    scheduled = _schedule(ops)
    store_pass = _lookup(scheduled, output).avail
    program = FlashRowEpilogueProgram(
        tuple(scheduled),
        output,
        tuple(aux_names),
        tuple(aux_dtypes),
        tuple(scalar_exprs),
        store_pass,
    )
    if program.passes > 8:
        return None
    return program


def _lookup(ops: Sequence[RowEpilogueOp], name: str) -> RowEpilogueOp:
    return next(op for op in ops if op.name == name)


def _schedule(ops: Sequence[RowEpilogueOp]) -> list[RowEpilogueOp]:
    """Assign pass indices: reductions close a pass, their consumers open the next."""
    scheduled: list[RowEpilogueOp] = []
    avail: dict[str, int] = {}
    for op in ops:
        if op.op in ("o_norm", "aux", "scalar", "const"):
            pass_index = 0
            reduce_pass = -1
        elif op.op == "reduce":
            reduce_pass = max(avail[name] for name in op.inputs)
            pass_index = reduce_pass + 1
        else:
            pass_index = max((avail[name] for name in op.inputs), default=0)
            reduce_pass = -1
        avail[op.name] = pass_index
        scheduled.append(
            dataclasses.replace(op, avail=pass_index, reduce_pass=reduce_pass)
        )
    return scheduled


# ---------------------------------------------------------------------------
# Source emission
# ---------------------------------------------------------------------------


def _needed_vec_ops(
    program: FlashRowEpilogueProgram, pass_index: int
) -> list[RowEpilogueOp]:
    """Vec ops recomputed in this pass: those feeding this pass's reductions or
    the output (final pass), stopping at row values (available) and leaves."""
    targets = [
        op.inputs[0]
        for op in program.ops
        if op.op == "reduce" and op.reduce_pass == pass_index
    ]
    if pass_index == program.store_pass:
        targets.append(program.output)
    needed: set[str] = set()
    stack = list(targets)
    while stack:
        name = stack.pop()
        if name in needed:
            continue
        op = program.op(name)
        if op.kind != VEC:
            continue
        needed.add(name)
        stack.extend(op.inputs)
    return [op for op in program.ops if op.name in needed]


def _row_ops_available_at(
    program: FlashRowEpilogueProgram, pass_index: int
) -> list[RowEpilogueOp]:
    """Row (scalar) ops computed once the reductions of pass ``pass_index - 1`` closed."""
    return [
        op
        for op in program.ops
        if op.kind == ROW
        and op.op not in ("reduce", "scalar", "const")
        and op.avail == pass_index
    ]


_SCALAR_MATH = {
    "sqrt": "cute.math.sqrt({a})",
    "rsqrt": "cute.math.rsqrt({a})",
    "exp": "cute.math.exp({a})",
    "exp2": "cute.math.exp2({a})",
    "log": "cute.math.log({a})",
    "log2": "cute.math.log2({a})",
    "tanh": "cute.math.tanh({a})",
    "abs": "cute.math.abs({a})",
}


def render_op(op: RowEpilogueOp, atom: Callable[[str], str]) -> str:
    """Render the right-hand side of ``op`` as scalar DSL code.

    Vec values are evaluated one element at a time inside the chunk loop, so
    every op is rendered on ``cutlass`` scalars; that keeps the live state of
    the epilogue to a handful of registers, which matters inside the FA4
    correction warps (64-80 register budget).
    """
    a = [atom(name) for name in op.inputs]
    kind = op.op
    if kind == "cast":
        dtype = op.attrs[0]
        assert isinstance(dtype, torch.dtype)
        return f"{_DTYPE_STR[dtype]}({a[0]})"
    if kind == "add":
        return f"({a[0]} + {a[1]})"
    if kind == "sub":
        return f"({a[0]} - {a[1]})"
    if kind == "mul":
        return f"({a[0]} * {a[1]})"
    if kind == "div":
        return f"({a[0]} / {a[1]})"
    if kind == "neg":
        return f"(-{a[0]})"
    if kind == "square":
        return f"({a[0]} * {a[0]})"
    if kind == "reciprocal":
        return f"(cutlass.Float32(1.0) / {a[0]})"
    if kind == "pow":
        exponent = _attr_float(op)
        if exponent == 2.0:
            return f"({a[0]} * {a[0]})"
        if exponent == 0.5:
            return f"cute.math.sqrt({a[0]})"
        if exponent == -0.5:
            return f"cute.math.rsqrt({a[0]})"
        if exponent == -1.0:
            return f"(cutlass.Float32(1.0) / {a[0]})"
        return a[0]
    if kind in ("gt", "lt", "ge", "le", "eq", "ne"):
        symbol = {"gt": ">", "lt": "<", "ge": ">=", "le": "<=", "eq": "==", "ne": "!="}[
            kind
        ]
        return f"({a[0]} {symbol} {a[1]})"
    if kind == "where":
        return f"cutlass.select_({a[0]}, {a[1]}, {a[2]})"
    if kind == "relu":
        x = a[0]
        # torch.relu: NaN stays NaN, every non-positive value (incl. -0.0) -> +0.0.
        return (
            f"cutlass.select_({x} != {x}, {x}, "
            f"cutlass.select_({x} > 0.0, {x}, cutlass.Float32(0.0)))"
        )
    if kind in ("maximum", "minimum"):
        fn = "cute.math.max" if kind == "maximum" else "cute.math.min"
        return f"{fn}({a[0]}, {a[1]}, propagate_nan=True)"
    if kind == "sigmoid":
        return (
            f"cute.math.rcp(1.0 + cute.math.exp2(({a[0]}) * -1.4426950408889634, "
            "fastmath=True), approx=True, ftz=True)"
        )
    if kind in _SCALAR_MATH:
        return _SCALAR_MATH[kind].format(a=a[0])
    raise AssertionError(f"unrenderable row epilogue op {kind!r}")


ROW_EPILOGUE_AUX_PREFETCH = 3
"""Aux chunks loaded ahead of the chunk being computed.

Aux rows exceed the L1 left beside FA4's shared memory, so every pass pays L2
latency per chunk unless the loads run ahead.  Each unit of distance costs one
extra aux fragment buffer.  ``HELION_CUTE_FLASH_ROW_EPILOGUE_PREFETCH``
overrides the default for experiments.
"""


def row_epilogue_aux_prefetch() -> int:
    import os

    value = os.environ.get("HELION_CUTE_FLASH_ROW_EPILOGUE_PREFETCH", "").strip()
    return int(value) if value else ROW_EPILOGUE_AUX_PREFETCH


def row_epilogue_hoist_enabled() -> bool:
    """``HELION_CUTE_FLASH_ROW_EPILOGUE_HOIST=0`` keeps every pass in the epilogue."""
    import os

    value = os.environ.get("HELION_CUTE_FLASH_ROW_EPILOGUE_HOIST", "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def hoistable_passes(program: FlashRowEpilogueProgram) -> int:
    """Number of leading passes that never read the accumulator.

    Those passes (e.g. a norm of an aux row) can run before the main loop; the
    store pass always stays in the epilogue.
    """
    count = 0
    for pass_index in range(program.store_pass):
        needed = _needed_vec_ops(program, pass_index)
        if any(op.op == "o_norm" for op in needed):
            break
        count += 1
    return count


def emit_row_epilogue(
    program: FlashRowEpilogueProgram,
    *,
    chunks: int,
    chunk_width: int,
    elem_var: str,
    o_alloc: str,
    o_load: Sequence[str],
    o_elem: str,
    aux_allocs: Sequence[str],
    aux_loads: Sequence[Sequence[str]],
    aux_elems: Sequence[str],
    out_alloc: str,
    store_elem: str,
    store: Sequence[str],
    scalar_names: Sequence[str],
    indent: str,
    prefix: str = "_ep",
    aux_prefetch: int | None = None,
    split: bool = False,
) -> tuple[str, str]:
    """Emit the program's passes as kernel-body source.

    Returns ``(prologue, epilogue)``.  With ``split`` the leading passes that
    do not read the accumulator (see ``hoistable_passes``) plus their row
    values form the prologue, meant to run before the main loop; otherwise the
    prologue is empty.  Templates: ``{o}`` / ``{a}`` / ``{out}`` name the
    fragment buffer of the chunk being processed, ``{i}`` is the literal chunk
    index, ``{j}`` the element index and ``{value}`` (``store_elem`` only) the
    fp32 output scalar.  ``o_load[0]`` issues the TMEM load of a chunk and the
    remaining ``o_load`` statements finish it (scaling); loads run ahead of the
    math by one chunk (O) and ``aux_prefetch`` chunks (aux).
    """
    assert len(aux_loads) == len(program.aux_names) == len(aux_elems) == len(aux_allocs)
    assert len(scalar_names) == len(program.scalar_exprs)
    # Chunk indices are literal ints: the DSL only unrolls literal loops and a
    # dynamic index would turn fragment accesses into local-memory traffic.
    assert isinstance(chunks, int) and isinstance(chunk_width, int) and chunks >= 1
    if aux_prefetch is None:
        aux_prefetch = row_epilogue_aux_prefetch()
    aux_prefetch = max(0, min(aux_prefetch, chunks - 1))
    aux_buffers = aux_prefetch + 1
    o_buffers = 2 if chunks > 1 else 1
    hoisted = hoistable_passes(program) if split and row_epilogue_hoist_enabled() else 0

    def var(name: str) -> str:
        return f"{prefix}_{name}"

    def atom(name: str) -> str:
        op = program.op(name)
        if op.op == "scalar":
            return scalar_names[_attr_int(op)]
        if op.op == "const":
            return f"cutlass.Float32({_attr_float(op)!r})"
        return var(name)

    def fill(
        text: str,
        *,
        i: int | None = None,
        j: str = "",
        o_slot: int = 0,
        a_slot: int = 0,
        aux: int = 0,
    ) -> str:
        return (
            text.replace("{i}", "" if i is None else str(i))
            .replace("{j}", j)
            .replace("{o}", f"{prefix}_o{o_slot}")
            .replace("{a}", f"{prefix}_a{aux}_{a_slot}")
            .replace("{out}", f"{prefix}_out{o_slot}")
        )

    reduce_inits = {
        "sum": "cutlass.Float32(0.0)",
        "amax": "cutlass.Float32(-cutlass.Float32.inf)",
        "amin": "cutlass.Float32(cutlass.Float32.inf)",
    }

    def emit_passes(pass_range: range, lines: list[str]) -> None:
        def add(text: str, extra: str = "") -> None:
            for line in text.split("\n"):
                lines.append(f"{indent}{extra}{line}" if line.strip() else "")

        for pass_index in pass_range:
            reductions = [
                op
                for op in program.ops
                if op.op == "reduce" and op.reduce_pass == pass_index
            ]
            needed = _needed_vec_ops(program, pass_index)
            needed_names = {op.name for op in needed}
            uses_o = any(op.op == "o_norm" for op in needed)
            aux_needed = sorted({_attr_int(op) for op in needed if op.op == "aux"})
            is_store = pass_index == program.store_pass
            accumulators = 2 if chunks > 1 else 1
            for op in reductions:
                init = reduce_inits[str(op.attrs[0])]
                for slot in range(accumulators):
                    add(f"{var(op.name)}_{slot} = {init}")
            # Prime the load pipelines.
            if uses_o:
                add(fill(o_load[0], i=0, o_slot=0))
            for index in aux_needed:
                for ahead in range(min(aux_buffers, chunks)):
                    for line in aux_loads[index]:
                        add(fill(line, i=ahead, a_slot=ahead % aux_buffers, aux=index))
            for i in range(chunks):
                o_slot = i % o_buffers
                a_slot = i % aux_buffers
                if uses_o and i + 1 < chunks:
                    add(fill(o_load[0], i=i + 1, o_slot=(i + 1) % o_buffers))
                if uses_o:
                    for line in o_load[1:]:
                        add(fill(line, i=i, o_slot=o_slot))
                add(f"for {elem_var} in cutlass.range_constexpr({chunk_width}):")
                inner = "    "
                for op in needed:
                    if op.op == "o_norm":
                        add(
                            f"{var(op.name)} = {fill(o_elem, i=i, j=elem_var, o_slot=o_slot)}",
                            inner,
                        )
                    elif op.op == "aux":
                        index = _attr_int(op)
                        add(
                            f"{var(op.name)} = "
                            f"{fill(aux_elems[index], i=i, j=elem_var, a_slot=a_slot, aux=index)}",
                            inner,
                        )
                    else:
                        add(f"{var(op.name)} = {render_op(op, atom)}", inner)
                for op in reductions:
                    source = var(op.inputs[0])
                    assert op.inputs[0] in needed_names
                    acc = f"{var(op.name)}_{i % accumulators}"
                    reduce_kind = str(op.attrs[0])
                    if reduce_kind == "sum":
                        add(f"{acc} = {acc} + {source}", inner)
                    else:
                        fn = (
                            "cute.math.max"
                            if reduce_kind == "amax"
                            else "cute.math.min"
                        )
                        add(f"{acc} = {fn}({acc}, {source}, propagate_nan=True)", inner)
                if is_store:
                    add(
                        fill(store_elem, i=i, j=elem_var, o_slot=o_slot).replace(
                            "{value}", var(program.output)
                        ),
                        inner,
                    )
                if is_store:
                    for line in store:
                        add(fill(line, i=i, o_slot=o_slot))
                # Refill this chunk's aux slot with the chunk ``aux_buffers`` ahead
                # only after its math consumed it.
                for index in aux_needed:
                    if i + aux_buffers < chunks:
                        for line in aux_loads[index]:
                            add(
                                fill(
                                    line,
                                    i=i + aux_buffers,
                                    a_slot=(i + aux_buffers) % aux_buffers,
                                    aux=index,
                                )
                            )
            for op in reductions:
                reduce_kind = str(op.attrs[0])
                parts = [f"{var(op.name)}_{slot}" for slot in range(accumulators)]
                if len(parts) == 1:
                    add(f"{var(op.name)} = {parts[0]}")
                elif reduce_kind == "sum":
                    add(f"{var(op.name)} = {parts[0]} + {parts[1]}")
                else:
                    fn = "cute.math.max" if reduce_kind == "amax" else "cute.math.min"
                    add(
                        f"{var(op.name)} = {fn}({parts[0]}, {parts[1]}, propagate_nan=True)"
                    )
            for op in _row_ops_available_at(program, pass_index + 1):
                add(f"{var(op.name)} = {render_op(op, atom)}")

    prologue: list[str] = []
    epilogue: list[str] = []

    def alloc_lines(lines: list[str], *, include_o: bool) -> None:
        for index, alloc in enumerate(aux_allocs):
            if any(_attr_int(op) == index for op in program.ops if op.op == "aux"):
                for slot in range(aux_buffers):
                    lines.append(f"{indent}{fill(alloc, a_slot=slot, aux=index)}")
        if include_o:
            for slot in range(o_buffers):
                lines.extend(
                    (
                        f"{indent}{fill(o_alloc, o_slot=slot)}",
                        f"{indent}{fill(out_alloc, o_slot=slot)}",
                    )
                )

    uses_o_anywhere = any(op.op == "o_norm" for op in program.ops)
    first_target = prologue if hoisted else epilogue
    alloc_lines(first_target, include_o=False)
    # Row values that depend on nothing but scalars/consts.
    for op in _row_ops_available_at(program, 0):
        first_target.append(f"{indent}{var(op.name)} = {render_op(op, atom)}")
    if hoisted:
        emit_passes(range(hoisted), prologue)
    alloc_lines(epilogue, include_o=uses_o_anywhere)
    emit_passes(range(hoisted, program.passes), epilogue)
    return "\n".join(prologue), "\n".join(epilogue)

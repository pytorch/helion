"""AST peephole that fuses adjacent per-lane fp8 decodes into one
``cvt.rn.f16x2.e4m3x2`` (decode TWO e4m3 bytes per instruction).

The SIMT matmul fallback, after the packed-load change, emits a constexpr
V-loop that decodes one fp8 byte per operand per lane::

    for vec_lane in cutlass.range_constexpr(8):
        load = cutlass.Uint8(PK0 >> 8 * vec_lane & 255) if M else cutlass.Uint8(0)
        load_1 = cutlass.Uint8(PK1 >> 8 * vec_lane & 255) if M1 else cutlass.Uint8(0)
        dot_product_0 = cutlass.Float32(
            _cute_fp8e4m3fn_to_float32(load) * _cute_fp8e4m3fn_to_float32(load_1)
        )
        dot_acc = dot_acc + dot_product_0

Each ``_cute_fp8e4m3fn_to_float32`` is a separate ``cvt.f16x2`` that throws
away one of the two decoded halves.  Decoding the bytes in PAIRS halves the
decode instruction count.  This pass rewrites the loop to step by two and
decode lanes ``2j`` and ``2j+1`` of each operand with a single
``fp8e4m3fn_x2_to_float32`` call::

    for vp in cutlass.range_constexpr(4):
        _pair_x = fp8e4m3fn_x2_to_float32(cutlass.Uint16(PK0 >> 16 * vp & 65535))
        _pair_y = fp8e4m3fn_x2_to_float32(cutlass.Uint16(PK1 >> 16 * vp & 65535))
        dot_acc = dot_acc + (_pair_x[0] * _pair_y[0] + _pair_x[1] * _pair_y[1])

The pass is conservative: it only triggers for the exact
``packed-shift -> fp8 decode -> product -> accumulate`` shape with an even
constexpr trip count and a self-accumulating ``dot_acc`` (the matmul
fallback's running sum, whose terms commute so reordering by pairs is
exact).  Anything unexpected leaves the loop untouched.

Drops the per-lane OOB mask (``if M else 0``): the packed load already only
fires when the V-aligned chunk is fully in bounds (the dispatcher requires
``numel % V == 0``), so every lane in the chunk is valid.
"""

from __future__ import annotations

import ast
import re

from ..ast_extension import statement_from_string

_DECODE_FN = "_cute_fp8e4m3fn_to_float32"
_DECODE2_FN = "_cute_fp8e4m3fn_x2_to_float32"

# ``<PACKED> >> 8 * <lane> & 255`` (parenthesisation as Python unparses it:
# ``PACKED >> 8 * lane & 255``).  Capture the packed-register name and lane var.
_SHIFT_RE = re.compile(
    r"^(?P<pk>[A-Za-z_][A-Za-z0-9_]*)\s*>>\s*8\s*\*\s*(?P<lane>[A-Za-z_][A-Za-z0-9_]*)\s*&\s*255$"
)


def _is_constexpr_v_loop(node: ast.stmt) -> tuple[ast.For, str, int] | None:
    if not isinstance(node, ast.For):
        return None
    it = node.iter
    if not (
        isinstance(it, ast.Call)
        and isinstance(it.func, ast.Attribute)
        and it.func.attr == "range_constexpr"
        and isinstance(it.func.value, ast.Name)
        and it.func.value.id == "cutlass"
        and len(it.args) == 1
        and isinstance(it.args[0], ast.Constant)
        and isinstance(it.args[0].value, int)
        and isinstance(node.target, ast.Name)
    ):
        return None
    return node, node.target.id, it.args[0].value


def _byte_extract_packed(rhs: ast.expr, lane_var: str) -> str | None:
    """If ``rhs`` is ``cutlass.Uint8(<pk> >> 8 * <lane_var> & 255)`` (optionally
    wrapped in an ``... if mask else cutlass.Uint8(0)``), return the packed
    register name ``<pk>``.  Otherwise None.
    """
    inner = rhs
    if isinstance(inner, ast.IfExp):
        inner = inner.body
    if not (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "Uint8"
        and len(inner.args) == 1
    ):
        return None
    m = _SHIFT_RE.match(ast.unparse(inner.args[0]))
    if m is None or m.group("lane") != lane_var:
        return None
    return m.group("pk")


def _match_decode_product(rhs: ast.expr) -> tuple[str, str] | None:
    """If ``rhs`` is ``cutlass.Float32(_dec(A) * _dec(B))`` return ``(A, B)``."""
    inner = rhs
    if (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Attribute)
        and inner.func.attr == "Float32"
        and len(inner.args) == 1
    ):
        inner = inner.args[0]
    if not (isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Mult)):
        return None

    def decoded_name(node: ast.expr) -> str | None:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == _DECODE_FN
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
        ):
            return node.args[0].id
        return None

    a = decoded_name(inner.left)
    b = decoded_name(inner.right)
    if a is None or b is None:
        return None
    return a, b


def _try_fuse_one_vloop(vloop: ast.For, lane_var: str, v: int) -> ast.For | None:
    if v < 2 or v % 2 != 0:
        return None
    body = vloop.body
    # Expect exactly: load=..., load_1=..., dot_product=..., dot_acc = dot_acc + dot_product
    # (plus optional ``indices``/``mask`` scalar setup we can drop).
    assigns: dict[str, ast.expr] = {}
    load_names: list[str] = []
    product_name: str | None = None
    product_operands: tuple[str, str] | None = None
    acc_stmt: ast.Assign | None = None
    for stmt in body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            return None
        if not isinstance(stmt.targets[0], ast.Name):
            return None
        name = stmt.targets[0].id
        rhs = stmt.value
        pk = _byte_extract_packed(rhs, lane_var)
        if pk is not None:
            assigns[name] = rhs
            load_names.append(name)
            continue
        prod = _match_decode_product(rhs)
        if prod is not None:
            if product_name is not None:
                return None
            product_name = name
            product_operands = prod
            continue
        # ``dot_acc = dot_acc + <product>`` — self-accumulating running sum.
        if (
            isinstance(rhs, ast.BinOp)
            and isinstance(rhs.op, ast.Add)
            and isinstance(rhs.left, ast.Name)
            and rhs.left.id == name
            and isinstance(rhs.right, ast.Name)
            and rhs.right.id == product_name
        ):
            acc_stmt = stmt
            continue
        # Allow inert scalar setup (indices/mask) that we will drop.
        if name in ("indices_1", "indices_2", "indices_0") or name.startswith("mask"):
            continue
        return None

    if product_operands is None or acc_stmt is None or len(load_names) != 2:
        return None
    acc_name = acc_stmt.targets[0].id  # type: ignore[union-attr]
    # Map the two decoded operands back to their packed registers.
    pk_x = _byte_extract_packed(assigns[product_operands[0]], lane_var)
    pk_y = _byte_extract_packed(assigns[product_operands[1]], lane_var)
    if pk_x is None or pk_y is None:
        return None

    # Build the step-2 paired-decode loop.
    half = v // 2
    px = f"_pair_x_{lane_var}"
    py = f"_pair_y_{lane_var}"
    new_body = [
        statement_from_string(
            f"{px} = {_DECODE2_FN}(cutlass.Uint16({pk_x} >> 16 * {lane_var} & 65535))"
        ),
        statement_from_string(
            f"{py} = {_DECODE2_FN}(cutlass.Uint16({pk_y} >> 16 * {lane_var} & 65535))"
        ),
        statement_from_string(
            f"{acc_name} = {acc_name} + ({px}[0] * {py}[0] + {px}[1] * {py}[1])"
        ),
    ]
    new_loop = ast.parse(
        f"for {lane_var} in cutlass.range_constexpr({half}):\n    pass"
    ).body[0]
    assert isinstance(new_loop, ast.For)
    new_loop.body = new_body
    ast.copy_location(new_loop, vloop)
    ast.fix_missing_locations(new_loop)
    return new_loop


def _walk(body: list[ast.stmt]) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    for stmt in body:
        match = _is_constexpr_v_loop(stmt)
        if match is not None:
            loop, lane_var, v = match
            fused = _try_fuse_one_vloop(loop, lane_var, v)
            if fused is not None:
                out.append(fused)
                continue
        for attr in ("body", "orelse"):
            inner = getattr(stmt, attr, None)
            if (
                isinstance(inner, list)
                and inner
                and all(isinstance(s, ast.stmt) for s in inner)
            ):
                setattr(stmt, attr, _walk(inner))
        out.append(stmt)
    return out


def fuse_fp8_pair_decode(body: list[ast.stmt]) -> list[ast.stmt]:
    """Rewrite per-lane fp8 scalar decodes into paired ``cvt.f16x2.e4m3x2``.

    Requires ``_cute_fp8e4m3fn_x2_to_float32`` to be imported in the kernel
    module (the device-function preamble adds it when this pass may fire).
    """
    return _walk(body)

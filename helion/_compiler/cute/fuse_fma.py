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
  forward dataflow over ``cutlass.Float32(...)``, ``cute.math.*`` calls and
  float constants) that is read exactly once in the whole kernel — integer
  index arithmetic never matches.
- Operands must be plain names or constants, and any name operand must be
  assigned at most once in the kernel (codegen temporaries are SSA;
  loop-carried accumulators are reassigned and therefore excluded).
- The consumer must be a top-level ``w = t + c`` / ``c + t`` / ``t - c`` /
  ``c - t`` assignment in the SAME statement list as the multiply.
"""

from __future__ import annotations

import ast

from ..ast_extension import create
from ..ast_extension import expr_from_string


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


_FLOAT_CALL_PREFIXES = ("cutlass.Float32", "cute.math.")


def _is_float_expr(node: ast.expr, float_names: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float)
    if isinstance(node, ast.Name):
        return node.id in float_names
    if isinstance(node, ast.BinOp):
        return _is_float_expr(node.left, float_names) and _is_float_expr(
            node.right, float_names
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _is_float_expr(node.operand, float_names)
    if isinstance(node, ast.Call):
        func = ast.unparse(node.func)
        return func.startswith(_FLOAT_CALL_PREFIXES)
    return False


def _simple_operand(
    node: ast.expr, float_names: set[str], writes: dict[str, int]
) -> bool:
    """A name/constant that provably holds the same fp32 value at any later
    point in the kernel (constants, or names written exactly once)."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, float)
    if isinstance(node, ast.Name):
        return node.id in float_names and writes.get(node.id, 0) <= 1
    return False


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
    variable, so a group write would change the value read at the fma)."""
    written_c = canon.get(written, written)
    for name in [*muls]:
        a, b = muls[name][1], muls[name][2]
        for operand in (a, b):
            if isinstance(operand, ast.Name) and (
                canon.get(operand.id, operand.id) == written_c
            ):
                del muls[name]
                break


def _process_list(
    stmts: list[ast.stmt],
    reads: dict[str, int],
    writes: dict[str, int],
    float_names: set[str],
    canon: dict[str, str],
) -> None:
    # Forward float-name dataflow + collect fusible multiplies as we go.
    muls: dict[str, tuple[int, ast.expr, ast.expr]] = {}
    to_delete: list[int] = []

    def _drop_for_subtree_writes(node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                _drop_stale_muls(muls, sub.id, canon)

    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, (ast.For, ast.While)):
            _process_list(stmt.body, reads, writes, float_names, canon)
            _drop_for_subtree_writes(stmt)
            continue
        if isinstance(stmt, ast.If):
            _process_list(stmt.body, reads, writes, float_names, canon)
            _process_list(stmt.orelse, reads, writes, float_names, canon)
            _drop_for_subtree_writes(stmt)
            continue
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            _drop_for_subtree_writes(stmt)
            continue
        name = stmt.targets[0].id
        value = stmt.value
        if _is_float_expr(value, float_names):
            float_names.add(name)
        if (
            isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Mult)
            and writes.get(name, 0) == 1
            and reads.get(name, 0) == 1
            and _simple_operand(value.left, float_names, writes)
            and _simple_operand(value.right, float_names, writes)
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
            if not _simple_operand(c_node, float_names, writes):
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
    float_names: set[str] = set()
    _process_list(body, reads, writes, float_names, rename_groups or {})
    return body

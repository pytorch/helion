"""AST peephole: the exact f16x2 dot product of two packed E2M1 groups.

A scalar fp4 x fp4 body decodes one byte of each operand at a time
(``_cute_float4_e2m1fn_x2_to_float32``), multiplies the two decoded values
pairwise in f32 and adds the products to a running f32 sum that starts at
zero (``examples/nvfp4_gemv.py``, fp4-in)::

    contrib = cutlass.Float32(0.0)
    load = cutlass.Uint8(W[0])
    fp4_pair_0 = _cute_float4_e2m1fn_x2_to_float32(load)
    weight_lo = fp4_pair_0[0]
    weight_hi = fp4_pair_0[1]
    load_1 = cutlass.Uint8(X[0])
    fp4_pair_1 = _cute_float4_e2m1fn_x2_to_float32(load_1)
    x_lo = fp4_pair_1[0]
    x_hi = fp4_pair_1[1]
    v_0 = weight_lo * x_lo
    v_1 = contrib + v_0
    v_2 = weight_hi * x_hi
    v_3 = v_1 + v_2
    ... bytes 1 to 7 of W and X ...

Per byte that is two ``e2m1x2`` conversions, four f16 -> f32 conversions,
two multiplies and two adds for two products.  Every product of two E2M1
values is a multiple of 0.25 of magnitude at most 36 and the sixteen-term
sum at most 576, so the f32 chain rounds nothing; the same sum kept as two
interleaved f16 sums of eight products each (magnitude at most 288, and f16
resolves quarters up to 512) followed by two exact f32 conversions and one
add is identical bit for bit.  When one block's statements decode all eight
bytes of two 8-byte vector loads ``W`` and ``X``, pair the halves of equal
byte index, and add the sixteen products to a zero-initialised sum whose
intermediate names are read nowhere else, those statements are replaced at
the position of the final add by::

    v_31 = cutlass.Float32(_cute_float4_e2m1fn_x16_dot_float32(W, X))

(``quantized_helpers.float4_e2m1fn_x16_dot_float32``: sixteen ``e2m1x2``
conversions, eight ``fma.rn.f16x2``, two conversions and one add).  The
vector loads and the zero initialisation stay where they are; anything that
does not match exactly is left alone.
"""

from __future__ import annotations

import ast
from collections import Counter
import dataclasses

from ..ast_extension import statement_from_string

DECODE_FN = "_cute_float4_e2m1fn_x2_to_float32"
DOT_FN = "_cute_float4_e2m1fn_x16_dot_float32"
GROUP_BYTES = 8


def _assigned_name(stmt: ast.stmt) -> str | None:
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    return None


def _is_zero_f32(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Call)
        and ast.unparse(value.func) == "cutlass.Float32"
        and len(value.args) == 1
        and not value.keywords
        and isinstance(value.args[0], ast.Constant)
        and type(value.args[0].value) in (int, float)
        and value.args[0].value == 0
    )


def _byte_extract(value: ast.expr) -> tuple[str, int] | None:
    """``cutlass.Uint8(VEC[b])`` -> ``(VEC, b)``."""
    if not (
        isinstance(value, ast.Call)
        and ast.unparse(value.func) == "cutlass.Uint8"
        and len(value.args) == 1
        and not value.keywords
    ):
        return None
    sub = value.args[0]
    if (
        isinstance(sub, ast.Subscript)
        and isinstance(sub.value, ast.Name)
        and isinstance(sub.slice, ast.Constant)
        and type(sub.slice.value) is int
    ):
        return sub.value.id, sub.slice.value
    return None


def _decode_call(value: ast.expr) -> str | None:
    """``_cute_float4_e2m1fn_x2_to_float32(load)`` -> ``load``."""
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == DECODE_FN
        and len(value.args) == 1
        and not value.keywords
        and isinstance(value.args[0], ast.Name)
    ):
        return value.args[0].id
    return None


def _half_extract(value: ast.expr) -> tuple[str, int] | None:
    """``pair[h]`` -> ``(pair, h)``."""
    if (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and isinstance(value.slice, ast.Constant)
        and value.slice.value in (0, 1)
    ):
        return value.value.id, value.slice.value
    return None


def _binop(value: ast.expr, op: type[ast.operator]) -> tuple[str, str] | None:
    if (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, op)
        and isinstance(value.left, ast.Name)
        and isinstance(value.right, ast.Name)
    ):
        return value.left.id, value.right.id
    return None


def _is_byte_vector_load(value: ast.expr) -> bool:
    """``cute.arch.load(ptr, ir.VectorType.get([8], cutlass.Uint8.mlir_type))``."""
    return (
        isinstance(value, ast.Call)
        and ast.unparse(value.func) == "cute.arch.load"
        and len(value.args) == 2
        and ast.unparse(value.args[1])
        == f"ir.VectorType.get([{GROUP_BYTES}], cutlass.Uint8.mlir_type)"
    )


def _writes(stmt: ast.stmt) -> set[str]:
    """Every name a statement (with its nested statements) assigns."""
    return {
        node.id
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)
    }


@dataclasses.dataclass
class _Facts:
    """Single-assignment facts of one block, by assigned name."""

    index: dict[str, int]
    byte: dict[str, tuple[str, int]]
    decode: dict[str, str]
    half: dict[str, tuple[str, int]]
    mul: dict[str, tuple[str, str]]
    add: dict[str, tuple[str, str]]
    vector: set[str]
    # Statement index -> the name assigned to a zero f32 literal there.
    zero_at: dict[int, str]
    # Statement index -> every name it writes (nested statements included).
    writes_at: list[set[str]]


def _collect(block: list[ast.stmt], defs: Counter[str]) -> _Facts:
    facts = _Facts({}, {}, {}, {}, {}, {}, set(), {}, [_writes(s) for s in block])
    for position, stmt in enumerate(block):
        name = _assigned_name(stmt)
        if name is None:
            continue
        assert isinstance(stmt, ast.Assign)
        value = stmt.value
        if _is_zero_f32(value):
            facts.zero_at[position] = name
        if defs[name] != 1:
            continue
        facts.index[name] = position
        if _is_zero_f32(value):
            continue
        if (byte := _byte_extract(value)) is not None:
            facts.byte[name] = byte
        elif (load := _decode_call(value)) is not None:
            facts.decode[name] = load
        elif (half := _half_extract(value)) is not None:
            facts.half[name] = half
        elif (mul := _binop(value, ast.Mult)) is not None:
            facts.mul[name] = mul
        elif (add := _binop(value, ast.Add)) is not None:
            facts.add[name] = add
        elif _is_byte_vector_load(value):
            facts.vector.add(name)
    return facts


def _decoded_halves(facts: _Facts) -> dict[str, tuple[str, int, int, str, str]]:
    """half name -> (vector, byte, half, load name, pair name)."""
    result: dict[str, tuple[str, int, int, str, str]] = {}
    for half_name, (pair, half) in facts.half.items():
        load = facts.decode.get(pair)
        if load is None:
            continue
        byte = facts.byte.get(load)
        if byte is None or byte[0] not in facts.vector:
            continue
        result[half_name] = (byte[0], byte[1], half, load, pair)
    return result


def _match_chain(
    final: str, facts: _Facts, halves: dict[str, tuple[str, int, int, str, str]]
) -> tuple[str, str, list[str], str, int] | None:
    """Walk the add chain ending at ``final`` back to its start.

    Returns ``(W, X, consumed names, base name, index of the first add)``
    when the chain adds exactly the sixteen half products of the eight bytes
    of two vectors, halves paired by equal byte index and position.  Whether
    the base reads as zero is the caller's check."""
    consumed: list[str] = [final]
    products: list[str] = []
    name = final
    first_add = facts.index[final]
    while True:
        operands = facts.add.get(name)
        if operands is None:
            break
        first_add = facts.index[name]
        lhs, rhs = operands
        if lhs in facts.mul and rhs not in facts.mul:
            product, name = lhs, rhs
        elif rhs in facts.mul and lhs not in facts.mul:
            product, name = rhs, lhs
        else:
            return None
        products.append(product)
        consumed.append(product)
        if name != final and name in facts.add:
            consumed.append(name)
        if len(products) > 2 * GROUP_BYTES:
            return None
    base = name
    if len(products) != 2 * GROUP_BYTES:
        return None
    pairs: dict[tuple[int, int], tuple[str, str]] = {}
    vectors: set[str] = set()
    for product in products:
        a, b = facts.mul[product]
        if a not in halves or b not in halves:
            return None
        va, ba, ha, _la, _pa = halves[a]
        vb, bb, hb, _lb, _pb = halves[b]
        if (ba, ha) != (bb, hb) or va == vb:
            return None
        key = (ba, ha)
        if key in pairs:
            return None
        pairs[key] = (a, b) if va < vb else (b, a)
        vectors.update((va, vb))
        for half_name in (a, b):
            _v, _b, _h, load, pair = halves[half_name]
            consumed.extend((half_name, load, pair))
    if len(vectors) != 2 or len(pairs) != 2 * GROUP_BYTES:
        return None
    (w, x) = sorted(vectors)
    # Every (byte, half) must come from W on one side and X on the other.
    for a, b in pairs.values():
        if halves[a][0] != w or halves[b][0] != x:
            return None
    return w, x, consumed, base, first_add


def _zero_base(facts: _Facts, base: str, first_add: int, visible: set[str]) -> bool:
    """Whether ``base`` reads as a zero f32 literal at statement ``first_add``:
    the nearest earlier assignment of it in this block is a zero literal, or
    an enclosing block assigned it zero and nothing here writes it before."""
    for position in range(first_add - 1, -1, -1):
        if base in facts.writes_at[position]:
            return facts.zero_at.get(position) == base
    return base in visible


def _fuse_block(
    block: list[ast.stmt],
    reads: Counter[str],
    defs: Counter[str],
    visible: set[str],
) -> None:
    facts = _collect(block, defs)
    # Zeros that reach a nested statement: assigned zero in this or an
    # enclosing block and not written again before it, nor by it.
    for position, stmt in enumerate(block):
        candidates = visible | {
            name for z, name in facts.zero_at.items() if z < position
        }
        nested = {
            name
            for name in candidates
            if _zero_base(facts, name, position, visible)
            and name not in facts.writes_at[position]
        }
        for field in ("body", "orelse", "finalbody"):
            child = getattr(stmt, field, None)
            if isinstance(child, list) and child and isinstance(child[0], ast.stmt):
                _fuse_block(child, reads, defs, nested)
    if not facts.vector or len(facts.add) < 2 * GROUP_BYTES:
        return
    halves = _decoded_halves(facts)
    if len(halves) < 4 * GROUP_BYTES:
        return
    chain_inputs = {operand for operands in facts.add.values() for operand in operands}
    finals = [name for name in facts.add if name not in chain_inputs]
    removed: set[str] = set()
    replacement: dict[str, ast.stmt] = {}
    for final in finals:
        match = _match_chain(final, facts, halves)
        if match is None:
            continue
        w, x, consumed, base, first_add = match
        if not _zero_base(facts, base, first_add, visible):
            continue
        consumed_set = set(consumed)
        # Each intermediate value is read only by the matched statements
        # (a decoded pair by its two half extracts, everything else once);
        # the final sum keeps its name and its readers.
        pairs = {halves[name][4] for name in consumed_set if name in halves}
        if any(
            reads[name] != (2 if name in pairs else 1)
            for name in consumed_set
            if name != final
        ):
            continue
        removed.update(consumed_set - {final})
        replacement[final] = statement_from_string(
            f"{final} = cutlass.Float32({DOT_FN}({w}, {x}))"
        )
    if not replacement:
        return
    new_block: list[ast.stmt] = []
    for stmt in block:
        name = _assigned_name(stmt)
        if name in removed:
            continue
        if name in replacement:
            new_block.append(replacement[name])
            continue
        new_block.append(stmt)
    block[:] = new_block


def fuse_fp4_pair_dot(body: list[ast.stmt]) -> list[ast.stmt]:
    """Apply the fusion to every block of ``body`` (mutates in place)."""
    module = ast.Module(body=body, type_ignores=[])
    reads: Counter[str] = Counter()
    defs: Counter[str] = Counter()
    for node in ast.walk(module):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load):
                reads[node.id] += 1
            else:
                defs[node.id] += 1
    _fuse_block(body, reads, defs, set())
    return body

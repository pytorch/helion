"""Emit the admitted matrix transports and the existing typed scalar recipes."""

from __future__ import annotations

import ast
import textwrap
from typing import TYPE_CHECKING

from .row_matrix_transport import ROW_MATRIX_BARRIERS
from .row_resident_codegen import _scalar_epilogue

if TYPE_CHECKING:
    from .row_matrix_transport import RowMatrixPlan
    from .row_resident import RowStore


# Per-instruction primitives. No borrowed kernel body or arithmetic recipe is
# embedded here. In particular, all elementwise floating-point work below is
# rendered from the typed graph, and the MMA K sequence is emitted explicitly.
_PRIMITIVES = """
@dsl_user_op
def _row_matrix_load(address, count: cutlass.Constexpr, transpose: cutlass.Constexpr, *, loc=None, ip=None):
    assert count in (1, 2, 4)
    word = ir_types.IntegerType.get_signless(32)
    outputs = ", ".join(f"${i}" for i in range(count))
    trans = ".trans" if transpose else ""
    instruction = f"ldmatrix.sync.aligned.m8n8.x{count}{trans}.shared.b16 {{{outputs}}}, [${count}];"
    result = llvm.inline_asm(llvm.StructType.get_literal([word] * count), [cutlass.Int32(address).ir_value(loc=loc, ip=ip)], instruction, ",".join(["=r"] * count + ["r", "~{memory}"]), has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return tuple(cutlass.Uint32(llvm.extractvalue(word, result, [i], loc=loc, ip=ip)) for i in range(count))

@dsl_user_op
def _row_matrix_store(address, words, *, loc=None, ip=None):
    count = len(words)
    assert count in (1, 2, 4)
    values = ", ".join(f"${i + 1}" for i in range(count))
    instruction = f"stmatrix.sync.aligned.m8n8.x{count}.shared.b16 [$0], {{{values}}};"
    llvm.inline_asm(None, [cutlass.Int32(address).ir_value(loc=loc, ip=ip), *[cutlass.Uint32(value).ir_value(loc=loc, ip=ip) for value in words]], instruction, ",".join(["r"] * (count + 1) + ["~{memory}"]), has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

@dsl_user_op
def _row_global_load(address, count: cutlass.Constexpr, *, loc=None, ip=None):
    assert count in (1, 4)
    word = ir_types.IntegerType.get_signless(32)
    outputs = ", ".join(f"${i}" for i in range(count))
    vector = ".v4" if count == 4 else ""
    operands = "{" + outputs + "}" if count == 4 else outputs
    instruction = "{ .reg .b64 policy; createpolicy.fractional.L2::evict_last.b64 policy, 1.0; " + f"ld.global.L1::evict_last.L2::cache_hint{vector}.b32 {operands}, [${count}], policy; }}"
    result = llvm.inline_asm(llvm.StructType.get_literal([word] * count), [cutlass.Int64(address).ir_value(loc=loc, ip=ip)], instruction, ",".join(["=r"] * count + ["l", "~{memory}"]), has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return tuple(cutlass.Uint32(llvm.extractvalue(word, result, [i], loc=loc, ip=ip)) for i in range(count))

@dsl_user_op
def _row_global_half(address, *, loc=None, ip=None):
    bits = llvm.inline_asm(ir_types.IntegerType.get_signless(16), [cutlass.Int64(address).ir_value(loc=loc, ip=ip)], "{ .reg .b64 policy; createpolicy.fractional.L2::evict_last.b64 policy, 1.0; ld.global.L1::evict_last.L2::cache_hint.b16 $0, [$1], policy; }", "=h,l,~{memory}", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Float16(llvm.bitcast(ir_types.F16Type.get(), bits, loc=loc, ip=ip))

@dsl_user_op
def _row_shared_packet(address, words, *, loc=None, ip=None):
    assert len(words) == 4
    llvm.inline_asm(None, [cutlass.Int32(address).ir_value(loc=loc, ip=ip), *[cutlass.Uint32(value).ir_value(loc=loc, ip=ip) for value in words]], "st.shared::cta.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r,~{memory}", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

@dsl_user_op
def _row_global_store(address, word, *, loc=None, ip=None):
    llvm.inline_asm(None, [cutlass.Int64(address).ir_value(loc=loc, ip=ip), cutlass.Uint32(word).ir_value(loc=loc, ip=ip)], "st.global.b32 [$0], $1;", "l,r,~{memory}", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)

@dsl_user_op
def _row_mma(a0, a1, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
    f32 = ir_types.F32Type.get()
    values = [*[cutlass.Uint32(value).ir_value(loc=loc, ip=ip) for value in (a0, a1, b0, b1)], *[cutlass.Float32(value).ir_value(loc=loc, ip=ip) for value in (c0, c1, c2, c3)]]
    result = llvm.inline_asm(llvm.StructType.get_literal([f32] * 4), values, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0, $1, $2, $3}, {$4, $4, $5, $5}, {$6, $7}, {$8, $9, $10, $11};", "=f,=f,=f,=f,r,r,r,r,f,f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return tuple(cutlass.Float32(llvm.extractvalue(f32, result, [i], loc=loc, ip=ip)) for i in range(4))

@dsl_user_op
def _row_pack(low, high, *, loc=None, ip=None):
    result = llvm.inline_asm(ir_types.IntegerType.get_signless(32), [cutlass.Float32(high).ir_value(loc=loc, ip=ip), cutlass.Float32(low).ir_value(loc=loc, ip=ip)], "cvt.rn.f16x2.f32 $0, $1, $2;", "=r,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Uint32(result)

@dsl_user_op
def _row_product(left, right, *, loc=None, ip=None):
    result = llvm.inline_asm(ir_types.IntegerType.get_signless(32), [cutlass.Uint32(left).ir_value(loc=loc, ip=ip), cutlass.Uint32(right).ir_value(loc=loc, ip=ip)], "mul.rn.f16x2 $0, $1, $2;", "=r,r,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return cutlass.Uint32(result)
"""


def _compact(row: str, column: str, width: int) -> str:
    return f"(2 * (({row}) * {width} + ({column})) ^ (({row}) * 16))"


def _middle(row: str, column: str, width: int) -> str:
    permuted = f"(({column}) // 64 * 64 + ({column}) % 32 * 2 + ({column}) // 32 % 2)"
    return f"(2 * (({row}) * {width} + {permuted}) ^ (({row}) * 4))"


def _weights(plan: RowMatrixPlan, index: int, argument: str) -> str:
    layout = plan.layouts[index]
    n = layout.columns
    lines = []
    # Retain the complete group of loads before publication, just as for the
    # register fragments below. No new cut of the original K sum is introduced.
    for packet in range(layout.weight_packets):
        offset = f"(tid * 8 + {packet * 2048})"
        lines.append(
            f"s{index}_weight_{packet} = _row_global_load(({argument}.iterator + {offset}).toint(), 4)\n"
        )
    for packet in range(layout.weight_packets):
        offset = f"(tid * 8 + {packet * 2048})"
        mapped = f"(2 * {offset} ^ (({offset} // {n} % 8) * 16))"
        lines.append(
            f"_row_shared_packet((arena_ptr + {mapped} // 2).toint(), s{index}_weight_{packet})\n"
        )
    return "".join(lines)


def _contraction(plan: RowMatrixPlan, index: int) -> str:
    layout = plan.layouts[index]
    n, k = layout.columns, layout.reduction
    p = f"s{index}"
    lines = []
    for packet in range(k // 32):
        column = f"({packet * 32} + lane // 8 * 8)"
        address = _compact("lane % 2", column, k)
        lines.append(
            f"{p}_a{packet} = _row_matrix_load((operand_a_ptr + {address} // 2).toint(), 4, False)\n"
        )
    for column_packet in range(layout.column_packets):
        for packet in range(k // 32):
            reduction = f"({packet * 32} + lane)"
            column = f"(warp * 8 + {column_packet * 64})"
            address = f"(2 * ({reduction} * {n} + {column}) ^ (({reduction} % 8) * 16))"
            lines.append(
                f"{p}_b{column_packet}_{packet} = _row_matrix_load((arena_ptr + {address} // 2).toint(), 4, True)\n"
            )
    for column_packet in range(layout.column_packets):
        for word in range(4):
            lines.append(f"{p}_c{column_packet}_{word} = cutlass.Float32(0.0)\n")
    # Interleave independent column fragments, retaining ascending K for each
    # accumulator and one explicit positive-zero seed for every physical row.
    for step in range(k // 16):
        packet, half = divmod(step, 2)
        for column_packet in range(layout.column_packets):
            names = ", ".join(f"{p}_c{column_packet}_{word}" for word in range(4))
            a = f"{p}_a{packet}"
            b = f"{p}_b{column_packet}_{packet}"
            lines.append(
                f"{names} = _row_mma({a}[{half * 2}], {a}[{half * 2 + 1}], "
                f"{b}[{half * 2}], {b}[{half * 2 + 1}], {names})\n"
            )
    return "".join(lines)


def _epilogue(store: RowStore, index: int, packets: int) -> str:
    lines = []
    counter = 0

    def fresh(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"s{index}_{prefix}_{counter}"

    assert not store.chain.auxiliary_tensor_loads
    for packet in range(packets):
        for half in range(2):
            value = f"s{index}_c{packet}_{half}"
            if store.chain.steps:
                prelude, value = store.chain.render_prelude_and_expr(value, fresh, "")
                lines.append(_scalar_epilogue(prelude))
            # This is the existing materialized store cast, not a new rounding.
            lines.append(
                f"s{index}_rounded{packet}_{half} = cutlass.Float16({value})\n"
            )
    return "".join(lines)


def row_matrix_module(plan: RowMatrixPlan, min_blocks: int = 0) -> str:
    row = plan.row
    first, second = plan.layouts
    arguments = {name: f"arg{i}" for i, name in enumerate(row.arguments)}
    x = arguments[row.stages[0].lhs_name]
    middle = arguments[row.stages[0].stores[0].name]
    materialized = arguments[plan.materialized.name]
    product = arguments[plan.product.name]
    auxiliary = arguments[plan.product_input]
    k0, n0, n1 = first.reduction, first.columns, second.columns
    body = [
        "tid = cutlass.Int32(cute.arch.thread_idx()[0])\n",
        "lane = tid % 32\nwarp = tid // 32\nlogical_row = lane // 4 % 2\n",
        "row_start = cutlass.Int32(cute.arch.block_idx()[0]) * 2\n",
        f"arena_ptr = cute.arch.alloc_smem(cutlass.Float16, {plan.shared_bytes // 2}, alignment=1024)\n",
        f"operand_a_ptr = arena_ptr + {plan.weight_bytes // 2}\n",
        f"arena = cute.make_tensor(arena_ptr, cute.make_layout(({plan.weight_bytes // 2},), stride=(1,)))\n",
        f"operand_a = cute.make_tensor(operand_a_ptr, cute.make_layout(({2 * max(row.reductions)},), stride=(1,)))\n",
    ]
    phase = 0

    def barrier() -> None:
        nonlocal phase
        assert phase < len(ROW_MATRIX_BARRIERS)
        body.append(f"# {ROW_MATRIX_BARRIERS[phase]}\ncute.arch.sync_threads()\n")
        phase += 1

    # Warp-uniform collective participation, with the original odd-row load
    # mask inside it. All lanes reconverge before the matrix instruction.
    segment = "(warp * 64 + lane % 8 * 8)"
    address = _compact(f"{segment} // {k0}", f"{segment} % {k0}", k0)
    body.extend(
        (
            f"""
if warp < {k0 // 32}:
    input_row = tid * 2 // {k0}
    input_column = tid * 2 % {k0}
    input_word = cutlass.Uint32(0)
    if row_start + input_row < {row.rows}:
        input_word = _row_global_load(({x}.iterator + (row_start + input_row) * {k0} + input_column).toint(), 1)[0]
    _row_matrix_store((operand_a_ptr + {address} // 2).toint(), (input_word,))
""",
            _weights(plan, 0, arguments[row.stages[0].rhs_name]),
        )
    )
    barrier()
    body.extend(
        (
            _contraction(plan, 0),
            _epilogue(row.stages[0].stores[0], 0, first.column_packets),
        )
    )
    barrier()
    # Canonical physical rows 0/1, rather than aliased scalar shared writes.
    for packet in range(first.column_packets):
        for half in range(2):
            column = f"(warp * 8 + {packet * 64} + lane % 4 * 2 + {half})"
            address = _middle("logical_row", column, n0)
            body.append(f"""
if lane < 8:
    arena[{address} // 2] = s0_rounded{packet}_{half}
""")
    barrier()
    for slot in range((2 * n0 + 255) // 256):
        flat = f"(tid + {slot * 256})"
        middle_row, column = f"{flat} // {n0}", f"{flat} % {n0}"
        address = _middle(middle_row, column, n0)
        body.append(f"""
if {flat} < {2 * n0} and row_start + {middle_row} < {row.rows}:
    {middle}[row_start + {middle_row}, {column}] = arena[{address} // 2]
""")
    barrier()
    for slot in range((2 * n0 + 255) // 256):
        flat = f"(tid + {slot * 256})"
        middle_row, column = f"{flat} // {n0}", f"{flat} % {n0}"
        address = _compact(middle_row, column, n0)
        body.append(f"""
if {flat} < {2 * n0}:
    middle_value = cutlass.Float16(0.0)
    if row_start + {middle_row} < {row.rows}:
        middle_value = _row_global_half(({middle}.iterator + (row_start + {middle_row}) * {n0} + {column}).toint())
    operand_a[{address} // 2] = middle_value
""")
    body.append(_weights(plan, 1, arguments[row.stages[1].rhs_name]))
    barrier()
    body.extend(
        (_contraction(plan, 1), _epilogue(plan.materialized, 1, second.column_packets))
    )
    for packet in range(second.column_packets):
        body.append(
            f"output_word{packet} = _row_pack(cutlass.Float32(s1_rounded{packet}_0), cutlass.Float32(s1_rounded{packet}_1))\n"
        )
    barrier()
    column = f"(warp * 8 + lane // 8 % {second.column_packets} * 64)"
    address = _compact("lane % 2", column, n1)
    words = ", ".join(f"output_word{packet}" for packet in range(second.column_packets))
    body.append(
        f"_row_matrix_store((arena_ptr + {address} // 2).toint(), ({words},))\n"
    )
    barrier()
    address = _compact(f"{segment} // {n1}", f"{segment} % {n1}", n1)
    body.append(f"""
auxiliary_word = cutlass.Uint32(0)
if warp < {n1 // 32}:
    final_row = tid * 2 // {n1}
    final_column = tid * 2 % {n1}
    final_word = _row_matrix_load((arena_ptr + {address} // 2).toint(), 1, False)[0]
    if row_start + final_row < {row.rows}:
        _row_global_store(({materialized}.iterator + (row_start + final_row) * {n1} + final_column).toint(), final_word)
        auxiliary_word = _row_global_load(({auxiliary}.iterator + (row_start + final_row) * {n1} + final_column).toint(), 1)[0]
""")
    barrier()
    operands = (
        "materialized_word, auxiliary_word"
        if plan.materialized_is_left_operand
        else "auxiliary_word, materialized_word"
    )
    body.append(f"""
if tid < {n1}:
    final_row = tid * 2 // {n1}
    final_column = tid * 2 % {n1}
    if row_start + final_row < {row.rows}:
        materialized_word = _row_global_load(({materialized}.iterator + (row_start + final_row) * {n1} + final_column).toint(), 1)[0]
        product_word = _row_product({operands})
        _row_global_store(({product}.iterator + (row_start + final_row) * {n1} + final_column).toint(), product_word)
""")
    assert phase == len(ROW_MATRIX_BARRIERS) == 8
    source = (
        "from __future__ import annotations\nimport cutlass\nimport cutlass.cute as cute\n"
        "from cutlass._mlir.dialects import llvm\nfrom cutlass._mlir.extras import types as ir_types\n"
        "from cutlass.cutlass_dsl import dsl_user_op\n"
        + _PRIMITIVES
        + f"\n@cute.kernel\ndef _helion_row_resident({', '.join(arguments.values())}):\n"
        + textwrap.indent("".join(body), "    ")
    )
    if min_blocks:
        source += (
            f"\n_helion_row_resident._helion_cute_min_blocks_per_mp = {min_blocks}\n"
        )
    return ast.unparse(ast.parse(source)) + "\n"

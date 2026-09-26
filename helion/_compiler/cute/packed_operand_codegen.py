"""Generate an ordinary CuTe launch for a proved staged packed operand.

The M32/N4 layout uses four physical m16n8k16 warps with replicated columns
and row ownership. Only the unique publishers write the shared output. All
staging widths, row strides, loop counts and arena offsets derive from the
guarded complete reduction extent, never from a kernel name or sample shape.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from typing import TYPE_CHECKING

from ... import exc
from .full_slice_matmul import _bound_names
from .packed_operand import KEY
from .packed_operand import PackedOperandPlan
from .packed_operand import prove_packed_operand

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..host_function import HostFunction


def _intrinsics(row_stride: int) -> str:
    """Architectural copy/MMA primitives; the planner supplies shared stride."""
    return f'''
@dsl_user_op
def _packed_copy(smem, gmem, width: int, *, loc=None, ip=None):
    assert width in (4, 16)
    policy = "cg" if width == 16 else "ca"
    llvm.inline_asm(
        None,
        [smem.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), gmem.llvm_ptr],
        f"cp.async.{{policy}}.shared.global [$0], [$1], {{width}};",
        "r,l", has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )

@dsl_user_op
def _packed_mma(a0, a2, b0, c0, c1, c2, c3, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([cutlass.Float32.mlir_type] * 4),
        [a0.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
         a2.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
         b0.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
         cutlass.Float32(c0).ir_value(loc=loc, ip=ip),
         cutlass.Float32(c1).ir_value(loc=loc, ip=ip),
         cutlass.Float32(c2).ir_value(loc=loc, ip=ip),
         cutlass.Float32(c3).ir_value(loc=loc, ip=ip)],
        """{{
        .reg .b32 a<4>, b<2>;
        .reg .b16 h<4>;
        ld.shared.b32 a0, [$4];
        ld.shared.b32 a1, [$4+{8 * row_stride}];
        ld.shared.b32 a2, [$5];
        ld.shared.b32 a3, [$5+{8 * row_stride}];
        ld.shared.b16 h0, [$6];
        ld.shared.b16 h1, [$6+8];
        ld.shared.b16 h2, [$6+128];
        ld.shared.b16 h3, [$6+136];
        mov.b32 b0, {{h0, h1}};
        mov.b32 b1, {{h2, h3}};
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
            {{$0, $1, $2, $3}}, {{a0, a1, a2, a3}}, {{b0, b1}}, {{$7, $8, $9, $10}};
        }}""",
        "=f,=f,=f,=f,r,r,r,f,f,f,f", has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return tuple(cutlass.Float32(llvm.extractvalue(
        cutlass.Float32.mlir_type, result, [i], loc=loc, ip=ip)) for i in range(4))
'''


def packed_operand_module(plan: PackedOperandPlan, min_blocks: int = 0) -> str:
    k = plan.reduction
    # ScalarRecipe owns reaching definitions. Names in emitted slices cannot
    # capture generated coordinates or overwrite another replay's definitions.
    occupied = {"packed_value"}

    def fresh(hint: str) -> str:
        while hint in occupied:
            hint += "_"
        occupied.add(hint)
        return hint

    recipe_lines: list[str] = []
    for index, recipe in enumerate(plan.recipes):
        statements, value = recipe.emit({}, fresh)
        recipe_lines.extend(ast.unparse(stmt) for stmt in statements)
        recipe_lines.append(f"field_{index} = {ast.unparse(value)}")
    recipes = textwrap.indent("\n".join(recipe_lines), "        ")
    body = f"""
@cute.kernel
def _helion_packed_operand(A, B, C):
    t = cutlass.Int32(cute.arch.thread_idx()[0])
    warp = t // 32
    lane = t % 32
    g = lane // 4
    u = lane % 4
    block_m = cutlass.Int32(cute.arch.block_idx()[0])
    block_n = cutlass.Int32(cute.arch.block_idx()[1])
    shared = cute.arch.alloc_smem(cutlass.Uint8, {plan.shared_bytes}, alignment=128)
    packed = cute.recast_ptr(shared + {plan.a_bytes + plan.decoded_bytes}, dtype=cutlass.Int8)
    decoded = cute.recast_ptr(shared + {plan.a_bytes}, dtype=cutlass.BFloat16)
    output_shared = cute.recast_ptr(shared, dtype=cutlass.BFloat16)
    for j in cutlass.range_constexpr({k // 32}):
        packet = t + 128 * j
        m = packet // {k // 8}
        column = (packet % {k // 8}) * 8
        a_byte = {2 * k} * m + ((2 * column) ^ (32 * (m % 4)))
        _packed_copy(shared + a_byte, A.iterator + (32 * block_m + m) * {k} + column, 16)
    if t < {k // 2}:
        _packed_copy(packed + 4 * t, B.iterator + t * {plan.columns} + 4 * block_n, 4)
    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.sync_threads()
    for j in cutlass.range_constexpr({k // 32}):
        logical = t + 128 * j
        packed_index = (logical // 8) * 4 + logical % 4
        packed_value = (packed + packed_index).load()
{recipes}
        unpacked = field_0 if (logical // 4) % 2 == 0 else field_1
        (decoded + logical).store(unpacked)
    cute.arch.sync_threads()
    c0 = cutlass.Float32(0)
    c1 = cutlass.Float32(0)
    c2 = cutlass.Float32(0)
    c3 = cutlass.Float32(0)
    for step in cutlass.range_constexpr({k // 16}):
        m = g + 16 * (warp % 2)
        column = 32 * (step // 2) + 2 * (step % 2) + 4 * u
        a_byte = {2 * k} * m + ((2 * column) ^ (32 * (m % 4)))
        b_byte = 8 * column + 2 * (g % 4)
        c0, c1, c2, c3 = _packed_mma(
            shared + a_byte, shared + (a_byte ^ 32),
            shared + {plan.a_bytes} + b_byte, c0, c1, c2, c3)
    cute.arch.sync_threads()
    if warp < 2 and u < 2:
        m = g + 16 * warp
        n = 2 * u
        (output_shared + m * 4 + n).store(cutlass.BFloat16(c0))
        (output_shared + m * 4 + n + 1).store(cutlass.BFloat16(c1))
        (output_shared + (m + 8) * 4 + n).store(cutlass.BFloat16(c2))
        (output_shared + (m + 8) * 4 + n + 1).store(cutlass.BFloat16(c3))
    cute.arch.sync_threads()
    m = t // 4
    n = t % 4
    value = (output_shared + t).load()
    (C.iterator + (block_m * 32 + m) * {plan.columns} + block_n * 4 + n).store(value)
"""
    source = (
        "from __future__ import annotations\n"
        "import operator\nimport cutlass\nimport cutlass.cute as cute\n"
        "from cutlass._mlir.dialects import llvm\n"
        "from cutlass.cutlass_dsl import dsl_user_op\n" + _intrinsics(2 * k) + body
    )
    if min_blocks:
        source += (
            f"\n_helion_packed_operand._helion_cute_min_blocks_per_mp = {min_blocks}\n"
        )
    return ast.unparse(ast.parse(source)) + "\n"


def generate_packed_operand(
    func: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.Module:
    from ..compile_environment import CompileEnvironment
    from ..generate_ast import generate_ast

    env = CompileEnvironment.current()
    if config.get(KEY) != "warp_narrow4":
        raise exc.InvalidConfig("unsupported packed-operand schedule")
    with func:
        plan = prove_packed_operand(env, func)
    if plan is None:
        raise exc.InvalidConfig(
            "packed-operand schedule requires a proved complete byte producer and contraction"
        )
    ordinary = type(config).from_dict(
        {key: value for key, value in config.config.items() if key != KEY}
    )
    module = generate_ast(func, ordinary, emit_repro_caller)
    host = next(
        stmt
        for stmt in module.body
        if isinstance(stmt, ast.FunctionDef) and stmt.name == func.name
    )
    metadata = [
        stmt
        for stmt in module.body
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Attribute)
        and isinstance(stmt.targets[0].value, ast.Name)
        and stmt.targets[0].value.id == func.name
        and stmt.targets[0].attr == "_helion_cute_kernels"
    ]
    assert len(metadata) == 1 and isinstance(metadata[0].value, ast.Tuple)
    devices = metadata[0].value.elts
    assert len(devices) == 2 and all(
        isinstance(node, ast.Attribute) for node in devices
    )
    modules = [
        ast.unparse(node.value) for node in devices if isinstance(node, ast.Attribute)
    ]
    calls = [
        stmt
        for stmt in host.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and ast.unparse(stmt.value.func.value) in modules
    ]
    if len(calls) != 2 or host.body.index(calls[1]) != host.body.index(calls[0]) + 1:
        raise exc.InvalidConfig(
            "packed-operand caller requires adjacent owned region launches"
        )
    allocations = [
        stmt
        for stmt in host.body
        if isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and stmt.targets[0].id == plan.materialized_name
    ]
    if len(allocations) != 1 or host.body.index(allocations[0]) >= host.body.index(
        calls[0]
    ):
        raise exc.InvalidConfig(
            "packed-operand caller requires one private materialized allocation"
        )
    used = _bound_names(func) | {
        node.id for node in ast.walk(module) if isinstance(node, ast.Name)
    }

    def fresh(prefix: str) -> str:
        while prefix in used:
            prefix += "_"
        used.add(prefix)
        return prefix

    module_name = fresh("_helion_packed_module")
    code_cache = fresh("_helion_packed_code_cache")
    min_blocks = config.get("cute_min_blocks_per_mp", 0)
    assert isinstance(min_blocks, int)
    source = packed_operand_module(plan, min_blocks)
    kernel = f"{module_name}._helion_packed_operand"
    arguments = ", ".join(plan.arguments)
    replacement = ast.parse(
        f"_launcher({kernel}, ({plan.rows // 32}, {plan.columns // 4}, 1), {arguments}, block=(128, 1, 1))"
    ).body[0]
    start = host.body.index(calls[0])
    host.body[start : start + 2] = [replacement]
    host.body.remove(allocations[0])
    if any(
        isinstance(node, ast.Name) and node.id == plan.materialized_name
        for node in ast.walk(host)
    ):
        raise exc.InvalidConfig("materialized operand escapes its two owned launches")
    module.body = [
        stmt
        for stmt in module.body
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and (
                isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id in modules
                or isinstance(stmt.targets[0], ast.Attribute)
                and isinstance(stmt.targets[0].value, ast.Attribute)
                and ast.unparse(stmt.targets[0].value.value) in modules
            )
        )
    ]
    position = module.body.index(host)
    module.body[position:position] = ast.parse(
        f"from torch._inductor.codecache import PyCodeCache as {code_cache}\n"
        f"{module_name} = {code_cache}.load({source!r})\n"
        f"{kernel}._helion_cute_source_hash = {hashlib.sha256(source.encode()).hexdigest()!r}\n"
    ).body
    metadata[0].value = ast.parse(f"({kernel},)", mode="eval").body
    env.cute_resolved_wrapper_plans = []
    return module

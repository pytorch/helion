"""One ordinary CuTe launch for a proved compact materialized row schedule.

The row/packet/MMA layout is independent of the scalar epilogue. Epilogues use
the existing typed CuTe recipes, including each original low precision cast.
No source from another backend or generated-name matcher participates here.
"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from typing import TYPE_CHECKING

import torch

from ... import exc
from ._ast_pass_utils import _bound_names
from .row_matrix_transport import ROW_MATRIX_MODE
from .row_matrix_transport import prove_row_matrix_transport
from .row_resident import ROW_RESIDENT_KEY
from .row_resident import prove_row_resident

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..host_function import HostFunction
    from .row_resident import RowResidentPlan


# A single architecture intrinsic, not a whole kernel/body from another
# compiler. The typed address/lane proof covers both transpose choices.
_MATRIX_LOAD = """
@dsl_user_op
def _row_ldmatrix(address, transpose: cutlass.Constexpr, *, loc=None, ip=None):
    word_type = ir_types.IntegerType.get_signless(32)
    instruction = "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];" if transpose else "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];"
    result = llvm.inline_asm(llvm.StructType.get_literal([word_type] * 4), [cutlass.Int32(address).ir_value(loc=loc, ip=ip)], instruction, "=r,=r,=r,=r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return tuple(cutlass.Uint32(llvm.extractvalue(word_type, result, [i], loc=loc, ip=ip)) for i in range(4))
"""


def _scalar_epilogue(source: str) -> str:
    """Use the ordinary scalar conditional for the proved tensor recipe.

    The shared epilogue renderer describes elementwise expressions on a
    TensorSSA. Every value here is one FP32 element. Its only tensor-only
    operation in our admitted FX vocabulary is ``cute.where``; the ordinary
    CuTe scalar lowering uses the same conditional expression. Keep both
    nested ReLU predicates, including its explicit NaN propagation.
    """

    class ScalarWhere(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call) -> ast.AST:
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "cute"
                and node.func.attr == "where"
            ):
                assert len(node.args) == 3 and not node.keywords
                condition, positive, negative = node.args
                return ast.copy_location(
                    ast.IfExp(test=condition, body=positive, orelse=negative), node
                )
            return node

    return ast.unparse(ScalarWhere().visit(ast.parse(source))) + "\n"


def _packet_copy(plan: RowResidentPlan, index: int, *, operand_a: bool) -> str:
    stage = plan.stages[index]
    n, k = plan.columns[index], plan.reductions[index]
    p = f"s{index}"
    arguments = {name: f"arg{i}" for i, name in enumerate(plan.arguments)}
    if operand_a and index == 1:
        return f"""
for flat_slot in cutlass.range_constexpr({(2 * k + 255) // 256}):
    flat = tid + flat_slot * 256
    if flat < {2 * k}:
        r = flat // {k}
        c = flat % {k}
        {p}_a[r, c] = row_cache[r, c] if row_start + r < {plan.rows} else cutlass.Float16(0.0)
"""
    name = arguments[stage.lhs_name if operand_a else stage.rhs_name]
    elements = 2 * k if operand_a else n * k
    target = f"{p}_a" if operand_a else f"{p}_b"
    axes = (
        f"r = flat // {k}\nc = flat % {k}"
        if operand_a
        else f"r = flat % {n}\nc = flat // {n}"
    )
    valid = f"row_start + r < {plan.rows}" if operand_a else "True"
    offset = f"(row_start + r) * {k} + c" if operand_a else f"c * {n} + r"
    target_index = "r, c + tail" if operand_a else "r + tail, c"
    return f"""
for packet_slot in cutlass.range_constexpr({(elements + 2047) // 2048}):
    flat = (tid + packet_slot * 256) * 8
    if flat < {elements}:
{textwrap.indent(axes, "        ")}
        if {valid}:
            source = cute.make_tensor(cute.make_ptr(cutlass.Float16, ({name}.iterator + {offset}).toint(), cute.AddressSpace.gmem, assumed_align=16), cute.make_layout((8,), stride=(1,)))
            destination = cute.make_tensor({target}.iterator + cute.assume(cute.crd2idx((r, c), {target}.layout), divby=8), cute.make_layout((8,), stride=(1,)))
            cute.copy(cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(cute.nvgpu.LoadCacheMode.GLOBAL), cutlass.Float16, num_bits_per_copy=128), source, destination)
        else:
            for tail in cutlass.range_constexpr(8):
                {target}[{target_index}] = cutlass.Float16(0.0)
"""


def _stage_source(plan: RowResidentPlan, index: int) -> str:
    stage = plan.stages[index]
    n, k = plan.columns[index], plan.reductions[index]
    p = f"s{index}"
    a_swizzle = min(3, (k // 8).bit_length() - 1)
    arguments = {name: f"arg{i}" for i, name in enumerate(plan.arguments)}
    setup = f"""
{p}_a = cute.make_tensor(cute.recast_ptr(operand_a_ptr, cute.make_swizzle({a_swizzle}, 4, 3)), cute.make_layout(((2, 8), {k}), stride=(({k}, 0), 1)))
{p}_b = cute.make_tensor(cute.recast_ptr(operand_b_ptr, cute.make_swizzle(3, 4, 3)), cute.make_layout(({n}, {k}), stride=(1, {n})))
{p}_mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op(cutlass.Float16, cutlass.Float32, (16, 8, 16)), atom_layout_mnk=(1, 8, 1), permutation_mnk=(16, cute.make_ordered_layout((8, 2, 4), order=(0, 2, 1)), 16))
{p}_thr = {p}_mma.get_slice(tid)
{p}_acc = cute.make_rmem_tensor({p}_mma.partition_shape_C((16, {n})), cutlass.Float32)
{p}_ra = {p}_thr.make_fragment_A({p}_thr.partition_shape_A((16, {k})))
{p}_rb = {p}_thr.make_fragment_B({p}_thr.partition_shape_B(({n}, {k})))
{p}_acc.fill(0.0)
"""
    copies = _packet_copy(plan, index, operand_a=True) + _packet_copy(
        plan, index, operand_a=False
    )
    shared_ready = "cute.arch.cp_async_commit_group()\ncute.arch.cp_async_wait_group(0)\ncute.arch.sync_threads()\n"
    load_a = f"""
{p}_ra_words = cute.recast_tensor({p}_ra, cutlass.Uint32)
for pair in cutlass.range_constexpr({k // 32}):
    compact_row = tid % 8 % 2
    compact_k = pair * 32 + tid % 32 // 8 * 8
    address = (operand_a_ptr + compact_row * {k} + compact_k).toint()
    address = address ^ ((address >> 3) & {((1 << a_swizzle) - 1) << 4})
    word0, word1, word2, word3 = _row_ldmatrix(address, False)
    {p}_ra_words[pair * 8] = word0
    {p}_ra_words[pair * 8 + 1] = word0
    {p}_ra_words[pair * 8 + 2] = word1
    {p}_ra_words[pair * 8 + 3] = word1
    {p}_ra_words[pair * 8 + 4] = word2
    {p}_ra_words[pair * 8 + 5] = word2
    {p}_ra_words[pair * 8 + 6] = word3
    {p}_ra_words[pair * 8 + 7] = word3
"""
    if n == 64:
        load_b = f"""
{p}_rb_words = cute.recast_tensor({p}_rb, cutlass.Uint32)
{p}_b_coordinates = {p}_thr.partition_B(cute.make_identity_tensor(({n}, {k})))
compact_column = {p}_b_coordinates[0][0] // 8 * 8
for pair in cutlass.range_constexpr({k // 32}):
    compact_k = pair * 32 + tid % 32
    address = (operand_b_ptr + compact_k * {n} + compact_column).toint()
    address = address ^ ((address >> 3) & 112)
    word0, word1, word2, word3 = _row_ldmatrix(address, True)
    {p}_rb_words[pair * 4] = word0
    {p}_rb_words[pair * 4 + 1] = word1
    {p}_rb_words[pair * 4 + 2] = word2
    {p}_rb_words[pair * 4 + 3] = word3
"""
    else:
        load_b = f"""
{p}_cb = cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), cutlass.Float16), {p}_mma)
{p}_tcb = {p}_cb.get_slice(tid)
cute.copy({p}_cb, {p}_tcb.partition_S({p}_b), {p}_tcb.retile({p}_rb))
"""
    mma = f"""
for mma_k in cutlass.range_constexpr({k // 16}):
    cute.gemm({p}_mma, {p}_acc, {p}_ra[None, None, mma_k], {p}_rb[None, None, mma_k], {p}_acc)
{p}_coordinates = {p}_thr.partition_C(cute.make_identity_tensor((16, {n})))
for slot in cutlass.range_constexpr(cute.size({p}_acc.shape)):
    output_row = {p}_coordinates[slot][0]
    output_column = {p}_coordinates[slot][1]
    if output_row < 2 and row_start + output_row < {plan.rows}:
        accumulated = cutlass.Float32({p}_acc[slot])
"""
    epilogue = []
    counter = 0

    def fresh(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{p}_{prefix}_{counter}"

    for store in stage.stores:
        aux_locals = {}
        for aux in store.chain.auxiliary_tensor_loads:
            node = aux.load_node.args[0]
            assert isinstance(node, torch.fx.Node)
            aux_name = fresh("aux")
            assert isinstance(node.args[0], str)
            input_name = arguments[node.args[0]]
            epilogue.append(
                f"{aux_name} = cutlass.Float32({input_name}[row_start + output_row, output_column])\n"
            )
            aux_locals[aux] = aux_name
        value = "accumulated"
        if store.chain.steps:
            prelude, value = store.chain.render_prelude_and_expr(
                value, fresh, "", aux_locals or None
            )
            epilogue.append(_scalar_epilogue(prelude))
        rounded = fresh("rounded")
        epilogue.extend(
            (
                f"{rounded} = cutlass.Float16({value})\n",
                f"{arguments[store.name]}[row_start + output_row, output_column] = {rounded}\n",
            )
        )
        if index == 0:
            epilogue.append(f"row_cache[output_row, output_column] = {rounded}\n")
    return (
        setup
        + copies
        + shared_ready
        + load_a
        + load_b
        + mma
        + textwrap.indent("".join(epilogue), "        ")
    )


def row_resident_module(plan: RowResidentPlan, min_blocks: int = 0) -> str:
    arguments = ", ".join(f"arg{i}" for i in range(len(plan.arguments)))
    prelude = f"""
tid = cutlass.Int32(cute.arch.thread_idx()[0])
row_start = cutlass.Int32(cute.arch.block_idx()[0]) * 2
row_cache_ptr = cute.arch.alloc_smem(cutlass.Float16, {2 * plan.columns[0]}, alignment=128)
row_cache = cute.make_tensor(row_cache_ptr, cute.make_layout((2, {plan.columns[0]}), stride=({plan.columns[0]}, 1)))
operand_a_ptr = cute.arch.alloc_smem(cutlass.Float16, {2 * max(plan.reductions)}, alignment=1024)
operand_b_ptr = cute.arch.alloc_smem(cutlass.Float16, {max(n * k for n, k in zip(plan.columns, plan.reductions, strict=True))}, alignment=1024)
"""
    body = (
        prelude
        + _stage_source(plan, 0)
        + "\ncute.arch.sync_threads()\n"
        + _stage_source(plan, 1)
    )
    source = (
        "from __future__ import annotations\nimport cutlass\nimport cutlass.cute as cute\n"
        "from cutlass._mlir.dialects import llvm\nfrom cutlass._mlir.extras import types as ir_types\n"
        "from cutlass.cutlass_dsl import dsl_user_op\n"
        + _MATRIX_LOAD
        + f"\n@cute.kernel\ndef _helion_row_resident({arguments}):\n"
        + textwrap.indent(body, "    ")
    )
    if min_blocks:
        source += (
            f"\n_helion_row_resident._helion_cute_min_blocks_per_mp = {min_blocks}\n"
        )
    return ast.unparse(ast.parse(source)) + "\n"


def generate_row_resident(
    func: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.Module:
    from ..compile_environment import CompileEnvironment
    from ..generate_ast import generate_ast

    env = CompileEnvironment.current()
    mode = config.get(ROW_RESIDENT_KEY)
    if mode not in ("warp_rows2", ROW_MATRIX_MODE):
        raise exc.InvalidConfig("unsupported row-resident schedule")
    with func:
        plan = prove_row_resident(env, func)
    if plan is None:
        raise exc.InvalidConfig(
            "row-resident schedule requires two proved complete materialized row contractions"
        )
    matrix_plan = None
    if mode == ROW_MATRIX_MODE:
        with func:
            matrix_plan = prove_row_matrix_transport(env, func, plan)
        if matrix_plan is None:
            raise exc.InvalidConfig(
                "matrix row transport requires sm90+, aligned complete tiles, "
                "and an exact materialized FP16 product"
            )
    ordinary = type(config).from_dict(
        {key: value for key, value in config.config.items() if key != ROW_RESIDENT_KEY}
    )
    module = generate_ast(func, ordinary, emit_repro_caller)
    host = next(
        stmt
        for stmt in module.body
        if isinstance(stmt, ast.FunctionDef) and stmt.name == func.name
    )
    fission = env.cute_fission_plan
    assert fission is not None
    # The ordinary fission emitter records its exact two owned module callees.
    # Verify that metadata and replace only their calls; keep the complete
    # generated prelude, allocation order, return and public default launcher.
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
            "row-resident caller requires adjacent owned region launches"
        )
    used = _bound_names(module)

    def fresh(prefix: str) -> str:
        while prefix in used:
            prefix += "_"
        used.add(prefix)
        return prefix

    module_name = fresh("_helion_row_module")
    code_cache = fresh("_helion_row_code_cache")
    min_blocks = config.get("cute_min_blocks_per_mp", 0)
    assert isinstance(min_blocks, int)
    if matrix_plan is not None:
        from .row_matrix_transport_codegen import row_matrix_module

        source = row_matrix_module(matrix_plan, min_blocks)
    else:
        source = row_resident_module(plan, min_blocks)
    kernel = f"{module_name}._helion_row_resident"
    argument_text = ", ".join(plan.arguments)
    replacement = ast.parse(
        f"_launcher({kernel}, ({(plan.rows + 1) // 2}, 1, 1), {argument_text}, block=(256, 1, 1))"
    ).body[0]
    start = host.body.index(calls[0])
    host.body[start : start + 2] = [replacement]
    # Remove now-dead ordinary stage modules, retaining all host imports and
    # statements. The generated nested source is replaced, never searched for
    # a particular kernel or tensor name.
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
        f"from torch._inductor.codecache import PyCodeCache as {code_cache}\n{module_name} = {code_cache}.load({source!r})\n{kernel}._helion_cute_source_hash = {hashlib.sha256(source.encode()).hexdigest()!r}\n"
    ).body
    metadata[0].value = ast.parse(f"({kernel},)", mode="eval").body
    env.cute_resolved_wrapper_plans = []
    return module

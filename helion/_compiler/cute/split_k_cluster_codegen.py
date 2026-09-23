"""One ordinary cluster launch for a typed private split-K reduction."""

from __future__ import annotations

import ast
import hashlib
import symtable
from typing import TYPE_CHECKING

import torch

from ... import exc
from ...runtime.cute.source_dependencies import wrapper_source_dependencies
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..output_header import get_needed_import_lines
from ..type_info import CallableType
from .materialized_fission_codegen import _empty_graph
from .memory_ops import tensor_has_specialized_tma_alignment
from .split_k_cluster import CLUSTER_K_SCHEDULE
from .split_k_cluster import checked_cluster_proof
from .split_k_cluster_config import CLUSTER8_K4
from .split_k_cluster_config import FINALIZER_KEY
from .split_k_cluster_config import FINALIZER_WARPS

if TYPE_CHECKING:
    from ...runtime.config import Config
    from ..host_function import HostFunction
    from .split_k_cluster import ClusterKFacts
    from .split_k_workspace import SplitKWorkspaceProof


def _a_copy_coordinates(bk: int) -> str:
    # A warp copies 32 contiguous eight-element packets per iteration. If BK
    # divides that iteration stride, the K coordinate is loop invariant and
    # the row coordinate advances by a constant number of rows.
    if bk > 0 and bk % 8 == 0 and (32 * 8) % bk == 0:
        return (
            f"    m = lane // {bk // 8} + ai * {32 * 8 // bk}\n"
            f"    k = (lane % {bk // 8}) * 8"
        )
    return f"    flat = (lane + ai * 32) * 8\n    m = flat // {bk}\n    k = flat % {bk}"


def kernel_source(
    proof: SplitKWorkspaceProof, facts: ClusterKFacts, *, finalizer_warps: int = 1
) -> str:
    assert type(finalizer_warps) is int and finalizer_warps in FINALIZER_WARPS
    s = CLUSTER_K_SCHEDULE
    dtype = (
        "cutlass.Float16" if proof.lhs.dtype is torch.float16 else "cutlass.BFloat16"
    )
    output_type = (
        "cutlass.Float16" if proof.output_dtype is torch.float16 else "cutlass.Float32"
    )
    chunk = facts.k // s.cluster_ctas
    warp_chunk = chunk // s.k_warps
    waves = warp_chunk // s.bk
    bias_parameter = ", bias" if proof.bias is not None else ""
    bias_code = ""
    if proof.bias is not None:
        coordinates = ("tile_m + m", "tile_n + n")
        index = " + ".join(
            f"({coordinates[axis]}) * {stride}"
            for axis, stride in zip(
                proof.bias.dimensions, proof.bias.strides, strict=True
            )
        )
        indent = "    " if finalizer_warps == 4 else "            "
        bias_code = (
            f"{indent}if rank == 0:\n"
            f"{indent}    value = value + cutlass.Float32((bias.iterator + ({index})).load())\n"
        )
    # Both ownership maps retain the same left-associated local and rank sums.
    # Four warps each finalize one value per lane instead of serializing four
    # values in warp zero. Publication and final DSM retirement are unchanged.
    if finalizer_warps == 4:
        finalizer = f"""    m = lane // {s.bn} + warp * {s.bm // s.k_warps}
    n = lane % {s.bn}
    value = cutlass.Float32(c[m, n, 0])
    for local_peer in cutlass.range_constexpr(1, 4):
        value = value + cutlass.Float32(c[m, n, local_peer])
{bias_code}    c[m, n, 0] = value
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    if rank == 0:
        m = lane // {s.bn} + warp * {s.bm // s.k_warps}
        n = lane % {s.bn}
        element = cute.crd2idx((m, n, 0), c.layout)
        value = cutlass.Float32(c[m, n, 0])
        for peer in cutlass.range_constexpr(1, 8):
            value = value + load_shared_remote_f32(cp + element, cutlass.Int32(peer))
        (out.iterator + cute.crd2idx((tile_m + m, tile_n + n), out.layout)).store({output_type}(value))
"""
    else:
        finalizer = f"""    if warp == 0:
        for item in cutlass.range_constexpr(4):
            m = lane // 8 + item * 4
            n = lane % 8
            value = cutlass.Float32(c[m, n, 0])
            for local_peer in cutlass.range_constexpr(1, 4):
                value = value + cutlass.Float32(c[m, n, local_peer])
{bias_code}            c[m, n, 0] = value
    cute.arch.sync_threads()
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    if rank == 0 and warp == 0:
        for item in cutlass.range_constexpr(4):
            m = lane // 8 + item * 4
            n = lane % 8
            element = cute.crd2idx((m, n, 0), c.layout)
            value = cutlass.Float32(c[m, n, 0])
            for peer in cutlass.range_constexpr(1, 8):
                value = value + load_shared_remote_f32(cp + element, cutlass.Int32(peer))
            (out.iterator + cute.crd2idx((tile_m + m, tile_n + n), out.layout)).store({output_type}(value))
"""
    a_coordinates = _a_copy_coordinates(s.bk)
    copy_body = f"""
for ai in cutlass.range_constexpr({s.bm * s.bk // (32 * 8)}):
{a_coordinates}
    cute.arch.cp_async_shared_global(a.iterator + cute.crd2idx((m, k, warp), a.layout), x.iterator + (tile_m + m) * cutlass.Int32(x.layout.stride[0]) + (start_k + k) * cutlass.Int32(x.layout.stride[1]), 16, 'cg')
for bi in cutlass.range_constexpr({s.bn * s.bk // (32 * 8)}):
    flat = (lane + bi * 32) * 8
    n = flat % {s.bn}
    k = flat // {s.bn}
    cute.arch.cp_async_shared_global(b.iterator + cute.crd2idx((n, k, warp), b.layout), y.iterator + (start_k + k) * cutlass.Int32(y.layout.stride[0]) + (tile_n + n) * cutlass.Int32(y.layout.stride[1]), 16, 'cg')
cute.arch.cp_async_commit_group()
cute.arch.cp_async_wait_group(0)
cute.arch.sync_threads()
for ki in cutlass.range_constexpr({s.bk // 16}):
    cute.copy(ca, pa[None, None, ki, warp], da[None, None, 0])
    cute.copy(cb, pb[None, None, ki, warp], db[None, None, 0])
    cute.gemm(mma, acc, ra[None, None, 0], rb[None, None, 0], acc)
"""
    if waves > 1:
        # All CTAs/warps run the same number of waves. Retire old operand reads
        # before the next copy can reuse either slab. No barrier is added to
        # the accepted one-wave contraction.
        copy_body = (
            f"for wave in range({waves}):\n"
            f"    start_k = rank * {chunk} + warp * {warp_chunk} + wave * {s.bk}\n"
            + "\n".join("    " + line for line in copy_body.strip().splitlines())
            + "\n    cute.arch.sync_threads()\n"
        )
    body = "\n".join("    " + line for line in copy_body.strip().splitlines())
    return f"""from __future__ import annotations
import cutlass
import cutlass.cute as cute
from helion._compiler.cute.cluster_helpers import load_shared_remote_f32

@cute.kernel
def _split_k_cluster(x, y, out{bias_parameter}):
    pid = cutlass.Int32(cute.arch.block_idx()[0])
    rank = pid % {s.cluster_ctas}
    tile_n = (pid // {s.cluster_ctas} % {facts.n // s.bn}) * {s.bn}
    tile_m = (pid // {s.cluster_ctas * (facts.n // s.bn)}) * {s.bm}
    lane = cutlass.Int32(cute.arch.thread_idx()[0])
    warp = cutlass.Int32(cute.arch.thread_idx()[1])
    start_k = rank * {chunk} + warp * {warp_chunk}
    ap = cute.arch.alloc_smem({dtype}, {s.bm * s.bk * s.k_warps}, alignment=1024)
    bp = cute.arch.alloc_smem({dtype}, {s.bn * s.bk * s.k_warps}, alignment=1024)
    cp = cute.arch.alloc_smem(cutlass.Float32, {s.bm * s.bn * s.k_warps}, alignment=128)
    a = cute.make_tensor(ap, cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, cute.make_layout(({s.bm}, {s.bk}, {s.k_warps}), stride=({s.bk}, 1, {s.bm * s.bk}))))
    b = cute.make_tensor(bp, cute.make_composed_layout(cute.make_swizzle(0, 3, 3), 0, cute.make_layout(({s.bn}, {s.bk}, {s.k_warps}), stride=(1, {s.bn}, {s.bn * s.bk}))))
    c = cute.make_tensor(cp, cute.make_layout(({s.bm}, {s.bn}, {s.k_warps}), stride=({s.bn}, 1, {s.bm * s.bn})))
    mma = cute.make_tiled_mma(cute.nvgpu.warp.MmaF16BF16Op({dtype}, cutlass.Float32, (16, 8, 16)), atom_layout_mnk=(1, 1, 1), permutation_mnk=(16, 8, 16))
    thread = mma.get_slice(lane)
    acc = cute.make_rmem_tensor(mma.partition_shape_C(({s.bm}, {s.bn})), cutlass.Float32)
    ra = thread.make_fragment_A(thread.partition_shape_A(({s.bm}, 16)))
    rb = thread.make_fragment_B(thread.partition_shape_B(({s.bn}, 16)))
    ca = cute.make_tiled_copy_A(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), {dtype}), mma)
    cb = cute.make_tiled_copy_B(cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=2), {dtype}), mma)
    ta = ca.get_slice(lane)
    tb = cb.get_slice(lane)
    pa = ta.partition_S(a)
    pb = tb.partition_S(b)
    da = ta.retile(ra)
    db = tb.retile(rb)
    acc.fill(0.0)
{body}
    tc = thread.partition_C(c)
    cute.autovec_copy(acc, tc[None, None, None, warp])
    cute.arch.sync_threads()
{finalizer}    cute.arch.cluster_arrive_relaxed()
    cute.arch.cluster_wait()
_split_k_cluster._helion_cute_cluster_shape = ({s.cluster_ctas}, 1, 1)
"""


def generate_split_k_cluster(
    func: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.Module:
    from ..generate_ast import GenerateAST
    from ..generate_ast import emit_main_def

    env = CompileEnvironment.current()
    proof, facts = checked_cluster_proof(env, func.device_ir, config)
    if not all(
        tensor_has_specialized_tma_alignment(env, tensor)
        for tensor in (proof.lhs, proof.rhs)
    ):
        raise exc.BackendUnsupported("cute", "cluster split-K requires aligned inputs")
    shell = GenerateAST(
        func,
        config,
        codegen_graphs=[_empty_graph(info.graph_id) for info in func.device_ir.graphs],
    )
    marker = ast.Pass()
    with shell.device_function:
        for statement in func.body:
            if (
                isinstance(statement, ExtendedAST)
                and statement._loop_type is LoopType.GRID
            ):
                shell.add_statement(marker)
            else:
                shell.add_statement(shell.visit(statement))
    host = func.codegen_function_def(shell.host_statements)
    output_dtype = (
        "torch.float16" if proof.output_dtype is torch.float16 else "torch.float32"
    )
    allocations = 0
    for statement in host.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == proof.output_name
        ):
            allocation = statement.value
            assert isinstance(allocation, ast.Call)
            assert isinstance(allocation.func, ExtendedAST)
            assert isinstance(allocation.func._type_info, CallableType)
            factory = allocation.func._type_info.value
            assert factory in (torch.zeros, torch.zeros_like)
            lowered = ast.parse(ast.unparse(allocation), mode="eval").body
            assert isinstance(lowered, ast.Call)
            lowered.func = ast.parse(
                "torch.empty_like" if factory is torch.zeros_like else "torch.empty",
                mode="eval",
            ).body
            lowered.keywords = [
                keyword for keyword in lowered.keywords if keyword.arg != "dtype"
            ] + [
                ast.keyword(
                    arg="dtype", value=ast.parse(output_dtype, mode="eval").body
                )
            ]
            statement.value = lowered
            allocations += 1
        elif isinstance(statement, ast.Return):
            statement.value = ast.Name(id=proof.output_name, ctx=ast.Load())
    assert allocations == 1 and host.body.count(marker) == 1
    # Include the function binding and every Python parameter/local binder,
    # even unused positional-only arguments that do not appear as ast.Name.
    tables = [symtable.symtable(ast.unparse(host), "<helion split-K host>", "exec")]
    used: set[str] = set()
    while tables:
        table = tables.pop()
        used.update(table.get_identifiers())
        tables.extend(table.get_children())
    module_name = "_helion_cluster_module"
    while module_name in used:
        module_name += "_"
    finalizer_warps = config.config.get(FINALIZER_KEY, 1)
    assert type(finalizer_warps) is int
    source = kernel_source(proof, facts, finalizer_warps=finalizer_warps)
    kernel = f"{module_name}._split_k_cluster"
    s = CLUSTER_K_SCHEDULE
    grid = (facts.m // s.bm) * (facts.n // s.bn) * s.cluster_ctas
    bias_arg = f", {proof.bias.host_expression}" if proof.bias is not None else ""
    launch = statement_from_string(
        f"_launcher({kernel}, ({grid},), {proof.lhs_expression}, {proof.rhs_expression}, "
        f"{proof.output_name}{bias_arg}, block=(32, 4, 1))"
    )
    host.body[host.body.index(marker)] = launch
    metadata = {
        "schedule": CLUSTER8_K4,
        "carrier_tile": s.carrier_tile,
        "carrier_threads": (4, 8, 4),
        "tile": (s.bm, s.bn, s.bk),
        "block": (32, 4, 1),
        "cta_order": ("k", "n", "m"),
        "cluster": (s.cluster_ctas, 1, 1),
        "k_warps": s.k_warps,
        "chunk_k": facts.chunk_size(config),
        "waves": facts.k // (s.cluster_ctas * s.k_warps * s.bk),
        "shared_upper_bound": s.shared_upper_bound,
        "bias_partition": 0 if proof.bias is not None else None,
        "output_dtype": str(proof.output_dtype),
    }
    if finalizer_warps != 1:
        metadata["finalizer_warps"] = finalizer_warps
    dependencies = wrapper_source_dependencies(("split_k_cluster",))
    source_hash = (
        hashlib.sha256(repr((source, metadata, dependencies)).encode()).hexdigest()
        if dependencies is not None
        else None
    )
    result = ast.Module(
        body=[
            *func.codegen_imports(),
            statement_from_string("from torch._inductor.codecache import PyCodeCache"),
            statement_from_string(f"{module_name} = PyCodeCache.load({source!r})"),
            statement_from_string(
                f"{kernel}._helion_cute_source_hash = {source_hash!r}"
            ),
            host,
            statement_from_string(f"{func.name}._helion_cute_kernels = ({kernel},)"),
            statement_from_string(
                f"{func.name}._helion_cute_split_k_schedule = {metadata!r}"
            ),
            *(
                [func.codegen_call_function(), emit_main_def()]
                if emit_repro_caller
                else []
            ),
        ],
        type_ignores=[],
    )
    imports = {
        ast.unparse(statement)
        for statement in result.body
        if isinstance(statement, (ast.Import, ast.ImportFrom))
    }
    result.body[:0] = [
        statement_from_string(line)
        for line in get_needed_import_lines(result)
        if line not in imports
    ]
    return result

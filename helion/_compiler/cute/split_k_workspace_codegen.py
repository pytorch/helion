"""Emit a source-hashed native producer/reducer bundle for a proved split K.

The producer is compiled from a private batched-GEMM template using the existing
native lowering. No user function is retraced and no nested BoundKernel cache is
created. The outer host program keeps its original metadata and assertions.
"""

from __future__ import annotations

import ast
import hashlib
import symtable
from typing import TYPE_CHECKING
from typing import cast

import torch

from ... import exc
from ... import language as hl
from ...runtime.config import Config
from ..ast_extension import ExtendedAST
from ..ast_extension import LoopType
from ..ast_extension import statement_from_string
from ..compile_environment import CompileEnvironment
from ..inductor_lowering_extra import patch_inductor_lowerings
from ..kernel_compiler import KernelCompiler
from ..output_header import get_needed_import_lines
from ..type_info import CallableType
from ..variable_origin import ArgumentOrigin
from .cute_mma import analyze_cute_mma_node
from .materialized_fission_codegen import _empty_graph
from .materialized_fission_codegen import _module_source
from .memory_ops import tensor_has_specialized_tma_alignment
from .pipeline_smem import analyze_pipeline_smem_facts
from .split_k_workspace import analyze_split_k_workspace
from .split_k_workspace import workspace_schedule
from .tcgen05_config import CuteTcgen05Config

if TYPE_CHECKING:
    from ..host_function import HostFunction
    from .split_k_workspace import SplitKWorkspaceProof
    from .split_k_workspace import SplitKWorkspaceSchedule


def _native_partition_gemm(
    a: torch.Tensor, b: torch.Tensor, partials: torch.Tensor
) -> torch.Tensor:
    partitions, rows, inner_size = a.shape
    columns = b.size(2)
    for partition, row, column in hl.tile(
        (partitions, rows, columns), block_size=(1, None, None)
    ):
        accumulator = hl.zeros([partition, row, column], dtype=torch.float32)
        for inner in hl.tile(inner_size):
            accumulator = torch.baddbmm(
                accumulator, a[partition, row, inner], b[partition, inner, column]
            )
        partials[partition, row, column] = accumulator
    return partials


def _producer_source(
    proof: SplitKWorkspaceProof,
    schedule: SplitKWorkspaceSchedule,
    owner: CompileEnvironment,
    capacity: int,
) -> tuple[str, list[dict[str, object]]]:
    """Compile metadata-only batch views after proving the owning inputs."""
    from ...runtime.kernel import _maybe_skip_dtype_check_in_meta_registrations

    child = CompileEnvironment(
        owner.device,
        owner.settings.copy(static_shapes=True),
        index_dtype=owner.index_dtype,
        is_distributed=False,
    )
    child.config_spec.num_sm = owner.config_spec.num_sm
    partitions, chunk = schedule.partitions, schedule.chunk
    metadata = (
        (
            (partitions, proof.m, chunk),
            (chunk * proof.lhs_strides[1], *proof.lhs_strides),
            proof.lhs.dtype,
        ),
        (
            (partitions, chunk, proof.n),
            (chunk * proof.rhs_strides[0], *proof.rhs_strides),
            proof.rhs.dtype,
        ),
        (
            (partitions, proof.m, proof.n),
            (proof.m * proof.n, proof.n, 1),
            torch.float32,
        ),
    )
    with owner.suspend(), child:
        arguments = [
            torch.empty_strided(shape, strides, dtype=dtype, device=child.device)
            for shape, strides, dtype in metadata
        ]
        for name, tensor in zip(("a", "b", "partials"), arguments, strict=True):
            child.input_sources[tensor] = ArgumentOrigin(name).to_source()
        # A/B inherit the cache-key-backed base/stride proof checked below;
        # the full chunk is BK-aligned, so every batch offset is 16B-aligned.
        # Partials are a new dense allocation in the outer host function.
        child.cute_proven_tma_inputs.update(cast("list[torch.Tensor]", arguments))
        compiler = KernelCompiler(child)
        with (
            _maybe_skip_dtype_check_in_meta_registrations(),
            patch_inductor_lowerings(),
        ):
            host = compiler.compile(_native_partition_gemm, list(arguments), {})
        with host:
            candidates = [
                (node, candidate)
                for info in host.device_ir.graphs
                for node in info.graph.nodes
                if (candidate := analyze_cute_mma_node(node, device_ir=host.device_ir))
                is not None
            ]
            if len(candidates) != 1:
                raise exc.BackendUnsupported(
                    "cute", "native partition producer proof declined"
                )
            node, candidate = candidates[0]
            facts = analyze_pipeline_smem_facts(
                candidate,
                node,
                host.device_ir.graphs,
                capacity_bytes=capacity,
                allow_leading_passthrough=True,
            )
            if facts is None:
                raise exc.BackendUnsupported(
                    "cute", "native partition producer SMEM proof declined"
                )
            child.config_spec._cute_tcgen05_config.register_pipeline_smem_facts(facts)
            config = child.config_spec.normalized_config(
                Config(
                    block_sizes=[schedule.bm, schedule.bn, schedule.bk],
                    tcgen05_cta_group="two",
                    tcgen05_cluster_m=2,
                    tcgen05_cluster_n=1,
                    tcgen05_ab_stages=schedule.stages,
                    tcgen05_acc_stages=2,
                    tcgen05_c_stages=2,
                    pid_type="persistent_blocked",
                )
            )
            from ..generate_ast import generate_ast

            module = generate_ast(host, config, False)
            return _module_source(module), child.cute_resolved_wrapper_plans


def _reducer_source(
    proof: SplitKWorkspaceProof, schedule: SplitKWorkspaceSchedule
) -> str:
    """Four disjoint partition lanes per output vector and one final cast.

    Every partial is loaded exactly once. The two butterfly additions combine
    those four FP32 lane sums; only their owner writes the output. The producer
    and reducer are ordered launches on the same stream, including graph replay.
    """
    bias_parameter = ", bias" if proof.bias is not None else ""
    bias_code = ""
    if proof.bias is not None:
        coordinates = (
            f"(output_begin + element) // {proof.n}",
            f"(output_begin + element) % {proof.n}",
        )
        index = " + ".join(
            f"({coordinates[dimension]}) * {stride}"
            for dimension, stride in zip(
                proof.bias.dimensions, proof.bias.strides, strict=True
            )
        )
        bias_code = (
            "            if partition == cutlass.Int32(0):\n"
            "                for element in cutlass.range_constexpr(4):\n"
            f"                    bias_scalar = cute.make_tensor(bias.iterator + ({index}), cute.make_layout(1))\n"
            "                    bias_value = cutlass.Float32(bias_scalar[0])\n"
            "                    loaded[element] = loaded[element] + bias_value\n"
        )
    bits = proof.output_dtype.itemsize * 8
    output_type = (
        "cutlass.Float16" if proof.output_dtype is torch.float16 else "cutlass.Float32"
    )
    return (
        "from __future__ import annotations\n"
        "import cutlass\nimport cutlass.cute as cute\n"
        "@cute.kernel\n"
        f"def _reduce_partitions(partials, out{bias_parameter}):\n"
        "    thread = cutlass.Int32(cute.arch.thread_idx()[0])\n"
        "    lane = thread % cutlass.Int32(32)\n"
        "    warp = thread // cutlass.Int32(32)\n"
        "    partition_lane = lane // cutlass.Int32(8)\n"
        "    output_begin = cute.assume(cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(128) + warp * cutlass.Int32(32) + (lane % cutlass.Int32(8)) * cutlass.Int32(4), divby=4)\n"
        "    values = cute.make_rmem_tensor((4,), cutlass.Float32)\n"
        "    values.fill(cutlass.Float32(0))\n"
        "    loaded = cute.make_rmem_tensor((4,), cutlass.Float32)\n"
        "    load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)\n"
        f"    for step in cutlass.range_constexpr({(schedule.partitions + 3) // 4}):\n"
        "        partition = partition_lane + cutlass.Int32(step * 4)\n"
        f"        if partition < cutlass.Int32({schedule.partitions}) and output_begin < cutlass.Int32({proof.m * proof.n}):\n"
        f"            source_offset = cute.assume(partition * cutlass.Int32({proof.m * proof.n}) + output_begin, divby=4)\n"
        "            source = cute.make_tensor(partials.iterator + source_offset, cute.make_layout((4,), stride=(1,)))\n"
        "            cute.copy(load_atom, source, loaded)\n"
        + bias_code
        + "            values.store(values.load() + loaded.load())\n"
        "    for element in cutlass.range_constexpr(4):\n"
        "        value = values[element]\n"
        "        value = value + cute.arch.shuffle_sync_bfly(value, 8)\n"
        "        value = value + cute.arch.shuffle_sync_bfly(value, 16)\n"
        "        values[element] = value\n"
        f"    if partition_lane == cutlass.Int32(0) and output_begin < cutlass.Int32({proof.m * proof.n}):\n"
        f"        converted = cute.make_rmem_tensor((4,), {output_type})\n"
        f"        converted.store(values.load().to({output_type}))\n"
        "        destination = cute.make_tensor(out.iterator + output_begin, cute.make_layout((4,), stride=(1,)))\n"
        f"        store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {output_type}, num_bits_per_copy={bits * 4})\n"
        "        cute.copy(store_atom, converted, destination)\n"
    )


def generate_split_k_workspace(
    func: HostFunction, config: Config, emit_repro_caller: bool
) -> ast.Module:
    from ..generate_ast import GenerateAST
    from ..generate_ast import emit_main_def

    env = CompileEnvironment.current()
    proof = analyze_split_k_workspace(env, func.device_ir)
    if proof is None:
        raise exc.BackendUnsupported(
            "cute", "private FP32 split-K workspace proof declined"
        )
    capacity = CuteTcgen05Config.per_cta_smem_capacity_bytes(env.device)
    schedule = workspace_schedule(proof, env, config, capacity_bytes=capacity)
    if schedule is None:
        raise exc.BackendUnsupported(
            "cute",
            "native split-K workspace requires full aligned partitions and a feasible paired pipeline",
        )
    if not all(
        tensor_has_specialized_tma_alignment(env, tensor)
        for tensor in (proof.lhs, proof.rhs)
    ):
        raise exc.BackendUnsupported(
            "cute", "native split-K workspace inputs lack TensorMap alignment proof"
        )
    # Retain ordinary host AST lowering, scalar bindings and assertions. The
    # original device root is replaced before any device or launcher emission.
    shell = GenerateAST(
        func,
        config,
        codegen_graphs=[_empty_graph(info.graph_id) for info in func.device_ir.graphs],
    )
    launch_marker = ast.Pass()
    roots = 0
    with shell.device_function:
        for statement in func.body:
            if (
                isinstance(statement, ExtendedAST)
                and statement._loop_type is LoopType.GRID
            ):
                shell.add_statement(launch_marker)
                roots += 1
            else:
                shell.add_statement(shell.visit(statement))
    host = func.codegen_function_def(shell.host_statements)
    if roots != 1:
        raise exc.BackendUnsupported(
            "cute", "split-K workspace requires one ordered host launch"
        )
    native_source, plans = _producer_source(proof, schedule, env, capacity)
    reduce_source = _reducer_source(proof, schedule)
    # Include the function binding and every Python parameter/local binder,
    # even unused positional-only arguments that do not appear as ast.Name.
    tables = [symtable.symtable(ast.unparse(host), "<helion split-K host>", "exec")]
    used: set[str] = set()
    while tables:
        table = tables.pop()
        used.update(table.get_identifiers())
        tables.extend(table.get_children())

    def unique(name: str) -> str:
        while name in used:
            name += "_"
        used.add(name)
        return name

    native_module = unique("_helion_split_k_native")
    reduce_module = unique("_helion_split_k_reduce")
    partials = unique("_helion_split_k_partials")
    view_a = unique("_helion_split_k_a")
    view_b = unique("_helion_split_k_b")
    output_dtype = (
        "torch.float16" if proof.output_dtype is torch.float16 else "torch.float32"
    )
    allocation_count = 0
    for stmt in host.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == proof.output_name
        ):
            allocation = stmt.value
            assert isinstance(allocation, ast.Call)
            assert isinstance(allocation.func, ExtendedAST)
            assert isinstance(allocation.func._type_info, CallableType)
            factory = allocation.func._type_info.value
            assert factory in (torch.zeros, torch.zeros_like)
            lowered_allocation = ast.parse(ast.unparse(allocation), mode="eval").body
            assert isinstance(lowered_allocation, ast.Call)
            # Keep the original allocation's host expressions and evaluation
            # point. A matrix input may be a view bound later in the prelude.
            stmt.value = ast.Call(
                func=ast.parse(
                    "torch.empty_like"
                    if factory is torch.zeros_like
                    else "torch.empty",
                    mode="eval",
                ).body,
                args=lowered_allocation.args,
                keywords=[
                    *(
                        keyword
                        for keyword in lowered_allocation.keywords
                        if keyword.arg != "dtype"
                    ),
                    ast.keyword(
                        arg="dtype", value=ast.parse(output_dtype, mode="eval").body
                    ),
                ],
            )
            allocation_count += 1
        elif isinstance(stmt, ast.Return):
            stmt.value = ast.Name(id=proof.output_name, ctx=ast.Load())
    assert allocation_count == 1
    p, chunk = schedule.partitions, schedule.chunk
    a_stride = (chunk * proof.lhs_strides[1], *proof.lhs_strides)
    b_stride = (chunk * proof.rhs_strides[0], *proof.rhs_strides)
    new_calls = ast.parse(
        f"{partials} = torch.empty(({p}, {proof.m}, {proof.n}), dtype=torch.float32, device=({proof.lhs_expression}).device)\n"
        f"{view_a} = ({proof.lhs_expression}).as_strided(({p}, {proof.m}, {chunk}), {a_stride!r})\n"
        f"{view_b} = ({proof.rhs_expression}).as_strided(({p}, {chunk}, {proof.n}), {b_stride!r})\n"
        f"{native_module}._native_partition_gemm({view_a}, {view_b}, {partials}, _launcher=_launcher)\n"
        f"_launcher({reduce_module}._reduce_partitions, ({(proof.m * proof.n + 127) // 128},), {partials}, {proof.output_name}"
        + (f", {proof.bias.host_expression}" if proof.bias is not None else "")
        + ", block=(128, 1, 1))\n"
    ).body
    index = host.body.index(launch_marker)
    host.body[index : index + 1] = new_calls
    native_kernel = f"{native_module}._helion__native_partition_gemm"
    reduce_kernel = f"{reduce_module}._reduce_partitions"
    loads = [statement_from_string("from torch._inductor.codecache import PyCodeCache")]
    for module_name, kernel, source in (
        (native_module, native_kernel, native_source),
        (reduce_module, reduce_kernel, reduce_source),
    ):
        loads.extend(
            (
                statement_from_string(f"{module_name} = PyCodeCache.load({source!r})"),
                statement_from_string(
                    f"{kernel}._helion_cute_source_hash = {hashlib.sha256(source.encode()).hexdigest()!r}"
                ),
            )
        )
    metadata = {
        "partitions": p,
        "chunk_k": chunk,
        "tile": (schedule.bm, schedule.bn, schedule.bk),
        "ab_stages": schedule.stages,
        "partial_dtype": "torch.float32",
        "output_dtype": str(proof.output_dtype),
        "bias_partition": 0 if proof.bias is not None else None,
    }
    result = ast.Module(
        body=[
            *func.codegen_imports(),
            *loads,
            host,
            statement_from_string(
                f"{func.name}._helion_cute_kernels = ({native_kernel}, {reduce_kernel})"
            ),
            statement_from_string(
                f"{func.name}._helion_cute_split_k_workspace = {metadata!r}"
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
        ast.unparse(stmt)
        for stmt in result.body
        if isinstance(stmt, (ast.Import, ast.ImportFrom))
    }
    result.body[:0] = [
        statement_from_string(line)
        for line in get_needed_import_lines(result)
        if line not in imports
    ]
    env.cute_resolved_wrapper_plans = plans
    return result

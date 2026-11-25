"""Emit a source-hashed native producer/reducer bundle for a proved split K.

The producer is compiled from a private batched-GEMM template using the existing
native lowering. No user function is retraced and no nested BoundKernel cache is
created. The outer host program keeps its original metadata and assertions.
"""

from __future__ import annotations

import ast
import copy
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
from ..type_info import CallableType
from ..variable_origin import ArgumentOrigin
from . import target_facts
from ._ast_pass_utils import assigned_name
from .compile_options import CONFIG_COMPILE_OPTION_KEYS
from .compile_options import compile_options_keyword
from .compile_options import config_compile_options
from .cute_mma import analyze_cute_mma_node
from .host_zero_fill_launches import zero_fill_kernel_launch
from .materialized_fission_codegen import _empty_graph
from .materialized_fission_codegen import _module_source
from .materialized_fission_codegen import prepend_missing_imports
from .memory_ops import tensor_has_specialized_tma_alignment
from .pdl_release import ZERO_FILL_KERNEL_PREFIX
from .pipeline_smem import analyze_pipeline_smem_facts
from .split_k_workspace import analyze_split_k_workspace
from .split_k_workspace import atomic_epilogue_geometry
from .split_k_workspace import cutlass_dtype_name
from .split_k_workspace import torch_dtype_name
from .split_k_workspace import workspace_schedule
from .state import cute_env_state
from .tcgen05_config import facts

if TYPE_CHECKING:
    from ..host_function import HostFunction
    from .split_k_workspace import AtomicEpilogueGeometry
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
    owner_config: Config,
) -> tuple[str, list[dict[str, object]]]:
    """Compile metadata-only batch views after proving the owning inputs."""
    from ...runtime.kernel import maybe_skip_dtype_check_in_meta_registrations

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
        child_state = cute_env_state(child)
        child_state.proven_tma_inputs.update(cast("list[torch.Tensor]", arguments))
        compiler = KernelCompiler(child)
        with (
            maybe_skip_dtype_check_in_meta_registrations(),
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
            pipeline_facts = analyze_pipeline_smem_facts(
                candidate,
                node,
                host.device_ir.graphs,
                capacity_bytes=capacity,
                allow_leading_passthrough=True,
            )
            if pipeline_facts is None:
                raise exc.BackendUnsupported(
                    "cute", "native partition producer SMEM proof declined"
                )
            facts.register_pipeline_smem_facts(
                child.config_spec._cute_tcgen05_config, pipeline_facts
            )
            producer_config = Config(
                block_sizes=[schedule.bm, schedule.bn, schedule.bk],
                tcgen05_cta_group="two",
                tcgen05_cluster_m=2,
                tcgen05_cluster_n=1,
                tcgen05_ab_stages=schedule.stages,
                tcgen05_acc_stages=2,
                tcgen05_c_stages=2,
                pid_type="persistent_blocked",
            )
            for key in CONFIG_COMPILE_OPTION_KEYS:
                if key in owner_config:
                    producer_config.config[key] = owner_config[key]
            config = child.config_spec.normalized_config(producer_config)
            from ..generate_ast import generate_ast

            module = generate_ast(host, config, False)
            plans = child_state.resolved_wrapper_plans
            if schedule.k_offset:
                _offset_partition_k_coordinates(module, schedule, plans)
            units = (
                schedule.partitions
                * -(-proof.m // schedule.bm)
                * -(-proof.n // schedule.bn)
            )
            if units > max(1, (child.config_spec.num_sm or 0) // 2):
                _order_partition_units_output_major(module, schedule, proof)
            if schedule.atomic_epilogue:
                plans = _rewrite_producer_for_atomic_epilogue(
                    module, proof, schedule, plans
                )
            return _module_source(module), plans


def _offset_partition_k_coordinates(
    module: ast.Module,
    schedule: SplitKWorkspaceSchedule,
    plans: list[dict[str, object]],
) -> None:
    """Address each partition's K range on the operands as they are.

    The producer is compiled over batch views ``a[p, m, k]`` (a ``[P, M,
    chunk]`` view of ``[M, K]``), whose partition coordinate its TMA loads
    take as a third TensorMap mode:

        gA_tma = cute.local_tile(tma_tensor_a, (128, 64, 1), (m_tile, None, cutlass.Int32(indices_0)))

    Where ``P * chunk`` exceeds K (``plain_store_chunk``: K = 20008 in three
    partitions of 6720) no strided view holds the last partition: its rows
    would run into the next row's first columns instead of zeros. The host
    passes ``[1, M, K]`` and ``[1, K, N]`` views instead, and each
    partition's loads shift their K coordinate by ``partition * chunk``:

        gA_tma = cute.local_tile(cute.domain_offset((0, cutlass.Int32(indices_0) * 6720, 0), tma_tensor_a), (128, 64, 1), (m_tile, None, 0))

    so the TensorMap's own K extent zero-fills the 152 columns of the last
    partition past K.
    """
    kernel_args: list[object] = []
    for plan in plans:
        if plan.get("kind") == "tcgen05_ab_tma":
            kernel_args = cast("list[object]", plan["kernel_args"])
    if len(kernel_args) != 4:
        raise exc.BackendUnsupported(
            "cute", "split-K K-offset partitions expect the A/B TensorMap producer"
        )
    tensors = {str(kernel_args[1]), str(kernel_args[3])}
    rewritten: list[str] = []
    for node in ast.walk(module):
        if not (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "cute.local_tile"
            and len(node.args) == 3
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in tensors
            and isinstance(node.args[2], ast.Tuple)
            and len(node.args[2].elts) == 3
        ):
            continue
        tensor = node.args[0].id
        m_or_n, k, partition = node.args[2].elts
        if not (isinstance(k, ast.Constant) and k.value is None):
            raise exc.BackendUnsupported(
                "cute", "split-K K-offset partitions expect whole-K operand tiles"
            )
        node.args[0] = ast.parse(
            f"cute.domain_offset((0, ({ast.unparse(partition)}) * {schedule.chunk}, 0), {tensor})",
            mode="eval",
        ).body
        node.args[2] = ast.Tuple(
            elts=[m_or_n, k, ast.Constant(value=0)], ctx=ast.Load()
        )
        rewritten.append(tensor)
    if sorted(rewritten) != sorted(tensors):
        raise exc.BackendUnsupported(
            "cute", "split-K K-offset partitions expect one A and one B operand tile"
        )
    ast.fix_missing_locations(module)


def _order_partition_units_output_major(
    module: ast.Module, schedule: SplitKWorkspaceSchedule, proof: SplitKWorkspaceProof
) -> None:
    """Hand out a multi-wave producer's work units partition by partition.

    The batched producer decodes its persistent scheduler's linear unit index
    partition-fastest:

        virtual_pid = (work_tile.tile_idx[0] >> 1) + work_tile.tile_idx[1] * 3 + ...
        pid_0 = virtual_pid % num_blocks_0      # partition
        pid_1 = virtual_pid // num_blocks_0 & cutlass.Int32(15)  # M tile
        pid_2 = virtual_pid // (num_blocks_0 * num_blocks_1)     # N tile

    so the units running together share no operand rows: bf16 4096 x 1536 x
    128104 in three partitions of 256x256 tiles runs 76 of its 288 units at
    once, three K slices of 16 M tiles over one N tile, and reads its 1 GB A
    once per N tile, six times in all. Remapping the index first,

        virtual_pid = virtual_pid // 96 + 3 * (virtual_pid // 6 % 16 + 16 * (virtual_pid % 6))

    hands out partition 0's 96 output tiles N-fastest, then partition 1's:
    the six N tiles of an M tile read their A slice together (GB300, one CUDA
    graph over cold inputs, of cuBLAS: 1.029 -> 1.068; 8192 x 768 x 50272 in
    three partitions 0.902 -> 1.027). A producer of one wave or less (every
    unit of its own CTA pair) keeps the decode.
    """
    partitions = schedule.partitions
    m_tiles = -(-proof.m // schedule.bm)
    n_tiles = -(-proof.n // schedule.bn)
    remap = ast.parse(
        f"virtual_pid = virtual_pid // {m_tiles * n_tiles} + {partitions} * "
        f"(virtual_pid // {n_tiles} % {m_tiles} + {m_tiles} * (virtual_pid % {n_tiles}))"
    ).body[0]
    remapped = 0
    for statements in _kernel_statement_lists(module):
        index = 0
        while index < len(statements):
            if assigned_name(statements[index]) == "virtual_pid":
                statements.insert(index + 1, copy.deepcopy(remap))
                remapped += 1
                index += 1
            index += 1
    if remapped == 0:
        raise exc.BackendUnsupported(
            "cute", "split-K producer ordering expects the persistent unit decode"
        )
    ast.fix_missing_locations(module)


# Threads per reducer CTA; each owns one 4-wide output vector.
REDUCER_THREADS = 256
# The partitions a reducer thread loads before it folds them into its total.
# One sum over every partition,
#
#     converted.store((loaded_0.load() + ... + loaded_127.load()).to(...))
#
# keeps 4 * P loaded floats live: 512 at the 128 partitions the atomic split-K
# proofs seed, past the 255 registers a thread has. Groups of eight keep 32.
REDUCER_GROUP = 8


def _reducer_grid(outputs: int) -> int:
    """CTAs of the reducer launch over ``outputs`` elements (``_reducer_source``)."""
    return -(-(outputs // 4) // REDUCER_THREADS)


def _reducer_source(
    proof: SplitKWorkspaceProof, schedule: SplitKWorkspaceSchedule
) -> str:
    """The reducer launch: a thread owns one 4-wide output vector, issues the
    loads of its partitions before the first addition and stores their sum in
    the output's dtype. For a 768 x 768 bf16 output of two partitions (147456
    vectors, 576 CTAs of 256 threads):

        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()
        vector = block * cutlass.Int32(256) + thread
        output_begin = cute.assume(vector * cutlass.Int32(4), divby=4)
        cute.copy(load_atom, <partials + 0 + output_begin>, loaded_0)
        cute.copy(load_atom, <partials + 589824 + output_begin>, loaded_1)
        converted.store((loaded_0.load() + loaded_1.load()).to(cutlass.BFloat16))
        cute.copy(store_atom, converted, <out + output_begin>)

    The partitions are summed in order, the bias (if any) added to partition
    zero first: the fp32 rounding order of the original epilogue.

    It triggers its dependents before it waits for the producer: the next
    kernel of the stream launches once every reducer CTA is resident, so its
    prologue runs under the reduction, and its own wait still orders it after
    the reducer. The schedule on GB300 (CUDA graph over cold inputs, two runs
    each), the trigger after the wait instead: bf16 768^2 x 8192 in two
    partitions 11.28-11.37 us against 10.83-10.94 with it first, 384 x 768 x
    16384 in four 12.92 against 12.55-12.62.

    One vector a thread keeps the grid wide. More vectors a thread (all of
    their loads in flight together, on fewer CTAs) read 768^2 x 8192 at
    12.06-12.13 us for two vectors, 13.74-13.88 for four and 15.50-15.75 for
    eight; spreading one vector's partitions over four lanes (each loading
    partitions lane, lane + 4, ... and butterfly shuffles combining them)
    issues one load a thread on a 4608-CTA grid: [4, 768, 768] fp32 partials
    alone (L2-hot, one launch included) reduce in 3.11 us that way, 1.83 us
    this way."""
    bias_parameter = ", bias" if proof.bias is not None else ""
    outputs = proof.m * proof.n
    vectors = outputs // 4
    ragged = vectors % REDUCER_THREADS != 0
    bits = proof.output_dtype.itemsize * 8
    output_type = cutlass_dtype_name(proof.output_dtype)
    lines = [
        "from __future__ import annotations",
        "import cutlass",
        "import cutlass.cute as cute",
        "@cute.kernel",
        f"def _reduce_partitions(partials, out{bias_parameter}):",
        "    cute.arch.griddepcontrol_launch_dependents()",
        "    cute.arch.griddepcontrol_wait()",
        "    thread = cutlass.Int32(cute.arch.thread_idx()[0])",
        "    block = cutlass.Int32(cute.arch.block_idx()[0])",
        "    load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=128)",
        f"    vector = block * cutlass.Int32({REDUCER_THREADS}) + thread",
        "    output_begin = cute.assume(vector * cutlass.Int32(4), divby=4)",
        *(
            f"    loaded_{q} = cute.make_rmem_tensor((4,), cutlass.Float32)"
            for q in range(schedule.partitions)
        ),
    ]
    indent = "    "
    if ragged:
        lines.append(f"    if vector < cutlass.Int32({vectors}):")
        indent = "        "
    groups = [
        range(start, min(start + REDUCER_GROUP, schedule.partitions))
        for start in range(0, schedule.partitions, REDUCER_GROUP)
    ]

    def loads(group: range) -> list[str]:
        return [
            f"{indent}cute.copy(load_atom, cute.make_tensor(partials.iterator + cute.assume(cutlass.Int32({q * outputs}) + output_begin, divby=4), cute.make_layout((4,), stride=(1,))), loaded_{q})"
            for q in group
        ]

    lines.extend(loads(groups[0]))
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
        lines.extend(
            (
                f"{indent}for element in cutlass.range_constexpr(4):",
                f"{indent}    bias_scalar = cute.make_tensor(bias.iterator + ({index}), cute.make_layout(1))",
                f"{indent}    loaded_0[element] = loaded_0[element] + cutlass.Float32(bias_scalar[0])",
            )
        )
    total = " + ".join(f"loaded_{q}.load()" for q in groups[0])
    if len(groups) > 1:
        # Past REDUCER_GROUP partitions each group folds into the running
        # total, in partition order (the same sum as one expression), before
        # the next group's loads issue:
        #
        #     total = loaded_0.load() + ... + loaded_7.load()
        #     cute.copy(load_atom, <partials + 8 * outputs + output_begin>, loaded_8)
        #     ...
        #     total = total + loaded_8.load() + ... + loaded_15.load()
        lines.append(f"{indent}total = {total}")
        for group in groups[1:]:
            lines.extend(loads(group))
            lines.append(
                f"{indent}total = total + "
                + " + ".join(f"loaded_{q}.load()" for q in group)
            )
        total = "total"
    lines.extend(
        (
            f"{indent}converted = cute.make_rmem_tensor((4,), {output_type})",
            f"{indent}converted.store(({total}).to({output_type}))",
            f"{indent}store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), {output_type}, num_bits_per_copy={bits * 4})",
            f"{indent}cute.copy(store_atom, converted, cute.make_tensor(out.iterator + output_begin, cute.make_layout((4,), stride=(1,))))",
        )
    )
    return "\n".join(lines) + "\n"


_RED_HELPERS = {
    torch.float16: "red_add_f16_vec",
    torch.bfloat16: "red_add_bf16_vec",
    torch.float32: "red_add_f32_vec",
}


def _kernel_statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
    """Every statement list nested in ``node`` (bodies and else branches)."""
    found: list[list[ast.stmt]] = []
    for child in ast.walk(node):
        for field in ("body", "orelse", "finalbody"):
            statements = getattr(child, field, None)
            if (
                isinstance(statements, list)
                and statements
                and isinstance(statements[0], ast.stmt)
            ):
                found.append(statements)
    return found


def _rewrite_producer_for_atomic_epilogue(
    module: ast.Module,
    proof: SplitKWorkspaceProof,
    schedule: SplitKWorkspaceSchedule,
    plans: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Turn the generated partition producer's TMA-store epilogue into packed
    atomic adds of the staged tile into the output.

    The native lowering stages each epilogue subtile of the fp32 accumulator
    in shared memory and has warp 0 issue one TMA store per subtile into the
    partials; here every epilogue thread instead reads back
    ``width`` consecutive values of a row (bias added on partition zero,
    hoisted when a lane's column is fixed) and issues one packed relaxed
    ``red`` into the output, whose coordinates come from an identity tensor
    partitioned exactly like the stored tile. The TMA-store descriptor, its
    kernel parameters, its prefetch and its wrapper plan disappear; the raw
    ``partials`` parameter becomes ``out`` (``bias`` follows it). The TMA
    warp's dependent-launch wait goes: the output is zeroed by a fill kernel
    that waits for every earlier kernel of the stream before releasing this
    one and writes only the output, so the operand loads may start under it
    while the epilogue warps keep their wait ahead of the first atomic.
    Returns the wrapper plans without the store plan.
    """
    kernels = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_helion__native_partition_gemm"
    ]
    wrappers = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_native_partition_gemm"
    ]
    if len(kernels) != 1 or len(wrappers) != 1:
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue expects one producer kernel"
        )
    kernel, wrapper = kernels[0], wrappers[0]
    geometry = _emitted_epilogue_geometry(kernel, proof, schedule)
    store_params = ("tcgen05_tma_store_atom", "tcgen05_tma_store_tensor")
    names = [argument.arg for argument in kernel.args.args]
    if (
        names[:3] != ["a", "b", "partials"]
        or names[-2:] != list(store_params)
        or kernel.args.kwonlyargs
    ):
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue expects the TMA-store producer form"
        )
    raw = ["a", "b", "out"] + (["bias"] if proof.bias is not None else [])
    kernel.args.args = [ast.arg(arg=name) for name in raw] + kernel.args.args[3:-2]

    def kernel_source_text(node: ast.AST) -> str:
        return ast.unparse(node)

    # The statements the TMA store needed and nothing else does.
    removed_prefetch = removed_epi = removed_partition = removed_gd = 0
    rewrote_tile = 0
    for statements in _kernel_statement_lists(kernel):
        kept: list[ast.stmt] = []
        for statement in statements:
            text = kernel_source_text(statement)
            if text == "cute.nvgpu.cpasync.prefetch_descriptor(tcgen05_tma_store_atom)":
                removed_prefetch += 1
                continue
            name = assigned_name(statement)
            if name == "tcgen05_tCgC_epi":
                removed_epi += 1
                continue
            if name == "tcgen05_bSG_gD":
                removed_gd += 1
                continue
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Tuple)
                and [
                    element.id
                    for element in statement.targets[0].elts
                    if isinstance(element, ast.Name)
                ]
                == ["tcgen05_bSG_sD", "tcgen05_bSG_gD_partitioned"]
            ):
                removed_partition += 1
                continue
            if (
                name == "tcgen05_gC3d"
                and isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Call)
                and statement.value.args
                and isinstance(statement.value.args[0], ast.Name)
                and statement.value.args[0].id == "tcgen05_tma_store_tensor"
            ):
                # Coordinates (m, n, partition) of the pair tile's elements in
                # the output, partitioned like the stored tile below.
                statement.value.args[0] = ast.parse(
                    f"cute.make_identity_tensor(({proof.m}, {proof.n}, {schedule.partitions}))",
                    mode="eval",
                ).body
                rewrote_tile += 1
            kept.append(statement)
        if not kept:
            # The store descriptor's prefetch was the branch's only statement.
            kept.append(ast.Pass())
        statements[:] = kept
    if (removed_prefetch, removed_epi, removed_partition, removed_gd, rewrote_tile) != (
        1,
        1,
        1,
        2,
        1,
    ):
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue could not retarget the producer epilogue"
        )
    # The store itself: ``if tcgen05_warp_idx == cutlass.Int32(0): cute.copy(
    # tcgen05_tma_store_atom, ...); tcgen05_c_pipeline.producer_commit()``.
    replaced_store = 0
    for statements in _kernel_statement_lists(kernel):
        stores = [
            index
            for index, statement in enumerate(statements)
            if isinstance(statement, ast.If)
            and kernel_source_text(statement.test)
            == "tcgen05_warp_idx == cutlass.Int32(0)"
            and any(
                "cute.copy(tcgen05_tma_store_atom" in kernel_source_text(inner)
                for inner in statement.body
            )
        ]
        for index in reversed(stores):
            statements[index : index + 1] = ast.parse(
                _atomic_epilogue_statements(proof, geometry)
            ).body
            replaced_store += 1
    if replaced_store != 1:
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue found no tile store to replace"
        )
    _insert_after_assignment(
        kernel, "tcgen05_store_epi_tile", _tiler_check_statement(geometry)
    )
    # The TMA warp's wait: the fill kernel ahead of this launch is the barrier.
    tma_branches = [
        statement
        for statement in kernel.body
        if isinstance(statement, ast.If)
        and kernel_source_text(statement.test)
        == "cute.arch.make_warp_uniform(cute.arch.warp_idx()) == cutlass.Int32(5)"
    ]
    removed_wait = 0
    for branch in tma_branches:
        for statements in _kernel_statement_lists(branch):
            for index, statement in enumerate(statements):
                if kernel_source_text(statement) == "cute.arch.griddepcontrol_wait()":
                    del statements[index]
                    removed_wait += 1
                    break
            if removed_wait:
                break
    if removed_wait != 1:
        raise exc.BackendUnsupported(
            "cute",
            "split-K atomic epilogue expects the TMA warp's dependent-launch wait",
        )
    # The epilogue warps' own wait stays and must dominate the reductions.
    _check_epilogue_wait_precedes_reductions(
        kernel, f"_cute_{_RED_HELPERS[proof.output_dtype]}"
    )
    # The wrapper: raw tensor parameters and the launch call.
    wrapper_names = [argument.arg for argument in wrapper.args.args]
    if wrapper_names != ["a", "b", "partials"]:
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue expects the producer wrapper form"
        )
    annotation = wrapper.args.args[2].annotation
    wrapper.args.args = [ast.arg(arg=name, annotation=annotation) for name in raw]
    for node in ast.walk(wrapper):
        if isinstance(node, ast.Call) and any(
            isinstance(argument, ast.Name) and argument.id == "partials"
            for argument in node.args
        ):
            position = next(
                index
                for index, argument in enumerate(node.args)
                if isinstance(argument, ast.Name) and argument.id == "partials"
            )
            node.args[position : position + 1] = [
                ast.Name(id=name, ctx=ast.Load()) for name in raw[2:]
            ]
        elif (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "partials"
        ):
            node.value = ast.Name(id="out", ctx=ast.Load())
    # The wrapper plans: the store descriptor is gone.
    kept_plans = [plan for plan in plans if plan.get("kind") != "tcgen05_d_tma"]
    if len(kept_plans) != len(plans) - 1:
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue expects one TMA-store wrapper plan"
        )
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Attribute)
            and statement.targets[0].attr == "_helion_cute_wrapper_plans"
        ):
            statement.value = ast.parse(repr(kept_plans), mode="eval").body
    helper = _RED_HELPERS[proof.output_dtype]
    module.body[:0] = [
        statement_from_string(
            f"from helion._compiler.cute.vec_utils import {helper} as _cute_{helper}"
        ),
        statement_from_string(
            f"import helion._compiler.cute._split_k_atomic_runtime as {_ATOMIC_RUNTIME}"
        ),
    ]
    ast.fix_missing_locations(module)
    return kept_plans


_EPILOGUE_TILE_CALL = "cutlass.utils.blackwell_helpers.compute_epilogue_tile_shape"
_CUTLASS_WIDTHS = {
    "cutlass.Float32": 32,
    "cutlass.Float16": 16,
    "cutlass.BFloat16": 16,
}


def _emitted_epilogue_geometry(
    kernel: ast.FunctionDef,
    proof: SplitKWorkspaceProof,
    schedule: SplitKWorkspaceSchedule,
) -> AtomicEpilogueGeometry:
    """The staged-subtile geometry of the generated producer, read off its
    ``tcgen05_store_epi_tile = compute_epilogue_tile_shape(...)`` statement
    (the lowering picks the CTA shape, two-CTA flag and source-C description
    per tile mode; the staged smem tile has exactly this subtile shape)."""
    calls = [
        statement.value
        for statement in ast.walk(kernel)
        if isinstance(statement, ast.Assign)
        and assigned_name(statement) == "tcgen05_store_epi_tile"
        and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value.func) == _EPILOGUE_TILE_CALL
    ]
    if len(calls) != 1:
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue expects one staged epilogue tile"
        )
    call = calls[0]
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    shape = call.args[0] if call.args else None
    if (
        len(call.args) != 4
        or not isinstance(shape, ast.Tuple)
        or len(shape.elts) != 2
        or not all(
            isinstance(element, ast.Constant) and type(element.value) is int
            for element in shape.elts
        )
        or not isinstance(call.args[1], ast.Constant)
        or type(call.args[1].value) is not bool
        or ast.unparse(call.args[3]) != "cutlass.Float32"
        or set(keywords) - {"layout_c", "elem_ty_c"}
    ):
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue could not read the epilogue tile"
        )
    cta_m, cta_n = (cast("ast.Constant", element).value for element in shape.elts)
    source_c = keywords.get("elem_ty_c")
    geometry = atomic_epilogue_geometry(
        rows=_emitted_cta_rows(kernel, schedule),
        cta_m=cast("int", cta_m),
        cta_n=cast("int", cta_n),
        use_2cta_instrs=bool(call.args[1].value),
        elem_width_c=None
        if source_c is None
        else _CUTLASS_WIDTHS[ast.unparse(source_c)],
        d_is_m_major=ast.unparse(call.args[2]).endswith("COL_MAJOR"),
        c_is_m_major=(
            "layout_c" not in keywords
            or ast.unparse(keywords["layout_c"]).endswith("COL_MAJOR")
        ),
        itemsize=proof.output_dtype.itemsize,
    )
    if geometry is None:
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue geometry declined"
        )
    return geometry


_TILED_MMA_CALL = "cutlass.utils.blackwell_helpers.make_trivial_tiled_mma"
_TWO_CTA_GROUP = "cute.nvgpu.tcgen05.CtaGroup.TWO"


def _emitted_cta_rows(
    kernel: ast.FunctionDef, schedule: SplitKWorkspaceSchedule
) -> int:
    """The output rows one CTA of the generated producer owns, read off its
    ``tiled_mma = make_trivial_tiled_mma(..., CtaGroup.TWO, (M, N), ...)``:
    a two-CTA pair splits the ``M`` rows of its MMA tile between the cluster
    ranks (``thr_mma = tiled_mma.get_slice(mma_slice_tidx)`` with the rank
    in ``mma_slice_tidx``), which is how the epilogue's row base
    ``tile_offset_1 + mma_slice_tidx * rows`` addresses them.  Anything but
    the two-CTA pair over the schedule's ``(bm, bn)`` tile is declined: a
    one-CTA producer owns every row and has no rank selector."""
    calls = [
        statement.value
        for statement in ast.walk(kernel)
        if isinstance(statement, ast.Assign)
        and assigned_name(statement) == "tiled_mma"
        and isinstance(statement.value, ast.Call)
        and ast.unparse(statement.value.func) == _TILED_MMA_CALL
    ]
    if len(calls) != 1:
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue expects one tcgen05 tiled MMA"
        )
    arguments = [ast.unparse(argument) for argument in calls[0].args]
    tiler = calls[0].args[6] if len(calls[0].args) > 6 else None
    if (
        len(arguments) != 8
        or arguments[5] != _TWO_CTA_GROUP
        or not isinstance(tiler, ast.Tuple)
        or [
            element.value if isinstance(element, ast.Constant) else None
            for element in tiler.elts
        ]
        != [schedule.bm, schedule.bn]
    ):
        raise exc.BackendUnsupported(
            "cute",
            "split-K atomic epilogue expects a two-CTA tcgen05 MMA over the "
            f"schedule's ({schedule.bm}, {schedule.bn}) tile",
        )
    selector = [
        ast.unparse(statement.value)
        for statement in ast.walk(kernel)
        if isinstance(statement, ast.Assign)
        and assigned_name(statement) == "mma_slice_tidx"
    ]
    if selector != [
        "_helion_tc_pro.mod_nonneg(cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()), 2)"
    ]:
        raise exc.BackendUnsupported(
            "cute", "split-K atomic epilogue expects the cluster rank MMA slice"
        )
    return schedule.bm // 2


_ATOMIC_RUNTIME = "_helion_split_k_atomic"


def _tiler_check_statement(geometry: AtomicEpilogueGeometry) -> ast.stmt:
    """The trace-time check of the emitted epilogue tiler against the
    arithmetic the atomic epilogue addresses the output with
    (``_split_k_atomic_runtime.check_epilogue_tiler``)."""
    return statement_from_string(
        f"{_ATOMIC_RUNTIME}.check_epilogue_tiler(tcgen05_store_epi_tile, "
        f"rows={geometry.rows}, tile_m={geometry.tile_m}, tile_n={geometry.tile_n}, "
        f"run={geometry.run}, run_stride={geometry.run_stride})"
    )


def _insert_after_assignment(
    kernel: ast.FunctionDef, name: str, statement: ast.stmt
) -> None:
    """Place ``statement`` right after the kernel's assignment to ``name``."""
    for statements in _kernel_statement_lists(kernel):
        for index, candidate in enumerate(statements):
            if assigned_name(candidate) == name:
                statements.insert(index + 1, statement)
                return
    raise exc.BackendUnsupported(
        "cute", f"split-K atomic epilogue expects the producer to bind {name}"
    )


def _statement_contains(statement: ast.AST, text: str) -> bool:
    return text in ast.unparse(statement)


def _check_epilogue_wait_precedes_reductions(
    kernel: ast.FunctionDef, helper: str
) -> None:
    """The reductions into the fill-zeroed output must run behind the
    epilogue warps' dependent-launch wait: the first top-level statement of
    the kernel whose subtree issues ``helper`` must be a role branch whose
    own statement list has ``cute.arch.griddepcontrol_wait()`` ahead of the
    statement holding the reductions, and no reduction may sit outside that
    branch."""
    wait = "cute.arch.griddepcontrol_wait()"
    holders = [
        statement
        for statement in kernel.body
        if _statement_contains(statement, f"{helper}(")
    ]
    if len(holders) != 1 or not isinstance(holders[0], ast.If):
        raise exc.BackendUnsupported(
            "cute",
            "split-K atomic epilogue expects its reductions inside one epilogue role branch",
        )
    branch = holders[0]
    first = next(
        index
        for index, statement in enumerate(branch.body)
        if _statement_contains(statement, f"{helper}(")
    )
    if not any(ast.unparse(statement) == wait for statement in branch.body[:first]):
        raise exc.BackendUnsupported(
            "cute",
            "split-K atomic epilogue expects the epilogue warps' dependent-launch "
            "wait ahead of the first reduction",
        )


def _atomic_epilogue_statements(
    proof: SplitKWorkspaceProof, geometry: AtomicEpilogueGeometry
) -> str:
    """The epilogue warps' packed atomic adds of one staged subtile."""
    helper = f"_cute_{_RED_HELPERS[proof.output_dtype]}"
    element_type = cutlass_dtype_name(proof.output_dtype)
    width = geometry.width
    per_warp = geometry.tile_m * geometry.chunks_per_row // 4
    bias = proof.bias
    hoist = bias is not None and bias.dimensions == (1,) and geometry.lane_column_fixed

    def bias_index(row: str, column: str) -> str:
        assert bias is not None
        coordinates = (row, column)
        return " + ".join(
            f"({coordinates[dimension]}) * {stride}"
            for dimension, stride in zip(bias.dimensions, bias.strides, strict=True)
        )

    def bias_load(index: str) -> str:
        return f"cutlass.Float32(cute.make_tensor(bias.iterator + ({index}), cute.make_layout(1))[0])"

    # The CTA's rows start at its cluster rank's half of the pair tile; the
    # subtile's rows and columns follow the static epilogue tiler.
    subtile_m, subtile_n = geometry.subtile_expressions("_tcgen05_subtile")
    row = geometry.row_expression(subtile_m, "tcgen05_red_row")
    column = geometry.column_expression(subtile_n, "tcgen05_red_col")
    lines = [
        f"tcgen05_red_row_base = tile_offset_1 + mma_slice_tidx * cutlass.Int32({geometry.rows})"
    ]
    if hoist:
        lines.extend(
            (
                f"tcgen05_red_col = (tcgen05_lane_idx % cutlass.Int32({geometry.chunks_per_row})) * cutlass.Int32({width})",
                f"tcgen05_red_column = tile_offset_2 + ({column})",
                *(
                    f"tcgen05_red_bias_{element} = cutlass.Float32(0.0)"
                    for element in range(width)
                ),
                "if tile_offset_0 == cutlass.Int32(0):",
                *(
                    f"    tcgen05_red_bias_{element} = "
                    + bias_load(
                        bias_index(
                            "tcgen05_red_row_base",
                            f"tcgen05_red_column + cutlass.Int32({element})",
                        )
                    )
                    for element in range(width)
                ),
            )
        )
    lines.extend(
        (
            f"for tcgen05_red_i in cutlass.range_constexpr({geometry.iterations}):",
            f"    tcgen05_red_chunk = tcgen05_warp_idx * cutlass.Int32({per_warp}) + cutlass.Int32(tcgen05_red_i * 32) + tcgen05_lane_idx",
            f"    tcgen05_red_row = tcgen05_red_chunk // cutlass.Int32({geometry.chunks_per_row})",
        )
    )
    if not hoist:
        lines.extend(
            (
                f"    tcgen05_red_col = (tcgen05_red_chunk % cutlass.Int32({geometry.chunks_per_row})) * cutlass.Int32({width})",
                f"    tcgen05_red_column = tile_offset_2 + ({column})",
            )
        )
    lines.append(f"    tcgen05_red_out_row = tcgen05_red_row_base + ({row})")
    for element in range(width):
        value = f"cutlass.Float32(tcgen05_sD[(tcgen05_red_row, tcgen05_red_col + cutlass.Int32({element}), tcgen05_c_buffer)])"
        if hoist:
            value += f" + tcgen05_red_bias_{element}"
        lines.append(f"    tcgen05_red_v{element} = {value}")
    if bias is not None and not hoist:
        lines.append("    if tile_offset_0 == cutlass.Int32(0):")
        lines.extend(
            f"        tcgen05_red_v{element} = tcgen05_red_v{element} + "
            + bias_load(
                bias_index(
                    "tcgen05_red_out_row",
                    f"tcgen05_red_column + cutlass.Int32({element})",
                )
            )
            for element in range(width)
        )
    values = ", ".join(
        f"{element_type}(tcgen05_red_v{element})" for element in range(width)
    )
    lines.extend(
        (
            f"    {helper}((out.iterator + tcgen05_red_out_row * cutlass.Int32({proof.n}) + tcgen05_red_column).llvm_ptr, [{values}])",
            "if tcgen05_warp_idx == cutlass.Int32(0):",
            "    tcgen05_c_pipeline.producer_commit()",
        )
    )
    return "\n".join(lines) + "\n"


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
    capacity = target_facts.per_cta_smem_capacity_bytes(env.device)
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
    native_source, plans = _producer_source(proof, schedule, env, capacity, config)
    reduce_source = (
        None if schedule.atomic_epilogue else _reducer_source(proof, schedule)
    )
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
    fill_kernel = unique(f"{ZERO_FILL_KERNEL_PREFIX}split_k")
    partials = unique("_helion_split_k_partials")
    view_a = unique("_helion_split_k_a")
    view_b = unique("_helion_split_k_b")
    output_dtype = torch_dtype_name(proof.output_dtype)
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
            if proof.store:
                # The plain store pattern keeps whatever fresh factory the
                # host wrote; the reducer writes every element.
                assert factory in (
                    torch.empty,
                    torch.empty_like,
                    torch.zeros,
                    torch.zeros_like,
                    torch.ones,
                    torch.ones_like,
                    torch.full,
                    torch.full_like,
                )
            else:
                assert factory in (torch.zeros, torch.zeros_like)
            lowered_allocation = ast.parse(ast.unparse(allocation), mode="eval").body
            assert isinstance(lowered_allocation, ast.Call)
            # Keep the original allocation's host expressions and evaluation
            # point. A matrix input may be a view bound later in the prelude.
            like_factory = factory in (
                torch.zeros_like,
                torch.empty_like,
                torch.ones_like,
                torch.full_like,
            )
            stmt.value = ast.Call(
                func=ast.parse(
                    "torch.empty_like" if like_factory else "torch.empty",
                    mode="eval",
                ).body,
                args=(
                    # ``torch.full(size, fill_value)``: the fill is the last
                    # positional argument unless it was passed by keyword.
                    lowered_allocation.args[:-1]
                    if factory in (torch.full, torch.full_like)
                    and not any(
                        keyword.arg == "fill_value"
                        for keyword in lowered_allocation.keywords
                    )
                    else lowered_allocation.args
                ),
                keywords=[
                    *(
                        keyword
                        for keyword in lowered_allocation.keywords
                        if keyword.arg not in ("dtype", "fill_value")
                    ),
                    ast.keyword(
                        arg="dtype", value=ast.parse(output_dtype, mode="eval").body
                    ),
                ],
            )
            allocation_count += 1
        elif isinstance(stmt, ast.Return) and not proof.argument_output:
            stmt.value = ast.Name(id=proof.output_name, ctx=ast.Load())
    # An argument output has no allocation to keep; the reducer writes the
    # argument itself.
    assert allocation_count == (0 if proof.argument_output else 1)
    p, chunk = schedule.partitions, schedule.chunk
    a_stride = (chunk * proof.lhs_strides[1], *proof.lhs_strides)
    b_stride = (chunk * proof.rhs_strides[0], *proof.rhs_strides)
    # K-offset partitions read the operands whole: one batch of all of K
    # (``_offset_partition_k_coordinates``).
    views_p, views_k = (1, proof.k) if schedule.k_offset else (p, chunk)
    views = (
        f"{view_a} = ({proof.lhs_expression}).as_strided(({views_p}, {proof.m}, {views_k}), {a_stride!r})\n"
        f"{view_b} = ({proof.rhs_expression}).as_strided(({views_p}, {views_k}, {proof.n}), {b_stride!r})\n"
    )
    bias_argument = f", {proof.bias.host_expression}" if proof.bias is not None else ""
    native_kernel = f"{native_module}._helion__native_partition_gemm"
    reduce_kernel = f"{reduce_module}._reduce_partitions"
    compile_options = config_compile_options(config)
    option = compile_options_keyword(compile_options)
    fill_definitions: list[ast.stmt] = []
    if schedule.atomic_epilogue:
        # The producer reduces into the output: zero it with a fill kernel
        # the producer launch depends on (the producer's operand loads run
        # under the fill, its epilogue waits for it), no partials, no reducer.
        fill_definitions, fill_launch = zero_fill_kernel_launch(
            fill_kernel,
            proof.output_dtype,
            proof.output_name,
            num_sm=env.config_spec.num_sm,
            numel=proof.m * proof.n,
            compile_options=compile_options,
        )
        new_calls = [
            fill_launch,
            *ast.parse(
                views
                + f"{native_module}._native_partition_gemm({view_a}, {view_b}, {proof.output_name}{bias_argument}, _launcher=_launcher)\n"
            ).body,
        ]
        kernels = (native_kernel, fill_kernel)
    else:
        reducer_grid = _reducer_grid(proof.m * proof.n)
        new_calls = ast.parse(
            f"{partials} = torch.empty(({p}, {proof.m}, {proof.n}), dtype=torch.float32, device=({proof.lhs_expression}).device)\n"
            + views
            + f"{native_module}._native_partition_gemm({view_a}, {view_b}, {partials}, _launcher=_launcher)\n"
            f"_launcher({reduce_module}._reduce_partitions, ({reducer_grid},), {partials}, {proof.output_name}{bias_argument}, block=({REDUCER_THREADS}, 1, 1){option})\n"
        ).body
        kernels = (native_kernel, reduce_kernel)
    index = host.body.index(launch_marker)
    host.body[index : index + 1] = new_calls
    loads = [statement_from_string("from torch._inductor.codecache import PyCodeCache")]
    for module_name, kernel, source in (
        (native_module, native_kernel, native_source),
        (reduce_module, reduce_kernel, reduce_source),
    ):
        if source is None:
            continue
        loads.extend(
            (
                statement_from_string(f"{module_name} = PyCodeCache.load({source!r})"),
                statement_from_string(
                    f"{kernel}._helion_cute_source_hash = {hashlib.sha256(source.encode()).hexdigest()!r}"
                ),
            )
        )
    if reduce_source is not None:
        # The reducer launches as a programmatic dependent of the producer
        # (its wait is its first statement).
        loads.append(
            statement_from_string(f"{reduce_kernel}._helion_cute_use_pdl = True")
        )
    if fill_definitions:
        loads.extend(
            (
                statement_from_string("import cutlass"),
                statement_from_string("import cutlass.cute as cute"),
                *fill_definitions,
            )
        )
    metadata = {
        "partitions": p,
        "chunk_k": chunk,
        # (recorded where the partitions run past K, so exact partitions'
        # modules stay as they were)
        **({"k_offset": True} if schedule.k_offset else {}),
        "tile": (schedule.bm, schedule.bn, schedule.bk),
        "ab_stages": schedule.stages,
        "partial_dtype": None if schedule.atomic_epilogue else "torch.float32",
        "output_dtype": str(proof.output_dtype),
        "bias_partition": 0 if proof.bias is not None else None,
        "epilogue": "atomic" if schedule.atomic_epilogue else "reducer",
    }
    result = ast.Module(
        body=[
            *func.codegen_imports(),
            *loads,
            host,
            statement_from_string(
                f"{func.name}._helion_cute_kernels = ({kernels[0]}, {kernels[1]})"
            ),
            # Both kernels launch on every call (the benchmark timers read the
            # count: ``benchmarking._launches_several_kernels``).
            statement_from_string(f"{func.name}._helion_cute_launches = 2"),
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
    prepend_missing_imports(result)
    cute_env_state(env).resolved_wrapper_plans = plans
    return result

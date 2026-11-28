"""Lower a complete certified page transition while preserving its host loop."""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import cast

import torch
from torch._inductor.virtualized import V

from ... import exc
from . import gathered_panel_layout as panels
from .analysis.gathered_softmax import GatheredSoftmax
from .analysis.gathered_softmax import prove_gathered_softmax
from .fragment_context import DenseSharedIndex
from .fragment_context import Fragment
from .gathered_launch import GATHERED_SEED_THREADS
from .memory_ops import _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
from .memory_ops import tensor_proven_base_alignment

if TYPE_CHECKING:
    from torch.fx import Node

    from ..compile_environment import CompileEnvironment
    from ..generate_ast import GenerateAST
    from .computed_fragment import FragmentCompiler


_RUNTIME = "_helion_gathered_softmax"


def emit_gathered_softmax_preamble(cg: GenerateAST) -> None:
    if not any(
        isinstance(statement, ast.Import)
        and any(alias.asname == _RUNTIME for alias in statement.names)
        for statement in cg.module_statements
    ):
        cg.module_statements.extend(
            ast.parse(
                f"import helion._compiler.cute._gathered_softmax_runtime as {_RUNTIME}"
            ).body
        )


def can_pack_bf16_k_pairs(env: CompileEnvironment, tensor: torch.Tensor) -> bool:
    return _aligned_packets(env, tensor, 4)


def _seed_base_is_aligned(
    env: CompileEnvironment, tensor: torch.Tensor, size: int
) -> bool:
    # Discovery runs before the ordinary persistent-vector specialization
    # snapshots its pointer residues. The bound code later checks that snapshot;
    # the seed can already exclude a misaligned example without retaining it.
    def aligned_input(env: CompileEnvironment, source: torch.Tensor, size: int) -> bool:
        arg = env.tensor_input_source(source)
        specialization = env.runtime_input_specializations.get(
            _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
        )
        if (
            arg is None
            or specialization is None
            or arg not in specialization.sources
            or not env.input_residues_hold_modulo(size)
        ):
            return False
        live = env.live_example_input(arg)
        return live is not None and live.data_ptr() % size == 0

    return (
        tensor_proven_base_alignment(env, tensor, (size,), input_aligned=aligned_input)
        == size
    )


def _aligned_packets(
    env: CompileEnvironment,
    tensor: torch.Tensor,
    size: int,
    *,
    require_bound_pointer: bool = True,
) -> bool:
    # Packet-sized row/page strides preserve base alignment for every
    # possible index. A graph with dynamic placeholder strides must take
    # scalar loads, even if the particular traced layout happens to qualify.
    storage = tensor.untyped_storage()
    origins = env.tensor_storage_provenance(tensor)
    # A host branch can join aligned and misaligned views of the same shape.
    # Its synthetic storage has no runtime pointer of its own. Trust only a
    # direct guarded input storage or a positively tracked fresh allocation.
    if origins is None or len(origins) != 1 or origins[0][0] != storage:
        return False
    if not origins[0][1] and not any(
        source.untyped_storage() == storage for source in env.input_sources
    ):
        return False
    return (
        tensor.dtype == torch.bfloat16
        and tensor.ndim >= 1
        and env.traced_strides_hold(tensor)
        and type(tensor.stride(-1)) is int
        and tensor.stride(-1) == 1
        and all(
            type(stride) is int and stride % (size // 2) == 0
            for stride in tensor.stride()[:-1]
        )
        and (
            tensor_proven_base_alignment(env, tensor, (size,)) == size
            if require_bound_pointer
            else _seed_base_is_aligned(env, tensor, size)
        )
    )


def can_pipeline_page(
    env: CompileEnvironment,
    proof: GatheredSoftmax,
    width: int,
    threads: int,
    *,
    require_bound_pointer: bool = True,
) -> bool:
    capability = env.config_spec.target_device_capability
    return (
        capability is not None
        and capability[0] >= 8
        and threads == GATHERED_SEED_THREADS
        and proof.qk.rows % panels.MMA_ROWS == 0
        and proof.qk.columns == panels.PAGE_ROWS
        and proof.qk.depth > 0
        and proof.qk.depth % panels.DEPTH_PANEL == 0
        and width == panels.VALUE_PANEL
        and all(
            _aligned_packets(
                env,
                cast("torch.Tensor", node.meta["val"]),
                16,
                require_bound_pointer=require_bound_pointer,
            )
            for node in (proof.qk.lhs, proof.qk.rhs, proof.value)
        )
    )


def gathered_softmax_page(
    compiler: FragmentCompiler,
    graph: torch.fx.Graph,
    values: dict[Node, object],
    private_loop_outputs: dict[int, tuple[Node, Fragment]] | None = None,
) -> list[Fragment] | None:
    capability = compiler.env.config_spec.target_device_capability
    if (
        capability is None
        or capability[0] < 8
        or compiler.threads % 32
        or not compiler.offsets
    ):
        return None
    proof = prove_gathered_softmax(graph, compiler.graphs, compiler.env)
    if proof is None or proof.value_bid not in compiler.offsets:
        return None
    old_acc = values[proof.previous_numerator]
    if not isinstance(old_acc, Fragment) or len(old_acc.shape) != 2:
        return None
    m, d = old_acc.shape
    n, k = proof.qk.columns, proof.qk.depth
    w = compiler.threads // 32
    if m != proof.qk.rows or n % 16 or d % (8 * w) or n % (8 * w):
        return None
    # A rounded last value tile can carry out-of-domain columns to a later
    # reduction. Only full value panels have a trivial projected output
    # domain, regardless of how the parent consumes the three results.
    value_width = cast("torch.Tensor", proof.value.meta["val"]).size(2)
    source_extent = compiler.env.block_sizes[proof.value_bid].size
    if (
        type(value_width) is not int
        or type(source_extent) is not int
        or value_width != source_extent
        or value_width % d
        or compiler.bounds.get(proof.value_bid) != str(source_extent)
    ):
        return None
    # Only the table, scalar scale and logical page are evaluated by the ordinary
    # fragment path; this retains both table address guards and signed indexing.
    compiler.needs_gathered_softmax_runtime = True
    compiler.used_gathered_bf16_fold = True
    held: list[object] = [*values.values()]
    compiler.held.append(held)

    def evaluate(node: Node) -> object:
        if node not in values:
            for source in node.all_input_nodes:
                evaluate(source)
            with node.meta["location"], V.set_current_node(node):
                values[node] = compiler.node(node, values)
            held.append(values[node])
        return values[node]

    def reserve(shape: tuple[int, ...], dtype: torch.dtype) -> Fragment:
        buf = compiler.allocate(Fragment(shape, dtype, lambda _: "0"))
        held.append(buf)
        return buf

    def stage(node: Node) -> Fragment:
        f = cast("Fragment", values[node])
        # Resident lazy views can have arbitrary coordinate transforms. Request
        # a plain allocation unless its own row-major storage is identified.
        source = compiler.materialize(
            f, copy=not isinstance(f.element, DenseSharedIndex)
        )
        held.append(source)
        return source

    try:
        phys = evaluate(proof.qk.page)
        scale = evaluate(proof.scale)
        logical = evaluate(proof.logical)

        q, keys, v = (
            compiler.host_tensor(x) for x in (proof.qk.lhs, proof.qk.rhs, proof.value)
        )
        limit, old_m, old_l, old_a = (
            stage(x)
            for x in (
                proof.limit,
                proof.previous_maximum,
                proof.previous_denominator,
                proof.previous_numerator,
            )
        )

        def output(
            index: int, param: Node, old: Fragment, shape: tuple[int, ...]
        ) -> Fragment:
            # The warp program loads the old row or accumulator coordinates
            # into registers before the same owning lanes store their updates.
            # A scratch copy created by stage() is not the private loop carry.
            target = (private_loop_outputs or {}).get(index)
            if target is not None and target[0] is param and target[1] is old:
                return old
            return reserve(shape, torch.float32)

        out_m = output(0, proof.previous_maximum, old_m, (m,))
        out_l = output(1, proof.previous_denominator, old_l, (m,))
        out_a = output(2, proof.previous_numerator, old_a, (m, d))
        probs = reserve((panels.MMA_ROWS, n), torch.bfloat16)
        scratch = reserve((panels.row_scratch_size(w),), torch.float32)
        pipeline = bool(compiler.df.config.get("cute_gathered_page_pipeline"))
        if pipeline and not can_pipeline_page(compiler.env, proof, d, compiler.threads):
            raise exc.InvalidConfig(
                "asynchronous gathered panels require an aligned four-warp BF16 page"
            )
        stages = compiler.df.config.get("cute_gathered_page_stages", 2)
        assert type(stages) is int and stages in panels.RING_STAGES
        # The QK stages share this slab with the later V panel. The fragment
        # allocator aligns buffers to 16 bytes; include enough space to round
        # up to the 128-byte bank swizzle origin inside the helper.
        slab = reserve((panels.slab_bytes(stages),), torch.uint8) if pipeline else None
        compiler.flush_pending_barrier()
        dyn = [
            q.name,
            keys.name,
            v.name,
            *(
                str(x.storage)
                for x in (
                    limit,
                    old_m,
                    old_l,
                    old_a,
                    out_m,
                    out_l,
                    out_a,
                    probs,
                    scratch,
                )
            ),
            compiler.thread,
            compiler.scalar(phys),
            compiler.scalar(logical),
            compiler.offsets[proof.value_bid],
            compiler.scalar(scale),
            compiler.sym(keys.value.size(0)),
            compiler.sym(v.value.size(0)),
            compiler.sym(v.value.size(2)),
        ]
        strides = [
            compiler.host_stride(host, i)
            for host in (q, keys, v)
            for i in range(host.value.ndim)
        ]
        assert (
            old_a.physical_row is not None
            and out_a.physical_row is not None
            and probs.physical_row is not None
        )
        const = [
            m,
            n,
            k,
            d,
            w,
            old_a.physical_row[0] // old_a.dtype.itemsize,
            out_a.physical_row[0] // out_a.dtype.itemsize,
            probs.physical_row[0] // probs.dtype.itemsize,
            can_pack_bf16_k_pairs(compiler.env, q.value),
            can_pack_bf16_k_pairs(compiler.env, keys.value),
        ]
        args = [*dyn, *strides, *map(str, const)]
        if slab is not None:
            compiler.used_async_gathered_page = True
            args.extend([str(slab.storage), str(stages)])
        if compiler.env.settings.fast_math:
            args.append("fast_math=True")
        compiler.emit(f"{_RUNTIME}.gathered_softmax_page({', '.join(args)})")
        return [out_m, out_l, out_a]
    finally:
        compiler.held.pop()

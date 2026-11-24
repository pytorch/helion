from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING
from typing import Any
from typing import cast
import weakref

import sympy
import torch
from torch._inductor import config as inductor_config
from torch._inductor.choices import FusionScore
from torch._inductor.choices import InductorChoices
from torch._inductor.choices import Sortable
from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import Pointwise
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TensorBox
from torch._inductor.loop_body import MemoryUsageType
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.scheduler import FusedMixOrderReductions
from torch._inductor.scheduler import FusedSchedulerNode
from torch._inductor.scheduler import MixOrderReduction
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.scheduler import refresh_group_node_dependencies
from torch._inductor.utils import sympy_product
from torch._inductor.utils import sympy_subs
from torch._inductor.virtualized import V
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Identity

from . import config
from .codegen.kernel import RDIM_SEGMENT_MIN
from .codegen.kernel import RDIM_SEGMENTS_MAX
from .codegen.kernel import HelionKernel
from .codegen.tiling import REDUCTION_RUNS_MAX
from .codegen.tiling import UNROLLED_RDIM_MAX
from .support import MATMUL_DTYPES
from .support import mix_order_split
from .support import names_rdim
from .support import numerics_modes
from .support import python_wrapper

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch._inductor.codegen.simd import SIMDScheduling
    from torch._inductor.ir import IRNode
    from torch._inductor.scheduler import Scheduler


# The reductions that keep Triton's rule for a persistent rdim
# (should_use_persistent_reduction()): a matmul's dot, whose K loop accumulates
# into acc, and those whose loop over tile_r reads each rdim once, combining as
# it goes, where the rdim written as one block reduces it twice (an argmax's
# max, then the first index of it: HelionKernel.arg_reduce()).
LOOPED_REDUCTIONS = OrderedSet(
    [
        "dot",
        "welford_reduce",
        "welford_combine",
        "online_softmax_reduce",
        "argmax",
        "argmin",
    ]
)

# The longest rdim of an online-softmax reduction that the CuTe backend reads as
# one block all the same, in a kernel that reads each rdim again after it
# (rereads_softmax_rdim()): its max, its sum of exps and the rest each pass over
# an rdim the autotuner can keep in registers (cute_reduction_reloads), or in a
# cluster's, where the loop reads it twice. Rdims of 262144 (a softmax over
# [8192, 262144] bf16) are the longest measured faster so.
PERSISTENT_SOFTMAX_RDIM = 2**18


def rereads_softmax_rdim(features: SIMDKernelFeatures) -> bool:
    """Whether a kernel with an online softmax reads each rdim again after it:
    another reduction over the rdim or a value of each element of it that
    depends on the softmax's results, as the output of softmax(x, -1) or the
    sum of it. Its loop then reads the rdim twice, where one block reads it
    once; a kernel that only reduces the rdim, as cross-entropy's max and
    log-sum-exp, reads it once either way, and one block of 262144 lanes is
    slower than the loop for them. So does one that also writes a
    value of each element the softmax's results do not reach, as the logits
    y = x * 2 that a cross-entropy of y returns beside its loss: the loop
    writes it in the pass that reduces the rdim."""
    total = sympy.Mul(features.numel, features.reduction_numel)
    nodes = features.scheduler_nodes()
    # The buffers the softmax's results reach, in kernel order: its own, then
    # those of the nodes that read one of them.
    reached = OrderedSet(
        name
        for node in nodes
        if isinstance(node.node, ComputedBuffer)
        and node.node.get_reduction_type() == "online_softmax_reduce"
        for name in node.get_buffer_names()
    )
    for node in nodes:
        if not isinstance(node.node, ComputedBuffer) or not any(
            dep.name in reached for dep in node.read_writes.reads
        ):
            continue
        if node.is_reduction() or V.graph.sizevars.statically_known_equals(
            sympy_product(node.get_ranges()[0]), total
        ):
            return True
        reached.update(node.get_buffer_names())
    return False


# The shortest contiguous run a split reduction's chunk is made of
# (HelionChoices.layout_split()): over fewer values, the first kernel's
# partials cost about what its reductions over r do.
MIN_SPLIT_RUN = 32
# The fewest values a reduction that Inductor would not split reads for
# HelionChoices.layout_split() to split it all the same, which adds a kernel
# and a workspace: var_mean_42fad's NCHW [32, 128, 28, 28], 3.2M values, went
# from 0.20x to 0.88x of inductor_cd split, and a BatchNorm over [2, 64, 8,
# 8], 8192 values, keeps its one kernel, whose launch the second's would
# double.
MIN_LAYOUT_SPLIT_VALUES = 2**20
# The reductions HelionKernel reads as a tile of their runs (rdim_runs()):
# those whose partials of the runs combine as their blocks' do.
RUN_REDUCTIONS = OrderedSet(
    [
        torch.ops.aten.sum,
        torch.ops.aten.mean,
        torch.ops.aten.prod,
        torch.ops.aten.amax,
        torch.ops.aten.amin,
        torch.ops.aten.any,
    ]
)

# The most chunks a layout split takes where no count of them is within a
# factor of 2 of Inductor's split, as a multiple of that split
# (HelionChoices.split_runs()): the reduction over the partials, which
# Inductor does not split again (create_multilayer_helper()), reads that many
# per output.
MAX_SPLIT_RUNS_OVERSHOOT = 16


def contiguous_run(value: torch.Tensor, dims: Sequence[int]) -> int:
    """The number of values in `value`'s last reduced `dims`, by dim index,
    that are one contiguous run of memory: H and W, 3136 values, of an NCHW
    [32, 256, 56, 56] reduced over N, H and W; 1 when the last reduced dim
    is not the innermost of memory, as NHWC's C, or of a permuted layout
    whose dims are not in memory order (x.transpose(2, 3)), which then keeps
    Inductor's split. Dims of size 1 do not count."""
    run = 1
    for dim in sorted({d % value.dim() for d in dims}, reverse=True):
        if value.shape[dim] == 1:
            continue
        if value.stride(dim) != run:
            break
        run *= value.shape[dim]
    return run


def memory_order(
    index: sympy.Expr, variables: Sequence[sympy.Symbol], ranges: dict[Any, Any]
) -> list[sympy.Symbol] | None:
    """`variables` (those of more than one element) in the order of their
    strides in `index`, outermost first, when `index` lays them out in C order
    and contiguous, each element once at an offset of 0: 100352*y + c + 64*r
    over y of 1024, c of 64 and r of 1568 is [y, r, c]. None for any other
    index: a stride that leaves gaps or overlaps, a // or %, an offset."""
    index = sympy.expand(index.replace(Identity, lambda arg: arg))
    index = index.xreplace({v: sympy.S.Zero for v in variables if ranges[v] == 1})
    kept = [v for v in variables if ranges[v] != 1]
    if not all(isinstance(ranges[v], (int, sympy.Integer)) for v in kept):
        return None
    strides = {v: index.coeff(v) for v in kept}
    if not all(isinstance(s, sympy.Integer) and s > 0 for s in strides.values()):
        return None
    if sympy.expand(index - sum(s * v for v, s in strides.items())) != 0:
        return None
    order = sorted(kept, key=lambda v: -strides[v])
    step = 1
    for v in reversed(order):
        if strides[v] != step:
            return None
        step *= int(ranges[v])
    return order


def body_indices(node: SchedulerNode, name: str, *, loads: bool) -> list[sympy.Expr]:
    """The indices at which `node`'s loop body loads (or stores) buffer `name`."""
    body = node._body
    kinds = (
        (MemoryUsageType.LOAD,)
        if loads
        else (MemoryUsageType.STORE, MemoryUsageType.STORE_REDUCTION)
    )
    return [
        body.indexing_exprs[entry.index_name]
        for kind in kinds
        for entry in body.memory_usage[kind]
        if entry.buffer_name == name
    ]


def shared_inner_loop(snodes: list[SchedulerNode], name: str) -> int | None:
    """The position, among the loops of the nodes of a pointwise node all
    reindexed onto a reduction's loops, of the loop along the contiguous dim
    of buffer `name`, read from the first node that accesses it; None when
    none does in a layout of its loops (memory_order())."""
    for sn in snodes:
        body = sn._body
        iter_vars = list(body.iter_vars)
        shared = [
            *body_indices(sn, name, loads=False),
            *body_indices(sn, name, loads=True),
        ]
        order = memory_order(shared[0], iter_vars, body.var_ranges) if shared else None
        if order:
            return iter_vars.index(order[-1])
    return None


def transposes_onto(node: SchedulerNode, name: str, inner_loop: int) -> bool:
    """Whether `node`, reindexed onto a reduction's loops, accesses some buffer
    other than `name` (the buffer the reduction shares) along loop
    `inner_loop`, the one along the contiguous dim of `name`
    (shared_inner_loop()), with a stride of more than one element, while
    another of its loops is the buffer's contiguous dim: a transposing copy.
    A node of a fused one that does not access `name` itself is checked
    against the loop of the node that does.

    The [32, 4, 128, 768] copy of sum_35b8acdece58 reads its input contiguous
    along r (x0[z, y, r, x] at 393216*z + 128*y + r + 512*x) and writes the
    buffer the split sum over r reads contiguous along x. On its own the copy is
    a pointwise kernel, which the CuTe backend tiles with a transpose through
    shared memory (7.6 us on GB300), and the split reads the copy back (3.1
    us). Fused, the reduction kernel reads x0 512 elements apart along x, one
    element of each 32-byte sector it fetches, in 18.0 us."""
    body = node._body
    iter_vars = list(body.iter_vars)
    inner = iter_vars[inner_loop]
    loops = [v for v in iter_vars if body.var_ranges[v] != 1]
    origin: dict[sympy.Expr, Any] = dict.fromkeys(loops, sympy.S.Zero)
    for entry in (
        body.memory_usage[MemoryUsageType.LOAD]
        + body.memory_usage[MemoryUsageType.STORE]
    ):
        if entry.buffer_name == name:
            continue
        # The index's step along each loop from the first element: the
        # input's view's strides, which a // or % of the reindexed loops
        # (512*p1 + 393216*(p0//4) + ModularIndexing(128*p0 + p2, 1, 512) for
        # the copy above) keeps from sympy's coefficients.
        index = body.indexing_exprs[entry.index_name]
        first = sympy_subs(index, origin)
        steps = {
            v: sympy_subs(index, {**origin, v: 1}) - first  # pyrefly: ignore [unsupported-operation]
            for v in loops
        }
        step = steps.get(inner)
        if (
            isinstance(step, sympy.Integer)
            and abs(step) > 1
            and any(steps[v] == 1 for v in loops if v != inner)
        ):
            return True
    return False


def reindex_onto_reduction(
    scheduler: Scheduler, node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> int:
    """With cuda_backend "helion", reindex a pointwise node onto the loops of
    a reduction it shares a buffer with, when both access the buffer once per
    element in different orders, so that the two fuse; the bytes the fusion
    then saves (Scheduler.score_fusion_memory()), or 0 when it declines. The
    scheduler asks can_fuse() of every candidate pair, and keeps the loops
    of a pair it accepts whether or not it fuses it later: a pointwise node
    that no reduction takes gets its own loops back before it is emitted
    (restore_unfused_reindex()), where its kernel would otherwise walk the
    reduction's order, the r of each split innermost.

    The first kernel of a split sum over r reads the [x, r] buffer a pointwise
    node writes in an order of its own: the SiLU backward that writes the
    channels-last [128, 64, 112, 112] output y of better-benchmark's
    sum_4a4493837e6e, whose channel sums Inductor splits in 1024 chunks of
    1568 elements along r for the 64 channels (c), writes y[p] over a flat p and reads
    it back at 100352*y + c + 64*r. Inductor fuses a pointwise node into a
    reduction only where their accesses already match (or match after a
    permutation of the loops, or after a flat reindex, which keeps the
    order), so it runs three kernels and reads y back:

        for tile_x in hl.tile(102760448):                    # y = f(x0, x1)
            out0[tile_x] = tmp8.to(torch.bfloat16)
        for tile_y, tile_x in hl.tile([1024, 64]):           # partial sums
            for tile_r in hl.tile(1568):
                tmp0 = in0[tile_y, tile_r, tile_x].to(torch.float32)
                acc = acc + torch.sum(tmp0, -2)

    The pointwise node's loops, reindexed to the buffer's dims in memory order
    ([1024, 1568, 64]) and permuted to the reduction's ([1024, 64, 1568]),
    write y where the reduction reads it, so the two become one kernel that
    computes y, stores it and sums it:

        for tile_y, tile_x in hl.tile([1024, 64]):
            for tile_r in hl.tile(1568):
                tmp0 = in0[tile_y, tile_r, tile_x].to(torch.float32)
                ...
                out0[tile_y, tile_r, tile_x] = tmp8.to(torch.bfloat16)
                acc = acc + torch.sum(tmp8, -2)

    which reads x0 and x1 and writes y once each, as the separate pointwise
    kernel does, and no longer reads y.

    Declines unless the pointwise node (or every node of a fused one, all of
    one iteration) and the reduction, a node of its own, access one buffer as
    contiguous C-order layouts of all their loops (memory_order()): a
    pointwise node that writes or reads the buffer once per element in some
    order of its loops, a reduction that reads it once per element. That is
    what makes the reindex a bijection between their iterations. A pointwise
    node that mutates or aliases a buffer keeps its loops too."""
    device = node1.get_device()
    if not (
        device is not None
        and device.type == "cuda"
        and inductor_config.cuda_backend == "helion"
        and node1.is_reduction() != node2.is_reduction()
    ):
        return 0
    pointwise, reduction = (node2, node1) if node1.is_reduction() else (node1, node2)
    if not isinstance(reduction, SchedulerNode) or reduction._body is None:
        return 0
    if not all(isinstance(sn, SchedulerNode) for sn in pointwise.get_nodes()):
        return 0
    # Only a reduction that depends on the pointwise node (reads its output,
    # directly or through other nodes, whose fusion the scheduler's cycle
    # check then decides) takes it. Two nodes that only read a common input
    # fuse horizontally, which Inductor decides on its own count of their
    # shared data (InductorChoices.can_fuse_horizontal()), 0 here: the
    # reindex would be rolled back.
    if not pointwise.get_operation_names() & reduction.ancestors:
        return 0
    # Only the first kernel of a split takes the pointwise node (Inductor
    # marks it, _split_size, with config.triton.mix_order_reduction on, its
    # default outside fbcode): its chunks give the fused kernel a tile_r loop
    # and the grid of a pointwise kernel. A reduction over its whole rdim in
    # one kernel keeps its producer apart, whatever reads its result: the GELU
    # backward and sums over y of better-benchmark's sum_139a3020e8bf over
    # [2048, 10240] fused into one persistent kernel that loads all of y for
    # each tile_x (in0[:, tile_x]) of both inputs and took 37.0 us, against
    # 25.3 us for the pointwise kernel and 5.0 us for the sums over y that
    # read its output back (GB300).
    if not (
        isinstance(reduction.node, ComputedBuffer)
        and reduction.node._split_size is not None
    ):
        return 0
    # The reindex sticks once the pair passes can_fuse(), which the scheduler
    # asks of every candidate pair before it picks one: a pointwise node that
    # reads another pointwise node's output, not yet fused with it, keeps its
    # loops, which that fusion needs. Reindexed onto a split mean's first
    # kernel, the SiLU of a BatchNorm's output y = silu(bn(x)) (y read by the
    # mean) fused with the split instead of with bn, and bn's fp32 output was
    # stored and read back where y alone, in bf16, had been. Once bn and silu
    # are one node, its inputs are the graph's or reductions' and the reindex
    # takes both, when their loops have one size: with
    # config.loop_ordering_after_fusion off, Inductor fuses pointwise nodes
    # of one numel that keep loops of different sizes, and such a group keeps
    # its kernel.
    for dep in pointwise.unmet_dependencies:
        # A buffer's defining op is the node that wrote it, never a fused group.
        producer = scheduler.name_to_buf[dep.name].defining_op
        if isinstance(producer, SchedulerNode) and not producer.is_reduction():
            return 0
    snodes = cast("list[SchedulerNode]", pointwise.get_nodes())
    if pointwise.has_aliasing_or_mutation() or not all(
        sn._body is not None and not sn.is_reduction() and sn._sizes == snodes[0]._sizes
        for sn in snodes
    ):
        return 0
    # A pointwise node that loads under a mask (a cat's inputs, a pad's
    # source: the masked subblocks of its body) keeps its loops. The q, k and
    # v that sum_6d68a671ec4a concatenates and permutes before its sums
    # come from three inputs, each under a mask of the reindexed loops,
    #     tmp0 = 128*tile_y.index + (tile_r.index + 128*tile_z.index) // 197
    #     tmp5 = hl.load(in0, [...], extra_mask=tmp0 < 128)
    # which the CuTe reduction kernel evaluates and loads one element at a
    # time: 23.5 us fused, against 5.6 us for the pointwise kernel, which
    # loads its flat tiles in packets, and 3.0 us for the split's first
    # kernel (GB300). A pad's source in sum_30fd5c612a69 lost 6-19% the
    # same way.
    if loads_under_mask(pointwise):
        return 0
    body = reduction._body
    red_vars = [*body.iter_vars, *body.reduce_vars]
    red_ranges = body.var_ranges
    total = sympy_product(red_ranges.values())
    pw_sizes = snodes[0]._sizes
    if pw_sizes[1] or not V.graph.sizevars.statically_known_equals(
        sympy_product(pw_sizes[0]), total
    ):
        return 0
    common = sorted(
        pointwise.read_writes.buffer_names() & reduction.read_writes.buffer_names(),
        key=lambda name: (
            -V.graph.sizevars.optimization_hint(
                V.graph.get_buffer(name).get_numel(), fallback=0
            )
        ),
    )
    plan = None
    for name in common:
        red_indices = body_indices(reduction, name, loads=True)
        if not red_indices or any(i != red_indices[0] for i in red_indices):
            continue
        red_order = memory_order(red_indices[0], red_vars, red_ranges)
        if red_order is None:
            continue
        for sn in snodes:
            pw_indices = [
                *body_indices(sn, name, loads=False),
                *body_indices(sn, name, loads=True),
            ]
            if not pw_indices or any(i != pw_indices[0] for i in pw_indices):
                continue
            pw_vars = list(sn._body.iter_vars)
            pw_order = memory_order(pw_indices[0], pw_vars, sn._body.var_ranges)
            if pw_order is not None:
                plan = (name, red_order, pw_vars, pw_order)
                break
        if plan is not None:
            break
    if plan is None:
        return 0
    shared, red_order, pw_vars, pw_order = plan
    states = [(sn, sn.snapshot_loop_state()) for sn in snodes]
    group = pointwise.group

    def restore() -> int:
        for sn, state in states:
            sn.restore_loop_state(state)
        if isinstance(pointwise, FusedSchedulerNode):
            pointwise.group = group
            refresh_group_node_dependencies(pointwise)
        return 0

    # 1. The pointwise loops in the shared buffer's memory order (loops of one
    # element first), so that the flat index of the loops is the element's.
    in_memory_order = [
        *(i for i, v in enumerate(pw_vars) if v not in pw_order),
        *(pw_vars.index(v) for v in pw_order),
    ]
    # 2. The flat index cut into the reduction's dims, in memory order, and
    # 3. those dims permuted into the reduction's loop order.
    red_dims = [red_ranges[v] for v in red_order]
    red_loops = [v for v in red_vars if red_ranges[v] != 1]
    to_reduction = [red_order.index(v) for v in red_loops]
    for sn in snodes:
        if in_memory_order != sorted(in_memory_order):
            sn.apply_new_loop_order(in_memory_order)
        sn.apply_loop_reindexing(red_dims)
        if to_reduction != sorted(to_reduction):
            sn.apply_new_loop_order(to_reduction)
    inner_loop = shared_inner_loop(snodes, shared)
    if inner_loop is None or any(
        transposes_onto(sn, shared, inner_loop) for sn in snodes
    ):
        return restore()
    if not inductor_config.loop_ordering_after_fusion:
        # The reduction's dependencies are normalized then, and the
        # reindexed ones are not (apply_loop_reindexing()), which
        # score_fusion_memory() compares exactly; no loop is reordered
        # later (Scheduler._try_reindex_pointwise_for_reduction() does the
        # same).
        for sn in snodes:
            sn.refresh_dependencies(normalize=True, need_clear_tiling_cache=False)
    if isinstance(pointwise, FusedSchedulerNode):
        pointwise.group = snodes[0].group
        refresh_group_node_dependencies(pointwise)
    score = scheduler.score_fusion_memory(node1, node2)
    if score <= 0:
        return restore()
    names = frozenset(pointwise.get_operation_names())
    for sn, state in states:
        # The loops before the first reindex of the node, kept until a
        # reduction takes it or its kernel is emitted
        # (restore_unfused_reindex()).
        _UNFUSED_REINDEX.setdefault(sn, (names, state))
    return score


# The pointwise nodes reindex_onto_reduction() reindexed: the operation names
# of the node it reindexed them as (a fused group or the node), and each one's
# loops before. Neither holds a scheduler node, so an entry no kernel claims
# goes with its node and its scheduler.
_UNFUSED_REINDEX: weakref.WeakKeyDictionary[
    SchedulerNode, tuple[frozenset[str], tuple[Any, ...]]
] = weakref.WeakKeyDictionary()


def fused_reindex(node: BaseSchedulerNode) -> None:
    """Forget the loops before the reindex of the nodes of `node` when it
    holds a reduction: they fused as reindexed."""
    if node.is_reduction():
        for sn in node.get_nodes():
            if isinstance(sn, SchedulerNode):
                _UNFUSED_REINDEX.pop(sn, None)


def restore_unfused_reindex(node: BaseSchedulerNode) -> None:
    """Give the nodes of a pointwise kernel that reindex_onto_reduction()
    reindexed, and no reduction then took, their own loops back, when the
    kernel holds the nodes reindexed together and only them, and forget the
    loops of every node of the kernel: its loops are final once it is
    emitted. A kernel that fused them with other nodes keeps the loops they
    fused with."""
    nodes = [sn for sn in node.get_nodes() if isinstance(sn, SchedulerNode)]
    entries = [_UNFUSED_REINDEX.pop(sn, None) for sn in nodes]
    if node.is_reduction() or not entries or any(e is None for e in entries):
        return
    names = frozenset(node.get_operation_names())
    if any(entry[0] != names for entry in entries if entry is not None):
        return
    for sn, entry in zip(nodes, entries, strict=True):
        assert entry is not None
        sn.restore_loop_state(entry[1])
    if isinstance(node, FusedSchedulerNode):
        refresh_group_node_dependencies(node)


def holds_reduction_prologue(node: BaseSchedulerNode) -> bool:
    """Whether `node` holds a pointwise node one of its reductions reads, a
    producer fused before it (reindex_onto_reduction()): Inductor's
    mix-order codegen emits every pointwise node of a pair's non-contiguous
    node after the pair's reductions, as their epilogue
    (SIMDScheduling._split_mix_order_reduction_epilogue()); the contiguous
    node's run before them."""
    nodes = node.get_nodes()
    reductions = [sn for sn in nodes if sn.is_reduction()]
    return any(
        not sn.is_reduction() and sn.get_operation_names() & reduction.ancestors
        for sn in nodes
        for reduction in reductions
    )


def static_layout(node: IRNode) -> bool:
    """Whether node's sizes are constants, as HelionKernel's tiles need
    (supports_tiling()), and for a buffer or view, its offset and the strides of
    its dims of more than one element, as its indices do (unsupported()'s
    dynamic_index). An operand that is still a Pointwise, a prologue, has no
    layout yet."""
    exprs = [*node.get_size()]
    if (layout := node.maybe_get_layout()) is not None:
        dims = zip(layout.size, layout.stride, strict=True)
        exprs += [stride for size, stride in dims if size != 1]
        exprs.append(layout.offset)
    return all(isinstance(expr, (int, sympy.Integer)) for expr in exprs)


def is_pad(node: IRNode) -> bool:
    """Whether operand node is itself a pad not yet computed: a
    TensorBox(StorageBox(Pointwise)) that reads one buffer through ops.masked, as
    constant_pad_nd lowers to (e.g. the pad that Inductor's pad_mm adds to align a
    matmul's dim). A pad behind a view, as the view that folds a padded 3-D
    activation into a Linear's mm, is not recognized and stays inlined."""
    if not (isinstance(node, TensorBox) and isinstance(node.data, StorageBox)):
        return False
    pointwise = node.data.data
    return (
        isinstance(pointwise, Pointwise)
        and "masked" in pointwise.inner_fn_opcount().used_ops
        and len(pointwise.get_read_names()) == 1
    )


def masked_loads(sn: BaseSchedulerNode) -> int:
    """How many masked subblocks (ops.masked(): a cat's parts, a pad's
    source) the body of pointwise node `sn` has; 0 for any other node."""
    if not (isinstance(sn, SchedulerNode) and sn._body is not None):
        return 0
    return 0 if sn.is_reduction() else len(sn._body.subblocks)


def loads_under_mask(node: BaseSchedulerNode) -> bool:
    """Whether a pointwise node of `node` loads under a mask: a cat's parts or
    a pad's source, the masked subblocks of its body."""
    return any(masked_loads(sn) for sn in node.get_nodes())


# The comparisons of an Inductor loop body (its op_counts).
COMPARISONS = ("lt", "le", "gt", "ge", "eq", "ne")


def segments_masked_parts(node: BaseSchedulerNode) -> bool:
    """Whether `node` is a pointwise node, or a group of them, with one
    member that loads under a mask (loads_under_mask()), computing one cat
    whose parts are all it loads under a mask: 2 to RDIM_SEGMENTS_MAX equal
    parts of RDIM_SEGMENT_MIN or more elements along one dim, a masked
    subblock each (Inductor's pointwise_cat()), which the kernel of `node`
    emits as segments of that dim whose loads carry no mask
    (HelionKernel.pointwise_segments()). That kernel cuts an axis at every
    comparison of an index with an integer in any member
    (HelionKernel.note_predicate()), and gets no segments when another
    comparison cuts a second axis, when two cats cut theirs where the cuts
    leave unequal segments, or when the cat is read through a gather. So the
    cat's own masks, a >= and a < per part as pointwise_cat() builds them,
    must be all the members compare, and none may index indirectly: z =
    torch.where(w < 4608, cat, 0) over a W index w fuses as before, in the
    cat's node or beside it. The count is deliberately conservative: a
    comparison of values (a ReLU backward's <=) keeps the veto off as well,
    and its cat fuses as before. A pad, a cat of unequal parts and other
    masked loads keep their masks in a kernel of their own too."""
    members = node.get_nodes()
    if any(sn.is_reduction() for sn in members):
        # A group that holds a reduction shares the first kernel's chunks,
        # past UNROLLED_RDIM_MAX, where no reduction kernel emits segments.
        return False
    cats = [sn for sn in members if masked_loads(sn)]
    if len(cats) != 1:
        return False
    (sn,) = cats
    masked = masked_loads(sn)
    bodies = [m._body for m in members if isinstance(m, SchedulerNode)]
    if (
        len(bodies) != len(members)
        or sn.node is None
        or any(body is None or body.indirect_vars for body in bodies)
        or sum(body.op_counts[op] for body in bodies if body for op in COMPARISONS)
        != 2 * masked
    ):
        return False
    origins = [
        origin
        for origin in sn.node.get_origins()
        if origin.target is torch.ops.aten.cat.default
    ]
    if len(origins) != 1:
        return False
    cat = origins[0]
    parts = cat.args[0]
    dim = cat.args[1] if len(cat.args) > 1 else cat.kwargs.get("dim", 0)
    if not isinstance(parts, (list, tuple)) or not isinstance(dim, int):
        return False
    # A symbolic size (a SymInt, which does not hash) is no length.
    sizes = OrderedSet(
        size
        if isinstance(part, torch.fx.Node)
        and isinstance(value := part.meta.get("val"), torch.Tensor)
        and value.dim() > 0
        and isinstance(size := value.shape[dim % value.dim()], int)
        else None
        for part in parts
    )
    return (
        masked == len(parts)
        and 2 <= len(parts) <= RDIM_SEGMENTS_MAX
        and len(sizes) == 1
        and isinstance(size := next(iter(sizes)), int)
        and size >= RDIM_SEGMENT_MIN
    )


def looped_split_first_kernel(node: BaseSchedulerNode) -> bool:
    """Whether `node` holds the first kernel of a split reduction (Inductor's
    _split_size) whose chunks are longer than UNROLLED_RDIM_MAX values. Such
    a kernel keeps Triton's rule (HelionChoices.should_use_persistent_reduction()),
    under which it loops over chunks that long whatever its reduction hint.
    The rule loops over shorter chunks too for a hint other than INNER (past
    64 values): those first kernels are not held here."""
    return any(
        isinstance(sn, SchedulerNode)
        and isinstance(sn.node, ComputedBuffer)
        and sn.node._split_size is not None
        and V.graph.sizevars.statically_known_gt(
            sympy_product(sn.node.get_reduction_size()), UNROLLED_RDIM_MAX
        )
        for sn in node.get_nodes()
    )


def reads_long_runs(node: BaseSchedulerNode) -> bool:
    """Whether `node` holds reductions whose kernel reads their runs as a
    tile, runs of more than UNROLLED_RDIM_MAX values that it reads in a loop
    or in a block Helion rolls, as a looped split's first kernel reads its
    chunks: HelionKernel.rdim_runs(), asked of the kernel of the nodes of
    `node` (a kernel with a Welford or an argmax reads the flat rdim)."""
    if not node.is_reduction() or isinstance(node, FusedMixOrderReductions):
        return False
    # masked_into_looped_split() asks on a CUDA device with cuda_backend
    # "helion": HelionScheduling, a SIMD backend, whose group is the node's
    # numel and reduction numel.
    scheduling = cast(
        "SIMDScheduling", V.graph.scheduler.get_backend(node.get_device())
    )
    numel, rnumel = cast("tuple[sympy.Expr, sympy.Expr]", node.group[1])
    node_schedule = scheduling.generate_node_schedule(node.get_nodes(), numel, rnumel)
    features = SIMDKernelFeatures(node_schedule, numel, rnumel)
    runs = HelionKernel.rdim_runs(features)
    return runs is not None and runs[1] > UNROLLED_RDIM_MAX


def loops_over_long_rdim(node: BaseSchedulerNode) -> bool:
    """Whether `node` holds a reduction kernel that reads chunks or runs of
    more than UNROLLED_RDIM_MAX values, a split's first kernel
    (looped_split_first_kernel()) or a reduction over runs
    (reads_long_runs())."""
    return looped_split_first_kernel(node) or reads_long_runs(node)


def masked_into_looped_split(
    node1: BaseSchedulerNode, node2: BaseSchedulerNode
) -> bool:
    """Whether fusing node1 and node2 puts the cat of equal parts a pointwise
    node computes under masks (segments_masked_parts()) into a reduction
    kernel that loops over chunks or runs of more than UNROLLED_RDIM_MAX
    values (loops_over_long_rdim()). The pointwise kernel emits a cat's
    equal parts as segments of their axis, whose loads carry no mask and
    serve every part (HelionScheduling.with_pointwise_segments()); the loop
    evaluates each part's masked loads per element, and each part's own:
    sum_9680481803dd's GLU backward, the cat of two [4, 64, 95696] halves
    computed from three inputs and its sum over N and W, took 100.9 us in one
    kernel over 512 runs of 95696, against 40.3 us for the cat's kernel and
    11.0 us for the split's first kernel reading its output (GB300). A
    persistent first kernel, of chunks of up to UNROLLED_RDIM_MAX, emits the
    parts as segments too: sum_456bcf55ec29's cat of two [4, 1024, 372]
    halves stays in its split's first kernel. A reduction over runs reads
    them the same way: the GLU backward of sum_39ea00f83135, the cat of two
    [4, 256, 5802] halves and its sum over N and W, took 29.1 us with the cat
    in the kernel that reads the 4 runs of 5802 of each channel as a tile,
    where the layout split's first kernel had taken 24.06 us with it. A
    pad's source, or a cat of unequal parts, keeps its masked loads in a
    kernel of its own as well, and fuses: the loop then saves the pass that
    reads its output back."""
    # HelionChoices handles every compile once helion.inductor is imported:
    # stock Inductor keeps its fusion.
    device = node1.get_device()
    if not (
        device is not None
        and device.type == "cuda"
        and inductor_config.cuda_backend == "helion"
    ):
        return False
    return (segments_masked_parts(node1) and loops_over_long_rdim(node2)) or (
        segments_masked_parts(node2) and loops_over_long_rdim(node1)
    )


# The aten reductions of a post-grad graph (reads_reduction()).
REDUCTIONS_OF_GRAPH = OrderedSet(
    [
        *RUN_REDUCTIONS,
        torch.ops.aten.var,
        torch.ops.aten.var_mean,
        torch.ops.aten.std,
        torch.ops.aten.std_mean,
        torch.ops.aten.max,
        torch.ops.aten.min,
        torch.ops.aten.argmax,
        torch.ops.aten.argmin,
        torch.ops.aten.logsumexp,
        torch.ops.prims.var,
    ]
)
# The FX nodes reads_reduction() looks through before it gives up (and
# answers yes).
READS_REDUCTION_NODES = 64


def reads_reduction(node: torch.fx.Node) -> bool:
    """Whether FX node `node` computes on the result of a reduction: one
    among the nodes its inputs are computed from, followed through up to
    READS_REDUCTION_NODES nodes (a graph that has more counts as one that
    does)."""
    seen: OrderedSet[torch.fx.Node] = OrderedSet()
    stack = list(node.all_input_nodes)
    while stack:
        current = stack.pop()
        if current in seen or current.op != "call_function":
            continue
        if len(seen) >= READS_REDUCTION_NODES:
            return True
        seen.add(current)
        target = current.target
        if (
            isinstance(target, torch._ops.OpOverload)
            and target.overloadpacket in REDUCTIONS_OF_GRAPH
        ):
            return True
        stack.extend(current.all_input_nodes)
    return False


class HelionChoices(InductorChoices):
    """Inductor's heuristics, with the changes the methods below make for
    cuda_backend "helion". The first: the matmuls a Helion kernel emits are
    lowered to dot reductions. relu(a @ b + c) is then one Helion kernel:

        helion_fused_add_mm_relu_0(arg0_1, arg1_1, arg2_1, buf1)

    instead of cuBLAS and a Helion epilogue:

        extern_kernels.mm(arg0_1, arg1_1, out=buf0)
        helion_fused_add_relu_0(reinterpret_tensor(buf1, ...), arg2_1, ...)
    """

    def use_native_matmul(self, mat1: IRNode, mat2: IRNode) -> bool:
        """With cuda_backend "helion", lower mat1 @ mat2 to a dot when a Helion
        kernel emits it; otherwise decide as InductorChoices does.

        A dot that HelionScheduling cannot emit fails the compile, so this
        declines what is known at lowering time not to become a Helion kernel,
        which keeps its extern kernel: e.g. relu(a @ b) with a dynamic dim of a,
        or relu(h[:, -1] @ b) with a dynamic dim 1 of h, whose size s is in the
        stride (64*s) and offset (64*s - 64) of h[:, -1]. What only scheduling
        shows still fails: an operand with an indirect load that needs a bounds
        check, as in a[i] @ b with an i of unknown range, or a fused epilogue
        that Helion does not emit, as in a @ b + torch.tensor(2.0) with the
        scalar on the CPU.

        With the CuTe backend, it also gives a pad operand (is_pad()) a kernel of
        its own."""
        device = mat1.get_device_or_error()
        if not (device.type == "cuda" and inductor_config.cuda_backend == "helion"):
            return super().use_native_matmul(mat1, mat2)
        native = (
            config.native_matmul
            and python_wrapper()
            and not numerics_modes()
            and mat1.get_dtype() in MATMUL_DTYPES
            and static_layout(mat1)
            and static_layout(mat2)
            # InductorChoices' default declines dims of size 1 too: Inductor's
            # native matmul does not take them yet. The sizes are static here.
            and min(int(n) for n in (*mat1.get_size()[-2:], mat2.get_size()[-1])) > 1
        )
        if native and config.backend == "cute":
            # A pad is computed by its own kernel, and the dot reads its buffer
            # as tiles. For a @ b with b [512, 197951] column-major, whose N
            # pad_mm pads to 197952:
            #     # helion_fused_mm_0: the pad
            #     for tile_y, tile_x in hl.tile([512, 197952]):
            #         ...
            #         out0[tile_y, tile_x] = tmp12.to(torch.bfloat16)
            #     # helion_fused_mm_1: the matmul
            #         tmp1 = in1[tile_r, tile_x]
            #         acc = torch.addmm(acc, tmp0, tmp1)
            # Inlined in the dot, each [tile_r, tile_x] tile of the pad is
            # recomputed for every tile_y, by a masked flat load of b:
            #     tmp6 = hl.load(in1, [tile_r.index.reshape([tile_r, 1]) + 512*(tile_x.index)], extra_mask=tmp5).to(torch.float32)
            #     tmp8 = torch.where(tmp5, tmp6, tmp7)
            #     ...
            #     acc = torch.addmm(acc, tmp0, tmp14)
            # which CuTe runs on its SIMT dot: 50.5 ms against 1.0 ms for both
            # kernels (256 x 197951 x 512 on a B200, default configs). Triton
            # keeps the pad in the dot, where it costs the mask arithmetic of
            # loads the dot makes anyway: 125 us, against 486 us for both
            # kernels, whose pad reads and writes all of b.
            for operand in (mat1, mat2):
                if is_pad(operand):
                    operand.realize()
        return native

    @staticmethod
    def should_use_persistent_reduction(
        features: SIMDKernelFeatures, cooperative_reduction: bool
    ) -> bool:
        """Whether a reduction kernel reads each row as one block, as
        HelionKernel.should_use_persistent_reduction() asks. With cuda_backend
        "helion" the answer is yes for a kernel of sums, maxes and the like,
        whatever the rdim's length or stride: the kernel writes the rdim as one
        block and leaves the choice between reading it whole and a loop over
        chunks of it to Config.reduction_loops, which the autotuner picks.
        x.sum(-1) of a [8192, 262144] x:

            for tile_x in hl.tile(8192):
                tmp0 = in0[tile_x, :]
                tmp1 = torch.sum(tmp0, -1)

        where the loop Triton's rule asks for (an inner reduction of up to 1024
        elements, another of up to 64) fixes the loop, its chunk and the
        reduction of each chunk in the generated code:

            for tile_x in hl.tile(8192):
                acc = hl.zeros([tile_x], dtype=torch.float32)
                for tile_r in hl.tile(262144):
                    tmp0 = in0[tile_x, tile_r]
                    acc = acc + torch.sum(tmp0, -1)

        A kernel with a matmul keeps Triton's rule, which loops over K, and so
        does one with a Welford or online-softmax reduction: its loop reads each
        rdim once, combining as it goes, where an rdim written as one block
        computes the mean before the variance (and the max before the sum of
        exps), so a loop over the rdim reads it twice. The first kernel of every
        split reduction keeps Triton's rule too (Inductor marks it, _split_size,
        only with config.triton.mix_order_reduction on, its default outside
        fbcode): its chunks loop, the form measured, as the chunks of 125 of
        x.sum(0) over [100000, 96] do (reduction_split_factor(),
        test_split_reduction_loops); a chunk past UNROLLED_RDIM_MAX that no
        input's full rdim names loops anyway
        (HelionScheduling.create_kernel_choices()). With the CuTe backend, an
        online-softmax rdim of up to PERSISTENT_SOFTMAX_RDIM is one block all
        the same in a kernel that reads
        each rdim again after it (rereads_softmax_rdim()), which the autotuner can
        keep in registers across those passes; a kernel that only reduces the
        rdim, as cross-entropy's, keeps the loop, which reads it once. Other
        devices and backends keep Triton's rule, and
        config.triton.persistent_reductions off still asks for loops. The kernel
        keeps an rdim it cannot write as one block in a loop, whatever this says
        (names_rdim()), and so does an rdim past UNROLLED_RDIM_MAX that no
        input's full rdim names (HelionScheduling.create_kernel_choices())."""
        devices = {
            device.type
            for node in features.scheduler_nodes()
            if (device := node.get_device()) is not None
        }
        if (
            devices == {"cuda"}
            and inductor_config.cuda_backend == "helion"
            and inductor_config.triton.persistent_reductions
            and not cooperative_reduction
        ):
            looped = LOOPED_REDUCTIONS
            if (
                config.backend == "cute"
                and V.graph.sizevars.statically_known_leq(
                    features.reduction_numel, PERSISTENT_SOFTMAX_RDIM
                )
                and rereads_softmax_rdim(features)
            ):
                looped = looped - OrderedSet(["online_softmax_reduce"])
            if not any(
                isinstance(node.node, ComputedBuffer)
                and (
                    node.node.get_reduction_type() in looped
                    or node.node._split_size is not None
                )
                for node in features.scheduler_nodes()
            ):
                return True
        return InductorChoices.should_use_persistent_reduction(
            features, cooperative_reduction
        )

    @staticmethod
    def mix_order_reductions(
        node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """Whether, with cuda_backend "helion", node1 and node2 reduce one
        [x, r] array along both dims, over r at each x and over x at each r,
        which HelionScheduling fuses into one kernel that reads the array once
        (codegen_mix_order_reduction()). RMSNorm's backward reads grad and x
        in its reduction over r, sum(grad * w * x, -1), and in its weight's
        gradient, sum(grad * x * rstd, 0), over x: unfused, the second reads
        them again. Inductor's check (MixOrderReduction.can_fuse()), which
        it applies to its Triton kernels only: sibling reductions that read a
        common buffer in full, one over a contiguous r of up to 16384, the
        other a sum or product in a float dtype, over enough data that the
        second read would not come from L2 (5 * 2**20 elements, with an x of
        4096 or more), with config.triton.mix_order_reduction on. Helion's
        conditions: an rdim Helion reads as one block (names_rdim()) and a
        split of x that divides it (mix_order_split()). Neither node may
        already be in a mix-order kernel, nor may the node that is not
        contiguous (FusedMixOrderReductions' node2) hold a producer of its
        reductions (holds_reduction_prologue())."""
        device = node1.get_device()
        if not (
            node1.is_reduction()
            and node2.is_reduction()
            and device is not None
            and device.type == "cuda"
            and inductor_config.cuda_backend == "helion"
            and not any(
                isinstance(node.scheduler.get_fused_node(n), FusedMixOrderReductions)
                for node in (node1, node2)
                for n in node.get_nodes()
            )
        ):
            return False
        # Inductor's check, whose gate on the Triton backend
        # (_is_gpu_triton_backend()) is all that keeps it from Helion's
        # kernels.
        with inductor_config.patch(cuda_backend="triton"):
            if not MixOrderReduction.can_fuse(node1, node2):
                return False
        contiguous = node1 if MixOrderReduction.is_contiguous_node(node1) else node2
        if holds_reduction_prologue(node2 if contiguous is node1 else node1):
            return False
        xnumel, rnumel = MixOrderReduction.get_numel_rnumel(contiguous)
        return (
            isinstance(xnumel, sympy.Integer)
            and isinstance(rnumel, sympy.Integer)
            and names_rdim(rnumel)
            and mix_order_split(int(xnumel), device) is not None
        )

    @staticmethod
    def can_fuse(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """InductorChoices.can_fuse(), which declines nodes that share no data
        by its count: a mix-order pair (mix_order_reductions()) reads a common
        buffer along different dims, which the count misses. It counts for the
        pair the elements of its array, as Inductor counts its Triton pairs. A
        pointwise node and a reduction that access a buffer in different
        orders share no data either until the pointwise node is reindexed onto
        the reduction's loops (reindex_onto_reduction()). A pointwise node
        computing a cat of equal parts stays out of a reduction kernel that
        loops over long chunks or runs (masked_into_looped_split())."""
        if masked_into_looped_split(node1, node2):
            return False
        if HelionChoices.mix_order_reductions(node1, node2):
            shared_data_score = max(
                shared_data_score, MixOrderReduction.get_fusion_score(node1, node2)
            )
        elif shared_data_score == 0:
            shared_data_score = reindex_onto_reduction(scheduler, node1, node2)
        return InductorChoices.can_fuse(scheduler, node1, node2, shared_data_score)

    @staticmethod
    def can_fuse_horizontal(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """InductorChoices.can_fuse_horizontal(), and a mix-order pair
        (mix_order_reductions()) whatever the data it shares."""
        return HelionChoices.mix_order_reductions(
            node1, node2
        ) or InductorChoices.can_fuse_horizontal(
            scheduler, node1, node2, shared_data_score
        )

    @staticmethod
    def score_fusion(
        scheduler: Scheduler, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> Sortable:
        """InductorChoices.score_fusion(), which ranks a mix-order pair
        (mix_order_reductions()) by the elements of its array, as Inductor
        ranks its Triton pairs (MixOrderReduction.get_fusion_score()), with
        the template and node-type scores Inductor gives them. Its
        buffer-overlap score stays the one InductorChoices computed, where
        Inductor's score_fusion_memory() gives such a pair 0: it only breaks
        ties of the memory score."""
        # InductorChoices.score_fusion() returns a FusionScore, typed Sortable.
        score = cast(
            "FusionScore", InductorChoices.score_fusion(scheduler, node1, node2)
        )
        if HelionChoices.mix_order_reductions(node1, node2):
            return dataclasses.replace(
                score,
                template_score=1
                + (node1.is_template() == inductor_config.epilogue_fusion_first),
                node_type_score=1,
                memory_score=MixOrderReduction.get_fusion_score(node1, node2),
            )
        return score

    def reduction_split_factor(
        self,
        device: torch.device,
        reduction_numel_hint: int,
        numel_hint: int,
        inner_reduction: bool,
    ) -> int:
        """Inductor's number of chunks for a split reduction, moved to the
        nearest number, within a factor of 2, that divides the rdim: the first
        kernel then reads whole chunks, with no mask past the end of the rdim.
        var_mean over N, H and W of a channels-last [128, 144, 56, 56], an rdim
        of 401408, in Inductor's 591 chunks of 680 (472 past the end):

            for tile_y, tile_x in hl.tile([591, 144]):
                for tile_r in hl.tile(680):
                    tmp3 = hl.load(in0, [tile_x.index.reshape([tile_x, 1])
                        + 144*(((tile_r.index + 680*tile_y.index.reshape(
                        [tile_y, 1, 1])) % 401408))], extra_mask=tmp2...)

        and in 512 chunks of 784, a tile of a view of in0 with no mask:

            for tile_y, tile_x in hl.tile([512, 144]):
                for tile_r in hl.tile(784):
                    tmp0 = in0[tile_y, tile_r, tile_x]"""
        split = super().reduction_split_factor(
            device, reduction_numel_hint, numel_hint, inner_reduction
        )
        if device.type != "cuda" or inductor_config.cuda_backend != "helion":
            return split
        layout = HelionChoices.layout_split(reduction_numel_hint, numel_hint, split)
        if layout is not None and (
            split > 1
            or (
                reduction_numel_hint * numel_hint >= MIN_LAYOUT_SPLIT_VALUES
                and not HelionChoices.reads_runs(layout)
            )
        ):
            return layout
        if split <= 1 or reduction_numel_hint % split == 0:
            return split
        divisors = [
            d
            for d in sympy.divisors(reduction_numel_hint)
            if max(2, split // 2) <= d <= 2 * split
        ]
        return min(divisors, key=lambda d: abs(d - split), default=split)

    @staticmethod
    def reads_runs(runs: int) -> bool:
        """Whether the reduction V.graph.current_node lowers, over `runs` runs
        of its input (layout_split()), is one HelionKernel reads as a tile of
        its runs (HelionKernel.rdim_runs()): a sum, mean, prod, amax, amin or
        any over at most REDUCTION_RUNS_MAX runs. Where Inductor would not
        split such a reduction, it keeps its one kernel rather than take the
        layout split: the sum over N and W of a [4, 1024, 364] bf16 tensor
        beside its masked copy (Demucs's sum_a05c10ed018a) read the runs
        side by side in one kernel of 2.30 us on a GB300, where the split's
        two kernels took 1.82 and 1.09 us and a launch. Under the CuTe
        backend, not one whose input reads another reduction (the sum of (x -
        x.mean(...)) ** 2, a variance): HelionKernel.rdim_runs() leaves such a
        kernel to the flat rdim there, and the layout split reads its runs as
        views."""
        node = V.graph.current_node
        return (
            runs <= REDUCTION_RUNS_MAX
            and node is not None
            and isinstance(node.target, torch._ops.OpOverload)
            and node.target.overloadpacket in RUN_REDUCTIONS
            and not (config.backend == "cute" and reads_reduction(node))
        )

    @staticmethod
    def layout_split(reduction_numel: int, numel: int, split: int) -> int | None:
        """The split of the reduction V.graph.current_node lowers into chunks
        of one contiguous run of its input each (contiguous_run()), as
        Inductor asks whether to split it: sum, var_mean and the like over N,
        H and W of an NCHW [32, 256, 56, 56] in 32 chunks of H * W. Inductor's
        split then reduces each run of 3136 values first, a tile of a view of
        the input, and the 32 partials of each channel after, with its own
        math (create_multilayer(), split Welford), where its split of the flat
        reduction index runs its chunks across N. The first kernel reads the
        runs as the 8192 rdims of a view (N and C, whose strides join):

            for tile_r in hl.tile(3136):
                tmp0 = in0[tile_x, tile_r].to(torch.float32)

        rather than in0[3136*tile_x + 802816*(tile_r.index // 3136) + ...],
        whose division and remainder keep the loads from being vectors. The
        split is asked of whether Inductor would split (one kernel reduces
        over N and H * W apart only as a tile of their runs,
        HelionKernel.rdim_runs(); reduction_split_factor() takes the split
        where Inductor would not only for MIN_LAYOUT_SPLIT_VALUES values or
        more, and not where the kernel reads the runs as that tile,
        reads_runs()), but not of the second reduction, over the partials,
        nor of another reduction than the node's own: None then, and when the
        run is the whole reduction (a contiguous rdim), shorter than
        MIN_SPLIT_RUN, or the input's layout is not static.

        Where Inductor's `split` asks for more chunks than there are runs,
        each run is cut into equal pieces (split_runs()): the first kernel
        then has about Inductor's parallelism, and each chunk is still one
        contiguous piece of a run, a tile of a view. The sum over N and W of
        [4, 128, 23212], whose 4 runs per channel Inductor would split in 7,
        is split in 8 chunks of 11606: 1024 rdims for the first kernel where
        512 rdims of 23212 left the 152 SMs of a GB300 half idle."""
        node = V.graph.current_node
        if node is None or node.op != "call_function" or isinstance(node.target, str):
            return None
        normalized = normalize_function(
            node.target, node.args, node.kwargs, normalize_to_only_use_kwargs=True
        )
        if normalized is None:
            return None
        x, dims = normalized.kwargs.get("input"), normalized.kwargs.get("dim")
        if isinstance(dims, int):
            dims = [dims]
        if (
            not isinstance(x, torch.fx.Node)
            or not isinstance(value := x.meta.get("val"), torch.Tensor)
            or not isinstance(dims, (list, tuple))
            or not dims
            or not all(isinstance(n, int) for n in (*value.shape, *value.stride()))
        ):
            return None
        rnumel = math.prod(value.shape[d % value.dim()] for d in set(dims))
        run = contiguous_run(value, dims)
        if (
            rnumel != reduction_numel
            or value.numel() != rnumel * numel
            or not MIN_SPLIT_RUN <= run < rnumel
        ):
            return None
        return HelionChoices.split_runs(rnumel // run, run, split)

    @staticmethod
    def split_runs(runs: int, run: int, split: int) -> int:
        """The number of chunks of a layout split of `runs` contiguous runs
        of `run` values each, where Inductor's split asks for `split` chunks:
        the runs as they are when there are as many, else each run cut into
        k equal pieces (k a divisor of the run that leaves MIN_SPLIT_RUN
        values or more per piece), so that no chunk crosses a run and the
        first kernel reads the chunks as a tile of a view. Of the counts of
        chunks that gives:

        - the nearest to `split` within a factor of 2 of it, the window
          reduction_split_factor() also moves Inductor's split within to
          divide an rdim: x.mean((0, 2, 3)) of an NCHW [8, 32, 256, 256],
          8 runs of 65536 per channel, takes Inductor's 16 chunks as 2 per
          run (26 us on a B200, quick autotune, where one chunk per run took
          53 us); 4 runs of 23212 = 4 * 7 * 829 at a split of 64 take 56
          chunks of 1658, not 112. The nearest can lie below `split` and,
          before sm100, leave chunks that Inductor splits again (4 runs of
          20000 at a split of 10 take 8 chunks of 10000, past its 8192
          threshold for their 64 outputs), as reduction_split_factor()'s
          pieces did before this rule;
        - where the run's divisors skip that window, the fewest that reach
          `split`, up to MAX_SPLIT_RUNS_OVERSHOOT times it: the sum over N
          and W of a bf16 [4, 8, 371372] (better-benchmark's
          sum_a200e61c94a5), 4 runs of 371372 = 4 * 227 * 409 per channel at
          Inductor's split of 76, takes 908 chunks of 1636, two kernels with
          no division (4.8 us for the first on a GB300). The most below the
          window, 16 chunks of 92843, leave chunks Inductor splits again, in
          5 of 18569 under a tail mask, read through // and % of a flat view,
          with a third kernel (13.0 us for the first);
        - else the most up to twice `split`: 3 runs of 3165152 = 2**5 * 98911
          at a split of 203 take 96 chunks, not 296733 of 32 values, whose
          partials the second kernel would read 296733 per output."""
        if runs >= split:
            return runs
        counts = [runs * k for k in sympy.divisors(run) if run // k >= MIN_SPLIT_RUN]
        near = [c for c in counts if split <= 2 * c <= 4 * split]
        if near:
            return min(near, key=lambda c: abs(c - split))
        above = [c for c in counts if split <= c <= MAX_SPLIT_RUNS_OVERSHOOT * split]
        if above:
            return min(above)
        return max(c for c in counts if c <= 2 * split)

    def uuid(self) -> None:
        # The FX graph cache key already holds what these choices read:
        # Inductor's config (cuda_backend, triton.*), and helion.inductor.config,
        # which helion.inductor registers with Inductor.
        return None

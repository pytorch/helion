from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

import sympy
import torch
from torch._inductor import config as inductor_config
from torch._inductor.choices import FusionScore
from torch._inductor.choices import InductorChoices
from torch._inductor.choices import Sortable
from torch._inductor.ir import ComputedBuffer
from torch._inductor.scheduler import FusedMixOrderReductions
from torch._inductor.scheduler import MixOrderReduction
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import V
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet

from . import config
from .support import MATMUL_DTYPES
from .support import mix_order_split
from .support import names_row_dim
from .support import numerics_modes
from .support import python_wrapper

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
    from torch._inductor.ir import IRNode
    from torch._inductor.scheduler import BaseSchedulerNode
    from torch._inductor.scheduler import Scheduler


# The reductions that keep Triton's rule for a persistent row
# (should_use_persistent_reduction()): a matmul's dot, whose K loop accumulates
# into acc, and those whose loop over tile_r reads each row once, combining as
# it goes, where the row written as one block reduces it twice.
LOOPED_REDUCTIONS = OrderedSet(
    ["dot", "welford_reduce", "welford_combine", "online_softmax_reduce"]
)

# The longest row of an online-softmax reduction that the CuTe backend reads as
# one block all the same, in a kernel that reads each row again after it
# (rereads_softmax_row()): its max, its sum of exps and the rest each pass over
# a row the autotuner can keep in registers (cute_reduction_reloads), or in a
# cluster's, where the loop reads it twice. Rows of 262144 (amax_sum_7079's
# softmax) are the longest measured faster so.
PERSISTENT_SOFTMAX_ROW = 2**18


def rereads_softmax_row(features: SIMDKernelFeatures) -> bool:
    """Whether a kernel with an online softmax reads each row again after it:
    another reduction over the row or a value of each element of it, as the
    output of softmax(x, -1) or the sum of it. Its loop then reads the row
    twice, where one block reads it once; a kernel that only reduces the row,
    as cross-entropy's max and log-sum-exp, reads it once either way, and one
    block of 262144 lanes (amax_sum_3aad) is slower than the loop."""
    total = sympy.Mul(features.numel, features.reduction_numel)
    return any(
        isinstance(node.node, ComputedBuffer)
        and node.node.get_reduction_type() != "online_softmax_reduce"
        and (
            node.is_reduction()
            or V.graph.sizevars.statically_known_equals(
                sympy_product(node.get_ranges()[0]), total
            )
        )
        for node in features.scheduler_nodes()
    )


# The shortest contiguous run a split reduction's chunk is made of
# (HelionChoices.layout_split()): over fewer values, the first kernel's
# partials cost about what its rows do.
MIN_SPLIT_RUN = 32


def contiguous_run(value: torch.Tensor, dims: Sequence[int]) -> int:
    """The number of values in the last of `value`'s reduced `dims` that are
    one contiguous run of memory, the innermost first: H and W, 3136 values,
    of an NCHW [32, 256, 56, 56] reduced over N, H and W; 1 when the
    innermost dim of memory is not reduced, as NHWC's C. Dims of size 1 do
    not count."""
    run = 1
    for dim in sorted({d % value.dim() for d in dims}, reverse=True):
        if value.shape[dim] == 1:
            continue
        if value.stride(dim) != run:
            break
        run *= value.shape[dim]
    return run


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


class HelionChoices(InductorChoices):
    """Inductor's heuristics, except that with cuda_backend "helion" the matmuls a
    Helion kernel emits are lowered to dot reductions. relu(a @ b + c) is then one
    Helion kernel:

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
        scalar on the CPU."""
        device = mat1.get_device_or_error()
        if not (device.type == "cuda" and inductor_config.cuda_backend == "helion"):
            return super().use_native_matmul(mat1, mat2)
        return (
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

    @staticmethod
    def should_use_persistent_reduction(
        features: SIMDKernelFeatures, cooperative_reduction: bool
    ) -> bool:
        """Whether a reduction kernel reads each row as one block, as
        HelionKernel.should_use_persistent_reduction() asks. With cuda_backend
        "helion" the answer is yes for a kernel of sums, maxes and the like,
        whatever the row's length or stride: the kernel writes the row as one
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
        row once, combining as it goes, where a row written as one block
        computes the mean before the variance (and the max before the sum of
        exps), so a loop over the row reads it twice. The first kernel of every
        split reduction keeps Triton's rule too (Inductor marks it, _split_size,
        only with config.triton.mix_order_reduction on, its default outside
        fbcode). Where its chunks run past the row's end (no count near
        Inductor's divides it, reduction_split_factor()), its loads are masked
        and flat, no full row of an operand names the row's dim, and a row named
        by a factory's dim cannot be rolled (HelionKernel.row_dim()): the chunks
        of a column sum over [65537, 512], a prime number of rows that no count
        near Inductor's divides, would each be read as one block. The chunks
        that divide the row (layout_split()) loop as well, the form measured.
        With the CuTe backend, an online-softmax row of up to
        PERSISTENT_SOFTMAX_ROW is one block all the same in a kernel that reads
        each row again after it (rereads_softmax_row()), which the autotuner can
        keep in registers across those passes; a kernel that only reduces the
        row, as cross-entropy's, keeps the loop, which reads it once. Other
        devices and backends keep Triton's rule, and
        config.triton.persistent_reductions off still asks for loops. The kernel
        keeps a row it cannot write as one block in a loop, whatever this says
        (names_row_dim()), and so does a row past UNROLLED_ROW_MAX that no
        input's full row names (HelionScheduling.create_kernel_choices())."""
        devices = {
            device.type
            for node in features.scheduler_nodes()
            if (device := node.get_device()) is not None
        }
        looped = LOOPED_REDUCTIONS
        if (
            config.backend == "cute"
            and V.graph.sizevars.statically_known_leq(
                features.reduction_numel, PERSISTENT_SOFTMAX_ROW
            )
            and rereads_softmax_row(features)
        ):
            looped = looped - OrderedSet(["online_softmax_reduce"])
        if (
            devices == {"cuda"}
            and inductor_config.cuda_backend == "helion"
            and inductor_config.triton.persistent_reductions
            and not cooperative_reduction
            and not any(
                isinstance(node.node, ComputedBuffer)
                and (
                    node.node.get_reduction_type() in looped
                    or node.node._split_size is not None
                )
                for node in features.scheduler_nodes()
            )
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
        [rows, cols] array along its two dims, the cols of each row (a row
        reduction) and the rows of each col (a column reduction), which
        HelionScheduling fuses into one kernel that reads the array once
        (codegen_mix_order_reduction()). RMSNorm's backward reads grad and x
        in its row reduction, sum(grad * w * x, -1), and in its weight's
        gradient, sum(grad * x * rstd, 0): unfused, the column reduction reads
        them again. Inductor's check (MixOrderReduction.can_fuse()), which
        it applies to its Triton kernels only: sibling reductions that read a
        common buffer in full, one along contiguous rows of up to 16384, the
        other a sum or product in a float dtype, over enough data that the
        second read would not come from L2 (5 * 2**20 elements, in 4096 rows
        or more), with config.triton.mix_order_reduction on. Helion's
        conditions: a row Helion reads as one block (names_row_dim()) and a
        split of the rows that divides them (mix_order_split()). Neither node
        may already be in a mix-order kernel."""
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
        row = node1 if MixOrderReduction.is_contiguous_node(node1) else node2
        rows, cols = MixOrderReduction.get_numel_rnumel(row)
        return (
            isinstance(rows, sympy.Integer)
            and isinstance(cols, sympy.Integer)
            and names_row_dim(cols)
            and mix_order_split(int(rows), device) is not None
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
        pair the elements of its array, as Inductor counts its Triton pairs."""
        if HelionChoices.mix_order_reductions(node1, node2):
            shared_data_score = max(
                shared_data_score, MixOrderReduction.get_fusion_score(node1, node2)
            )
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
        ranks its Triton pairs."""
        score = InductorChoices.score_fusion(scheduler, node1, node2)
        if isinstance(score, FusionScore) and HelionChoices.mix_order_reductions(
            node1, node2
        ):
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
        nearest number, within a factor of 2, that divides the row: the first
        kernel then reads whole chunks, with no mask past the end of the row.
        var_mean over N, H and W of a channels-last [128, 144, 56, 56], a row
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
        if (
            layout := HelionChoices.layout_split(reduction_numel_hint, numel_hint)
        ) is not None:
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
    def layout_split(reduction_numel: int, numel: int) -> int | None:
        """The split of the reduction V.graph.current_node lowers into chunks
        of one contiguous run of its input each (contiguous_run()), as
        Inductor asks whether to split it: sum, var_mean and the like over N,
        H and W of an NCHW [32, 256, 56, 56] in 32 chunks of H * W. Inductor's
        split then reduces each run of 3136 values first, a tile of a view of
        the input, and the 32 partials of each channel after, with its own
        math (create_multilayer(), split Welford), where its split of the flat
        reduction index runs its chunks across N. The first kernel reads the
        runs as the 8192 rows of a view (N and C, whose strides join):

            for tile_r in hl.tile(3136):
                tmp0 = in0[tile_x, tile_r].to(torch.float32)

        rather than in0[3136*tile_x + 802816*(tile_r.index // 3136) + ...],
        whose division and remainder keep the loads from being vectors. The
        split is asked of whether Inductor would split (Helion cannot reduce
        over N and H * W apart in one kernel), but not of the second
        reduction, over the partials, nor of another reduction than the
        node's own: None then, and when the run is the whole reduction (a
        contiguous row), shorter than MIN_SPLIT_RUN, or the input's layout is
        not static."""
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
        return rnumel // run

    def uuid(self) -> None:
        # The FX graph cache key already holds what use_native_matmul() reads:
        # Inductor's cuda_backend, and helion.inductor.config, which
        # helion.inductor registers with Inductor.
        return None

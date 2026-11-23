from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
from torch._inductor import config as inductor_config
from torch._inductor.choices import InductorChoices
from torch._inductor.ir import ComputedBuffer
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet

from . import config
from .support import MATMUL_DTYPES
from .support import numerics_modes
from .support import python_wrapper

if TYPE_CHECKING:
    from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
    from torch._inductor.ir import IRNode


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
        fbcode): its chunks loop, the form measured, as the chunks of 128 of
        x.sum(0) over [100000, 96] do (test_split_reduction_loops); a chunk
        past UNROLLED_RDIM_MAX that no input's full rdim names loops anyway
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

    def uuid(self) -> None:
        # The FX graph cache key already holds what these choices read:
        # Inductor's config (cuda_backend, triton.*), and helion.inductor.config,
        # which helion.inductor registers with Inductor.
        return None

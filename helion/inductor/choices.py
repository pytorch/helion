from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import sympy
import torch
from torch._inductor import config as inductor_config
from torch._inductor.choices import FusionScore
from torch._inductor.choices import InductorChoices
from torch._inductor.choices import Sortable
from torch._inductor.ir import BaseView
from torch._inductor.ir import ComputedBuffer
from torch._inductor.ir import InputBuffer
from torch._inductor.ir import MutableBox
from torch._inductor.ir import Pointwise
from torch._inductor.ir import Reduction
from torch._inductor.ir import StorageBox
from torch._inductor.ir import TensorBox
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.scheduler import FusedMixOrderReductions
from torch._inductor.scheduler import MixOrderReduction
from torch._inductor.scheduler import SchedulerNode
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import V
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet

from .._compiler.backend import Backend
from .._compiler.backend_registry import get_backend
from .._compiler.inductor_matmul import MatmulFacts
from .._compiler.inductor_matmul import MatmulPlan
from . import config
from .support import MATMUL_DTYPES
from .support import mix_order_split
from .support import names_rdim
from .support import numerics_modes
from .support import python_wrapper

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Sequence

    from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
    from torch._inductor.ir import IRNode
    from torch._inductor.scheduler import BaseSchedulerNode
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


def _backend() -> Backend | None:
    """The backend whose kernels helion.inductor emits (config.backend), if it
    plans matmuls (overrides Backend.inductor_matmul_plan)."""
    backend = get_backend(config.backend)
    if type(backend).inductor_matmul_plan is Backend.inductor_matmul_plan:
        return None
    return backend


def _tf32(dtype: torch.dtype) -> bool:
    return (
        dtype == torch.float32 and torch.backends.cuda.matmul.fp32_precision == "tf32"
    )


def _facts(
    dtype: torch.dtype,
    device: torch.device,
    out_dtype: torch.dtype | None = None,
    **sizes: Any,  # noqa: ANN401
) -> MatmulFacts:
    props = DeviceProperties.create(device)
    return MatmulFacts(
        dtype=dtype,
        out_dtype=dtype if out_dtype is None else out_dtype,
        tf32=_tf32(dtype),
        major=props.major,
        sm_count=props.multi_processor_count,
        **sizes,
    )


def _dot_plans(features: SIMDKernelFeatures) -> Iterator[MatmulPlan]:
    """The backend's plan for each dot of a kernel on a GPU, with cuda_backend
    "helion"."""
    backend = _backend()
    if backend is None or inductor_config.cuda_backend != "helion":
        return
    for node in features.node_schedule:
        if not (
            isinstance(node, SchedulerNode)
            and isinstance(node.node, ComputedBuffer)
            and node.node.get_reduction_type() == "dot"
        ):
            continue
        reduction = node.node.data
        assert isinstance(reduction, Reduction)
        device = node.node.get_device_or_error()
        if device.type != "cuda":
            continue
        facts = _facts(reduction.src_dtype, device, reduction.dtype)
        if not backend.inductor_matmul_plan(facts).may_rewrite:
            continue
        size = node.node.get_size()
        m, n = (int(dim) for dim in size[-2:])
        (k,) = (int(dim) for dim in node.node.get_reduction_size())
        yield backend.inductor_matmul_plan(
            dataclasses.replace(facts, rank=len(size), m=m, n=n, k=k)
        )


def dot_keeps_k_loop(features: SIMDKernelFeatures) -> bool:
    """Whether a kernel's dot keeps its loop over K however short K is
    (MatmulPlan.keep_k_loop): K in one block is written as a stored
    torch.mm / torch.bmm, which a backend may run on another path."""
    return any(plan.keep_k_loop for plan in _dot_plans(features))


def dot_reads_k_whole(features: SIMDKernelFeatures) -> bool:
    """Whether a kernel's dot reads K as one block where Inductor's rule would
    loop over it (MatmulPlan.whole_k)."""
    return inductor_config.triton.persistent_reductions and any(
        plan.whole_k for plan in _dot_plans(features)
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


def storage_offset(node: IRNode) -> int:
    """The storage offset, in elements, of the tensor a kernel receives for
    operand `node`: its layout's offset plus that of the graph input or
    constant it is or views, whose buffer Inductor builds at offset 0 (a
    symbolic one by its hint, which Dynamo does not guard on)."""
    hint = V.graph.sizevars.optimization_hint
    offset = hint(node.get_layout().offset)
    while isinstance(node, (MutableBox, BaseView)):
        node = node.data
    if isinstance(node, InputBuffer):
        name = node.get_name()
        if name in V.graph.constants:  # (a real tensor)
            offset += int(V.graph.constants[name].storage_offset())
        else:
            offset += hint(V.graph.graph_input_storage_offsets.get(name, 0))
    return offset


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
    def _plan(
        mat1: IRNode, mat2: IRNode | None = None, operands: Sequence[int] = ()
    ) -> MatmulPlan:
        """The backend's plan for mat1 @ mat2 (torch.bmm at rank 3). Without mat2
        it reads mat1's dtype and device only; with it, the sizes, which
        static_layout() proved constant, and the strides and storage offsets of
        `operands` (0: mat1, 1: mat2), whose layouts the caller has."""
        backend = _backend()
        if backend is None:
            return MatmulPlan()
        sizes: dict[str, Any] = {}
        if mat2 is not None:
            *batch, m, k = (int(size) for size in mat1.get_size())
            sizes.update(
                rank=2 + len(batch),
                batch=batch[0] if batch else None,
                m=m,
                n=int(mat2.get_size()[-1]),
                k=k,
            )
            for operand in operands:
                mat = (mat1, mat2)[operand]
                name = "ab"[operand]
                strides = tuple(int(stride) for stride in mat.get_layout().stride)
                sizes[f"{name}_strides"] = strides
                sizes[f"{name}_offset"] = storage_offset(mat)
        facts = _facts(mat1.get_dtype(), mat1.get_device_or_error(), **sizes)
        return backend.inductor_matmul_plan(facts)

    def transpose_matmul(self, mat1: IRNode, mat2: IRNode) -> bool:
        """Whether helion.inductor lowers mat1 @ mat2 as its transposed product
        (lowering/mm.py, MatmulPlan.transpose). Asks use_native_matmul() first,
        which may realize a pad operand. Inert outside cuda_backend "helion",
        as are the other matmul questions below."""
        if not (
            mat1.get_device_or_error().type == "cuda"
            and inductor_config.cuda_backend == "helion"
            and self._plan(mat1).may_transpose
            and self.use_native_matmul(mat1, mat2)
        ):
            return False
        if mat1.maybe_get_layout() is None or mat2.maybe_get_layout() is None:
            return False
        return self._plan(mat1, mat2, (0, 1)).transpose

    def transpose_bmm(self, mat1: IRNode, mat2: IRNode) -> bool:
        """Whether helion.inductor lowers torch.bmm(mat1, mat2) as its transposed
        product (lowering/mm.py, MatmulPlan.transpose)."""
        if not self._rewritable(mat1, mat2):
            return False
        if mat1.maybe_get_layout() is None or mat2.maybe_get_layout() is None:
            return False
        if int(mat1.get_size()[0]) == 1:
            # (one matrix is no batch, and static_layout() leaves its stride open)
            return False
        return self._plan(mat1, mat2, (0, 1)).transpose

    def copy_matmul_b(self, mat1: IRNode, mat2: IRNode) -> bool:
        """Whether helion.inductor lowers mat1 @ mat2 over a K-major copy of mat2
        (lowering/mm.py, MatmulPlan.copy_b_k_major)."""
        if not self._rewritable(mat1, mat2):
            return False
        if mat1.maybe_get_layout() is None or mat2.maybe_get_layout() is None:
            return False
        return self._plan(mat1, mat2, (0, 1)).copy_b_k_major

    @staticmethod
    def may_rewrite(mat1: IRNode) -> bool:
        """Whether the backend may copy, stage or pad the operands of a matmul of
        mat1, or transpose it as a bmm (MatmulPlan.may_rewrite). Reads no layout
        and realizes nothing, so lowering/mm.py asks it before it reads a zero
        pad's FX nodes."""
        return (
            mat1.get_device_or_error().type == "cuda"
            and inductor_config.cuda_backend == "helion"
            and HelionChoices._plan(mat1).may_rewrite
        )

    def _rewritable(self, mat1: IRNode, mat2: IRNode) -> bool:
        """may_rewrite(), then use_native_matmul(), which may realize a pad
        operand."""
        return self.may_rewrite(mat1) and self.use_native_matmul(mat1, mat2)

    def stage_matmul_operand(
        self, mat1: IRNode, mat2: IRNode, operand: int
    ) -> int | None:
        """The row stride of an aligned copy of operand `operand` of mat1 @ mat2
        (0: mat1, 1: mat2) that helion.inductor writes ahead of the product, or
        None to read the operand as it is (lowering/mm.py, MatmulPlan.stage)."""
        if not self._rewritable(mat1, mat2):
            return None
        if (mat1, mat2)[operand].maybe_get_layout() is None:
            # a Pointwise not yet realized, whose strides are Inductor's to pick
            return None
        return self._plan(mat1, mat2, (operand,)).stage[operand]

    def pad_matmul_output(
        self, mat1: IRNode, mat2: IRNode, operand: int, *, exact: bool
    ) -> bool:
        """Whether helion.inductor computes mat1 @ mat2, the product of the source
        of pad_mm's zero pad of mat1's M (operand 0) or mat2's N (operand 1),
        into a padded output instead (lowering/mm.py, MatmulPlan.pad_output),
        for a product it lowers as written. With `exact`, something can read
        past the slices, so the pad holds eager's values, a reduction over the
        other operand: declined where that operand is larger than the source,
        which the pad kernel would copy."""
        if not self._rewritable(mat1, mat2):
            return False
        source, other = (mat1, mat2)[operand], (mat1, mat2)[1 - operand]
        if source.maybe_get_layout() is None:
            return False
        if exact and int(other.get_numel()) > int(source.get_numel()):
            return False
        return (
            self._plan(mat1, mat2, (operand,)).pad_output[operand]
            and self.stage_matmul_operand(mat1, mat2, 1 - operand) is None
            and not self.transpose_matmul(mat1, mat2)
            and not self.copy_matmul_b(mat1, mat2)
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

        A kernel with a matmul keeps Triton's rule, which loops over K, unless the
        backend's plan keeps its K loop or reads K whole (dot_keeps_k_loop(),
        dot_reads_k_whole()); so does a kernel with a Welford or online-softmax
        reduction: its loop reads each
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
        if dot_keeps_k_loop(features):
            return False
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
        ) or dot_reads_k_whole(features)

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
        already be in a mix-order kernel."""
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
        layout = HelionChoices.layout_split(reduction_numel_hint, numel_hint)
        if layout is not None and (
            split > 1 or reduction_numel_hint * numel_hint >= MIN_LAYOUT_SPLIT_VALUES
        ):
            if split <= layout:
                return layout
            # Fewer runs than Inductor's chunks, which it counts to fill the
            # GPU: each run cut into k chunks, k a divisor of the run, so that
            # the chunks number about Inductor's and each stays in one run, a
            # tile of a view all the same. x.mean((0, 2, 3)) of an NCHW
            # [8, 32, 256, 256], 8 runs of 65536 per channel, takes Inductor's
            # 16 chunks as 2 per run,
            #     for tile_z, tile_y, tile_x in hl.tile([32, 8, 2]):
            #         for tile_r in hl.tile(32768):
            #             tmp0 = in0[tile_z, tile_y, tile_x, tile_r]
            # 26 us on a B200 (quick autotune), where one chunk per run,
            # hl.tile([32, 8]) over runs of 65536, took 53 us.
            run = reduction_numel_hint // layout
            pieces = [d for d in sympy.divisors(run) if layout * d <= 2 * split]
            return layout * min(pieces, key=lambda d: abs(layout * d - split))
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
        runs as the 8192 rdims of a view (N and C, whose strides join):

            for tile_r in hl.tile(3136):
                tmp0 = in0[tile_x, tile_r].to(torch.float32)

        rather than in0[3136*tile_x + 802816*(tile_r.index // 3136) + ...],
        whose division and remainder keep the loads from being vectors. The
        split is asked of whether Inductor would split (Helion cannot reduce
        over N and H * W apart in one kernel; reduction_split_factor() takes
        it where Inductor would not only for MIN_LAYOUT_SPLIT_VALUES values
        or more), but not of the second reduction, over the partials, nor of another reduction than the
        node's own: None then, and when the run is the whole reduction (a
        contiguous rdim), shorter than MIN_SPLIT_RUN, or the input's layout is
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
        # The FX graph cache key already holds what these choices read:
        # Inductor's config (cuda_backend, triton.*), helion.inductor.config
        # (which helion.inductor registers with Inductor; its _source_hash covers
        # the backends' matmul rules), the device, by its name in the system
        # info, and torch.backends.cuda.matmul.fp32_precision (Inductor's own
        # cuda_matmul_settings).
        return None

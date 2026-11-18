from __future__ import annotations

from typing import TYPE_CHECKING

import sympy
from torch._inductor import config as inductor_config
from torch._inductor.choices import InductorChoices

from . import config
from .scheduling import MATMUL_DTYPES
from .scheduling import numerics_modes
from .scheduling import python_wrapper

if TYPE_CHECKING:
    from torch._inductor.codegen.simd_kernel_features import SIMDKernelFeatures
    from torch._inductor.ir import IRNode


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
        HelionKernel.should_use_persistent_reduction() asks: the hook for a rule
        of Helion's own, e.g. one that leaves a long row to
        Config.reduction_loops. For now Triton's rule (an inner reduction of up
        to 1024 elements, another of up to 64). HelionChoices handles every
        backend, so such a rule checks cuda_backend, as use_native_matmul()
        does. The kernel keeps a row it cannot write as one block in a loop,
        whatever this says (names_row_dim())."""
        return InductorChoices.should_use_persistent_reduction(
            features, cooperative_reduction
        )

    def uuid(self) -> None:
        # The FX graph cache key already holds what use_native_matmul() reads:
        # Inductor's cuda_backend, and helion.inductor.config, which
        # helion.inductor registers with Inductor.
        return None

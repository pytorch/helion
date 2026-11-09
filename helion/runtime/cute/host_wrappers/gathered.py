"""Host wrapper emitter of the gathered (indexed-row) tcgen05 MMA: the
gather-atom tensor maps over the source rows and the one-CTA-per-group grid."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

from .... import exc
from .context import HostWrapperContext
from .context import HostWrapperEmitter

if TYPE_CHECKING:
    from collections.abc import Sequence


def emit_gathered(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``gathered_mma_tma``: the gather TMA arguments and the grouped grid."""
    bn = ctx.plan_int("bn")
    stages = ctx.plan_int("stages")
    kernel_args = plan["kernel_args"]
    if (
        bn not in (128, 256, 512)
        or stages not in (2, 3, 4)
        or (128 + bn) * 64 * 2 * stages + 18432 > 232448
        or not isinstance(kernel_args, (list, tuple))
        or len(kernel_args) != 6
        or not all(isinstance(name, str) for name in kernel_args)
    ):
        raise exc.BackendUnsupported("cute", "invalid gathered MMA wrapper geometry")
    parameters = cast("Sequence[str]", kernel_args)
    extent = (
        f"arg{ctx.plan_int('m_extent_idx')}"
        if "m_extent_idx" in plan
        else str(ctx.plan_int("m_extent"))
    )
    ctx.body.extend(
        (
            (
                f"    {', '.join(parameters)} = _helion_make_gathered_tma("
                f"arg{ctx.plan_int('lhs_idx')}, arg{ctx.plan_int('rhs_idx')}, {bn}, {stages})"
            ),
            (
                f"    grid_x = cutlass.Int32({ctx.plan_int('groups') * ((ctx.plan_int('n_size') + bn - 1) // bn)}) * "
                f"cutlass.Int32(cutlass.min({ctx.plan_int('grid_cap')}, cutlass.max(1, "
                f"(cutlass.Int64(cutlass.Int32({extent})) + 127) // 128)))"
            ),
            "    grid_y = cutlass.Int32(1)",
            "    grid_z = cutlass.Int32(1)",
        )
    )
    ctx.call_args.extend(parameters)


def gathered_namespace(plan: dict[str, object]) -> dict[str, object]:
    from ...._compiler.cute.gathered_mma_runtime import make_tma_arguments

    return {"_helion_make_gathered_tma": make_tma_arguments}


EMITTERS = (
    HostWrapperEmitter(
        "gathered_mma_tma",
        emit_gathered,
        namespace=gathered_namespace,
        source_dependencies=(
            "_compiler/cute/_mlir_compat.py",
            "_compiler/cute/gathered_mma_runtime.py",
        ),
    ),
)

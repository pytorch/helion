"""Host wrapper emitter of the native block-scaled (NVFP4 / MXFP8) GEMM.

The device root is ``pass``: the wrapper calls the rendered preparation and
the SM100 block-scaled GEMM host entry (``block_scaled_prepare``) with the
argument tensors, so the emitter re-proves the plan's geometry, binds the
entry into the wrapper namespace and spells the host call.
"""

from __future__ import annotations

from .... import exc
from .context import HostWrapperContext
from .context import HostWrapperEmitter


def emit_block_scaled(ctx: HostWrapperContext, plan: dict[str, object]) -> None:
    """``block_scaled_mma``: re-prove the wrapper geometry; the host call replaces the launch."""
    if (
        ctx.plan_int("bn")
        not in (
            (8, 16, 32, 64, 128, 224, 256)
            if plan.get("format") == "mxfp8"
            else (64, 128, 256)
        )
        or ctx.plan_int("bk") not in (64, 128, 256)
        or ctx.plan_int("stages") not in (2, 3, 4, 5, 6, 8, 10, 12)
        or ctx.plan_int("cluster_m") not in (1, 2)
        or (ctx.plan_int("bn") == 8 and ctx.plan_int("cluster_m") != 1)
        or any(ctx.plan_int(key) <= 0 for key in ("m", "n", "k", "workspace_bytes"))
        or ctx.plan_int("k") % 16
        or type(plan.get("persistent")) is not bool
    ):
        raise exc.BackendUnsupported("cute", "invalid block-scaled wrapper geometry")


def block_scaled_namespace(plan: dict[str, object]) -> dict[str, object]:
    from ...._compiler.cute.block_scaled_prepare import make_block_scaled_entry

    namespace: dict[str, object] = {
        "_helion_block_scaled_entry": make_block_scaled_entry(plan)
    }
    if "scale_value" in plan:
        namespace["_helion_block_scaled_alpha"] = plan["scale_value"]
    return namespace


def block_scaled_host_call(body: list[str], plan: dict[str, object]) -> None:
    arguments = [
        f"arg{plan[key]}"
        for key in (
            "lhs_idx",
            "rhs_idx",
            "lhs_scale_idx",
            "rhs_scale_idx",
            "workspace_idx",
            "out_idx",
        )
    ]
    alpha = (
        f"arg{plan['scale_idx']}"
        if "scale_idx" in plan
        else "_helion_block_scaled_alpha"
    )
    arguments.extend((f"cutlass.Float32({alpha})", "stream"))
    body.append("    _helion_block_scaled_entry(" + ", ".join(arguments) + ")")


EMITTERS = (
    HostWrapperEmitter(
        "block_scaled_mma",
        emit_block_scaled,
        namespace=block_scaled_namespace,
        host_call=block_scaled_host_call,
        # The entry is rendered and exec'd (``__module__`` None) and its
        # imports are function-local, so the helper walk cannot reach these.
        source_dependencies=(
            "exc.py",
            "_compiler/cute/block_scaled_config.py",
            "_compiler/cute/block_scaled_prepare.py",
            "_compiler/cute/block_scaled_runtime.py",
            "_compiler/cute/mxfp8_mma.py",
            "_compiler/cute/mxfp8_prepare.py",
        ),
    ),
)

"""Single source of truth for tcgen05 matmul support constraints.

Intentionally dependency-free (only ``dataclasses``) so both the config-shaping
side (``helion.language.matmul_ops.enforce_dot_requirements``) and the codegen
side (``helion._compiler.cute.cute_mma._emit_mma_pipeline``) can import it at
module level without crossing into the heavy CuTe codegen module or forming an
import cycle (``cute_mma -> aux_tensor -> matmul_ops``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tcgen05MatmulEnvelope:
    """Resolved feature set of a tcgen05 matmul config, for support validation.

    Every field is consumed by a rule in :func:`tcgen05_unsupported_reason`;
    there are no aspirational/unused fields, so both callers must populate all
    of them with values accurate for their context (codegen from the resolved
    config, autotune from the candidate search config). Add a field only
    together with the rule that needs it, and update both call sites -- that is
    what keeps this a genuine single source of truth rather than two predicates
    that can drift apart.
    """

    has_leading_passthrough: bool
    cta_group: int  # 1 (CtaGroup.ONE) or 2 (CtaGroup.TWO)
    cluster_n: int
    persistent: bool
    partial_axes: frozenset[str]  # subset of {"m", "n", "k"} that are not full


def tcgen05_unsupported_reason(env: Tcgen05MatmulEnvelope) -> str | None:
    """Return a reason string if ``env`` is unsupported, else ``None``."""
    # Batched (leading-passthrough) CtaGroup.TWO is validated only for static
    # full tiles: the output-edge scheduler linearizes the virtual pid over
    # M/N only (so a leading batch axis misclassifies partial tiles) and the
    # K-tail reduction is batch-unaware -- either silently miscomputes. This is
    # deliberately TMA-independent so no fallback layout can slip past.
    if (
        env.has_leading_passthrough
        and env.cta_group == 2
        and env.cluster_n == 1
        and env.persistent
        and env.partial_axes
    ):
        axes = "/".join(ax.upper() for ax in ("m", "n", "k") if ax in env.partial_axes)
        return (
            "batched (leading-passthrough) CtaGroup.TWO tcgen05 matmul does "
            f"not support partial {axes} tiles; pad M/N/K to the block sizes "
            "(static full tiles) or use tcgen05_cluster_m=1."
        )
    return None

"""FLA forward kernels keyed by linear-attention variant.

FLA is optional for these examples. When it is installed, each wrapper adapts
the variant's FLA op to the shared Helion-native wrapper signature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

from .linear_attention_engine import LinearAttentionVariant

if TYPE_CHECKING:
    import torch


class FlaForwardKernel(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


_FLA_FWD: dict[LinearAttentionVariant, FlaForwardKernel] = {}

try:
    from fla.ops.delta_rule import chunk_delta_rule  # pyrefly: ignore[missing-import]
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,  # pyrefly: ignore[missing-import]
    )
    from fla.ops.gla import chunk_gla  # pyrefly: ignore[missing-import]
    from fla.ops.kda import chunk_kda  # pyrefly: ignore[missing-import]
    from fla.ops.linear_attn import chunk_linear_attn  # pyrefly: ignore[missing-import]
    from fla.ops.retention import chunk_retention  # pyrefly: ignore[missing-import]
    from fla.ops.simple_gla import chunk_simple_gla  # pyrefly: ignore[missing-import]
except ImportError:
    pass
else:

    def fla_chunk_linear_attn(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_linear_attn(
            q,
            k,
            v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=return_final_state,
            normalize=False,
        )

    def fla_chunk_simple_gla(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert g is not None
        return chunk_simple_gla(
            q,
            k,
            v,
            g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=return_final_state,
        )

    def fla_chunk_retention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return chunk_retention(
            q,
            k,
            v,
            scale=scale,
            initial_state=initial_state,
            output_final_state=return_final_state,
        )

    def fla_chunk_gla(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert g is not None
        return chunk_gla(
            q,
            k,
            v,
            g,
            scale=scale,
            initial_state=initial_state,
            output_final_state=return_final_state,
        )

    def fla_chunk_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert beta is not None
        return chunk_delta_rule(
            q,
            k,
            v,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=return_final_state,
        )

    def fla_chunk_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert g is not None
        assert beta is not None
        return chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=return_final_state,
        )

    def fla_chunk_kda(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert g is not None
        assert beta is not None
        return chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=return_final_state,
        )

    _FLA_FWD = {
        LinearAttentionVariant.VANILLA: fla_chunk_linear_attn,
        LinearAttentionVariant.SIMPLE_GLA: fla_chunk_simple_gla,
        LinearAttentionVariant.RETENTION: fla_chunk_retention,
        LinearAttentionVariant.FULL_GLA: fla_chunk_gla,
        LinearAttentionVariant.DELTA_RULE: fla_chunk_delta_rule,
        LinearAttentionVariant.GATED_DELTA_RULE: fla_chunk_gated_delta_rule,
        LinearAttentionVariant.KDA: fla_chunk_kda,
    }


def get_fla_fwd_kernel(variant: LinearAttentionVariant) -> FlaForwardKernel | None:
    return _FLA_FWD.get(variant)

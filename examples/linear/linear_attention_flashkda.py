"""FlashKDA forward kernel, adapted to the shared wrapper signature.

FlashKDA is a hand-written CUDA extension and is optional for these examples. It
implements KDA only, forward only, at D == 128, and it takes the model's input
transforms itself, so it pairs with the fused-preamble flags rather than with
pre-transformed inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol

from .linear_attention_engine import LinearAttentionVariant

if TYPE_CHECKING:
    import torch


class FlashKdaForwardKernel(Protocol):
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
        cu_seqlens: torch.Tensor | None = None,
        **preamble: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


_FLASHKDA_FWD: dict[LinearAttentionVariant, FlashKdaForwardKernel] = {}

try:
    import flash_kda  # pyrefly: ignore[missing-import]
except ImportError:
    pass
else:
    import torch

    def flashkda_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        *,
        scale: float = 1.0,
        initial_state: torch.Tensor | None = None,
        return_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **preamble: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """flash_kda.fwd, which writes o and the final state into out params.

        State is [N, H, V, K], the state_v_first orientation. cu_seqlens is int64
        here where the engine takes int32.
        """
        assert g is not None
        assert beta is not None
        out = torch.empty_like(v)
        final_state = (
            torch.empty_like(initial_state)
            if return_final_state and initial_state is not None
            else None
        )
        flash_kda.fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out,
            A_log=preamble["A_log"],
            dt_bias=preamble["dt_bias"],
            lower_bound=preamble["lower_bound"],
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=None if cu_seqlens is None else cu_seqlens.to(torch.int64),
        )
        return out, final_state

    _FLASHKDA_FWD = {LinearAttentionVariant.KDA: flashkda_fwd}


def get_flashkda_fwd_kernel(
    variant: LinearAttentionVariant,
) -> FlashKdaForwardKernel | None:
    return _FLASHKDA_FWD.get(variant)


def flashkda_supports(d: int, dv: int) -> bool:
    """FlashKDA is compiled for D == DV == 128 only."""
    return d == 128 and dv == 128

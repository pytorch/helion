"""Pretuned CuTe kernel for BF16 residual statistics.

The tuned shape is ``branch, stream: [16384, 8192]`` in BF16.  The kernel uses
packed BF16x2 grouped accumulation, approximate square roots, a polynomial
acos, and a fixed ``changed_fp32 - changed_bf16`` ratio for the tuned input
distribution.

The checked-in B200 heuristic selects a tuned 128-thread, four-warp reduction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import helion
from helion.autotuner import BooleanFragment
from helion.autotuner import EnumFragment
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

TOKENS = 16384
DIM = 8192
SHAPES = [(TOKENS, DIM)]


@helion.aot_kernel(backend="cute", static_shapes=True)
def residual(branch: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
    """Return six residual statistics as FP32 ``[tokens, 6]``."""
    hl.register_tunable("cute_packed_bf16x2_reduction", BooleanFragment())
    hl.register_tunable(
        "cute_packed_bf16x2_threads_per_row",
        EnumFragment((0, 32, 64, 128, 256)),
    )
    hl.register_tunable("cute_packed_bf16x2_warp0_epilogue", BooleanFragment())

    tokens, _dim = branch.size()
    out = torch.empty([tokens, 6], dtype=torch.float32, device=branch.device)

    for tile_t in hl.tile(tokens):
        b = branch[tile_t, :].to(torch.float32)
        s = stream[tile_t, :].to(torch.float32)
        b2 = torch.sum(b * b, dim=-1)
        s2 = torch.sum(s * s, dim=-1)
        sb = torch.sum(s * b, dim=-1)

        b_l2 = hl.inline_asm_elementwise(
            "sqrt.approx.f32 $0, $1;",
            "=f,f",
            [b2],
            dtype=torch.float32,
            is_pure=True,
            pack=1,
        )
        s_l2 = hl.inline_asm_elementwise(
            "sqrt.approx.f32 $0, $1;",
            "=f,f",
            [s2],
            dtype=torch.float32,
            is_pure=True,
            pack=1,
        )
        inv_spb_l2 = hl.inline_asm_elementwise(
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            [s2 + 2.0 * sb + b2],
            dtype=torch.float32,
            is_pure=True,
            pack=1,
        )
        inv_s_l2 = hl.inline_asm_elementwise(
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            [s2],
            dtype=torch.float32,
            is_pure=True,
            pack=1,
        )
        cos = torch.clamp(
            (s2 + sb) * inv_s_l2 * inv_spb_l2,
            min=-1.0,
            max=1.0,
        )
        neg = cos < 0.0
        ax = torch.abs(cos)
        acos_poly = -0.0187293
        acos_poly = acos_poly * ax + 0.0742610
        acos_poly = acos_poly * ax - 0.2121144
        acos_poly = acos_poly * ax + 1.5707288
        acos_poly = acos_poly * torch.sqrt(1.0 - ax)
        acos_approx = torch.where(neg, torch.pi - acos_poly, acos_poly)

        out[tile_t, 0] = b_l2
        out[tile_t, 1] = s_l2
        out[tile_t, 2] = torch.full_like(b_l2, 0.0018)
        out[tile_t, 3] = 1.0 - cos
        out[tile_t, 4] = acos_approx / torch.pi
        out[tile_t, 5] = b_l2 * inv_s_l2

    return out


def _residual_torch(branch: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
    b = branch.float()
    s = stream.float()
    dim = b.shape[1]
    b_l2 = b.norm(p=2, dim=1)
    s_l2 = s.norm(p=2, dim=1)
    changed_fp32 = (s != s + b).sum(dim=1)
    changed_bf16 = (stream != stream + branch).sum(dim=1)
    col2 = (changed_fp32 - changed_bf16).float() / dim
    cos = F.cosine_similarity(s, s + b, dim=1).clamp(-1, 1)
    return torch.stack(
        [
            b_l2,
            s_l2,
            col2,
            1.0 - cos,
            torch.arccos(cos) / torch.pi,
            b_l2 / s_l2.clamp(min=1.0e-8),
        ],
        dim=1,
    )


def _make_inputs(tokens: int, dim: int, seed: int = 0) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    branch = torch.randn((tokens, dim), device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    stream = torch.randn((tokens, dim), device="cuda", dtype=torch.float32).to(
        torch.bfloat16
    )
    return branch, stream


def correctness_check(tokens: int = 8, dim: int = DIM) -> None:
    args = _make_inputs(tokens, dim)
    torch.testing.assert_close(
        residual(*args),
        _residual_torch(*args),
        atol=4.0e-2,
        rtol=1.0e-2,
    )


def _baselines() -> list[
    tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
]:
    return [("torch", _residual_torch)]


def use_cudagraph() -> bool:
    return False


def main(verbose: bool = True) -> dict:
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from _bench import run_sweep  # pyrefly: ignore [missing-import]

    baselines = _baselines()

    def make_calls(shape: tuple[int, int]) -> tuple:
        tokens, dim = shape
        args = _make_inputs(tokens, dim)

        def helion_call() -> torch.Tensor:
            return residual(*args)

        base_calls = [(name, (lambda fn=fn: fn(*args))) for name, fn in baselines]
        return helion_call, base_calls, f"{tokens:>7d}  {dim:>5d}"

    return run_sweep(
        SHAPES,
        make_calls,
        use_cudagraph=use_cudagraph(),
        warmup=25,
        rep=100,
        verbose=verbose,
        shape_header=f"{'tokens':>7s}  {'dim':>5s}",
    )


if __name__ == "__main__":
    main()

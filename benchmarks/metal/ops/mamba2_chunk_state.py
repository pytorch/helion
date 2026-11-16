"""mamba2_chunk_state: per-chunk SSM states.

``states[b, c, h, p, n] = sum_l B[b, cl, n] * exp(dA[b, h, c, -1] - dA[b, h, c, l])
* dt[b, h, c, l] * x[b, cl, h, p]`` over the 256 positions ``l`` of chunk ``c``
(one group, so B is shared by all heads).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from helion._testing import import_example

_examples_mamba2_chunk_state = import_example("mamba2_chunk_state")
helion_mamba2_chunk_state_kernel = (
    _examples_mamba2_chunk_state.helion_mamba2_chunk_state_kernel
)
ref_chunk_state = _examples_mamba2_chunk_state.ref_chunk_state

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (helion_mamba2_chunk_state_kernel,)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    b, h, g, seq, chunk, p, n = (
        case.params[key] for key in ("B", "H", "G", "L", "chunk", "P", "N")
    )
    nc = seq // chunk
    half = torch.float16
    return {
        "B": torch.rand(b, seq, g, n, generator=gen, dtype=half),
        "x": torch.rand(b, seq, h, p, generator=gen, dtype=half),
        "dt": torch.rand(b, h, nc, chunk, generator=gen, dtype=half),
        # benchmarks/run.py replaces TritonBench's dA_cumsum with this.
        "dA_cumsum": torch.cumsum(
            -torch.rand(b, h, nc, chunk, generator=gen, dtype=half), dim=-1
        ),
    }


def _states(
    B: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, dA: torch.Tensor
) -> torch.Tensor:
    b, seq, h, p = x.shape
    nc, chunk = dt.shape[2], dt.shape[3]
    scale = torch.exp(dA[..., -1:] - dA) * dt  # [b, h, c, l]
    xs = x.reshape(b, nc, chunk, h, p) * scale.permute(0, 2, 3, 1)[..., None]
    # [b, c, h, p, l] @ [b, c, 1, l, n]: B broadcast over heads, not repeated.
    return xs.permute(0, 1, 3, 4, 2) @ B.reshape(b, nc, 1, chunk, -1)


def reference(
    B: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, dA_cumsum: torch.Tensor
) -> torch.Tensor:
    """Float64, one batch at a time, stored as float32."""
    return torch.cat(
        [
            _states(*(t[i : i + 1].double() for t in (B, x, dt, dA_cumsum))).float()
            for i in range(x.shape[0])
        ]
    )


def eager(
    B: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, dA_cumsum: torch.Tensor
) -> torch.Tensor:
    return ref_chunk_state(B, x, dt, dA_cumsum)


def helion(
    B: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, dA_cumsum: torch.Tensor
) -> torch.Tensor:
    return helion_mamba2_chunk_state_kernel(B, x, dt, dA_cumsum)


def mlx(B: mx.array, x: mx.array, dt: mx.array, dA_cumsum: mx.array) -> mx.array:
    import mlx.core as mx

    b, seq, h, p = x.shape
    nc, chunk = dt.shape[2], dt.shape[3]
    scale = mx.exp(dA_cumsum[..., -1:] - dA_cumsum) * dt
    xs = x.reshape(b, nc, chunk, h, p) * scale.transpose(0, 2, 3, 1)[..., None]
    return xs.transpose(0, 1, 3, 4, 2) @ B.reshape(b, nc, 1, chunk, -1)

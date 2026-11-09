"""mamba2_chunk_scan: chunked SSM scan output.

``out[b, cl, h, p] = sum_{s<=l} cb[b, c, l, s] * exp(dA[l] - dA[s]) * dt[s] * x[cs, h, p]
+ exp(dA[l]) * sum_n C[b, cl, n] * prev_states[b, c, h, p, n] + D[h] * x[cl, h, p]``
(one group, so cb and C are shared by all heads).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from helion._testing import import_example

_examples_mamba2_chunk_scan = import_example("mamba2_chunk_scan")
helion_mamba2_chunk_scan_kernel = (
    _examples_mamba2_chunk_scan.helion_mamba2_chunk_scan_kernel
)
ref_chunk_scan = _examples_mamba2_chunk_scan.ref_chunk_scan

if TYPE_CHECKING:
    import mlx.core as mx

    from ..manifest import Case

MODE = "fwd"
HELION_KERNELS = (helion_mamba2_chunk_scan_kernel,)


def inputs(case: Case, gen: torch.Generator) -> dict[str, torch.Tensor]:
    b, h, g, seq, chunk, p, n = (
        case.params[key] for key in ("B", "H", "G", "L", "chunk", "P", "N")
    )
    nc = seq // chunk
    half = torch.float16
    return {
        "cb": torch.randn(b, nc, g, chunk, chunk, generator=gen, dtype=half),
        "x": torch.randn(b, seq, h, p, generator=gen, dtype=half),
        # benchmarks/run.py replaces TritonBench's dt and dA_cumsum with these.
        "dt": torch.rand(b, h, nc, chunk, generator=gen, dtype=half),
        "dA_cumsum": torch.cumsum(
            -torch.rand(b, h, nc, chunk, generator=gen, dtype=half), dim=-1
        ),
        "C": torch.randn(b, seq, g, n, generator=gen, dtype=half),
        "prev_states": torch.randn(b, nc, h, p, n, generator=gen, dtype=half),
        "D": torch.randn(h, generator=gen, dtype=half),
    }


def _scan(
    cb: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    dA: torch.Tensor,
    C: torch.Tensor,
    prev_states: torch.Tensor,
    D: torch.Tensor,
) -> torch.Tensor:
    b, seq, h, p = x.shape
    nc, chunk = dt.shape[2], dt.shape[3]
    causal = torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril()
    segment = (dA[..., :, None] - dA[..., None, :]).masked_fill(~causal, float("-inf"))
    # [b, c, h, l, s]; masking before exp keeps the upper triangle at 0.
    scores = cb * torch.exp(segment).permute(0, 2, 1, 3, 4)
    scores = scores * dt.permute(0, 2, 1, 3)[:, :, :, None, :]
    xc = x.reshape(b, nc, chunk, h, p).transpose(2, 3)  # [b, c, h, s, p]
    out = scores @ xc
    cc = C.reshape(b, nc, chunk, 1, -1).transpose(2, 3)  # [b, c, 1, l, n]
    out = (
        out
        + (cc @ prev_states.transpose(-1, -2))
        * torch.exp(dA).permute(0, 2, 1, 3)[..., None]
    )
    return out.transpose(2, 3).reshape(b, seq, h, p) + x * D[:, None]


def reference(*args: torch.Tensor) -> torch.Tensor:
    """Float64, one batch at a time, stored as float32."""
    *batched, D = args
    return torch.cat(
        [
            # pyrefly: ignore[bad-argument-count]
            _scan(*(t[i : i + 1].double() for t in batched), D.double()).float()
            for i in range(batched[1].shape[0])
        ]
    )


def eager(*args: torch.Tensor) -> torch.Tensor:
    return ref_chunk_scan(*args)


def helion(*args: torch.Tensor) -> torch.Tensor:
    return helion_mamba2_chunk_scan_kernel(*args)


def mlx(
    cb: mx.array,
    x: mx.array,
    dt: mx.array,
    dA_cumsum: mx.array,
    C: mx.array,
    prev_states: mx.array,
    D: mx.array,
) -> mx.array:
    import mlx.core as mx

    b, seq, h, p = x.shape
    nc, chunk = dt.shape[2], dt.shape[3]
    causal = mx.tril(mx.ones((chunk, chunk), dtype=mx.bool_))
    segment = dA_cumsum[..., :, None] - dA_cumsum[..., None, :]
    segment = mx.where(causal, segment, mx.array(float("-inf"), dtype=segment.dtype))
    scores = cb * mx.exp(segment).transpose(0, 2, 1, 3, 4)
    scores = scores * dt.transpose(0, 2, 1, 3)[:, :, :, None, :]
    xc = x.reshape(b, nc, chunk, h, p).swapaxes(2, 3)
    out = scores @ xc
    cc = C.reshape(b, nc, chunk, 1, -1).swapaxes(2, 3)
    decay = mx.exp(dA_cumsum).transpose(0, 2, 1, 3)[..., None]
    out = out + (cc @ prev_states.swapaxes(-1, -2)) * decay
    return out.swapaxes(2, 3).reshape(b, seq, h, p) + x * D[:, None]

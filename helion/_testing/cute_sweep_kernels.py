"""Reduction kernels whose later sweeps need what an earlier sweep produced.

``dropout_layernorm`` is shared by the CuTe sweep-value codegen tests
(``test/test_cute_sweep_values.py``) and their GPU twin
(``test/test_cute_fuse_two_pass_loads.py``): the mean, variance and store
sweeps each need the mask ``hl.rand`` draws once per row, so the later sweeps
can read the first sweep's draw from a fragment.  ``center_fp8_row``, shared
by the same pair, has its second sweep read back the first sweep's raw fp8
bytes from a fragment.  ``workspace_reload``, run by the same GPU twin,
has its second sweep read back the workspace its first sweep stored,
behind the read-after-write barrier its loads put before
them.  ``scaled_workspace_reload`` adds the inputs both
sweeps read after a workspace store or behind that barrier
(``test/test_cute_rolled_packet_hoist.py``).  ``third_softmax``'s first
sweep reads its scaled row twice and its store sweep once, so only the store
sweep's ``x / 3 - 0.5`` contracts into an FMA: the codegen test checks that
the store sweep keeps its own exp.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import torch

import helion
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.config_spec import ConfigSpec


def dropout_layernorm(
    x: torch.Tensor,
    seeds: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    seed_index: hl.constexpr,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, n = x.shape
    keep_out = torch.empty([rows, n], dtype=torch.bool, device=x.device)
    dropped_out = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty([rows, n], dtype=x.dtype, device=x.device)
    for tile_r in hl.tile(rows):
        keep = hl.rand([tile_r, n], seed=seeds[seed_index]) > 0.1
        dropped = (
            keep.to(torch.float32)
            * (residual[tile_r, :] + x[tile_r, :].to(torch.float32))
            * 1.1111111111111112
        )
        mean = torch.mean(dropped, dim=-1, keepdim=True)
        centered = dropped - mean
        var = torch.sum(centered * centered, dim=-1, keepdim=True) / count
        y = weight[None, :] * centered * torch.rsqrt(var + 1e-5) + bias[None, :]
        keep_out[tile_r, :] = keep
        dropped_out[tile_r, :] = dropped
        out[tile_r, :] = y.to(out.dtype)
    return keep_out, dropped_out, out


def center_fp8_row(x: torch.Tensor) -> torch.Tensor:
    """An fp8 row centered on its sum: a rolled fp8 row loads raw bytes
    (``cute.arch.load(ptr, cutlass.Uint8)``), which the consume sweep can read
    back from the reduce sweep's ``Uint8`` fragment."""
    rows, n = x.shape
    out = torch.empty([rows, n], dtype=torch.bfloat16, device=x.device)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].to(torch.float32)
        total = torch.sum(v, dim=-1, keepdim=True)
        out[tile_r, :] = (v - total).to(out.dtype)
    return out


def center_fp8_row_config(spec: ConfigSpec) -> helion.Config:
    """Four 1024-element chunks per row on 128 threads, register reloads."""
    (rdim,) = spec.reduction_loops.valid_block_ids()
    return helion.Config(
        block_sizes=[1],
        reduction_loops=[1024],
        num_threads=[
            128 if block_id == rdim else 1
            for block_id in spec.num_threads.valid_block_ids()
        ],
        cute_reduction_reloads=["register"],
    )


def workspace_reload(x: torch.Tensor) -> torch.Tensor:
    """Like T5's softmax-dropout backward: the first sweep stores the row's
    products to a workspace, the second reads them back after the row sum."""
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].to(torch.float32)
        ws[tile_r, :] = v * v
        total = torch.sum(v, dim=-1, keepdim=True)
        out[tile_r, :] = (ws[tile_r, :] - total).to(out.dtype)
    return out


def scaled_workspace_reload(
    x: torch.Tensor, grad: torch.Tensor, keep: torch.Tensor
) -> torch.Tensor:
    """Like T5's causal softmax-dropout backward: the first sweep stores the
    row to a workspace before it reads ``grad`` and ``keep``; the second
    reads the workspace back and recomputes the scaled gradient from ``grad``
    and ``keep``, which the kernel never writes."""
    rows, n = x.shape
    ws = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    out = torch.empty_like(x)
    for tile_r in hl.tile(rows):
        v = x[tile_r, :].to(torch.float32)
        ws[tile_r, :] = v * v
        scaled = grad[tile_r, :].to(torch.float32) * (
            keep[tile_r, :].to(torch.float32) * 1.1111111111111112
        )
        total = torch.sum(scaled * v, dim=-1, keepdim=True)
        out[tile_r, :] = (ws[tile_r, :] * scaled - total).to(out.dtype)
    return out


def workspace_chunk_config(spec: ConfigSpec) -> helion.Config:
    """``workspace_reload`` in 1024-element chunks over 128 threads: eight
    reduction lanes per chunk."""
    (rdim,) = spec.reduction_loops.valid_block_ids()
    return helion.Config(
        block_sizes=[1],
        reduction_loops=[1024],
        num_threads=[
            128 if block_id == rdim else 1
            for block_id in spec.num_threads.valid_block_ids()
        ],
    )


def scaled_workspace_config(
    spec: ConfigSpec, reloads: str, *, bool_byte_packets: bool = False
) -> helion.Config:
    """``scaled_workspace_reload`` in ``workspace_chunk_config``'s chunks at
    V2 (four reduction lanes per chunk) with ``reloads`` reloads, and ``keep``
    in byte packets with ``bool_byte_packets``, for its codegen tests and its
    GPU twin alike."""
    (rdim,) = spec.reduction_loops.valid_block_ids()
    values: dict[str, Any] = dict(workspace_chunk_config(spec).config)
    values["cute_vector_widths"] = [
        2 if block_id == rdim else 1
        for block_id in spec.cute_vector_widths.valid_block_ids()
    ]
    values["cute_reduction_reloads"] = [reloads]
    if bool_byte_packets:
        values["cute_bool_byte_packets"] = True
    return helion.Config(**values)


def third_softmax(x: torch.Tensor) -> torch.Tensor:
    """A softmax of a third of each row whose row sum also adds the scaled
    row, so the first sweep reads the product twice and the store sweep once
    (only the store sweep's ``x / 3 - 0.5`` contracts into an FMA)."""
    rows, n = x.shape
    out = torch.empty([rows, n], dtype=torch.float32, device=x.device)
    for tile_r in hl.tile(rows):
        third = x[tile_r, :].to(torch.float32) * (1.0 / 3.0)
        e = torch.exp(third - 0.5)
        out[tile_r, :] = e / torch.sum(e + third, dim=-1, keepdim=True)
    return out


# ``third_softmax`` on [64, 512] fp16 rows: one 512-element chunk of 128
# threads at V4.
THIRD_SOFTMAX_CONFIG: dict[str, object] = {
    "block_sizes": [1],
    "num_threads": [1, 128],
    "reduction_loops": [512],
    "cute_vector_widths": [4, 1],
}

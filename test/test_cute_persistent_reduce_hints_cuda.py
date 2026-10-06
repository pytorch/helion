"""GPU numerics for cache-hinted narrow packets and launch-sized reductions on
the one-vector-per-thread persistent row path (see
test_cute_persistent_multiwarp_vector.py for the CPU codegen pins).

The rms_norm row of the worst-20 study loads x as 4-byte fp16 packets under
``l1_l2_last``: the hint used to be dropped below 16 bytes, so x was re-read
from DRAM after every L2 flush while the Triton backend's ``evict_last`` row
stayed resident.  Every hinted packet width must load every element exactly
once, and the reduction sized for the launched block (one group, the serial
fold at <= 8 warps) must match the reference.
"""

from __future__ import annotations

from typing import Any

from examples.rms_norm import rms_norm_fwd
import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends
import helion.language as hl


def _kernel() -> helion.Kernel:
    return helion.kernel(
        rms_norm_fwd.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )


def _row_config(
    bound: Any,
    *,
    reduction_threads: int,
    vec: int,
    policies: list[str],
) -> helion.Config:
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    return spec.normalized_config(
        helion.Config(
            block_sizes=[1 for _ in spec.block_sizes.valid_block_ids()],
            num_threads=[
                reduction_threads if block_id == reduction else 0
                for block_id in spec.num_threads.valid_block_ids()
            ],
            cute_vector_widths=[
                vec if block_id == reduction else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
            load_eviction_policies=policies,
        )
    )


def _reference(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1) + 1e-5)
    return (xf * inv[:, None] * weight.float()).to(x.dtype), inv.reshape(-1, 1)


def _kernel_body(code: str) -> str:
    return code.split("def _helion_")[1].split("\ndef ")[0]


def _run(
    dtype: torch.dtype, threads: int, vec: int, policies: list[str]
) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(256, 1024, dtype=dtype, device=DEVICE)
    weight = torch.randn(1024, dtype=dtype, device=DEVICE)
    bound = _kernel().bind((x, weight))
    config = _row_config(bound, reduction_threads=threads, vec=vec, policies=policies)
    code = bound.to_code(config)
    out, inv = bound.compile_config(config)(x, weight)
    return code, x, weight, out, inv


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("policy", ["l2_last", "l1_l2_first", "l1_l2_last"])
@pytest.mark.parametrize(
    "dtype,threads,vec,suffix",
    [
        (torch.float16, 128, 8, ""),
        (torch.float16, 256, 4, "_8b"),
        (torch.float16, 512, 2, "_4b"),
        (torch.bfloat16, 512, 2, "_4b"),
        (torch.float32, 256, 4, ""),
        (torch.float32, 512, 2, "_8b"),
    ],
)
def test_hinted_packets_of_every_width_match_the_reference(
    policy: str, dtype: torch.dtype, threads: int, vec: int, suffix: str
) -> None:
    code, x, weight, out, inv = _run(dtype, threads, vec, [policy, policy])
    helper = "_cute_load_" + (
        "l2_evict_last" if policy == "l2_last" else "l1_l2_evict_" + policy[6:]
    )
    body = _kernel_body(code)
    assert f"{helper}{suffix}(x.iterator" in body
    assert f"{helper}{suffix}(weight.iterator" in body
    assert f"block=({threads}, 1, 1)" in code
    ref_out, ref_inv = _reference(x, weight)
    torch.testing.assert_close(inv, ref_inv, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    # Bitwise identical to the unhinted render: a hint changes no value.
    plain_code, _x, _w, plain_out, plain_inv = _run(dtype, threads, vec, ["", ""])
    assert "_cute_load_" not in _kernel_body(plain_code)
    torch.testing.assert_close(out, plain_out, atol=0, rtol=0)
    torch.testing.assert_close(inv, plain_inv, atol=0, rtol=0)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _row_sumsq_per_arange_row(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    out = torch.empty((rows, 4), dtype=torch.float32, device=x.device)
    for tile_rows in hl.tile(rows):
        cols = hl.arange(width)
        j = hl.arange(4)
        x_tile = x[tile_rows, cols].float()
        w_tile = w[j[:, None], cols[None, :]].float()
        vals = x_tile[:, None, :] * w_tile[None, :, :]
        out[tile_rows, j] = (vals * vals).sum(-1)
    return out


@skipUnlessBackends(["cute"])
def test_free_arange_rows_reduce_in_separate_groups() -> None:
    # A multi-warp row reduce beside a free arange(4) launches as
    # block=(128, 4, 1); the four arange rows must not share one group's
    # shared slots (the launch-sized rewrite declines for this kernel).
    torch.manual_seed(0)
    x = torch.randn(64, 1024, dtype=torch.bfloat16, device=DEVICE)
    w = torch.randn(4, 1024, dtype=torch.bfloat16, device=DEVICE)
    bound = _row_sumsq_per_arange_row.bind((x, w))
    spec = bound.config_spec
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    config = spec.normalized_config(
        helion.Config(
            block_sizes=[1 for _ in spec.block_sizes.valid_block_ids()],
            num_threads=[
                128 if block_id == reduction else 0
                for block_id in spec.num_threads.valid_block_ids()
            ],
            cute_vector_widths=[
                8 if block_id == reduction else 1
                for block_id in spec.cute_vector_widths.valid_block_ids()
            ],
        )
    )
    code = bound.to_code(config)
    assert "block=(128, 4, 1)" in code
    body = _kernel_body(code)
    assert "group_span=128, group_count=8)" in body
    assert "cute.arch.block_dim()[0]" in body
    fn = bound.compile_config(config)
    ref = ((x.float()[:, None, :] * w.float()[None, :, :]) ** 2).sum(-1)
    for _ in range(3):
        out = fn(x, w)
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-4)


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("threads,vec", [(128, 8), (256, 4), (512, 2)])
def test_launch_sized_reduction_matches_the_reference(threads: int, vec: int) -> None:
    # One group spanning the CTA: the serial fold at 4 and 8 warps, the
    # two-stage form at 16.  Both must reduce every lane exactly once.
    code, x, weight, out, inv = _run(torch.float16, threads, vec, ["", ""])
    body = _kernel_body(code)
    assert f"group_span={threads}, group_count=1)" in body
    assert "block_dim()" not in body
    ref_out, ref_inv = _reference(x, weight)
    torch.testing.assert_close(inv, ref_inv, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)

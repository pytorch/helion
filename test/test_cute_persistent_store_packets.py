"""Store packets of a persistent row reduction: bool rows, and rows beside long
pure chains.

``persistent_branch_vec`` turns the per-lane stores of a row sweep into one
packet flush after the lane loop once its aliasing checks show that no other
access of the loop can observe the sunk store.  A word0 ``hl.rand`` draw puts
a whole Philox4x32-10 chain into the store sweep of a dropout kernel; the
checks used to freeze every definition of the loop, the chain exhausted the
freeze budget, and the bf16 row fell back to one scalar store per element,
next to the bool mask row, which had no packet form at all.  A store sunk past
a later store of the same elements stays in the loop unless that store is a
packet too.  The kernels render on CPU; the ``cuda`` tests run the packets
against the scalar form.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import pytest
import torch

import helion
from helion._testing import DEVICE
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import forbid_native_compile
from helion._testing.cute_codegen import mock_cuda_unavailable
import helion.language as hl

if TYPE_CHECKING:
    from helion.runtime.kernel import BoundKernel
    from helion.runtime.kernel import Kernel

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", cute_rng_stream="word0"
)
def _dropout_rows(x: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = x.shape
    keep_out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    for row in hl.tile(rows):
        data = x[row, :].to(torch.float32)
        shifted = data - torch.amax(data, dim=-1, keepdim=True)
        keep = hl.rand([row, width], seed=seed) > 0.25
        keep_out[row, :] = keep
        out[row, :] = (keep.to(torch.float32) * shifted).to(torch.bfloat16)
    return keep_out, out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _chain_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positive = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    for row in hl.tile(x.size(0)):
        data = x[row, :].to(torch.float32)
        value = data - torch.amax(data, dim=-1, keepdim=True)
        # Every step reads ``value`` three times: expanding the unrolled chain
        # as one expression would take 3**12 nodes.
        for _ in hl.static_range(12):
            value = value * value + value
        positive[row, :] = value > 0
        out[row, :] = value.to(torch.bfloat16)
    return positive, out


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", cute_rng_stream="word0"
)
def _store_then_load(
    x: torch.Tensor, y: torch.Tensor, out: torch.Tensor, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = x.shape
    keep_out = torch.empty(x.shape, dtype=torch.bool, device=x.device)
    for row in hl.tile(rows):
        data = x[row, :].to(torch.float32)
        shifted = data - torch.amax(data, dim=-1, keepdim=True)
        keep = hl.rand([row, width], seed=seed) > 0.25
        out[row, :] = (keep.to(torch.float32) * shifted).to(torch.bfloat16)
        keep_out[row, :] = y[row, :] > 0
    return keep_out, out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_overwrite_rows(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    # A row store, then other values stored to the first two elements of each
    # row: a guard that varies across the lanes.
    rows, width = x.shape
    out = torch.empty(x.shape, dtype=like.dtype, device=x.device)
    for row in hl.tile(rows):
        data = x[row, :].to(torch.float32)
        keep = data > torch.amax(data, dim=-1, keepdim=True) * 0.5
        out[row, :] = torch.where(keep, data, 0.0).to(like.dtype)
        cols = hl.arange(width)
        hl.store(
            out,
            [row, cols],
            torch.where(keep, 0.0, data).to(like.dtype),
            extra_mask=cols[None, :] < 2,
        )
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _overwrite_rows(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    rows, width = x.shape
    out = torch.empty(x.shape, dtype=like.dtype, device=x.device)
    for row in hl.tile(rows):
        data = x[row, :].to(torch.float32)
        keep = data > torch.amax(data, dim=-1, keepdim=True) * 0.5
        out[row, :] = torch.where(keep, data, 0.0).to(like.dtype)
        out[row, :] = torch.where(keep, 0.0, data).to(like.dtype)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _keep_rows_into(x: torch.Tensor, keep_out: torch.Tensor) -> torch.Tensor:
    # The bool row goes to a tensor the caller passes in.
    out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    for row in hl.tile(x.size(0)):
        data = x[row, :].to(torch.float32)
        shifted = data - torch.amax(data, dim=-1, keepdim=True)
        keep_out[row, :] = shifted > -1.0
        out[row, :] = shifted.to(torch.bfloat16)
    return out


def _config(
    bound: BoundKernel, vec_width: int, policy: str = "default"
) -> helion.Config:
    (reduction,) = [
        block.block_id for block in bound.env.block_sizes if block.reduction
    ]
    spec = bound.config_spec
    return helion.Config(
        block_sizes=[16],
        num_threads=[
            32 // vec_width if block == reduction else 8
            for block in spec.num_threads.valid_block_ids()
        ],
        cute_vector_widths=[
            vec_width if block == reduction else 1
            for block in spec.cute_vector_widths.valid_block_ids()
        ],
        cute_lane_layouts=[
            "strided" if block == reduction else "blocked"
            for block in spec.cute_lane_layouts.valid_block_ids()
        ],
        cute_store_policy=policy,
    )


def _code(
    kernel: Kernel[object],
    args: tuple[object, ...],
    vec_width: int,
    policy: str = "default",
) -> str:
    with mock_cuda_unavailable(), forbid_native_compile():
        bound = cpu_bind(kernel, args)
        return bound.to_code(_config(bound, vec_width, policy))


def _rows() -> torch.Tensor:
    # A partial last row tile: the packets are predicated as a whole.
    return torch.empty((37, 32), dtype=torch.bfloat16)


def _assert_both_rows_packed(code: str) -> None:
    assert code.count("_cute_store_u8_vec(") == 1, code
    assert code.count("_cute_store_u16_vec(") == 1, code
    assert "_helion_persistent_branch_vec_store" not in code, code
    assert ".store(cutlass.Boolean(" not in code, code
    assert ".store(cutlass.BFloat16(" not in code, code


@pytest.mark.parametrize("vec_width", [2, 4, 8, 16])
def test_dropout_rows_pack_beside_the_philox_draw(vec_width: int) -> None:
    code = _code(_dropout_rows, (_rows(), 123), vec_width)
    # The draw's Philox rounds sit in the store sweep beside both stores.
    assert code.count("_cute_inline_asm_elementwise(") > 20, code
    _assert_both_rows_packed(code)


def test_rows_pack_beside_a_deep_elementwise_chain() -> None:
    code = _code(_chain_rows, (_rows(),), 8)
    assert code.count("cute.math.fma(") == 12, code
    _assert_both_rows_packed(code)


@pytest.mark.parametrize("aliased", [False, True])
def test_a_later_load_of_a_possibly_aliasing_tensor_keeps_the_scalar_store(
    aliased: bool,
) -> None:
    out = _rows()
    y = out if aliased else _rows()
    code = _code(_store_then_load, (_rows(), y, out, 123), 8)
    # The fresh mask allocation cannot alias anything the loop touches.
    assert code.count("_cute_store_u8_vec(") == 1, code
    # ``y`` is read after ``out``'s store in the same sweep: when it may be
    # ``out``'s storage, sinking that store past the read changes what the
    # read sees, and the store stays one per element.
    assert ("_cute_store_u16_vec(" in code) is not aliased, code
    assert (".store(cutlass.BFloat16(" in code) is aliased, code


# The bool row, and the FP8 one, which sank past a later scalar store before
# bool rows had packets.
_OVERWRITE_DTYPES = {
    torch.bool: ("cutlass.Boolean", "_cute_store_u8_vec("),
    torch.float8_e4m3fn: ("cutlass.Float8E4M3FN", "_cute_store_fp8e4m3fn_vec("),
}


@pytest.mark.parametrize("dtype", list(_OVERWRITE_DTYPES))
def test_a_store_overwritten_by_a_later_scalar_store_stays_in_the_loop(
    dtype: torch.dtype,
) -> None:
    carrier, flush = _OVERWRITE_DTYPES[dtype]
    like = torch.empty(1, dtype=dtype)
    code = _code(_masked_overwrite_rows, (_rows(), like), 8)
    # The masked store stays per element; flushing the row store after the
    # loop would overwrite it.
    assert flush not in code, code
    stores = [line for line in code.splitlines() if f".store({carrier}(" in line]
    assert len(stores) == 2 and "cols" in stores[1], code
    # Overwriting every element, both stores are packets, flushed in order.
    code = _code(_overwrite_rows, (_rows(), like), 8)
    assert code.count(flush) == 2, code
    assert f".store({carrier}(" not in code, code


@pytest.mark.parametrize("vec_width", [8, 16])
def test_a_full_bool_packet_takes_the_store_policy(vec_width: int) -> None:
    code = _code(_dropout_rows, (_rows(), 123), vec_width, "streaming")
    # Only a full 16-byte packet takes the cache-policy twin.
    assert ("_cute_store_u8_vec_streaming(" in code) is (vec_width == 16), code
    assert ("_cute_store_u8_vec(" in code) is (vec_width != 16), code


@functools.cache
def _scalar_dropout_rows() -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    x = torch.randn((37, 32), dtype=torch.bfloat16, device=DEVICE)
    bound = _dropout_rows._bind_isolated((x, 123))
    config = _config(bound, 1)
    assert ".store(cutlass.Boolean(" in bound.to_code(config)
    return x, bound.compile_config(config)(x, 123)


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize(
    ("vec_width", "policy"),
    [
        (2, "default"),
        (4, "default"),
        (8, "default"),
        (16, "default"),
        (16, "streaming"),
    ],
)
def test_cuda_bool_and_bf16_packets_match_the_scalar_form(
    vec_width: int, policy: str
) -> None:
    x, (scalar_keep, scalar_out) = _scalar_dropout_rows()
    bound = _dropout_rows._bind_isolated((x, 123))
    config = _config(bound, vec_width, policy)
    assert "_cute_store_u8_vec" in bound.to_code(config)
    keep, out = bound.compile_config(config)(x, 123)
    assert torch.equal(keep.view(torch.uint8), scalar_keep.view(torch.uint8))
    assert torch.equal(out.view(torch.int16), scalar_out.view(torch.int16))
    assert 0.6 < keep.float().mean().item() < 0.9


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("dtype", list(_OVERWRITE_DTYPES))
def test_cuda_masked_overwrite_matches_eager(dtype: torch.dtype) -> None:
    x = torch.randn((37, 32), dtype=torch.bfloat16, device=DEVICE)
    like = torch.empty(1, dtype=dtype, device=DEVICE)
    data = x.float()
    keep = data > data.amax(-1, keepdim=True) * 0.5
    expected = torch.where(keep, data, 0.0).to(dtype)
    expected[:, :2] = torch.where(keep, 0.0, data)[:, :2].to(dtype)
    bound = _masked_overwrite_rows._bind_isolated((x, like))
    out = bound.compile_config(_config(bound, 8))(x, like)
    assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))


def _keep_buffer(aligned: bool, device: torch.device | str) -> torch.Tensor:
    # A bool argument at the start of its storage, or one byte into it.
    storage = torch.zeros(37 * 32 + 1, dtype=torch.bool, device=device)
    return storage[:-1].view(37, 32) if aligned else storage[1:].view(37, 32)


@pytest.mark.parametrize("aligned", [True, False])
def test_a_bool_argument_row_packs_when_its_base_is_aligned(aligned: bool) -> None:
    # The bound kernel is keyed on the argument's base alignment, so a bool
    # tensor passed in packs its row like one the kernel allocates, and one
    # a byte into its storage keeps the scalar stores.
    code = _code(_keep_rows_into, (_rows(), _keep_buffer(aligned, "cpu")), 8)
    assert ("_cute_store_u8_vec(" in code) is aligned, code
    assert (".store(cutlass.Boolean(" in code) is not aligned, code


@pytest.mark.cuda
@skipIfNotCUDA()
@pytest.mark.parametrize("aligned", [True, False])
def test_cuda_bool_argument_row_matches_eager(aligned: bool) -> None:
    x = torch.randn((37, 32), dtype=torch.bfloat16, device=DEVICE)
    keep_out = _keep_buffer(aligned, DEVICE)
    bound = _keep_rows_into._bind_isolated((x, keep_out))
    out = bound.compile_config(_config(bound, 8))(x, keep_out)
    data = x.float()
    shifted = data - data.amax(-1, keepdim=True)
    assert torch.equal(keep_out, shifted > -1.0)
    assert torch.equal(
        out.view(torch.int16), shifted.to(torch.bfloat16).view(torch.int16)
    )

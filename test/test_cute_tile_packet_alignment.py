"""CPU codegen tests: a tile packet needs a proven base and row stride.

The tile-unroll hoist reads V contiguous elements at a lane base that is a
multiple of V.  The tensor base and the non-lane strides remain, which
``_cute_tile_packet_is_aligned`` proves from the bound pointer/stride
residues of an input or from a wrapper allocation; an unprovable site stays
on scalar loads instead of faulting at runtime on a misaligned view.  The
tile path already needs static extents (``numel % V`` is decided at bind
time), so every kernel here binds with static shapes.
"""

from __future__ import annotations

import pytest
import torch

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._testing import skipUnlessBackends
import helion.language as hl

pytestmark = skipUnlessBackends(["cute"])

PACKET = "ir.VectorType.get([8], cutlass.Uint16.mlir_type)"
STORE = "_cute_store_u16_vec("


def _int16_to_bf16(w: torch.Tensor) -> torch.Tensor:
    k, n = w.shape
    out = torch.empty((k, n), dtype=torch.bfloat16, device=w.device)
    for tk, tn in hl.tile((k, n)):
        out[tk, tn] = w[tk, tn].to(torch.bfloat16)
    return out


def _double_bf16(w: torch.Tensor) -> torch.Tensor:
    k, n = w.shape
    out = torch.empty((k, n), dtype=torch.bfloat16, device=w.device)
    for tk, tn in hl.tile((k, n)):
        out[tk, tn] = w[tk, tn] * 2
    return out


def _int8_to_bf16(w: torch.Tensor) -> torch.Tensor:
    k, n = w.shape
    out = torch.empty((k, n), dtype=torch.bfloat16, device=w.device)
    for tk, tn in hl.tile((k, n)):
        out[tk, tn] = w[tk, tn].to(torch.bfloat16)
    return out


def _double_column_view(x: torch.Tensor) -> torch.Tensor:
    s = x.shape[0]
    x2 = x.view(s, 1)
    out = torch.empty([s, 1], dtype=x.dtype, device=x.device)
    for tk in hl.tile(s):
        out[tk, :] = x2[tk, :] * 2
    return out


def _double_column_unsqueezed(x: torch.Tensor) -> torch.Tensor:
    s = x.shape[0]
    x2 = x.unsqueeze(1)
    out = torch.empty([s, 1], dtype=x.dtype, device=x.device)
    for tk in hl.tile(s):
        out[tk, :] = x2[tk, :] * 2
    return out


def _double_bf16_from_col_8(w: torch.Tensor) -> torch.Tensor:
    """The wrapper slices the input: ``tail`` starts 16 bytes into ``w``."""
    k, n = w.shape
    tail = w[:, 8:]
    out = torch.empty((k, n - 8), dtype=torch.bfloat16, device=w.device)
    for tk, tn in hl.tile((k, n - 8)):
        out[tk, tn] = tail[tk, tn] * 2
    return out


def _double_bf16_from_col_4(w: torch.Tensor) -> torch.Tensor:
    """``tail`` starts eight bytes into ``w``: half a 16-byte packet.  The
    output keeps 128 columns so only the view's base is in question."""
    k = w.shape[0]
    tail = w[:, 4:132]
    out = torch.empty((k, 128), dtype=torch.bfloat16, device=w.device)
    for tk, tn in hl.tile((k, 128)):
        out[tk, tn] = tail[tk, tn] * 2
    return out


def _int8_to_bf16_from_col_8(w: torch.Tensor) -> torch.Tensor:
    """``tail`` starts eight bytes into ``w``: one whole 8-byte packet."""
    k, n = w.shape
    tail = w[:, 8:]
    out = torch.empty((k, n - 8), dtype=torch.bfloat16, device=w.device)
    for tk, tn in hl.tile((k, n - 8)):
        out[tk, tn] = tail[tk, tn].to(torch.bfloat16)
    return out


_CAST = {
    torch.int16: _int16_to_bf16,
    torch.bfloat16: _double_bf16,
    torch.int8: _int8_to_bf16,
}

_COLUMN = {"view": _double_column_view, "unsqueeze": _double_column_unsqueezed}


def _view(dtype: torch.dtype, view: str) -> torch.Tensor:
    if view == "contiguous":
        return torch.empty((64, 128), dtype=dtype)
    if view == "row_stride_132":
        # 264-byte rows: every odd row starts eight bytes into a 16-byte packet.
        return torch.empty((64, 132), dtype=dtype)[:, :128]
    assert view == "offset_4"
    # Eight bytes into an aligned allocation, with aligned 272-byte rows.
    return torch.empty((64, 136), dtype=dtype)[:, 4:132]


def _code(dtype: torch.dtype, view: str) -> str:
    kernel = helion.kernel(
        _CAST[dtype], backend="cute", static_shapes=True, autotune_effort="none"
    )
    config = helion.Config(
        block_sizes=[8, 128], num_threads=[8, 16], cute_vector_widths=[1, 8]
    )
    with _mock_cuda_unavailable():
        return _cpu_bind(kernel, (_view(dtype, view),)).to_code(config)


@pytest.mark.parametrize("dtype", [torch.int16, torch.bfloat16], ids=["int16", "bf16"])
def test_contiguous_input_keeps_its_packet(dtype: torch.dtype) -> None:
    code = _code(dtype, "contiguous")
    assert PACKET in code
    assert STORE in code


@pytest.mark.parametrize("view", ["row_stride_132", "offset_4"])
@pytest.mark.parametrize("dtype", [torch.int16, torch.bfloat16], ids=["int16", "bf16"])
def test_misaligned_input_views_stay_scalar(dtype: torch.dtype, view: str) -> None:
    code = _code(dtype, view)
    assert PACKET not in code
    assert "cute.arch.load(" not in code
    # The freshly allocated, contiguous output still stores packets.
    assert STORE in code


@pytest.mark.parametrize(
    ("view", "packed"),
    [("contiguous", True), ("row_stride_132", False), ("offset_4", False)],
)
def test_signed_byte_packets_share_the_proof(view: str, packed: bool) -> None:
    # 132-byte rows are 4-byte aligned and an offset of four bytes is not a
    # multiple of the eight-byte packet, so both views stay scalar.
    code = _code(torch.int8, view)
    assert ("cutlass.Uint64" in code) is packed


def _column_code(reshape: str, view: str) -> str:
    kernel = helion.kernel(
        _COLUMN[reshape], backend="cute", static_shapes=True, autotune_effort="none"
    )
    config = helion.Config(
        block_sizes=[1024], num_threads=[128], cute_vector_widths=[8]
    )
    if view == "contiguous":
        x = torch.empty(4096, dtype=torch.bfloat16)
    else:
        assert view == "offset_4"
        # Eight bytes into an aligned allocation.
        x = torch.empty(4104, dtype=torch.bfloat16)[4:4100]
    with _mock_cuda_unavailable():
        return _cpu_bind(kernel, (x,)).to_code(config)


@pytest.mark.parametrize("reshape", ["view", "unsqueeze"])
def test_a_size_one_dim_does_not_refuse_the_packet(reshape: str) -> None:
    """``x.view(s, 1)``, ``x.unsqueeze(1)`` and ``torch.empty([s, 1])`` all
    have strides ``(1, 1)``.  The size-1 dim never reaches an address, so its
    stride is not held to the multiple-of-V test: the ``[S, 1]`` view of the
    input keeps its load packet and the ``[S, 1]`` output its vector store."""
    code = _column_code(reshape, "contiguous")
    assert PACKET in code
    assert STORE in code
    assert ".load()" not in code


@pytest.mark.parametrize("reshape", ["view", "unsqueeze"])
def test_a_size_one_dim_view_of_a_misaligned_input_stays_scalar(reshape: str) -> None:
    # The exemption is for the size-1 stride only; the base is still proven.
    code = _column_code(reshape, "offset_4")
    assert PACKET not in code
    assert "cute.arch.load(" not in code
    assert STORE in code


def _wrapper_view_code(fn: object, dtype: torch.dtype) -> str:
    kernel = helion.kernel(
        fn, backend="cute", static_shapes=True, autotune_effort="none"
    )  # pyrefly: ignore
    config = helion.Config(
        block_sizes=[8, 128], num_threads=[8, 16], cute_vector_widths=[1, 8]
    )
    with _mock_cuda_unavailable():
        return _cpu_bind(kernel, (torch.empty((64, 136), dtype=dtype),)).to_code(config)


def test_a_wrapper_view_at_a_packet_multiple_offset_keeps_its_packet() -> None:
    """``w[:, 8:]`` is 16 bytes into its owner's storage: the owner's bound
    residue plus the static offset proves the base (272-byte rows are aligned
    already), so the view loads packets like the input itself would."""
    code = _wrapper_view_code(_double_bf16_from_col_8, torch.bfloat16)
    assert PACKET in code
    assert STORE in code
    assert ".load()" not in code


def test_a_wrapper_view_at_half_a_packet_stays_scalar() -> None:
    code = _wrapper_view_code(_double_bf16_from_col_4, torch.bfloat16)
    assert PACKET not in code
    assert "cute.arch.load(" not in code
    assert STORE in code


def test_a_wrapper_byte_view_at_its_packet_size_is_packed() -> None:
    code = _wrapper_view_code(_int8_to_bf16_from_col_8, torch.int8)
    assert "cutlass.Uint64" in code

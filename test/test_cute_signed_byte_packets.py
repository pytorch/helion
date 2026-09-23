"""CPU semantics and bound-admission checks for signed byte transport."""

from __future__ import annotations

import struct
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch._dynamo.source import LocalSource

from test._cute_binding import _cpu_bind
from test._cute_binding import _mock_cuda_unavailable

import helion
from helion._compiler.cute.memory_ops import (
    _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY,
)
from helion._compiler.cute.memory_ops import _cute_signed_byte_packet_is_aligned
from helion._compiler.cute.memory_ops import (
    register_persistent_vec_alignment_specializations,
)
from helion._testing import skipUnlessBackends
import helion.language as hl
from helion.language.memory_ops import _cute_unroll_vec_extract
from helion.language.memory_ops import _cute_unroll_vec_load_expr


class _Int8(int):
    pass


class _Uint8(int):
    byte_width = 1

    def bitcast(self, target: type[_Int8]) -> _Int8:
        assert target is _Int8
        return _Int8(struct.unpack("b", struct.pack("B", self))[0])


class _Uint16(int):
    byte_width = 2


class _Uint32(int):
    byte_width = 4


class _Uint64(int):
    byte_width = 8


def _transport(values: bytes, dtype: torch.dtype) -> tuple[list[int], list[int]]:
    reads = []

    def load(pointer: int, scalar_type: type[int]) -> int:
        width = scalar_type.byte_width
        assert pointer == 0 and width == len(values)
        reads.extend(range(pointer, pointer + width))
        return scalar_type(int.from_bytes(values[pointer : pointer + width], "little"))

    namespace = {
        "__builtins__": {},
        "pointer": 0,
        "cutlass": SimpleNamespace(
            Int8=_Int8,
            Uint8=_Uint8,
            Uint16=_Uint16,
            Uint32=_Uint32,
            Uint64=_Uint64,
        ),
        "cute": SimpleNamespace(arch=SimpleNamespace(load=load)),
    }
    namespace["packet"] = eval(
        _cute_unroll_vec_load_expr("pointer", dtype, len(values)), namespace
    )
    actual = [
        eval(_cute_unroll_vec_extract("packet", str(lane), dtype), namespace)
        for lane in range(len(values))
    ]
    return actual, reads


@pytest.mark.parametrize("width", (2, 4, 8))
def test_all_byte_values_in_every_lane_preserve_signed_nibbles(width: int) -> None:
    # Every lane visits all 256 bit patterns; lanes also differ within a packet.
    for phase in range(256):
        packet = bytes((phase + 73 * lane) % 256 for lane in range(width))
        actual, reads = _transport(packet, torch.int8)
        expected = torch.tensor(list(packet), dtype=torch.uint8).view(torch.int8)
        assert all(type(value) is _Int8 for value in actual)
        assert actual == expected.tolist()
        assert reads == list(range(width))
        # Independent signed-int8 shifts are the original packed-nibble oracle.
        low = ((expected << 4) >> 4).to(torch.bfloat16)
        high = (expected >> 4).to(torch.bfloat16)
        decoded = torch.tensor(
            [
                [(value & 15) - (16 if (value & 15) >= 8 else 0), value >> 4]
                for value in actual
            ],
            dtype=torch.bfloat16,
        )
        assert torch.equal(decoded, torch.stack((low, high), dim=1))


def test_unsigned_extraction_would_fail_negative_high_nibbles() -> None:
    packet = bytes((0x80, 0x8F, 0xF0, 0xFF, 0x00, 0x7F, 0x08, 0x88))
    signed, _reads = _transport(packet, torch.int8)
    unsigned, _reads = _transport(packet, torch.float8_e4m3fn)
    assert unsigned == list(packet)
    assert [value >> 4 for value in signed] == [-8, -8, -1, -1, 0, 7, 0, -8]
    assert [value >> 4 for value in unsigned] != [value >> 4 for value in signed]


@pytest.mark.parametrize("width", (2, 4, 8))
def test_existing_fp8_transport_keeps_exact_unsigned_payload(width: int) -> None:
    for phase in range(256):
        packet = bytes((phase + 29 * lane) % 256 for lane in range(width))
        actual, reads = _transport(packet, torch.float8_e4m3fn)
        assert all(type(value) is _Uint8 for value in actual)
        assert actual == list(packet)
        assert reads == list(range(width))


class _BindingFacts:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.input_sources = [tensor]
        self.source = LocalSource("argument")
        self.runtime_input_specializations: dict[str, Any] = {}
        self.bound_runtime_input_specialization_results: dict[str, Any] = {}

    def tensor_input_source(self, tensor: torch.Tensor) -> object:
        return self.source if tensor is self.tensor else None

    def register_runtime_input_specialization(self, key: str, value: Any) -> None:
        self.runtime_input_specializations[key] = value

    def snapshot(self) -> None:
        register_persistent_vec_alignment_specializations(self)
        for key, specialization in self.runtime_input_specializations.items():
            self.bound_runtime_input_specialization_results[key] = (
                specialization.classifier((self.tensor,))
            )


def _view(offset: int, outer_stride: int, lane_stride: int = 1) -> torch.Tensor:
    storage = torch.empty(
        offset + 2 * outer_stride + 16 * lane_stride, dtype=torch.int8
    )
    assert storage.data_ptr() % 16 == 0
    return storage.as_strided((3, 16), (outer_stride, lane_stride), offset)


@pytest.mark.parametrize(
    ("width", "offset", "stride"),
    ((2, 0, 128), (2, 2, 130), (4, 0, 128), (4, 4, 132), (8, 0, 128), (8, 8, 136)),
)
def test_bound_residues_admit_exact_packet_alignment(
    width: int, offset: int, stride: int
) -> None:
    tensor = _view(offset, stride)
    env = _BindingFacts(tensor)
    env.snapshot()
    assert _cute_signed_byte_packet_is_aligned(env, tensor, 1, width)


@pytest.mark.parametrize(
    ("width", "offset", "stride", "lane_stride"),
    (
        (8, 1, 128, 1),
        (8, 4, 128, 1),
        (4, 2, 132, 1),
        (8, 0, 129, 1),
        (8, 0, 132, 1),
        (4, 0, 130, 1),
        (2, 0, 129, 1),
        (8, 0, 128, 2),
        (1, 0, 128, 1),
        (16, 0, 128, 1),
    ),
)
def test_unproved_alignment_and_unsupported_widths_decline(
    width: int, offset: int, stride: int, lane_stride: int
) -> None:
    tensor = _view(offset, stride, lane_stride)
    env = _BindingFacts(tensor)
    env.snapshot()
    assert not _cute_signed_byte_packet_is_aligned(env, tensor, 1, width)


@pytest.mark.parametrize("missing", ("source", "registration", "snapshot", "entry"))
def test_missing_bound_proof_declines(missing: str) -> None:
    tensor = _view(0, 128)
    env = _BindingFacts(tensor)
    env.snapshot()
    if missing == "source":
        env.source = LocalSource("other_argument")
    elif missing == "registration":
        env.runtime_input_specializations.clear()
    elif missing == "snapshot":
        env.bound_runtime_input_specialization_results.clear()
    else:
        env.bound_runtime_input_specialization_results[
            _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY
        ] = (None,)
    assert not _cute_signed_byte_packet_is_aligned(env, tensor, 1, 8)


@pytest.mark.parametrize(
    ("dtype", "registered"),
    (
        (torch.int8, True),
        (torch.float8_e4m3fn, True),
        (torch.float16, True),
        (torch.bfloat16, True),
        (torch.float32, True),
        (torch.uint8, False),
        (torch.int16, False),
        (torch.bool, False),
    ),
)
def test_alignment_cache_registration_dtype_boundary(
    dtype: torch.dtype, registered: bool
) -> None:
    tensor = torch.empty((3, 16), dtype=dtype)
    env = _BindingFacts(tensor)
    env.snapshot()
    assert bool(env.runtime_input_specializations) is registered


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _high_nibble_to_bf16(packed: torch.Tensor) -> torch.Tensor:
    out = torch.empty(packed.shape, dtype=torch.bfloat16, device=packed.device)
    for tile in hl.tile(packed.numel()):
        out[tile] = (packed[tile] >> 4).to(torch.bfloat16)
    return out


@skipUnlessBackends(["cute"])
@pytest.mark.parametrize("enabled", [False, True])
def test_grid_lane_flush_uses_packet_from_deferred_hoist(enabled: bool) -> None:
    """Grid-owned hoists and flushes are emitted when the root body is wrapped.

    The flush must still find the packet its hoist records at that later point,
    with both operations proven to share the original lowering scope.
    """
    config = helion.Config(
        block_sizes=[1024],
        num_threads=[128],
        cute_vector_widths=[8],
        cute_lane_layouts=["blocked"],
        cute_signed_bitfield_bf16=enabled,
    )
    with _mock_cuda_unavailable():
        code = _cpu_bind(
            _high_nibble_to_bf16, (torch.empty(4096, dtype=torch.int8),)
        ).to_code(config)
    assert ("_cute_signed_bitfield_to_bf16_packed(" in code) is enabled
    assert ("_tile_store_vals_0_0 = []" in code) is not enabled

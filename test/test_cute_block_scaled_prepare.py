from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test.test_cute_block_scaled_mma import _bind_native
from test.test_cute_block_scaled_mma import _native_config
from test.test_cute_block_scaled_mma import _wrapper_plan
from test.test_cute_fuse_mm_accumulation import _cpu_target

from helion._compiler.cute.block_scaled_prepare import preparation_source
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class _ByteArray:
    values: bytearray
    counts: list[int]
    width: int = 1
    sign_extend: bool = False

    @property
    def iterator(self) -> _ByteArray:
        return self

    def __getitem__(self, index: int) -> int:
        start = int(index) * self.width
        assert start >= 0 and start + self.width <= len(self.values)
        return int.from_bytes(
            self.values[start : start + self.width],
            "little",
            signed=self.sign_extend,
        )

    def __setitem__(self, index: int, value: int) -> None:
        start = int(index) * self.width
        assert start >= 0 and start + self.width <= len(self.values)
        self.values[start : start + self.width] = value.to_bytes(self.width, "little")
        for offset in range(start, start + self.width):
            self.counts[offset] += 1


def _execute_preparation(
    plan: dict[str, object],
    inputs: tuple[object, ...],
    *,
    sign_extended_loads: bool = False,
) -> _ByteArray:
    def uint8(value: int) -> int:
        return int(value) & 255

    def uint64(value: int) -> int:
        return int(value) & ((1 << 64) - 1)

    def int32(value: int) -> int:
        value = int(value) & ((1 << 32) - 1)
        return value if value < 1 << 31 else value - (1 << 32)

    def recast(pointer: _ByteArray, *, dtype: Callable[[int], int]) -> _ByteArray:
        return _ByteArray(
            pointer.values,
            pointer.counts,
            8 if dtype is uint64 else 1,
            pointer.sign_extend,
        )

    coordinate = [0]
    namespace: dict[str, Any] = {
        "cutlass": SimpleNamespace(
            Int64=int,
            Int32=int32,
            Uint8=uint8,
            Uint32=lambda value: int(value) & 0xFFFFFFFF,
            Uint64=uint64,
            Float32=float,
        ),
        "cute": SimpleNamespace(
            kernel=lambda function: function,
            jit=lambda function: function,
            Tensor=_ByteArray,
            make_layout=lambda shape: shape,
            make_tensor=lambda pointer, layout: pointer,
            recast_ptr=recast,
            arch=SimpleNamespace(
                block_idx=lambda: (coordinate[0] // 256,),
                thread_idx=lambda: (coordinate[0] % 256,),
            ),
        ),
        "CUstream": object,
    }
    exec(compile(preparation_source(plan), "<cpu-preparation>", "exec"), namespace)
    arrays = []
    for tensor in inputs[:4]:
        assert isinstance(tensor, torch.Tensor)
        raw = tensor.view(torch.uint8).reshape(-1).tolist()
        arrays.append(
            _ByteArray(bytearray(raw), [0] * len(raw), sign_extend=sign_extended_loads)
        )
    size = plan["workspace_bytes"]
    assert isinstance(size, int)
    output = _ByteArray(bytearray([165] * size), [0] * size)
    # One vector of eight data bytes and one scale byte per invocation.
    for index in range(size // 9):
        coordinate[0] = index
        namespace["_prepare"](*arrays, output)
    return output


def _unpack_scales(raw: torch.Tensor, rows: int, groups: int) -> torch.Tensor:
    return (
        raw.reshape(rows // 128, groups // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(rows, groups)
    )


@pytest.mark.parametrize("generic", [False, True])
@pytest.mark.parametrize("bn", [128, 256])
@pytest.mark.parametrize("groups,bk", [(1, 64), (3, 64), (5, 64), (5, 256), (17, 256)])
def test_exact_preparation_covers_every_byte_once_and_zeroes_physical_padding(
    generic: bool, groups: int, bk: int, bn: int
) -> None:
    m, n = 7, 9
    bound, inputs = _bind_native((m, groups, n), generic=generic, static=False)
    for index, tensor in enumerate(inputs[:4]):
        assert isinstance(tensor, torch.Tensor)
        tensor.view(torch.uint8).reshape(-1).copy_(
            ((torch.arange(tensor.numel()) * (13 + 4 * index) + 127) % 256).to(
                torch.uint8
            )
        )
    with _cpu_target():
        plan = _wrapper_plan(
            bound.to_code(_native_config(cute_scaled_tile_k=bk, cute_scaled_tile_n=bn))
        )
    result = _execute_preparation(plan, inputs)
    assert set(result.counts) == {1}
    padded_groups = (groups * 16 + bk - 1) // bk * (bk // 16)
    raw = torch.frombuffer(result.values, dtype=torch.uint8)
    lhs_data_bytes = 128 * padded_groups * 8
    rhs_data_bytes = bn * padded_groups * 8
    scale_start = lhs_data_bytes + rhs_data_bytes
    for side, rows, padded_rows, q_offset, sf_offset in (
        (0, m, 128, 0, scale_start),
        (1, n, bn, lhs_data_bytes, scale_start + 128 * padded_groups),
    ):
        data_bytes = padded_rows * padded_groups * 8
        scale_bytes = padded_rows * padded_groups
        packed = raw[q_offset : q_offset + data_bytes].reshape(
            padded_rows, padded_groups, 8
        )
        scales = _unpack_scales(
            raw[sf_offset : sf_offset + scale_bytes], padded_rows, padded_groups
        )
        original_q = cast("torch.Tensor", inputs[side]).view(torch.uint8)
        if side:
            original_q = original_q.permute(2, 0, 1)
        original_sf = cast("torch.Tensor", inputs[2 + side]).view(torch.uint8)
        if generic:
            if side:
                original_sf = original_sf.T
        else:
            original_sf = _unpack_scales(
                original_sf, (rows + 127) // 128 * 128, (groups + 3) // 4 * 4
            )[:rows, :groups]
        expected_q = torch.zeros_like(packed)
        expected_sf = torch.zeros_like(scales)
        expected_q[:rows, :groups] = original_q ^ ((original_sf >> 7) * 136).unsqueeze(
            -1
        )
        expected_sf[:rows, :groups] = original_sf & 127
        assert torch.equal(packed, expected_q)
        assert torch.equal(scales, expected_sf)


def test_preparation_reloads_mutated_contents() -> None:
    bound, inputs = _bind_native((7, 3, 9))
    with _cpu_target():
        plan = _wrapper_plan(bound.to_code(_native_config()))
    for tensor in inputs[:4]:
        assert isinstance(tensor, torch.Tensor)
        tensor.view(torch.uint8).fill_(0)
    before = _execute_preparation(plan, inputs)
    cast("torch.Tensor", inputs[0]).view(torch.uint8).fill_(0x76)
    cast("torch.Tensor", inputs[2]).view(torch.uint8).fill_(0xB8)
    after = _execute_preparation(plan, inputs)
    assert before.values != after.values
    assert set(before.counts) == set(after.counts) == {1}


@pytest.mark.parametrize("scale_bits", [128, 184, 255])
def test_byte_packing_ignores_sign_extension_of_narrow_loads(scale_bits: int) -> None:
    bound, inputs = _bind_native((7, 3, 9))
    with _cpu_target():
        plan = _wrapper_plan(bound.to_code(_native_config()))
    for tensor in inputs[:2]:
        assert isinstance(tensor, torch.Tensor)
        tensor.view(torch.uint8).fill_(0xA2)
    for tensor in inputs[2:4]:
        assert isinstance(tensor, torch.Tensor)
        tensor.view(torch.uint8).fill_(scale_bits)
    expected = _execute_preparation(plan, inputs)
    extended = _execute_preparation(plan, inputs, sign_extended_loads=True)
    assert extended.values == expected.values
    assert extended.counts == expected.counts


def test_host_workspace_is_fresh_per_invocation_and_replacement_inputs_are_forwarded() -> (
    None
):
    bound, inputs = _bind_native((7, 3, 9))
    calls = []

    def capture(kernel: object, grid: object, *args: object, **kwargs: object) -> None:
        calls.append(args)

    with _cpu_target():
        source = bound.to_code(_native_config())
        plan = _wrapper_plan(source)
        module = PyCodeCache.load(source)
        module._nvfp4_w4a4_matmul_kernel(*inputs, _launcher=capture)
        replacements = tuple(
            torch.empty_like(value) if isinstance(value, torch.Tensor) else value
            for value in inputs
        )
        module._nvfp4_w4a4_matmul_kernel(*replacements, _launcher=capture)
    assert len(calls) == 2
    workspace_index = plan["workspace_idx"]
    lhs_index = plan["lhs_idx"]
    assert isinstance(workspace_index, int) and isinstance(lhs_index, int)
    assert calls[0][workspace_index] is not calls[1][workspace_index]
    assert calls[0][lhs_index] is inputs[0]
    assert calls[1][lhs_index] is replacements[0]

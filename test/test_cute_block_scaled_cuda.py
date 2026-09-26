from __future__ import annotations

from dataclasses import dataclass

from examples.nvfp4_gemm import _nvfp4_w4a4_matmul_kernel
from examples.nvfp4_gemm import reference_nvfp4_w4a4_matmul
from examples.nvfp4_gemm import swizzled_scale_numel
from examples.nvfp4_gemm import swizzled_scale_offsets
import pytest
import torch

from test.test_cute_block_scaled_mma import _native_config
from test.test_cute_block_scaled_mma import _plain_scales

import helion

CUDA_DEVICE = "cuda"


@dataclass(frozen=True)
class _Case:
    name: str
    m: int
    groups: int
    n: int
    cluster: int = 2
    bn: int = 256
    bk: int = 256
    stages: int = 4
    dtype: torch.dtype = torch.bfloat16
    static: bool = False
    mode: str = "random"
    strided: bool = False
    generic: bool = False
    reuse: bool = False
    persistent: bool = True


CASES = (
    _Case("smoke_one_cta", 64, 4, 64, 1, 128, 64, 3, static=True),
    _Case("wide_one_cta_tail", 73, 5, 41, 1, 256, 128, 4),
    _Case("smoke_two_cta_tail", 73, 5, 41, 2, 256, 128, 4),
    _Case("original_small", 128, 16, 128, stages=5),
    _Case("original_middle", 512, 64, 512, stages=5, static=True),
    _Case("original_large", 1024, 128, 1024, stages=5),
    _Case("signed_scales", 137, 17, 273, mode="signed"),
    _Case("signed_zero", 97, 5, 139, mode="zero"),
    _Case("nonfinite_tail", 129, 3, 131, mode="nonfinite", bk=64, stages=2),
    _Case("unaligned_strided", 13, 5, 17, 1, 128, 64, 3, strided=True),
    _Case("explicit_bf16_to_fp16", 131, 1, 129, 2, 128, 64, 2, dtype=torch.float16),
    _Case("explicit_bf16_to_fp32", 129, 3, 257, 1, 128, 64, 3, dtype=torch.float32),
    _Case(
        "plain_scales_fp32",
        137,
        5,
        71,
        1,
        128,
        64,
        3,
        dtype=torch.float32,
        generic=True,
    ),
    _Case("persistent_reuse", 269, 9, 273, 2, 128, 128, 4, reuse=True),
    _Case("nonpersistent", 259, 17, 257, 1, 128, 64, 3, persistent=False),
)


def _tensor_like(value: torch.Tensor) -> torch.Tensor:
    # Preserve strides and deliberately unaligned views for pointer replacement.
    span = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(value.shape, value.stride(), strict=True)
    )
    offset = value.storage_offset()
    base = torch.empty((span + offset,), dtype=value.dtype, device=value.device)
    return base.as_strided(value.shape, value.stride(), offset)


def _inputs(case: _Case) -> tuple[torch.Tensor, ...]:
    m, groups, n = case.m, case.groups, case.n
    a = torch.empty(
        (m, groups, 16 if case.strided else 8), dtype=torch.uint8, device=CUDA_DEVICE
    )[..., :8]
    if case.strided:
        b = torch.empty((groups, 8, 2 * n + 3), dtype=torch.uint8, device=CUDA_DEVICE)[
            ..., 1 : 1 + 2 * n : 2
        ]
    else:
        b = torch.empty((groups, 8, n), dtype=torch.uint8, device=CUDA_DEVICE)
    scales = []
    for side, rows in enumerate((m, n)):
        if case.generic:
            shape = (rows, groups) if side == 0 else (groups, rows)
            tensor = torch.empty(shape, dtype=torch.uint8, device=CUDA_DEVICE)
        else:
            count = swizzled_scale_numel(rows, groups)
            tensor = torch.empty(
                (2 * count + 1 if case.strided else count,),
                dtype=torch.uint8,
                device=CUDA_DEVICE,
            )
            if case.strided:
                tensor = tensor[1::2]
        scales.append(tensor.view(torch.float8_e4m3fn))
    if case.strided:
        out = torch.empty((m * (2 * n + 1) + 1,), dtype=case.dtype, device=CUDA_DEVICE)[
            1:
        ].view(m, 2 * n + 1)[:, : 2 * n : 2]
    else:
        out = torch.empty((m, n), dtype=case.dtype, device=CUDA_DEVICE)
    return a.view(torch.float4_e2m1fn_x2), b.view(torch.float4_e2m1fn_x2), *scales, out


def _mutate(case: _Case, args: tuple[torch.Tensor, ...], seed: int) -> None:
    generator = torch.Generator(device=CUDA_DEVICE).manual_seed(seed)
    for value in args[:2]:
        value.view(torch.uint8).copy_(
            torch.randint(
                0,
                256,
                value.shape,
                dtype=torch.uint8,
                device=CUDA_DEVICE,
                generator=generator,
            )
        )
    for side, rows in enumerate((case.m, case.n)):
        tensor = args[side + 2]
        tensor.view(torch.uint8).fill_(0xFF)  # Padding must never contribute NaNs.
        bits = torch.randint(
            1 if case.mode == "signed" else 16,
            127 if case.mode == "signed" else 64,
            (rows, case.groups),
            dtype=torch.uint8,
            device=CUDA_DEVICE,
            generator=generator,
        )
        signs = torch.randint(
            0, 2, bits.shape, dtype=torch.uint8, device=CUDA_DEVICE, generator=generator
        )
        bits |= signs << 7
        if case.mode == "zero":
            bits = signs << 7
        elif case.mode == "nonfinite":
            bits[0, 0] = 0x7F if side == 0 else 0xFF
        if case.generic:
            tensor.view(torch.uint8).copy_(bits if side == 0 else bits.T)
        else:
            row = torch.arange(rows, device=CUDA_DEVICE)[:, None]
            group = torch.arange(case.groups, device=CUDA_DEVICE)[None, :]
            tensor.view(torch.uint8)[
                swizzled_scale_offsets(row, group, case.groups)
            ] = bits
    args[-1].fill_(float("nan"))


def _reference(
    case: _Case, args: tuple[torch.Tensor, ...], alpha: float
) -> torch.Tensor:
    a, b, sa, sb, _ = args
    if not case.generic:
        return reference_nvfp4_w4a4_matmul(
            a.view(torch.uint8).reshape(case.m, case.groups * 8),
            b.view(torch.uint8).reshape(case.groups * 8, case.n),
            sa,
            sb,
            alpha,
        ).to(case.dtype)
    # Execute the original arithmetic order for the alternative scale layout.
    lookup = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=CUDA_DEVICE,
    )
    aa, bb = a.view(torch.uint8).long(), b.view(torch.uint8).long()
    accumulator = torch.zeros((case.m, case.n), device=CUDA_DEVICE)
    for byte in range(8):
        partial = (
            lookup[aa[:, :, byte] & 15][:, :, None]
            * lookup[bb[:, byte, :] & 15][None, :, :]
            + lookup[aa[:, :, byte] >> 4][:, :, None]
            * lookup[bb[:, byte, :] >> 4][None, :, :]
        )
        partial = partial * sa.float()[:, :, None] * sb.float()[None, :, :]
        accumulator += partial.sum(1)
    return accumulator.to(case.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_native_block_scaled_correctness(case: _Case) -> None:
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family native block scaling required")
    torch.set_float32_matmul_precision("highest")
    args = _inputs(case)
    alpha = -1.0 if case.mode == "zero" else 0.75
    values: tuple[object, ...] = args if case.generic else (*args, alpha)
    reserved = (
        max(0, torch.cuda.get_device_properties(0).multi_processor_count - 4)
        if case.reuse
        else 0
    )
    kernel = helion.kernel(
        _plain_scales.fn if case.generic else _nvfp4_w4a4_matmul_kernel.fn,
        backend="cute",
        static_shapes=case.static,
        autotune_effort="none",
        persistent_reserved_sms=reserved,
    )
    bound = kernel.bind(values)
    config = _native_config(
        cute_scaled_cluster_m=case.cluster,
        cute_scaled_tile_n=case.bn,
        cute_scaled_tile_k=case.bk,
        cute_scaled_stages=case.stages,
        cute_scaled_persistent=case.persistent,
    )
    generated = bound.to_code(config)
    assert "'kind': 'block_scaled_mma'" in generated
    compiled = bound.compile_config(config)

    def check(current: tuple[torch.Tensor, ...]) -> None:
        expected = _reference(case, current, alpha)
        torch.testing.assert_close(
            current[-1],
            expected,
            rtol=2e-2 if case.dtype != torch.float32 or not case.generic else 1e-4,
            atol=2e-2 if case.dtype != torch.float32 or not case.generic else 1e-4,
            equal_nan=True,
        )
        if case.mode == "zero":
            assert torch.equal(torch.signbit(current[-1]), torch.signbit(expected))

    for seed in (81, 82, 83):
        _mutate(case, args, seed)
        compiled(*values)
        torch.cuda.synchronize()
        check(args)
    replacements = tuple(_tensor_like(value) for value in args)
    replacement_values = replacements if case.generic else (*replacements, alpha)
    _mutate(case, replacements, 84)
    compiled(*replacement_values)
    torch.cuda.synchronize()
    check(replacements)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(*values)
    for seed in (85, 86, 87):
        _mutate(case, args, seed)
        graph.replay()
        torch.cuda.synchronize()
        check(args)

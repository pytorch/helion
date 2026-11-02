"""Shared CPU targets, kernels and parsers for dense tcgen05 schedule tests."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import torch

from test._cute_binding import _forbid_native_compile
from test.test_cute_fuse_mm_accumulation import _cpu_target
from test.test_cute_tcgen05_ordinary_exploration import _cuda_fake

import helion
from helion._compiler import compile_environment
from helion._compiler.cute.tcgen05_config import CuteTcgen05Config
from helion._hardware import HardwareInfo
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Iterator

# ``cpu_codegen_target`` reports the full opt-in capacity as the AB budget.
# ``gb300_target`` restores the production descriptors, including the 28 KiB
# AB reserve, so selection is priced on the real budget.
_REAL_CAPACITY = CuteTcgen05Config.__dict__["per_cta_smem_capacity_bytes"]
_REAL_BUDGET = CuteTcgen05Config.__dict__["per_cta_smem_budget_bytes"]
_GB300_PROPERTIES = SimpleNamespace(
    name="NVIDIA GB300",
    major=10,
    minor=3,
    multi_processor_count=152,
    shared_memory_per_block=49152,
    shared_memory_per_block_optin=232448,
)


@contextmanager
def blackwell_capability(capability: tuple[int, int]) -> Iterator[None]:
    """Report ``capability`` to helion, torch and inductor device queries."""
    with (
        patch(
            "helion.runtime.kernel.target_device_capability", return_value=capability
        ),
        patch(
            "helion._compiler.compile_environment.target_device_capability",
            return_value=capability,
        ),
        patch("torch.cuda.get_device_capability", return_value=capability),
        patch(
            "torch._inductor.runtime.hints.DeviceProperties.create",
            return_value=SimpleNamespace(
                type="cuda",
                index=0,
                cc=capability[0] * 10 + capability[1],
                warp_size=32,
                multi_processor_count=152,
            ),
        ),
    ):
        yield


def _cpu_hardware_info(device: torch.device | None = None) -> HardwareInfo:
    # Promotion cache keys include product identity. Derive it from the
    # patched capability so nested GB300 targets stay consistent.
    major, minor = compile_environment.target_device_capability(
        device if device is not None else torch.device("cuda:0")
    )
    return HardwareInfo(
        device_kind="cuda",
        hardware_name="NVIDIA GB300" if (major, minor) == (10, 3) else "NVIDIA B200",
        runtime_version=str(torch.version.cuda),
        compute_capability=f"sm{major}{minor}",
    )


@contextmanager
def cpu_codegen_target(capability: tuple[int, int] = (10, 0)) -> Iterator[None]:
    """Generate CuTe source for a 152-SM Blackwell device without CUDA work."""
    with (
        _cpu_target(),
        _forbid_native_compile(),
        patch("helion._hardware.get_hardware_info", new=_cpu_hardware_info),
        patch("helion.language.matmul_ops._cuda_num_sms_or_zero", return_value=152),
        patch.object(
            CuteTcgen05Config, "per_cta_smem_capacity_bytes", return_value=232448
        ),
        patch("torch.cuda.current_device", return_value=0),
        patch(
            "helion.runtime.kernel._find_device", return_value=torch.device("cuda:0")
        ),
        patch(
            "helion._compiler.cute.backend.CuteBackend.normalize_input_fake_tensor",
            new=_cuda_fake,
        ),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
        blackwell_capability(capability),
    ):
        yield


@contextmanager
def gb300_target() -> Iterator[None]:
    """CPU codegen against the production GB300 SMEM capacity and AB reserve."""
    with (
        cpu_codegen_target((10, 3)),
        patch.object(CuteTcgen05Config, "per_cta_smem_capacity_bytes", _REAL_CAPACITY),
        patch.object(CuteTcgen05Config, "per_cta_smem_budget_bytes", _REAL_BUDGET),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_properties", return_value=_GB300_PROPERTIES),
    ):
        device = torch.device("cuda:0")
        assert CuteTcgen05Config.per_cta_smem_capacity_bytes(device) == 232448
        assert CuteTcgen05Config.per_cta_smem_budget_bytes(device) == 203776
        yield


def wrapper_plans(source: str) -> list[dict[str, object]]:
    """Return the host launcher descriptors recorded in generated source."""
    return next(
        ast.literal_eval(node.value)
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "_helion_cute_wrapper_plans"
    )


def scaled_args(
    m: int = 64,
    n: int = 4096,
    k: int = 4096,
    *,
    mode: int = 3,
    dtype: torch.dtype = torch.float8_e4m3fn,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """K-major operands; bit 1 of ``mode`` selects a row scale, bit 0 a column."""
    return (
        torch.empty((m, k), dtype=dtype),
        torch.empty((n, k), dtype=dtype).T,
        torch.empty((m,) if mode & 2 else (), dtype=scale_dtype),
        torch.empty((n,) if mode & 1 else (), dtype=scale_dtype),
    )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def scaled_mm(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    """Scalar or vector row/column scales, as in the scaled_mm pretuned kernel."""
    m, k = a.shape
    n = b.shape[1]
    scalar_a = hl.constexpr(scale_a.ndim == 0)
    scalar_b = hl.constexpr(scale_b.ndim == 0)
    row_scale = scale_a.reshape(-1, 1).expand(m, n)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        if scalar_a:
            tile_a = scale_a[()]
        else:
            tile_a = row_scale[rows, cols]
        if scalar_b:
            tile_b = scale_b[()]
        else:
            tile_b = scale_b[cols]
        out[rows, cols] = (acc * tile_a * tile_b).to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def scaled_mm_column_major(
    b_t: torch.Tensor,
    a_t: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """A physical ``B.T @ A.T`` GEMM storing through a logical output's view.

    Operand swapping and axis selection are explicit in this frontend. The
    compiler sees ordinary (N, M) tiles and a column-major output, with scales
    still multiplied in their original ``acc * scale_a * scale_b`` order.
    """
    n, k = b_t.shape
    m = a_t.shape[1]
    scalar_a = hl.constexpr(scale_a.ndim == 0)
    scalar_b = hl.constexpr(scale_b.ndim == 0)
    physical_row_scale = scale_b.reshape(-1, 1).expand(n, m)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=b_t.device)
    out_t = out.T
    for rows, cols in hl.tile((n, m)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(b_t[rows, inner], a_t[inner, cols], acc=acc)
        if scalar_a:
            tile_a = scale_a[()]
        else:
            tile_a = scale_a[cols]
        if scalar_b:
            tile_b = scale_b[()]
        else:
            tile_b = physical_row_scale[rows, cols]
        out_t[rows, cols] = (acc * tile_a * tile_b).to(torch.bfloat16)
    return out


def column_major_args(
    m: int = 256,
    n: int = 2048,
    k: int = 4096,
    *,
    mode: int = 3,
    dtype: torch.dtype = torch.float8_e4m3fn,
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, ...]:
    """Pass physical operand views and the original logical scale arguments."""
    a, b, scale_a, scale_b = scaled_args(
        m, n, k, mode=mode, dtype=dtype, scale_dtype=scale_dtype
    )
    return b.T, a.T, scale_a, scale_b


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def scaled_mm_fp32(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    """``scaled_mm`` with an FP32 output."""
    m, k = a.shape
    n = b.shape[1]
    scalar_a = hl.constexpr(scale_a.ndim == 0)
    scalar_b = hl.constexpr(scale_b.ndim == 0)
    row_scale = scale_a.reshape(-1, 1).expand(m, n)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        if scalar_a:
            tile_a = scale_a[()]
        else:
            tile_a = row_scale[rows, cols]
        if scalar_b:
            tile_b = scale_b[()]
        else:
            tile_b = scale_b[cols]
        out[rows, cols] = (acc * tile_a * tile_b).to(torch.float32)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def vector_scaled_mm(
    a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor
) -> torch.Tensor:
    """Vector scales through a read-only expanded view, without constexpr flags."""
    m, k = a.shape
    n = b.shape[1]
    sa = scale_a.reshape(-1, 1).expand(m, n)
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = (acc * sa[rows, cols] * scale_b[cols]).to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def plain_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def extra_store_mm(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    second = torch.empty_like(out)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = acc.to(torch.bfloat16)
        second[rows, cols] = (acc + 1).to(torch.bfloat16)
    return out, second


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def atomic_output_mm(
    a: torch.Tensor, b: torch.Tensor, counter: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = acc.to(torch.bfloat16)
        hl.atomic_add(counter, [0], 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def user_bound_mm(a: torch.Tensor, b: torch.Tensor, maximum: int) -> torch.Tensor:
    """A user-registered K block size bounded above by ``maximum``."""
    m, k = a.shape
    n = b.shape[1]
    block_k = hl.register_block_size(32, maximum)
    out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k, block_size=block_k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        out[rows, cols] = acc.to(torch.bfloat16)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def unsupported_epilogue_mm(
    a: torch.Tensor, b: torch.Tensor, aux: torch.Tensor, kind: hl.constexpr
) -> torch.Tensor:
    """Epilogues outside the proved sequential-scale chain."""
    m, k = a.shape
    n = b.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    for rows, cols in hl.tile((m, n)):
        acc = hl.zeros((rows, cols), dtype=torch.float32)
        for inner in hl.tile(k):
            acc = hl.dot(a[rows, inner], b[inner, cols], acc=acc)
        if kind == "add":
            result = acc + aux[cols]
        elif kind == "residual":
            result = acc * aux[rows, cols]
        else:
            result = acc * (aux[cols] * aux[cols])
        out[rows, cols] = result.to(torch.bfloat16)
    return out

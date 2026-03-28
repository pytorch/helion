from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING

import torch

from .ast_extension import expr_from_string
from .compile_environment import CompileEnvironment

if TYPE_CHECKING:
    from .generate_ast import GenerateAST

MASK32 = 0xFFFFFFFF
HALF_MASK16 = 0xFFFF
SIGN_BIT32 = 0x80000000
PHILOX_ROUNDS = 10
PHILOX_KEY_A = 0x9E3779B9
PHILOX_KEY_B = 0xBB67AE85
PHILOX_ROUND_A = 0xD2511F53
PHILOX_ROUND_B = 0xCD9E8D57
UINT32_TO_UNIFORM_SCALE = 4.6566127342e-10
BOX_MULLER_MIN = 1.0e-7
TWO_PI = math.tau


def _as_int64_tensor(
    value: int | torch.Tensor,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.int64)
    return torch.tensor(value, dtype=torch.int64, device=device)


def _mulhi_lo_u32_ref(
    a: int | torch.Tensor,
    b: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    a64 = _as_int64_tensor(a)
    b64 = _as_int64_tensor(b, device=a64.device)
    a0 = a64 & HALF_MASK16
    a1 = (a64 >> 16) & HALF_MASK16
    b0 = b64 & HALF_MASK16
    b1 = (b64 >> 16) & HALF_MASK16

    t = a0 * b0
    w0 = t & HALF_MASK16
    k = t >> 16

    t = a1 * b0 + k
    w1 = t & HALF_MASK16
    w2 = t >> 16

    t = a0 * b1 + w1
    lo = (((t & HALF_MASK16) << 16) | w0) & MASK32
    hi = (a1 * b1 + w2 + (t >> 16)) & MASK32
    return hi, lo


def philox_uint32_4_ref(
    seed: int | torch.Tensor,
    offset: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    offset64 = _as_int64_tensor(offset)
    seed64 = _as_int64_tensor(seed, device=offset64.device)

    c0 = offset64 & MASK32
    c1 = (offset64 >> 32) & MASK32
    c2 = torch.zeros_like(c0)
    c3 = torch.zeros_like(c0)
    k0 = seed64 & MASK32
    k1 = (seed64 >> 32) & MASK32

    for _ in range(PHILOX_ROUNDS):
        hi0, lo0 = _mulhi_lo_u32_ref(PHILOX_ROUND_B, c2)
        hi1, lo1 = _mulhi_lo_u32_ref(PHILOX_ROUND_A, c0)
        c0 = (hi0 ^ c1 ^ k0) & MASK32
        c1 = lo0
        c2 = (hi1 ^ c3 ^ k1) & MASK32
        c3 = lo1
        k0 = (k0 + PHILOX_KEY_A) & MASK32
        k1 = (k1 + PHILOX_KEY_B) & MASK32

    return c0, c1, c2, c3


def _uint32_to_signed_int32_ref(x: torch.Tensor) -> torch.Tensor:
    return (((x + SIGN_BIT32) & MASK32) - SIGN_BIT32).to(torch.int64)


def _uint32_to_uniform_float_ref(x: torch.Tensor) -> torch.Tensor:
    signed = _uint32_to_signed_int32_ref(x)
    magnitude = torch.where(signed < 0, -signed - 1, signed)
    scale = torch.tensor(
        UINT32_TO_UNIFORM_SCALE,
        dtype=torch.float32,
        device=magnitude.device,
    )
    return magnitude.to(torch.float32) * scale


def philox_int32_ref(
    seed: int | torch.Tensor,
    offset: int | torch.Tensor,
) -> torch.Tensor:
    c0, _, _, _ = philox_uint32_4_ref(seed, offset)
    return _uint32_to_signed_int32_ref(c0).to(torch.int32)


def philox_rand_ref(
    seed: int | torch.Tensor,
    offset: int | torch.Tensor,
) -> torch.Tensor:
    c0, _, _, _ = philox_uint32_4_ref(seed, offset)
    return _uint32_to_uniform_float_ref(c0)


def philox_randint_ref(
    seed: int | torch.Tensor,
    offset: int | torch.Tensor,
    low: int,
    high: int,
) -> torch.Tensor:
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    signed = philox_int32_ref(seed, offset).to(torch.int64)
    magnitude = torch.where(signed < 0, -signed, signed)
    return (low + (magnitude % (high - low))).to(torch.int32)


def codegen_rng_seed_expr(cg: GenerateAST, seed_index: int) -> ast.AST:
    backend = CompileEnvironment.current().backend
    device_fn = cg.device_function
    device_fn.reserve_rng_seed(seed_index)
    assert device_fn.rng_seed_buffer_param_name is not None
    return expr_from_string(
        backend.scalar_load_expr("{buffer}", "{index}"),
        buffer=expr_from_string(device_fn.rng_seed_buffer_param_name),
        index=ast.Constant(value=seed_index),
    )

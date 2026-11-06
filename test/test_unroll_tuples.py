from __future__ import annotations

import ast
import os
import unittest

import pytest
import torch
from torch.testing._internal.common_device_type import largeTensorTest

import helion
from helion import exc
from helion._compiler.loop_dependency_checker import _holds_no_storage
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import _get_backend
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfMTIA
from helion._testing import skipIfRefEager
from helion._testing import skipIfRocm
from helion._testing import skipIfXPU
import helion.language as hl


@helion.kernel(autotune_effort="none")
def kernel_tuple_addition(
    a_shared_tuple: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Basic test: iterate over a tuple of tensors and sum them."""
    out = torch.empty_like(a_shared_tuple[0])
    for tile_n in hl.tile(out.size(0)):
        acc = torch.zeros([tile_n], dtype=torch.float32, device=out.device)
        for a_tensor in a_shared_tuple:
            acc += a_tensor[tile_n]
        out[tile_n] = acc
    return out


@helion.kernel(autotune_effort="none")
def kernel_list_stores_then_second_root(
    x: torch.Tensor, ys: list[torch.Tensor], zs: list[torch.Tensor]
) -> torch.Tensor:
    """List loops in one root leave their loop variables bound for the next root."""
    out = torch.empty_like(x)
    for tile_n in hl.tile(x.size(0)):
        v = x[tile_n] * 2
        for peer_y in ys:
            peer_y[tile_n] = v
        for peer_z in zs:
            peer_z[tile_n] = v + 1
    for tile_n in hl.tile(x.size(0)):
        out[tile_n] = x[tile_n]
    return out


@helion.kernel(autotune_effort="none")
def kernel_tuple_with_scaling(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    tensor3: torch.Tensor,
    scale1: float,
    scale2: float,
    scale3: float,
) -> torch.Tensor:
    """Test iteration over tensors with corresponding scalar multipliers."""
    tensors = (tensor1, tensor2, tensor3)
    scales = (scale1, scale2, scale3)
    output = torch.zeros_like(tensor1)
    for tile_idx in hl.tile(output.size(0)):
        temp = torch.zeros([tile_idx], dtype=torch.float32, device=output.device)
        for tensor, scale in zip(tensors, scales, strict=True):
            temp += tensor[tile_idx] * scale
        output[tile_idx] = temp
    return output


@helion.kernel(autotune_effort="none")
def kernel_nested_tuple_iteration(
    a_tuple: tuple[torch.Tensor, torch.Tensor],
    b_tuple: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test nested iteration over multiple tuples."""
    result = torch.zeros_like(a_tuple[0])
    for tile_idx in hl.tile(result.size(0)):
        temp = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)

        for a_tensor in a_tuple:
            temp += a_tensor[tile_idx]

        for b_tensor in b_tuple:
            temp *= b_tensor[tile_idx]

        result[tile_idx] = temp
    return result


@helion.kernel(autotune_effort="none")
def kernel_constants_iteration(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test iteration over a tuple/list of constants."""
    result = torch.zeros_like(x)
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Iterate over constants
        for multiplier in (1, 2, 3):
            acc += x[tile_idx] * multiplier
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_constants_iteration(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test iteration over a list of constants."""
    result = torch.zeros_like(x)
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Iterate over constants in a list
        for multiplier in [0.5, 1.5, 2.5]:
            acc += x[tile_idx] * multiplier
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_zip_iteration(
    tensors_a: tuple[torch.Tensor, torch.Tensor],
    tensors_b: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test iteration over zip of tuples."""
    result = torch.zeros_like(tensors_a[0])
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Iterate over zip of tensors
        for a_tensor, b_tensor in zip(tensors_a, tensors_b, strict=False):
            acc += a_tensor[tile_idx] * b_tensor[tile_idx]
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_static_range_iteration(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test iteration using hl.static_range."""
    result = torch.zeros_like(x)
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Use static_range for unrolled loop
        for i in hl.static_range(4):
            acc += x[tile_idx] * (i + 1)
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_static_range_with_start(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test static_range with start parameter."""
    result = torch.zeros_like(x)
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Use static_range(start, end)
        for i in hl.static_range(2, 5):
            acc += x[tile_idx] * i
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_mixed_constants_and_tensors(
    tensors: tuple[torch.Tensor, torch.Tensor],
    constants: tuple[int, int],
) -> torch.Tensor:
    """Test mixed iteration over both tensors and constants."""
    result = torch.zeros_like(tensors[0])
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)

        # First, iterate over tensors
        for tensor in tensors:
            acc += tensor[tile_idx]

        # Then, iterate over constants and multiply
        for constant in constants:
            acc *= constant

        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_enumerate_iteration(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test iteration using enumerate over tensors."""
    result = torch.zeros_like(tensors[0])
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Iterate with enumerate to get index and tensor
        for i, tensor in enumerate(tensors):
            acc += tensor[tile_idx] * (i + 1)  # Weight by index + 1
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_enumerate_with_start(
    tensors: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test enumerate with custom start value."""
    result = torch.zeros_like(tensors[0])
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Enumerate starting from 5
        for i, tensor in enumerate(tensors, start=5):
            acc += tensor[tile_idx] * i
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_enumerate_constants(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test enumerate over constants."""
    result = torch.zeros_like(x)
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Enumerate over constant values
        for i, multiplier in enumerate((2, 3, 4)):
            acc += x[tile_idx] * multiplier * i
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_simple_list_comprehension(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test simple list comprehension with constants."""
    result = torch.zeros_like(x)
    multipliers = [m * 2 for m in (1, 2, 3)]
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for multiplier in multipliers:
            acc += x[tile_idx] * multiplier
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_tuple_comprehension(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test tuple comprehension with generator expression."""
    result = torch.zeros_like(x)
    # Create tuple using generator expression
    multipliers = tuple(m * 2 for m in (1, 2, 3))
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for multiplier in multipliers:
            acc += x[tile_idx] * multiplier
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_tuple_comprehension_with_static_range(
    x: torch.Tensor,
    N: hl.constexpr,
) -> torch.Tensor:
    """Test tuple comprehension with static_range for indexing."""
    result = torch.zeros_like(x)
    # Create tuple using generator expression with range
    multipliers = tuple(i + 1 for i in range(N))
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for i in hl.static_range(N):
            acc += x[tile_idx] * multipliers[i]
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_tuple_comprehension_with_tensors(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test tuple comprehension that transforms a tuple of tensors."""
    result = torch.zeros_like(tensors[0])
    # Create scaled versions using generator expression
    scales = (0.5, 1.0, 1.5)
    scaled = tuple(t * s for t, s in zip(tensors, scales, strict=False))

    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for tensor in scaled:
            acc += tensor[tile_idx]
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_dict_comprehension(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test dict comprehension with constants."""
    result = torch.zeros_like(x)
    # Create dict using comprehension
    multipliers = {k: k * 2 for k in (1, 2, 3)}
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Access dict with literal keys
        acc += x[tile_idx] * multipliers[1]
        acc += x[tile_idx] * multipliers[2]
        acc += x[tile_idx] * multipliers[3]
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_dict_comprehension_with_range(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test dict comprehension with range for key generation."""
    result = torch.zeros_like(x)
    # Create dict using comprehension with range
    multipliers = {i: (i + 1) * 2 for i in range(4)}
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Access dict with literal keys
        acc += x[tile_idx] * multipliers[0]
        acc += x[tile_idx] * multipliers[1]
        acc += x[tile_idx] * multipliers[2]
        acc += x[tile_idx] * multipliers[3]
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_comprehension_with_function(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test list comprehension with expressions."""
    result = torch.zeros_like(x)

    # Test list comprehension with more complex expressions
    squared_values = [i * i for i in (1, 2, 3)]
    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for value in squared_values:
            acc += x[tile_idx] * value
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_comprehension_with_tensors(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test list comprehension that produces a list of tensors."""
    result = torch.zeros_like(tensors[0])

    # This should work - creating a list of tensor references
    tensor_list = list(tensors)

    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for tensor in tensor_list:
            acc += tensor[tile_idx]
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_nested_list_comprehension(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test nested list comprehension (flattened)."""
    result = torch.zeros_like(x)

    # Create pairs manually first since nested comprehensions with multiple generators
    # are not supported yet
    base_pairs = ((1, 3), (1, 4), (2, 3), (2, 4))
    pairs = list(base_pairs)

    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        for i, j in pairs:
            acc += x[tile_idx] * (i + j)
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_comprehension_with_tuple_unrolling(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Test interaction between list comprehension and tuple unrolling."""
    result = torch.zeros_like(tensors[0])

    # Create scaled versions of tensors using list comprehension
    scales = [0.5, 1.0, 1.5]
    scaled_tensors = [t * scale for t, scale in zip(tensors, scales, strict=False)]

    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)
        # Iterate over the scaled tensors (both list comp and tuple unrolling)
        for scaled_tensor in scaled_tensors:
            acc += scaled_tensor[tile_idx]
        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_comprehension_host_and_device(
    x: torch.Tensor,
) -> torch.Tensor:
    """Test list comprehension that works in both host and device code."""
    result = torch.zeros_like(x)

    # Host code - create list comprehension
    host_multipliers = [i * 2 for i in (1, 2, 3, 4)]

    for tile_idx in hl.tile(result.size(0)):
        acc = torch.zeros([tile_idx], dtype=torch.float32, device=result.device)

        # Device code - create list comprehension (should be unrolled)
        device_multipliers = [i + 1 for i in (0, 1, 2)]

        # Use both host and device comprehensions
        for host_mult in host_multipliers:
            for device_mult in device_multipliers:
                acc += x[tile_idx] * host_mult * device_mult

        result[tile_idx] = acc
    return result


@helion.kernel(autotune_effort="none")
def kernel_list_register_cache_layernorm(
    input_list: list[torch.Tensor],
    ln_eps: float = 1e-5,
) -> torch.Tensor:
    """Two-pass layernorm over concatenated list elements using register cache."""
    G = len(input_list)
    M, D = input_list[0].shape
    FULL_D = G * D
    FULL_D = hl.specialize(FULL_D)
    D = hl.specialize(D)
    output = torch.empty(
        [M, FULL_D], dtype=input_list[0].dtype, device=input_list[0].device
    )
    d_block = hl.register_block_size(helion.next_power_of_2(D))
    for tile_m, tile_d in hl.tile([M, D], block_size=[None, d_block]):
        row_sum = hl.zeros([tile_m], dtype=torch.float32)
        row_sq_sum = hl.zeros([tile_m], dtype=torch.float32)
        cached = [hl.zeros([tile_m, tile_d], dtype=input_list[0].dtype)] * G
        for i in hl.static_range(G):
            val = input_list[i][tile_m, tile_d]
            cached[i] = val
            val_f32 = val.to(torch.float32)
            row_sum = row_sum + torch.sum(val_f32, dim=-1)
            row_sq_sum = row_sq_sum + torch.sum(val_f32 * val_f32, dim=-1)
        mean = row_sum / FULL_D
        var = (row_sq_sum / FULL_D) - (mean * mean)
        rstd = 1.0 / torch.sqrt(var + ln_eps)
        for i in hl.static_range(G):
            normalized = (cached[i].to(torch.float32) - mean[:, None]) * rstd[:, None]
            output[tile_m, tile_d + i * D] = normalized.to(output.dtype)
    return output


@helion.kernel(autotune_effort="none")
def kernel_list_no_cache_layernorm(
    input_list: list[torch.Tensor],
    ln_eps: float = 1e-5,
) -> torch.Tensor:
    """Two-pass layernorm without register cache (re-gathers in pass 2)."""
    G = len(input_list)
    M, D = input_list[0].shape
    FULL_D = G * D
    FULL_D = hl.specialize(FULL_D)
    D = hl.specialize(D)
    output = torch.empty(
        [M, FULL_D], dtype=input_list[0].dtype, device=input_list[0].device
    )
    d_block = hl.register_block_size(helion.next_power_of_2(D))
    for tile_m, tile_d in hl.tile([M, D], block_size=[None, d_block]):
        row_sum = hl.zeros([tile_m], dtype=torch.float32)
        row_sq_sum = hl.zeros([tile_m], dtype=torch.float32)
        for i in hl.static_range(G):
            val = input_list[i][tile_m, tile_d]
            val_f32 = val.to(torch.float32)
            row_sum = row_sum + torch.sum(val_f32, dim=-1)
            row_sq_sum = row_sq_sum + torch.sum(val_f32 * val_f32, dim=-1)
        mean = row_sum / FULL_D
        var = (row_sq_sum / FULL_D) - (mean * mean)
        rstd = 1.0 / torch.sqrt(var + ln_eps)
        for i in hl.static_range(G):
            val = input_list[i][tile_m, tile_d]
            normalized = (val.to(torch.float32) - mean[:, None]) * rstd[:, None]
            output[tile_m, tile_d + i * D] = normalized.to(output.dtype)
    return output


@helion.kernel(autotune_effort="none")
def filtered_values(x: torch.Tensor, limit: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i + 1 for i in (0, 1, 2, 3) if i < limit if i % 2 == 0]
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for value in values:
            acc = acc + x[tile] * value
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def constant_filters(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [8 // i for i in (0, 1, 2, 4) if i != 0 and 8 // i >= 4]
        chained = [i + 1 for i in (0, 1, 2) if 0 < i <= 8 // i]
        members = [i + 1 for i in hl.static_range(4) if i in (0, 2)]
        nested = [i + 1 for i in (0, 1, 2) if (i != 0 and 8 // i >= 4) == 1]
        truth = [i for i in (1, 2) if (i,) if not ()]
        out[tile] = (
            x[tile] * sum(values)
            + x[tile] * sum(chained)
            + x[tile] * sum(members)
            + x[tile] * sum(nested)
            + x[tile] * sum(truth)
        )
    return out


@helion.kernel(autotune_effort="none")
def filtered_generator(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = tuple(i + 1 for i in (0, 1, 2, 3) if i % 2 == 0)
        out[tile] = x[tile] * (values[0] + values[1])
    return out


@helion.kernel(autotune_effort="none")
def none_values(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [None for i in (0, 1, 2) if i > 0]
        if values[1] is None:
            out[tile] = x[tile]
    return out


@helion.kernel(autotune_effort="none")
def dynamic_filter(x: torch.Tensor, limit: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (0, 1, 2) if i < limit]
        out[tile] = x[tile] * values[0]
    return out


@helion.kernel(autotune_effort="none")
def dynamic_membership(x: torch.Tensor, limit: int) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (0, 1, 2) if i in (limit, 5)]
        for value in values:
            out[tile] = x[tile] * value
    return out


@helion.kernel(autotune_effort="none")
def tensor_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (0, 1, 2) if x[tile] > 0]
        for value in values:
            out[tile] = x[tile] * value
    return out


@helion.kernel(autotune_effort="none")
def identity_filter(x: torch.Tensor, limit: hl.constexpr) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (10000, 20000) if i is limit]
        for value in values:
            out[tile] = x[tile] * value
    return out


@helion.kernel(autotune_effort="none")
def shadowed_target(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    i = 7
    for tile in hl.tile(x.numel()):
        values = [i + 1 for i in (0, 1, 2) if i > 0]
        out[tile] = x[tile] * (values[0] + values[1] + i)
    return out


@helion.kernel(autotune_effort="none")
def filtered_tensors(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        pairs = ((0, 1), (1, 2), (2, 3))
        values = [x[tile] * factor for index, factor in pairs if index != 1]
        out[tile] = values[0] + values[1]
    return out


@helion.kernel(autotune_effort="none")
def computed_literal_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for j in (0, 1):
            n = j + 1
            values = [i for i in (1, 2) if i == n]
            acc = acc + x[tile] * values[0]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def computed_static_range_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for j in hl.static_range(2):
            n = j + 1
            values = [i for i in (1, 2) if i == n]
            acc = acc + x[tile] * values[0]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def iteration_dependent_iterable(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for j in (1, 2):
            values = [i for i in (j,) if i > 0]
            acc = acc + x[tile] * values[0]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def heterogeneous_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [v for v in (1, None, 2) if v is not None]
        mixed = [v for v in (1, "keep", 2) if v != 2]
        if mixed[1] == "keep":
            out[tile] = x[tile] * (values[0] + values[1])
    return out


@helion.kernel(autotune_effort="none")
def tensor_and_scalar_result(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [v for v in (x, 1) if v is not None]
        out[tile] = values[0][tile] * values[1]
    return out


@helion.kernel(autotune_effort="none")
def changing_comprehension_length(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = torch.zeros([tile], dtype=x.dtype, device=x.device)
        for limit in hl.static_range(1, 4):
            values = [i for i in (0, 1, 2) if i < limit]
            for value in values:
                acc = acc + x[tile] * value
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def filtered_multidimensional_tensor(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.size(0)):
        values = [x[tile] * i for i in (1, 2, 3) if i != 2]
        out[tile] = values[0] + values[1]
    return out


@helion.kernel(autotune_effort="none")
def generator_enclosing_filtered_dict(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (1, 2))
        first = {0: len([i for i in (1, 2) if i > 0]) for _ in generator}  # noqa: B035
        second = {0: len([i for i in (1, 2) if i > 0]) for _ in generator}  # noqa: B035
        out[tile] = x[tile] * (len(first) + len(second))
    return out


@helion.kernel(autotune_effort="none")
def generator_enclosing_nested_filtered_list(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (1, 1))
        first = [len([i for i in (1, 2) if value]) for value in generator]
        second = [len([i for i in (1, 2) if value]) for value in generator]
        out[tile] = x[tile] * (len(first) + len(second))
    return out


@helion.kernel(autotune_effort="none")
def generator_enclosing_filtered_list(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (1, 1))
        acc = x[tile] * 0
        for value in generator:
            selected = [i for i in (1, 2) if value]
            acc = acc + len(selected)
        for value in generator:
            selected = [i for i in (1, 2) if value]
            acc = acc + len(selected)
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def consumed_generator_enclosing_filtered_list(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (1, 1))
        acc = x[tile] * 0
        for value in tuple(generator):
            selected = [i for i in (1, 2) if value]
            acc = acc + len(selected)
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def consumed_generator_in_iterable(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        first = [0 for _ in tuple(generator) if 1]
        second = [0 for _ in tuple(generator) if 1]
        out[tile] = x[tile] * (len(first) + len(second))
    return out


@helion.kernel(autotune_effort="none")
def consumed_generator_in_element(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        values = [len(tuple(generator)) for i in (1, 2) if i]
        out[tile] = x[tile] * (values[0] + 10 * values[1])
    return out


@helion.kernel(autotune_effort="none")
def host_generator_as_filtered_iterable(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    generator = (j + 0 for j in (3, 4))
    for tile in hl.tile(x.numel()):
        first = [0 for _ in generator if 1]
        second = [0 for _ in generator if 1]
        out[tile] = x[tile] * (len(first) + len(second))
    return out


@helion.kernel(autotune_effort="none")
def generator_as_filtered_iterable(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        first = [0 for _ in generator if 1]
        second = [0 for _ in generator if 1]
        out[tile] = x[tile] * (len(first) + len(second))
    return out


@helion.kernel(autotune_effort="none")
def nested_generator_in_filter(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (j + 0 for j in (3, 4))
        values = [i for i in (1, 2) if [generator] == [(3, 4)]]
        out[tile] = x[tile] * len(values)
    return out


@helion.kernel(autotune_effort="none")
def mutate_comprehension_list(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        cached = [x[tile] * 0 for _ in (0, 1)]
        cached[0] = x[tile]
        cached[1] = x[tile] * 2
        out[tile] = cached[0] + cached[1]
    return out


@helion.kernel(autotune_effort="none")
def unpack_list_comprehensions(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        a, b = [i + 0 for i in (1, 2)]
        c, d = [i for i in (0, 3, 4) if i > 0]
        out[tile] = x[tile] * (a + b + c + d)
    return out


@helion.kernel(autotune_effort="none")
def mixed_sequence_tuple_comparison(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = x[tile] * 0
        for values in ([1, 2], (1, 2)):
            if values == (1, 2):
                acc = acc + x[tile]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def mixed_sequence_list_comparison(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        acc = x[tile] * 0
        for values in ((1, 2), [1, 2]):
            if values == [1, 2]:
                acc = acc + x[tile]
        out[tile] = acc
    return out


@helion.kernel(autotune_effort="none")
def generator_tuple_comparison(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        generator = (i + 0 for i in (1, 2))
        if generator == (1, 2):
            out[tile] = x[tile] * 7
        else:
            out[tile] = x[tile] * 3
    return out


@helion.kernel(autotune_effort="none")
def filtered_list_tuple_comparison(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        values = [i for i in (1, 2) if i > 0]
        if values == (1, 2):
            out[tile] = x[tile] * 7
        else:
            out[tile] = x[tile] * 3
    return out


@helion.kernel(autotune_effort="none")
def filtered_list_comparisons(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(x.numel()):
        first = [i for i in (1, 2) if i > 0]
        selected = [i for i in (1, 2) if first == [1, 2]]
        different_type = [i for i in (3, 4) if first == (1, 2)]
        acc = x[tile] * 0
        for value in selected:
            acc = acc + x[tile] * value
        for value in different_type:
            acc = acc + x[tile] * value
        out[tile] = acc
    return out


@onlyBackends(["triton", "cute"])
class TestUnrollTuples(RefEagerTestBase, TestCase):
    def test_basic_tuple_addition(self):
        """Test basic iteration over tuple of tensors with addition."""
        size = (32,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        tuple_arg = (tensor1, tensor2, tensor3)

        code, result = code_and_output(kernel_tuple_addition, (tuple_arg,))

        # Validate generated code

        # Test correctness
        expected = tensor1 + tensor2 + tensor3
        torch.testing.assert_close(result, expected)

    def test_tuple_with_scaling_factors(self):
        """Test iteration with corresponding scalar values."""
        size = (48,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        scale1, scale2, scale3 = 2.0, 0.5, 1.5

        code, result = code_and_output(
            kernel_tuple_with_scaling,
            (tensor1, tensor2, tensor3, scale1, scale2, scale3),
        )

        # Validate generated code

        # Test correctness
        expected = tensor1 * scale1 + tensor2 * scale2 + tensor3 * scale3
        torch.testing.assert_close(result, expected)

    def test_nested_tuple_iteration(self):
        """Test nested loops over multiple tuples."""
        size = (40,)
        a1 = torch.randn(size, device=DEVICE)
        a2 = torch.randn(size, device=DEVICE)
        b1 = torch.randn(size, device=DEVICE) + 1.0  # Avoid zeros for multiplication
        b2 = torch.randn(size, device=DEVICE) + 1.0

        a_tuple = (a1, a2)
        b_tuple = (b1, b2)

        code, result = code_and_output(
            kernel_nested_tuple_iteration, (a_tuple, b_tuple)
        )

        # Validate generated code

        # Test correctness
        temp = a1 + a2
        expected = temp * b1 * b2
        torch.testing.assert_close(result, expected)

    def test_single_element_tuple(self):
        """Test with single-element tuple."""
        size = (16,)
        tensor = torch.randn(size, device=DEVICE)

        tuple_arg = (tensor,)

        code, result = code_and_output(kernel_tuple_addition, (tuple_arg,))

        # Validate generated code

        # Test correctness - should just copy the tensor
        torch.testing.assert_close(result, tensor)

    @onlyBackends(["triton"])
    def test_single_element_list_stores_before_second_root(self):
        """One-element lists' leaked loop variables are not host arguments."""
        x = torch.randn(16, device=DEVICE)
        for count in (1, 3):
            ys = [torch.zeros_like(x) for _ in range(count)]
            zs = [torch.zeros_like(x) for _ in range(count)]
            code, result = code_and_output(
                kernel_list_stores_then_second_root, (x, ys, zs)
            )
            if not self._in_ref_eager_mode:
                self.assertNotIn("peer_", code.split("_launcher(")[-1])
            torch.testing.assert_close(result, x)
            for y, z in zip(ys, zs, strict=True):
                torch.testing.assert_close(y, 2 * x)
                torch.testing.assert_close(z, 2 * x + 1)

    def test_constants_iteration(self):
        """Test iteration over tuple of constants."""
        size = (24,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_constants_iteration, (x,))

        # Validate generated code

        # Test correctness - should be x * (1 + 2 + 3) = x * 6
        expected = x * 6
        torch.testing.assert_close(result, expected)

    def test_list_constants_iteration(self):
        """Test iteration over list of constants."""
        size = (20,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_list_constants_iteration, (x,))

        # Validate generated code

        # Test correctness - should be x * (0.5 + 1.5 + 2.5) = x * 4.5
        expected = x * 4.5
        torch.testing.assert_close(result, expected)

    def test_zip_iteration(self):
        """Test iteration over zip of tuples."""
        # Create one reference tensor and use it to create others with same size
        reference = torch.randn((36,), device=DEVICE)
        a1 = torch.randn_like(reference)
        a2 = torch.randn_like(reference)
        b1 = torch.randn_like(reference)
        b2 = torch.randn_like(reference)

        tensors_a = (a1, a2)
        tensors_b = (b1, b2)

        code, result = code_and_output(kernel_zip_iteration, (tensors_a, tensors_b))

        # Validate generated code

        # Test correctness - should be a1*b1 + a2*b2
        expected = a1 * b1 + a2 * b2
        torch.testing.assert_close(result, expected)

    def test_static_range_iteration(self):
        """Test iteration using hl.static_range."""
        size = (28,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_static_range_iteration, (x,))

        # Validate generated code

        # Test correctness - should be x * (1 + 2 + 3 + 4) = x * 10
        expected = x * 10
        torch.testing.assert_close(result, expected)

    def test_static_range_with_start(self):
        """Test static_range with start parameter."""
        size = (18,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_static_range_with_start, (x,))

        # Validate generated code

        # Test correctness - should be x * (2 + 3 + 4) = x * 9
        expected = x * 9
        torch.testing.assert_close(result, expected)

    def test_static_range_tuple_indexing(self):
        @helion.kernel(autotune_effort="none")
        def kernel_static_range_tuple_indexing(
            buf_tuple: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            WORLD_SIZE: hl.constexpr,
        ) -> torch.Tensor:
            """Test tuple indexing with static_range - iterating over tuple elements."""
            (M,) = buf_tuple[0].shape
            result = torch.zeros_like(buf_tuple[0])

            for tile_m in hl.tile(M):
                acc = hl.zeros([tile_m], dtype=torch.float32)

                # Use static_range to index into tuple
                for i in hl.static_range(WORLD_SIZE):
                    acc += buf_tuple[i][tile_m]

                result[tile_m] = acc

            return result

        size = (32,)
        world_size = 4

        tensors = tuple(
            torch.ones(size, device=DEVICE, dtype=torch.float32) * (i + 1)
            for i in range(world_size)
        )

        code, result = code_and_output(
            kernel_static_range_tuple_indexing, (tensors, world_size)
        )

        # Test correctness - should be sum of all tensors: 1 + 2 + 3 + 4 = 10
        expected = sum(tensors)
        torch.testing.assert_close(result, expected)

    @skipIfRefEager("Type inference errors are not raised in ref eager mode")
    def test_static_range_tuple_indexing_requires_uniform_types(self):
        @helion.kernel(autotune_effort="none")
        def kernel_static_range_tuple_mismatch(x: torch.Tensor) -> torch.Tensor:
            heterogeneous = (x, 1)

            for _tile_n in hl.tile(x.size(0)):
                for idx in hl.static_range(2):
                    _ = heterogeneous[idx]
            return x

        x = torch.ones((8,), device=DEVICE)

        with self.assertRaisesRegex(
            exc.TypeInferenceError,
            r"Sequence indexing with non-literal index requires all elements to have the same type",
        ):
            code_and_output(kernel_static_range_tuple_mismatch, (x,))

    def test_mixed_constants_and_tensors(self):
        """Test mixed iteration over both tensors and constants."""
        size = (22,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2)
        constants = (2, 3)

        code, result = code_and_output(
            kernel_mixed_constants_and_tensors, (tensors, constants)
        )

        # Validate generated code

        # Test correctness - should be (tensor1 + tensor2) * 2 * 3
        expected = (tensor1 + tensor2) * 2 * 3
        torch.testing.assert_close(result, expected)

    def test_enumerate_iteration(self):
        """Test iteration using enumerate over tensors."""
        size = (24,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2, tensor3)

        code, result = code_and_output(kernel_enumerate_iteration, (tensors,))

        # Validate generated code

        # Test correctness - should be tensor1*1 + tensor2*2 + tensor3*3
        expected = tensor1 * 1 + tensor2 * 2 + tensor3 * 3
        torch.testing.assert_close(result, expected)

    def test_enumerate_with_start(self):
        """Test enumerate with custom start value."""
        size = (18,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2)

        code, result = code_and_output(kernel_enumerate_with_start, (tensors,))

        # Validate generated code

        # Test correctness - should be tensor1*5 + tensor2*6 (start=5)
        expected = tensor1 * 5 + tensor2 * 6
        torch.testing.assert_close(result, expected)

    def test_enumerate_constants(self):
        """Test enumerate over constants."""
        size = (20,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_enumerate_constants, (x,))

        # Validate generated code

        # Test correctness - should be x*(2*0 + 3*1 + 4*2) = x*(0 + 3 + 8) = x*11
        expected = x * 11
        torch.testing.assert_close(result, expected)

    def test_simple_list_comprehension(self):
        """Test simple list comprehension with constants."""
        size = (16,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_simple_list_comprehension, (x,))

        # Validate generated code

        # Test correctness - should be x * (2 + 4 + 6) = x * 12
        expected = x * 12
        torch.testing.assert_close(result, expected)

    def test_tuple_comprehension(self):
        """Test tuple comprehension with generator expression."""
        size = (16,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_tuple_comprehension, (x,))

        # Validate generated code

        # Test correctness - should be x * (2 + 4 + 6) = x * 12
        expected = x * 12
        torch.testing.assert_close(result, expected)

    def test_tuple_comprehension_with_static_range(self):
        """Test tuple comprehension with static_range for indexing."""
        size = (16,)
        x = torch.randn(size, device=DEVICE)
        N = 4

        code, result = code_and_output(
            kernel_tuple_comprehension_with_static_range, (x, N)
        )

        # Validate generated code

        # Test correctness - should be x * (1 + 2 + 3 + 4) = x * 10
        expected = x * 10
        torch.testing.assert_close(result, expected)

    def test_tuple_comprehension_with_tensors(self):
        """Test tuple comprehension that transforms a tuple of tensors."""
        size = (18,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2, tensor3)

        code, result = code_and_output(
            kernel_tuple_comprehension_with_tensors, (tensors,)
        )

        # Validate generated code

        # Test correctness - should be tensor1*0.5 + tensor2*1.0 + tensor3*1.5
        expected = tensor1 * 0.5 + tensor2 * 1.0 + tensor3 * 1.5
        torch.testing.assert_close(result, expected)

    def test_dict_comprehension(self):
        """Test dict comprehension with constants."""
        size = (16,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_dict_comprehension, (x,))

        # Validate generated code

        # Test correctness - multipliers = {1: 2, 2: 4, 3: 6}
        # should be x * (2 + 4 + 6) = x * 12
        expected = x * 12
        torch.testing.assert_close(result, expected)

    def test_dict_comprehension_with_range(self):
        """Test dict comprehension with range for key generation."""
        size = (16,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_dict_comprehension_with_range, (x,))

        # Validate generated code

        # Test correctness - multipliers = {0: 2, 1: 4, 2: 6, 3: 8}
        # should be x * (2 + 4 + 6 + 8) = x * 20
        expected = x * 20
        torch.testing.assert_close(result, expected)

    def test_list_comprehension_with_function(self):
        """Test list comprehension with expressions."""
        size = (14,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_list_comprehension_with_function, (x,))

        # Validate generated code

        # Test correctness - should be x * (1 + 4 + 9) = x * 14
        expected = x * 14
        torch.testing.assert_close(result, expected)

    def test_list_comprehension_with_tensors(self):
        """Test list comprehension that produces a list of tensors."""
        size = (18,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2, tensor3)

        code, result = code_and_output(
            kernel_list_comprehension_with_tensors, (tensors,)
        )

        # Validate generated code

        # Test correctness - should be sum of all tensors
        expected = tensor1 + tensor2 + tensor3
        torch.testing.assert_close(result, expected)

    def test_nested_list_comprehension(self):
        """Test nested list comprehension."""
        size = (12,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_nested_list_comprehension, (x,))

        # Validate generated code

        # Test correctness - pairs are (1,3), (1,4), (2,3), (2,4)
        # should be x * (4 + 5 + 5 + 6) = x * 20
        expected = x * 20
        torch.testing.assert_close(result, expected)

    def test_list_comprehension_with_tuple_unrolling(self):
        """Test interaction between list comprehension and tuple unrolling."""
        size = (22,)
        tensor1 = torch.randn(size, device=DEVICE)
        tensor2 = torch.randn(size, device=DEVICE)
        tensor3 = torch.randn(size, device=DEVICE)

        tensors = (tensor1, tensor2, tensor3)

        code, result = code_and_output(
            kernel_list_comprehension_with_tuple_unrolling, (tensors,)
        )

        # Validate generated code

        # Test correctness - should be tensor1*0.5 + tensor2*1.0 + tensor3*1.5
        expected = tensor1 * 0.5 + tensor2 * 1.0 + tensor3 * 1.5
        torch.testing.assert_close(result, expected)

    def test_list_comprehension_host_and_device(self):
        """Test list comprehension that works in both host and device code."""
        size = (26,)
        x = torch.randn(size, device=DEVICE)

        code, result = code_and_output(kernel_list_comprehension_host_and_device, (x,))

        # Validate generated code

        # Test correctness
        # host_multipliers = [2, 4, 6, 8]
        # device_multipliers = [1, 2, 3]
        # Total should be x * (2*1 + 2*2 + 2*3 + 4*1 + 4*2 + 4*3 + 6*1 + 6*2 + 6*3 + 8*1 + 8*2 + 8*3)
        # = x * (2 + 4 + 6 + 4 + 8 + 12 + 6 + 12 + 18 + 8 + 16 + 24) = x * 120
        expected = x * 120
        torch.testing.assert_close(result, expected)

    @largeTensorTest("8GB", device=DEVICE)
    @skipIfRefEager("RuntimeError in ref eager mode")
    @skipIfMTIA(
        "Triton-MTIA: register-cache layernorm (G=8) needs >14 circular buffers; see T280008478"
    )
    # MI350X CI intermittently returns wrong values for a few rows of this
    # two-pass layernorm (up to ~900 elements off by up to 0.26) while
    # every CUDA job passes; test_examples.py::test_softmax_bwd has the
    # full description.
    @skipIfRocm("intermittent wrong rows from looped row reductions on ROCm")
    def test_list_register_cache_layernorm(self):
        """Test two-pass layernorm with register-cached list elements."""
        M, D, G = 1024 * 1024, 32, 8
        tensors = [
            torch.randn(M, D, device=DEVICE, dtype=torch.bfloat16) for _ in range(G)
        ]

        code, result = code_and_output(kernel_list_register_cache_layernorm, (tensors,))

        # Reference: concat then layernorm (no affine)
        concatenated = torch.cat(tensors, dim=1).float()
        mean = concatenated.mean(dim=-1)
        var = ((concatenated - mean[:, None]) ** 2).mean(dim=-1)
        rstd = 1.0 / torch.sqrt(var + 1e-5)
        expected = ((concatenated - mean[:, None]) * rstd[:, None]).to(tensors[0].dtype)

        torch.testing.assert_close(result, expected, atol=5 * 1e-2, rtol=5 * 1e-2)

        # Verify register caching: G loads in pass 1, no re-loads in pass 2
        if _get_backend() == "triton":
            triton_kernel = code[: code.index("\ndef kernel_list_register_cache")]
            load_count = triton_kernel.count("tl.load")
            assert load_count == G, f"Expected {G} loads, got {load_count}"

    @largeTensorTest("8GB", device=DEVICE)
    @skipIfRefEager("RuntimeError in ref eager mode")
    @skipIfMTIA(
        "Triton-MTIA: no-cache layernorm (G=8, 2*G loads) exceeds the 14 circular-buffer limit; see T280008478"
    )
    # MI350X CI intermittently returns wrong values for a few rows of this
    # two-pass layernorm (up to ~900 elements off by up to 0.26) while
    # every CUDA job passes; test_examples.py::test_softmax_bwd has the
    # full description.
    @skipIfRocm("intermittent wrong rows from looped row reductions on ROCm")
    def test_list_no_cache_layernorm(self):
        """Test two-pass layernorm without register cache (re-gathers in pass 2)."""
        M, D, G = 1024 * 1024, 32, 8
        tensors = [
            torch.randn(M, D, device=DEVICE, dtype=torch.bfloat16) for _ in range(G)
        ]

        code, result = code_and_output(kernel_list_no_cache_layernorm, (tensors,))

        # Reference: concat then layernorm (no affine)
        concatenated = torch.cat(tensors, dim=1).float()
        mean = concatenated.mean(dim=-1)
        var = ((concatenated - mean[:, None]) ** 2).mean(dim=-1)
        rstd = 1.0 / torch.sqrt(var + 1e-5)
        expected = ((concatenated - mean[:, None]) * rstd[:, None]).to(tensors[0].dtype)

        torch.testing.assert_close(result, expected, atol=5 * 1e-2, rtol=5 * 1e-2)

        # No cache: G loads in pass 1 + G loads in pass 2
        if _get_backend() == "triton":
            triton_kernel = code[: code.index("\ndef kernel_list_no_cache")]
            load_count = triton_kernel.count("tl.load")
            assert load_count == 2 * G, f"Expected {2 * G} loads, got {load_count}"

    @largeTensorTest("12GB", device=DEVICE)
    @skipIfRefEager("Benchmark not applicable in ref eager mode")
    @skipIfRocm("Benchmark timing unreliable on ROCm")
    @skipIfXPU("Benchmark timing unreliable on XPU")
    @unittest.skipIf(
        "PYTEST_XDIST_WORKER" in os.environ,
        "Benchmark timing unreliable under pytest-xdist",
    )
    @skipIfMTIA(
        "Triton-MTIA: both layernorm kernels exceed the 14 circular-buffer limit; see T280008478"
    )
    def test_register_cache_faster_than_no_cache(self):
        """Verify register-cached layernorm is faster than re-gathering."""
        from triton.testing import do_bench

        M, D, G = 1024 * 1024, 32, 8
        tensors = [
            torch.randn(M, D, device=DEVICE, dtype=torch.bfloat16) for _ in range(G)
        ]

        # Warmup and correctness check
        cached_result = kernel_list_register_cache_layernorm(tensors)
        no_cache_result = kernel_list_no_cache_layernorm(tensors)
        torch.testing.assert_close(
            cached_result, no_cache_result, atol=5 * 1e-2, rtol=5 * 1e-2
        )

        ms_cached = do_bench(lambda: kernel_list_register_cache_layernorm(tensors))
        ms_no_cache = do_bench(lambda: kernel_list_no_cache_layernorm(tensors))

        speedup = ms_no_cache / ms_cached
        print(
            f"\nRegister cache: {ms_cached:.3f} ms, "
            f"No cache: {ms_no_cache:.3f} ms, "
            f"Speedup: {speedup:.2f}x"
        )
        assert speedup > 1.0, (
            f"Expected register cache to be faster, got speedup={speedup:.2f}x"
        )


@pytest.mark.skipif(
    os.environ.get("HELION_INTERPRET") == "1", reason="Requires compiled mode"
)
class TestFilteredComprehensions:
    @pytest.mark.parametrize("limit, factors", [(0, ()), (1, (1,)), (4, (1, 3))])
    def test_filtered_values_bind(self, limit: int, factors: tuple[int, ...]) -> None:
        bound = filtered_values.bind((torch.ones(32), limit))
        graph = str(bound.host_function.device_ir.graphs[0].graph)
        for factor in factors:
            assert f", {factor})" in graph
        assert graph.count("aten.mul.Tensor") == len(factors)

    def test_constant_filters(self) -> None:
        bound = constant_filters.bind((torch.ones(32),))
        graph = bound.host_function.device_ir.graphs[0].graph
        factors = [
            node.args[1]
            for node in graph.nodes
            if node.target is torch.ops.aten.mul.Tensor
        ]
        assert factors == [12, 5, 4, 5, 3]

    def test_none_is_an_element(self) -> None:
        bound = none_values.bind((torch.ones(32),))
        assert any(
            node.target is hl.store
            for node in bound.host_function.device_ir.graphs[0].graph.nodes
        )

    def test_runtime_filter_is_rejected(self) -> None:
        with pytest.raises(exc.StatementNotSupported, match="compile-time constant"):
            dynamic_filter.bind((torch.ones(32), 2))

    def test_runtime_membership_is_rejected(self) -> None:
        with pytest.raises(exc.StatementNotSupported, match="compile-time constant"):
            dynamic_membership.bind((torch.ones(32), 2))

    def test_tensor_filter_is_rejected(self) -> None:
        with pytest.raises(exc.StatementNotSupported, match="compile-time constant"):
            tensor_filter.bind((torch.ones(32),))

    def test_identity_filter_is_rejected(self) -> None:
        with pytest.raises(exc.StatementNotSupported, match="Identity comparisons"):
            identity_filter.bind((torch.ones(32), 10000))

    def test_comprehension_target_does_not_shadow_outer_value(self) -> None:
        bound = shadowed_target.bind((torch.ones(32),))
        graph = str(bound.host_function.device_ir.graphs[0].graph)
        assert ", 12)" in graph

    def test_filtered_tensor_bodies_are_traced_only_for_selected_elements(self) -> None:
        bound = filtered_tensors.bind((torch.ones(32),))
        graph = str(bound.host_function.device_ir.graphs[0].graph)
        assert graph.count("aten.mul.Tensor") == 2
        assert ", 1)" in graph
        assert ", 3)" in graph

    @pytest.mark.parametrize(
        "kernel", [computed_literal_filter, computed_static_range_filter]
    )
    def test_computed_filter_per_iteration(self, kernel: helion.Kernel) -> None:
        bound = kernel.bind((torch.ones(32),))
        graph = str(bound.host_function.device_ir.graphs[0].graph)
        assert graph.count("aten.mul.Tensor") == 2
        assert all(f", {factor})" in graph for factor in (1, 2))

    @pytest.mark.parametrize(
        "kernel", [iteration_dependent_iterable, heterogeneous_filter]
    )
    def test_filtered_sequence_types_per_iteration(self, kernel: helion.Kernel) -> None:
        bound = kernel.bind((torch.ones(32),))
        graph = str(bound.host_function.device_ir.graphs[0].graph)
        if kernel is iteration_dependent_iterable:
            assert graph.count("aten.mul.Tensor") == 2
            assert all(f", {factor})" in graph for factor in (1, 2))
        else:
            assert ", 3)" in graph

    def test_filtered_mixed_result_preserves_tensor_alias_information(self) -> None:
        bound = tensor_and_scalar_result.bind((torch.ones(32),))
        comprehension = next(
            node
            for stmt in bound.host_function.body
            for node in ast.walk(stmt)
            if isinstance(node, ast.ListComp)
        )
        assert isinstance(comprehension.elt, ast.Name)
        assert not _holds_no_storage(comprehension.elt)

    def test_changing_comprehension_length_is_rejected(self) -> None:
        with pytest.raises(exc.StatementNotSupported, match="changing lengths"):
            changing_comprehension_length.bind((torch.ones(32),))

    @pytest.mark.parametrize(
        ("kernel", "message"),
        [
            (filtered_generator, "Filtered generator expressions"),
            (
                host_generator_as_filtered_iterable,
                "Generator values in filtered comprehensions",
            ),
            (
                generator_as_filtered_iterable,
                "Generator values in filtered comprehensions",
            ),
            (
                consumed_generator_in_iterable,
                "Generator values in filtered comprehensions",
            ),
            (
                consumed_generator_in_element,
                "Generator values in filtered comprehensions",
            ),
            (nested_generator_in_filter, "Generator values in filtered comprehensions"),
        ],
    )
    def test_generator_restrictions(self, kernel: helion.Kernel, message: str) -> None:
        with pytest.raises(exc.StatementNotSupported, match=message):
            kernel.bind((torch.ones(32),))

    @pytest.mark.parametrize(
        ("kernel", "message"),
        [
            (
                generator_enclosing_filtered_list,
                "Generator values in loops containing filtered comprehensions",
            ),
            (
                consumed_generator_enclosing_filtered_list,
                "Generator values in loops containing filtered comprehensions",
            ),
            (
                generator_enclosing_nested_filtered_list,
                "Generator values in comprehensions containing filtered lists",
            ),
            (
                generator_enclosing_filtered_dict,
                "Generator values in comprehensions containing filtered lists",
            ),
            (generator_tuple_comparison, "Generator equality comparisons"),
        ],
    )
    def test_generator_enclosing_filtered_lists(
        self, kernel: helion.Kernel, message: str
    ) -> None:
        with pytest.raises(exc.StatementNotSupported, match=message):
            kernel.bind((torch.ones(32),))

    @pytest.mark.parametrize(
        "kernel", [mixed_sequence_tuple_comparison, mixed_sequence_list_comparison]
    )
    def test_mixed_sequence_comparisons(self, kernel: helion.Kernel) -> None:
        bound = kernel.bind((torch.ones(32),))
        assert bound.host_function is not None
        graph = bound.host_function.device_ir.graphs[0].graph
        assert (
            sum(node.target is torch.ops.aten.add.Tensor for node in graph.nodes) == 1
        )

    def test_filtered_list_comparisons(self) -> None:
        bound = filtered_list_comparisons.bind((torch.ones(32),))
        assert bound.host_function is not None
        graph = bound.host_function.device_ir.graphs[0].graph
        factors = [
            node.args[1]
            for node in graph.nodes
            if node.target is torch.ops.aten.mul.Tensor
        ]
        assert factors == [0, 1, 2]

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    @pytest.mark.parametrize("limit, multiplier", [(0, 0), (4, 4)])
    def test_filtered_values_cuda(self, limit: int, multiplier: int) -> None:
        x = torch.randn(128, device=DEVICE)
        torch.testing.assert_close(filtered_values(x, limit), x * multiplier)

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    def test_filtered_multidimensional_tensor_cuda(self) -> None:
        x = torch.randn(16, 4, device=DEVICE)
        torch.testing.assert_close(filtered_multidimensional_tensor(x), x * 4)

    @pytest.mark.cuda
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    @pytest.mark.parametrize(
        ("kernel", "factor"),
        [
            (filtered_tensors, 4),
            (mutate_comprehension_list, 3),
            (unpack_list_comprehensions, 10),
            (mixed_sequence_tuple_comparison, 1),
            (filtered_list_tuple_comparison, 3),
            (computed_static_range_filter, 3),
            (computed_literal_filter, 3),
        ],
    )
    def test_comprehension_results_cuda(
        self, kernel: helion.Kernel, factor: int
    ) -> None:
        x = torch.randn(128, device=DEVICE)
        torch.testing.assert_close(kernel(x), x * factor)


if __name__ == "__main__":
    unittest.main()

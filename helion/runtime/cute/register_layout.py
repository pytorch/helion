"""Register ownership conversions within a contiguous power-of-two subgroup.

Cyclic ownership assigns column ``register * lanes + lane``. Vector ownership
assigns ``(register // vector * lanes + output_lane) * vector + register % vector``.
Conversions preserve every register bit; callers retain logical-tail masks and
prove memory alignment separately. All lanes in each subgroup must participate.
"""

from __future__ import annotations

import cutlass
from cutlass import Int32
import cutlass.cute as cute


@cute.jit
def _exchange_lane_register_bits(
    fragment: cute.Tensor,
    block_width: cutlass.Constexpr[int],
    exchange_width: cutlass.Constexpr[int],
    register_span: cutlass.Constexpr[int],
) -> cute.Tensor:
    size = cute.size(fragment.shape)
    assert fragment.element_type.width in (1, 8, 16, 32, 64)
    output = cute.make_rmem_tensor(size, fragment.element_type)
    for index in cutlass.range_constexpr(size):
        output[index] = fragment[index]
    lane = cute.arch.lane_idx()
    for block in cutlass.range_constexpr(size // block_width):
        for stage in cutlass.range_constexpr(exchange_width.bit_length() - 1):
            lane_bit = 1 << stage
            register_bit = lane_bit * register_span
            for group in cutlass.range_constexpr(block_width // (2 * register_bit)):
                for offset in cutlass.range_constexpr(register_bit):
                    low = block * block_width + group * 2 * register_bit + offset
                    high = low + register_bit
                    a = fragment.element_type(output[low])
                    b = fragment.element_type(output[high])
                    if cutlass.const_expr(fragment.element_type.width in (8, 16)):
                        # CuTe promotes narrow floats to FP32 before a shuffle.
                        # Reinterpret first to retain NaN payloads and zero signs.
                        word_type = (
                            cutlass.Uint16
                            if cutlass.const_expr(fragment.element_type.width == 16)
                            else cutlass.Uint8
                        )
                        peer_b = (
                            cute.arch.shuffle_sync_bfly(
                                b.bitcast(word_type).to(cutlass.Uint32), offset=lane_bit
                            )
                            .to(word_type)
                            .bitcast(fragment.element_type)
                        )
                        peer_a = (
                            cute.arch.shuffle_sync_bfly(
                                a.bitcast(word_type).to(cutlass.Uint32), offset=lane_bit
                            )
                            .to(word_type)
                            .bitcast(fragment.element_type)
                        )
                    else:
                        peer_b = cute.arch.shuffle_sync_bfly(b, offset=lane_bit)
                        peer_a = cute.arch.shuffle_sync_bfly(a, offset=lane_bit)
                    if (lane & Int32(lane_bit)) != 0:
                        output[low], output[high] = peer_b, b
                    else:
                        output[low], output[high] = a, peer_a
    return output


@cute.jit
def cyclic_to_vector_narrow(
    fragment: cute.Tensor, vector_width: cutlass.Constexpr[int]
) -> cute.Tensor:
    """Convert when the containing subgroup width is a multiple of vector_width."""
    assert 1 < vector_width <= 32
    assert (vector_width & (vector_width - 1)) == 0
    assert cute.size(fragment.shape) % vector_width == 0
    return _exchange_lane_register_bits(fragment, vector_width, vector_width, 1)


@cute.jit
def cyclic_to_vector_wide(
    fragment: cute.Tensor,
    vector_width: cutlass.Constexpr[int],
    lanes_per_row: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Convert vectors wider than the containing subgroup, preserving its lane."""
    size = cute.size(fragment.shape)
    assert 1 <= lanes_per_row < vector_width <= 32
    assert (vector_width & (vector_width - 1)) == 0
    assert (lanes_per_row & (lanes_per_row - 1)) == 0
    assert size % vector_width == 0
    if cutlass.const_expr(lanes_per_row == 1):
        return fragment
    span = vector_width // lanes_per_row
    output = _exchange_lane_register_bits(fragment, vector_width, lanes_per_row, span)
    result = cute.make_rmem_tensor(size, fragment.element_type)
    for block in cutlass.range_constexpr(size // vector_width):
        for index in cutlass.range_constexpr(vector_width):
            source = (index % lanes_per_row) * span + index // lanes_per_row
            result[block * vector_width + index] = output[block * vector_width + source]
    return result


@cute.jit
def subgroup_vectorize(
    fragment: cute.Tensor,
    vector_width: cutlass.Constexpr[int],
    lanes_per_row: cutlass.Constexpr[int],
) -> tuple[cute.Tensor, Int32]:
    """Return contiguous vectors and their row-local output lane.

    The fragment contains a complete number of vectors, including any padded
    slots. Its scalar representation must support CuTe's butterfly shuffle.
    """
    assert cute.rank(fragment.shape) == 1
    assert 1 <= lanes_per_row <= 32
    assert lanes_per_row & (lanes_per_row - 1) == 0
    assert 1 <= vector_width <= 32
    assert vector_width & (vector_width - 1) == 0
    assert cute.size(fragment.shape) % vector_width == 0
    lane = cute.arch.lane_idx() % Int32(lanes_per_row)
    if cutlass.const_expr(vector_width == 1):
        return fragment, lane
    if cutlass.const_expr(vector_width <= lanes_per_row):
        output = cyclic_to_vector_narrow(fragment, vector_width)
        output_lane = (lane % Int32(vector_width)) * Int32(
            lanes_per_row // vector_width
        ) + lane // Int32(vector_width)
        return output, output_lane
    return cyclic_to_vector_wide(fragment, vector_width, lanes_per_row), lane


@cute.jit
def subgroup_unvectorize(
    fragment: cute.Tensor,
    vector_width: cutlass.Constexpr[int],
    lanes_per_row: cutlass.Constexpr[int],
) -> cute.Tensor:
    """Invert subgroup_vectorize using its same physical-lane convention."""
    assert cute.rank(fragment.shape) == 1
    assert 1 <= lanes_per_row <= 32
    assert lanes_per_row & (lanes_per_row - 1) == 0
    assert 1 <= vector_width <= 32
    assert vector_width & (vector_width - 1) == 0
    size = cute.size(fragment.shape)
    assert size % vector_width == 0
    if cutlass.const_expr(vector_width == 1 or lanes_per_row == 1):
        return fragment
    if cutlass.const_expr(vector_width <= lanes_per_row):
        return cyclic_to_vector_narrow(fragment, vector_width)
    span = vector_width // lanes_per_row
    unrotated = cute.make_rmem_tensor(size, fragment.element_type)
    for block in cutlass.range_constexpr(size // vector_width):
        for index in cutlass.range_constexpr(vector_width):
            source = (index % lanes_per_row) * span + index // lanes_per_row
            unrotated[block * vector_width + source] = fragment[
                block * vector_width + index
            ]
    return _exchange_lane_register_bits(unrotated, vector_width, lanes_per_row, span)

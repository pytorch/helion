"""One-CTA host sum/mean of a 1-D FP32 tensor with ATen's addition tree.

A host ``torch.sum(x)`` / ``x.mean()`` of a short 1-D FP32 tensor after a
Helion kernel launches ATen's generic ``reduce_kernel``, whose fixed cost is
about a microsecond above a minimal kernel of the same geometry (jsd s0: a
2.4 us graph node after a 1.5 us kernel). This kernel performs exactly the
addition tree ATen's ``gpu_reduce_kernel`` would (Reduce.cuh, pinned by the
paired-sum fingerprint) for a 16-byte aligned contiguous input of at least
128 elements reduced by one CTA: ``input_vectorized_thread_reduce_impl``
(four accumulators per lane of a ``width``-lane block, the tail elements into
the first accumulator, the increasing accumulator fold) and
``block_x_reduce`` (descending shared-memory halving down to one warp, then
the descending shuffle tree). The tree is a dataflow; this kernel evaluates
it in shared memory, so the output is bit-identical to the ATen result.
``mean`` multiplies by ATen's ``factor``.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute

from ...runtime.cute.source_dependencies import set_helper_source_hash
from .resident_reduction_runtime import _cute_resident_load_vector


@cute.kernel
def full_sum_kernel(
    source: cute.Tensor,
    result: cute.Tensor,
    numel: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
    factor: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[bool],
) -> None:
    lane = cutlass.Int32(cute.arch.thread_idx()[0])
    partial = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, width, alignment=16),
        cute.make_layout(width),
    )
    # Lane ``lane`` of ATen's block: the four-element vectors ``lane,
    # lane + width, ...`` into four independent +0 accumulators, the tail
    # element (if any) into the first, then the increasing fold.
    accum0 = cutlass.Float32(0)
    accum1 = cutlass.Float32(0)
    accum2 = cutlass.Float32(0)
    accum3 = cutlass.Float32(0)
    vectors: cutlass.Constexpr = numel // 4
    for step in cutlass.range_constexpr(cute.ceil_div(vectors, width)):
        index = step * width + lane
        if index < vectors:
            values = _cute_resident_load_vector(source, index * 4, 4)
            accum0 = accum0 + values[0]
            accum1 = accum1 + values[1]
            accum2 = accum2 + values[2]
            accum3 = accum3 + values[3]
    tail_count: cutlass.Constexpr = numel % 4
    if cutlass.const_expr(tail_count > 0):
        if lane < tail_count:
            tail = (source.iterator + (numel - tail_count + lane)).load()  # pyrefly: ignore[missing-attribute]
            accum0 = accum0 + tail
    partial[lane] = ((accum0 + accum1) + accum2) + accum3
    # block_x_reduce: shared halving while more than one warp remains, then
    # the warp's descending shuffle tree; both read the previous level's
    # values of lanes [offset, 2 * offset), so one shared tree is the same
    # dataflow.
    for level in cutlass.range_constexpr(width.bit_length() - 1):
        offset: cutlass.Constexpr = width >> (level + 1)
        cute.arch.sync_threads()
        if lane < offset:
            partial[lane] = partial[lane] + partial[lane + offset]
    cute.arch.sync_threads()
    if lane == 0:
        total = partial[0]
        if cutlass.const_expr(scale):
            total = total * cutlass.Float32(factor)
        result.iterator.store(total)  # pyrefly: ignore[missing-attribute]


set_helper_source_hash(full_sum_kernel, "full_host_sum")

"""MSL device-side reduction preamble for the Metal backend.

The heavy lifting is done by Inductor's ``c10/metal/reduction_utils.h``, which
already ships with PyTorch and is reachable from ``torch.mps.compile_shader``
(it embeds the ``c10`` headers).  It provides, for every dtype Helion supports:

- ``c10::metal::simd_{sum,prod,max,min}`` -- 32-lane SIMD-group reductions with
  bfloat upcasting, a ``long`` fallback (Metal has no 64-bit SIMD reduction),
  and NaN propagation for floating-point min/max.
- ``c10::metal::simd_arg{min,max}(value, index)`` -- ``simd_ballot``-based
  index selection returning a ``pair``.

This module wraps those helpers for Helion's needs: leading barriers make
threadgroup scratch reusable inside device loops, segmented butterflies isolate
sub-SIMD-group reductions, and argreductions compare carried global indices
rather than assuming that SIMD lane order is index order.

The one primitive it does *not* reuse is ``c10::metal::threadgroup_*``: that
helper re-reduces with a partially populated SIMD group, which the ``long``
emulation mis-handles.  ``HELION_TG_REDUCE`` below builds the same two-stage
reduction on top of ``c10::metal::simd_*``; its second stage never runs on a
partially populated SIMD group.
"""

from __future__ import annotations

#: ``#include`` lines added to the MSL preamble when a kernel reduces.
REDUCTION_INCLUDES: tuple[str, ...] = ("#include <c10/metal/reduction_utils.h>",)

#: Reduction kinds that map onto a ``c10::metal::`` primitive.
SUPPORTED_REDUCTIONS = frozenset({"sum", "prod", "max", "min"})
SUPPORTED_ARGREDUCTIONS = frozenset({"argmin", "argmax"})

#: MSL namespace holding Helion's reduction entry points.  Every reduction the
#: Metal backend emits goes through it -- including the ones that are a plain
#: forward to ``c10::metal`` -- so ``metal_jit`` can decide whether a kernel
#: needs the reduction preamble by looking for this single name.
REDUCTION_NAMESPACE = "helion_red"

#: How ``helion_red::group_*`` lays out a group reduction, chosen per
#: reduction site by ``reduction.group_path`` once ``metal_jit`` knows the
#: launch.  The preamble names each as ``path_<name>`` with its index here.
GROUP_PATHS = ("contiguous", "strided", "masked", "shared_tree")

REDUCTION_PREAMBLE = """
namespace helion_red {

@GROUP_PATHS@

// Whole-SIMD-group reductions, straight from Inductor.
#define HELION_SIMD_REDUCE(NAME)                                   \\
  template <typename T>                                            \\
  inline T simd_##NAME(T v) {                                      \\
    return ::c10::metal::simd_##NAME(v);                           \\
  }
HELION_SIMD_REDUCE(sum)
HELION_SIMD_REDUCE(prod)
HELION_SIMD_REDUCE(max)
HELION_SIMD_REDUCE(min)
#undef HELION_SIMD_REDUCE

// Metal has no 64-bit simd_shuffle_xor overload; shuffle the halves instead.
template <typename T>
inline T sxor(T v, ushort m) {
  return ::metal::simd_shuffle_xor(v, m);
}
inline long sxor(long v, ushort m) {
  uint2 p = as_type<uint2>(v);
  return as_type<long>(uint2(::metal::simd_shuffle_xor(p.x, m),
                             ::metal::simd_shuffle_xor(p.y, m)));
}
inline ulong sxor(ulong v, ushort m) {
  uint2 p = as_type<uint2>(v);
  return as_type<ulong>(uint2(::metal::simd_shuffle_xor(p.x, m),
                              ::metal::simd_shuffle_xor(p.y, m)));
}

// Butterfly over `span` contiguous lanes (a power of two dividing the SIMD
// width), leaving the group result in every lane of the group.  Used when a
// SIMD group holds more than one reduction group, where c10::metal::simd_*
// would fold unrelated groups together.
#define HELION_SEG_REDUCE(NAME, EXPR)                              \\
  template <typename T>                                            \\
  inline T seg_##NAME(T v, uint span) {                            \\
    for (uint off = span >> 1; off > 0; off >>= 1) {               \\
      T o = sxor(v, ushort(off));                                  \\
      v = (EXPR);                                                  \\
    }                                                              \\
    return v;                                                      \\
  }
HELION_SEG_REDUCE(sum, v + o)
HELION_SEG_REDUCE(prod, v * o)
HELION_SEG_REDUCE(max, ::c10::metal::max(v, o))
HELION_SEG_REDUCE(min, ::c10::metal::min(v, o))
#undef HELION_SEG_REDUCE

// "Is this the extremum?" -- NaN counts as one, matching torch and
// c10::metal::simd_arg*.  The integral case has no NaN.
template <typename T>
inline bool is_extremum(T v, T best) {
  return v == best;
}
inline bool is_extremum(float v, float best) {
  return v == best || ::metal::isnan(v);
}
inline bool is_extremum(half v, half best) {
  return v == best || ::metal::isnan(v);
}
inline bool is_extremum(bfloat v, bfloat best) {
  return v == best || ::metal::isnan(float(v));
}

// Whole-SIMD-group argmin/argmax.  c10::metal::simd_arg* selects the first
// winning SIMD lane, which is only the lowest index when each lane carries its
// own lane index.  A rolled reduction instead carries the winner from several
// chunks, so reduce the value first and then the global candidate index.
#define HELION_SIMD_ARGREDUCE(NAME, VALUE_OP)                      \\
  template <typename T, typename I>                                \\
  inline I simd_##NAME(T v, I idx) {                               \\
    T best = ::c10::metal::simd_##VALUE_OP(v);                     \\
    I cand = is_extremum(v, best)                                  \\
                 ? idx                                             \\
                 : ::metal::numeric_limits<I>::max();              \\
    return ::c10::metal::simd_min(cand);                           \\
  }
HELION_SIMD_ARGREDUCE(argmax, max)
HELION_SIMD_ARGREDUCE(argmin, min)
#undef HELION_SIMD_ARGREDUCE

// Segmented argmin/argmax.  c10::metal::simd_arg* ballots over all 32 lanes,
// which would span several reduction groups when span < 32; reduce the value
// within the segment, then take the lowest index attaining it (torch's
// tie-break).
#define HELION_SEG_ARGREDUCE(NAME, VALUE_OP)                       \\
  template <typename T, typename I>                                \\
  inline I seg_##NAME(T v, I idx, uint span) {                     \\
    T best = seg_##VALUE_OP(v, span);                              \\
    I cand = is_extremum(v, best)                                  \\
                 ? idx                                             \\
                 : ::metal::numeric_limits<I>::max();              \\
    return seg_min(cand, span);                                    \\
  }
HELION_SEG_ARGREDUCE(argmax, max)
HELION_SEG_ARGREDUCE(argmin, min)
#undef HELION_SEG_ARGREDUCE

// max/min identities for masked positions and padded slots.  Floating
// reductions accumulate in float, whose identities are -inf/+inf, so an
// all-inf group returns inf rather than the finite extreme.  Integral
// types have no infinity and use lowest()/max().
template <typename T>
inline T max_identity() { return ::metal::numeric_limits<T>::lowest(); }
template <>
inline float max_identity<float>() { return -::metal::numeric_limits<float>::infinity(); }
template <typename T>
inline T min_identity() { return ::metal::numeric_limits<T>::max(); }
template <>
inline float min_identity<float>() { return ::metal::numeric_limits<float>::infinity(); }

// Cross-SIMD-group reduction, in two stages: every SIMD group folds its own
// lanes and writes one scratch slot, then every thread of the reduction group
// folds those slots itself, so the result needs no broadcast (and no third
// barrier).  Every thread reads the same slots in the same order, so they
// agree bit for bit.  Up to 4 slots (spans <= 128) a plain fold over the
// slots is fastest; beyond that, a SIMD reduction over them (at 8 slots the
// fold wins on some kernels and loses on others).
// ``size / 32`` is a constant at every call site, so the branch folds away.
//
// Deliberately not a forward to ``c10::metal::threadgroup_*``.  That helper
// guards its second stage with ``idx < ceil(size / 32)``, so a span of
// 64..512 re-reduces with only 2..16 of the 32 lanes live.  Metal has no
// 64-bit SIMD reduction, so c10 emulates the ``long`` case with
// ``simd_shuffle_and_fill_down``, whose fill is ``int2(0)`` -- the identity
// for sum and for nothing else.  An int64 ``amax`` over an all-negative span
// therefore returned 0.  Reading a padded slot instead keeps all 32 lanes
// live, which is correct for every dtype and costs one select.
//
// The leading barrier makes the scratch buffer safe to reuse across
// device-loop iterations: without it a fast thread's stage-1 write for
// iteration i+1 can clobber a slow thread's stage-2 read for iteration i.
#define HELION_TG_REDUCE(NAME, IDENTITY, COMBINE)                            \\
  template <typename T>                                                      \\
  inline T tg_##NAME(threadgroup T* buf, T v, uint idx, uint size) {         \\
    constexpr uint w = ::c10::metal::simdgroup_size;                         \\
    const uint groups = size / w;                                            \\
    ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);       \\
    T part = ::c10::metal::simd_##NAME(v);                                   \\
    if (idx % w == 0) {                                                      \\
      buf[idx / w] = part;                                                   \\
    }                                                                        \\
    ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);       \\
    if (groups <= 4) {                                                       \\
      T acc = buf[0];                                                        \\
      for (uint g = 1; g < groups; ++g) {                                    \\
        T o = buf[g];                                                        \\
        acc = (COMBINE);                                                     \\
      }                                                                      \\
      return acc;                                                            \\
    }                                                                        \\
    const uint lane = idx % w;                                               \\
    T slot = lane < groups ? buf[lane] : T(IDENTITY);                        \\
    return ::c10::metal::simd_##NAME(slot);                                  \\
  }
HELION_TG_REDUCE(sum, 0, acc + o)
HELION_TG_REDUCE(prod, 1, acc * o)
HELION_TG_REDUCE(max, max_identity<T>(), ::c10::metal::max(acc, o))
HELION_TG_REDUCE(min, min_identity<T>(), ::c10::metal::min(acc, o))
#undef HELION_TG_REDUCE

// Reduction over runs of `span` consecutive lanes: the seg/simd/tg tier for
// that span.  `idx` is the lane's position in its run; `partials` (one slot
// per SIMD group) is only touched above the SIMD width, at the run's slice.
#define HELION_LANE_REDUCE(NAME)                                             \\
  template <typename T>                                                      \\
  inline T lanes_##NAME(threadgroup T* partials, T v, uint idx, uint span,   \\
                        uint simd_group) {                                   \\
    constexpr uint w = ::c10::metal::simdgroup_size;                         \\
    if (span < w) {                                                          \\
      return seg_##NAME(v, span);                                            \\
    }                                                                        \\
    if (span == w) {                                                         \\
      return simd_##NAME(v);                                                 \\
    }                                                                        \\
    const uint groups = span / w;                                            \\
    return tg_##NAME(partials + (simd_group / groups) * groups, v, idx, span); \\
  }
HELION_LANE_REDUCE(sum)
HELION_LANE_REDUCE(prod)
HELION_LANE_REDUCE(max)
HELION_LANE_REDUCE(min)
#undef HELION_LANE_REDUCE

// Reduction over the runs of positions `below` apart in aligned groups of
// `size` lanes, for a power-of-two `below`.  A butterfly folds the lanes of a
// SIMD group that share a run.  Above the SIMD width each SIMD group writes one
// partial per run and every thread folds its own run's partials in order, so
// a run's threads agree bit for bit.  All runs share two barriers, where one
// collective per run would pay two each (CuTe's
// _cute_grouped_reduce_shared_columns makes the same trade).
#define HELION_STRIDED_REDUCE(NAME, COMBINE)                                 \\
  template <typename T>                                                      \\
  inline T strided_##NAME(threadgroup T* scratch, T v, uint lane, uint below, \\
                          uint size) {                                       \\
    constexpr uint w = ::c10::metal::simdgroup_size;                         \\
    for (uint off = below; off < ::metal::min(size, w); off <<= 1) {         \\
      T o = sxor(v, ushort(off));                                            \\
      v = (COMBINE);                                                         \\
    }                                                                        \\
    if (size <= w) {                                                         \\
      return v;                                                              \\
    }                                                                        \\
    const uint runs = ::metal::min(below, w);                                \\
    const uint stride = ::metal::max(below, w);                              \\
    ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);       \\
    if (lane % w < runs) {                                                   \\
      scratch[lane / w * runs + lane % w] = v;                               \\
    }                                                                        \\
    ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);       \\
    const uint base = lane - lane % size;                                    \\
    const uint first = base + lane % below;                                  \\
    v = scratch[first / w * runs + first % w];                               \\
    for (uint p = first + stride; p < base + size; p += stride) {            \\
      T o = scratch[p / w * runs + p % w];                                   \\
      v = (COMBINE);                                                         \\
    }                                                                        \\
    return v;                                                                \\
  }
HELION_STRIDED_REDUCE(sum, v + o)
HELION_STRIDED_REDUCE(prod, v * o)
HELION_STRIDED_REDUCE(max, ::c10::metal::max(v, o))
HELION_STRIDED_REDUCE(min, ::c10::metal::min(v, o))
#undef HELION_STRIDED_REDUCE

// Reduce the launched axis. Non-contributing positions enter with the identity.
// Each thread retains its own row's result.  Rows a power of two apart take
// strided_* above; any other stride uses one masked ordinary collective per
// row, like CuTe's grouped helpers.
#define HELION_GROUP_REDUCE(NAME, IDENTITY, COMBINE)                          \\
  template <typename T>                                                      \\
  inline T group_##NAME(uint path, threadgroup T* scratch, T v, uint3 tid,    \\
                        uint axis, uint span, uint tx, uint ty, uint tz) {   \\
    constexpr uint w = ::c10::metal::simdgroup_size;                         \\
    const uint extent = axis == 0 ? tx : (axis == 1 ? ty : tz);              \\
    const uint below = axis == 0 ? 1 : (axis == 1 ? tx : tx * ty);            \\
    const uint lane = tid.x + tx * (tid.y + ty * tid.z);                     \\
    if (span < extent) {                                                    \\
      v = tid[axis] < span ? v : T(IDENTITY);                                \\
    }                                                                       \\
    if (path == path_contiguous) {                                           \\
      return lanes_##NAME(scratch, v, lane % extent, extent, lane / w);       \\
    }                                                                        \\
    const uint size = extent * below;                                        \\
    if (path == path_strided) {                                              \\
      return strided_##NAME(scratch, v, lane, below, size);                  \\
    }                                                                        \\
    const uint pos = lane % size;                                            \\
    const uint row = pos % below;                                            \\
    T selected = T(IDENTITY);                                                \\
    for (uint p = 0; p < below; ++p) {                                       \\
      T input = row == p ? v : T(IDENTITY);                                   \\
      T result;                                                              \\
      if (path == path_masked) {                                             \\
        result = lanes_##NAME(scratch, input, pos, size, lane / w);           \\
      } else {                                                               \\
        ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);     \\
        scratch[lane] = input;                                               \\
        ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);     \\
        for (uint stride = 1; stride < size; stride *= 2) {                  \\
          if (pos % (2 * stride) == 0 && pos + stride < size) {              \\
            T v = scratch[lane];                                            \\
            T o = scratch[lane + stride];                                   \\
            scratch[lane] = (COMBINE);                                       \\
          }                                                                  \\
          ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);   \\
        }                                                                    \\
        result = scratch[lane - pos];                                        \\
      }                                                                      \\
      if (row == p) { selected = result; }                                   \\
    }                                                                        \\
    return selected;                                                         \\
  }
HELION_GROUP_REDUCE(sum, 0, v + o)
HELION_GROUP_REDUCE(prod, 1, v * o)
HELION_GROUP_REDUCE(max, max_identity<T>(), ::c10::metal::max(v, o))
HELION_GROUP_REDUCE(min, min_identity<T>(), ::c10::metal::min(v, o))
#undef HELION_GROUP_REDUCE

// Argreductions compose the same collective twice. Masked identities cannot
// supply candidate indices, even when they tie a valid infinity.
#define HELION_GROUP_ARGREDUCE(NAME, VALUE_OP)                                \\
  template <typename T, typename I>                                          \\
  inline I group_##NAME(uint path, threadgroup T* vpartials,                 \\
                        threadgroup I* ipartials, T v, I i, uint3 tid,        \\
                        uint axis, uint span, uint tx, uint ty, uint tz) {   \\
    T best = group_##VALUE_OP(path, vpartials, v, tid, axis, span, tx, ty, tz);\\
    I cand = tid[axis] < span && is_extremum(v, best)                          \\
                 ? i : ::metal::numeric_limits<I>::max();                    \\
    return group_min(path, ipartials, cand, tid, axis, span, tx, ty, tz);      \\
  }
HELION_GROUP_ARGREDUCE(argmax, max)
HELION_GROUP_ARGREDUCE(argmin, min)
#undef HELION_GROUP_ARGREDUCE

}  // namespace helion_red
""".strip().replace(
    "@GROUP_PATHS@",
    "\n".join(
        f"constexpr constant uint path_{name} = {index};"
        for index, name in enumerate(GROUP_PATHS)
    ),
)

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
reduction on top of ``c10::metal::simd_*`` with the second stage padded out to
a full SIMD group.
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

REDUCTION_PREAMBLE = """
namespace helion_red {

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

// Cross-SIMD-group reduction, in two stages: every SIMD group folds its own
// lanes and writes one scratch slot, then the reduction group's first SIMD
// group folds those slots and broadcasts slot 0 back to every thread.
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
// iteration i+1 can clobber a slow thread's broadcast read for iteration i.
#define HELION_TG_REDUCE(NAME, IDENTITY)                                     \\
  template <typename T>                                                      \\
  inline T tg_##NAME(threadgroup T* buf, T v, uint idx, uint size) {         \\
    constexpr uint w = ::c10::metal::simdgroup_size;                         \\
    const uint groups = (size + w - 1) / w;                                  \\
    ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);       \\
    T part = ::c10::metal::simd_##NAME(v);                                   \\
    if (idx % w == 0) {                                                      \\
      buf[idx / w] = part;                                                   \\
    }                                                                        \\
    ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);       \\
    if (idx < w) {                                                           \\
      T slot = idx < groups ? buf[idx] : T(IDENTITY);                        \\
      T total = ::c10::metal::simd_##NAME(slot);                             \\
      if (idx == 0) {                                                        \\
        buf[0] = total;                                                      \\
      }                                                                      \\
    }                                                                        \\
    ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);       \\
    return buf[0];                                                           \\
  }
HELION_TG_REDUCE(sum, 0)
HELION_TG_REDUCE(prod, 1)
HELION_TG_REDUCE(max, ::metal::numeric_limits<T>::lowest())
HELION_TG_REDUCE(min, ::metal::numeric_limits<T>::max())
#undef HELION_TG_REDUCE

// Each tg_* below carries its own leading barrier, which is what keeps the
// value pass and the index pass from racing on their separate buffers.
#define HELION_TG_ARGREDUCE(NAME, VALUE_OP)                                  \\
  template <typename T, typename I>                                          \\
  inline I tg_##NAME(threadgroup T* vbuf, threadgroup I* ibuf, T v, I i,      \\
                     uint idx, uint size) {                                  \\
    T best = tg_##VALUE_OP(vbuf, v, idx, size);                              \\
    I cand = is_extremum(v, best)                                            \\
                 ? i                                                        \\
                 : ::metal::numeric_limits<I>::max();                       \\
    return tg_min(ibuf, cand, idx, size);                                    \\
  }
HELION_TG_ARGREDUCE(argmax, max)
HELION_TG_ARGREDUCE(argmin, min)
#undef HELION_TG_ARGREDUCE

}  // namespace helion_red
""".strip()

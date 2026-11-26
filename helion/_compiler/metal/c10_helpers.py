"""Helion's copies of the ``c10::metal`` helpers its Metal kernels call most.

Helion's MSL calls a few helpers from torch's ``c10/metal`` headers, and
including the headers brings about 160 KB of source with them, which every
compile parses and an MLX-delegate host copies on every call.  This module
holds copies of those helpers: integer and floating ``floor_divide`` and
``remainder``, NaN-propagating ``max`` and ``min``, and the SIMD-group
reductions and ``simdgroup_size`` the reduction preamble
(``msl_reduction``) uses, each with what it needs.  A kernel that names
only these, qualified or in an unqualified call the headers would resolve
to them (``UNQUALIFIED``), carries the ones it names inside ``namespace c10
{ namespace metal {``, so that its calls read and resolve as before.  A
kernel that names any other ``c10`` helper (``erf`` and the other special
functions) or reduces 64-bit integers (c10's ``long`` SIMD reductions derive
from MLX's) includes the headers as before, and none of the copies
(``c10_source``).  Over the Metal test suite's kernels and the study's
matmuls, GEMVs, attention and norm kernels, those names cover all but a few.

``test/test_metal_c10_helpers_mps.py`` checks each copy against torch's own
on edge cases, so a change in torch's headers shows up there.
"""

# The helpers below are copied from PyTorch's c10/metal/common.h, utils.h and
# reduction_utils.h (https://github.com/pytorch/pytorch), BSD-3-Clause,
# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import re

from .msl_reduction import REDUCTION_C10
from .msl_reduction import REDUCTION_INCLUDES

#: Each helper's MSL and the helpers it uses, in the order a preamble emits
#: them: a helper follows everything it uses.
HELPERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "simdgroup_size": ("constant constexpr unsigned simdgroup_size = 32;", ()),
    "is_complex_v": (
        """\
template <typename T>
constexpr constant bool is_complex_v =
    ::metal::is_same_v<T, float2> || ::metal::is_same_v<T, half2>;""",
        (),
    ),
    "is_scalar_floating_point_v": (
        """\
template <typename T>
constexpr constant bool is_scalar_floating_point_v =
    ::metal::is_floating_point_v<T> && ::metal::is_scalar_v<T>;""",
        (),
    ),
    "is_scalar_integral_v": (
        """\
template <typename T>
constexpr constant bool is_scalar_integral_v =
    ::metal::is_integral_v<T> && ::metal::is_scalar_v<T>;""",
        (),
    ),
    "common_dtype": (
        """\
template <typename U, typename V>
using common_dtype = decltype(U(0) + V(0));""",
        (),
    ),
    "max": (
        """\
template <typename T>
::metal::enable_if_t<::metal::is_floating_point_v<T>, T> max(T a, T b) {
  return ::metal::isunordered(a, b) ? NAN : ::metal::max(b, a);
}

template <typename T, typename U>
::metal::enable_if_t<::metal::is_integral_v<T>&& ::metal::is_integral_v<U>, T>
max(T a, U b) {
  return ::metal::max(a, static_cast<T>(b));
}

template <>
inline bfloat max(bfloat a, bfloat b) {
  return bfloat(
      ::metal::isunordered(a, b) ? NAN : ::metal::max(float(b), float(a)));
}""",
        (),
    ),
    "min": (
        """\
template <typename T>
::metal::enable_if_t<::metal::is_floating_point_v<T>, T> min(T a, T b) {
  return ::metal::isunordered(a, b) ? NAN : ::metal::min(b, a);
}

template <typename T, typename U>
::metal::enable_if_t<::metal::is_integral_v<T>&& ::metal::is_integral_v<U>, T>
min(T a, U b) {
  return ::metal::min(a, static_cast<T>(b));
}

template <>
inline bfloat min(bfloat a, bfloat b) {
  return bfloat(
      ::metal::isunordered(a, b) ? NAN : ::metal::min(float(b), float(a)));
}""",
        (),
    ),
    "floor_divide": (
        """\
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && is_scalar_integral_v<U>,
        bool> = true>
inline common_dtype<T, U> floor_divide(T x, U y) {
  const auto quot = x / y;
  return (x < 0) == (y < 0) ? quot : (x % y != 0) ? quot - 1 : quot;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_floating_point_v<T> && is_scalar_floating_point_v<U>,
        bool> = true>
inline common_dtype<T, U> floor_divide(T x, U y) {
  return ::metal::floor(x / y);
}""",
        ("is_scalar_integral_v", "is_scalar_floating_point_v", "common_dtype"),
    ),
    "remainder": (
        """\
template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_floating_point_v<T> || is_scalar_floating_point_v<U>,
        bool> = true>
inline float remainder(const T x, const U y) {
  const auto x_f = static_cast<float>(x);
  const auto y_f = static_cast<float>(y);
  auto mod = ::metal::fmod(x_f, y_f);
  if (mod != 0 && (y_f < 0) != (mod < 0)) {
    mod += y_f;
  }
  return mod;
}

template <
    typename T,
    typename U,
    ::metal::enable_if_t<
        is_scalar_integral_v<T> && is_scalar_integral_v<U>,
        bool> = true>
inline common_dtype<T, U> remainder(const T x, const U y) {
  auto rc = x % y;
  return rc == 0 || (x ^ y) > 0 ? rc : rc + y;
}""",
        ("is_scalar_integral_v", "is_scalar_floating_point_v", "common_dtype"),
    ),
    "simd_type": (
        """\
namespace detail {
template <typename T>
struct simd_type {
  using t = T;
};

// Helper that allows one to run simd ops over bfl16 by upcasting them to fp32
template <typename T>
using simd_type_t = typename simd_type<T>::t;

template <>
struct simd_type<bfloat> {
  using t = float;
};
} // namespace detail""",
        (),
    ),
    "simd_sum": (
        """\
template <typename T>
inline ::metal::
    enable_if_t<!::metal::is_same_v<T, long> && !c10::metal::is_complex_v<T>, T>
    simd_sum(T val) {
  return T(::metal::simd_sum(detail::simd_type_t<T>(val)));
}""",
        ("is_complex_v", "simd_type"),
    ),
    "simd_prod": (
        """\
template <typename T>
inline ::metal::
    enable_if_t<!::metal::is_same_v<T, long> && !c10::metal::is_complex_v<T>, T>
    simd_prod(T val) {
  return T(::metal::simd_product(detail::simd_type_t<T>(val)));
}""",
        ("is_complex_v", "simd_type"),
    ),
    "simd_max": (
        """\
template <
    typename T,
    ::metal::enable_if_t<::metal::is_floating_point_v<T>, bool> = true>
inline T simd_max(T val) {
  if (::metal::simd_any(::metal::isnan(val))) {
    return ::metal::numeric_limits<T>::quiet_NaN();
  }
  return T(::metal::simd_max(detail::simd_type_t<T>(val)));
}

template <
    typename T,
    ::metal::enable_if_t<::metal::is_integral_v<T> && sizeof(T) != 8, bool> =
        true>
inline T simd_max(T val) {
  return ::metal::simd_max(val);
}""",
        ("simd_type",),
    ),
    "simd_min": (
        """\
template <
    typename T,
    ::metal::enable_if_t<::metal::is_floating_point_v<T>, bool> = true>
inline T simd_min(T val) {
  if (::metal::simd_any(::metal::isnan(val))) {
    return ::metal::numeric_limits<T>::quiet_NaN();
  }
  return T(::metal::simd_min(detail::simd_type_t<T>(val)));
}

template <
    typename T,
    ::metal::enable_if_t<::metal::is_integral_v<T> && sizeof(T) != 8, bool> =
        true>
inline T simd_min(T val) {
  return ::metal::simd_min(val);
}""",
        ("simd_type",),
    ),
}

#: The headers a kernel that names a helper this module lacks includes.
C10_INCLUDES: tuple[str, ...] = (
    "#include <c10/metal/utils.h>",
    "#include <c10/metal/special_math.h>",
)

#: Metal's functions that ``c10::metal`` overloads.  The headers' ``using
#: namespace c10::metal`` (c10/metal/igamma.h, through special_math.h) puts
#: c10's overloads beside Metal's for an unqualified call, so such a call
#: names c10 as well: ``min(offset, end)`` takes c10's integral ``min`` for
#: mixed operands, which Metal's overloads leave ambiguous.
UNQUALIFIED: frozenset[str] = frozenset(
    {"min", "max", "fma", "fmod", "pow", "sqrt"}
    | {"simd_sum", "simd_max", "simd_min", "simd_broadcast"}
)

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
#: A name in ``c10`` and the name after it, ``c10::metal::floor_divide``.
_C10_NAME = re.compile(r"\bc10\s*::\s*(\w+)(?:\s*::\s*(\w+))?")
#: An unqualified call of one of ``UNQUALIFIED``.
_UNQUALIFIED_CALL = re.compile(rf"(?<![\w:.])({'|'.join(sorted(UNQUALIFIED))})\s*\(")
#: Identifiers torch's headers define outside ``namespace c10``.
_C10_MACRO = re.compile(r"\b(?:C10_\w+|IF_CONSTEXPR)\b")
#: A 64-bit integer value in a reduction (``helion_red``) call.
_LONG_REDUCTION = re.compile(r"helion_red\s*::[^;]*\bstatic_cast\s*<\s*u?long\s*>")


def c10_names(code: str) -> set[str] | None:
    """The ``c10::metal`` names *code* (MSL, comments included) uses,
    qualified or in an unqualified call (``UNQUALIFIED``), or None if it
    uses ``c10`` otherwise: another namespace in ``c10``, or a macro the
    headers define."""
    code = _COMMENT.sub("", code)
    if _C10_MACRO.search(code):
        return None
    names = set(_UNQUALIFIED_CALL.findall(code))
    for outer, name in _C10_NAME.findall(code):
        if outer != "metal" or not name or name == "detail":
            return None
        names.add(name)
    return names


def helpers_source(names: set[str]) -> list[str]:
    """The lines of ``namespace c10::metal`` holding the helpers *names*
    and every helper they use, in ``HELPERS`` order."""
    needed: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name not in needed:
            needed.add(name)
            pending.extend(HELPERS[name][1])
    return [
        "namespace c10 {",
        "namespace metal {",
        "",
        *(f"{source}\n" for name, (source, _) in HELPERS.items() if name in needed),
        "} // namespace metal",
        "} // namespace c10",
        # As the headers have it, for unqualified calls (``UNQUALIFIED``).
        "using namespace c10::metal;",
    ]


def c10_source(code: str, reduces: bool) -> tuple[list[str], list[str]]:
    """The lines that give *code* (a kernel's MSL but these lines, and but
    the reduction preamble where *reduces*) the ``c10::metal`` helpers it
    uses, in place of the c10 includes and of the reduction's: this module's
    copies of them, or where it uses one they lack, torch's headers
    (``C10_INCLUDES``, and ``REDUCTION_INCLUDES`` for a reduction)."""
    names = c10_names(code)
    if (
        names is None
        or not names <= HELPERS.keys()
        or (reduces and _LONG_REDUCTION.search(code))
    ):
        return list(C10_INCLUDES), list(REDUCTION_INCLUDES if reduces else ())
    if reduces:
        names |= REDUCTION_C10
    return (helpers_source(names) if names else []), []

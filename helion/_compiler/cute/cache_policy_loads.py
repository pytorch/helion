"""Names of the generated L2-policy vector load helpers.

``load_eviction_policies`` values that ``cute.arch.load`` cannot express
(``l2_last``, ``l1_l2_first``, ``l1_l2_last``) are emitted as calls to the
inline-PTX helpers of ``l2_policy.py`` (``createpolicy.fractional``) with the
same call shape as the packet load,
``NAME(PTR, ir.VectorType.get([V], ELEM.mlir_type))``; one helper per packet
width (16, 8 and 4 bytes), chosen by ``cache_hinted_load_helper``.  The lane-loop passes
treat every listed name exactly like ``cute.arch.load``: relocatable,
collected as a memory load and measured by the aliasing proofs.
``CuteBackend.library_imports`` binds these names in generated code; a helper
added there must be added here (``test_load_helper_names_match_the_backend_
imports`` pins the two lists together).
"""

from __future__ import annotations

# The eviction suffix ``memory_ops`` attaches to a hinted load site (it is not
# a ``cute.arch.load`` kwarg but a marker for the helper form).
_CUTE_L2_LAST_SUFFIX = "__l2_last__"
# Eviction suffix -> 16-byte helper.
_CUTE_CACHE_LOAD_HELPERS = {
    _CUTE_L2_LAST_SUFFIX: "_cute_load_l2_evict_last",
    "__l1_l2_first__": "_cute_load_l1_l2_evict_first",
    "__l1_l2_last__": "_cute_load_l1_l2_evict_last",
}
# Eviction suffix -> 8-byte helper.
_CUTE_CACHE_LOAD_8B_HELPERS = {
    suffix: f"{name}_8b" for suffix, name in _CUTE_CACHE_LOAD_HELPERS.items()
}
# Eviction suffix -> 4-byte helper.
_CUTE_CACHE_LOAD_4B_HELPERS = {
    suffix: f"{name}_4b" for suffix, name in _CUTE_CACHE_LOAD_HELPERS.items()
}
# Packet byte width -> the helper table for it.  A hinted packet of any other
# width (2-byte scalars, byte-packed dtypes) keeps the plain load.
_CUTE_CACHE_LOAD_HELPERS_BY_BYTES = {
    16: _CUTE_CACHE_LOAD_HELPERS,
    8: _CUTE_CACHE_LOAD_8B_HELPERS,
    4: _CUTE_CACHE_LOAD_4B_HELPERS,
}
# Every generated-code name of an L2-policy vector load helper.
_CUTE_CACHE_LOAD_HELPER_NAMES = frozenset(
    name
    for helpers in _CUTE_CACHE_LOAD_HELPERS_BY_BYTES.values()
    for name in helpers.values()
)


def cache_hinted_load_helper(eviction_suffix: str, packet_bytes: int) -> str | None:
    """The L2-policy load helper for a ``packet_bytes``-wide packet whose
    load site carries ``eviction_suffix``, or None when the suffix is not an
    L2 policy marker or no helper exists for that width (the caller then
    emits the plain ``cute.arch.load``)."""
    helpers = _CUTE_CACHE_LOAD_HELPERS_BY_BYTES.get(packet_bytes)
    if helpers is None:
        return None
    return helpers.get(eviction_suffix)

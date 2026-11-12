"""The PTX forms of a CuTe load site's cache hint.

Role: general-purpose emitter.  The device helpers that write hinted loads as
inline PTX (``l2_policy``, ``masked_packet_load``) and the MMA DAG's hinted
``cp.async`` packets (``cute_mma_dag``) take their L1 and L2 qualifiers from
these tables.  The ``createpolicy`` descriptor setup is still written out in
``l2_policy._policy_assembly`` and ``masked_packet_load``, and
``l2_policy.scalar_policy_load``, ``readonly_packet_load_helpers``' 32-byte
load, ``device/ptx.py`` and ``row_matrix_transport_codegen`` keep literal
forms.  The module is DSL-free and imports nothing, so the passes import it
too.  The CuTe disk-cache key follows a helper's globals only into modules
that define the functions and classes it names (``__module__``), never into
a constants-only module like this one: it is listed in
``source_dependencies._COMMON_DEPENDENCIES`` instead.
"""

from __future__ import annotations

# L1 eviction priority -> its PTX qualifier: the values ``cute.arch.load``
# accepts as ``level1_eviction_priority``, which are the searched ``first`` /
# ``last`` and every explicit ``hl.load(..., eviction_policy=...)`` spelling.
L1_PRIORITY_QUALIFIERS: dict[str, str] = {
    "evict_normal": ".L1::evict_normal",
    "evict_first": ".L1::evict_first",
    "evict_last": ".L1::evict_last",
    "evict_unchanged": ".L1::evict_unchanged",
    "evict_no_allocate": ".L1::no_allocate",
}

# L2 cache-policy family (``CachePolicy.l2_helper``) -> the qualifier its
# loads put ahead of ``.L2::cache_hint`` (the family's L1 eviction priority,
# ``.nc`` for the read-only form) and the eviction policy of the
# ``createpolicy.fractional`` descriptor they pass.
L2_POLICY_FORMS: dict[str, tuple[str, str]] = {
    "l2_last": ("", "evict_last"),
    "l1_l2_first": (".L1::evict_first", "evict_first"),
    "l1_l2_last": (".L1::evict_last", "evict_last"),
    "nc_l2_first": (".nc", "evict_first"),
}

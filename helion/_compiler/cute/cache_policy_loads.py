"""The cache hint of a CuTe load site (``CachePolicy``) and the L2-policy
vector load helper names it selects.

A load site's hint is one of: nothing; an L1 eviction priority
(``cute.arch.load(ptr, T, level1_eviction_priority='evict_last')``); the
streaming cache operator (``cop='cs'``, evict-first at L1 and L2); or an L2
policy ``cute.arch.load`` cannot express (``l2_last``, ``l1_l2_first``,
``l1_l2_last`` and, for read-only sources, ``nc_l2_first``), which is emitted
as a call to the inline-PTX helpers of ``l2_policy.py`` /
``readonly_packet_load_helpers.py`` (``createpolicy.fractional``) with the
packet load's call shape, ``NAME(PTR, ir.VectorType.get([V], ELEM.mlir_type))``,
one helper per packet width (``CachePolicy.vector_helper``; the 32-byte row
of the L2 policies -- the 256-bit ``.v8.b32`` load, SM100+ PTX -- only behind
the ``cute_wide_packet_hints`` config knob, ``hint_wide_packets``).  The
predicated packet loads of ``masked_packet_load`` take the hint as their
priority argument instead (``CachePolicy.masked_priority``), at every width.

The load lowering threads the hint through its load-site facts
(``CuteScalarLoadSite``, ``CuteVloopLoadFact``, the persistent branch-vec
marker) as the ``eviction_suffix`` string: the ``cute.arch.load`` keyword
fragment for the first two forms and the ``__<policy>__`` marker for the L2
helpers.  ``CachePolicy.suffix`` renders that string and
``CachePolicy.from_suffix`` parses it back, so every emitter reads the hint
through one object instead of inspecting the string.

``PACKET_LOAD_HELPERS`` lists the load helpers generated code binds:
``CuteBackend.library_imports`` takes its rows from it, and the width tables
below (the L2-policy helpers by packet width, the ``nc_l2_first`` pair, the
predicated ``_cute_load_vNb_if`` loads of ``masked_packet_load``) derive from
it, so a helper is spelled once.  ``test_cute_packet_load_helper_table.py``
pins the rows to the names and import order generated code had before the
table.  The lane-loop passes treat every L2-policy name exactly like
``cute.arch.load``: relocatable, collected as a memory load and measured by
the aliasing proofs.
"""

from __future__ import annotations

import ast
import dataclasses
from typing import TYPE_CHECKING
from typing import ClassVar

import torch

from ... import exc
from .cache_hint_forms import L1_PRIORITY_QUALIFIERS
from .cutedsl_compat import readonly_packet_load_supported

if TYPE_CHECKING:
    from collections.abc import Sequence


def _l2_marker(policy: str) -> str:
    """The ``eviction_suffix`` marker of an L2 helper policy (``__l2_last__``):
    the one spelling ``CachePolicy.suffix`` renders and ``from_suffix``
    parses."""
    return f"__{policy}__"


# ``load_eviction_policies`` value -> ``level1_eviction_priority`` keyword value.
_CUTE_EVICTION_POLICY_MAP = {
    "": "",
    "first": "evict_first",
    "last": "evict_last",
}
# L2-policy helper family (a ``load_eviction_policies`` value) -> the stem of
# its helpers' names: ``load_v<bytes>b_<stem>`` in ``l2_policy`` and
# ``_cute_load_<stem>[_<bytes>b]`` in generated code.
_L2_HELPER_STEMS = {
    "l2_last": "l2_evict_last",
    "l1_l2_first": "l1_l2_evict_first",
    "l1_l2_last": "l1_l2_evict_last",
}
# The ``load_eviction_policies`` values emitted as L2-policy helper markers.
_L2_HELPER_POLICIES = frozenset(_L2_HELPER_STEMS)
_NC_L2_FIRST_POLICY = "nc_l2_first"
_STREAMING_KWARGS = ", cop='cs'"
_L1_KWARG_PREFIX = ", level1_eviction_priority="
# The ``cute.arch.load`` keywords that spell a cache hint.
LOAD_POLICY_KEYWORDS = frozenset({"level1_eviction_priority", "cop"})
# ``level1_eviction_priority`` value -> ``_cute_scalar_policy_load`` policy name.
_SCALAR_POLICY_NAMES = {"evict_first": "first", "evict_last": "last"}


@dataclasses.dataclass(frozen=True)
class PacketLoadHelper:
    """One load helper ``CuteBackend.library_imports`` binds in generated code.

    ``name`` is the generated-code name and ``symbol`` the binding of
    ``helion._compiler.cute.<module>`` behind it.  ``family`` is the
    ``CachePolicy.l2_helper`` family of an L2-policy packet load
    (``l2_last`` / ``l1_l2_first`` / ``l1_l2_last`` / ``nc_l2_first``),
    ``predicated`` for the ``_if`` loads (their priority is an argument) or
    ``scalar`` for the scalar policy load; ``packet_bytes`` is the packet
    width, 0 for the scalar load (its width follows its dtype).
    """

    name: str
    module: str
    symbol: str
    family: str
    packet_bytes: int

    @property
    def import_line(self) -> str:
        """The ``library_imports`` row binding ``name``."""
        return (
            f"from helion._compiler.cute.{self.module} import {self.symbol} "
            f"as {self.name}"
        )


def _l2_helper(policy: str, packet_bytes: int) -> PacketLoadHelper:
    stem = _L2_HELPER_STEMS[policy]
    # The 16-byte helpers predate the width suffix.
    suffix = "" if packet_bytes == 16 else f"_{packet_bytes}b"
    return PacketLoadHelper(
        f"_cute_load_{stem}{suffix}",
        "l2_policy",
        f"load_v{packet_bytes}b_{stem}",
        policy,
        packet_bytes,
    )


def _predicated_helper(packet_bytes: int) -> PacketLoadHelper:
    return PacketLoadHelper(
        f"_cute_load_v{packet_bytes}b_if",
        "masked_packet_load",
        f"load_v{packet_bytes}b_if",
        "predicated",
        packet_bytes,
    )


def _nc_helper(packet_bytes: int) -> PacketLoadHelper:
    return PacketLoadHelper(
        f"_cute_load_nc_l2_evict_first_{packet_bytes}b",
        "readonly_packet_load_helpers",
        f"load_nc_l2_evict_first_{packet_bytes}b",
        _NC_L2_FIRST_POLICY,
        packet_bytes,
    )


# Every ``_cute_load_*`` helper and the scalar policy load, in the order
# ``CuteBackend.library_imports`` binds them.  The preamble emits a kernel's
# imports in that order, so it is part of the rendered source: the rows keep
# the order the helpers were added in rather than their widths, and the scalar
# policy load sits among them.
PACKET_LOAD_HELPERS = (
    _l2_helper("l2_last", 16),
    _predicated_helper(64),
    _predicated_helper(32),
    _predicated_helper(16),
    _predicated_helper(8),
    _predicated_helper(4),
    _nc_helper(32),
    _nc_helper(16),
    PacketLoadHelper(
        "_cute_scalar_policy_load", "l2_policy", "scalar_policy_load", "scalar", 0
    ),
    _l2_helper("l1_l2_first", 16),
    _l2_helper("l1_l2_first", 8),
    _l2_helper("l1_l2_last", 16),
    _l2_helper("l1_l2_last", 8),
    _l2_helper("l2_last", 8),
    _l2_helper("l2_last", 4),
    _l2_helper("l1_l2_first", 4),
    _l2_helper("l1_l2_last", 4),
    _l2_helper("l2_last", 32),
    _l2_helper("l1_l2_first", 32),
    _l2_helper("l1_l2_last", 32),
)
# Packet byte width -> noncoherent L2 evict-first helper.
_CUTE_NC_L2_FIRST_HELPERS = {
    helper.packet_bytes: helper.name
    for helper in PACKET_LOAD_HELPERS
    if helper.family == _NC_L2_FIRST_POLICY
}
# (packet byte width, L2-policy family) -> helper.  A hinted packet of any
# other width (2-byte scalars, byte-packed dtypes) keeps the plain load.  The
# 32-byte rows (one ``ld.global...v8.b32``, SM100+) are opt-in:
# ``vector_helper`` consults them only with ``wide_packets``
# (``hint_wide_packets`` of the config), so a config without the knob renders
# the plain load a hinted 32-byte packet always had.
_CUTE_CACHE_LOAD_HELPERS = {
    (helper.packet_bytes, helper.family): helper.name
    for helper in PACKET_LOAD_HELPERS
    if helper.family in _L2_HELPER_POLICIES
}
_CUTE_WIDE_PACKET_BYTES = 32
# Every generated-code name of an L2-policy vector load helper.
_CUTE_CACHE_LOAD_HELPER_NAMES = frozenset(
    _CUTE_CACHE_LOAD_HELPERS.values()
) | frozenset(_CUTE_NC_L2_FIRST_HELPERS.values())
# Lane byte width -> predicated packet load (32 and 64 bytes: two and four
# consecutive 16-byte packets under one predicate).  The AST passes that
# recognize these calls read the names here, wherever Helion imports; the
# helpers themselves import the CuTe DSL.
MASKED_PACKET_LOAD_HELPERS = {
    helper.packet_bytes: helper.name
    for helper in PACKET_LOAD_HELPERS
    if helper.family == "predicated"
}
MASKED_PACKET_LOAD_HELPER_NAMES = frozenset(MASKED_PACKET_LOAD_HELPERS.values())


@dataclasses.dataclass(frozen=True)
class CachePolicy:
    """The cache hint of one load site.

    ``l1_priority`` is the ``level1_eviction_priority`` keyword value
    (``evict_first`` / ``evict_last``, or the explicit ``hl.load(...,
    eviction_policy=)`` spelling, which ``cute.arch.load`` shares with
    Triton); ``streaming`` is the ``cop='cs'`` cache operator; ``l2_helper``
    names the inline-PTX helper family (``l2_last``, ``l1_l2_first``,
    ``l1_l2_last``, ``nc_l2_first``).  At most one is set.
    """

    l1_priority: str = ""
    streaming: bool = False
    l2_helper: str = ""

    NONE: ClassVar[CachePolicy]

    @classmethod
    def explicit(cls, eviction_policy: str) -> CachePolicy:
        """The hint of an explicit ``hl.load(..., eviction_policy=...)``: an
        L1 eviction priority (``L1_PRIORITY_QUALIFIERS``), forwarded as the
        L1 priority; "" is no hint.  Any other spelling is refused: the
        predicated packet loads take an L2 policy family's name in the same
        argument, so ``nc_l2_first`` would load noncoherently with no check
        that the kernel never writes the source."""
        if not eviction_policy:
            return cls.NONE
        if eviction_policy not in L1_PRIORITY_QUALIFIERS:
            raise exc.BackendUnsupported(
                "cute",
                f"hl.load(..., eviction_policy={eviction_policy!r}): a CuTe load "
                f"takes an L1 eviction priority ({', '.join(L1_PRIORITY_QUALIFIERS)})",
            )
        return cls(l1_priority=eviction_policy)

    @classmethod
    def tunable(cls, policy: str) -> CachePolicy:
        """The hint of a ``load_eviction_policies`` value.

        ``streaming`` is the cache operator; ``nc_l2_first`` and the three
        L2 policies name an inline-PTX helper family (the caller has already
        checked that ``nc_l2_first``'s source is read-only); ``first`` /
        ``last`` map to the L1 priorities; anything else is no hint.
        """
        if policy == "streaming":
            return cls(streaming=True)
        if policy == _NC_L2_FIRST_POLICY or policy in _L2_HELPER_POLICIES:
            return cls(l2_helper=policy)
        if mapped := _CUTE_EVICTION_POLICY_MAP.get(policy, ""):
            return cls(l1_priority=mapped)
        return cls.NONE

    @classmethod
    def from_suffix(cls, eviction_suffix: str) -> CachePolicy:
        """Parse the ``eviction_suffix`` wire format (``suffix`` inverted)."""
        if not eviction_suffix:
            return cls.NONE
        if eviction_suffix == _STREAMING_KWARGS:
            return cls(streaming=True)
        if eviction_suffix.startswith(_L1_KWARG_PREFIX):
            value = ast.literal_eval(eviction_suffix[len(_L1_KWARG_PREFIX) :])
            assert isinstance(value, str)
            return cls(l1_priority=value)
        policy = eviction_suffix[2:-2]
        assert eviction_suffix == _l2_marker(policy) and (
            policy == _NC_L2_FIRST_POLICY or policy in _L2_HELPER_POLICIES
        ), eviction_suffix
        return cls(l2_helper=policy)

    @classmethod
    def from_load_keywords(cls, keywords: Sequence[ast.keyword]) -> CachePolicy | None:
        """The hint spelled by a generated ``cute.arch.load`` call's keywords,
        or None when a keyword is not a string-valued cache hint."""
        policy = cls.NONE
        for keyword in keywords:
            if not (
                isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                return None
            if keyword.arg == "level1_eviction_priority":
                policy = dataclasses.replace(policy, l1_priority=keyword.value.value)
            elif keyword.arg == "cop" and keyword.value.value == "cs":
                policy = dataclasses.replace(policy, streaming=True)
            else:
                return None
        return policy

    @property
    def suffix(self) -> str:
        """The ``eviction_suffix`` wire format of this hint."""
        if self.l2_helper:
            return _l2_marker(self.l2_helper)
        if self.streaming:
            return _STREAMING_KWARGS
        if self.l1_priority:
            return f"{_L1_KWARG_PREFIX}{self.l1_priority!r}"
        return ""

    @property
    def is_l2_marker(self) -> bool:
        """Whether the hint is an inline-PTX helper family, which
        ``cute.arch.load`` cannot express (its keyword fragment is empty)."""
        return bool(self.l2_helper)

    @property
    def load_kwargs(self) -> str:
        """The ``cute.arch.load`` keyword fragment (``", cop='cs'"`` /
        ``", level1_eviction_priority='evict_last'"``); "" for no hint and for
        the L2 helper families."""
        return "" if self.l2_helper else self.suffix

    def vector_helper(
        self, packet_bytes: int, *, wide_packets: bool = False
    ) -> str | None:
        """The L2-policy load helper for a ``packet_bytes``-wide packet, or
        None when the hint is not an L2 helper family or no helper exists for
        that width (the caller then emits the plain ``cute.arch.load``).  The
        32-byte row of the L2 policies exists only under ``wide_packets``
        (``hint_wide_packets``); the 32-byte ``nc_l2_first`` helper has its
        own gate at the load site."""
        if not self.l2_helper:
            return None
        if self.l2_helper == _NC_L2_FIRST_POLICY:
            return _CUTE_NC_L2_FIRST_HELPERS.get(packet_bytes)
        if packet_bytes == _CUTE_WIDE_PACKET_BYTES and not wide_packets:
            return None
        return _CUTE_CACHE_LOAD_HELPERS.get((packet_bytes, self.l2_helper))

    @property
    def masked_priority(self) -> str:
        """The priority argument of the predicated packet loads
        (``_cute_load_vNb_if``): the L2 helper family (the predicated loads
        carry its ``createpolicy`` descriptor, at every width, 32 and 64
        bytes too without ``cute_wide_packet_hints``), the L1 priority,
        ``cs`` for streaming, "" otherwise."""
        if self.l2_helper:
            return self.l2_helper
        if self.l1_priority:
            return self.l1_priority
        if self.streaming:
            return "cs"
        return ""

    @property
    def scalar_policy(self) -> str | None:
        """The ``_cute_scalar_policy_load`` policy name of this hint
        (``first`` / ``last`` / ``streaming``), or None when it has none."""
        if self.streaming:
            return "streaming"
        return _SCALAR_POLICY_NAMES.get(self.l1_priority)


CachePolicy.NONE = CachePolicy()


def hint_wide_packets(cute_wide_packet_hints: bool) -> bool:
    """Whether a hinted 32-byte packet takes its ``_32b`` helper: the
    ``cute_wide_packet_hints`` config knob (the argument) on a target whose
    toolchain has the 256-bit hinted load (SM100+ with CUDA 13, the gate of
    the ``nc_l2_first`` 32-byte helper).  Off, the packet keeps the plain
    ``cute.arch.load`` it had before the row existed.  Reads the compile
    environment only when the knob is set, so a pass run outside one (the
    standalone pass tests) sees the default."""
    if not cute_wide_packet_hints:
        return False
    # Imported here: this module is loaded while the compile environment's
    # own imports are still resolving.
    from ..compile_environment import CompileEnvironment

    return readonly_packet_load_supported(
        CompileEnvironment.current().config_spec.target_device_capability,
        torch.version.cuda,
    )

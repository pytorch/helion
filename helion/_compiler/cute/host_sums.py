"""The host-sum pass: the three host-sum tails in their fixed order.

Under the ``cute_host_paired_sum`` knob three host rewrites run after the
launch, in this order: :mod:`host_paired_sum` (two independent ``sum(0).to``
casts), :mod:`host_single_sum` (one returned ``sum(0).to`` cast) and
:mod:`host_full_sum` (a 1-D ``sum``/``mean`` after the device loops).  Each
kind owns its proof, helper name and temporaries over the shared prelude,
recognizer and :class:`~helion._compiler.cute.host_sum_site.HostSumSite`
emitter of :mod:`host_sum_site`; this module is the one entry point the
generator (:func:`lower_host_sums`) and the knob's heuristic
(:func:`host_sums_available`) call, and the knob's key and choices are
re-exported here for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .host_full_sum import lower_host_full_sum
from .host_full_sum import prove_host_full_sum
from .host_paired_sum import lower_host_sum_pairs
from .host_paired_sum import prove_host_sum_pairs
from .host_single_sum import lower_host_single_sum
from .host_single_sum import prove_host_single_sum
from .host_sum_site import PAIRED_SUM_CHOICES
from .host_sum_site import PAIRED_SUM_KEY

if TYPE_CHECKING:
    import ast

    from ..host_function import HostFunction

__all__ = [
    "PAIRED_SUM_CHOICES",
    "PAIRED_SUM_KEY",
    "host_sums_available",
    "lower_host_sums",
]


def lower_host_sums(
    host: HostFunction, body: list[ast.AST], layout: str
) -> list[ast.AST]:
    """Apply the three host-sum rewrites in their fixed order (pairs, then the
    returned single sum, then the full sums); ``layout`` is the knob value."""
    for lower in (lower_host_sum_pairs, lower_host_single_sum, lower_host_full_sum):
        body = lower(host, body, layout)
    return body


def host_sums_available(host: HostFunction) -> bool:
    """Whether any of the three proofs admits a site (the knob's availability)."""
    return bool(
        prove_host_sum_pairs(host).pairs
        or prove_host_single_sum(host).site is not None
        or prove_host_full_sum(host).sites
    )

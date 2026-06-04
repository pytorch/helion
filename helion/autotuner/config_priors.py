from __future__ import annotations

import random
from typing import TYPE_CHECKING
from typing import Callable

if TYPE_CHECKING:
    from .config_fragment import ConfigSpecFragment

# A ``ValuePrior`` biases the random sampling of a single flat-config slot that
# belongs to one config key. It receives the slot's fragment and the slot's
# position within the key (``0`` for scalar keys; ``0..n-1`` for sequence keys
# such as ``indexing`` or ``block_sizes``) and returns a biased value, or
# ``None`` to fall back to the fragment's uniform ``random()``.
#
# Returned values must be representable by the fragment (e.g. a member of an
# ``EnumFragment``'s ``choices``, or within an integer fragment's range);
# :func:`weighted_choice` enforces this, so a backend can list aspirational
# values without worrying about which a given kernel actually exposes.
ValuePrior = Callable[["ConfigSpecFragment", int], object]


def _fragment_accepts(fragment: ConfigSpecFragment, value: object) -> bool:
    """Best-effort check that ``value`` lies in this fragment's value space."""
    choices = getattr(fragment, "choices", None)
    if choices is not None:
        return value in choices
    low = getattr(fragment, "low", None)
    high = getattr(fragment, "high", None)
    if isinstance(low, int) and isinstance(high, int) and isinstance(value, int):
        return low <= value <= high
    return True


def weighted_choice(weighted_values: dict[object, float]) -> ValuePrior:
    """Build a position-agnostic prior that samples from ``weighted_values``.

    Values the slot's fragment cannot represent are dropped before sampling; if
    nothing representable remains, the prior returns ``None`` so the caller falls
    back to the fragment's uniform ``random()``. This lets a backend express one
    bias (e.g. ``{"tensor_descriptor": 4.0, "pointer": 1.0}``) that applies to
    whichever slots of a key the live kernel actually exposes.
    """
    items = [(v, w) for v, w in weighted_values.items() if w > 0]

    def _prior(fragment: ConfigSpecFragment, position: int) -> object:
        allowed = [(v, w) for v, w in items if _fragment_accepts(fragment, v)]
        if not allowed:
            return None
        values, weights = zip(*allowed, strict=True)
        return random.choices(values, weights=weights, k=1)[0]

    return _prior

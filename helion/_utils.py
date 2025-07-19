from __future__ import annotations

import collections

counters: collections.defaultdict[str, collections.Counter[str]] = (
    collections.defaultdict(collections.Counter)
)

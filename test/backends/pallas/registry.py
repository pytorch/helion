"""Known Pallas gaps in the portable test suite."""

from __future__ import annotations

from test.backends import backend_gap

GAPS = [
    backend_gap(
        "test_indexing",
        "TestIndexing::test_arange",
        reason="a padded tile does not mask the partial output store",
    )
]

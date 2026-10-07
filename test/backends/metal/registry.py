"""Known Metal gaps in the portable test suite."""

from __future__ import annotations

from test.backends import xfail_test

xfail_test(
    "metal",
    "test_indexing",
    "TestIndexing::test_arange",
    reason="a bare tile.index store fails Metal code generation",
)

"""FlyDSL backend — portable test registration and gap declarations.

To triage gaps: run ``HELION_BACKEND=flydsl pytest test/portable/`` and replace
the whole-file xfail entries below with per-test ``xfail_test()`` calls (or
remove them for tests that pass).

Once FlyDSL is added to CI (see ``.github/matrix.json``), do the triage above
so that any unexpected passes surface as XPASS failures immediately rather than
accumulating silently.
"""

from __future__ import annotations

from test.backends import register_backend
from test.backends import xfail_test

_backend = register_backend("flydsl")

xfail_test(
    _backend,
    "test_views",
    "",
    reason="test_views not yet evaluated on the FlyDSL backend",
)

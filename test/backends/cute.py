"""CuTe (CUTLASS CuTe DSL) backend — portable test registration.

All tests in test/portable/ are expected to pass on the CuTe backend.
"""

from __future__ import annotations

from test.backends import register_backend

register_backend("cute")

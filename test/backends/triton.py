"""Triton backend — portable test registration.

All tests in test/portable/ are expected to pass on the Triton backend.
Tensor-descriptor tests additionally require sm90+ CUDA hardware and carry
@skipUnlessTensorDescriptor directly in the test file.
"""

from __future__ import annotations

from test.backends import register_backend

register_backend("triton")

"""TileIR backend — portable test registration.

TileIR extends TritonBackend and shares its codegen. All tests in
test/portable/ are expected to pass on the TileIR backend.
"""

from __future__ import annotations

from test.backends import register_backend

register_backend("tileir")

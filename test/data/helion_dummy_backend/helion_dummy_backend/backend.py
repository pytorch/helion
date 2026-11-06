"""Reference out-of-tree backend used by test/test_backend_plugins.py.

Imports Helion only through ``helion.backend_api`` and reuses Triton codegen
under a new name (the TileIRBackend pattern).
"""

from __future__ import annotations

from helion import backend_api


class TritonOOTBackend(backend_api.TritonBackend):
    HELION_BACKEND_API_VERSION = 1

    @property
    def name(self) -> str:
        return "triton_oot"

    @property
    def codegen_name(self) -> str:
        return "triton"
